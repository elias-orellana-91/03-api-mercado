"""Ranking de tickers y resumen por sector."""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import Price, Ticker
from app.errors import ApiError
from app.services import metrics as M
from app.services.queries import price_frame

METRIC_FIELDS = {"return": "total_return", "volatility": "volatility", "sharpe": "sharpe"}


def period_start(period: str, end: date) -> date:
    """Primer día calendario del periodo que contiene ``end``."""
    if period == "month":
        return end.replace(day=1)
    if period == "quarter":
        return end.replace(month=3 * ((end.month - 1) // 3) + 1, day=1)
    if period == "ytd":
        return end.replace(month=1, day=1)
    raise ValueError(period)


def resolve_window(
    db: Session, period: str, start: date | None, end: date | None, latest: date
) -> tuple[date, date]:
    """Devuelve (fecha_base, fecha_fin) para consultar precios.

    * month/quarter/ytd: acumulado del periodo hasta la última fecha con datos. La base es el último
      cierre ANTERIOR al inicio del periodo (así el retorno incluye el primer día del periodo).
    * custom: requiere ``start``; ``end`` por defecto es la última fecha.
    """
    if period == "custom":
        if start is None:
            raise ApiError(422, "missing_start", "period=custom requiere el parámetro 'start'.")
        end = end or latest
        if start > end:
            raise ApiError(422, "invalid_range", "'start' no puede ser posterior a 'end'.")
        return start, end
    if start is not None or end is not None:
        raise ApiError(422, "unexpected_dates", "'start'/'end' solo se aceptan con period=custom.")
    first_day = period_start(period, latest)
    base = db.scalar(select(func.max(Price.date)).where(Price.date < first_day))
    return base or first_day, latest


def rank_tickers(
    db: Session, period: str, metric: str, order: str, start: date | None, end: date | None, limit: int
) -> dict:
    latest = db.scalar(select(func.max(Price.date)))
    if latest is None:
        raise ApiError(404, "no_data", "No hay datos de precios cargados.")
    base, end_date = resolve_window(db, period, start, end, latest)
    tickers = {t.symbol: t for t in db.scalars(select(Ticker))}
    frame = price_frame(db, list(tickers), base, end_date)
    if len(frame) < 2:
        raise ApiError(422, "insufficient_data", "El rango no tiene al menos 2 fechas con datos.")

    values: dict[str, float | None] = {}
    for sym in frame.columns:
        series = frame[sym].dropna()
        values[sym] = getattr(M.compute_metrics(series), METRIC_FIELDS[metric]) if len(series) >= 2 else None

    ranked = sorted(
        values.items(),
        key=lambda kv: (kv[1] is None, -(kv[1] or 0) if order == "desc" else (kv[1] or 0), kv[0]),
    )[:limit]
    items = [
        {"rank": i, "ticker": s, "name": tickers[s].name, "sector": tickers[s].sector, "value": v}
        for i, (s, v) in enumerate(ranked, start=1)
    ]
    return {
        "period": period,
        "metric": metric,
        "order": order,
        "start": frame.index[0].date(),
        "end": frame.index[-1].date(),
        "best": items[0] if items and items[0]["value"] is not None else None,
        "items": items,
    }


def sectors_summary(db: Session, start: date | None, end: date | None) -> dict:
    lo, hi = db.execute(select(func.min(Price.date), func.max(Price.date))).one()
    if lo is None:
        raise ApiError(404, "no_data", "No hay datos de precios cargados.")
    start, end = start or lo, end or hi
    if start > end:
        raise ApiError(422, "invalid_range", "'start' no puede ser posterior a 'end'.")
    tickers = list(db.scalars(select(Ticker).order_by(Ticker.sector, Ticker.symbol)))
    frame = price_frame(db, [t.symbol for t in tickers], start, end)
    if len(frame) < 2:
        raise ApiError(422, "insufficient_data", "El rango no tiene al menos 2 fechas con datos.")

    totals = {c: M.total_return(frame[c].dropna()) for c in frame.columns if frame[c].dropna().size >= 2}
    out = []
    for sector in sorted({t.sector for t in tickers}):
        syms = [t.symbol for t in tickers if t.sector == sector and t.symbol in totals]
        if not syms:
            continue
        rets = frame[syms].pct_change(fill_method=None).iloc[1:].mean(axis=1)  # igual ponderación
        index = pd.concat([pd.Series([1.0], index=[frame.index[0]]), (1 + rets).cumprod()])
        m = M.compute_metrics(index)
        out.append(
            {
                "sector": sector,
                "tickers": syms,
                "total_return": m.total_return,
                "volatility": m.volatility,
                "max_drawdown": m.max_drawdown,
                "sharpe": m.sharpe,
                "best_ticker": max(syms, key=lambda s: totals[s]),
                "worst_ticker": min(syms, key=lambda s: totals[s]),
            }
        )
    return {"start": frame.index[0].date(), "end": frame.index[-1].date(), "sectors": out}
