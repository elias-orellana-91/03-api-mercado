"""Acceso a datos (solo ORM) y utilidades de lectura."""
from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import Price, Ticker, Vix


def get_ticker(db: Session, symbol: str) -> Ticker | None:
    return db.get(Ticker, symbol.upper())


def list_tickers(db: Session) -> list[Ticker]:
    return list(db.scalars(select(Ticker).order_by(Ticker.sector, Ticker.symbol)))


def date_bounds(db: Session) -> tuple[date | None, date | None]:
    lo, hi = db.execute(select(func.min(Price.date), func.max(Price.date))).one()
    return lo, hi


def price_frame(
    db: Session, symbols: list[str], start: date | None = None, end: date | None = None
) -> pd.DataFrame:
    """Cierres ajustados en formato ancho (índice fecha, columnas tickers)."""
    stmt = select(Price.date, Price.ticker, Price.close).where(Price.ticker.in_(symbols))
    if start:
        stmt = stmt.where(Price.date >= start)
    if end:
        stmt = stmt.where(Price.date <= end)
    rows = db.execute(stmt.order_by(Price.date)).all()
    if not rows:
        return pd.DataFrame(columns=symbols, dtype=float)
    df = pd.DataFrame(rows, columns=["date", "ticker", "close"])
    out = df.pivot(index="date", columns="ticker", values="close").sort_index()
    out.index = pd.to_datetime(out.index)
    return out


def last_rows_frames(
    db: Session, symbols: list[str], as_of: date, n_rows: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Últimas ``n_rows`` fechas (<= as_of) de precios, volumen y VIX para ``symbols``."""
    dates = list(
        db.scalars(
            select(Price.date)
            .where(Price.ticker.in_(symbols), Price.date <= as_of)
            .group_by(Price.date)
            .order_by(Price.date.desc())
            .limit(n_rows)
        )
    )
    if not dates:
        empty = pd.DataFrame(columns=symbols, dtype=float)
        return empty, empty.copy(), pd.Series(dtype=float)
    first = min(dates)
    rows = db.execute(
        select(Price.date, Price.ticker, Price.close, Price.volume).where(
            Price.ticker.in_(symbols), Price.date >= first, Price.date <= as_of
        )
    ).all()
    df = pd.DataFrame(rows, columns=["date", "ticker", "close", "volume"])
    prices = df.pivot(index="date", columns="ticker", values="close").sort_index()
    volume = df.pivot(index="date", columns="ticker", values="volume").sort_index().astype(float)
    for frame in (prices, volume):
        frame.index = pd.to_datetime(frame.index)
    vix_rows = db.execute(select(Vix.date, Vix.value).where(Vix.date >= first, Vix.date <= as_of)).all()
    vix = pd.Series(
        [v for _, v in vix_rows], index=pd.to_datetime([d for d, _ in vix_rows]), name="VIX", dtype=float
    ).sort_index()
    return prices, volume, vix
