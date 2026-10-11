"""Endpoints de datos de mercado: tickers, precios, métricas, ranking y sectores (solo lectura de la BD)."""
from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import Price, Ticker
from app.deps import get_db, valid_ticker
from app.errors import ERROR_RESPONSES, ApiError
from app.schemas import (
    MetricsResponse,
    PricePoint,
    PricesResponse,
    RankingResponse,
    SectorsSummaryResponse,
    TickerOut,
)
from app.services import metrics as M
from app.services import queries
from app.services.ranking import rank_tickers, sectors_summary

router = APIRouter()


def _check_range(start: date | None, end: date | None) -> None:
    if start and end and start > end:
        raise ApiError(422, "invalid_range", "'start' no puede ser posterior a 'end'.")


@router.get("/tickers", response_model=list[TickerOut], tags=["Mercado"], summary="Lista de tickers con su sector")
def get_tickers(db: Session = Depends(get_db)):
    return queries.list_tickers(db)


@router.get(
    "/prices/{symbol}",
    response_model=PricesResponse,
    responses={k: ERROR_RESPONSES[k] for k in (404, 422)},
    tags=["Mercado"],
    summary="Precios diarios ajustados de un ticker",
)
def get_prices(
    ticker: Ticker = Depends(valid_ticker),
    start: date | None = Query(None, description="Fecha inicial (YYYY-MM-DD), inclusive"),
    end: date | None = Query(None, description="Fecha final (YYYY-MM-DD), inclusive"),
    limit: int = Query(500, ge=1, le=5000, description="Máximo de filas por página"),
    offset: int = Query(0, ge=0),
    order: Literal["asc", "desc"] = Query("asc", description="Orden por fecha"),
    db: Session = Depends(get_db),
):
    _check_range(start, end)
    where = [Price.ticker == ticker.symbol]
    if start:
        where.append(Price.date >= start)
    if end:
        where.append(Price.date <= end)
    total = db.scalar(select(func.count()).select_from(Price).where(*where)) or 0
    sort = Price.date.asc() if order == "asc" else Price.date.desc()
    rows = db.scalars(select(Price).where(*where).order_by(sort).limit(limit).offset(offset)).all()
    return PricesResponse(
        ticker=ticker.symbol,
        start=start,
        end=end,
        total=total,
        limit=limit,
        offset=offset,
        items=[PricePoint(date=r.date, close=r.close, volume=r.volume) for r in rows],
    )


@router.get(
    "/metrics/{symbol}",
    response_model=MetricsResponse,
    responses={k: ERROR_RESPONSES[k] for k in (404, 422)},
    tags=["Mercado"],
    summary="Retorno total, volatilidad, max drawdown y Sharpe",
)
def get_metrics(
    ticker: Ticker = Depends(valid_ticker),
    start: date | None = Query(None, description="Por defecto, el inicio del historial"),
    end: date | None = Query(None, description="Por defecto, la última fecha con datos"),
    db: Session = Depends(get_db),
):
    _check_range(start, end)
    frame = queries.price_frame(db, [ticker.symbol], start, end)
    series = frame[ticker.symbol].dropna()
    if len(series) < 2:
        raise ApiError(422, "insufficient_data", "El rango no tiene al menos 2 precios para calcular métricas.")
    m = M.compute_metrics(series)
    return MetricsResponse(
        ticker=ticker.symbol,
        start=series.index[0].date(),
        end=series.index[-1].date(),
        n_days=m.n_days,
        total_return=m.total_return,
        volatility=m.volatility,
        max_drawdown=m.max_drawdown,
        sharpe=m.sharpe,
    )


@router.get(
    "/ranking",
    response_model=RankingResponse,
    responses={k: ERROR_RESPONSES[k] for k in (404, 422)},
    tags=["Mercado"],
    summary="Ranking de tickers (¿qué acción rindió mejor este mes?)",
    description=(
        "`period=month|quarter|ytd` calcula el acumulado del mes/trimestre/año en curso hasta la última fecha "
        "con datos (base = último cierre anterior al periodo). `period=custom` requiere `start` (y opcionalmente `end`). "
        "`order=desc` (por defecto) devuelve primero el valor más alto."
    ),
)
def get_ranking(
    period: Literal["month", "quarter", "ytd", "custom"] = Query("month"),
    metric: Literal["return", "volatility", "sharpe"] = Query("return"),
    order: Literal["desc", "asc"] = Query("desc"),
    start: date | None = None,
    end: date | None = None,
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    return rank_tickers(db, period, metric, order, start, end, limit)


@router.get(
    "/sectors/summary",
    response_model=SectorsSummaryResponse,
    responses={k: ERROR_RESPONSES[k] for k in (404, 422)},
    tags=["Mercado"],
    summary="Resumen por sector (cartera de igual ponderación)",
)
def get_sectors_summary(start: date | None = None, end: date | None = None, db: Session = Depends(get_db)):
    return sectors_summary(db, start, end)
