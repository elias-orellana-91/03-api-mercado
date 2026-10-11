"""Modelos Pydantic de respuesta (con ejemplos para Swagger)."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


def _ex(example: dict) -> ConfigDict:
    return ConfigDict(json_schema_extra={"examples": [example]})


class TickerOut(BaseModel):
    symbol: str
    name: str
    sector: str
    model_config = _ex({"symbol": "AAPL", "name": "Apple Inc.", "sector": "Tecnología"})


class PricePoint(BaseModel):
    date: date
    close: float = Field(description="Cierre ajustado por dividendos y splits (USD)")
    volume: int | None = None


class PricesResponse(BaseModel):
    ticker: str
    start: date | None
    end: date | None
    total: int = Field(description="Filas que cumplen el filtro (antes de paginar)")
    limit: int
    offset: int
    items: list[PricePoint]
    model_config = _ex(
        {
            "ticker": "AAPL", "start": "2025-01-02", "end": "2025-01-06", "total": 2, "limit": 500, "offset": 0,
            "items": [
                {"date": "2025-01-02", "close": 243.85, "volume": 55740700},
                {"date": "2025-01-03", "close": 243.36, "volume": 40244100},
            ],
        }
    )


class MetricsResponse(BaseModel):
    ticker: str
    start: date
    end: date
    n_days: int
    total_return: float = Field(description="Retorno total del periodo (0.10 = +10 %)")
    volatility: float = Field(description="Volatilidad anualizada (desv. estándar diaria × √252)")
    max_drawdown: float = Field(description="Máxima caída desde un máximo previo (negativa)")
    sharpe: float | None = Field(description="Sharpe anualizado con tasa libre de riesgo 0; null si no definido")
    model_config = _ex(
        {"ticker": "AAPL", "start": "2025-01-02", "end": "2025-12-30", "n_days": 250, "total_return": 0.0912,
         "volatility": 0.318, "max_drawdown": -0.3057, "sharpe": 0.41}
    )


class RankingItem(BaseModel):
    rank: int
    ticker: str
    name: str
    sector: str
    value: float | None


class RankingResponse(BaseModel):
    period: Literal["month", "quarter", "ytd", "custom"]
    metric: Literal["return", "volatility", "sharpe"]
    order: Literal["desc", "asc"]
    start: date = Field(description="Primera fecha de precio usada (base del cálculo)")
    end: date
    best: RankingItem | None
    items: list[RankingItem]
    model_config = _ex(
        {
            "period": "month", "metric": "return", "order": "desc", "start": "2025-11-28", "end": "2025-12-30",
            "best": {"rank": 1, "ticker": "GS", "name": "The Goldman Sachs Group, Inc.", "sector": "Bancos", "value": 0.071},
            "items": [{"rank": 1, "ticker": "GS", "name": "The Goldman Sachs Group, Inc.", "sector": "Bancos", "value": 0.071}],
        }
    )


class SectorSummary(BaseModel):
    sector: str
    tickers: list[str]
    total_return: float = Field(description="Cartera de igual ponderación del sector")
    volatility: float
    max_drawdown: float
    sharpe: float | None
    best_ticker: str
    worst_ticker: str


class SectorsSummaryResponse(BaseModel):
    start: date
    end: date
    sectors: list[SectorSummary]


class PredictionOut(BaseModel):
    ticker: str
    as_of_date: date = Field(description="Fecha de cierre usada para calcular las features")
    horizon_days: int = 5
    probability_high_vol: float = Field(description="Probabilidad de volatilidad alta en los próximos 5 días hábiles")
    predicted_class: int = Field(description="1 = volatilidad alta, 0 = normal (probabilidad >= umbral)")
    label: Literal["alta", "normal"]
    decision_threshold: float
    model_version: str
    created_at: datetime
    model_config = _ex(
        {"ticker": "NVDA", "as_of_date": "2025-12-30", "horizon_days": 5, "probability_high_vol": 0.62,
         "predicted_class": 1, "label": "alta", "decision_threshold": 0.5, "model_version": "0.1.0",
         "created_at": "2026-01-02T10:00:00Z"}
    )


class PredictionsLatestResponse(BaseModel):
    as_of_date: date
    count: int
    items: list[PredictionOut]


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    trained_at: str
    trained_on: str
    sha256: str
    decision_threshold: float
    features: list[str]
    test_metrics: dict
    baselines_test_metrics: dict
    required_inputs: dict
    library_versions: dict
    warnings: list[str]


class ModelStatus(BaseModel):
    loaded: bool
    hash_ok: bool
    sha256: str | None
    version: str | None
    error: str | None
    warnings: list[str]


class EtlRunOut(BaseModel):
    id: int
    mode: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    rows: int
    error: str | None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    database: dict
    model: ModelStatus
    etl: dict
    data: dict
    model_config = _ex(
        {
            "status": "ok",
            "database": {"ok": True},
            "model": {"loaded": True, "hash_ok": True, "sha256": "ab12…", "version": "0.1.0", "error": None, "warnings": []},
            "etl": {"last_run": {"id": 1, "mode": "seed", "status": "success", "rows": 77000}},
            "data": {"last_date": "2025-12-30"},
        }
    )


class EtlTriggerResponse(BaseModel):
    status: Literal["accepted"]
    message: str
