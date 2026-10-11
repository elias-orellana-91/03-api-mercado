"""Predicciones de volatilidad y metadatos del modelo."""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import Price, Ticker, utcnow
from app.deps import get_db, get_store, is_admin, valid_ticker
from app.errors import ERROR_RESPONSES, ApiError
from app.ml.store import ModelStore
from app.schemas import ModelInfo, PredictionOut, PredictionsLatestResponse
from app.services import predictor

router = APIRouter(tags=["Modelo"])
RESP = {k: ERROR_RESPONSES[k] for k in (404, 422, 503)}


def _out(r: predictor.PredictionResult, created_at) -> PredictionOut:
    return PredictionOut(
        ticker=r.ticker,
        as_of_date=r.as_of_date,
        probability_high_vol=r.probability,
        predicted_class=r.predicted_class,
        label="alta" if r.predicted_class else "normal",
        decision_threshold=r.decision_threshold,
        in_sample=r.in_sample,
        model_version=r.model_version,
        created_at=created_at,
    )


@router.get(
    "/predictions/latest",
    response_model=PredictionsLatestResponse,
    responses=RESP,
    summary="Predicción para todos los tickers en la última fecha",
)
def predictions_latest(db: Session = Depends(get_db), store: ModelStore = Depends(get_store)):
    predictor.require_model(store)
    last = db.scalar(select(func.max(Price.date)))
    if last is None:
        raise ApiError(404, "no_data", "No hay datos de precios cargados.")
    symbols = list(db.scalars(select(Price.ticker).where(Price.date == last).order_by(Price.ticker)))
    results = predictor.predict(db, store, symbols, last)
    now = utcnow()  # los GET solo calculan: no se escribe en la BD
    return PredictionsLatestResponse(as_of_date=last, count=len(results), items=[_out(r, now) for r in results])


@router.get(
    "/predictions/{symbol}",
    response_model=PredictionOut,
    responses=RESP,
    summary="Probabilidad de volatilidad alta en los próximos 5 días hábiles",
    description="Por defecto usa la última fecha con datos del ticker. La predicción se calcula al vuelo; este GET no escribe en la base de datos (`created_at` es el momento del cálculo). `in_sample=true` indica que la fecha cae dentro del periodo de entrenamiento del modelo (ya la vio al entrenar).",
)
def prediction_for_ticker(
    ticker: Ticker = Depends(valid_ticker),
    date_: date | None = Query(None, alias="date", description="Fecha de cierre (YYYY-MM-DD); por defecto la última"),
    db: Session = Depends(get_db),
    store: ModelStore = Depends(get_store),
):
    predictor.require_model(store)
    if date_ is None:
        date_ = db.scalar(select(func.max(Price.date)).where(Price.ticker == ticker.symbol))
        if date_ is None:
            raise ApiError(404, "no_data", f"No hay precios cargados para {ticker.symbol}.")
    elif db.get(Price, (ticker.symbol, date_)) is None:
        raise ApiError(404, "no_data_for_date", f"{ticker.symbol} no tiene datos el {date_}.")
    results = predictor.predict(db, store, [ticker.symbol], date_)
    if not results:
        raise ApiError(422, "features_unavailable", "No se pudo calcular la predicción para esa fecha.")
    return _out(results[0], utcnow())  # solo calcula: este GET no escribe en la BD


@router.get(
    "/model/info",
    response_model=ModelInfo,
    responses={503: ERROR_RESPONSES[503]},
    summary="Versión, entrenamiento, métricas de test, features y hash del modelo",
    description="Las versiones exactas de las librerías y los avisos del modelo solo se muestran con `X-API-Key`.",
    response_model_exclude_unset=True,
)
def model_info(store: ModelStore = Depends(get_store), admin: bool = Depends(is_admin)):
    predictor.require_model(store)
    md = store.metadata
    extra = {"library_versions": md.get("library_versions", {}), "warnings": store.version_warnings} if admin else {}
    return ModelInfo(
        model_name=md["model_name"],
        model_version=md["model_version"],
        trained_at=md["trained_at"],
        trained_on=md["trained_on"],
        sha256=store.sha256 or "",
        decision_threshold=md["decision_threshold"],
        features=md["features"],
        test_metrics=md["test_metrics"],
        baselines_test_metrics=md.get("baselines_test_metrics", {}),
        required_inputs=md["required_inputs"],
        **extra,
    )
