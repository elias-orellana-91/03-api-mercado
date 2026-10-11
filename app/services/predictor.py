"""Inferencia: features (copia de P2) sobre el historial leído de NUESTRA base de datos."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from app.errors import ApiError
from app.ml.features import build_features
from app.ml.store import ModelStore
from app.services.queries import last_rows_frames

HISTORY_MARGIN = 20  # filas extra sobre el mínimo (las features solo usan ventanas hacia atrás)


@dataclass
class PredictionResult:
    ticker: str
    as_of_date: date
    probability: float
    predicted_class: int
    model_version: str
    decision_threshold: float


def require_model(store: ModelStore) -> None:
    if not store.loaded:
        raise ApiError(503, "model_unavailable", store.error or "El modelo no está cargado.")


def predict(db: Session, store: ModelStore, symbols: list[str], target: date) -> list[PredictionResult]:
    """Predice para ``symbols`` en la fecha ``target`` (debe tener datos). Calcula con una sola pasada."""
    require_model(store)
    ri = store.metadata["required_inputs"]
    market, min_days = ri["market_ticker"], int(ri["min_history_days"])
    needed = sorted(set(symbols) | {market})
    prices, volume, vix = last_rows_frames(db, needed, target, min_days + HISTORY_MARGIN)
    if len(prices) < min_days:
        raise ApiError(
            422,
            "insufficient_history",
            f"Se necesitan al menos {min_days} días de historial hasta {target}; hay {len(prices)}.",
        )
    if vix.empty:
        raise ApiError(422, "features_unavailable", "No hay datos de VIX para el periodo requerido.")

    feats = build_features(prices, vix, volume, market=market)
    ts = pd.Timestamp(target)
    if ts not in feats.index.get_level_values("Date"):
        raise ApiError(404, "no_data_for_date", f"No hay datos de precios para {target}.")
    row = feats.xs(ts, level="Date").reindex(symbols)[store.metadata["features"]]
    ok = row.notna().all(axis=1)
    if not ok.any():
        raise ApiError(422, "features_unavailable", f"No se pudieron calcular las features para {target}.")
    row = row[ok]
    proba = store.model.predict_proba(row)[:, 1]
    version = store.metadata["model_version"]
    thr = float(store.metadata["decision_threshold"])
    return [
        PredictionResult(t, target, float(p), int(p >= thr), version, thr)
        for t, p in zip(row.index, proba, strict=True)
    ]
