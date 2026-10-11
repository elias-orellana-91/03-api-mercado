"""Paridad con el proyecto 2 usando los datos semilla reales (data/seed): la API no altera la inferencia."""
from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.etl import SEED_DIR, load_seed, run_seed
from app.main import create_app
from app.ml.features import build_features
from tests.conftest import forbidden_source_factory, make_settings, new_factory


@pytest.fixture(scope="module")
def real_client(model_store):
    factory = new_factory()
    run_seed(factory, SEED_DIR)
    with TestClient(create_app(make_settings(), factory, model_store, forbidden_source_factory)) as c:
        yield c


def test_api_predictions_equal_direct_pipeline_on_real_data(real_client, model_store):
    data = load_seed()
    feats = build_features(data.prices, data.vix, data.volume)
    t = data.prices.index[-1]
    X = feats.xs(t, level="Date")[model_store.metadata["features"]]
    expected = pd.Series(model_store.model.predict_proba(X)[:, 1], index=X.index)
    body = real_client.get("/predictions/latest").json()
    got = {i["ticker"]: i["probability_high_vol"] for i in body["items"]}
    assert body["as_of_date"] == str(t.date()) and set(got) == set(expected.index)
    for ticker, p in expected.items():
        assert got[ticker] == pytest.approx(p, abs=1e-9)


def test_historical_date_prediction_equals_direct_pipeline(real_client, model_store):
    data = load_seed()
    feats = build_features(data.prices, data.vix, data.volume)
    t = pd.Timestamp("2024-06-14")
    expected = model_store.model.predict_proba(feats.xs(t, level="Date").loc[["NVDA"]][model_store.metadata["features"]])[0, 1]
    got = real_client.get("/predictions/NVDA", params={"date": "2024-06-14"}).json()["probability_high_vol"]
    assert got == pytest.approx(expected, abs=1e-9)


def test_real_data_ranking_smoke(real_client):
    body = real_client.get("/ranking", params={"period": "ytd"}).json()
    assert body["end"] == "2025-12-30" and len(body["items"]) == 16