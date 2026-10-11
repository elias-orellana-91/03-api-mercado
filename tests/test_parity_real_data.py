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

def test_in_sample_flag_on_real_data(real_client, model_store):
    rng = model_store.metadata["trained_on_range"]  # el límite sale del metadata, no del código
    assert rng["end"] == "2022-12-22"
    inside = real_client.get("/predictions/AAPL", params={"date": "2020-03-16"}).json()
    assert inside["in_sample"] is True
    last_train = real_client.get("/predictions/AAPL", params={"date": rng["end"]}).json()
    assert last_train["in_sample"] is True
    after = real_client.get("/predictions/AAPL", params={"date": "2024-06-14"}).json()
    assert after["in_sample"] is False
    latest = real_client.get("/predictions/latest").json()
    assert latest["items"] and all(i["in_sample"] is False for i in latest["items"])


def test_is_in_sample_boundaries():
    from datetime import date

    from app.services.predictor import is_in_sample

    md = {"trained_on_range": {"start": "2016-01-08", "end": "2022-12-22"}}
    assert is_in_sample(md, date(2016, 1, 8)) and is_in_sample(md, date(2022, 12, 22))
    assert not is_in_sample(md, date(2016, 1, 7)) and not is_in_sample(md, date(2022, 12, 23))


def test_in_sample_in_openapi_schema(real_client):
    spec = real_client.get("/openapi.json").json()
    prop = spec["components"]["schemas"]["PredictionOut"]["properties"]["in_sample"]
    assert prop["type"] == "boolean" and "entrenamiento" in prop["description"]
    assert "in_sample" in spec["components"]["schemas"]["PredictionOut"]["required"]