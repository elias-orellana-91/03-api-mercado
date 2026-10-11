"""Endpoints con el dataset sintético completo (16 tickers, 330 días)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from app.db import Prediction
from app.ml.universe import TICKER_TO_SECTOR, TICKERS


def test_health_ok(client, market):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["database"]["ok"] is True
    assert body["model"]["loaded"] and body["model"]["hash_ok"]
    assert body["etl"]["last_run"]["status"] == "success" and body["etl"]["last_run"]["mode"] == "seed"
    assert body["data"]["last_date"] == str(market.prices.index[-1].date())


def test_tickers(client):
    body = client.get("/tickers").json()
    assert {t["symbol"] for t in body} == set(TICKERS)
    assert all(t["sector"] == TICKER_TO_SECTOR[t["symbol"]] for t in body)


def test_prices_default_and_pagination(client, market):
    r = client.get("/prices/aapl", params={"limit": 5}).json()  # el símbolo no distingue mayúsculas
    assert r["ticker"] == "AAPL" and r["total"] == len(market.prices) and len(r["items"]) == 5
    assert r["items"][0]["close"] == pytest.approx(market.prices["AAPL"].iloc[0])
    page2 = client.get("/prices/AAPL", params={"limit": 5, "offset": 5}).json()["items"]
    assert page2[0]["date"] == str(market.prices.index[5].date())
    desc = client.get("/prices/AAPL", params={"limit": 1, "order": "desc"}).json()["items"][0]
    assert desc["date"] == str(market.prices.index[-1].date())


def test_prices_date_filter(client, market):
    s, e = market.prices.index[10], market.prices.index[20]
    r = client.get("/prices/MSFT", params={"start": str(s.date()), "end": str(e.date())}).json()
    assert r["total"] == 11 and r["items"][0]["date"] == str(s.date()) and r["items"][-1]["date"] == str(e.date())


@pytest.mark.parametrize(
    "path,params,status,code",
    [
        ("/prices/ZZZ", {}, 404, "ticker_not_found"),
        ("/prices/AAPL", {"start": "no-es-fecha"}, 422, "validation_error"),
        ("/prices/AAPL", {"start": "2025-02-01", "end": "2025-01-01"}, 422, "invalid_range"),
        ("/prices/AAPL", {"limit": 0}, 422, "validation_error"),
        ("/prices/AAPL", {"limit": 100000}, 422, "validation_error"),
        ("/prices/AAPL", {"offset": -1}, 422, "validation_error"),
        ("/prices/A%20B", {}, 422, "validation_error"),  # símbolo con formato inválido
        ("/metrics/ZZZ", {}, 404, "ticker_not_found"),
        ("/predictions/ZZZ", {}, 404, "ticker_not_found"),
        ("/ruta-inexistente", {}, 404, "not_found"),
    ],
)
def test_error_schema(client, path, params, status, code):
    r = client.get(path, params=params)
    assert r.status_code == status
    err = r.json()["error"]
    assert err["code"] == code and err["message"] and "request_id" in err
    if code == "validation_error":
        assert err["details"]


def test_metrics_full_range_matches_numpy(client, market):
    p = market.prices["NVDA"]
    r = p.pct_change().dropna()
    body = client.get("/metrics/NVDA").json()
    assert body["total_return"] == pytest.approx(p.iloc[-1] / p.iloc[0] - 1)
    assert body["volatility"] == pytest.approx(r.std() * np.sqrt(252))
    assert body["max_drawdown"] == pytest.approx((p / p.cummax() - 1).min())


def test_ranking_and_sectors_basic(client):
    body = client.get("/ranking", params={"period": "ytd", "limit": 3}).json()
    assert len(body["items"]) == 3
    values = [i["value"] for i in body["items"]]
    assert values == sorted(values, reverse=True)
    sectors = client.get("/sectors/summary").json()["sectors"]
    assert {s["sector"] for s in sectors} == set(TICKER_TO_SECTOR.values())


# ---------- predicciones ----------
def test_prediction_default_is_last_date(client, market):
    r = client.get("/predictions/AAPL")
    assert r.status_code == 200
    b = r.json()
    assert b["as_of_date"] == str(market.prices.index[-1].date())
    assert 0 <= b["probability_high_vol"] <= 1 and b["horizon_days"] == 5
    assert b["predicted_class"] == int(b["probability_high_vol"] >= b["decision_threshold"])
    assert b["label"] == ("alta" if b["predicted_class"] else "normal")


def test_prediction_get_does_not_write_to_db(client, seeded_factory, market):
    """Un GET anónimo solo calcula: la tabla predictions no cambia (ni con fechas distintas ni con /latest)."""

    def n_rows():
        with seeded_factory() as s:
            return s.scalar(select(func.count()).select_from(Prediction))

    before = n_rows()
    client.get("/predictions/JPM")
    for d in market.prices.index[-5:]:
        assert client.get("/predictions/JPM", params={"date": str(d.date())}).status_code == 200
    assert client.get("/predictions/latest").status_code == 200
    assert n_rows() == before == 0


def test_prediction_with_explicit_date(client, market):
    d = market.prices.index[-10]
    r = client.get("/predictions/KO", params={"date": str(d.date())})
    assert r.status_code == 200 and r.json()["as_of_date"] == str(d.date())


def test_prediction_matches_direct_pipeline(client, market, model_store):
    """La API debe dar lo mismo que build_features sobre todo el histórico + modelo."""
    from app.ml.features import build_features

    feats = build_features(market.prices, market.vix, market.volume)
    t = market.prices.index[-1]
    row = feats.xs(t, level="Date").loc[["WMT"]][model_store.metadata["features"]]
    expected = model_store.model.predict_proba(row)[0, 1]
    got = client.get("/predictions/WMT").json()["probability_high_vol"]
    assert got == pytest.approx(expected, abs=1e-9)


def test_prediction_errors(client, market):
    assert client.get("/predictions/AAPL", params={"date": "2030-01-01"}).json()["error"]["code"] == "no_data_for_date"
    early = str(market.prices.index[100].date())  # < 257 días de historial
    r = client.get("/predictions/AAPL", params={"date": early})
    assert r.status_code == 422 and r.json()["error"]["code"] == "insufficient_history"
    assert client.get("/predictions/AAPL", params={"date": "ayer"}).status_code == 422


def test_predictions_latest(client, market):
    b = client.get("/predictions/latest").json()
    assert b["as_of_date"] == str(market.prices.index[-1].date()) and b["count"] == len(TICKERS)
    assert {i["ticker"] for i in b["items"]} == set(TICKERS)
    one = client.get("/predictions/GS").json()["probability_high_vol"]
    assert {i["ticker"]: i["probability_high_vol"] for i in b["items"]}["GS"] == pytest.approx(one)


def test_model_info(client, model_store):
    b = client.get("/model/info").json()
    assert b["model_version"] == model_store.metadata["model_version"]
    assert b["sha256"] == model_store.sha256 and len(b["sha256"]) == 64
    assert b["features"] == model_store.metadata["features"]
    assert b["test_metrics"]["roc_auc"] == pytest.approx(0.646, abs=0.001)
    assert b["required_inputs"]["min_history_days"] == 257


def test_openapi_documents_error_schema_and_examples(client):
    spec = client.get("/openapi.json").json()
    assert "ErrorResponse" in spec["components"]["schemas"]
    assert "examples" in spec["components"]["schemas"]["PredictionOut"]
    assert "/admin/etl" in spec["paths"] and "/predictions/latest" in spec["paths"]


def test_pandas_dtypes_sanity(market):
    assert isinstance(market.prices, pd.DataFrame)


def test_huge_offset_is_422_not_500(client):
    for off in ("100000000000000000000", "1000001", "-1"):
        r = client.get("/prices/AAPL", params={"offset": off})
        assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert client.get("/prices/AAPL", params={"offset": 1_000_000}).status_code == 200


def test_unhandled_500_keeps_request_id(seeded_factory, model_store):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import forbidden_source_factory, make_settings

    app = create_app(make_settings(), seeded_factory, model_store, forbidden_source_factory)

    @app.get("/boom")
    def boom():
        raise ValueError("secreto interno")

    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.get("/boom", headers={"X-Request-ID": "req-500-abcdef"})
    assert r.status_code == 500 and "secreto" not in r.text
    assert r.json()["error"]["request_id"] == "req-500-abcdef" and r.headers["x-request-id"] == "req-500-abcdef"