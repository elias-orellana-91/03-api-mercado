"""Métricas y ranking contra cálculos hechos a mano (dataset mínimo determinista)."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.db import Price, Ticker
from app.main import create_app
from app.services import metrics as M
from tests.conftest import forbidden_source_factory, make_settings, new_factory

DATES = [date(2025, 11, 27), date(2025, 11, 28), date(2025, 12, 1), date(2025, 12, 2), date(2025, 12, 3), date(2025, 12, 4)]
SERIES = {
    "AAPL": [100, 100, 110, 121, 121, 110],
    "MSFT": [50, 50, 55, 60, 66, 60],
    "XOM": [80, 80, 78, 76, 75.2, 76],
}
SECTOR = {"AAPL": "Tecnología", "MSFT": "Tecnología", "XOM": "Energía"}


@pytest.fixture(scope="module")
def mini(model_store):
    factory = new_factory()
    with factory() as s:
        for sym, closes in SERIES.items():
            s.add(Ticker(symbol=sym, name=sym, sector=SECTOR[sym]))
            s.flush()
            s.add_all(Price(ticker=sym, date=d, close=float(c), volume=1000) for d, c in zip(DATES, closes, strict=True))
        s.commit()
    with TestClient(create_app(make_settings(), factory, model_store, forbidden_source_factory)) as c:
        yield c


# ---------- funciones puras ----------
def test_pure_metrics_by_hand():
    p = pd.Series([100.0, 110.0, 99.0, 99.0, 120.0])
    assert M.total_return(p) == pytest.approx(0.20)
    assert M.max_drawdown(p) == pytest.approx(99 / 110 - 1)
    r = p.pct_change().dropna()
    assert M.annualized_volatility(r) == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert M.sharpe_ratio(r) == pytest.approx(r.mean() * 252 / (r.std(ddof=1) * np.sqrt(252)))


def test_sharpe_none_when_no_volatility():
    assert M.sharpe_ratio(pd.Series([0.01] * 10)) is None
    assert M.sharpe_ratio(pd.Series([0.0] * 10)) is None


# ---------- /metrics ----------
def test_metrics_endpoint_matches_hand_calculation(mini):
    p = pd.Series(SERIES["AAPL"], dtype=float)
    r = p.pct_change().dropna()
    body = mini.get("/metrics/AAPL").json()
    assert body["n_days"] == 6 and body["start"] == "2025-11-27" and body["end"] == "2025-12-04"
    assert body["total_return"] == pytest.approx(0.10)
    assert body["max_drawdown"] == pytest.approx(110 / 121 - 1)
    assert body["volatility"] == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert body["sharpe"] == pytest.approx(r.mean() * 252 / (r.std(ddof=1) * np.sqrt(252)))


def test_metrics_date_range_and_errors(mini):
    body = mini.get("/metrics/MSFT", params={"start": "2025-12-01", "end": "2025-12-03"}).json()
    assert body["total_return"] == pytest.approx(66 / 55 - 1)
    assert mini.get("/metrics/MSFT", params={"start": "2026-01-01"}).status_code == 422  # sin datos
    assert mini.get("/metrics/MSFT", params={"start": "2025-12-03", "end": "2025-12-01"}).json()["error"]["code"] == "invalid_range"


# ---------- /ranking ----------
def test_ranking_month_return(mini):
    body = mini.get("/ranking", params={"period": "month", "metric": "return"}).json()
    assert body["start"] == "2025-11-28" and body["end"] == "2025-12-04"  # base = último cierre de noviembre
    assert [i["ticker"] for i in body["items"]] == ["MSFT", "AAPL", "XOM"]
    assert [round(i["value"], 6) for i in body["items"]] == [round(60 / 50 - 1, 6), round(110 / 100 - 1, 6), round(76 / 80 - 1, 6)]
    assert body["best"]["ticker"] == "MSFT" and body["items"][0]["rank"] == 1


def test_ranking_asc_and_limit(mini):
    body = mini.get("/ranking", params={"period": "month", "order": "asc", "limit": 2}).json()
    assert [i["ticker"] for i in body["items"]] == ["XOM", "AAPL"]


def test_ranking_volatility_and_sharpe_by_hand(mini):
    window = {s: pd.Series(v[1:], dtype=float) for s, v in SERIES.items()}  # base 28-nov .. 4-dic
    vols = {s: p.pct_change().dropna().std(ddof=1) * np.sqrt(252) for s, p in window.items()}
    body = mini.get("/ranking", params={"period": "month", "metric": "volatility"}).json()
    got = {i["ticker"]: i["value"] for i in body["items"]}
    for s in vols:
        assert got[s] == pytest.approx(vols[s])
    assert [i["ticker"] for i in body["items"]] == sorted(vols, key=lambda s: -vols[s])
    sharpe = mini.get("/ranking", params={"period": "month", "metric": "sharpe"}).json()["items"]
    r = window["AAPL"].pct_change().dropna()
    assert {i["ticker"]: i["value"] for i in sharpe}["AAPL"] == pytest.approx(r.mean() * 252 / (r.std(ddof=1) * np.sqrt(252)))


def test_ranking_ytd_and_quarter_use_first_available_base(mini):
    for period in ("ytd", "quarter"):
        body = mini.get("/ranking", params={"period": period}).json()
        assert body["start"] == "2025-11-27"  # no hay cierre anterior al periodo: se usa el primer dato
        assert body["items"][0]["ticker"] == "MSFT"


def test_ranking_custom(mini):
    body = mini.get("/ranking", params={"period": "custom", "start": "2025-12-01", "end": "2025-12-03"}).json()
    got = {i["ticker"]: i["value"] for i in body["items"]}
    assert got["AAPL"] == pytest.approx(121 / 110 - 1)
    assert got["XOM"] == pytest.approx(75.2 / 78 - 1)


@pytest.mark.parametrize(
    "params,code",
    [
        ({"period": "custom"}, "missing_start"),
        ({"period": "custom", "start": "2025-12-04", "end": "2025-12-01"}, "invalid_range"),
        ({"period": "month", "start": "2025-12-01"}, "unexpected_dates"),
        ({"period": "custom", "start": "2030-01-01", "end": "2030-02-01"}, "insufficient_data"),
    ],
)
def test_ranking_errors(mini, params, code):
    r = mini.get("/ranking", params=params)
    assert r.status_code == 422 and r.json()["error"]["code"] == code


def test_ranking_invalid_enum_is_422(mini):
    r = mini.get("/ranking", params={"period": "decade"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"


# ---------- /sectors/summary ----------
def test_sectors_summary_by_hand(mini):
    body = mini.get("/sectors/summary").json()
    by = {s["sector"]: s for s in body["sectors"]}
    tech = pd.DataFrame({k: SERIES[k] for k in ("AAPL", "MSFT")}, dtype=float).pct_change().iloc[1:].mean(axis=1)
    idx = np.concatenate([[1.0], (1 + tech).cumprod().to_numpy()])
    assert by["Tecnología"]["total_return"] == pytest.approx(idx[-1] - 1)
    assert by["Tecnología"]["max_drawdown"] == pytest.approx((idx / np.maximum.accumulate(idx) - 1).min())
    assert by["Tecnología"]["best_ticker"] == "MSFT" and by["Tecnología"]["worst_ticker"] == "AAPL"
    assert by["Energía"]["total_return"] == pytest.approx(76 / 80 - 1)
    assert by["Energía"]["tickers"] == ["XOM"]
