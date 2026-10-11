"""Fixtures: SQLite en memoria + dataset sintético. Sin red ni Docker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db import Base, make_engine, make_session_factory
from app.etl import run_seed
from app.main import create_app
from app.ml.store import ModelStore
from app.ml.universe import TICKERS
from app.sources.base import MarketData

N_DAYS = 330  # > 257 (historial mínimo del modelo) + margen
ADMIN_KEY = "test-admin-key-not-a-secret"


class ForbiddenSource:
    """Fuente que falla si se usa: los endpoints de lectura jamás deben llamarla."""

    def fetch(self, *args, **kwargs):
        raise AssertionError("¡Se llamó a la fuente externa desde un endpoint!")


def forbidden_source_factory():
    return ForbiddenSource()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Cualquier intento de descargar de Yahoo durante los tests rompe el test."""
    import yfinance

    def boom(*a, **k):
        raise AssertionError("Acceso a la red (yfinance.download) en un test")

    monkeypatch.setattr(yfinance, "download", boom)


def synthetic_market(n_days: int = N_DAYS, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days, name="Date")
    rets = rng.normal(0.0004, 0.013, (n_days, len(TICKERS)))
    prices = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=TICKERS)
    volume = pd.DataFrame(rng.integers(1_000_000, 5_000_000, (n_days, len(TICKERS))).astype(float), index=idx, columns=TICKERS)
    vix = pd.Series(16 + 4 * np.abs(rng.normal(size=n_days)), index=idx, name="VIX")
    return MarketData(prices, volume, vix)


def write_seed(directory, market: MarketData) -> None:
    market.prices.to_parquet(directory / "prices.parquet")
    market.volume.to_parquet(directory / "volume.parquet")
    market.vix.to_frame().to_parquet(directory / "vix.parquet")


def new_factory():
    engine = make_engine("sqlite://")
    Base.metadata.create_all(engine)
    return make_session_factory(engine)


def make_settings(**kw) -> Settings:
    base = {"_env_file": None, "database_url": "sqlite://", "admin_api_key": ADMIN_KEY, "rate_limit_enabled": False}
    base.update(kw)
    return Settings(**base)


@pytest.fixture(scope="session")
def market() -> MarketData:
    return synthetic_market()


@pytest.fixture(scope="session")
def model_store() -> ModelStore:
    return ModelStore().load()


@pytest.fixture(scope="session")
def seeded_factory(tmp_path_factory, market):
    seed_dir = tmp_path_factory.mktemp("seed")
    write_seed(seed_dir, market)
    factory = new_factory()
    run_seed(factory, seed_dir)
    return factory


@pytest.fixture
def client(seeded_factory, model_store):
    app = create_app(make_settings(), seeded_factory, model_store, forbidden_source_factory)
    with TestClient(app) as c:
        yield c
