"""ETL: idempotencia, atomicidad, modo incremental y errores de la fuente."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import func, select

from app import etl
from app.db import EtlRun, Price, Ticker, Vix
from app.ml.universe import TICKERS
from app.sources.base import MarketData
from tests.conftest import new_factory, synthetic_market, write_seed


def counts(factory):
    with factory() as s:
        return tuple(s.scalar(select(func.count()).select_from(m)) for m in (Ticker, Price, Vix, EtlRun))


class FakeSource:
    """Devuelve el tramo pedido de un MarketData sintético y registra las llamadas."""

    def __init__(self, market: MarketData):
        self.market, self.calls = market, []

    def fetch(self, tickers, start, end):
        self.calls.append((list(tickers), start, end))
        m = self.market
        keep = (m.prices.index >= pd.Timestamp(start)) & (m.prices.index < pd.Timestamp(end))
        return MarketData(m.prices[keep], m.volume[keep], m.vix[keep])


@pytest.fixture
def seed_dir(tmp_path):
    write_seed(tmp_path, synthetic_market(n_days=60))
    return tmp_path


def test_seed_is_idempotent(seed_dir):
    f = new_factory()
    etl.run_seed(f, seed_dir)
    first = counts(f)
    with f() as s:
        snapshot = s.execute(select(Price.ticker, Price.date, Price.close, Price.volume).order_by(Price.ticker, Price.date)).all()
    etl.run_seed(f, seed_dir)
    second = counts(f)
    assert first[:3] == (16, 60 * 16, 60) and second[:3] == first[:3]  # mismos datos
    assert second[3] == 2  # pero dos corridas registradas
    with f() as s:
        again = s.execute(select(Price.ticker, Price.date, Price.close, Price.volume).order_by(Price.ticker, Price.date)).all()
    assert again == snapshot
    with f() as s:
        runs = s.scalars(select(EtlRun)).all()
    assert all(r.status == "success" and r.finished_at and r.rows > 0 and r.mode == "seed" for r in runs)


def test_upsert_updates_changed_values(seed_dir):
    f = new_factory()
    etl.run_seed(f, seed_dir)
    with f() as s, s.begin():
        row = s.get(Price, ("AAPL", pd.Timestamp("2024-01-02").date()))
        row.close = -1.0
    etl.run_seed(f, seed_dir)  # el upsert corrige el valor alterado
    with f() as s:
        assert s.get(Price, ("AAPL", pd.Timestamp("2024-01-02").date())).close > 0


def test_failed_seed_rolls_back_and_records_error(seed_dir, monkeypatch):
    f = new_factory()

    def boom(*a, **k):
        raise RuntimeError("fallo a mitad de la carga")

    monkeypatch.setattr(etl, "upsert_vix", boom)  # falla DESPUÉS de insertar tickers y precios
    with pytest.raises(RuntimeError):
        etl.run_seed(f, seed_dir)
    t, p, v, r = counts(f)
    assert (t, p, v) == (0, 0, 0)  # nada a medias
    with f() as s:
        run = s.scalars(select(EtlRun)).one()
    assert run.status == "failed" and "fallo a mitad" in run.error and run.finished_at is not None


def test_missing_seed_files_are_recorded(tmp_path):
    f = new_factory()
    with pytest.raises(FileNotFoundError):
        etl.run_seed(f, tmp_path)
    with f() as s:
        assert s.scalars(select(EtlRun)).one().status == "failed"


def test_incremental_fetches_only_missing_dates(seed_dir):
    full = synthetic_market(n_days=75)
    f = new_factory()
    etl.run_seed(f, seed_dir)  # primeros 60 días
    with f() as s:
        last = s.scalar(select(func.max(Price.date)))
    src = FakeSource(full)
    etl.run_incremental(f, src, today=full.prices.index[-1].date())
    assert src.calls[0][0] == TICKERS
    assert src.calls[0][1] == last - timedelta(days=etl.OVERLAP_DAYS)  # solo desde lo que falta (+ solape)
    t, p, v, r = counts(f)
    assert p == 75 * 16 and v == 75 and r == 2
    etl.run_incremental(f, src, today=full.prices.index[-1].date())  # idempotente
    assert counts(f)[:3] == (16, 75 * 16, 75)


def test_incremental_without_new_data_is_ok(seed_dir):
    market = synthetic_market(n_days=60)
    f = new_factory()
    etl.run_seed(f, seed_dir)
    with f() as s:
        last = s.scalar(select(func.max(Price.date)))
    before = counts(f)[:3]
    etl.run_incremental(f, FakeSource(market), today=last)
    assert counts(f)[:3] == before


def test_incremental_source_error_is_recorded_and_leaves_data(seed_dir):
    f = new_factory()
    etl.run_seed(f, seed_dir)
    before = counts(f)[:3]

    class Broken:
        def fetch(self, *a, **k):
            raise ConnectionError("Yahoo no responde")

    with pytest.raises(ConnectionError):
        etl.run_incremental(f, Broken(), today=date(2030, 1, 1))
    assert counts(f)[:3] == before
    with f() as s:
        last = s.scalars(select(EtlRun).order_by(EtlRun.id.desc())).first()
    assert last.status == "failed" and "Yahoo no responde" in last.error


def test_incremental_on_empty_db_starts_from_default(seed_dir):
    f = new_factory()
    src = FakeSource(synthetic_market(n_days=10))
    etl.run_incremental(f, src, today=date(2024, 1, 20))
    assert src.calls[0][1] == date(2015, 1, 1)
    assert counts(f)[1] == 10 * 16


def test_cli_seed_loads_real_seed_files_without_network(tmp_path, monkeypatch):
    """`python -m app.etl --seed` con los parquet reales de data/seed (sin red)."""
    from app.config import get_settings
    from app.db import Base, make_engine

    url = f"sqlite:///{tmp_path / 'cli.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        Base.metadata.create_all(make_engine(url))
        assert etl.main(["--seed"]) == 0
        f = make_session_factory_for(url)
        t, p, v, r = counts(f)
        assert (t, p, v, r) == (16, 16 * 2765, 2765, 1)
    finally:
        get_settings.cache_clear()


def make_session_factory_for(url):
    from app.db import make_engine, make_session_factory

    return make_session_factory(make_engine(url))
def test_cli_seed_if_empty_is_noop_when_data_exists(tmp_path, monkeypatch):
    from app.config import get_settings
    from app.db import Base, make_engine, make_session_factory

    url = f"sqlite:///{tmp_path / 'e.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    try:
        Base.metadata.create_all(make_engine(url))
        assert etl.main(["--seed", "--if-empty"]) == 0  # vacía: carga
        f = make_session_factory(make_engine(url))
        assert counts(f)[3] == 1
        assert etl.main(["--seed", "--if-empty"]) == 0  # ya hay datos: no hace nada
        assert counts(f)[3] == 1
    finally:
        get_settings.cache_clear()

def _empty_market() -> MarketData:
    empty = pd.DataFrame(columns=TICKERS, dtype=float)
    return MarketData(empty, empty.copy(), pd.Series(dtype=float, name="VIX"))


class StaticSource:
    def __init__(self, data: MarketData):
        self.data = data

    def fetch(self, *a, **k):
        return self.data


def _assert_failed_without_writes(f, expected_msg):
    with f() as s:
        run = s.scalars(select(EtlRun).order_by(EtlRun.id.desc())).first()
    assert run.status == "failed" and expected_msg in run.error and run.finished_at is not None
    assert counts(f)[:3] == (0, 0, 0)  # nada escrito (ni siquiera los tickers)


def test_incremental_empty_source_is_failed_not_success():
    f = new_factory()
    with pytest.raises(RuntimeError, match="no devolvió datos"):
        etl.run_incremental(f, StaticSource(_empty_market()), today=date(2024, 6, 1))
    _assert_failed_without_writes(f, "La fuente no devolvió datos")


def test_incremental_empty_vix_is_failed():
    m = synthetic_market(n_days=10)
    f = new_factory()
    with pytest.raises(RuntimeError, match="no devolvió datos"):
        etl.run_incremental(f, StaticSource(MarketData(m.prices, m.volume, m.vix.iloc[0:0])), today=date(2024, 6, 1))
    _assert_failed_without_writes(f, "La fuente no devolvió datos")


def test_incremental_missing_tickers_is_failed():
    m = synthetic_market(n_days=10)
    f = new_factory()
    partial = MarketData(m.prices.drop(columns=["AAPL"]), m.volume, m.vix)
    all_nan = m.prices.copy()
    all_nan["MSFT"] = float("nan")
    for prices, missing in ((partial, "AAPL"), (MarketData(all_nan, m.volume, m.vix), "MSFT")):
        data = prices if isinstance(prices, MarketData) else MarketData(prices, m.volume, m.vix)
        with pytest.raises(RuntimeError, match="Faltan tickers") as exc:
            etl.run_incremental(f, StaticSource(data), today=date(2024, 6, 1))
        assert missing in str(exc.value)
    assert counts(f)[:3] == (0, 0, 0)


def test_health_reflects_failed_etl_when_source_is_empty(model_store):
    from fastapi.testclient import TestClient

    from app.main import create_app
    from tests.conftest import ADMIN_KEY, make_settings

    f = new_factory()
    app = create_app(make_settings(), f, model_store, lambda: StaticSource(_empty_market()))
    with TestClient(app) as c:
        assert c.post("/admin/etl", headers={"X-API-Key": ADMIN_KEY}).status_code == 202
        last = c.get("/health", headers={"X-API-Key": ADMIN_KEY}).json()["etl"]["last_run"]
    assert last["status"] == "failed" and "La fuente no devolvió datos" in last["error"]