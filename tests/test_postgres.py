"""Integración con PostgreSQL real. Se salta si no hay TEST_POSTGRES_URL (CI la define con un service container)."""
from __future__ import annotations

import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import func, select, text

from app import etl
from app.db import Price, Ticker, make_engine, make_session_factory
from tests.conftest import synthetic_market, write_seed

URL = os.environ.get("TEST_POSTGRES_URL", "")
pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not URL.startswith(("postgres", "postgresql")), reason="TEST_POSTGRES_URL no definida"),
]


@pytest.fixture
def pg(monkeypatch):
    from app.config import Settings, get_settings

    url = Settings(_env_file=None, database_url=URL).database_url  # normaliza al driver psycopg 3
    engine = make_engine(url)
    with engine.begin() as c:
        c.execute(text("DROP SCHEMA public CASCADE"))
        c.execute(text("CREATE SCHEMA public"))
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    yield engine
    get_settings.cache_clear()
    engine.dispose()


def test_migrations_and_idempotent_etl_on_postgres(pg, tmp_path):
    command.upgrade(Config("alembic.ini"), "head")
    write_seed(tmp_path, synthetic_market(n_days=40))
    factory = make_session_factory(pg)
    etl.run_seed(factory, tmp_path)
    etl.run_seed(factory, tmp_path)
    with factory() as s:
        assert s.scalar(select(func.count()).select_from(Ticker)) == 16
        assert s.scalar(select(func.count()).select_from(Price)) == 16 * 40
    command.downgrade(Config("alembic.ini"), "base")