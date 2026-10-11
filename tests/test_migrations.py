"""Migraciones de Alembic en SQLite: upgrade/downgrade reversibles (el CI repite lo esencial en Postgres)."""
from __future__ import annotations

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect

from app.config import get_settings
from app.db import make_engine


@pytest.fixture
def sqlite_url(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'mig.db'}"
    monkeypatch.setenv("DATABASE_URL", url)
    get_settings.cache_clear()
    yield url
    get_settings.cache_clear()


def _tables(url):
    engine = make_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def _predictions_shape(url):
    engine = make_engine(url)
    try:
        insp = inspect(engine)
        cols = [(c["name"], str(c["type"]), c["nullable"]) for c in insp.get_columns("predictions")]
        uniques = [(u["name"], u["column_names"]) for u in insp.get_unique_constraints("predictions")]
        return cols, uniques, [i["name"] for i in insp.get_indexes("predictions")]
    finally:
        engine.dispose()


def test_0002_drops_predictions_and_downgrade_recreates_it_identically(sqlite_url):
    cfg = Config("alembic.ini")
    command.upgrade(cfg, "0001")
    original = _predictions_shape(sqlite_url)
    command.upgrade(cfg, "head")
    assert _tables(sqlite_url) == {"tickers", "prices", "vix", "etl_runs", "alembic_version"}
    command.downgrade(cfg, "-1")
    assert "predictions" in _tables(sqlite_url)
    assert _predictions_shape(sqlite_url) == original  # misma definición que en 0001
    command.upgrade(cfg, "head")  # y vuelve a subir sin problema
    assert "predictions" not in _tables(sqlite_url)
    command.downgrade(cfg, "base")
    assert _tables(sqlite_url) <= {"alembic_version"}


def test_orm_metadata_matches_migrated_schema(sqlite_url):
    from app.db import Base

    command.upgrade(Config("alembic.ini"), "head")
    assert _tables(sqlite_url) - {"alembic_version"} == set(Base.metadata.tables)