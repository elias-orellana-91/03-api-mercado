"""ETL: carga precios, volumen y VIX en la base de datos propia.

Uso::

    python -m app.etl --seed     # desde data/seed/*.parquet (sin red)
    python -m app.etl            # incremental: solo las fechas que faltan, desde la fuente (Yahoo)

* Idempotente: todo se escribe con UPSERT; correr dos veces deja los mismos datos.
* Atómico: las escrituras de una corrida van en una sola transacción; si algo falla se hace
  rollback (sin datos a medias) y la corrida queda registrada como ``failed`` en ``etl_runs``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.db import EtlRun, Price, Ticker, Vix, make_engine, make_session_factory, utcnow
from app.ml.universe import START_DATE, TICKER_NAMES, TICKER_TO_SECTOR, TICKERS
from app.sources.base import MarketData, PriceSource

log = logging.getLogger("app.etl")
SEED_DIR = Path(__file__).resolve().parents[1] / "data" / "seed"
OVERLAP_DAYS = 7  # el incremental re-pide unos días: el upsert corrige revisiones recientes
CHUNK = 5000


# ------------------------------------------------------------------ upserts
def _insert(session: Session, table):
    name = session.get_bind().dialect.name
    if name == "postgresql":
        return postgresql.insert(table)
    if name == "sqlite":
        return sqlite.insert(table)
    raise NotImplementedError(f"Dialecto no soportado para upsert: {name}")


def _upsert(session: Session, model, rows: list[dict], keys: list[str]) -> int:
    if not rows:
        return 0
    table = model.__table__
    for i in range(0, len(rows), CHUNK):
        stmt = _insert(session, table)
        update = {c.name: stmt.excluded[c.name] for c in table.columns if c.name not in keys}
        session.execute(stmt.on_conflict_do_update(index_elements=keys, set_=update), rows[i : i + CHUNK])
    return len(rows)


def upsert_tickers(session: Session) -> int:
    rows = [{"symbol": t, "name": TICKER_NAMES[t], "sector": TICKER_TO_SECTOR[t]} for t in TICKERS]
    return _upsert(session, Ticker, rows, ["symbol"])


def upsert_prices(session: Session, prices: pd.DataFrame, volume: pd.DataFrame) -> int:
    prices = prices[[c for c in prices.columns if c in TICKER_TO_SECTOR]]
    vol = volume.reindex(index=prices.index, columns=prices.columns)
    long = prices.stack().rename("close").to_frame()  # descarta NaN
    long["volume"] = vol.stack().reindex(long.index)
    long = long.reset_index()
    long.columns = ["date", "ticker", "close", "volume"]
    rows = [
        {
            "ticker": r.ticker,
            "date": pd.Timestamp(r.date).date(),
            "close": float(r.close),
            "volume": None if pd.isna(r.volume) else int(r.volume),
        }
        for r in long.itertuples(index=False)
    ]
    return _upsert(session, Price, rows, ["ticker", "date"])


def upsert_vix(session: Session, vix: pd.Series) -> int:
    rows = [{"date": pd.Timestamp(d).date(), "value": float(v)} for d, v in vix.dropna().items()]
    return _upsert(session, Vix, rows, ["date"])


def _write_market_data(session: Session, data: MarketData) -> int:
    return upsert_prices(session, data.prices, data.volume) + upsert_vix(session, data.vix)


# ------------------------------------------------------------------ corridas
def _start_run(factory: sessionmaker[Session], mode: str) -> int:
    with factory() as s:
        run = EtlRun(mode=mode, status="running")
        s.add(run)
        s.commit()
        return run.id


def _finish_run(factory: sessionmaker[Session], run_id: int, status: str, rows: int, error: str | None) -> None:
    with factory() as s:
        run = s.get(EtlRun, run_id)
        run.status, run.rows, run.error, run.finished_at = status, rows, error, utcnow()
        s.commit()


def load_seed(seed_dir: Path = SEED_DIR) -> MarketData:
    prices = pd.read_parquet(seed_dir / "prices.parquet")
    volume = pd.read_parquet(seed_dir / "volume.parquet")
    vix = pd.read_parquet(seed_dir / "vix.parquet")["VIX"]
    return MarketData(prices, volume, vix)


def run_seed(factory: sessionmaker[Session], seed_dir: Path = SEED_DIR) -> int:
    """Carga inicial desde parquet. Devuelve filas escritas."""
    run_id = _start_run(factory, "seed")
    try:
        data = load_seed(seed_dir)
        with factory() as s, s.begin():  # una transacción: todo o nada
            rows = upsert_tickers(s) + _write_market_data(s, data)
    except Exception as exc:  # noqa: BLE001 - se registra y se relanza
        _finish_run(factory, run_id, "failed", 0, f"{type(exc).__name__}: {exc}"[:2000])
        raise
    _finish_run(factory, run_id, "success", rows, None)
    return rows


def run_incremental(factory: sessionmaker[Session], source: PriceSource, today: date | None = None) -> int:
    """Trae de la fuente solo las fechas que faltan (con un pequeño solapamiento)."""
    run_id = _start_run(factory, "incremental")
    try:
        today = today or date.today()
        with factory() as s:
            last = s.scalar(select(func.max(Price.date)))
        start = date.fromisoformat(START_DATE) if last is None else last - timedelta(days=OVERLAP_DAYS)
        data = source.fetch(TICKERS, start, today + timedelta(days=1))  # fuera de la transacción de BD
        with factory() as s, s.begin():
            rows = upsert_tickers(s) + _write_market_data(s, data)
    except Exception as exc:  # noqa: BLE001
        _finish_run(factory, run_id, "failed", 0, f"{type(exc).__name__}: {exc}"[:2000])
        raise
    _finish_run(factory, run_id, "success", rows, None)
    return rows


def has_prices(factory: sessionmaker[Session]) -> bool:
    with factory() as s:
        return s.scalar(select(func.count()).select_from(Price)) > 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Carga de datos de mercado en la base de datos.")
    ap.add_argument("--seed", action="store_true", help="cargar desde data/seed/*.parquet (sin red)")
    ap.add_argument("--if-empty", action="store_true", help="con --seed: no hace nada si ya hay precios cargados")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = get_settings()
    factory = make_session_factory(make_engine(settings.database_url))
    try:
        if args.seed:
            if args.if_empty and has_prices(factory):
                log.info("La base ya tiene precios: se omite el seed (--if-empty)")
                return 0
            rows = run_seed(factory)
        else:
            from app.sources.yahoo import YahooSource

            rows = run_incremental(factory, YahooSource())
    except Exception as exc:  # noqa: BLE001
        log.error("ETL falló: %s", exc)
        return 1
    log.info("ETL OK (%s): %d filas escritas", "seed" if args.seed else "incremental", rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())

