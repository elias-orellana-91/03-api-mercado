"""Dependencias compartidas de FastAPI."""
from __future__ import annotations

import secrets
from collections.abc import Iterator

from fastapi import Depends, Path, Request
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.db import Ticker, session_scope
from app.errors import ApiError
from app.ml.store import ModelStore
from app.services import queries

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Clave de administración")

SymbolPath = Path(pattern=r"^[A-Za-z][A-Za-z0-9.\-]{0,15}$", description="Símbolo del ticker (p. ej. AAPL)", examples=["AAPL"])


def get_db(request: Request) -> Iterator[Session]:
    yield from session_scope(request.app.state.session_factory)


def get_store(request: Request) -> ModelStore:
    return request.app.state.model_store


def valid_ticker(symbol: str = SymbolPath, db: Session = Depends(get_db)) -> Ticker:
    ticker = queries.get_ticker(db, symbol)
    if ticker is None:
        raise ApiError(404, "ticker_not_found", f"El ticker '{symbol.upper()}' no existe.")
    return ticker


def require_api_key(request: Request, key: str | None = Depends(api_key_header)) -> None:
    expected = request.app.state.settings.admin_api_key
    if not expected:
        raise ApiError(503, "admin_disabled", "El endpoint de administración está deshabilitado (ADMIN_API_KEY no configurada).")
    if not key or not secrets.compare_digest(key.encode(), expected.encode()):
        raise ApiError(401, "invalid_api_key", "API key ausente o inválida.")
