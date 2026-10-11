"""Aplicación FastAPI: única puerta de acceso a los datos del portafolio.

Arranque: ``uvicorn app.main:create_app --factory``.

Los endpoints leen SOLO de la base de datos propia; la fuente externa (Yahoo) vive aislada en
``app/sources`` y la usa únicamente el ETL.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, get_settings
from app.db import make_engine, make_session_factory
from app.errors import _payload, register_handlers
from app.logging_conf import RequestIdMiddleware, setup_logging
from app.ml.store import ModelStore
from app.ratelimit import RateLimitMiddleware
from app.routers import market, ml, system
from app.sources.base import PriceSource

log = logging.getLogger("app")

DESCRIPTION = """
API propia de datos de mercado (16 acciones de EE.UU. y SPY, 2015-2025) con predicción de volatilidad.

* Los datos salen de **nuestra base de datos**; ningún endpoint llama a Yahoo.
* `POST /admin/etl` requiere la cabecera `X-API-Key`.
* Errores con esquema común: `{"error": {"code", "message", "details", "request_id"}}`.
* No es una recomendación de inversión.
"""


def _default_source() -> PriceSource:
    from app.sources.yahoo import YahooSource

    return YahooSource()


def create_app(
    settings: Settings | None = None,
    session_factory: sessionmaker[Session] | None = None,
    model_store: ModelStore | None = None,
    source_factory: Callable[[], PriceSource] = _default_source,
) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    owns_engine = session_factory is None  # solo se cierra el motor que creamos nosotros
    factory = session_factory or make_session_factory(make_engine(settings.database_url))
    store = model_store or ModelStore().load()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        log.info("API iniciada", extra={"extra_fields": {"environment": settings.environment, "model": store.status()}})
        yield
        if owns_engine:
            factory.kw["bind"].dispose()

    app = FastAPI(
        title="API de mercado",
        version="1.0.0",
        description=DESCRIPTION,
        lifespan=lifespan,
        license_info={"name": "MIT"},
    )
    app.state.settings = settings
    app.state.session_factory = factory
    app.state.model_store = store
    app.state.source_factory = source_factory  # solo lo usa el ETL disparado desde /admin/etl

    register_handlers(app)

    @app.exception_handler(SQLAlchemyError)
    async def _db_error(_: Request, exc: SQLAlchemyError):
        log.error("Error de base de datos: %s", type(exc).__name__)
        return JSONResponse(_payload("database_unavailable", "Base de datos no disponible."), status_code=503)

    # El último middleware añadido es el más externo
    app.add_middleware(RateLimitMiddleware, rate=settings.rate_limit, enabled=settings.rate_limit_enabled)
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
        )
    app.add_middleware(RequestIdMiddleware)

    for r in (system.router, market.router, ml.router):
        app.include_router(r)
    return app

