"""Salud del servicio y administración (ETL)."""
from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.db import EtlRun, Price, utcnow
from app.deps import get_db, get_store, is_admin, require_api_key
from app.errors import ApiError
from app.etl import EtlLock
from app.ml.store import ModelStore
from app.schemas import EtlRunOut, EtlTriggerResponse, HealthResponse

log = logging.getLogger("app.admin")
router = APIRouter()
STALE_RUN = timedelta(minutes=30)


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Sistema"],
    summary="Estado del servicio (detalle de BD, modelo, ETL y datos solo con X-API-Key)",
    description=(
        "200 si la BD responde (status `ok`, o `degraded` si el modelo no cargó); 503 si la BD no está disponible. "
        "Público: solo `status`. Con `X-API-Key` válida añade `database`, `model`, `etl` y `data`."
    ),
    response_model_exclude_unset=True,
)
def health(
    db: Session = Depends(get_db), store: ModelStore = Depends(get_store), admin: bool = Depends(is_admin)
):
    model = store.status()
    try:
        db.execute(select(1))
        last_run = db.scalar(select(EtlRun).order_by(EtlRun.id.desc()).limit(1))
        last_date = db.scalar(select(func.max(Price.date)))
    except SQLAlchemyError:
        body = HealthResponse(status="unavailable")
        if admin:
            body = HealthResponse(
                status="unavailable", database={"ok": False}, model=model, etl={"last_run": None}, data={"last_date": None}
            )
        return JSONResponse(body.model_dump(mode="json", exclude_unset=True), status_code=503)
    ok = model["loaded"] and model["hash_ok"]
    status = "ok" if ok else "degraded"
    if not admin:
        return HealthResponse(status=status)
    return HealthResponse(
        status=status,
        database={"ok": True},
        model=model,
        etl={"last_run": EtlRunOut.model_validate(last_run, from_attributes=True) if last_run else None},
        data={"last_date": last_date},
    )


def _run_etl(factory: sessionmaker[Session], source_factory, lock: EtlLock) -> None:
    from app.etl import run_incremental

    try:
        rows = run_incremental(factory, source_factory())
        log.info("ETL incremental finalizado (%d filas)", rows)
    except Exception:  # noqa: BLE001 - ya registrado en etl_runs
        log.exception("ETL incremental falló")
    finally:
        lock.release()


@router.post(
    "/admin/etl",
    response_model=EtlTriggerResponse,
    status_code=202,
    dependencies=[Depends(require_api_key)],
    tags=["Administración"],
    summary="Dispara una carga incremental (requiere X-API-Key)",
    responses={
        401: {"description": "API key ausente o inválida"},
        409: {"description": "Ya hay un ETL en curso"},
        503: {"description": "Endpoint deshabilitado (sin ADMIN_API_KEY)"},
    },
)
def trigger_etl(request: Request, background: BackgroundTasks, db: Session = Depends(get_db)):
    factory = request.app.state.session_factory
    lock = EtlLock(factory)
    if not lock.acquire():  # atómico: de varias peticiones simultáneas solo una lo obtiene
        raise ApiError(409, "etl_in_progress", "Ya hay un ETL en curso.")
    try:
        running = db.scalar(
            select(EtlRun).where(EtlRun.status == "running", EtlRun.started_at > utcnow() - STALE_RUN).limit(1)
        )
        if running:
            raise ApiError(409, "etl_in_progress", f"Ya hay un ETL en curso (id {running.id}).")
    except BaseException:
        lock.release()
        raise
    background.add_task(_run_etl, factory, request.app.state.source_factory, lock)  # el task libera el lock
    return EtlTriggerResponse(status="accepted", message="Carga incremental iniciada; revisa /health para el resultado.")
