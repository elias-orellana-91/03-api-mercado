"""Esquema de error común y manejadores de excepciones."""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.logging_conf import request_id_ctx

log = logging.getLogger("app.errors")


class ErrorBody(BaseModel):
    code: str = Field(description="Código estable legible por máquina", examples=["ticker_not_found"])
    message: str = Field(description="Mensaje para humanos", examples=["El ticker 'ZZZ' no existe."])
    details: list[dict] | None = Field(default=None, description="Detalle de validación (solo 422)")
    request_id: str | None = Field(default=None, description="Id de la petición (también en X-Request-ID)")


class ErrorResponse(BaseModel):
    error: ErrorBody


class ApiError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code, self.code, self.message = status_code, code, message
        super().__init__(message)


ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Recurso no encontrado (p. ej. ticker desconocido)"},
    422: {"model": ErrorResponse, "description": "Parámetros inválidos"},
    503: {"model": ErrorResponse, "description": "Modelo o base de datos no disponibles"},
}


def _payload(code: str, message: str, details: list[dict] | None = None) -> dict:
    return {"error": {"code": code, "message": message, "details": details, "request_id": request_id_ctx.get()}}


def register_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError):
        return JSONResponse(_payload(exc.code, exc.message), status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError):
        details = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(_payload("validation_error", "Parámetros inválidos.", details), status_code=422)

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException):
        code = {404: "not_found", 405: "method_not_allowed", 401: "unauthorized"}.get(exc.status_code, "http_error")
        return JSONResponse(_payload(code, str(exc.detail)), status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception):
        log.exception("Error no controlado")  # el traceback va al log, nunca a la respuesta
        return JSONResponse(_payload("internal_error", "Error interno del servidor."), status_code=500)
