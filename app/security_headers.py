"""Cabeceras de seguridad para todas las respuestas de la API.

* ``Strict-Transport-Security``: la API solo se sirve por HTTPS (Render termina TLS en su proxy).
* ``X-Content-Type-Options: nosniff``: el navegador no adivina el tipo MIME.
* ``Cache-Control: no-store`` por defecto: los datos cambian con el ETL y /health y /admin no deben cachearse.
  Un endpoint que fije su propio ``Cache-Control`` lo conserva.

La cabecera ``Server`` se quita arrancando uvicorn con ``--no-server-header`` (ver Dockerfile).
"""
from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

HSTS = "max-age=31536000; includeSubDomains"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = HSTS
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers.setdefault("Cache-Control", "no-store")
        return response