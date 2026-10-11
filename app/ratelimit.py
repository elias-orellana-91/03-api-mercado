"""Rate limiting simple por IP de cliente (ventana fija, almacenamiento en memoria).

Se implementa con la librería ``limits`` (el motor que usa slowapi) en un middleware propio: el
``SlowAPIMiddleware`` no detecta las rutas con las versiones recientes de FastAPI (que envuelve los
routers incluidos) y dejaría el límite sin aplicar silenciosamente. Aquí no depende de las rutas.

Nota de despliegue: detrás de un proxy (Render) la IP real llega en ``X-Forwarded-For``; uvicorn la
resuelve con ``--proxy-headers`` (ver Dockerfile). El almacenamiento en memoria es por proceso: con
varias réplicas el límite efectivo se multiplica (usar Redis en ese caso).
"""
from __future__ import annotations

from limits import parse
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.logging_conf import request_id_ctx


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate: str, enabled: bool = True, exempt_paths: tuple[str, ...] = ("/health",)) -> None:
        super().__init__(app)
        self.item = parse(rate)
        self.enabled = enabled
        self.exempt = exempt_paths
        self.limiter = FixedWindowRateLimiter(MemoryStorage())

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.enabled or request.url.path in self.exempt:
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        if not self.limiter.hit(self.item, client):
            return JSONResponse(
                {
                    "error": {
                        "code": "rate_limited",
                        "message": "Demasiadas peticiones; intenta de nuevo más tarde.",
                        "details": None,
                        "request_id": request_id_ctx.get(),
                    }
                },
                status_code=429,
                headers={"Retry-After": str(self.item.get_expiry())},
            )
        return await call_next(request)
