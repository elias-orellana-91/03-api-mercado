"""Rate limiting simple por IP de cliente (ventana fija, almacenamiento en memoria).

Se implementa con la librería ``limits`` (el motor que usa slowapi) en un middleware propio: el
``SlowAPIMiddleware`` no detecta las rutas con las versiones recientes de FastAPI (que envuelve los
routers incluidos) y dejaría el límite sin aplicar silenciosamente. Aquí no depende de las rutas.

Nota de despliegue: detrás de un proxy (Render) la IP real llega en ``X-Forwarded-For``. uvicorn NO se
arranca con ``--proxy-headers`` (confiaría en una cabecera falsificable); aquí se usa la ÚLTIMA IP de la
lista, la que agrega el proxy de confianza (ver ``client_ip``). El almacenamiento en memoria es por proceso: con
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


def client_ip(request: Request) -> str:
    """IP del cliente para el límite: la última de ``X-Forwarded-For`` (la agrega el proxy, no el cliente).

    Las anteriores las puede inventar el cliente, así que rotarlas no evita el bloqueo. Sin la cabecera
    (acceso directo) se usa la IP del socket.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        last = xff.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate: str, enabled: bool = True, exempt_paths: tuple[str, ...] = ("/health",)) -> None:
        super().__init__(app)
        self.item = parse(rate)
        self.enabled = enabled
        self.exempt = exempt_paths
        self.limiter = FixedWindowRateLimiter(MemoryStorage())

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Los paths exentos (/health) dejan de estarlo si traen X-API-Key: así no sirven para adivinar la clave
        if not self.enabled or (request.url.path in self.exempt and "x-api-key" not in request.headers):
            return await call_next(request)
        if not self.limiter.hit(self.item, client_ip(request)):
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
