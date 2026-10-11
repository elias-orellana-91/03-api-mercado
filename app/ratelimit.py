"""Rate limiting simple por IP de cliente (ventana fija, almacenamiento en memoria).

Se implementa con la librería ``limits`` (el motor que usa slowapi) en un middleware propio: el
``SlowAPIMiddleware`` no detecta las rutas con las versiones recientes de FastAPI (que envuelve los
routers incluidos) y dejaría el límite sin aplicar silenciosamente. Aquí no depende de las rutas.

Nota de despliegue: detrás de un proxy (Render) la IP real llega en ``X-Forwarded-For``. uvicorn NO se
arranca con ``--proxy-headers`` (confiaría en una cabecera falsificable); aquí se lee la cabecera solo con
``TRUST_PROXY=true`` y se toma la IP que agrega el proxy de confianza (``PROXY_HOPS``, ver ``client_ip``). El almacenamiento en memoria es por proceso: con
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


def client_ip(request: Request, trust_proxy: bool = False, proxy_hops: int = 1) -> str:
    """IP del cliente para el límite.

    * ``trust_proxy=False`` (por defecto): la IP del socket. ``X-Forwarded-For`` se ignora porque, sin un proxy
      delante, la controla el cliente por completo.
    * ``trust_proxy=True``: se unen TODAS las líneas de ``X-Forwarded-For`` (``getlist``: el cliente puede
      repetir la cabecera), se descartan los elementos vacíos y se toma ``parts[-proxy_hops]``, es decir, la
      IP que el proxy de confianza vio como cliente. Las anteriores las puede inventar el cliente. Si hay menos
      partes que saltos se usa la primera disponible; sin partes, la IP del socket.
    """
    if trust_proxy:
        parts = [p.strip() for line in request.headers.getlist("x-forwarded-for") for p in line.split(",")]
        parts = [p for p in parts if p]
        if parts:
            return parts[-proxy_hops] if len(parts) >= proxy_hops else parts[0]
    return request.client.host if request.client else "unknown"

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        rate: str,
        enabled: bool = True,
        exempt_paths: tuple[str, ...] = ("/health",),
        trust_proxy: bool = False,
        proxy_hops: int = 1,
    ) -> None:
        super().__init__(app)
        self.item = parse(rate)
        self.enabled = enabled
        self.exempt = exempt_paths
        self.trust_proxy = trust_proxy
        self.proxy_hops = proxy_hops
        self.limiter = FixedWindowRateLimiter(MemoryStorage())

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Los paths exentos (/health) dejan de estarlo si traen X-API-Key: así no sirven para adivinar la clave
        if not self.enabled or (request.url.path in self.exempt and "x-api-key" not in request.headers):
            return await call_next(request)
        if not self.limiter.hit(self.item, client_ip(request, self.trust_proxy, self.proxy_hops)):
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
