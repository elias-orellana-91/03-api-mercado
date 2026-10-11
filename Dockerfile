# syntax=docker/dockerfile:1

# ---- Etapa 1: dependencias en un venv aislado -------------------------------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---- Etapa 2: imagen final mínima, sin root ----------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=8000
RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app --home-dir /app --shell /usr/sbin/nologin app
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app migrations ./migrations
COPY --chown=app:app app ./app
COPY --chown=app:app data/seed ./data/seed
USER app
EXPOSE 8000

# 200 si la BD responde (el modelo degradado no tumba el contenedor); 503/timeout => unhealthy
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD ["python", "-c", "import os,sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health', timeout=4).status==200 else 1)"]

# Migra la BD, opcionalmente carga el seed si está vacía (AUTO_SEED=true) y arranca la API.
# Detrás de un proxy, define FORWARDED_ALLOW_IPS (uvicorn lo lee del entorno) para ver la IP real del cliente.
CMD ["sh", "-c", "alembic upgrade head && if [ \"$AUTO_SEED\" = \"true\" ]; then python -m app.etl --seed --if-empty; fi && exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers"]