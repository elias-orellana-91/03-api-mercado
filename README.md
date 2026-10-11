# 🔌 API de mercado (FastAPI + PostgreSQL)

Proyecto 3 del portafolio de finanzas. API propia que sirve datos de mercado de 16 acciones de EE.UU. (más el ETF `SPY`) y las predicciones de volatilidad del [proyecto 2](../02-ml-volatilidad). Es la **única puerta de acceso a los datos** del portafolio: el agente (proyecto 4) y la web (proyecto 5) solo hablarán con ella.

> **Regla de diseño:** los endpoints **nunca llaman a Yahoo**. Leen de nuestra base de datos. La única dependencia externa es la fuente de precios (`yfinance`), aislada en un adaptador (`app/sources/yahoo.py`, detrás de la interfaz `PriceSource`) que solo usa el proceso de carga (ETL).

## 🏗️ Arquitectura

```mermaid
flowchart LR
    Y[(Yahoo Finance)] -->|solo ETL| A[app/sources/yahoo.py<br/>PriceSource]
    S[data/seed/*.parquet] -->|--seed, sin red| E
    A --> E[ETL<br/>python -m app.etl<br/>upsert + transacción]
    E --> D[(PostgreSQL<br/>tickers · prices · vix<br/>predictions · etl_runs)]
    D --> F[FastAPI<br/>app/main.py]
    M[modelo P2<br/>SHA-256 verificado] --> F
    F --> G[Agente IA · proyecto 4]
    F --> W[Web · proyecto 5]
```

## 📡 Endpoints

Documentación interactiva (Swagger, con ejemplos y esquemas de error): **`/docs`** (ReDoc en `/redoc`). Con `ENVIRONMENT=production` la documentación y `/openapi.json` están **desactivados** (solo en local/desarrollo).

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/health` | Público: solo `status` (`ok`/`degraded`/`unavailable`). Con cabecera `X-API-Key`: además BD, modelo (hash ok), última corrida del ETL y última fecha con datos |
| GET | `/tickers` | Lista de tickers con su sector |
| GET | `/prices/{ticker}?start=&end=&limit=&offset=&order=` | Precios ajustados, validación de fechas y paginación (`limit` ≤ 5000) |
| GET | `/metrics/{ticker}?start=&end=` | Retorno total, volatilidad anualizada, max drawdown y Sharpe |
| GET | `/ranking?period=month\|quarter\|ytd\|custom&metric=return\|volatility\|sharpe&order=&start=&end=&limit=` | "¿Qué acción rindió mejor este mes?" (pensado para el agente) |
| GET | `/sectors/summary?start=&end=` | Resumen por sector (cartera de igual ponderación) |
| GET | `/predictions/{ticker}?date=` | Probabilidad de volatilidad alta en los próximos 5 días hábiles (por defecto, la última fecha). Solo calcula: no escribe en la BD |
| GET | `/predictions/latest` | Todos los tickers en la última fecha |
| GET | `/model/info` | Versión, fecha de entrenamiento, métricas de test, features y hash (las versiones de librerías y avisos solo con `X-API-Key`) |
| POST | `/admin/etl` | Dispara una carga incremental. Requiere cabecera `X-API-Key` |

**Errores** con esquema común: `404` ticker desconocido, `422` validación, `429` rate limit, `503` modelo o BD no disponibles.

```json
{"error": {"code": "ticker_not_found", "message": "El ticker 'ZZZ' no existe.", "details": null, "request_id": "5dcc2892e21f…"}}
```

### Ejemplos `curl`

```bash
curl http://localhost:8000/health                                   # solo {"status": "ok"}
curl -H "X-API-Key: $ADMIN_API_KEY" http://localhost:8000/health    # con detalle
curl "http://localhost:8000/prices/AAPL?start=2025-12-01&end=2025-12-30&limit=10"
curl "http://localhost:8000/metrics/NVDA?start=2025-01-01"
curl "http://localhost:8000/ranking?period=month&metric=return&limit=3"
curl "http://localhost:8000/ranking?period=custom&start=2025-06-01&end=2025-09-30&metric=sharpe"
curl "http://localhost:8000/sectors/summary?start=2025-01-01"
curl http://localhost:8000/predictions/AAPL
curl "http://localhost:8000/predictions/NVDA?date=2025-06-13"
curl http://localhost:8000/predictions/latest
curl http://localhost:8000/model/info
curl -X POST http://localhost:8000/admin/etl -H "X-API-Key: $ADMIN_API_KEY"
```

**Semántica del ranking:** `month`, `quarter` y `ytd` calculan el acumulado del periodo en curso hasta la última fecha con datos; la base es el **último cierre anterior** al inicio del periodo (así el retorno incluye el primer día). `custom` requiere `start`. `order=desc` (por defecto) devuelve primero el valor más alto; para "la acción menos volátil" usa `metric=volatility&order=asc`. **Ojo con `best`:** es siempre el elemento de rank 1 según `order`, no un juicio de calidad; con `order=asc` o `metric=volatility` no significa "mejor" (con volatilidad y `order=desc` es la más volátil). El campo no se renombra para no cambiar el contrato.

## 🚀 Cómo ejecutarla

### Sin Docker (SQLite, lo más rápido)

```powershell
python -m venv .venv
.venv\Scripts\activate                 # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
copy .env.example .env                 # DATABASE_URL=sqlite:///./local.db ya viene por defecto

alembic upgrade head                   # crea las tablas
python -m app.etl --seed               # carga data/seed/*.parquet (sin red, ~5 s)
uvicorn app.main:create_app --factory --reload
# http://localhost:8000/docs
```

Para actualizar con datos nuevos de Yahoo (único paso con red): `python -m app.etl` (incremental) o `POST /admin/etl`.

### Con Docker (PostgreSQL)

```bash
cp .env.example .env     # completa POSTGRES_USER / POSTGRES_PASSWORD / POSTGRES_DB y ADMIN_API_KEY
docker compose up --build
```

`db` (postgres:16 con volumen y healthcheck) → `api` (aplica migraciones al iniciar) → `etl` (tarea puntual con `--seed`). La API queda en `http://localhost:8000`.

> ⚠️ Los archivos Docker no se han probado a mano en una máquina con Docker; el CI (job `docker`) construye la imagen, la arranca contra Postgres y comprueba `/health` en cada push.

### Variables de entorno

| Variable | Descripción | Por defecto |
|---|---|---|
| `DATABASE_URL` | URL SQLAlchemy (`postgresql+psycopg://…` o `sqlite:///…`). Las URLs `postgres://`/`postgresql://` se normalizan | `sqlite:///./local.db` |
| `ADMIN_API_KEY` | Clave de `POST /admin/etl`. Vacía = endpoint deshabilitado (503) | _(vacía)_ |
| `CORS_ORIGINS` | Orígenes permitidos, separados por coma. Vacío = CORS desactivado. `*` prohibido si `ENVIRONMENT=production` | _(vacío)_ |
| `ENVIRONMENT` | `development` \| `production` | `development` |
| `RATE_LIMIT` | Límite por IP, p. ej. `60/minute` | `60/minute` |
| `TRUST_PROXY` | `true` solo si hay un proxy de confianza delante (Render): entonces la IP del cliente se lee de `X-Forwarded-For`. Sin proxy (p. ej. `docker-compose` con el puerto abierto) esa cabecera la controla el cliente y se ignora | `false` |
| `PROXY_HOPS` | N (≥ 1): número de IPs que agrega el proxy a `X-Forwarded-For`; se cuenta `parts[-N]`. En `render.yaml` es `1`, **a confirmar desplegando** (registrar el header completo de una petición desde una IP conocida y contar las IPs que Render agrega tras la del cliente) | `1` |
| `LOG_LEVEL` | Nivel de logging | `INFO` |
| `AUTO_SEED` | Solo en Docker: `true` carga el seed al iniciar si la BD está vacía | _(vacío)_ |

## 🔐 Decisiones de seguridad

- **Modelo verificado antes de cargarse.** `joblib` usa pickle, que puede ejecutar código arbitrario. `app/ml/artifacts/manifest.json` (versionado) guarda el SHA-256 de `model.joblib` y `metadata.json`; al arrancar se verifica el hash **antes** de `joblib.load`. Si no coincide, el modelo no se carga, `/health` pasa a `degraded` con el motivo y `/model/info` y `/predictions/*` responden 503. La ruta es fija dentro del paquete (nunca viene de input). Se avisa si las versiones de scikit-learn/pandas instaladas difieren de las del entrenamiento, y `requirements.txt` las fija exactamente. Tras reentrenar: `python -m app.ml.store --write-manifest`.
- **API key de administración** en `X-API-Key`, leída del entorno y comparada con `secrets.compare_digest`. Sin clave configurada, el endpoint está deshabilitado. La clave nunca se loguea.
- **CORS configurable**; nunca `*` en producción (la app no arranca).
- **Rate limiting por IP** (ventana fija; `/health` exento salvo que lleve `X-API-Key`, para que no sirva de fuerza bruta). Se implementa con la librería `limits` en un middleware propio: el middleware de `slowapi` no encuentra las rutas con las versiones recientes de FastAPI y dejaba el límite sin aplicar. Por defecto se usa la IP del socket e `X-Forwarded-For` se ignora (sin proxy lo controla el cliente). Con `TRUST_PROXY=true` (Render) se unen todas las líneas de esa cabecera y se toma la IP `parts[-PROXY_HOPS]`, la que agrega el proxy; las anteriores las puede inventar el cliente. uvicorn no se arranca con `--proxy-headers`. Todas las respuestas llevan `Strict-Transport-Security`, `X-Content-Type-Options: nosniff` y `Cache-Control: no-store`, y uvicorn arranca con `--no-server-header`. Hay que confirmar en el despliegue cuántas IPs agrega Render (`PROXY_HOPS`); el contador es por proceso (con varias réplicas usar Redis).
- **Hash del modelo público a propósito.** /model/info expone el SHA-256 del modelo: permite a cualquiera verificar su integridad, y la API lo comprueba contra manifest.json antes de cargar el pickle.
- **Cadena de suministro.** El CI corre con `permissions: contents: read`, las actions están fijadas por SHA de commit y la imagen base de Docker por digest; Dependabot (`.github/dependabot.yml`) propone las actualizaciones de pip, actions y Docker.
- **Sin secretos en el repo:** configuración solo por entorno; `.env` está en `.gitignore`; `.env.example` no contiene valores reales.
- **Solo ORM/parámetros** (cero SQL con f-strings), contenedor sin root, errores 500 sin detalles internos y logging JSON con `X-Request-ID` (sin cabeceras ni cuerpos).
- **Aislamiento de la fuente externa:** `yfinance` solo se importa en `app/sources/yahoo.py`; un test lo verifica y otros comprueban que ningún endpoint de lectura lo invoca.

## 🧠 Modelo (proyecto 2)

Se copiaron del proyecto 2 (`02-ml-volatilidad`, commit `73bddeb68bd30ce811667ddea59d56dce73c74c7`): `models/model.joblib`, `models/metadata.json` y `src/features.py` (→ `app/ml/features.py`, con un único cambio: el import del universo de tickers). La inferencia usa `build_features` sobre el historial leído de la base de datos y respeta `required_inputs` (SPY, VIX, volumen y **≥ 257 días**). Un test comprueba que, con los datos reales, las probabilidades de la API coinciden con ejecutar el pipeline directamente. Si se reentrena el modelo, hay que volver a copiar código y artefactos **juntos** y regenerar el manifiesto.

El modelo (regresión logística) tiene ROC-AUC 0,646 en test: es una señal modesta de riesgo relativo, **no** una recomendación de inversión.

## 🗄️ Base de datos y ETL

Tablas (migraciones con Alembic): `tickers`, `prices` (PK ticker+fecha), `vix`, `predictions` (única por ticker+fecha+versión del modelo) y `etl_runs`.

- `python -m app.etl --seed`: carga desde los parquet de `data/seed/` (copiados del proyecto 2), sin red.
- `python -m app.etl`: incremental; pide a la fuente solo desde la última fecha cargada (con 7 días de solape).
- **Idempotente** (UPSERT) y **atómico** (una transacción por corrida: si falla, rollback y la corrida queda como `failed` en `etl_runs` con el error).
- Limitación: los precios son *ajustados*; Yahoo reescribe el historial cuando hay dividendos o splits. El solape de 7 días solo corrige revisiones recientes; para una recarga completa, vacía las tablas y vuelve a ejecutar el ETL.

## ☁️ Despliegue en Render

El repo incluye un blueprint (`render.yaml`) con un web service Docker y una base PostgreSQL gestionada.

1. Sube el repositorio a GitHub.
2. En Render: **New → Blueprint** y selecciona el repo. Render crea la base de datos y el servicio.
3. Variables: `DATABASE_URL` se conecta sola a la base; `ADMIN_API_KEY` la **genera Render** (léela en el panel); define `CORS_ORIGINS` con la URL de la web (sin `*`).
4. En el primer arranque, `AUTO_SEED=true` aplica las migraciones y carga el seed si la BD está vacía. Después, la actualización de datos se hace con `POST /admin/etl` (o un cron externo que llame a ese endpoint).
5. Comprueba `https://<tu-servicio>.onrender.com/health` (devuelve `{"status": "ok"}`; con `X-API-Key` el detalle). `/docs` está desactivado en producción.

Notas: el plan gratuito duerme el servicio tras inactividad (la primera petición tarda) y la base gratuita tiene límites; revisa la política vigente de Render.

## 🧪 Tests y calidad

```bash
pytest -q          # 90 tests, sin red ni Docker (SQLite en memoria); 1 integración Postgres se salta sin TEST_POSTGRES_URL
ruff check .
# Integración con PostgreSQL real (opcional):
TEST_POSTGRES_URL=postgresql+psycopg://usuario:clave@localhost:5432/prueba pytest -m postgres
```

Cubren: cada endpoint (casos felices y errores), métricas y ranking contra cálculos a mano, ETL idempotente y atómico, verificación de hash (artefacto alterado ⇒ no carga), API key (ausente / incorrecta / correcta), CORS, rate limit y que ningún endpoint llame a la fuente externa. CI en `.github/workflows/ci.yml` (ruff + pytest con un contenedor de Postgres).

## 🗂️ Estructura

```
03-api-mercado/
├── app/
│   ├── main.py            # create_app (factory), middlewares, routers
│   ├── config.py · db.py · deps.py · errors.py · logging_conf.py · ratelimit.py
│   ├── etl.py             # python -m app.etl [--seed]
│   ├── sources/           # base.py (PriceSource) · yahoo.py (único import de yfinance)
│   ├── services/          # metrics.py (lógica del P1) · ranking.py · predictor.py · queries.py
│   ├── routers/           # market.py · ml.py · system.py
│   └── ml/                # features.py (copia del P2) · store.py (hash) · universe.py · artifacts/
├── migrations/ · alembic.ini
├── data/seed/             # parquet del proyecto 2 (precios, volumen, VIX)
├── tests/
├── Dockerfile · docker-compose.yml · render.yaml · .github/workflows/ci.yml
└── requirements.txt · requirements-dev.txt · pyproject.toml · .env.example
```

## 📄 Datos y aviso de uso

- Los precios, volúmenes y VIX provienen de Yahoo Finance a través de `yfinance`; este proyecto **no tiene afiliación** con Yahoo.
- Los datos (`data/seed/*.parquet` y lo que sirve la API) se incluyen **solo con fines educativos y no comerciales**; los términos de uso de Yahoo aplican a esos datos.
- La licencia MIT cubre el **código**, no los datos.
- La información y las predicciones **no constituyen una recomendación de inversión**.

## ⚖️ Licencia

Código bajo licencia [MIT](LICENSE) © 2026 Elias Orellana.

## 🛠️ Stack

Python 3.12 · FastAPI · Pydantic v2 · SQLAlchemy 2 · Alembic · PostgreSQL (psycopg 3) · pandas · scikit-learn · pytest · Docker · Render
