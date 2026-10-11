"""API key, CORS, rate limiting, request id, errores de BD y aislamiento de la fuente externa."""
from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select

from app.config import Settings
from app.db import EtlRun, make_engine, make_session_factory
from app.main import create_app
from app.sources.base import MarketData
from tests.conftest import ADMIN_KEY, forbidden_source_factory, make_settings, new_factory, synthetic_market


class RecordingSource:
    def __init__(self):
        self.called = 0

    def fetch(self, tickers, start, end):
        self.called += 1
        m = synthetic_market(n_days=5)
        return MarketData(m.prices, m.volume, m.vix)


# ---------- /admin/etl ----------
def admin_client(model_store, source=None, **settings_kw):
    factory = new_factory()
    src = source or RecordingSource()
    app = create_app(make_settings(**settings_kw), factory, model_store, lambda: src)
    return TestClient(app), factory, src


def test_admin_without_key_is_401(model_store):
    c, _, src = admin_client(model_store)
    r = c.post("/admin/etl")
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert src.called == 0


def test_admin_wrong_key_is_401(model_store):
    c, _, src = admin_client(model_store)
    for bad in ("wrong", ADMIN_KEY + "x", ADMIN_KEY[:-1], ""):
        r = c.post("/admin/etl", headers={"X-API-Key": bad})
        assert r.status_code == 401
    assert src.called == 0


def test_admin_correct_key_runs_etl(model_store):
    c, factory, src = admin_client(model_store)
    r = c.post("/admin/etl", headers={"X-API-Key": ADMIN_KEY})
    assert r.status_code == 202 and r.json()["status"] == "accepted"
    assert src.called == 1  # el background task ya corrió (TestClient)
    with factory() as s:
        run = s.scalars(select(EtlRun)).one()
    assert run.status == "success" and run.mode == "incremental"


def test_admin_get_not_allowed(model_store):
    c, _, _ = admin_client(model_store)
    assert c.get("/admin/etl", headers={"X-API-Key": ADMIN_KEY}).status_code == 405


def test_admin_disabled_without_configured_key(model_store):
    c, _, src = admin_client(model_store, admin_api_key=None)
    r = c.post("/admin/etl", headers={"X-API-Key": "lo-que-sea"})
    assert r.status_code == 503 and r.json()["error"]["code"] == "admin_disabled"
    assert src.called == 0


def test_admin_conflict_when_etl_running(model_store):
    c, factory, src = admin_client(model_store)
    with factory() as s:
        s.add(EtlRun(mode="incremental", status="running"))
        s.commit()
    r = c.post("/admin/etl", headers={"X-API-Key": ADMIN_KEY})
    assert r.status_code == 409 and r.json()["error"]["code"] == "etl_in_progress" and src.called == 0


def test_admin_etl_failure_is_recorded_not_raised(model_store):
    class Broken:
        def fetch(self, *a, **k):
            raise ConnectionError("sin red")

    c, factory, _ = admin_client(model_store, source=Broken())
    assert c.post("/admin/etl", headers={"X-API-Key": ADMIN_KEY}).status_code == 202
    with factory() as s:
        assert s.scalars(select(EtlRun)).one().status == "failed"


def test_api_key_never_logged(model_store, caplog):
    c, _, _ = admin_client(model_store)
    with caplog.at_level(logging.DEBUG):
        c.post("/admin/etl", headers={"X-API-Key": ADMIN_KEY})
        c.post("/admin/etl", headers={"X-API-Key": "otra-clave-secreta"})
    text = "\n".join(r.getMessage() + str(getattr(r, "extra_fields", "")) for r in caplog.records)
    assert ADMIN_KEY not in text and "otra-clave-secreta" not in text


# ---------- CORS ----------
def test_cors_allows_only_configured_origins(seeded_factory, model_store):
    app = create_app(make_settings(cors_origins="https://mi-web.example"), seeded_factory, model_store, forbidden_source_factory)
    with TestClient(app) as c:
        ok = c.get("/tickers", headers={"Origin": "https://mi-web.example"})
        bad = c.get("/tickers", headers={"Origin": "https://evil.example"})
    assert ok.headers.get("access-control-allow-origin") == "https://mi-web.example"
    assert "access-control-allow-origin" not in bad.headers


def test_cors_disabled_by_default(client):
    r = client.get("/tickers", headers={"Origin": "https://cualquiera.example"})
    assert "access-control-allow-origin" not in r.headers


def test_wildcard_cors_rejected_in_production():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, environment="production", cors_origins="*")
    assert Settings(_env_file=None, environment="development", cors_origins="*").cors_origins == ["*"]


def test_database_url_normalized_for_psycopg3():
    assert Settings(_env_file=None, database_url="postgres://u:p@h/db").database_url == "postgresql+psycopg://u:p@h/db"
    assert Settings(_env_file=None, database_url="postgresql://u:p@h/db").database_url.startswith("postgresql+psycopg://")


# ---------- rate limit ----------
def test_rate_limit_returns_429_with_error_schema(seeded_factory, model_store):
    app = create_app(
        make_settings(rate_limit="3/minute", rate_limit_enabled=True), seeded_factory, model_store, forbidden_source_factory
    )
    with TestClient(app) as c:
        codes = [c.get("/tickers").status_code for _ in range(5)]
        limited = c.get("/tickers")
        health = [c.get("/health").status_code for _ in range(6)]  # /health está exento
    assert codes[:3] == [200, 200, 200] and codes[3:] == [429, 429]
    assert limited.json()["error"]["code"] == "rate_limited"
    assert set(health) == {200}


# ---------- request id y errores ----------
def test_request_id_generated_and_propagated(client):
    r = client.get("/tickers")
    assert re.fullmatch(r"[0-9a-f]{32}", r.headers["x-request-id"])
    mine = "abc-12345678"
    r2 = client.get("/prices/ZZZ", headers={"X-Request-ID": mine})
    assert r2.headers["x-request-id"] == mine and r2.json()["error"]["request_id"] == mine
    r3 = client.get("/tickers", headers={"X-Request-ID": "mal id con espacios\n"})
    assert r3.headers["x-request-id"] != "mal id con espacios"


def test_unavailable_database_gives_503(model_store):
    empty = make_session_factory(make_engine("sqlite://"))  # sin tablas => OperationalError
    with TestClient(create_app(make_settings(), empty, model_store, forbidden_source_factory)) as c:
        pub = c.get("/health")
        assert pub.status_code == 503 and pub.json() == {"status": "unavailable"}
        h = c.get("/health", headers={"X-API-Key": ADMIN_KEY})
        assert h.status_code == 503 and h.json()["status"] == "unavailable" and h.json()["database"]["ok"] is False
        r = c.get("/tickers")
        assert r.status_code == 503 and r.json()["error"]["code"] == "database_unavailable"
        assert "sqlite" not in r.text.lower()  # no se filtran detalles internos


# ---------- los endpoints nunca llaman a Yahoo ----------
READ_ENDPOINTS = [
    "/health", "/tickers", "/prices/AAPL", "/metrics/AAPL", "/ranking", "/ranking?period=ytd&metric=sharpe",
    "/sectors/summary", "/predictions/AAPL", "/predictions/latest", "/model/info", "/prices/ZZZ", "/openapi.json",
]


@pytest.mark.parametrize("path", READ_ENDPOINTS)
def test_read_endpoints_never_call_external_source(client, path):
    # forbidden_source_factory y yfinance.download (autouse) fallan si se invocan
    assert client.get(path).status_code in (200, 404)


def test_yfinance_is_isolated_in_a_single_adapter():
    root = Path(__file__).resolve().parents[1] / "app"
    offenders = [
        str(p.relative_to(root)) for p in root.rglob("*.py") if "yfinance" in p.read_text(encoding="utf-8") and p.name != "yahoo.py"
    ]
    # solo se permite mencionar yfinance en comentarios/docstrings de módulos no-adaptador
    for rel in offenders:
        text = (root / rel).read_text(encoding="utf-8")
        assert not re.search(r"^\s*(import|from)\s+yfinance", text, re.M), f"{rel} importa yfinance"
    imports = [p for p in root.rglob("*.py") if re.search(r"^\s*(import|from)\s+yfinance", p.read_text(encoding="utf-8"), re.M)]
    assert [p.name for p in imports] == ["yahoo.py"]


def test_rate_limit_not_bypassed_by_fake_forwarded_for(seeded_factory, model_store):
    """Rotar IPs falsas al inicio de X-Forwarded-For no evita el bloqueo: cuenta la última (la del proxy)."""
    app = create_app(
        make_settings(rate_limit="3/minute", rate_limit_enabled=True), seeded_factory, model_store, forbidden_source_factory
    )
    with TestClient(app) as c:
        codes = [c.get("/tickers", headers={"X-Forwarded-For": f"10.9.8.{i}, 203.0.113.7"}).status_code for i in range(6)]
        # otro cliente real (otra IP añadida por el proxy) tiene su propio contador
        other = c.get("/tickers", headers={"X-Forwarded-For": "10.9.8.1, 198.51.100.9"}).status_code
    assert codes == [200, 200, 200, 429, 429, 429]
    assert other == 200


def test_client_ip_falls_back_to_socket_address():
    from starlette.requests import Request

    from app.ratelimit import client_ip

    scope = {"type": "http", "headers": [], "client": ("192.0.2.5", 1234)}
    assert client_ip(Request(scope)) == "192.0.2.5"
    scope["headers"] = [(b"x-forwarded-for", b"1.1.1.1, 2.2.2.2 ")]
    assert client_ip(Request(scope)) == "2.2.2.2"

def test_concurrent_admin_etl_only_one_gets_202(model_store):
    """4 POST /admin/etl simultáneos: solo uno obtiene 202, el resto 409 (bloqueo atómico)."""
    import threading
    import time

    codes: list[int] = []
    codes_lock = threading.Lock()

    class SlowSource(RecordingSource):
        def fetch(self, tickers, start, end):
            # mantiene el ETL "en curso" hasta que las otras 3 peticiones hayan sido rechazadas
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                with codes_lock:
                    if codes.count(409) >= 3:
                        break
                time.sleep(0.02)
            return super().fetch(tickers, start, end)

    src = SlowSource()
    c, factory, _ = admin_client(model_store, source=src)
    start = threading.Barrier(4)

    def hit():
        start.wait()
        status = c.post("/admin/etl", headers={"X-API-Key": ADMIN_KEY}).status_code
        with codes_lock:
            codes.append(status)

    threads = [threading.Thread(target=hit) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(codes) == [202, 409, 409, 409]
    assert src.called == 1
    with factory() as s:
        assert len(s.scalars(select(EtlRun)).all()) == 1
    # el bloqueo se libera al terminar: un nuevo ETL vuelve a ser posible
    assert c.post("/admin/etl", headers={"X-API-Key": ADMIN_KEY}).status_code == 202


def test_etl_lock_released_even_if_etl_fails(model_store):
    class Broken:
        def fetch(self, *a, **k):
            raise ConnectionError("sin red")

    c, _, _ = admin_client(model_store, source=Broken())
    assert c.post("/admin/etl", headers={"X-API-Key": ADMIN_KEY}).status_code == 202
    assert c.post("/admin/etl", headers={"X-API-Key": ADMIN_KEY}).status_code == 202

# ---------- información interna ----------
def test_health_public_only_status_admin_gets_detail(client):
    pub = client.get("/health")
    assert pub.status_code == 200 and pub.json() == {"status": "ok"}
    for bad in ({"X-API-Key": "mala"}, {"X-API-Key": ""}):
        assert client.get("/health", headers=bad).json() == {"status": "ok"}
    full = client.get("/health", headers={"X-API-Key": ADMIN_KEY}).json()
    assert set(full) == {"status", "database", "model", "etl", "data"} and full["model"]["sha256"]


def test_health_detail_hidden_even_when_etl_failed(model_store):
    c, factory, _ = admin_client(model_store)
    with factory() as s:
        s.add(EtlRun(mode="incremental", status="failed", error="OperationalError: host secreto.internal"))
        s.commit()
    assert "secreto" not in c.get("/health").text
    assert "secreto" in c.get("/health", headers={"X-API-Key": ADMIN_KEY}).text


def test_model_info_hides_library_versions_without_key(client):
    pub = client.get("/model/info").json()
    assert "library_versions" not in pub and "warnings" not in pub and pub["sha256"]
    full = client.get("/model/info", headers={"X-API-Key": ADMIN_KEY}).json()
    assert full["library_versions"] and isinstance(full["warnings"], list)


def test_docs_disabled_in_production_enabled_otherwise(seeded_factory, model_store):
    prod = create_app(make_settings(environment="production"), seeded_factory, model_store, forbidden_source_factory)
    with TestClient(prod) as c:
        assert [c.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [404, 404, 404]
        assert c.get("/health").status_code == 200 and c.get("/tickers").status_code == 200
    dev = create_app(make_settings(environment="development"), seeded_factory, model_store, forbidden_source_factory)
    with TestClient(dev) as c:
        assert [c.get(p).status_code for p in ("/docs", "/redoc", "/openapi.json")] == [200, 200, 200]


def test_health_with_api_key_is_rate_limited(seeded_factory, model_store):
    """/health sigue exento sin clave (healthchecks), pero con X-API-Key cuenta: no sirve para fuerza bruta."""
    app = create_app(
        make_settings(rate_limit="3/minute", rate_limit_enabled=True), seeded_factory, model_store, forbidden_source_factory
    )
    with TestClient(app) as c:
        assert {c.get("/health").status_code for _ in range(8)} == {200}
        codes = [c.get("/health", headers={"X-API-Key": "adivinando"}).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]

# ---------- cabeceras de seguridad ----------
def test_security_headers_on_all_kinds_of_responses(seeded_factory, model_store):
    app = create_app(
        make_settings(rate_limit="2/minute", rate_limit_enabled=True), seeded_factory, model_store, forbidden_source_factory
    )
    with TestClient(app) as c:
        responses = [c.get("/health"), c.get("/tickers"), c.get("/prices/ZZZ"), c.get("/tickers"), c.get("/tickers")]
    assert [r.status_code for r in responses] == [200, 200, 404, 429, 429]  # OK, 404 y 429
    for r in responses:
        assert r.headers["strict-transport-security"].startswith("max-age=")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["cache-control"] == "no-store"


def test_dockerfile_hides_server_header_and_does_not_trust_proxy_headers():
    text = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")
    cmd = text[text.rindex("CMD ["):]
    assert "--no-server-header" in cmd and "--proxy-headers" not in cmd
    assert "FORWARDED_ALLOW_IPS" not in (Path(__file__).resolve().parents[1] / "render.yaml").read_text(encoding="utf-8")

# ---------- CI / Docker fijados ----------
def test_ci_and_docker_supply_chain_pinning():
    root = Path(__file__).resolve().parents[1]
    ci = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert re.search(r"^permissions:\s*\n\s+contents: read", ci, re.M)
    uses = re.findall(r"uses:\s*(\S+)", ci)
    assert uses and all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u) for u in uses), uses  # actions por SHA de commit
    docker = (root / "Dockerfile").read_text(encoding="utf-8")
    froms = re.findall(r"^FROM\s+(\S+)", docker, re.M)
    assert froms and all(re.fullmatch(r"[\w./:-]+@sha256:[0-9a-f]{64}", f) for f in froms), froms
    dependabot = (root / ".github" / "dependabot.yml").read_text(encoding="utf-8")
    for eco in ("pip", "github-actions", "docker"):
        assert f"package-ecosystem: {eco}" in dependabot