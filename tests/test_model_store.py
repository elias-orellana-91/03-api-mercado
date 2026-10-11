"""Carga segura del modelo: hash SHA-256 verificado ANTES de joblib.load."""
from __future__ import annotations

import json
import shutil

import joblib
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.ml import store as store_mod
from app.ml.store import ARTIFACTS_DIR, ModelStore, sha256_file, write_manifest
from tests.conftest import forbidden_source_factory, make_settings, new_factory


@pytest.fixture
def artifacts(tmp_path):
    d = tmp_path / "artifacts"
    shutil.copytree(ARTIFACTS_DIR, d)
    return d


def test_manifest_matches_packaged_artifacts():
    manifest = json.loads((ARTIFACTS_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["files"]["model.joblib"] == sha256_file(ARTIFACTS_DIR / "model.joblib")
    assert manifest["files"]["metadata.json"] == sha256_file(ARTIFACTS_DIR / "metadata.json")


def test_valid_artifact_loads(artifacts):
    s = ModelStore(artifacts).load()
    assert s.loaded and s.hash_ok and s.error is None and len(s.sha256) == 64
    assert s.status()["version"] == s.metadata["model_version"]


def test_tampered_model_is_not_loaded_and_joblib_never_called(artifacts, monkeypatch):
    with (artifacts / "model.joblib").open("ab") as fh:
        fh.write(b"\x00payload-malicioso")

    def forbidden(*a, **k):
        raise AssertionError("joblib.load llamado con un artefacto cuyo hash no coincide")

    monkeypatch.setattr(joblib, "load", forbidden)
    monkeypatch.setattr(store_mod.joblib, "load", forbidden)
    s = ModelStore(artifacts).load()
    assert not s.loaded and not s.hash_ok and "model.joblib" in s.error and "NO cargado" in s.error


def test_tampered_metadata_is_rejected(artifacts):
    meta = artifacts / "metadata.json"
    meta.write_text(meta.read_text(encoding="utf-8").replace('"decision_threshold": 0.5', '"decision_threshold": 0.01'), encoding="utf-8")
    s = ModelStore(artifacts).load()
    assert not s.loaded and "metadata.json" in s.error


def test_missing_manifest_or_files_do_not_raise(artifacts):
    (artifacts / "manifest.json").unlink()
    s = ModelStore(artifacts).load()
    assert not s.loaded and s.error
    s2 = ModelStore(artifacts / "no-existe").load()
    assert not s2.loaded and s2.error


def test_write_manifest_roundtrip(artifacts):
    (artifacts / "model.joblib").write_bytes((artifacts / "model.joblib").read_bytes())  # sin cambios
    manifest = write_manifest(artifacts)
    assert ModelStore(artifacts).load().loaded
    assert set(manifest["files"]) == {"model.joblib", "metadata.json"}


def test_version_mismatch_warns_but_loads(artifacts):
    meta_path = artifacts / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["library_versions"]["scikit-learn"] = "0.0.1"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    write_manifest(artifacts)  # el manifiesto coincide: lo único que difiere es la versión declarada
    s = ModelStore(artifacts).load()
    assert s.loaded and any("scikit-learn" in w and "0.0.1" in w for w in s.version_warnings)


def test_installed_versions_match_training(model_store):
    assert model_store.version_warnings == []  # requirements.txt fija las versiones del P2


def test_api_reports_and_blocks_when_model_is_tampered(artifacts):
    (artifacts / "model.joblib").write_bytes(b"corrupto")
    bad = ModelStore(artifacts).load()
    app = create_app(make_settings(), new_factory(), bad, forbidden_source_factory)
    with TestClient(app) as c:
        h = c.get("/health").json()
        assert h["status"] == "degraded" and h["model"]["hash_ok"] is False and h["model"]["loaded"] is False
        assert h["model"]["error"]
        for path in ("/model/info", "/predictions/latest"):
            r = c.get(path)
            assert r.status_code == 503 and r.json()["error"]["code"] == "model_unavailable"
        # el resto de la API sigue funcionando
        assert c.get("/tickers").status_code == 200
