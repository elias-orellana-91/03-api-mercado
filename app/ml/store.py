"""Carga segura del modelo: verificación SHA-256 ANTES de ``joblib.load``.

* El manifiesto (``manifest.json``) está versionado en el repositorio y contiene el hash de
  ``model.joblib`` y de ``metadata.json``.
* El modelo se carga solo desde una ruta fija del paquete (``ARTIFACTS_DIR``); nunca desde input.
* Si el hash no coincide, NO se deserializa nada (joblib/pickle puede ejecutar código arbitrario)
  y ``/health`` lo reporta.
* Para regenerar el manifiesto tras reentrenar: ``python -m app.ml.store --write-manifest``.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib

log = logging.getLogger("app.model")

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
MANIFEST_FILE = "manifest.json"
# Librerías cuya versión debe coincidir con la del entrenamiento (clave de metadata -> paquete)
VERSIONED_LIBS = {"scikit-learn": "scikit-learn", "pandas": "pandas", "xgboost": "xgboost"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(directory: Path = ARTIFACTS_DIR) -> dict:
    manifest = {
        "algorithm": "sha256",
        "files": {name: sha256_file(directory / name) for name in (MODEL_FILE, METADATA_FILE)},
    }
    (directory / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


@dataclass
class ModelStore:
    directory: Path = ARTIFACTS_DIR
    model: Any = None
    metadata: dict = field(default_factory=dict)
    sha256: str | None = None
    hash_ok: bool = False
    error: str | None = None
    version_warnings: list[str] = field(default_factory=list)

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> ModelStore:
        """Verifica hashes y, solo si coinciden, carga modelo y metadata. Nunca lanza."""
        self.model, self.metadata, self.hash_ok, self.error, self.version_warnings = None, {}, False, None, []
        try:
            manifest = json.loads((self.directory / MANIFEST_FILE).read_text(encoding="utf-8"))
            expected = manifest["files"]
            model_path, meta_path = self.directory / MODEL_FILE, self.directory / METADATA_FILE
            actual = {MODEL_FILE: sha256_file(model_path), METADATA_FILE: sha256_file(meta_path)}
            self.sha256 = actual[MODEL_FILE]
            bad = [n for n in actual if actual[n] != expected.get(n)]
            if bad:
                self.error = f"Hash SHA-256 no coincide con el manifiesto: {', '.join(bad)}. Modelo NO cargado."
                log.error(self.error)
                return self
            self.hash_ok = True
            self.metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self.model = joblib.load(model_path)  # solo tras verificar el hash
            self.version_warnings = self._check_versions()
            for w in self.version_warnings:
                log.warning(w)
        except Exception as exc:  # noqa: BLE001 - el estado se reporta en /health
            self.model, self.error = None, f"{type(exc).__name__}: {exc}"
            log.error("No se pudo cargar el modelo: %s", self.error)
        return self

    def _check_versions(self) -> list[str]:
        warnings = []
        trained = self.metadata.get("library_versions", {})
        for key, package in VERSIONED_LIBS.items():
            if key not in trained:
                continue
            try:
                installed = importlib_metadata.version(package)
            except importlib_metadata.PackageNotFoundError:
                continue  # librería no instalada: solo importa si el modelo la necesita
            if installed != trained[key]:
                warnings.append(
                    f"{package}: instalada {installed} != entrenamiento {trained[key]}; "
                    "las predicciones podrían diferir."
                )
        return warnings

    def status(self) -> dict:
        return {
            "loaded": self.loaded,
            "hash_ok": self.hash_ok,
            "sha256": self.sha256,
            "version": self.metadata.get("model_version"),
            "error": self.error,
            "warnings": self.version_warnings,
        }


if __name__ == "__main__":
    import sys

    if "--write-manifest" in sys.argv:
        print(json.dumps(write_manifest(), indent=2))
    else:
        print(json.dumps(ModelStore().load().status(), indent=2))
