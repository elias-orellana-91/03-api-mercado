"""Configuración exclusivamente por variables de entorno (ver .env.example)."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "sqlite:///./local.db"
    admin_api_key: str | None = None
    cors_origins: Annotated[list[str], NoDecode] = []
    environment: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "60/minute"
    rate_limit_enabled: bool = True

    @field_validator("database_url")
    @classmethod
    def _normalize_db_url(cls, v: str) -> str:
        # Render/Heroku entregan postgres:// o postgresql://; SQLAlchemy necesita el driver psycopg 3.
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix) :]
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("admin_api_key", mode="before")
    @classmethod
    def _empty_key_is_none(cls, v):
        return v or None

    @model_validator(mode="after")
    def _no_wildcard_cors_in_production(self):
        if self.environment.lower() == "production" and "*" in self.cors_origins:
            raise ValueError("CORS_ORIGINS no puede ser '*' en producción: lista los orígenes explícitos.")
        return self

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    return Settings()
