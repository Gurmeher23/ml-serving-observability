"""Runtime configuration, read from environment variables (12-factor)."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", extra="ignore", protected_namespaces=())

    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    model_name: str = "energy-load-model"
    # Exactly one of these selects the model. A pinned version wins over the alias,
    # which makes "roll back to v1" a one-line config change.
    model_version: str | None = None
    model_alias: str = "champion"

    max_batch_size: int = Field(default=1000, ge=1)
    capture_max_rows: int = Field(default=5000, ge=0)
    log_level: str = "INFO"

    @property
    def model_uri(self) -> str:
        if self.model_version:
            return f"models:/{self.model_name}/{self.model_version}"
        return f"models:/{self.model_name}@{self.model_alias}"
