"""Resolve and load a model from the MLflow Model Registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
from mlflow.tracking import MlflowClient

from service.config import Settings


@dataclass(frozen=True)
class LoadedModel:
    estimator: Any
    name: str
    version: str
    uri: str
    run_id: str | None


def load_from_registry(settings: Settings) -> LoadedModel:
    """Load by pinned version if set, otherwise by alias; raises if it cannot be resolved."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_registry_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    if settings.model_version:
        mv = client.get_model_version(settings.model_name, settings.model_version)
    else:
        mv = client.get_model_version_by_alias(settings.model_name, settings.model_alias)
    # Load the exact resolved version so the reported version always matches the weights.
    uri = f"models:/{settings.model_name}/{mv.version}"
    estimator = mlflow.sklearn.load_model(uri)
    return LoadedModel(
        estimator, settings.model_name, str(mv.version), settings.model_uri, mv.run_id
    )
