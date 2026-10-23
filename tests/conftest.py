import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from service.app import create_app  # noqa: E402
from service.config import Settings  # noqa: E402
from training import train  # noqa: E402

VALID_ROW = {
    "temperature_c": 4.5,
    "hour_of_day": 18,
    "day_of_week": 2,
    "is_holiday": 0,
    "wind_speed_ms": 5.0,
    "cloud_cover": 0.4,
}


@pytest.fixture(scope="session")
def registry_dir(tmp_path_factory) -> Path:
    """A real MLflow registry (SQLite) with v1 + v2 trained on a small dataset."""
    store = tmp_path_factory.mktemp("registry")
    train.run(store, reference_path=None, n_rows=800)
    return store


@pytest.fixture
def settings(registry_dir) -> Settings:
    return Settings(mlflow_tracking_uri=train.sqlite_uri(registry_dir))


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:
        yield c
