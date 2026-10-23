import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlflow.exceptions import MlflowException

from service.app import create_app
from service.registry import load_from_registry
from training.data import FEATURES
from training.train import MODEL_NAME


def test_alias_resolves_to_v2_champion(settings):
    model = load_from_registry(settings)
    assert (model.name, model.version) == (MODEL_NAME, "2")
    assert model.run_id


def test_pinned_version_overrides_alias_for_rollback(settings):
    pinned = settings.model_copy(update={"model_version": "1"})
    assert load_from_registry(pinned).version == "1"
    with TestClient(create_app(pinned)) as c:
        assert c.get("/model").json()["model_version"] == "1"
        assert 'model_version{model_name="energy-load-model"} 1.0' in c.get("/metrics").text


def test_versions_are_genuinely_different_models(settings):
    row = pd.DataFrame([[4.5, 18, 2, 0, 5.0, 0.4]], columns=FEATURES)
    v1 = load_from_registry(settings.model_copy(update={"model_version": "1"}))
    v2 = load_from_registry(settings)
    assert type(v1.estimator).__name__ != type(v2.estimator).__name__
    assert v1.estimator.predict(row)[0] != v2.estimator.predict(row)[0]


def test_unknown_version_and_alias_fail_loudly(settings):
    with pytest.raises(MlflowException):
        load_from_registry(settings.model_copy(update={"model_version": "99"}))
    with pytest.raises(MlflowException):
        load_from_registry(settings.model_copy(update={"model_alias": "nope"}))


def test_v2_beats_baseline_on_logged_metrics(registry_dir, settings):
    from mlflow.tracking import MlflowClient

    load_from_registry(settings)  # sets the tracking URI
    client = MlflowClient()
    mae = {
        v: client.get_run(client.get_model_version(MODEL_NAME, v).run_id).data.metrics["mae"]
        for v in ("1", "2")
    }
    assert mae["2"] < mae["1"]
