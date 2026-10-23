import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.config import Settings
from tests.conftest import VALID_ROW


def test_health_is_always_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_when_model_loaded(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json()["model_version"] == "2"


def test_not_ready_when_model_load_fails():
    def broken(_settings):
        raise RuntimeError("registry unreachable")

    with TestClient(create_app(Settings(), model_loader=broken)) as c:
        assert c.get("/health").status_code == 200  # process is alive...
        assert c.get("/ready").status_code == 503  # ...but must not receive traffic
        assert c.post("/predict", json={"instances": [VALID_ROW]}).status_code == 503


def test_predict_batch(client):
    r = client.post("/predict", json={"instances": [VALID_ROW, {**VALID_ROW, "hour_of_day": 3}]})
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "energy-load-model"
    assert body["model_version"] == "2"
    assert len(body["predictions"]) == 2
    assert all(30_000 < p < 90_000 for p in body["predictions"])
    assert body["predictions"][0] != body["predictions"][1]


@pytest.mark.parametrize(
    "bad",
    [
        {"temperature_c": 500},
        {"hour_of_day": 24},
        {"is_holiday": 2},
        {"cloud_cover": 1.5},
        {"wind_speed_ms": -1},
        {"unexpected": 1},
    ],
)
def test_predict_rejects_invalid_input(client, bad):
    assert client.post("/predict", json={"instances": [{**VALID_ROW, **bad}]}).status_code == 422


def test_predict_rejects_missing_field_and_empty_batch(client):
    row = {k: v for k, v in VALID_ROW.items() if k != "temperature_c"}
    assert client.post("/predict", json={"instances": [row]}).status_code == 422
    assert client.post("/predict", json={"instances": []}).status_code == 422


def test_predict_enforces_batch_limit(settings):
    small = settings.model_copy(update={"max_batch_size": 2})
    with TestClient(create_app(small)) as c:
        assert c.post("/predict", json={"instances": [VALID_ROW] * 2}).status_code == 200
        assert c.post("/predict", json={"instances": [VALID_ROW] * 3}).status_code == 413


def test_request_id_is_echoed_or_generated(client):
    assert client.get("/health", headers={"x-request-id": "abc"}).headers["x-request-id"] == "abc"
    assert len(client.get("/health").headers["x-request-id"]) == 32


def test_inference_failure_returns_500_without_leaking_details(settings):
    from service.registry import LoadedModel

    class Boom:
        def predict(self, _):
            raise ValueError("secret internal detail")

    def loader(_s):
        return LoadedModel(Boom(), "m", "7", "models:/m/7", None)

    with TestClient(create_app(settings, model_loader=loader)) as c:
        r = c.post("/predict", json={"instances": [VALID_ROW]})
        assert r.status_code == 500
        assert "secret" not in r.text
        assert 'model_inference_errors_total{reason="predict"} 1.0' in c.get("/metrics").text


def test_captured_rows_feed_drift_job(client):
    client.post("/predict", json={"instances": [VALID_ROW] * 3})
    body = client.get("/captured").json()
    assert body["count"] == 3
    assert body["rows"][0] == VALID_ROW


def test_capture_is_a_rolling_window(settings):
    with TestClient(create_app(settings.model_copy(update={"capture_max_rows": 5}))) as c:
        c.post("/predict", json={"instances": [VALID_ROW] * 4})
        c.post("/predict", json={"instances": [{**VALID_ROW, "hour_of_day": 9}] * 4})
        rows = c.get("/captured").json()["rows"]
        assert len(rows) == 5
        assert rows[-1]["hour_of_day"] == 9
