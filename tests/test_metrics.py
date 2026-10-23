from tests.conftest import VALID_ROW


def scrape(client) -> str:
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    return r.text


def test_all_required_metrics_exposed_after_traffic(client):
    client.post("/predict", json={"instances": [VALID_ROW] * 3})
    text = scrape(client)
    for name in (
        "http_request_duration_seconds_bucket",
        "http_requests_total",
        "model_prediction_value_bucket",
        "model_predict_batch_size_sum",
        "model_version",
        "model_inference_errors_total",
    ):
        assert name in text, name
    assert 'http_requests_total{method="POST",path="/predict",status="200"} 1.0' in text
    assert 'model_version{model_name="energy-load-model"} 2.0' in text
    assert "model_prediction_value_count 3.0" in text


def test_validation_errors_are_counted_as_422(client):
    client.post("/predict", json={"instances": [{**VALID_ROW, "hour_of_day": 99}]})
    assert 'path="/predict",status="422"' in scrape(client)


def test_unknown_paths_do_not_create_unbounded_labels(client):
    for i in range(5):
        client.get(f"/does-not-exist/{i}")
    text = scrape(client)
    assert "does-not-exist" not in text
    assert 'path="unmatched"' in text


def test_model_load_failure_is_counted():
    from fastapi.testclient import TestClient

    from service.app import create_app
    from service.config import Settings

    def broken(_s):
        raise RuntimeError("nope")

    with TestClient(create_app(Settings(), model_loader=broken)) as c:
        assert 'model_inference_errors_total{reason="model_load"} 1.0' in c.get("/metrics").text
