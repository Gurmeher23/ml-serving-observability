import json

import numpy as np
import pandas as pd
import pytest

from monitoring import drift, drift_job
from training.data import FEATURES, make_energy_dataset


@pytest.fixture(scope="module")
def reference():
    return make_energy_dataset(2000, seed=1)


def test_psi_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    assert drift.psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.05


def test_psi_is_large_for_shifted_distribution():
    rng = np.random.default_rng(0)
    assert drift.psi(rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)) > 0.5


def test_psi_handles_discrete_and_unseen_categories():
    ref = np.array([0] * 97 + [1] * 3)
    assert drift.psi(ref, np.array([0] * 97 + [1] * 3)) < 0.01
    assert drift.psi(ref, np.array([2] * 100)) > 1.0  # entirely unseen value


def test_no_drift_on_fresh_sample_from_same_process(reference):
    report = drift.build_report(reference, make_energy_dataset(500, seed=2), FEATURES)
    assert report["dataset_drift"] is False
    assert report["drifted_share"] == 0.0


def test_heatwave_drift_is_detected_on_the_right_features(reference):
    current = make_energy_dataset(500, seed=3, scenario="heatwave")
    report = drift.build_report(reference, current, FEATURES)
    drifted = {f["feature"] for f in report["features"] if f["drifted"]}
    assert report["dataset_drift"] is True
    assert {"temperature_c", "wind_speed_ms", "cloud_cover"} <= drifted
    assert "is_holiday" not in drifted  # untouched features are not flagged


def test_single_feature_drift_below_dataset_threshold(reference):
    current = make_energy_dataset(500, seed=4)
    current["temperature_c"] += 12
    report = drift.build_report(reference, current, FEATURES)
    assert [f["feature"] for f in report["features"] if f["drifted"]] == ["temperature_c"]
    assert report["dataset_drift"] is False  # 1 of 6 features < 30%


def test_html_report_renders_and_escapes(reference):
    report = drift.build_report(
        reference, make_energy_dataset(300, seed=5, scenario="heatwave"), FEATURES
    )
    html = drift.render_html(report)
    assert "DATASET DRIFT DETECTED" in html and "temperature_c" in html
    report["features"][0]["feature"] = "<script>"
    assert "<script>" not in drift.render_html(report)


def _write(tmp_path, reference, current):
    ref_p, cur_p = tmp_path / "ref.csv", tmp_path / "cur.csv"
    reference[FEATURES].to_csv(ref_p, index=False)
    current[FEATURES].to_csv(cur_p, index=False)
    return ["--reference", str(ref_p), "--source", str(cur_p), "--out-dir", str(tmp_path / "out")]


def test_job_writes_reports_and_exits_2_on_drift_when_asked(tmp_path, reference, capsys):
    args = _write(tmp_path, reference, make_energy_dataset(400, seed=6, scenario="heatwave"))
    assert drift_job.main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dataset_drift"] is True
    assert (tmp_path / "out" / "drift_report.html").exists()
    saved = json.loads((tmp_path / "out" / "drift_report.json").read_text())
    assert saved["dataset_drift"] is True
    assert drift_job.main([*args, "--fail-on-drift"]) == 2


def test_job_reports_no_drift_with_exit_0(tmp_path, reference, capsys):
    args = _write(tmp_path, reference, make_energy_dataset(400, seed=7))
    assert drift_job.main([*args, "--fail-on-drift"]) == 0
    assert json.loads(capsys.readouterr().out)["dataset_drift"] is False


def test_job_skips_when_too_little_data(tmp_path, reference, capsys):
    args = _write(tmp_path, reference, make_energy_dataset(50, seed=8))
    assert drift_job.main(args) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "insufficient_data"
    assert not (tmp_path / "out").exists()


def test_job_returns_1_on_unreadable_source(tmp_path, reference):
    ref_p = tmp_path / "ref.csv"
    reference[FEATURES].to_csv(ref_p, index=False)
    assert (
        drift_job.main(["--reference", str(ref_p), "--source", str(tmp_path / "missing.csv")]) == 1
    )


def test_job_pushes_expected_gauges(tmp_path, reference, monkeypatch):
    captured = {}

    def fake_push(gateway, job, registry):
        captured["gateway"], captured["job"] = gateway, job
        captured["names"] = {m.name for m in registry.collect()}

    monkeypatch.setattr(drift_job, "push_to_gateway", fake_push)
    args = _write(tmp_path, reference, make_energy_dataset(400, seed=9, scenario="heatwave"))
    drift_job.main([*args, "--pushgateway", "pg:9091"])
    assert captured["gateway"] == "pg:9091"
    assert {"model_drift_share", "model_dataset_drift", "model_feature_psi"} <= captured["names"]


def test_end_to_end_service_capture_to_drift(settings, reference, tmp_path):
    """Requests through the real API -> /captured -> drift report."""
    from fastapi.testclient import TestClient

    from service.app import create_app

    rows = make_energy_dataset(300, seed=10, scenario="heatwave")[FEATURES].to_dict("records")
    with TestClient(create_app(settings)) as c:
        for i in range(0, 300, 100):
            assert c.post("/predict", json={"instances": rows[i : i + 100]}).status_code == 200
        captured = pd.DataFrame(c.get("/captured").json()["rows"], columns=FEATURES)
    assert len(captured) == 300
    assert drift.build_report(reference, captured, FEATURES)["dataset_drift"] is True
