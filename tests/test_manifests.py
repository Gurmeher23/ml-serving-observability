"""Static checks of the deploy artefacts.

Rendering uses ``kubectl kustomize`` and ``helm template`` when installed; the tests skip
(with the reason) otherwise. Nothing here talks to a cluster: live behaviour is exercised
by the kind job in .github/workflows/ci.yml.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> list[dict]:
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=ROOT).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def _by_kind(docs: list[dict], kind: str) -> list[dict]:
    return [d for d in docs if d["kind"] == kind]


needs_kubectl = pytest.mark.skipif(not shutil.which("kubectl"), reason="kubectl not installed")
needs_helm = pytest.mark.skipif(not shutil.which("helm"), reason="helm not installed")


def check_workload(docs: list[dict]) -> None:
    (dep,) = _by_kind(docs, "Deployment")
    pod = dep["spec"]["template"]
    (container,) = pod["spec"]["containers"]
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert "startupProbe" in container
    assert container["resources"]["requests"] and container["resources"]["limits"]
    assert pod["spec"]["securityContext"]["runAsNonRoot"] is True
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert pod["metadata"]["annotations"]["prometheus.io/path"] == "/metrics"
    assert dep["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    # Service and Deployment must agree on selector labels, or traffic goes nowhere.
    (svc,) = _by_kind(docs, "Service")
    for key, value in svc["spec"]["selector"].items():
        assert pod["metadata"]["labels"][key] == value
    assert dep["spec"]["selector"]["matchLabels"].items() <= pod["metadata"]["labels"].items()
    port_names = {p["name"] for p in container["ports"]}
    assert svc["spec"]["ports"][0]["targetPort"] in port_names
    # HPA points at the Deployment.
    (hpa,) = _by_kind(docs, "HorizontalPodAutoscaler")
    assert hpa["spec"]["scaleTargetRef"]["name"] == dep["metadata"]["name"]
    assert hpa["spec"]["minReplicas"] <= hpa["spec"]["maxReplicas"]
    # Every envFrom ConfigMap exists.
    cms = {c["metadata"]["name"] for c in _by_kind(docs, "ConfigMap")}
    assert {e["configMapRef"]["name"] for e in container["envFrom"]} <= cms


def check_cronjob(docs: list[dict]) -> None:
    (cj,) = _by_kind(docs, "CronJob")
    assert cj["spec"]["concurrencyPolicy"] == "Forbid"
    c = cj["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]
    assert c["command"] == ["python", "-m", "monitoring.drift_job"]
    assert any(a.startswith("--source=http://") and a.endswith("/captured") for a in c["args"])
    assert c["resources"]["limits"]


@needs_kubectl
@pytest.mark.parametrize("path", ["deploy/k8s/base", "deploy/k8s/overlays/ci"])
def test_kustomize_renders_valid_workload(path):
    docs = _run(["kubectl", "kustomize", path])
    check_workload(docs)
    check_cronjob(docs)
    assert {d["kind"] for d in docs} >= {
        "Namespace",
        "ConfigMap",
        "Deployment",
        "Service",
        "HorizontalPodAutoscaler",
        "CronJob",
        "PodDisruptionBudget",
    }


@needs_kubectl
def test_ci_overlay_uses_local_image_and_single_replica():
    docs = _run(["kubectl", "kustomize", "deploy/k8s/overlays/ci"])
    (dep,) = _by_kind(docs, "Deployment")
    assert dep["spec"]["replicas"] == 1
    assert dep["spec"]["template"]["spec"]["containers"][0]["image"] == (
        "ml-serving-observability:ci"
    )
    assert _by_kind(docs, "HorizontalPodAutoscaler")[0]["spec"]["minReplicas"] == 1
    (cj,) = _by_kind(docs, "CronJob")
    job_image = cj["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0]["image"]
    assert job_image == "ml-serving-observability:ci"


@needs_kubectl
def test_prometheus_operator_overlay_adds_servicemonitor():
    docs = _run(["kubectl", "kustomize", "deploy/k8s/overlays/prometheus-operator"])
    (sm,) = _by_kind(docs, "ServiceMonitor")
    assert sm["spec"]["endpoints"][0]["path"] == "/metrics"


@needs_helm
def test_helm_chart_lints_and_renders():
    subprocess.run(["helm", "lint", "charts/ml-service"], check=True, cwd=ROOT, capture_output=True)
    docs = _run(["helm", "template", "rel", "charts/ml-service"])
    check_workload(docs)
    check_cronjob(docs)
    assert not _by_kind(docs, "Ingress")  # off by default


@needs_helm
def test_helm_aks_style_ingress_and_servicemonitor():
    docs = _run(
        [
            "helm",
            "template",
            "rel",
            "charts/ml-service",
            "--set",
            "ingress.enabled=true",
            "--set",
            "prometheus.serviceMonitor.enabled=true",
            "--set",
            "config.MODEL_VERSION=1",
        ]
    )
    (ing,) = _by_kind(docs, "Ingress")
    assert ing["spec"]["ingressClassName"] == "webapprouting.kubernetes.azure.com"
    paths = {p["path"] for p in ing["spec"]["rules"][0]["http"]["paths"]}
    assert paths == {"/predict", "/ready"}  # /metrics and /captured are not exposed
    assert _by_kind(docs, "ServiceMonitor")
    (cm,) = _by_kind(docs, "ConfigMap")
    assert cm["data"]["MODEL_VERSION"] == "1"


@pytest.mark.skipif(
    not shutil.which("kubeconform") or not shutil.which("kubectl"),
    reason="kubeconform/kubectl not installed (schema check runs in CI)",
)
def test_kubeconform_schema_validation():
    rendered = subprocess.run(
        ["kubectl", "kustomize", "deploy/k8s/overlays/ci"],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout
    result = subprocess.run(
        ["kubeconform", "-strict", "-summary", "-kubernetes-version", "1.30.0"],
        input=rendered,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_grafana_dashboard_only_queries_metrics_that_exist(settings):
    from service.metrics import Metrics

    emitted: set[str] = set()
    suffixes = {"counter": ["_total"], "histogram": ["_bucket", "_sum", "_count"], "gauge": [""]}
    for metric in Metrics().registry.collect():
        base = metric.name
        emitted.update(base + sfx for sfx in suffixes[metric.type])
    # Gauges pushed by monitoring.drift_job.
    emitted |= {"model_drift_share", "model_dataset_drift", "model_feature_psi"}
    dash = json.loads((ROOT / "deploy/grafana/dashboards/ml-service.json").read_text())
    for panel in dash["panels"]:
        for target in panel["targets"]:
            idents = set(re.findall(r"[a-zA-Z_:][a-zA-Z0-9_:]*", target["expr"]))
            names = {i for i in idents if i.startswith(("http_", "model_"))}
            assert names, target["expr"]
            unknown = {n for n in names if n not in emitted}
            assert not unknown, (panel["title"], unknown)


def test_compose_and_prometheus_configs_are_consistent():
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    prom = yaml.safe_load((ROOT / "deploy/prometheus/prometheus.yml").read_text())
    targets = {
        t for sc in prom["scrape_configs"] for c in sc["static_configs"] for t in c["targets"]
    }
    for target in targets:
        assert target.split(":")[0] in compose["services"], target
    alerts = yaml.safe_load((ROOT / "deploy/prometheus/alerts.yml").read_text())
    assert {r["alert"] for g in alerts["groups"] for r in g["rules"]} >= {"MlModelDriftDetected"}
    for volume in compose["services"]["prometheus"]["volumes"]:
        assert (ROOT / volume.split(":")[0]).exists(), volume


def test_workflow_is_valid_yaml_with_expected_jobs():
    wf = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    assert {"lint-and-test", "manifests", "kind-e2e"} <= set(wf["jobs"])
