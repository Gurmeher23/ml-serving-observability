# ml-serving-observability

A model-serving service with the operational pieces that usually get skipped in ML demos:
a versioned model registry, health/readiness probes, Prometheus metrics, a Grafana
dashboard, a drift-detection job, and Kubernetes manifests that are actually deployed to a
cluster (a local `kind` cluster, in CI) on every push.

The model itself is deliberately simple (electricity-load regression on **synthetic**,
deterministic data). The project is about everything around the model.

## Real-world use

Typical situation: a forecasting or scoring model is trained by a data scientist and has to
run for months behind an API. Things that go wrong, and what this repo does about them:

| Problem in production | What is here |
| --- | --- |
| "Which model is live?" | MLflow Model Registry versions + aliases, version exported as a metric and returned in every response |
| A new model is worse | Rollback by pinning `MODEL_VERSION` in config and restarting; no rebuild (tested in CI) |
| The service is up but not usable | Separate liveness (`/health`) and readiness (`/ready`) probes; a failed model load keeps the pod out of rotation |
| Slow or failing requests | Latency histogram, request counter by status, inference error counter, Grafana panels, alert rules |
| Inputs change over time | Requests are captured in a rolling window; a CronJob compares them with the training reference (PSI + KS) |
| Garbage input | Pydantic bounds on every field, batch size limit, rejected with 422/413 |

## Architecture

```
 training/train.py                         requests
 (synthetic data)                             |
        |                                     v
        v                         +-----------------------+   /metrics    +------------+
 MLflow Model Registry  --load--> |  FastAPI ml-service   | ------------> | Prometheus |--> Grafana
 energy-load-model                |  /predict /health     |               +-----^------+
  v1 Ridge   (previous)           |  /ready /metrics      |                     | push
  v2 GBM     (champion)           |  rolling capture      |  GET /captured      |
                                  +-----------------------+ <------------ drift CronJob
                                                                       (PSI + KS vs reference)
                                                                       -> JSON + HTML report
                                                                       -> Pushgateway gauges
```

Layout:

```
service/      FastAPI app, config, registry loader, metrics, JSON logging, request capture
training/     synthetic dataset + train/register script
monitoring/   drift.py (PSI, KS, report) and drift_job.py (batch CLI)
deploy/       k8s/ (Kustomize base + overlays), prometheus/, grafana/ (provisioned dashboard)
charts/       Helm chart ml-service (AKS-style ingress values)
tests/        pytest suite (API, metrics, registry, drift, manifests)
scripts/      send_traffic.py (stdlib-only load/drift generator)
```

## Run it

Python 3.12.

### Locally (uvicorn)

```bash
python -m venv .venv && source .venv/bin/activate
make install        # pip install -e ".[dev]"
make train          # registers energy-load-model v1 (Ridge) and v2 (GBM, alias "champion")
make serve          # uvicorn on :8000, model = models:/energy-load-model@champion

curl -s localhost:8000/ready
curl -s -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"instances":[
  {"temperature_c":2,"hour_of_day":18,"day_of_week":2,"is_holiday":0,"wind_speed_ms":5,"cloud_cover":0.5}]}'
make traffic        # 400 normal rows
make drift          # drift job against the running service -> reports/drift_report.{json,html}
```

Pin a version instead of the alias (rollback): `MODEL_VERSION=1 make serve`.

### With Docker Compose (service + Prometheus + Pushgateway + Grafana)

```bash
docker compose up --build -d
python scripts/send_traffic.py --rows 400 --scenario heatwave
docker compose --profile drift run --rm drift-check
```

Service `:8000`, Prometheus `:9090`, Grafana `:3000` (dashboard "ML service: latency,
errors, drift" is provisioned; admin password from `GRAFANA_ADMIN_PASSWORD`, default `admin`,
change it for anything that is not your laptop).

### On a local kind cluster

```bash
docker build -t ml-serving-observability:ci .
kind create cluster --name ci
kind load docker-image ml-serving-observability:ci --name ci
kubectl apply -k deploy/k8s/overlays/ci
kubectl -n ml-serving rollout status deployment/ml-service
kubectl -n ml-serving port-forward svc/ml-service 8000:80
```

This is exactly what the `kind-e2e` job in `.github/workflows/ci.yml` does. Helm:
`helm template rel charts/ml-service` (add `--set ingress.enabled=true` for the AKS-style ingress).

## Endpoints

| Route | Purpose |
| --- | --- |
| `POST /predict` | Batched prediction. Body `{"instances":[{...6 features...}]}`, 1 to 1000 rows |
| `GET /health` | Liveness: process is up |
| `GET /ready` | Readiness: 200 only when a model is loaded, else 503 |
| `GET /model` | Name, version, registry URI and MLflow run id of the loaded model |
| `GET /metrics` | Prometheus exposition |
| `GET /captured` | Recent request features for the drift job (internal, see limitations) |

Metrics: `http_request_duration_seconds` (histogram), `http_requests_total`,
`model_prediction_value` (histogram), `model_predict_batch_size` (histogram),
`model_version` (gauge), `model_inference_errors_total{reason}`. The drift job adds
`model_dataset_drift`, `model_drift_share`, `model_feature_psi{feature}` through a Pushgateway.
Route labels use the route template, so unknown URLs cannot blow up label cardinality (tested).

## Actual local output

Command: `pytest` (Python 3.12.10, macOS, mlflow-skinny 3.17, scikit-learn 1.9, FastAPI 0.161).

```
$ pytest -v
...
============================= 48 passed in 11.67s ==============================
tests/test_api.py: 16   tests/test_drift.py: 13   tests/test_manifests.py: 10
tests/test_metrics.py: 4   tests/test_registry.py: 5

$ ruff check . && ruff format --check .
All checks passed!
22 files already formatted

$ kubectl kustomize deploy/k8s/overlays/ci | kubeconform -strict -summary -kubernetes-version 1.30.0
Summary: 7 resources found parsing stdin - Valid: 7, Invalid: 0, Errors: 0, Skipped: 0

$ helm lint charts/ml-service
1 chart(s) linted, 0 chart(s) failed
```

Training (`make train`), metrics on a held-out 20 % split of the synthetic data:

```
registered energy-load-model v1 alias=previous metrics={'mae': 2415.37, 'rmse': 2972.92}
registered energy-load-model v2 alias=champion metrics={'mae': 774.30, 'rmse': 983.89}
```

Drift job against a running service. First 300 rows from the training-like scenario, then
1000 rows from the shifted "heatwave" scenario (hotter, calmer, clearer):

```
$ python scripts/send_traffic.py --rows 300 --scenario none
$ python -m monitoring.drift_job --source http://localhost:8123/captured
{"status": "ok", "dataset_drift": false, "drifted_share": 0.0, "drifted_features": [], "current_rows": 300}

$ python scripts/send_traffic.py --rows 1000 --batch 50 --scenario heatwave
$ python -m monitoring.drift_job --source http://localhost:8123/captured --min-samples 500
{"status": "ok", "dataset_drift": true, "drifted_share": 0.5, "drifted_features": ["temperature_c", "wind_speed_ms", "cloud_cover"], "current_rows": 1301}
```

(The second window still contains the first 300 normal rows, which is why the shift is
diluted but still detected. PSI per feature: temperature 0.54, wind 0.76, cloud cover 1.40;
the untouched features stay below 0.02.)

A JSON log line from the service:

```
{"ts": "2026-10-23T12:38:26.386+00:00", "level": "INFO", "logger": "service", "message": "request", "request_id": "da2e06294bc143fb80070f947f64c45a", "method": "POST", "path": "/predict", "status": 200, "duration_ms": 1.68}
```

Feature values are never logged.

I also checked the Docker build's dependency set without Docker: a clean virtualenv with
`pip install .`, training into a different directory, serving from that directory, then
traffic and the drift job all worked.

## What is and is not tested

| Area | Status |
| --- | --- |
| API, validation, metrics, registry loading and rollback, drift maths, drift CLI | Unit/integration tests, run locally and in CI (real MLflow SQLite registry, no mocks of the registry) |
| Kustomize/Helm output | Rendered and checked for probes, resources, selectors, HPA target, security context, ingress paths, CronJob command (tests); schema-validated with kubeconform |
| Grafana dashboard | Test checks that every metric it queries is actually emitted. **Not** opened in a Grafana instance by me |
| Docker image, Compose stack | **Not run locally** (no Docker daemon on the author's machine while building). Built and run in CI only (image); the Compose stack is only config-validated |
| Kubernetes deployment | **Deployed to a local `kind` cluster in GitHub Actions only** (rollout, probes, port-forward, `/predict`, `/metrics`, drift CronJob run, config rollback). Not run locally, no persistent cluster |
| Cloud / AKS | **None.** The Helm values are AKS-style (ingress class, TLS) and "AKS-ready, not AKS-tested" |
| HPA behaviour | Object is valid; kind has no metrics-server, so autoscaling itself is never exercised |
| Load / performance | Not measured. No latency numbers are claimed |

CI (`.github/workflows/ci.yml`) was written without being executed locally; the first run on
GitHub is its first real run. Check the Actions tab for its actual status.

## Design notes

**Model versioning and rollback.** `training/train.py` registers every model in the MLflow
Model Registry (SQLite backend, because MLflow's file store is in maintenance mode) and moves
aliases: `champion` is what serves, `previous` is the last known good. The service resolves
the alias to a concrete version at startup, loads that exact version, and reports it in
`/model`, in every `/predict` response, and as the `model_version` gauge. Promotion is
"move the alias, restart pods"; rollback is "set `MODEL_VERSION` in the ConfigMap, restart",
with `maxUnavailable: 0` so serving continues during the roll. The CI job exercises this.
Models are loaded once at startup, so a version change is always an explicit, observable
rollout rather than a silent in-process swap.

**Probes.** Startup probe (allows model load), liveness on `/health` (never depends on the
model, so a bad model cannot cause restart loops), readiness on `/ready` (depends on the
model, so a bad model never receives traffic).

**Suggested SLOs** (targets to design alerts around, not measured results): 99.5 % of
`/predict` requests non-5xx over 30 days; p95 `/predict` latency under 250 ms; model
version consistent across replicas; drift job succeeded within the last 2 hours.
`deploy/prometheus/alerts.yml` encodes the first two plus a drift alert.

**Drift method.** PSI on reference deciles (category frequencies for discrete features),
plus a two-sample KS test. A feature is flagged if PSI >= 0.2, or KS p < 0.01 **and** KS
statistic >= 0.1 (otherwise large samples flag trivial differences). The dataset is flagged
when at least 30 % of features drift. Thresholds are conventions, not universal truths; tune
them per model. I implemented this directly instead of using Evidently so each threshold is
visible and unit-tested. Evidently is **not** used here.

**Security basics.** Non-root container, dropped capabilities, `RuntimeDefault` seccomp, no
privilege escalation, no secrets in the repo, request features never logged.

## Limitations (honest list)

* **Demo registry is baked into the image.** The Dockerfile trains and registers v1/v2 at
  build time so the image is self-contained. A real setup points `MLFLOW_TRACKING_URI` at a
  shared tracking server with a database and object storage, and training runs elsewhere.
* **Rolling capture is per pod and in memory.** With more than one replica, the drift job
  sees one random pod's recent requests, and the window is lost on restart. The CI overlay
  runs one replica for this reason. A production design would stream features to a shared
  store (Kafka, a table) instead.
* **`/captured` has no authentication** and exposes recent request features. It is meant to
  stay cluster-internal; the Helm ingress only exposes `/predict` and `/ready`. Add a
  NetworkPolicy or a token before using real data.
* **No labels, so no performance monitoring.** Drift is input drift. Real accuracy tracking
  needs ground truth to arrive later and be joined to predictions.
* **Synthetic data**, so absolute metrics say nothing about real energy load.
* **Drift gauges need a Pushgateway**; without it the job still writes JSON/HTML and logs the
  summary. Reports in the CronJob go to the pod's temp directory (read the log line, or mount
  a volume).
* Single-model service: no canary/shadow traffic splitting, no authentication on `/predict`,
  no rate limiting, no tracing.
* HPA needs metrics-server; ServiceMonitor needs the Prometheus Operator CRDs (separate
  overlay, only schema-checked).
