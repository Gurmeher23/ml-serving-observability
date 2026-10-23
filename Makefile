.PHONY: install train test lint serve traffic traffic-drift drift up down kind-note

PY ?= python
export MLFLOW_DISABLE_AGENT_HINT=1

install:
	$(PY) -m pip install -e ".[dev]"

# Starts from a clean registry (re-running would otherwise add v3, v4). Trains two versions and registers them: v1 (Ridge, alias "previous"), v2 (GBM, alias "champion").
train:
	rm -rf mlruns artifacts
	$(PY) -m training.train

lint:
	ruff check . && ruff format --check .

test:
	pytest

serve: train
	MLFLOW_TRACKING_URI=sqlite:///$(CURDIR)/mlruns/mlflow.db uvicorn service.app:app_factory --factory --port 8000

traffic:
	$(PY) scripts/send_traffic.py --rows 400

traffic-drift:
	$(PY) scripts/send_traffic.py --rows 400 --scenario heatwave

drift:
	$(PY) -m monitoring.drift_job --source http://localhost:8000/captured --out-dir reports

up:
	docker compose up --build -d

down:
	docker compose down -v
