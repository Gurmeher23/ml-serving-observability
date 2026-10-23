# syntax=docker/dockerfile:1.7

# ---- builder: install dependencies and train the demo registry ----------------------
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY service ./service
COPY monitoring ./monitoring
COPY training ./training
RUN pip install .

# Demo only: train and register v1/v2 at build time so the image is self-contained.
# The registry path (/app/mlruns) is the same in the runtime stage, because MLflow records
# absolute artifact paths. In a real deployment MLFLOW_TRACKING_URI points at a shared
# tracking server and this step does not exist.
ENV MLFLOW_DISABLE_AGENT_HINT=1
RUN python -m training.train --store-dir /app/mlruns --reference-path /app/reference/reference.csv

# ---- runtime ----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/opt/venv/bin:$PATH" \
    MLFLOW_TRACKING_URI="sqlite:////app/mlruns/mlflow.db" MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /app

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=app:app /app/mlruns /app/mlruns
COPY --from=builder --chown=app:app /app/reference /app/reference
COPY service ./service
COPY monitoring ./monitoring
COPY training ./training

USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"
CMD ["uvicorn", "service.app:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
