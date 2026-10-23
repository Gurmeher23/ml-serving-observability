"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from service.capture import RollingCapture
from service.config import Settings
from service.logging_config import configure_logging
from service.metrics import Metrics
from service.registry import LoadedModel, load_from_registry
from service.schemas import ModelInfo, PredictRequest, PredictResponse
from training.data import FEATURES

log = logging.getLogger("service")

ModelLoader = Callable[[Settings], LoadedModel]


def create_app(
    settings: Settings | None = None, model_loader: ModelLoader = load_from_registry
) -> FastAPI:
    settings = settings or Settings()
    metrics = Metrics()
    capture = RollingCapture(settings.capture_max_rows)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(settings.log_level)
        app.state.model = None
        try:
            model = model_loader(settings)
            app.state.model = model
            metrics.model_version.labels(model.name).set(float(model.version))
            log.info(
                "model loaded",
                extra={"model_name": model.name, "model_version": model.version, "uri": model.uri},
            )
        except Exception:
            # Stay up so /health and /metrics work, but report not-ready (503) to the
            # orchestrator so no traffic is routed here.
            metrics.inference_errors.labels("model_load").inc()
            log.exception("model load failed", extra={"uri": settings.model_uri})
        yield

    app = FastAPI(title="ml-serving-observability", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["x-request-id"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            # Use the route template, never the raw path, to keep label cardinality bounded.
            path = getattr(route, "path", "unmatched")
            metrics.request_latency.labels(request.method, path).observe(elapsed)
            metrics.requests.labels(request.method, path, str(status)).inc()
            if path not in ("/metrics", "/health"):
                log.info(
                    "request",
                    extra={
                        "request_id": request_id,
                        "method": request.method,
                        "path": path,
                        "status": status,
                        "duration_ms": round(elapsed * 1000, 2),
                    },
                )

    @app.get("/health", summary="Liveness: the process is up")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", summary="Readiness: a model is loaded and can serve")
    def ready() -> JSONResponse:
        model: LoadedModel | None = app.state.model
        if model is None:
            return JSONResponse({"status": "model_not_loaded"}, status_code=503)
        return JSONResponse({"status": "ready", "model_version": model.version})

    @app.get("/model", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        model: LoadedModel | None = app.state.model
        if model is None:
            raise HTTPException(503, "model not loaded")
        return ModelInfo(
            model_name=model.name,
            model_version=model.version,
            model_uri=model.uri,
            run_id=model.run_id,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(body: PredictRequest) -> PredictResponse:
        model: LoadedModel | None = app.state.model
        if model is None:
            metrics.inference_errors.labels("not_ready").inc()
            raise HTTPException(503, "model not loaded")
        if len(body.instances) > settings.max_batch_size:
            raise HTTPException(
                413, f"batch of {len(body.instances)} exceeds limit {settings.max_batch_size}"
            )
        rows = [i.model_dump() for i in body.instances]
        frame = pd.DataFrame(rows, columns=FEATURES).astype(float)
        try:
            values = [float(v) for v in model.estimator.predict(frame)]
        except Exception:
            metrics.inference_errors.labels("predict").inc()
            log.exception("inference failed", extra={"batch_size": len(rows)})
            raise HTTPException(500, "inference failed") from None
        metrics.batch_size.observe(len(rows))
        for v in values:
            metrics.prediction_value.observe(v)
        capture.add(rows)
        return PredictResponse(
            model_name=model.name, model_version=model.version, predictions=values
        )

    @app.get("/captured", summary="Recent request features, for the drift job (internal)")
    def captured(limit: int = Query(default=5000, ge=1, le=100_000)) -> dict:
        return {"count": len(capture), "rows": capture.recent(limit)}

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    app.state.metrics = metrics
    app.state.capture = capture
    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn service.app:app_factory --factory``."""
    return create_app()
