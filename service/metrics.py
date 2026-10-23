"""Prometheus instruments. One registry per app instance keeps tests isolated."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
# Load in MW for the demo model; adjust the buckets with the target's range.
PREDICTION_BUCKETS = tuple(range(30_000, 80_001, 5_000))
BATCH_BUCKETS = (1, 2, 5, 10, 25, 50, 100, 250, 500, 1000)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.request_latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency by route",
            ["method", "path"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.requests = Counter(
            "http_requests_total",
            "HTTP requests by route and status code",
            ["method", "path", "status"],
            registry=self.registry,
        )
        self.prediction_value = Histogram(
            "model_prediction_value",
            "Distribution of predicted values (watch for shifts)",
            buckets=PREDICTION_BUCKETS,
            registry=self.registry,
        )
        self.batch_size = Histogram(
            "model_predict_batch_size",
            "Rows per /predict request",
            buckets=BATCH_BUCKETS,
            registry=self.registry,
        )
        self.model_version = Gauge(
            "model_version",
            "Registry version of the model currently served",
            ["model_name"],
            registry=self.registry,
        )
        self.inference_errors = Counter(
            "model_inference_errors_total",
            "Inference failures by reason",
            ["reason"],
            registry=self.registry,
        )
