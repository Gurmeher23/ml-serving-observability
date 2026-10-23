"""Batch drift check: reference sample vs recent live requests.

Typical use:

    python -m monitoring.drift_job --reference artifacts/reference.csv \\
        --source http://localhost:8000/captured --out-dir reports

Outputs ``drift_report.json`` and ``drift_report.html`` in ``--out-dir``, prints a one-line
JSON summary to stdout, and (optionally) pushes gauges to a Prometheus Pushgateway, because a
short-lived job cannot be scraped.

Exit codes: 0 ok / not enough data, 2 drift detected and ``--fail-on-drift`` set, 1 error.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from monitoring.drift import build_report, render_html
from training.data import FEATURES


def fetch_current(source: str) -> pd.DataFrame:
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310 (trusted config)
            rows = json.load(resp)["rows"]
        return pd.DataFrame(rows, columns=FEATURES)
    return pd.read_csv(source)[FEATURES]


def push_metrics(report: dict, gateway: str, job: str = "drift_job") -> None:
    registry = CollectorRegistry()
    Gauge("model_drift_share", "Share of features flagged as drifted", registry=registry).set(
        report["drifted_share"]
    )
    Gauge("model_dataset_drift", "1 if dataset drift was detected", registry=registry).set(
        1 if report["dataset_drift"] else 0
    )
    psi_gauge = Gauge("model_feature_psi", "PSI per feature", ["feature"], registry=registry)
    for f in report["features"]:
        psi_gauge.labels(f["feature"]).set(f["psi"])
    Gauge(
        "drift_job_last_run_timestamp_seconds", "Unix time of last run", registry=registry
    ).set_to_current_time()
    push_to_gateway(gateway, job=job, registry=registry)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--reference", default="artifacts/reference.csv")
    p.add_argument("--source", required=True, help="service /captured URL or a CSV path")
    p.add_argument("--out-dir", default="reports")
    p.add_argument("--min-samples", type=int, default=200)
    p.add_argument("--pushgateway", default=None, help="host:port of a Prometheus Pushgateway")
    p.add_argument("--fail-on-drift", action="store_true")
    args = p.parse_args(argv)

    try:
        reference = pd.read_csv(args.reference)[FEATURES]
        current = fetch_current(args.source)
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 1

    if len(current) < args.min_samples:
        print(
            json.dumps(
                {
                    "status": "insufficient_data",
                    "current_rows": len(current),
                    "min_samples": args.min_samples,
                }
            )
        )
        return 0

    report = build_report(reference, current, FEATURES)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(report, indent=2))
    (out / "drift_report.html").write_text(render_html(report))
    if args.pushgateway:
        push_metrics(report, args.pushgateway)

    summary = {
        "status": "ok",
        "dataset_drift": report["dataset_drift"],
        "drifted_share": report["drifted_share"],
        "drifted_features": [f["feature"] for f in report["features"] if f["drifted"]],
        "current_rows": report["current_rows"],
    }
    print(json.dumps(summary))
    return 2 if (args.fail_on_drift and report["dataset_drift"]) else 0


if __name__ == "__main__":
    sys.exit(main())
