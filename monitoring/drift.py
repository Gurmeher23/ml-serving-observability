"""Self-contained feature drift detection: Population Stability Index + two-sample KS test.

Why not Evidently: this keeps the dependency footprint small and, more importantly,
makes every threshold visible and testable. The statistics are standard:

* PSI (industry rule of thumb): < 0.1 stable, 0.1 to 0.2 moderate shift, >= 0.2 major shift.
* Kolmogorov-Smirnov: distribution-free test. With large samples it flags tiny, irrelevant
  differences, so a feature only counts as drifted from KS when the statistic (the effect
  size) is also at least ``ks_min_stat``.
"""

from __future__ import annotations

import html
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-4
PSI_THRESHOLD = 0.2
KS_ALPHA = 0.01
KS_MIN_STAT = 0.1
DATASET_DRIFT_SHARE = 0.3
MAX_DISCRETE = 10


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    drifted: bool
    ref_mean: float
    cur_mean: float


def _proportions(ref: np.ndarray, cur: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.unique(ref)
    if len(values) <= MAX_DISCRETE:
        # Discrete feature: compare category frequencies. Unseen values land in an extra bin.
        edges_ref = np.array([(ref == v).mean() for v in values] + [0.0])
        edges_cur = np.array([(cur == v).mean() for v in values] + [(~np.isin(cur, values)).mean()])
        return edges_ref, edges_cur
    # Continuous feature: decile bins from the reference, open-ended at both tails.
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)))
    edges[0], edges[-1] = -np.inf, np.inf
    return (
        np.histogram(ref, edges)[0] / len(ref),
        np.histogram(cur, edges)[0] / len(cur),
    )


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    p, q = _proportions(np.asarray(ref, float), np.asarray(cur, float))
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def feature_drift(name: str, ref: pd.Series, cur: pd.Series) -> FeatureDrift:
    ref_a, cur_a = ref.to_numpy(float), cur.to_numpy(float)
    score = psi(ref_a, cur_a)
    ks = stats.ks_2samp(ref_a, cur_a)
    drifted = score >= PSI_THRESHOLD or (ks.pvalue < KS_ALPHA and ks.statistic >= KS_MIN_STAT)
    return FeatureDrift(
        name,
        round(score, 4),
        round(float(ks.statistic), 4),
        float(ks.pvalue),
        bool(drifted),
        round(float(ref_a.mean()), 4),
        round(float(cur_a.mean()), 4),
    )


def build_report(reference: pd.DataFrame, current: pd.DataFrame, features: list[str]) -> dict:
    """Compare ``current`` against ``reference`` on ``features``."""
    results = [feature_drift(f, reference[f], current[f]) for f in features]
    share = sum(r.drifted for r in results) / len(results)
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "thresholds": {
            "psi": PSI_THRESHOLD,
            "ks_alpha": KS_ALPHA,
            "ks_min_stat": KS_MIN_STAT,
            "dataset_drift_share": DATASET_DRIFT_SHARE,
        },
        "drifted_share": round(share, 4),
        "dataset_drift": share >= DATASET_DRIFT_SHARE,
        "features": [asdict(r) for r in results],
    }


def render_html(report: dict) -> str:
    rows = "\n".join(
        f'<tr class="{"bad" if f["drifted"] else "ok"}"><td>{html.escape(f["feature"])}</td>'
        f"<td>{f['psi']:.3f}</td><td>{f['ks_stat']:.3f}</td><td>{f['ks_pvalue']:.2e}</td>"
        f"<td>{f['ref_mean']:.2f}</td><td>{f['cur_mean']:.2f}</td>"
        f"<td>{'DRIFT' if f['drifted'] else 'ok'}</td></tr>"
        for f in report["features"]
    )
    verdict = "DATASET DRIFT DETECTED" if report["dataset_drift"] else "No dataset drift"
    t = report["thresholds"]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Drift report</title>
<style>body{{font:14px system-ui;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 10px;text-align:right}}td:first-child{{text-align:left}}
.bad{{background:#fde2e2}}.ok{{background:#e6f4ea}}</style></head><body>
<h1>{verdict}</h1>
<p>Generated {report["generated_at"]}. Reference rows {report["reference_rows"]},
current rows {report["current_rows"]}. Drifted features: {report["drifted_share"]:.0%}
(dataset drift at {t["dataset_drift_share"]:.0%}).</p>
<p>Feature drifts if PSI &ge; {t["psi"]}
or (KS p &lt; {t["ks_alpha"]} and KS stat &ge; {t["ks_min_stat"]}).</p>
<table><tr><th>feature</th><th>PSI</th><th>KS stat</th><th>KS p</th><th>ref mean</th>
<th>current mean</th><th>verdict</th></tr>
{rows}
</table></body></html>"""


def is_finite_report(report: dict) -> bool:
    return all(math.isfinite(f["psi"]) for f in report["features"])
