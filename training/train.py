"""Train two model versions and register them in the MLflow Model Registry.

``make train`` runs this module (SQLite-backed registry in ``mlruns/``).
It registers ``energy-load-model``:

* v1: Ridge regression baseline (alias ``previous``)
* v2: gradient boosting (alias ``champion``)

The service resolves ``models:/energy-load-model@champion`` by default, so rollback
is a registry/config change, not a rebuild. The reference sample used for drift
detection is written to ``artifacts/reference.csv`` and logged with the v2 run.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.data import FEATURES, TARGET, make_energy_dataset

log = logging.getLogger("training")

MODEL_NAME = "energy-load-model"
EXPERIMENT = "energy-load"
TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]


@dataclass(frozen=True)
class TrainedVersion:
    version: int
    run_id: str
    metrics: dict[str, float]


def _candidates(seed: int):
    return [
        ("ridge-baseline", make_pipeline(StandardScaler(), Ridge(alpha=1.0)), "previous"),
        (
            "gradient-boosting",
            GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.08, subsample=0.8, random_state=seed
            ),
            "champion",
        ),
    ]


def sqlite_uri(store_dir: Path) -> str:
    """SQLite tracking/registry URI inside ``store_dir`` (MLflow's file store is deprecated)."""
    return f"sqlite:///{store_dir.resolve() / 'mlflow.db'}"


def run(
    store_dir: Path,
    reference_path: Path | None = Path("artifacts/reference.csv"),
    n_rows: int = 5000,
    seed: int = 42,
) -> list[TrainedVersion]:
    """Train and register both versions. Returns the registered versions in order."""
    store_dir = store_dir.resolve()
    store_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = sqlite_uri(store_dir)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient()
    if client.get_experiment_by_name(EXPERIMENT) is None:
        client.create_experiment(EXPERIMENT, artifact_location=(store_dir / "artifacts").as_uri())
    mlflow.set_experiment(EXPERIMENT)

    df = make_energy_dataset(n=n_rows, seed=seed)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=seed)
    x_train, y_train = train_df[FEATURES], train_df[TARGET]
    x_test, y_test = test_df[FEATURES], test_df[TARGET]

    results: list[TrainedVersion] = []
    for run_name, estimator, alias in _candidates(seed):
        with mlflow.start_run(run_name=run_name) as active:
            estimator.fit(x_train, y_train)
            pred = estimator.predict(x_test)
            metrics = {
                "mae": float(mean_absolute_error(y_test, pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            }
            mlflow.log_params({"model": run_name, "n_train": len(x_train), "seed": seed})
            mlflow.log_metrics(metrics)
            mlflow.set_tag("data", "synthetic energy load, see training/data.py")
            if reference_path is not None and alias == "champion":
                reference_path.parent.mkdir(parents=True, exist_ok=True)
                x_train.sample(n=min(2000, len(x_train)), random_state=seed).to_csv(
                    reference_path, index=False
                )
                mlflow.log_artifact(str(reference_path), artifact_path="reference")
            signature = infer_signature(x_train.astype(float), pred)
            info = mlflow.sklearn.log_model(
                estimator,
                name="model",
                signature=signature,
                input_example=x_train.head(3).astype(float),
                registered_model_name=MODEL_NAME,
                # MLflow 3 serialises sklearn models with skops, which refuses unreviewed
                # types. The tree node storage is the one type our own models need.
                skops_trusted_types=TRUSTED_TYPES,
            )
            version = int(info.registered_model_version)
            client.set_registered_model_alias(MODEL_NAME, alias, str(version))
            results.append(TrainedVersion(version, active.info.run_id, metrics))
            log.info("registered %s v%s alias=%s metrics=%s", MODEL_NAME, version, alias, metrics)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", default="mlruns", help="SQLite DB + artifacts live here")
    parser.add_argument("--reference-path", default="artifacts/reference.csv")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    versions = run(Path(args.store_dir), Path(args.reference_path), args.rows, args.seed)
    print(json.dumps([v.__dict__ for v in versions], indent=2))


if __name__ == "__main__":
    main()
