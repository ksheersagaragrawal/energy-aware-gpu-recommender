"""Train static/reduced feature baselines for Phase 4 external data."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from phase4_external_benchmark import (
    BenchmarkConfig,
    build_prediction_metrics,
    get_default_paths,
    load_gpu_mangrove_dataset,
)


def main() -> None:
    paths = get_default_paths()
    dataset_path = paths.processed_dir / "phase4_gpu_mangrove_power.csv"
    metrics_path = paths.tables_dir / "phase4_prediction_metrics.csv"
    preds_path = paths.tables_dir / "phase4_predictions_gpu_mangrove.csv"

    bundle = load_gpu_mangrove_dataset(dataset_path)
    if bundle is None:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        preds_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_cols = [
            "dataset",
            "target",
            "feature_set",
            "model",
            "backend",
            "n_train",
            "n_test",
            "mae",
            "rmse",
            "r2",
            "mape",
            "train_time_sec",
            "infer_time_sec",
            "notes",
        ]
        preds_cols = [
            "row_id",
            "dataset",
            "target",
            "feature_set",
            "model",
            "backend",
            "split",
            "prediction",
        ]
        pd.DataFrame(columns=metrics_cols).to_csv(metrics_path, index=False)
        pd.DataFrame(columns=preds_cols).to_csv(preds_path, index=False)
        return

    config = BenchmarkConfig()
    metrics_df, preds_df = build_prediction_metrics(bundle, config)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)

    preds_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(preds_path, index=False)


if __name__ == "__main__":
    main()
