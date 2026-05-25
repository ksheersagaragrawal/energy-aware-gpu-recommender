"""Run recommendation-style benchmarks for Phase 4."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from phase4_external_benchmark import (
    BenchmarkConfig,
    get_default_paths,
    load_gpu_mangrove_dataset,
    run_recommendation_benchmark,
)


def main() -> None:
    paths = get_default_paths()
    dataset_path = paths.processed_dir / "phase4_gpu_mangrove_power.csv"
    preds_path = paths.tables_dir / "phase4_predictions_gpu_mangrove.csv"
    output_path = paths.tables_dir / "phase4_recommendation_metrics.csv"

    bundle = load_gpu_mangrove_dataset(dataset_path)
    if bundle is None or not preds_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "dataset",
            "feature_set",
            "method",
            "ndcg_at_5",
            "recall_at_5",
            "feasible_hit_rate",
            "mean_true_ppw_at_5",
            "mean_true_power_at_5",
            "mean_true_perf_at_5",
            "efficiency_regret_at_5",
            "top1_share",
            "unique_candidates",
        ]).to_csv(output_path, index=False)
        return

    preds_df = pd.read_csv(preds_path)
    if preds_df.empty:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=[
            "dataset",
            "feature_set",
            "method",
            "ndcg_at_5",
            "recall_at_5",
            "feasible_hit_rate",
            "mean_true_ppw_at_5",
            "mean_true_power_at_5",
            "mean_true_perf_at_5",
            "efficiency_regret_at_5",
            "top1_share",
            "unique_candidates",
        ]).to_csv(output_path, index=False)
        return

    config = BenchmarkConfig()
    results_df = run_recommendation_benchmark(bundle, preds_df, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
