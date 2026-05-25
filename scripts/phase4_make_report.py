"""Create Phase 4 report and plots."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from phase4_external_benchmark import get_default_paths
from phase4_reporting import (
    build_latex_insert,
    build_summary_report,
    plot_prediction_metrics,
    plot_pred_vs_true,
    plot_recommendation_metrics,
)


def main() -> None:
    paths = get_default_paths()
    report_dir = paths.repo_root / "results" / "reports"
    plots_dir = paths.repo_root / "results" / "plots"

    report_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_prediction_metrics(
        paths.tables_dir / "phase4_prediction_metrics.csv",
        plots_dir / "phase4_prediction_metrics_bar.png",
    )
    plot_recommendation_metrics(
        paths.tables_dir / "phase4_recommendation_metrics.csv",
        plots_dir / "phase4_recommendation_metrics_bar.png",
    )
    plot_pred_vs_true(
        paths.processed_dir / "phase4_gpu_mangrove_power.csv",
        paths.tables_dir / "phase4_predictions_gpu_mangrove.csv",
        plots_dir / "phase4_pred_vs_true_power.png",
    )

    build_summary_report(paths, report_dir / "phase4_summary.md")
    build_latex_insert(report_dir / "phase4_latex_insert.tex")


if __name__ == "__main__":
    main()
