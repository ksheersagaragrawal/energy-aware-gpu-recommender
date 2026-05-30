"""Generate clean, slide-ready result figures.

Figures:
1) Power-model MAE across models (TDP vs PSU).
2) Feature-set ablation impact (test MAE) for a selected model.
3) Observability gap diagram (conceptual positioning).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_TDP_METRICS = "data/results/tdp_model_metrics.csv"
DEFAULT_PSU_METRICS = "data/results/psu_model_metrics.csv"
DEFAULT_ABLATION_SUMMARY = "data/results/ablation_power_model_summary.csv"
DEFAULT_OUTPUT_DIR = "results/plots/slide_figures"
DEFAULT_MODEL_FOR_ABLATION = "XGBoost"

PASTEL_COLORS = [
    "#AEC6CF",  # pastel blue
    "#FFB347",  # pastel orange
    "#B39EB5",  # pastel purple
    "#77DD77",  # pastel green
    "#FF6961",  # pastel red
    "#FDFD96",  # pastel yellow
]


def _configure_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    })


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_model_metrics(tdp_path: str, psu_path: str) -> pd.DataFrame:
    tdp_df = pd.read_csv(tdp_path)
    psu_df = pd.read_csv(psu_path)
    tdp_df = tdp_df.assign(target="tdp_w")
    psu_df = psu_df.assign(target="psu_w")
    return pd.concat([tdp_df, psu_df], ignore_index=True)


def plot_power_model_mae(metrics: pd.DataFrame, output_path: Path) -> None:
    order = [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "MLP",
    ]

    tdp = metrics[metrics["target"] == "tdp_w"].set_index("model")
    psu = metrics[metrics["target"] == "psu_w"].set_index("model")

    tdp = tdp.loc[[m for m in order if m in tdp.index]]
    psu = psu.loc[[m for m in order if m in psu.index]]

    models = tdp.index.tolist()
    x = np.arange(len(models))
    width = 0.38

    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=600)
    ax.bar(x - width / 2, tdp["test_mae"], width, label="TDP MAE", color=PASTEL_COLORS[0])
    ax.bar(x + width / 2, psu["test_mae"], width, label="PSU MAE", color=PASTEL_COLORS[1])

    ax.set_title("Power-model accuracy across models")
    ax.set_ylabel("Test MAE (W)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_ablation_feature_sets(
    ablation_summary: pd.DataFrame,
    output_path: Path,
    model_name: str,
) -> None:
    subset = ablation_summary[ablation_summary["model"] == model_name].copy()
    if subset.empty:
        raise ValueError(f"No ablation rows found for model '{model_name}'.")

    feature_order = ["Full", "Clocks-only", "Minimal", "Core-only", "Memory-only"]
    targets = ["tdp_w", "psu_w"]

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2), dpi=600, sharey=False)
    for idx, target in enumerate(targets):
        ax = axes[idx]
        data = subset[subset["target"] == target].set_index("feature_set")
        data = data.loc[[f for f in feature_order if f in data.index]]

        x = np.arange(len(data.index))
        ax.plot(
            x,
            data["test_mae"],
            marker="o",
            linewidth=2.0,
            color=PASTEL_COLORS[2 + idx],
        )
        ax.set_title(f"{model_name} ablation: {target}")
        ax.set_ylabel("Test MAE (W)")
        ax.set_xticks(x)
        ax.set_xticklabels(data.index, rotation=25, ha="right", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_observability_gap(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 2.2), dpi=600)

    ax.hlines(0, 0, 1, color="#CCCCCC", linewidth=3)
    ax.scatter([0.15], [0], s=140, color=PASTEL_COLORS[0], label="Our method")
    ax.scatter([0.8, 0.9, 0.95], [0, 0, 0], s=110, color=PASTEL_COLORS[3], label="Runtime baselines")

    ax.text(0.15, 0.08, "Static specs\n(pre-deployment)", ha="center", fontweight="bold")
    ax.text(0.9, 0.08, "Runtime counters\n/profiling", ha="center", fontweight="bold")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.2, 0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Observability gap vs prior work")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate slide-ready result figures.")
    parser.add_argument("--tdp-metrics", default=DEFAULT_TDP_METRICS)
    parser.add_argument("--psu-metrics", default=DEFAULT_PSU_METRICS)
    parser.add_argument("--ablation-summary", default=DEFAULT_ABLATION_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ablation-model", default=DEFAULT_MODEL_FOR_ABLATION)
    args = parser.parse_args()

    _configure_style()

    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    metrics = load_model_metrics(args.tdp_metrics, args.psu_metrics)
    ablation_summary = pd.read_csv(args.ablation_summary)

    plot_power_model_mae(metrics, output_dir / "figure_power_model_mae.png")
    plot_ablation_feature_sets(
        ablation_summary,
        output_dir / "figure_ablation_feature_sets.png",
        model_name=args.ablation_model,
    )
    plot_observability_gap(output_dir / "figure_observability_gap.png")

    print("Saved figures:")
    print(f"  {output_dir / 'figure_power_model_mae.png'}")
    print(f"  {output_dir / 'figure_ablation_feature_sets.png'}")
    print(f"  {output_dir / 'figure_observability_gap.png'}")


if __name__ == "__main__":
    main()
