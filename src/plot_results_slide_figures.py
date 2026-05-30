"""Generate high-resolution, slide-ready result figures (single-slide friendly).

Figures:
1) Power model quality (best TDP/PSU models, MAE with R2 annotations).
2) Uncertainty quality (90% interval coverage and interval width).
3) Recommendation outcomes (efficiency + diversity).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_TDP_METRICS = "data/results/tdp_model_metrics.csv"
DEFAULT_PSU_METRICS = "data/results/psu_model_metrics.csv"
DEFAULT_METHOD_SUMMARY = "data/results/passmark_method_comparison_summary.csv"
DEFAULT_METHOD_SUMMARY_FALLBACK = "data/results/passmark_recommender_summary.csv"
DEFAULT_OUTPUT_DIR = "results/plots/slide_figures"

PASTEL_COLORS = [
    "#9FB9C7",  # muted pastel blue
    "#D8B79A",  # muted pastel sand
    "#B9A9C9",  # muted pastel lavender
    "#9DBEA8",  # muted pastel green
    "#D6A6A6",  # muted pastel rose
    "#CFC9A9",  # muted pastel khaki
]


def _configure_style() -> None:
    plt.rcParams.update({
        "font.size": 12,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
    })


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, output_path: Path) -> None:
    # High resolution for slide compression robustness.
    fig.savefig(output_path, dpi=900, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def load_power_metrics(tdp_path: str, psu_path: str) -> pd.DataFrame:
    tdp_df = pd.read_csv(tdp_path)
    psu_df = pd.read_csv(psu_path)
    if "target" not in tdp_df.columns:
        tdp_df = tdp_df.assign(target="tdp_w")
    if "target" not in psu_df.columns:
        psu_df = psu_df.assign(target="psu_w")
    return pd.concat([tdp_df, psu_df], ignore_index=True)


def _best_row_for_target(metrics: pd.DataFrame, target: str) -> pd.Series:
    sub = metrics[metrics["target"] == target].copy()
    if sub.empty:
        raise ValueError(f"No rows found for target '{target}'")
    return sub.sort_values("test_mae", ascending=True).iloc[0]


def plot_power_quality(metrics: pd.DataFrame, output_path: Path) -> None:
    tdp_best = _best_row_for_target(metrics, "tdp_w")
    psu_best = _best_row_for_target(metrics, "psu_w")

    labels = [
        f"TDP\n{tdp_best['model']}",
        f"PSU\n{psu_best['model']}",
    ]
    maes = [float(tdp_best["test_mae"]), float(psu_best["test_mae"])]
    r2s = [float(tdp_best["test_r2"]), float(psu_best["test_r2"])]

    fig, ax = plt.subplots(figsize=(4.1, 3.1))
    bars = ax.bar(labels, maes, color=[PASTEL_COLORS[0], PASTEL_COLORS[1]], width=0.62)
    ax.set_title("Best Power Model Accuracy")
    ax.set_ylabel("Test MAE (W)")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, mae, r2 in zip(bars, maes, r2s):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"MAE {mae:.1f}\nR2 {r2:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
            fontweight="bold",
        )

    fig.tight_layout()
    _save(fig, output_path)


def plot_uncertainty_quality(metrics: pd.DataFrame, output_path: Path) -> None:
    uq = metrics.dropna(subset=["coverage_90", "mean_interval_width"]).copy()
    if uq.empty:
        raise ValueError("No UQ rows with coverage/interval-width found in metrics.")

    agg = (
        uq.groupby("model", as_index=False)[["coverage_90", "mean_interval_width"]]
        .mean()
        .sort_values("coverage_90", ascending=False)
        .head(5)
    )

    x = np.arange(len(agg))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(4.2, 3.1))
    ax2 = ax1.twinx()

    b1 = ax1.bar(x - width / 2, agg["coverage_90"], width, color=PASTEL_COLORS[2], label="Coverage@90")
    b2 = ax2.bar(x + width / 2, agg["mean_interval_width"], width, color=PASTEL_COLORS[3], label="Interval Width")

    ax1.set_title("Uncertainty Quality")
    ax1.set_ylabel("Empirical Coverage (target=0.90)")
    ax2.set_ylabel("Mean Interval Width (W)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg["model"], rotation=22, ha="right", fontweight="bold")
    ax1.grid(axis="y", alpha=0.22)
    ax1.axhline(0.90, color="#666666", linestyle="--", linewidth=1.2)
    ax1.set_axisbelow(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right")

    for bar in b1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.2f}",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")

    fig.tight_layout()
    _save(fig, output_path)


def _method_order(df: pd.DataFrame) -> List[str]:
    preferred = [
        "LTR-Utility-Top5",
        "ML-Utility-Top5",
        "UtilityFormula-Top5",
        "Power-Top5",
        "KNN50-Feasible-PPW-Top5",
        "KNN50-Feasible",
        "PassMark G3D",
        "Proxy perf_score",
    ]
    present = set(df["method"].astype(str))
    out = [m for m in preferred if m in present]
    leftovers = [m for m in df["method"].astype(str).tolist() if m not in out]
    return out + leftovers


def load_method_summary(primary_path: str, fallback_path: str) -> pd.DataFrame:
    primary = Path(primary_path)
    fallback = Path(fallback_path)
    if primary.exists():
        return pd.read_csv(primary)
    if fallback.exists():
        return pd.read_csv(fallback)
    raise FileNotFoundError(f"Neither method summary file exists: {primary} or {fallback}")


def plot_recommender_outcomes(summary_df: pd.DataFrame, output_path: Path) -> None:
    df = summary_df.copy()
    if "avg_ppw" not in df.columns:
        if "avg_score_per_watt" in df.columns:
            df["avg_ppw"] = df["avg_score_per_watt"]
        else:
            raise ValueError("No avg_ppw/avg_score_per_watt column in method summary.")

    has_diversity = "top1_share" in df.columns
    order = _method_order(df)
    df["method"] = pd.Categorical(df["method"], categories=order, ordered=True)
    df = df.sort_values("method")

    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(4.3, 3.1))
    ax1.bar(x, df["avg_ppw"], color=PASTEL_COLORS[4], width=0.62)
    ax1.set_title("Recommendation Outcomes")
    ax1.set_ylabel("Average PPW")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["method"], rotation=25, ha="right", fontweight="bold")
    ax1.grid(axis="y", alpha=0.24)
    ax1.set_axisbelow(True)

    if has_diversity:
        ax2 = ax1.twinx()
        ax2.plot(x, df["top1_share"], marker="o", color=PASTEL_COLORS[0], linewidth=2.0, label="Top-1 Share")
        ax2.set_ylabel("Top-1 Share (lower better)")
        ax2.set_ylim(0, min(1.0, float(np.nanmax(df["top1_share"]) * 1.15)))
        ax2.legend(frameon=False, loc="upper right")

    best_idx = int(np.nanargmax(df["avg_ppw"].values))
    ax1.text(
        x[best_idx],
        df["avg_ppw"].iloc[best_idx],
        "Best PPW",
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
    )

    fig.tight_layout()
    _save(fig, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-resolution slide result figures.")
    parser.add_argument("--tdp-metrics", default=DEFAULT_TDP_METRICS)
    parser.add_argument("--psu-metrics", default=DEFAULT_PSU_METRICS)
    parser.add_argument("--method-summary", default=DEFAULT_METHOD_SUMMARY)
    parser.add_argument("--method-summary-fallback", default=DEFAULT_METHOD_SUMMARY_FALLBACK)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    _configure_style()

    output_dir = Path(args.output_dir)
    _ensure_output_dir(output_dir)

    power_metrics = load_power_metrics(args.tdp_metrics, args.psu_metrics)
    method_summary = load_method_summary(args.method_summary, args.method_summary_fallback)

    plot_power_quality(power_metrics, output_dir / "figure_power_model_quality.png")
    plot_uncertainty_quality(power_metrics, output_dir / "figure_uncertainty_quality.png")
    plot_recommender_outcomes(method_summary, output_dir / "figure_recommender_outcomes.png")

    print("Saved figures:")
    print(f"  {output_dir / 'figure_power_model_quality.png'}")
    print(f"  {output_dir / 'figure_uncertainty_quality.png'}")
    print(f"  {output_dir / 'figure_recommender_outcomes.png'}")


if __name__ == "__main__":
    main()
