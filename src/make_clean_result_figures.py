"""Create clean, publication-ready result figures for the GPU recommender project.

Outputs both vector PDF and high-resolution PNG versions of:
1) power model accuracy,
2) uncertainty quality,
3) recommendation outcomes.

Example
-------
python make_clean_result_figures.py --mode paper
python make_clean_result_figures.py --mode slide
python make_clean_result_figures.py --mode paper --uq-target tdp_w
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_TDP_METRICS = "data/results/tdp_model_metrics.csv"
DEFAULT_PSU_METRICS = "data/results/psu_model_metrics.csv"
DEFAULT_METHOD_SUMMARY = "data/results/passmark_method_comparison_summary.csv"
DEFAULT_METHOD_SUMMARY_FALLBACK = "data/results/passmark_recommender_summary.csv"
DEFAULT_OUTPUT_DIR = "results/plots/clean_figures"

PALETTE = {
    "primary": "#9DB7D5",
    "secondary": "#B8D8C4",
    "accent": "#E6C8A8",
    "muted": "#CFCFEA",
}


def configure_style(mode: str) -> None:
    """Apply readable sizing. DPI controls sharpness, not readable font size."""
    if mode == "slide":
        base, title, label, tick, legend = 7, 9, 8, 7, 7
    else:
        base, title, label, tick, legend = 8, 9, 8, 7, 7

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": base,
            "axes.titlesize": title,
            "axes.titleweight": "semibold",
            "axes.labelsize": label,
            "xtick.labelsize": tick,
            "ytick.labelsize": tick,
            "legend.fontsize": legend,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 120,
            "savefig.transparent": False,
        }
    )


def figure_size(mode: str, kind: str) -> tuple[float, float]:
    """Return dimensions suited to a full-width paper figure or a slide."""
    if mode == "slide":
        return {
            "power": (8.0, 3.6),
            "uq": (9.0, 4.0),
            "recommender": (9.0, 5.8),
        }[kind]
    return {
        "power": (5.8, 2.6),
        "uq": (6.8, 2.8),
        "recommender": (6.8, 4.2),
    }[kind]


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    """Save a vector version for papers and a PNG for previews or slides."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def read_power_metrics(tdp_path: str, psu_path: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path, target in [(tdp_path, "tdp_w"), (psu_path, "psu_w")]:
        df = pd.read_csv(path)
        if "target" not in df.columns:
            df = df.assign(target=target)
        frames.append(df)
    metrics = pd.concat(frames, ignore_index=True)

    required = {"target", "model", "test_mae", "test_r2"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Power metrics are missing columns: {sorted(missing)}")
    return metrics


def clean_model_name(name: object) -> str:
    rename = {
        "Gradient Boosting": "Gradient Boosting",
        "Bayesian Ridge": "Bayesian Ridge",
        "Gaussian Process": "Gaussian Process",
        "Quantile XGBoost": "Quantile XGBoost",
        "QXGB_Q0.05": "QXGB",
        "QXGB_Q0.50": "QXGB",
        "QXGB_Q0.95": "QXGB",
    }
    value = str(name)
    return rename.get(value, value)


def plot_power_quality(metrics: pd.DataFrame, output_stem: Path, mode: str) -> None:
    """Show MAE for all evaluated models across TDP/PSU targets."""
    plot_df = metrics.copy()
    plot_df["model_label"] = plot_df["model"].map(clean_model_name)
    plot_df["target_label"] = plot_df["target"].astype(str).map({"tdp_w": "TDP", "psu_w": "PSU"})
    plot_df = plot_df.dropna(subset=["model_label", "target_label", "test_mae"]).copy()
    if plot_df.empty:
        raise ValueError("No valid rows available for power-model plotting.")

    order = (
        plot_df.groupby("model_label", as_index=False)["test_mae"]
        .mean()
        .sort_values("test_mae", ascending=True)["model_label"]
        .tolist()
    )
    pivot = (
        plot_df.pivot_table(index="model_label", columns="target_label", values="test_mae", aggfunc="mean")
        .reindex(order)
    )

    fig, ax = plt.subplots(figsize=figure_size(mode, "power"), layout="constrained")
    x = np.arange(len(pivot.index))
    width = 0.38
    tdp_vals = pivot.get("TDP", pd.Series(index=pivot.index, dtype=float)).to_numpy(dtype=float)
    psu_vals = pivot.get("PSU", pd.Series(index=pivot.index, dtype=float)).to_numpy(dtype=float)
    tdp_bars = ax.bar(x - width / 2, tdp_vals, width=width, color=PALETTE["primary"], label="TDP")
    psu_bars = ax.bar(x + width / 2, psu_vals, width=width, color=PALETTE["secondary"], label="PSU")

    ax.set_title("Power Model Quality Across All Tested Models")
    ax.set_ylabel("Test MAE (W) ↓")
    ax.set_xticks(x, pivot.index.tolist(), rotation=22, ha="right")
    ax.yaxis.grid(True, linewidth=0.7, alpha=0.25)
    ax.set_axisbelow(True)
    max_mae = np.nanmax(np.concatenate([tdp_vals, psu_vals]))
    ax.set_ylim(0, float(max_mae) * 1.24)

    def annotate_bars(bars: Iterable, values: np.ndarray) -> None:
        for bar, val in zip(bars, values):
            if np.isnan(val):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + float(max_mae) * 0.02,
                f"{val:.1f}",
                ha="center",
                va="bottom",
            )

    annotate_bars(tdp_bars, tdp_vals)
    annotate_bars(psu_bars, psu_vals)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)

    save_figure(fig, output_stem)


def build_uq_table(metrics: pd.DataFrame, uq_target: str) -> pd.DataFrame:
    required = {"model", "coverage_90", "mean_interval_width"}
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(f"Uncertainty metrics are missing columns: {sorted(missing)}")

    uq = metrics.dropna(subset=["coverage_90", "mean_interval_width"]).copy()
    if uq_target != "all":
        uq = uq.loc[uq["target"].astype(str) == uq_target].copy()
    if uq.empty:
        raise ValueError(f"No uncertainty rows found for uq_target={uq_target!r}.")

    agg = (
        uq.groupby("model", as_index=False, observed=True)[["coverage_90", "mean_interval_width"]]
        .mean()
        .copy()
    )
    agg["model_label"] = agg["model"].map(clean_model_name)

    preferred = ["Bayesian Ridge", "Gaussian Process", "Quantile XGBoost", "QXGB"]
    rank = {name: idx for idx, name in enumerate(preferred)}
    agg["_order"] = agg["model_label"].map(lambda value: rank.get(value, len(rank)))
    return agg.sort_values(["_order", "model_label"]).reset_index(drop=True)


def plot_uncertainty_quality(
    metrics: pd.DataFrame, output_stem: Path, mode: str, uq_target: str
) -> None:
    """Separate coverage and width so readers do not infer that higher coverage is always better."""
    agg = build_uq_table(metrics, uq_target)
    y = np.arange(len(agg))
    title_suffix = "" if uq_target == "all" else f" ({uq_target})"

    fig, (ax_cov, ax_width) = plt.subplots(
        1,
        2,
        figsize=figure_size(mode, "uq"),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1]},
        layout="constrained",
    )
    fig.suptitle(f"90% Prediction Interval Quality{title_suffix}", fontweight="semibold")

    coverage = agg["coverage_90"].to_numpy(dtype=float)
    widths = agg["mean_interval_width"].to_numpy(dtype=float)
    model_labels = agg["model_label"].tolist()

    cov_bars = ax_cov.barh(y, coverage, color=PALETTE["primary"])
    ax_cov.axvline(0.90, linestyle="--", linewidth=1.0, color=PALETTE["accent"], label="Nominal=0.90")
    ax_cov.set_xlabel("Empirical coverage")
    ax_cov.set_yticks(y, model_labels)
    ax_cov.set_xlim(0, 1.05)
    ax_cov.xaxis.grid(True, linewidth=0.7, alpha=0.25)
    ax_cov.set_axisbelow(True)
    ax_cov.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    for bar, value in zip(cov_bars, coverage):
        ax_cov.text(
            min(value + 0.018, 1.01),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
        )

    width_bars = ax_width.barh(y, widths, color=PALETTE["secondary"])
    ax_width.set_xlabel("Mean interval width (W) ↓")
    ax_width.xaxis.grid(True, linewidth=0.7, alpha=0.25)
    ax_width.set_axisbelow(True)
    ax_width.tick_params(axis="y", left=False, labelleft=False)
    ax_width.set_xlim(0, float(widths.max()) * 1.23)

    for bar, value in zip(width_bars, widths):
        ax_width.text(
            value + float(widths.max()) * 0.03,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            ha="left",
        )

    ax_cov.invert_yaxis()
    save_figure(fig, output_stem)


def read_method_summary(primary_path: str, fallback_path: str) -> pd.DataFrame:
    for path in (Path(primary_path), Path(fallback_path)):
        if path.exists():
            df = pd.read_csv(path)
            break
    else:
        raise FileNotFoundError(
            f"Could not find either method-summary file: {primary_path}, {fallback_path}"
        )

    if "avg_ppw" not in df.columns:
        if "avg_score_per_watt" in df.columns:
            df["avg_ppw"] = df["avg_score_per_watt"]
        else:
            raise ValueError("Method summary must include avg_ppw or avg_score_per_watt.")
    if "method" not in df.columns:
        raise ValueError("Method summary must include a method column.")
    return df


def ordered_methods(df: pd.DataFrame) -> pd.DataFrame:
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
    present = list(dict.fromkeys(df["method"].astype(str).tolist()))
    order = [method for method in preferred if method in present]
    order.extend(method for method in present if method not in order)
    out = df.copy()
    out["method"] = pd.Categorical(out["method"].astype(str), categories=order, ordered=True)
    return out.sort_values("method").reset_index(drop=True)


def short_method_label(method: object) -> str:
    labels = {
        "LTR-Utility-Top5": "LTR",
        "ML-Utility-Top5": "ML-U",
        "UtilityFormula-Top5": "Formula",
        "Power-Top5": "Power",
        "KNN50-Feasible-PPW-Top5": "KNN+PPW",
        "KNN50-Feasible": "KNN",
        "PassMark G3D": "PassMark",
        "Proxy perf_score": "Proxy",
    }
    value = str(method)
    return labels.get(value, value)


def plot_recommender_outcomes(summary_df: pd.DataFrame, output_stem: Path, mode: str) -> None:
    """Make the efficiency/diversity trade-off explicit and easy to infer."""
    df = ordered_methods(summary_df)
    labels = [short_method_label(value) for value in df["method"].astype(str)]
    ppw = df["avg_ppw"].to_numpy(dtype=float)
    has_diversity = "top1_share" in df.columns and df["top1_share"].notna().any()

    if has_diversity:
        fig, (ax_rank, ax_trade) = plt.subplots(
            1,
            2,
            figsize=figure_size(mode, "recommender"),
            gridspec_kw={"width_ratios": [1.25, 1.0]},
            layout="constrained",
        )
    else:
        fig, ax_rank = plt.subplots(figsize=figure_size(mode, "recommender"), layout="constrained")
        ax_trade = None

    fig.suptitle("Recommendation Outcomes: Efficiency vs Diversity", fontweight="semibold")

    # Panel A: PPW ranking with delta vs KNN baseline.
    order_idx = np.argsort(-ppw)
    rank_labels = [labels[i] for i in order_idx]
    rank_ppw = ppw[order_idx]
    y = np.arange(len(rank_ppw))
    colors = [PALETTE["primary"] if i == 0 else PALETTE["secondary"] for i in range(len(rank_ppw))]
    bars = ax_rank.barh(y, rank_ppw, color=colors)
    ax_rank.set_yticks(y, rank_labels)
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("Average PPW (higher is better)")
    ax_rank.xaxis.grid(True, linewidth=0.7, alpha=0.25)
    ax_rank.set_axisbelow(True)

    baseline_idx = None
    for i, method_name in enumerate(df["method"].astype(str)):
        if method_name == "KNN50-Feasible":
            baseline_idx = i
            break
    baseline = ppw[baseline_idx] if baseline_idx is not None else np.nan

    max_ppw = float(np.nanmax(rank_ppw))
    ax_rank.set_xlim(0, max_ppw * 1.32)
    for i, (bar, value) in enumerate(zip(bars, rank_ppw)):
        txt = f"{value:.4f}"
        if i == 0:
            txt += "  (best)"
        if np.isfinite(baseline) and baseline > 0:
            gain = (value / baseline - 1.0) * 100.0
            txt += f"\n{gain:+.0f}% vs KNN"
        ax_rank.text(
            value + max_ppw * 0.03,
            bar.get_y() + bar.get_height() / 2,
            txt,
            va="center",
            ha="left",
        )
    ax_rank.set_title("A) Efficiency Ranking")

    # Panel B: trade-off map.
    if ax_trade is not None:
        diversity = df["top1_share"].to_numpy(dtype=float)
        x_vals = diversity
        y_vals = ppw

        ax_trade.scatter(
            x_vals,
            y_vals,
            s=58,
            color=PALETTE["muted"],
            edgecolor="white",
            linewidth=0.7,
        )
        for x_val, y_val, label in zip(x_vals, y_vals, labels):
            ax_trade.text(x_val + 0.01, y_val, label, va="center", ha="left")

        med_x = float(np.nanmedian(x_vals))
        med_y = float(np.nanmedian(y_vals))
        ax_trade.axvline(med_x, color=PALETTE["accent"], linestyle="--", linewidth=0.9)
        ax_trade.axhline(med_y, color=PALETTE["accent"], linestyle="--", linewidth=0.9)
        ax_trade.text(
            0.03,
            0.97,
            "Better zone:\nhigh PPW + low top-1 share",
            transform=ax_trade.transAxes,
            ha="left",
            va="top",
        )
        ax_trade.set_xlabel("Top-1 share (lower is more diverse)")
        ax_trade.set_ylabel("Average PPW")
        ax_trade.grid(True, linewidth=0.7, alpha=0.25)
        ax_trade.set_axisbelow(True)
        ax_trade.set_title("B) Trade-off Map")

    save_figure(fig, output_stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate clean result figures.")
    parser.add_argument("--tdp-metrics", default=DEFAULT_TDP_METRICS)
    parser.add_argument("--psu-metrics", default=DEFAULT_PSU_METRICS)
    parser.add_argument("--method-summary", default=DEFAULT_METHOD_SUMMARY)
    parser.add_argument("--method-summary-fallback", default=DEFAULT_METHOD_SUMMARY_FALLBACK)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=["paper", "slide"],
        default="paper",
        help="Use paper for report figures or slide for presentation figures.",
    )
    parser.add_argument(
        "--uq-target",
        choices=["all", "tdp_w", "psu_w"],
        default="all",
        help="Plot pooled uncertainty metrics or restrict uncertainty to one target.",
    )
    args = parser.parse_args()

    configure_style(args.mode)
    output_dir = Path(args.output_dir)

    metrics = read_power_metrics(args.tdp_metrics, args.psu_metrics)
    method_summary = read_method_summary(args.method_summary, args.method_summary_fallback)

    plot_power_quality(metrics, output_dir / "figure_power_model_quality", args.mode)
    plot_uncertainty_quality(
        metrics,
        output_dir / "figure_uncertainty_quality",
        args.mode,
        args.uq_target,
    )
    plot_recommender_outcomes(
        method_summary,
        output_dir / "figure_recommender_outcomes",
        args.mode,
    )

    print(f"Saved PDF and PNG figures to: {output_dir}")


if __name__ == "__main__":
    main()
