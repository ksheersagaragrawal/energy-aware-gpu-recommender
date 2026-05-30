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


def configure_style(mode: str) -> None:
    """Apply readable sizing. DPI controls sharpness, not readable font size."""
    if mode == "slide":
        base, title, label, tick, legend = 13, 16, 14, 12, 11
    else:
        base, title, label, tick, legend = 9, 10, 9, 8, 8

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


def best_row(metrics: pd.DataFrame, target: str) -> pd.Series:
    rows = metrics.loc[metrics["target"].astype(str) == target].copy()
    if rows.empty:
        raise ValueError(f"No rows found for target={target!r}")
    return rows.sort_values("test_mae", ascending=True).iloc[0]


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
    """Show the two prediction targets compactly, with metrics outside the bars."""
    rows = [best_row(metrics, "tdp_w"), best_row(metrics, "psu_w")]
    labels = [
        f"TDP\n{clean_model_name(rows[0]['model'])}",
        f"PSU\n{clean_model_name(rows[1]['model'])}",
    ]
    maes = np.array([float(row["test_mae"]) for row in rows])
    r2_values = np.array([float(row["test_r2"]) for row in rows])

    fig, ax = plt.subplots(figsize=figure_size(mode, "power"), layout="constrained")
    x = np.arange(len(labels))
    bars = ax.bar(x, maes, width=0.52)

    ax.set_title("Best Power Prediction Models")
    ax.set_ylabel("Test MAE (W) ↓")
    ax.set_xticks(x, labels)
    ax.yaxis.grid(True, linewidth=0.7, alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(0, float(maes.max()) * 1.27)

    for bar, mae, r2 in zip(bars, maes, r2_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mae + float(maes.max()) * 0.035,
            f"{mae:.1f} W\n$R^2$ = {r2:.3f}",
            ha="center",
            va="bottom",
        )

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

    cov_bars = ax_cov.barh(y, coverage)
    ax_cov.axvline(0.90, linestyle="--", linewidth=1.0, label="Nominal = 0.90")
    ax_cov.set_xlabel("Empirical coverage")
    ax_cov.set_yticks(y, model_labels)
    ax_cov.set_xlim(0, 1.05)
    ax_cov.xaxis.grid(True, linewidth=0.7, alpha=0.25)
    ax_cov.set_axisbelow(True)
    ax_cov.legend(loc="lower right", frameon=False)

    for bar, value in zip(cov_bars, coverage):
        ax_cov.text(
            min(value + 0.018, 1.01),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha="left",
        )

    width_bars = ax_width.barh(y, widths)
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
    """Use aligned panels rather than a dual-axis overlay."""
    df = ordered_methods(summary_df)
    labels = [short_method_label(value) for value in df["method"].astype(str)]
    ppw = df["avg_ppw"].to_numpy(dtype=float)
    x = np.arange(len(df))
    has_diversity = "top1_share" in df.columns and df["top1_share"].notna().any()

    if has_diversity:
        fig, (ax_ppw, ax_div) = plt.subplots(
            2,
            1,
            figsize=figure_size(mode, "recommender"),
            sharex=True,
            gridspec_kw={"height_ratios": [1.7, 1.05]},
            layout="constrained",
        )
    else:
        fig, ax_ppw = plt.subplots(
            figsize=figure_size(mode, "recommender"), layout="constrained"
        )
        ax_div = None

    fig.suptitle("Recommendation Outcomes", fontweight="semibold")
    bars = ax_ppw.bar(x, ppw, width=0.62)
    ax_ppw.set_ylabel("Average PPW ↑")
    ax_ppw.yaxis.grid(True, linewidth=0.7, alpha=0.25)
    ax_ppw.set_axisbelow(True)
    ax_ppw.set_ylim(0, float(np.nanmax(ppw)) * 1.24)

    best_idx = int(np.nanargmax(ppw))
    for idx, (bar, value) in enumerate(zip(bars, ppw)):
        label = f"{value:.4f}"
        if idx == best_idx:
            label += "\nBest"
        ax_ppw.text(
            bar.get_x() + bar.get_width() / 2,
            value + float(np.nanmax(ppw)) * 0.035,
            label,
            ha="center",
            va="bottom",
        )

    if ax_div is not None:
        diversity = df["top1_share"].to_numpy(dtype=float)
        div_bars = ax_div.bar(x, diversity, width=0.62)
        ax_div.set_ylabel("Top-1 share ↓")
        ax_div.set_xlabel("Recommendation method")
        ax_div.set_xticks(x, labels)
        ax_div.set_ylim(0, 1.05)
        ax_div.yaxis.grid(True, linewidth=0.7, alpha=0.25)
        ax_div.set_axisbelow(True)
        ax_div.text(
            1.0,
            0.96,
            "Lower means more diverse",
            transform=ax_div.transAxes,
            ha="right",
            va="top",
        )
        for bar, value in zip(div_bars, diversity):
            ax_div.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.035,
                f"{value:.2f}",
                ha="center",
                va="bottom",
            )
    else:
        ax_ppw.set_xlabel("Recommendation method")
        ax_ppw.set_xticks(x, labels)

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
