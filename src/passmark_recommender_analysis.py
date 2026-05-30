"""Compare proxy perf_score vs PassMark G3D-based ranking.

Generates per-game and aggregate tables plus a few clean plots to support
report insights on whether a benchmark-backed signal changes recommendations.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from recommender import GAME_VECTORS, GPU_VECTORS, SOFT_THRESHOLD
from recommender import build_gpu_features_for_ml, load_ml_model

PREDICTIONS_PATH = "data/results/gpu_power_predictions.csv"
TDP_METRICS_PATH = "data/results/tdp_model_metrics.csv"
PSU_METRICS_PATH = "data/results/psu_model_metrics.csv"

TDP_USED_COL = "tdp_w_used"
PSU_USED_COL = "psu_w_used"


PASTEL_COLORS = [
    "#AEC6CF",  # pastel blue
    "#FFB347",  # pastel orange
    "#B39EB5",  # pastel purple
    "#77DD77",  # pastel green
]

DEFAULT_OUTPUT_DIR = "data/results"
DEFAULT_PLOTS_DIR = "results/plots/passmark_analysis"


@dataclass(frozen=True)
class Config:
    mode: str = "recom"
    k_top: int = 5
    soft_threshold: float = 0.80


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


def hard_filter(gpus: pd.DataFrame, game_row: pd.Series) -> pd.DataFrame:
    mask = pd.Series(True, index=gpus.index)
    vram_req = game_row.get("min_vram_mb")
    if pd.notna(vram_req) and vram_req > 0:
        mask &= gpus["memory_mb"] >= vram_req

    dx_req = game_row.get("min_direct_x")
    if pd.notna(dx_req) and dx_req > 0:
        mask &= gpus["direct_x"] >= dx_req

    return gpus[mask].copy()


def soft_filter(gpus: pd.DataFrame, game_row: pd.Series, threshold: float) -> pd.DataFrame:
    mask = pd.Series(True, index=gpus.index)
    for col in ["texture_rate", "pixel_rate", "memory_bandwidth_gbs", "tmus", "rops"]:
        req = game_row.get(col)
        if pd.isna(req) or req <= 0:
            continue
        min_val = req * threshold
        mask &= gpus[col] >= min_val
    return gpus[mask].copy()


def _rank_topk(df: pd.DataFrame, score_col: str, tdp_col: str, k: int) -> pd.DataFrame:
    df = df.copy()
    df["score_per_watt"] = np.where(df[tdp_col] > 0, df[score_col] / df[tdp_col], np.nan)
    ranked = df.sort_values("score_per_watt", ascending=False)
    return ranked.head(k).copy()


def _compute_metrics(topk: pd.DataFrame, score_col: str) -> Dict[str, float]:
    if topk.empty:
        return {
            "avg_tdp": np.nan,
            "avg_psu": np.nan,
            "avg_score": np.nan,
            "avg_score_per_watt": np.nan,
        }
    return {
        "avg_tdp": topk[TDP_USED_COL].mean(),
        "avg_psu": topk[PSU_USED_COL].mean(),
        "avg_score": topk[score_col].mean(),
        "avg_score_per_watt": topk["score_per_watt"].mean(),
    }


def _top1_share(names: pd.Series) -> float:
    top1 = names.dropna().apply(lambda v: v.split(", ")[0] if v else "")
    if top1.empty:
        return np.nan
    return float(top1.value_counts(normalize=True).iloc[0])


def load_data(mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(GAME_VECTORS[mode])
    gpus = pd.read_csv(GPU_VECTORS)
    gpus = _attach_power_predictions(gpus)
    return games, gpus


def attach_passmark_predictions(gpus: pd.DataFrame) -> pd.DataFrame:
    payload = load_ml_model()
    feature_cols = payload["feature_cols"]
    mem_type_cols = payload["mem_type_cols"]
    model = payload["model"]

    X = build_gpu_features_for_ml(gpus, feature_cols, mem_type_cols)
    X = np.nan_to_num(X, nan=0.0)

    gpus = gpus.copy()
    gpus["pred_g3d"] = model.predict(X)
    gpus["pred_g3d_per_watt"] = np.where(gpus[TDP_USED_COL] > 0, gpus["pred_g3d"] / gpus[TDP_USED_COL], np.nan)
    return gpus


def _load_best_prediction_columns() -> Tuple[str | None, str | None]:
    tdp_col = None
    psu_col = None

    try:
        tdp_metrics = pd.read_csv(TDP_METRICS_PATH)
        if not tdp_metrics.empty and "prediction_column_to_use" in tdp_metrics.columns:
            tdp_col = tdp_metrics.iloc[0]["prediction_column_to_use"]
    except FileNotFoundError:
        tdp_col = None

    try:
        psu_metrics = pd.read_csv(PSU_METRICS_PATH)
        if not psu_metrics.empty and "prediction_column_to_use" in psu_metrics.columns:
            psu_col = psu_metrics.iloc[0]["prediction_column_to_use"]
    except FileNotFoundError:
        psu_col = None

    return tdp_col, psu_col


def _attach_power_predictions(gpu_df: pd.DataFrame) -> pd.DataFrame:
    df = gpu_df.copy()
    tdp_col, psu_col = _load_best_prediction_columns()

    try:
        preds = pd.read_csv(PREDICTIONS_PATH)
    except FileNotFoundError:
        preds = None

    if preds is not None:
        df = df.merge(preds, on=["brand", "name"], how="left")

    if tdp_col and tdp_col in df.columns:
        df[TDP_USED_COL] = df[tdp_col]
    elif "actual_tdp_w" in df.columns:
        df[TDP_USED_COL] = df["actual_tdp_w"]
    else:
        df[TDP_USED_COL] = df["tdp_w"]

    if psu_col and psu_col in df.columns:
        df[PSU_USED_COL] = df[psu_col]
    elif "actual_psu_w" in df.columns:
        df[PSU_USED_COL] = df["actual_psu_w"]
    else:
        df[PSU_USED_COL] = df["psu_w"]

    df[TDP_USED_COL] = df[TDP_USED_COL].astype(float)
    df[PSU_USED_COL] = df[PSU_USED_COL].astype(float)
    return df


def run_analysis(games: pd.DataFrame, gpus: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    per_game_rows: List[Dict[str, object]] = []

    for _, game_row in games.iterrows():
        feasible = soft_filter(hard_filter(gpus, game_row), game_row, config.soft_threshold)
        if feasible.empty:
            continue

        proxy_topk = _rank_topk(feasible, "perf_score", TDP_USED_COL, config.k_top)
        passmark_topk = _rank_topk(feasible, "pred_g3d", TDP_USED_COL, config.k_top)

        proxy_metrics = _compute_metrics(proxy_topk, "perf_score")
        passmark_metrics = _compute_metrics(passmark_topk, "pred_g3d")

        proxy_names = ", ".join(proxy_topk["name"].tolist())
        passmark_names = ", ".join(passmark_topk["name"].tolist())

        overlap = len(set(proxy_topk["name"]) & set(passmark_topk["name"]))
        overlap_rate = overlap / config.k_top if config.k_top > 0 else np.nan

        per_game_rows.append({
            "game_name": game_row.get("name"),
            "proxy_gpu_names": proxy_names,
            "passmark_gpu_names": passmark_names,
            "proxy_avg_tdp": proxy_metrics["avg_tdp"],
            "proxy_avg_psu": proxy_metrics["avg_psu"],
            "proxy_avg_score": proxy_metrics["avg_score"],
            "proxy_avg_score_per_watt": proxy_metrics["avg_score_per_watt"],
            "passmark_avg_tdp": passmark_metrics["avg_tdp"],
            "passmark_avg_psu": passmark_metrics["avg_psu"],
            "passmark_avg_score": passmark_metrics["avg_score"],
            "passmark_avg_score_per_watt": passmark_metrics["avg_score_per_watt"],
            "overlap_at_k": overlap,
            "overlap_rate": overlap_rate,
        })

    per_game_df = pd.DataFrame(per_game_rows)
    if per_game_df.empty:
        return per_game_df, pd.DataFrame()

    summary_rows = []
    for prefix, label in [("proxy", "Proxy perf_score"), ("passmark", "PassMark G3D")]:
        summary_rows.append({
            "method": label,
            "avg_tdp": per_game_df[f"{prefix}_avg_tdp"].mean(),
            "avg_psu": per_game_df[f"{prefix}_avg_psu"].mean(),
            "avg_score": per_game_df[f"{prefix}_avg_score"].mean(),
            "avg_score_per_watt": per_game_df[f"{prefix}_avg_score_per_watt"].mean(),
            "unique_gpus": per_game_df[f"{prefix}_gpu_names"].str.split(", ").explode().nunique(),
            "top1_share": _top1_share(per_game_df[f"{prefix}_gpu_names"]),
            "avg_overlap_rate": per_game_df["overlap_rate"].mean(),
        })

    summary_df = pd.DataFrame(summary_rows)
    return per_game_df, summary_df


def save_outputs(per_game_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_game_path = output_dir / "passmark_recommender_per_game.csv"
    summary_path = output_dir / "passmark_recommender_summary.csv"
    per_game_df.to_csv(per_game_path, index=False)
    summary_df.to_csv(summary_path, index=False)


def plot_summary(summary_df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=600)
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["avg_score_per_watt"], color=PASTEL_COLORS[:2])
    ax.set_title("Efficiency score per watt")
    ax.set_ylabel("Avg score per watt")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"], rotation=15, ha="right", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "passmark_efficiency_bar.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=600)
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["unique_gpus"], color=PASTEL_COLORS[2:4])
    ax.set_title("Unique GPUs selected")
    ax.set_ylabel("Unique GPU count")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"], rotation=15, ha="right", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "passmark_unique_gpus_bar.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=600)
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["avg_overlap_rate"], color=PASTEL_COLORS[:2])
    ax.set_title("Overlap@5: proxy vs PassMark")
    ax.set_ylabel("Average overlap rate")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"], rotation=15, ha="right", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "passmark_overlap_bar.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PassMark vs proxy recommendation analysis")
    parser.add_argument("--mode", choices=["min", "recom"], default="recom")
    parser.add_argument("--k-top", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=SOFT_THRESHOLD)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--plots-dir", default=DEFAULT_PLOTS_DIR)
    args = parser.parse_args()

    _configure_style()

    games, gpus = load_data(args.mode)
    gpus = attach_passmark_predictions(gpus)

    config = Config(mode=args.mode, k_top=args.k_top, soft_threshold=args.threshold)
    per_game_df, summary_df = run_analysis(games, gpus, config)

    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)

    save_outputs(per_game_df, summary_df, output_dir)
    if not summary_df.empty:
        plot_summary(summary_df, plots_dir)

    print("Saved outputs:")
    print(f"  {output_dir / 'passmark_recommender_per_game.csv'}")
    print(f"  {output_dir / 'passmark_recommender_summary.csv'}")
    print(f"  {plots_dir / 'passmark_efficiency_bar.png'}")
    print(f"  {plots_dir / 'passmark_unique_gpus_bar.png'}")
    print(f"  {plots_dir / 'passmark_overlap_bar.png'}")


if __name__ == "__main__":
    main()
