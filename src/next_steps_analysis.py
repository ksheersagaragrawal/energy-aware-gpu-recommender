"""Post-hoc analyses for Task 4 next steps.

Generates:
- Weighted-sum sensitivity sweep summary
- Perf-score feasibility ablation summary
- Method diversity report
- Min vs recom rank correlation report
- Markdown report tying everything together

Usage:
    python src/next_steps_analysis.py --mode both
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

GAME_VECTORS = {
    "min": ROOT / "data" / "vectors" / "game_vectors_min.csv",
    "recom": ROOT / "data" / "vectors" / "game_vectors_recom.csv",
}
GPU_VECTORS = ROOT / "data" / "vectors" / "gpu_power_vectors.csv"

RESULTS_DIR = ROOT / "data" / "results"

SOFT_FEATURE_MAP = {
    "texture_rate": "texture_rate",
    "pixel_rate": "pixel_rate",
    "bandwidth": "memory_bandwidth_gbs",
    "tmus": "tmus",
    "rops": "rops",
}

KNN_FEATURE_MAP = {
    "texture_rate": ("texture_rate", "texture_rate"),
    "pixel_rate": ("pixel_rate", "pixel_rate"),
    "bandwidth": ("bandwidth", "memory_bandwidth_gbs"),
    "tmus": ("tmus", "tmus"),
    "rops": ("rops", "rops"),
    "memory_clock": ("memory_clock", "memory_speed_mhz"),
    "boost_clock": ("boost_clock", "boost_clock"),
}

EPSILON = 1e-6


@dataclass(frozen=True)
class SweepConfig:
    mode: str
    random_runs: int = 30
    random_seed: int = 42
    soft_threshold: float = 0.80
    safety_alpha: float = 1.10
    perf_alpha: float = 1.00


def load_vectors(mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(GAME_VECTORS[mode])
    gpus = pd.read_csv(GPU_VECTORS)
    return games, gpus


def hard_filter(game: pd.Series, gpus: pd.DataFrame, mode: str) -> pd.DataFrame:
    mask = pd.Series(True, index=gpus.index)
    if mode == "min":
        vram_req = game.get("min_vram_mb")
        dx_req = game.get("min_direct_x")
    else:
        vram_req = game.get("recom_vram_mb")
        dx_req = game.get("recom_direct_x")

    if pd.notna(vram_req) and vram_req > 0:
        mask &= gpus["memory_mb"] >= vram_req
    if pd.notna(dx_req) and dx_req > 0:
        mask &= gpus["direct_x"] >= dx_req
    return gpus[mask].copy()


def perf_feasible_filter(game: pd.Series, gpus: pd.DataFrame, alpha: float) -> pd.DataFrame:
    req = game.get("perf_score")
    if pd.isna(req) or req <= 0:
        return gpus
    return gpus[gpus["perf_score"] >= (req * alpha)].copy()


def compute_best_ppw(feasible: pd.DataFrame) -> float:
    if feasible.empty:
        return np.nan
    ppw = feasible["perf_score"] / feasible["tdp_w"]
    ppw = ppw.replace([np.inf, -np.inf], np.nan).dropna()
    if ppw.empty:
        return np.nan
    return float(ppw.max())


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def select_random(feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    return feasible.iloc[int(rng.integers(0, len(feasible)))]


def select_lowest_tdp(feasible: pd.DataFrame) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    return feasible.sort_values(["tdp_w", "perf_score"], ascending=[True, False]).iloc[0]


def select_lowest_psu(feasible: pd.DataFrame) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    return feasible.sort_values(["psu_w", "perf_score"], ascending=[True, False]).iloc[0]


def select_highest_perf(feasible: pd.DataFrame) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    return feasible.sort_values(["perf_score", "tdp_w"], ascending=[False, True]).iloc[0]


def select_perf_per_tdp(feasible: pd.DataFrame) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    scores = feasible["perf_score"] / feasible["tdp_w"]
    idx = scores.replace([np.inf, -np.inf], np.nan).fillna(-np.inf).idxmax()
    return feasible.loc[idx]


def select_smallest_margin(game: pd.Series, feasible: pd.DataFrame) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    req = game.get("perf_score")
    if pd.isna(req) or req <= 0:
        return None
    margins = (feasible["perf_score"] - req) / req
    margins = margins.where(margins >= 0, np.inf)
    idx = margins.idxmin()
    if pd.isna(idx) or np.isinf(margins.loc[idx]):
        return feasible.sort_values(["perf_score", "tdp_w"], ascending=[False, True]).iloc[0]
    return feasible.loc[idx]


def select_safety_perf_per_tdp(game: pd.Series, feasible: pd.DataFrame, alpha: float) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    req = game.get("perf_score")
    if pd.isna(req) or req <= 0:
        return None
    filtered = feasible[feasible["perf_score"] >= (req * alpha)]
    if filtered.empty:
        return None
    scores = filtered["perf_score"] / filtered["tdp_w"]
    idx = scores.replace([np.inf, -np.inf], np.nan).fillna(-np.inf).idxmax()
    return filtered.loc[idx]


def select_pareto_knee(feasible: pd.DataFrame) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    perf = feasible["perf_score"].values
    tdp = feasible["tdp_w"].values
    psu = feasible["psu_w"].values
    n = len(feasible)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        if dominated[i]:
            continue
        better_or_equal_perf = perf >= perf[i]
        better_or_equal_tdp = tdp <= tdp[i]
        better_or_equal_psu = psu <= psu[i]
        strictly_better = (perf > perf[i]) | (tdp < tdp[i]) | (psu < psu[i])
        dominates = better_or_equal_perf & better_or_equal_tdp & better_or_equal_psu & strictly_better
        if dominates.any():
            dominated[i] = True

    pareto = feasible.loc[~dominated].copy()
    if pareto.empty:
        return None

    perf_norm = pareto["perf_score"] - pareto["perf_score"].min()
    if perf_norm.max() > 0:
        perf_norm = perf_norm / perf_norm.max()
    tdp_norm = pareto["tdp_w"] - pareto["tdp_w"].min()
    if tdp_norm.max() > 0:
        tdp_norm = tdp_norm / tdp_norm.max()
    psu_norm = pareto["psu_w"] - pareto["psu_w"].min()
    if psu_norm.max() > 0:
        psu_norm = psu_norm / psu_norm.max()

    distances = np.sqrt((1 - perf_norm) ** 2 + tdp_norm ** 2 + psu_norm ** 2)
    idx = distances.idxmin()
    return pareto.loc[idx]


def select_weighted_sum(game: pd.Series, feasible: pd.DataFrame, weights: Dict[str, float]) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    perf_z = zscore(feasible["perf_score"])
    tdp_z = zscore(feasible["tdp_w"])
    psu_z = zscore(feasible["psu_w"])

    req = game.get("perf_score")
    if pd.isna(req) or req <= 0:
        overprov_z = pd.Series(0.0, index=feasible.index)
    else:
        overprov = feasible["perf_score"] - req
        overprov_z = zscore(overprov)

    score = (
        weights.get("perf", 1.0) * perf_z
        - weights.get("tdp", 1.0) * tdp_z
        - weights.get("psu", 0.5) * psu_z
        - weights.get("overprov", 0.5) * overprov_z
    )
    idx = score.idxmax()
    return feasible.loc[idx]


def select_knn(game: pd.Series, feasible: pd.DataFrame) -> Optional[pd.Series]:
    if feasible.empty:
        return None

    game_vec = []
    gpu_matrix = []

    for _, (game_col, gpu_col) in KNN_FEATURE_MAP.items():
        gpu_vals = feasible[gpu_col].values.astype(float)
        game_val = float(game.get(game_col) or 0)

        combined = np.concatenate([gpu_vals[~np.isnan(gpu_vals)], [game_val]])
        f_min, f_max = combined.min(), combined.max()
        if f_max > f_min:
            gpu_norm = np.where(
                np.isnan(gpu_vals),
                0.0,
                np.clip((gpu_vals - f_min) / (f_max - f_min), EPSILON, 1.0),
            )
            game_norm = 0.0 if np.isnan(game_val) else float(
                np.clip((game_val - f_min) / (f_max - f_min), EPSILON, 1.0)
            )
        else:
            gpu_norm = np.where(np.isnan(gpu_vals), 0.0, 1.0)
            game_norm = 0.0 if np.isnan(game_val) else 1.0

        gpu_matrix.append(gpu_norm)
        game_vec.append(game_norm)

    gpu_matrix = np.column_stack(gpu_matrix)
    game_vec = np.array(game_vec, dtype=float).reshape(1, -1)

    distances = np.linalg.norm(gpu_matrix - game_vec, axis=1)
    idx = int(np.argmin(distances))
    return feasible.iloc[idx]


def select_proposed(game: pd.Series, feasible: pd.DataFrame, threshold: float) -> Optional[pd.Series]:
    if feasible.empty:
        return None
    mask = pd.Series(True, index=feasible.index)
    for game_col, gpu_col in SOFT_FEATURE_MAP.items():
        req = game.get(game_col)
        if pd.isna(req) or req <= 0:
            continue
        mask &= feasible[gpu_col] >= (req * threshold)
    filtered = feasible[mask]
    if filtered.empty:
        return None
    scores = filtered["perf_score"] / filtered["tdp_w"]
    idx = scores.replace([np.inf, -np.inf], np.nan).fillna(-np.inf).idxmax()
    return filtered.loc[idx]


def build_recommendation_row(
    game: pd.Series,
    feasible: pd.DataFrame,
    selected: Optional[pd.Series],
    method: str,
    mode: str,
    run_idx: Optional[int],
    best_ppw: float,
) -> Dict[str, object]:
    base = {
        "game_name": game.get("name"),
        "track": mode,
        "method": method,
        "run": run_idx,
        "feasible_set_size": len(feasible),
        "game_perf_score": game.get("perf_score"),
        "best_ppw": best_ppw,
    }
    if selected is None:
        base.update(
            {
                "selected_gpu": None,
                "selected_brand": None,
                "selected_tdp_w": np.nan,
                "selected_psu_w": np.nan,
                "selected_perf_score": np.nan,
                "selected_perf_per_watt": np.nan,
            }
        )
        return base

    perf_score = selected.get("perf_score")
    tdp = selected.get("tdp_w")
    perf_per_watt = perf_score / tdp if pd.notna(perf_score) and pd.notna(tdp) and tdp > 0 else np.nan
    base.update(
        {
            "selected_gpu": selected.get("name"),
            "selected_brand": selected.get("brand"),
            "selected_tdp_w": tdp,
            "selected_psu_w": selected.get("psu_w"),
            "selected_perf_score": perf_score,
            "selected_perf_per_watt": perf_per_watt,
        }
    )
    return base


def compute_metrics(recs: pd.DataFrame) -> pd.DataFrame:
    metrics = recs.copy()
    metrics["coverage"] = metrics["selected_gpu"].notna().astype(int)

    req = metrics["game_perf_score"]
    sel_perf = metrics["selected_perf_score"]
    metrics["overprov_abs"] = sel_perf - req
    metrics["overprov_rel"] = (sel_perf / req) - 1
    metrics.loc[(req.isna()) | (req <= 0), ["overprov_abs", "overprov_rel"]] = np.nan

    metrics["eff_regret_abs"] = metrics["best_ppw"] - metrics["selected_perf_per_watt"]
    metrics["eff_regret_rel"] = metrics["eff_regret_abs"] / metrics["best_ppw"]

    return metrics


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "coverage",
        "selected_tdp_w",
        "selected_psu_w",
        "selected_perf_per_watt",
        "overprov_abs",
        "overprov_rel",
        "eff_regret_abs",
        "eff_regret_rel",
    ]

    rows = []
    for (track, method), group in metrics.groupby(["track", "method"], dropna=False):
        row = {"track": track, "method": method}
        for col in metric_cols:
            series = group[col].dropna()
            row[f"{col}_mean"] = series.mean() if not series.empty else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def eval_strategies(
    mode: str,
    rng: np.random.Generator,
    soft_threshold: float,
    safety_alpha: float,
    perf_alpha: float,
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    games, gpus = load_vectors(mode)
    rows = []

    for _, game in games.iterrows():
        feasible = hard_filter(game, gpus, mode)
        feasible = perf_feasible_filter(game, feasible, perf_alpha)
        best_ppw = compute_best_ppw(feasible)

        strategies = [
            ("random_feasible", lambda g, f: select_random(f, rng), True),
            ("lowest_tdp", lambda g, f: select_lowest_tdp(f), False),
            ("lowest_psu", lambda g, f: select_lowest_psu(f), False),
            ("highest_perf", lambda g, f: select_highest_perf(f), False),
            ("perf_per_tdp", lambda g, f: select_perf_per_tdp(f), False),
            ("smallest_margin", lambda g, f: select_smallest_margin(g, f), False),
            ("safety_factor_perf_per_tdp", lambda g, f: select_safety_perf_per_tdp(g, f, safety_alpha), False),
            ("pareto_knee", lambda g, f: select_pareto_knee(f), False),
            ("weighted_sum", lambda g, f: select_weighted_sum(g, f, weights or {}), False),
            ("knn_retrieval", lambda g, f: select_knn(g, f), False),
            ("proposed_recommender", lambda g, f: select_proposed(g, f, soft_threshold), False),
        ]

        for name, selector, is_random in strategies:
            runs = 30 if is_random else 1
            for run_idx in range(runs):
                selected = selector(game, feasible)
                rows.append(
                    build_recommendation_row(
                        game,
                        feasible,
                        selected,
                        name,
                        mode,
                        run_idx if runs > 1 else None,
                        best_ppw,
                    )
                )

    recs = pd.DataFrame(rows)
    return recs


def weighted_sum_sweep(config: SweepConfig) -> pd.DataFrame:
    modes = [config.mode] if config.mode in {"min", "recom"} else ["min", "recom"]
    rng = np.random.default_rng(config.random_seed)

    weights_grid = []
    for tdp_w in [0.25, 0.5, 1.0, 2.0]:
        for psu_w in [0.25, 0.5, 1.0, 2.0]:
            for overprov_w in [0.0, 0.5]:
                weights_grid.append({"perf": 1.0, "tdp": tdp_w, "psu": psu_w, "overprov": overprov_w})

    rows = []
    for mode in modes:
        for weights in weights_grid:
            recs = eval_strategies(
                mode,
                rng,
                config.soft_threshold,
                config.safety_alpha,
                config.perf_alpha,
                weights,
            )
            metrics = compute_metrics(recs)
            summary = summarize_metrics(metrics)
            summary = summary[summary["method"] == "weighted_sum"].copy()
            summary["weight_perf"] = weights["perf"]
            summary["weight_tdp"] = weights["tdp"]
            summary["weight_psu"] = weights["psu"]
            summary["weight_overprov"] = weights["overprov"]
            rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def perf_feasible_ablation(config: SweepConfig) -> pd.DataFrame:
    modes = [config.mode] if config.mode in {"min", "recom"} else ["min", "recom"]
    rng = np.random.default_rng(config.random_seed)

    rows = []
    for mode in modes:
        recs = eval_strategies(
            mode,
            rng,
            config.soft_threshold,
            config.safety_alpha,
            config.perf_alpha,
            None,
        )
        metrics = compute_metrics(recs)
        summary = summarize_metrics(metrics)
        summary["perf_alpha"] = config.perf_alpha
        rows.append(summary)

    return pd.concat(rows, ignore_index=True)


def method_diversity() -> pd.DataFrame:
    recs_path = RESULTS_DIR / "baseline_recommendations.csv"
    recs = pd.read_csv(recs_path)
    recs = recs[recs["selected_gpu"].notna()]

    rows = []
    for (track, method), group in recs.groupby(["track", "method"], dropna=False):
        total = len(group)
        unique = group["selected_gpu"].nunique()
        top_share = group["selected_gpu"].value_counts(normalize=True).iloc[0] if total > 0 else np.nan
        rows.append(
            {
                "track": track,
                "method": method,
                "total_recommendations": total,
                "unique_gpus": unique,
                "unique_share": unique / total if total > 0 else np.nan,
                "top1_share": top_share,
            }
        )

    return pd.DataFrame(rows)


def rank_correlation() -> pd.DataFrame:
    summary_path = RESULTS_DIR / "baseline_metrics_summary.csv"
    summary = pd.read_csv(summary_path)

    metrics = [
        ("selected_tdp_w_mean", True),
        ("selected_psu_w_mean", True),
        ("selected_perf_per_watt_mean", False),
        ("overprov_abs_mean", True),
        ("eff_regret_abs_mean", True),
    ]

    rows = []
    methods = set(summary[summary["track"] == "min"]["method"]) & set(
        summary[summary["track"] == "recom"]["method"]
    )

    for metric, lower_is_better in metrics:
        min_df = summary[(summary["track"] == "min") & (summary["method"].isin(methods))].copy()
        recom_df = summary[(summary["track"] == "recom") & (summary["method"].isin(methods))].copy()

        min_df = min_df.set_index("method")
        recom_df = recom_df.set_index("method")

        min_vals = min_df[metric]
        recom_vals = recom_df[metric]

        min_rank = min_vals.rank(ascending=lower_is_better)
        recom_rank = recom_vals.rank(ascending=lower_is_better)

        if min_rank.isna().any() or recom_rank.isna().any():
            continue

        corr = np.corrcoef(min_rank.values, recom_rank.values)[0, 1]
        rows.append(
            {
                "metric": metric,
                "spearman_r": float(corr),
                "lower_is_better": lower_is_better,
            }
        )

    return pd.DataFrame(rows)


def write_report(
    diversity: pd.DataFrame,
    sweep: pd.DataFrame,
    perf_ablation: pd.DataFrame,
    rank_corr: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = []
    lines.append("# Task 4 Next Steps Analysis")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Weighted-sum sweep summary: {RESULTS_DIR / 'weighted_sum_sweep_summary.csv'}")
    lines.append(f"- Perf-score feasibility ablation: {RESULTS_DIR / 'perf_feasible_ablation_summary.csv'}")
    lines.append(f"- Method diversity: {RESULTS_DIR / 'method_diversity_summary.csv'}")
    lines.append(f"- Min vs recom rank correlation: {RESULTS_DIR / 'min_recom_rank_correlation.csv'}")
    lines.append("")

    lines.append("## Method diversity (top findings)")
    lines.append("")
    for _, row in diversity.sort_values(["track", "method"]).head(10).iterrows():
        lines.append(
            f"- {row['track']} / {row['method']}: unique GPUs={row['unique_gpus']}, "
            f"unique share={row['unique_share']:.3f}, top1 share={row['top1_share']:.3f}"
        )

    lines.append("")
    lines.append("## Min vs recom rank correlation")
    lines.append("")
    lines.append("Spearman rank correlation is used to compare method ordering across tracks [2].")
    for _, row in rank_corr.iterrows():
        lines.append(f"- {row['metric']}: Spearman r={row['spearman_r']:.3f}")

    lines.append("")
    lines.append("## Perf-score feasibility ablation")
    lines.append("")
    lines.append("Summary stored in perf_feasible_ablation_summary.csv.")

    lines.append("")
    lines.append("## Weighted-sum sensitivity sweep")
    lines.append("")
    lines.append("Summary stored in weighted_sum_sweep_summary.csv. Weight sweeps follow standard sensitivity analysis practice [1].")

    lines.append("")
    lines.append("## References")
    lines.append("")
    lines.append("[1] Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana, M., and Tarantola, S., \"Global Sensitivity Analysis: The Primer,\" Wiley, 2008. https://onlinelibrary.wiley.com/doi/book/10.1002/9780470725184")
    lines.append("[2] Spearman, C., \"The proof and measurement of association between two things,\" The American Journal of Psychology, 1904. https://doi.org/10.2307/1412159")

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Next steps analysis for Task 4")
    parser.add_argument("--mode", choices=["min", "recom", "both"], default="both")
    parser.add_argument("--soft-threshold", type=float, default=0.80)
    parser.add_argument("--safety-alpha", type=float, default=1.10)
    parser.add_argument("--perf-alpha", type=float, default=1.00)
    parser.add_argument("--random-runs", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = SweepConfig(
        mode=args.mode,
        random_runs=args.random_runs,
        random_seed=args.random_seed,
        soft_threshold=args.soft_threshold,
        safety_alpha=args.safety_alpha,
        perf_alpha=args.perf_alpha,
    )

    diversity = method_diversity()
    diversity_out = RESULTS_DIR / "method_diversity_summary.csv"
    diversity.to_csv(diversity_out, index=False)

    rank_corr = rank_correlation()
    rank_corr_out = RESULTS_DIR / "min_recom_rank_correlation.csv"
    rank_corr.to_csv(rank_corr_out, index=False)

    perf_ablation = perf_feasible_ablation(config)
    perf_ablation_out = RESULTS_DIR / "perf_feasible_ablation_summary.csv"
    perf_ablation.to_csv(perf_ablation_out, index=False)

    sweep = weighted_sum_sweep(config)
    sweep_out = RESULTS_DIR / "weighted_sum_sweep_summary.csv"
    sweep.to_csv(sweep_out, index=False)

    report_path = ROOT / "task4_next_steps_report.md"
    write_report(diversity, sweep, perf_ablation, rank_corr, report_path)

    print(f"Saved: {diversity_out}")
    print(f"Saved: {rank_corr_out}")
    print(f"Saved: {perf_ablation_out}")
    print(f"Saved: {sweep_out}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
