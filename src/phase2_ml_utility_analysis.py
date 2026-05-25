"""Phase 2: ML utility top-k analysis.

Notes:
- KNN50_All tests raw feature-affinity retrieval from the full GPU database.
- KNN50_Feasible tests feature-affinity retrieval after enforcing feasibility.
- KNN50_Feasible_PPW_Top5 tests KNN retrieval plus PPW reranking.
- Power_Top5 tests direct power-aware recommendation from the feasible set.
- UtilityFormula_Top5 and ML_Utility_Top5 provide multi-objective alternatives.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from phase1_topk_knn_analysis import (
    Phase1Config,
    TopKRecommendationAnalyzer,
    _attach_power_predictions,
)


PREDICTION_METRICS_OUT = "phase2_model_regression_metrics.csv"
PER_GAME_OUT = "phase2_per_game_ml_utility_topk_analysis.csv"
AGG_OUT = "phase2_aggregate_ml_utility_topk_summary.csv"


GPU_ID_COL = "gpu_id"
TDP_USED_COL = "tdp_w_used"
PSU_USED_COL = "psu_w_used"
PPW_COL = "perf_per_watt"


@dataclass(frozen=True)
class Phase2Config:
    mode: str = "recom"
    k_top: int = 5
    knn_k: int = 50
    soft_threshold: float = 0.80
    output_dir: str = "phase2_outputs"
    random_seed: int = 42
    train_split: float = 0.8


def _safe_minmax(series: pd.Series) -> Tuple[float, float]:
    s_min = series.min()
    s_max = series.max()
    return s_min, s_max


def _minmax_score(values: pd.Series) -> pd.Series:
    v_min, v_max = _safe_minmax(values)
    if pd.isna(v_min) or pd.isna(v_max) or v_max == v_min:
        return pd.Series(0.5, index=values.index)
    score = (values - v_min) / (v_max - v_min)
    return score.clip(0.0, 1.0)


def _low_is_good_score(values: pd.Series) -> pd.Series:
    return (1.0 - _minmax_score(values)).clip(0.0, 1.0)


def _compute_margin(gpu_perf: pd.Series, game_perf: float) -> pd.Series:
    if pd.isna(game_perf) or game_perf == 0:
        return pd.Series(np.nan, index=gpu_perf.index)
    return (gpu_perf - game_perf) / game_perf


def _utility_formula_scores(
    game_row: pd.Series,
    feasible: pd.DataFrame,
) -> pd.DataFrame:
    df = feasible.copy()

    ppw = df[PPW_COL]
    df["ppw_score"] = _minmax_score(ppw)
    df["low_tdp_score"] = _low_is_good_score(df[TDP_USED_COL])
    df["low_psu_score"] = _low_is_good_score(df[PSU_USED_COL])

    # Overprovisioning proxy: game and GPU perf scores are normalized on
    # different datasets, so treat margin-based signals as relative proxies.
    game_perf = game_row.get("perf_score")
    margin = _compute_margin(df["perf_score"], game_perf)

    margin_score = margin.clip(lower=0.0, upper=0.25) / 0.25
    margin_score = margin_score.fillna(0.5)
    df["margin_score"] = margin_score.clip(0.0, 1.0)

    target_margin = 0.15
    right_size = 1.0 - (margin - target_margin).abs() / target_margin
    right_size = right_size.clip(lower=0.0, upper=1.0).fillna(0.5)
    df["right_size_score"] = right_size

    df["utility_formula_score"] = (
        0.50 * df["ppw_score"]
        + 0.15 * df["low_tdp_score"]
        + 0.10 * df["low_psu_score"]
        + 0.15 * df["margin_score"]
        + 0.10 * df["right_size_score"]
    )

    return df


def _build_pair_features(game_row: pd.Series, gpu_df: pd.DataFrame) -> pd.DataFrame:
    required_vram = game_row.get("min_vram_mb")
    required_directx = game_row.get("min_direct_x")

    features = pd.DataFrame({
        "required_vram": required_vram,
        "required_bandwidth": game_row.get("bandwidth"),
        "required_texture_rate": game_row.get("texture_rate"),
        "required_pixel_rate": game_row.get("pixel_rate"),
        "required_directx": required_directx,
        "gpu_vram": gpu_df["memory_mb"].values,
        "gpu_bandwidth": gpu_df["memory_bandwidth_gbs"].values,
        "gpu_texture_rate": gpu_df["texture_rate"].values,
        "gpu_pixel_rate": gpu_df["pixel_rate"].values,
        "gpu_directx": gpu_df["direct_x"].values,
        "gpu_perf_score": gpu_df["perf_score"].values,
        "predicted_tdp": gpu_df[TDP_USED_COL].values,
        "predicted_psu": gpu_df[PSU_USED_COL].values,
        "ppw": gpu_df[PPW_COL].values,
        "mode_recom": 1.0,
    })

    features["vram_margin"] = features["gpu_vram"] - features["required_vram"]
    features["bandwidth_margin"] = features["gpu_bandwidth"] - features["required_bandwidth"]
    features["texture_margin"] = features["gpu_texture_rate"] - features["required_texture_rate"]
    features["pixel_margin"] = features["gpu_pixel_rate"] - features["required_pixel_rate"]

    def _relative_margin(margin: pd.Series, base: float) -> pd.Series:
        if pd.isna(base) or base == 0:
            return pd.Series(np.nan, index=margin.index)
        return margin / base

    features["relative_vram_margin"] = _relative_margin(features["vram_margin"], required_vram)
    features["relative_bandwidth_margin"] = _relative_margin(features["bandwidth_margin"], game_row.get("bandwidth"))
    features["relative_texture_margin"] = _relative_margin(features["texture_margin"], game_row.get("texture_rate"))
    features["relative_pixel_margin"] = _relative_margin(features["pixel_margin"], game_row.get("pixel_rate"))

    perf_margin = _compute_margin(gpu_df["perf_score"], game_row.get("perf_score"))
    features["performance_margin_proxy"] = perf_margin.values
    features["overprovisioning_proxy"] = perf_margin.values

    return features


def _train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[object, Dict[str, object]]:
    metrics = {}

    # Try XGBoost first (GPU if available), then LightGBM, then sklearn GBDT.
    model = None
    backend = "cpu"

    try:
        import xgboost as xgb  # type: ignore

        params = {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 42,
        }

        def _fit_xgb(local_params):
            nonlocal model, backend
            start = time.time()
            model = xgb.XGBRegressor(**local_params)
            model.fit(X_train, y_train)
            train_time = time.time() - start
            backend = local_params.get("device", "cpu")
            return train_time

        try:
            params_gpu = params | {"tree_method": "hist", "device": "cuda"}
            train_time = _fit_xgb(params_gpu)
        except Exception:
            try:
                params_gpu_hist = params | {"tree_method": "gpu_hist"}
                train_time = _fit_xgb(params_gpu_hist)
            except Exception:
                params_cpu = params | {"tree_method": "hist", "device": "cpu"}
                train_time = _fit_xgb(params_cpu)
                backend = "cpu"

        start = time.time()
        preds = model.predict(X_test)
        infer_time = time.time() - start

        metrics.update({
            "model": "XGBoostRegressor",
            "backend": backend,
            "train_time_sec": train_time,
            "inference_time_sec": infer_time,
            "mae": mean_absolute_error(y_test, preds),
            "rmse": mean_squared_error(y_test, preds, squared=False),
            "r2": r2_score(y_test, preds),
        })
        print(
            f"[phase2] model={metrics['model']} backend={metrics['backend']} "
            f"train_time={metrics['train_time_sec']:.2f}s "
            f"infer_time={metrics['inference_time_sec']:.2f}s "
            f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} r2={metrics['r2']:.4f}"
        )
        return model, metrics

    except Exception:
        model = None

    try:
        import lightgbm as lgb  # type: ignore

        params = {
            "n_estimators": 400,
            "learning_rate": 0.05,
            "random_state": 42,
        }

        def _fit_lgb(local_params):
            nonlocal model, backend
            start = time.time()
            model = lgb.LGBMRegressor(**local_params)
            model.fit(X_train, y_train)
            train_time = time.time() - start
            backend = "gpu" if local_params.get("device_type") == "gpu" else "cpu"
            return train_time

        try:
            train_time = _fit_lgb(params | {"device_type": "gpu"})
        except Exception:
            train_time = _fit_lgb(params | {"device_type": "cpu"})

        start = time.time()
        preds = model.predict(X_test)
        infer_time = time.time() - start

        metrics.update({
            "model": "LightGBMRegressor",
            "backend": backend,
            "train_time_sec": train_time,
            "inference_time_sec": infer_time,
            "mae": mean_absolute_error(y_test, preds),
            "rmse": mean_squared_error(y_test, preds, squared=False),
            "r2": r2_score(y_test, preds),
        })
        print(
            f"[phase2] model={metrics['model']} backend={metrics['backend']} "
            f"train_time={metrics['train_time_sec']:.2f}s "
            f"infer_time={metrics['inference_time_sec']:.2f}s "
            f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} r2={metrics['r2']:.4f}"
        )
        return model, metrics

    except Exception:
        model = None

    start = time.time()
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    preds = model.predict(X_test)
    infer_time = time.time() - start

    metrics.update({
        "model": "GradientBoostingRegressor",
        "backend": "cpu",
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "mae": mean_absolute_error(y_test, preds),
        "rmse": mean_squared_error(y_test, preds, squared=False),
        "r2": r2_score(y_test, preds),
    })
    print(
        f"[phase2] model={metrics['model']} backend={metrics['backend']} "
        f"train_time={metrics['train_time_sec']:.2f}s "
        f"infer_time={metrics['inference_time_sec']:.2f}s "
        f"mae={metrics['mae']:.4f} rmse={metrics['rmse']:.4f} r2={metrics['r2']:.4f}"
    )
    return model, metrics


def _compute_overlap_matrix(method_sets: Dict[str, List[set]], k: int = 5) -> pd.DataFrame:
    methods = list(method_sets.keys())
    matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

    for m1 in methods:
        for m2 in methods:
            overlaps = []
            for s1, s2 in zip(method_sets[m1], method_sets[m2]):
                overlaps.append(len(s1 & s2) / max(k, 1))
            matrix.loc[m1, m2] = float(np.nanmean(overlaps))
    return matrix


def _top1_share(names_series: pd.Series) -> float:
    top1 = names_series.dropna().apply(lambda v: v.split(", ")[0] if v else "")
    if top1.empty:
        return np.nan
    return top1.value_counts(normalize=True).iloc[0]


def _aggregate_summary(per_game_df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    rows = []
    for method in methods:
        prefix = method.lower()
        rows.append({
            "method": method,
            "avg_tdp": per_game_df[f"{prefix}_avg_tdp"].mean(),
            "avg_psu": per_game_df[f"{prefix}_avg_psu"].mean(),
            "avg_ppw": per_game_df[f"{prefix}_avg_ppw"].mean(),
            "avg_efficiency_regret": per_game_df[f"{prefix}_avg_efficiency_regret"].mean(),
            "avg_overprovisioning_proxy": per_game_df[f"{prefix}_avg_overprovisioning"].mean(),
            "unique_gpus": per_game_df[f"{prefix}_gpu_names"].str.split(", ").explode().nunique(),
            "top1_share": _top1_share(per_game_df[f"{prefix}_gpu_names"]),
        })
    return pd.DataFrame(rows)


def _normalize_metrics(summary_df: pd.DataFrame, metrics: List[str], higher_better: List[str]) -> pd.DataFrame:
    norm = summary_df.set_index("method")[metrics].copy()
    for col in metrics:
        v_min, v_max = norm[col].min(), norm[col].max()
        if v_max == v_min:
            norm[col] = 1.0
            continue
        scaled = (norm[col] - v_min) / (v_max - v_min)
        if col in higher_better:
            norm[col] = scaled
        else:
            norm[col] = 1.0 - scaled
    return norm


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: ML utility top-k analysis")
    parser.add_argument("--mode", choices=["min", "recom"], default="recom")
    parser.add_argument("--k-top", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--output-dir", default="phase2_outputs")
    args = parser.parse_args()

    config = Phase2Config(
        mode=args.mode,
        k_top=args.k_top,
        knn_k=args.knn_k,
        soft_threshold=args.threshold,
        output_dir=args.output_dir,
    )

    games_df = pd.read_csv(f"data/vectors/game_vectors_{config.mode}.csv")
    gpu_df = pd.read_csv("data/vectors/gpu_power_vectors.csv")
    gpu_df = _attach_power_predictions(gpu_df)

    analyzer = TopKRecommendationAnalyzer(
        games_df,
        gpu_df,
        Phase1Config(
            mode=config.mode,
            k_top=config.k_top,
            knn_k=config.knn_k,
            soft_threshold=config.soft_threshold,
        ),
    )

    # Train/test split by games
    train_games, test_games = train_test_split(
        games_df["name"].values,
        train_size=config.train_split,
        random_state=config.random_seed,
        shuffle=True,
    )

    train_games = set(train_games)
    test_games = set(test_games)

    pair_rows = []
    pair_targets = []

    for _, game_row in games_df.iterrows():
        if game_row["name"] not in train_games:
            continue
        feasible = analyzer.get_feasible_gpus(game_row)
        if feasible.empty:
            continue
        scored = _utility_formula_scores(game_row, feasible)
        features = _build_pair_features(game_row, scored)
        pair_rows.append(features)
        pair_targets.append(scored["utility_formula_score"].values)

    if not pair_rows:
        raise RuntimeError("No feasible game-GPU pairs found for training.")

    X_train = pd.concat(pair_rows, ignore_index=True).fillna(0.0)
    y_train = pd.Series(np.concatenate(pair_targets))

    # Build a test set for model metrics
    test_pair_rows = []
    test_pair_targets = []
    for _, game_row in games_df.iterrows():
        if game_row["name"] not in test_games:
            continue
        feasible = analyzer.get_feasible_gpus(game_row)
        if feasible.empty:
            continue
        scored = _utility_formula_scores(game_row, feasible)
        features = _build_pair_features(game_row, scored)
        test_pair_rows.append(features)
        test_pair_targets.append(scored["utility_formula_score"].values)

    X_test = pd.concat(test_pair_rows, ignore_index=True).fillna(0.0)
    y_test = pd.Series(np.concatenate(test_pair_targets))

    model, model_metrics = _train_model(X_train, y_train, X_test, y_test)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([model_metrics]).to_csv(out_dir / PREDICTION_METRICS_OUT, index=False)

    methods = [
        "KNN50_Feasible",
        "KNN50_Feasible_PPW_Top5",
        "Power_Top5",
        "UtilityFormula_Top5",
        "ML_Utility_Top5",
    ]

    per_game_rows = []
    overlap_sets: Dict[str, List[set]] = {m: [] for m in methods}
    utility_components = []

    for _, game_row in games_df.iterrows():
        if game_row["name"] not in test_games:
            continue

        feasible = analyzer.get_feasible_gpus(game_row)
        if feasible.empty:
            continue

        best_ppw = feasible[PPW_COL].max()

        power_topk = analyzer.get_power_topk(game_row, k=config.k_top)
        knn_feasible = analyzer.get_knn_candidates_feasible(game_row, n_neighbors=config.knn_k)
        knn_feasible_ppw = analyzer.get_knn_feasible_ppw_topk(game_row, n_neighbors=config.knn_k, k=config.k_top)

        utility_scored = _utility_formula_scores(game_row, feasible)
        utility_topk = utility_scored.sort_values("utility_formula_score", ascending=False).head(config.k_top)

        utility_components.append(
            utility_topk[[
                "name",
                "ppw_score",
                "low_tdp_score",
                "low_psu_score",
                "margin_score",
                "right_size_score",
                "utility_formula_score",
            ]].assign(game_name=game_row["name"])
        )

        # ML utility predictions
        ml_features = _build_pair_features(game_row, feasible).fillna(0.0)
        preds = model.predict(ml_features)
        ml_scored = feasible.copy()
        ml_scored["ml_utility_score"] = preds
        ml_topk = ml_scored.sort_values("ml_utility_score", ascending=False).head(config.k_top)

        power_metrics = analyzer.compute_candidate_metrics(power_topk, game_row, best_ppw)
        knn_metrics = analyzer.compute_candidate_metrics(knn_feasible, game_row, best_ppw)
        knn_ppw_metrics = analyzer.compute_candidate_metrics(knn_feasible_ppw, game_row, best_ppw)
        utility_metrics = analyzer.compute_candidate_metrics(utility_topk, game_row, best_ppw)
        ml_metrics = analyzer.compute_candidate_metrics(ml_topk, game_row, best_ppw)

        per_game_rows.append({
            "game_name": game_row["name"],
            "knn50_feasible_avg_tdp": knn_metrics["avg_tdp"],
            "knn50_feasible_avg_psu": knn_metrics["avg_psu"],
            "knn50_feasible_avg_ppw": knn_metrics["avg_ppw"],
            "knn50_feasible_avg_efficiency_regret": knn_metrics["avg_eff_regret"],
            "knn50_feasible_avg_overprovisioning": knn_metrics["avg_overprov"],
            "knn50_feasible_gpu_names": ", ".join(knn_feasible["name"].tolist()),
            "knn50_feasible_ppw_top5_avg_tdp": knn_ppw_metrics["avg_tdp"],
            "knn50_feasible_ppw_top5_avg_psu": knn_ppw_metrics["avg_psu"],
            "knn50_feasible_ppw_top5_avg_ppw": knn_ppw_metrics["avg_ppw"],
            "knn50_feasible_ppw_top5_avg_efficiency_regret": knn_ppw_metrics["avg_eff_regret"],
            "knn50_feasible_ppw_top5_avg_overprovisioning": knn_ppw_metrics["avg_overprov"],
            "knn50_feasible_ppw_top5_gpu_names": ", ".join(knn_feasible_ppw["name"].tolist()),
            "power_top5_avg_tdp": power_metrics["avg_tdp"],
            "power_top5_avg_psu": power_metrics["avg_psu"],
            "power_top5_avg_ppw": power_metrics["avg_ppw"],
            "power_top5_avg_efficiency_regret": power_metrics["avg_eff_regret"],
            "power_top5_avg_overprovisioning": power_metrics["avg_overprov"],
            "power_top5_gpu_names": ", ".join(power_topk["name"].tolist()),
            "utilityformula_top5_avg_tdp": utility_metrics["avg_tdp"],
            "utilityformula_top5_avg_psu": utility_metrics["avg_psu"],
            "utilityformula_top5_avg_ppw": utility_metrics["avg_ppw"],
            "utilityformula_top5_avg_efficiency_regret": utility_metrics["avg_eff_regret"],
            "utilityformula_top5_avg_overprovisioning": utility_metrics["avg_overprov"],
            "utilityformula_top5_gpu_names": ", ".join(utility_topk["name"].tolist()),
            "ml_utility_top5_avg_tdp": ml_metrics["avg_tdp"],
            "ml_utility_top5_avg_psu": ml_metrics["avg_psu"],
            "ml_utility_top5_avg_ppw": ml_metrics["avg_ppw"],
            "ml_utility_top5_avg_efficiency_regret": ml_metrics["avg_eff_regret"],
            "ml_utility_top5_avg_overprovisioning": ml_metrics["avg_overprov"],
            "ml_utility_top5_gpu_names": ", ".join(ml_topk["name"].tolist()),
        })

        overlap_sets["KNN50_Feasible"].append(set(knn_feasible[GPU_ID_COL].tolist()))
        overlap_sets["KNN50_Feasible_PPW_Top5"].append(set(knn_feasible_ppw[GPU_ID_COL].tolist()))
        overlap_sets["Power_Top5"].append(set(power_topk[GPU_ID_COL].tolist()))
        overlap_sets["UtilityFormula_Top5"].append(set(utility_topk[GPU_ID_COL].tolist()))
        overlap_sets["ML_Utility_Top5"].append(set(ml_topk[GPU_ID_COL].tolist()))

    per_game_df = pd.DataFrame(per_game_rows)
    per_game_df.to_csv(out_dir / PER_GAME_OUT, index=False)

    if utility_components:
        pd.concat(utility_components, ignore_index=True).to_csv(
            out_dir / "phase2_utility_formula_top5_components.csv",
            index=False,
        )

    summary_df = _aggregate_summary(per_game_df, methods)
    summary_df.to_csv(out_dir / AGG_OUT, index=False)

    overlap_matrix = _compute_overlap_matrix(
        {
            "Power_Top5": overlap_sets["Power_Top5"],
            "UtilityFormula_Top5": overlap_sets["UtilityFormula_Top5"],
            "ML_Utility_Top5": overlap_sets["ML_Utility_Top5"],
            "KNN50_Feasible_PPW_Top5": overlap_sets["KNN50_Feasible_PPW_Top5"],
        },
        k=config.k_top,
    )

    # Plots (300 dpi)
    metrics = [
        "avg_tdp",
        "avg_psu",
        "avg_ppw",
        "avg_efficiency_regret",
        "unique_gpus",
        "top1_share",
    ]
    higher_better = ["avg_ppw", "unique_gpus"]
    norm = _normalize_metrics(summary_df, metrics, higher_better)

    plt.figure(figsize=(8, 3.5))
    plt.imshow(norm.values, aspect="auto", cmap="viridis")
    plt.xticks(range(len(metrics)), metrics, rotation=30, ha="right")
    plt.yticks(range(len(norm.index)), norm.index)
    plt.colorbar(label="Normalized score (higher is better)")
    plt.title("Phase 2 metric heatmap (normalized)")
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_metric_heatmap.png", dpi=300)
    plt.close()

    # Efficiency trade-off
    plt.figure(figsize=(7, 5))
    for _, row in summary_df.iterrows():
        plt.scatter(row["avg_ppw"], row["avg_efficiency_regret"], s=70)
        plt.text(row["avg_ppw"], row["avg_efficiency_regret"], row["method"], fontsize=9)
    plt.xlabel("avg_ppw")
    plt.ylabel("avg_efficiency_regret")
    plt.title("Phase 2 efficiency trade-off")
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_efficiency_tradeoff.png", dpi=300)
    plt.close()

    # Diversity vs efficiency trade-off
    plt.figure(figsize=(7, 5))
    for _, row in summary_df.iterrows():
        x_val = 1.0 - row["top1_share"] if pd.notna(row["top1_share"]) else row["unique_gpus"]
        plt.scatter(x_val, row["avg_ppw"], s=70)
        plt.text(x_val, row["avg_ppw"], row["method"], fontsize=9)
    plt.xlabel("1 - top1_share (or unique_gpus)")
    plt.ylabel("avg_ppw")
    plt.title("Phase 2 diversity-efficiency trade-off")
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_diversity_efficiency_tradeoff.png", dpi=300)
    plt.close()

    # Overlap heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(overlap_matrix.values, cmap="Blues", vmin=0.0, vmax=1.0)
    plt.xticks(range(len(overlap_matrix.columns)), overlap_matrix.columns, rotation=30, ha="right")
    plt.yticks(range(len(overlap_matrix.index)), overlap_matrix.index)
    plt.colorbar(label="Overlap@5")
    plt.title("Phase 2 method overlap heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_method_overlap_heatmap.png", dpi=300)
    plt.close()

    # Frequency bars
    def _topk_freq(names_series: pd.Series, label: str) -> pd.DataFrame:
        freq = names_series.dropna().str.split(", ").explode().value_counts().head(15)
        df = freq.reset_index()
        df.columns = ["gpu_name", "count"]
        df["method"] = label
        return df

    power_freq = _topk_freq(per_game_df["power_top5_gpu_names"], "Power_Top5")
    ml_freq = _topk_freq(per_game_df["ml_utility_top5_gpu_names"], "ML_Utility_Top5")

    freq_df = pd.concat([power_freq, ml_freq], ignore_index=True)
    plt.figure(figsize=(10, 5))
    methods_list = list(freq_df["method"].unique())
    gpu_names = sorted(set(freq_df["gpu_name"]))
    x = np.arange(len(gpu_names))
    width = 0.4

    for idx, method in enumerate(methods_list):
        sub = freq_df[freq_df["method"] == method].set_index("gpu_name").reindex(gpu_names).fillna(0)
        offset = (idx - (len(methods_list) - 1) / 2) * width
        plt.bar(x + offset, sub["count"].values, width=width, label=method, alpha=0.8)

    plt.xticks(x, gpu_names, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Top 15 GPU frequency: Power vs ML Utility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phase2_frequency_bar_power_vs_ml.png", dpi=300)
    plt.close()

    # Save method notes
    notes = (
        "# Phase 2 Method Notes\n"
        "- KNN50_All tests raw feature-affinity retrieval from the full GPU database.\n"
        "- KNN50_Feasible tests feature-affinity retrieval after enforcing feasibility.\n"
        "- KNN50_Feasible_PPW_Top5 tests KNN retrieval plus PPW reranking.\n"
        "- Power_Top5 tests direct power-aware recommendation from the feasible set.\n"
        "- UtilityFormula_Top5 and ML_Utility_Top5 provide multi-objective alternatives.\n"
    )
    (out_dir / "phase2_method_notes.md").write_text(notes)


if __name__ == "__main__":
    main()
