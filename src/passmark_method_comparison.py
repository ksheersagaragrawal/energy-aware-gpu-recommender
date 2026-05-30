"""Run PassMark-based method comparison across recommendation baselines.

Produces per-game and aggregate tables for PPW, diversity, and top-1 share using
PassMark G3D predictions as the performance signal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

try:
    from sklearn.base import InconsistentVersionWarning
except Exception:
    InconsistentVersionWarning = None

if InconsistentVersionWarning is not None:
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from recommender import GAME_VECTORS, GPU_VECTORS, SOFT_THRESHOLD
from recommender import build_gpu_features_for_ml, load_ml_model


# Resolve paths relative to the repository root (two levels up from this file: repo/src -> repo)
BASE_DIR = Path(__file__).resolve().parent.parent

PREDICTIONS_PATH = BASE_DIR / "data" / "results" / "gpu_power_predictions.csv"
TDP_METRICS_PATH = BASE_DIR / "data" / "results" / "tdp_model_metrics.csv"
PSU_METRICS_PATH = BASE_DIR / "data" / "results" / "psu_model_metrics.csv"

TDP_USED_COL = "tdp_w_used"
PSU_USED_COL = "psu_w_used"
PPW_COL = "perf_per_watt"
GPU_ID_COL = "gpu_id"

OUTPUT_SUMMARY = BASE_DIR / "data" / "results" / "passmark_method_comparison_summary.csv"
OUTPUT_PER_GAME = BASE_DIR / "data" / "results" / "passmark_method_comparison_per_game.csv"
PLOTS_DIR = BASE_DIR / "results" / "plots" / "passmark_analysis"

PASTEL_COLORS = [
    "#AEC6CF",
    "#FFB347",
    "#B39EB5",
    "#77DD77",
    "#FF6961",
    "#FDFD96",
]

KNN_FEATURES = [
    "texture_rate",
    "pixel_rate",
    "memory_bandwidth_gbs",
    "tmus",
    "rops",
    "memory_speed_mhz",
    "boost_clock_mhz",
]

SOFT_FILTER_COLS = ["texture_rate", "pixel_rate", "memory_bandwidth_gbs", "tmus", "rops"]


@dataclass(frozen=True)
class Config:
    mode: str = "recom"
    k_top: int = 5
    knn_k: int = 50
    soft_threshold: float = 0.80
    train_split: float = 0.8
    random_seed: int = 42


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


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [passmark_method_comparison] {msg}", flush=True)


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


def attach_power_predictions(gpu_df: pd.DataFrame) -> pd.DataFrame:
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


def attach_passmark_perf_score(gpu_df: pd.DataFrame) -> pd.DataFrame:
    payload = load_ml_model()
    feature_cols = payload["feature_cols"]
    mem_type_cols = payload["mem_type_cols"]
    model = payload["model"]

    X = build_gpu_features_for_ml(gpu_df, feature_cols, mem_type_cols)
    X = np.nan_to_num(X, nan=0.0)

    df = gpu_df.copy()
    df["pred_g3d"] = model.predict(X)
    valid = df["pred_g3d"].replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        raise RuntimeError("No valid PassMark predictions available.")

    v_min, v_max = valid.min(), valid.max()
    if v_max == v_min:
        df["perf_score"] = 1.0
    else:
        df["perf_score"] = (df["pred_g3d"] - v_min) / (v_max - v_min)
        df["perf_score"] = df["perf_score"].clip(0.0, 1.0)

    df[PPW_COL] = np.where(df[TDP_USED_COL] > 0, df["perf_score"] / df[TDP_USED_COL], np.nan)
    df[GPU_ID_COL] = df["brand"].astype(str) + "||" + df["name"].astype(str)
    return df


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
    for col in SOFT_FILTER_COLS:
        req = game_row.get(col)
        if pd.isna(req) or req <= 0:
            continue
        min_val = req * threshold
        mask &= gpus[col] >= min_val
    return gpus[mask].copy()


def knn_candidates(gpu_pool: pd.DataFrame, game_row: pd.Series, n_neighbors: int) -> pd.DataFrame:
    game_vec = []
    gpu_matrix = []

    for col in KNN_FEATURES:
        gpu_vals = gpu_pool[col].values.astype(float)
        game_val = float(game_row.get(col) or 0)

        combined = np.concatenate([gpu_vals[~np.isnan(gpu_vals)], [game_val]])
        f_min, f_max = combined.min(), combined.max()

        if f_max > f_min:
            gpu_norm = np.where(
                np.isnan(gpu_vals),
                0.0,
                np.clip((gpu_vals - f_min) / (f_max - f_min), 1e-6, 1.0),
            )
            game_norm = 0.0 if np.isnan(game_val) else float(
                np.clip((game_val - f_min) / (f_max - f_min), 1e-6, 1.0)
            )
        else:
            gpu_norm = np.where(np.isnan(gpu_vals), 0.0, 1.0)
            game_norm = 0.0 if np.isnan(game_val) else 1.0

        gpu_matrix.append(gpu_norm)
        game_vec.append(game_norm)

    gpu_matrix = np.column_stack(gpu_matrix)
    game_vec = np.array(game_vec, dtype=float).reshape(1, -1)

    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(gpu_pool)), metric="euclidean")
    nn.fit(gpu_matrix)
    distances, indices = nn.kneighbors(game_vec)

    result = gpu_pool.iloc[indices[0]].copy()
    result["distance"] = distances[0]
    return result


def compute_candidate_metrics(candidate_df: pd.DataFrame) -> Dict[str, float]:
    if candidate_df.empty:
        return {
            "avg_ppw": np.nan,
            "avg_tdp": np.nan,
            "avg_psu": np.nan,
        }
    return {
        "avg_ppw": candidate_df[PPW_COL].mean(),
        "avg_tdp": candidate_df[TDP_USED_COL].mean(),
        "avg_psu": candidate_df[PSU_USED_COL].mean(),
    }


def _minmax_score(values: pd.Series) -> pd.Series:
    v_min, v_max = values.min(), values.max()
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


def utility_formula_scores(game_row: pd.Series, feasible: pd.DataFrame) -> pd.DataFrame:
    df = feasible.copy()

    df["ppw_score"] = _minmax_score(df[PPW_COL])
    df["low_tdp_score"] = _low_is_good_score(df[TDP_USED_COL])
    df["low_psu_score"] = _low_is_good_score(df[PSU_USED_COL])

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


def build_pair_features(game_row: pd.Series, gpu_df: pd.DataFrame) -> pd.DataFrame:
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
    features["directx_margin"] = features["gpu_directx"] - features["required_directx"]
    return features


def train_ml_utility_regressor(
    games_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
    train_games: set,
    test_games: set,
    config: Config,
) -> object:
    from sklearn.ensemble import GradientBoostingRegressor

    train_rows = []
    train_targets = []
    test_rows = []
    test_targets = []

    total_games = len(games_df)
    for i, (_, game_row) in enumerate(games_df.iterrows(), start=1):
        feasible = soft_filter(hard_filter(gpu_df, game_row), game_row, config.soft_threshold)
        if feasible.empty:
            if i % 500 == 0 or i == total_games:
                _log(f"ML utility training data pass: processed {i}/{total_games} games")
            continue
        scored = utility_formula_scores(game_row, feasible)
        # include feature affinity distance so training and prediction feature sets match
        affinity = feature_affinity_distance(game_row, feasible)
        features = build_pair_features(game_row, scored)
        features["feature_affinity_distance"] = affinity.values
        features = features.fillna(0.0).infer_objects(copy=False)
        targets = scored["utility_formula_score"].values

        if np.isnan(targets).all():
            continue

        if game_row["name"] in train_games:
            mask = ~np.isnan(targets)
            train_rows.append(features[mask])
            train_targets.append(targets[mask])
        elif game_row["name"] in test_games:
            mask = ~np.isnan(targets)
            test_rows.append(features[mask])
            test_targets.append(targets[mask])

        if i % 500 == 0 or i == total_games:
            _log(f"ML utility training data pass: processed {i}/{total_games} games")

    if not train_rows or not train_targets:
        raise RuntimeError("No valid training samples available for ML utility model.")

    X_train = pd.concat(train_rows, ignore_index=True).fillna(0.0).infer_objects(copy=False)
    y_train = pd.Series(np.concatenate(train_targets)).replace([np.inf, -np.inf], np.nan)
    valid_mask = y_train.notna()
    X_train = X_train.loc[valid_mask].reset_index(drop=True)
    y_train = y_train.loc[valid_mask]

    if y_train.empty:
        raise RuntimeError("ML utility model target values are all NaN after filtering.")

    model = GradientBoostingRegressor(random_state=config.random_seed)
    model.fit(X_train, y_train)
    return model


def compute_pareto_front(objectives: np.ndarray, max_size: int = 2000) -> np.ndarray | None:
    n = objectives.shape[0]
    if n == 0:
        return None
    if n > max_size:
        return None

    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominates_i = np.all(objectives <= objectives[i], axis=1) & np.any(objectives < objectives[i], axis=1)
        if np.any(dominates_i):
            is_efficient[i] = False
    return is_efficient


def assign_relevance_labels(base_scores: pd.Series, pareto_mask: np.ndarray | None) -> pd.Series:
    scores = base_scores.fillna(base_scores.median())
    if pareto_mask is None:
        p85, p60, p30 = np.percentile(scores, [85, 60, 30])
        labels = pd.Series(0, index=scores.index, dtype=int)
        labels[scores >= p30] = 1
        labels[scores >= p60] = 2
        labels[scores >= p85] = 3
        return labels

    p80 = np.percentile(scores, 80)
    p60 = np.percentile(scores, 60)
    median = np.percentile(scores, 50)

    labels = pd.Series(0, index=scores.index, dtype=int)
    pareto_idx = scores.index[pareto_mask]
    labels.loc[pareto_idx] = 2
    labels[(scores >= p60) & (labels == 0)] = 2
    labels[(scores >= median) & (labels == 0)] = 1
    labels[(scores >= p80) & (scores.index.isin(pareto_idx))] = 3
    return labels


def feature_affinity_distance(game_row: pd.Series, gpu_pool: pd.DataFrame) -> pd.Series:
    game_vec = []
    gpu_matrix = []

    for col in KNN_FEATURES:
        gpu_vals = gpu_pool[col].values.astype(float)
        game_val = float(game_row.get(col) or 0)

        combined = np.concatenate([gpu_vals[~np.isnan(gpu_vals)], [game_val]])
        f_min, f_max = combined.min(), combined.max()

        if f_max > f_min:
            gpu_norm = np.where(
                np.isnan(gpu_vals),
                0.0,
                np.clip((gpu_vals - f_min) / (f_max - f_min), 1e-6, 1.0),
            )
            game_norm = 0.0 if np.isnan(game_val) else float(
                np.clip((game_val - f_min) / (f_max - f_min), 1e-6, 1.0)
            )
        else:
            gpu_norm = np.where(np.isnan(gpu_vals), 0.0, 1.0)
            game_norm = 0.0 if np.isnan(game_val) else 1.0

        gpu_matrix.append(gpu_norm)
        game_vec.append(game_norm)

    gpu_matrix = np.column_stack(gpu_matrix)
    game_vec = np.array(game_vec, dtype=float).reshape(1, -1)
    distances = np.linalg.norm(gpu_matrix - game_vec, axis=1)
    return pd.Series(distances, index=gpu_pool.index)


def compute_base_scores(game_row: pd.Series, feasible: pd.DataFrame, affinity_distance: pd.Series) -> pd.DataFrame:
    df = feasible.copy()

    df["ppw_score"] = _minmax_score(df[PPW_COL])
    df["low_tdp_score"] = _low_is_good_score(df[TDP_USED_COL])
    df["low_psu_score"] = _low_is_good_score(df[PSU_USED_COL])

    affinity = affinity_distance.copy()
    if affinity.isna().all():
        df["feature_affinity_score"] = 0.5
    else:
        df["feature_affinity_score"] = _low_is_good_score(affinity.fillna(affinity.median()))

    game_perf = game_row.get("perf_score")
    margin = _compute_margin(df["perf_score"], game_perf)
    margin_score = margin.clip(lower=0.0, upper=0.25) / 0.25
    margin_score = margin_score.fillna(0.5).clip(0.0, 1.0)
    df["margin_score"] = margin_score

    target_margin = 0.15
    right_size = 1.0 - (margin - target_margin).abs() / target_margin
    right_size = right_size.clip(lower=0.0, upper=1.0).fillna(0.5)
    df["right_size_score"] = right_size

    df["base_score"] = (
        0.55 * df["ppw_score"]
        + 0.15 * df["low_tdp_score"]
        + 0.10 * df["low_psu_score"]
        + 0.10 * df["feature_affinity_score"]
        + 0.10 * df["right_size_score"]
    )

    return df


def build_pair_dataset(
    games_df: pd.DataFrame,
    gpu_df: pd.DataFrame,
    config: Config,
) -> Tuple[pd.DataFrame, pd.Series, List[int], Dict[str, Dict[str, int]]]:
    feature_rows = []
    label_rows = []
    group_sizes = []
    label_lookup: Dict[str, Dict[str, int]] = {}

    total_games = len(games_df)
    for i, (_, game_row) in enumerate(games_df.iterrows(), start=1):
        feasible = soft_filter(hard_filter(gpu_df, game_row), game_row, config.soft_threshold)
        if feasible.empty:
            if i % 500 == 0 or i == total_games:
                _log(f"LTR pair-build pass: processed {i}/{total_games} games")
            continue

        affinity = feature_affinity_distance(game_row, feasible)
        scored = compute_base_scores(game_row, feasible, affinity)

        best_ppw = feasible[PPW_COL].max()
        eff_regret = (best_ppw - feasible[PPW_COL]).fillna(best_ppw)
        dist_vals = affinity.fillna(affinity.median())
        objectives = np.column_stack([
            -feasible[PPW_COL].values,
            feasible[TDP_USED_COL].values,
            feasible[PSU_USED_COL].values,
            eff_regret.values,
            dist_vals.values,
        ])

        pareto_mask = compute_pareto_front(objectives)
        labels = assign_relevance_labels(scored["base_score"], pareto_mask)

        label_lookup[game_row["name"]] = {
            gid: int(lbl) for gid, lbl in zip(feasible[GPU_ID_COL], labels.values)
        }

        pair_features = build_pair_features(game_row, scored)
        pair_features["feature_affinity_distance"] = affinity.values
        pair_features = pair_features.fillna(0.0).infer_objects(copy=False)

        feature_rows.append(pair_features)
        label_rows.append(labels.values)
        group_sizes.append(len(labels))

        if i % 500 == 0 or i == total_games:
            _log(f"LTR pair-build pass: processed {i}/{total_games} games")

    X = pd.concat(feature_rows, ignore_index=True)
    y = pd.Series(np.concatenate(label_rows))
    return X, y, group_sizes, label_lookup


def train_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: List[int],
) -> object:
    import xgboost as xgb

    params = {
        "objective": "rank:ndcg",
        "eval_metric": "ndcg@5",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbosity": 1,
        "n_jobs": 1,
    }

    # GPU options (will be used if xgboost is built with GPU support)
    params.update({
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "gpu_id": 0,
    })

    model = xgb.XGBRanker(**params)
    model.fit(X_train, y_train, group=group_train)
    return model


def compute_methods_for_game(
    game_row: pd.Series,
    gpu_df: pd.DataFrame,
    config: Config,
    ml_model: object,
    ltr_model: object,
    label_lookup: Dict[str, Dict[str, int]],
) -> Dict[str, Dict[str, float]]:
    feasible = soft_filter(hard_filter(gpu_df, game_row), game_row, config.soft_threshold)
    if feasible.empty:
        return {}

    best_ppw = feasible[PPW_COL].max()

    knn_feasible = knn_candidates(feasible, game_row, config.knn_k)
    knn_feasible_ppw = knn_feasible.sort_values(PPW_COL, ascending=False).head(config.k_top)
    power_topk = feasible.sort_values(PPW_COL, ascending=False).head(config.k_top)

    affinity = feature_affinity_distance(game_row, feasible)
    utility_scored = compute_base_scores(game_row, feasible, affinity)
    utility_topk = utility_scored.sort_values("base_score", ascending=False).head(config.k_top)

    ml_features = build_pair_features(game_row, feasible).fillna(0.0).infer_objects(copy=False)
    ml_features["feature_affinity_distance"] = affinity.values
    ml_preds = ml_model.predict(ml_features)
    ml_scored = feasible.copy()
    ml_scored["ml_utility_score"] = ml_preds
    ml_topk = ml_scored.sort_values("ml_utility_score", ascending=False).head(config.k_top)

    ltr_features = build_pair_features(game_row, feasible).fillna(0.0).infer_objects(copy=False)
    ltr_features["feature_affinity_distance"] = affinity.values
    ltr_preds = ltr_model.predict(ltr_features)
    ltr_scored = feasible.copy()
    ltr_scored["ltr_score"] = ltr_preds
    ltr_topk = ltr_scored.sort_values("ltr_score", ascending=False).head(config.k_top)

    def _metrics(df: pd.DataFrame) -> Dict[str, float]:
        return compute_candidate_metrics(df)

    return {
        "KNN50-Feasible": _metrics(knn_feasible),
        "KNN50-Feasible-PPW-Top5": _metrics(knn_feasible_ppw),
        "Power-Top5": _metrics(power_topk),
        "UtilityFormula-Top5": _metrics(utility_topk),
        "ML-Utility-Top5": _metrics(ml_topk),
        "LTR-Utility-Top5": _metrics(ltr_topk),
        "_names": {
            "KNN50-Feasible": ", ".join(knn_feasible["name"].tolist()),
            "KNN50-Feasible-PPW-Top5": ", ".join(knn_feasible_ppw["name"].tolist()),
            "Power-Top5": ", ".join(power_topk["name"].tolist()),
            "UtilityFormula-Top5": ", ".join(utility_topk["name"].tolist()),
            "ML-Utility-Top5": ", ".join(ml_topk["name"].tolist()),
            "LTR-Utility-Top5": ", ".join(ltr_topk["name"].tolist()),
        },
    }


def summarize(per_game_rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(per_game_rows)
    methods = [
        "KNN50-Feasible",
        "KNN50-Feasible-PPW-Top5",
        "Power-Top5",
        "UtilityFormula-Top5",
        "ML-Utility-Top5",
        "LTR-Utility-Top5",
    ]

    rows = []
    for method in methods:
        rows.append({
            "method": method,
            "avg_ppw": df[f"{method}_avg_ppw"].mean(),
            "unique_gpus": df[f"{method}_gpu_names"].str.split(", ").explode().nunique(),
            "top1_share": _top1_share(df[f"{method}_gpu_names"]),
        })

    return pd.DataFrame(rows)


def _top1_share(names: pd.Series) -> float:
    top1 = names.dropna().apply(lambda v: v.split(", ")[0] if v else "")
    if top1.empty:
        return np.nan
    return float(top1.value_counts(normalize=True).iloc[0])


def plot_summary(summary_df: pd.DataFrame) -> None:
    plots_dir = Path(PLOTS_DIR)
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=600)
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["avg_ppw"], color=PASTEL_COLORS[: len(summary_df)])
    ax.set_title("PassMark-based PPW across methods")
    ax.set_ylabel("Avg PPW")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"], rotation=25, ha="right", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "passmark_methods_ppw.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=600)
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["unique_gpus"], color=PASTEL_COLORS[: len(summary_df)])
    ax.set_title("PassMark-based unique GPUs")
    ax.set_ylabel("Unique GPU count")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"], rotation=25, ha="right", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "passmark_methods_unique.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=600)
    x = np.arange(len(summary_df))
    ax.bar(x, summary_df["top1_share"], color=PASTEL_COLORS[: len(summary_df)])
    ax.set_title("PassMark-based top-1 share")
    ax.set_ylabel("Top-1 share")
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["method"], rotation=25, ha="right", fontweight="bold")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(plots_dir / "passmark_methods_top1_share.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="PassMark method comparison")
    parser.add_argument("--mode", choices=["min", "recom"], default="recom")
    parser.add_argument("--k-top", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=SOFT_THRESHOLD)
    args = parser.parse_args()

    _configure_style()
    _log("Run started")

    _log(f"Loading game vectors ({args.mode}) and GPU vectors")
    games = pd.read_csv(GAME_VECTORS[args.mode])
    gpus = pd.read_csv(GPU_VECTORS)
    _log(f"Loaded games={len(games)}, gpus={len(gpus)}")
    gpus = attach_power_predictions(gpus)
    gpus = attach_passmark_perf_score(gpus)
    _log("Attached power predictions and PassMark-based perf scores")

    config = Config(
        mode=args.mode,
        k_top=args.k_top,
        knn_k=args.knn_k,
        soft_threshold=args.threshold,
    )

    train_games, test_games = _train_test_split_games(games, config)
    _log(f"Train/test split: train_games={len(train_games)}, test_games={len(test_games)}")

    _log(f"Training ML utility regressor on {len(train_games)} train games")
    ml_model = train_ml_utility_regressor(games, gpus, train_games, test_games, config)
    _log("ML utility regressor trained")

    _log("Building pair dataset for LTR training")
    X_train, y_train, group_train, label_lookup = build_pair_dataset(
        games[games["name"].isin(train_games)],
        gpus,
        config,
    )
    _log(f"Built pair dataset: X={X_train.shape}, y={y_train.shape}, groups={len(group_train)}")

    _log("Training LTR ranker (XGBRanker)")
    ltr_model = train_ranker(X_train, y_train, group_train)
    _log("LTR ranker trained")

    per_game_rows: List[Dict[str, object]] = []

    eval_total = len(test_games)
    eval_seen = 0
    for _, game_row in games.iterrows():
        if game_row["name"] not in test_games:
            continue
        eval_seen += 1

        metrics = compute_methods_for_game(game_row, gpus, config, ml_model, ltr_model, label_lookup)
        if not metrics:
            if eval_seen % 250 == 0 or eval_seen == eval_total:
                _log(f"Evaluation pass: processed {eval_seen}/{eval_total} test games")
            continue

        row = {"game_name": game_row["name"]}
        names = metrics.pop("_names")
        for method, vals in metrics.items():
            row[f"{method}_avg_ppw"] = vals["avg_ppw"]
            row[f"{method}_avg_tdp"] = vals["avg_tdp"]
            row[f"{method}_avg_psu"] = vals["avg_psu"]
            row[f"{method}_gpu_names"] = names[method]
        per_game_rows.append(row)
        if eval_seen % 250 == 0 or eval_seen == eval_total:
            _log(f"Evaluation pass: processed {eval_seen}/{eval_total} test games")

    per_game_df = pd.DataFrame(per_game_rows)
    summary_df = summarize(per_game_rows)

    Path(OUTPUT_SUMMARY).parent.mkdir(parents=True, exist_ok=True)
    per_game_df.to_csv(OUTPUT_PER_GAME, index=False)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    plot_summary(summary_df)

    _log("Saved outputs:")
    _log(f"  {OUTPUT_PER_GAME}")
    _log(f"  {OUTPUT_SUMMARY}")
    _log(f"  {Path(PLOTS_DIR) / 'passmark_methods_ppw.png'}")
    _log(f"  {Path(PLOTS_DIR) / 'passmark_methods_unique.png'}")
    _log(f"  {Path(PLOTS_DIR) / 'passmark_methods_top1_share.png'}")


def _train_test_split_games(games: pd.DataFrame, config: Config) -> Tuple[set, set]:
    from sklearn.model_selection import train_test_split

    train_games, test_games = train_test_split(
        games["name"].values,
        train_size=config.train_split,
        random_state=config.random_seed,
        shuffle=True,
    )
    return set(train_games), set(test_games)


if __name__ == "__main__":
    main()
