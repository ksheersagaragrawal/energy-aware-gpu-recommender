"""Shared helpers for the recommendation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

try:
    from src.recommender import (
        EPSILON,
        GAME_VECTORS,
        GPU_VECTORS,
        KNN_FEATURE_MAP,
        SOFT_FILTER_MAP,
        build_gpu_features_for_ml,
        load_ml_model,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from recommender import (  # type: ignore
        EPSILON,
        GAME_VECTORS,
        GPU_VECTORS,
        KNN_FEATURE_MAP,
        SOFT_FILTER_MAP,
        build_gpu_features_for_ml,
        load_ml_model,
    )


PREDICTIONS_PATH = "data/results/gpu_power_predictions.csv"
TDP_METRICS_PATH = "data/results/tdp_model_metrics.csv"
PSU_METRICS_PATH = "data/results/psu_model_metrics.csv"

GPU_ID_COL = "gpu_id"
TDP_USED_COL = "tdp_w_used"
PSU_USED_COL = "psu_w_used"
PPW_COL = "perf_per_watt"


@dataclass(frozen=True)
class Phase1Config:
    mode: str = "recom"
    k_top: int = 5
    knn_k: int = 50
    soft_threshold: float = 0.80
    output_dir: str = "phase1_outputs"
    num_workers: int = 1
    log_every: int = 200


def _safe_minmax(values: pd.Series) -> pd.Series:
    v_min, v_max = values.min(), values.max()
    if pd.isna(v_min) or pd.isna(v_max) or v_max == v_min:
        return pd.Series(0.5, index=values.index)
    return ((values - v_min) / (v_max - v_min)).clip(0.0, 1.0)


def _low_is_good(values: pd.Series) -> pd.Series:
    return (1.0 - _safe_minmax(values)).clip(0.0, 1.0)


def _compute_margin(gpu_perf: pd.Series, game_perf: float) -> pd.Series:
    if pd.isna(game_perf) or game_perf == 0:
        return pd.Series(np.nan, index=gpu_perf.index)
    return (gpu_perf - game_perf) / game_perf


def _load_best_prediction_columns() -> Tuple[Optional[str], Optional[str]]:
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

    df[PPW_COL] = np.where(df[TDP_USED_COL] > 0, df["perf_score"] / df[TDP_USED_COL], np.nan)
    df[GPU_ID_COL] = df["brand"].astype(str) + "||" + df["name"].astype(str)
    return df


class TopKRecommendationAnalyzer:
    def __init__(self, games_df: pd.DataFrame, gpu_df: pd.DataFrame, config: Phase1Config):
        self.games_df = games_df.copy()
        self.gpu_df = gpu_df.copy()
        self.config = config

    def get_feasible_gpus(self, game_row: pd.Series) -> pd.DataFrame:
        gpus = self.gpu_df.copy()
        mask = pd.Series(True, index=gpus.index)

        vram_req = game_row.get("min_vram_mb")
        if pd.notna(vram_req) and vram_req > 0:
            mask &= gpus["memory_mb"] >= vram_req

        dx_req = game_row.get("min_direct_x")
        if pd.notna(dx_req) and dx_req > 0:
            mask &= gpus["direct_x"] >= dx_req

        gpus = gpus[mask].copy()

        mask = pd.Series(True, index=gpus.index)
        for game_col, gpu_col in SOFT_FILTER_MAP.items():
            req = game_row.get(game_col)
            if pd.isna(req) or req <= 0:
                continue
            min_val = req * self.config.soft_threshold
            mask &= gpus[gpu_col] >= min_val

        return gpus[mask].copy()

    def get_power_topk(self, game_row: pd.Series, k: int = 5) -> pd.DataFrame:
        feasible = self.get_feasible_gpus(game_row)
        if feasible.empty:
            return feasible
        return feasible.sort_values(PPW_COL, ascending=False).head(k).copy()

    def get_knn_candidates(self, game_row: pd.Series, n_neighbors: int = 50) -> pd.DataFrame:
        return self._knn_candidates_from_pool(self.gpu_df, game_row, n_neighbors)

    def get_knn_candidates_feasible(self, game_row: pd.Series, n_neighbors: int = 50) -> pd.DataFrame:
        feasible = self.get_feasible_gpus(game_row)
        if feasible.empty:
            return feasible
        return self._knn_candidates_from_pool(feasible, game_row, n_neighbors)

    def _knn_candidates_from_pool(
        self,
        gpu_pool: pd.DataFrame,
        game_row: pd.Series,
        n_neighbors: int,
    ) -> pd.DataFrame:
        game_vec = []
        gpu_matrix = []

        for _, (game_col, gpu_col) in KNN_FEATURE_MAP.items():
            gpu_vals = gpu_pool[gpu_col].values.astype(float)
            game_val = float(game_row.get(game_col) or 0)

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

        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(gpu_pool)), metric="euclidean")
        nn.fit(gpu_matrix)
        distances, indices = nn.kneighbors(game_vec)

        result = gpu_pool.iloc[indices[0]].copy()
        result["distance"] = distances[0]
        return result

    def compute_candidate_metrics(
        self,
        candidate_df: pd.DataFrame,
        game_row: pd.Series,
        best_ppw: float,
    ) -> Dict[str, float]:
        if candidate_df.empty:
            return {
                "avg_tdp": np.nan,
                "avg_psu": np.nan,
                "avg_ppw": np.nan,
                "avg_overprov": np.nan,
                "avg_eff_regret": np.nan,
            }

        game_perf = game_row.get("perf_score")
        selected_perf = candidate_df["perf_score"]
        overprov_abs = selected_perf - game_perf

        if pd.notna(game_perf) and game_perf != 0:
            overprov_rel = overprov_abs / game_perf
        else:
            overprov_rel = pd.Series(np.nan, index=candidate_df.index)

        if pd.notna(best_ppw) and best_ppw > 0:
            eff_regret_abs = best_ppw - candidate_df[PPW_COL]
            eff_regret_rel = eff_regret_abs / best_ppw
        else:
            eff_regret_rel = pd.Series(np.nan, index=candidate_df.index)

        return {
            "avg_tdp": candidate_df[TDP_USED_COL].mean(),
            "avg_psu": candidate_df[PSU_USED_COL].mean(),
            "avg_ppw": candidate_df[PPW_COL].mean(),
            "avg_overprov": overprov_rel.mean(),
            "avg_eff_regret": eff_regret_rel.mean(),
        }


def _minmax_score(values: pd.Series) -> pd.Series:
    return _safe_minmax(values)


def _low_is_good_score(values: pd.Series) -> pd.Series:
    return _low_is_good(values)


def _utility_formula_scores(game_row: pd.Series, feasible: pd.DataFrame) -> pd.DataFrame:
    df = feasible.copy()

    df["ppw_score"] = _minmax_score(df[PPW_COL])
    df["low_tdp_score"] = _low_is_good_score(df[TDP_USED_COL])
    df["low_psu_score"] = _low_is_good_score(df[PSU_USED_COL])

    game_perf = game_row.get("perf_score")
    margin = _compute_margin(df["perf_score"], game_perf)

    margin_score = margin.clip(lower=0.0, upper=0.25) / 0.25
    df["margin_score"] = margin_score.fillna(0.5).clip(0.0, 1.0)

    target_margin = 0.15
    right_size = 1.0 - (margin - target_margin).abs() / target_margin
    df["right_size_score"] = right_size.clip(lower=0.0, upper=1.0).fillna(0.5)

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
    metrics: Dict[str, object] = {}
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
            model = xgb.XGBRegressor(**local_params)
            model.fit(X_train, y_train)
            backend = local_params.get("device", "cpu")

        try:
            _fit_xgb(params | {"tree_method": "hist", "device": "cuda"})
        except Exception:
            try:
                _fit_xgb(params | {"tree_method": "gpu_hist"})
            except Exception:
                _fit_xgb(params | {"tree_method": "hist", "device": "cpu"})
                backend = "cpu"

        preds = model.predict(X_test)
        metrics.update({
            "model": "XGBoostRegressor",
            "backend": backend,
            "train_time_sec": np.nan,
            "infer_time_sec": np.nan,
            "test_mae": mean_absolute_error(y_test, preds),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "test_r2": r2_score(y_test, preds),
        })
        return model, metrics
    except Exception:
        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics.update({
            "model": "GradientBoostingRegressor",
            "backend": "cpu",
            "train_time_sec": np.nan,
            "infer_time_sec": np.nan,
            "test_mae": mean_absolute_error(y_test, preds),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
            "test_r2": r2_score(y_test, preds),
        })
        return model, metrics
