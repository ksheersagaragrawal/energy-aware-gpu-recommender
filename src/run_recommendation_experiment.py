"""Unified recommendation experiment runner.

Runs the report-aligned recommendation pipeline twice:
1. static scoring mode
2. g3d scoring mode

Both modes use the same feasibility filtering, the same train/test split by game,
and the same LTR setup. Outputs are written into one root directory with a
per-mode subdirectory plus top-level comparison CSVs.
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    from src.phase1_topk_knn_analysis import Phase1Config, TopKRecommendationAnalyzer, _attach_power_predictions
    from src.phase2_ml_utility_analysis import _build_pair_features, _utility_formula_scores
except ImportError:  # pragma: no cover - direct script execution fallback
    from recommender import (
        EPSILON,
        GAME_VECTORS,
        GPU_VECTORS,
        KNN_FEATURE_MAP,
        SOFT_FILTER_MAP,
        build_gpu_features_for_ml,
        load_ml_model,
    )
    from phase1_topk_knn_analysis import Phase1Config, TopKRecommendationAnalyzer, _attach_power_predictions
    from phase2_ml_utility_analysis import _build_pair_features, _utility_formula_scores


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "recommendation_final"
SOFT_THRESHOLD = 0.80
RANDOM_SEED = 42
GPU_ID_COL = "gpu_id"
TDP_USED_COL = "tdp_w_used"
PSU_USED_COL = "psu_w_used"
PPW_COL = "perf_per_watt"
MAX_CANDIDATES_PER_GAME = 150


@dataclass(frozen=True)
class ExperimentConfig:
    requirements_mode: str = "recom"
    scoring_mode: str = "static"
    k_top: int = 5
    knn_k: int = 50
    soft_threshold: float = SOFT_THRESHOLD
    train_split: float = 0.8
    random_seed: int = RANDOM_SEED
    output_dir: Path = DEFAULT_OUTPUT_DIR


def _log(message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [run_recommendation_experiment] {message}", flush=True)


def _require_columns(df: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


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


def _compute_feature_affinity_distance(game_row: pd.Series, gpu_pool: pd.DataFrame) -> pd.Series:
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
    distances = np.linalg.norm(gpu_matrix - game_vec, axis=1)
    return pd.Series(distances, index=gpu_pool.index)


def _compute_base_scores(game_row: pd.Series, feasible: pd.DataFrame, affinity_distance: pd.Series) -> pd.DataFrame:
    df = feasible.copy()

    df["ppw_score"] = _safe_minmax(df[PPW_COL])
    df["low_tdp_score"] = _low_is_good(df[TDP_USED_COL])
    df["low_psu_score"] = _low_is_good(df[PSU_USED_COL])

    affinity = affinity_distance.copy()
    if affinity.isna().all():
        df["feature_affinity_score"] = 0.5
    else:
        df["feature_affinity_score"] = _low_is_good(affinity.fillna(affinity.median()))

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


def _compute_pareto_front(objectives: np.ndarray, max_size: int = 300) -> Optional[np.ndarray]:
    n = objectives.shape[0]
    if n == 0 or n > max_size:
        return None

    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominates_i = np.all(objectives <= objectives[i], axis=1) & np.any(objectives < objectives[i], axis=1)
        if np.any(dominates_i):
            is_efficient[i] = False
    return is_efficient


def _assign_relevance_labels(base_scores: pd.Series, pareto_mask: Optional[np.ndarray]) -> pd.Series:
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


def _prepare_games(requirements_mode: str) -> pd.DataFrame:
    if requirements_mode not in GAME_VECTORS:
        raise KeyError(f"Unknown requirements mode: {requirements_mode}")

    games_path = Path(GAME_VECTORS[requirements_mode])
    if not games_path.exists():
        raise FileNotFoundError(f"Game vectors not found: {games_path}")
    games_df = pd.read_csv(games_path)

    _require_columns(
        games_df,
        [
            "name",
            "perf_score",
            "min_vram_mb",
            "min_direct_x",
            "texture_rate",
            "pixel_rate",
            "memory_bandwidth_gbs",
            "tmus",
            "rops",
            "memory_speed_mhz",
            "boost_clock_mhz",
        ],
        f"game vectors ({requirements_mode})",
    )
    return games_df


def _prepare_gpus(scoring_mode: str) -> pd.DataFrame:
    gpu_path = Path(GPU_VECTORS)
    if not gpu_path.exists():
        raise FileNotFoundError(f"GPU vectors not found: {gpu_path}")

    gpu_df = pd.read_csv(gpu_path)
    _require_columns(
        gpu_df,
        [
            "brand",
            "name",
            "memory_mb",
            "direct_x",
            "tdp_w",
            "psu_w",
            "perf_score",
            "texture_rate",
            "pixel_rate",
            "memory_bandwidth_gbs",
            "tmus",
            "rops",
            "memory_speed_mhz",
            "boost_clock_mhz",
            "memory_type",
        ],
        "gpu vectors",
    )

    gpu_df = _attach_power_predictions(gpu_df)

    if scoring_mode == "static":
        gpu_df["perf_score_mode"] = pd.to_numeric(gpu_df["perf_score"], errors="coerce")
        gpu_df["score_source"] = "static"
        return gpu_df

    if scoring_mode != "g3d":
        raise ValueError(f"Unsupported scoring mode: {scoring_mode}")

    payload_path = ROOT / "models" / "gpu_performance_model.pkl"
    if not payload_path.exists():
        raise FileNotFoundError(
            f"PassMark performance model not found: {payload_path}. "
            "Run src/train_ml_recommender.py first."
        )

    payload = load_ml_model()
    feature_cols = payload["feature_cols"]
    mem_type_cols = payload["mem_type_cols"]
    model = payload["model"]

    X = build_gpu_features_for_ml(gpu_df, feature_cols, mem_type_cols)
    X = np.nan_to_num(X, nan=0.0)
    gpu_df["pred_g3d"] = model.predict(X)

    valid = pd.Series(gpu_df["pred_g3d"]).replace([np.inf, -np.inf], np.nan).dropna()
    if valid.empty:
        raise RuntimeError("PassMark model produced no valid predictions.")

    v_min, v_max = valid.min(), valid.max()
    if v_max == v_min:
        gpu_df["perf_score_mode"] = 1.0
    else:
        gpu_df["perf_score_mode"] = ((gpu_df["pred_g3d"] - v_min) / (v_max - v_min)).clip(0.0, 1.0)
    gpu_df["score_source"] = "g3d"
    return gpu_df


def _validate_feasible_inputs(games_df: pd.DataFrame, gpu_df: pd.DataFrame) -> None:
    _require_columns(
        games_df,
        ["name", "perf_score", "min_vram_mb", "min_direct_x", "texture_rate", "pixel_rate", "memory_bandwidth_gbs", "tmus", "rops"],
        "game vectors",
    )
    _require_columns(
        gpu_df,
        ["brand", "name", "memory_mb", "direct_x", "tdp_w", "psu_w", "texture_rate", "pixel_rate", "memory_bandwidth_gbs", "tmus", "rops", "memory_speed_mhz", "boost_clock_mhz"],
        "gpu vectors",
    )


def _train_ranker(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    group_train: List[int],
    X_test: pd.DataFrame,
) -> Tuple[object, Dict[str, object], np.ndarray]:
    metrics: Dict[str, object] = {}
    backend = "cpu"

    try:
        import xgboost as xgb  # type: ignore

        params = {
            "objective": "rank:ndcg",
            "eval_metric": "ndcg@5",
            "learning_rate": 0.05,
            "max_depth": 6,
            "n_estimators": 100,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": RANDOM_SEED,
            "verbosity": 0,
            "n_jobs": 1,
            "tree_method": "hist",
            "device": "cpu",
        }

        start = time.time()
        model = xgb.XGBRanker(**params)
        model.fit(X_train, y_train, group=group_train)
        train_time = time.time() - start
        backend = "cpu"

        infer_start = time.time()
        preds = model.predict(X_test)
        infer_time = time.time() - infer_start

        metrics.update({
            "model": "XGBRanker",
            "backend": backend,
            "train_time_sec": train_time,
            "inference_time_sec": infer_time,
        })
        return model, metrics, preds

    except Exception as error:
        _log(f"XGBRanker unavailable; using fallback ranker. Reason: {error}")

    try:
        import lightgbm as lgb  # type: ignore

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.05,
            "n_estimators": 100,
            "random_state": RANDOM_SEED,
        }

        start = time.time()
        try:
            model = lgb.LGBMRanker(**params, device_type="gpu")
            model.fit(X_train, y_train, group=group_train)
            backend = "gpu"
        except Exception:
            model = lgb.LGBMRanker(**params, device_type="cpu")
            model.fit(X_train, y_train, group=group_train)
            backend = "cpu"
        train_time = time.time() - start

        infer_start = time.time()
        preds = model.predict(X_test)
        infer_time = time.time() - infer_start

        metrics.update({
            "model": "LGBMRanker",
            "backend": backend,
            "train_time_sec": train_time,
            "inference_time_sec": infer_time,
        })
        return model, metrics, preds

    except Exception as error:
        _log(f"LightGBM ranker unavailable; using pointwise fallback. Reason: {error}")

    from sklearn.ensemble import GradientBoostingRegressor

    start = time.time()
    model = GradientBoostingRegressor(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    train_time = time.time() - start

    infer_start = time.time()
    preds = model.predict(X_test)
    infer_time = time.time() - infer_start

    metrics.update({
        "model": "GradientBoostingRegressor",
        "backend": "cpu",
        "train_time_sec": train_time,
        "inference_time_sec": infer_time,
        "warning": "pointwise_fallback",
    })
    return model, metrics, preds


def _compute_ndcg_at_k(labels: List[int], k: int = 5) -> float:
    labels_k = labels[:k]
    if not labels_k:
        return float("nan")
    gains = np.array([2 ** rel - 1 for rel in labels_k], dtype=float)
    discounts = np.log2(np.arange(2, len(labels_k) + 2))
    dcg = (gains / discounts).sum()
    ideal = sorted(labels, reverse=True)[:k]
    ideal_gains = np.array([2 ** rel - 1 for rel in ideal], dtype=float)
    ideal_dcg = (ideal_gains / discounts).sum()
    return float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0


def _compute_recall_label3_at_k(labels: List[int], k: int = 5) -> float:
    total = sum(1 for label in labels if label == 3)
    if total == 0:
        return float("nan")
    hit = sum(1 for label in labels[:k] if label == 3)
    return float(hit / total)


def _top1_share(names_series: pd.Series) -> float:
    top1 = names_series.dropna().apply(lambda value: value.split(", ")[0] if value else "")
    if top1.empty:
        return float("nan")
    return float(top1.value_counts(normalize=True).iloc[0])


def _series_stats(values: pd.Series) -> Dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _pearson_corr(a: pd.Series, b: pd.Series) -> float:
    a_clean = pd.to_numeric(a, errors="coerce")
    b_clean = pd.to_numeric(b, errors="coerce")
    valid = a_clean.notna() & b_clean.notna()
    if valid.sum() < 2:
        return float("nan")
    return float(a_clean[valid].corr(b_clean[valid]))


def _label_distribution_frame(label_counts: List[Dict[str, object]]) -> pd.DataFrame:
    if not label_counts:
        return pd.DataFrame(columns=["split", "game_name", "label_0", "label_1", "label_2", "label_3"])
    return pd.DataFrame(label_counts)


class RecommendationExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.games_df = _prepare_games(config.requirements_mode)
        self.gpu_df = _prepare_gpus(config.scoring_mode)
        self.gpu_df["perf_score"] = self.gpu_df["perf_score_mode"].astype(float)
        self.gpu_df[PPW_COL] = np.where(
            self.gpu_df[TDP_USED_COL] > 0,
            self.gpu_df["perf_score"] / self.gpu_df[TDP_USED_COL],
            np.nan,
        )
        _validate_feasible_inputs(self.games_df, self.gpu_df)

        self.analyzer = TopKRecommendationAnalyzer(
            self.games_df,
            self.gpu_df,
            Phase1Config(
                mode=config.requirements_mode,
                k_top=config.k_top,
                knn_k=config.knn_k,
                soft_threshold=config.soft_threshold,
            ),
        )

        self.train_games, self.test_games = train_test_split(
            self.games_df["name"].values,
            train_size=self.config.train_split,
            random_state=self.config.random_seed,
            shuffle=True,
        )
        self.train_games = set(self.train_games)
        self.test_games = set(self.test_games)
        self.game_cache = self._build_game_cache()

    def _build_game_cache(self) -> Dict[str, Dict[str, object]]:
        cache: Dict[str, Dict[str, object]] = {}
        total = len(self.games_df)
        for i, (_, game_row) in enumerate(self.games_df.iterrows(), start=1):
            feasible = self.analyzer.get_feasible_gpus(game_row)
            if feasible.empty:
                cache[game_row["name"]] = {"feasible_idx": np.array([], dtype=int)}
            else:
                affinity_distance = _compute_feature_affinity_distance(game_row, feasible)
                cache[game_row["name"]] = {
                    "feasible_idx": feasible.index.to_numpy(),
                    "affinity_distance": affinity_distance,
                }

            if i % 500 == 0 or i == total:
                _log(f"cached feasibility for {i}/{total} games")

        return cache

    def _cached_feasible(self, game_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        item = self.game_cache.get(game_name)
        if not item:
            return pd.DataFrame(), pd.Series(dtype=float)

        idx = item.get("feasible_idx")
        if idx is None or len(idx) == 0:
            return pd.DataFrame(), pd.Series(dtype=float)

        feasible = self.gpu_df.loc[idx].copy()
        affinity_distance = item.get("affinity_distance")
        if not isinstance(affinity_distance, pd.Series):
            affinity_distance = pd.Series(dtype=float)
        return feasible, affinity_distance

    def _build_pair_dataset(self, game_subset: pd.DataFrame, split: str) -> Tuple[pd.DataFrame, pd.Series, List[int], Dict[str, Dict[str, int]], pd.DataFrame]:
        feature_rows = []
        label_rows = []
        group_sizes = []
        label_dist_rows = []
        label_lookup: Dict[str, Dict[str, int]] = {}
        total = len(game_subset)

        for i, (_, game_row) in enumerate(game_subset.iterrows(), start=1):
            feasible, affinity_distance = self._cached_feasible(game_row["name"])
            if feasible.empty:
                continue

            scored = _compute_base_scores(game_row, feasible, affinity_distance)

            best_ppw = feasible[PPW_COL].max()
            eff_regret = (best_ppw - feasible[PPW_COL]).fillna(best_ppw)
            dist_vals = affinity_distance.fillna(affinity_distance.median())
            objectives = np.column_stack([
                -feasible[PPW_COL].values,
                feasible[TDP_USED_COL].values,
                feasible[PSU_USED_COL].values,
                eff_regret.values,
                dist_vals.values,
            ])

            pareto_mask = _compute_pareto_front(objectives)
            labels = _assign_relevance_labels(scored["base_score"], pareto_mask)

            label_dist_rows.append({
                "split": split,
                "game_name": game_row["name"],
                "label_0": int((labels == 0).sum()),
                "label_1": int((labels == 1).sum()),
                "label_2": int((labels == 2).sum()),
                "label_3": int((labels == 3).sum()),
            })

            label_lookup[game_row["name"]] = {
                gid: int(lbl) for gid, lbl in zip(feasible[GPU_ID_COL], labels.values)
            }

            if len(scored) > MAX_CANDIDATES_PER_GAME:
                selected_idx = scored["base_score"].nlargest(MAX_CANDIDATES_PER_GAME).index
                train_scored = scored.loc[selected_idx]
                train_affinity = affinity_distance.loc[selected_idx]
                train_labels = labels.loc[selected_idx]
            else:
                train_scored = scored
                train_affinity = affinity_distance
                train_labels = labels

            pair_features = _build_pair_features(game_row, train_scored)
            pair_features["feature_affinity_distance"] = train_affinity.values
            pair_features = _numeric_frame(pair_features)

            feature_rows.append(pair_features)
            label_rows.append(train_labels.values)
            group_sizes.append(len(train_labels))

            if i % 500 == 0 or i == total:
                _log(f"{split} pair-build progress: {i}/{total} games")

        if not feature_rows:
            raise RuntimeError(f"No feasible game-GPU pairs found for {split} split.")

        X = pd.concat(feature_rows, ignore_index=True)
        y = pd.Series(np.concatenate(label_rows))
        label_dist = _label_distribution_frame(label_dist_rows)
        return X, y, group_sizes, label_lookup, label_dist

    def _evaluate_game(
        self,
        game_row: pd.Series,
        model: object,
        label_lookup: Dict[str, Dict[str, int]],
    ) -> Dict[str, object]:
        feasible, affinity_distance = self._cached_feasible(game_row["name"])
        if feasible.empty:
            return {}

        best_ppw = feasible[PPW_COL].max()
        power_topk = feasible.sort_values(PPW_COL, ascending=False).head(self.config.k_top).copy()

        ltr_features = _build_pair_features(game_row, feasible)
        ltr_features["feature_affinity_distance"] = affinity_distance.values
        ltr_features = _numeric_frame(ltr_features)
        ltr_preds = model.predict(ltr_features)
        ltr_scored = feasible.copy()
        ltr_scored["ltr_score"] = ltr_preds
        ltr_topk = ltr_scored.sort_values("ltr_score", ascending=False).head(self.config.k_top)

        power_metrics = self.analyzer.compute_candidate_metrics(power_topk, game_row, best_ppw)
        ltr_metrics = self.analyzer.compute_candidate_metrics(ltr_topk, game_row, best_ppw)

        label_map = label_lookup.get(game_row["name"], {})

        def _labels_for(df: pd.DataFrame) -> List[int]:
            return [label_map.get(gid, 0) for gid in df[GPU_ID_COL].tolist()]

        power_labels = _labels_for(power_topk)
        ltr_labels = _labels_for(ltr_topk)

        return {
            "game_name": game_row["name"],
            "power_top5_avg_tdp": power_metrics["avg_tdp"],
            "power_top5_avg_psu": power_metrics["avg_psu"],
            "power_top5_avg_ppw": power_metrics["avg_ppw"],
            "power_top5_avg_efficiency_regret": power_metrics["avg_eff_regret"],
            "power_top5_avg_overprovisioning": power_metrics["avg_overprov"],
            "power_top5_gpu_names": ", ".join(power_topk["name"].tolist()),
            "ltr_utility_top5_avg_tdp": ltr_metrics["avg_tdp"],
            "ltr_utility_top5_avg_psu": ltr_metrics["avg_psu"],
            "ltr_utility_top5_avg_ppw": ltr_metrics["avg_ppw"],
            "ltr_utility_top5_avg_efficiency_regret": ltr_metrics["avg_eff_regret"],
            "ltr_utility_top5_avg_overprovisioning": ltr_metrics["avg_overprov"],
            "ltr_utility_top5_gpu_names": ", ".join(ltr_topk["name"].tolist()),
            "power_top5_ndcg@5": _compute_ndcg_at_k(power_labels, k=self.config.k_top),
            "power_top5_recall_label3@5": _compute_recall_label3_at_k(power_labels, k=self.config.k_top),
            "ltr_utility_top5_ndcg@5": _compute_ndcg_at_k(ltr_labels, k=self.config.k_top),
            "ltr_utility_top5_recall_label3@5": _compute_recall_label3_at_k(ltr_labels, k=self.config.k_top),
            "power_top5_set": set(power_topk[GPU_ID_COL].tolist()),
            "ltr_utility_top5_set": set(ltr_topk[GPU_ID_COL].tolist()),
        }

    def _plot_metric_heatmap(self, summary_df: pd.DataFrame, out_path: Path) -> None:
        metrics = [
            "avg_tdp",
            "avg_psu",
            "avg_ppw",
            "avg_efficiency_regret",
            "unique_gpus",
            "top1_share",
            "ndcg@5",
            "recall_label3@5",
        ]
        higher_better = {"avg_ppw", "unique_gpus", "ndcg@5", "recall_label3@5"}

        data = summary_df.set_index("method")[metrics].copy()
        norm = data.copy()
        for col in metrics:
            v_min, v_max = data[col].min(), data[col].max()
            if v_max == v_min:
                norm[col] = 1.0
                continue
            scaled = (data[col] - v_min) / (v_max - v_min)
            norm[col] = scaled if col in higher_better else 1.0 - scaled

        plt.figure(figsize=(9, 4))
        plt.imshow(norm.values, aspect="auto", cmap="viridis")
        plt.xticks(range(len(metrics)), metrics, rotation=30, ha="right")
        plt.yticks(range(len(norm.index)), norm.index)
        plt.colorbar(label="Normalized score (higher is better)")
        plt.title(f"{self.config.scoring_mode} metric heatmap")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()

    def run(self) -> Dict[str, Path]:
        _log(
            f"requirements_mode={self.config.requirements_mode} scoring_mode={self.config.scoring_mode} "
            f"games={len(self.games_df)} candidate_gpus={len(self.gpu_df)} "
            f"train_games={len(self.train_games)} test_games={len(self.test_games)} "
            f"k_top={self.config.k_top} knn_k={self.config.knn_k} soft_threshold={self.config.soft_threshold:.2f}"
        )

        X_train, y_train, group_train, label_lookup_train, label_dist_train = self._build_pair_dataset(
            self.games_df[self.games_df["name"].isin(self.train_games)],
            split="train",
        )
        X_test, y_test, group_test, label_lookup_test, label_dist_test = self._build_pair_dataset(
            self.games_df[self.games_df["name"].isin(self.test_games)],
            split="test",
        )

        _log(
            f"built pair datasets for {self.config.scoring_mode}: "
            f"X_train={X_train.shape} X_test={X_test.shape} groups_train={len(group_train)} groups_test={len(group_test)}"
        )

        model, model_metrics, test_preds = _train_ranker(X_train, y_train, group_train, X_test)

        model_metrics = dict(model_metrics)
        model_metrics["ndcg@5"] = float("nan")
        model_metrics["recall_label3@5"] = float("nan")

        out_dir = self.output_dir / self.config.scoring_mode
        out_dir.mkdir(parents=True, exist_ok=True)

        label_dist = pd.concat([label_dist_train, label_dist_test], ignore_index=True)
        label_dist.to_csv(out_dir / "label_distribution.csv", index=False)

        per_game_rows: List[Dict[str, object]] = []
        for _, game_row in self.games_df.iterrows():
            if game_row["name"] not in self.test_games:
                continue
            row = self._evaluate_game(game_row, model, label_lookup_test)
            if row:
                per_game_rows.append(row)

        if not per_game_rows:
            raise RuntimeError(f"No evaluation rows produced for scoring mode {self.config.scoring_mode}.")

        per_game_df = pd.DataFrame(per_game_rows)
        per_game_df["power_top5_overlap_ltr"] = per_game_df.apply(
            lambda row: len(row["power_top5_set"] & row["ltr_utility_top5_set"]) / max(self.config.k_top, 1),
            axis=1,
        )

        method_rows = []
        for method, prefix in [("Power_Top5", "power_top5"), ("LTR_Utility_Top5", "ltr_utility_top5")]:
            method_rows.append({
                "method": method,
                "avg_tdp": per_game_df[f"{prefix}_avg_tdp"].mean(),
                "avg_psu": per_game_df[f"{prefix}_avg_psu"].mean(),
                "avg_ppw": per_game_df[f"{prefix}_avg_ppw"].mean(),
                "avg_efficiency_regret": per_game_df[f"{prefix}_avg_efficiency_regret"].mean(),
                "avg_overprovisioning": per_game_df[f"{prefix}_avg_overprovisioning"].mean(),
                "unique_gpus": per_game_df[f"{prefix}_gpu_names"].str.split(", ").explode().nunique(),
                "top1_share": _top1_share(per_game_df[f"{prefix}_gpu_names"]),
                "ndcg@5": per_game_df[f"{prefix}_ndcg@5"].mean(),
                "recall_label3@5": per_game_df[f"{prefix}_recall_label3@5"].mean(),
            })

        summary_df = pd.DataFrame(method_rows)
        comparison_df = summary_df.copy()

        per_game_output = out_dir / "per_game_top5.csv"
        summary_output = out_dir / "aggregate_summary.csv"
        comparison_output = self.output_dir / f"method_comparison_{self.config.scoring_mode}.csv"
        metrics_output = out_dir / "ltr_model_metrics.csv"
        feature_output = out_dir / "ltr_feature_importance.csv"
        heatmap_output = out_dir / "metric_heatmap.png"

        per_game_df.drop(columns=["power_top5_set", "ltr_utility_top5_set"], inplace=True)
        per_game_df.to_csv(per_game_output, index=False)
        summary_df.to_csv(summary_output, index=False)
        comparison_df.to_csv(comparison_output, index=False)

        if hasattr(model, "feature_importances_"):
            feature_rows = [
                {"feature": feature, "importance": float(score)}
                for feature, score in zip(X_train.columns, model.feature_importances_)
            ]
        elif hasattr(model, "get_booster"):
            booster = model.get_booster()
            feature_rows = [
                {"feature": feature, "importance": float(score)}
                for feature, score in booster.get_score(importance_type="gain").items()
            ]
        else:
            feature_rows = []

        if feature_rows:
            pd.DataFrame(feature_rows).sort_values("importance", ascending=False).to_csv(feature_output, index=False)

        model_metrics.update({
            "model_role": "ltr_ranker",
            "ndcg@5": float(per_game_df["ltr_utility_top5_ndcg@5"].mean()),
            "recall_label3@5": float(per_game_df["ltr_utility_top5_recall_label3@5"].mean()),
        })
        pd.DataFrame([model_metrics]).to_csv(metrics_output, index=False)

        self._plot_metric_heatmap(summary_df, heatmap_output)

        label_counts = label_dist[["label_0", "label_1", "label_2", "label_3"]].sum().to_dict()
        _log(f"label distribution for {self.config.scoring_mode}: {label_counts}")
        _log(f"wrote outputs to {out_dir}")
        _log(f"comparison file: {comparison_output}")

        return {
            "out_dir": out_dir,
            "per_game": per_game_output,
            "summary": summary_output,
            "comparison": comparison_output,
            "metrics": metrics_output,
            "feature_importance": feature_output,
            "heatmap": heatmap_output,
            "label_distribution": out_dir / "label_distribution.csv",
        }


def _build_full_labels(
    experiment: RecommendationExperiment,
    game_row: pd.Series,
) -> Tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    feasible, affinity_distance = experiment._cached_feasible(game_row["name"])
    if feasible.empty:
        return pd.Series(dtype=int), feasible, affinity_distance, pd.DataFrame()

    scored = _compute_base_scores(game_row, feasible, affinity_distance)
    best_ppw = feasible[PPW_COL].max()
    eff_regret = (best_ppw - feasible[PPW_COL]).fillna(best_ppw)
    dist_vals = affinity_distance.fillna(affinity_distance.median())
    objectives = np.column_stack([
        -feasible[PPW_COL].values,
        feasible[TDP_USED_COL].values,
        feasible[PSU_USED_COL].values,
        eff_regret.values,
        dist_vals.values,
    ])
    pareto_mask = _compute_pareto_front(objectives)
    labels = _assign_relevance_labels(scored["base_score"], pareto_mask)
    return labels, feasible, affinity_distance, scored


def run_scoring_diagnostic(output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    static_exp = RecommendationExperiment(
        ExperimentConfig(requirements_mode="recom", scoring_mode="static", output_dir=output_dir)
    )
    g3d_exp = RecommendationExperiment(
        ExperimentConfig(requirements_mode="recom", scoring_mode="g3d", output_dir=output_dir)
    )

    static_scores = static_exp.gpu_df["perf_score"]
    g3d_scores = g3d_exp.gpu_df["perf_score"]
    g3d_pred = g3d_exp.gpu_df["pred_g3d"] if "pred_g3d" in g3d_exp.gpu_df.columns else pd.Series(dtype=float)

    static_ppw = static_exp.gpu_df[PPW_COL]
    g3d_ppw = g3d_exp.gpu_df[PPW_COL]

    if not np.allclose(g3d_exp.gpu_df["perf_score"], g3d_exp.gpu_df["perf_score_mode"], equal_nan=True):
        _log("warning: g3d perf_score does not exactly match perf_score_mode")

    if np.allclose(g3d_exp.gpu_df["perf_score"], static_exp.gpu_df["perf_score"], equal_nan=True):
        _log("warning: g3d perf_score matches static perf_score row-for-row")

    test_games = sorted(static_exp.test_games)
    overlap_rows = []
    label_diff_rows = []
    label_rows = []
    examples = []

    total_labels = 0
    differing_labels = 0
    games_with_any_label_diff = 0

    for idx, game_name in enumerate(test_games):
        game_row = static_exp.games_df[static_exp.games_df["name"] == game_name].iloc[0]

        static_labels, static_feasible, _, _ = _build_full_labels(static_exp, game_row)
        g3d_labels, g3d_feasible, _, _ = _build_full_labels(g3d_exp, game_row)

        static_topk = static_feasible.sort_values(PPW_COL, ascending=False).head(static_exp.config.k_top)
        g3d_topk = g3d_feasible.sort_values(PPW_COL, ascending=False).head(g3d_exp.config.k_top)

        static_top5_names = static_topk["name"].tolist()
        g3d_top5_names = g3d_topk["name"].tolist()
        overlap_count = len(set(static_top5_names) & set(g3d_top5_names))
        top1_same = bool(static_top5_names[:1] and g3d_top5_names[:1] and static_top5_names[0] == g3d_top5_names[0])

        overlap_rows.append({
            "game_name": game_name,
            "static_top5": ", ".join(static_top5_names),
            "g3d_top5": ", ".join(g3d_top5_names),
            "overlap_count": overlap_count,
            "top1_same": top1_same,
        })

        if idx < 5:
            examples.append({
                "game_name": game_name,
                "static_top5": static_top5_names,
                "g3d_top5": g3d_top5_names,
                "overlap_count": overlap_count,
            })

        static_label_map = {gid: int(lbl) for gid, lbl in zip(static_feasible[GPU_ID_COL], static_labels.values)}
        g3d_label_map = {gid: int(lbl) for gid, lbl in zip(g3d_feasible[GPU_ID_COL], g3d_labels.values)}
        common_ids = [gid for gid in static_feasible[GPU_ID_COL].tolist() if gid in g3d_label_map]
        for gid in common_ids:
            total_labels += 1
            s_lbl = static_label_map.get(gid)
            g_lbl = g3d_label_map.get(gid)
            if s_lbl != g_lbl:
                differing_labels += 1

        if any(static_label_map.get(gid) != g3d_label_map.get(gid) for gid in common_ids):
            games_with_any_label_diff += 1

        label_rows.append({
            "game_name": game_name,
            "static_label_0": int((static_labels == 0).sum()),
            "static_label_1": int((static_labels == 1).sum()),
            "static_label_2": int((static_labels == 2).sum()),
            "static_label_3": int((static_labels == 3).sum()),
            "g3d_label_0": int((g3d_labels == 0).sum()),
            "g3d_label_1": int((g3d_labels == 1).sum()),
            "g3d_label_2": int((g3d_labels == 2).sum()),
            "g3d_label_3": int((g3d_labels == 3).sum()),
        })

    overlap_df = pd.DataFrame(overlap_rows)
    overlap_csv = output_dir / "scoring_diagnostic_overlap.csv"
    overlap_df.to_csv(overlap_csv, index=False)

    static_perf_stats = _series_stats(static_scores)
    g3d_perf_stats = _series_stats(g3d_scores)
    g3d_pred_stats = _series_stats(g3d_pred)
    static_ppw_stats = _series_stats(static_ppw)
    g3d_ppw_stats = _series_stats(g3d_ppw)

    static_vs_g3d_perf_corr = _pearson_corr(static_scores, g3d_scores)
    static_vs_g3d_ppw_corr = _pearson_corr(static_ppw, g3d_ppw)
    perf_diff_pct = float((static_scores.reset_index(drop=True) != g3d_scores.reset_index(drop=True)).mean() * 100.0)
    ppw_diff_pct = float((static_ppw.reset_index(drop=True) != g3d_ppw.reset_index(drop=True)).mean() * 100.0)

    identical_top5_pct = float((overlap_df["overlap_count"] == static_exp.config.k_top).mean() * 100.0)
    identical_top1_pct = float(overlap_df["top1_same"].mean() * 100.0)
    avg_overlap = float(overlap_df["overlap_count"].mean())

    static_label_counts = {
        "label_0": int(sum(row["static_label_0"] for row in label_rows)),
        "label_1": int(sum(row["static_label_1"] for row in label_rows)),
        "label_2": int(sum(row["static_label_2"] for row in label_rows)),
        "label_3": int(sum(row["static_label_3"] for row in label_rows)),
    }
    g3d_label_counts = {
        "label_0": int(sum(row["g3d_label_0"] for row in label_rows)),
        "label_1": int(sum(row["g3d_label_1"] for row in label_rows)),
        "label_2": int(sum(row["g3d_label_2"] for row in label_rows)),
        "label_3": int(sum(row["g3d_label_3"] for row in label_rows)),
    }

    label_diff_pct = float((differing_labels / max(total_labels, 1)) * 100.0)
    games_with_label_diff_pct = float((games_with_any_label_diff / max(len(test_games), 1)) * 100.0)

    lines = [
        "Scoring diagnostic summary",
        f"Games evaluated: {len(test_games)}",
        f"GPU rows: {len(static_exp.gpu_df)}",
        "",
        "Score-level diagnostics",
        f"Static perf_score: mean={static_perf_stats['mean']:.6f} std={static_perf_stats['std']:.6f} min={static_perf_stats['min']:.6f} max={static_perf_stats['max']:.6f}",
        f"G3D perf_score: mean={g3d_perf_stats['mean']:.6f} std={g3d_perf_stats['std']:.6f} min={g3d_perf_stats['min']:.6f} max={g3d_perf_stats['max']:.6f}",
        f"G3D pred_g3d: mean={g3d_pred_stats['mean']:.6f} std={g3d_pred_stats['std']:.6f} min={g3d_pred_stats['min']:.6f} max={g3d_pred_stats['max']:.6f}",
        f"Static vs G3D perf_score diff pct: {perf_diff_pct:.2f}%",
        f"Static vs G3D perf_score Pearson r: {static_vs_g3d_perf_corr:.6f}",
        f"Static PPW: mean={static_ppw_stats['mean']:.6f} std={static_ppw_stats['std']:.6f} min={static_ppw_stats['min']:.6f} max={static_ppw_stats['max']:.6f}",
        f"G3D PPW: mean={g3d_ppw_stats['mean']:.6f} std={g3d_ppw_stats['std']:.6f} min={g3d_ppw_stats['min']:.6f} max={g3d_ppw_stats['max']:.6f}",
        f"Static vs G3D PPW Pearson r: {static_vs_g3d_ppw_corr:.6f}",
        "",
        "Power-Top5 overlap diagnostics",
        f"Average overlap@5: {avg_overlap:.6f}",
        f"Identical top-5 sets: {identical_top5_pct:.2f}%",
        f"Identical top-1 recommendations: {identical_top1_pct:.2f}%",
        "Example games:",
    ]

    for example in examples:
        lines.extend([
            f"- {example['game_name']}",
            f"  static: {example['static_top5']}",
            f"  g3d: {example['g3d_top5']}",
            f"  overlap_count: {example['overlap_count']}",
        ])

    lines.extend([
        "",
        "Label diagnostics",
        f"Static label counts: {static_label_counts}",
        f"G3D label counts: {g3d_label_counts}",
        f"Label difference pct: {label_diff_pct:.2f}%",
        f"Games with any label difference pct: {games_with_label_diff_pct:.2f}%",
        "",
        "Checks",
        f"G3D perf_score is copied from perf_score_mode before PPW/ranking: {'yes' if np.allclose(g3d_exp.gpu_df['perf_score'], g3d_exp.gpu_df['perf_score_mode'], equal_nan=True) else 'no'}",
        "Power-Top5 uses mode-specific PPW: yes",
        "Labels use mode-specific utility values: yes",
    ])

    summary_path = output_dir / "scoring_diagnostic_summary.txt"
    summary_path.write_text("\n".join(lines).rstrip() + "\n")

    _log(f"diagnostic summary written: {summary_path}")
    _log(f"diagnostic overlap csv written: {overlap_csv}")

    return {
        "summary": summary_path,
        "overlap": overlap_csv,
    }


def _write_run_summary(root_dir: Path, mode_outputs: Dict[str, Dict[str, Path]], config: ExperimentConfig) -> Path:
    lines = [
        "Recommendation experiment summary",
        f"Command: python -m src.run_recommendation_experiment --output-dir {config.output_dir}",
        f"Scoring modes: static, g3d",
        f"Requirements mode: {config.requirements_mode}",
        f"Games split: 80/20 by game name, random_seed={config.random_seed}",
        f"Top-k: {config.k_top}",
        f"Soft threshold: {config.soft_threshold:.2f}",
        "",
    ]

    for mode, outputs in mode_outputs.items():
        lines.extend([
            f"[{mode}]",
            f"  per_game_top5: {outputs['per_game']}",
            f"  aggregate_summary: {outputs['summary']}",
            f"  method_comparison: {outputs['comparison']}",
            f"  ltr_model_metrics: {outputs['metrics']}",
            f"  ltr_feature_importance: {outputs.get('feature_importance', Path(''))}",
            f"  heatmap: {outputs['heatmap']}",
            f"  label_distribution: {outputs['label_distribution']}",
            "",
        ])

    path = root_dir / "run_summary.txt"
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def run_experiment(output_dir: Path) -> Dict[str, Dict[str, Path]]:
    config = ExperimentConfig(output_dir=output_dir)
    _log(f"starting unified experiment in {output_dir}")

    outputs: Dict[str, Dict[str, Path]] = {}
    for mode in ["static", "g3d"]:
        mode_config = ExperimentConfig(
            requirements_mode=config.requirements_mode,
            scoring_mode=mode,
            k_top=config.k_top,
            knn_k=config.knn_k,
            soft_threshold=config.soft_threshold,
            train_split=config.train_split,
            random_seed=config.random_seed,
            output_dir=output_dir,
        )
        experiment = RecommendationExperiment(mode_config)
        outputs[mode] = experiment.run()

    run_summary_path = _write_run_summary(output_dir, outputs, config)
    _log(f"wrote run summary: {run_summary_path}")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified energy-aware GPU recommendation experiment")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Root output directory for both scoring modes",
    )
    parser.add_argument(
        "--diagnose-scoring",
        action="store_true",
        help="Run scoring diagnostics only and skip the full experiment",
    )
    args = parser.parse_args()
    if args.diagnose_scoring:
        run_scoring_diagnostic(Path(args.output_dir))
        return
    run_experiment(Path(args.output_dir))


if __name__ == "__main__":
    main()
