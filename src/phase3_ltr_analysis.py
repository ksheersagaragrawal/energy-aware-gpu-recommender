"""Phase 3: Pareto-aware learning-to-rank GPU recommender.

Notes:
- LTR_Utility_Top5 is trained using proxy relevance labels (not real user-choice labels).
- Each game is a query and feasible GPUs are candidate items.
- Labels are derived from power-efficiency-first Pareto/utility criteria.
- NDCG@5 and Recall@5 evaluate whether high-relevance proxy candidates appear near the top.
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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

try:
    from src.recommender import KNN_FEATURE_MAP, EPSILON
    from src.phase1_topk_knn_analysis import Phase1Config, TopKRecommendationAnalyzer, _attach_power_predictions
    from src.phase2_ml_utility_analysis import _build_pair_features, _train_model, _utility_formula_scores
except ImportError:  # pragma: no cover - direct script execution fallback
    from recommender import KNN_FEATURE_MAP, EPSILON
    from phase1_topk_knn_analysis import Phase1Config, TopKRecommendationAnalyzer, _attach_power_predictions
    from phase2_ml_utility_analysis import _build_pair_features, _train_model, _utility_formula_scores


GPU_ID_COL = "gpu_id"
TDP_USED_COL = "tdp_w_used"
PSU_USED_COL = "psu_w_used"
PPW_COL = "perf_per_watt"


@dataclass(frozen=True)
class Phase3Config:
    mode: str = "recom"
    k_top: int = 5
    knn_k: int = 50
    soft_threshold: float = 0.80
    output_dir: str = "phase3_outputs"
    random_seed: int = 42
    train_split: float = 0.8


class Phase3LTRAnalyzer:
    def __init__(self, games_df: pd.DataFrame, gpu_df: pd.DataFrame, config: Phase3Config):
        self.games_df = games_df.copy()
        self.gpu_df = gpu_df.copy()
        self.config = config
        self.analyzer = TopKRecommendationAnalyzer(
            self.games_df,
            self.gpu_df,
            Phase1Config(
                mode=config.mode,
                k_top=config.k_top,
                knn_k=config.knn_k,
                soft_threshold=config.soft_threshold,
            ),
        )

    def _compute_feature_affinity_distance(self, game_row: pd.Series, gpu_pool: pd.DataFrame) -> pd.Series:
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

    def _minmax_score(self, values: pd.Series) -> pd.Series:
        v_min, v_max = values.min(), values.max()
        if pd.isna(v_min) or pd.isna(v_max) or v_max == v_min:
            return pd.Series(0.5, index=values.index)
        score = (values - v_min) / (v_max - v_min)
        return score.clip(0.0, 1.0)

    def _low_is_good(self, values: pd.Series) -> pd.Series:
        return (1.0 - self._minmax_score(values)).clip(0.0, 1.0)

    def _compute_margin(self, gpu_perf: pd.Series, game_perf: float) -> pd.Series:
        if pd.isna(game_perf) or game_perf == 0:
            return pd.Series(np.nan, index=gpu_perf.index)
        return (gpu_perf - game_perf) / game_perf

    def compute_base_scores(
        self,
        game_row: pd.Series,
        feasible: pd.DataFrame,
        affinity_distance: pd.Series,
    ) -> pd.DataFrame:
        df = feasible.copy()

        df["ppw_score"] = self._minmax_score(df[PPW_COL])
        df["low_tdp_score"] = self._low_is_good(df[TDP_USED_COL])
        df["low_psu_score"] = self._low_is_good(df[PSU_USED_COL])

        affinity = affinity_distance.copy()
        if affinity.isna().all():
            df["feature_affinity_score"] = 0.5
        else:
            df["feature_affinity_score"] = self._low_is_good(affinity.fillna(affinity.median()))

        game_perf = game_row.get("perf_score")
        margin = self._compute_margin(df["perf_score"], game_perf)
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

    def compute_pareto_front(self, objectives: np.ndarray, max_size: int = 2000) -> Optional[np.ndarray]:
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

    def assign_relevance_labels(
        self,
        base_scores: pd.Series,
        pareto_mask: Optional[np.ndarray],
    ) -> pd.Series:
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

    def build_pair_dataset(
        self,
        game_subset: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series, List[int], Dict[str, Dict[str, int]], pd.DataFrame]:
        feature_rows = []
        label_rows = []
        group_sizes = []
        label_dist_rows = []
        label_lookup: Dict[str, Dict[str, int]] = {}

        for _, game_row in game_subset.iterrows():
            feasible = self.analyzer.get_feasible_gpus(game_row)
            if feasible.empty:
                continue

            affinity_distance = self._compute_feature_affinity_distance(game_row, feasible)
            scored = self.compute_base_scores(game_row, feasible, affinity_distance)

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

            pareto_mask = self.compute_pareto_front(objectives)
            labels = self.assign_relevance_labels(scored["base_score"], pareto_mask)

            label_dist_rows.append({
                "game_name": game_row["name"],
                "label_0": int((labels == 0).sum()),
                "label_1": int((labels == 1).sum()),
                "label_2": int((labels == 2).sum()),
                "label_3": int((labels == 3).sum()),
            })

            label_lookup[game_row["name"]] = {
                gid: int(lbl) for gid, lbl in zip(feasible[GPU_ID_COL], labels.values)
            }

            pair_features = _build_pair_features(game_row, scored)
            pair_features["feature_affinity_distance"] = affinity_distance.values
            pair_features = pair_features.fillna(0.0)

            feature_rows.append(pair_features)
            label_rows.append(labels.values)
            group_sizes.append(len(labels))

        if not feature_rows:
            raise RuntimeError("No feasible game-GPU pairs found.")

        X = pd.concat(feature_rows, ignore_index=True)
        y = pd.Series(np.concatenate(label_rows))
        label_dist = pd.DataFrame(label_dist_rows)
        return X, y, group_sizes, label_lookup, label_dist

    def train_ranker(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        group_train: List[int],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        group_test: List[int],
    ) -> Tuple[object, Dict[str, object]]:
        model = None
        metrics = {}

        # XGBoost ranker with GPU fallback
        try:
            import xgboost as xgb  # type: ignore

            params = {
                "objective": "rank:ndcg",
                "eval_metric": "ndcg@5",
                "learning_rate": 0.05,
                "max_depth": 6,
                "n_estimators": 300,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": 42,
            }

            backend = "cpu"
            start = time.time()

            def _fit(local_params):
                nonlocal model, backend
                model = xgb.XGBRanker(**local_params)
                model.fit(X_train, y_train, group=group_train)
                backend = local_params.get("device", "cpu")

            try:
                _fit(params | {"tree_method": "hist", "device": "cuda"})
                backend = "cuda"
            except Exception:
                try:
                    _fit(params | {"tree_method": "gpu_hist"})
                    backend = "cuda"
                except Exception:
                    _fit(params | {"tree_method": "hist", "device": "cpu"})
                    backend = "cpu"

            train_time = time.time() - start

            start = time.time()
            preds = model.predict(X_test)
            infer_time = time.time() - start

            metrics.update({
                "model": "XGBRanker",
                "backend": backend,
                "train_time_sec": train_time,
                "inference_time_sec": infer_time,
            })
            print(
                f"[phase3] model={metrics['model']} backend={metrics['backend']} "
                f"train_time={metrics['train_time_sec']:.2f}s "
                f"infer_time={metrics['inference_time_sec']:.2f}s"
            )
            return model, metrics

        except Exception:
            model = None

        # LightGBM ranker fallback
        try:
            import lightgbm as lgb  # type: ignore

            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "learning_rate": 0.05,
                "n_estimators": 300,
                "random_state": 42,
            }

            backend = "cpu"
            start = time.time()
            try:
                model = lgb.LGBMRanker(**params, device_type="gpu")
                model.fit(X_train, y_train, group=group_train)
                backend = "gpu"
            except Exception:
                model = lgb.LGBMRanker(**params, device_type="cpu")
                model.fit(X_train, y_train, group=group_train)

            train_time = time.time() - start

            start = time.time()
            preds = model.predict(X_test)
            infer_time = time.time() - start

            metrics.update({
                "model": "LGBMRanker",
                "backend": backend,
                "train_time_sec": train_time,
                "inference_time_sec": infer_time,
            })
            print(
                f"[phase3] model={metrics['model']} backend={metrics['backend']} "
                f"train_time={metrics['train_time_sec']:.2f}s "
                f"infer_time={metrics['inference_time_sec']:.2f}s"
            )
            return model, metrics

        except Exception:
            model = None

        # Pointwise fallback
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
            "warning": "pointwise_fallback",
        })
        print(
            f"[phase3] model={metrics['model']} backend={metrics['backend']} "
            f"train_time={metrics['train_time_sec']:.2f}s "
            f"infer_time={metrics['inference_time_sec']:.2f}s"
        )
        return model, metrics

    def train_ml_utility_regressor(
        self,
        train_games: set,
        test_games: set,
    ) -> Tuple[object, Dict[str, object]]:
        train_rows = []
        train_targets = []
        test_rows = []
        test_targets = []

        for _, game_row in self.games_df.iterrows():
            if game_row["name"] not in train_games:
                continue
            feasible = self.analyzer.get_feasible_gpus(game_row)
            if feasible.empty:
                continue
            scored = _utility_formula_scores(game_row, feasible)
            features = _build_pair_features(game_row, scored)
            affinity_distance = self._compute_feature_affinity_distance(game_row, feasible)
            features["feature_affinity_distance"] = affinity_distance.values
            train_rows.append(features)
            train_targets.append(scored["utility_formula_score"].values)

        for _, game_row in self.games_df.iterrows():
            if game_row["name"] not in test_games:
                continue
            feasible = self.analyzer.get_feasible_gpus(game_row)
            if feasible.empty:
                continue
            scored = _utility_formula_scores(game_row, feasible)
            features = _build_pair_features(game_row, scored)
            affinity_distance = self._compute_feature_affinity_distance(game_row, feasible)
            features["feature_affinity_distance"] = affinity_distance.values
            test_rows.append(features)
            test_targets.append(scored["utility_formula_score"].values)

        X_train = pd.concat(train_rows, ignore_index=True).fillna(0.0)
        y_train = pd.Series(np.concatenate(train_targets))
        X_test = pd.concat(test_rows, ignore_index=True).fillna(0.0)
        y_test = pd.Series(np.concatenate(test_targets))

        model, metrics = _train_model(X_train, y_train, X_test, y_test)
        return model, metrics

    def recommend_ltr_topk(self, game_row: pd.Series, model: object) -> pd.DataFrame:
        feasible = self.analyzer.get_feasible_gpus(game_row)
        if feasible.empty:
            return feasible

        affinity_distance = self._compute_feature_affinity_distance(game_row, feasible)
        features = _build_pair_features(game_row, feasible)
        features["feature_affinity_distance"] = affinity_distance.values
        features = features.fillna(0.0)

        scores = model.predict(features)
        ranked = feasible.copy()
        ranked["ltr_score"] = scores
        return ranked.sort_values("ltr_score", ascending=False).head(self.config.k_top)

    def compute_ndcg_at_k(self, labels: List[int], k: int = 5) -> float:
        labels_k = labels[:k]
        if not labels_k:
            return np.nan
        gains = np.array([2 ** rel - 1 for rel in labels_k], dtype=float)
        discounts = np.log2(np.arange(2, len(labels_k) + 2))
        dcg = (gains / discounts).sum()
        ideal = sorted(labels, reverse=True)[:k]
        ideal_gains = np.array([2 ** rel - 1 for rel in ideal], dtype=float)
        ideal_dcg = (ideal_gains / discounts).sum()
        return float(dcg / ideal_dcg) if ideal_dcg > 0 else 0.0

    def compute_recall_label3_at_k(self, labels: List[int], k: int = 5) -> float:
        total = sum(1 for l in labels if l == 3)
        if total == 0:
            return np.nan
        hit = sum(1 for l in labels[:k] if l == 3)
        return hit / total

    def build_aggregate_summary(self, per_game_df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
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
                "top1_share": self._top1_share(per_game_df[f"{prefix}_gpu_names"]),
                "ndcg@5": per_game_df[f"{prefix}_ndcg@5"].mean(),
                "recall_label3@5": per_game_df[f"{prefix}_recall_label3@5"].mean(),
                "overlap@5_with_power": per_game_df.get(f"{prefix}_overlap_power", pd.Series(dtype=float)).mean(),
                "overlap@5_with_utility": per_game_df.get(f"{prefix}_overlap_utility", pd.Series(dtype=float)).mean(),
                "overlap@5_with_ml": per_game_df.get(f"{prefix}_overlap_ml", pd.Series(dtype=float)).mean(),
            })
        return pd.DataFrame(rows)

    def _top1_share(self, names_series: pd.Series) -> float:
        top1 = names_series.dropna().apply(lambda v: v.split(", ")[0] if v else "")
        if top1.empty:
            return np.nan
        return top1.value_counts(normalize=True).iloc[0]

    def plot_metric_heatmap(self, summary_df: pd.DataFrame, out_dir: Path) -> None:
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
            if col in higher_better:
                norm[col] = scaled
            else:
                norm[col] = 1.0 - scaled

        plt.figure(figsize=(9, 4))
        plt.imshow(norm.values, aspect="auto", cmap="viridis")
        plt.xticks(range(len(metrics)), metrics, rotation=30, ha="right")
        plt.yticks(range(len(norm.index)), norm.index)
        plt.colorbar(label="Normalized score (higher is better)")
        plt.title("Phase 3 metric heatmap (normalized)")
        plt.tight_layout()
        plt.savefig(out_dir / "phase3_metric_heatmap.png", dpi=300)
        plt.close()

    def plot_efficiency_tradeoff(self, summary_df: pd.DataFrame, out_dir: Path) -> None:
        plt.figure(figsize=(7, 5))
        for _, row in summary_df.iterrows():
            plt.scatter(row["avg_ppw"], row["avg_efficiency_regret"], s=70)
            plt.text(row["avg_ppw"], row["avg_efficiency_regret"], row["method"], fontsize=9)
        plt.xlabel("avg_ppw")
        plt.ylabel("avg_efficiency_regret")
        plt.title("Phase 3 efficiency trade-off")
        plt.tight_layout()
        plt.savefig(out_dir / "phase3_efficiency_tradeoff.png", dpi=300)
        plt.close()

    def plot_diversity_efficiency_tradeoff(self, summary_df: pd.DataFrame, out_dir: Path) -> None:
        plt.figure(figsize=(7, 5))
        for _, row in summary_df.iterrows():
            x_val = 1.0 - row["top1_share"] if pd.notna(row["top1_share"]) else row["unique_gpus"]
            plt.scatter(x_val, row["avg_ppw"], s=70)
            plt.text(x_val, row["avg_ppw"], row["method"], fontsize=9)
        plt.xlabel("1 - top1_share (or unique_gpus)")
        plt.ylabel("avg_ppw")
        plt.title("Phase 3 diversity-efficiency trade-off")
        plt.tight_layout()
        plt.savefig(out_dir / "phase3_diversity_efficiency_tradeoff.png", dpi=300)
        plt.close()

    def plot_ndcg_recall_bar(self, summary_df: pd.DataFrame, out_dir: Path) -> None:
        plt.figure(figsize=(8, 4))
        x = np.arange(len(summary_df))
        width = 0.35
        plt.bar(x - width / 2, summary_df["ndcg@5"], width=width, label="NDCG@5")
        plt.bar(x + width / 2, summary_df["recall_label3@5"], width=width, label="Recall@5 (label=3)")
        plt.xticks(x, summary_df["method"], rotation=30, ha="right")
        plt.ylabel("Score")
        plt.title("Phase 3 ranking metrics")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "phase3_ndcg_recall_bar.png", dpi=300)
        plt.close()

    def plot_method_overlap_heatmap(self, overlap_matrix: pd.DataFrame, out_dir: Path) -> None:
        plt.figure(figsize=(6, 5))
        plt.imshow(overlap_matrix.values, cmap="Blues", vmin=0.0, vmax=1.0)
        plt.xticks(range(len(overlap_matrix.columns)), overlap_matrix.columns, rotation=30, ha="right")
        plt.yticks(range(len(overlap_matrix.index)), overlap_matrix.index)
        plt.colorbar(label="Overlap@5")
        plt.title("Phase 3 method overlap heatmap")
        plt.tight_layout()
        plt.savefig(out_dir / "phase3_method_overlap_heatmap.png", dpi=300)
        plt.close()

    def plot_frequency_bar(self, per_game_df: pd.DataFrame, out_dir: Path) -> None:
        def _freq(names_series: pd.Series, label: str) -> pd.DataFrame:
            freq = names_series.dropna().str.split(", ").explode().value_counts().head(15)
            df = freq.reset_index()
            df.columns = ["gpu_name", "count"]
            df["method"] = label
            return df

        power_freq = _freq(per_game_df["power_top5_gpu_names"], "Power_Top5")
        ltr_freq = _freq(per_game_df["ltr_utility_top5_gpu_names"], "LTR_Utility_Top5")
        freq_df = pd.concat([power_freq, ltr_freq], ignore_index=True)

        plt.figure(figsize=(10, 5))
        methods = list(freq_df["method"].unique())
        gpu_names = sorted(set(freq_df["gpu_name"]))
        x = np.arange(len(gpu_names))
        width = 0.4

        for idx, method in enumerate(methods):
            sub = freq_df[freq_df["method"] == method].set_index("gpu_name").reindex(gpu_names).fillna(0)
            offset = (idx - (len(methods) - 1) / 2) * width
            plt.bar(x + offset, sub["count"].values, width=width, label=method, alpha=0.8)

        plt.xticks(x, gpu_names, rotation=45, ha="right")
        plt.ylabel("Count")
        plt.title("Top 15 GPU frequency: Power vs LTR")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "phase3_frequency_bar_ltr_vs_power.png", dpi=300)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: LTR utility top-k analysis")
    parser.add_argument("--mode", choices=["min", "recom"], default="recom")
    parser.add_argument("--k-top", type=int, default=5)
    parser.add_argument("--knn-k", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--output-dir", default="phase3_outputs")
    args = parser.parse_args()

    config = Phase3Config(
        mode=args.mode,
        k_top=args.k_top,
        knn_k=args.knn_k,
        soft_threshold=args.threshold,
        output_dir=args.output_dir,
    )

    games_df = pd.read_csv(f"data/vectors/game_vectors_{config.mode}.csv")
    gpu_df = pd.read_csv("data/vectors/gpu_power_vectors.csv")
    gpu_df = _attach_power_predictions(gpu_df)

    analyzer = Phase3LTRAnalyzer(games_df, gpu_df, config)

    train_games, test_games = train_test_split(
        games_df["name"].values,
        train_size=config.train_split,
        random_state=config.random_seed,
        shuffle=True,
    )
    train_games = set(train_games)
    test_games = set(test_games)

    X_train, y_train, group_train, label_lookup_train, label_dist_train = analyzer.build_pair_dataset(
        games_df[games_df["name"].isin(train_games)]
    )
    X_test, y_test, group_test, label_lookup_test, label_dist_test = analyzer.build_pair_dataset(
        games_df[games_df["name"].isin(test_games)]
    )

    model, model_metrics = analyzer.train_ranker(X_train, y_train, group_train, X_test, y_test, group_test)

    ml_utility_model, ml_utility_metrics = analyzer.train_ml_utility_regressor(train_games, test_games)

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_dist = pd.concat([label_dist_train, label_dist_test], ignore_index=True)
    label_dist.to_csv(out_dir / "phase3_relevance_label_distribution.csv", index=False)

    metrics_rows = [
        {**model_metrics, "model_role": "ltr_ranker"},
        {**ml_utility_metrics, "model_role": "ml_utility_regressor"},
    ]

    methods = [
        "KNN50_Feasible",
        "KNN50_Feasible_PPW_Top5",
        "Power_Top5",
        "UtilityFormula_Top5",
        "ML_Utility_Top5",
        "LTR_Utility_Top5",
    ]

    per_game_rows = []
    overlap_sets: Dict[str, List[set]] = {m: [] for m in methods}
    ndcg_rows: Dict[str, List[float]] = {m: [] for m in methods}
    recall_rows: Dict[str, List[float]] = {m: [] for m in methods}

    for _, game_row in games_df.iterrows():
        if game_row["name"] not in test_games:
            continue

        feasible = analyzer.analyzer.get_feasible_gpus(game_row)
        if feasible.empty:
            continue

        best_ppw = feasible[PPW_COL].max()

        power_topk = analyzer.analyzer.get_power_topk(game_row, k=config.k_top)
        knn_feasible = analyzer.analyzer.get_knn_candidates_feasible(game_row, n_neighbors=config.knn_k)
        knn_feasible_ppw = analyzer.analyzer.get_knn_feasible_ppw_topk(game_row, n_neighbors=config.knn_k, k=config.k_top)

        affinity_distance = analyzer._compute_feature_affinity_distance(game_row, feasible)
        utility_scored = analyzer.compute_base_scores(game_row, feasible, affinity_distance)
        utility_topk = utility_scored.sort_values("base_score", ascending=False).head(config.k_top)

        ml_scored = _utility_formula_scores(game_row, feasible)
        ml_features = _build_pair_features(game_row, ml_scored)
        ml_features["feature_affinity_distance"] = affinity_distance.values
        ml_features = ml_features.fillna(0.0)
        ml_preds = ml_utility_model.predict(ml_features)
        ml_scored = feasible.copy()
        ml_scored["ml_utility_score"] = ml_preds
        ml_topk = ml_scored.sort_values("ml_utility_score", ascending=False).head(config.k_top)

        ltr_topk = analyzer.recommend_ltr_topk(game_row, model)

        power_metrics = analyzer.analyzer.compute_candidate_metrics(power_topk, game_row, best_ppw)
        knn_metrics = analyzer.analyzer.compute_candidate_metrics(knn_feasible, game_row, best_ppw)
        knn_ppw_metrics = analyzer.analyzer.compute_candidate_metrics(knn_feasible_ppw, game_row, best_ppw)
        utility_metrics = analyzer.analyzer.compute_candidate_metrics(utility_topk, game_row, best_ppw)
        ml_metrics = analyzer.analyzer.compute_candidate_metrics(ml_topk, game_row, best_ppw)
        ltr_metrics = analyzer.analyzer.compute_candidate_metrics(ltr_topk, game_row, best_ppw)

        label_map = label_lookup_test.get(game_row["name"], {})

        def _labels_for(df: pd.DataFrame) -> List[int]:
            return [label_map.get(gid, 0) for gid in df[GPU_ID_COL].tolist()]

        for method_name, df in [
            ("KNN50_Feasible", knn_feasible),
            ("KNN50_Feasible_PPW_Top5", knn_feasible_ppw),
            ("Power_Top5", power_topk),
            ("UtilityFormula_Top5", utility_topk),
            ("ML_Utility_Top5", ml_topk),
            ("LTR_Utility_Top5", ltr_topk),
        ]:
            labels = _labels_for(df)
            ndcg_rows[method_name].append(analyzer.compute_ndcg_at_k(labels, k=config.k_top))
            recall_rows[method_name].append(analyzer.compute_recall_label3_at_k(labels, k=config.k_top))

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
            "ltr_utility_top5_avg_tdp": ltr_metrics["avg_tdp"],
            "ltr_utility_top5_avg_psu": ltr_metrics["avg_psu"],
            "ltr_utility_top5_avg_ppw": ltr_metrics["avg_ppw"],
            "ltr_utility_top5_avg_efficiency_regret": ltr_metrics["avg_eff_regret"],
            "ltr_utility_top5_avg_overprovisioning": ltr_metrics["avg_overprov"],
            "ltr_utility_top5_gpu_names": ", ".join(ltr_topk["name"].tolist()),
        })

        overlap_sets["KNN50_Feasible"].append(set(knn_feasible.head(config.k_top)[GPU_ID_COL].tolist()))
        overlap_sets["KNN50_Feasible_PPW_Top5"].append(set(knn_feasible_ppw[GPU_ID_COL].tolist()))
        overlap_sets["Power_Top5"].append(set(power_topk[GPU_ID_COL].tolist()))
        overlap_sets["UtilityFormula_Top5"].append(set(utility_topk[GPU_ID_COL].tolist()))
        overlap_sets["ML_Utility_Top5"].append(set(ml_topk[GPU_ID_COL].tolist()))
        overlap_sets["LTR_Utility_Top5"].append(set(ltr_topk[GPU_ID_COL].tolist()))

    per_game_df = pd.DataFrame(per_game_rows)

    for method in methods:
        per_game_df[f"{method.lower()}_ndcg@5"] = ndcg_rows[method]
        per_game_df[f"{method.lower()}_recall_label3@5"] = recall_rows[method]

    # Overlap@5 with key baselines
    power_sets = overlap_sets["Power_Top5"]
    utility_sets = overlap_sets["UtilityFormula_Top5"]
    ml_sets = overlap_sets["ML_Utility_Top5"]
    for method in methods:
        overlaps_power = []
        overlaps_utility = []
        overlaps_ml = []
        for s1, s_power, s_util, s_ml in zip(
            overlap_sets[method], power_sets, utility_sets, ml_sets
        ):
            overlaps_power.append(len(s1 & s_power) / max(config.k_top, 1))
            overlaps_utility.append(len(s1 & s_util) / max(config.k_top, 1))
            overlaps_ml.append(len(s1 & s_ml) / max(config.k_top, 1))
        per_game_df[f"{method.lower()}_overlap_power"] = overlaps_power
        per_game_df[f"{method.lower()}_overlap_utility"] = overlaps_utility
        per_game_df[f"{method.lower()}_overlap_ml"] = overlaps_ml

    per_game_df.to_csv(out_dir / "phase3_per_game_ltr_topk_analysis.csv", index=False)

    summary_df = analyzer.build_aggregate_summary(per_game_df, methods)
    summary_df.to_csv(out_dir / "phase3_aggregate_ltr_topk_summary.csv", index=False)

    # Update model metrics with ranking diagnostics
    for row in metrics_rows:
        if row.get("model_role") == "ltr_ranker":
            row["ndcg@5"] = float(np.nanmean(ndcg_rows["LTR_Utility_Top5"]))
            row["recall_label3@5"] = float(np.nanmean(recall_rows["LTR_Utility_Top5"]))

    pd.DataFrame(metrics_rows).to_csv(out_dir / "phase3_ltr_model_metrics.csv", index=False)

    # Overlap heatmap between key methods
    overlap_matrix = pd.DataFrame(index=[
        "Power_Top5",
        "UtilityFormula_Top5",
        "ML_Utility_Top5",
        "LTR_Utility_Top5",
        "KNN50_Feasible_PPW_Top5",
    ])
    for m1 in overlap_matrix.index:
        for m2 in overlap_matrix.index:
            overlaps = []
            for s1, s2 in zip(overlap_sets[m1], overlap_sets[m2]):
                overlaps.append(len(s1 & s2) / max(config.k_top, 1))
            overlap_matrix.loc[m1, m2] = float(np.nanmean(overlaps))

    analyzer.plot_metric_heatmap(summary_df, out_dir)
    analyzer.plot_efficiency_tradeoff(summary_df, out_dir)
    analyzer.plot_diversity_efficiency_tradeoff(summary_df, out_dir)
    analyzer.plot_ndcg_recall_bar(summary_df, out_dir)
    analyzer.plot_method_overlap_heatmap(overlap_matrix, out_dir)
    analyzer.plot_frequency_bar(per_game_df, out_dir)

    # Feature importance (if available)
    feature_importance_rows = []
    if hasattr(model, "feature_importances_"):
        for name, score in zip(X_train.columns, model.feature_importances_):
            feature_importance_rows.append({"feature": name, "importance": float(score)})
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
        scores = booster.get_score(importance_type="gain")
        for name, score in scores.items():
            feature_importance_rows.append({"feature": name, "importance": float(score)})

    if feature_importance_rows:
        pd.DataFrame(feature_importance_rows).sort_values("importance", ascending=False).to_csv(
            out_dir / "phase3_ltr_feature_importance.csv",
            index=False,
        )

    notes = (
        "# Phase 3 Interpretation Notes\n"
        "1. LTR_Utility_Top5 is trained using proxy relevance labels, not real user-choice labels.\n"
        "2. Relevance labels are derived from power-efficiency-first Pareto/utility criteria.\n"
        "3. Each game is treated as a query and feasible GPUs are candidate items.\n"
        "4. NDCG@5 and Recall@5 evaluate whether high-relevance proxy candidates appear near the top.\n"
        "5. If LTR_Utility_Top5 does not beat Power_Top5 on PPW, that is expected.\n"
        "6. The key question is whether LTR preserves efficiency while improving diversity or ranking quality.\n"
    )
    (out_dir / "phase3_method_notes.md").write_text(notes)


if __name__ == "__main__":
    main()
