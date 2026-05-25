"""Phase 1: Top-k Recommendation vs KNN Feature-Affinity Retrieval.

This module compares power-aware top-k recommendations against KNN-50 retrieval
for each game. KNN50 represents feature-affinity retrieval. Power_Top5 represents
power-aware recommendation. The overlap analysis checks alignment between the two.
KNN50_PPW_Top5 is a reranked KNN baseline using power-aware PPW.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from recommender import GAME_VECTORS, GPU_VECTORS, KNN_FEATURE_MAP, SOFT_FILTER_MAP, EPSILON


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


class TopKRecommendationAnalyzer:
    def __init__(self, games_df: pd.DataFrame, gpu_df: pd.DataFrame, config: Phase1Config):
        self.games_df = games_df.copy()
        self.gpu_df = gpu_df.copy()
        self.config = config
        self._log("initialized analyzer")

    def _log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[phase1 {timestamp}] {message}")

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

        ranked = feasible.sort_values(PPW_COL, ascending=False)
        return ranked.head(k).copy()

    def get_knn_candidates(self, game_row: pd.Series, n_neighbors: int = 50) -> pd.DataFrame:
        game_vec = []
        gpu_matrix = []

        for _, (game_col, gpu_col) in KNN_FEATURE_MAP.items():
            gpu_vals = self.gpu_df[gpu_col].values.astype(float)
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

        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(self.gpu_df)), metric="euclidean")
        nn.fit(gpu_matrix)
        distances, indices = nn.kneighbors(game_vec)

        result = self.gpu_df.iloc[indices[0]].copy()
        result["distance"] = distances[0]
        return result

    def get_knn_ppw_topk(self, game_row: pd.Series, n_neighbors: int = 50, k: int = 5) -> pd.DataFrame:
        knn = self.get_knn_candidates(game_row, n_neighbors=n_neighbors)
        if knn.empty:
            return knn
        return knn.sort_values(PPW_COL, ascending=False).head(k).copy()

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
            eff_regret_abs = pd.Series(np.nan, index=candidate_df.index)
            eff_regret_rel = pd.Series(np.nan, index=candidate_df.index)

        return {
            "avg_tdp": candidate_df[TDP_USED_COL].mean(),
            "avg_psu": candidate_df[PSU_USED_COL].mean(),
            "avg_ppw": candidate_df[PPW_COL].mean(),
            "avg_overprov": overprov_rel.mean(),
            "avg_eff_regret": eff_regret_rel.mean(),
        }

    def compute_overlap(self, topk_df: pd.DataFrame, knn_df: pd.DataFrame) -> Tuple[int, float]:
        if topk_df.empty or knn_df.empty:
            return 0, 0.0

        topk_ids = set(topk_df[GPU_ID_COL].tolist())
        knn_ids = set(knn_df[GPU_ID_COL].tolist())
        overlap = len(topk_ids & knn_ids)
        return overlap, overlap / max(self.config.k_top, 1)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        per_game_rows: List[Dict[str, object]] = []
        skipped_games: List[Dict[str, object]] = []
        total = len(self.games_df)
        start = time.time()
        self._log(f"processing {total} games (workers={self.config.num_workers})")

        def _process_game(game_row: pd.Series) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]]]:
            game_name = game_row.get("name")
            feasible = self.get_feasible_gpus(game_row)

            if feasible.empty:
                return None, {"game_name": game_name, "reason": "no_feasible_gpus"}

            best_ppw = feasible[PPW_COL].max()

            power_topk = self.get_power_topk(game_row, k=self.config.k_top)
            knn = self.get_knn_candidates(game_row, n_neighbors=self.config.knn_k)
            knn_ppw_topk = self.get_knn_ppw_topk(game_row, n_neighbors=self.config.knn_k, k=self.config.k_top)

            overlap_count, overlap_rate = self.compute_overlap(power_topk, knn)

            power_metrics = self.compute_candidate_metrics(power_topk, game_row, best_ppw)
            knn_metrics = self.compute_candidate_metrics(knn, game_row, best_ppw)
            knn_ppw_metrics = self.compute_candidate_metrics(knn_ppw_topk, game_row, best_ppw)

            return {
                "game_name": game_name,
                "actual_power_topk_count": len(power_topk),
                "actual_knn_count": len(knn),
                "overlap_count": overlap_count,
                "overlap_rate": overlap_rate,
                "power_top5_gpu_names": ", ".join(power_topk["name"].tolist()),
                "knn50_ppw_top5_gpu_names": ", ".join(knn_ppw_topk["name"].tolist()),
                "power_top5_avg_tdp": power_metrics["avg_tdp"],
                "power_top5_avg_psu": power_metrics["avg_psu"],
                "power_top5_avg_ppw": power_metrics["avg_ppw"],
                "power_top5_avg_overprovisioning": power_metrics["avg_overprov"],
                "power_top5_avg_efficiency_regret": power_metrics["avg_eff_regret"],
                "knn50_avg_tdp": knn_metrics["avg_tdp"],
                "knn50_avg_psu": knn_metrics["avg_psu"],
                "knn50_avg_ppw": knn_metrics["avg_ppw"],
                "knn50_avg_overprovisioning": knn_metrics["avg_overprov"],
                "knn50_avg_efficiency_regret": knn_metrics["avg_eff_regret"],
                "knn50_ppw_top5_avg_tdp": knn_ppw_metrics["avg_tdp"],
                "knn50_ppw_top5_avg_psu": knn_ppw_metrics["avg_psu"],
                "knn50_ppw_top5_avg_ppw": knn_ppw_metrics["avg_ppw"],
                "knn50_ppw_top5_avg_overprovisioning": knn_ppw_metrics["avg_overprov"],
                "knn50_ppw_top5_avg_efficiency_regret": knn_ppw_metrics["avg_eff_regret"],
            }, None

        if self.config.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {executor.submit(_process_game, row): idx for idx, row in self.games_df.iterrows()}
                for i, future in enumerate(as_completed(futures), start=1):
                    result, skipped = future.result()
                    if result is not None:
                        per_game_rows.append(result)
                    if skipped is not None:
                        skipped_games.append(skipped)
                    if i % self.config.log_every == 0 or i == total:
                        elapsed = time.time() - start
                        self._log(f"processed {i}/{total} games in {elapsed:.1f}s")
        else:
            for i, (_, game_row) in enumerate(self.games_df.iterrows(), start=1):
                result, skipped = _process_game(game_row)
                if result is not None:
                    per_game_rows.append(result)
                if skipped is not None:
                    skipped_games.append(skipped)
                if i % self.config.log_every == 0 or i == total:
                    elapsed = time.time() - start
                    self._log(f"processed {i}/{total} games in {elapsed:.1f}s")

        per_game_df = pd.DataFrame(per_game_rows)
        skipped_df = pd.DataFrame(skipped_games)
        summary_df = self.build_aggregate_summary(per_game_df)
        return per_game_df, summary_df, skipped_df

    def build_aggregate_summary(self, per_game_df: pd.DataFrame) -> pd.DataFrame:
        def _top1_share(names_series: pd.Series) -> float:
            top1 = names_series.dropna().apply(lambda v: v.split(", ")[0] if v else "")
            if top1.empty:
                return np.nan
            return top1.value_counts(normalize=True).iloc[0]

        power_cols = [
            "power_top5_avg_tdp",
            "power_top5_avg_psu",
            "power_top5_avg_ppw",
            "power_top5_avg_overprovisioning",
            "power_top5_avg_efficiency_regret",
        ]
        knn_cols = [
            "knn50_avg_tdp",
            "knn50_avg_psu",
            "knn50_avg_ppw",
            "knn50_avg_overprovisioning",
            "knn50_avg_efficiency_regret",
        ]
        knn_ppw_cols = [
            "knn50_ppw_top5_avg_tdp",
            "knn50_ppw_top5_avg_psu",
            "knn50_ppw_top5_avg_ppw",
            "knn50_ppw_top5_avg_overprovisioning",
            "knn50_ppw_top5_avg_efficiency_regret",
        ]

        summary_rows = []

        power_unique = per_game_df["power_top5_gpu_names"].str.split(", ").explode().nunique()
        knn_ppw_unique = per_game_df["knn50_ppw_top5_gpu_names"].str.split(", ").explode().nunique()

        knn_unique = np.nan
        try:
            knn_ids = set()
            for _, game_row in self.games_df.iterrows():
                knn_df = self.get_knn_candidates(game_row, n_neighbors=self.config.knn_k)
                knn_ids.update(knn_df[GPU_ID_COL].tolist())
            knn_unique = len(knn_ids)
        except Exception:
            knn_unique = np.nan

        summary_rows.append({
            "method": "Power_Top5",
            "avg_tdp": per_game_df[power_cols[0]].mean(),
            "avg_psu": per_game_df[power_cols[1]].mean(),
            "avg_ppw": per_game_df[power_cols[2]].mean(),
            "avg_overprovisioning": per_game_df[power_cols[3]].mean(),
            "avg_efficiency_regret": per_game_df[power_cols[4]].mean(),
            "unique_gpus": power_unique,
            "top1_share": _top1_share(per_game_df["power_top5_gpu_names"]),
            "avg_overlap_with_knn50": per_game_df["overlap_rate"].mean(),
        })

        summary_rows.append({
            "method": "KNN50",
            "avg_tdp": per_game_df[knn_cols[0]].mean(),
            "avg_psu": per_game_df[knn_cols[1]].mean(),
            "avg_ppw": per_game_df[knn_cols[2]].mean(),
            "avg_overprovisioning": per_game_df[knn_cols[3]].mean(),
            "avg_efficiency_regret": per_game_df[knn_cols[4]].mean(),
            "unique_gpus": knn_unique,
            "top1_share": np.nan,
            "avg_overlap_with_knn50": np.nan,
        })

        summary_rows.append({
            "method": "KNN50_PPW_Top5",
            "avg_tdp": per_game_df[knn_ppw_cols[0]].mean(),
            "avg_psu": per_game_df[knn_ppw_cols[1]].mean(),
            "avg_ppw": per_game_df[knn_ppw_cols[2]].mean(),
            "avg_overprovisioning": per_game_df[knn_ppw_cols[3]].mean(),
            "avg_efficiency_regret": per_game_df[knn_ppw_cols[4]].mean(),
            "unique_gpus": knn_ppw_unique,
            "top1_share": _top1_share(per_game_df["knn50_ppw_top5_gpu_names"]),
            "avg_overlap_with_knn50": np.nan,
        })

        return pd.DataFrame(summary_rows)

    def save_outputs(self, per_game_df: pd.DataFrame, summary_df: pd.DataFrame, skipped_df: pd.DataFrame) -> None:
        out_dir = Path(self.config.output_dir)
        figs_dir = out_dir / "figs"
        out_dir.mkdir(parents=True, exist_ok=True)
        figs_dir.mkdir(parents=True, exist_ok=True)

        per_game_df.to_csv(out_dir / "phase1_per_game_topk_knn_analysis.csv", index=False)
        summary_df.to_csv(out_dir / "phase1_aggregate_topk_knn_summary.csv", index=False)
        if not skipped_df.empty:
            skipped_df.to_csv(out_dir / "phase1_skipped_games.csv", index=False)

        self.plot_overlap_histogram(per_game_df, figs_dir)
        self.plot_metric_heatmap(summary_df, figs_dir)
        self.plot_representative_scatter(per_game_df, figs_dir)

    def plot_overlap_histogram(self, per_game_df: pd.DataFrame, figs_dir: Path) -> None:
        counts = per_game_df["overlap_count"].value_counts().reindex(range(0, 6), fill_value=0)

        plt.figure(figsize=(8, 5))
        plt.bar(counts.index.astype(str), counts.values, color="#4C72B0")
        plt.title("Overlap between Power-aware Top-5 and KNN-50")
        plt.xlabel("Overlap count")
        plt.ylabel("Number of games")
        plt.tight_layout()
        plt.savefig(figs_dir / "overlap_histogram_power_top5_vs_knn50.png", dpi=160)
        plt.close()

    def plot_metric_heatmap(self, summary_df: pd.DataFrame, figs_dir: Path) -> None:
        metrics = [
            "avg_tdp",
            "avg_psu",
            "avg_ppw",
            "avg_overprovisioning",
            "avg_efficiency_regret",
        ]

        data = summary_df.set_index("method")[metrics].copy()

        # Normalize so higher is better. For metrics where lower is better,
        # invert the min-max scaling to keep darker = better in the heatmap.
        norm = pd.DataFrame(index=data.index, columns=metrics, dtype=float)
        for col in metrics:
            series = data[col]
            v_min, v_max = series.min(), series.max()
            if v_max == v_min:
                norm[col] = 1.0
                continue

            scaled = (series - v_min) / (v_max - v_min)
            if col in {"avg_tdp", "avg_psu", "avg_overprovisioning", "avg_efficiency_regret"}:
                norm[col] = 1.0 - scaled
            else:
                norm[col] = scaled

        plt.figure(figsize=(8, 3))
        plt.imshow(norm.values, aspect="auto", cmap="viridis")
        plt.xticks(range(len(metrics)), metrics, rotation=30, ha="right")
        plt.yticks(range(len(norm.index)), norm.index)
        plt.colorbar(label="Normalized score (higher is better)")
        plt.title("Top-k vs KNN aggregate metric comparison")
        plt.tight_layout()
        plt.savefig(figs_dir / "metric_heatmap_topk_vs_knn.png", dpi=160)
        plt.close()

    def plot_representative_scatter(self, per_game_df: pd.DataFrame, figs_dir: Path) -> None:
        if per_game_df.empty:
            return

        candidates = per_game_df[(per_game_df["overlap_count"] >= 2) & (per_game_df["overlap_count"] <= 4)]
        if not candidates.empty:
            row = candidates.iloc[0]
        else:
            med = per_game_df["overlap_count"].median()
            row = per_game_df.iloc[(per_game_df["overlap_count"] - med).abs().argsort().iloc[0]]

        game_name = row["game_name"]
        game_row = self.games_df[self.games_df["name"] == game_name].iloc[0]
        knn = self.get_knn_candidates(game_row, n_neighbors=self.config.knn_k)
        topk = self.get_power_topk(game_row, k=self.config.k_top)

        if knn.empty or topk.empty:
            return
        game_perf = game_row.get("perf_score")
        overprov = knn["perf_score"] - game_perf
        if pd.notna(game_perf) and game_perf != 0:
            overprov = overprov / game_perf
        else:
            overprov = np.full(len(knn), np.nan)

        plt.figure(figsize=(7, 5))
        plt.scatter(overprov, knn[PPW_COL], s=20, alpha=0.5, label="KNN50")
        if pd.notna(game_perf) and game_perf != 0:
            topk_overprov = (topk["perf_score"] - game_perf) / game_perf
        else:
            topk_overprov = np.full(len(topk), np.nan)

        plt.scatter(
            topk_overprov,
            topk[PPW_COL],
            s=80,
            color="#E24A33",
            label="Power_Top5",
        )

        for _, row in topk.iterrows():
            if pd.notna(game_perf) and game_perf != 0:
                x_val = (row["perf_score"] - game_perf) / game_perf
            else:
                x_val = np.nan
            plt.annotate(row["name"], (x_val, row[PPW_COL]), fontsize=7)

        plt.xlabel("Overprovisioning (relative)")
        plt.ylabel("Performance per watt")
        plt.title("Representative game: PPW vs Overprovisioning")
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / "representative_game_scatter_ppw_vs_overprov.png", dpi=160)
        plt.close()


def _load_best_prediction_columns() -> Tuple[Optional[str], Optional[str]]:
    tdp_col = None
    psu_col = None

    try:
        tdp_metrics = pd.read_csv(TDP_METRICS_PATH)
        if not tdp_metrics.empty:
            tdp_col = tdp_metrics.iloc[0]["prediction_column_to_use"]
    except FileNotFoundError:
        tdp_col = None

    try:
        psu_metrics = pd.read_csv(PSU_METRICS_PATH)
        if not psu_metrics.empty:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Top-k vs KNN-50 analysis")
    parser.add_argument("--mode", choices=["min", "recom"], default="recom", help="Game vector mode")
    parser.add_argument("--k-top", type=int, default=5, help="Top-k size (default: 5)")
    parser.add_argument("--knn-k", type=int, default=50, help="KNN candidate size (default: 50)")
    parser.add_argument("--threshold", type=float, default=0.80, help="Soft filter threshold")
    parser.add_argument("--output-dir", default="phase1_outputs", help="Output directory")
    parser.add_argument("--num-workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--log-every", type=int, default=200, help="Progress log interval")
    args = parser.parse_args()

    games_df = pd.read_csv(GAME_VECTORS[args.mode])
    gpu_df = pd.read_csv(GPU_VECTORS)
    gpu_df = _attach_power_predictions(gpu_df)

    config = Phase1Config(
        mode=args.mode,
        k_top=args.k_top,
        knn_k=args.knn_k,
        soft_threshold=args.threshold,
        output_dir=args.output_dir,
        num_workers=max(args.num_workers, 1),
        log_every=max(args.log_every, 1),
    )

    analyzer = TopKRecommendationAnalyzer(games_df, gpu_df, config)
    per_game_df, summary_df, skipped_df = analyzer.run()
    analyzer.save_outputs(per_game_df, summary_df, skipped_df)


if __name__ == "__main__":
    main()
