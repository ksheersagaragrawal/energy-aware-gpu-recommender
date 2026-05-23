"""Baseline evaluation for the energy-aware GPU recommender.

Implements the Task 4 baseline suite with shared hard-feasibility filtering,
per-game recommendations, and aggregate evaluation metrics.

Usage:
    python src/baselines.py --mode min
    python src/baselines.py --mode recom --random-runs 50
    python src/baselines.py --mode both --bootstrap-ci
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from task4_plots import PlotConfig, PlotGenerator
from task4_subgroups import SubgroupConfig, SubgroupEvaluator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

GAME_VECTORS = {
    "min": ROOT / "data" / "vectors" / "game_vectors_min.csv",
    "recom": ROOT / "data" / "vectors" / "game_vectors_recom.csv",
}
GPU_VECTORS = ROOT / "data" / "vectors" / "gpu_power_vectors.csv"

RESULTS_DIR = ROOT / "data" / "results"
FIGURES_DIR = ROOT / "figures"

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
class EvaluationConfig:
    mode: str
    random_runs: int = 30
    random_seed: int = 42
    soft_threshold: float = 0.80
    safety_alpha: float = 1.10
    weights: Dict[str, float] = None
    bootstrap_ci: bool = False
    bootstrap_samples: int = 500
    log_level: str = "INFO"

    def resolved_weights(self) -> Dict[str, float]:
        if self.weights is None:
            return {
                "perf": 1.0,
                "tdp": 1.0,
                "psu": 0.5,
                "overprov": 0.5,
            }
        return self.weights


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def build_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("baseline_eval")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------


class DataLoader:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def load_game_vectors(self, mode: str) -> pd.DataFrame:
        path = GAME_VECTORS[mode]
        self.logger.info("Loading game vectors: %s", path)
        df = pd.read_csv(path)
        self.logger.info("Loaded games: %d rows, %d columns", len(df), len(df.columns))
        return df

    def load_gpu_vectors(self) -> pd.DataFrame:
        self.logger.info("Loading GPU vectors: %s", GPU_VECTORS)
        df = pd.read_csv(GPU_VECTORS)
        self.logger.info("Loaded GPUs: %d rows, %d columns", len(df), len(df.columns))
        return df

    def validate_columns(self, games: pd.DataFrame, gpus: pd.DataFrame) -> None:
        required_game = {
            "name",
            "min_vram_mb",
            "min_direct_x",
            "recom_vram_mb",
            "recom_direct_x",
            "perf_score",
        }
        missing_game = required_game - set(games.columns)
        if missing_game:
            raise ValueError(f"Missing required game columns: {sorted(missing_game)}")

        required_gpu = {
            "brand",
            "name",
            "memory_mb",
            "direct_x",
            "tdp_w",
            "psu_w",
            "perf_score",
        }
        missing_gpu = required_gpu - set(gpus.columns)
        if missing_gpu:
            raise ValueError(f"Missing required GPU columns: {sorted(missing_gpu)}")

        for _, gpu_col in SOFT_FEATURE_MAP.items():
            if gpu_col not in gpus.columns:
                raise ValueError(f"Missing GPU feature column: {gpu_col}")

        for _, (_, gpu_col) in KNN_FEATURE_MAP.items():
            if gpu_col not in gpus.columns:
                raise ValueError(f"Missing GPU KNN column: {gpu_col}")

        self.logger.info("Column validation passed")


# ---------------------------------------------------------------------------
# Feasibility filter
# ---------------------------------------------------------------------------


class FeasibilityFilter:
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def get_requirements(self, game: pd.Series, mode: str) -> Tuple[Optional[float], Optional[float]]:
        if mode == "min":
            vram = game.get("min_vram_mb")
            dx = game.get("min_direct_x")
        else:
            vram = game.get("recom_vram_mb")
            dx = game.get("recom_direct_x")
        vram = float(vram) if pd.notna(vram) else None
        dx = float(dx) if pd.notna(dx) else None
        return vram, dx

    def filter(self, game: pd.Series, gpus: pd.DataFrame, mode: str) -> pd.DataFrame:
        vram_req, dx_req = self.get_requirements(game, mode)
        mask = pd.Series(True, index=gpus.index)

        if vram_req is not None and vram_req > 0:
            mask &= gpus["memory_mb"] >= vram_req

        if dx_req is not None and dx_req > 0:
            mask &= gpus["direct_x"] >= dx_req

        feasible = gpus[mask].copy()
        self.logger.debug(
            "Feasible GPUs for %s (%s): %d / %d (vram>=%s, dx>=%s)",
            game.get("name"),
            mode,
            mask.sum(),
            len(gpus),
            vram_req,
            dx_req,
        )
        return feasible


# ---------------------------------------------------------------------------
# Baseline strategies
# ---------------------------------------------------------------------------


class BaselineStrategy:
    name: str = "base"

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
        raise NotImplementedError


class RandomFeasible(BaselineStrategy):
    name = "random_feasible"

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
        if feasible.empty:
            return None
        idx = rng.integers(0, len(feasible))
        return feasible.iloc[int(idx)]


class LowestTDP(BaselineStrategy):
    name = "lowest_tdp"

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
        if feasible.empty:
            return None
        return feasible.sort_values(["tdp_w", "perf_score"], ascending=[True, False]).iloc[0]


class LowestPSU(BaselineStrategy):
    name = "lowest_psu"

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
        if feasible.empty:
            return None
        return feasible.sort_values(["psu_w", "perf_score"], ascending=[True, False]).iloc[0]


class HighestPerformance(BaselineStrategy):
    name = "highest_perf"

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
        if feasible.empty:
            return None
        return feasible.sort_values(["perf_score", "tdp_w"], ascending=[False, True]).iloc[0]


class PerfPerTDP(BaselineStrategy):
    name = "perf_per_tdp"

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
        if feasible.empty:
            return None
        scores = feasible["perf_score"] / feasible["tdp_w"]
        idx = scores.replace([np.inf, -np.inf], np.nan).fillna(-np.inf).idxmax()
        return feasible.loc[idx]


class SmallestMargin(BaselineStrategy):
    name = "smallest_margin"

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
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


class SafetyFactorPerfPerTDP(BaselineStrategy):
    name = "safety_factor_perf_per_tdp"

    def __init__(self, alpha: float):
        self.alpha = alpha

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
        if feasible.empty:
            return None
        req = game.get("perf_score")
        if pd.isna(req) or req <= 0:
            return None
        cutoff = req * self.alpha
        filtered = feasible[feasible["perf_score"] >= cutoff]
        if filtered.empty:
            return None
        scores = filtered["perf_score"] / filtered["tdp_w"]
        idx = scores.replace([np.inf, -np.inf], np.nan).fillna(-np.inf).idxmax()
        return filtered.loc[idx]


class ParetoKnee(BaselineStrategy):
    name = "pareto_knee"

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
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

        perf_norm = (pareto["perf_score"] - pareto["perf_score"].min())
        if perf_norm.max() > 0:
            perf_norm = perf_norm / perf_norm.max()
        tdp_norm = pareto["tdp_w"] - pareto["tdp_w"].min()
        if tdp_norm.max() > 0:
            tdp_norm = tdp_norm / tdp_norm.max()
        psu_norm = pareto["psu_w"] - pareto["psu_w"].min()
        if psu_norm.max() > 0:
            psu_norm = psu_norm / psu_norm.max()

        # Ideal point: max perf (1), min tdp (0), min psu (0)
        distances = np.sqrt((1 - perf_norm) ** 2 + tdp_norm ** 2 + psu_norm ** 2)
        idx = distances.idxmin()
        return pareto.loc[idx]


class WeightedSumUtility(BaselineStrategy):
    name = "weighted_sum"

    def __init__(self, weights: Dict[str, float]):
        self.weights = weights

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
        if feasible.empty:
            return None

        def zscore(series: pd.Series) -> pd.Series:
            std = series.std()
            if pd.isna(std) or std == 0:
                return pd.Series(0.0, index=series.index)
            return (series - series.mean()) / std

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
            self.weights.get("perf", 1.0) * perf_z
            - self.weights.get("tdp", 1.0) * tdp_z
            - self.weights.get("psu", 0.5) * psu_z
            - self.weights.get("overprov", 0.5) * overprov_z
        )
        idx = score.idxmax()
        return feasible.loc[idx]


class KNNRetrieval(BaselineStrategy):
    name = "knn_retrieval"

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
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


class ProposedRecommender(BaselineStrategy):
    name = "proposed_recommender"

    def __init__(self, threshold: float):
        self.threshold = threshold

    def select(self, game: pd.Series, feasible: pd.DataFrame, rng: np.random.Generator) -> Optional[pd.Series]:
        if feasible.empty:
            return None

        mask = pd.Series(True, index=feasible.index)
        for game_col, gpu_col in SOFT_FEATURE_MAP.items():
            req = game.get(game_col)
            if pd.isna(req) or req <= 0:
                continue
            mask &= feasible[gpu_col] >= (req * self.threshold)

        filtered = feasible[mask]
        if filtered.empty:
            return None

        scores = filtered["perf_score"] / filtered["tdp_w"]
        idx = scores.replace([np.inf, -np.inf], np.nan).fillna(-np.inf).idxmax()
        return filtered.loc[idx]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class BaselineEvaluator:
    def __init__(self, config: EvaluationConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.loader = DataLoader(logger)
        self.filter = FeasibilityFilter(logger)

    def build_strategies(self) -> List[BaselineStrategy]:
        weights = self.config.resolved_weights()
        return [
            RandomFeasible(),
            LowestTDP(),
            LowestPSU(),
            HighestPerformance(),
            PerfPerTDP(),
            SmallestMargin(),
            SafetyFactorPerfPerTDP(self.config.safety_alpha),
            ParetoKnee(),
            WeightedSumUtility(weights),
            KNNRetrieval(),
            ProposedRecommender(self.config.soft_threshold),
        ]

    def evaluate_mode(self, mode: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        games = self.loader.load_game_vectors(mode)
        gpus = self.loader.load_gpu_vectors()
        self.loader.validate_columns(games, gpus)

        strategies = self.build_strategies()
        rng = np.random.default_rng(self.config.random_seed)

        rows = []
        feasible_stats = []

        for _, game in games.iterrows():
            feasible = self.filter.filter(game, gpus, mode)
            best_ppw = self._best_ppw(feasible)
            feasible_stats.append(
                {
                    "game_name": game.get("name"),
                    "track": mode,
                    "feasible_set_size": len(feasible),
                }
            )

            for strat in strategies:
                runs = self.config.random_runs if strat.name == "random_feasible" else 1
                for run_idx in range(runs):
                    selected = strat.select(game, feasible, rng)
                    rows.append(
                        self._build_recommendation_row(
                            game,
                            feasible,
                            selected,
                            strat.name,
                            mode,
                            run_idx if runs > 1 else None,
                            best_ppw,
                        )
                    )

        recs = pd.DataFrame(rows)
        feasible_stats_df = pd.DataFrame(feasible_stats)
        metrics = self.compute_metrics(recs, feasible_stats_df)
        return recs, feasible_stats_df, metrics

    def _build_recommendation_row(
        self,
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

    def compute_metrics(self, recs: pd.DataFrame, feasible_stats: pd.DataFrame) -> pd.DataFrame:
        metrics = recs.copy()
        metrics["coverage"] = metrics["selected_gpu"].notna().astype(int)

        req = metrics["game_perf_score"]
        sel_perf = metrics["selected_perf_score"]
        metrics["overprov_abs"] = sel_perf - req
        metrics["overprov_rel"] = (sel_perf / req) - 1
        metrics.loc[(req.isna()) | (req <= 0), ["overprov_abs", "overprov_rel"]] = np.nan

        metrics["eff_regret_abs"] = metrics["best_ppw"] - metrics["selected_perf_per_watt"]
        metrics["eff_regret_rel"] = metrics["eff_regret_abs"] / metrics["best_ppw"]

        metrics["feature_slack_mean"] = metrics.apply(self._feature_slack, axis=1)

        return metrics

    def _best_ppw(self, feasible: pd.DataFrame) -> float:
        if feasible.empty:
            return np.nan
        ppw = feasible["perf_score"] / feasible["tdp_w"]
        ppw = ppw.replace([np.inf, -np.inf], np.nan).dropna()
        if ppw.empty:
            return np.nan
        return float(ppw.max())

    def _feature_slack(self, row: pd.Series) -> float:
        if pd.isna(row.get("selected_gpu")):
            return np.nan

        game_name = row.get("game_name")
        track = row.get("track")
        # Feature slack is computed later with full rows, fallback to NaN here
        if game_name is None or track is None:
            return np.nan
        return np.nan


# ---------------------------------------------------------------------------
# Metric augmentation with feature slack
# ---------------------------------------------------------------------------


def compute_feature_slack(
    metrics: pd.DataFrame,
    games: pd.DataFrame,
    gpus: pd.DataFrame,
) -> pd.Series:
    game_lookup = games.set_index("name")
    gpu_lookup = gpus.set_index("name")

    slack_vals = []
    for _, row in metrics.iterrows():
        gpu_name = row.get("selected_gpu")
        game_name = row.get("game_name")
        if pd.isna(gpu_name) or pd.isna(game_name):
            slack_vals.append(np.nan)
            continue

        if game_name not in game_lookup.index or gpu_name not in gpu_lookup.index:
            slack_vals.append(np.nan)
            continue

        game = game_lookup.loc[game_name]
        if isinstance(game, pd.DataFrame):
            game = game.iloc[0]

        gpu = gpu_lookup.loc[gpu_name]
        if isinstance(gpu, pd.DataFrame):
            gpu = gpu.iloc[0]

        slacks = []
        for game_col, gpu_col in SOFT_FEATURE_MAP.items():
            g_req = game.get(game_col)
            g_val = gpu.get(gpu_col)
            if pd.isna(g_req) or g_req <= 0 or pd.isna(g_val):
                continue
            slacks.append(max(0.0, (g_val - g_req) / g_req))

        slack_vals.append(float(np.mean(slacks)) if slacks else np.nan)

    return pd.Series(slack_vals, index=metrics.index)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize_metrics(metrics: pd.DataFrame, bootstrap_ci: bool, samples: int) -> pd.DataFrame:
    metric_cols = [
        "coverage",
        "selected_tdp_w",
        "selected_psu_w",
        "selected_perf_per_watt",
        "overprov_abs",
        "overprov_rel",
        "eff_regret_abs",
        "eff_regret_rel",
        "feature_slack_mean",
    ]

    def agg(group: pd.DataFrame) -> Dict[str, float]:
        out = {}
        for col in metric_cols:
            series = group[col].dropna()
            out[f"{col}_mean"] = series.mean() if not series.empty else np.nan
            out[f"{col}_median"] = series.median() if not series.empty else np.nan
            out[f"{col}_std"] = series.std() if not series.empty else np.nan
            out[f"{col}_p90"] = series.quantile(0.9) if not series.empty else np.nan

            if bootstrap_ci and not series.empty:
                out[f"{col}_ci_low"], out[f"{col}_ci_high"] = bootstrap_ci_mean(series.values, samples)
        return out

    rows = []
    for (track, method), group in metrics.groupby(["track", "method"], dropna=False):
        row = {"track": track, "method": method}
        row.update(agg(group))
        rows.append(row)

    return pd.DataFrame(rows)


def bootstrap_ci_mean(values: np.ndarray, samples: int) -> Tuple[float, float]:
    rng = np.random.default_rng(123)
    means = []
    for _ in range(samples):
        resample = rng.choice(values, size=len(values), replace=True)
        means.append(np.mean(resample))
    lower = np.percentile(means, 2.5)
    upper = np.percentile(means, 97.5)
    return float(lower), float(upper)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Task 4 baseline evaluation")
    parser.add_argument("--mode", choices=["min", "recom", "both"], default="min")
    parser.add_argument("--random-runs", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--soft-threshold", type=float, default=0.80)
    parser.add_argument("--safety-alpha", type=float, default=1.10)
    parser.add_argument("--bootstrap-ci", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    config = EvaluationConfig(
        mode=args.mode,
        random_runs=args.random_runs,
        random_seed=args.random_seed,
        soft_threshold=args.soft_threshold,
        safety_alpha=args.safety_alpha,
        bootstrap_ci=args.bootstrap_ci,
        bootstrap_samples=args.bootstrap_samples,
        log_level=args.log_level,
    )

    logger = build_logger(config.log_level)
    evaluator = BaselineEvaluator(config, logger)
    subgroup_eval = SubgroupEvaluator(SubgroupConfig(), logger)
    plotter = PlotGenerator(PlotConfig(FIGURES_DIR), logger)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_recs = []
    all_metrics = []
    all_feasible = []

    modes = [config.mode] if config.mode in {"min", "recom"} else ["min", "recom"]
    for mode in modes:
        logger.info("Evaluating mode: %s", mode)
        recs, feasible_stats, metrics = evaluator.evaluate_mode(mode)

        # Load data again for feature slack calculation
        games = evaluator.loader.load_game_vectors(mode)
        gpus = evaluator.loader.load_gpu_vectors()
        metrics["feature_slack_mean"] = compute_feature_slack(metrics, games, gpus)

        subgroup_summary = subgroup_eval.evaluate(metrics, games, mode)
        subgroup_out = RESULTS_DIR / f"subgroup_metrics_summary_{mode}.csv"
        subgroup_summary.to_csv(subgroup_out, index=False)
        logger.info("Saved subgroup metrics: %s", subgroup_out)

        all_recs.append(recs)
        all_metrics.append(metrics)
        all_feasible.append(feasible_stats)

    recs_df = pd.concat(all_recs, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    feasible_df = pd.concat(all_feasible, ignore_index=True)

    summary_df = summarize_metrics(metrics_df, config.bootstrap_ci, config.bootstrap_samples)

    recs_out = RESULTS_DIR / "baseline_recommendations.csv"
    metrics_out = RESULTS_DIR / "baseline_metrics_per_game.csv"
    summary_out = RESULTS_DIR / "baseline_metrics_summary.csv"
    feasible_out = RESULTS_DIR / "feasible_set_stats.csv"

    recs_df.to_csv(recs_out, index=False)
    metrics_df.to_csv(metrics_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    feasible_df.to_csv(feasible_out, index=False)

    logger.info("Saved recommendations: %s", recs_out)
    logger.info("Saved per-game metrics: %s", metrics_out)
    logger.info("Saved summary metrics: %s", summary_out)
    logger.info("Saved feasible stats: %s", feasible_out)

    try:
        plotter.generate_all(summary_df, metrics_df, modes)
    except ImportError as exc:
        logger.warning("Plotting skipped: %s", exc)


if __name__ == "__main__":
    main()
