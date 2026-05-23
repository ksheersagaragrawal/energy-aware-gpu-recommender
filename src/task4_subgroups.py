"""Subgroup evaluation for Task 4 baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SubgroupConfig:
    vram_bins_mb: Tuple[int, int, int, int] = (2048, 4096, 8192, 16384)


class SubgroupEvaluator:
    def __init__(self, config: SubgroupConfig, logger):
        self.config = config
        self.logger = logger

    def evaluate(self, metrics_df: pd.DataFrame, games_df: pd.DataFrame, mode: str) -> pd.DataFrame:
        games = games_df[["name", "perf_score", "min_vram_mb", "recom_vram_mb", "min_direct_x", "recom_direct_x"]].copy()
        games = games.rename(columns={"name": "game_name", "perf_score": "game_perf_score"})

        merged = metrics_df.merge(games, on="game_name", how="left", suffixes=("", "_game"))

        perf_col = "game_perf_score"
        if perf_col not in merged.columns and "game_perf_score_game" in merged.columns:
            perf_col = "game_perf_score_game"

        merged["difficulty_bucket"] = self._difficulty_bucket(merged[perf_col])
        merged["vram_bucket"] = self._vram_bucket(merged, mode)
        merged["directx_bucket"] = self._directx_bucket(merged, mode)

        summary = []
        for subgroup_col in ["difficulty_bucket", "vram_bucket", "directx_bucket"]:
            for (track, method, bucket), group in merged.groupby(["track", "method", subgroup_col], dropna=False):
                row = {
                    "track": track,
                    "method": method,
                    "subgroup": subgroup_col,
                    "bucket": bucket,
                }
                row.update(self._aggregate(group))
                summary.append(row)

        summary_df = pd.DataFrame(summary)
        self.logger.info("Built subgroup summary: %d rows", len(summary_df))
        return summary_df

    def _aggregate(self, group: pd.DataFrame) -> Dict[str, float]:
        def safe(series: pd.Series) -> Dict[str, float]:
            series = series.dropna()
            if series.empty:
                return {"mean": np.nan, "median": np.nan, "p90": np.nan}
            return {
                "mean": series.mean(),
                "median": series.median(),
                "p90": series.quantile(0.9),
            }

        metrics = {
            "coverage": safe(group["coverage"]),
            "tdp": safe(group["selected_tdp_w"]),
            "psu": safe(group["selected_psu_w"]),
            "ppw": safe(group["selected_perf_per_watt"]),
            "overprov": safe(group["overprov_rel"]),
            "eff_regret": safe(group["eff_regret_rel"]),
        }

        flat = {}
        for key, values in metrics.items():
            for stat, val in values.items():
                flat[f"{key}_{stat}"] = val
        return flat

    def _difficulty_bucket(self, perf: pd.Series) -> pd.Series:
        return pd.qcut(perf, q=[0, 0.33, 0.66, 1.0], labels=["low", "mid", "high"], duplicates="drop")

    def _vram_bucket(self, merged: pd.DataFrame, mode: str) -> pd.Series:
        vram_col = "min_vram_mb" if mode == "min" else "recom_vram_mb"
        bins = [0] + list(self.config.vram_bins_mb) + [np.inf]
        labels = [
            "<=2GB",
            "2-4GB",
            "4-8GB",
            "8-16GB",
            ">16GB",
        ]
        return pd.cut(merged[vram_col], bins=bins, labels=labels, include_lowest=True)

    def _directx_bucket(self, merged: pd.DataFrame, mode: str) -> pd.Series:
        dx_col = "min_direct_x" if mode == "min" else "recom_direct_x"
        dx = merged[dx_col]
        buckets = pd.Series(index=merged.index, dtype=object)
        buckets.loc[dx < 11] = "<=10"
        buckets.loc[(dx >= 11) & (dx < 12)] = "11"
        buckets.loc[dx >= 12] = ">=12"
        return buckets
