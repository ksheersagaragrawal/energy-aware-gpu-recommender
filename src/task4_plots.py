"""Plotting utilities for Task 4 baseline evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency
    plt = None
    _IMPORT_ERROR = exc


@dataclass(frozen=True)
class PlotConfig:
    output_dir: Path
    dpi: int = 300


class PlotGenerator:
    def __init__(self, config: PlotConfig, logger):
        self.config = config
        self.logger = logger

    def ensure_matplotlib(self) -> None:
        if plt is None:
            raise ImportError("matplotlib is required for plotting") from _IMPORT_ERROR

    def plot_bar_by_method(
        self,
        summary_df: pd.DataFrame,
        track: str,
        metric_col: str,
        title: str,
        ylabel: str,
        filename: str,
    ) -> None:
        self.ensure_matplotlib()
        data = summary_df[summary_df["track"] == track].copy()
        data = data.sort_values(metric_col, ascending=False)
        if data.empty:
            self.logger.warning("No summary data for track=%s metric=%s", track, metric_col)
            return

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(data["method"], data[metric_col], color="#2d6a4f")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("method")
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        fig.tight_layout()

        out = self.config.output_dir / filename
        fig.savefig(out, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Saved plot: %s", out)

    def plot_scatter_pareto(
        self,
        metrics_df: pd.DataFrame,
        track: str,
        filename: str,
    ) -> None:
        self.ensure_matplotlib()
        data = metrics_df[metrics_df["track"] == track].copy()
        data = data.dropna(subset=["selected_tdp_w", "selected_perf_score"])
        if data.empty:
            self.logger.warning("No scatter data for track=%s", track)
            return

        fig, ax = plt.subplots(figsize=(6, 5))
        methods = data["method"].unique().tolist()
        for method in methods:
            subset = data[data["method"] == method]
            ax.scatter(
                subset["selected_tdp_w"],
                subset["selected_perf_score"],
                s=10,
                alpha=0.4,
                label=method,
            )

        ax.set_xlabel("TDP (W)")
        ax.set_ylabel("perf_score")
        ax.set_title("Pareto scatter: TDP vs perf_score")
        ax.legend(loc="best", fontsize=7, frameon=False)
        fig.tight_layout()

        out = self.config.output_dir / filename
        fig.savefig(out, dpi=self.config.dpi, bbox_inches="tight")
        plt.close(fig)
        self.logger.info("Saved plot: %s", out)

    def generate_all(self, summary_df: pd.DataFrame, metrics_df: pd.DataFrame, tracks: List[str]) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for track in tracks:
            self.plot_bar_by_method(
                summary_df,
                track,
                "selected_tdp_w_mean",
                f"Average TDP by method ({track})",
                "TDP (W)",
                f"avg_tdp_by_method_{track}.png",
            )
            self.plot_bar_by_method(
                summary_df,
                track,
                "selected_psu_w_mean",
                f"Average PSU by method ({track})",
                "PSU (W)",
                f"avg_psu_by_method_{track}.png",
            )
            self.plot_bar_by_method(
                summary_df,
                track,
                "selected_perf_per_watt_mean",
                f"Performance per watt by method ({track})",
                "perf_score / TDP",
                f"perf_per_watt_by_method_{track}.png",
            )
            self.plot_bar_by_method(
                summary_df,
                track,
                "overprov_rel_mean",
                f"Over-provisioning by method ({track})",
                "relative over-provisioning",
                f"overprovisioning_by_method_{track}.png",
            )
            self.plot_bar_by_method(
                summary_df,
                track,
                "eff_regret_rel_mean",
                f"Efficiency regret by method ({track})",
                "relative efficiency regret",
                f"efficiency_regret_by_method_{track}.png",
            )
            self.plot_scatter_pareto(metrics_df, track, f"pareto_scatter_{track}.png")
