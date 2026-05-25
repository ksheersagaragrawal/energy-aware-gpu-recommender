"""Phase 4: Reporting utilities for external benchmark audit and summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from phase4_external_benchmark import BenchmarkPaths


@dataclass(frozen=True)
class ReportPaths:
    tables_dir: Path
    plots_dir: Path
    reports_dir: Path


def get_report_paths(repo_root: Path) -> ReportPaths:
    return ReportPaths(
        tables_dir=repo_root / "results" / "tables",
        plots_dir=repo_root / "results" / "plots",
        reports_dir=repo_root / "results" / "reports",
    )


def build_artifact_audit_table(output_path: Path) -> pd.DataFrame:
    rows = [
        {
            "paper_id": "wu2015gpgpu",
            "title": "GPGPU Performance and Power Estimation Using Machine Learning",
            "year": 2015,
            "venue": "HPCA",
            "doi": "10.1109/HPCA.2015.7056041",
            "public_pdf": "unknown",
            "public_code": "unknown",
            "public_dataset": "unknown",
            "dataset_url": "",
            "code_url": "",
            "runtime_counters_required": "yes",
            "workload_execution_required": "yes",
            "static_features_available": "partial",
            "direct_reproduction_possible": "unknown",
            "recommended_use_in_our_report": "citation_only",
            "notes": "No public row-level data confirmed in this audit.",
        },
        {
            "paper_id": "dutta2018gpu_power",
            "title": "GPU Power Prediction via Ensemble Machine Learning for DVFS Space Exploration",
            "year": 2018,
            "venue": "Computing Frontiers",
            "doi": "10.1145/3203217.3203226",
            "public_pdf": "unknown",
            "public_code": "unknown",
            "public_dataset": "unknown",
            "dataset_url": "",
            "code_url": "",
            "runtime_counters_required": "yes",
            "workload_execution_required": "yes",
            "static_features_available": "partial",
            "direct_reproduction_possible": "unknown",
            "recommended_use_in_our_report": "citation_only",
            "notes": "No public row-level data confirmed in this audit.",
        },
        {
            "paper_id": "moolchandani2022concurrent",
            "title": "Performance and Power Prediction for Concurrent Execution on GPUs",
            "year": 2022,
            "venue": "ACM TACO",
            "doi": "10.1145/3524096",
            "public_pdf": "unknown",
            "public_code": "unknown",
            "public_dataset": "unknown",
            "dataset_url": "",
            "code_url": "",
            "runtime_counters_required": "yes",
            "workload_execution_required": "yes",
            "static_features_available": "partial",
            "direct_reproduction_possible": "unknown",
            "recommended_use_in_our_report": "citation_only",
            "notes": "No public row-level data confirmed in this audit.",
        },
        {
            "paper_id": "braun2021mangrove",
            "title": "GPU Mangrove: A Simple Model for Portable and Fast Prediction of Execution Time and Power Consumption of GPU Kernels",
            "year": 2021,
            "venue": "arXiv",
            "doi": "",
            "public_pdf": "unknown",
            "public_code": "yes",
            "public_dataset": "unknown",
            "dataset_url": "",
            "code_url": "https://github.com/lorenzbraun/gpu-mangrove",
            "runtime_counters_required": "yes",
            "workload_execution_required": "yes",
            "static_features_available": "partial",
            "direct_reproduction_possible": "unknown",
            "recommended_use_in_our_report": "primary_benchmark",
            "notes": "Preferred runnable external benchmark if data downloads succeed.",
        },
        {
            "paper_id": "mlenergy2025",
            "title": "ML.ENERGY Benchmark",
            "year": 2025,
            "venue": "NeurIPS Datasets and Benchmarks",
            "doi": "",
            "public_pdf": "unknown",
            "public_code": "unknown",
            "public_dataset": "unknown",
            "dataset_url": "",
            "code_url": "https://github.com/ml-energy/benchmark",
            "runtime_counters_required": "varies",
            "workload_execution_required": "varies",
            "static_features_available": "unknown",
            "direct_reproduction_possible": "unknown",
            "recommended_use_in_our_report": "citation_only",
            "notes": "Use as motivation if dataset is not directly reusable.",
        },
    ]

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def build_related_work_table(output_path: Path) -> pd.DataFrame:
    rows = [
        {
            "work": "Wu et al. 2015",
            "year": 2015,
            "venue": "HPCA",
            "problem_setting": "Kernel-level power/performance prediction",
            "input_signals": "Runtime counters + workload execution",
            "requires_runtime_data": "yes",
            "requires_hardware_counters": "yes",
            "output": "Power/performance estimates",
            "our_use": "Related work citation",
            "difference_from_our_project": "Our inputs are static GPU specs and game requirements.",
        },
        {
            "work": "Dutta et al. 2018",
            "year": 2018,
            "venue": "Computing Frontiers",
            "problem_setting": "DVFS-aware power prediction",
            "input_signals": "Runtime profiling and DVFS states",
            "requires_runtime_data": "yes",
            "requires_hardware_counters": "yes",
            "output": "Power prediction under DVFS",
            "our_use": "Related work citation",
            "difference_from_our_project": "Our method is pre-deployment and static-only.",
        },
        {
            "work": "Moolchandani et al. 2022",
            "year": 2022,
            "venue": "ACM TACO",
            "problem_setting": "Concurrent kernel power/performance",
            "input_signals": "Runtime counters and traces",
            "requires_runtime_data": "yes",
            "requires_hardware_counters": "yes",
            "output": "Power/performance for concurrent kernels",
            "our_use": "Related work citation",
            "difference_from_our_project": "We do not assume runtime traces or counters.",
        },
        {
            "work": "GPU Mangrove",
            "year": 2021,
            "venue": "arXiv",
            "problem_setting": "Kernel execution time and power",
            "input_signals": "Runtime counters and workload execution",
            "requires_runtime_data": "yes",
            "requires_hardware_counters": "yes",
            "output": "Time/power prediction",
            "our_use": "Runnable benchmark + reduced feature baseline",
            "difference_from_our_project": "We focus on GPU recommendation from static specs.",
        },
        {
            "work": "ML.ENERGY",
            "year": 2025,
            "venue": "NeurIPS Datasets and Benchmarks",
            "problem_setting": "Energy benchmarking",
            "input_signals": "Varies by benchmark",
            "requires_runtime_data": "yes",
            "requires_hardware_counters": "varies",
            "output": "Energy measurements",
            "our_use": "Motivation and context",
            "difference_from_our_project": "Our pipeline targets pre-deployment ranking.",
        },
    ]

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def plot_prediction_metrics(metrics_path: Path, output_path: Path) -> Optional[Path]:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None

    if df["feature_set"].nunique() < 2:
        return None

    metric = "mape" if df["mape"].notna().any() else "rmse"
    pivot = df.pivot_table(index="feature_set", values=metric, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(6, 4))
    pivot.plot(kind="bar", legend=False, ax=ax)
    ax.set_ylabel(metric.upper())
    ax.set_xlabel("Feature set")
    ax.set_title("Prediction accuracy by feature set")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_recommendation_metrics(metrics_path: Path, output_path: Path) -> Optional[Path]:
    if not metrics_path.exists():
        return None
    df = pd.read_csv(metrics_path)
    if df.empty:
        return None

    metric = "mean_true_ppw_at_5" if "mean_true_ppw_at_5" in df.columns else "recall_at_5"
    pivot = df.pivot_table(index="method", values=metric, aggfunc="mean")

    fig, ax = plt.subplots(figsize=(6, 4))
    pivot.plot(kind="bar", legend=False, ax=ax)
    ax.set_ylabel(metric)
    ax.set_xlabel("Method")
    ax.set_title("Recommendation metrics (mean)")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def plot_pred_vs_true(
    dataset_path: Path,
    predictions_path: Path,
    output_path: Path,
) -> Optional[Path]:
    if not dataset_path.exists() or not predictions_path.exists():
        return None

    df = pd.read_csv(dataset_path)
    preds = pd.read_csv(predictions_path)
    power_preds = preds[preds["target"] == "power_w"]
    if power_preds.empty or df.empty:
        return None

    best = power_preds.groupby("feature_set")["prediction"].count().idxmax()
    best_preds = power_preds[power_preds["feature_set"] == best]

    merged = df.merge(best_preds[["row_id", "prediction"]], on="row_id", how="inner")
    if merged.empty:
        return None

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(merged["prediction"], merged["power_w"], alpha=0.6)
    min_val = min(merged["prediction"].min(), merged["power_w"].min())
    max_val = max(merged["prediction"].max(), merged["power_w"].max())
    ax.plot([min_val, max_val], [min_val, max_val])
    ax.set_xlabel("Predicted power")
    ax.set_ylabel("True power")
    ax.set_title(f"Predicted vs true power ({best})")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def build_summary_report(
    paths: BenchmarkPaths,
    output_path: Path,
) -> None:
    audit_path = paths.tables_dir / "phase4_artifact_audit.csv"
    related_path = paths.tables_dir / "phase4_related_work_positioning.csv"
    metrics_path = paths.tables_dir / "phase4_prediction_metrics.csv"
    rec_path = paths.tables_dir / "phase4_recommendation_metrics.csv"

    audit_df = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    rec_df = pd.read_csv(rec_path) if rec_path.exists() else pd.DataFrame()

    missing_rows = audit_df[audit_df["public_dataset"] != "yes"] if not audit_df.empty else pd.DataFrame()

    lines = []
    lines.append("# Phase 4 Summary")
    lines.append("")
    lines.append("## Executive summary")
    lines.append("- Public row-level data for Wu/Dutta/Moolchandani was not confirmed in this audit.")
    lines.append("- GPU Mangrove is the preferred runnable benchmark when its data download succeeds.")
    lines.append("- ML.ENERGY is included as context unless its dataset is directly reusable.")
    lines.append("")
    lines.append("## Why this benchmark matters")
    lines.append("Runtime-heavy papers rely on counters/profiling/execution; our pipeline uses static specs and game requirements for pre-deployment screening. This lower-observability setting is different from measured runtime prediction.")
    lines.append("")
    lines.append("## Artifact audit table")
    lines.append(f"See {audit_path.name} for detailed availability.")
    lines.append("Download outcomes are logged in phase4_download_log.txt.")
    lines.append("")
    lines.append("## Feature-set comparison")
    if metrics_df.empty:
        lines.append("No external dataset metrics were produced.")
    else:
        lines.append("Static-only feature sets are expected to be less accurate than runtime-full features.")
    lines.append("")
    lines.append("## Recommendation comparison")
    if rec_df.empty:
        lines.append("Recommendation metrics were not produced because prediction outputs were unavailable.")
    else:
        lines.append("Power-aware ranking with a performance floor avoids selecting underpowered low-power candidates.")
    lines.append("")
    lines.append("## Key takeaway")
    lines.append(
        "External runtime-power papers remain stronger for their original measured-power tasks because they use richer observability. Our contribution is complementary: using only static/pre-deployment information, the proposed pipeline can produce feasible power-aware shortlists and avoids the failure mode of selecting the lowest-power but underpowered candidate."
    )
    lines.append("")
    lines.append("## Limitations")
    lines.append("- No exact reproduction of Wu/Dutta/Moolchandani without public row-level data.")
    lines.append("- GPU Mangrove is kernel/configuration prediction, not game-to-GPU catalog recommendation.")
    lines.append("- Static features are less informative than runtime counters.")
    lines.append("- TDP/PSU and static performance are proxies, not true runtime energy or FPS.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def build_latex_insert(output_path: Path) -> None:
    lines = [
        "We audited three GPU power/performance prediction papers; their exact row-level datasets were not publicly available in our audit.",
        "We therefore used the public GPU Mangrove benchmark when available for reduced-feature/static-feature comparison.",
        "This comparison is not a claim of SOTA replacement; it demonstrates the trade-off between runtime-heavy accuracy and static pre-deployment recommendation and supports the use of PPW and performance floors.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
