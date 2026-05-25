"""Phase 4: reported baseline comparison and static-spec positioning report."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
REPORTS_DIR = RESULTS_DIR / "reports"


def _ensure_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _find_first(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _load_csv_optional(paths: List[Path]) -> Optional[pd.DataFrame]:
    path = _find_first(paths)
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _log(msg: str) -> None:
    print(f"[phase4] {msg}")


def build_published_baseline_table(our_summary: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = [
        {
            "paper_id": "wu2015gpgpu",
            "citation_key": "wu2015gpgpu",
            "year": 2015,
            "venue": "HPCA",
            "task": "GPGPU performance and dynamic power estimation across hardware configurations",
            "target": "execution time and dynamic power",
            "input_features": "hardware configuration plus base-run performance counters",
            "requires_runtime_data": "yes",
            "requires_profiling": "yes",
            "requires_hardware_counters": "yes",
            "requires_workload_execution": "yes",
            "reported_metric": "average percent error",
            "reported_value": "performance 15%, dynamic power 10%",
            "reported_unit": "percent error",
            "public_row_level_data_found": "no",
            "direct_reproduction_attempted": "no",
            "directly_comparable_to_ours": "no",
            "comparison_type": "reported-only, different task",
            "notes": "Uses runtime/base-run counters and dynamic power; useful related work but not a static recommendation benchmark.",
        },
        {
            "paper_id": "dutta2018gpu_power",
            "citation_key": "dutta_gpu_power",
            "year": 2018,
            "venue": "Computing Frontiers",
            "task": "GPU power prediction for DVFS space exploration",
            "target": "application power across core/memory frequency settings",
            "input_features": "DVFS settings plus nvprof utilization counters",
            "requires_runtime_data": "yes",
            "requires_profiling": "yes",
            "requires_hardware_counters": "yes",
            "requires_workload_execution": "yes",
            "reported_metric": "MAE percentage",
            "reported_value": "3.5% ensemble MAE, 11% max error",
            "reported_unit": "percent",
            "public_row_level_data_found": "no",
            "direct_reproduction_attempted": "no",
            "directly_comparable_to_ours": "no",
            "comparison_type": "reported-only, different task",
            "notes": "Strong runtime/DVFS power prediction baseline; not directly comparable to static catalog recommendation.",
        },
        {
            "paper_id": "moolchandani2022concurrent",
            "citation_key": "moolchandani2022concurrent",
            "year": 2022,
            "venue": "ACM TACO",
            "task": "performance and power prediction for concurrent GPU applications",
            "target": "concurrent execution time/performance and power",
            "input_features": "standalone CPU/GPU measurements, CPU-side counters, fairness signal",
            "requires_runtime_data": "yes",
            "requires_profiling": "yes",
            "requires_hardware_counters": "partially",
            "requires_workload_execution": "yes",
            "reported_metric": "MAPE / accuracy",
            "reported_value": "about 9% MAPE performance and about 4% MAPE power for concurrency-2 setting",
            "reported_unit": "percent",
            "public_row_level_data_found": "no",
            "direct_reproduction_attempted": "no",
            "directly_comparable_to_ours": "no",
            "comparison_type": "reported-only, different task",
            "notes": "Lower GPU-counter dependence than Wu/Dutta, but still requires profiling and workload execution.",
        },
        {
            "paper_id": "braun2021mangrove",
            "citation_key": "braun2021mangrove",
            "year": 2021,
            "venue": "ACM TACO",
            "task": "portable prediction of GPU kernel execution time and power",
            "target": "kernel time and power",
            "input_features": "kernel features, benchmark configuration, GPU measurements/databases",
            "requires_runtime_data": "yes",
            "requires_profiling": "yes",
            "requires_hardware_counters": "depends on feature set",
            "requires_workload_execution": "yes",
            "reported_metric": "paper-reported prediction errors",
            "reported_value": "see paper",
            "reported_unit": "mixed",
            "public_row_level_data_found": "repo available, database download required",
            "direct_reproduction_attempted": "no",
            "directly_comparable_to_ours": "partially",
            "comparison_type": "public artifact, possible future benchmark",
            "notes": "Good future runnable benchmark, but not required for this report.",
        },
        {
            "paper_id": "mlenergy2025",
            "citation_key": "chung2025mlenergy",
            "year": 2025,
            "venue": "NeurIPS Datasets and Benchmarks",
            "task": "inference energy measurement and optimization benchmark",
            "target": "inference energy/performance across configurations",
            "input_features": "measured inference configurations",
            "requires_runtime_data": "yes",
            "requires_profiling": "yes",
            "requires_hardware_counters": "no or benchmark-dependent",
            "requires_workload_execution": "yes",
            "reported_metric": "benchmark/leaderboard energy metrics",
            "reported_value": "benchmark-specific",
            "reported_unit": "mixed",
            "public_row_level_data_found": "repo available",
            "direct_reproduction_attempted": "no",
            "directly_comparable_to_ours": "partially",
            "comparison_type": "recent energy-aware benchmark motivation",
            "notes": "Use as recent NeurIPS citation motivating energy-aware metrics and optimization.",
        },
    ]

    ours_value = ""
    if our_summary is not None and not our_summary.empty:
        ltr_row = our_summary[our_summary["method"] == "LTR_Utility_Top5"]
        if not ltr_row.empty:
            row = ltr_row.iloc[0]
            parts = []
            for col in ["avg_ppw", "avg_tdp", "avg_psu", "ndcg_at_5", "recall_at_5"]:
                if col in row and pd.notna(row[col]):
                    parts.append(f"{col}={row[col]:.3f}")
            ours_value = "; ".join(parts)

    rows.append(
        {
            "paper_id": "ours_static_gpu_recommender",
            "citation_key": "ours",
            "year": 2026,
            "venue": "ECE 228 project",
            "task": "static pre-deployment energy-aware GPU recommendation",
            "target": "top-k feasible GPU shortlist",
            "input_features": "static GPU specifications plus game requirements",
            "requires_runtime_data": "no",
            "requires_profiling": "no",
            "requires_hardware_counters": "no",
            "requires_workload_execution": "no",
            "reported_metric": "PPW, TDP, PSU, efficiency regret, diversity, NDCG@5",
            "reported_value": ours_value or "from phase3_aggregate_ltr_topk_summary.csv when available",
            "reported_unit": "mixed",
            "public_row_level_data_found": "yes, project data",
            "direct_reproduction_attempted": "yes",
            "directly_comparable_to_ours": "yes",
            "comparison_type": "our setting",
            "notes": "Lower-observability recommendation setting; not a replacement for runtime power predictors.",
        }
    )

    return pd.DataFrame(rows)


def build_our_recommendation_summary(phase3_summary: Optional[pd.DataFrame]) -> pd.DataFrame:
    methods = [
        "Power_Top5",
        "UtilityFormula_Top5",
        "ML_Utility_Top5",
        "LTR_Utility_Top5",
        "KNN50_Feasible",
        "KNN50_Feasible_PPW_Top5",
    ]

    interpretations = {
        "KNN50_Feasible": "Feature-affinity retrieval is diverse but inefficient.",
        "KNN50_Feasible_PPW_Top5": "PPW reranking improves KNN but remains weaker than direct power-aware methods.",
        "Power_Top5": "Strong pure efficiency baseline but collapses to few GPUs.",
        "UtilityFormula_Top5": "Transparent multi-objective score close to Power_Top5.",
        "ML_Utility_Top5": "Pointwise ML mostly reproduces the hand-designed utility formula.",
        "LTR_Utility_Top5": "Best balanced method; lower TDP/PSU and much less collapse while preserving high PPW.",
    }

    rows = []
    if phase3_summary is not None and not phase3_summary.empty:
        for method in methods:
            subset = phase3_summary[phase3_summary["method"] == method]
            if subset.empty:
                rows.append({
                    "method": method,
                    "avg_tdp": np.nan,
                    "avg_psu": np.nan,
                    "avg_ppw": np.nan,
                    "avg_efficiency_regret": np.nan,
                    "unique_gpus": np.nan,
                    "top1_share": np.nan,
                    "ndcg_at_5": np.nan,
                    "recall_at_5": np.nan,
                    "interpretation": interpretations.get(method, ""),
                })
                continue

            row = subset.iloc[0]
            rows.append({
                "method": method,
                "avg_tdp": row.get("avg_tdp"),
                "avg_psu": row.get("avg_psu"),
                "avg_ppw": row.get("avg_ppw"),
                "avg_efficiency_regret": row.get("avg_efficiency_regret"),
                "unique_gpus": row.get("unique_gpus"),
                "top1_share": row.get("top1_share"),
                "ndcg_at_5": row.get("ndcg_at_5"),
                "recall_at_5": row.get("recall_at_5"),
                "interpretation": interpretations.get(method, ""),
            })
    else:
        for method in methods:
            rows.append({
                "method": method,
                "avg_tdp": np.nan,
                "avg_psu": np.nan,
                "avg_ppw": np.nan,
                "avg_efficiency_regret": np.nan,
                "unique_gpus": np.nan,
                "top1_share": np.nan,
                "ndcg_at_5": np.nan,
                "recall_at_5": np.nan,
                "interpretation": interpretations.get(method, ""),
            })

    return pd.DataFrame(rows)


def build_feature_availability_table() -> pd.DataFrame:
    rows = [
        {
            "method_or_paper": "Wu et al. 2015",
            "static_gpu_specs": "no",
            "game_or_workload_requirements": "yes",
            "runtime_counters": "yes",
            "dvfs_states": "no",
            "profiling_required": "yes",
            "measured_power_required": "yes",
            "measured_runtime_required": "yes",
            "supports_predeployment": "no",
            "supports_recommendation": "no",
            "main_output": "power/performance prediction",
        },
        {
            "method_or_paper": "Dutta et al. 2018",
            "static_gpu_specs": "no",
            "game_or_workload_requirements": "yes",
            "runtime_counters": "yes",
            "dvfs_states": "yes",
            "profiling_required": "yes",
            "measured_power_required": "yes",
            "measured_runtime_required": "yes",
            "supports_predeployment": "no",
            "supports_recommendation": "no",
            "main_output": "DVFS power prediction",
        },
        {
            "method_or_paper": "Moolchandani et al. 2022",
            "static_gpu_specs": "no",
            "game_or_workload_requirements": "yes",
            "runtime_counters": "partially",
            "dvfs_states": "no",
            "profiling_required": "yes",
            "measured_power_required": "yes",
            "measured_runtime_required": "yes",
            "supports_predeployment": "no",
            "supports_recommendation": "no",
            "main_output": "concurrent performance/power",
        },
        {
            "method_or_paper": "Braun/GPU Mangrove",
            "static_gpu_specs": "no",
            "game_or_workload_requirements": "yes",
            "runtime_counters": "yes",
            "dvfs_states": "no",
            "profiling_required": "yes",
            "measured_power_required": "yes",
            "measured_runtime_required": "yes",
            "supports_predeployment": "no",
            "supports_recommendation": "no",
            "main_output": "kernel time/power prediction",
        },
        {
            "method_or_paper": "ML.ENERGY 2025",
            "static_gpu_specs": "no",
            "game_or_workload_requirements": "yes",
            "runtime_counters": "varies",
            "dvfs_states": "varies",
            "profiling_required": "yes",
            "measured_power_required": "yes",
            "measured_runtime_required": "yes",
            "supports_predeployment": "no",
            "supports_recommendation": "no",
            "main_output": "energy benchmarking",
        },
        {
            "method_or_paper": "Our Power_Top5",
            "static_gpu_specs": "yes",
            "game_or_workload_requirements": "yes",
            "runtime_counters": "no",
            "dvfs_states": "no",
            "profiling_required": "no",
            "measured_power_required": "no",
            "measured_runtime_required": "no",
            "supports_predeployment": "yes",
            "supports_recommendation": "yes",
            "main_output": "top-k recommendation",
        },
        {
            "method_or_paper": "Our LTR_Utility_Top5",
            "static_gpu_specs": "yes",
            "game_or_workload_requirements": "yes",
            "runtime_counters": "no",
            "dvfs_states": "no",
            "profiling_required": "no",
            "measured_power_required": "no",
            "measured_runtime_required": "no",
            "supports_predeployment": "yes",
            "supports_recommendation": "yes",
            "main_output": "top-k recommendation",
        },
    ]

    return pd.DataFrame(rows)


def build_ta_feedback_table() -> pd.DataFrame:
    rows = [
        {
            "ta_feedback": "NeurIPS GPU efficiency papers should be considered.",
            "risk_if_unaddressed": "Related work may look incomplete.",
            "project_response": "Add ML.ENERGY 2025 as recent NeurIPS Datasets and Benchmarks citation for energy-aware inference measurement and optimization.",
            "where_addressed": "Related Work and Phase 4 comparison.",
            "remaining_limitation": "ML.ENERGY targets GenAI inference, not gaming GPU recommendation.",
        },
        {
            "ta_feedback": "Lowest PSU alone may select an old weak GPU.",
            "risk_if_unaddressed": "Recommender may optimize power while ignoring capability.",
            "project_response": "Use feasibility filtering and PPW/utility/LTR ranking after enforcing performance and compatibility constraints.",
            "where_addressed": "Methodology, Recommendation Objective, Top-k Evaluation.",
            "remaining_limitation": "Static performance score is a proxy, not measured FPS.",
        },
        {
            "ta_feedback": "TMUs/ROPs and detailed architecture features can be inconsistent across GPU generations.",
            "risk_if_unaddressed": "Model may overtrust non-comparable raw hardware features.",
            "project_response": "Treat detailed architectural specs as soft features or ranking signals, not strict universal ground truth. Use tolerant filtering and performance proxies.",
            "where_addressed": "Feasibility Filtering, Performance Score, Limitations.",
            "remaining_limitation": "True cross-generation comparability requires FPS/runtime benchmarks.",
        },
        {
            "ta_feedback": "Need relevant benchmark or baseline.",
            "risk_if_unaddressed": "Project may look isolated from existing GPU power literature.",
            "project_response": "Add artifact audit and reported-baseline table for Wu, Dutta, Moolchandani, GPU Mangrove, and ML.ENERGY.",
            "where_addressed": "Phase 4 comparison section.",
            "remaining_limitation": "Direct reproduction is limited by missing public row-level datasets and different task definitions.",
        },
    ]

    return pd.DataFrame(rows)


def plot_observability_vs_output(output_path: Path) -> Optional[Path]:
    rows = [
        {"name": "Our LTR_Utility_Top5", "observability": 1, "output": "recommendation"},
        {"name": "ML.ENERGY", "observability": 3, "output": "benchmark"},
        {"name": "GPU Mangrove", "observability": 3, "output": "prediction"},
        {"name": "Moolchandani", "observability": 2, "output": "prediction"},
        {"name": "Dutta", "observability": 3, "output": "prediction"},
        {"name": "Wu", "observability": 3, "output": "prediction"},
    ]

    df = pd.DataFrame(rows)
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(df["name"], df["observability"])
    ax.set_ylabel("Observability level")
    ax.set_xlabel("Method or paper")
    ax.set_title("Observability vs output setting")
    ax.set_ylim(0, 3.5)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def build_report_markdown(
    baseline_path: Path,
    feature_path: Path,
    ta_path: Path,
    output_path: Path,
) -> None:
    lines = []
    lines.append("# Phase 4 Reported Baseline Comparison")
    lines.append("")
    lines.append("## Purpose")
    lines.append("This Phase 4 report provides a reported-baseline and artifact audit comparison, not a direct reproduction. Metrics are taken from published papers and are not apples-to-apples accuracy benchmarks against our static-spec recommendation task.")
    lines.append("")
    lines.append("## Artifact Availability")
    lines.append("Public row-level datasets and maintained code are not consistently available for older GPU power/performance papers, which limits direct reproduction.")
    lines.append("")
    lines.append("## Published Baselines")
    lines.append(f"See {baseline_path.name} for the reported-only baseline table.")
    lines.append("")
    lines.append("## Feature Availability and Observability")
    lines.append("Prior methods typically depend on runtime counters, profiling traces, DVFS sweeps, or workload execution. Our approach uses lower-observability static GPU specifications and game requirements.")
    lines.append("")
    lines.append("## Why Our Approach Is Still Relevant")
    lines.append("Runtime power/performance predictors are expected to be more accurate on their original measured-power tasks because they use richer information. Our method addresses a different earlier-stage problem: pre-deployment GPU recommendation when runtime telemetry, FPS, and measured energy are unavailable.")
    lines.append("")
    lines.append("## Response to TA Feedback")
    lines.append(f"See {ta_path.name} for the feedback-response table.")
    lines.append("")
    lines.append("## Recommended Report Text")
    lines.append("You can cite the Phase 4 tables in related work and limitations to clarify that the comparison is reported-only and task-specific.")
    lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def build_latex_insert(output_path: Path) -> None:
    lines = []
    lines.append("Prior GPU power/performance predictors use runtime counters, DVFS sweeps, or profiling traces; they are stronger on their measured-power tasks but require richer runtime observability. Our method targets a different setting: pre-deployment GPU recommendation with static specifications and game requirements when runtime telemetry is unavailable.")
    lines.append("")
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append("\\begin{tabular}{llllll}")
    lines.append("\\toprule")
    lines.append("Work & Year & Input Signals & Runtime Data? & Output & How We Use It \\")
    lines.append("\\midrule")
    lines.append("Wu et al. & 2015 & Counters + config & Yes & Prediction & Reported-only \\")
    lines.append("Dutta et al. & 2018 & DVFS + counters & Yes & Prediction & Reported-only \\")
    lines.append("Moolchandani et al. & 2022 & Profiling + fairness & Yes & Prediction & Reported-only \\")
    lines.append("GPU Mangrove & 2021 & Kernel features + DB & Yes & Prediction & Future benchmark \\")
    lines.append("ML.ENERGY & 2025 & Inference configs & Yes & Benchmark & Motivation \\")
    lines.append("Ours & 2026 & Static specs + requirements & No & Recommendation & This project \\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\caption{Positioning against published GPU power/performance work. Reported metrics are task-specific and not a direct accuracy comparison.}")
    lines.append("\\label{tab:phase4-positioning}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("The comparison is not an apples-to-apples accuracy benchmark; it highlights differences in observability and task definition between runtime prediction and static pre-deployment recommendation.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    _ensure_dirs()

    phase3_paths = [
        REPO_ROOT / "results" / "tables" / "phase3_aggregate_ltr_topk_summary.csv",
        REPO_ROOT / "phase3_aggregate_ltr_topk_summary.csv",
        REPO_ROOT / "results" / "phase3_aggregate_ltr_topk_summary.csv",
        REPO_ROOT / "outputs" / "phase3_aggregate_ltr_topk_summary.csv",
        REPO_ROOT / "data" / "processed" / "phase3_aggregate_ltr_topk_summary.csv",
    ]

    phase2_metrics_paths = [
        REPO_ROOT / "results" / "tables" / "phase2_model_regression_metrics.csv",
        REPO_ROOT / "phase2_model_regression_metrics.csv",
    ]

    phase3_metrics_paths = [
        REPO_ROOT / "results" / "tables" / "phase3_ltr_model_metrics.csv",
        REPO_ROOT / "phase3_ltr_model_metrics.csv",
    ]

    phase3_feature_paths = [
        REPO_ROOT / "results" / "tables" / "phase3_ltr_feature_importance.csv",
        REPO_ROOT / "phase3_ltr_feature_importance.csv",
    ]

    phase3_summary = _load_csv_optional(phase3_paths)
    phase2_metrics = _load_csv_optional(phase2_metrics_paths)
    phase3_metrics = _load_csv_optional(phase3_metrics_paths)
    phase3_features = _load_csv_optional(phase3_feature_paths)

    if phase3_summary is None:
        _log("missing phase3_aggregate_ltr_topk_summary.csv")
    if phase2_metrics is None:
        _log("missing phase2_model_regression_metrics.csv")
    if phase3_metrics is None:
        _log("missing phase3_ltr_model_metrics.csv")
    if phase3_features is None:
        _log("missing phase3_ltr_feature_importance.csv")

    our_summary = build_our_recommendation_summary(phase3_summary)
    baseline_df = build_published_baseline_table(our_summary)
    feature_df = build_feature_availability_table()
    ta_df = build_ta_feedback_table()

    baseline_path = TABLES_DIR / "phase4_published_baseline_comparison.csv"
    our_summary_path = TABLES_DIR / "phase4_our_recommendation_summary.csv"
    feature_path = TABLES_DIR / "phase4_feature_availability.csv"
    ta_path = TABLES_DIR / "phase4_ta_feedback_response.csv"

    baseline_df.to_csv(baseline_path, index=False)
    our_summary.to_csv(our_summary_path, index=False)
    feature_df.to_csv(feature_path, index=False)
    ta_df.to_csv(ta_path, index=False)

    plot_path = plot_observability_vs_output(PLOTS_DIR / "phase4_observability_vs_output.png")

    report_path = REPORTS_DIR / "phase4_reported_baseline_comparison.md"
    latex_path = REPORTS_DIR / "phase4_latex_insert.tex"

    build_report_markdown(baseline_path, feature_path, ta_path, report_path)
    build_latex_insert(latex_path)

    created_paths = [baseline_path, our_summary_path, feature_path, ta_path, report_path, latex_path]
    if plot_path is not None:
        created_paths.append(plot_path)

    for path in created_paths:
        _log(f"created {path}")


if __name__ == "__main__":
    main()
