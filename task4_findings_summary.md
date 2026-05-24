# Task 4 Findings Summary

## Data Sources Used
- Baseline summaries: data/results/baseline_metrics_summary.csv
- Subgroups: data/results/subgroup_metrics_summary_min.csv, data/results/subgroup_metrics_summary_recom.csv
- Model metrics: data/results/tdp_model_metrics.csv, data/results/psu_model_metrics.csv
- Next-steps diagnostics (Downloads): method_diversity_summary.csv, min_recom_rank_correlation.csv, perf_feasible_ablation_summary.csv, weighted_sum_sweep_summary.csv
- Figures: figures/avg_tdp_by_method_*.png, figures/avg_psu_by_method_*.png, figures/perf_per_watt_by_method_*.png, figures/overprovisioning_by_method_*.png, figures/efficiency_regret_by_method_*.png, figures/pareto_scatter_*.png

## Core Results (Baselines)
- Coverage is ~1.0 for almost all methods, indicating the hard-feasible filter rarely leaves empty sets.
- Highest performance always selects the max-power GPU (TDP 600, PSU 1000), causing high overprovisioning and poor efficiency regret.
- Lowest TDP / lowest PSU / perf-per-TDP / weighted-sum collapse to the same low-power GPU in most cases.
- Pareto-knee sits in a mid-range power region, providing a balanced trade-off.
- Proposed recommender stays close to perf-per-TDP but slightly increases TDP/PSU, indicating a conservative shift.

## Model Quality (Power Predictions)
- TDP: tree-based models (XGBoost/GB/RF) achieve ~15–16 W MAE and R2 ~0.92, suggesting solid predictive support.
- PSU: best models show ~26–28 W MAE and R2 ~0.91–0.92, weaker than TDP but still useful.

## Subgroup Stability
- Harder subgroups (difficulty, VRAM, DirectX) push KNN/random/lowest-margin toward higher power.
- Efficiency baselines (perf-per-TDP, lowest TDP/PSU) barely respond to subgroup difficulty, indicating low adaptivity.

## Diagnostic Findings (Next Steps)
- Method diversity shows strong collapse for most baselines: top-1 GPU share ~1.0 for several methods.
- Only KNN, random, and smallest-margin show meaningful variety.
- Min vs recom ranks are highly correlated (Spearman ~0.97–0.99), reflecting rigidity rather than adaptive behavior.
- Perf-feasible ablation with alpha=1.10 changes magnitudes but does not break the collapse.
- Weighted-sum sweep (alpha=1.10) shows two regimes: a high-power corner (TDP ~300, PSU ~700) when TDP/PSU weights are low, and a low-power corner (min: TDP ~51–54, PSU ~214–219; recom: TDP ~82–99, PSU ~275–304) for most other weight settings. This indicates weak smooth trade-offs and continued collapse to a dominant GPU selection.

## Critical Assessment
- The results are consistent and stable, but many baselines converge to a single GPU, reducing comparative value.
- This appears driven by dataset structure and objective similarity, not by implementation errors.
- The recommender framework is functioning as designed; the observed collapse is a property of the objective landscape.

## Industry Relevance
- The metrics map directly to build cost, power supply sizing, thermal limits, and sustainability targets.
- The analysis highlights when simple energy heuristics may under-provision and when balanced approaches are needed.
- This supports practical deployment for OEM configuration, PC build guidance, and energy-aware hardware procurement.

## What We Added
- task4_results_report.md: comprehensive report with figures and SOTA-style references.
- task4_tables_section.tex: full LaTeX subsection with figures, tables, analysis, and references.
- src/next_steps_analysis.py: diagnostics script (sensitivity sweep, ablation, diversity, rank correlation).

## Recommended Next Actions
- Run perf-feasible ablation with multiple alpha values (1.00–1.15) and compare diversity shifts.
- Expand weighted-sum grid to explore a broader trade-off range.
- Consider adding a strict perf-score feasibility option to reduce under-provisioning risk.
