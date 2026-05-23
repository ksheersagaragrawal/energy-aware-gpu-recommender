# Task 4 Results Report

## Scope and data sources

This report summarizes Task 4 baseline evaluation results for the energy-aware GPU recommender. It uses the outputs produced by the baseline evaluation pipeline and model training scripts, and it follows the evaluation framing and metric definitions in [ece228_task4_baselines_evaluation_plan.md](ece228_task4_baselines_evaluation_plan.md).

Data sources used:
- Baseline summary metrics: [data/results/baseline_metrics_summary.csv](data/results/baseline_metrics_summary.csv)
- Feasible set statistics: [data/results/feasible_set_stats.csv](data/results/feasible_set_stats.csv)
- Subgroup summaries (min track): [data/results/subgroup_metrics_summary_min.csv](data/results/subgroup_metrics_summary_min.csv)
- Subgroup summaries (recom track): [data/results/subgroup_metrics_summary_recom.csv](data/results/subgroup_metrics_summary_recom.csv)
- Model metrics (TDP): [data/results/tdp_model_metrics.csv](data/results/tdp_model_metrics.csv)
- Model metrics (PSU): [data/results/psu_model_metrics.csv](data/results/psu_model_metrics.csv)
- Plots: [figures/avg_tdp_by_method_min.png](figures/avg_tdp_by_method_min.png), [figures/avg_tdp_by_method_recom.png](figures/avg_tdp_by_method_recom.png), [figures/avg_psu_by_method_min.png](figures/avg_psu_by_method_min.png), [figures/avg_psu_by_method_recom.png](figures/avg_psu_by_method_recom.png), [figures/perf_per_watt_by_method_min.png](figures/perf_per_watt_by_method_min.png), [figures/perf_per_watt_by_method_recom.png](figures/perf_per_watt_by_method_recom.png), [figures/overprovisioning_by_method_min.png](figures/overprovisioning_by_method_min.png), [figures/overprovisioning_by_method_recom.png](figures/overprovisioning_by_method_recom.png), [figures/efficiency_regret_by_method_min.png](figures/efficiency_regret_by_method_min.png), [figures/efficiency_regret_by_method_recom.png](figures/efficiency_regret_by_method_recom.png), [figures/pareto_scatter_min.png](figures/pareto_scatter_min.png), [figures/pareto_scatter_recom.png](figures/pareto_scatter_recom.png).

## Why these baselines and metrics

The baseline set and metrics follow the rationale in [ece228_task4_baselines_evaluation_plan.md](ece228_task4_baselines_evaluation_plan.md):
- Hard-feasible candidate sets ensure fair comparison.
- Energy-aware and multi-objective baselines (lowest TDP, lowest PSU, perf per TDP, Pareto knee, weighted sum) represent standard, defensible competing strategies in energy-aware recommendation.
- Coverage, TDP, PSU, performance-per-watt, overprovisioning, and efficiency regret capture feasibility, energy cost, efficiency, and right-sizing tradeoffs.

## Model quality (TDP and PSU predictors)

These models provide the predicted power and PSU values used in evaluation.

TDP model quality:
- Best test MAE is ~15.56 W for XGBoost with $R^2 \approx 0.926$ in [data/results/tdp_model_metrics.csv](data/results/tdp_model_metrics.csv).
- Gradient Boosting and Random Forest are close, with $R^2 \approx 0.916$ to $0.919$.
- Linear and L1/L2 baselines are much weaker, confirming non-linear effects in the feature space.

PSU model quality:
- Best test MAE is ~26.39 W for Gradient Boosting with $R^2 \approx 0.909$ in [data/results/psu_model_metrics.csv](data/results/psu_model_metrics.csv).
- Random Forest and XGBoost are comparable with $R^2 \approx 0.918$.
- MLP and linear baselines lag substantially.

Inference: The tree-based models are strong enough for downstream evaluation. Errors remain non-trivial for PSU, so interpretations that depend heavily on small PSU differences should be made cautiously.

## Feasible set sanity check

- All baselines show coverage at or near 1.0 (see summary in [data/results/baseline_metrics_summary.csv](data/results/baseline_metrics_summary.csv)).
- Feasible set sizes are consistently non-zero for sampled games in [data/results/feasible_set_stats.csv](data/results/feasible_set_stats.csv), which explains the full coverage across methods.

Inference: The hard-feasible filtering is stable and does not collapse the candidate pool.

## Aggregate baseline behavior (min and recom tracks)

Key takeaways from [data/results/baseline_metrics_summary.csv](data/results/baseline_metrics_summary.csv):

- Highest performance baseline:
  - Always selects the max-power GPU (TDP 600 W, PSU 1000 W), producing the highest overprovisioning and efficiency regret.
  - This provides a strong upper anchor for performance-first selection.

- Lowest TDP and lowest PSU baselines:
  - Consistently choose the same ultra-low-power GPU (TDP 43 W, PSU 200 W) across both tracks.
  - They maximize efficiency, but can under-provision for demanding games, which is visible as negative overprovisioning in some subgroup buckets.

- Perf per TDP and weighted sum:
  - Collapse to the same low-power GPU in the min track, and largely the same in the recom track.
  - This indicates a dominant low-power GPU in the candidate pool under the current constraints and objective weights.

- Pareto knee:
  - Sits around mid-range (TDP ~140 W, PSU ~300 W) with moderate efficiency regret and moderate overprovisioning.
  - This acts as a balanced multi-objective baseline, as intended in the plan.

- Safety factor perf per TDP:
  - Coverage dips slightly below 1.0 (min and recom). This is expected when a safety margin removes candidates for some games.
  - The tradeoff is higher TDP/PSU but reduced under-provisioning risk.

- Proposed recommender:
  - Closest to perf per TDP / weighted sum but with slightly higher TDP and PSU, indicating a softer trade toward feasibility margin.
  - This is a reasonable compromise if the goal is to reduce under-provisioning while staying efficient.

These tradeoffs are visualized in:
- TDP/PSU averages: [figures/avg_tdp_by_method_min.png](figures/avg_tdp_by_method_min.png), [figures/avg_tdp_by_method_recom.png](figures/avg_tdp_by_method_recom.png), [figures/avg_psu_by_method_min.png](figures/avg_psu_by_method_min.png), [figures/avg_psu_by_method_recom.png](figures/avg_psu_by_method_recom.png)
- Efficiency and regret: [figures/perf_per_watt_by_method_min.png](figures/perf_per_watt_by_method_min.png), [figures/perf_per_watt_by_method_recom.png](figures/perf_per_watt_by_method_recom.png), [figures/efficiency_regret_by_method_min.png](figures/efficiency_regret_by_method_min.png), [figures/efficiency_regret_by_method_recom.png](figures/efficiency_regret_by_method_recom.png)
- Overprovisioning: [figures/overprovisioning_by_method_min.png](figures/overprovisioning_by_method_min.png), [figures/overprovisioning_by_method_recom.png](figures/overprovisioning_by_method_recom.png)
- Pareto geometry: [figures/pareto_scatter_min.png](figures/pareto_scatter_min.png), [figures/pareto_scatter_recom.png](figures/pareto_scatter_recom.png)

## Subgroup analysis (difficulty, VRAM, DirectX)

Subgroups were defined to probe stability across game requirement regimes, per [ece228_task4_baselines_evaluation_plan.md](ece228_task4_baselines_evaluation_plan.md). Summaries are in [data/results/subgroup_metrics_summary_min.csv](data/results/subgroup_metrics_summary_min.csv) and [data/results/subgroup_metrics_summary_recom.csv](data/results/subgroup_metrics_summary_recom.csv).

Observed patterns:
- Difficulty buckets:
  - KNN and random feasible scale TDP/PSU upward with higher difficulty, indicating that they are sensitive to requirement intensity.
  - Lowest TDP/PSU and perf per TDP remain fixed across difficulty, implying a dominant low-power choice that does not adapt to increased requirements.
  - Safety factor and smallest margin baselines increase power usage for high difficulty, consistent with their design to ensure margin.

- VRAM buckets:
  - For higher VRAM buckets (8-16 GB, >16 GB) in the recom track, safety factor and smallest margin push to high TDP/PSU regions, while perf per TDP remains fixed at lower values.
  - This suggests that in high VRAM regimes, perf per TDP may under-provision unless soft constraints include perf score or VRAM-aware thresholds.

- DirectX buckets:
  - DirectX >= 12 games lead to higher TDP/PSU for KNN and random feasible, which aligns with higher requirement complexity.
  - The fixed baselines (lowest TDP/PSU, perf per TDP, weighted sum) do not respond to DirectX tiers, signaling limited adaptivity.

Inference: The subgroup analysis supports that methods with explicit margin or Pareto balancing respond to harder requirement tiers, while pure efficiency baselines can become insensitive to the requirement regime.

## Comparison to common SOTA-style approaches (qualitative)

No external benchmark or SOTA dataset is included in this project, so we avoid numerical comparisons to external methods. However, we can contextualize the baselines against common approaches reported in multi-objective recommendation and energy-efficiency literature, as described in [ece228_task4_baselines_evaluation_plan.md](ece228_task4_baselines_evaluation_plan.md):

- Scalarized multi-objective rankings (weighted sum) and Pareto-front selection are standard SOTA-style baselines for multi-objective recommendation [1][2][3].
- Pareto-based and multi-objective learning approaches are widely used to balance competing objectives, including efficiency and utility [1][2][4].
- Performance-per-watt and energy-efficiency analyses are standard in GPU evaluation, motivating the perf per watt and efficiency regret metrics used here [5][6].

Against these SOTA-style baselines:
- The proposed recommender matches or exceeds efficiency-oriented baselines in efficiency regret while reducing under-provisioning relative to pure perf per TDP or lowest TDP.
- Pareto knee and safety factor baselines remain strong competitors, providing balanced tradeoffs that are challenging to beat without explicit objective weighting.

For a formal SOTA comparison, we would need to evaluate on a shared benchmark dataset or reproduce metrics from the cited papers.

## Key conclusions

- The pipeline is stable: coverage is 1.0 for most methods and feasible sets are non-empty.
- Tree-based models provide solid predictive power for TDP and PSU, which supports downstream evaluation.
- Multiple baselines collapse to a dominant low-power GPU, indicating a data-driven efficiency frontier that is hard to beat under the current objective definitions.
- Methods with explicit margin control (safety factor, smallest margin) adapt better to high-difficulty and high-VRAM regimes.
- The proposed recommender behaves as a practical compromise between efficiency and safety, remaining close to perf per TDP while modestly increasing power to reduce under-provisioning risk.

## Recommendations for next iteration

- Add sensitivity analysis over weighted sum coefficients to show robustness.
- Introduce a hard perf-score feasibility option to evaluate the impact of strict performance constraints.
- Add external benchmark references if a formal SOTA comparison is required.

## References

[1] Ribeiro, M. T., Lacerda, A., Veloso, A., and Ziviani, N., "Pareto-efficient hybridization for multi-objective recommender systems," RecSys 2012. https://dl.acm.org/doi/abs/10.1145/2365952.2365962

[2] Lin, X., Chen, H., Pei, C., Sun, F., Xiao, X., and Sun, H., "A pareto-efficient algorithm for multiple objective optimization in e-commerce recommendation," RecSys 2019. https://dl.acm.org/doi/abs/10.1145/3298689.3346998

[3] Wu, H., Ma, C., Mitra, B., Diaz, F., and Liu, X., "A multi-objective optimization framework for multi-stakeholder fairness-aware recommendation," TOIS 2022. https://dl.acm.org/doi/abs/10.1145/3564285

[4] Li, P., and Tuzhilin, A., "Deep pareto reinforcement learning for multi-objective recommender systems," arXiv 2024. https://arxiv.org/abs/2407.03580

[5] Cebrian, J. M., Guerrero, G. D., and others, "Energy efficiency analysis of GPUs," IEEE Cluster 2012. https://ieeexplore.ieee.org/abstract/document/6270749/

[6] Coplin, J., and Burtscher, M., "Energy, power, and performance characterization of GPGPU benchmark programs," IEEE IPDPS Workshops 2016. https://ieeexplore.ieee.org/abstract/document/7530002/
