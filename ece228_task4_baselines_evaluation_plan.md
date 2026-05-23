# Task 4 Plan: Baselines, Evaluation, Report, and Poster

## Project framing

This project should be presented as a **constraint-aware, energy-aware GPU recommendation system** for PC games. The goal is not simply to recommend the most powerful GPU. The goal is to recommend a **feasible desktop discrete GPU** that satisfies a game’s hardware requirements while reducing unnecessary power demand and over-provisioning.

A defensible framing is:

> Given a game requirement vector and a database of desktop discrete GPUs, we select a feasible GPU that balances compatibility, performance, power demand, and over-provisioning.

This framing is stronger than saying “we use ML to recommend GPUs,” because the final recommendation task is naturally a **constrained multi-objective ranking problem**. It involves several competing objectives:

- Compatibility must be satisfied.
- Performance should be sufficient.
- TDP should be low.
- PSU requirement should be low.
- Over-provisioning should be avoided.
- Efficiency should be high.

This is defensible because recent recommender-system and optimization literature commonly treats recommendation as a multi-objective ranking problem, often using scalarization, Pareto-front approximation, and trade-off analysis. It is also consistent with hardware evaluation practice, where GPU performance, power, and performance-per-watt are commonly reported.

---

## Are the proposed baselines doable?

Yes. All proposed baselines are doable with the current pipeline, assuming the existing cleaned GPU table includes columns such as:

- `gpu_name`
- `vram_mb`
- `directx`
- `tdp_w`
- `psu_w`
- `perf_score`
- optional soft features such as memory bandwidth, texture rate, pixel rate, TMUs, and ROPs

The baselines do **not** require new labels such as actual FPS or measured game power. They operate on the feasible GPU set already produced by the recommender.

The only optional addition that requires external data is a sanity check against PassMark, 3DMark, or another public benchmark score. That is useful, but not required for Task 4.

---

## Core evaluation principle

For each game, every method must choose from the **same hard-feasible GPU set**.

This is important for fairness.

Hard feasibility should include only non-negotiable constraints, such as:

```text
GPU is feasible if:
1. gpu_vram_mb >= game_required_vram_mb
2. gpu_directx >= game_required_directx, if DirectX is available
3. GPU is desktop discrete, based on the project scope
```

Soft constraints, such as texture rate, pixel rate, memory bandwidth, TMUs, ROPs, and `perf_score`, should be used carefully. They can be used by the proposed recommender or by specific baselines, but they should not secretly define the candidate pool for all methods unless clearly stated.

Recommended rule:

> Use hard filters to define the shared candidate pool. Use soft features only inside ranking functions or explicitly named ablations.

This prevents the proposed method from receiving unfair credit for both filtering and ranking.

---

## Baselines to implement

### Baseline 1: Random feasible

**Selection rule:** Randomly select one GPU from the feasible set.

```text
selected_gpu = random_choice(feasible_gpus)
```

**Why we do it:**  
This is a sanity baseline. It shows how much improvement we get over arbitrary feasible selection.

**Why it is defensible:**  
Random baselines are common in recommender and ranking evaluations because they provide a lower-bound reference.

**Implementation note:**  
Run random feasible with multiple seeds, for example 30 or 100 runs, and report the mean and standard deviation.

---

### Baseline 2: Lowest TDP feasible

**Selection rule:** Select the feasible GPU with the lowest TDP.

```text
selected_gpu = argmin_gpu tdp_w
```

**Why we do it:**  
This tests pure GPU-side power minimization.

**Why it is defensible:**  
Since the project is energy-aware, the simplest competing strategy is “choose the lowest-power feasible GPU.”

**Expected behavior:**  
This baseline should achieve low average TDP, but it may choose GPUs with very little performance headroom.

---

### Baseline 3: Lowest PSU feasible

**Selection rule:** Select the feasible GPU with the lowest PSU requirement.

```text
selected_gpu = argmin_gpu psu_w
```

**Why we do it:**  
This tests system-level power/capacity minimization from a user build perspective.

**Why it is defensible:**  
Vendor recommended PSU is not the same as measured energy, but it is a practical compatibility and build-planning signal.

**Expected behavior:**  
This may behave similarly to lowest TDP, but not always. PSU recommendations include assumptions about system-level power requirements.

---

### Baseline 4: Highest performance feasible

**Selection rule:** Select the feasible GPU with the highest `perf_score`.

```text
selected_gpu = argmax_gpu perf_score
```

**Why we do it:**  
This is the performance-first baseline.

**Why it is defensible:**  
It represents the common naive approach of recommending the strongest compatible GPU.

**Expected behavior:**  
This should produce high performance, but it will likely have high TDP, high PSU, and high over-provisioning.

---

### Baseline 5: Performance per TDP

**Selection rule:** Select the feasible GPU with the highest performance-per-watt proxy.

```text
selected_gpu = argmax_gpu perf_score / tdp_w
```

**Why we do it:**  
This is the strongest simple efficiency baseline.

**Why it is defensible:**  
Performance-per-watt is a standard energy-efficiency metric in computing hardware evaluation. It is also widely used in hardware benchmarking and green computing contexts.

**Expected behavior:**  
This baseline may be hard to beat on efficiency regret. That is okay. The proposed method should aim to reduce over-provisioning or provide safer requirement matching while staying competitive in efficiency.

---

### Baseline 6: Smallest-margin feasible

**Selection rule:** Select the feasible GPU with the smallest positive performance surplus over the game requirement.

```text
selected_gpu = argmin_gpu max(0, gpu_perf_score - game_required_perf_score)
```

A relative version is usually better:

```text
margin = (gpu_perf_score - game_required_perf_score) / game_required_perf_score
selected_gpu = feasible GPU with smallest non-negative margin
```

**Why we do it:**  
This is a right-sizing baseline. It asks: “Can we choose the GPU that is just strong enough?”

**Why it is defensible:**  
Right-sizing is a natural baseline for energy-aware selection. It directly tests whether the proposed method does more than simply avoiding overpowered GPUs.

**Expected behavior:**  
This baseline should have low over-provisioning, but it may be fragile if the estimated requirement vector is noisy or optimistic.

**Recommended tie-break:**  
If two GPUs have similar margins, choose the lower TDP, then lower PSU.

---

### Baseline 7: Safety-factor Perf/TDP

**Selection rule:** Select the GPU with highest `perf_score / tdp_w`, but only among GPUs that exceed the requirement by a safety factor.

```text
candidate_gpus = feasible GPUs where perf_score >= alpha * game_required_perf_score
selected_gpu = argmax_gpu perf_score / tdp_w
```

Recommended values:

```text
alpha ∈ {1.00, 1.05, 1.10, 1.15}
```

Default:

```text
alpha = 1.10
```

**Why we do it:**  
Game requirement data is noisy and incomplete. A GPU that barely meets the requirement may not be robust. The safety factor creates a more realistic efficiency baseline.

**Why it is defensible:**  
Energy-aware systems often optimize power while satisfying a quality or performance constraint. This baseline follows that logic: reduce power while maintaining a margin above required performance.

**Expected behavior:**  
This should be more conservative than pure Perf/TDP and less overpowered than highest-performance feasible.

---

### Baseline 8: Pareto-knee feasible

**Selection rule:** First compute the Pareto frontier among feasible GPUs using:

```text
maximize perf_score
minimize tdp_w
minimize psu_w
```

A GPU is dominated if another feasible GPU has:

```text
higher or equal performance
lower or equal TDP
lower or equal PSU
and is strictly better in at least one dimension
```

After removing dominated GPUs, select the GPU closest to the ideal point:

```text
ideal = (max performance, min TDP, min PSU)
selected_gpu = Pareto GPU closest to ideal after normalization
```

**Why we do it:**  
This is the cleanest multi-objective baseline. It avoids selecting GPUs that are clearly worse than another option.

**Why it is defensible:**  
Pareto-front methods are standard for multi-objective optimization. Recent recommender-system work also uses Pareto-front approximation to handle multiple objectives.

**Expected behavior:**  
This baseline should produce balanced selections and is likely one of the strongest comparisons for the proposed method.

---

### Baseline 9: Weighted-sum utility

**Selection rule:** Normalize performance, TDP, PSU, and optionally over-provisioning. Then choose the GPU with maximum utility.

Example:

```text
score =
  1.00 * z(perf_score)
- 1.00 * z(tdp_w)
- 0.50 * z(psu_w)
- 0.50 * z(overprovisioning)
```

Alternative simpler version:

```text
score = z(perf_score) - lambda_tdp * z(tdp_w) - lambda_psu * z(psu_w)
```

**Why we do it:**  
This is a scalarized multi-objective baseline.

**Why it is defensible:**  
Scalarization is one of the simplest and most common ways to handle multi-objective optimization and ranking. It is easy to explain and easy to reproduce.

**Expected behavior:**  
This should be a strong balanced baseline, but the result depends on weight choices.

**Recommended weight policy:**  
Use fixed weights for the main results, and optionally include a sensitivity plot over weights.

---

### Baseline 10: KNN retrieval baseline

**Selection rule:** Use the existing KNN mode, if available.

For each game:

```text
1. Represent the game as a normalized requirement vector.
2. Represent each feasible GPU as a normalized GPU vector.
3. Select the feasible GPU nearest to the game vector.
4. Tie-break by lower TDP or higher Perf/TDP.
```

**Why we do it:**  
This tests a retrieval-style recommender instead of a hand-written ranking rule.

**Why it is defensible:**  
Nearest-neighbor retrieval is a simple, interpretable recommendation baseline. Since the current project already supports KNN mode, it should be evaluated explicitly.

**Expected behavior:**  
KNN may perform well on requirement matching, but may not optimize power unless the distance function or tie-break includes power.

---

## Recommended final baseline suite

For the final report, use this order:

| Method | Category | Keep/Add | Main purpose |
|---|---|---:|---|
| Random feasible | Sanity | Keep | Lower-bound reference |
| Lowest TDP feasible | Energy | Keep | Pure TDP minimization |
| Lowest PSU feasible | Build constraint | Keep | System power/capacity minimization |
| Highest performance feasible | Performance | Keep | Performance-first upper anchor |
| Perf/TDP | Efficiency | Keep | Standard efficiency baseline |
| Smallest-margin feasible | Right-sizing | Add | Avoid over-provisioning |
| Safety-factor Perf/TDP | Robust efficiency | Add | Efficient but not barely feasible |
| Pareto-knee feasible | Multi-objective | Add | Balanced Pareto trade-off |
| Weighted-sum utility | Multi-objective | Add | Scalarized ranking comparator |
| KNN retrieval | ML/retrieval | Add if implemented | Similarity-based recommendation |
| Proposed recommender | Proposed | Existing | Final method |

If time is short, implement only these eight:

1. Random feasible  
2. Lowest TDP feasible  
3. Lowest PSU feasible  
4. Highest performance feasible  
5. Perf/TDP  
6. Smallest-margin feasible  
7. Pareto-knee feasible  
8. Proposed recommender  

If time allows, add Safety-factor Perf/TDP, Weighted-sum utility, and KNN.

---

## Evaluation metrics

For each game and each method, compute one selected GPU. Then compute the following metrics.

### Metric 1: Coverage

```text
coverage = number of games with valid recommendation / total number of games
```

**Higher is better.**

This is important if any method can fail to return a GPU, especially safety-factor baselines.

---

### Metric 2: Average selected TDP

```text
avg_selected_tdp = mean(selected_gpu_tdp_w)
```

**Lower is better.**

This measures GPU-side power demand.

---

### Metric 3: Average selected PSU

```text
avg_selected_psu = mean(selected_gpu_psu_w)
```

**Lower is better.**

This measures build-level power/capacity burden.

---

### Metric 4: Performance per watt proxy

```text
ppw = selected_gpu_perf_score / selected_gpu_tdp_w
```

**Higher is better.**

This is the main efficiency metric.

---

### Metric 5: Performance over-provisioning

Absolute version:

```text
overprov_abs = selected_gpu_perf_score - game_required_perf_score
```

Relative version:

```text
overprov_rel = selected_gpu_perf_score / game_required_perf_score - 1
```

**Lower is better, assuming feasibility is satisfied.**

This measures how much stronger the selected GPU is than needed.

---

### Metric 6: Efficiency regret

For each game, define the best possible efficiency among feasible GPUs:

```text
best_ppw = max(perf_score / tdp_w among feasible GPUs)
```

Then compute:

```text
efficiency_regret_abs = best_ppw - selected_ppw
```

Relative version:

```text
efficiency_regret_rel = (best_ppw - selected_ppw) / best_ppw
```

**Lower is better.**

This metric asks:

> How far is this method from the best efficiency choice available for the same game?

---

### Metric 7: Feature slack

If the project has soft features such as bandwidth, texture rate, pixel rate, TMUs, and ROPs, compute average relative slack:

```text
feature_slack_mean =
mean over soft features:
max(0, (gpu_feature - game_required_feature) / game_required_feature)
```

**Lower is better, assuming feasibility is satisfied.**

This is useful because over-provisioning may happen in features other than the aggregate `perf_score`.

---

## Main result table

The final report should include one main table like this:

| Method | Coverage ↑ | Avg TDP ↓ | Avg PSU ↓ | Perf/TDP ↑ | Overprov. ↓ | Eff. Regret ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Random feasible |  |  |  |  |  |  |
| Lowest TDP feasible |  |  |  |  |  |  |
| Lowest PSU feasible |  |  |  |  |  |  |
| Highest performance feasible |  |  |  |  |  |  |
| Perf/TDP |  |  |  |  |  |  |
| Smallest-margin feasible |  |  |  |  |  |  |
| Pareto-knee feasible |  |  |  |  |  |  |
| Weighted-sum utility |  |  |  |  |  |  |
| KNN retrieval |  |  |  |  |  |  |
| Proposed recommender |  |  |  |  |  |  |

Recommended reporting:

- Mean
- Median
- 90th percentile for regret and over-provisioning
- Standard deviation for random feasible
- 95% bootstrap confidence intervals if time allows

---

## Recommended plots

### Plot 1: Average selected TDP by method

**Purpose:** Show how much power demand each strategy creates.

Expected pattern:

- Highest-performance feasible likely has high TDP.
- Lowest-TDP feasible should be lowest.
- Proposed method should be competitive without being too fragile.

---

### Plot 2: Performance-per-watt by method

**Purpose:** Show efficiency.

This should be one of the main poster figures.

---

### Plot 3: Over-provisioning by method

**Purpose:** Show whether a method recommends unnecessarily powerful GPUs.

Expected pattern:

- Highest-performance feasible likely over-provisions heavily.
- Smallest-margin feasible likely has lowest over-provisioning.
- Proposed method should balance low over-provisioning with good efficiency.

---

### Plot 4: Efficiency regret by method

**Purpose:** Show how close each method is to the best possible efficient feasible GPU.

This is probably the cleanest “summary” plot.

---

### Plot 5: Pareto scatter

Plot selected GPUs as:

```text
x-axis = TDP
y-axis = perf_score
```

Optionally show the Pareto frontier.

**Purpose:** Visually show which methods pick dominated or non-dominated GPUs.

---

### Plot 6: Sensitivity plot

For safety-factor baseline:

```text
x-axis = alpha
y-axis = coverage, PPW, or regret
```

For weighted-sum baseline:

```text
x-axis = lambda_tdp
y-axis = regret or over-provisioning
```

**Purpose:** Show the conclusions are not due to one arbitrary hyperparameter.

---

## Execution steps

### Step 1: Load cleaned data

Inputs:

```text
data/vectors/game_vectors_min.csv
data/vectors/game_vectors_recom.csv
data/vectors/gpu_power_vectors.csv
```

Output:

```text
data/results/eval_inputs_summary.csv
```

Tasks:

1. Load game vectors.
2. Load GPU vectors.
3. Confirm required columns exist.
4. Remove rows with missing critical values.
5. Keep only desktop discrete GPUs, if not already enforced.

---

### Step 2: Define shared feasibility function

Create one function used by every method:

```python
def get_feasible_gpus(game_row, gpu_df):
    # Hard constraints only
    # VRAM, DirectX, desktop discrete scope
    return feasible_gpu_df
```

Output:

```text
data/results/feasible_set_stats.csv
```

Include:

- number of feasible GPUs per game
- min/median/max feasible set size
- number of games with zero feasible GPUs

Why this matters:

> It shows whether the evaluation is meaningful and whether methods are compared on the same candidate set.

---

### Step 3: Implement baseline selectors

Create:

```text
src/evaluation/baselines.py
```

Suggested functions:

```python
select_random_feasible(game, feasible_gpus, seed)
select_lowest_tdp(game, feasible_gpus)
select_lowest_psu(game, feasible_gpus)
select_highest_perf(game, feasible_gpus)
select_perf_per_tdp(game, feasible_gpus)
select_smallest_margin(game, feasible_gpus)
select_safety_factor_ppw(game, feasible_gpus, alpha=1.10)
select_pareto_knee(game, feasible_gpus)
select_weighted_sum(game, feasible_gpus, weights)
select_knn(game, feasible_gpus)
select_proposed(game, feasible_gpus)
```

Output:

```text
data/results/baseline_recommendations.csv
```

Each row should include:

```text
game_name
track
method
selected_gpu
selected_tdp_w
selected_psu_w
selected_perf_score
feasible_set_size
```

---

### Step 4: Compute evaluation metrics

Create:

```text
src/evaluation/evaluate_baselines.py
```

Output:

```text
data/results/baseline_metrics_per_game.csv
data/results/baseline_metrics_summary.csv
```

Per-game metrics:

```text
coverage
selected_tdp
selected_psu
selected_ppw
overprov_abs
overprov_rel
efficiency_regret_abs
efficiency_regret_rel
feature_slack_mean
```

Aggregate metrics:

```text
mean
median
std
p90
95% bootstrap CI
```

---

### Step 5: Run subgroup evaluation

Recommended subgroups:

1. Minimum requirements vs recommended requirements
2. Low, medium, high game requirement difficulty
3. VRAM requirement bins
4. DirectX generation, if available

Output:

```text
data/results/subgroup_metrics_summary.csv
```

Why this matters:

> A method may look good on average but fail on harder games. Subgroup results make the evaluation more honest.

---

### Step 6: Generate plots

Create:

```text
src/evaluation/plot_results.py
```

Outputs:

```text
figures/avg_tdp_by_method.png
figures/avg_psu_by_method.png
figures/perf_per_watt_by_method.png
figures/overprovisioning_by_method.png
figures/efficiency_regret_by_method.png
figures/pareto_scatter.png
figures/sensitivity_alpha.png
```

Use high resolution:

```python
plt.savefig("figures/name.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/name.pdf", bbox_inches="tight")
```

---

### Step 7: Add optional external benchmark sanity check

Optional but useful if time allows.

Goal:

> Check whether internal `perf_score` correlates with an external benchmark score.

Possible external sources:

- PassMark G3D Mark
- 3DMark Time Spy graphics score
- Tom’s Hardware GPU hierarchy
- TechPowerUp relative performance

Output:

```text
data/results/external_benchmark_match.csv
figures/internal_perf_vs_external_benchmark.png
```

Metric:

```text
Spearman correlation between internal perf_score and external benchmark score
```

Why this matters:

> It validates that the project’s hand-designed `perf_score` is at least directionally aligned with external GPU performance rankings.

This is not required, but it would make the report stronger.

---

## Report structure for Task 4

### Section: Baselines

Write:

> We evaluate the proposed recommender against a suite of feasible-selection baselines. All methods operate on the same hard-feasible GPU set for each game, ensuring that differences arise from the ranking strategy rather than from different compatibility filters. The baselines include power-minimizing, performance-maximizing, efficiency-oriented, right-sizing, retrieval-style, and multi-objective ranking methods.

Then include the baseline table.

---

### Section: Metrics

Write:

> Since direct per-game FPS and runtime energy measurements are unavailable, we evaluate recommendation quality using proxy metrics derived from cleaned GPU specifications and game requirement vectors. We report selected TDP and PSU as power-related proxies, performance-per-TDP as an efficiency proxy, over-provisioning relative to the game requirement, and efficiency regret relative to the best feasible efficiency choice.

This is important because it is honest and defensible.

---

### Section: Results

Use the main result table and plots.

Suggested analysis language:

> The highest-performance baseline selects powerful GPUs but substantially increases TDP, PSU requirement, and over-provisioning. Lowest-TDP and lowest-PSU baselines reduce power demand but may provide minimal performance headroom. Perf/TDP is a strong efficiency baseline, while smallest-margin feasible is a strong right-sizing baseline. The proposed recommender is evaluated by whether it achieves competitive efficiency while reducing unnecessary over-provisioning and maintaining full coverage.

---

### Section: Limitations

Include:

> This evaluation uses specification-derived proxy metrics rather than measured FPS or in-game power. TDP is not identical to runtime power consumption, and vendor PSU recommendations depend on assumptions about the full system configuration. Therefore, results should be interpreted as energy-aware recommendation quality under available structured hardware specifications, not as direct measurement of real game energy use.

This limitation makes the project look more honest, not weaker.

---

## Poster structure

Use four blocks:

### Block 1: Problem

```text
Goal: Recommend a feasible, power-efficient GPU for each game.
Challenge: Fastest GPU wastes power; lowest-power GPU may barely meet requirements.
```

### Block 2: Method

Show pipeline:

```text
Game requirements → hard feasibility filters → candidate GPUs → ranking method → selected GPU
```

### Block 3: Baselines and metrics

List:

```text
Baselines: Random, Lowest TDP, Lowest PSU, Highest Perf, Perf/TDP, Smallest Margin, Pareto Knee, Proposed
Metrics: TDP, PSU, Perf/TDP, Over-provisioning, Efficiency Regret
```

### Block 4: Results

Use:

1. One main result table
2. One efficiency-regret plot
3. One over-provisioning plot
4. One short takeaway

Example takeaway:

> The proposed method reduces unnecessary over-provisioning while staying competitive with performance-per-watt baselines.

---

## Why this plan is defensible

This plan is defensible for four reasons.

First, every method is compared on the same hard-feasible GPU set. This avoids unfair comparisons.

Second, the baselines cover meaningful extremes:

- random selection
- lowest power
- lowest system PSU
- highest performance
- best efficiency ratio
- right-sizing
- multi-objective trade-off

Third, the metrics align with the project goal. The project is not evaluated only on performance. It is evaluated on the trade-off between performance, power, and over-provisioning.

Fourth, the limitations are clearly stated. The project does not claim to measure actual FPS or real in-game energy. It evaluates structured, spec-based recommendation quality.

---

## Recommended citations and how to use them

### Multi-objective recommendation and ranking

Use these to justify Pareto-knee and weighted-sum baselines.

1. Timo Wilm, Philipp Normann, Felix Stepprath. **Pareto Front Approximation for Multi-Objective Session-Based Recommender Systems.** arXiv, 2024.  
   URL: https://arxiv.org/abs/2407.16828  
   Use for: Pareto-front recommendation baseline and multi-objective recommendation framing.

2. Chongming Gao et al. **A Survey of Multi-Objective Recommender Systems.** arXiv, 2023.  
   URL: https://arxiv.org/abs/2307.04923  
   Use for: General claim that recommender systems often optimize multiple competing objectives.

3. Ali Jadbabaie, Devavrat Shah, Sean R. Sinclair. **Multi-Objective LQR with Linear Scalarization.** arXiv, 2024.  
   URL: https://arxiv.org/abs/2408.04488  
   Use for: Scalarization as a standard multi-objective optimization approach.

---

### Hardware efficiency and power-aware evaluation

Use these to justify TDP, PSU, and performance-per-watt metrics.

4. Dan Zhao et al. **Sustainable Supercomputing for AI: GPU Power Capping at HPC Scale.** arXiv, 2024.  
   URL: https://arxiv.org/abs/2402.18593  
   Use for: Power-aware GPU evaluation and energy-performance trade-offs.

5. Maria Patrou et al. **Power-Capping Metric Evaluation for Improving Energy Efficiency in HPC Applications.** arXiv, 2025.  
   URL: https://arxiv.org/abs/2505.21758  
   Use for: Multi-objective energy-performance metrics and power-capping evaluation.

6. Performance per watt overview.  
   URL: https://en.wikipedia.org/wiki/Performance_per_watt  
   Use for: Basic definition of performance-per-watt as a hardware efficiency metric. Prefer a more formal source if your professor expects academic-only citations, but this is useful for quick justification.

---

### GPU benchmark and external validation sources

Use these to justify external sanity checks and benchmark-proxy framing.

7. UL Solutions. **3DMark benchmark.**  
   URL: https://benchmarks.ul.com/3dmark  
   Use for: Standardized GPU benchmarking and graphics performance comparison.

8. UL Solutions Support. **Estimating game performance from 3DMark scores.**  
   URL: https://support.benchmarks.ul.com/support/solutions/articles/44002125238-estimating-game-performance-from-3dmark-scores  
   Use for: Connecting benchmark scores to game performance estimates.

9. PassMark Software. **PerformanceTest and video card benchmarks.**  
   URL: https://www.passmark.com/products/performancetest/  
   Use for: External GPU benchmark score source.

10. PassMark Software. **High End Video Card Chart.**  
    URL: https://www.videocardbenchmark.net/high_end_gpus.html  
    Use for: External GPU performance ranking source.

11. Tom’s Hardware. **Graphics Card Power Consumption Tested.**  
    URL: https://www.tomshardware.com/features/graphics-card-power-consumption-tested  
    Use for: GPU power comparison and the difference between specs and measured power.

---

### Evaluation methodology

Use these if you include statistical comparison or split justification.

12. Maurizio Ferrari Dacrema, Paolo Cremonesi, Dietmar Jannach. **Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches.** RecSys, 2019.  
    URL: https://dl.acm.org/doi/10.1145/3298689.3347058  
    Use for: Careful recommender evaluation and strong baseline comparison.

13. Joeran Beel et al. **Offline Recommender-System Evaluation: Some Lessons Learned.** arXiv, 2020.  
    URL: https://arxiv.org/abs/2010.11060  
    Use for: Offline recommendation evaluation design and pitfalls.

14. Janez Demšar. **Statistical Comparisons of Classifiers over Multiple Data Sets.** JMLR, 2006.  
    URL: https://www.jmlr.org/papers/volume7/demsar06a/demsar06a.pdf  
    Use for: Friedman tests, average ranks, and critical-difference diagrams.

---

## Minimal implementation checklist

Use this as the Task 4 checklist.

```text
[ ] Implement shared hard-feasible GPU function
[ ] Save feasible set statistics
[ ] Implement Random feasible
[ ] Implement Lowest TDP feasible
[ ] Implement Lowest PSU feasible
[ ] Implement Highest performance feasible
[ ] Implement Perf/TDP
[ ] Implement Smallest-margin feasible
[ ] Implement Pareto-knee feasible
[ ] Optional: Implement Safety-factor Perf/TDP
[ ] Optional: Implement Weighted-sum utility
[ ] Optional: Add KNN baseline from existing recommender mode
[ ] Run all methods on min game vectors
[ ] Run all methods on recommended game vectors
[ ] Save per-game selected GPU outputs
[ ] Compute coverage, TDP, PSU, PPW, over-provisioning, regret
[ ] Create main result table
[ ] Create subgroup result table
[ ] Create TDP, PSU, PPW, over-provisioning, regret plots
[ ] Optional: external benchmark sanity check
[ ] Write report section
[ ] Build poster result block
```

---

## Final recommendation

For the strongest and most realistic Task 4, implement this exact set:

```text
1. Random feasible
2. Lowest TDP feasible
3. Lowest PSU feasible
4. Highest performance feasible
5. Perf/TDP
6. Smallest-margin feasible
7. Pareto-knee feasible
8. Proposed recommender
```

Then add these only if time permits:

```text
9. Safety-factor Perf/TDP
10. Weighted-sum utility
11. KNN retrieval
```

This is enough to make the task look rigorous, fair, and defensible for an ECE 228 project.