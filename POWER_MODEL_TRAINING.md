# Power Model Training: Plan and Design

This document describes how we train the TDP and PSU prediction models that turn cleaned GPU specifications into power estimates for the recommender. It is the bridge between `DATA_CLEANING.md` (which produces `data/vectors/gpu_power_vectors.csv`) and the actual training code in `src/train_gpu_specs_models.py`. The training step has three goals that pull in slightly different directions, so the design here is explicit about which model serves which goal.

## What we are training

Two regression targets:

- **`tdp_w`** — GPU board thermal design point in watts. Primary target. Required by the recommender's performance-per-watt ranking.
- **`psu_w`** — vendor-recommended total system PSU in watts. Secondary target. Includes non-GPU components plus a safety margin, so it is a system-level provisioning number, not a GPU-only power figure.

Input is `data/vectors/gpu_power_vectors.csv` (1,441 rows × 136 columns). The training subset is the 1,117 rows where both `tdp_w` and `psu_w` are populated. The remaining 324 rows (predominantly pre-2006 cards with no published TDP) are **prediction-only** — the model fits on the 1,117 and predicts for all 1,441. This is the Option 3 design from `DATA_CLEANING.md`.

## Model lineup

Ten models in total: seven point-prediction models we already had (some with feature-set changes), plus three new uncertainty-quantifying models.

| # | Model | Family | Target | Role |
|---|---|---|---|---|
| 1 | Linear Regression | linear | point | baseline |
| 2 | Ridge Regression | linear | point | regularized baseline |
| 3 | Lasso Regression | linear | point | sparsity baseline |
| 4 | Random Forest | tree | point | nonlinear baseline |
| 5 | Gradient Boosting | tree | point | strong tabular |
| 6 | XGBoost | tree | point | best point estimator |
| 7 | MLP (PyTorch) | neural | point | dl baseline |
| 8 | **Bayesian Ridge** | linear | point + σ | UQ, ~10 LOC |
| 9 | **Quantile XGBoost** | tree | quantile (5%, 50%, 95%) | UQ, best ROI |
| 10 | **Gaussian Process** | kernel | point + σ | UQ, course-aligned |

Each model is trained twice, once for each target (tdp_w, psu_w), giving 20 fitted models total.

## Feature set per model

The vector CSV exposes 136 columns. Not every model uses every column — feature subsets are picked to match each model's NaN tolerance and computational budget.

**Tiered columns (recap):**

- **Tier 1 (0% NaN, 9 cols):** `process_nm, memory_speed_mhz, memory_mb, memory_bandwidth_gbs, tmus, rops, pixel_rate, texture_rate, direct_x`
- **Tier 2 (1–25% NaN, 7 cols):** `transistors_m, die_size_mm2, density_kmm2, memory_bus_bits, release_year, shading_units, fp32_gflops`
- **Tier 3 (35–90% NaN, 5 cols):** `gpu_clock_mhz, base_clock_mhz, boost_clock_mhz, tensor_cores, rt_cores`
- **Categorical one-hot:** `memory_type_*` (~23 cols), `architecture_*` (~66 cols)

**Per-model assignments:**

| Model class | Numeric features | Categorical | NaN strategy | Count |
|---|---|---|---|---|
| Linear / Ridge / Lasso / MLP | Tier 1 + Tier 2 + `boost_clock_mhz` | memory_type + architecture | use `standard_*` columns (median-imputed in vector build) | ~106 |
| Random Forest / Gradient Boosting | Tier 1 + Tier 2 + `boost_clock_mhz` (raw) | memory_type + architecture | pre-impute NaN with column median before fit | ~106 |
| **Bayesian Ridge** | Tier 1 + Tier 2 + `boost_clock_mhz` (standardized) + 8 missing indicators | memory_type + architecture | median-impute + add `is_missing_<col>` indicators so σ reflects imputation uncertainty | ~114 |
| **XGBoost (point)** | Tier 1 + Tier 2 + **all of Tier 3** (raw) | memory_type + architecture | XGBoost native NaN handling — no imputation | ~110 |
| **Quantile XGBoost** | A/B between (A) native NaN and (B) median-impute + missing indicators | memory_type + architecture | tested both ways; kept variant with better validation MAE + calibration | ~110 or ~115 |
| **Gaussian Process** | Tier 1 + Tier 2 + `boost_clock_mhz` (standardized) + 8 missing indicators | (skip — sparse one-hots hurt RBF kernel) | median-impute + missing indicators | 25 |

**Why XGBoost (point) gets the heavy-NaN columns and others don't:** XGBoost learns a default direction for missing values at each tree split, so it can extract signal from columns like `boost_clock_mhz` (65% NaN), `tensor_cores` (90% NaN), and `rt_cores` (85% NaN) without imputation. Imputing these columns for non-XGBoost models would inject a large amount of synthetic data (median-filling 85% of `rt_cores`, for example, replaces real "no RT cores on this card" signal with "this card has the median RT core count," which is wrong for most cards). So we deliberately give XGBoost a richer feature set and accept that this couples the headline point-prediction result more tightly to XGBoost than to the linear or RF/GB models.

**Why Bayesian Ridge and GP use median imputation + missing indicators:** missing-value status is itself a signal that the UQ models should learn from — a card with NaN `boost_clock_mhz` is genuinely more uncertain than one with a measured value. For every Tier 2 column and `boost_clock_mhz` that gets median-imputed, we add a binary `is_missing_<col>` indicator. The model then learns two things in one shot: (a) the typical feature → TDP relationship, and (b) how much extra posterior variance to assign to rows where that feature was imputed. The GP's posterior σ becomes a function of both location uncertainty in feature space *and* imputation uncertainty. Bayesian Ridge captures it differently (through the weight posterior over the indicator columns), but the effect is the same.

**Why Tier 3 (other than `boost_clock_mhz`) is still excluded from Bayesian Ridge and GP:** at 65–90% NaN, the indicator column becomes the dominant signal and the imputed median is mostly noise. Net negative information for these models. XGBoost can still extract value here via its native NaN handling, which is why XGBoost keeps them.

**Why GP skips the categorical one-hots:** sparse binary features (memory_type 23-way, architecture 66-way) actively hurt RBF kernel distances — every pair of rows that share no category differs by exactly √2 on those columns, which dominates the kernel and washes out the continuous-feature signal. A learned low-dimensional embedding (PCA, autoencoder) would fix this, but adding it is more engineering than the present pass warrants. Logged as a future improvement.

**Quantile XGBoost A/B test on NaN strategy:** Quantile XGBoost is fundamentally a tree model, so it benefits from XGBoost's native NaN routing. But the missing-indicator trick is also well-defined for it. Rather than guess which works better at this dataset size, we train both variants and pick the one with lower validation MAE on the median quantile *and* better empirical coverage of the 90% prediction interval. The selection criterion: (a) coverage must be within ±5% of the nominal 90%, and (b) among variants meeting (a), pick the lower median-MAE. Decision is recorded in the metrics CSV and called out in the writeup.

## Train / val / test split

- 70% train / 10% validation / 20% test, random seed 42 (matches prior pipeline).
- The split is taken from the 1,117 training-eligible rows. The 324 prediction-only rows are never in any split — they are only ever inference targets.
- **Validation set** drives all model-selection decisions:
  - Hyperparameter grid search per model (validation MAE).
  - Quantile XGBoost A/B (validation coverage in [85%, 95%], then validation MAE on the median).
- **Test set** is used only for the final reported numbers: MAE / RMSE / R² in the metrics CSVs, and the calibration / coverage / σ-distribution plots that go into the writeup. Touched exactly once per model per target.
- σ percentile cut-offs for the confidence flag are computed on the **training fold**, not validation or test, so they reflect the distribution the model was actually fitted to.
- **Known limitation:** the split is pure random, not stratified by TDP quartile. With n=1,117 the failure mode is unlikely but the proposal flagged stratification as a follow-up improvement. Not blocking submission; will be addressed in a separate pass.

## NaN handling — summary

| Source of NaN | Handled where | How |
|---|---|---|
| Raw spec missing (older cards) | cleaner | parser returns `NaN` |
| Sentinel string (`"unknown"`, etc.) | cleaner | parser maps to `NaN` |
| Out-of-range value | cleaner | coerced to `NaN`, row logged |
| NaN in `tdp_w` or `psu_w` | trainer | row excluded from train/val/test split |
| NaN in Tier 1/2 features (for linear/MLP) | vector builder | median-imputed *before* standardization |
| NaN in Tier 1/2 features (for RF/GB) | trainer | pre-imputed with column median before fit |
| NaN in Tier 3 features (XGBoost only) | trainer | passed as-is; XGBoost routes them at split time |
| NaN in one-hot columns | n/a | one-hot encoder treats `NaN` as `"unknown"` bucket |

There are no legitimate zero values in the GPU dataset — anything that came in as `0` or `"0"` was caught by the cleaner's sentinel set. So zero-handling is not a separate concern in training.

## Hyperparameter tuning

Manual grid search over a small per-model dict, validation MAE as the selection criterion. Keeping the same grids as the prior pipeline for the seven point models:

- **Ridge / Lasso:** `alpha ∈ {0.0001, 0.001, 0.01, 0.1, 0.5, 0.75, 1.0, 5.0, 10.0, 100.0}` (subset per model)
- **Random Forest:** `n_estimators ∈ {300, 500, 800}`, `max_depth ∈ {None, 10, 20, 30}`, `min_samples_leaf ∈ {1, 2, 4}`, `max_features ∈ {"sqrt", 0.7, 1.0}`
- **Gradient Boosting:** `n_estimators ∈ {300, 500, 800}`, `learning_rate ∈ {0.03, 0.05, 0.075, 0.1}`, `max_depth ∈ {3, 4, 5, 6}`, `min_samples_leaf ∈ {1, 2, 5}`
- **XGBoost (and Quantile XGBoost per quantile):** `n_estimators ∈ {300, 500, 800}`, `learning_rate ∈ {0.03, 0.05, 0.075, 0.1}`, `max_depth ∈ {4, 5, 6}`, `min_child_weight ∈ {1, 3, 5}`, `reg_lambda ∈ {1.0, 5.0, 10.0}`
- **MLP:** `hidden_layers ∈ {(32,16), (64,32), (128,64), (256,128)}`, `learning_rate ∈ {1e-4, 1e-3, 5e-3}`, `weight_decay ∈ {1e-4, 1e-3}`, `dropout ∈ {0.0, 0.1, 0.2}`, `epochs = 1000`
- **Bayesian Ridge:** sklearn default priors (`alpha_1=alpha_2=lambda_1=lambda_2=1e-6`) — no tuning needed.
- **Gaussian Process:** kernel = `ConstantKernel * RBF + WhiteKernel`, optimizer auto-fits length scale and noise floor by max-likelihood. Restart optimizer 5 times to escape local minima.

## The three UQ models

### 8. Bayesian Ridge Regression

`sklearn.linear_model.BayesianRidge` — drop-in replacement for `Ridge`. After fitting, calling `predict(X, return_std=True)` returns both the mean prediction and its posterior standard deviation. The σ comes from the Gaussian posterior over the weight vector; it captures the model's uncertainty about *where the regression line is*, given the data.

Feature set: Tier 1 + Tier 2 + `boost_clock_mhz` (standardized via `standard_*` columns) + 8 missing-value indicator columns (one each for the 7 Tier 2 features plus `boost_clock_mhz`). Categorical one-hots from memory_type and architecture are also included. Total: ~114 features.

The missing indicators give Bayesian Ridge a way to propagate imputation uncertainty into σ — rows where features were imputed get nonzero coefficients on their `is_missing_*` columns, and the posterior over those coefficients widens the prediction σ accordingly.

Expected behavior: lower σ in dense regions of feature space (mid-range GPUs with all features measured), higher σ near the edges (very low or very high TDP, or cards with several imputed feature values). Cheapest possible UQ — ~10 lines of code beyond the existing Ridge implementation.

### 9. Quantile XGBoost

Three XGBoost models trained at three quantiles using `objective="reg:quantileerror"` and `quantile_alpha = 0.05, 0.5, 0.95`. At inference, the three give a (lower, median, upper) tuple. The prediction interval `[lower, upper]` is a direct 90% empirical estimate — no Gaussian assumption.

**A/B on the NaN strategy.** We train Quantile XGBoost twice and keep the better variant:
- **Variant A — native NaN:** Tier 1 + Tier 2 + all Tier 3 (raw, no imputation), plus categorical one-hots. XGBoost routes NaN at split time as for the point model.
- **Variant B — median impute + missing indicators:** same numeric feature list, but NaN-filled with column median and accompanied by `is_missing_<col>` indicator columns for every column that had any NaN. Same categorical one-hots.

Selection rule: empirical 90% coverage on the validation set must fall in [85%, 95%], and among variants that meet this, the one with lower validation MAE on the median quantile wins. If neither variant meets the coverage band, the better-coverage variant is kept and the failure is flagged in the metrics CSV. The chosen variant per target is recorded as a row in `tdp_model_metrics.csv` / `psu_model_metrics.csv`.

This is the strongest UQ for this dataset because:
- XGBoost is already our best point estimator, so the median quantile is a strong point prediction on its own.
- Quantile loss is asymmetric; the 5% and 95% models learn the *actual shape* of the TDP error distribution, which is heavy-tailed (a 500 W card is rare and may have very wide intervals).
- Heteroscedastic by construction: different rows get different interval widths based on what XGBoost learned at their location in feature space.

Three separate fits per variant × two variants = six XGBoost fits per target — still seconds at this dataset size.

### 10. Gaussian Process

`sklearn.gaussian_process.GaussianProcessRegressor` with `ConstantKernel * RBF + WhiteKernel`. The RBF kernel is anisotropic (ARD) — one length scale per feature, learned by marginal-likelihood maximization alongside the noise floor and signal variance. Predictions return both posterior mean and posterior standard deviation with proper Bayesian interpretation.

Feature set: Tier 1 (9 cols) + Tier 2 (7 cols) + `boost_clock_mhz` (1 col) + 8 missing-value indicators = **25 features**, all standardized. Categorical one-hots are excluded because sparse binary features confuse RBF distance computation — every pair of rows that share no architecture differs by √2 on those columns and washes out continuous-feature signal.

The missing indicators are the key piece that makes the GP a real UQ model rather than a smoothed regressor on imputed data:
- A row with `boost_clock_mhz` measured will, on the relevant kernel dimension, look distinct from a row with `boost_clock_mhz` imputed.
- The ARD length scale on `is_missing_boost_clock_mhz` will be learned from data — if the indicator carries no information, its length scale grows large and the column effectively drops out; if it does carry information (which it should, given that pre-3D cards systematically lack boost_clock), the GP downweights those rows' similarity to fully-observed rows.
- The resulting posterior σ is **higher for rows whose features were imputed**, which is the calibrated behavior we want.

Computational footprint: O(n³) training cost and O(n²) memory. At n=1,117 training rows, fit time is a few seconds and memory ~10 MB. ARD over 25 features is well-conditioned at this n; restart the optimizer 5 times to escape local minima in the kernel hyperparameter landscape.

The GP's headline value here is conceptual:
- Canonical UQ method in the course (May 21 GP lecture, May 28 UQ lecture).
- Posterior σ is calibrated under the GP modeling assumptions — principled credible intervals.
- Will likely lose on point-prediction MAE to XGBoost (fewer features, simpler smoothing), and that's expected. Its job is the σ, not the μ.

## What σ buys us — broader than the recommender

UQ outputs are useful in several places. Listing them here so the training output schema is designed to support all of them, not just the recommender:

1. **Recommendation explanations** — "we estimate this card uses 115–135 W (90% confidence)" is operationally more useful than "this card uses 120 W" for someone provisioning a PSU.
2. **Out-of-distribution detection** — a test card with σ much larger than the training-set median is likely an architecture or generation the model has never seen. The recommender can downrank or flag it.
3. **Anomaly / data-quality detection** — training rows where the actual `tdp_w` falls outside the model's ±2σ band are either label errors in the spec sheet or genuinely unusual cards. Either way, surfacing them is valuable data audit and is free once we have σ.
4. **Stratified evaluation** — report MAE separately for high-σ vs low-σ predictions. A well-calibrated model has low MAE where σ is small and higher MAE where σ is large.
5. **Active learning / data curation** — if we add more training data later, prefer the cards the model is most uncertain about.
6. **Risk-adjusted reporting in the writeup** — "we predict TDP to within ±X watts at 90% confidence on held-out cards" is a much stronger physical-applications claim than "MAE = 15 W."
7. **Recommender-level downstream uses** — e.g. ranking by a lower-confidence-bound perf-per-watt, or abstaining when every feasible candidate has high σ. These are designed in detail in a separate document, not here.

The point of this section is that σ is not just a recommender artifact — it carries scientific weight on its own. The training output schema below is designed so all of these uses are possible without re-running training.

## Output schema

The training script produces three CSVs and a set of figures. The schema is designed so every σ-related downstream use enumerated above is supported without re-running training.

```
data/results/
├── tdp_model_metrics.csv
│   columns: target, model, params, val_mae, test_mae, test_rmse, test_r2,
│            coverage_90 (UQ models only), mean_interval_width (UQ models only),
│            qxgb_variant (Quantile XGB only — "native_nan" or "impute_indicator"),
│            inference_latency_ms_per_row
│
├── psu_model_metrics.csv
│   same schema for the PSU target
│
├── gpu_power_predictions.csv  (1,441 rows — every cleaned GPU)
│   identity columns:
│     brand, name, actual_tdp_w, actual_psu_w
│
│   per-model point predictions (10 models × 2 targets = 20 columns):
│     pred_tdp_w_linear_regression, pred_tdp_w_ridge, pred_tdp_w_lasso,
│     pred_tdp_w_random_forest, pred_tdp_w_gradient_boosting, pred_tdp_w_xgboost,
│     pred_tdp_w_mlp, pred_tdp_w_bayesian_ridge, pred_tdp_w_quantile_xgb,
│     pred_tdp_w_gaussian_process
│     (and the parallel 10 pred_psu_w_* columns)
│
│   per-UQ-model σ (3 UQ models × 2 targets = 6 columns):
│     sigma_tdp_w_bayesian_ridge, sigma_tdp_w_quantile_xgb, sigma_tdp_w_gaussian_process
│     sigma_psu_w_bayesian_ridge, sigma_psu_w_quantile_xgb, sigma_psu_w_gaussian_process
│
│     IMPORTANT: for Quantile XGB the σ column is a Gaussian-EQUIVALENT scaled
│     from the (5%, 95%) quantile gap as (upper - lower) / (2 × 1.645). This is
│     a convenience so all three UQ models can be compared on the same numeric
│     scale and combined later (e.g. for an inverse-variance ensemble). It is
│     NOT the native Quantile XGB uncertainty representation — the canonical
│     representation is the [lower, upper] interval itself, which is what the
│     calibration / coverage plots use. The writeup must call this out.
│
│   prediction intervals (Quantile XGB only):
│     lower_tdp_w_quantile_xgb, upper_tdp_w_quantile_xgb   (5% and 95% bands)
│     lower_psu_w_quantile_xgb, upper_psu_w_quantile_xgb
│
│   confidence flags (one per UQ model per target = 6 columns):
│     confidence_tdp_w_bayesian_ridge       ∈ {"high", "medium", "low"}
│     confidence_tdp_w_quantile_xgb         ∈ {"high", "medium", "low"}
│     confidence_tdp_w_gaussian_process     ∈ {"high", "medium", "low"}
│     (and parallel for psu_w)
│     Thresholds: "high"   if σ ≤ 33rd percentile of training-fold σ for that model
│                 "medium" if 33rd < σ ≤ 67th percentile
│                 "low"    if σ > 67th percentile
│
│     CRITICAL: the 33rd / 67th σ percentiles are computed EXCLUSIVELY on the
│     training fold (not on val, not on test, not on the prediction-only rows).
│     The two cut-off values per (model × target) are saved to
│     data/results/confidence_thresholds.json so:
│       (a) val / test / prediction-only rows are flagged against fixed cut-offs,
│       (b) the cut-offs are reproducible if we re-run inference later, and
│       (c) downstream code (recommender) can read the same thresholds without
│           re-deriving them from the predictions CSV.
│
│   anomaly flags (per UQ model + consensus = 4 per target = 8 columns):
│     residual_outlier_tdp_w_bayesian_ridge      ∈ {0, 1}
│     residual_outlier_tdp_w_quantile_xgb        ∈ {0, 1}
│     residual_outlier_tdp_w_gaussian_process    ∈ {0, 1}
│     residual_outlier_tdp_w_consensus           ∈ {0, 1}   ← 1 iff ≥ 2 of the 3
│                                                              individual flags are 1
│     (and the parallel 4 psu_w columns)
│
│     Per-model rule:
│       Bayesian Ridge / GP:  1 if |actual − μ| > 2 · σ_model
│       Quantile XGB:         1 if actual is outside [lower_5%, upper_95%]
│
│     Computed only where the actual target is non-NaN (i.e. on the 1,117
│     training-eligible rows). Prediction-only rows have NaN here because there
│     is nothing to compare against. The consensus column is the most useful
│     for data-quality auditing — single-model flags can disagree, but rows
│     flagged by 2/3 of the models are strong candidates for label review.
│
└── best_power_model_summary.csv
    best model per target on:
      • test MAE (point-prediction winner)
      • coverage-MAE Pareto efficiency (UQ winner)

figures/
├── <model>_<target>_actual_vs_pred.png         # residual scatter per model+target (existing)
├── mlp_<target>_train_loss.png                 # MLP loss curves (existing)
│
├── calibration_<target>_<uq_model>.png         # NEW: required for all 3 UQ models × 2 targets = 6 plots.
│                                               # x-axis: predicted σ binned into 10 quantile bins
│                                               # y-axis: mean |actual − predicted| in each bin
│                                               # overlaid: y = x reference line (ideal calibration)
│                                               # if the points hug the line, σ is well-calibrated.
│
├── coverage_<target>_<uq_model>.png            # NEW: cumulative coverage curve.
│                                               # x-axis: nominal coverage level α ∈ [0.5, 0.99]
│                                               # y-axis: empirical fraction of test rows where
│                                               #         |actual − predicted| ≤ z_α · σ
│                                               # overlaid: y = x reference (ideal calibration)
│
└── sigma_distribution_<target>_<uq_model>.png  # NEW: histogram of σ on the training set with the
                                                #       33rd / 67th percentile cut-offs marked.
                                                #       Shows where the confidence-flag thresholds land.
```

The calibration figures (`calibration_*` and `coverage_*`) are **mandatory deliverables** — if we claim uncertainty quantification in the writeup, we must be able to show whether the uncertainty is reliable. A model can have a beautiful low-MAE point prediction and completely uncalibrated σ; both are useless for the recommender's downstream uses.

**Final-report selection.** All 18 plots (6 calibration + 6 coverage + 6 σ-distribution) are generated and saved as artifacts in `figures/`. The main writeup surfaces only the most informative subset — typically the best-calibrated UQ model's calibration + coverage pair per target (4 plots total) — and the remaining 14 stay as appendix / supplementary artifacts. The decision of which plots are surfaced is made *after* training, based on which model wins per target and which figure most clearly shows the calibration story.

## Evaluation metrics

Two layers of evaluation.

**Point-prediction accuracy (all 10 models):**
- MAE, RMSE, R² on the held-out test set
- Inference latency per row

**UQ-specific (Bayesian Ridge, Quantile XGBoost, GP):**
- **Empirical 90% coverage.** Fraction of test rows where the true `tdp_w` falls inside the predicted [μ − 1.645σ, μ + 1.645σ] interval (or [Q5, Q95] for Quantile XGB). Should be ≈ 0.90 if the model is well-calibrated. Reported in the metrics CSV; visualized in the coverage plot.
- **Mean interval width.** Average size of the prediction interval. Tighter is better, but only conditional on coverage holding — a model with 60% coverage and tiny intervals is not "efficient," it is broken.
- **Calibration plot** (`calibration_<target>_<model>.png`). Predicted σ binned into deciles vs. mean absolute residual in each bin. A well-calibrated model gives a roughly linear (y = x) relationship.
- **Coverage curve** (`coverage_<target>_<model>.png`). Empirical coverage vs. nominal coverage across α ∈ [0.5, 0.99]. The closer the curve hugs y = x, the better calibrated the σ is across all confidence levels — not just at 90%.
- **σ distribution** (`sigma_distribution_<target>_<model>.png`). Histogram of σ over the training set, with the 33rd / 67th percentile cuts marked. Shows the operating range of the confidence flag.

If a UQ model achieves 90% target coverage with much wider intervals than another, it is more honest but less useful. The combination of the calibration plot + interval width is the call: prefer the *narrowest* intervals that still hit the coverage target.

## Compute caveat

XGBoost requires `libomp` on macOS. The Framework Python 3.11 at `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3` has sklearn and torch but fails to load `libxgboost.dylib` because `/opt/homebrew/opt/libomp/lib/libomp.dylib` is missing. **Before running training, run `brew install libomp`** or training will error out at the XGBoost step.

The data cleaner and vector builder do not need xgboost or torch and run fine under the existing Python.

## Implementation plan

When you give the go-ahead, the work on `src/train_gpu_specs_models.py` is roughly:

1. **Update feature lists** (per-model assignments above). Adds `boost_clock_mhz` to the linear/RF/GB/MLP standardized feature list. Adds Tier 3 to XGBoost (point) only. Adds the `architecture_` one-hot prefix to the categorical detection.
2. **Build the missing-indicator helper.** A small utility that takes a list of columns and produces, for each column, a paired `is_missing_<col>` binary column. Used by Bayesian Ridge, GP, and the Variant B of Quantile XGBoost.
3. **Refactor to single-df flow.** Remove `load_prediction_vectors()`, the `pred_df` parameter, and the two-file CSV handling. The trainer reads the vector CSV once; the training step drops NaN-target rows internally; predictions are computed for every row including prediction-only ones.
4. **Add Bayesian Ridge model.** New `bayesian_ridge(...)` function. Median-imputes Tier 2 + `boost_clock_mhz`, adds missing indicators, fits `sklearn.linear_model.BayesianRidge`, returns mean and σ. σ stored in the predictions CSV.
5. **Add Quantile XGBoost (both variants).** Two paths trained per target:
   - Variant A: native NaN, three XGB fits at q ∈ {0.05, 0.5, 0.95}.
   - Variant B: median impute + missing indicators, three XGB fits at the same quantiles.
   Select per the rule above; record the selection.
6. **Add Gaussian Process.** New `gaussian_process(...)` function. Uses Tier 1 + Tier 2 + `boost_clock_mhz` + missing indicators (25 features), ARD-RBF + WhiteKernel, 5 optimizer restarts.
7. **Compute confidence flags.** Per UQ model per target, compute the 33rd / 67th σ percentiles on the training fold, save them, apply them to flag every row in the predictions CSV.
8. **Compute residual-outlier flags.** Per target, mark rows where `|actual − pred_gp| > 2 · sigma_gp` (only computable for the 1,117 training-eligible rows; NaN for prediction-only rows).
9. **Generate calibration plots** for the three UQ models × two targets = 6 figures, plus the coverage curves and σ distribution histograms.
10. **Update metrics CSVs** with coverage, interval width, QXGB variant choice, and inference latency.
11. **Run end-to-end** and verify outputs match the schema above.

(Ensemble combining the three UQ models via inverse-variance weighting is deferred — see Out of Scope.)

## Out of scope (deferred to future work)

- **All recommender-side uses of σ** — including but not limited to lower-confidence-bound perf-per-watt, abstention, and confidence-aware explanations. Designed in a separate document once this training pass is complete.
- **Inverse-variance-weighted ensemble** of the three UQ models. Optional / nice-to-have once Bayesian Ridge, Quantile XGB, and GP are individually working and their calibration plots have been inspected. Easy to add later — the predictions CSV already carries all the inputs the ensemble would need.
- **Stratified train/test split** by TDP quartile.
- **K-fold cross-validation** (proposed in `PROPOSAL_ksa.md` §3).
- **Log-transformed target** (proposed in `PROPOSAL_ksa.md` §3).
- **Train-fold-only standardization** — current vector builder standardizes over the full dataset, mild leakage, low-priority fix.
- **Architecture as a learned low-dim embedding** rather than 66-way one-hot. Would also let the GP use architecture info via the embedded representation.
- **MLP with MC Dropout** — would give a fourth UQ method. Possible future addition.
- **Conformal prediction wrapper** around any point-prediction model — would give distribution-free prediction intervals with formal coverage guarantees. Worth adding if the three principled UQ models all turn out to be poorly calibrated.
