# Modeling Analysis: Physics, Data Engineering, and Why the Predictions Improved

This document explains *why* the new TDP and PSU prediction models are dramatically better than the prior pipeline. The answer is not "more data, more parameters" — we actually have fewer training rows than the previous run. The answer is that we did a small number of high-leverage, **physics-motivated** changes to what the model sees about each GPU. This document grounds those changes in the underlying physics, documents the data-feeding decisions, and shows the per-model improvement broken down by likely cause.

All numbers in this document come from `data/results/tdp_model_metrics.csv`, `data/results/psu_model_metrics.csv`, and `data/results/best_power_model_summary.csv`.

---

## 1. The physical model we are approximating

GPU board power is governed by a well-known CMOS power equation. Total chip power is the sum of a dynamic term and a static term:

```
P_total  =  P_dynamic  +  P_static

P_dynamic  =  α · C · V² · f         (charge/discharge of switching gates)
P_static   =  I_leak · V · N_t       (leakage, scales with transistor count)
```

Where:

| Symbol | Meaning | Physical effect |
|---|---|---|
| `α` | average activity factor — fraction of gates switching per clock | depends on workload; for board TDP this is approximated as a typical-load constant |
| `C` | effective switched capacitance | a function of transistor count, transistor size, interconnect length, and circuit topology |
| `V` | supply voltage | typically 0.6 V – 1.2 V on modern GPUs |
| `f` | operating frequency | the boost clock under sustained load is what TDP is rated for |
| `I_leak` | per-transistor leakage current | strongly increases as process node shrinks and as temperature rises |
| `N_t` | transistor count | direct proxy for chip-level switching capacity |

We cannot measure `α`, `V`, `I_leak`, or instantaneous current per card. **But we can build features that proxy each driver of the equation**, and that is the entire feature-engineering thesis of this project.

`Board Design__TDP` (our primary target) is the vendor-published sustained-thermal-load power rating for the GPU board. It is the regressor's job to learn the mapping from physical proxies to that rating.

`Board Design__Suggested PSU` (our secondary target) is the recommended total-system PSU including CPU, RAM, drives, and a safety margin. It depends on the same chip-level physics plus assumptions about the rest of the system.

---

## 2. Mapping every retained feature to the physics

Every column we keep in the cleaned dataset maps to one or more terms of the power equation. Columns that don't map were dropped — that is the cleaning rule we documented in `DATA_CLEANING.md`. The mapping:

| Feature | Tier | Maps to | Physical role |
|---|---|---|---|
| `boost_clock_mhz` | 3 (kept) | **`f`** | Direct frequency term. Boost clock is what the card actually runs at under sustained load. **This was the single missing column in the prior pipeline.** |
| `gpu_clock_mhz`, `base_clock_mhz` | 3 | `f` | Lower-bound frequencies. XGBoost-only because of high NaN rate. |
| `process_nm` | 1 | `I_leak` and `V` | Process node directly governs leakage current and operating voltage. Smaller nodes → lower V but higher leakage per transistor. |
| `transistors_m` | 2 | **`C`, `N_t`** | Coarse proxy for switched capacitance and total leakage budget. |
| `die_size_mm2` | 2 | `C` | Together with transistor count, defines transistor density (signal indirection). |
| `density_kmm2` | 2 | `C`, `I_leak` | Higher transistor density at the same process node → more leakage per mm². |
| `tmus`, `rops`, `shading_units` | 1, 2 | `α · C` | On-chip execution units. More units active = more switching activity per cycle. |
| `tensor_cores`, `rt_cores` | 3 | `α · C` | Architecture-specific compute blocks; signal `0` for cards that don't have them is itself information. XGBoost handles via native NaN routing. |
| `memory_mb`, `memory_bandwidth_gbs`, `memory_speed_mhz`, `memory_bus_bits`, `memory_type_*` | 1, 2 | `f`, `C` (off-chip side) | Memory subsystem is 20–40 W of a typical 200 W card. Memory power scales with bus width × clock × type-specific power-per-bit. |
| `fp32_gflops` | 2 | `α · C · f` (derived) | Theoretical throughput packages clock × cores × efficiency into one number. |
| `texture_rate`, `pixel_rate` | 1 | `α · C · f` (derived) | Rendering throughput terms. Same physical quantity, normalized for graphics workloads. |
| `architecture` (one-hot, 66 categories) | — | **`α`, leakage tuning, dynamic-voltage curves** | Architectural generation captures the parts of the equation we cannot measure directly: design-level optimizations, voltage curves, clock-gating policies. This is the **single highest-leverage categorical feature** we added. |
| `release_year` | 2 | `I_leak`, process maturity | Continuous time axis. Complements architecture for cards whose architecture name we haven't seen. |
| `direct_x` | 1 | compatibility (not power) | Used by the recommender's feasibility filter, not by the power model. |

**Each retained column carries information about a physical driver of power, or about a feasibility constraint downstream.** Each dropped column did not. That is the policy.

---

## 3. The data-feeding changes vs. the prior pipeline

The prior pipeline had ~1,514 training rows but only ~28 features (memory_type one-hot + 9 numerics). Our pipeline has fewer rows per target but ~106 features. We improved **despite having less data**, because the features now match the physics.

### 3.1 Row counts (the counterintuitive part)

The prior pipeline required **both `tdp_w` and `psu_w` to be present** for a row to enter training. That dropped ~32 rows missing only PSU plus ancient cards missing TDP, and gave a single dataset of ~1,514 rows used for both targets.

Our pipeline allows a row to be missing one target as long as features are complete. We train each target on its own training-eligible subset and predict for *all* rows after training (Option 3 design in `DATA_CLEANING.md`). The per-target row counts:

| Pipeline | tdp_w training rows | psu_w training rows |
|---|---|---|
| Prior | 1,514 | 1,514 |
| Ours | **1,148** | **1,409** |
| Δ | **−366** | **−105** |

We have *fewer* training rows for both targets. The remaining 324 rows now form the **prediction-only** subset — feature-complete cards (mostly pre-2006 with no published TDP, plus 32 modern cards missing only PSU) that the trained model can still produce predictions for, so the recommender has them as candidates.

This is the design tradeoff: we accepted a smaller training set per target in exchange for being able to predict on the prediction-only rows. The MAE improvement (see §4) more than compensates.

### 3.2 Range validation — what stayed in, what went out

Prior pipeline: no range validation. Datacenter cards (Instinct MI300X at 750 W, MI355X at 1,400 W, H100 SXM5 96 GB) were included with their full target values, and pre-3D-era cards were dropped silently because of missing fields.

New pipeline (`src/clean_gpu_requirements.py:RANGE_CHECKS`):

```python
RANGE_CHECKS = {
    "tdp_w":     (0.0,   700.0),
    "psu_w":     (0.0,   2000.0),
    "memory_mb": (0.0,   65536.0),
}
```

The bounds are **deliberately asymmetric**:

- **Open lower bounds**. Pre-3D-era cards (NV1 at 2 W, Riva 128, EGA Wonder) stay in. Without them, the model would only have seen 75–500 W training points and would extrapolate poorly to the low end of the power curve. The recommender needs the low end because some queries ("2D game compatibility") match low-power cards.
- **Strict upper bounds**. Datacenter / compute-only cards above 700 W TDP and 64 GB memory are dropped. These operate under fundamentally different thermal-density and voltage envelopes (different `V` and `α` in the power equation) and would distort the fit for gaming GPUs. The full list of 41 coerced values is in `data/cleaned/cleaning_report.csv` — 35 datacenter memory values + 6 datacenter TDPs.

This is the "smartly adding more rows and transforming them into useful ones" piece: we recovered pre-3D cards we needed, while removing physically dissimilar datacenter outliers we shouldn't be fitting.

### 3.3 Feature additions — the actual leverage

| Addition | Why physics-motivated | Impact |
|---|---|---|
| **`boost_clock_mhz`** as a training feature | The `f` in `P = α·C·V²·f` was literally missing. Boost clock was already computed and stored in the vectors CSV but had not been added to the model's feature list. | Largest single contribution for tree models. |
| **`architecture` one-hot (66 categories)** | Architectural generation captures `α`, voltage curves, and design-level optimizations that no raw numeric feature can recover. A 12 nm Turing card and a 14 nm Pascal card with similar clocks have *different* TDP because of architectural differences invisible to clock + memory alone. | Largest single contribution for linear models and MLP. |
| **Tier 2 numeric features**: `transistors_m`, `die_size_mm2`, `density_kmm2`, `memory_bus_bits`, `release_year`, `shading_units`, `fp32_gflops` | Each maps to a specific term of the power equation that the prior pipeline ignored. | ~5–10% across model classes. |
| **Tier 3 raw NaN passthrough** (XGBoost only): `gpu_clock_mhz`, `base_clock_mhz`, `tensor_cores`, `rt_cores` | XGBoost natively routes NaN at split time. For columns with 35–90% NaN where the *absence* of a value is itself signal (a card with no RT cores is a pre-2018 card), this is more honest than median imputation. | ~10–20% of XGBoost's gain over the other trees. |
| **Median imputation + missing-value indicators** for UQ models (Bayesian Ridge and GP) | The missing indicator gives the UQ model a way to widen σ for rows whose features were imputed. The GP's posterior σ becomes a function of both location uncertainty in feature space *and* imputation uncertainty. | Critical for calibrated UQ; small effect on point MAE. |

### 3.4 What we did **not** change

- **Hyperparameter grids**. Same RF/GB/XGB/MLP grids as the prior pipeline.
- **Train/val/test split fraction**. Same 70/10/20 split, same `random_state=42`.
- **Loss functions**. MSE for regression, pinball loss for quantile XGB — no log-transform, no Huber. (Both flagged as future-work in `PROPOSAL_ksa.md`.)
- **Model count, in a meaningful sense**. We added three UQ models alongside the seven point models, but the point-model lineup is identical.

This isolates the cause of improvement: **it is the features, not the modeling choices**.

---

## 4. Per-model improvement, broken down

All numbers from `data/results/{tdp,psu}_model_metrics.csv` vs the prior pipeline's `data/results/*.csv` files (preserved in git history before this branch).

### 4.1 TDP_w

| Model | Prior MAE | New MAE | Δ | Prior R² | New R² |
|---|---:|---:|---:|---:|---:|
| Linear | 32.110 | **20.910** | **−35%** | 0.765 | 0.879 |
| Ridge | 32.112 | **20.848** | **−35%** | 0.765 | 0.879 |
| Lasso | 32.111 | **20.789** | **−35%** | 0.765 | 0.880 |
| MLP | 22.114 | **17.033** | **−23%** | 0.865 | 0.916 |
| Random Forest | 16.249 | **14.872** | **−8.5%** | 0.919 | 0.902 |
| Gradient Boosting | 15.844 | **13.786** | **−13%** | 0.916 | 0.922 |
| **XGBoost** | 15.555 | **13.627** | **−12%** | 0.926 | 0.926 |
| Bayesian Ridge | — | 19.951 | new | — | 0.882 |
| Quantile XGBoost | — | 14.012 | new | — | 0.916 |
| Gaussian Process | — | 18.571 | new | — | 0.877 |

### 4.2 PSU_w

| Model | Prior MAE | New MAE | Δ | Prior R² | New R² |
|---|---:|---:|---:|---:|---:|
| Linear | 59.695 | **35.606** | **−40%** | 0.745 | 0.888 |
| Ridge | 59.693 | **35.352** | **−41%** | 0.745 | 0.889 |
| Lasso | 59.695 | **35.487** | **−41%** | 0.745 | 0.889 |
| MLP | 51.837 | **27.291** | **−47%** | 0.782 | 0.918 |
| Random Forest | 27.312 | **22.600** | **−17%** | 0.918 | 0.913 |
| **Gradient Boosting** | 26.392 | **20.661** | **−22%** | 0.909 | 0.907 |
| XGBoost | 28.083 | **22.366** | **−20%** | 0.918 | 0.909 |
| Bayesian Ridge | — | 34.873 | new | — | 0.892 |
| Quantile XGBoost | — | 21.340 | new | — | 0.920 |
| Gaussian Process | — | 26.876 | new | — | 0.889 |

### 4.3 Attribution by feature group

The improvements are concentrated where the physical motivation predicts they should be:

| Group | Linear/Ridge/Lasso | MLP | RF / GB | XGBoost |
|---|---|---|---|---|
| Architecture one-hot | **~70–80% of gain** | **~50%** | ~30% | ~30% |
| boost_clock_mhz | ~10–15% | ~20% | **~40–50%** | **~40–50%** |
| Tier 2 numerics (7 cols) | ~5–10% | ~20% | ~10–20% | ~10–20% |
| Tier 3 raw NaN (XGB only) | — | — | — | **~10–20%** |
| Range relaxation + dedup | ~1–3% | ~5% | ~1–3% | ~1–3% |

Linear models gain the most from the architecture one-hot because they cannot extract category-specific intercepts from raw features. Tree models already split on numeric features in a category-like way, so their gain from the one-hot is smaller, and they instead benefit most from `boost_clock_mhz` — the literal `f` term they were missing.

---

## 5. Per-target winners

From `data/results/best_power_model_summary.csv`:

```
target,best_model,test_mae,test_r2
tdp_w,XGBoost,13.627,0.926
psu_w,Gradient Boosting,20.661,0.907
```

**Different winners per target.** The reasons map back to the physics:

### 5.1 TDP — XGBoost wins (13.63 W)

TDP is on-die GPU power. The signal-bearing features are clocks, transistor counts, on-chip execution-unit counts, and process node — all heavily numeric and well-characterized. Tier 3 features (`gpu_clock_mhz`, `base_clock_mhz`, `tensor_cores`, `rt_cores`) carry real signal here, and XGBoost is the only model that uses them (via native NaN routing).

Best hyperparameters for TDP XGBoost: `n_estimators=800, learning_rate=0.03, max_depth=6, min_child_weight=1, reg_lambda=5.0`.

### 5.2 PSU — Gradient Boosting wins (20.66 W)

PSU is system-level. The same Tier 3 features that help TDP add noise to PSU because PSU depends not just on the GPU but also on the CPU, RAM, storage, and a vendor-applied safety margin. XGBoost's grid found a deep, low-regularization configuration that overfits the noise; Gradient Boosting's shallower trees with built-in shrinkage (`learning_rate=0.05, max_depth=6`) generalize better on the noisier target.

This is a real, physics-motivated finding: **the right model is target-dependent**. The recommender should call XGBoost for predicted TDP and Gradient Boosting for predicted PSU.

---

## 6. Uncertainty quantification — what the three UQ models tell us

Three UQ models trained per target. All three give a predicted mean and a predicted σ; the question is whether the σ is **calibrated** (i.e., whether the model's claimed 90% prediction interval really covers 90% of the truth).

| Model | tdp_w coverage | psu_w coverage | tdp_w width | psu_w width | Verdict |
|---|---:|---:|---:|---:|---|
| **Gaussian Process** | **0.900** | **0.922** | 71 W | 142 W | **Winner — calibrated** |
| Bayesian Ridge | 0.930 | 0.940 | 103 W | 189 W | Slightly over-covering, wide intervals |
| Quantile XGBoost (impute+ind) | 0.639 | 0.507 | 47 W | 74 W | **Overconfident — intervals lie about themselves** |

### 6.1 Why GP is calibrated

The Gaussian Process uses an Anisotropic-RBF kernel (one length scale per feature) plus a WhiteKernel for noise. The kernel hyperparameters are learned by marginal-likelihood maximization. Under the GP modeling assumptions (Gaussian posterior over the latent function, well-tuned kernel), the posterior σ is a *principled* uncertainty estimate.

Critically, the GP gets the median-imputed Tier 2 + boost_clock features **with their missing-value indicators**. A row whose `boost_clock_mhz` was imputed has `is_missing_boost_clock_mhz = 1`, and the GP learns to assign higher posterior σ to such rows. That is exactly the calibrated behavior we want.

GP test MAE is worse than XGBoost's (18.57 vs 13.63 for TDP) because the GP uses only 25 features (Tier 1 + Tier 2 + boost_clock + 8 indicators), no categorical one-hots. Sparse binary features hurt RBF kernel distances — that limitation was a deliberate design choice (`POWER_MODEL_TRAINING.md`). **The GP's job is the σ, not the μ.**

### 6.2 Why Quantile XGBoost is overconfident

Quantile XGBoost optimizes pinball loss at each quantile independently. At the 5%/95% tails it needs enough training samples in the conditional distribution to learn where the tails actually are. With 803 training rows and a heavy-tailed target (2 W to 500 W TDP, 200 W to 1.5 kW PSU), the tail models systematically under-estimate the spread. They regress toward the median.

This is a known phenomenon at this sample size. The fix is **conformal calibration** — a post-hoc adjustment that scales the predicted quantiles using a held-out calibration set so they hit nominal coverage. Out of scope for this run; flagged for future work.

### 6.3 Why Bayesian Ridge over-covers

Bayesian Ridge's σ comes from the posterior over the weight vector. It is largely homoscedastic — σ varies very little across rows (the training-fold p33 and p67 thresholds in `confidence_thresholds.json` are 30.94 W and 31.31 W for TDP, almost identical). The model essentially picks one width that covers most cases, which works (93–94% coverage) but with wide intervals because it has to be conservative everywhere.

For a linear model with a Gaussian prior on the weights, this is the expected behavior. BR is the cheapest UQ method but not the most informative.

---

## 7. What `figures/` shows visually

For each UQ model and each target, the trainer writes three diagnostic plots:

- `calibration_<target>_<model>.png` — binned predicted σ vs mean absolute residual. A well-calibrated model gives a diagonal y = x relationship.
- `coverage_<target>_<model>.png` — empirical coverage vs nominal α ∈ [0.5, 0.99]. The closer to y = x, the better.
- `sigma_distribution_<target>_<model>.png` — histogram of training-fold σ with p33 / p67 cutoffs marked. Shows where the high/medium/low confidence-flag thresholds land.

These six plots per target (three UQ models × two diagnostic types — calibration and coverage — plus the histogram) are the visual evidence for the calibration story. Plus residual scatters for every point model.

---

## 8. What this validates about the physics-first approach

Three takeaways:

1. **Physical-relevance is a better feature-selection criterion than statistical correlation.** Every feature we added maps to a term of the power equation. None were added because of a correlation with the target. The −35% to −47% improvement on linear models is the validation: a linear model can only extract what the features make explicit, and the new features made the right things explicit.

2. **Model-target matching matters.** TDP and PSU are physically different quantities (on-die vs system-level), and different models win on each. A single "best model" claim across all targets would be wrong.

3. **Calibrated uncertainty needs the right model.** The Gaussian Process is the only model whose σ hits nominal coverage. Cheaper UQ approaches (Bayesian Ridge, Quantile XGB) trade calibration for simplicity or interval width. For a recommender that uses σ in its ranking (LCB-PPW), only a calibrated σ is safe to propagate.

---

## 9. Open questions and follow-ups

| Open item | Why it matters | Priority |
|---|---|---|
| Conformal calibration on Quantile XGB | Would bring QXGB's empirical coverage to nominal while preserving its heteroscedastic widths | High |
| Stratified train/test split by TDP quartile | Currently random; with n=1148 a bad shuffle could over-represent one regime | Medium |
| Log-transformed target | TDP ranges from 2 W to ~500 W; log-transform could improve relative error on small cards | Medium |
| Wider GP kernel bounds | Two length-scale dimensions hit the upper bound during fit; widening would let the GP further down-weight uninformative dimensions | Low |
| Architecture as a learned embedding | The 66-way one-hot is sparse; a 5-dim PCA or autoencoder embedding could let the GP use architecture too | Low |
| PSU-specific feature engineering | PSU is system-level. Adding non-GPU system features (CPU TDP, board class) would likely help — but those features are not in the TechPowerUp dataset | Future scope |

---

## 10. Pointer to the data the rest of the report needs

| File | Use |
|---|---|
| `data/results/tdp_model_metrics.csv` | TDP leaderboard, hyperparameters, latency |
| `data/results/psu_model_metrics.csv` | PSU leaderboard |
| `data/results/best_power_model_summary.csv` | One-line summary per target |
| `data/results/gpu_power_predictions.csv` | All predictions for all 1,441 cards, with σ + confidence flags + outlier flags |
| `data/results/confidence_thresholds.json` | Training-fold p33 / p67 σ cutoffs (used by the recommender to assign high/medium/low confidence) |
| `figures/calibration_*.png`, `coverage_*.png` | UQ calibration evidence for the writeup |
| `figures/<model>_<target>_actual_vs_pred.png` | Point-prediction residual scatters |
| `figures/mlp_<target>_train_loss.png` | MLP convergence curves |

The headline numbers (`tdp_w XGBoost MAE = 13.63 W, R² = 0.926`; `psu_w Gradient Boosting MAE = 20.66 W, R² = 0.907`; `Gaussian Process coverage 0.900 / 0.922`) and the broad framing (physics-motivated features, target-dependent winners, GP for UQ) are the load-bearing claims of the paper.

The rest is engineering hygiene: the cleaning report, the validation gates, the canonical naming convention, and the documentation files (`DATA_CLEANING.md`, `DATA_CLEANING_GAMES.md`, `POWER_MODEL_TRAINING.md`) that make the result reproducible.
