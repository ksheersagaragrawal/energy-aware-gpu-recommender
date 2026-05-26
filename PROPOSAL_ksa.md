# Proposal: Improving the Energy-Aware GPU Recommender (Pre-Submission Refactor)

This proposal walks through the project pipeline end-to-end — data cleaning, feature engineering, power-model training, the performance score, feasibility filtering, the recommendation methods in Section 20, and evaluation — and identifies what is currently incorrect, what is hand-wavy, what is missing, and what should be added so the work credibly fits the "ML for Physical Applications" course framing.

---

## 1. Data cleaning (`src/clean_gpu_requirements.py`)

The cleaning step is functional but is doing less work than it should, and a few small bugs need fixing before any downstream improvement is meaningful.

**Bugs and hygiene.** `clean()` accepts a `df` parameter and then immediately overwrites it by re-reading `data/raw/gpu_specs.csv` on line 26 — the parameter is dead code and makes the function untestable. Drop the re-read. The filtering step uses `Mobile Graphics__Release Date.isna()` as the negative signal for "is desktop"; this is fragile (a future raw export with empty strings rather than NaN would silently include mobile parts). Replace with an explicit positive predicate on `Graphics Card__Production` plus a sanity check that we didn't drop more than ~X% of rows in any single filter, and log the drop count at each step so future regressions are visible.

**Missing features that should land here, not later.** Today the cleaner emits process_nm, TMUs, ROPs, texture_rate, pixel_rate, direct_x, memory_mb, memory_speed_mhz, memory_bandwidth_gbs, psu_w, tdp_w, memory_type_raw, brand, name. Three columns that are in the raw CSV and matter physically/architecturally are *not* being extracted: `Clock Speeds__Boost Clock`, `Graphics Processor__Architecture`, and `Graphics Card__Release Date`. Boost clock is currently joined back in much later via a separate merge inside `build_gpu_power_vectors.py:add_boost_clock` — that's both a layering violation and a source of silent row-loss when the merge fails. Move boost_clock parsing into the cleaner, alongside `architecture` (kept as raw string for one-hot encoding later) and `release_year` (parsed from the release date). Process node alone does not capture architectural efficiency improvements — a 14nm Pascal card and a 12nm Turing card differ in efficiency in ways `process_nm` cannot explain, but `architecture` can.

**Validation.** Add explicit numeric range checks: TDP in (5, 700) W, PSU in (100, 2000) W, memory in (128, 65536) MB. Any row violating these should be flagged in a `cleaning_report.csv` rather than silently dropped, so we can audit whether our cleaning is throwing away legitimate data. Also add deduplication by `name` keeping the most-recent variant — TechPowerUp has multiple SKUs of the same chip and our model currently double-counts.

---

## 2. Feature vectors (`src/build_gpu_power_vectors.py`)

Two issues here. **First, train/test leakage.** The current `stand_cols()` and `normalize_perf_features()` fit means, standard deviations, and min/max over the entire dataset including the eventual test split. This biases the standardized features and the perf_score, and inflates downstream test metrics. Standardization and min-max normalization must be fit on the training split only and applied to validation/test. The cleanest fix is to drop in `sklearn.preprocessing.StandardScaler` and `MinMaxScaler` inside an `sklearn` Pipeline, and only `.fit_transform` on the train fold inside the CV loop (see §3).

**Second, the perf_score in code does not match the perf_score in the report.** Equation 12 in the report is an arithmetic weighted mean `0.30·FP32 + 0.25·BW + 0.20·TR + 0.20·PR + 0.05·VRAM`. The actual code in `compute_perf_score()` is an unweighted *geometric* mean over `{texture_rate, pixel_rate, tmus, rops, memory_bandwidth_gbs, memory_speed_mhz, boost_clock}` — a different feature set, a different aggregation, and equal weights. This contradiction needs resolving before submission. Recommendation: keep the geometric mean (it's more robust to scale-mismatched features) but (a) update the report to match Eq. 12, (b) make the feature list explicit, and (c) add a one-line justification for geometric over arithmetic mean in the report — for capability proxies, geometric mean penalizes lopsided GPUs (huge BW but tiny TMUs) more correctly than a linear combination.

Boost clock parsing already exists but the `add_boost_clock` merge is on `name` only — collisions between brands with same model names will drop rows. Merge on `(brand, name)`.

---

## 3. Power model training (`src/train_gpu_specs_models.py`)

This is where the largest unrealized gains are.

**Add boost_clock to the feature set.** The `normal_cols` list on line 365 still excludes `boost_clock`. Physically, GPU dynamic power is `P ∝ f · C · V²`, so frequency (boost_clock) is one of the three primary drivers. It is already joined into the vector CSV. This is a one-line change and is expected to drop TDP MAE meaningfully. This was flagged in the last review and still hasn't landed.

**Log-transform the TDP target.** TDP ranges from ~5 W to ~700 W — heavy-tailed. Tree models and MLPs both fit log(y) more cleanly. Train on `log1p(tdp_w)`, predict, and `expm1` at inference. Same for `psu_w`. Expect another 1–2 W MAE improvement and a much smaller relative error on low-TDP cards (where the absolute error is currently small but the percentage error is large).

**Add architecture and release_year features.** One-hot encode `architecture` (Pascal/Turing/Ampere/Ada/RDNA1/2/3/etc.). Include `release_year` as a continuous feature. This addresses the systematic mis-prediction on newer-arch / lower-process-node cards that current `process_nm` alone cannot explain.

**Stratified split, not pure random.** Replace `train_test_split(..., random_state=42)` with `StratifiedShuffleSplit` over TDP quartiles. The current random split can dump all the 400W+ cards into one side, which is a real problem at n ≈ 1500.

**K-fold cross-validation.** A single 70/10/20 split gives a point estimate with no error bar. Switch to 5-fold CV (stratified by TDP quartile), report mean ± std for MAE/RMSE/R² in Table 11 of the report. The current "XGBoost MAE 15.55 W, R² 0.926" is unfalsifiable as written — we don't know if shuffling the split would give 13 W or 18 W.

**MLP fixes.** The current MLP runs for a fixed 1000 epochs with no early stopping and no target standardization. Add early stopping on val loss with patience=50, and standardize the target. The recent regression in MLP test MAE (20.76 → 22.11) is partly attributable to the fixed-epoch protocol picking up overfitting that early stopping would have avoided.

**Joint TDP/PSU prediction.** TDP and PSU are tightly correlated (PSU ≈ TDP + system overhead + safety margin). Training them independently throws away that correlation. A multi-output XGBoost or a small two-headed MLP that shares trunk features and produces both `tdp_w` and `psu_w` is more sample-efficient and should improve both. Optional but cheap.

**Move toward the syllabus.** This is where the project earns its course title. Add a **Gaussian Process regressor** for TDP using `sklearn.gaussian_process.GaussianProcessRegressor` with an RBF + WhiteKernel — this gives calibrated `σ(TDP̂)` per GPU. GPs are on the syllabus (May 21) and uncertainty quantification is on the syllabus (May 28). The GP doesn't need to beat XGBoost on point MAE; its value is the predicted variance, which we'll use downstream in §6. Add a **physics-informed soft constraint**: penalize the regressor when predicted TDP fails to increase monotonically as a function of `boost_clock`, `memory_bandwidth`, or `TMUs` holding other features fixed. This is implementable as an extra loss term on synthetic perturbed examples and is the closest analog to a PINN that this static dataset supports without runtime telemetry. Both additions give the writeup something real to say about "physical applications" beyond "we predicted a physical quantity."

---

## 4. Performance score

The PerfScore is the most under-justified piece of the pipeline. Two fixes.

**Resolve the code/report inconsistency** (see §2 — geometric mean in code vs weighted arithmetic in report). Pick one, document the choice, and add a one-paragraph justification.

**Sensitivity analysis on weights.** Whatever weighting we use, the report has to demonstrate the *ranking* is robust to perturbations of those weights — otherwise the entire recommendation is an artifact of five magic numbers. Run the full top-k pipeline at three weight configurations (the current one, a uniform one, and one weighted toward bandwidth and FP32 only) and report how often the top-5 set changes. If it changes a lot, the weights need to be learned; if it doesn't, the result is robust and we can leave them.

**Optional: learn the weights.** Fit the weights jointly with the LTR objective in §6 by parameterizing PerfScore as a small linear layer and optimizing it inside the LambdaMART loss. This makes the whole pipeline end-to-end rather than hand-tuned-then-learned.

---

## 5. Feasibility filtering

Section 10 of the report describes hard vs. soft constraints and a tolerance factor α = 0.9. Two correctness items:

**Confirm the implementation matches the spec.** The report's Hard/Soft table (Table 6) describes VRAM and DirectX as hard, others as tolerant. The code needs an audit that this is actually what runs — based on the current codebase I cannot tell whether the α tolerance is applied to bandwidth/texture/pixel as the report claims. Make this explicit in a `feasibility.py` module with unit tests for each constraint.

**Unit alignment.** `memory_mb` is in MB; game requirement `Min VRAM` is in GB. The constraint must convert. Same for `direct_x` — both sides need to be integers.

**Sanity check at the end.** After feasibility filtering, log the distribution of |G_q| per game. If many games have |G_q| < 5, top-5 collapses for trivial reasons. If many games have |G_q| > 500, the filter is too permissive. Both extremes change how to interpret Section 20's results.

---

## 6. Recommendation strategies (Section 20)

This is the section with the most serious correctness issues, and also the easiest section to repair.

**The proxy NDCG/Recall numbers are not informative.** Table 13 reports `NDCG@5 = 1.000` and `Recall@5 = 1.000` for four of the six methods. Those numbers are not evidence of recommendation quality — they reflect that the relevance labels were constructed from the same PPW, TDP, PSU, and efficiency-regret quantities that the methods rank on. A model that ranks on X is trivially good at recovering labels derived from X. Two acceptable fixes: (a) **drop NDCG/Recall from Table 13 entirely** and replace them with the operational metrics that actually matter (TDP, PSU, PPW, efficiency regret, diversity); or (b) **construct labels from an independent source** — e.g., the PassMark / 3DMark scores per GPU (publicly available, not used by our pipeline), or use the game dataset's own `Recom_GPU_GD_RATING` if the GPU named in the recommendation can be mapped to TechPowerUp. The latter is more ambitious but much more defensible. Without one of these fixes, the LTR-Utility-Top5 result claiming "NDCG@5 = 1.000" should not appear in the submission.

**Add the heuristic baselines the report already promised.** Section 14 lists five baselines — Lowest-TDP feasible, Lowest-PSU feasible, Highest-Performance feasible, Random feasible, Rule-based score — and Table 13 evaluates none of them. This is the single largest gap between what the report claims and what it shows. Add all five to Table 13. Without them, the report cannot defend the claim that ML methods help, because Power-Top5 is itself already a strong heuristic (direct PPW ranking with no learning). The interesting comparison is "do the ML methods improve over `Random feasible` or `Highest-Performance feasible`?" — which the current results table does not answer.

**ML-Utility-Top5 is a regression onto a formula and should be reframed or dropped.** The report itself acknowledges this in §20.3 ("if it produces nearly the same ranking, then the learned model is mostly reproducing the hand-designed objective"). Table 13 confirms it does. There are two honest paths: (a) drop ML-Utility-Top5 from the final results — it's not actually doing ML work; or (b) keep it explicitly as a diagnostic showing that pointwise regression cannot improve over its training target, and lean on LTR-Utility-Top5 as the actual learned method. Path (b) is fine if framed honestly.

**LTR-Utility-Top5 needs label fixes.** Currently the relevance labels {0,1,2,3} are derived from PPW, predicted TDP, predicted PSU, efficiency regret, feature-affinity distance, and Pareto membership — all of which are functions of the same hardware features the ranker sees. The ranker is essentially being trained to predict scores computed from its own inputs. To make the LTR result meaningful, the labels should come from a held-out signal: external GPU benchmark scores (3DMark, PassMark), real per-game FPS data if any can be scraped (Tom's Hardware GPU hierarchy is one option), or the game dataset's GD-rating mapped to GPUs by name. At minimum, the labels should be constructed from features the ranker *does not* see — e.g., release year, MSRP, average user rating — so there is something for the ranker to actually learn rather than recover.

**Utility weights (0.50, 0.15, 0.10, 0.15, 0.10) need to be either justified or made learnable.** Today they are unsourced. Either cite where they come from (a small grid search on a held-out set, an ablation), or fold them into the LTR objective so they are learned jointly. As-is they are five magic numbers driving the headline result.

**Reconcile argmax vs Top-5.** The problem statement (Eq. 1) is `argmax`; the evaluation uses Top-5. The report should either evolve the problem statement to "top-k feasible energy-efficient GPU recommendation" early in the document, or justify why the formal problem is single-pick but the evaluation is multi-pick (e.g., to measure diversity collapse). Currently the mismatch makes the framing feel inconsistent.

**Course-aligned recommendation upgrade — UQ-aware ranking.** If we add the GP TDP model from §3, we get per-GPU `σ(TDP̂)`. Use this in the ranker: a GPU with predicted `TDP = 110W ± 5W` is operationally safer than one with `TDP = 110W ± 30W` for a user provisioning a PSU. Implement a lower-confidence-bound (LCB) variant of PPW: `PPW_LCB(g) = PerfScore(g) / (TDP̂(g) + κ·σ(g))`, and report it as a separate row in Table 13. This is a *direct* application of uncertainty quantification (May 28 lecture) to the recommendation problem, ties the GP model's variance output to a concrete downstream decision, and gives the report a clean physics-applications storyline: "we don't just predict TDP, we propagate the uncertainty of that physical prediction into the ranking."

---

## 7. Evaluation framework

**Split games (queries), not GPUs.** The TDP model splits GPUs; the recommender currently evaluates on the same game set without a train/test split over games. Hold out ~20% of games as a test query set so the LTR model is evaluated on unseen game requirements.

**Confidence intervals.** Every metric in Table 13 is a single point estimate. Bootstrap over games (resample with replacement, recompute) to get 95% CIs, and report them. Without CIs we cannot tell whether LTR's 50.33 W average TDP is meaningfully different from Power-Top5's 56.70 W or just noise.

**Inference latency.** The report mentions it as a metric in §15.1 but never reports it. Include it — for a recommender that's intended to run pre-deployment, latency per query is a legitimate metric.

**Failure cases.** Pick 3–5 games where the LTR method recommends very differently from Power-Top5 and discuss qualitatively which is "better" and why. Right now the report claims LTR is better-balanced; a few concrete game-level examples would substantiate that beyond aggregate metrics.

---

## 8. Course-alignment summary

To credibly position this as an "ML for Physical Applications" project rather than an applied recsys project, the minimum lift is:

1. **GP regressor for TDP with calibrated uncertainty** (May 21 GP lecture).
2. **Propagate that uncertainty into the recommender via UQ-aware ranking** (May 28 UQ lecture).
3. **Physics-informed soft constraint** on the TDP model (monotonicity in clock and bandwidth — closest analog to a PINN this dataset allows).
4. **Optional: end-to-end differentiable top-k via OptNet-style relaxation** (May 26 lecture). This would replace the two-stage predict-then-rank pipeline with a single objective.

Any one of (1)+(2) is enough to make the course-alignment claim defensible; doing (3) on top of that makes it strong. (4) is a stretch goal for the final week and shouldn't gate the submission.

---

## 9. Suggested ordering and effort

Roughly in execution order, smallest to largest effort, blocking work first:

| # | Item | Effort | Blocks submission? |
|---|------|--------|---------------------|
| 1 | Add `boost_clock` to `normal_cols` | 1 line | yes |
| 2 | Log-transform TDP/PSU targets | ~10 lines | no |
| 3 | Fix train/test leakage in standardization | ~30 lines | no |
| 4 | Reconcile perf_score code vs report | doc + code | yes |
| 5 | Add heuristic baselines to Table 13 | ~50 lines | **yes** |
| 6 | Drop or replace NDCG/Recall labels | doc + code | **yes** |
| 7 | Stratified K-fold CV in power model | ~40 lines | no |
| 8 | Architecture + release_year features | ~30 lines | no |
| 9 | Split games train/test in §20 | ~20 lines | yes |
| 10 | Bootstrap CIs in Table 13 | ~20 lines | no |
| 11 | GP regressor + σ(TDP̂) | ~80 lines | no |
| 12 | UQ-aware ranking variant in Table 13 | ~40 lines | no |
| 13 | Physics-informed monotonicity loss | ~60 lines | no |
| 14 | Reframe ML-Utility-Top5 or drop | doc + code | yes |
| 15 | Reconcile argmax vs Top-k in problem statement | doc | yes |

Items 1, 4, 5, 6, 9, 14, 15 are the ones I'd treat as hard blockers for a credible final submission. Items 11–13 are the ones that make the project actually fit the course title.
