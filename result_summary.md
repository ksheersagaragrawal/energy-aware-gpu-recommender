# Energy-Aware GPU Recommender - Result Summary and Context Log

## Purpose of this document
This file is a durable project memory log. If a session closes, this document should let a new session quickly recover:
- what the project does end to end,
- why each major component exists,
- where key code/results live,
- what conclusions are currently supported.

---

## 1) Project objective
Build an energy-aware GPU recommendation pipeline for games that:
1. predicts GPU power attributes from static specs (TDP/PSU),
2. uses robust performance signals (PassMark-based) rather than only engineered proxies,
3. recommends feasible GPUs per game workload,
4. evaluates recommendation quality beyond a single metric (efficiency + robustness/diversity).

Core principle: recommend GPUs that satisfy workload requirements with better power efficiency and less overprovisioning.

---

## 2) User understanding check (validated)
The user’s understanding was broadly correct. Final aligned view:

1. Regression models for TDP/PSU from static GPU features: **yes**.
2. Uncertainty models: **yes**, for confidence/risk estimation around power predictions.
3. PassMark training model replacing engineered performance proxy: **yes**.
4. Recommendation frameworks on top of game dataset with robustness analysis: **yes**.

---

## 3) End-to-end pipeline context

### Data sources
- Game requirements vectors:
  - `data/vectors/game_vectors_min.csv`
  - `data/vectors/game_vectors_recom.csv`
- GPU spec vectors:
  - `data/vectors/gpu_power_vectors.csv`
- PassMark benchmark data:
  - `data/raw/passmark_benchmarks.csv`
- Joined training set for PassMark model:
  - `data/training/gpu_benchmark_dataset.csv`

### High-level flow
1. Build/clean vectors and benchmark joins.
2. Train power models (`tdp_w`, `psu_w`) from static specs.
3. Train PassMark model (predict `g3d_mark`).
4. For each game:
   - hard filter by VRAM and DirectX,
   - soft filter by throughput-style requirements,
   - rank candidate GPUs by different recommenders.
5. Compare methods using PPW, power metrics, and robustness/diversity metrics.

---

## 4) Component-wise summary (what, why, results, inference)

## 4.1 Power prediction models (TDP and PSU)

### What we do
Train multiple regressors on static GPU features to predict:
- `tdp_w`
- `psu_w`

Main script:
- `src/train_gpu_specs_models.py`

Outputs:
- `data/results/tdp_model_metrics.csv`
- `data/results/psu_model_metrics.csv`
- `data/results/gpu_power_predictions.csv`
- `data/results/best_power_model_summary.csv`
- supporting figures in `figures/`

### Why it matters
Recommendation ranking needs power estimates for all candidate GPUs. Real measured power may be incomplete, so predictive power models are required to operationalize energy-aware ranking.

### Typical result artifacts
- test MAE/RMSE/R2 per model
- inference latency
- selected “best model” per target

### Inference
Static hardware features can provide useful power estimates, enabling scalable energy-aware recommendation in absence of full measured power labels.

---

## 4.2 Uncertainty quantification (UQ) for power models

### What we do
Add uncertainty-aware variants and diagnostics:
- Bayesian approaches,
- Gaussian Process,
- Quantile XGBoost style intervals,
- calibration and coverage analysis.

Main script:
- `src/train_gpu_specs_models.py`

Outputs:
- interval metrics (coverage/width) in model metrics tables
- calibration/coverage/sigma plots in `figures/`
- `data/results/confidence_thresholds.json`

### Why it matters
Point estimates alone hide risk. UQ indicates where predictions are less reliable, enabling confidence-aware recommendation or risk gating.

### Inference
Uncertainty signals allow safer decision-making and can reduce silent failure from overconfident but wrong power predictions.

---

## 4.3 PassMark-based performance modeling

### What we do
Replace or complement engineered `perf_score` with benchmark-grounded predicted performance:
- scrape/match PassMark benchmark data,
- build joined training set,
- train model to predict `g3d_mark` from static features.

Main scripts:
- `src/scrape_passmark.py`
- `src/build_benchmark_dataset.py`
- `src/train_ml_recommender.py`

Key artifacts:
- `data/raw/passmark_benchmarks.csv`
- `data/training/gpu_benchmark_dataset.csv`
- `models/gpu_performance_model.pkl`

### Why it matters
Handcrafted performance proxies are interpretable but can be biased. Benchmark-backed prediction provides a more empirical performance signal for ranking.

### Inference
PassMark-based signal improves realism and credibility of efficiency ranking (performance per watt).

---

## 4.4 Recommendation frameworks and robustness analysis

### What we do
Compare multiple recommendation strategies after feasibility filtering:
- KNN retrieval variants,
- direct PPW ranking,
- utility-formula ranking,
- ML utility imitation model,
- LTR ranker with proxy relevance labels.

Main scripts:
- `src/phase2_ml_utility_analysis.py`
- `src/phase3_ltr_analysis.py`
- `src/passmark_method_comparison.py`

Main outputs:
- per-game and aggregate CSVs in `data/results/` and phase outputs
- plots in `results/plots/passmark_analysis/`

### Why it matters
A method can look good on one metric but fail on robustness. Evaluation includes:
- efficiency metrics (PPW, TDP/PSU),
- robustness/diversity (unique GPU count, top-1 share, overlap),
- retrieval quality (NDCG@5, Recall@5 for LTR setting).

### Inference
The project is positioned as a robust recommendation study, not only a single-model regression exercise.

---

## 5) Current run context (PassMark method comparison)

Command being run:
```bash
python src/passmark_method_comparison.py --mode recom --k-top 5 --knn-k 50
```

What this run does:
1. Loads games and GPUs.
2. Attaches power predictions and PassMark-based performance.
3. Train/test split by games (default 80/20).
4. Trains:
   - ML utility regressor (pointwise),
   - LTR ranker (pair/query-style).
5. Evaluates multiple methods on test games.
6. Saves per-game summary, aggregate summary, and plots.

---

## 6) Fixes made during this session

### 6.1 Warning suppression for sklearn pickle version mismatch
- Updated `src/recommender.py` to load model payload under a scoped warning filter for `InconsistentVersionWarning`.
- Reason: Colab/runtime sklearn version may differ from model serialization version.

### 6.2 Memory type feature robustness at inference
- Updated `src/recommender.py` to use:
  - `memory_type_raw` if present, else
  - `memory_type`.
- Prevents all-zero memory-type one-hots when schema differs.

### 6.3 Progress logging for long runs
- Added timestamped `_log(...)` in `src/passmark_method_comparison.py`.
- Added periodic progress prints for:
  - ML utility data pass,
  - LTR pair-build pass,
  - evaluation pass.

### 6.4 Pandas FutureWarning flood fix
- Replaced repeated `fillna(0.0).infer_objects(copy=False)` patterns with numeric coercion helper:
  - `_numeric_frame(df) -> df.apply(pd.to_numeric, errors="coerce").fillna(0.0)`
- File: `src/passmark_method_comparison.py`
- Effect: removes massive downcasting warnings and keeps logs readable.

---

## 7) One-slide result framing (draft structure)

Suggested title:
- **Energy-Aware GPU Recommendation from Static Specs: Accurate, Uncertainty-Aware, and Robust**

Suggested 4 blocks:
1. **Power Modeling**  
   Predict TDP/PSU from static features -> enables scalable energy metrics.
2. **Uncertainty Modeling**  
   Calibrated prediction intervals -> confidence-aware recommendations.
3. **PassMark Performance Modeling**  
   Benchmark-grounded performance proxy -> more realistic PPW ranking.
4. **Recommendation Robustness**  
   Multi-method comparison on thousands of games -> efficiency + diversity trade-off analysis.

Suggested bottom-line takeaway:
- The pipeline operationalizes practical energy-aware GPU recommendation by combining static-feature power prediction, uncertainty awareness, and robust multi-metric evaluation.

---

## 8) Important code locations for quick resume

- Main recommender logic:
  - `src/recommender.py`
- PassMark comparison experiment:
  - `src/passmark_method_comparison.py`
- Power modeling + UQ:
  - `src/train_gpu_specs_models.py`
- PassMark data + training:
  - `src/scrape_passmark.py`
  - `src/build_benchmark_dataset.py`
  - `src/train_ml_recommender.py`
- Phase analyses:
  - `src/phase2_ml_utility_analysis.py`
  - `src/phase3_ltr_analysis.py`
- Reporting comparison script:
  - `scripts/phase4_reported_baseline_comparison.py`

---

## 9) Risks / caveats to keep in mind

1. Proxy-label dependency for LTR:
- Relevance labels are derived from engineered Pareto/utility logic, not user click/choice logs.

2. Cross-environment reproducibility:
- Model pickle and library version mismatches can produce warnings.

3. Runtime backend variability:
- XGBoost GPU availability differs by environment; fallback behavior should be maintained.

4. Scraping fragility:
- PassMark parsing may break if website HTML structure changes.

---

## 10) Session note
This summary captures:
- conceptual understanding,
- implementation structure,
- currently implemented fixes,
- slide-ready narrative.

It is intended to be enough context to restart discussion and continue result packaging without re-scanning the entire repo.
