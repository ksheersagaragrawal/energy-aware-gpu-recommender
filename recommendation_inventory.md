# Recommendation Pipeline Inventory

| File | Purpose | Used by final pipeline | Recommendation | Reason |
|---|---|---:|---|---|
| `src/run_recommendation_experiment.py` | Unified static/G3D recommendation runner, diagnostics, run summary | Yes | Keep | This is the report-aligned entry point. |
| `src/recommender.py` | Shared feasibility and ranking helpers; legacy CLI | Yes | Keep/refactor | Final runner imports helper code from here. |
| `src/phase1_topk_knn_analysis.py` | Feasibility filtering, PPW top-k, overlap metrics | Yes | Keep | Final runner uses feasibility and helper metrics from phase 1. |
| `src/phase2_ml_utility_analysis.py` | Feature construction and utility-score helpers | Yes | Keep | Final runner reuses pairwise feature and utility logic. |
| `src/train_ml_recommender.py` | Trains the PassMark/G3D model payload | Yes | Keep | Needed to rebuild `models/gpu_performance_model.pkl`. |
| `src/build_benchmark_dataset.py` | Builds the PassMark training dataset | Yes | Keep | Source of the model training labels. |
| `src/scrape_passmark.py` | Collects PassMark benchmark data | Yes | Keep | Upstream data acquisition for the model target. |
| `src/build_gpu_power_vectors.py` | Builds static GPU performance vectors | Yes | Keep | Source of the static `perf_score`. |
| `src/build_vectors.py` | Builds cleaned vector tables | Yes | Keep | Upstream preprocessing for the recommender inputs. |
| `src/clean_game_requirements.py` | Cleans game requirement vectors | Yes | Keep | Upstream preprocessing for the game side. |
| `src/clean_gpu_requirements.py` | Cleans GPU specification vectors | Yes | Keep | Upstream preprocessing for the GPU side. |
| `src/phase3_ltr_analysis.py` | Legacy LTR experiment script | No | Archive | Superseded by `src/run_recommendation_experiment.py`. |
| `src/passmark_method_comparison.py` | Legacy PassMark vs baseline comparison | No | Archive | Superseded by the unified runner and diagnostic mode. |
| `src/passmark_recommender_analysis.py` | Legacy proxy-vs-PassMark analysis | No | Archive | Not part of the report pipeline. |
| `src/make_clean_result_figures.py` | Report figure generation | Unclear | Keep | Used for report visuals, but not by the main experiment runner. |
| `src/plot_results_slide_figures.py` | Slide-friendly figure generation | Unclear | Keep | Ancillary reporting script. |
| `src/ablation_power_models.py` | Ablation / analysis utility | No | Archive | Not part of the final report pipeline. |
| `scripts/phase4_reported_baseline_comparison.py` | Report-era baseline comparison script | No | Archive | Historical reporting helper, not final runner input. |

Notes:
- The final report flow is `src/run_recommendation_experiment.py`.
- Legacy scripts are retained for reference but are not part of the main report path.
- Report outputs should continue to be generated in a separate cleanup/test folder unless a rerun is explicitly approved.
