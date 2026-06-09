# Energy-Aware GPU Recommender

This project builds an ML-based GPU recommender for desktop gaming workloads. Given a game requirement and a candidate GPU catalog, the system recommends GPUs that are strong enough to run the game while avoiding unnecessary power usage.

The pipeline combines:

1. **Power prediction**: predicts GPU TDP and suggested PSU from static GPU specs.
2. **Performance prediction**: predicts PassMark-style G3D benchmark score from GPU hardware features.
3. **Recommendation**: filters compatible GPUs and ranks them using power-aware efficiency metrics.

This project does **not** predict exact FPS, runtime latency, or measured in-game power draw. The performance score is a static hardware-based proxy.

## Datasets

The project uses three datasets:

| Dataset                       | Goal                                                                                    |
| ----------------------------- | --------------------------------------------------------------------------------------- |
| Game Eequirements             | Provides minimum and recommended GPU requirements for games                             |
| TechPowerUp GPU specs         | Provides GPU specs such as VRAM, bandwidth, clocks, TMUs, ROPs, TDP, and suggested PSU  |
| PassMark benchmark            | Provides G3D Mark scores used to train the ML performance model                         |

The game requirement data is cleaned into:

```text
data/cleaned/game_reqs_min.csv
data/cleaned/game_reqs_recom.csv
```

The benchmark training data is created at:

```text
data/training/gpu_benchmark_dataset.csv
```

The GPU spec data is cleaned into:

```text
data/cleaned/gpu_specs_cleaned.csv
```

## Repository Structure

```text
energy-aware-gpu-recommender/
├── data/
│   ├── raw/          # Original GPU, game, and benchmark data
│   ├── cleaned/      # Cleaned GPU and game requirement files
│   ├── vectors/      # Game and GPU feature vectors
│   ├── training/     # Benchmark training dataset
│   └── results/      # Model metrics, predictions, and ablation results
│
├── src/
│   ├── clean_game_requirements.py
│   ├── clean_gpu_requirements.py
│   ├── scrape_passmark.py
│   ├── build_vectors.py
│   ├── build_gpu_power_vectors.py
│   ├── build_benchmark_dataset.py
│   ├── train_gpu_specs_models.py
│   ├── train_ml_recommender.py
│   ├── ablation_power_models.py
│   ├── recommender.py
│   ├── recommender_utility.py
│   └── run_recommendation_experiment.py
│
├── models/           # Trained models
├── figures/          #  plots
└── outputs/          # Final recommendation experiment outputs
```


## Setup

```bash
git clone git@github.com:ksheersagaragrawal/energy-aware-gpu-recommender.git
cd energy-aware-gpu-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

Run the ML recommender:

```bash
python3 src/recommender.py --game "Cyberpunk 2077" --method ml --k 5
```

Compare against the rule-based/static method:

```bash
python3 src/recommender.py --game "Cyberpunk 2077" --method top_k --k 5
```

Use recommended requirements and return top 10:

```bash
python3 src/recommender.py --game "Cyberpunk 2077" --method ml --mode recom --k 10
```

Use a stricter soft filter:

```bash
python3 src/recommender.py --game "Cyberpunk 2077" --method ml --threshold 0.90
```

### Recommender Arguments

| Argument      | Example            | Description                                                                             |
| ------------- | ------------------ | --------------------------------------------------------------------------------------- |
| `--game`      | `"Cyberpunk 2077"` | game name or partial game name                                                          |
| `--method`    | `ml`               | `ml` uses predicted G3D score per watt; `top_k` uses static performance score per watt  |
| `--mode`      | `recom`            | requirement mode: `min` or `recom`                                                      |
| `--k`         | `5`                | number of GPU recommendations returned                                                  |
| `--threshold` | `0.80`             | soft-filter threshold for hardware requirement matching                                 |

---

## Recommendation Flow

```text
game name
→ load game requirement vector
→ apply hard compatibility filters
→ apply soft capability filters
→ score feasible GPUs
→ rank by power-aware efficiency
→ return top-k recommendations
```

### Hard Filters

| Requirement | Description                                    |
| ----------- | ---------------------------------------------- |
| VRAM        | GPU must satisfy the game memory requirement   |
| DirectX     | GPU must support the required DirectX version  |

### Soft Filters

| Feature                | Description                     |
| ---------------------- | ------------------------------- |
| `texture_rate`         | Texture processing capability   |
| `pixel_rate`           | Pixel/raster output capability  |
| `memory_bandwidth_gbs` | Memory bandwidth                |
| `tmus`                 | Texture Mapping Units           |
| `rops`                 | Render Output Units             |

A threshold of `0.80` means the GPU must meet at least 80% of each soft requirement.

## ML Performance Model

The ML performance model is trained by first joining cleaned GPU specs with PassMark G3D benchmark labels, then fitting an XGBoost regression model.

### Build the Benchmark Dataset

```bash
python3 src/build_benchmark_dataset.py
```

## Power Modeling

The power model predicts two targets from static GPU specs:

| Target  | Description                           |
| ------- | ------------------------------------- |
| `tdp_w` | GPU board-level Thermal Design Power  |
| `psu_w` | Suggested system PSU capacity         |

Run:

```bash
python3 src/train_gpu_specs_models.py
```

This trains multiple model families, including linear models, tree-based models, MLP, XGBoost, Quantile XGBoost, Bayesian Ridge, and Gaussian Process Regression.

Main outputs:

| File                                        | Description                    |
| ------------------------------------------- | ------------------------------ |
| `data/results/tdp_model_metrics.csv`        | TDP model metrics              |
| `data/results/psu_model_metrics.csv`        | PSU model metrics              |
| `data/results/gpu_power_predictions.csv`    | Predicted TDP/PSU values       |
| `data/results/best_power_model_summary.csv` | Best model summary             |
| `data/results/confidence_thresholds.json`   | Confidence threshold metadata  |

Predicted TDP/PSU values are merged into the recommendation experiment and used for performance-per-watt, ranking utilities, and evaluation metrics.


## Baselines and Metrics

### Baselines

| Baseline                 | Description                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| Minimum GPU baseline     | Closest GPU to the game’s minimum requirement profile              |
| Recommended GPU baseline | Closest GPU to the game’s recommended requirement profile          |
| Lowest TDP feasible      | Lowest-TDP GPU among feasible candidates                           |
| Lowest PSU feasible      | Lowest suggested-PSU GPU among feasible candidates                 |
| Performance-per-watt     | Feasible GPU with best performance score per watt                  |
| Power-Top5               | Top 5 feasible GPUs by performance-per-watt                        |
| LTR-Top5                 | Learning-to-rank top 5 using efficiency and right-sizing features  |

### Evaluation Metrics

| Metric                       | Description                                            |
| ---------------------------- | ------------------------------------------------------ |
| Average selected TDP         | Average TDP of recommended GPUs                        |
| Average selected PSU         | Average suggested PSU of recommended GPUs              |
| Average performance-per-watt | Performance score divided by TDP                       |
| Over-provisioning            | Extra capability beyond the game requirement           |
| Efficiency regret            | Gap from the best feasible efficiency option           |
| Unique GPU count             | Diversity of recommendations                           |
| Top-1 share                  | Whether the recommender collapses to one GPU           |
| NDCG@5                       | Ranking quality against energy-aware relevance labels  |
| Recall@5                     | Whether high-relevance GPUs appear in the top 5        |

## Reproduce the Full Pipeline

Run all commands from the repository root.

### 1. Optional PassMark scrape

The repo already includes `data/raw/passmark_benchmarks.csv`, so scraping is optional.

```bash
python3 src/scrape_passmark.py
```

### 2. Clean data

```bash
python3 src/clean_gpu_requirements.py
python3 src/clean_game_requirements.py
```

### 3. Build vectors

```bash
python3 src/build_vectors.py
python3 src/build_gpu_power_vectors.py
```

### 4. Train power models

```bash
python3 src/train_gpu_specs_models.py
```

### 5. Build benchmark dataset

```bash
python3 src/build_benchmark_dataset.py
```

### 6. Train ML performance model

```bash
python3 src/train_ml_recommender.py
```

### 7. Run final recommendation experiment

```bash
python3 -m src.run_recommendation_experiment --output-dir outputs/recommendation_final
```

Root-level outputs:

```text
outputs/recommendation_final/method_comparison_static.csv
outputs/recommendation_final/method_comparison_g3d.csv
outputs/recommendation_final/run_summary.txt
```

```text
outputs/recommendation_final/static/
outputs/recommendation_final/g3d/
```

### 8. Run ablation study (Optional)

```bash
python3 -m src.ablation_power_models \
  --input-path data/vectors/gpu_power_vectors.csv \
  --output-dir data/results \
  --tdp-metrics data/results/tdp_model_metrics.csv \
  --psu-metrics data/results/psu_model_metrics.csv \
  --no-gpu \
  --n-jobs 1
```

### Ablation Arguments

| Argument        | Example                              | Description                             |
| --------------- | ------------------------------------ | --------------------------------------- |
| `--input-path`  | `data/vectors/gpu_power_vectors.csv` | Input GPU feature vector file           |
| `--output-dir`  | `data/results`                       | Directory for ablation result files     |
| `--tdp-metrics` | `data/results/tdp_model_metrics.csv` | TDP metrics file                        |
| `--psu-metrics` | `data/results/psu_model_metrics.csv` | PSU metrics file                        |
| `--no-gpu`      | flag                                 | Runs ablation without GPU acceleration  |
| `--n-jobs`      | `1`                                  | Number of parallel jobs                 |

---

## Notes

* Full model training can take a long time because several model families are trained and evaluated
* GPU acceleration is needed for full retraining, specifically for ML model training
* The included processed files, trained model artifact, and final outputs allow quick recommendation testing without retraining
* The PassMark scraper is optional; the benchmark CSV is already included with the reproducible data source

---

## Contributors

Team 7: Avinash Gondela, Bhavya Gupta, Ksheer Sagar Agrawal, and Saachi Shenoy.
