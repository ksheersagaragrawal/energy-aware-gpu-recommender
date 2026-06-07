# Energy-Efficient GPU Power Modeling

This project explores GPU power modeling with static hardware data.

The main idea is to use GPU specifications and benchmark labels to study how GPU performance and power relate across different datasets.

## Project Direction

We use two datasets:

1. **Game Requirements Dataset**
   - Provides minimum and recommended GPU requirements for games.
   - Used to understand workload requirements.

2. **TechPowerUp GPU Specs Dataset**
   - Provides GPU specifications such as memory, bandwidth, clocks, TMUs, ROPs, TDP, and suggested PSU.
   - Used as the candidate GPU database and feature source.

## Current Scope

We are focusing on **desktop discrete GPUs**.

We are not using mobile GPUs, integrated GPUs, data center GPUs, or workstation GPUs in the main analysis because they follow different power, memory, and deployment assumptions.

## Game Dataset Cleaning

For the game requirement dataset:

- Removed CPU columns because this project is focused on GPU analysis.
- Ignored columns that were mostly missing, redundant, or not useful for GPU selection.
- Split the dataset into:
  - minimum requirement dataset
  - recommended requirement dataset
- Kept around 16 useful GPU-related columns.
- Categorized columns into:
  - hard filters
  - soft filters
  - performance score features
  - optional metadata
  - not-needed columns

## Requirement Vector

The requirement vector represents what the game needs.

Example fields:

- GPU memory / VRAM
- DirectX
- texture rate
- pixel rate
- memory bandwidth
- TMUs
- ROPs

These are used to filter candidate GPUs.

## Performance Score

After filtering, the analysis uses a hardware-based performance score.

The score is built from performance-related GPU specs such as:

- texture rate
- pixel rate
- memory bandwidth
- TMUs
- ROPs
- memory speed
- boost clock

We do not include columns like DirectX, PSU, power connector, OS, RAM, or HDD in the performance score because they are not direct GPU performance features.

## Power Modeling

The power model is trained on the TechPowerUp GPU specs dataset.

Primary target:

- `Board Design__TDP`

Secondary target:

- `Board Design__Suggested PSU`

TDP is closer to actual GPU board power, while suggested PSU is more of a system-level provisioning metric.

## Important Notes

This project does not claim to predict exact FPS or runtime gaming latency.

The performance score is a static hardware-based proxy.

The goal is to study GPU hardware behavior in a simple and interpretable way.

## Usage

### 1. Scrape PassMark benchmarks

Fetches GPU G3D Mark scores from PassMark and matches them against the GPU specs dataset.

```bash
python3 src/scrape_passmark.py
```

Output: `data/raw/passmark_benchmarks.csv`

### 2. Build training dataset

Joins PassMark scores with GPU specs and one-hot encodes memory types.

```bash
python3 src/build_benchmark_dataset.py
```

Output: `data/training/gpu_benchmark_dataset.csv`

### 3. Train the XGBoost model

Trains a regression model to predict G3D Mark from hardware specs.

```bash
python3 src/train_ml_recommender.py
```

Output: `models/gpu_performance_model.pkl`

### 4. Run the recommender

```bash
# ML-based ranking (predicted G3D Mark per watt)
python3 src/recommender.py --game "Cyberpunk 2077" --method ml --k 5

# Static ranking (geometric mean perf score per watt)
python3 src/recommender.py --game "Cyberpunk 2077" --method top_k --k 5

# Use recommended requirements instead of minimum
python3 src/recommender.py --game "Cyberpunk 2077" --method ml --mode recom --k 5
```

**Arguments:**

| Argument | Default | Description |
|---|---|---|
| `--game` | required | Game name (partial match supported) |
| `--k` | 5 | Number of GPUs to return |
| `--mode` | `min` | `min` or `recom` requirements |
| `--method` | `top_k` | `top_k` (static) or `ml` (XGBoost) |
| `--threshold` | 0.80 | Soft filter threshold (0–1) |

---

## Unified Recommendation Experiment

The command `python3 -m src.run_recommendation_experiment --output-dir outputs/recommendation_final` runs the report-aligned GPU recommendation experiment for both static and G3D scoring modes, using the same feasibility filters and train/test split.

It also writes a `run_summary.txt` file in the output directory that logs the exact command and the main experiment settings.

## Power-Model Ablation

The command `python3 -m src.ablation_power_models --output-dir data/results` runs the power-model feature ablation study used in the report.

It compares feature subsets for the TDP and PSU prediction models and writes the ablation tables into `data/results/`.
