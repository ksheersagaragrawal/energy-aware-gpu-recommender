# Energy-Efficient GPU Recommendation

This project explores ML-based GPU power modeling for energy-efficient GPU recommendation.

The main idea is simple:

Given a game requirement and a list of candidate GPUs, we want to recommend a GPU that is strong enough to run the game, but does not waste unnecessary power.

## Project Direction

We use two datasets:

1. **Game Requirements Dataset**
   - Gives minimum and recommended GPU requirements for games.
   - Used to understand what level of GPU capability a game needs.

2. **TechPowerUp GPU Specs Dataset**
   - Gives real GPU specifications such as memory, bandwidth, clocks, TMUs, ROPs, TDP, and suggested PSU.
   - Used as the candidate GPU database.
   - Also used to train power prediction models.

## Current Scope

We are focusing on **desktop discrete GPUs**.

We are not using mobile GPUs, integrated GPUs, data center GPUs, or workstation GPUs in the main experiment because they follow different power, memory, and deployment assumptions.

## Game Dataset Cleaning

For the game requirement dataset:

- Removed CPU columns because this project is focused on GPU recommendation.
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

After filtering, we need a way to rank GPUs.

We create a performance score using performance-related GPU specs such as:

- texture rate
- pixel rate
- memory bandwidth
- TMUs
- ROPs
- memory speed
- boost clock

We do not include columns like DirectX, PSU, power connector, OS, RAM, or HDD in the performance score because they are not direct GPU performance features.

## Power Modeling

The power model will be trained on the TechPowerUp GPU specs dataset.

Primary target:

- `Board Design__TDP`

Secondary target:

- `Board Design__Suggested PSU`

TDP is closer to actual GPU board power, while suggested PSU is more of a system-level provisioning metric.

## Recommendation Flow

The final flow is:

1. Clean the game requirement dataset.
2. Clean the TechPowerUp GPU specs dataset.
3. Train a model to predict GPU TDP / PSU.
4. Create a requirement vector for each game.
5. Filter GPUs that do not satisfy the game requirement.
6. Compute performance score for remaining GPUs.
7. Rank feasible GPUs by performance-per-watt.
8. Recommend the best GPU.

## Baselines

We plan to compare our method against these baselines:

1. **Minimum GPU baseline**
   - Find the GPU closest to the game’s minimum requirement profile.

2. **Recommended GPU baseline**
   - Find the GPU closest to the game’s recommended requirement profile.

3. **Lowest TDP feasible**
   - Among feasible GPUs, choose the one with the lowest TDP.

4. **Lowest PSU feasible**
   - Among feasible GPUs, choose the one with the lowest suggested PSU.

5. **Performance-per-watt**
   - Among feasible GPUs, choose the GPU with the best performance score per watt.

## Evaluation Metrics

We will evaluate each method using:

- average selected TDP
- average selected PSU
- average performance score
- average performance-per-watt
- performance over-provisioning
- power over-provisioning
- efficiency regret

## Important Notes

This project does not claim to predict exact FPS or runtime gaming latency.

The performance score is a static hardware-based proxy.

The goal is to build a simple and interpretable energy-aware GPU recommender using game requirements and GPU specifications.

---

## Final Recommendation Experiment

The report uses a single unified experiment runner that evaluates both scoring modes:

- `static` — interpretable hardware-based performance score
- `g3d` — predicted PassMark G3D score from the trained model

### Final command

```bash
python -m src.run_recommendation_experiment --output-dir outputs/recommendation_final
```

### Diagnostic command

```bash
python -m src.run_recommendation_experiment --diagnose-scoring --output-dir outputs/recommendation_cleanup_test
```

This diagnostic checks static-vs-G3D score correlation, PPW correlation, Power-Top5 overlap, and label differences without running the full LTR training path.

---

## ML Recommender

An ML-based recommendation method has been added on top of the existing pipeline. Instead of ranking GPUs using the hand-crafted geometric mean performance score, it uses an **XGBoost regression model** trained on real benchmark data to predict GPU performance.

### New Dataset: PassMark Benchmarks

- **Source:** VideoCardBenchmark.net (PassMark Software) — scraped May 2026
- **File:** `data/raw/passmark_benchmarks.csv`
- **Size:** 901 GPUs with real measured benchmark scores
- **Scores:** G3D Mark, DX9 FPS, DX10 FPS, DX11 FPS, DX12 FPS, GPU Compute

The G3D Mark scores are real measurements submitted by users running PassMark's PerformanceTest software. They are independently measured — not derived from hardware specs — which makes them valid as training labels.

### XGBoost Model

**Task:** Regression — predict G3D Mark score from GPU hardware specs

**Training data:** 1,027 GPUs (PassMark benchmarks joined with GPU specs), split 80/20

#### Input Features (X)

| Feature | Description | Unit |
|---|---|---|
| `pixel_rate` | Pixels rendered per second | GPixel/s |
| `texture_rate` | Textures rendered per second | GTex/s |
| `tmus` | Texture Mapping Units | count |
| `rops` | Render Output Units | count |
| `process_nm` | Manufacturing process node size | nm |
| `memory_mb` | VRAM size | MB |
| `memory_speed_mhz` | Memory clock speed | MHz |
| `memory_bandwidth_gbs` | Memory bandwidth | GB/s |
| `direct_x` | DirectX version supported | version |
| `tdp_w` | Thermal Design Power | watts |
| `mem_gddr5`, `mem_gddr6`, `mem_hbm2`, ... | Memory type one-hot encoded | 16 binary flags |

**Total: 26 features** (10 continuous + 16 one-hot memory type flags)

#### Output (y)

| Output | Description | Range in dataset |
|---|---|---|
| `G3D Mark` | PassMark overall GPU gaming benchmark score | 1 → 41,587 |

#### Model Performance

| Split | MAE | RMSE | R² |
|---|---|---|---|
| Train (821 GPUs) | 221.8 | 716.8 | 0.992 |
| Test (206 GPUs) | 721.7 | 1,699.3 | 0.958 |

#### Top Feature Importances

| Feature | Importance |
|---|---|
| `pixel_rate` | 43.1% |
| `process_nm` | 12.0% |
| `texture_rate` | 10.1% |
| `mem_gddr5` | 6.3% |
| `mem_hbm2` | 4.8% |

`pixel_rate` is the most important feature — rendering pixels is the core gaming workload.

### Full Recommendation Flow (ML Method)

```
User inputs: game name (e.g. "Cyberpunk 2077")
                        |
                        v
         Look up game in game_vectors_min.csv
         (7,292 games with hardware requirement vectors)
                        |
                        v
              Extract game requirements:
              VRAM, DirectX, texture rate, pixel rate,
              memory bandwidth, TMUs, ROPs
                        |
                        v
         ┌──── HARD FILTER (must pass both) ────┐
         │  GPU VRAM    ≥ game VRAM requirement  │
         │  GPU DirectX ≥ game DirectX req       │
         └───────────────────────────────────────┘
                        |
                        v
         ┌──── SOFT FILTER (must meet ≥ alpha%) ───┐
         │  GPU texture rate  ≥ alpha% of game req  │
         │  GPU pixel rate    ≥ alpha% of game req  │
         │  GPU bandwidth     ≥ alpha% of game req  │
         │  GPU TMUs          ≥ alpha% of game req  │
         │  GPU ROPs          ≥ alpha% of game req  │
         └───────────────────────────────────────┘
                        |
                        v
         For each surviving GPU:
         Feed GPU hardware specs → XGBoost model
         → predicted G3D Mark score
                        |
                        v
         Compute: predicted_G3D / TDP
                        |
                        v
         Sort descending → return top-k GPUs
```

> The game requirements are used **only for filtering**. They are never fed into the XGBoost model. The model only sees GPU hardware specs.

### Scripts

| Script | Purpose |
|---|---|
| `src/scrape_passmark.py` | Scrapes G3D Mark + DX FPS scores from VideoCardBenchmark.net |
| `src/build_benchmark_dataset.py` | Joins PassMark data with GPU specs → training dataset |
| `src/train_ml_recommender.py` | Trains XGBoost model and saves to `models/gpu_performance_model.pkl` |

### How to Run the ML Recommender

**Step 1: Scrape PassMark data** (takes ~12 min, saves incrementally)
```bash
python src/scrape_passmark.py
```

**Step 2: Build training dataset**
```bash
python src/build_benchmark_dataset.py
```

**Step 3: Train the model**
```bash
python src/train_ml_recommender.py
```

**Step 4: Run recommendations**
```bash
python src/recommender.py --game "GAME NAME" --method ml [options]
```

#### Arguments

| Argument | Required | Default | Options | Description |
|---|---|---|---|---|
| `--game` | Yes | — | any string | Game name, partial match supported |
| `--method` | No | `top_k` | `top_k`, `knn`, `ml` | Ranking method |
| `--k` | No | `5` | any int | Number of GPUs to return |
| `--mode` | No | `min` | `min`, `recom` | Use minimum or recommended game requirements |
| `--threshold` | No | `0.80` | 0.0 – 1.0 | Soft filter strictness (used with `top_k` and `ml`) |

#### Examples

```bash
# ML recommender — top 5 most power-efficient GPUs for Cyberpunk 2077
python src/recommender.py --game "Cyberpunk 2077" --method ml --k 5

# Compare against rule-based method
python src/recommender.py --game "Cyberpunk 2077" --method top_k --k 5

# Use recommended requirements, return top 10
python src/recommender.py --game "Cyberpunk 2077" --method ml --mode recom --k 10

# Stricter soft filter — GPU must meet 90% of game requirements
python src/recommender.py --game "Cyberpunk 2077" --method ml --threshold 0.9
```
