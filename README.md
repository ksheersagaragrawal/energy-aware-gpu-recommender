# Energy-Aware GPU Recommender

ML-based GPU power modeling and energy-aware GPU recommendation for PC game workloads.

## Project Goal

This project builds a clean, validated dataset and performs exploratory data analysis (EDA)
as the foundation for an energy-aware GPU recommendation system. **No ML models are trained
at this stage.** The focus is entirely on making the data clean, validated, and analysis-ready.

## Datasets

1. **TechPowerUp GPU Specs** — GPU hardware specifications scraped from
   [https://www.techpowerup.com/gpu-specs/](https://www.techpowerup.com/gpu-specs/).
   Key features: memory size, memory type, memory clock, GPU clock, TMUs, ROPs, cores, PSU.

2. **Kaggle PC Video Game Requirements v2** — `videogame_requirements.csv` containing
   minimum and recommended GPU requirements per game title.

## Project Structure

```
energy-aware-gpu-recommender/
├── data/
│   ├── raw/                  # Raw, unmodified source data
│   └── processed/            # Cleaned, analysis-ready CSVs
├── notebooks/
│   └── eda_cleaning.ipynb    # Step-by-step EDA and cleaning notebook
├── reports/
│   └── figures/              # Saved visualisation images
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py      # Reusable cleaning functions
│   └── validation.py         # Dataset validation helpers
├── tests/
│   ├── __init__.py
│   └── test_cleaned_data.py  # Pytest-based validation tests
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add raw data

Place the following files in `data/raw/`:

- `videogame_requirements.csv` — downloaded from Kaggle.
- `gpu_specs.csv` — TechPowerUp GPU specs (scraped or manually exported).

### 3. Run the cleaning pipeline

```python
from src.data_cleaning import clean_game_requirements, clean_gpu_specs

games_min, games_rec = clean_game_requirements("data/raw/videogame_requirements.csv")
gpu_specs = clean_gpu_specs("data/raw/gpu_specs.csv")

games_min.to_csv("data/processed/games_min_clean.csv", index=False)
games_rec.to_csv("data/processed/games_rec_clean.csv", index=False)
gpu_specs.to_csv("data/processed/gpu_specs_clean.csv", index=False)
```

### 4. Run tests

```bash
pytest tests/
```

### 5. Open the notebook

```bash
jupyter notebook notebooks/eda_cleaning.ipynb
```

## Cleaning Tasks

- Load and inspect raw game requirements and GPU specs data.
- Extract GPU features: memory size, memory type, memory clock, GPU clock, TMUs, ROPs, cores, PSU.
- Handle minimum and recommended GPU requirement columns separately.
- Standardise column names (snake_case, lowercase).
- Convert units (GB, MB, MHz, bit, watts) to plain numeric values.
- Handle missing values and obvious invalid zeros.
- Remove or flag unusable rows.
- Save cleaned outputs as CSV files.

## EDA Visuals

- Distribution of PSU requirements.
- Distribution of GPU memory size.
- Distribution of memory type.
- Correlation heatmap of numeric GPU features.
- Minimum vs recommended PSU comparison.
- Missing value summary.
- Top common memory types and GPU feature ranges.

## License

MIT
