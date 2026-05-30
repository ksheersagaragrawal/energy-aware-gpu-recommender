"""Build model-ready GPU feature vectors from cleaned specs.

Reads `gpu_specs_cleaned.csv`, adds:
 - standardized numeric features (for linear / MLP models)
 - one-hot encoded categorical features (memory_type, architecture, generation)
 - min-max normalized performance features
 - geometric-mean perf_score

Rows with NaN targets are passed through unchanged so the trainer can predict
on them after fitting.

Usage:
    python src/build_gpu_power_vectors.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
CLEANED_PATH = ROOT / "data" / "cleaned" / "gpu_specs_cleaned.csv"
OUT_PATH = ROOT / "data" / "vectors" / "gpu_power_vectors.csv"


# Numeric features that get standardized (mean 0, std 1) for linear / MLP / BR / GP models.
# NaN values are median-imputed before standardization so downstream models don't choke.
# Tree models use the raw (NaN-preserving) columns instead.
#
# Includes Tier 1 (all 9, 0% NaN), Tier 2 (7 cols, 1-25% NaN), and boost_clock_mhz from
# Tier 3. The other Tier 3 columns (gpu_clock_mhz, base_clock_mhz, tensor_cores, rt_cores)
# are excluded — their NaN rate (35-90%) is too high for imputation to be meaningful.
STANDARDIZE_COLS = [
    # Tier 1 — 0% NaN, required features
    "process_nm",
    "tmus",
    "rops",
    "texture_rate",
    "pixel_rate",
    "direct_x",
    "memory_mb",
    "memory_speed_mhz",
    "memory_bandwidth_gbs",
    # Tier 2 — <25% NaN, median-imputable
    "transistors_m",
    "die_size_mm2",
    "density_kmm2",
    "memory_bus_bits",
    "release_year",
    "shading_units",
    "fp32_gflops",
    # Boost clock from Tier 3 — highest-signal of the heavy-NaN cols
    "boost_clock_mhz",
]

# Categorical columns to one-hot encode. `generation` is kept as a raw string
# column in the cleaned CSV but excluded here — it has 250+ distinct values
# (mostly product-line labels like "GeForce 600") and one-hot encoding it
# blows up the feature space without adding signal beyond `architecture`.
ONEHOT_COLS = ["memory_type", "architecture"]

# Performance features for perf_score — must match the game side
PERF_FEATURES = [
    "texture_rate",
    "pixel_rate",
    "memory_bandwidth_gbs",
    "tmus",
    "rops",
    "memory_speed_mhz",
    "boost_clock_mhz",
]

EPSILON = 1e-6


def standardize(df):
    """Add `standard_<col>` for each STANDARDIZE_COLS column.

    NaN values are imputed with the column median before computing mean/std so
    the standardized column has no missing values. The raw column is kept
    untouched alongside for tree models.
    """
    for col in STANDARDIZE_COLS:
        vals = df[col]
        median = vals.median()
        filled = vals.fillna(median)
        mu = filled.mean()
        sigma = filled.std()
        if sigma == 0 or pd.isna(sigma):
            df[f"standard_{col}"] = 0.0
        else:
            df[f"standard_{col}"] = (filled - mu) / sigma
    return df


def onehot(df):
    """One-hot encode categorical columns, treating NaN as its own 'unknown' bucket."""
    for col in ONEHOT_COLS:
        if col not in df.columns:
            continue
        vals = df[col].fillna("unknown").astype(str).str.strip()
        enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = enc.fit_transform(vals.values.reshape(-1, 1))
        names = enc.get_feature_names_out([col])
        df[names] = encoded
        print(f"[onehot] {col}: {len(names)} categories")
    return df


def normalize_perf(df):
    """Min-max normalize each PERF_FEATURES column into `norm_<col>` in [eps, 1.0]."""
    for col in PERF_FEATURES:
        vals = df[col].dropna()
        if len(vals) == 0:
            df[f"norm_{col}"] = np.nan
            continue
        cmin, cmax = vals.min(), vals.max()
        if cmax > cmin:
            df[f"norm_{col}"] = ((df[col] - cmin) / (cmax - cmin)).clip(EPSILON, 1.0)
        else:
            df[f"norm_{col}"] = np.where(df[col].notna(), 1.0, np.nan)
    return df


def compute_perf_score(df):
    """Geometric mean of available norm_<perf> columns; tracks feature count per row."""
    norm_cols = [f"norm_{c}" for c in PERF_FEATURES]
    log_vals = df[norm_cols].apply(np.log)
    df["perf_feature_count"] = df[norm_cols].notna().sum(axis=1).astype(int)
    df["perf_score"] = np.exp(log_vals.mean(axis=1, skipna=True))
    pcount = (df["perf_score"].notna()).sum()
    print(f"[perf_score] computed for {pcount} / {len(df)} rows")
    return df


def main():
    print("=" * 60)
    print("GPU VECTOR BUILD")
    print("=" * 60)
    df = pd.read_csv(CLEANED_PATH)
    print(f"[load] {len(df)} rows x {len(df.columns)} cols  ({CLEANED_PATH})")

    df = standardize(df)
    df = onehot(df)
    df = normalize_perf(df)
    df = compute_perf_score(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[save] {len(df)} rows x {len(df.columns)} cols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
