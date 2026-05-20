"""makes GPU power-model vectors from cleaned TechPowerUp GPU specs

script loads the cleaned GPU candidate dataset, separates metadata,
model features, and prediction targets, one-hot encodes categorical GPU
features, standardizes numeric features for linear models, and saves a
model-ready vector dataset for TDP and PSU prediction

python src/build_gpu_power_vectors.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

CLEANED_PATH = "data/cleaned/gpu_specs_cleaned.csv"
RAW_PATH = "data/raw/gpu_specs.csv"
OUT_DIR = "data/vectors/gpu_power_vectors.csv"

# Maps perf_score feature names to actual GPU vector column names
GPU_PERF_COL_MAP = {
    "texture_rate": "texture_rate",
    "pixel_rate": "pixel_rate",
    "tmus": "tmus",
    "rops": "rops",
    "bandwidth": "memory_bandwidth_gbs",
    "memory_clock": "memory_speed_mhz",
    "boost_clock": "boost_clock",
}

EPSILON = 1e-6


def stand_cols(df):
    """standardizes values for a certain set of columns """
    needs_norm = ["process_nm","tmus", "rops","texture_rate","pixel_rate","direct_x","memory_mb","memory_speed_mhz","memory_bandwidth_gbs"]
    for feature in needs_norm:
        std = df[feature].std()
        if std == 0 or pd.isna(std):
            df[f"standard_{feature}"] = 0.0
        else:
            df[f"standard_{feature}"] = (df[feature] - df[feature].mean()) / std
    return df


def onehot_help(df):
    """onehot encode memory"""
    oneh = OneHotEncoder(sparse_output=False)
    cols = oneh.fit_transform(df[["memory_type_raw"]])
    encoded_col_names = oneh.get_feature_names_out(["memory_type_raw"])
    df[encoded_col_names] = cols
    return df


def add_boost_clock(df):
    """parse boost_clock (MHz) from raw GPU specs and join into df on name"""
    raw = pd.read_csv(RAW_PATH, usecols=["Name", "Clock Speeds__Boost Clock"])
    raw = raw.rename(columns={"Name": "name", "Clock Speeds__Boost Clock": "boost_clock_raw"})
    raw = raw.dropna(subset=["boost_clock_raw"]).drop_duplicates(subset=["name"])
    raw["boost_clock"] = (
        raw["boost_clock_raw"]
        .str.replace(r"[^\d.]", "", regex=True)
        .astype(float)
    )
    df = df.merge(raw[["name", "boost_clock"]], on="name", how="left")
    print(f"[boost_clock] non-null after join: {df['boost_clock'].notna().sum()} / {len(df)}")
    return df


def normalize_perf_features(df):
    """min-max normalize each perf feature into norm_<perf_name> columns"""
    for perf_name, col in GPU_PERF_COL_MAP.items():
        vals = df[col].dropna()
        col_min, col_max = vals.min(), vals.max()
        norm_col = f"norm_{perf_name}"
        if col_max > col_min:
            df[norm_col] = (df[col] - col_min) / (col_max - col_min)
            df[norm_col] = df[norm_col].clip(lower=EPSILON, upper=1.0)
        else:
            df[norm_col] = np.where(df[col].notna(), 1.0, np.nan)
    return df


def compute_perf_score(df):
    """geometric mean of available normalized perf features, same method as game vectors"""
    norm_cols = [f"norm_{name}" for name in GPU_PERF_COL_MAP]
    log_vals = df[norm_cols].apply(np.log)
    df["perf_feature_count"] = df[norm_cols].notna().sum(axis=1).astype(int)
    log_mean = log_vals.mean(axis=1, skipna=True)
    df["perf_score"] = np.exp(log_mean)
    print(f"[perf_score] computed for {df['perf_score'].notna().sum()} / {len(df)} rows")
    return df


def load_data():
    """loads in cleaned data"""
    df = pd.read_csv(CLEANED_PATH)
    return df

def build_vectors(df):
    """get the data ready for models"""
    df = stand_cols(df)
    df = onehot_help(df)
    df = add_boost_clock(df)
    df = normalize_perf_features(df)
    df = compute_perf_score(df)
    return df


def main():
    df = load_data()
    df = build_vectors(df)
    df.to_csv(OUT_DIR, index=False)
    print(f"\nSaved GPU power vectors to {OUT_DIR}")


if __name__ == "__main__":
    main()



