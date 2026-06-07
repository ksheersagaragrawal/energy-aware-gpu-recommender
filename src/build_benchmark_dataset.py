"""
Joins passmark_benchmarks.csv with gpu_specs_cleaned.csv to produce the
training dataset for the XGBoost model.

Output: data/training/gpu_benchmark_dataset.csv
"""

import pandas as pd
from pathlib import Path

specs_path    = "data/cleaned/gpu_specs_cleaned.csv"
passmark_path = "data/raw/passmark_benchmarks.csv"
out_path      = "data/training/gpu_benchmark_dataset.csv"

memory_types = [
    "DDR", "DDR2", "DDR3", "SDR",
    "GDDR2", "GDDR3", "GDDR4", "GDDR5", "GDDR5X",
    "GDDR6", "GDDR6X", "GDDR7",
    "HBM", "HBM2", "HBM2e", "HBM3",
]

feature_cols = [
    "process_nm", "tmus", "rops", "texture_rate", "pixel_rate",
    "direct_x", "memory_mb", "memory_speed_mhz", "memory_bandwidth_gbs", "tdp_w",
]


# joins gpu specs with passmark scores, one-hot encodes memory type, and saves the training dataset
def build():
    specs    = pd.read_csv(specs_path)
    passmark = pd.read_csv(passmark_path)

    df = specs.merge(passmark[["spec_name", "g3d_mark", "match_score"]], left_on="name", right_on="spec_name", how="inner")
    df = df.dropna(subset=["g3d_mark", "texture_rate", "pixel_rate", "tmus", "rops", "tdp_w"])

    for mem in memory_types:
        df[f"mem_{mem.lower()}"] = (df["memory_type"].str.upper() == mem.upper()).astype(int)

    mem_cols  = [f"mem_{m.lower()}" for m in memory_types]
    keep_cols = ["name", "brand"] + feature_cols + mem_cols + ["g3d_mark", "match_score"]
    df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows → {out_path}")
    return df


if __name__ == "__main__":
    build()
