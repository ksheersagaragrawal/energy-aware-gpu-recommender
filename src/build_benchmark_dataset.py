"""
Joins passmark_benchmarks.csv with gpu_specs_cleaned.csv to produce a
training dataset for the ML recommender model.

Features : GPU hardware specs (process_nm, tmus, rops, texture_rate,
           pixel_rate, direct_x, memory_mb, memory_speed_mhz, memory_type,
           memory_bandwidth_gbs, tdp_w)
Labels   : g3d_mark   — PassMark G3D Mark (real measured benchmark)
           dx9_fps    — DirectX 9  synthetic FPS
           dx10_fps   — DirectX 10 synthetic FPS
           dx11_fps   — DirectX 11 synthetic FPS
           dx12_fps   — DirectX 12 synthetic FPS

Output: data/training/gpu_benchmark_dataset.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
SPECS_PATH    = ROOT / "data" / "cleaned" / "gpu_specs_cleaned.csv"
PASSMARK_PATH = ROOT / "data" / "raw" / "passmark_benchmarks.csv"
OUT_DIR       = ROOT / "data" / "training"
OUT_PATH      = OUT_DIR / "gpu_benchmark_dataset.csv"

MEMORY_TYPE_CATEGORIES = [
    "DDR", "DDR2", "DDR3", "SDR",
    "GDDR2", "GDDR3", "GDDR4", "GDDR5", "GDDR5X",
    "GDDR6", "GDDR6X", "GDDR7",
    "HBM", "HBM2", "HBM2e", "HBM3",
]

GPU_FEATURE_COLS = [
    "process_nm",
    "tmus",
    "rops",
    "texture_rate",
    "pixel_rate",
    "direct_x",
    "memory_mb",
    "memory_speed_mhz",
    "memory_bandwidth_gbs",
    "tdp_w",
]

LABEL_COLS = ["g3d_mark", "dx9_fps", "dx10_fps", "dx11_fps", "dx12_fps", "gpu_compute_ops"]


def encode_memory_type(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode memory_type_raw into known categories."""
    for cat in MEMORY_TYPE_CATEGORIES:
        df[f"mem_{cat.lower()}"] = (
            df["memory_type_raw"].str.upper() == cat.upper()
        ).astype(int)
    return df


def build():
    specs    = pd.read_csv(SPECS_PATH)
    passmark = pd.read_csv(PASSMARK_PATH)

    print(f"GPU specs   : {len(specs)} rows")
    print(f"PassMark    : {len(passmark)} rows")

    # passmark has spec_name (our GPU name) and pm_name (PassMark's name)
    # Join on spec_name == specs.name
    merged = specs.merge(
        passmark[["spec_name", "g3d_mark", "dx9_fps", "dx10_fps",
                  "dx11_fps", "dx12_fps", "gpu_compute_ops", "match_score"]],
        left_on="name",
        right_on="spec_name",
        how="inner",
    )
    print(f"After join  : {len(merged)} rows")

    # Drop rows with missing G3D Mark (our primary label)
    merged = merged.dropna(subset=["g3d_mark"])
    print(f"With G3D    : {len(merged)} rows")

    # One-hot encode memory type
    merged = encode_memory_type(merged)
    mem_cols = [f"mem_{c.lower()}" for c in MEMORY_TYPE_CATEGORIES]

    # Final feature set
    feature_cols = GPU_FEATURE_COLS + mem_cols

    # Drop rows missing critical features
    before = len(merged)
    merged = merged.dropna(subset=["texture_rate", "pixel_rate", "tmus", "rops", "tdp_w"])
    print(f"After feature dropna: {len(merged)} rows (dropped {before - len(merged)})")

    # Select and order columns
    keep_cols = ["name", "brand"] + feature_cols + LABEL_COLS + ["match_score"]
    out = merged[[c for c in keep_cols if c in merged.columns]].copy()
    out = out.reset_index(drop=True)

    # Summary stats
    print(f"\nLabel distributions:")
    for col in LABEL_COLS:
        if col in out.columns:
            valid = out[col].dropna()
            print(f"  {col:20s}: n={len(valid):4d}  "
                  f"min={valid.min():.0f}  median={valid.median():.0f}  max={valid.max():.0f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows x {len(out.columns)} cols → {OUT_PATH}")
    return out


if __name__ == "__main__":
    build()