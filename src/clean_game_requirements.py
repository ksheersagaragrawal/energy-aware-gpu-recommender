"""Clean videogame_requirements.csv for the energy-aware GPU recommender pipeline.

Drops CPU columns, removes high-sparsity/irrelevant GPU columns, replaces
zero sentinels with NaN, parses system columns, deduplicates, and splits
into min/recom requirement datasets with aligned row counts.

Usage:
    python src/clean_game_requirements.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "videogame_requirements.csv"
OUT_DIR = ROOT / "data" / "cleaned"

DROP_GPU_SUFFIXES = [
    "Tensor_Cores",
    "GD_RATING",
    "DisplayPort_Connection",
    "Release_Price",
    "DVI_Connection",
    "HDMI_Connection",
    "Resolution",
    "Best_RAM_Match",
    "Best_Resolution",
    "Power_Connector",
]

ZERO_IS_MISSING_SUFFIXES = [
    "Process",
    "TMUs",
    "Texture_Rate",
    "ROPs",
    "Pixel_Rate",
    "Shader",
    "Open_GL",
    "Memory",
    "Memory_Speed",
    "Memory_Type",
    "Memory_Bandwidth",
    "Boost_Clock",
    "PSU",
]

GPU_SUFFIX_RENAME = {
    "Process": "process_nm",
    "TMUs": "tmus",
    "Texture_Rate": "texture_rate",
    "ROPs": "rops",
    "Pixel_Rate": "pixel_rate",
    "Direct_X": "direct_x",
    "Shader": "shader",
    "Open_GL": "open_gl",
    "Memory": "memory_mb",
    "Memory_Speed": "memory_speed_mhz",
    "Memory_Type": "memory_type",
    "Memory_Bandwidth": "memory_bandwidth_gbs",
    "Boost_Clock": "boost_clock_mhz",
    "PSU": "psu_w",
}

SYSTEM_RENAME = {
    "Min_RAM": "min_ram_mb",
    "Recom_RAM": "recom_ram_mb",
    "Min_VRAM": "min_vram_mb",
    "Recom_VRAM": "recom_vram_mb",
    "Min_OS": "min_os",
    "Recom_OS": "recom_os",
    "Min_Direct_X": "min_direct_x",
    "Recom_Direct_X": "recom_direct_x",
    "Min_HDD_Space": "min_hdd_mb",
    "Recom_HDD_Space": "recom_hdd_mb",
    "Release_Date": "release_date",
    "Name": "name",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_size_to_mb(value):
    """Parse a size string like '1 GB', '512MB', '1.953125GB' to float MB."""
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    m = re.match(r"^(-?\d+\.?\d*)\s*(GB|MB|KB|TB)$", s, re.IGNORECASE)
    if not m:
        return np.nan
    num = float(m.group(1))
    unit = m.group(2).upper()
    if num < 0:
        return np.nan
    if unit == "TB":
        return num * 1024 * 1024
    if unit == "GB":
        return num * 1024
    if unit == "MB":
        return num
    if unit == "KB":
        return num / 1024
    return np.nan


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_raw():
    df = pd.read_csv(RAW_PATH)
    assert df.shape == (10849, 90), f"Unexpected shape: {df.shape}"
    print(f"[load] {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def drop_cpu_columns(df):
    cpu_cols = [c for c in df.columns if c.startswith("Min_CPU_") or c.startswith("Recom_CPU_")]
    df = df.drop(columns=cpu_cols)
    print(f"[drop_cpu] dropped {len(cpu_cols)} CPU columns -> {df.shape[1]} columns remain")
    return df


def drop_sparse_gpu_columns(df):
    drop_cols = []
    for suffix in DROP_GPU_SUFFIXES:
        for prefix in ("Min_GPU_", "Recom_GPU_"):
            col = f"{prefix}{suffix}"
            if col in df.columns:
                drop_cols.append(col)
    df = df.drop(columns=drop_cols)
    print(f"[drop_sparse] dropped {len(drop_cols)} GPU columns -> {df.shape[1]} columns remain")
    return df


def replace_zeros_with_nan(df):
    count = 0
    for suffix in ZERO_IS_MISSING_SUFFIXES:
        for prefix in ("Min_GPU_", "Recom_GPU_"):
            col = f"{prefix}{suffix}"
            if col in df.columns:
                mask = df[col] == 0
                count += mask.sum()
                df.loc[mask, col] = np.nan
    # Memory_Type also has -1 as an invalid sentinel
    for prefix in ("Min_GPU_", "Recom_GPU_"):
        col = f"{prefix}Memory_Type"
        if col in df.columns:
            mask = df[col] < 0
            count += mask.sum()
            df.loc[mask, col] = np.nan
    print(f"[zeros_to_nan] replaced {count} zero-sentinels with NaN")
    return df


def clean_system_columns(df):
    # Replace sentinels
    for col in ["Min_RAM", "Recom_RAM", "Min_VRAM", "Recom_VRAM", "Min_HDD_Space", "Recom_HDD_Space"]:
        df[col] = df[col].replace(["0", "0MB"], np.nan)

    for col in ["Min_OS", "Recom_OS"]:
        df[col] = df[col].replace("0", np.nan)
        df[col] = df[col].str.strip()

    for col in ["Min_Direct_X", "Recom_Direct_X"]:
        df[col] = df[col].replace(0, np.nan)

    # Parse size strings to MB
    for col in ["Min_RAM", "Recom_RAM", "Min_VRAM", "Recom_VRAM", "Min_HDD_Space", "Recom_HDD_Space"]:
        df[col] = df[col].apply(parse_size_to_mb)

    print(f"[clean_system] cleaned system columns")
    return df


def deduplicate(df):
    before = len(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    string_cols = [c for c in df.columns if c not in numeric_cols and c != "Name"]

    agg_dict = {}
    for col in numeric_cols:
        agg_dict[col] = "max"
    for col in string_cols:
        agg_dict[col] = "first"

    df = df.groupby("Name", as_index=False).agg(agg_dict)
    print(f"[dedup] {before} -> {len(df)} rows (removed {before - len(df)} duplicates)")
    return df


def drop_incomplete_gpu_rows(df):
    min_gpu_cols = [c for c in df.columns if c.startswith("Min_GPU_")]
    recom_gpu_cols = [c for c in df.columns if c.startswith("Recom_GPU_")]

    min_all_nan = df[min_gpu_cols].isna().all(axis=1)
    recom_all_nan = df[recom_gpu_cols].isna().all(axis=1)
    drop_mask = min_all_nan | recom_all_nan

    before = len(df)
    df = df[~drop_mask].reset_index(drop=True)
    print(f"[drop_incomplete] {before} -> {len(df)} rows (dropped {drop_mask.sum()} with missing min or recom GPU data)")
    return df


def split_and_rename(df):
    min_gpu_cols = [c for c in df.columns if c.startswith("Min_GPU_")]
    recom_gpu_cols = [c for c in df.columns if c.startswith("Recom_GPU_")]
    system_cols = list(SYSTEM_RENAME.keys())

    # Build rename maps
    min_rename = {f"Min_GPU_{suffix}": name for suffix, name in GPU_SUFFIX_RENAME.items()}
    min_rename.update(SYSTEM_RENAME)

    recom_rename = {f"Recom_GPU_{suffix}": name for suffix, name in GPU_SUFFIX_RENAME.items()}
    recom_rename.update(SYSTEM_RENAME)

    # Select columns for each split (Name is already in system_cols via SYSTEM_RENAME)
    min_keep = min_gpu_cols + [c for c in system_cols if c in df.columns]
    recom_keep = recom_gpu_cols + [c for c in system_cols if c in df.columns]

    df_min = df[min_keep].rename(columns=min_rename)
    df_recom = df[recom_keep].rename(columns=recom_rename)

    # Parse release_date
    df_min["release_date"] = pd.to_datetime(df_min["release_date"], format="%Y-%m-%d")
    df_recom["release_date"] = pd.to_datetime(df_recom["release_date"], format="%Y-%m-%d")

    print(f"[split] min: {df_min.shape}, recom: {df_recom.shape}")
    return df_min, df_recom


def validate(df_min, df_recom):
    gpu_cols = [c for c in df_min.columns if c not in SYSTEM_RENAME.values()]

    # 1. No fully-null GPU rows
    assert df_min[gpu_cols].notna().any(axis=1).all(), "min has rows with all-NaN GPU cols"
    assert df_recom[gpu_cols].notna().any(axis=1).all(), "recom has rows with all-NaN GPU cols"

    # 2. No duplicate game names
    assert df_min["name"].nunique() == len(df_min), "min has duplicate game names"
    assert df_recom["name"].nunique() == len(df_recom), "recom has duplicate game names"

    # 3. DirectX never NaN
    assert df_min["direct_x"].notna().all(), "min has NaN in direct_x"
    assert df_recom["direct_x"].notna().all(), "recom has NaN in direct_x"

    # 4. Memory >95% present
    assert df_min["memory_mb"].notna().mean() > 0.95, "min memory_mb coverage < 95%"
    assert df_recom["memory_mb"].notna().mean() > 0.95, "recom memory_mb coverage < 95%"

    # 5. No negative values in numeric columns
    numeric_cols = df_min.select_dtypes(include="number").columns
    for col in numeric_cols:
        vals = df_min[col].dropna()
        if len(vals) > 0:
            assert (vals >= 0).all(), f"min has negative values in {col}"
    for col in df_recom.select_dtypes(include="number").columns:
        vals = df_recom[col].dropna()
        if len(vals) > 0:
            assert (vals >= 0).all(), f"recom has negative values in {col}"

    # 6. Row count sanity
    assert 6000 < len(df_min) < 8000, f"min row count unexpected: {len(df_min)}"
    assert 6000 < len(df_recom) < 8000, f"recom row count unexpected: {len(df_recom)}"

    # 7. Both files have identical row count and game set
    assert len(df_min) == len(df_recom), "min and recom have different row counts"
    assert set(df_min["name"]) == set(df_recom["name"]), "min and recom have different game sets"

    # 8. Column count
    assert len(df_min.columns) == 26, f"min has {len(df_min.columns)} columns, expected 26"
    assert len(df_recom.columns) == 26, f"recom has {len(df_recom.columns)} columns, expected 26"

    # 9. No CPU columns remain
    assert not any("cpu" in c.lower() for c in df_min.columns), "min still has CPU columns"
    assert not any("cpu" in c.lower() for c in df_recom.columns), "recom still has CPU columns"

    print("[validate] all assertions passed")


def print_summary(df_min, df_recom):
    print("\n" + "=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)

    for label, df in [("MIN", df_min), ("RECOM", df_recom)]:
        print(f"\n--- {label} dataset: {df.shape[0]} rows x {df.shape[1]} columns ---")
        print(f"\nNull % per column:")
        null_pct = (df.isnull().sum() / len(df) * 100).round(1)
        for col, pct in null_pct.items():
            print(f"  {col:30s} {pct:6.1f}%")

        print(f"\nNumeric column ranges:")
        for col in df.select_dtypes(include="number").columns:
            vals = df[col].dropna()
            if len(vals) > 0:
                print(f"  {col:30s} min={vals.min():10.2f}  median={vals.median():10.2f}  max={vals.max():10.2f}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df = load_raw()
    df = drop_cpu_columns(df)
    df = drop_sparse_gpu_columns(df)
    df = replace_zeros_with_nan(df)
    df = clean_system_columns(df)
    df = deduplicate(df)
    df = drop_incomplete_gpu_rows(df)
    df_min, df_recom = split_and_rename(df)
    validate(df_min, df_recom)
    print_summary(df_min, df_recom)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_min.to_csv(OUT_DIR / "game_reqs_min.csv", index=False)
    df_recom.to_csv(OUT_DIR / "game_reqs_recom.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'game_reqs_min.csv'}")
    print(f"Saved to {OUT_DIR / 'game_reqs_recom.csv'}")


if __name__ == "__main__":
    main()
