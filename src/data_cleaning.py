"""
src/data_cleaning.py
====================
Reusable data-cleaning functions for the energy-aware GPU recommender project.

Two datasets are cleaned here:
  1. Kaggle PC Video Game Requirements v2 (videogame_requirements.csv)
  2. TechPowerUp GPU Specs (gpu_specs.csv)

No ML models are trained in this module — the sole purpose is to produce
clean, analysis-ready DataFrames and CSV files.
"""

from __future__ import annotations

import re
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column-name constants
# ---------------------------------------------------------------------------

# GPU feature columns that must be numeric after cleaning
NUMERIC_GPU_COLS = [
    "memory_size_mb",
    "memory_clock_mhz",
    "gpu_clock_mhz",
    "tmus",
    "rops",
    "shader_cores",
    "psu_watts",
]

# Required columns in the cleaned game-requirements datasets
GAME_REQ_COLS = ["game_title", "min_gpu", "rec_gpu", "min_psu_watts", "rec_psu_watts"]

# Required columns in the cleaned GPU-specs dataset
GPU_SPECS_COLS = [
    "gpu_name",
    "memory_size_mb",
    "memory_type",
    "memory_clock_mhz",
    "gpu_clock_mhz",
    "tmus",
    "rops",
    "shader_cores",
    "psu_watts",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def standardise_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return *df* with column names converted to lowercase snake_case."""
    df = df.copy()
    df.columns = [
        re.sub(r"\s+", "_", col.strip().lower().replace("-", "_"))
        for col in df.columns
    ]
    return df


def parse_memory_size(value: object) -> float:
    """
    Convert a memory-size string such as '8 GB', '512 MB', '4096' to megabytes.

    Rules
    -----
    - Already a number → returned as-is (assumed MB).
    - Contains 'GB' or 'G' → multiply by 1024.
    - Contains 'MB' or 'M' → keep as-is.
    - Unparseable → NaN.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    number_match = re.search(r"[\d.]+", text)
    if not number_match:
        return np.nan
    number = float(number_match.group())
    if "GB" in text or text.endswith("G"):
        return number * 1024
    if "MB" in text or text.endswith("M"):
        return number
    # Plain numeric — assume MB
    return number


def parse_clock_mhz(value: object) -> float:
    """
    Convert a clock-speed string such as '1750 MHz', '1.75 GHz' to MHz.

    Rules
    -----
    - Contains 'GHz' or 'GHZ' → multiply by 1000.
    - Contains 'MHz' or 'MHZ' → keep as-is.
    - Plain numeric → assumed MHz.
    - Unparseable → NaN.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    number_match = re.search(r"[\d.]+", text)
    if not number_match:
        return np.nan
    number = float(number_match.group())
    if "GHZ" in text:
        return number * 1000
    # MHz or plain numeric
    return number


def parse_power_watts(value: object) -> float:
    """
    Convert a power string such as '350 W', '350W', '350' to watts (float).

    Unparseable values → NaN.
    """
    if pd.isna(value):
        return np.nan
    text = str(value).strip().upper()
    number_match = re.search(r"[\d.]+", text)
    if not number_match:
        return np.nan
    return float(number_match.group())


def parse_integer_feature(value: object) -> float:
    """
    Extract the first integer from *value*.  Returns NaN if unparseable.
    Useful for TMUs, ROPs, shader core counts.
    """
    if pd.isna(value):
        return np.nan
    number_match = re.search(r"\d+", str(value))
    if not number_match:
        return np.nan
    return float(number_match.group())


def drop_invalid_rows(
    df: pd.DataFrame,
    required_cols: list[str],
    zero_invalid_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Remove rows that are unusable:
      1. All *required_cols* must be non-null.
      2. Values of *zero_invalid_cols* must be > 0 (zeros treated as missing).

    Parameters
    ----------
    df : pd.DataFrame
    required_cols : list[str]
        Rows missing any of these columns are dropped.
    zero_invalid_cols : list[str] | None
        Subset of *required_cols* where 0 is considered invalid.

    Returns
    -------
    pd.DataFrame
        Cleaned copy of *df* with a reset index.
    """
    df = df.copy()
    before = len(df)

    # Drop rows missing required columns
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)

    # Replace 0 with NaN in zero-invalid columns, then drop again
    if zero_invalid_cols:
        for col in zero_invalid_cols:
            if col in df.columns:
                df[col] = df[col].replace(0, np.nan)
        df = df.dropna(subset=[c for c in zero_invalid_cols if c in df.columns])

    after = len(df)
    logger.info("drop_invalid_rows: %d → %d rows (removed %d)", before, after, before - after)
    print(f"  Rows before cleaning : {before}")
    print(f"  Rows after  cleaning : {after}")
    print(f"  Rows removed         : {before - after}")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Game requirements cleaning
# ---------------------------------------------------------------------------

# Column aliases used by the Kaggle dataset (lowercase snake_case after standardisation)
_GAME_TITLE_ALIASES = ["game_title", "title", "name", "game"]
_MIN_GPU_ALIASES = ["minimum_gpu", "min_gpu", "minimum_graphics", "min_graphics", "req_min_gpu"]
_REC_GPU_ALIASES = ["recommended_gpu", "rec_gpu", "recommended_graphics", "rec_graphics", "req_rec_gpu"]
_MIN_PSU_ALIASES = ["minimum_psu", "min_psu", "minimum_psu_watts", "min_psu_watts", "minimum_power"]
_REC_PSU_ALIASES = ["recommended_psu", "rec_psu", "recommended_psu_watts", "rec_psu_watts", "recommended_power"]


def _resolve_column(df: pd.DataFrame, aliases: list[str], target_name: str) -> pd.DataFrame:
    """Rename the first matching alias column to *target_name* and return df."""
    for alias in aliases:
        if alias in df.columns:
            if alias != target_name:
                df = df.rename(columns={alias: target_name})
            return df
    logger.warning("Column '%s' not found; creating empty column.", target_name)
    df[target_name] = np.nan
    return df


def clean_game_requirements(
    filepath: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and clean the Kaggle PC Video Game Requirements CSV.

    Steps
    -----
    1. Load CSV and standardise column names.
    2. Resolve aliased column names to canonical names.
    3. Parse PSU columns to numeric watts.
    4. Split into *minimum* and *recommended* requirement DataFrames.
    5. Drop unusable rows (missing game title or PSU).
    6. Save cleaned CSVs to data/processed/.

    Parameters
    ----------
    filepath : str
        Path to the raw ``videogame_requirements.csv``.

    Returns
    -------
    (games_min, games_rec) : tuple[pd.DataFrame, pd.DataFrame]
        Cleaned minimum and recommended requirement DataFrames.
    """
    print(f"\n=== Loading game requirements from: {filepath} ===")
    df = pd.read_csv(filepath)
    print(f"  Raw shape: {df.shape}")

    df = standardise_column_names(df)

    # Resolve canonical column names
    for aliases, target in [
        (_GAME_TITLE_ALIASES, "game_title"),
        (_MIN_GPU_ALIASES, "min_gpu"),
        (_REC_GPU_ALIASES, "rec_gpu"),
        (_MIN_PSU_ALIASES, "min_psu_watts"),
        (_REC_PSU_ALIASES, "rec_psu_watts"),
    ]:
        df = _resolve_column(df, aliases, target)

    # Parse PSU columns
    for col in ["min_psu_watts", "rec_psu_watts"]:
        if col in df.columns:
            df[col] = df[col].apply(parse_power_watts)

    # Build minimum requirements DataFrame
    min_cols = ["game_title", "min_gpu", "min_psu_watts"]
    existing_min = [c for c in min_cols if c in df.columns]
    games_min = df[existing_min].copy()
    games_min = games_min.rename(columns={"min_gpu": "gpu_name", "min_psu_watts": "psu_watts"})
    games_min["requirement_type"] = "minimum"

    # Build recommended requirements DataFrame
    rec_cols = ["game_title", "rec_gpu", "rec_psu_watts"]
    existing_rec = [c for c in rec_cols if c in df.columns]
    games_rec = df[existing_rec].copy()
    games_rec = games_rec.rename(columns={"rec_gpu": "gpu_name", "rec_psu_watts": "psu_watts"})
    games_rec["requirement_type"] = "recommended"

    # Drop rows without a game title or PSU value
    print("\n--- Minimum requirements ---")
    games_min = drop_invalid_rows(games_min, ["game_title"], zero_invalid_cols=["psu_watts"])
    print("\n--- Recommended requirements ---")
    games_rec = drop_invalid_rows(games_rec, ["game_title"], zero_invalid_cols=["psu_watts"])

    return games_min, games_rec


# ---------------------------------------------------------------------------
# GPU specs cleaning
# ---------------------------------------------------------------------------

# Column aliases used by TechPowerUp / scraped GPU-specs CSV
_GPU_NAME_ALIASES = ["gpu_name", "name", "gpu", "model", "product_name"]
_MEM_SIZE_ALIASES = ["memory_size_mb", "memory_size", "vram", "memory", "mem_size"]
_MEM_TYPE_ALIASES = ["memory_type", "mem_type", "vram_type"]
_MEM_CLOCK_ALIASES = ["memory_clock_mhz", "memory_clock", "mem_clock", "memory_speed"]
_GPU_CLOCK_ALIASES = ["gpu_clock_mhz", "gpu_clock", "base_clock", "core_clock", "boost_clock"]
_TMUS_ALIASES = ["tmus", "texture_units", "texture_mapping_units"]
_ROPS_ALIASES = ["rops", "render_output_units", "render_outputs"]
_CORES_ALIASES = ["shader_cores", "shaders", "cuda_cores", "stream_processors", "shader_processors"]
_PSU_ALIASES = ["psu_watts", "psu", "tdp", "power", "recommended_psu", "power_supply"]


def clean_gpu_specs(filepath: str) -> pd.DataFrame:
    """
    Load and clean the TechPowerUp GPU specs CSV.

    Steps
    -----
    1. Load CSV and standardise column names.
    2. Resolve aliased column names to canonical names.
    3. Parse memory size (→ MB), clocks (→ MHz), PSU (→ watts), integer counts.
    4. Drop unusable rows (missing GPU name or PSU).
    5. Return cleaned DataFrame.

    Parameters
    ----------
    filepath : str
        Path to the raw GPU specs CSV (scraped or manually exported).

    Returns
    -------
    pd.DataFrame
        Cleaned GPU specs DataFrame.
    """
    print(f"\n=== Loading GPU specs from: {filepath} ===")
    df = pd.read_csv(filepath)
    print(f"  Raw shape: {df.shape}")

    df = standardise_column_names(df)

    # Resolve canonical column names
    for aliases, target in [
        (_GPU_NAME_ALIASES, "gpu_name"),
        (_MEM_SIZE_ALIASES, "memory_size_mb"),
        (_MEM_TYPE_ALIASES, "memory_type"),
        (_MEM_CLOCK_ALIASES, "memory_clock_mhz"),
        (_GPU_CLOCK_ALIASES, "gpu_clock_mhz"),
        (_TMUS_ALIASES, "tmus"),
        (_ROPS_ALIASES, "rops"),
        (_CORES_ALIASES, "shader_cores"),
        (_PSU_ALIASES, "psu_watts"),
    ]:
        df = _resolve_column(df, aliases, target)

    # Parse numeric columns
    df["memory_size_mb"] = df["memory_size_mb"].apply(parse_memory_size)
    df["memory_clock_mhz"] = df["memory_clock_mhz"].apply(parse_clock_mhz)
    df["gpu_clock_mhz"] = df["gpu_clock_mhz"].apply(parse_clock_mhz)
    df["psu_watts"] = df["psu_watts"].apply(parse_power_watts)
    df["tmus"] = df["tmus"].apply(parse_integer_feature)
    df["rops"] = df["rops"].apply(parse_integer_feature)
    df["shader_cores"] = df["shader_cores"].apply(parse_integer_feature)

    # Standardise memory type
    if "memory_type" in df.columns:
        df["memory_type"] = df["memory_type"].astype(str).str.strip().str.upper()
        df["memory_type"] = df["memory_type"].replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})

    # Drop unusable rows
    print("\n--- GPU specs ---")
    df = drop_invalid_rows(df, ["gpu_name"], zero_invalid_cols=["psu_watts"])

    # Keep only canonical columns that exist
    keep_cols = [c for c in GPU_SPECS_COLS if c in df.columns]
    df = df[keep_cols]

    return df
