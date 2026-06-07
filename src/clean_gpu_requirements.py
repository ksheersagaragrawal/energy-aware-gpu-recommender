"""Clean gpu_specs.csv for the energy-aware GPU recommender pipeline.

Filters to desktop discrete single-GPU rows, parses raw spec strings into
canonical numeric units, applies physical range checks, deduplicates by GPU
name, and writes the cleaned dataset. Rows with missing TDP/PSU targets are
KEPT (as NaN) so the trained model can predict them post-training.

Usage:
    python src/clean_gpu_requirements.py
"""
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "gpu_specs.csv"
OUT_PATH = ROOT / "data" / "cleaned" / "gpu_specs_cleaned.csv"
REPORT_PATH = ROOT / "data" / "cleaned" / "cleaning_report.csv"


# Sentinel strings that mean "missing" — matched after lowercase + strip
MISSING_SENTINELS = {
    "", "unknown", "n/a", "na", "nan", "none", "null",
    "tbd", "tba", "system shared", "system dependent",
}


# Numeric column keep-list: (raw_col, clean_col, parser_name)
KEEP_NUMERIC = [
    ("Graphics Processor__Process Size",       "process_nm",            "unit"),
    ("Graphics Card__Release Date",            "release_year",          "year"),
    ("Graphics Processor__Transistors",        "transistors_m",         "transistors_m"),
    ("Graphics Processor__Die Size",           "die_size_mm2",          "die_size"),
    ("Graphics Processor__Density",            "density_kmm2",          "density"),
    ("Clock Speeds__GPU Clock",                "gpu_clock_mhz",         "unit"),
    ("Clock Speeds__Memory Clock",             "memory_speed_mhz",      "unit"),
    ("Clock Speeds__Base Clock",               "base_clock_mhz",        "unit"),
    ("Clock Speeds__Boost Clock",              "boost_clock_mhz",       "unit"),
    ("Memory__Memory Size",                    "memory_mb",             "memory_size"),
    ("Memory__Memory Bus",                     "memory_bus_bits",       "unit"),
    ("Memory__Bandwidth",                      "memory_bandwidth_gbs",  "bandwidth"),
    ("Render Config__Shading Units",           "shading_units",         "plain"),
    ("Render Config__TMUs",                    "tmus",                  "plain"),
    ("Render Config__ROPs",                    "rops",                  "plain"),
    ("Render Config__Tensor Cores",            "tensor_cores",          "plain"),
    ("Render Config__RT Cores",                "rt_cores",              "plain"),
    ("Theoretical Performance__FP32 (float)",  "fp32_gflops",           "fp32"),
    ("Theoretical Performance__Pixel Rate",    "pixel_rate",            "pixel_rate"),
    ("Theoretical Performance__Texture Rate",  "texture_rate",          "texture_rate"),
    ("Graphics Features__DirectX",             "direct_x",              "unit"),
    ("Board Design__TDP",                      "tdp_w",                 "unit"),
    ("Board Design__Suggested PSU",            "psu_w",                 "unit"),
]


# String column keep-list: (raw_col, clean_col)
KEEP_STRING = [
    ("Brand",                               "brand"),
    ("Name",                                "name"),
    ("Graphics Processor__Architecture",    "architecture"),
    ("Graphics Card__Generation",           "generation"),
    ("Memory__Memory Type",                 "memory_type"),
]


# Transient columns used only for row filtering, then discarded
FILTER_COLS = [
    "Graphics Card__Production",
    "Mobile Graphics__Release Date",
    "Integrated Graphics__Release Date",
    "Top__TMUS",
    "Top__ROPS",
]


# Rows missing any of these features are dropped — all are required for downstream
REQUIRED_FEATURES = [
    "process_nm", "tmus", "rops", "texture_rate", "pixel_rate",
    "direct_x", "memory_mb", "memory_bandwidth_gbs",
]


# Out-of-range values get coerced to NaN and logged. Lower bounds are kept
# permissive on purpose so pre-3D cards (NV1, Riva 128, EGA Wonder, etc.) and
# early integrated frame buffers stay in the dataset; upper bounds remain
# strict because datacenter / compute-only cards (Instinct MI300+, etc.)
# have a different power profile from gaming GPUs and would confound the model.
RANGE_CHECKS = {
    "tdp_w":     (0.0,   700.0),
    "psu_w":     (0.0,   2000.0),
    "memory_mb": (0.0,   65536.0),
}


# ----------------------------------------------------------------------------
# Value parsers
# ----------------------------------------------------------------------------

def _is_missing(value):
    if pd.isna(value):
        return True
    return str(value).strip().lower() in MISSING_SENTINELS


def parse_plain(value):
    """Convert numeric-like value to float; sentinels to NaN."""
    if _is_missing(value):
        return np.nan
    try:
        return float(str(value).replace(",", "").strip())
    except (ValueError, TypeError):
        return np.nan


def parse_unit(value):
    """Extract first number from a string like '1750 MHz', '11 W', '14 nm', '12.1'."""
    if _is_missing(value):
        return np.nan
    s = str(value).replace(",", "").strip()
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else np.nan


def parse_year(value):
    """Extract a 4-digit year from a date string like 'Aug 4th, 1986' or '1998'."""
    if _is_missing(value):
        return np.nan
    m = re.search(r"(19|20)\d{2}", str(value))
    return int(m.group(0)) if m else np.nan


def parse_transistors_m(value):
    """Parse '959 million' -> 959.0, '17 billion' -> 17000.0 (units of millions)."""
    if _is_missing(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*(million|billion|m|b)?\b", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    if (m.group(2) or "") in ("billion", "b"):
        n *= 1000.0
    return n


def parse_die_size(value):
    """Parse '90 mm²' -> 90.0. Handles '90 mm² | unknown' by taking the first variant."""
    if _is_missing(value):
        return np.nan
    head = str(value).split("|")[0].strip()
    if _is_missing(head):
        return np.nan
    m = re.search(r"[\d.]+", head)
    return float(m.group(0)) if m else np.nan


def parse_density(value):
    """Parse '7.8K / mm²' -> 7.8 (units of K transistors / mm²)."""
    if _is_missing(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*([kmb]?)\s*/", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    suffix = m.group(2) or "k"  # default to K if no suffix (raw transistors/mm² is rare here)
    mult = {"k": 1.0, "m": 1000.0, "b": 1_000_000.0}.get(suffix, 1.0)
    return n * mult


def parse_memory_size(value):
    """Parse '8 GB' -> 8192.0, '512 MB' -> 512.0 (units of MB)."""
    if _is_missing(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*(tb|gb|mb|kb)?", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    mult = {"tb": 1024 * 1024, "gb": 1024, "mb": 1.0, "kb": 1.0 / 1024}.get(m.group(2) or "mb", 1.0)
    return n * mult


def parse_bandwidth(value):
    """Parse '448 GB/s' -> 448.0, '1.5 TB/s' -> 1500.0 (units of GB/s)."""
    if _is_missing(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*(tb/s|gb/s|mb/s)?", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    mult = {"tb/s": 1000.0, "gb/s": 1.0, "mb/s": 0.001}.get(m.group(2) or "gb/s", 1.0)
    return n * mult


def parse_pixel_rate(value):
    """Parse '124.4 GPixel/s' -> 124.4 (units of GPixel/s)."""
    if _is_missing(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*(gpixel/s|mpixel/s)?", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    mult = {"gpixel/s": 1.0, "mpixel/s": 0.001}.get(m.group(2) or "gpixel/s", 1.0)
    return n * mult


def parse_texture_rate(value):
    """Parse '248.8 GTexel/s' -> 248.8 (units of GTexel/s)."""
    if _is_missing(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*(gtexel/s|mtexel/s)?", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    mult = {"gtexel/s": 1.0, "mtexel/s": 0.001}.get(m.group(2) or "gtexel/s", 1.0)
    return n * mult


def parse_fp32(value):
    """Parse '1,200 GFLOPS' -> 1200.0, '12.3 TFLOPS' -> 12300.0 (units of GFLOPS)."""
    if _is_missing(value):
        return np.nan
    s = str(value).strip().lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*(tflops|gflops|mflops)?", s)
    if not m:
        return np.nan
    n = float(m.group(1))
    mult = {"tflops": 1000.0, "gflops": 1.0, "mflops": 0.001}.get(m.group(2) or "gflops", 1.0)
    return n * mult


def parse_string(value):
    """Strip whitespace; sentinel-aware NaN coercion."""
    if _is_missing(value):
        return np.nan
    return str(value).strip()


PARSERS = {
    "plain":         parse_plain,
    "unit":          parse_unit,
    "year":          parse_year,
    "transistors_m": parse_transistors_m,
    "die_size":      parse_die_size,
    "density":       parse_density,
    "memory_size":   parse_memory_size,
    "bandwidth":     parse_bandwidth,
    "pixel_rate":    parse_pixel_rate,
    "texture_rate":  parse_texture_rate,
    "fp32":          parse_fp32,
}


# ----------------------------------------------------------------------------
# Pipeline steps
# ----------------------------------------------------------------------------

def _is_multi_gpu(value):
    """Detect dual-GPU SKUs like '2 x 2304' in Top__TMUS / Top__ROPS."""
    if pd.isna(value):
        return False
    s = str(value).replace(",", "").strip().lower()
    return bool(re.search(r"\d+(?:\.\d+)?\s*x\s*\d+(?:\.\d+)?", s))


def filter_rows(df):
    """Keep desktop, discrete, single-GPU rows."""
    n0 = len(df)
    multi      = df["Top__TMUS"].apply(_is_multi_gpu) | df["Top__ROPS"].apply(_is_multi_gpu)
    mobile     = df["Mobile Graphics__Release Date"].notna()
    integrated = df["Integrated Graphics__Release Date"].notna()
    is_card    = df["Graphics Card__Production"].notna()
    keep = (~multi) & (~mobile) & (~integrated) & is_card
    out = df[keep].copy()
    print(f"[filter] {n0} -> {len(out)}  "
          f"(dropped multi={int(multi.sum())}, mobile={int(mobile.sum())}, "
          f"integrated={int(integrated.sum())}, non-card={int((~is_card).sum())})")
    return out


def parse_columns(df):
    """Apply parsers to the keep-list; build a fresh DataFrame with clean column names."""
    out = pd.DataFrame(index=df.index)
    for raw, clean, parser_name in KEEP_NUMERIC:
        out[clean] = df[raw].apply(PARSERS[parser_name])
    for raw, clean in KEEP_STRING:
        out[clean] = df[raw].apply(parse_string)
    # Reorder so identity sits first for readability
    leading = ["brand", "name", "architecture", "generation", "release_year"]
    rest = [c for c in out.columns if c not in leading]
    out = out[leading + rest]
    return out


def range_validate(df):
    """Coerce out-of-range numeric values to NaN; write a per-row report."""
    flagged = []
    for col, (lo, hi) in RANGE_CHECKS.items():
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        for idx in df.index[bad]:
            flagged.append({
                "name": df.at[idx, "name"],
                "column": col,
                "value": df.at[idx, col],
                "expected_range": f"[{lo}, {hi}]",
            })
        df.loc[bad, col] = np.nan
    if flagged:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(flagged).to_csv(REPORT_PATH, index=False)
        print(f"[range_validate] coerced {len(flagged)} out-of-range values -> {REPORT_PATH}")
    else:
        print("[range_validate] no range violations")
    return df


def drop_missing_features(df):
    """Drop rows missing any required-feature column."""
    n0 = len(df)
    out = df.dropna(subset=REQUIRED_FEATURES).copy()
    print(f"[required_features] {n0} -> {len(out)}  "
          f"(dropped {n0 - len(out)} with NaN in any of {REQUIRED_FEATURES})")
    return out


def deduplicate(df):
    """One row per name. Prefer rows that have both targets present and more non-NaN columns."""
    n0 = len(df)
    df = df.assign(
        _has_tdp=df["tdp_w"].notna().astype(int),
        _has_psu=df["psu_w"].notna().astype(int),
        _nn=df.notna().sum(axis=1),
    )
    df = df.sort_values(
        by=["name", "_has_tdp", "_has_psu", "_nn"],
        ascending=[True, False, False, False],
    )
    df = df.drop_duplicates(subset="name", keep="first")
    df = df.drop(columns=["_has_tdp", "_has_psu", "_nn"])
    print(f"[dedupe] {n0} -> {len(df)}  (removed {n0 - len(df)} duplicate names)")
    return df


def report_targets(df):
    """Log how many rows have NaN targets (the prediction-only subset)."""
    n = len(df)
    tdp_nan = int(df["tdp_w"].isna().sum())
    psu_nan = int(df["psu_w"].isna().sum())
    both    = int(df[["tdp_w", "psu_w"]].notna().all(axis=1).sum())
    print(f"[targets] total={n}  both-present={both}  tdp NaN={tdp_nan}  psu NaN={psu_nan}")


def main():
    print("=" * 60)
    print("GPU SPECS CLEANING")
    print("=" * 60)
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    print(f"[load] {len(raw)} rows x {len(raw.columns)} cols  ({RAW_PATH})")

    df = filter_rows(raw)
    df = parse_columns(df)
    df = range_validate(df)
    df = drop_missing_features(df)
    df = deduplicate(df)
    report_targets(df)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[save] {len(df)} rows x {len(df.columns)} cols -> {OUT_PATH}")


if __name__ == "__main__":
    main()
