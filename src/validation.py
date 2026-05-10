"""
src/validation.py
=================
Dataset-validation helpers for the energy-aware GPU recommender project.

Each public function either:
  - Returns ``True`` / ``False`` (predicate style), or
  - Prints a summary to stdout and returns a dict of results.

These are used directly in tests/test_cleaned_data.py and in the EDA notebook.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column presence
# ---------------------------------------------------------------------------


def check_required_columns(df: pd.DataFrame, required_cols: list[str]) -> bool:
    """
    Return ``True`` if all *required_cols* exist in *df*.

    Logs a warning listing any missing columns.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        logger.warning("Missing required columns: %s", missing)
        return False
    return True


# ---------------------------------------------------------------------------
# Numeric dtype checks
# ---------------------------------------------------------------------------


def check_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> bool:
    """
    Return ``True`` if every column in *numeric_cols* (that exists in *df*)
    has a numeric dtype.

    Logs a warning for each non-numeric column found.
    """
    all_ok = True
    for col in numeric_cols:
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            logger.warning("Column '%s' is not numeric (dtype=%s)", col, df[col].dtype)
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Null checks
# ---------------------------------------------------------------------------


def check_no_unexpected_nulls(
    df: pd.DataFrame,
    required_cols: list[str],
) -> bool:
    """
    Return ``True`` if none of the *required_cols* contain NaN values.

    Logs each column that contains nulls.
    """
    all_ok = True
    for col in required_cols:
        if col not in df.columns:
            continue
        null_count = df[col].isna().sum()
        if null_count > 0:
            logger.warning("Column '%s' has %d unexpected null(s).", col, null_count)
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Domain-specific range checks
# ---------------------------------------------------------------------------


def check_psu_positive(df: pd.DataFrame, psu_col: str = "psu_watts") -> bool:
    """
    Return ``True`` if all non-null values in *psu_col* are strictly positive.
    """
    if psu_col not in df.columns:
        logger.warning("Column '%s' not found — skipping PSU check.", psu_col)
        return True
    non_null = df[psu_col].dropna()
    invalid = (non_null <= 0).sum()
    if invalid:
        logger.warning("%d row(s) have non-positive PSU values.", invalid)
        return False
    return True


def check_memory_size_range(
    df: pd.DataFrame,
    col: str = "memory_size_mb",
    min_mb: float = 128,
    max_mb: float = 131_072,  # 128 GB upper bound
) -> bool:
    """
    Return ``True`` if all non-null values in *col* are within [*min_mb*, *max_mb*].

    Default range: 128 MB – 128 GB.
    """
    if col not in df.columns:
        logger.warning("Column '%s' not found — skipping memory-size range check.", col)
        return True
    non_null = df[col].dropna()
    out_of_range = ((non_null < min_mb) | (non_null > max_mb)).sum()
    if out_of_range:
        logger.warning(
            "%d row(s) have memory_size_mb outside [%s, %s].",
            out_of_range,
            min_mb,
            max_mb,
        )
        return False
    return True


def check_memory_clock_range(
    df: pd.DataFrame,
    col: str = "memory_clock_mhz",
    min_mhz: float = 100,
    max_mhz: float = 50_000,
) -> bool:
    """
    Return ``True`` if all non-null values in *col* are within [*min_mhz*, *max_mhz*].

    Default range: 100 MHz – 50 000 MHz (50 GHz).
    """
    if col not in df.columns:
        logger.warning("Column '%s' not found — skipping memory-clock range check.", col)
        return True
    non_null = df[col].dropna()
    out_of_range = ((non_null < min_mhz) | (non_null > max_mhz)).sum()
    if out_of_range:
        logger.warning(
            "%d row(s) have memory_clock_mhz outside [%s, %s].",
            out_of_range,
            min_mhz,
            max_mhz,
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Requirement-type split check
# ---------------------------------------------------------------------------


def check_requirement_type_split(
    games_min: pd.DataFrame,
    games_rec: pd.DataFrame,
) -> bool:
    """
    Return ``True`` if:
      - Both DataFrames have a ``requirement_type`` column.
      - *games_min* contains only ``'minimum'`` values.
      - *games_rec* contains only ``'recommended'`` values.
    """
    all_ok = True
    for df, expected_type, label in [
        (games_min, "minimum", "games_min"),
        (games_rec, "recommended", "games_rec"),
    ]:
        if "requirement_type" not in df.columns:
            logger.warning("'%s' is missing 'requirement_type' column.", label)
            all_ok = False
            continue
        wrong = df[df["requirement_type"] != expected_type]
        if len(wrong):
            logger.warning(
                "'%s' has %d row(s) with unexpected requirement_type values.",
                label,
                len(wrong),
            )
            all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising null counts and percentages per column.

    Columns: ``column``, ``null_count``, ``null_pct``.
    """
    null_counts = df.isna().sum()
    null_pct = (null_counts / len(df) * 100).round(2)
    summary = pd.DataFrame(
        {"column": null_counts.index, "null_count": null_counts.values, "null_pct": null_pct.values}
    )
    return summary.sort_values("null_count", ascending=False).reset_index(drop=True)


def run_all_validations(
    gpu_specs: pd.DataFrame,
    games_min: pd.DataFrame,
    games_rec: pd.DataFrame,
    gpu_required_cols: list[str],
    game_required_cols: list[str],
    numeric_gpu_cols: list[str],
) -> dict[str, bool]:
    """
    Run all validation checks and return a results dict.

    Keys are human-readable check names; values are ``True`` (pass) / ``False`` (fail).

    Parameters
    ----------
    gpu_specs : pd.DataFrame
        Cleaned GPU specs DataFrame.
    games_min : pd.DataFrame
        Cleaned minimum game requirements DataFrame.
    games_rec : pd.DataFrame
        Cleaned recommended game requirements DataFrame.
    gpu_required_cols : list[str]
        Columns that must exist and be non-null in *gpu_specs*.
    game_required_cols : list[str]
        Columns that must exist and be non-null in *games_min* / *games_rec*.
    numeric_gpu_cols : list[str]
        Columns in *gpu_specs* that must have numeric dtypes.

    Returns
    -------
    dict[str, bool]
    """
    results: dict[str, bool] = {}

    results["gpu_specs_required_columns"] = check_required_columns(gpu_specs, gpu_required_cols)
    results["games_min_required_columns"] = check_required_columns(games_min, game_required_cols)
    results["games_rec_required_columns"] = check_required_columns(games_rec, game_required_cols)

    results["gpu_specs_numeric_columns"] = check_numeric_columns(gpu_specs, numeric_gpu_cols)

    results["gpu_specs_psu_positive"] = check_psu_positive(gpu_specs, "psu_watts")
    results["games_min_psu_positive"] = check_psu_positive(games_min, "psu_watts")
    results["games_rec_psu_positive"] = check_psu_positive(games_rec, "psu_watts")

    results["gpu_memory_size_range"] = check_memory_size_range(gpu_specs)
    results["gpu_memory_clock_range"] = check_memory_clock_range(gpu_specs)

    results["requirement_type_split"] = check_requirement_type_split(games_min, games_rec)

    # Print summary table
    print("\n=== Validation Results ===")
    pad = max(len(k) for k in results) + 2
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check:<{pad}} {status}")
    overall = all(results.values())
    print(f"\n  Overall: {'ALL CHECKS PASSED' if overall else 'SOME CHECKS FAILED'}")

    return results
