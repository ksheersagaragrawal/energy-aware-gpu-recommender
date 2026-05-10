"""
tests/test_cleaned_data.py
==========================
Pytest-based tests for the data-cleaning and validation pipeline.

These tests work entirely with synthetic (in-memory) DataFrames — no real
CSV files are required to run the test suite.

Run with:
    pytest tests/test_cleaned_data.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import (
    NUMERIC_GPU_COLS,
    GPU_SPECS_COLS,
    standardise_column_names,
    parse_memory_size,
    parse_clock_mhz,
    parse_power_watts,
    parse_integer_feature,
    drop_invalid_rows,
    clean_game_requirements,
    clean_gpu_specs,
)
from src.validation import (
    check_required_columns,
    check_numeric_columns,
    check_no_unexpected_nulls,
    check_psu_positive,
    check_memory_size_range,
    check_memory_clock_range,
    check_requirement_type_split,
    missing_value_summary,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic raw data
# ---------------------------------------------------------------------------


@pytest.fixture()
def raw_game_requirements_csv(tmp_path: pytest.TempPathFactory) -> str:
    """Write a minimal synthetic game-requirements CSV and return its path."""
    data = {
        "game_title": ["Game A", "Game B", "Game C", "Game D"],
        "min_gpu": ["GTX 1060", "RX 580", "GTX 970", None],
        "rec_gpu": ["RTX 2080", "RX 6700 XT", "RTX 3070", "RTX 4090"],
        "min_psu_watts": ["450 W", "500 W", "0", "350 W"],
        "rec_psu_watts": ["650 W", "700 W", "750 W", "850 W"],
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "videogame_requirements.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture()
def raw_gpu_specs_csv(tmp_path: pytest.TempPathFactory) -> str:
    """Write a minimal synthetic GPU-specs CSV and return its path."""
    data = {
        "gpu_name": ["RTX 3080", "RX 6800 XT", "GTX 1080 Ti", "Bad GPU"],
        "memory_size_mb": ["10 GB", "16 GB", "11264 MB", None],
        "memory_type": ["GDDR6X", "GDDR6", "GDDR5X", "GDDR6"],
        "memory_clock_mhz": ["9501 MHz", "2000 MHz", "1376 MHz", "1000 MHz"],
        "gpu_clock_mhz": ["1440 MHz", "1825 MHz", "1480 MHz", "1200 MHz"],
        "tmus": ["272", "288", "224", "0"],
        "rops": ["96", "128", "88", "0"],
        "shader_cores": ["8704", "4608", "3584", "512"],
        "psu_watts": ["320 W", "300 W", "250 W", "0 W"],
    }
    df = pd.DataFrame(data)
    filepath = tmp_path / "gpu_specs.csv"
    df.to_csv(filepath, index=False)
    return str(filepath)


@pytest.fixture()
def clean_gpu_df() -> pd.DataFrame:
    """Return a minimal already-clean GPU specs DataFrame."""
    return pd.DataFrame(
        {
            "gpu_name": ["RTX 3080", "RX 6800 XT", "GTX 1080 Ti"],
            "memory_size_mb": [10240.0, 16384.0, 11264.0],
            "memory_type": ["GDDR6X", "GDDR6", "GDDR5X"],
            "memory_clock_mhz": [9501.0, 2000.0, 1376.0],
            "gpu_clock_mhz": [1440.0, 1825.0, 1480.0],
            "tmus": [272.0, 288.0, 224.0],
            "rops": [96.0, 128.0, 88.0],
            "shader_cores": [8704.0, 4608.0, 3584.0],
            "psu_watts": [320.0, 300.0, 250.0],
        }
    )


@pytest.fixture()
def clean_games_min() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_title": ["Game A", "Game B"],
            "gpu_name": ["GTX 1060", "RX 580"],
            "psu_watts": [450.0, 500.0],
            "requirement_type": ["minimum", "minimum"],
        }
    )


@pytest.fixture()
def clean_games_rec() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_title": ["Game A", "Game B"],
            "gpu_name": ["RTX 2080", "RX 6700 XT"],
            "psu_watts": [650.0, 700.0],
            "requirement_type": ["recommended", "recommended"],
        }
    )


# ---------------------------------------------------------------------------
# Unit tests — parsing helpers
# ---------------------------------------------------------------------------


class TestParseMemorySize:
    def test_gb_string(self):
        assert parse_memory_size("8 GB") == pytest.approx(8192.0)

    def test_mb_string(self):
        assert parse_memory_size("512 MB") == pytest.approx(512.0)

    def test_plain_number(self):
        assert parse_memory_size("4096") == pytest.approx(4096.0)

    def test_nan_input(self):
        assert np.isnan(parse_memory_size(np.nan))

    def test_none_input(self):
        assert np.isnan(parse_memory_size(None))

    def test_unparseable(self):
        assert np.isnan(parse_memory_size("unknown"))

    def test_g_suffix(self):
        assert parse_memory_size("12G") == pytest.approx(12 * 1024)

    def test_m_suffix(self):
        assert parse_memory_size("2048M") == pytest.approx(2048.0)


class TestParseClockMhz:
    def test_mhz_string(self):
        assert parse_clock_mhz("1750 MHz") == pytest.approx(1750.0)

    def test_ghz_string(self):
        assert parse_clock_mhz("1.75 GHz") == pytest.approx(1750.0)

    def test_plain_number(self):
        assert parse_clock_mhz("2000") == pytest.approx(2000.0)

    def test_nan_input(self):
        assert np.isnan(parse_clock_mhz(np.nan))

    def test_unparseable(self):
        assert np.isnan(parse_clock_mhz("N/A"))


class TestParsePowerWatts:
    def test_watts_string(self):
        assert parse_power_watts("350 W") == pytest.approx(350.0)

    def test_watts_no_space(self):
        assert parse_power_watts("350W") == pytest.approx(350.0)

    def test_plain_number(self):
        assert parse_power_watts("750") == pytest.approx(750.0)

    def test_nan_input(self):
        assert np.isnan(parse_power_watts(np.nan))

    def test_unparseable(self):
        assert np.isnan(parse_power_watts("TBD"))


class TestParseIntegerFeature:
    def test_plain_integer(self):
        assert parse_integer_feature("272") == pytest.approx(272.0)

    def test_string_with_prefix(self):
        assert parse_integer_feature("x288") == pytest.approx(288.0)

    def test_nan(self):
        assert np.isnan(parse_integer_feature(np.nan))

    def test_no_digits(self):
        assert np.isnan(parse_integer_feature("N/A"))


# ---------------------------------------------------------------------------
# Unit tests — standardise_column_names
# ---------------------------------------------------------------------------


class TestStandardiseColumnNames:
    def test_lowercase(self):
        df = pd.DataFrame(columns=["GPU Name", "Memory Size"])
        result = standardise_column_names(df)
        assert list(result.columns) == ["gpu_name", "memory_size"]

    def test_spaces_to_underscores(self):
        df = pd.DataFrame(columns=["Memory Type", "GPU Clock"])
        result = standardise_column_names(df)
        assert list(result.columns) == ["memory_type", "gpu_clock"]

    def test_hyphens_to_underscores(self):
        df = pd.DataFrame(columns=["mem-size", "gpu-clock"])
        result = standardise_column_names(df)
        assert list(result.columns) == ["mem_size", "gpu_clock"]


# ---------------------------------------------------------------------------
# Unit tests — drop_invalid_rows
# ---------------------------------------------------------------------------


class TestDropInvalidRows:
    def test_drops_missing_required(self):
        df = pd.DataFrame({"name": ["A", None, "C"], "val": [1, 2, 3]})
        result = drop_invalid_rows(df, required_cols=["name"])
        assert len(result) == 2
        assert result["name"].notna().all()

    def test_drops_zero_in_zero_invalid_cols(self):
        df = pd.DataFrame({"name": ["A", "B", "C"], "psu": [350.0, 0.0, 500.0]})
        result = drop_invalid_rows(df, required_cols=["name"], zero_invalid_cols=["psu"])
        assert len(result) == 2
        assert (result["psu"] > 0).all()

    def test_keeps_valid_rows(self):
        df = pd.DataFrame({"name": ["A", "B"], "psu": [350.0, 500.0]})
        result = drop_invalid_rows(df, required_cols=["name"], zero_invalid_cols=["psu"])
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration tests — cleaning pipelines
# ---------------------------------------------------------------------------


class TestCleanGameRequirements:
    def test_returns_two_dataframes(self, raw_game_requirements_csv):
        result = clean_game_requirements(raw_game_requirements_csv)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_minimum_has_requirement_type(self, raw_game_requirements_csv):
        games_min, _ = clean_game_requirements(raw_game_requirements_csv)
        assert "requirement_type" in games_min.columns
        assert (games_min["requirement_type"] == "minimum").all()

    def test_recommended_has_requirement_type(self, raw_game_requirements_csv):
        _, games_rec = clean_game_requirements(raw_game_requirements_csv)
        assert "requirement_type" in games_rec.columns
        assert (games_rec["requirement_type"] == "recommended").all()

    def test_zero_psu_removed_from_min(self, raw_game_requirements_csv):
        games_min, _ = clean_game_requirements(raw_game_requirements_csv)
        if "psu_watts" in games_min.columns:
            non_null_psu = games_min["psu_watts"].dropna()
            assert (non_null_psu > 0).all()

    def test_psu_column_is_numeric(self, raw_game_requirements_csv):
        games_min, games_rec = clean_game_requirements(raw_game_requirements_csv)
        for df in (games_min, games_rec):
            if "psu_watts" in df.columns:
                assert pd.api.types.is_numeric_dtype(df["psu_watts"])

    def test_row_count_reduced(self, raw_game_requirements_csv):
        """Cleaning should remove at least one row (zero PSU or null GPU)."""
        games_min, _ = clean_game_requirements(raw_game_requirements_csv)
        # Raw had 4 rows; Game C has PSU=0 which should be dropped
        assert len(games_min) < 4


class TestCleanGpuSpecs:
    def test_returns_dataframe(self, raw_gpu_specs_csv):
        result = clean_gpu_specs(raw_gpu_specs_csv)
        assert isinstance(result, pd.DataFrame)

    def test_memory_size_in_mb(self, raw_gpu_specs_csv):
        df = clean_gpu_specs(raw_gpu_specs_csv)
        if "memory_size_mb" in df.columns:
            # 10 GB → 10240 MB
            rtx_row = df[df["gpu_name"] == "RTX 3080"]
            if not rtx_row.empty:
                assert rtx_row["memory_size_mb"].values[0] == pytest.approx(10240.0)

    def test_numeric_columns_are_numeric(self, raw_gpu_specs_csv):
        df = clean_gpu_specs(raw_gpu_specs_csv)
        for col in NUMERIC_GPU_COLS:
            if col in df.columns:
                assert pd.api.types.is_numeric_dtype(df[col]), f"{col} is not numeric"

    def test_zero_psu_removed(self, raw_gpu_specs_csv):
        df = clean_gpu_specs(raw_gpu_specs_csv)
        if "psu_watts" in df.columns:
            non_null = df["psu_watts"].dropna()
            assert (non_null > 0).all()

    def test_gpu_name_column_exists(self, raw_gpu_specs_csv):
        df = clean_gpu_specs(raw_gpu_specs_csv)
        assert "gpu_name" in df.columns


# ---------------------------------------------------------------------------
# Validation function tests
# ---------------------------------------------------------------------------


class TestCheckRequiredColumns:
    def test_all_present(self, clean_gpu_df):
        assert check_required_columns(clean_gpu_df, ["gpu_name", "psu_watts"]) is True

    def test_missing_column(self, clean_gpu_df):
        assert check_required_columns(clean_gpu_df, ["nonexistent_col"]) is False


class TestCheckNumericColumns:
    def test_numeric_dtype(self, clean_gpu_df):
        assert check_numeric_columns(clean_gpu_df, ["psu_watts", "tmus"]) is True

    def test_non_numeric_dtype(self):
        df = pd.DataFrame({"col": ["a", "b", "c"]})
        assert check_numeric_columns(df, ["col"]) is False

    def test_missing_column_skipped(self, clean_gpu_df):
        # A missing column should not fail the check (it is simply skipped)
        assert check_numeric_columns(clean_gpu_df, ["missing_col"]) is True


class TestCheckNoUnexpectedNulls:
    def test_no_nulls(self, clean_gpu_df):
        assert check_no_unexpected_nulls(clean_gpu_df, ["gpu_name", "psu_watts"]) is True

    def test_with_nulls(self):
        df = pd.DataFrame({"gpu_name": ["A", None], "psu_watts": [300.0, 250.0]})
        assert check_no_unexpected_nulls(df, ["gpu_name"]) is False


class TestCheckPsuPositive:
    def test_all_positive(self, clean_gpu_df):
        assert check_psu_positive(clean_gpu_df) is True

    def test_zero_psu(self):
        df = pd.DataFrame({"psu_watts": [0.0, 350.0]})
        assert check_psu_positive(df) is False

    def test_negative_psu(self):
        df = pd.DataFrame({"psu_watts": [-100.0, 350.0]})
        assert check_psu_positive(df) is False

    def test_missing_column(self, clean_gpu_df):
        df = clean_gpu_df.drop(columns=["psu_watts"])
        # Missing column → returns True (check skipped)
        assert check_psu_positive(df) is True


class TestCheckMemorySizeRange:
    def test_valid_range(self, clean_gpu_df):
        assert check_memory_size_range(clean_gpu_df) is True

    def test_too_small(self):
        df = pd.DataFrame({"memory_size_mb": [64.0]})
        assert check_memory_size_range(df) is False

    def test_too_large(self):
        df = pd.DataFrame({"memory_size_mb": [999_999.0]})
        assert check_memory_size_range(df) is False


class TestCheckMemoryClockRange:
    def test_valid_range(self, clean_gpu_df):
        assert check_memory_clock_range(clean_gpu_df) is True

    def test_too_low(self):
        df = pd.DataFrame({"memory_clock_mhz": [10.0]})
        assert check_memory_clock_range(df) is False

    def test_too_high(self):
        df = pd.DataFrame({"memory_clock_mhz": [100_000.0]})
        assert check_memory_clock_range(df) is False


class TestCheckRequirementTypeSplit:
    def test_correctly_split(self, clean_games_min, clean_games_rec):
        assert check_requirement_type_split(clean_games_min, clean_games_rec) is True

    def test_wrong_type_in_min(self, clean_games_min, clean_games_rec):
        bad_min = clean_games_min.copy()
        bad_min.loc[0, "requirement_type"] = "recommended"
        assert check_requirement_type_split(bad_min, clean_games_rec) is False

    def test_missing_column(self, clean_games_min, clean_games_rec):
        bad_min = clean_games_min.drop(columns=["requirement_type"])
        assert check_requirement_type_split(bad_min, clean_games_rec) is False


class TestMissingValueSummary:
    def test_returns_dataframe(self, clean_gpu_df):
        summary = missing_value_summary(clean_gpu_df)
        assert isinstance(summary, pd.DataFrame)
        assert "column" in summary.columns
        assert "null_count" in summary.columns
        assert "null_pct" in summary.columns

    def test_zero_nulls_for_clean_df(self, clean_gpu_df):
        summary = missing_value_summary(clean_gpu_df)
        assert summary["null_count"].sum() == 0

    def test_detects_nulls(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 3]})
        summary = missing_value_summary(df)
        assert summary.loc[summary["column"] == "b", "null_count"].values[0] == 2
