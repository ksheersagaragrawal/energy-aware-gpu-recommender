"""Build requirement vectors, performance vectors, and normalized performance
scores from cleaned game requirements data.

Adds hard/soft filter counts, min-max-normalized perf features, and a
geometric-mean perf_score. Column names are kept as produced by the cleaning
step — they already match the GPU-side canonical naming, so the feasibility
filter is a column-name comparison rather than a translation.

Usage:
    python src/build_vectors.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLEANED_DIR = ROOT / "data" / "cleaned"
OUT_DIR = ROOT / "data" / "vectors"

HARD_FILTER_COLS = ["memory_mb", "direct_x"]
SOFT_FILTER_COLS = ["texture_rate", "pixel_rate", "memory_bandwidth_gbs", "tmus", "rops"]
PERF_COLS = [
    "texture_rate",
    "pixel_rate",
    "memory_bandwidth_gbs",
    "tmus",
    "rops",
    "memory_speed_mhz",
    "boost_clock_mhz",
]

EPSILON = 1e-6


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def load_cleaned():
    df_min = pd.read_csv(CLEANED_DIR / "game_reqs_min.csv")
    df_recom = pd.read_csv(CLEANED_DIR / "game_reqs_recom.csv")
    assert df_min.shape == (7292, 26), f"min shape unexpected: {df_min.shape}"
    assert df_recom.shape == (7292, 26), f"recom shape unexpected: {df_recom.shape}"
    print(f"[load] min: {df_min.shape}, recom: {df_recom.shape}")
    return df_min, df_recom


def add_filter_counts(df):
    df["hard_filter_count"] = df[HARD_FILTER_COLS].notna().sum(axis=1).astype(int)
    df["soft_filter_count"] = df[SOFT_FILTER_COLS].notna().sum(axis=1).astype(int)
    return df


def normalize_perf_features(df):
    ranges = {}
    for col in PERF_COLS:
        vals = df[col].dropna()
        col_min = vals.min()
        col_max = vals.max()
        ranges[col] = (col_min, col_max)

        norm_col = f"norm_{col}"
        if col_max > col_min:
            df[norm_col] = (df[col] - col_min) / (col_max - col_min)
            df[norm_col] = df[norm_col].clip(lower=EPSILON, upper=1.0)
        else:
            df[norm_col] = np.where(df[col].notna(), 1.0, np.nan)

    print(f"[normalize] normalization ranges:")
    for col, (cmin, cmax) in ranges.items():
        print(f"  {col:20s}  min={cmin:10.2f}  max={cmax:10.2f}")
    return df


def compute_perf_score(df):
    norm_cols = [f"norm_{col}" for col in PERF_COLS]

    log_vals = df[norm_cols].apply(np.log)
    df["perf_feature_count"] = df[norm_cols].notna().sum(axis=1).astype(int)
    log_mean = log_vals.mean(axis=1, skipna=True)
    df["perf_score"] = np.exp(log_mean)
    # Rows with 0 features get NaN (log_mean is NaN when all values are NaN)

    print(f"[perf_score] computed for {(df['perf_score'].notna()).sum()} / {len(df)} rows")
    return df


def validate(df_min, df_recom):
    for label, df in [("min", df_min), ("recom", df_recom)]:
        # Row count
        assert len(df) == 7292, f"{label} row count: {len(df)}"

        # perf_score in [0, 1] or NaN
        scores = df["perf_score"].dropna()
        assert (scores >= 0).all(), f"{label} has negative perf_score"
        assert (scores <= 1.0 + EPSILON).all(), f"{label} has perf_score > 1"

        # perf_feature_count in [0, 7]
        assert df["perf_feature_count"].between(0, 7).all(), f"{label} perf_feature_count out of range"

        # norm_* in [EPSILON, 1.0] or NaN
        for col in PERF_COLS:
            norm_col = f"norm_{col}"
            vals = df[norm_col].dropna()
            if len(vals) > 0:
                assert (vals >= EPSILON - 1e-10).all(), f"{label} {norm_col} below epsilon"
                assert (vals <= 1.0 + 1e-10).all(), f"{label} {norm_col} above 1.0"

        # filter counts in range
        assert df["hard_filter_count"].between(0, 2).all(), f"{label} hard_filter_count out of range"
        assert df["soft_filter_count"].between(0, 5).all(), f"{label} soft_filter_count out of range"

        # Column count
        assert len(df.columns) == 37, f"{label} has {len(df.columns)} columns, expected 37"

    # Same game sets
    assert len(df_min) == len(df_recom), "row counts differ"
    assert set(df_min["name"]) == set(df_recom["name"]), "game sets differ"

    print("[validate] all assertions passed")


def print_summary(df_min, df_recom):
    print("\n" + "=" * 60)
    print("VECTOR BUILD SUMMARY")
    print("=" * 60)

    for label, df in [("MIN", df_min), ("RECOM", df_recom)]:
        print(f"\n--- {label} dataset: {df.shape[0]} rows x {df.shape[1]} columns ---")

        scores = df["perf_score"].dropna()
        print(f"\nPerformance score distribution (n={len(scores)}):")
        print(f"  min={scores.min():.4f}  p25={scores.quantile(0.25):.4f}  "
              f"median={scores.median():.4f}  p75={scores.quantile(0.75):.4f}  max={scores.max():.4f}")

        print(f"\nFeature count distribution:")
        vc = df["perf_feature_count"].value_counts().sort_index()
        for count, n in vc.items():
            print(f"  {count} features: {n} games ({n/len(df)*100:.1f}%)")

        print(f"\nFilter count distribution:")
        print(f"  hard_filter_count: mean={df['hard_filter_count'].mean():.2f}")
        print(f"  soft_filter_count: mean={df['soft_filter_count'].mean():.2f}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    df_min, df_recom = load_cleaned()

    results = {}
    for label, df in [("min", df_min), ("recom", df_recom)]:
        print(f"\n--- Processing {label} ---")
        df = add_filter_counts(df)
        df = normalize_perf_features(df)
        df = compute_perf_score(df)
        results[label] = df

    df_min, df_recom = results["min"], results["recom"]
    validate(df_min, df_recom)
    print_summary(df_min, df_recom)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_min.to_csv(OUT_DIR / "game_vectors_min.csv", index=False)
    df_recom.to_csv(OUT_DIR / "game_vectors_recom.csv", index=False)
    print(f"\nSaved to {OUT_DIR / 'game_vectors_min.csv'}")
    print(f"Saved to {OUT_DIR / 'game_vectors_recom.csv'}")


if __name__ == "__main__":
    main()
