"""Prepare external benchmarks for Phase 4."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from phase4_external_benchmark import (
    BenchmarkPaths,
    build_canonical_gpu_mangrove,
    get_default_paths,
    list_external_files,
    _safe_read_csv,
    _find_sqlite_tables,
    _read_sqlite_table,
)


def _load_best_external_dataframe(repo_dir: Path) -> pd.DataFrame:
    candidates = []
    for path in list_external_files(repo_dir):
        if path.suffix == ".csv":
            df = _safe_read_csv(path)
            if df is not None and not df.empty:
                candidates.append(df)
        elif path.suffix in {".sqlite", ".db"}:
            tables = _find_sqlite_tables(path)
            for table in tables:
                df = _read_sqlite_table(path, table)
                if df is not None and not df.empty:
                    candidates.append(df)
    if not candidates:
        return pd.DataFrame()
    return max(candidates, key=lambda df: df.shape[0] * df.shape[1])


def main() -> None:
    paths = get_default_paths()

    gpu_repo = paths.external_dir / "gpu-mangrove"
    canonical_path = paths.processed_dir / "phase4_gpu_mangrove_power.csv"
    schema_path = paths.tables_dir / "phase4_gpu_mangrove_schema_map.csv"

    if gpu_repo.exists():
        source_df = _load_best_external_dataframe(gpu_repo)
        if not source_df.empty:
            canonical, schema = build_canonical_gpu_mangrove(source_df, "gpu-mangrove")
            canonical.to_csv(canonical_path, index=False)
            schema.to_csv(schema_path, index=False)
            return

    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(columns=[
        "source_repo",
        "gpu_name",
        "bench",
        "app",
        "dataset",
        "lseq",
        "power_w",
        "time_ms",
        "split",
        "row_id",
    ]).to_csv(canonical_path, index=False)

    pd.DataFrame(columns=["canonical_col", "source_col", "notes"]).to_csv(schema_path, index=False)


if __name__ == "__main__":
    main()
