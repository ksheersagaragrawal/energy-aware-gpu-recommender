"""Build Phase 4 artifact audit and related-work tables."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT / "src"))

from phase4_external_benchmark import get_default_paths
from phase4_reporting import build_artifact_audit_table, build_related_work_table


def main() -> None:
    paths = get_default_paths()
    build_artifact_audit_table(paths.tables_dir / "phase4_artifact_audit.csv")
    build_related_work_table(paths.tables_dir / "phase4_related_work_positioning.csv")


if __name__ == "__main__":
    main()
