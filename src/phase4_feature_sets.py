"""Phase 4: Feature-set selection for external benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import pandas as pd


FEATURE_SET_STATIC = "static_only"
FEATURE_SET_STATIC_PROXY = "static_plus_proxy"
FEATURE_SET_RUNTIME = "runtime_full"
FEATURE_SET_ORDER = [FEATURE_SET_STATIC, FEATURE_SET_STATIC_PROXY, FEATURE_SET_RUNTIME]


DEFAULT_TARGET_COLUMNS = {
    "power",
    "power_w",
    "power_watt",
    "avg_power",
    "time",
    "time_ms",
    "runtime",
    "duration",
}


STATIC_PATTERNS = (
    "gpu",
    "device",
    "arch",
    "vendor",
    "family",
    "sm",
    "cores",
    "clock",
    "mem",
    "vram",
    "bandwidth",
    "tmu",
    "rop",
)

PROXY_PATTERNS = (
    "dataset",
    "input",
    "problem",
    "size",
    "batch",
    "grid",
    "block",
    "workgroup",
    "tile",
    "dim",
    "shape",
)

RUNTIME_PATTERNS = (
    "counter",
    "ipc",
    "throughput",
    "inst",
    "instruction",
    "occupancy",
    "util",
    "warp",
    "stall",
    "cache",
    "dram",
    "gld",
    "gst",
)

GROUP_PATTERNS = ("app", "bench", "benchmark", "kernel", "dataset")


@dataclass(frozen=True)
class FeatureRoleSets:
    target_cols: List[str]
    id_cols: List[str]
    group_cols: List[str]
    static_cols: List[str]
    proxy_cols: List[str]
    runtime_cols: List[str]
    categorical_cols: List[str]


def _normalize_columns(columns: Iterable[str]) -> Dict[str, str]:
    return {col: col.strip().lower() for col in columns}


def _matches_pattern(col: str, patterns: Iterable[str]) -> bool:
    return any(pattern in col for pattern in patterns)


def infer_feature_roles(df: pd.DataFrame, target_columns: Iterable[str] | None = None) -> FeatureRoleSets:
    target_columns = set(target_columns or DEFAULT_TARGET_COLUMNS)
    col_map = _normalize_columns(df.columns)

    target_cols = [col for col, low in col_map.items() if low in target_columns]
    id_cols = [col for col, low in col_map.items() if low.endswith("_id") or low in {"id", "row_id"}]
    group_cols = [col for col, low in col_map.items() if _matches_pattern(low, GROUP_PATTERNS)]

    static_cols = []
    proxy_cols = []
    runtime_cols = []

    for col, low in col_map.items():
        if col in target_cols or col in id_cols:
            continue
        if low.startswith("feature_static_"):
            static_cols.append(col)
            continue
        if low.startswith("feature_proxy_"):
            proxy_cols.append(col)
            continue
        if low.startswith("feature_runtime_"):
            runtime_cols.append(col)
            continue
        if _matches_pattern(low, STATIC_PATTERNS):
            static_cols.append(col)
        elif _matches_pattern(low, PROXY_PATTERNS):
            proxy_cols.append(col)
        elif _matches_pattern(low, RUNTIME_PATTERNS):
            runtime_cols.append(col)

    categorical_cols = [
        col
        for col in df.columns
        if df[col].dtype == "object" or df[col].dtype.name.startswith("category")
    ]

    return FeatureRoleSets(
        target_cols=target_cols,
        id_cols=id_cols,
        group_cols=group_cols,
        static_cols=sorted(set(static_cols)),
        proxy_cols=sorted(set(proxy_cols)),
        runtime_cols=sorted(set(runtime_cols)),
        categorical_cols=sorted(set(categorical_cols)),
    )


def select_feature_columns(
    df: pd.DataFrame,
    feature_set: str,
    target_columns: Iterable[str] | None = None,
) -> List[str]:
    roles = infer_feature_roles(df, target_columns)
    candidate_cols: List[str] = []

    if feature_set == FEATURE_SET_STATIC:
        candidate_cols = roles.static_cols + roles.group_cols
    elif feature_set == FEATURE_SET_STATIC_PROXY:
        candidate_cols = roles.static_cols + roles.proxy_cols + roles.group_cols
    elif feature_set == FEATURE_SET_RUNTIME:
        candidate_cols = [
            col
            for col in df.columns
            if col not in roles.target_cols and col not in roles.id_cols
        ]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    candidate_cols = [col for col in candidate_cols if col not in roles.target_cols and col not in roles.id_cols]

    if not candidate_cols:
        candidate_cols = [
            col
            for col in df.columns
            if col not in roles.target_cols and col not in roles.id_cols
        ]

    return sorted(set(candidate_cols))


def build_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    feature_df = df[feature_cols].copy()

    categorical_cols = [
        col
        for col in feature_cols
        if feature_df[col].dtype == "object" or feature_df[col].dtype.name.startswith("category")
    ]

    if categorical_cols:
        feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=True)

    feature_df = feature_df.fillna(feature_df.median(numeric_only=True))
    feature_names = list(feature_df.columns)
    return feature_df, feature_names
