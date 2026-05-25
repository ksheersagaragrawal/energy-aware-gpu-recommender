"""Phase 4: External benchmark preparation, modeling, and recommendation evaluation."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.ensemble import RandomForestRegressor

from phase4_feature_sets import (
    FEATURE_SET_ORDER,
    FEATURE_SET_RUNTIME,
    build_feature_matrix,
    select_feature_columns,
)


SOURCE_REPO_COL = "source_repo"
GPU_NAME_COL = "gpu_name"
BENCH_COL = "bench"
APP_COL = "app"
DATASET_COL = "dataset"
LSEQ_COL = "lseq"
POWER_COL = "power_w"
TIME_COL = "time_ms"
SPLIT_COL = "split"
ROW_ID_COL = "row_id"


@dataclass(frozen=True)
class BenchmarkPaths:
    repo_root: Path
    external_dir: Path
    processed_dir: Path
    tables_dir: Path


@dataclass(frozen=True)
class BenchmarkConfig:
    random_seed: int = 42
    test_size: float = 0.2
    power_budget_percentile: int = 75
    perf_floor_percentile: int = 50


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    df: pd.DataFrame
    group_col: Optional[str]
    available_targets: List[str]


@dataclass(frozen=True)
class ModelResult:
    model_name: str
    backend: str
    train_time_sec: float
    infer_time_sec: float
    predictions: np.ndarray


def get_default_paths() -> BenchmarkPaths:
    repo_root = Path(__file__).resolve().parents[1]
    return BenchmarkPaths(
        repo_root=repo_root,
        external_dir=repo_root / "external",
        processed_dir=repo_root / "data" / "processed",
        tables_dir=repo_root / "results" / "tables",
    )


def list_external_files(repo_dir: Path) -> List[Path]:
    if not repo_dir.exists():
        return []
    patterns = ["**/*.csv", "**/*.sqlite", "**/*.db"]
    files = []
    for pattern in patterns:
        files.extend(repo_dir.glob(pattern))
    return sorted({path for path in files if path.is_file()})


def _safe_read_csv(path: Path, max_rows: int | None = None) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path, nrows=max_rows)
    except Exception:
        return None


def _read_sqlite_table(path: Path, table: str) -> Optional[pd.DataFrame]:
    try:
        conn = sqlite3.connect(str(path))
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        conn.close()
        return df
    except Exception:
        return None


def _find_sqlite_tables(path: Path) -> List[str]:
    try:
        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        return tables
    except Exception:
        return []


def _pick_best_dataframe(dataframes: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if not dataframes:
        return None
    dataframes = [df for df in dataframes if df is not None and not df.empty]
    if not dataframes:
        return None
    return max(dataframes, key=lambda df: df.shape[0] * df.shape[1])


def _normalize_col(col: str) -> str:
    return col.strip().lower()


def _guess_column(columns: Iterable[str], patterns: Iterable[str]) -> Optional[str]:
    lowered = {_normalize_col(col): col for col in columns}
    for pattern in patterns:
        for low, original in lowered.items():
            if pattern in low:
                return original
    return None


def build_canonical_gpu_mangrove(
    source_df: pd.DataFrame,
    source_repo: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = source_df.copy()
    columns = list(df.columns)

    mapping = {
        GPU_NAME_COL: _guess_column(columns, ["gpu_name", "gpu", "device"]),
        BENCH_COL: _guess_column(columns, ["bench", "benchmark"]),
        APP_COL: _guess_column(columns, ["app", "application", "kernel"]),
        DATASET_COL: _guess_column(columns, ["dataset", "input", "problem"]),
        LSEQ_COL: _guess_column(columns, ["lseq", "sequence"]),
        POWER_COL: _guess_column(columns, ["power", "watt"]),
        TIME_COL: _guess_column(columns, ["time", "runtime", "duration"]),
    }

    canonical = pd.DataFrame({
        SOURCE_REPO_COL: source_repo,
        GPU_NAME_COL: df[mapping[GPU_NAME_COL]] if mapping[GPU_NAME_COL] else np.nan,
        BENCH_COL: df[mapping[BENCH_COL]] if mapping[BENCH_COL] else np.nan,
        APP_COL: df[mapping[APP_COL]] if mapping[APP_COL] else np.nan,
        DATASET_COL: df[mapping[DATASET_COL]] if mapping[DATASET_COL] else np.nan,
        LSEQ_COL: df[mapping[LSEQ_COL]] if mapping[LSEQ_COL] else np.nan,
        POWER_COL: df[mapping[POWER_COL]] if mapping[POWER_COL] else np.nan,
        TIME_COL: df[mapping[TIME_COL]] if mapping[TIME_COL] else np.nan,
        SPLIT_COL: np.nan,
    })

    used_columns = {col for col in mapping.values() if col}
    feature_columns = [col for col in df.columns if col not in used_columns]

    feature_df = pd.DataFrame(index=df.index)
    for col in feature_columns:
        safe_name = _normalize_col(col).replace(" ", "_")
        feature_df[f"feature_{safe_name}"] = df[col].values

    canonical = pd.concat([canonical, feature_df], axis=1)
    canonical[ROW_ID_COL] = np.arange(len(canonical))

    schema_rows = []
    for canonical_col, source_col in mapping.items():
        schema_rows.append({
            "canonical_col": canonical_col,
            "source_col": source_col or "",
            "notes": "" if source_col else "missing",
        })

    for col in feature_columns:
        safe_name = _normalize_col(col).replace(" ", "_")
        schema_rows.append({
            "canonical_col": f"feature_{safe_name}",
            "source_col": col,
            "notes": "feature",
        })

    schema_df = pd.DataFrame(schema_rows)
    return canonical, schema_df


def load_gpu_mangrove_dataset(processed_path: Path) -> Optional[DatasetBundle]:
    if not processed_path.exists():
        return None
    df = pd.read_csv(processed_path)
    if df.empty:
        return None

    group_col = select_group_column(df)
    available_targets = [col for col in [POWER_COL, TIME_COL] if col in df.columns]
    return DatasetBundle(name="gpu_mangrove", df=df, group_col=group_col, available_targets=available_targets)


def select_group_column(df: pd.DataFrame) -> Optional[str]:
    for col in [APP_COL, BENCH_COL, DATASET_COL]:
        if col in df.columns and df[col].nunique(dropna=True) > 1:
            return col
    return None


def split_train_test(
    df: pd.DataFrame,
    group_col: Optional[str],
    test_size: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray, str]:
    if group_col and df[group_col].nunique(dropna=True) > 1:
        groups = df[group_col].fillna("unknown").astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(splitter.split(df, groups=groups))
        return train_idx, test_idx, f"group_{group_col}"

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=test_size,
        random_state=random_seed,
    )
    return train_idx, test_idx, "random"


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | None]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    if np.all(y_true > 0):
        mape = np.mean(np.abs((y_true - y_pred) / y_true))
    else:
        mape = None

    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def train_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    random_seed: int,
) -> ModelResult:
    try:
        import xgboost as xgb  # type: ignore

        params = {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_lambda": 1.0,
            "objective": "reg:squarederror",
            "random_state": random_seed,
        }

        def _fit(local_params: Dict[str, object]) -> ModelResult:
            start = time.time()
            model = xgb.XGBRegressor(**local_params)
            model.fit(X_train, y_train)
            train_time = time.time() - start
            infer_start = time.time()
            preds = model.predict(X_test)
            infer_time = time.time() - infer_start
            backend = local_params.get("device", "cpu")
            return ModelResult("XGBoost", str(backend), train_time, infer_time, preds)

        try:
            return _fit(params | {"tree_method": "hist", "device": "cuda"})
        except Exception:
            try:
                return _fit(params | {"tree_method": "gpu_hist"})
            except Exception:
                return _fit(params)

    except Exception:
        start = time.time()
        model = RandomForestRegressor(
            n_estimators=500,
            n_jobs=-1,
            random_state=random_seed,
        )
        model.fit(X_train, y_train)
        train_time = time.time() - start
        infer_start = time.time()
        preds = model.predict(X_test)
        infer_time = time.time() - infer_start
        return ModelResult("RandomForest", "cpu", train_time, infer_time, preds)


def train_feature_set_models(
    dataset: DatasetBundle,
    feature_set: str,
    target_col: str,
    config: BenchmarkConfig,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    df = dataset.df.copy()

    feature_cols = select_feature_columns(df, feature_set, target_columns=[target_col])
    features, _ = build_feature_matrix(df, feature_cols)
    y = df[target_col].values

    train_idx, test_idx, split_method = split_train_test(
        df,
        dataset.group_col,
        config.test_size,
        config.random_seed,
    )

    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model_result = train_regressor(X_train, pd.Series(y_train), X_test, config.random_seed)

    print(
        "[phase4]"
        f" dataset={dataset.name} target={target_col} feature_set={feature_set}"
        f" model={model_result.model_name} backend={model_result.backend}"
        f" train_sec={model_result.train_time_sec:.2f} infer_sec={model_result.infer_time_sec:.2f}"
    )

    metrics = _compute_metrics(y_test, model_result.predictions)

    metrics_row = {
        "dataset": dataset.name,
        "target": target_col,
        "feature_set": feature_set,
        "model": model_result.model_name,
        "backend": model_result.backend,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "r2": metrics["r2"],
        "mape": metrics["mape"],
        "train_time_sec": model_result.train_time_sec,
        "infer_time_sec": model_result.infer_time_sec,
        "notes": split_method,
    }

    pred_rows = pd.DataFrame({
        ROW_ID_COL: df[ROW_ID_COL].values[test_idx],
        "dataset": dataset.name,
        "target": target_col,
        "feature_set": feature_set,
        "model": model_result.model_name,
        "backend": model_result.backend,
        SPLIT_COL: "test",
        "prediction": model_result.predictions,
    })

    return metrics_row, pred_rows


def build_prediction_metrics(
    dataset: DatasetBundle,
    config: BenchmarkConfig,
    feature_sets: Iterable[str] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    feature_sets = list(feature_sets or FEATURE_SET_ORDER)
    metrics_rows = []
    prediction_rows = []

    for target_col in dataset.available_targets:
        if dataset.df[target_col].dropna().empty:
            continue
        for feature_set in feature_sets:
            metrics_row, pred_rows = train_feature_set_models(
                dataset,
                feature_set,
                target_col,
                config,
            )
            metrics_rows.append(metrics_row)
            prediction_rows.append(pred_rows)

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(prediction_rows, ignore_index=True) if prediction_rows else pd.DataFrame()
    return metrics_df, preds_df


def _label_relevance(ppw: pd.Series) -> pd.Series:
    if ppw.empty:
        return pd.Series(dtype=int)
    p80, p60, p30 = np.percentile(ppw, [80, 60, 30])
    labels = pd.Series(0, index=ppw.index)
    labels[ppw >= p30] = 1
    labels[ppw >= p60] = 2
    labels[ppw >= p80] = 3
    return labels


def run_recommendation_benchmark(
    dataset: DatasetBundle,
    predictions: pd.DataFrame,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    df = dataset.df.copy()
    df = df.merge(predictions, on=[ROW_ID_COL], how="inner")

    methods = [
        "LowestPowerTop5",
        "PowerTop5",
        "UtilityFormulaTop5",
        "MLUtilityTop5",
        "LTRUtilityTop5",
    ]

    results = []

    for feature_set in predictions["feature_set"].unique():
        df_feature = df[df["feature_set"] == feature_set].copy()

        power_pred = df_feature[df_feature["target"] == POWER_COL][[ROW_ID_COL, "prediction"]].rename(
            columns={"prediction": "pred_power"}
        )
        time_pred = df_feature[df_feature["target"] == TIME_COL][[ROW_ID_COL, "prediction"]].rename(
            columns={"prediction": "pred_time"}
        )

        base = dataset.df.copy()
        base = base.merge(power_pred, on=ROW_ID_COL, how="left")
        if not time_pred.empty:
            base = base.merge(time_pred, on=ROW_ID_COL, how="left")

        if POWER_COL not in base.columns or TIME_COL not in base.columns:
            continue

        base = base.dropna(subset=[POWER_COL, TIME_COL])
        if base.empty:
            continue

        base = base.copy()
        base["true_perf"] = 1.0 / base[TIME_COL].replace(0, np.nan)
        base["true_ppw"] = base["true_perf"] / base[POWER_COL].replace(0, np.nan)

        if base["true_perf"].isna().all():
            continue

        group_col = dataset.group_col or APP_COL
        if group_col not in base.columns:
            base[group_col] = "all"

        ml_scores = None
        ltr_scores = None
        train_idx, test_idx, _ = split_train_test(
            base,
            group_col,
            config.test_size,
            config.random_seed,
        )

        if len(test_idx) > 0:
            features_cols = select_feature_columns(
                base,
                feature_set,
                target_columns=[POWER_COL, TIME_COL],
            )
            feature_matrix, _ = build_feature_matrix(base, features_cols)

            try:
                target = base["true_ppw"].values
                X_train = feature_matrix.iloc[train_idx]
                y_train = target[train_idx]
                X_test = feature_matrix.iloc[test_idx]
                ml_result = train_regressor(X_train, pd.Series(y_train), X_test, config.random_seed)
                ml_scores = pd.Series(ml_result.predictions, index=base.index[test_idx])
            except Exception:
                ml_scores = None

            try:
                import xgboost as xgb  # type: ignore

                group_train = base.iloc[train_idx][group_col].fillna("unknown").astype(str)
                group_test = base.iloc[test_idx][group_col].fillna("unknown").astype(str)

                train_sizes = group_train.value_counts().sort_index()
                test_sizes = group_test.value_counts().sort_index()

                if train_sizes.sum() >= 20 and len(train_sizes) >= 2:
                    y_labels = _label_relevance(base["true_ppw"])
                    dtrain = xgb.DMatrix(
                        feature_matrix.iloc[train_idx],
                        label=y_labels.iloc[train_idx].values,
                    )
                    dtest = xgb.DMatrix(feature_matrix.iloc[test_idx])

                    dtrain.set_group(train_sizes.values)
                    params = {
                        "objective": "rank:ndcg",
                        "learning_rate": 0.1,
                        "max_depth": 6,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "random_state": config.random_seed,
                    }

                    try:
                        params_gpu = params | {"tree_method": "hist", "device": "cuda"}
                        model = xgb.train(params_gpu, dtrain, num_boost_round=300)
                    except Exception:
                        try:
                            params_gpu = params | {"tree_method": "gpu_hist"}
                            model = xgb.train(params_gpu, dtrain, num_boost_round=300)
                        except Exception:
                            model = xgb.train(params, dtrain, num_boost_round=300)

                    ltr_scores = pd.Series(model.predict(dtest), index=base.index[test_idx])
            except Exception:
                ltr_scores = None

        for group, group_df in base.groupby(group_col):
            group_df = group_df.copy()
            if group_df["true_perf"].dropna().empty:
                continue
            perf_floor = np.nanpercentile(group_df["true_perf"], config.perf_floor_percentile)
            feasible = group_df[group_df["true_perf"] >= perf_floor].copy()
            if feasible.empty:
                continue

            power_budget = np.nanpercentile(feasible[POWER_COL], config.power_budget_percentile)

            relevance = _label_relevance(feasible["true_ppw"])

            for method in methods:
                ranked = feasible.copy()
                score_col = None

                if method == "LowestPowerTop5":
                    if "pred_power" not in ranked.columns:
                        continue
                    ranked = ranked.dropna(subset=["pred_power"])
                    ranked["score"] = -ranked["pred_power"]
                    score_col = "score"
                    ranked = ranked.sort_values("pred_power", ascending=True)
                elif method == "PowerTop5":
                    if "pred_power" not in ranked.columns or "pred_time" not in ranked.columns:
                        continue
                    ranked = ranked.dropna(subset=["pred_power", "pred_time"])
                    ranked["pred_perf"] = 1.0 / ranked["pred_time"].replace(0, np.nan)
                    ranked["pred_ppw"] = ranked["pred_perf"] / ranked["pred_power"].replace(0, np.nan)
                    ranked["score"] = ranked["pred_ppw"]
                    score_col = "score"
                    ranked = ranked.sort_values("pred_ppw", ascending=False)
                elif method == "UtilityFormulaTop5":
                    if "pred_power" not in ranked.columns or "pred_time" not in ranked.columns:
                        continue
                    ranked = ranked.dropna(subset=["pred_power", "pred_time"])
                    ranked["pred_perf"] = 1.0 / ranked["pred_time"].replace(0, np.nan)
                    ranked["pred_ppw"] = ranked["pred_perf"] / ranked["pred_power"].replace(0, np.nan)

                    def _minmax(series: pd.Series) -> pd.Series:
                        s_min, s_max = series.min(), series.max()
                        if pd.isna(s_min) or pd.isna(s_max) or s_min == s_max:
                            return pd.Series(0.5, index=series.index)
                        return (series - s_min) / (s_max - s_min)

                    ranked["ppw_score"] = _minmax(ranked["pred_ppw"])
                    ranked["perf_score"] = _minmax(ranked["pred_perf"])
                    ranked["low_power_score"] = 1.0 - _minmax(ranked["pred_power"])
                    ranked["power_slack"] = (power_budget - ranked["pred_power"]).clip(lower=0.0)
                    ranked["power_slack_score"] = _minmax(ranked["power_slack"])

                    ranked["utility_score"] = (
                        0.50 * ranked["ppw_score"]
                        + 0.25 * ranked["perf_score"]
                        + 0.15 * ranked["low_power_score"]
                        + 0.10 * ranked["power_slack_score"]
                    )
                    ranked["score"] = ranked["utility_score"]
                    score_col = "score"
                    ranked = ranked.sort_values("utility_score", ascending=False)
                elif method == "MLUtilityTop5":
                    if ml_scores is None:
                        continue
                    ranked = ranked[ranked.index.isin(ml_scores.index)].copy()
                    ranked["score"] = ml_scores.loc[ranked.index]
                    score_col = "score"
                    ranked = ranked.sort_values("score", ascending=False)
                elif method == "LTRUtilityTop5":
                    if ltr_scores is None:
                        continue
                    ranked = ranked[ranked.index.isin(ltr_scores.index)].copy()
                    ranked["score"] = ltr_scores.loc[ranked.index]
                    score_col = "score"
                    ranked = ranked.sort_values("score", ascending=False)
                else:
                    continue

                topk = ranked.head(5)
                if topk.empty:
                    continue

                true_ppw_at_5 = topk["true_ppw"].mean()
                true_power_at_5 = topk[POWER_COL].mean()
                true_perf_at_5 = topk["true_perf"].mean()

                best_ppw = feasible["true_ppw"].max()
                efficiency_regret = best_ppw - true_ppw_at_5

                top1_gpu = topk[GPU_NAME_COL].iloc[0] if GPU_NAME_COL in topk.columns else str(topk.index[0])

                if score_col is not None:
                    ndcg = ndcg_score(
                        [relevance.loc[ranked.index].values],
                        [ranked[score_col].fillna(0.0).values],
                        k=5,
                    )
                else:
                    ndcg = None

                results.append({
                    "dataset": dataset.name,
                    "feature_set": feature_set,
                    "method": method,
                    "group": str(group),
                    "ndcg_at_5": ndcg,
                    "recall_at_5": 1.0 if (relevance.loc[topk.index] == 3).any() else 0.0,
                    "feasible_hit_rate": 1.0,
                    "mean_true_ppw_at_5": true_ppw_at_5,
                    "mean_true_power_at_5": true_power_at_5,
                    "mean_true_perf_at_5": true_perf_at_5,
                    "efficiency_regret_at_5": efficiency_regret,
                    "top1_candidate": top1_gpu,
                })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)
    summary_rows = []

    for (dataset_name, feature_set, method), subset in results_df.groupby([
        "dataset",
        "feature_set",
        "method",
    ]):
        top1_counts = subset["top1_candidate"].value_counts(normalize=True)
        top1_share = top1_counts.iloc[0] if not top1_counts.empty else np.nan
        unique_candidates = subset["top1_candidate"].nunique()

        summary_rows.append({
            "dataset": dataset_name,
            "feature_set": feature_set,
            "method": method,
            "ndcg_at_5": subset["ndcg_at_5"].mean(),
            "recall_at_5": subset["recall_at_5"].mean(),
            "feasible_hit_rate": subset["feasible_hit_rate"].mean(),
            "mean_true_ppw_at_5": subset["mean_true_ppw_at_5"].mean(),
            "mean_true_power_at_5": subset["mean_true_power_at_5"].mean(),
            "mean_true_perf_at_5": subset["mean_true_perf_at_5"].mean(),
            "efficiency_regret_at_5": subset["efficiency_regret_at_5"].mean(),
            "top1_share": top1_share,
            "unique_candidates": unique_candidates,
        })

    return pd.DataFrame(summary_rows)
