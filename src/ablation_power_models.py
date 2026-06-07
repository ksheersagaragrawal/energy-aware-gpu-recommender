"""Run feature-set ablations for GPU power prediction.

This script reuses the vector dataset produced by build_gpu_power_vectors.py and the
best model hyperparameters from train_gpu_specs_models.py metrics. The only change
across runs is the feature subset.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost.core import XGBoostError


DEFAULT_INPUT_PATH = "data/vectors/gpu_power_vectors.csv"
DEFAULT_RESULTS_DIR = "data/results"
DEFAULT_TDP_METRICS = "data/results/tdp_model_metrics.csv"
DEFAULT_PSU_METRICS = "data/results/psu_model_metrics.csv"
DEFAULT_RANDOM_STATE = 42

FULL_FEATURE_SET = [
    "process_nm",
    "tmus",
    "rops",
    "texture_rate",
    "pixel_rate",
    "direct_x",
    "memory_mb",
    "memory_speed_mhz",
    "memory_bandwidth_gbs",
]

FEATURE_SET_DEFS = {
    "Full": FULL_FEATURE_SET,
    "Memory-only": [
        "memory_mb",
        "memory_bandwidth_gbs",
    ],
    "Core-only": [
        "process_nm",
        "tmus",
        "rops",
    ],
    "Clocks-only": [
        "memory_speed_mhz",
        "texture_rate",
        "pixel_rate",
    ],
    "Minimal": [
        "process_nm",
        "memory_mb",
        "memory_bandwidth_gbs",
    ],
}

TARGETS = ["tdp_w", "psu_w"]
MODELS = ["Gradient Boosting", "XGBoost"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: Dict[str, object]


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def load_power_vector_dataset(path: str) -> pd.DataFrame:
    """Load the GPU power vector dataset."""
    return pd.read_csv(path)


def _memory_type_one_hot_cols(df: pd.DataFrame) -> List[str]:
    """Return the memory-type indicator columns used by the current vector dataset."""
    raw_cols = [col for col in df.columns if col.startswith("memory_type_raw_")]
    if raw_cols:
        return raw_cols
    return [col for col in df.columns if col.startswith("memory_type_") and col != "memory_type"]


def get_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Return feature sets with dynamic memory type one-hot columns for Full."""
    memory_type_cols = _memory_type_one_hot_cols(df)
    feature_sets = {}
    for name, cols in FEATURE_SET_DEFS.items():
        if name == "Full":
            feature_sets[name] = cols + memory_type_cols
        else:
            feature_sets[name] = cols.copy()
    return feature_sets


def _parse_params(params_text: str) -> Dict[str, object]:
    if not params_text:
        return {}
    try:
        return ast.literal_eval(params_text)
    except (ValueError, SyntaxError):
        raise ValueError(f"Could not parse params: {params_text}")


def _load_model_params_for_target(metrics_path: str, model_name: str, target: str) -> Dict[str, object]:
    """Load best params for a target from the metrics CSV for a given model."""
    metrics = pd.read_csv(metrics_path)
    model_rows = metrics[metrics["model"] == model_name].copy()
    if model_rows.empty:
        raise ValueError(f"No metrics found for model '{model_name}' in {metrics_path}")

    target_rows = model_rows[model_rows["target"] == target]
    if target_rows.empty:
        raise ValueError(f"No metrics found for target '{target}' in {metrics_path}")
    best_row = target_rows.sort_values("test_mae").iloc[0]
    return _parse_params(str(best_row["params"]))


def get_model_specs(tdp_metrics: str, psu_metrics: str) -> Dict[str, Dict[str, ModelSpec]]:
    """Return model specs keyed by target then model name."""
    specs: Dict[str, Dict[str, ModelSpec]] = {target: {} for target in TARGETS}

    tdp_params_gb = _load_model_params_for_target(tdp_metrics, "Gradient Boosting", "tdp_w")
    psu_params_gb = _load_model_params_for_target(psu_metrics, "Gradient Boosting", "psu_w")
    tdp_params_xgb = _load_model_params_for_target(tdp_metrics, "XGBoost", "tdp_w")
    psu_params_xgb = _load_model_params_for_target(psu_metrics, "XGBoost", "psu_w")

    specs["tdp_w"]["Gradient Boosting"] = ModelSpec("Gradient Boosting", tdp_params_gb)
    specs["psu_w"]["Gradient Boosting"] = ModelSpec("Gradient Boosting", psu_params_gb)
    specs["tdp_w"]["XGBoost"] = ModelSpec("XGBoost", tdp_params_xgb)
    specs["psu_w"]["XGBoost"] = ModelSpec("XGBoost", psu_params_xgb)

    return specs


def split_data(df: pd.DataFrame, target_col: str, features: Sequence[str], random_state: int) -> SplitData:
    """Match the train/val/test split from train_gpu_specs_models.py."""
    y = df[target_col]
    X = df[list(features)]
    X_train, X_combo, y_train, y_combo = train_test_split(
        X, y, test_size=0.3, random_state=random_state
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_combo, y_combo, test_size=1 / 3, random_state=random_state
    )
    return SplitData(X_train=X_train, X_val=X_val, X_test=X_test, y_train=y_train, y_val=y_val, y_test=y_test)


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> Tuple[float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def _make_model(model_spec: ModelSpec, use_gpu: bool) -> Tuple[object, bool]:
    if model_spec.name == "Gradient Boosting":
        return GradientBoostingRegressor(**model_spec.params), False

    params = dict(model_spec.params)
    if use_gpu:
        params = {**params, "tree_method": "gpu_hist", "predictor": "gpu_predictor"}
    return XGBRegressor(**params), use_gpu


def _fit_xgboost_with_fallback(
    model_spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_gpu: bool,
) -> Tuple[XGBRegressor, bool]:
    model, using_gpu = _make_model(model_spec, use_gpu)
    try:
        model.fit(X_train, y_train)
        return model, using_gpu
    except XGBoostError:
        if not use_gpu:
            raise
        cpu_model, _ = _make_model(model_spec, use_gpu=False)
        cpu_model.fit(X_train, y_train)
        return cpu_model, False


def evaluate_model(
    model_spec: ModelSpec,
    split: SplitData,
    use_gpu: bool,
) -> Dict[str, float]:
    X_train_full = pd.concat([split.X_train, split.X_val], axis=0)
    y_train_full = pd.concat([split.y_train, split.y_val], axis=0)

    if model_spec.name == "XGBoost":
        model, used_gpu = _fit_xgboost_with_fallback(model_spec, X_train_full, y_train_full, use_gpu)
    else:
        model, used_gpu = _make_model(model_spec, use_gpu)
        model.fit(X_train_full, y_train_full)

    train_pred = model.predict(X_train_full)
    test_pred = model.predict(split.X_test)

    train_mae, train_rmse, train_r2 = _evaluate_predictions(y_train_full, train_pred)
    test_mae, test_rmse, test_r2 = _evaluate_predictions(split.y_test, test_pred)

    return {
        "train_mae": train_mae,
        "train_rmse": train_rmse,
        "train_r2": train_r2,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "used_gpu": used_gpu,
    }


def validate_feature_set(df: pd.DataFrame, feature_set: Sequence[str], feature_set_name: str) -> None:
    missing = [col for col in feature_set if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for feature set '{feature_set_name}': {missing}")


def run_single_ablation(
    df: pd.DataFrame,
    target: str,
    feature_set_name: str,
    features: Sequence[str],
    model_spec: ModelSpec,
    random_state: int,
    use_gpu: bool,
) -> Dict[str, object]:
    validate_feature_set(df, features, feature_set_name)

    subset = df[list(features) + [target]].dropna()
    if subset.empty:
        raise ValueError(f"No rows left after dropping NaNs for {feature_set_name} / {target}")

    split = split_data(subset, target, features, random_state)
    metrics = evaluate_model(model_spec, split, use_gpu)

    result = {
        "target": target,
        "model": model_spec.name,
        "feature_set": feature_set_name,
        "num_features": len(features),
        "feature_names": "|".join(features),
        "train_mae": metrics["train_mae"],
        "test_mae": metrics["test_mae"],
        "train_rmse": metrics["train_rmse"],
        "test_rmse": metrics["test_rmse"],
        "train_r2": metrics["train_r2"],
        "test_r2": metrics["test_r2"],
        "train_size": len(split.X_train) + len(split.X_val),
        "test_size": len(split.X_test),
        "used_gpu": metrics["used_gpu"],
    }

    print(
        f"[ablation] target={target} model={model_spec.name} "
        f"feature_set={feature_set_name} features={len(features)} "
        f"test_mae={metrics['test_mae']:.3f} test_rmse={metrics['test_rmse']:.3f} "
        f"test_r2={metrics['test_r2']:.3f}"
    )

    return result


def run_all_ablations(
    df: pd.DataFrame,
    feature_sets: Dict[str, List[str]],
    model_specs: Dict[str, Dict[str, ModelSpec]],
    random_state: int,
    use_gpu: bool,
    n_jobs: int,
) -> pd.DataFrame:
    jobs: List[Tuple[str, str, str, List[str], ModelSpec]] = []
    for target in TARGETS:
        for model_name in MODELS:
            for feature_set_name, features in feature_sets.items():
                jobs.append((target, model_name, feature_set_name, features, model_specs[target][model_name]))

    results: List[Dict[str, object]] = []

    if use_gpu:
        for target, model_name, feature_set_name, features, model_spec in jobs:
            results.append(
                run_single_ablation(
                    df,
                    target,
                    feature_set_name,
                    features,
                    model_spec,
                    random_state,
                    use_gpu=True,
                )
            )
        return pd.DataFrame(results)

    if n_jobs == 1:
        for target, model_name, feature_set_name, features, model_spec in jobs:
            results.append(
                run_single_ablation(
                    df,
                    target,
                    feature_set_name,
                    features,
                    model_spec,
                    random_state,
                    use_gpu=False,
                )
            )
        return pd.DataFrame(results)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=None if n_jobs < 0 else n_jobs) as executor:
        futures = [
            executor.submit(
                run_single_ablation,
                df,
                target,
                feature_set_name,
                features,
                model_spec,
                random_state,
                False,
            )
            for target, model_name, feature_set_name, features, model_spec in jobs
        ]
        for future in as_completed(futures):
            results.append(future.result())

    return pd.DataFrame(results)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_results(df: pd.DataFrame, output_dir: Path) -> None:
    metrics_path = output_dir / "ablation_power_model_metrics.csv"
    summary_path = output_dir / "ablation_power_model_summary.csv"
    best_path = output_dir / "ablation_power_model_best_by_target_model.csv"
    insights_path = output_dir / "ablation_power_model_feature_group_insights.csv"

    _save_csv(df, metrics_path)

    summary = df[[
        "target",
        "model",
        "feature_set",
        "num_features",
        "test_mae",
        "test_rmse",
        "test_r2",
    ]].copy()
    summary = summary.sort_values(["target", "model", "test_mae"], ascending=[True, True, True])
    _save_csv(summary, summary_path)

    best_rows = summary.groupby(["target", "model"], as_index=False).first()
    best_rows = best_rows.rename(columns={"feature_set": "best_feature_set"})
    _save_csv(best_rows, best_path)

    full_rows = df[df["feature_set"] == "Full"]
    if full_rows.empty:
        raise ValueError("Full feature set results missing; cannot compute insights table.")

    insights = df.merge(
        full_rows[["target", "model", "test_mae", "test_r2"]],
        on=["target", "model"],
        suffixes=("_ablated", "_full"),
    )
    insights["mae_diff_from_full"] = insights["test_mae_ablated"] - insights["test_mae_full"]
    insights["mae_pct_change_from_full"] = (
        insights["mae_diff_from_full"] / insights["test_mae_full"] * 100.0
    )
    insights["r2_diff_from_full"] = insights["test_r2_ablated"] - insights["test_r2_full"]

    insights = insights[[
        "target",
        "model",
        "feature_set",
        "test_mae_full",
        "test_mae_ablated",
        "mae_diff_from_full",
        "mae_pct_change_from_full",
        "test_r2_full",
        "test_r2_ablated",
        "r2_diff_from_full",
    ]].sort_values(["target", "model", "mae_diff_from_full"])

    _save_csv(insights, insights_path)


def print_ranked_summary(df: pd.DataFrame) -> None:
    summary = df[[
        "target",
        "model",
        "feature_set",
        "test_mae",
        "test_rmse",
        "test_r2",
        "num_features",
    ]].copy()

    for target in TARGETS:
        for model in MODELS:
            subset = summary[(summary["target"] == target) & (summary["model"] == model)]
            if subset.empty:
                continue
            subset = subset.sort_values("test_mae")
            print(f"\nTarget: {target} | Model: {model}")
            print("Rank  Feature Set           Test MAE    Test RMSE   Test R2    Num Features")
            for i, row in enumerate(subset.itertuples(index=False), start=1):
                print(
                    f"{i:<5} {row.feature_set:<20} "
                    f"{row.test_mae:>9.3f} {row.test_rmse:>11.3f} "
                    f"{row.test_r2:>8.3f} {row.num_features:>12}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run power-model feature ablations.")
    parser.add_argument("--input-path", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tdp-metrics", default=DEFAULT_TDP_METRICS)
    parser.add_argument("--psu-metrics", default=DEFAULT_PSU_METRICS)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)

    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--use-gpu", action="store_true", help="Use GPU for XGBoost when available.")
    gpu_group.add_argument("--no-gpu", action="store_true", help="Force CPU mode.")

    args = parser.parse_args()

    use_gpu = True
    if args.no_gpu:
        use_gpu = False
    if args.use_gpu:
        use_gpu = True

    df = load_power_vector_dataset(args.input_path)
    feature_sets = get_feature_sets(df)

    if "Full" in feature_sets and not any(col.startswith("memory_type_") for col in feature_sets["Full"]):
        raise ValueError("Full feature set requires memory_type_* one-hot columns, but none were found.")

    model_specs = get_model_specs(args.tdp_metrics, args.psu_metrics)

    results = run_all_ablations(
        df=df,
        feature_sets=feature_sets,
        model_specs=model_specs,
        random_state=args.random_state,
        use_gpu=use_gpu,
        n_jobs=args.n_jobs,
    )

    output_dir = Path(args.output_dir)
    save_results(results, output_dir)
    print_ranked_summary(results)

    print("\nSaved ablation outputs:")
    print(f"  {output_dir / 'ablation_power_model_metrics.csv'}")
    print(f"  {output_dir / 'ablation_power_model_summary.csv'}")
    print(f"  {output_dir / 'ablation_power_model_best_by_target_model.csv'}")
    print(f"  {output_dir / 'ablation_power_model_feature_group_insights.csv'}")


if __name__ == "__main__":
    main()
