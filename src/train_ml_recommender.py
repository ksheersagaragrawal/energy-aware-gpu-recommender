"""
Trains an XGBoost regression model to predict GPU G3D Mark score from
hardware specifications.

Output: models/gpu_performance_model.pkl
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

dataset_path = "data/training/gpu_benchmark_dataset.csv"
model_path   = "models/gpu_performance_model.pkl"

memory_types = [
    "DDR", "DDR2", "DDR3", "SDR",
    "GDDR2", "GDDR3", "GDDR4", "GDDR5", "GDDR5X",
    "GDDR6", "GDDR6X", "GDDR7",
    "HBM", "HBM2", "HBM2e", "HBM3",
]

continuous_features = [
    "process_nm", "tmus", "rops", "texture_rate", "pixel_rate",
    "direct_x", "memory_mb", "memory_speed_mhz", "memory_bandwidth_gbs", "tdp_w",
]

mem_features = [f"mem_{m.lower()}" for m in memory_types]
all_features = continuous_features + mem_features


# loads the training dataset, fills missing values, and splits into train/test
def load_and_split():
    df = pd.read_csv(dataset_path)
    df = df.dropna(subset=["g3d_mark", "texture_rate", "pixel_rate", "tmus", "rops"])

    df[mem_features]        = df[mem_features].fillna(0)
    df[continuous_features] = df[continuous_features].apply(lambda col: col.fillna(col.median()))

    X = df[all_features].values.astype(float)
    y = df["g3d_mark"].values.astype(float)

    return train_test_split(X, y, test_size=0.2, random_state=42)


# prints MAE, RMSE, and R² for a given set of predictions
def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:6s}  MAE={mae:7.1f}  RMSE={rmse:7.1f}  R²={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


# trains the XGBoost model, prints evaluation and feature importances, saves to disk
def train():
    X_train, X_test, y_train, y_test = load_and_split()

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    model.fit(X_train, y_train)

    print("Model evaluation:")
    evaluate("Train", y_train, model.predict(X_train))
    test_metrics = evaluate("Test",  y_test,  model.predict(X_test))

    print("\nTop-10 feature importances:")
    feat_imp = sorted(zip(all_features, model.feature_importances_), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]:
        print(f"  {feat:30s}: {imp:.4f}")

    payload = {
        "model":         model,
        "feature_cols":  all_features,
        "mem_type_cols": mem_features,
        "test_metrics":  test_metrics,
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nModel saved → {model_path}")


if __name__ == "__main__":
    train()
