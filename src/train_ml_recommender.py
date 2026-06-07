"""
Trains an XGBoost model to predict GPU G3D Mark (PassMark benchmark score)
from hardware specs.

At recommendation time:
  - Hard/soft filter GPUs by game requirements
  - Predict G3D Mark for each passing GPU
  - Rank by predicted_G3D / TDP  (energy-efficient performance)

Model is saved to: models/gpu_performance_model.pkl

Usage:
    python src/train_ml_recommender.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

ROOT           = Path(__file__).resolve().parent.parent
DATASET_PATH   = ROOT / "data" / "training" / "gpu_benchmark_dataset.csv"
MODELS_DIR     = ROOT / "models"
MODEL_PATH     = MODELS_DIR / "gpu_performance_model.pkl"

MEMORY_TYPE_CATEGORIES = [
    "DDR", "DDR2", "DDR3", "SDR",
    "GDDR2", "GDDR3", "GDDR4", "GDDR5", "GDDR5X",
    "GDDR6", "GDDR6X", "GDDR7",
    "HBM", "HBM2", "HBM2e", "HBM3",
]

CONTINUOUS_FEATURES = [
    "process_nm",
    "tmus",
    "rops",
    "texture_rate",
    "pixel_rate",
    "direct_x",
    "memory_mb",
    "memory_speed_mhz",
    "memory_bandwidth_gbs",
    "tdp_w",
]

MEM_TYPE_FEATURES = [f"mem_{c.lower()}" for c in MEMORY_TYPE_CATEGORIES]
ALL_FEATURES      = CONTINUOUS_FEATURES + MEM_TYPE_FEATURES
TARGET            = "g3d_mark"


def load_and_split(test_size=0.2, random_state=42):
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset: {len(df)} GPUs, {df.columns.tolist()}")

    # Fill missing memory type one-hots with 0
    for col in MEM_TYPE_FEATURES:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0)

    # Drop rows missing target or critical continuous features
    df = df.dropna(subset=[TARGET] + ["texture_rate", "pixel_rate", "tmus", "rops", "tdp_w"])
    print(f"After dropna: {len(df)} rows")

    # Fill remaining missing continuous features with column median
    for col in CONTINUOUS_FEATURES:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[ALL_FEATURES].values.astype(float)
    y = df[TARGET].values.astype(float)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=test_size, random_state=random_state
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")
    return X_train, X_test, y_train, y_test, df


def evaluate(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:30s}  MAE={mae:7.1f}  RMSE={rmse:7.1f}  R²={r2:.4f}")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train():
    X_train, X_test, y_train, y_test, df = load_and_split()

    # Fit scaler on training data (used for feature reporting, XGBoost doesn't need it)
    scaler = StandardScaler()
    scaler.fit(X_train[:, :len(CONTINUOUS_FEATURES)])  # scale continuous only

    # XGBoost — hyperparameters tuned for G3D Mark range (~37 to ~40000)
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

    print("\nTraining XGBoost model...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    print("\nModel evaluation:")
    train_metrics = evaluate("Train", y_train, train_pred)
    test_metrics  = evaluate("Test",  y_test,  test_pred)

    # Feature importance
    print("\nTop-10 feature importances:")
    importances = model.feature_importances_
    feat_imp = sorted(zip(ALL_FEATURES, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imp[:10]:
        print(f"  {feat:30s}: {imp:.4f}")

    # Save model + metadata
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model":           model,
        "scaler":          scaler,
        "feature_cols":    ALL_FEATURES,
        "continuous_cols": CONTINUOUS_FEATURES,
        "mem_type_cols":   MEM_TYPE_FEATURES,
        "target":          TARGET,
        "test_metrics":    test_metrics,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nModel saved → {MODEL_PATH}")

    return model, test_metrics


if __name__ == "__main__":
    train()