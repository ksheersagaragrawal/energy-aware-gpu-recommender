"""train GPU TDP and PSU prediction models
loads GPU power-model vectors and trains linear regression, ridge regression, lasso regression,
random forest regresor, gradient boosted regresor, XGBoost and MLP models for:
    1. tdp prediction
    2. psu prediction

Hyperparameters are tuned manually using validation MAE
models are evaluated on a held-out test set using MAE, RMSE, and R2

python src/train_gpu_specs_models.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

RESULTS_OUT = "data/results"
METRICS_OUT = "data/results/power_model_metrics.csv"
PREDICTIONS_OUT = "data/results/gpu_power_predictions.csv"
TDP_METRICS_OUT = "data/results/tdp_model_metrics.csv"
PSU_METRICS_OUT = "data/results/psu_model_metrics.csv"
BEST_MODEL_SUMMARY_OUT = "data/results/best_power_model_summary.csv"

def load_vectors():
    """load data"""
    return pd.read_csv("data/vectors/gpu_power_vectors.csv")


def split(df, target_col, features):
    """train test split"""
    y = df[target_col]
    temp = df.copy()
    X = temp[features]
    X_train, X_combo, y_train, y_combo = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_combo, y_combo, test_size=1/3, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate(y_true, y_pred):
    """ mae, mse, and r^2"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rsq = r2_score(y_true, y_pred)

    return mae, mse, rsq


def finetuning_with_hyperparams(model, params, X_train, X_val, y_train, y_val, X_test, y_test, df, features):
    """ training for all imported models after we've decided on hyperparams """
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    tuned_model = model(**params)
    tuned_model.fit(X_full, y_full)
    temp=df.copy()
    df_feat = temp[features]
    test_pred = tuned_model.predict(X_test)
    df_preds = tuned_model.predict(df_feat)

    test_mae, test_rmse, test_rsq = evaluate(y_test, test_pred)

    return test_mae, test_rmse, test_rsq, df_preds


def lin_reg(X_train, y_train, X_val, y_val, X_test, y_test, df, features):
    """train Linear Regression, note: doesnt need hyperparameter tuning"""
    model = LinearRegression()
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    val_mae = mean_absolute_error(y_val, val_pred)
    test_mae, test_mse, test_rsq, df_preds = finetuning_with_hyperparams(LinearRegression, {}, X_train, X_val, y_train, y_val, X_test, y_test, df, features)
    return ("Linear Regression", {}, val_mae, test_mae, test_mse, test_rsq, df_preds)



def ridge_reg(X_train, y_train, X_val, y_val, X_test, y_test, df, features):
    """find the best hyperparams and train ridge regression"""
    alpha = 0
    best_score = 100000000000

    for a in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0]:
        model = Ridge(alpha=a)
        model.fit(X_train, y_train)
        vpreds = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, vpreds)

        if val_mae < best_score:
            best_score = val_mae
            alpha = a

    test_mae, test_mse, test_rsq, df_preds = finetuning_with_hyperparams(Ridge, {"alpha": alpha}, X_train, X_val, y_train, y_val, X_test, y_test, df, features)

    return ("Ridge Regression", {"alpha": alpha}, best_score, test_mae, test_mse, test_rsq, df_preds)

def lasso_reg(X_train, y_train, X_val, y_val, X_test, y_test, df, features):
    """find the best hyperparams and train lasso regression"""
    alpha = 0
    best_score = 100000000000

    for a in [0.0001, 0.001, 0.01, 0.1, 0.5, 0.75, 1.0]:
        model = Lasso( alpha=a, max_iter=10000)
        model.fit(X_train, y_train)
        vpreds = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, vpreds)

        if val_mae < best_score:
            best_score = val_mae
            alpha = a

    test_mae, test_mse, test_rsq, df_preds = finetuning_with_hyperparams(Lasso, {"alpha": alpha, "max_iter": 10000}, X_train, X_val, y_train, y_val, X_test, y_test, df, features)

    return ("Lasso Regression", {"alpha": alpha, "max_iter": 10000}, best_score, test_mae, test_mse, test_rsq, df_preds)

def random_forest(X_train, y_train, X_val, y_val, X_test, y_test, df, features):
    """find the best hyperparams and a random forest"""
    best_set = {}
    best_score = 100000000000

    for n_estimators in [300, 500, 800]:
        for max_depth in [None, 10, 20, 30]:
            for min_samples_leaf in [1, 2, 4]:
                for max_features in ["sqrt", 0.7, 1.0]:
                    params = { "n_estimators": n_estimators, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "max_features": max_features, "n_jobs": -1,  "random_state": 42}
                    model = RandomForestRegressor(**params)
                    model.fit(X_train, y_train)
                    vpreds = model.predict(X_val)
                    val_mae = mean_absolute_error(y_val, vpreds)

                    if val_mae < best_score:
                        best_score = val_mae
                        best_set = params

    test_mae, test_mse, test_rsq, df_preds = finetuning_with_hyperparams(RandomForestRegressor,best_set,  X_train, X_val, y_train, y_val, X_test, y_test, df, features)

    return ("Random Forest", best_set, best_score, test_mae, test_mse, test_rsq, df_preds)



def gradient_boosting(X_train, y_train, X_val, y_val, X_test, y_test, df, features):
    """find the best hyperparams and a Gradient Boosting model"""

    best_set = {}
    best_score = 100000000000

    for n_estimators in [300, 500, 800]:
        for learning_rate in [0.03, 0.05, 0.075, 0.1]:
            for max_depth in [3, 4, 5, 6]:
                for min_samples_leaf in [1, 2, 5]:
                    params = { "n_estimators": n_estimators, "learning_rate": learning_rate, "max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "random_state": 42}
                    model = GradientBoostingRegressor(**params)
                    model.fit(X_train, y_train)

                    val_pred = model.predict(X_val)
                    val_mae = mean_absolute_error(y_val, val_pred)

                    if val_mae < best_score:
                        best_score = val_mae
                        best_set = params

    test_mae, test_mse, test_rsq, df_preds = finetuning_with_hyperparams(GradientBoostingRegressor,best_set,  X_train, X_val, y_train, y_val, X_test, y_test, df, features)
    return ("Gradient Boosting", best_set, best_score, test_mae, test_mse, test_rsq, df_preds)


def xgboost(X_train, y_train, X_val, y_val, X_test, y_test, df, features):
    """find the best hyperparams and a XGBoost model"""
    best_set = {}
    best_score = 100000000000

    for n_estimators in [300, 500, 800]:
        for learning_rate in [0.03, 0.05, 0.075, 0.1]:
            for max_depth in [4, 5, 6]:
                for min_child_weight in [1, 3, 5]:
                    for reg_lambda in [1.0, 5.0, 10.0]:
                        params = { "objective": "reg:squarederror",  "subsample": 0.9,  "colsample_bytree": 0.9, "n_estimators": n_estimators, "learning_rate": learning_rate,  "max_depth": max_depth, "min_child_weight": min_child_weight, "reg_lambda": reg_lambda, "random_state": 42}
                        model = XGBRegressor(**params)
                        model.fit(X_train, y_train)
                        val_pred = model.predict(X_val)
                        val_mae = mean_absolute_error(y_val, val_pred)

                        if val_mae < best_score:
                            best_score = val_mae
                            best_set = params

    test_mae, test_mse, test_rsq, df_preds = finetuning_with_hyperparams(XGBRegressor,best_set,  X_train, X_val, y_train, y_val, X_test, y_test, df, features)
    return ("XGBoost", best_set, best_score, test_mae, test_mse, test_rsq, df_preds)

class MLPModel(nn.Module):
    """MLP custom implementation (heavily inspired by ECE 228 HW 2)"""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: int = (64, 32),
        output_dim: int = 1,
        dropout: float = 0.1,
    ):
        super(MLPModel, self).__init__()

        self.layer_1 = nn.Linear(input_dim, hidden_layers[0])
        self.activation_function_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(dropout)

        self.layer_2 = nn.Linear(hidden_layers[0], hidden_layers[1])
        self.activation_function_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

        self.layer_out = nn.Linear(hidden_layers[1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_layer_1 = self.layer_1(x)
        x_active_1 = self.activation_function_1(x_layer_1)
        x_final_1 = self.dropout_1(x_active_1)

        x_layer_2 = self.layer_2(x_final_1)
        x_active_2 = self.activation_function_2(x_layer_2)
        x_final_2 = self.dropout_2(x_active_2)

        y = self.layer_out(x_final_2)

        return y
    
def mlp_once(X_train, y_train, X_val, y_val, hidden_layers, learning_rate, weight_decay, dropout, epochs=1000):
    """train one MLP once and return model and val MAE. Once again heavily inspired by ECE 228 HW 2"""
    X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
    X_val_t = torch.tensor(X_val.values, dtype=torch.float32)

    model = MLPModel( input_dim=X_train.shape[1], hidden_layers=hidden_layers, dropout=dropout)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam( model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for i in range(0, epochs):
        model.train()
        optimizer.zero_grad()

        train_pred = model(X_train_t)
        train_loss = loss_fn(train_pred, y_train_t)

        train_loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).numpy().ravel()

    val_mae = mean_absolute_error(y_val, val_pred)

    return model, val_mae


def predict_mlp(model, X):
    """return mlp predictions"""
    model.eval()
    X_tensor = torch.tensor(X.values, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_tensor).numpy().ravel()

    return preds


def mlp(X_train, y_train, X_val, y_val, X_test, y_test, df, features):
    """find the best hyperparams and train a MLP model"""
    best_set = {}
    best_score = 100000000000
    X_full = df[features]

    for hidden_layers in [(32, 16), (64, 32), (128, 64), (256, 128)]:
        for learning_rate in [0.001, 0.0005, 0.0001]:
            for weight_decay in [0.0, 0.0001, 0.001]:
                for dropout in [0.0, 0.1, 0.2]:
                    for epochs in [500, 1000]:
                        model, val_mae = mlp_once(X_train, y_train, X_val, y_val, hidden_layers, learning_rate, weight_decay, dropout, epochs=epochs)
                        if val_mae < best_score:
                            best_score = val_mae
                            best_set = { "hidden_layers": hidden_layers,  "learning_rate": learning_rate, "weight_decay": weight_decay, "dropout": dropout, "epochs": epochs }

    model, _ = mlp_once( pd.concat([X_train, X_val], axis=0), pd.concat([y_train, y_val], axis=0), X_test,  y_test, best_set["hidden_layers"], best_set["learning_rate"], best_set["weight_decay"], best_set["dropout"], best_set["epochs"])
    df_preds = predict_mlp(model, X_full)

    test_pred = predict_mlp(model, X_test)
    test_mae, test_mse, test_rsq = evaluate(y_test, test_pred)

    return ("MLP", best_set, best_score, test_mae, test_mse, test_rsq, df_preds)

def run_models_for_target(df, target_col, raw_features, standard_features):
    """run all models for a target"""
    linear_models = [lin_reg, ridge_reg, lasso_reg, mlp]
    tree_models = [ random_forest, gradient_boosting, xgboost]

    output = []
    predictions = pd.DataFrame({  "brand": df["brand"], "name": df["name"],  "actual_tdp_w": df["tdp_w"],  "actual_psu_w": df["psu_w"] })
    for model in linear_models:
        X_train, X_val, X_test, y_train, y_val, y_test = split(df, target_col, standard_features)
        model_name, params, val_mae, test_mae, test_rmse, test_rsq, df_preds = model(X_train, y_train, X_val, y_val,  X_test, y_test, df, standard_features )
        output.append({ "target": target_col, "model": model_name, "params": params, "val_mae": val_mae,  "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_rsq})
        name = model_name.lower().replace(" ", "_")
        predictions[f"pred_{target_col}_{name}"] = df_preds

    for model in tree_models:
        X_train, X_val, X_test, y_train, y_val, y_test = split(  df, target_col, raw_features )
        model_name, params, val_mae, test_mae, test_rmse, test_rsq, df_preds = model(X_train, y_train, X_val, y_val, X_test, y_test,  df, raw_features )
        output.append({ "target": target_col, "model": model_name, "params": params, "val_mae": val_mae, "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_rsq})
        name = model_name.lower().replace(" ", "_")
        predictions[f"pred_{target_col}_{name}"] = df_preds

    return output, predictions

def main():
    df = load_vectors()

    memory_type_cols = [ col for col in df.columns if col.startswith("memory_type_raw_")]
    normal_cols = [ "process_nm", "tmus",  "rops", "texture_rate", "pixel_rate","direct_x", "memory_mb", "memory_speed_mhz",  "memory_bandwidth_gbs"]
    raw_features = normal_cols + memory_type_cols
    standard_features = [f"standard_{col}" for col in normal_cols] + memory_type_cols

    tdp_results, tdp_predictions = run_models_for_target(df,  "tdp_w",  raw_features, standard_features )
    psu_results, psu_predictions = run_models_for_target(df, "psu_w", raw_features, standard_features)

    all_results = tdp_results + psu_results
    metrics_df = pd.DataFrame(all_results)

    metrics_df["prediction_column_to_use"] = metrics_df.apply( lambda row: f"pred_{row['target']}_{row['model'].lower().replace(' ', '_')}", axis=1)
    metric_cols = ["target","model", "prediction_column_to_use","test_mae","test_rmse","test_r2","val_mae","params"]

    tdp_metrics_df = (metrics_df[metrics_df["target"] == "tdp_w"][metric_cols].sort_values("test_mae"))
    psu_metrics_df = (metrics_df[metrics_df["target"] == "psu_w"][metric_cols].sort_values("test_mae"))

    predictions_df = tdp_predictions.merge(psu_predictions.drop(columns=["brand", "name", "actual_tdp_w", "actual_psu_w"]),left_index=True,right_index=True)

    best_tdp_row = tdp_metrics_df.iloc[0]
    best_psu_row = psu_metrics_df.iloc[0]

    Path(RESULTS_OUT).mkdir(parents=True, exist_ok=True)

    tdp_metrics_df.to_csv(TDP_METRICS_OUT, index=False)
    psu_metrics_df.to_csv(PSU_METRICS_OUT, index=False)
    predictions_df.to_csv(PREDICTIONS_OUT, index=False)

    print("\ntdp model results:")
    print(tdp_metrics_df.to_string(index=False))

    print("\npsu model results:")
    print(psu_metrics_df.to_string(index=False))

    print(
        "\nBest TDP model:\n"
        f"  model: {best_tdp_row['model']}\n"
        f"  prediction column: {best_tdp_row['prediction_column_to_use']}\n"
        f"  mae: {best_tdp_row['test_mae']:.3f} W\n"
        f"  rmse: {best_tdp_row['test_rmse']:.3f} W\n"
        f"  r^2: {best_tdp_row['test_r2']:.3f}"
    )

    print(
        "\nBest PSU model:\n"
        f"  model: {best_psu_row['model']}\n"
        f"  prediction column: {best_psu_row['prediction_column_to_use']}\n"
        f"  mae: {best_psu_row['test_mae']:.3f} W\n"
        f"  rmse: {best_psu_row['test_rmse']:.3f} W\n"
        f"  r^2: {best_psu_row['test_r2']:.3f}"
    )

    print("\nSaved files:")
    print(f"  tdp metrics: {TDP_METRICS_OUT}")
    print(f"  psu metrics: {PSU_METRICS_OUT}")
    print(f"  predictions: {PREDICTIONS_OUT}")


if __name__ == "__main__":
    main()