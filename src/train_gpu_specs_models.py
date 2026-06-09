"""Train TDP and PSU prediction models with uncertainty quantification.

Reads `data/vectors/gpu_power_vectors.csv`, fits ten models (seven point + three UQ)
per target (tdp_w, psu_w), evaluates on a held-out test split, and writes:

    data/results/tdp_model_metrics.csv
    data/results/psu_model_metrics.csv
    data/results/gpu_power_predictions.csv     (one row per GPU; predictions for ALL rows)
    data/results/best_power_model_summary.csv
    data/results/confidence_thresholds.json    (training-fold σ percentiles per UQ model)
    figures/<model>_<target>_actual_vs_pred.png
    figures/mlp_<target>_train_loss.png
    figures/calibration_<target>_<uq_model>.png
    figures/coverage_<target>_<uq_model>.png
    figures/sigma_distribution_<target>_<uq_model>.png

Auto-detects CUDA via torch.cuda.is_available(). XGBoost and PyTorch use the GPU
when available; sklearn models stay on CPU. See POWER_MODEL_TRAINING.md for the
full design rationale.

Usage:
    python src/train_gpu_specs_models.py
"""
from pathlib import Path
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parent.parent
VECTORS_PATH = ROOT / "data" / "vectors" / "gpu_power_vectors.csv"
RESULTS_DIR = ROOT / "data" / "results"
FIGURES_DIR = ROOT / "figures"

TDP_METRICS_OUT = RESULTS_DIR / "tdp_model_metrics.csv"
PSU_METRICS_OUT = RESULTS_DIR / "psu_model_metrics.csv"
PREDICTIONS_OUT = RESULTS_DIR / "gpu_power_predictions.csv"
BEST_MODEL_SUMMARY_OUT = RESULTS_DIR / "best_power_model_summary.csv"
CONFIDENCE_THRESHOLDS_OUT = RESULTS_DIR / "confidence_thresholds.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[device] using {DEVICE}")

TIER_1 = [
    "process_nm", "memory_speed_mhz", "memory_mb", "memory_bandwidth_gbs",
    "tmus", "rops", "pixel_rate", "texture_rate", "direct_x",
]
TIER_2 = [
    "transistors_m", "die_size_mm2", "density_kmm2", "memory_bus_bits",
    "release_year", "shading_units", "fp32_gflops",
]
TIER_3_HEAVY = ["gpu_clock_mhz", "base_clock_mhz", "tensor_cores", "rt_cores"]
BOOST_CLOCK = "boost_clock_mhz"

INDICATOR_COLS = TIER_2 + [BOOST_CLOCK]

QXGB_QUANTILES = (0.05, 0.50, 0.95)
Z_90 = 1.6448536269514722 


def standardized(cols):
    return [f"standard_{c}" for c in cols]


def categorical_cols(df):
    return [c for c in df.columns if c.startswith(("memory_type_", "architecture_"))]


def add_missing_indicators(frame, cols, prefix="is_missing_"):
    """Append is_missing_<col> binary columns based on raw NaN status."""
    out = frame.copy()
    added = []
    for c in cols:
        col = f"{prefix}{c}"
        out[col] = out[c].isna().astype(int)
        added.append(col)
    return out, added


def median_impute(train_frame, *other_frames, cols):
    """Median-impute cols using training-fold medians. Returns imputed copies."""
    medians = train_frame[cols].median()
    out = [train_frame.copy()]
    for f in other_frames:
        out.append(f.copy())
    for frame in out:
        frame[cols] = frame[cols].fillna(medians)
    return tuple(out) + (medians,)


def index_split(n, val_frac=0.10, test_frac=0.20, seed=42):
    """Reproducible 70/10/20 train/val/test row-index split."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]
    return train_idx, val_idx, test_idx



def evaluate_point(y_true, y_pred):
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_uq_gaussian(y_true, mu, sigma):
    lower = mu - Z_90 * sigma
    upper = mu + Z_90 * sigma
    covered = ((y_true >= lower) & (y_true <= upper)).mean()
    width = (upper - lower).mean()
    return {"coverage_90": float(covered), "mean_interval_width": float(width)}


def evaluate_uq_quantile(y_true, lower, upper):
    covered = ((y_true >= lower) & (y_true <= upper)).mean()
    width = (upper - lower).mean()
    return {"coverage_90": float(covered), "mean_interval_width": float(width)}



def _slug(s):
    return s.lower().replace(" ", "_")


def save_residual_scatter(y_true, y_pred, model_name, target):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.5, s=12)
    plt.plot([lo, hi], [lo, hi], color="black", lw=1)
    plt.xlabel(f"actual {target}")
    plt.ylabel(f"predicted {target}")
    plt.title(f"{model_name} — {target}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"{_slug(model_name)}_{target}_actual_vs_pred.png", dpi=100)
    plt.close()


def save_mlp_loss(losses, target):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("training MSE loss")
    plt.title(f"MLP — {target}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"mlp_{target}_train_loss.png", dpi=100)
    plt.close()


def save_calibration(y_true, mu, sigma, target, model_name, n_bins=10):
    """Binned σ vs mean absolute residual on the test set."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    residual = np.abs(y_true - mu)
    edges = np.quantile(sigma, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_idx = np.digitize(sigma, edges[1:-1])
    xs, ys = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        xs.append(sigma[mask].mean())
        ys.append(residual[mask].mean())
    if not xs:
        return
    mx = max(max(xs), max(ys)) * 1.05
    plt.figure(figsize=(5, 5))
    plt.scatter(xs, ys, s=40)
    plt.plot([0, mx], [0, mx], color="black", lw=1, ls="--", label="y = x (ideal)")
    plt.xlabel("predicted σ (test, binned by σ-decile)")
    plt.ylabel("mean |actual − predicted|")
    plt.title(f"calibration — {model_name} — {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"calibration_{target}_{_slug(model_name)}.png", dpi=100)
    plt.close()


def save_coverage(y_true, mu, sigma, target, model_name):
    """Empirical coverage vs nominal α on the test set."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    from scipy.stats import norm
    alphas = np.linspace(0.5, 0.99, 25)
    empirical = []
    for a in alphas:
        z = norm.ppf(0.5 + a / 2)
        lower = mu - z * sigma
        upper = mu + z * sigma
        empirical.append(float(((y_true >= lower) & (y_true <= upper)).mean()))
    plt.figure(figsize=(5, 5))
    plt.plot(alphas, empirical, marker="o", label="empirical")
    plt.plot([0.5, 0.99], [0.5, 0.99], color="black", lw=1, ls="--", label="y = x (ideal)")
    plt.xlabel("nominal coverage α")
    plt.ylabel("empirical coverage")
    plt.title(f"coverage — {model_name} — {target}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"coverage_{target}_{_slug(model_name)}.png", dpi=100)
    plt.close()


def save_sigma_distribution(sigma_train, target, model_name, thresholds):
    """Histogram of training-fold σ with the 33/67 percentile cuts marked."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 3))
    plt.hist(sigma_train, bins=30)
    labels = ["33% (high→medium)", "67% (medium→low)"]
    for thr, label in zip(thresholds, labels):
        plt.axvline(thr, color="red", lw=1, ls="--", label=f"{label} = {thr:.3g}")
    plt.xlabel("training-fold σ")
    plt.ylabel("count")
    plt.title(f"σ distribution — {model_name} — {target}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"sigma_distribution_{target}_{_slug(model_name)}.png", dpi=100)
    plt.close()



def _fit_and_score(model_cls, params, X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    """Generic point-model fit-on-(train+val), evaluate test, predict all."""
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    fitted = model_cls(**params).fit(X_full, y_full)
    t0 = time.perf_counter()
    test_pred = fitted.predict(X_test)
    latency = (time.perf_counter() - t0) / len(X_test) * 1000.0
    preds_all = fitted.predict(X_all)
    m = evaluate_point(y_test, test_pred)
    return {
        "test_mae": m["mae"], "test_rmse": m["rmse"], "test_r2": m["r2"],
        "preds_all": preds_all, "test_pred": test_pred,
        "inference_latency_ms": float(latency),
    }


def lin_reg(X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    m = LinearRegression().fit(X_train, y_train)
    val_mae = float(mean_absolute_error(y_val, m.predict(X_val)))
    out = _fit_and_score(LinearRegression, {}, X_train, X_val, X_test, X_all, y_train, y_val, y_test)
    return {"model": "Linear Regression", "params": {}, "val_mae": val_mae, **out}


def ridge_reg(X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    best_alpha, best_val = None, float("inf")
    for a in (0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 100.0):
        m = Ridge(alpha=a).fit(X_train, y_train)
        v = float(mean_absolute_error(y_val, m.predict(X_val)))
        if v < best_val:
            best_val, best_alpha = v, a
    out = _fit_and_score(Ridge, {"alpha": best_alpha}, X_train, X_val, X_test, X_all, y_train, y_val, y_test)
    return {"model": "Ridge Regression", "params": {"alpha": best_alpha}, "val_mae": best_val, **out}


def lasso_reg(X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    best_alpha, best_val = None, float("inf")
    for a in (0.0001, 0.001, 0.01, 0.1, 0.5, 0.75, 1.0):
        m = Lasso(alpha=a, max_iter=10000).fit(X_train, y_train)
        v = float(mean_absolute_error(y_val, m.predict(X_val)))
        if v < best_val:
            best_val, best_alpha = v, a
    params = {"alpha": best_alpha, "max_iter": 10000}
    out = _fit_and_score(Lasso, params, X_train, X_val, X_test, X_all, y_train, y_val, y_test)
    return {"model": "Lasso Regression", "params": params, "val_mae": best_val, **out}


def random_forest(X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    best_params, best_val = None, float("inf")
    for n in (300, 500, 800):
        for d in (None, 10, 20, 30):
            for leaf in (1, 2, 4):
                for feat in ("sqrt", 0.7, 1.0):
                    p = {
                        "n_estimators": n, "max_depth": d, "min_samples_leaf": leaf,
                        "max_features": feat, "n_jobs": -1, "random_state": 42,
                    }
                    m = RandomForestRegressor(**p).fit(X_train, y_train)
                    v = float(mean_absolute_error(y_val, m.predict(X_val)))
                    if v < best_val:
                        best_val, best_params = v, p
    out = _fit_and_score(RandomForestRegressor, best_params, X_train, X_val, X_test, X_all, y_train, y_val, y_test)
    return {"model": "Random Forest", "params": best_params, "val_mae": best_val, **out}


def gradient_boosting(X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    best_params, best_val = None, float("inf")
    for n in (300, 500, 800):
        for lr in (0.03, 0.05, 0.075, 0.1):
            for d in (3, 4, 5, 6):
                for leaf in (1, 2, 5):
                    p = {
                        "n_estimators": n, "learning_rate": lr, "max_depth": d,
                        "min_samples_leaf": leaf, "random_state": 42,
                    }
                    m = GradientBoostingRegressor(**p).fit(X_train, y_train)
                    v = float(mean_absolute_error(y_val, m.predict(X_val)))
                    if v < best_val:
                        best_val, best_params = v, p
    out = _fit_and_score(GradientBoostingRegressor, best_params, X_train, X_val, X_test, X_all, y_train, y_val, y_test)
    return {"model": "Gradient Boosting", "params": best_params, "val_mae": best_val, **out}


def xgboost_point(X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    best_params, best_val = None, float("inf")
    base = {
        "objective": "reg:squarederror",
        "subsample": 0.9, "colsample_bytree": 0.9,
        "tree_method": "hist", "device": DEVICE,
        "random_state": 42,
    }
    for n in (300, 500, 800):
        for lr in (0.03, 0.05, 0.075, 0.1):
            for d in (4, 5, 6):
                for mcw in (1, 3, 5):
                    for rl in (1.0, 5.0, 10.0):
                        p = {**base, "n_estimators": n, "learning_rate": lr,
                             "max_depth": d, "min_child_weight": mcw, "reg_lambda": rl}
                        m = XGBRegressor(**p).fit(X_train, y_train)
                        v = float(mean_absolute_error(y_val, m.predict(X_val)))
                        if v < best_val:
                            best_val, best_params = v, p
    out = _fit_and_score(XGBRegressor, best_params, X_train, X_val, X_test, X_all, y_train, y_val, y_test)
    return {"model": "XGBoost", "params": best_params, "val_mae": best_val, **out}


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden=(256, 128), dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x)


def _mlp_train_once(X_train, y_train, X_val, y_val, hidden, lr, wd, dropout, epochs, record_losses=False):
    dev = torch.device(DEVICE)
    Xt = torch.tensor(X_train.values, dtype=torch.float32, device=dev)
    yt = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32, device=dev)
    Xv = torch.tensor(X_val.values, dtype=torch.float32, device=dev)
    model = MLPModel(X_train.shape[1], hidden=hidden, dropout=dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    losses = [] if record_losses else None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        if record_losses:
            losses.append(float(loss.item()))
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).detach().cpu().numpy().ravel()
    val_mae = float(mean_absolute_error(y_val, val_pred))
    if record_losses:
        return model, val_mae, losses
    return model, val_mae


def _mlp_predict(model, X):
    dev = torch.device(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(torch.tensor(X.values, dtype=torch.float32, device=dev)).detach().cpu().numpy().ravel()
    return out


def mlp(X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    best_params, best_val = None, float("inf")
    for hidden in [(32, 16), (64, 32), (128, 64), (256, 128)]:
        for lr in (1e-4, 1e-3, 5e-3):
            for wd in (1e-4, 1e-3):
                for dp in (0.0, 0.1, 0.2):
                    _, v = _mlp_train_once(X_train, y_train, X_val, y_val, hidden, lr, wd, dp, epochs=1000)
                    if v < best_val:
                        best_val = v
                        best_params = {"hidden_layers": hidden, "learning_rate": lr,
                                       "weight_decay": wd, "dropout": dp, "epochs": 1000}
    # Final fit on train+val
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    model, _, losses = _mlp_train_once(
        X_full, y_full, X_test, y_test,
        best_params["hidden_layers"], best_params["learning_rate"],
        best_params["weight_decay"], best_params["dropout"],
        best_params["epochs"], record_losses=True,
    )
    save_mlp_loss(losses, y_train.name)
    t0 = time.perf_counter()
    test_pred = _mlp_predict(model, X_test)
    latency = (time.perf_counter() - t0) / len(X_test) * 1000.0
    preds_all = _mlp_predict(model, X_all)
    m = evaluate_point(y_test, test_pred)
    return {
        "model": "MLP", "params": best_params, "val_mae": best_val,
        "test_mae": m["mae"], "test_rmse": m["rmse"], "test_r2": m["r2"],
        "preds_all": preds_all, "test_pred": test_pred,
        "inference_latency_ms": float(latency),
    }


def bayesian_ridge(X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    fitted = BayesianRidge().fit(X_full, y_full)
    val_pred = fitted.predict(X_val)
    val_mae = float(mean_absolute_error(y_val, val_pred))
    t0 = time.perf_counter()
    test_mu, test_sigma = fitted.predict(X_test, return_std=True)
    latency = (time.perf_counter() - t0) / len(X_test) * 1000.0
    mu_all, sigma_all = fitted.predict(X_all, return_std=True)
    _, sigma_train = fitted.predict(X_train, return_std=True)
    point = evaluate_point(y_test, test_mu)
    uq = evaluate_uq_gaussian(y_test.values, test_mu, test_sigma)
    return {
        "model": "Bayesian Ridge", "params": {}, "val_mae": val_mae,
        "test_mae": point["mae"], "test_rmse": point["rmse"], "test_r2": point["r2"],
        "coverage_90": uq["coverage_90"], "mean_interval_width": uq["mean_interval_width"],
        "preds_all": mu_all, "sigma_all": sigma_all,
        "test_pred": test_mu, "test_sigma": test_sigma,
        "sigma_train": sigma_train,
        "inference_latency_ms": float(latency),
    }


def _quantile_xgb_variant(X_train, X_val, X_test, X_all, y_train, y_val, y_test, variant_label):
    """Train a single Quantile XGB variant. Tunes hyperparams on q=0.5 only,
    then re-fits the three quantiles using the chosen hyperparams."""
    base = {
        "tree_method": "hist", "device": DEVICE,
        "subsample": 0.9, "colsample_bytree": 0.9,
        "random_state": 42,
    }
    best_h, best_val = None, float("inf")
    for n in (300, 500, 800):
        for lr in (0.03, 0.05, 0.075, 0.1):
            for d in (4, 5, 6):
                for mcw in (1, 3, 5):
                    for rl in (1.0, 5.0, 10.0):
                        p = {
                            **base, "objective": "reg:quantileerror", "quantile_alpha": 0.5,
                            "n_estimators": n, "learning_rate": lr,
                            "max_depth": d, "min_child_weight": mcw, "reg_lambda": rl,
                        }
                        m = XGBRegressor(**p).fit(X_train, y_train)
                        v = float(mean_absolute_error(y_val, m.predict(X_val)))
                        if v < best_val:
                            best_val = v
                            best_h = {k: v_ for k, v_ in p.items()
                                      if k not in ("objective", "quantile_alpha")}
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    fits = {}
    for q in QXGB_QUANTILES:
        p = {**best_h, "objective": "reg:quantileerror", "quantile_alpha": q}
        fits[q] = XGBRegressor(**p).fit(X_full, y_full)

    t0 = time.perf_counter()
    test_lower = fits[0.05].predict(X_test)
    test_med = fits[0.50].predict(X_test)
    test_upper = fits[0.95].predict(X_test)
    latency = (time.perf_counter() - t0) / len(X_test) * 1000.0

    all_lower = fits[0.05].predict(X_all)
    all_med = fits[0.50].predict(X_all)
    all_upper = fits[0.95].predict(X_all)

    train_lower = fits[0.05].predict(X_train)
    train_upper = fits[0.95].predict(X_train)

    point = evaluate_point(y_test, test_med)
    uq = evaluate_uq_quantile(y_test.values, test_lower, test_upper)
    sigma_test = (test_upper - test_lower) / (2 * Z_90)
    sigma_all = (all_upper - all_lower) / (2 * Z_90)
    sigma_train = (train_upper - train_lower) / (2 * Z_90)

    return {
        "variant": variant_label, "params": best_h, "val_mae": best_val,
        "test_mae": point["mae"], "test_rmse": point["rmse"], "test_r2": point["r2"],
        "coverage_90": uq["coverage_90"], "mean_interval_width": uq["mean_interval_width"],
        "preds_all": all_med, "sigma_all": sigma_all,
        "lower_all": all_lower, "upper_all": all_upper,
        "test_pred": test_med, "test_sigma": sigma_test,
        "test_lower": test_lower, "test_upper": test_upper,
        "sigma_train": sigma_train,
        "inference_latency_ms": float(latency),
    }


def quantile_xgboost_ab(A, B):
    """A/B select between native-NaN (A) and impute+indicator (B) variants.
    Selection: prefer variant with coverage_90 in [0.85, 0.95], then min val_mae."""
    def in_band(r):
        return 0.85 <= r["coverage_90"] <= 0.95

    if in_band(A) and in_band(B):
        chosen = A if A["val_mae"] <= B["val_mae"] else B
    elif in_band(A):
        chosen = A
    elif in_band(B):
        chosen = B
    else:
        chosen = A if abs(A["coverage_90"] - 0.9) <= abs(B["coverage_90"] - 0.9) else B
    chosen = dict(chosen)
    chosen["model"] = "Quantile XGBoost"
    return chosen


def gaussian_process(X_train, X_val, X_test, X_all, y_train, y_val, y_test):
    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)
    n_features = X_train.shape[1]
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e3))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e3))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=5,
        normalize_y=True, random_state=42,
    )
    gp.fit(X_full, y_full)
    val_pred = gp.predict(X_val)
    val_mae = float(mean_absolute_error(y_val, val_pred))
    t0 = time.perf_counter()
    test_mu, test_sigma = gp.predict(X_test, return_std=True)
    latency = (time.perf_counter() - t0) / len(X_test) * 1000.0
    mu_all, sigma_all = gp.predict(X_all, return_std=True)
    _, sigma_train = gp.predict(X_train, return_std=True)
    point = evaluate_point(y_test, test_mu)
    uq = evaluate_uq_gaussian(y_test.values, test_mu, test_sigma)
    return {
        "model": "Gaussian Process", "params": {"kernel": str(gp.kernel_)},
        "val_mae": val_mae,
        "test_mae": point["mae"], "test_rmse": point["rmse"], "test_r2": point["r2"],
        "coverage_90": uq["coverage_90"], "mean_interval_width": uq["mean_interval_width"],
        "preds_all": mu_all, "sigma_all": sigma_all,
        "test_pred": test_mu, "test_sigma": test_sigma,
        "sigma_train": sigma_train,
        "inference_latency_ms": float(latency),
    }


def compute_confidence_flags(sigma_all, sigma_train, low_pct=33, high_pct=67):
    low_thr = float(np.percentile(sigma_train, low_pct))
    high_thr = float(np.percentile(sigma_train, high_pct))
    flags = np.where(sigma_all <= low_thr, "high",
                     np.where(sigma_all <= high_thr, "medium", "low"))
    return flags, low_thr, high_thr


def outliers_gaussian(y_true_all, mu_all, sigma_all):
    """1 if |actual − μ| > 2σ. NaN where actual is NaN (prediction-only rows)."""
    out = np.full(len(y_true_all), np.nan, dtype="float64")
    mask = ~np.isnan(y_true_all)
    out[mask] = (np.abs(y_true_all[mask] - mu_all[mask]) > 2 * sigma_all[mask]).astype(int)
    return out


def outliers_quantile(y_true_all, lower_all, upper_all):
    out = np.full(len(y_true_all), np.nan, dtype="float64")
    mask = ~np.isnan(y_true_all)
    out[mask] = ((y_true_all[mask] < lower_all[mask]) | (y_true_all[mask] > upper_all[mask])).astype(int)
    return out



def run_target(df_vectors, target, predictions_acc):
    """Train all 10 models for a single target. Returns metrics list, updates
    predictions_acc in place with new pred / sigma / confidence / outlier columns.
    Also returns the confidence_thresholds dict to be saved to JSON."""

    print(f"\n{'=' * 60}\n  TARGET: {target}\n{'=' * 60}")

    train_mask = df_vectors[target].notna()
    train_df = df_vectors[train_mask].reset_index(drop=True)
    print(f"  training-eligible rows: {len(train_df)}  /  total: {len(df_vectors)}")

    train_idx, val_idx, test_idx = index_split(len(train_df))
    print(f"  split: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    y_full_train = train_df[target]
    y_full_train.name = target
    y_train = y_full_train.iloc[train_idx].reset_index(drop=True); y_train.name = target
    y_val = y_full_train.iloc[val_idx].reset_index(drop=True); y_val.name = target
    y_test = y_full_train.iloc[test_idx].reset_index(drop=True); y_test.name = target

    cat_cols = categorical_cols(df_vectors)

    #  Linear / MLP / Bayesian Ridge feature matrix: standardized + categorical
    std_cols = standardized(TIER_1 + TIER_2 + [BOOST_CLOCK])
    X_std_all = df_vectors[std_cols + cat_cols].copy()
    X_std_train_full = train_df[std_cols + cat_cols].copy()
    X_std_train = X_std_train_full.iloc[train_idx].reset_index(drop=True)
    X_std_val = X_std_train_full.iloc[val_idx].reset_index(drop=True)
    X_std_test = X_std_train_full.iloc[test_idx].reset_index(drop=True)

    #  Bayesian Ridge: same standardized features + missing indicators
    br_indicator_train = train_df[INDICATOR_COLS].isna().astype(int)
    br_indicator_train.columns = [f"is_missing_{c}" for c in INDICATOR_COLS]
    br_indicator_all = df_vectors[INDICATOR_COLS].isna().astype(int)
    br_indicator_all.columns = [f"is_missing_{c}" for c in INDICATOR_COLS]

    X_br_all = pd.concat([X_std_all, br_indicator_all], axis=1)
    X_br_train_full = pd.concat([X_std_train_full, br_indicator_train], axis=1)
    X_br_train = X_br_train_full.iloc[train_idx].reset_index(drop=True)
    X_br_val = X_br_train_full.iloc[val_idx].reset_index(drop=True)
    X_br_test = X_br_train_full.iloc[test_idx].reset_index(drop=True)

    #  GP: standardized Tier 1 + Tier 2 + boost_clock + missing indicators (no categorical)
    gp_std_cols = standardized(TIER_1 + TIER_2 + [BOOST_CLOCK])
    X_gp_all = pd.concat([df_vectors[gp_std_cols], br_indicator_all], axis=1)
    X_gp_train_full = pd.concat([train_df[gp_std_cols], br_indicator_train], axis=1)
    X_gp_train = X_gp_train_full.iloc[train_idx].reset_index(drop=True)
    X_gp_val = X_gp_train_full.iloc[val_idx].reset_index(drop=True)
    X_gp_test = X_gp_train_full.iloc[test_idx].reset_index(drop=True)

    #  RF / GB: raw Tier 1 + Tier 2 + boost_clock + categorical, median-imputed
    rf_raw = TIER_1 + TIER_2 + [BOOST_CLOCK]
    X_rf_train_full = train_df[rf_raw + cat_cols].copy()
    X_rf_train_pre = X_rf_train_full.iloc[train_idx].reset_index(drop=True)
    medians = X_rf_train_pre[rf_raw].median()
    X_rf_train = X_rf_train_pre.copy(); X_rf_train[rf_raw] = X_rf_train[rf_raw].fillna(medians)
    X_rf_val = X_rf_train_full.iloc[val_idx].reset_index(drop=True)
    X_rf_val[rf_raw] = X_rf_val[rf_raw].fillna(medians)
    X_rf_test = X_rf_train_full.iloc[test_idx].reset_index(drop=True)
    X_rf_test[rf_raw] = X_rf_test[rf_raw].fillna(medians)
    X_rf_all = df_vectors[rf_raw + cat_cols].copy()
    X_rf_all[rf_raw] = X_rf_all[rf_raw].fillna(medians)

    #  XGBoost / QXGB Variant A: raw Tier 1 + Tier 2 + Tier 3 + boost_clock + categorical, NaN preserved
    xgb_raw = TIER_1 + TIER_2 + TIER_3_HEAVY + [BOOST_CLOCK]
    X_xgb_all = df_vectors[xgb_raw + cat_cols].copy()
    X_xgb_train_full = train_df[xgb_raw + cat_cols].copy()
    X_xgb_train = X_xgb_train_full.iloc[train_idx].reset_index(drop=True)
    X_xgb_val = X_xgb_train_full.iloc[val_idx].reset_index(drop=True)
    X_xgb_test = X_xgb_train_full.iloc[test_idx].reset_index(drop=True)

    #  QXGB Variant B: median-impute (Tier 2 + boost_clock), keep Tier 3 native NaN, add indicators
    qb_raw_impute = TIER_2 + [BOOST_CLOCK]
    X_qb_train_full = train_df[xgb_raw + cat_cols].copy()
    X_qb_train_pre = X_qb_train_full.iloc[train_idx].reset_index(drop=True)
    qb_medians = X_qb_train_pre[qb_raw_impute].median()
    X_qb_train = X_qb_train_pre.copy()
    X_qb_train[qb_raw_impute] = X_qb_train[qb_raw_impute].fillna(qb_medians)
    X_qb_train = pd.concat([X_qb_train, br_indicator_train.iloc[train_idx].reset_index(drop=True)], axis=1)
    X_qb_val = X_qb_train_full.iloc[val_idx].reset_index(drop=True)
    X_qb_val[qb_raw_impute] = X_qb_val[qb_raw_impute].fillna(qb_medians)
    X_qb_val = pd.concat([X_qb_val, br_indicator_train.iloc[val_idx].reset_index(drop=True)], axis=1)
    X_qb_test = X_qb_train_full.iloc[test_idx].reset_index(drop=True)
    X_qb_test[qb_raw_impute] = X_qb_test[qb_raw_impute].fillna(qb_medians)
    X_qb_test = pd.concat([X_qb_test, br_indicator_train.iloc[test_idx].reset_index(drop=True)], axis=1)
    X_qb_all = df_vectors[xgb_raw + cat_cols].copy()
    X_qb_all[qb_raw_impute] = X_qb_all[qb_raw_impute].fillna(qb_medians)
    X_qb_all = pd.concat([X_qb_all, br_indicator_all], axis=1)

    results = []
    y_actual_all = df_vectors[target].values

    #  Point models
    for fn, name, X_train_, X_val_, X_test_, X_all_ in [
        (lin_reg, "Linear Regression", X_std_train, X_std_val, X_std_test, X_std_all),
        (ridge_reg, "Ridge Regression", X_std_train, X_std_val, X_std_test, X_std_all),
        (lasso_reg, "Lasso Regression", X_std_train, X_std_val, X_std_test, X_std_all),
        (mlp, "MLP", X_std_train, X_std_val, X_std_test, X_std_all),
        (random_forest, "Random Forest", X_rf_train, X_rf_val, X_rf_test, X_rf_all),
        (gradient_boosting, "Gradient Boosting", X_rf_train, X_rf_val, X_rf_test, X_rf_all),
        (xgboost_point, "XGBoost", X_xgb_train, X_xgb_val, X_xgb_test, X_xgb_all),
    ]:
        t0 = time.perf_counter()
        print(f"  [{name}] training...")
        res = fn(X_train_, X_val_, X_test_, X_all_, y_train, y_val, y_test)
        dt = time.perf_counter() - t0
        print(f"  [{name}] val_mae={res['val_mae']:.3f}  test_mae={res['test_mae']:.3f}  "
              f"r2={res['test_r2']:.3f}  time={dt:.1f}s")
        results.append(res)
        save_residual_scatter(y_test.values, res["test_pred"], name, target)
        col = f"pred_{target}_{_slug(name)}"
        predictions_acc[col] = res["preds_all"]

    #  Bayesian Ridge
    t0 = time.perf_counter()
    print("  [Bayesian Ridge] training...")
    br = bayesian_ridge(X_br_train, X_br_val, X_br_test, X_br_all, y_train, y_val, y_test)
    dt = time.perf_counter() - t0
    print(f"  [Bayesian Ridge] val_mae={br['val_mae']:.3f}  test_mae={br['test_mae']:.3f}  "
          f"coverage_90={br['coverage_90']:.3f}  width={br['mean_interval_width']:.2f}  time={dt:.1f}s")
    results.append(br)
    save_residual_scatter(y_test.values, br["test_pred"], "Bayesian Ridge", target)
    save_calibration(y_test.values, br["test_pred"], br["test_sigma"], target, "Bayesian Ridge")
    save_coverage(y_test.values, br["test_pred"], br["test_sigma"], target, "Bayesian Ridge")

    #  Quantile XGBoost A/B
    print("  [Quantile XGBoost] training Variant A (native NaN)...")
    t0 = time.perf_counter()
    qxgb_A = _quantile_xgb_variant(X_xgb_train, X_xgb_val, X_xgb_test, X_xgb_all,
                                   y_train, y_val, y_test, "native_nan")
    print(f"  [QXGB A] val_mae={qxgb_A['val_mae']:.3f}  test_mae={qxgb_A['test_mae']:.3f}  "
          f"coverage_90={qxgb_A['coverage_90']:.3f}  width={qxgb_A['mean_interval_width']:.2f}  "
          f"time={time.perf_counter() - t0:.1f}s")
    print("  [Quantile XGBoost] training Variant B (impute + indicators)...")
    t0 = time.perf_counter()
    qxgb_B = _quantile_xgb_variant(X_qb_train, X_qb_val, X_qb_test, X_qb_all,
                                   y_train, y_val, y_test, "impute_indicator")
    print(f"  [QXGB B] val_mae={qxgb_B['val_mae']:.3f}  test_mae={qxgb_B['test_mae']:.3f}  "
          f"coverage_90={qxgb_B['coverage_90']:.3f}  width={qxgb_B['mean_interval_width']:.2f}  "
          f"time={time.perf_counter() - t0:.1f}s")
    qxgb = quantile_xgboost_ab(qxgb_A, qxgb_B)
    print(f"  [Quantile XGBoost] chose variant: {qxgb['variant']}")
    results.append(qxgb)
    save_residual_scatter(y_test.values, qxgb["test_pred"], "Quantile XGBoost", target)
    save_calibration(y_test.values, qxgb["test_pred"], qxgb["test_sigma"], target, "Quantile XGBoost")
    save_coverage(y_test.values, qxgb["test_pred"], qxgb["test_sigma"], target, "Quantile XGBoost")

    #  Gaussian Process
    t0 = time.perf_counter()
    print("  [Gaussian Process] training...")
    gp = gaussian_process(X_gp_train, X_gp_val, X_gp_test, X_gp_all, y_train, y_val, y_test)
    dt = time.perf_counter() - t0
    print(f"  [Gaussian Process] val_mae={gp['val_mae']:.3f}  test_mae={gp['test_mae']:.3f}  "
          f"coverage_90={gp['coverage_90']:.3f}  width={gp['mean_interval_width']:.2f}  time={dt:.1f}s")
    results.append(gp)
    save_residual_scatter(y_test.values, gp["test_pred"], "Gaussian Process", target)
    save_calibration(y_test.values, gp["test_pred"], gp["test_sigma"], target, "Gaussian Process")
    save_coverage(y_test.values, gp["test_pred"], gp["test_sigma"], target, "Gaussian Process")

    #  Confidence thresholds and flags, outlier flags
    confidence_thresholds = {}
    individual_outlier_cols = []
    for res in (br, qxgb, gp):
        name = res["model"]
        slug = _slug(name)
        flags, low_thr, high_thr = compute_confidence_flags(
            res["sigma_all"], res["sigma_train"], low_pct=33, high_pct=67
        )
        confidence_thresholds[f"{target}__{slug}"] = {"p33": low_thr, "p67": high_thr}
        save_sigma_distribution(res["sigma_train"], target, name, [low_thr, high_thr])
        # Predictions, σ, confidence, lower/upper bands for QXGB
        predictions_acc[f"pred_{target}_{slug}"] = res["preds_all"]
        predictions_acc[f"sigma_{target}_{slug}"] = res["sigma_all"]
        predictions_acc[f"confidence_{target}_{slug}"] = flags
        if name == "Quantile XGBoost":
            predictions_acc[f"lower_{target}_quantile_xgboost"] = res["lower_all"]
            predictions_acc[f"upper_{target}_quantile_xgboost"] = res["upper_all"]
            outlier = outliers_quantile(y_actual_all, res["lower_all"], res["upper_all"])
        else:
            outlier = outliers_gaussian(y_actual_all, res["preds_all"], res["sigma_all"])
        predictions_acc[f"residual_outlier_{target}_{slug}"] = outlier
        individual_outlier_cols.append(f"residual_outlier_{target}_{slug}")

    # Consensus outlier: 1 iff ≥ 2 of 3 individual flags are 1; NaN where any source is NaN
    outlier_matrix = np.stack([predictions_acc[c].astype("float64") for c in individual_outlier_cols], axis=1)
    nan_row = np.isnan(outlier_matrix).any(axis=1)
    sums = np.nansum(outlier_matrix, axis=1)
    consensus = np.where(nan_row, np.nan, (sums >= 2).astype(int))
    predictions_acc[f"residual_outlier_{target}_consensus"] = consensus

    return results, confidence_thresholds



def main():
    print("=" * 60)
    print("POWER MODEL TRAINING")
    print(f"  device: {DEVICE}")
    print("=" * 60)
    df = pd.read_csv(VECTORS_PATH)
    print(f"[load] {len(df)} rows x {len(df.columns)} cols  ({VECTORS_PATH.name})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame({
        "brand": df["brand"],
        "name": df["name"],
        "actual_tdp_w": df["tdp_w"],
        "actual_psu_w": df["psu_w"],
    })

    all_metrics = []
    all_thresholds = {}
    for target in ("tdp_w", "psu_w"):
        results, thresholds = run_target(df, target, predictions)
        for r in results:
            row = {
                "target": target,
                "model": r["model"],
                "params": json.dumps(r["params"], default=str),
                "val_mae": r["val_mae"],
                "test_mae": r["test_mae"],
                "test_rmse": r["test_rmse"],
                "test_r2": r["test_r2"],
                "inference_latency_ms_per_row": r.get("inference_latency_ms"),
                "coverage_90": r.get("coverage_90"),
                "mean_interval_width": r.get("mean_interval_width"),
                "qxgb_variant": r.get("variant") if r["model"] == "Quantile XGBoost" else None,
            }
            all_metrics.append(row)
        all_thresholds.update(thresholds)

    # Per-target metrics CSV, sorted by test_mae
    metrics_df = pd.DataFrame(all_metrics)
    metric_cols = [
        "target", "model", "val_mae", "test_mae", "test_rmse", "test_r2",
        "coverage_90", "mean_interval_width", "qxgb_variant",
        "inference_latency_ms_per_row", "params",
    ]
    metrics_df = metrics_df[metric_cols]
    metrics_df[metrics_df["target"] == "tdp_w"].sort_values("test_mae").to_csv(TDP_METRICS_OUT, index=False)
    metrics_df[metrics_df["target"] == "psu_w"].sort_values("test_mae").to_csv(PSU_METRICS_OUT, index=False)

    predictions.to_csv(PREDICTIONS_OUT, index=False)

    # Best-model summary (lowest test_mae per target)
    best_rows = []
    for target in ("tdp_w", "psu_w"):
        sub = metrics_df[metrics_df["target"] == target].sort_values("test_mae").head(1).iloc[0]
        best_rows.append({
            "target": target,
            "best_model": sub["model"],
            "test_mae": sub["test_mae"],
            "test_r2": sub["test_r2"],
            "coverage_90": sub["coverage_90"],
        })
    pd.DataFrame(best_rows).to_csv(BEST_MODEL_SUMMARY_OUT, index=False)

    with open(CONFIDENCE_THRESHOLDS_OUT, "w") as f:
        json.dump(all_thresholds, f, indent=2)

    print(f"\n[save] metrics -> {TDP_METRICS_OUT.name}, {PSU_METRICS_OUT.name}")
    print(f"[save] predictions ({len(predictions)} rows x {len(predictions.columns)} cols) -> {PREDICTIONS_OUT.name}")
    print(f"[save] best-model summary -> {BEST_MODEL_SUMMARY_OUT.name}")
    print(f"[save] confidence thresholds -> {CONFIDENCE_THRESHOLDS_OUT.name}")
    print("\n[done]")


if __name__ == "__main__":
    main()
