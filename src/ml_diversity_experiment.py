"""ML-based ranking experiment with diversity controls to reduce collapse.

This script trains a lightweight MLP to predict a composite utility score for
(game, GPU) pairs and compares three selection policies:
- ml_ranker: pick the top predicted utility
- ml_ranker_softmax: sample from a softmax distribution (temperature)
- ml_ranker_diversity: penalize globally frequent GPUs to reduce collapse

Usage:
  python src/ml_diversity_experiment.py --mode both --use-gpu
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

ROOT = Path(__file__).resolve().parent.parent

GAME_VECTORS = {
    "min": ROOT / "data" / "vectors" / "game_vectors_min.csv",
    "recom": ROOT / "data" / "vectors" / "game_vectors_recom.csv",
}
GPU_VECTORS = ROOT / "data" / "vectors" / "gpu_power_vectors.csv"

RESULTS_DIR = ROOT / "data" / "results"

SOFT_FEATURES = [
    "texture_rate",
    "pixel_rate",
    "bandwidth",
    "tmus",
    "rops",
]


@dataclass(frozen=True)
class TrainConfig:
    mode: str
    max_candidates_per_game: int = 200
    batch_size: int = 2048
    epochs: int = 10
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    seed: int = 42
    use_gpu: bool = False
    temperature: float = 0.7
    diversity_lambda: float = 0.5


def load_vectors(mode: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.read_csv(GAME_VECTORS[mode])
    gpus = pd.read_csv(GPU_VECTORS)
    return games, gpus


def hard_filter(game: pd.Series, gpus: pd.DataFrame, mode: str) -> pd.DataFrame:
    mask = pd.Series(True, index=gpus.index)
    if mode == "min":
        vram_req = game.get("min_vram_mb")
        dx_req = game.get("min_direct_x")
    else:
        vram_req = game.get("recom_vram_mb")
        dx_req = game.get("recom_direct_x")

    if pd.notna(vram_req) and vram_req > 0:
        mask &= gpus["memory_mb"] >= vram_req
    if pd.notna(dx_req) and dx_req > 0:
        mask &= gpus["direct_x"] >= dx_req
    return gpus[mask].copy()


def build_features(game: pd.Series, gpu: pd.Series) -> np.ndarray:
    req_perf = float(game.get("perf_score") or 0.0)
    gpu_perf = float(gpu.get("perf_score") or 0.0)

    perf_ratio = gpu_perf / req_perf if req_perf > 0 else 0.0
    overprov = (gpu_perf - req_perf) / req_perf if req_perf > 0 else 0.0

    feats = [
        perf_ratio,
        overprov,
        float(gpu.get("tdp_w") or 0.0),
        float(gpu.get("psu_w") or 0.0),
        float(gpu.get("memory_mb") or 0.0),
        float(gpu.get("direct_x") or 0.0),
    ]

    for feat in SOFT_FEATURES:
        req = float(game.get(feat) or 0.0)
        val = float(gpu.get("memory_bandwidth_gbs") if feat == "bandwidth" else gpu.get(feat) or 0.0)
        ratio = val / req if req > 0 else 0.0
        feats.append(ratio)

    return np.array(feats, dtype=np.float32)


def utility_label(game: pd.Series, gpu: pd.Series) -> float:
    req_perf = float(game.get("perf_score") or 0.0)
    gpu_perf = float(gpu.get("perf_score") or 0.0)
    tdp = float(gpu.get("tdp_w") or 0.0)
    psu = float(gpu.get("psu_w") or 0.0)

    perf_ratio = min(gpu_perf / req_perf, 2.0) if req_perf > 0 else 0.0
    overprov = (gpu_perf - req_perf) / req_perf if req_perf > 0 else 0.0

    # Normalize power to reduce scale dominance.
    tdp_norm = tdp / 600.0
    psu_norm = psu / 1000.0

    # Heuristic target: meet perf, prefer efficient and avoid extreme overprovision.
    return float((0.7 * perf_ratio) - (0.25 * tdp_norm) - (0.15 * psu_norm) - (0.1 * max(0.0, overprov)))


def build_dataset(games: pd.DataFrame, gpus: pd.DataFrame, mode: str, cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.seed)
    feat_list: List[np.ndarray] = []
    label_list: List[float] = []

    for _, game in games.iterrows():
        feasible = hard_filter(game, gpus, mode)
        if feasible.empty:
            continue

        if len(feasible) > cfg.max_candidates_per_game:
            feasible = feasible.sample(cfg.max_candidates_per_game, random_state=cfg.seed)

        for _, gpu in feasible.iterrows():
            feat_list.append(build_features(game, gpu))
            label_list.append(utility_label(game, gpu))

    X = np.vstack(feat_list) if feat_list else np.empty((0, 6 + len(SOFT_FEATURES)), dtype=np.float32)
    y = np.array(label_list, dtype=np.float32)
    return X, y


def train_mlp(X: np.ndarray, y: np.ndarray, cfg: TrainConfig) -> nn.Module:
    if torch is None:
        raise RuntimeError("PyTorch is required for the ML experiment. Please install torch.")

    device = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")

    model = nn.Sequential(
        nn.Linear(X.shape[1], cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
        nn.ReLU(),
        nn.Linear(cfg.hidden_dim, 1),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y).unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for _ in range(cfg.epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()

    return model


def score_candidates(model: nn.Module, feats: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for the ML experiment. Please install torch.")
    device = torch.device("cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        scores = model(torch.from_numpy(feats).to(device)).squeeze(1).cpu().numpy()
    return scores


def select_softmax(scores: np.ndarray, temperature: float, rng: np.random.Generator) -> int:
    scaled = scores / max(temperature, 1e-6)
    scaled = scaled - scaled.max()
    probs = np.exp(scaled)
    probs = probs / probs.sum()
    return int(rng.choice(len(scores), p=probs))


def compute_metrics(recs: pd.DataFrame) -> pd.DataFrame:
    metrics = recs.copy()
    metrics["coverage"] = metrics["selected_gpu"].notna().astype(int)

    req = metrics["game_perf_score"]
    sel_perf = metrics["selected_perf_score"]
    metrics["overprov_abs"] = sel_perf - req
    metrics["overprov_rel"] = (sel_perf / req) - 1
    metrics.loc[(req.isna()) | (req <= 0), ["overprov_abs", "overprov_rel"]] = np.nan

    metrics["eff_regret_abs"] = metrics["best_ppw"] - metrics["selected_perf_per_watt"]
    metrics["eff_regret_rel"] = metrics["eff_regret_abs"] / metrics["best_ppw"]

    return metrics


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "coverage",
        "selected_tdp_w",
        "selected_psu_w",
        "selected_perf_per_watt",
        "overprov_abs",
        "overprov_rel",
        "eff_regret_abs",
        "eff_regret_rel",
    ]

    rows = []
    for (track, method), group in metrics.groupby(["track", "method"], dropna=False):
        row = {"track": track, "method": method}
        for col in metric_cols:
            series = group[col].dropna()
            row[f"{col}_mean"] = series.mean() if not series.empty else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compute_best_ppw(feasible: pd.DataFrame) -> float:
    if feasible.empty:
        return np.nan
    ppw = feasible["perf_score"] / feasible["tdp_w"]
    ppw = ppw.replace([np.inf, -np.inf], np.nan).dropna()
    return float(ppw.max()) if not ppw.empty else np.nan


def build_rec_row(game: pd.Series, feasible: pd.DataFrame, selected: Optional[pd.Series], method: str, mode: str, best_ppw: float) -> Dict[str, object]:
    base = {
        "game_name": game.get("name"),
        "track": mode,
        "method": method,
        "selected_gpu": None if selected is None else selected.get("name"),
        "selected_tdp_w": np.nan if selected is None else selected.get("tdp_w"),
        "selected_psu_w": np.nan if selected is None else selected.get("psu_w"),
        "selected_perf_score": np.nan if selected is None else selected.get("perf_score"),
        "selected_perf_per_watt": np.nan if selected is None else (selected.get("perf_score") / selected.get("tdp_w") if selected.get("tdp_w") else np.nan),
        "game_perf_score": game.get("perf_score"),
        "best_ppw": best_ppw,
    }
    return base


def run_mode(mode: str, cfg: TrainConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    games, gpus = load_vectors(mode)
    X, y = build_dataset(games, gpus, mode, cfg)

    model = train_mlp(X, y, cfg)
    rng = np.random.default_rng(cfg.seed)

    rows: List[Dict[str, object]] = []
    freq: Dict[str, int] = {}

    for _, game in games.iterrows():
        feasible = hard_filter(game, gpus, mode)
        if feasible.empty:
            rows.append(build_rec_row(game, feasible, None, "ml_ranker", mode, np.nan))
            rows.append(build_rec_row(game, feasible, None, "ml_ranker_softmax", mode, np.nan))
            rows.append(build_rec_row(game, feasible, None, "ml_ranker_diversity", mode, np.nan))
            continue

        if len(feasible) > cfg.max_candidates_per_game:
            feasible = feasible.sample(cfg.max_candidates_per_game, random_state=cfg.seed)

        feats = np.vstack([build_features(game, gpu) for _, gpu in feasible.iterrows()])
        scores = score_candidates(model, feats, cfg)

        best_ppw = compute_best_ppw(feasible)

        # Baseline ML: pick top score.
        idx_top = int(np.argmax(scores))
        selected_top = feasible.iloc[idx_top]
        rows.append(build_rec_row(game, feasible, selected_top, "ml_ranker", mode, best_ppw))

        # Softmax sampling for diversity.
        idx_soft = select_softmax(scores, cfg.temperature, rng)
        selected_soft = feasible.iloc[idx_soft]
        rows.append(build_rec_row(game, feasible, selected_soft, "ml_ranker_softmax", mode, best_ppw))

        # Global frequency penalty to prevent collapse.
        penalties = np.array([freq.get(name, 0) for name in feasible["name"].tolist()], dtype=np.float32)
        adj_scores = scores - (cfg.diversity_lambda * np.log1p(penalties))
        idx_div = int(np.argmax(adj_scores))
        selected_div = feasible.iloc[idx_div]
        rows.append(build_rec_row(game, feasible, selected_div, "ml_ranker_diversity", mode, best_ppw))

        # Update frequency with the diversity-aware pick.
        sel_name = selected_div.get("name")
        if sel_name:
            freq[sel_name] = freq.get(sel_name, 0) + 1

    recs = pd.DataFrame(rows)
    metrics = compute_metrics(recs)
    return recs, metrics


def method_diversity(recs: pd.DataFrame) -> pd.DataFrame:
    recs = recs[recs["selected_gpu"].notna()]
    rows = []
    for (track, method), group in recs.groupby(["track", "method"], dropna=False):
        total = len(group)
        unique = group["selected_gpu"].nunique()
        top_share = group["selected_gpu"].value_counts(normalize=True).iloc[0] if total > 0 else np.nan
        rows.append(
            {
                "track": track,
                "method": method,
                "total_recommendations": total,
                "unique_gpus": unique,
                "unique_share": unique / total if total > 0 else np.nan,
                "top1_share": top_share,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="ML diversity experiment for GPU selection")
    parser.add_argument("--mode", choices=["min", "recom", "both"], default="both")
    parser.add_argument("--max-candidates-per-game", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--diversity-lambda", type=float, default=0.5)
    args = parser.parse_args()

    cfg = TrainConfig(
        mode=args.mode,
        max_candidates_per_game=args.max_candidates_per_game,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
        use_gpu=args.use_gpu,
        temperature=args.temperature,
        diversity_lambda=args.diversity_lambda,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_recs = []
    all_metrics = []
    modes = [cfg.mode] if cfg.mode in {"min", "recom"} else ["min", "recom"]

    for mode in modes:
        recs, metrics = run_mode(mode, cfg)
        all_recs.append(recs)
        all_metrics.append(metrics)

    recs_df = pd.concat(all_recs, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)

    summary_df = summarize_metrics(metrics_df)
    diversity_df = method_diversity(recs_df)

    recs_out = RESULTS_DIR / "ml_ranker_recommendations.csv"
    summary_out = RESULTS_DIR / "ml_ranker_metrics_summary.csv"
    diversity_out = RESULTS_DIR / "ml_ranker_diversity_summary.csv"

    recs_df.to_csv(recs_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    diversity_df.to_csv(diversity_out, index=False)

    print(f"Saved: {recs_out}")
    print(f"Saved: {summary_out}")
    print(f"Saved: {diversity_out}")


if __name__ == "__main__":
    main()
