"""GPU recommender: filters feasible GPUs for a game and ranks by performance per watt.

Two ranking methods:

  top_k  — hard/soft filter feasible GPUs, rank by perf_score / tdp_w
  knn    — find k nearest GPUs to the game's normalized feature vector (euclidean distance)

Hard filters  — GPU must meet or exceed the game's minimum requirement:
    memory_mb, direct_x

Soft filters  — GPU must meet at least SOFT_THRESHOLD (80%) of the requirement:
    texture_rate, pixel_rate, memory_bandwidth_gbs, tmus, rops

Usage:
    python src/recommender.py --game "Cyberpunk 2077" --k 5
    python src/recommender.py --game "Cyberpunk 2077" --k 5 --method knn
    python src/recommender.py --game "Cyberpunk 2077" --k 5 --mode recom --threshold 0.7
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

GAME_VECTORS = {
    "min":   "data/vectors/game_vectors_min.csv",
    "recom": "data/vectors/game_vectors_recom.csv",
}
GPU_VECTORS = "data/vectors/gpu_power_vectors.csv"

SOFT_THRESHOLD = 0.80

# Shared canonical column names between GPU and game datasets — no translation
# needed because both pipelines emit the same names.
KNN_FEATURES = [
    "texture_rate",
    "pixel_rate",
    "memory_bandwidth_gbs",
    "tmus",
    "rops",
    "memory_speed_mhz",
    "boost_clock_mhz",
]

SOFT_FILTER_COLS = ["texture_rate", "pixel_rate", "memory_bandwidth_gbs", "tmus", "rops"]

EPSILON = 1e-6

DISPLAY_COLS       = ["brand", "name", "memory_mb", "direct_x", "tdp_w", "perf_score", "perf_per_watt"]
DISPLAY_COLS_KNN   = ["brand", "name", "memory_mb", "direct_x", "tdp_w", "perf_score", "distance"]


def load_data(mode: str):
    games = pd.read_csv(GAME_VECTORS[mode])
    gpus  = pd.read_csv(GPU_VECTORS)
    return games, gpus


def find_game(games: pd.DataFrame, game_name: str) -> pd.Series:
    exact = games[games["name"].str.lower() == game_name.lower()]
    if not exact.empty:
        return exact.iloc[0]

    partial = games[games["name"].str.lower().str.contains(game_name.lower(), regex=False)]
    if partial.empty:
        raise ValueError(f"No game found matching '{game_name}'.")
    if len(partial) > 1:
        print(f"[find_game] {len(partial)} partial matches — using: {partial['name'].iloc[0]!r}")
    return partial.iloc[0]


def hard_filter(gpus: pd.DataFrame, game: pd.Series) -> pd.DataFrame:
    mask = pd.Series(True, index=gpus.index)

    vram_req = game.get("min_vram_mb")
    if pd.notna(vram_req) and vram_req > 0:
        mask &= gpus["memory_mb"] >= vram_req

    dx_req = game.get("min_direct_x")
    if pd.notna(dx_req) and dx_req > 0:
        mask &= gpus["direct_x"] >= dx_req

    result = gpus[mask].copy()
    print(f"[hard_filter]  {mask.sum():4d} / {len(gpus)} GPUs pass  "
          f"(vram>={vram_req} MB, directx>={dx_req})")
    return result


def soft_filter(gpus: pd.DataFrame, game: pd.Series, threshold: float) -> pd.DataFrame:
    mask = pd.Series(True, index=gpus.index)

    applied = []
    for col in SOFT_FILTER_COLS:
        req = game.get(col)
        if pd.isna(req) or req <= 0:
            continue
        min_val = req * threshold
        mask &= gpus[col] >= min_val
        applied.append(f"{col}>={min_val:.1f}")

    result = gpus[mask].copy()
    print(f"[soft_filter]  {mask.sum():4d} / {len(gpus)} GPUs pass  "
          f"(threshold={threshold:.0%}, filters: {', '.join(applied)})")
    return result


def rank(gpus: pd.DataFrame) -> pd.DataFrame:
    gpus = gpus.copy()
    gpus["perf_per_watt"] = gpus["perf_score"] / gpus["tdp_w"]
    return gpus.sort_values("perf_per_watt", ascending=False)


def recommend_top_k(game: pd.Series, gpus: pd.DataFrame, k: int, threshold: float) -> pd.DataFrame:
    feasible = hard_filter(gpus, game)
    feasible = soft_filter(feasible, game, threshold)

    if feasible.empty:
        print("\nNo GPUs passed all filters.")
        return pd.DataFrame()

    ranked = rank(feasible)
    ranked["perf_per_watt"] = ranked["perf_score"] / ranked["tdp_w"]
    top_k  = ranked.head(k)[DISPLAY_COLS].reset_index(drop=True)
    top_k.index += 1

    print(f"\nTop-{k} GPUs by performance per watt:")
    print(top_k.to_string())
    return top_k


def recommend_knn(game: pd.Series, gpus: pd.DataFrame, k: int) -> pd.DataFrame:
    game_vec = []
    gpu_matrix = []

    for col in KNN_FEATURES:
        gpu_vals = gpus[col].values.astype(float)
        game_val = float(game.get(col) or 0)

        # Shared min/max across both datasets for this feature
        combined = np.concatenate([gpu_vals[~np.isnan(gpu_vals)], [game_val]])
        f_min, f_max = combined.min(), combined.max()

        if f_max > f_min:
            gpu_norm  = np.where(np.isnan(gpu_vals), 0.0,
                                 np.clip((gpu_vals - f_min) / (f_max - f_min), EPSILON, 1.0))
            game_norm = 0.0 if np.isnan(game_val) else float(
                np.clip((game_val - f_min) / (f_max - f_min), EPSILON, 1.0))
        else:
            gpu_norm  = np.where(np.isnan(gpu_vals), 0.0, 1.0)
            game_norm = 0.0 if np.isnan(game_val) else 1.0

        gpu_matrix.append(gpu_norm)
        game_vec.append(game_norm)

    gpu_matrix = np.column_stack(gpu_matrix)
    game_vec   = np.array(game_vec, dtype=float).reshape(1, -1)

    nn = NearestNeighbors(n_neighbors=min(k, len(gpus)), metric="euclidean")
    nn.fit(gpu_matrix)
    distances, indices = nn.kneighbors(game_vec)

    result = gpus.iloc[indices[0]].copy()
    result["distance"] = distances[0]
    result = result.reset_index(drop=True)
    result.index += 1

    print(f"\nTop-{k} GPUs by nearest neighbor (jointly normalized, euclidean distance):")
    print(result[DISPLAY_COLS_KNN].to_string())
    return result[DISPLAY_COLS_KNN]


def recommend(game_name: str, k: int = 5, mode: str = "min",
              method: str = "top_k", threshold: float = SOFT_THRESHOLD) -> pd.DataFrame:
    games, gpus = load_data(mode)
    game = find_game(games, game_name)

    print(f"\nGame   : {game['name']}")
    print(f"Mode   : {mode} requirements")
    print(f"Method : {method}\n")

    if method == "knn":
        return recommend_knn(game, gpus, k)
    return recommend_top_k(game, gpus, k, threshold)


def main():
    parser = argparse.ArgumentParser(description="Energy-aware GPU recommender")
    parser.add_argument("--game",      required=True,                      help="Game name (partial match supported)")
    parser.add_argument("--k",         type=int,   default=5,              help="Number of GPUs to return (default: 5)")
    parser.add_argument("--mode",      choices=["min", "recom"],           default="min",
                        help="Use minimum or recommended game requirements (default: min)")
    parser.add_argument("--method",    choices=["top_k", "knn"],           default="top_k",
                        help="Ranking method: top_k (filter+perf/watt) or knn (default: top_k)")
    parser.add_argument("--threshold", type=float, default=SOFT_THRESHOLD,
                        help="Soft filter threshold 0-1, only used with top_k (default: 0.80)")
    args = parser.parse_args()

    recommend(args.game, k=args.k, mode=args.mode, method=args.method, threshold=args.threshold)


if __name__ == "__main__":
    main()