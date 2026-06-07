"""
Energy-aware GPU recommender.
Output: top GPU recommendations for a given game
"""

import argparse
import pickle
import numpy as np
import pandas as pd

game_vectors = {
    "min":   "data/vectors/game_vectors_min.csv",
    "recom": "data/vectors/game_vectors_recom.csv",
}
gpu_vectors    = "data/vectors/gpu_power_vectors.csv"
model_path     = "models/gpu_performance_model.pkl"
soft_threshold = 0.80

soft_filter_cols = ["texture_rate", "pixel_rate", "memory_bandwidth_gbs", "tmus", "rops"]


# looks up the game by exact name first, falls back to partial match
def find_game(games, game_name):
    exact = games[games["name"].str.lower() == game_name.lower()]
    if not exact.empty:
        return exact.iloc[0]

    partial = games[games["name"].str.lower().str.contains(game_name.lower(), regex=False)]
    if partial.empty:
        raise ValueError(f"No game found matching '{game_name}'.")
    if len(partial) > 1:
        print(f"{len(partial)} partial matches — using: {partial['name'].iloc[0]!r}")
    return partial.iloc[0]


# removes GPUs that don't meet the game's minimum VRAM and DirectX requirements
def hard_filter(gpus, game):
    mask     = pd.Series(True, index=gpus.index)
    vram_req = game.get("min_vram_mb")
    dx_req   = game.get("min_direct_x")

    if pd.notna(vram_req) and vram_req > 0:
        mask &= gpus["memory_mb"] >= vram_req
    if pd.notna(dx_req) and dx_req > 0:
        mask &= gpus["direct_x"] >= dx_req

    print(f"[hard_filter]  {mask.sum():4d} / {len(gpus)} GPUs pass  (vram>={vram_req} MB, directx>={dx_req})")
    return gpus[mask].copy()


# removes GPUs that fall below the threshold percentage of the game's throughput requirements
def soft_filter(gpus, game, threshold):
    mask    = pd.Series(True, index=gpus.index)
    applied = []

    for col in soft_filter_cols:
        req = game.get(col)
        if pd.isna(req) or req <= 0:
            continue
        min_val = req * threshold
        mask   &= gpus[col] >= min_val
        applied.append(f"{col}>={min_val:.1f}")

    print(f"[soft_filter]  {mask.sum():4d} / {len(gpus)} GPUs pass  (threshold={threshold:.0%}, filters: {', '.join(applied)})")
    return gpus[mask].copy()


# builds the feature matrix the XGBoost model expects — reads pre-encoded memory type columns + median-fills continuous specs
def prepare_features(gpus, feature_cols, mem_type_cols):
    df = gpus.copy()

    # model uses "mem_gddr6" style names; gpu vectors have "memory_type_GDDR6" style — map between them
    for col in mem_type_cols:
        source = "memory_type_" + col.replace("mem_", "").upper()
        df[col] = df[source].fillna(0).astype(int) if source in df.columns else 0

    for col in [c for c in feature_cols if not c.startswith("mem_")]:
        df[col] = df[col].fillna(df[col].median()) if col in df.columns else 0.0

    return np.nan_to_num(df[feature_cols].values.astype(float), nan=0.0)


# ranks filtered GPUs by pre-computed perf_score / TDP and returns the top k
def recommend_top_k(game, gpus, k, threshold):
    feasible = hard_filter(gpus, game)
    feasible = soft_filter(feasible, game, threshold)

    if feasible.empty:
        print("No GPUs passed all filters.")
        return pd.DataFrame()

    feasible["perf_per_watt"] = feasible["perf_score"] / feasible["tdp_w"]
    result = feasible.sort_values("perf_per_watt", ascending=False).head(k)
    result = result[["brand", "name", "memory_mb", "direct_x", "tdp_w", "perf_score", "perf_per_watt"]].reset_index(drop=True)
    result.index += 1

    print(f"\nTop-{k} GPUs by performance per watt:")
    print(result.to_string())
    return result


# ranks filtered GPUs by XGBoost predicted G3D Mark / TDP and returns the top k
def recommend_ml(game, gpus, k, threshold, payload):
    feasible = hard_filter(gpus, game)
    feasible = soft_filter(feasible, game, threshold)

    if feasible.empty:
        print("No GPUs passed all filters.")
        return pd.DataFrame()

    feasible["pred_g3d"]          = payload["model"].predict(prepare_features(feasible, payload["feature_cols"], payload["mem_type_cols"]))
    feasible["pred_g3d_per_watt"] = feasible["pred_g3d"] / feasible["tdp_w"]

    result = feasible.sort_values("pred_g3d_per_watt", ascending=False).head(k)
    result = result[["brand", "name", "memory_mb", "direct_x", "tdp_w", "pred_g3d", "pred_g3d_per_watt"]].reset_index(drop=True)
    result.index += 1

    print(f"\nTop-{k} GPUs by predicted G3D Mark per watt (ML):")
    print(result.to_string())
    return result


# main entry point — loads data, finds the game, and dispatches to the right method
def recommend(game_name, k=5, mode="min", method="top_k", threshold=soft_threshold):
    games = pd.read_csv(game_vectors[mode])
    gpus  = pd.read_csv(gpu_vectors)
    game  = find_game(games, game_name)

    print(f"\nGame   : {game['name']}")
    print(f"Mode   : {mode} requirements")
    print(f"Method : {method}\n")

    if method == "ml":
        with open(model_path, "rb") as f:
            payload = pickle.load(f)
        return recommend_ml(game, gpus, k, threshold, payload)
    return recommend_top_k(game, gpus, k, threshold)


def main():
    parser = argparse.ArgumentParser(description="Energy-aware GPU recommender")
    parser.add_argument("--game",      required=True,                           help="Game name (partial match supported)")
    parser.add_argument("--k",         type=int,   default=5,                   help="Number of GPUs to return (default: 5)")
    parser.add_argument("--mode",      default="min",   choices=["min", "recom"], help="Use minimum or recommended requirements (default: min)")
    parser.add_argument("--method",    default="top_k", choices=["top_k", "ml"], help="Ranking method (default: top_k)")
    parser.add_argument("--threshold", type=float, default=soft_threshold,        help="Soft filter threshold 0-1 (default: 0.80)")
    args = parser.parse_args()

    recommend(args.game, k=args.k, mode=args.mode, method=args.method, threshold=args.threshold)


if __name__ == "__main__":
    main()
