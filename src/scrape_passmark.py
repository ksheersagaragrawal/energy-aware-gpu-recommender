"""
Scrapes GPU benchmark data from VideoCardBenchmark (PassMark):
  - G3D Mark score (overall gaming performance)
  - Per-DirectX FPS scores (DX9, DX10, DX11, DX12)
  - GPU Compute score (Ops/Sec)

Strategy:
  1. Fetch full GPU list (1 request) → all names + G3D Mark scores
  2. Fuzzy-match against gpu_specs_cleaned.csv to find overlapping GPUs
  3. Fetch detail pages only for matched GPUs → DX-specific FPS scores

Outputs:
  data/raw/passmark_benchmarks.csv  — matched GPUs with all benchmark scores
"""

import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from rapidfuzz import process, fuzz

BASE_URL   = "https://www.videocardbenchmark.net"
LIST_URL   = f"{BASE_URL}/gpu_list.php"
DETAIL_URL = f"{BASE_URL}/gpu.php"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

SPECS_PATH  = Path(__file__).resolve().parents[1] / "data" / "cleaned" / "gpu_specs_cleaned.csv"
OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "passmark_benchmarks.csv"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def fetch(url, params=None, retries=3, base_delay=2.0):
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                return r.text
            print(f"  HTTP {r.status_code} for {url} params={params}")
        except requests.RequestException as e:
            print(f"  Request error (attempt {attempt+1}): {e}")
        time.sleep(base_delay * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_gpu_list(html):
    """Parse GPU list table → list of {gpu_name, g3d_mark, gpu_id}."""
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="cputable")
    if not table:
        raise RuntimeError("Could not find #cputable on GPU list page.")

    rows = []
    for tr in table.select("tbody tr"):
        tr_id = tr.get("id", "")
        id_match = re.match(r"gpu(\d+)", tr_id)
        if not id_match:
            continue
        gpu_id = int(id_match.group(1))

        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        name_link = tds[0].find("a")
        if not name_link:
            continue
        gpu_name = name_link.get_text(strip=True)

        g3d_text = tds[1].get_text(strip=True).replace(",", "")
        if not g3d_text.isdigit():
            continue

        rows.append({
            "gpu_name": gpu_name,
            "g3d_mark": int(g3d_text),
            "gpu_id":   gpu_id,
        })
    return rows


def parse_gpu_detail(html):
    """
    Parse GPU detail page → {dx9_fps, dx10_fps, dx11_fps, dx12_fps, gpu_compute_ops}.
    Table: <table id="test-suite-results">
    """
    soup = BeautifulSoup(html, "html.parser")
    result = {
        "dx9_fps": None, "dx10_fps": None,
        "dx11_fps": None, "dx12_fps": None,
        "gpu_compute_ops": None,
    }
    table = soup.find("table", id="test-suite-results")
    if not table:
        return result

    for tr in table.find_all("tr"):
        th = tr.find("th")
        td = tr.find("td")
        if not th or not td:
            continue
        label = th.get_text(strip=True).lower()
        num_match = re.search(r"([\d,]+)", td.get_text(strip=True))
        if not num_match:
            continue
        value = int(num_match.group(1).replace(",", ""))

        if "directx 9" in label:
            result["dx9_fps"] = value
        elif "directx 10" in label:
            result["dx10_fps"] = value
        elif "directx 11" in label:
            result["dx11_fps"] = value
        elif "directx 12" in label:
            result["dx12_fps"] = value
        elif "gpu compute" in label:
            result["gpu_compute_ops"] = value

    return result


# ---------------------------------------------------------------------------
# Fuzzy matching helpers
# ---------------------------------------------------------------------------

def normalize(name: str) -> str:
    """Lowercase, strip vendor prefixes and punctuation for matching."""
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\b(nvidia|amd|ati|intel|geforce|radeon|arc|gecube|powercolor|sapphire|xfx)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def extract_model_numbers(name: str) -> set:
    """Extract numeric tokens that identify the GPU model (e.g. '3090', '6800')."""
    return set(re.findall(r"\d{3,}", name))


def match_gpu_list_to_specs(gpu_list: list[dict], specs_df: pd.DataFrame, threshold=72):
    """
    For each GPU in our specs, find the best-matching PassMark entry using
    rapidfuzz token_sort_ratio + model number agreement.

    threshold: rapidfuzz score 0-100 (72 is a good balance of precision/recall).
    Returns a DataFrame with columns: spec_name, pm_name, gpu_id, g3d_mark, match_score.
    """
    pm_names  = [e["gpu_name"] for e in gpu_list]
    pm_norms  = [normalize(n) for n in pm_names]

    matches = []
    for _, spec_row in specs_df.iterrows():
        spec_name = spec_row["name"]
        spec_norm = normalize(spec_name)
        spec_nums = extract_model_numbers(spec_norm)

        # rapidfuzz: find top candidates by token sort ratio
        results = process.extract(
            spec_norm, pm_norms,
            scorer=fuzz.token_sort_ratio,
            limit=5,
            score_cutoff=threshold,
        )
        if not results:
            continue

        # Among candidates, prefer those whose model numbers overlap
        best = None
        best_score = -1
        for pm_norm_match, score, idx in results:
            pm_nums = extract_model_numbers(pm_norm_match)
            # Require at least one shared model number (if spec has any)
            if spec_nums and pm_nums and not (spec_nums & pm_nums):
                continue
            if score > best_score:
                best_score = score
                best = gpu_list[idx]

        if best:
            matches.append({
                "spec_name":   spec_name,
                "pm_name":     best["gpu_name"],
                "gpu_id":      best["gpu_id"],
                "g3d_mark":    best["g3d_mark"],
                "match_score": round(best_score, 1),
            })

    return pd.DataFrame(matches)


# ---------------------------------------------------------------------------
# Main scrape
# ---------------------------------------------------------------------------

def scrape_all():
    # Step 1: Get full GPU list (1 request)
    print("Step 1: Fetching GPU list...", flush=True)
    list_html = fetch(LIST_URL)
    if not list_html:
        raise RuntimeError("Failed to fetch GPU list.")
    gpu_list = parse_gpu_list(list_html)
    print(f"  Parsed {len(gpu_list)} GPUs from PassMark list.", flush=True)

    # Step 2: Fuzzy-match against our GPU specs
    print("\nStep 2: Fuzzy-matching to gpu_specs_cleaned.csv...", flush=True)
    specs_df = pd.read_csv(SPECS_PATH)
    matches_df = match_gpu_list_to_specs(gpu_list, specs_df)
    print(f"  Matched {len(matches_df)} / {len(specs_df)} GPUs (threshold=72).", flush=True)
    print(matches_df.head(5).to_string(), flush=True)

    # Step 3: Fetch detail pages only for matched GPUs
    total = len(matches_df)
    print(f"\nStep 3: Fetching detail pages for {total} matched GPUs...", flush=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in matches_df.reset_index(drop=True).iterrows():
        if i % 25 == 0:
            print(f"  [{i}/{total}] {row['pm_name']}", flush=True)

        html = fetch(DETAIL_URL, params={"gpu": row["pm_name"], "id": row["gpu_id"]})
        detail = parse_gpu_detail(html) if html else {
            "dx9_fps": None, "dx10_fps": None,
            "dx11_fps": None, "dx12_fps": None,
            "gpu_compute_ops": None,
        }
        rows.append({**row.to_dict(), **detail})
        time.sleep(0.8)

        # Save incrementally every 50 rows so progress isn't lost if interrupted
        if (i + 1) % 50 == 0:
            pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
            print(f"  [checkpoint] saved {i+1} rows", flush=True)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result_df)} rows → {OUTPUT_PATH}", flush=True)

    # Summary
    complete = result_df.dropna(subset=["dx9_fps", "dx12_fps"])
    print(f"GPUs with complete DX scores: {len(complete)}")
    print(result_df[["spec_name", "pm_name", "g3d_mark", "dx9_fps", "dx11_fps", "dx12_fps", "match_score"]].head(10).to_string())
    return result_df


if __name__ == "__main__":
    scrape_all()