"""
Scrapes GPU G3D Mark scores from Passmark.

Output: data/raw/passmark_benchmarks.csv
"""

import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pathlib import Path
from rapidfuzz import process, fuzz

list_url = "https://www.videocardbenchmark.net/gpu_list.php"

# site blocks requests without a browser User-Agent
headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

specs_path  = "data/cleaned/gpu_specs_cleaned.csv"
output_path = "data/raw/passmark_benchmarks.csv"


# makes a GET request and returns the page HTML as a string
def fetch(url):
    r = requests.get(url, headers=headers, timeout=15)
    return r.text


# parses the PassMark GPU list page and returns each GPU's name and G3D Mark score
def parse_gpu_list(html):
    table = BeautifulSoup(html, "html.parser").find("table", id="cputable")
    rows  = []

    for tr in table.select("tbody tr"):
        if not tr.get("id", "").startswith("gpu"):
            continue

        tds       = tr.find_all("td")
        name_link = tds[0].find("a") if len(tds) >= 2 else None
        if not name_link:
            continue

        g3d = tds[1].get_text(strip=True).replace(",", "")
        if not g3d.isdigit():
            continue

        rows.append({
            "gpu_name": name_link.get_text(strip=True),
            "g3d_mark": int(g3d),
        })
    return rows


# strips brand/AIB prefixes so "NVIDIA GeForce RTX 3090" and "GeForce RTX 3090" both reduce to "rtx 3090" before matching
def strip_vendor(name):
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    name = re.sub(r"\b(nvidia|amd|ati|intel|geforce|radeon|arc|gecube|powercolor|sapphire|xfx)\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


# pulls out any 3+ digit numbers from a name (e.g. "rtx 3090" → {"3090"}) to confirm the match is actually the right GPU model
def model_numbers(name):
    return set(re.findall(r"\d{3,}", name))


# for each GPU in our specs dataset, finds the best matching GPU in the PassMark list by name similarity and model number agreement
def match_gpus(gpu_list, specs_df, threshold=72):
    pm_names = [g["gpu_name"] for g in gpu_list]
    pm_norms = [strip_vendor(n) for n in pm_names]

    matches = []
    for _, row in specs_df.iterrows():
        spec_norm  = strip_vendor(row["name"])
        spec_nums  = model_numbers(spec_norm)
        candidates = process.extract(spec_norm, pm_norms, scorer=fuzz.token_sort_ratio, limit=5, score_cutoff=threshold)

        if not candidates:
            continue

        best, best_score = None, -1
        for norm, score, idx in candidates:
            pm_nums = model_numbers(norm)
            if spec_nums and pm_nums and not (spec_nums & pm_nums):
                continue
            if score > best_score:
                best_score, best = score, gpu_list[idx]

        if best:
            matches.append({
                "spec_name":   row["name"],
                "pm_name":     best["gpu_name"],
                "g3d_mark":    best["g3d_mark"],
                "match_score": round(best_score, 1),
            })

    return pd.DataFrame(matches)


# main entry point — fetches the GPU list, matches it against our specs, and saves the result
def scrape_all():
    print("Fetching GPU list...", flush=True)
    gpu_list = parse_gpu_list(fetch(list_url))
    print(f"  {len(gpu_list)} GPUs parsed", flush=True)

    specs_df   = pd.read_csv(specs_path)
    matches_df = match_gpus(gpu_list, specs_df)
    print(f"  {len(matches_df)} / {len(specs_df)} GPUs matched", flush=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    matches_df.to_csv(output_path, index=False)
    print(f"Saved {len(matches_df)} rows → {output_path}", flush=True)


if __name__ == "__main__":
    scrape_all()
