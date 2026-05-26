# Game Requirements: Data Cleaning and Feature Engineering

This document describes how we turn the raw scraped video-game requirements file into the model-ready feature vectors used by the recommender. The GPU side gives us the candidate hardware; the game side gives us the workload constraints. Both sides have to speak the same language and use the same canonical names so the feasibility filter is just a column-name comparison, not a hand-mapped translation table.

## Why this stage matters

The recommender is a thresholds problem. A game says "you need at least this much memory bandwidth, this many texture rate, this version of DirectX." The recommender treats those numbers as **feasibility thresholds**: a GPU is a valid candidate only if its capabilities meet or exceed the game's thresholds. So the game cleaning step has two jobs:

1. Get the threshold numbers into **clean, comparable, numeric units** that match the GPU side exactly.
2. Build a **per-game capability vector** that lets us compute a "how hard is this game" score, mirroring the perf-score we build for each GPU.

Engineering-wise, most of the cleaning is about handling a noisy scrape: zeros where there should be NaN, size strings with different units, the same game listed multiple times for different platforms, and a huge number of columns we don't need because they describe CPU requirements (we model GPU power, not full-system power) or display connections (which don't affect feasibility).

## The raw data

The raw input is `data/raw/videogame_requirements.csv`, scraped from a PC-games requirements aggregator. It has **10,849 rows and 90 columns**. A "row" is a game-platform listing. Columns split into four groups:

- **Game identity (2 cols)** — `Name`, `Release_Date`.
- **Minimum CPU requirements (15 cols)** — `Min_CPU_CPU_Speed`, `Min_CPU_TDP`, `Min_CPU_L1_Cache`, etc.
- **Recommended CPU requirements (15 cols)** — mirror of the above with `Recom_CPU_*` prefix.
- **Minimum GPU requirements (24 cols)** — `Min_GPU_TMUs`, `Min_GPU_Texture_Rate`, `Min_GPU_Memory`, `Min_GPU_PSU`, etc.
- **Recommended GPU requirements (24 cols)** — mirror with `Recom_GPU_*` prefix.
- **System requirements (10 cols)** — `Min_RAM`, `Recom_RAM`, `Min_VRAM`, `Recom_VRAM`, `Min_OS`, `Recom_OS`, `Min_Direct_X`, `Recom_Direct_X`, `Min_HDD_Space`, `Recom_HDD_Space`.

Two structural facts shape every cleaning decision below. **First**, the same game appears multiple times because the scrape lists separate rows per platform (Steam, Epic, GOG) and per language edition. **Second**, the scrape uses literal zero values to encode "missing" for several columns — a 2D shooter doesn't have a meaningful "Texture Rate" requirement, so it gets `0`. We need to distinguish "the value is genuinely zero" from "the field is missing," and for our columns the latter is always the case.

## Column selection: what we drop and why

We keep **26 columns** out of the 90.

**Drop all CPU columns (30 cols).** The project scope is GPU power, not full-system power. We don't model CPU TDP, we don't recommend CPUs, and including CPU requirements as features would let the model leak CPU information into GPU decisions. The Min_CPU and Recom_CPU groups are dropped in one sweep.

**Drop non-power, non-capability GPU columns (20 cols, 10 per Min/Recom).** These are the columns that are either too sparse to be useful, too vendor-specific, or describe physical attributes that don't enter the feasibility check:

- `Tensor_Cores` — the game-side scrape rarely populates this and it's an architecture-dependent count anyway.
- `GD_RATING` — a third-party scalar score; we explicitly want to avoid relying on a black-box external rating as a feasibility signal.
- `DVI_Connection`, `HDMI_Connection`, `DisplayPort_Connection` — display output compatibility, not a power or capability constraint.
- `Power_Connector` — the type of PSU connector required (6-pin / 8-pin), correlated with TDP but partially a label leak.
- `Release_Price`, `Resolution`, `Best_RAM_Match`, `Best_Resolution` — marketing-derived fields that don't fit the thresholds-on-capability framework.

**Keep these 14 GPU columns per side (28 cols total).** They are the capability dimensions that directly mirror the GPU spec sheet: `Process`, `TMUs`, `Texture_Rate`, `ROPs`, `Pixel_Rate`, `Direct_X`, `Shader`, `Open_GL`, `Memory`, `Memory_Speed`, `Memory_Type`, `Memory_Bandwidth`, `Boost_Clock`, `PSU`. After splitting into min/recom files (next section), each side keeps 14 GPU columns + 12 shared columns = 26.

**Keep all 10 system columns plus identity** (12 cols total: Name, Release_Date, plus 10 system columns). These describe non-GPU thresholds the recommender will still want to know about (a game needing 16 GB system RAM is a deployment constraint even if it doesn't change the GPU choice).

## Value parsing: strings, sentinels, units

Three classes of value-level cleaning happen in this stage.

**Zero as a sentinel for missing.** For 13 GPU spec columns (`Process`, `TMUs`, `Texture_Rate`, `ROPs`, `Pixel_Rate`, `Shader`, `Open_GL`, `Memory`, `Memory_Speed`, `Memory_Type`, `Memory_Bandwidth`, `Boost_Clock`, `PSU`), a value of `0` means "this game didn't specify this requirement," not "this game requires zero." We replace those zeros with `NaN` so downstream models see a proper missing-value signal. `Memory_Type` additionally has `-1` as an invalid sentinel for unknown memory type, also coerced to `NaN`. The same logic applies to `Min_RAM`, `Recom_RAM`, etc. — string forms `"0"` and `"0MB"` become `NaN`.

**Size string parsing.** The system columns store RAM, VRAM, and HDD as strings with mixed units: `"1 GB"`, `"512MB"`, `"1.953125GB"`. The parser normalizes everything to MB:

| Concept | Raw example | Cleaned column | Unit |
|---|---|---|---|
| System RAM | `"8 GB"`, `"16384MB"` | `min_ram_mb`, `recom_ram_mb` | MB |
| Video RAM | `"4 GB"`, `"2048MB"` | `min_vram_mb`, `recom_vram_mb` | MB |
| Disk space | `"50 GB"`, `"40GB"`, `"40000MB"` | `min_hdd_mb`, `recom_hdd_mb` | MB |

Negative numbers and unparseable strings become `NaN`. Whitespace and OS strings (Windows 10, Windows 11) are stripped of leading/trailing spaces.

**Release date parsing.** `Release_Date` is converted from `"2021-04-23"` strings to proper `datetime64` timestamps with `pd.to_datetime`. We don't extract release_year as a separate feature here — the date itself is sortable.

## Deduplication: one row per game

After parsing, the scrape's per-platform duplication needs collapsing. We group rows by `Name` and aggregate:

- For numeric columns (every GPU spec, RAM/VRAM/HDD sizes, DirectX version), take the **max** across platforms. Reason: different platforms occasionally list slightly different requirements; the max is the conservative feasibility threshold (the GPU has to satisfy the strictest variant).
- For string columns (OS, memory type, release date), take the **first** non-null. These don't materially vary across platforms.

This collapses ~10,849 platform-rows into roughly 7,292 unique games.

## Drop rows missing all GPU requirements

A small fraction of games have either no `Min_GPU_*` data or no `Recom_GPU_*` data — usually older or extremely lightweight games where the scrape couldn't find a hardware requirement at all. Rows that are entirely NaN across all `Min_GPU_*` columns, or entirely NaN across all `Recom_GPU_*` columns, are dropped. A row that has *some* GPU info on both sides is kept; the feasibility filter can handle column-level NaN gracefully but cannot do anything with a row that has no GPU requirement signal at all.

## The min/recom split: why two files

A single game gives us **two different feasibility constraints**: a minimum-requirements vector (the game runs at all) and a recommended-requirements vector (the game runs well). These are different operating points and produce different recommendations — using minimum thresholds returns the broadest candidate set; using recommended thresholds returns the candidate set the user is actually expected to deploy.

Rather than carry both constraint vectors in one row and force every downstream consumer to remember which prefix to use, we split into two files at the cleaning stage:

- `data/cleaned/game_reqs_min.csv` — one row per game, GPU columns are the **minimum** thresholds.
- `data/cleaned/game_reqs_recom.csv` — one row per game, GPU columns are the **recommended** thresholds.

Both files contain the **same set of games** (validated by assertion) and the **same shared system columns** (RAM, VRAM, OS, DirectX, HDD, name, release_date). The only difference is which GPU thresholds are loaded. The recommender's "recommended requirement track" reads `game_reqs_recom.csv`; an alternate "minimum requirement track" reads `game_reqs_min.csv` with no other code changes.

## Canonical naming

The same canonical names used on the GPU side apply here. For shared concepts the column names match exactly so the feasibility filter is a column-name comparison rather than a translation step:

`process_nm`, `tmus`, `rops`, `texture_rate`, `pixel_rate`, `direct_x`, `memory_mb`, `memory_speed_mhz`, `memory_bandwidth_gbs`, `memory_type`, `boost_clock_mhz`, `psu_w`.

Game-only columns keep their own names: `min_ram_mb`, `recom_ram_mb`, `min_vram_mb`, `recom_vram_mb`, `min_os`, `recom_os`, `min_direct_x`, `recom_direct_x`, `min_hdd_mb`, `recom_hdd_mb`, `release_date`, `name`. The `min_` / `recom_` prefix here refers to **system-level requirements** that exist alongside the GPU spec vector — distinct from the file-level split which is about minimum-vs-recommended GPU specs.

Names that include units (`memory_mb`, `boost_clock_mhz`, `psu_w`) are intentional. They prevent silent unit mismatches when the GPU side joins.

## Feature engineering (vector-build step)

The cleaned CSVs are human-readable but not yet useful for ranking. The vector-build step adds three things:

**Hard/soft filter counts.** Each game has a `hard_filter_count` (how many of `memory_mb`, `direct_x` are specified) and a `soft_filter_count` (how many of `texture_rate`, `pixel_rate`, `memory_bandwidth_gbs`, `tmus`, `rops` are specified). These let the recommender know how strict the feasibility check can be for each game — a game that only specifies VRAM and DirectX has 2 hard filters and 0 soft filters, so we cannot reject candidates on texture rate for that game.

**Min-max normalization of performance features.** The same seven capability features the GPU side normalizes — `texture_rate`, `pixel_rate`, `memory_bandwidth_gbs`, `tmus`, `rops`, `memory_speed_mhz`, `boost_clock_mhz` — are min-max normalized per column to produce `norm_*` columns in `[ε, 1.0]`. The leakage caveat from the GPU side applies symmetrically: when we later move to proper train/test splits, these min/max values must be fit on the training fold only and re-applied to test.

**Performance score.** A geometric mean across the available `norm_*` columns gives each game a single `perf_score` value, representing "how demanding is this game on raw GPU capability." This score lives on the same `[ε, 1.0]` scale as the GPU-side perf_score. Because both sides use the same normalization recipe and the same canonical column names, a game's perf_score is directly comparable to a GPU's perf_score: roughly, a candidate is feasibility-overshooting if its GPU perf_score is much higher than the game's perf_score requirement.

We also track `perf_feature_count` per game so the recommender knows how many features contributed to that geometric mean — a perf_score computed from 7 features is more trustworthy than one computed from 2.

## Validation gates

The cleaning script asserts several invariants before saving, so silent corruption gets caught at build time rather than during training:

- No row has all-NaN GPU columns (we dropped those).
- Game names are unique within each file.
- `direct_x` is never NaN — every game must specify a DirectX version.
- `memory_mb` coverage is above 95% — VRAM requirement is a near-universal field; falling below 95% means the parser broke.
- All numeric columns are non-negative.
- The min and recom files have identical row counts and identical game sets.
- Column count matches the documented schema (26 cleaned, 37 vectors).

If any assertion fails the build halts. This protects downstream consumers — the recommender, the LTR ranker, and the evaluation script — from operating on subtly broken data.

## Pipeline summary

```
data/raw/videogame_requirements.csv     10,849 rows, 90 cols
        │
        │  src/clean_game_requirements.py
        │  → drop CPU columns, drop sparse GPU columns,
        │    zero-as-NaN replacement, size-string parsing,
        │    deduplicate by Name, drop all-NaN GPU rows,
        │    split into min/recom, canonical naming
        ▼
data/cleaned/game_reqs_min.csv          ~7,292 rows × 26 cols
data/cleaned/game_reqs_recom.csv        ~7,292 rows × 26 cols
        │
        │  src/build_vectors.py
        │  → hard/soft filter counts, min-max normalize
        │    perf features, geometric-mean perf_score
        ▼
data/vectors/game_vectors_min.csv       ~7,292 rows × 37 cols
data/vectors/game_vectors_recom.csv     ~7,292 rows × 37 cols
        │
        │  src/recommender.py (downstream)
        │  → joins with GPU vectors on canonical names,
        │    runs feasibility filter + top-k ranking
        ▼
recommendation results
```

The two-stage split (cleaned → vectors) exists for the same reason as on the GPU side. Cleaning rules change when the raw scrape evolves or new sentinel values appear; vector encoding changes when we add a new model class. Keeping the stages separate means a change to one does not require re-running the other, and the cleaned CSVs stay human-readable for debugging.

## Mirroring with the GPU side

By the end of cleaning + vector build, the game side and the GPU side share:

- The same canonical column names for every shared concept.
- The same normalization recipe (min-max → `[ε, 1.0]`).
- The same perf_score recipe (geometric mean of `norm_*` columns, with feature-count tracking).
- The same unit conventions (`memory_mb`, `boost_clock_mhz`, `memory_bandwidth_gbs`, `psu_w`).

This mirroring is deliberate. The recommender's job is to ask "is this GPU's capability at least as high as this game's requirement?" Doing that comparison correctly requires that both sides have spent equal care expressing their values in the same units, with the same scaling, under the same naming. The cleaning step is where that contract is established; everything downstream depends on it being right.
