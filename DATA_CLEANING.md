# GPU Specs: Data Cleaning and Feature Engineering

This document describes how we turn the raw scraped GPU specifications file into the model-ready feature matrix that the TDP and PSU prediction models train on. The data cleaning step is not just plumbing — every keep/drop decision is grounded in a physical, engineering, or modeling reason, and that reasoning is part of the project's contribution. The goal of this document is to make those reasons explicit so the design is easy to defend and easy to revisit.

## Why this stage matters

GPU power consumption follows a known physical relationship: dynamic power dissipation in a CMOS chip scales as `P ∝ f · C · V²`, where `f` is operating frequency, `C` is the switched capacitance (a function of how many transistors are switching and how big they are), and `V` is the supply voltage. Static power adds a leakage term that depends on transistor count, process node, and temperature.

We do not have direct measurements of `f`, `C`, or `V` for each GPU. What we have is a noisy spec sheet. Good cleaning means asking, for each column in the raw data: *does this column carry information about one of the physical drivers of power?* If yes, keep it; if not, drop it. This rule decides 80% of the cleaning policy without further argument.

## The raw data

The raw input is `data/raw/gpu_specs.csv` — a scrape of TechPowerUp's GPU database. It has **134 columns and 3,203 rows**. A "row" is a GPU model. Columns cover everything TechPowerUp displays: manufacturer, chip name, architecture, clock speeds, memory, render configuration (TMUs, ROPs, cores), theoretical FLOPs, board design (TDP, PSU, dimensions), graphics API support, and a lot of release-time marketing metadata.

Most of the 134 columns are not useful for predicting power. Three patterns account for most of the noise: duplicate columns scraped from the same page in different formats, columns for product lines we don't care about (mobile, integrated), and columns that exist only for one vendor or one generation (CUDA SDK version, RDNA3 chiplet sizes).

## Row filtering: which GPUs we keep

We model **desktop discrete single-GPU cards only**. Three filters apply, each with a physical reason:

- **Desktop only.** Mobile GPUs run at constrained voltages and frequencies to fit thermal envelopes of laptops. The `f · V²` operating point is fundamentally different, and `Board Design__TDP` for a laptop card reflects platform constraints, not the chip's intrinsic power profile. Including them would force the regressor to learn two different functions superimposed.
- **Discrete only.** Integrated graphics share power and thermal budget with the CPU. Their reported TDP is for the combined SoC, not the GPU alone, so the target variable is meaningfully different.
- **Single-GPU only.** Dual-GPU cards (GTX 690, R9 295X2) report aggregate TDP for two chips. Each spec column also doubles. Including them breaks the per-chip physical relationship we are trying to learn.

The filter is implemented by reading five raw columns — `Graphics Card__Production`, `Mobile Graphics__Release Date`, `Integrated Graphics__Release Date`, `Top__TMUS`, `Top__ROPS` — and dropping rows that don't pass. These five columns are only used to decide row membership and are not kept in the output.

After filtering, 1,646 rows remain.

## Column selection: what we keep, with reasons

We keep **28 columns** out of the 134. They fall into seven groups.

**Identity (2 cols)** — `Brand`, `Name`. Used to look up predictions later and to debug.

**Architecture and generation (3 cols)** — `Graphics Processor__Architecture` (e.g., Pascal, Turing, Ampere, RDNA2), `Graphics Card__Generation`, `Graphics Card__Release Date`. Architecture captures generational efficiency improvements that pure process-node shrinks miss — a 12nm Turing card and a 12nm Pascal refresh have different perf-per-watt because of design changes (improved schedulers, larger caches, lower-voltage operating points). Release date complements this with a continuous time axis, useful for newer cards whose architecture string we haven't seen before. The release date is a string like `"Aug 4th, 1986"` and gets parsed to a `release_year` integer.

**Physical chip properties (4 cols)** — `Process Size` (nm), `Transistors`, `Die Size`, `Density`. Process node directly affects leakage and operating voltage. Transistor count is a coarse proxy for switched capacitance `C` in the power equation. Die size and density relate to thermal density, which constrains the achievable `f · V²` operating point.

**Clocks (4 cols)** — `GPU Clock`, `Memory Clock`, `Base Clock`, `Boost Clock`. The `f` in `P ∝ f · V²`. Boost clock is the most important of the four — it's the frequency the card actually runs at under sustained load, which is what TDP is rated for.

**Memory subsystem (4 cols)** — `Memory Size`, `Memory Type`, `Memory Bus`, `Bandwidth`. Memory power is a significant fraction of total board power (often 20–40 W on a 200 W card). Memory type (GDDR6 vs GDDR6X vs HBM2) has very different power-per-bit profiles. Bus width and bandwidth determine how much memory traffic the card supports.

**Compute capacity (5 cols)** — `Shading Units`, `TMUs`, `ROPs`, `Tensor Cores`, `RT Cores`. These are the actual on-chip execution units. More units active = more switching activity = more power. Tensor and RT cores are zero on pre-2018 cards and that's fine — the model can use the absence as a signal.

**Theoretical performance (3 cols)** — `FP32 (float)`, `Pixel Rate`, `Texture Rate`. These are derived quantities (e.g., FP32 = cores × clock × 2), but they encode the relationship between hardware and throughput in a way the raw columns don't always make explicit. We keep FP32 only and drop FP64/FP16/BF16/TF32, because (a) FP32 is the gaming-relevant precision, (b) FP64 ratio is fixed by architecture so it adds no info once architecture is encoded, (c) BF16/TF32 are populated only for 7 datacenter cards.

**Power targets (2 cols)** — `Board Design__TDP` (the GPU board's thermal design point, our primary regression target) and `Board Design__Suggested PSU` (the vendor's recommended *total system* PSU, including CPU, RAM, drives, and a safety margin — our secondary system-level target). These are clearly different physical quantities and the report treats them as such.

**Compatibility (1 col)** — `Graphics Features__DirectX`. Not a power feature; we need it for feasibility filtering against game requirements (a game asking for DX12 cannot run on a DX11 card).

The remaining ~106 raw columns are dropped. The drop reasons cluster into a few categories: scraping artifacts (`source_file`), internal IDs (`Board Number`, `Part Number`), mobile/integrated columns (entirely NaN after our row filter), `Top__*` duplicates of detailed columns, physical board dimensions (TDP already captures the thermal envelope), legacy fixed-function shader counts (modern GPUs use unified shaders), vendor-specific niche fields (NVENC/NVDEC, CUDA versions, RDNA3 chiplet sub-die measurements), and cache hierarchy columns that are largely NaN and already captured by architecture.

## Value parsing: strings to numbers

Raw values are strings with units. The cleaner converts every kept numeric column to a single canonical unit and a plain float:

| Concept | Raw example | Cleaned column | Unit |
|---|---|---|---|
| Memory size | `'8 GB'`, `'512 MB'` | `memory_mb` | MB |
| Memory bandwidth | `'448.0 GB/s'`, `'1.5 TB/s'` | `memory_bandwidth_gbs` | GB/s |
| Texture rate | `'248.8 GTexel/s'` | `texture_rate` | GTexel/s |
| Pixel rate | `'124.4 GPixel/s'` | `pixel_rate` | GPixel/s |
| FP32 | `'1,200.0 GFLOPS'`, `'12.3 TFLOPS'` | `fp32_gflops` | GFLOPS |
| Boost clock | `'1750 MHz'` | `boost_clock_mhz` | MHz |
| Process node | `'14 nm'` | `process_nm` | nm |
| TDP | `'150 W'`, `'unknown'` | `tdp_w` | W |
| Release date | `'Aug 4th, 1986'` | `release_year` | year (int) |

A few columns contain explicit sentinel strings instead of numbers — most commonly the literal string `"unknown"` in `Board Design__TDP` and `Board Design__Suggested PSU`. The parser lowercases and strips whitespace, then checks for a sentinel set: `{"", "unknown", "n/a", "na", "nan", "none", "null", "tbd", "tba", "system shared", "system dependent"}`. Any match becomes `NaN`. After this, every kept numeric column passes through `pd.to_numeric(..., errors='coerce')` so anything else that didn't parse cleanly also becomes `NaN` rather than silently breaking downstream.

Categorical strings (architecture, generation, memory type) are kept as strings at this stage — they're encoded later in the vector-build step, not here.

## Range validation and deduplication

After parsing we apply lightweight physical sanity checks with **deliberately permissive lower bounds and strict upper bounds**:

- `tdp_w` ∈ (0, 700) W
- `psu_w` ∈ (0, 2000) W
- `memory_mb` ∈ (0, 65,536) MB

The lower bounds are kept open on purpose so pre-3D cards (NV1 at 2 W, Riva 128, EGA Wonder, etc.) and early integrated frame buffers stay in the dataset. This was an explicit choice — these old cards still teach the model something about the low end of the power curve, even if their architectures are no longer commercially relevant. We do not want a regressor that has only seen 75 W–500 W training points.

The upper bounds, however, are strict. Datacenter / compute-only cards (Instinct MI300 / MI350 at 750–1,400 W) are excluded because they operate under fundamentally different thermal-density and voltage envelopes from gaming GPUs and would distort the regression. Out-of-range values are coerced to NaN, and the affected (name, column, value) tuples are written to `data/cleaned/cleaning_report.csv` for audit.

TechPowerUp sometimes lists multiple SKUs of the same GPU (different board partners, minor revisions). We deduplicate on `name`, keeping the row with both targets present and the most populated columns.

## Handling missing targets: predict, don't drop

A meaningful fraction of desktop GPUs in this dataset have no published TDP or PSU — primarily pre-2006 cards (TDP simply wasn't a vendor-published spec back then) plus a handful of modern niche cards. The natural instinct is to drop these rows during cleaning. We don't.

The cleaner keeps any row that passes the GPU-type filter, the range checks, and the required-feature filter, even if `tdp_w` or `psu_w` is NaN. The training script then drops NaN-target rows internally before fitting. After training, the model predicts TDP and PSU for *every* row, including those with NaN ground truth. The final predictions CSV has both the actual value (which may be NaN) and the predicted value side-by-side, so we can flag these GPUs as "predicted only" downstream and still use them as candidates in the recommender.

Empirically, the current pipeline produces:
- **1,441 cleaned rows total**
- of which **1,117 have both targets populated** (training-eligible)
- and **324 are prediction-only** (293 missing TDP, 32 missing PSU, ~1 overlap)
- 296 of the 324 prediction-only rows are pre-3D-era cards (1996–2005)

This means NaN serves as the universal "unknown" signal throughout the pipeline. There's no parallel flag column and no second cleaned-data file.

## Canonical naming

Both the GPU side (this dataset) and the game requirements side use a single canonical naming convention for shared concepts, so the feasibility filter is a column-name intersection rather than a hand-mapped translation table. The shared names are:

`process_nm`, `tmus`, `rops`, `texture_rate`, `pixel_rate`, `direct_x`, `memory_mb`, `memory_speed_mhz`, `memory_bandwidth_gbs`, `memory_type`, `boost_clock_mhz`, `psu_w`.

GPU-only names: `tdp_w`, `fp32_gflops`, `transistors`, `die_size_mm2`, `density`, `architecture`, `generation`, `release_year`, `shading_units`, `tensor_cores`, `rt_cores`, `brand`, `name`.

Names that include units (`memory_mb`, `boost_clock_mhz`, `tdp_w`) are intentional — they prevent silent unit mismatches downstream.

## Feature engineering (vector-build step)

The cleaned CSV is human-readable but not yet model-ready. The vector-build step adds the numeric encodings models need:

- **One-hot encoding** for the three categorical string columns: `memory_type` (~19 categories), `architecture` (~25), `generation` (~15). Linear models (Ridge, Lasso, Linear Regression) and MLPs cannot accept strings as inputs, and label-encoding (Pascal=1, Turing=2, …) imposes a fake numerical ordering that doesn't reflect reality. One-hot encoding represents each category as its own binary column, which is faithful to the categorical-no-ordering structure of the data.
- **Standardization** (`(x - mean) / std`) for the numeric features, producing `standard_*` columns alongside the raw ones. Linear models and MLPs benefit from this; tree models (XGBoost, GB, RF) use the raw columns directly. Critically, the mean and standard deviation must be computed on the training fold only and applied to validation/test — computing them over the full dataset leaks test information into training and inflates reported metrics.
- **Min-max normalization** (`(x - min) / (max - min)`) for the perf-score input features, producing `norm_*` columns. Same train-fold-only fitting rule applies.
- **Derived features.** A geometric mean of the `norm_*` columns produces a single `perf_score` capability number per GPU. This serves as the "how capable is this GPU" signal that the recommender's performance-per-watt ranking divides predicted TDP into.

## Pipeline summary

```
data/raw/gpu_specs.csv                  3,203 rows, 134 cols
        │
        │  src/clean_gpu_requirements.py
        │  → row filters, column selection, value parsing,
        │    range validation, deduplication
        ▼
data/cleaned/gpu_specs_cleaned.csv      1,441 rows × 28 cols, human-readable
                                        (1,117 with both targets,
                                          324 prediction-only)
        │
        │  src/build_gpu_power_vectors.py
        │  → one-hot encoding, standardization, normalization,
        │    derived perf_score
        ▼
data/vectors/gpu_power_vectors.csv      1,441 rows × 136 cols, model-ready
        │
        │  src/train_gpu_specs_models.py
        │  → drop NaN-target rows internally, train, predict on all rows
        ▼
data/results/gpu_power_predictions.csv  1,441 rows with actual + predicted
                                        TDP and PSU side-by-side
```

The two-stage split (cleaned → vectors) exists because the two stages have different reasons to change. Cleaning rules change when we discover new sentinel strings, new edge cases in the raw scrape, or new physical filters. Vector encoding changes when we add a new model class with different input expectations. Keeping them separate means a change to one does not require re-running the other.
