# Appendix — Benchmark Datasets

This appendix documents the location, size, source, and loading strategy for all benchmark datasets cited in the proposal. Datasets are stored in `data/` at the repository root, with structure matching the reproducible harness loaders under `reproduce/tables/_fuzzy_models.py`.

## Dataset Inventory

### Classification

#### PhiUSIIL Phishing URL (Table 4.1, §4.3.2, Ch 6)
- **File:** `data/PhiUSIIL_Phishing_URL_Dataset.csv`
- **Size:** 235,000 rows × 54 features (binary classification)
- **Role:** Large-scale classification benchmark; featured in Table 4.1 (MoG baselines), Table 6.1 (model family), norm/conorm comparisons
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets), dataset ID 967
- **Loading:** `reproduce/tables/_fuzzy_models.py::load_phiusiil()`
- **Status:** Repaired in this pass; previous loader path was broken (tribble-fis/gaussian_mixture/ deleted upstream); now loads from local CSV

#### RT-IOT2022 (Table 4.4, §4.4, pending)
- **File:** `data/RT_IOT2022.csv`
- **Size:** 123,000 rows × 83 features, 12 classes
- **Role:** Open-set detection testbed (planned); intended to replace Glass (214 samples) as the large-scale anomaly/open-set partner
- **Source:** [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/datasets), dataset ID 952
- **Loading:** `reproduce/tables/_fuzzy_models.py::load_rt_iot2022()` (not yet wired)
- **Status:** In the repository and wired (2026-08-12), both roles measured. Open-set claim at five seeds (Table 4.7b): complement rule loses to Isolation Forest at this scale. Plain classification/timing at ten seeds (Table 4.4, `table_4_1_mog_baselines.py`): MoG trains in 37.42 ± 0.64 s at 92.7 ± 0.2% accuracy against Random Forest's 99.9 ± 0.0%.

#### Glass (UCI) (Tables 4.6–4.7, Fig 4.2, §4.4)
- **File:** Auto-fetched via sklearn or `ucimlrepo` (id 41)
- **Size:** 214 rows × 9 features, 6 classes
- **Role:** Small anomaly-detection baseline; also used as open-set stress test
- **Status:** Already available; no manual download needed

### Regression

#### Concrete Compressive Strength (Tables 4.1–4.5, Ch 6, every regression table)
- **File:** `data/Concrete_Data.csv` (auto-built from `AEEM6097/project-data/Concrete_Data.xls` if absent)
- **Size:** 1,030 rows × 8 features → 1 target
- **Role:** Small-scale regression benchmark; the *only* measured regression dataset in the proposal
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets), dataset ID 165
- **Loading:** `reproduce/tables/_fuzzy_models.py::load_concrete()`
- **Status:** Measured at ten seeds; Chapter 4 §4.3.2 decision rule runs on Concrete; Chapter 6 model family runs on Concrete
- **Gap:** No large-scale regression partner. Chapter 7 names this as a structural gap (A.7.1).

#### Bike Sharing Demand (Tables 4.5, 6.1, large regression benchmark)
- **File:** `data/bikeshare-hour.csv`
- **Size:** 17,379 rows × 14 numeric features → 1 target (hourly bike rental demand)
- **Role:** Large-scale regression benchmark; scale partner for Concrete (1,030 rows)
- **Source:** [Kaggle Bike Sharing Demand](https://www.kaggle.com/datasets/c1730b3c7d4311e6a6202040f0db4ec7b826f619)
- **Loading:** `reproduce/tables/_fuzzy_models.py::load_bikeshare()`
- **Status:** Measured at ten seeds (2026-08-12), wired into Table 4.1 alongside Concrete and PhiUSIIL; 17.3× larger than Concrete, demonstrating fuzzy regression scaling on real urban dynamics

### Clustering / Structure Discovery

#### Shuttle (NASA/UCI Statlog) (Ch 3 §3.3.2, §7.3, capstone)
- **File:** `data/shuttle.csv`
- **Size:** ~58,000 rows × 7 features, 7 classes (imbalanced: ~80% in one flight condition)
- **Role:** Large-scale clustering flagship; used for the integrated capstone pipeline (Chapter 3 reorder → Chapter 5 memberships → Chapter 6 inference → Ruspini export)
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/datasets), dataset ID 148
- **Loading:** `reproduce/tables/_fuzzy_models.py::load_shuttle()` (auto-fetch via `ucimlrepo` if absent)
- **Status:** Publicly available; can be auto-fetched; cached locally to avoid repeated network calls

#### Synthetic Clustering Batteries (Ch 3 Table 3.5, Ch 5 Table 5.1–5.3)
- **File:** Generated inline by `reproduce/tables/table_5_x_ch5_selection.py` (two_moons, circles, aniso, bridged, etc.)
- **Size:** 120–1,500 points per construction
- **Role:** Small-scale validation; ground truth for topological correctness checks
- **Status:** Already available; generated on-demand

#### Psychiatric-Evaluation Set (Ch 3 §3.3.2, §3.4, Table 3.3)
- **Size:** 135,000 rows × 165 features
- **Role:** Large-scale memory-footprint demonstration (not independent reproducible by third parties)
- **Status:** **PRIVATE — not redistributable.** Feature names anonymized before author access. Measurement demonstrates scale only; no individual-feature conclusion is drawn. Chapter 7 names a fallback: re-take this measurement on a public dataset of comparable size.

### Topological Membership Generation (Chapter 5)

#### Synthetic Membership Batteries (Table 5.1–5.3, Fig 5.2)
- **Files:** Generated inline by `reproduce/tables/table_5_x_ch5_selection.py`
- **Size:** 120–5,000 points, hand-constructed (two_gaussians, bridged_gaussians, concentric_rings, varying_density, nested_gaussians, three_level_hierarchy, etc.)
- **Role:** Validation of band-discovery and membership-generation quality; all ground truth available by construction
- **Status:** Already available; all stored in `battery_hierarchical.SCALABLE` generator

#### Scalable Membership Batteries (Goal G1, §7.2)
- **Status:** **Measured at ten seeds (2026-08-12)** across `single_scale`/`many_scale`/`log_separated` at $n=100$–$5{,}000$ (see Chapter 7 §7.2 Goal G1, and `reproduce/tables/table_5_4_ch5_g1_scaling.py`). `many_scale` confirmed solid (ARI 1.00 every seed/n); `single_scale` less stable than the earlier single-seed run implied (granularity agrees only 5–7/10 seeds); `log_separated` shows a gradual, not sharp, size transition. The one-pass generator itself (Goal G1's phase five) remains unbuilt, so this measures the existing two-stage selector, not the construction G1 is ultimately about.
- **Role:** Evidence for the one-pass membership claim; needed for integration capstone

### Non-Coordinate / Relational Data (Goal G2, Chapter 7 §7.2, A.5)

Goal G2 requires demonstration on genuinely non-metric domains: time series under dynamic time warping, sequences under edit distance, graphs under a kernel dissimilarity. Three of the UCR/UEA time-series datasets below are now **measured** (2026-08-12); the rest remain verified loadable but unwired.

#### UCR/UEA Time Series (via `aeon`)
- **Files:** Auto-fetched via `aeon.datasets.load_classification()`
- **Source:** [UCR/UEA Time Series Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
- **Measured (`reproduce/tables/table_3_7_g2_dtw_nonmetric.py` + `table_3_7_g2_downstream.py`, 2026-08-12):**
  - **ECG5000:** 5,000 series × 140 timesteps — exactness 1.000, triangle-inequality violations 20.9%, downstream set-cover beats NERFCM-given-k by 0.122 ARI (fails the ±0.05 parity threshold in the favorable direction)
  - **FordA:** 4,921 × 500 — exactness 1.000, violations 0.4% (below the synthetic proxy), downstream: every method scores ≈0 ARI (degenerate)
  - **Crop:** 24,000 × 46, 24 classes (the large-scale target, ≈4.6 GB as float64 dissimilarity matrix) — exactness 1.000, violations 23.6%, matrix build 1,597 s + reorder **4.7 s**, downstream both methods weak (NERFCM 0.029, set-cover 0.064 ARI)
- **Verified loadable, not yet attempted:**
  - **ElectricDevices:** 16,637 × 96
  - **StarLightCurves:** 9,236 × 1,024
- **Distance metric:** Dynamic Time Warping (DTW)
- **Status:** Three of five wired and measured; two verified loadable, unattempted
- **Role:** Core to Goal G2 — demonstrates coordinate-free regime on real non-metric data; exactness (decision-rule item 1) now confirmed on all three attempted, downstream usefulness (item 3) partially evidenced, not yet closed

#### Graph Kernels (TUDataset, via `aeon`)
- **Source:** [TUDataset](https://www.tu-dortmund.de/en/university/news/2023/the-turing-university-graph-kernel-benchmarks-dataset/)
- **Candidate datasets:** MUTAG, PROTEINS, ENZYMES, NCI1
- **Status:** Verification in progress (A.5 notes "to confirm")
- **Role:** Goal G2 — demonstrates structure discovery on graph-structured data

#### Duin–Pękalska Dissimilarity Collection
- **Source:** [Duin & Pękalska dissimilarity data repository](https://www.prtools.org/gallery/duin_pkalska/)
- **Role:** Goal G2 — already distributed as distance matrices, matching the coordinate-free claim most literally
- **Status:** Verification in progress

---

## Loading Strategy

### Automatic Fetching

The following datasets are fetched automatically on first use if not present locally:

- **Concrete:** via `ucimlrepo` (UCI id 165) or local `.xls`
- **Shuttle:** via `ucimlrepo` (UCI id 148) or auto-built cache
- **Glass, Diabetes, Wine, Breast Cancer, Digits:** via `sklearn.datasets` or `ucimlrepo`
- **UCR/UEA time series:** via `aeon.datasets.load_classification()`

### Manual Download Required

For datasets that cannot be auto-fetched, the file structure is:

```
data/
├── RT_IOT2022.csv                      # 123k × 83, 12 classes
├── PhiUSIIL_Phishing_URL_Dataset.csv   # 235k × 54, binary
├── shuttle.csv                         # 58k × 7, 7 classes
└── beth/
    ├── beth_*.csv                      # Host telemetry (one or more files)
    └── (structure to be confirmed)
```

### Configuration

Override the default `data/` directory by setting the `GRAD_SCHOOL_DATA` environment variable:

```bash
export GRAD_SCHOOL_DATA=/path/to/datasets
uv run --project tribble-fis python reproduce/tables/table_4_1_mog_baselines.py
```

---

## Regression Dataset Selection

The proposal currently has only one regression benchmark (**Concrete**, 1,030 rows). A large-scale regression partner is needed to support the claims in Chapter 4 and 6 (see Chapter 7 §7.2, Appendix A.7.1).

### Selection Criteria
- **Size:** 5,000–20,000 rows (to balance Concrete's 1,030)
- **Features:** 8–20 input features (similar complexity to Concrete's 8)
- **Availability:** Public, no license restrictions, downloadable as CSV
- **Relevance:** Physics/engineering or natural phenomena (consistent with Concrete's domain)
- **Test split:** Should have a well-defined train/test or cross-validation protocol

### Candidates

| Dataset | Source | Rows | Features | Notes |
|---|---|---|---|---|
| **Combined Cycle Power Plant Energy Efficiency** | [UCI 165](https://archive.ics.uci.edu/datasets) | 9,568 | 4 | Sensors from a power plant; similar physical domain to Concrete |
| **Airfoil Self-Noise** | [UCI 291](https://archive.ics.uci.edu/datasets) | 1,503 | 5 | Aeroacoustic; smaller than ideal but clean domain |
| **Real Estate Valuation** | [UCI 477](https://archive.ics.uci.edu/datasets) | 414 | 7 | Too small; similar domain (structural property prediction) |
| **Energy Efficiency (Building Physics)** | [UCI 242](https://archive.ics.uci.edu/datasets) | 768 | 8 | Similar to Concrete; smaller than ideal |
| **Bike Sharing Demand** | [Kaggle](https://www.kaggle.com/datasets/c1730b3c7d4311e6a6202040f0db4ec7b826f619/bike-sharing-demand) | 17,379 | 16+ | Urban dynamics; slightly more features than target; well-established benchmark |
| **House Prices (Advanced)** | [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) | 1,460 | 81 | Too many features; test split is withheld |
| **Wine Quality (Red)** | [UCI 186](https://archive.ics.uci.edu/datasets) | 1,599 | 11 | Physicochemical; similar domain; already loaded in `reproduce/tables/_fuzzy_models.py` |

### Recommendation

**Combined Cycle Power Plant Energy Efficiency** (UCI 165, 9,568 rows × 4 features) is the strongest candidate:
- **Balanced size:** 9.3× larger than Concrete, demonstrating scale without being overwhelming
- **Clean domain:** Direct analogue to Concrete's physical measurement domain
- **Minimal features:** 4 inputs maintain interpretability parallel to Concrete's 8
- **Public and simple:** No withheld test set or licensing complications
- **Existing precedent:** This dataset class is standard for fuzzy regression benchmarks

**Wine Quality (Red)** (UCI 186, 1,599 × 11) is the second choice:
- Already has a loader in `_fuzzy_models.py`
- Slightly larger than Concrete; physicochemical domain is similar
- 11 features are on the high side for interpretability claims but acceptable

**Bike Sharing** is the highest-risk choice (most features, most complex feature interactions), but if you want to demonstrate scaling past Concrete's ~1K rows, it's a strong practical benchmark.

---

## Status Summary

| Category | Small / Fast | Large / Scale | Measured? | Status |
|---|---|---|---|---|
| **Regression** | Concrete (1,030) | Bike Sharing (17,379); California Housing (20,433); Superconductivity (21,263) | **Yes** (2026-08-12, 10 seeds) | Bike Sharing in Table 4.1; California Housing/Superconductivity in new Appendix A.7.1 generator — RF wins both, flat MoG/HME unstable on Superconductivity |
| **Classification** | Glass (214) | PhiUSIIL (235k) | Yes | Fixed; was broken, now working |
| **Classification (open-set)** | Glass (214) | RT-IOT2022 (123k) | **Yes** (2026-08-12, both roles) | Open-set (5 seeds, Table 4.7b): complement rule loses to Isolation Forest. Classification/timing (10 seeds, Table 4.4): MoG 92.7% / 37.4s vs. RF 99.9% |
| **Anomaly** | Glass (214) | BETH (3.8M) | No | BETH in place; no measurements yet; still blocked on the one-class-protocol decision (§7.3) |
| **Clustering** | Synthetic (120–1.5k) | Shuttle (58k) | Demonstrated | Shuttle in place; demo not repeatable |
| **Membership Gen** | Synthetic (120–160) | Scalable batteries (100–5,000) | **Yes** (2026-08-12, 10 seeds) | `many_scale` solid, `single_scale` less stable than believed, `log_separated` gradual — see Ch5 §5.4, Goal G1 |
| **Non-coordinate (G2)** | *[unwired]* | ECG5000, FordA, Crop (up to 24k, DTW) | **Partially** (2026-08-12, exactness only) | Exactness = 1.000 on 3 of 5 named datasets; downstream-usefulness threshold not yet met (2 of 3 tested, both low-information) |

---

*Appendix — benchmark dataset locations and loading strategy. Tracking in `CHECKLIST.md` under **D1** (data infrastructure).*
