# Data Directory

This directory contains datasets used for experiments and reproducible research. Most datasets are tracked in git; some large files (>10 MB) are downloaded from public sources and ignored by git (documented below).

## Tracked Datasets

### Small datasets (tracked in git)

| File | Size | Source | Description |
|------|------|--------|-------------|
| `Concrete_Data.csv` | 60 KB | UCI ML Repository (Dataset ID 165) | 1,030 concrete mixtures with compressive strength. Features: cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, age. |
| `bikeshare-hour.csv` | 1.2 MB | UCI ML Repository (Dataset ID 275) | 17,379 hourly bike-sharing demand records. Features include hour, weather, temperature, humidity, wind speed. |
| `bodyfat.csv` | 20 KB | UCI ML Repository | 250 body measurements with percent body fat. Classic regression benchmark. |
| `glass.csv` | 11 KB | UCI ML Repository (Dataset ID 42) | 214 glass samples classified into 6 types. |
| `shuttle.csv` | 1.6 MB | UCI ML Repository (Dataset ID 148) | 58,000 space shuttle observations, 9 features, 7 classes. Multi-class classification benchmark. |
| `WEC_Sydney_100.csv` | 4.4 MB | Wave Energy Converter Buoy Farm | 2,319 samples with 300 features predicting total power from 100 buoys near Sydney, Australia. |

---

## Expected by a loader but not yet vendored

Small datasets that a shared loader in `repro_data` reads from `data/`, but which
are not currently committed. Drop the file into `data/` (its loader returns
`None` and prints a not-found line until then). Commit it once present (<10 MB).

| File | Source | Read by | Description |
|------|--------|---------|-------------|
| `darwin.csv` | UCI ML Repository (DARWIN — Alzheimer handwriting) | `repro_data.load_darwin` | Handwriting-task features, binary class (`P` Alzheimer's vs `H` healthy) in a `class` column; the ID column is dropped by `select_dtypes`. Consolidated from five identical `FuzzySystemsExperiments/darwin*` inline loaders. |
| `winequality-white.csv` | UCI ML Repository (Wine Quality, id 186) | `repro_data.load_wine_quality` | White-wine physicochemical features → continuous `quality` score (regression). Semicolon-delimited. |
| `powerconsumption.csv` | UCI / Kaggle (Tetouan City power consumption) | `repro_data.load_powerconsumption` | Environmental features → Zone-1 power (regression). All three `PowerConsumption_Zone{1,2,3}` columns are dropped from X — Zones 2/3 are alternate targets, not features. |

---

## Large Datasets (gitignored, must be downloaded)

These files are >10 MB and sourced from public repositories that may not be accessible from all environments (especially cloud runners). Each is documented with its source and download method.

### RT_IOT2022.csv
- **Size:** 53 MB
- **Shape:** 123,117 rows × 81 features + 1 target
- **Source:** Kaggle — [RT-IOT2022: Real-world IoT intrusion detection dataset](https://www.kaggle.com/datasets/christianamartins/rt-iot2022)
- **Description:** IoT network traffic with 12 attack types (DoS, DDoS, reconnaissance, MITM, etc.) plus normal traffic. Used for anomaly detection and multi-class classification benchmarks.
- **Target:** `Attack_type` (12 classes: `MQTT_Pub`, `MQTT_Sub`, `HTTP`, `COAP`, `DNS`, `Reconnaissance`, `Mirai`, `BASHLITE`, `Torii`, `DDoS_HTTP`, `DDoS_TCP`, `DOS_SYN_Hping`)
- **Preprocessing notes:** The CSV contains an unnamed index column (`Unnamed: 0`) which encodes the attack class and must be dropped before use.

### WEC_Perth_49.csv
- **Size:** 33 MB
- **Shape:** 2,318 samples × 98 features
- **Source:** Figshare — [WEC Perth Wave Energy Farm Dataset](https://figshare.com/) (UC Irvine source)
- **Description:** Wave Energy Converter (WEC) data from Perth, Australia buoy farm with 49 selected features after dimensionality reduction. Predicts total power output from ocean wave measurements.
- **Target:** `Total_Power` (continuous regression)
- **Related variants:** `WEC_Perth_100.csv` contains all 100 original features (larger, higher-dimensional variant used for feature selection studies).

### WEC_Perth_100.csv
- **Size:** 16 MB
- **Shape:** 2,318 samples × 100 features
- **Source:** Figshare — [WEC Perth Wave Energy Farm Dataset](https://figshare.com/) (UC Irvine source)
- **Description:** Full-dimensional variant of WEC Perth with all original features. Used as a challenging high-dimensional regression benchmark—the "signal-in-noise" problem where most features are noise.
- **Target:** `Total_Power` (continuous regression)
- **Notes:** Issue #97 resolved FIS viability on this dataset through aggressive feature selection, achieving R²=0.6475 with preprocessing.

### WEC_Sydney_49.csv
- **Size:** 16 MB
- **Shape:** 2,319 samples × 49 features
- **Source:** Figshare — [WEC Sydney Wave Energy Farm Dataset](https://figshare.com/) (UC Irvine source)
- **Description:** Wave Energy Converter data from Sydney, Australia buoy farm with 49 selected features. Predicts total power output. Companion to Perth site for cross-location validation.
- **Target:** `Total_Power` (continuous regression)
- **Related variants:** `WEC_Sydney_100.csv` (tracked, 4.4 MB) contains the 100-feature variant.

---

## Download Instructions

These large files are NOT tracked in git but are required to reproduce some experiments. **Note:** Kaggle and UCI ML Repository mirrors may be blocked on some networks (including Claude Cloud runners). For reproducibility, store them locally and pass `GRAD_SCHOOL_DATA=/path/to/data` to the Python environment.

### Via Kaggle API (requires API key)
```bash
# RT-IOT2022
kaggle datasets download -d christianamartins/rt-iot2022 -p data/
unzip data/rt-iot2022.zip -d data/
mv data/RT_IOT2022.csv data/
```

### Via UCI ML / Figshare (if accessible)
- WEC datasets: https://figshare.com/ (search for "Wave Energy Converter")
- Check UC Irvine ML Repository for archived versions

### Fallback: GitHub mirrors
If primary sources are blocked, check if archived versions exist in:
- https://github.com/[user]/[repo-name] (search "WEC_Perth" or "RT_IOT2022")
- Community ML dataset repos that mirror UCI and Kaggle data

---

## Environment Setup

Point the experiment loaders to your local data directory:
```bash
export GRAD_SCHOOL_DATA=/path/to/grad-school/data
python -m reproduce.tables.table_4_1  # Will use $GRAD_SCHOOL_DATA for dataset loading
```

If the environment variable is not set, loaders default to `data/` in the repository root.

---

## Adding New Datasets

When adding a new dataset:
1. If <10 MB → commit directly to git
2. If >10 MB → add to `.gitignore` and document here with:
   - Source URL and access method
   - Shape (rows × columns)
   - Description of features and target
   - Download or access instructions
   - Any preprocessing notes or caveats
