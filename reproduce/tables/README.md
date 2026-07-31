# Experiment-table generators

Each script regenerates one table from the proposal, writing both a Markdown and
a CSV file into [`../outputs/`](../outputs). Every number is a **mean ± std over
a fixed set of seeds** (`common.SEEDS`, default `0,1,2,3,4`), so the tables are
reproducible and carry error bars. A cell that reads `N/A` means that method or
dataset genuinely wasn't available on this machine — nothing is fabricated.

## Running

The fuzzy models live in the `tribble-fis` environment and pVAT in
`tribble-cluster`, so each table runs under the right submodule via `uv`:

```bash
# from the repo root
uv run --project tribble-fis     python reproduce/tables/table_4_1_mog_baselines.py
uv run --project tribble-fis     python reproduce/tables/table_6_1_model_family.py
uv run --project tribble-cluster python reproduce/tables/table_3_1_pvat_scaling.py
```

Outputs: `reproduce/outputs/table_4_1.{md,csv}`, `table_6_1.{md,csv}`, `table_3_1.{md,csv}`.

## Knobs (environment variables)

- `REPRO_SEEDS="0,1,2,3,4,5,6"` — widen the seed set (more seeds → tighter CIs).
- `REPRO_N_GRID="256,512,1024,2048,4096"` — the N values for Table 3.1.
- `REPRO_NAIVE_CAP="1024"` — largest N at which the cubic classical VAT is timed.

## What each table contains

| Script | Proposal table | Datasets | Models | Baselines |
|---|---|---|---|---|
| `table_3_1_pvat_scaling.py` | Table 3.1 | random 2-D point sets | exact pVAT | in-script classical O(N³) VAT |
| `table_4_1_mog_baselines.py` | Table 4.1 | Concrete, PhiUSIIL | MoG FIS (time + accuracy) | sklearn RF; ANFIS/GA-FIS optional |
| `table_6_1_model_family.py` | Table 6.1 | Concrete, PhiUSIIL | flat / fuzzy tree / HME | sklearn CART & RF; M5 optional |

## Adding the optional baselines (ANFIS, GA-FIS, M5)

These columns stay `N/A` until you drop in an adapter:

- **ANFIS / GA-FIS** (Table 4.1): create `reproduce/tables/_baseline_anfis.py` and
  `_baseline_gafis.py`, each exposing
  `fit_predict(X_tr, y_tr, X_te, *, kind, seed) -> predictions`
  (`kind` is `"reg"` or `"clf"`). The table auto-detects and fills the column.
- **M5** (Table 6.1): `pip install m5py` in the `tribble-fis` env; the script
  picks it up automatically for the regression rows.

## Notes / caveats

- **Datasets.** Concrete is built automatically from the `.xls` in
  `AEEM6097/project-data/` if the repo CSV is missing. PhiUSIIL is loaded via the
  repo's own `demo_phishing.load_data`, falling back to `ucimlrepo` (id 967).
- **API resolution.** The fuzzy-tree/HME class names and the pVAT entry point are
  resolved by name with fallbacks; if the repo API has drifted, the affected cell
  reports `N/A` and prints which symbol it looked for — adjust
  `_fuzzy_models.py` / `table_3_1`'s `_resolve_pvat()` accordingly.
- These are the *harness* numbers. The **citable** versions must be re-run under
  the G4 repeatability protocol (stable clocks, more seeds, datacenter FP64 GPU);
  see `../../research/proposal-defense/ACTION_ITEMS.md`.
