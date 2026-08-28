# `nn-cmapss` — FIS vs. network, and FIS quality, on N-CMAPSS DS02

Turbofan remaining-useful-life on DS02: a like-for-like benchmark of a ReLU
network against the TRIBBLE fuzzy inference system, and a follow-on study of how
to make the FIS's own RUL accurate, smooth, and monotone.

Reports and figures land in `outputs/nn-cmapss/` (gitignored — force-add what's
worth keeping):

- **[`BENCHMARK.md`](../../outputs/nn-cmapss/BENCHMARK.md)** — network vs. FIS.
- **[`REVIEW.md`](../../outputs/nn-cmapss/REVIEW.md)** — review of PR #111 and the
  existing `FuzzySystemsExperiments/cmapss_rul*.py` work.
- **[`MONOTONE.md`](../../outputs/nn-cmapss/MONOTONE.md)** — near-monotone RUL:
  post-hoc smoothing and a monotone-by-construction model.
- **[`FIS_QUALITY.md`](../../outputs/nn-cmapss/FIS_QUALITY.md)** — improving the
  FIS's RUL inside TribbleRegressor, and the recommended pipeline.

## How the code is organized

Five **library** modules (imported, no side effects) and a handful of **driver**
scripts (each runs one study and writes to `outputs/`). Start with the library,
then read whichever study interests you.

### Library

| module | what it owns |
|---|---|
| `cmapss_data.py` | the data: load DS02, condition-correct, aggregate (`whole_cycle` or `raw_memory`), scale, split into fit/val/train/test. Defines the named bundles (`honest`, `best`, `memory18`). |
| `models.py` | the FIS (`TribbleRegressor`) and the network arms (via `experiments/fis-to-neural-net/fis2nn.py`), plus the `FIS_CONFIGS`. |
| `metrics.py` | **all scoring, one home** — pooled/endpoint RMSE·MAE·NASA (`evaluate`) and per-engine trajectory accuracy + monotonicity (`per_cycle`, `score_engine`, `aggregate`). |
| `transforms.py` | the monotone post-processing operators (`out_cummin`, `out_mean_cummin`, `out_iso_offline`, …) and `apply_output`. |
| `report.py` | the shared plotting palette, `_style`, and the benchmark figure/table builders used by the report generators. |

### Drivers

**The benchmark (network vs. FIS)** — see `BENCHMARK.md`:

| script | study |
|---|---|
| `smoke.py` | smallest thing that can fail; run first. |
| `sweep.py` | network hyperparameter sweep (width / lr / batch). |
| `sweep_fis.py` | the FIS's Factor-D grid, selected on validation. |
| `arms.py` | the benchmark proper: every initialization, val-selected, scored once on test. |
| `fidelity.py` | why the warm start doesn't fire — conversion fidelity vs. FIS dimension. |
| `external_baselines.py` | sklearn MLP/GBM/RF — a check that the NumPy trainer isn't the ceiling. |
| `trajectories.py` | FIS-vs-network RUL overlay per test engine. |
| `write_benchmark.py`, `write_artifact.py` | generate `BENCHMARK.md` and the HTML summary from the run artifacts. |

**Making RUL monotone** — see `MONOTONE.md`:

| script | study |
|---|---|
| `monotone.py` | post-hoc smoothing: quantify the noise, compare causal clamps against the offline oracle. |
| `monotone_model.py` | a RUL model that is monotone *by construction* (accumulated non-negative damage). |

**FIS quality** (accuracy + smoothness inside the FIS) — see `FIS_QUALITY.md`.
One driver, three subcommands:

```bash
python fis_quality.py levers        # which FIS-native lever moves both axes (memory features)
python fis_quality.py memory-sweep  # tune the memory-window size
python fis_quality.py monotone      # the recommended memory18 FIS made hard-monotone
```

## Protocol

DS02's six development engines split into **fit** (2, 5, 10, 16) and **val**
(18, 20); the three official held-out engines (11, 14, 15) are **test**.
Everything tunable — network width/lr/stopping-epoch/read-out ridge, and the
FIS's Factor-D grid — is chosen on val; test is scored once. This is the one
intentional difference from the DOE in `FuzzySystemsExperiments/`, which selects
on `rmse_true`; both conventions are reported side by side in `BENCHMARK.md`.

Two RMSE conventions appear throughout and are not interchangeable: **pooled**
per-sample (the benchmark headline) and **per-engine mean** (each trajectory
weighted equally — the right one for smoothness). `metrics.py` computes both.

## Running

Needs `data/nasa-cmapps2/N-CMAPSS_DS02-006.h5` (2.4 GB, Kaggle release, not tracked).
Building the cached bundles is ~10 s each; the sweeps dominate runtime (~an hour
total on 8 CPU cores), everything else is seconds.

```bash
cd experiments/nn-cmapss
python cmapss_data.py honest && python cmapss_data.py best   # build + cache
python smoke.py honest

# the benchmark
python sweep.py --bundle honest --epochs 400 --seeds 5 --out sweep_honest_small.json \
    --grid '{"n_hidden":[1,2,3,4,6,8,12],"lr":[0.01,0.03,0.1,0.3],"batch_size":[32,128]}'
python sweep.py --bundle best --epochs 120 --seeds 3 \
    --grid '{"n_hidden":[4,8,16,32,64,128],"lr":[0.003,0.01,0.03,0.1],"batch_size":[128,512]}'
python sweep_fis.py && python fidelity.py && python external_baselines.py
python arms.py --bundle honest --epochs 400 --n-hidden 8  --seeds 5
python arms.py --bundle honest --epochs 400 --n-hidden 0  --seeds 5 --out arms_honest_convwidth.json
python arms.py --bundle best   --epochs 120 --n-hidden 32 --seeds 3
python arms.py --bundle best --fis-config honest --epochs 120 --n-hidden 32 --seeds 3 \
    --out arms_best_1storder.json
python trajectories.py && python write_benchmark.py

# making RUL monotone
python monotone.py && python monotone_model.py

# FIS quality (the recommendation lives here)
python fis_quality.py levers && python fis_quality.py memory-sweep && python fis_quality.py monotone
```

## The headlines

- **Network vs. FIS:** the network beats the FIS ~25% on the cheap `honest`
  pipeline and ties it on `best`, both seconds-scale. The FIS→network warm start
  loses to random init because the axis-aligned conversion carries only the
  FIS's additive part and DS02's FIS is strongly interacting (`fidelity.py`
  measures this against the best achievable additive fit).
- **FIS quality:** inside TribbleRegressor, **memory features** are the one lever
  that improves accuracy *and* smoothness at once (per-engine RMSE 10.05 → 6.19,
  up-noise 142 → 52, on the same 18 real sensors). Trend augmentation and
  hyperparameters don't help.
- **Monotone RUL:** the recommended `memory18` FIS plus a `cummin` clamp is
  accurate, smooth, and hard-monotone (6.15 RMSE, 0% up-cycles) — a one-line
  post-processor, no retraining.
