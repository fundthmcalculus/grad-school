# `nn-cmapss` — a neural network against TRIBBLE on N-CMAPSS DS02

Benchmarks a ReLU network against the fuzzy inference system on turbofan
remaining-useful-life prediction, using the FIS pipeline's own preprocessing,
and applies PR #111's FIS→network warm start to it.

Reports and figures land in `outputs/nn-cmapss/` (gitignored — force-add
anything worth keeping):

- **[`BENCHMARK.md`](../../outputs/nn-cmapss/BENCHMARK.md)** — the results.
- **[`REVIEW.md`](../../outputs/nn-cmapss/REVIEW.md)** — review of PR #111 and
  the existing `FuzzySystemsExperiments/cmapss_rul*.py` work.

## What each file does

| file | role |
|---|---|
| `cmapss_data.py` | the shared feature pipeline: load DS02, condition-correct, aggregate, scale, split. Copied from `cmapss_rul_best.py` so both arms see identical columns, with two tightenings — no test-unit RUL caps, and a validation split. |
| `models.py` | the arms and the metrics. Wraps `TribbleRegressor`, and `experiments/fis-to-neural-net/fis2nn.py` unmodified for the network. |
| `smoke.py` | smallest thing that can fail. Run this first. |
| `sweep.py` | the network hyperparameter sweep (width, learning rate, batch size). |
| `sweep_fis.py` | the same Factor-D grid the DOE runs, but selected on validation — so the FIS's number obeys the same rules as the network's. |
| `arms.py` | the benchmark proper: every initialization, selected on validation, scored once on test. |
| `fidelity.py` | why the hot start does not fire — conversion fidelity against FIS dimension, with a best-possible-additive reference. |
| `external_baselines.py` | sklearn MLP / GBM / random forest, as a check that the hand-rolled NumPy trainer is not the limiting factor. |
| `trajectories.py` | the FIS-vs-network overlay: predicted RUL against ground truth, per held-out engine, both pipelines. Re-derives each curve from the recorded configuration and asserts its RMSE against the run artifact before drawing. |
| `monotone.py` | reducing the FIS's cycle-to-cycle noise toward a monotone-decreasing RUL — quantifies the noise, evaluates causal vs. oracle smoothing, writes `MONOTONE.md`. |
| `report.py` | tables and figures from the artifacts. |
| `write_benchmark.py` | generates `BENCHMARK.md`. Every number is interpolated from JSON; nothing is transcribed. |

## Protocol

DS02's six development engines split into **fit** (2, 5, 10, 16) and **val**
(18, 20); the three official held-out engines (11, 14, 15) are **test**.
Hyperparameters — network width, learning rate, stopping epoch, read-out ridge,
and the FIS's Factor-D grid — are chosen on val. Test is scored once.

This is the one intentional difference from the DOE in
`FuzzySystemsExperiments/`, which selects its grid on `rmse_test_true`. Both
conventions are reported side by side in `BENCHMARK.md` so the size of that
selection advantage is visible rather than argued about.

## Running

```bash
cd experiments/nn-cmapss
python cmapss_data.py honest && python cmapss_data.py best   # ~10 s each, cached
python smoke.py honest
python sweep.py --bundle honest --epochs 400 --seeds 5 \
    --out sweep_honest_small.json \
    --grid '{"n_hidden":[1,2,3,4,6,8,12],"lr":[0.01,0.03,0.1,0.3],"batch_size":[32,128]}'
python sweep.py --bundle best --epochs 120 --seeds 3 \
    --grid '{"n_hidden":[4,8,16,32,64,128],"lr":[0.003,0.01,0.03,0.1],"batch_size":[128,512]}'
python sweep_fis.py
python fidelity.py
python arms.py --bundle honest --epochs 400 --n-hidden 8  --seeds 5
python arms.py --bundle honest --epochs 400 --n-hidden 0  --seeds 5 \
    --out arms_honest_convwidth.json
python arms.py --bundle best   --epochs 120 --n-hidden 32 --seeds 3
python arms.py --bundle best --fis-config honest --epochs 120 --n-hidden 32 \
    --seeds 3 --out arms_best_1storder.json
python external_baselines.py
python trajectories.py                                       # the overlay figure
python write_benchmark.py
```

Needs `NASA-CMAPSS/N-CMAPSS_DS02-006.h5` (2.4 GB, Kaggle N-CMAPSS release, not
tracked). Total runtime is roughly an hour on 8 CPU cores, dominated by the two
sweeps; everything else is seconds.

## The headline

The network beats the FIS by ~25% RMSE on the cheap `honest` pipeline and ties
it on the richer `best` one, both at seconds-scale cost. The FIS-derived warm
start loses to a plain random initialization on both, because the axis-aligned
conversion can only carry the FIS's *additive* part and DS02's FIS is strongly
interacting — a claim `fidelity.py` measures against the best additive fit
achievable, rather than infers.
