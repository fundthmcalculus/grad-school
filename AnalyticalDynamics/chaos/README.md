# chaos/ — FIS reproduction of arXiv:2504.13453

Reproduces the "time-step based approach" of Ramachandruni et al.,
*Using Machine Learning and Neural Networks to Analyze and Predict Chaos in
Multi-Pendulum and Chaotic Systems* (arXiv:2504.13453), replacing the paper's ten
neural / ML models with a collection of Takagi–Sugeno fuzzy inference systems
(one FIS per output angle, from `tribblefis`).

Read in this order:

| File | What it is |
|---|---|
| `METHOD_AND_PARAMETERS.md` | The paper's method, step by step, and every training parameter — separating what the PDF says from what the authors' code does. Read first. |
| `REPRODUCTION_REPORT.md` | What worked, what didn't, and how the FIS compares to the paper's reported numbers. |
| `results/comparison.md` | Generated cell-by-cell ranking tables: FIS vs the paper's models vs two no-learning baselines. |

**Short version:** the FIS beats all eight of the paper's time-step models in six of
the eight reported cells and places second in a seventh. But on the two friction
holdout cells, *averaging the two nearest training trajectories* — no parameters,
no fitting — beats every model in the paper by 6–18×, because the held-out
initial condition sits exactly halfway between two trained ones and damping keeps
them within 4% of the target's range for the whole window. The benchmark's
friction variant is an interpolation problem, not a chaos problem.

## Layout

```
pendulum_data.py       RK4 generator (paper-faithful) + a provenance check that
                       cross-validates the n=2 equations against this repo's
                       independent symbolic model every run
fis_timestep.py        the FIS operator, the paper's preprocessing protocol, the
                       three metric families, and the no-learning baselines
sweep.py               main hyperparameter sweep            -> results/sweep.csv
sweep_lowcap.py        low-capacity follow-up sweep         -> results/sweep_lowcap.csv
bracket_diagnostic.py  measures why the frictionless holdout is unlearnable
plots.py               figure generators (paper figure formats)
paper_results.py       the paper's reported numbers, transcribed
run_all.py             picks the winner per dataset, refits, emits all outputs
data/                  generated datasets (.npz + tidy .csv)
results/               sweep CSVs, best.csv, comparison.md, bracket.csv
figures/               PNGs
```

## Running

Everything runs from *this directory* with the repo's root virtualenv. There is
no `__init__.py`: the modules import each other flatly and `pendulum_data.py`
puts `AnalyticalDynamics/` on `sys.path` for `n_pendulum_symbolic`, matching the
convention of the sibling scripts.

```sh
cd AnalyticalDynamics/chaos
../../.venv/Scripts/python pendulum_data.py       #  13 s  -> data/
../../.venv/Scripts/python bracket_diagnostic.py  #   5 s  -> results/bracket.csv, fig_bracket
../../.venv/Scripts/python sweep.py               #  67 min, 4 datasets x 19 configs
../../.venv/Scripts/python sweep_lowcap.py        #   3 min, 4 x 24 configs
../../.venv/Scripts/python run_all.py             #  10 min, refits winners -> figures + tables
```

Timings are measured on 32 cores, though the FIS fit is single-threaded, so they
are effectively single-core numbers. The two sweeps are independent and can run
concurrently. `sweep.py --quick` runs a 4-config subset if you only want the
pipeline exercised.

`run_all.py` merges both sweep CSVs, picks a winner per (dataset, metric-family),
refits it, and **asserts the refit reproduces the swept score to 1e-9** before
reporting it.

Five `triple_friction` configurations fail by design and are recorded as error
rows: `output_partition="uniform"` at 300 buckets leaves empty buckets on that
target and the library's consequent indexing raises `IndexError`. Use `quantile`
there — see `REPRODUCTION_REPORT.md` §4.3.

## Datasets

Four, each 31 initial conditions × 2000 samples over 10 s at h = 0.005 s, plus a
held-out `[120°, …, 2.05°]` trajectory that is never trained on:

| | frictionless | friction (damping 0.15) |
|---|---|---|
| double (n=2) | `double_frictionless` | `double_friction` |
| triple (n=3) | `triple_frictionless` | `triple_friction` |

Double sweeps θ₂(0) over 0.0–3.0° at 0.1°, with θ₁(0) = 120°. Triple sweeps
θ₃(0) over the same grid with θ₁(0) = 120°, θ₂(0) = 0°, per the paper's Fig. 18B
caption.

## Two things to know before reading any number

1. **RMSE is dimensionless.** The paper min-max scales each trajectory to [0, 1]
   independently before pooling, so RMSE is a fraction of that trajectory's own
   angular range. 0.017 on the frictionless double-pendulum holdout is ≈ 16°, not
   0.017°. Every FIS metric here is reported in degrees as well.
2. **Three metric families, not one.** `pooled` is the paper's own random 80/20
   split over pooled rows — it interleaves samples 5 ms apart between train and
   test and measures interpolation, not generalisation. `trained_ic` scores a
   trajectory that was in training. `holdout_ic` scores the never-trained
   `2.05°` IC and is the only number that tests the paper's actual claim.

No animations are produced, by request.
