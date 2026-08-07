# chaos/ — FIS reproduction of arXiv:2504.13453

Reproduces the "time-step based approach" of Ramachandruni et al.,
*Using Machine Learning and Neural Networks to Analyze and Predict Chaos in
Multi-Pendulum and Chaotic Systems* (arXiv:2504.13453), replacing the paper's ten
neural / ML models with a collection of Takagi–Sugeno fuzzy inference systems
(one FIS per output angle, from `tribblefis`), and extending it from the paper's
two chain lengths to n = 2, 3 and 5.

Read in this order:

| File                    | What it is                                                                                                                                                                                                                                                                                                        |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `results/comparison.md` | Generated cell-by-cell ranking tables: FIS vs the paper's models vs two no-learning baselines.                                                                                                                                                                                                                    |
| `paper.md`              | Write-up as a NAFIPS-length paper (~5,400 words, 9 sections, 12 equations), with `references.bib` alongside. Lives here rather than under `papers/` on purpose: it is a draft against live code, and its figure links point at `figures/` in this directory. Move both to `papers/` when drafting for submission. |

**Short version:** the FIS beats all eight of the paper's time-step models in six of
the seven cells the paper reports and places second in the seventh. But on the
friction holdout cells, *averaging the two nearest training trajectories* — no
parameters, no fitting — beats every model in the paper by 6–18×, because the
held-out initial condition sits exactly halfway between two trained ones and
damping keeps them within 4% of the target's range for the whole window. The
benchmark's friction variant at n=2 and n=3 is an interpolation problem, not a
chaos problem.

Extending to n=5 (`REPRODUCTION_REPORT.md` §8) shows that dominance is
chain-length-dependent: at five links the same baseline is 20× worse, because the
bracketing pair separates by 24% of the target's range instead of 3%. It is still
ahead of the FIS, but by 1.3× rather than 4.8×. Across all three chain lengths the
FIS loses to the baseline on every friction holdout and beats both baselines on
every frictionless one — it wins where the answer is a blend over many training
initial conditions and loses where it is one nearby trajectory copied accurately.

## Layout

```
pendulum_data.py       RK4 generator (paper-faithful) + a provenance check that
                       cross-validates the n=2 equations against this repo's
                       independent symbolic model every run
fis_timestep.py        the FIS operator, the paper's preprocessing protocol, the
                       three metric families, and the no-learning baselines
sweep.py               main hyperparameter sweep, --n/--out -> results/sweep*.csv
sweep_lowcap.py        low-capacity follow-up sweep, --n/--out
bracket_diagnostic.py  measures why the frictionless holdout is unlearnable
plots.py               figure generators (paper figure formats)
paper_results.py       the paper's reported numbers, transcribed
run_all.py             picks the winner per dataset, refits, emits all outputs
data/                  generated datasets (.npz + tidy .csv)
results/               sweep{,_lowcap,_n5,_lowcap_n5}.csv, best.csv, comparison.md,
                       bracket.csv, extrapolation.csv  (tracked -- the evidence)
figures/               PNGs
```

## Running

Everything runs from *this directory* with the repo's root virtualenv. There is
no `__init__.py`: the modules import each other flatly and `pendulum_data.py`
puts `AnalyticalDynamics/` on `sys.path` for `n_pendulum_symbolic`, matching the
convention of the sibling scripts.

```sh
cd AnalyticalDynamics/chaos
../../.venv/Scripts/python pendulum_data.py       #  33 s  -> data/  (includes the n=5 checks)
../../.venv/Scripts/python bracket_diagnostic.py  #  10 s  -> results/bracket.csv, fig_bracket
../../.venv/Scripts/python sweep.py               # ~2.5 h, 6 datasets x 19 configs
../../.venv/Scripts/python sweep_lowcap.py        #   7 min, 6 datasets x 24 configs
../../.venv/Scripts/python run_all.py             #  20 min, refits winners -> figures + tables
```

To add a chain length without re-running what is already scored, sweep it into its
own file — `run_all.py` merges every sweep CSV it finds:

```sh
../../.venv/Scripts/python sweep.py --n 5 --out sweep_n5.csv                # 72 min
../../.venv/Scripts/python sweep_lowcap.py --n 5 --out sweep_lowcap_n5.csv  #  3 min
```

Timings are measured on 32 cores, though the FIS fit is single-threaded, so they
are effectively single-core numbers. The two sweeps are independent and can run
concurrently. `sweep.py --quick` runs a 4-config subset if you only want the
pipeline exercised.

`run_all.py` merges every sweep CSV named in its `SWEEP_FILES`, picks a winner per
(dataset, metric-family), refits it, and **asserts the refit reproduces the swept
score to 1e-9** before reporting it. 253 scored configurations at present.

Five `triple_friction` configurations fail by design and are recorded as error
rows: `output_partition="uniform"` at 300 buckets leaves empty buckets on that
target and the library's consequent indexing raises `IndexError`. Use `quantile`
there — see `REPRODUCTION_REPORT.md` §4.3.

## Datasets

Six, each 31 initial conditions × 2000 samples over 10 s at h = 0.005 s, plus a
held-out `[120°, …, 2.05°]` trajectory that is never trained on and runs to
**20 s** — twice the training window, so its second half tests extrapolation in
time as well as to an unseen angle:

| | frictionless | friction (damping 0.15) |
|---|---|---|
| double (n=2) | `double_frictionless` | `double_friction` |
| triple (n=3) | `triple_frictionless` | `triple_friction` |
| quintuple (n=5) | `quintuple_frictionless` | `quintuple_friction` |

θ₁(0) is pinned at 120°, the **last** link's angle is swept over 0.0–3.0° at 0.1°,
and every link in between starts hanging straight down — `[120, x]`,
`[120, 0, x]`, `[120, 0, 0, 0, x]`. The first two are the paper's own cases
(§2.2 and the Fig. 18B caption); n=5 continues the pattern and is this
repository's extension, since the paper stops at the triple pendulum. Chain length
comes from `pendulum_data.N_LINKS`; adding n=4 would be a one-tuple edit.

Equations of motion for n ≥ 3 come from `../n_pendulum_symbolic.py`, which is
n-generic and already validated at n=5. Nothing was derived by hand for this
extension.

## Two things to know before reading any number

1. **RMSE is dimensionless.** The paper min-max scales each trajectory to [0, 1]
   independently before pooling, so RMSE is a fraction of that trajectory's own
   angular range. 0.017 on the frictionless double-pendulum holdout is ≈ 16°, not
   0.017°. Every FIS metric here is reported in degrees as well. Both scalers are
   clipped differently on purpose: the **target** scaler is fitted on the 10 s
   training window, so in-window numbers stay comparable to the 10 s protocol,
   while the **input** scaler is unclipped and returns 2.0 for t = 20 s rather
   than saturating at 1.0 — which is why extrapolation diverges instead of
   plateauing.
2. **Four metric families, not one.** `pooled` is the paper's own random 80/20
   split over pooled rows — it interleaves samples 5 ms apart between train and
   test and measures interpolation, not generalisation. `trained_ic` scores a
   trajectory that was in training. `holdout_ic` scores the never-trained
   `2.05°` IC over 0–10 s and is the number that tests the paper's actual claim.
   `extrap_ic` scores the same trajectory over 10–20 s, beyond the training
   window entirely.
3. **Nothing survives the window edge.** On all three friction datasets the FIS is
   accurate right up to t = 10 s and then fails within one or two timesteps
   (R² +0.998 → −6.3e7 on the double pendulum). The time-step formulation makes
   `t` an input feature rather than integrating a state, which is why it is
   accurate inside the window and has no horizon at all outside it.
   `REPRODUCTION_REPORT.md` §9, `results/extrapolation.csv`.

Figures are per-metric: `fig_rmse_*` and `fig_r2_*`, each a bar chart sorted
best-first, rather than the paper's dual-axis combined chart.

No animations are produced, by request.
