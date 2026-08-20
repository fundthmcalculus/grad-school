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
damping keeps them within 6% of the target's range for the whole window. The
benchmark's friction variant at n=2 and n=3 is an interpolation problem, not a
chaos problem.

Extending to n=5 (paper.md §5.3, `results/bracket.json`) shows that dominance is
chain-length-dependent: at five links the same baseline's RMSE is 20× worse,
while the bracketing pair's own separation grows more gently, from 0.6% (n=2)
to 6.0% (n=3) to 7.0% (n=5) of the target's range. The baseline is still
ahead of the FIS, but by 1.3× rather than 4.8×. Across all three chain lengths the
FIS loses to the baseline on every friction holdout and beats both baselines on
every frictionless one — it wins where the answer is a blend over many training
initial conditions and loses where it is one nearby trajectory copied accurately.

## Layout

```
pipeline_cache.py       content-hash freshness helpers run_all.py's stages use
pendulum_data.py        RK4 generator (paper-faithful) + a provenance check that
                        cross-validates the n=2 equations against this repo's
                        independent symbolic model every run
fis_timestep.py         the FIS operator, the paper's preprocessing protocol, the
                        three metric families, and the no-learning baselines
sweep.py                main + low-capacity grids (--lowcap), --n/--out for a
                        chain length outside the pipeline
bracket_diagnostic.py   why the frictionless holdout is unlearnable (Tables 2-3)
compare_families.py     time-as-input vs state-as-input; run_all.py only uses its
                        time-as-input half (Table 4) via gen_n2_rollout_comparison.py
wrap_sweep.py           wrap / hysteresis / sin-cos representation study (Tables 6-8)
gen_n2_rollout_comparison.py / gen_rollout_error_vs_n.py
                        Table 4 / Table 7's figures, fitted or read from real
                        pipeline output (not synthesized or hardcoded)
plots.py                figure generators (paper figure formats)
paper_results.py        the paper's reported numbers, transcribed
run_all.py              the one entry point -- see Running, below
stable_extrapolation.py state-as-input surrogate; a comparison point for
                        compare_families.py, not part of the reproduction
                        run_all.py drives (paper.md doesn't cite its numbers)
data/                   generated datasets (.npz + tidy .csv)
results/                one JSON per stage (data, sweep, sweep_lowcap, select,
                        bracket, capacity, representation) + comparison.md
figures/                PNGs
```

## Running

Everything runs from *this directory* with the repo's root virtualenv. There is
no `__init__.py`: the modules import each other flatly and `pendulum_data.py`
puts `AnalyticalDynamics/` on `sys.path` for `n_pendulum_symbolic`, matching the
convention of the sibling scripts.

```sh
cd AnalyticalDynamics/chaos
../../.venv/Scripts/python run_all.py           # everything, from scratch, cached
```

That one command runs seven stages in order -- data generation, the main sweep,
the low-capacity sweep, refit-and-select (Table 1), the bracket diagnostic
(Tables 2-3), the capacity/extrapolation comparison (Table 4), and the
wrap/representation study (Tables 6-8) -- and leaves one JSON file per stage in
`results/`, plus every figure paper.md links to, in `figures/`. Each stage is
skipped if its cached JSON already matches a hash of that stage's inputs (fits
are deterministic, `random_state` fixed throughout), so a second run does
nothing:

```sh
../../.venv/Scripts/python run_all.py                      # skip anything unchanged
../../.venv/Scripts/python run_all.py --fresh               # force every stage
../../.venv/Scripts/python run_all.py --fresh sweep lowcap   # force just these two
../../.venv/Scripts/python run_all.py --stage bracket        # run one stage alone
```

The main sweep, low-capacity sweep, and representation stages are the expensive
ones (~2.5 h, ~10 min, and ~90 min respectively -- the representation stage's
cost comes from dozens of full-2nd-order 120-rule fits across the wrap/
hysteresis grid, not from any single slow step). `data`, `select`, and
`bracket`/`capacity` are seconds to a few minutes. `python sweep.py --quick` runs a
4-config smoke subset of the main grid if you only want the pipeline exercised
without paying for the full sweep; `python sweep.py --n 5 --out sweep_n5.csv`
sweeps an additional chain length outside the pipeline (which always requests
the fixed n in {2, 3, 5} grid paper.md reports).

The `select` stage picks a winner per (dataset, metric family) from the merged
sweep + low-capacity rows, refits it, and **asserts the refit reproduces the
swept score to 1e-9** before reporting it -- 253 scored configurations at
present. Five `triple_friction` configurations fail by design and are recorded
as error rows: `output_partition="uniform"` at 300 buckets leaves empty buckets
on that target and the library's consequent indexing raises `IndexError`. Use
`quantile` there.

Every module above still has its own `if __name__ == "__main__":` for standalone
debugging (`python bracket_diagnostic.py`, `python wrap_sweep.py`, ...) -- it
computes the same thing run_all.py's matching stage does and writes a plain
(non-cached) JSON dump of its own results, so it's never gated behind the
pipeline's freshness check.

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
   paper.md §6, `results/select.json` (each dataset's `extrapolation` field).

Figures are per-metric: `fig_rmse_*` and `fig_r2_*`, each a bar chart sorted
best-first, rather than the paper's dual-axis combined chart.

No animations are produced, by request.
