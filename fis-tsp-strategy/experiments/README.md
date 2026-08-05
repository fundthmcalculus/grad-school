# Experiments

Scripts that are **not** part of the reproducible pipeline. Nothing in `run_all.py` calls
anything in here, and no reported number depends on running any of it.

Each is here for one of three reasons, and the reason matters when reading it: a thing that was
superseded is not the same as a thing that was measured once and does not need measuring again.

| script | status | why it is here, not up a level |
|---|---|---|
| `tune_es.py` | **superseded** | The original hand-rolled (1+1)-ES tuner, before `tune_opt.py` and the `optimizers` GA. It fits consequents only — not membership functions — against a wall-clock objective rather than the deterministic cost proxy, so its runs are not reproducible and its results are not comparable with the current ones. Kept because it is the control that says the GA was worth adopting. Writes `results/legacy/tuned_es.npz`. |
| `features_probe.py` | **one-off, frozen** | The feature screen of FINDINGS §7: it instruments a fixed-parameter LK run, records candidate antecedents before each city scan and the outcome after, and scores every feature by AUC. It produced `results/feature_screen.json` over 12 278 city scans, and `feature_registry.py` checks itself against that file. Re-running it is only necessary if a new antecedent is proposed — the numbers for the existing ones will not change. |
| `figures_tours.py` | **illustrative** | Draws actual tours side by side. It supports no claim; it exists because a 1% length difference is invisible in a scalar and obvious in a picture. Expensive (it re-solves each instance at a large kick budget) for something that is not evidence. |
| `profile_kernels.py` | **diagnostic** | Where the solver's time goes, three ways that check each other (counter attribution, jitted microbenchmarks, ablation), plus `--cython` to build a C translation of the hottest kernel and race it against numba. Its findings are written up in FINDINGS §11; re-run it after any change to the hot path, or on new hardware, since both the cost model's calibration and the compiler comparison are machine-specific. |
| `cython_fis_eval.pyx` | **diagnostic** | The C translation `--cython` builds. Deliberately a transcription rather than an improvement, so the only variable between it and numba is code generation. |

## Running them

They import the top-level modules and write into the same `results/` tree as everything else,
so they work from anywhere:

```bash
python experiments/features_probe.py       # rewrites results/feature_screen.json
python experiments/figures_tours.py        # rewrites results/figures/fis_tsp_tours.png
python experiments/tune_es.py --seconds 600
python experiments/profile_kernels.py            # where the time goes
python experiments/profile_kernels.py --cython   # ...and whether Cython would help (it does not)
```

`--cython` needs Cython and a C compiler; it builds in place and reports the build error rather
than the answer if either is missing. The build products are gitignored — a compiled extension
is machine-specific and regenerated on demand.

## What is deliberately *not* here

`kick.py` keeps its own `main`, and `refine.py` keeps its `--demo`, even though both are
diagnostics rather than reported stages. They are library modules the pipeline imports; moving
the file would move the library with it, and splitting the CLI off from the code it exercises
puts a diagnostic one edit away from silently not matching what it diagnoses.

`kick.py --instance … --kicks …` is the plateau probe of FINDINGS §9: it carries the
`accept_equal` and `patience` switches that exist to tell a repertoire limit apart from an
accept-rule limit.
