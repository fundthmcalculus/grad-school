# A fuzzy inference system as a TSP strategy engine

A Lin-Kernighan solver spends its time unevenly and its parameters uniformly. Some
cities sit on edges that are already as short as the instance allows; others carry a
long edge or a sharp kink and a deep gain chain there pays for itself many times
over. A conventional LK cannot tell them apart, so it picks one candidate breadth
and one chain depth and applies them everywhere.

This directory replaces three of those fixed rules with fuzzy inference systems, and
measures what that buys.

| | fixed rule | rule base |
|---|---|---|
| **which city next**, during construction | take the nearest unvisited | `CONSTRUCT`: nearness, stranding risk, cost of coming back, heading continuity |
| **how hard to search a city** | one (breadth, deep breadth, depth, Or-opt) everywhere | `EFFORT`: edge excess, past failures, turn sharpness, progress |
| **deepen this chain or cut it** | run to a fixed depth | `CHAIN`: gain credit carried, depth so far, gain already banked, next step's trade |

The engine is small and plain on purpose: three linguistic terms per input, product
t-norm, singleton consequents, weighted-average defuzzification. All of it is
nopython-jitted and allocation-free, because it runs in the innermost loop of a local
search — the cost model priced a rule-base evaluation at roughly what the whole city scan
it decides about costs, so the membership bank is compiled to a lookup table and every
scratch buffer is owned by the caller. A useful consequence: membership shape is then
just data, so centres, widths and functional form (gaussian or triangular) are all things
the optimiser can fit. Every antecedent is a scale-free *ratio*, which is what lets one
fitted rule base transfer from a 52-city instance to an 85 900-city one.

## What is measured, and against what

The comparison is against the same LK, swept into a time-versus-quality **frontier**
over the parameters that trade one for the other. Beating one configuration of a
tunable solver proves nothing — there is always a slow one and always a weak one. So
`benchmark.py` reports the frontier and asks whether the fuzzy arm lands outside it.

Both arms call the same `lk.py`. The baseline *is* that module with `use_chain` off
and constant parameters, so the two cannot differ in their move repertoire or in the
speed of their arithmetic — only in strategy. Every reported tour is checked to be a
permutation of the cities and re-scored from the coordinates under TSPLIB rounding,
independently of the solver's own bookkeeping.

Reference values are the published TSPLIB optima. LKH (via `elkai`) is reported as an
external yardstick, not a competitor this work claims to beat.

See **[FINDINGS.md](FINDINGS.md)** for the results, including the two components that
did not work.

## Layout

Three kinds of thing live here, in three places, and the split is the point: a reader should be
able to tell a reported result from an exploration without reading the code.

**Library** — imported, never an entry point.

| file | |
|---|---|
| `paths.py` | every input and output location, in one place |
| `tsplib.py` | instance loading, TSPLIB distance rounding, published optima, tour validation |
| `core.py` | distances, candidate lists, nearest-neighbour and greedy-edge constructions, tour surgery |
| `lk.py` | the baseline Lin-Kernighan, shared by every arm |
| `fis.py` | the inference engine, the three rule bases, and the fitted-rule-base record |
| `fis_lk.py` | fuzzy construction, and the fuzzy-controlled local search |
| `kick.py` | the double-bridge move and the iterated loop; the EFFORT base can aim its kicks |
| `synth.py` | synthetic instance families, which is what made the training pool large enough |
| `refine.py` | the compass search that polishes what the GA hands over |
| `feature_registry.py` | the master record of every antecedent tried and where it ended up |

**Pipeline** — each stage writes an artifact the next one reads. `run_all.py` runs them in order.

| stage | writes |
|---|---|
| `test_invariants.py` | nothing; it either passes or stops the run |
| `costmodel.py` | `results/costmodel.npz` — the deterministic time proxy the tuner optimises against |
| `tune_opt.py --scale small\|large` | `results/tuned_<scale>.npz`, `results/tune_<scale>.json` |
| `benchmark.py --scale small\|large` | `results/results_<scale>.json` — the frontier-relative test-set comparison |
| `lkh_reference.py` | `results/lkh.json` — LKH once per test instance, as a yardstick |
| `lkh_compare.py` | `results/lkh_compare.json` — every arm and LKH, curve against curve, across a size ladder |
| `figures.py`, `figures_tuning.py`, `figures_lkh.py` | `results/figures/*.png` |

**`experiments/`** — superseded, one-off, or illustrative. Nothing reported depends on it; see
[experiments/README.md](experiments/README.md).

**`results/`** — every artifact any of the above writes, including `results/legacy/` for data
that has been superseded but still backs a claim in FINDINGS.md.

## Running it

```bash
pip install numpy scipy matplotlib numba elkai
pip install -e ../tribble-opt          # the optimizers library used by tune_opt.py
cd fis-tsp-strategy

python run_all.py                      # every stage, in order, except the full LKH ladder
python run_all.py --list               # what the stages are and what each writes
python run_all.py --dry-run            # the same, plus what the LKH stage would cost
python run_all.py --quick              # a small GA budget: proves it runs, not the result
python run_all.py --stages figs figs-lkh   # redraw from results already on disk
```

Or one stage at a time — every script is runnable on its own and takes `--help`.

Two ordering constraints are real rather than conventional. `costmodel.py` must run before
`tune_opt.py`, and again after any change to the solver's hot path: the tuner's objective is the
fitted cost proxy rather than wall clock, so that the search is deterministic and not corrupted
by its own CPU contention, and stale coefficients mean it is optimising against code that no
longer exists. And `test_invariants.py` runs first, because four of the bugs recorded in
FINDINGS §10 produced *plausible numbers* rather than crashes.

Wall clock is what `benchmark.py` and `lkh_compare.py` report. The proxy is only what the
search spends.

### The expensive stage

`lkh_compare.py` dominates everything else, and not because of our solver: LKH's cost grows as
roughly n^3.5, so one run on the largest instance of the ladder costs more than every other
stage combined. `--dry-run` prices it per instance before it starts, `--ladder` opts into the
full size range, and results are written after each instance so an interrupted run keeps
everything it measured. `--skip-lkh` re-measures our arms against an LKH curve already on disk.

## Instances

Instances come from `../ClusteringExperiments/tsplib/`, which already carries 111 TSPLIB files
and the published-optimum index. Training, validation and test lists are disjoint —
`tune_opt.py` asserts it at import rather than trusting the lists to stay right, and
`lkh_compare.py` refuses outright to measure on an instance the rule base was fitted or
selected on. `synth.py` generates four structural families, which is what took the training
pool from 9 instances to 20; the frontier-relative objective is a ratio of two tours on one
instance, so synthetic instances need no known optimum.
