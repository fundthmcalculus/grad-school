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
| **how hard to search a city** | one (breadth, deep breadth, depth, Or-opt) everywhere | `EFFORT`: a depth-1 look-ahead probe, candidate rank, edge excess, edge asymmetry |
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

Reference values are the published TSPLIB optima. **LKH** (via `elkai`) is measured as a
separate question by `lkh_compare.py`, curve against curve rather than point against point,
because a solver whose premise is choosing its own operating point cannot be judged at
somebody else's. The answer there is unambiguous and negative — see below and FINDINGS §6.2.

See **[FINDINGS.md](FINDINGS.md)** for the results, including the components that did not work.

## What this claims

One claim, and it is narrow:

> A **small, readable, scale-free** rule base can allocate Lin-Kernighan search effort per
> decision. It **transfers across instance families without refitting**. Against a properly
> configured LKH it loses on the instances LKH finds easy, and wins a bounded window on the
> instances LKH cannot solve.

* **It transfers.** Fitting on one structural family and testing on another costs −0.0006 in q
  against a ±0.0006 noise floor; on two of four families a family's own fit scored *worse* on it
  than a foreign one. The scale-free ratios are doing the work, not the 87 fitted parameters —
  the **hand-written** rules are best or near-best on three of four synthetic families
  (FINDINGS §4.2).
* **It wins where uniform perturbation has stopped working.** On the two TSPLIB instances
  whose uniform-kick control has *plateaued* — four times the budget returning the identical
  tour — an `EFFORT`-aimed perturbation is the best of a 2×2 factorial on both, by 5.3× at
  matched budget on fl1577. On d2103 it reaches a shorter tour than any measured LKH
  configuration below 39 s. Where the control still has headroom it wins 2 of 11 (§6.3).
  Instance size and kick density were each tried as the criterion first and neither survived —
  the retractions are in §6.3, and they are worth reading before trusting this one.
* **It loses elsewhere.** On the instances LKH finds easy, LKH reaches 10–100× better quality
  *inside our own time budget*, on six of seven (§6.2).

**Not claimed:** that this beats LKH in general, or that it is competitive with the learned-LKH
line — NeuroLKH reports 0.05–0.09% where these arms sit at 0.3–1.3%, under a stricter protocol.
See [references/BENCHMARKS.md](references/BENCHMARKS.md) for what that literature measures and
[references/PRIOR_ART.md](references/PRIOR_ART.md) for what is and is not novel here.

### Size, quality and time at a glance

```bash
python summary.py --sort n            # every measured (instance, arm), long form
python summary.py --arms iterated lkh --instances pr2392
```

`results/summary.csv` carries all of it. The three source files answer different questions and
print different shapes; this joins them into one table of *n*, gap over the published optimum,
and wall clock.

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
| `figures.py`, `figures_tuning.py`, `figures_fis.py`, `figures_lkh.py`, `figures_aim.py` | `results/figures/*.png` |
| `summary.py` | `results/summary.csv` — one flat size / quality / time row per (instance, arm) |

**`experiments/`** — superseded, one-off, or illustrative, plus two that produce reported
findings but re-fit from scratch and so do not belong in `run_all.py`
(`transfer.py` → §4.2, `profile_kernels.py` → §8). See
[experiments/README.md](experiments/README.md).

**`results/`** — every artifact any of the above writes, including `results/legacy/` for data
that has been superseded but still backs a claim in FINDINGS.md.

**`references/`** — what the comparable literature measures
([BENCHMARKS.md](references/BENCHMARKS.md)) and what is actually novel here
([PRIOR_ART.md](references/PRIOR_ART.md)). The PDFs themselves are gitignored.

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
FINDINGS §9 produced *plausible numbers* rather than crashes.

Wall clock is what `benchmark.py` and `lkh_compare.py` report. The proxy is only what the
search spends.

### The expensive stage

`lkh_compare.py` dominates everything else, and not because of our solver: LKH's cost grows as
steeply in n, so one run on the largest instance of the ladder costs more than every other
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
