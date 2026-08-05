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

The engine is small and plain on purpose: gaussian membership functions, three
linguistic terms per input, product t-norm, singleton consequents, weighted-average
defuzzification. All of it is nopython-jitted and allocation-free, because it runs in
the innermost loop of a local search. Every antecedent is a scale-free *ratio*, which
is what lets one tuned rule base transfer from a 52-city instance to an 85 900-city
one.

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

| file | |
|---|---|
| `tsplib.py` | instance loading, TSPLIB distance rounding, published optima, tour validation |
| `core.py` | distances, candidate lists, nearest-neighbour and greedy-edge constructions, tour surgery |
| `lk.py` | the baseline Lin-Kernighan, shared by both arms |
| `fis.py` | the inference engine and the three rule bases |
| `fis_lk.py` | fuzzy construction, and the fuzzy-controlled local search |
| `costmodel.py` | a deterministic cost proxy, fitted to measured wall clock, that the tuner optimises against |
| `tune_opt.py` | fits consequents *and* membership functions with the `optimizers` GA / PSO / ACO |
| `tune.py` | the earlier hand-rolled (1+1)-ES tuner, kept for comparison |
| `benchmark.py` | the reported comparison |
| `lkh_reference.py` | LKH numbers, in a subprocess with a timeout |
| `figures.py` | the time-versus-quality plane, and where the effort goes |
| `figures_tuning.py` | which optimiser won, and what fitting did to the membership functions |

## Running it

```bash
pip install numpy scipy matplotlib numba elkai
pip install -e ../tribble-opt          # the optimizers library used by tune_opt.py
cd fis-tsp-strategy

python costmodel.py                                    # writes costmodel.npz
python tune_opt.py --generations 25 --population 24    # writes tuned_opt.npz
python benchmark.py --reps 3 --tuned tuned_opt.npz     # writes results.json
python lkh_reference.py --max-n 2500 --timeout 90      # optional external reference
python figures.py && python figures_tuning.py          # writes figures/
```

`costmodel.py` has to run before `tune_opt.py`: the tuner's objective is the fitted cost
proxy, not wall clock, so that the search is deterministic, reproducible, and not
corrupted by its own CPU contention. Wall clock is what `benchmark.py` reports.

Instances come from `../ClusteringExperiments/tsplib/`, which already carries 111
TSPLIB files and the published-optimum index. The training, validation and test
instance lists are disjoint; the split is in `tune.py` and `benchmark.py`.
