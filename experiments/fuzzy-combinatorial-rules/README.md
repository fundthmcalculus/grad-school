# Experiment: C rules, a Ruspini grid, and a combinatorial choice of antecedents

**Status:** complete · **Started:** 2026-08-02

Findings are in [`RESULTS.md`](RESULTS.md). This file is the setup: what the
model is, why the search space has the size it does, and what each selector
actually does.

## Question

Fit a fuzzy classifier the ordinary way, then impose one hard structural
constraint — **exactly as many rules as there are classes** — and move all the
remaining freedom into a discrete choice. Partition every (normalised) input
with a *k*-term Ruspini partition of triangular membership functions, fix that
grid, and let the only decision be **which membership functions appear in which
rule**.

That decision is combinatorial, and the question is what combinatorial
machinery finds a good answer: minimum spanning trees on two different graphs,
greedy hill climbing, annealing, and — where the space is small enough to allow
it — brute force, which tells us how good "good" actually was.

## The model

Everything is normalised to `[0, 1]` on the training split (test points are
clipped in, never extrapolated) and partitioned with *k* triangular MFs whose
peaks sit at `j/(k-1)`. That is the Ruspini condition: the memberships of a
variable sum to 1 at every point, which pins the geometry completely and leaves
`k` as the single continuous-side knob. See `ruspini.py`.

Rule *c*, one per class:

```
IF x_1 is (A_1c) AND ... AND x_d is (A_dc) THEN class = c
```

`A_ic` is a **subset** `S[c, i, :]` of variable *i*'s *k* membership functions,
read disjunctively — "petal length is medium **or** high **or** very high". The
whole model is one boolean tensor `S` of shape `(C, d, k)`.

Three properties follow, and they are the point of the design:

- **Subsets are not exclusive.** The same MF can appear in any number of rules;
  figure `02-rule-masks.png` shows MFs used by all three iris rules sitting
  beside MFs used by one.
- **Different variables contribute different numbers of MFs to the same rule.**
  Nothing ties `|S[c, i]|` to `|S[c, i']|`.
- **A full row is an exact don't-care.** Under a Ruspini partition the
  memberships of a variable sum to 1, so summing over a subset is automatically
  in `[0, 1]` and summing over the full set is exactly 1. Plain summation is
  therefore an exact t-conorm on this geometry, and "select everything" means
  "ignore this variable" with no residue. (Under `max`, the same full set is
  worth only `max_j mu_ij(x) ∈ [0.5, 1]`, so don't-care would leak a penalty —
  which is why `sum` is the default disjunction and `max` is kept as an
  ablation.)

Firing strength is the disjunction per variable, then a t-norm across variables:

```
a_ic(x) = OR_{j in S[c,i]} mu_ij(x)        tau_c(x) = AND_i a_ic(x)
```

with `min` (default) or `product` for the `AND`. Prediction is `argmax_c tau_c`.
Rules are also optionally weighted by inverse mean firing (`inverse-mass`),
because a broad rule fires high everywhere and would otherwise win the argmax on
generality alone; both weighted and unweighted accuracy are reported.

## The search space

Per class, each of the *d* variables takes one of `2^k - 1` nonempty subsets, so
the per-class space is `(2^k - 1)^d` and the model's joint space is
`(2^k - 1)^(d·C)`:

| dataset | d | C | k=3 | k=5 | k=7 |
|---|---|---|---|---|---|
| iris | 4 | 3 | 1.4e10 | 7.9e17 | 1.8e25 |
| wine | 13 | 3 | 9.1e32 | 1.5e58 | 1.1e82 |
| glass | 9 | 6 | 4.3e45 | 3.4e80 | 4.0e113 |

Each class is optimised independently against its own one-vs-rest margin

```
J_c(S) = mean_{x in c} tau_c(x)  -  lambda * mean_{x not in c} tau_c(x)
```

so the rules interact only at prediction time, through the argmax. That
decoupling is what makes brute force possible at all: it turns one
`(2^k - 1)^(d·C)` problem into *C* separate `(2^k - 1)^d` problems, and
`31^4 = 923 521` is enumerable where `7.9e17` is not.

`J_c` is **not submodular**. Both of its terms are non-decreasing in `S` —
adding an MF can only raise `tau_c`, on in-class and out-of-class points alike —
so `J_c` is a difference of monotone set functions, and greedy carries no
approximation guarantee. That is exactly why the exhaustive column matters.

## Selectors

All of them tune their own hyper-parameters (`alpha`, the single-linkage cut
level) against `J_c` **on the training split only**. Nothing sees test data.

| selector | idea |
|---|---|
| `mass` | Per variable, take MFs in descending class-*c* membership mass until they cover `alpha` of it. Wang–Mendel-flavoured baseline: no interaction between variables at all. |
| `mst_mf` | **MST over the `d·k` membership functions.** Edge weight is `1 −` fuzzy Jaccard of how the two MFs co-fire on class-*c* samples. Sweep a single-linkage cut over the tree's own edge weights, and keep the component carrying the most class-discriminative mass (fuzzy precision above prior, times support). Variables absent from that component become don't-cares. |
| `mst_core` | **MST over the class's samples** in the `d·k`-dimensional membership space. Cut it, drop straggler components as outliers, and let the surviving core define the MF mass. Outlier-robust support estimation — the repository's VAT / single-linkage lineage applied to rule support rather than to cluster count. |
| `greedy` | Steepest-ascent hill climb from all-don't-care, one MF flip at a time. |
| `anneal` | Simulated annealing from the best of the above. |
| `exhaustive` | Every subset combination, when the per-class space fits the budget (2e6). |

`mst_mf` and `mst_core` are the two honest ways to get a spanning tree into this
problem, and they answer different questions: one asks *which membership
functions belong together*, the other asks *which samples should be allowed to
vote*.

## The convexity variant

`--convex` restricts every antecedent to one *contiguous* run of MFs — the
classical linguistic constraint, under which "x is low or medium" is a term and
"x is low or high (but not medium)" is not. It shrinks each variable's choice
from `2^k - 1` subsets to `k(k+1)/2` intervals (at k=7, 127 → 28), which is
enough to bring iris k=7 inside the enumeration budget. The free search does
*not* respect it on its own — `convex_frac` in the results JSON measures how far
it drifts — so this is a real constraint rather than a restatement, and
`RESULTS.md` reports what it costs.

## Layout

```
ruspini.py        partitions, scaling, partition-of-unity self-check
rules.py          the (C, d, k) rule base, firing, objective, incremental evaluator
selection.py      the selectors (named `selection`, not `selectors`, to avoid
                  shadowing the stdlib module that subprocess imports)
model.py          fit: normalise -> fuzzify -> select per class -> weight
datasets.py       iris, wine (sklearn), glass (repo-local csv, no network fallback)
run_experiment.py the sweep -> outputs/results-*.json, outputs/tables-*.md
landscape.py      exhaustive enumeration statistics -> outputs/landscape*.*
summarize.py      cross-run comparisons -> outputs/summary.md
figures.py        figures/*.png
```

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install numpy scipy scikit-learn pandas matplotlib
cd experiments/fuzzy-combinatorial-rules
python run_experiment.py --tag main          # the table of record, 141 s
python run_experiment.py --quick             # 3 seeds, iris+wine, k in {3,5}
python run_experiment.py --tag convex --convex     # interval antecedents only
python run_experiment.py --tag tnorm-product --tnorm product \
    --selectors mass,mst_mf,mst_core,greedy,anneal
python run_experiment.py --tag disj-max --disjunction max \
    --selectors mass,mst_mf,mst_core,greedy,anneal
python run_experiment.py --rules iris:5:greedy     # print the rules in words
python landscape.py --ks 3,5                       # enumeration statistics
python landscape.py --ks 3,5,7 --convex
python summarize.py > outputs/summary.md
python figures.py
```

Figures: `01` the partitions themselves, `02` which MFs each rule claims and how
often rules share one, `03` where the selectors land in the full enumeration,
`04` the co-firing MST over membership functions, `05` the drift away from
convex antecedents as k grows.

Seeds follow `reproduce/common.py`: ten of them, `REPRO_SEEDS` to override.
Every reported number is a mean ± population std over those seeds, on stratified
70/30 splits. `run_experiment.py` asserts the partition-of-unity invariant for
every `k` before it runs anything.
