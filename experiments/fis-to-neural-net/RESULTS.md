# Results

Run of record: `results.json`, ten seeds, 150 epochs, commit `d830022`,
`tribble-fis` `5b92ec8`. Tables quoted here are generated, not transcribed —
`results_summary.md`, `time_to_quality.md`, `triangularization.md`, `gating.md`.

## Short version

**The conversion works, and it is exact where the theorem says it should be.**
In one dimension the seed is backed out of the FIS analytically, sees no labels,
and reproduces the FIS to 3% of the FIS output's own standard deviation. On that
rung the warm-started network is the *only* arm that reaches any quality target
at all inside 150 epochs — the randomly initialized networks never arrive.

**Above one dimension the hot start does not pay for itself on time.** Not
because the conversion is wrong, but for two reasons that the ablations separate
cleanly:

1. **The speed comes from the architecture, not from the FIS's knots.** An
   identically-shaped network with knots at per-feature *quantiles* matches or
   beats the FIS-derived one on every rung above 1-D — 22× faster to `1.25x
   best` on Concrete, 149× to FIS parity on bikeshare. H5 is falsified, and
   falsified consistently.
2. **On these datasets the FIS fit costs about as much as the whole training
   run.** TRIBBLE fits Concrete in 0.19 s; a from-scratch network reaches the
   FIS's own accuracy in 0.02 s and finishes 150 epochs in 0.56 s. There is no
   room for a warm start to amortize. Bikeshare — 22× longer to train — is the
   only rung where the hot arms come out ahead at some targets (1.3–1.5×), which
   is where the regime boundary is.

**What TRIBBLE does contribute is feature selection, not knot placement.** On
WEC, the same random initialization trained on TRIBBLE's 12 columns reaches
`1.5x best` **9.2× faster** than the same network trained on all 301. That is
the FIS earning its keep — through a different mechanism than the one this
experiment set out to test.

**And one thing to fix before any of this goes further:** the triangularized FIS
is not a usable model above ~6 features, and bikeshare's shared loader leaks its
target. Both are below.

## Hypothesis scoring

| | hypothesis | verdict |
|---|---|---|
| H1 | 1-D equivalence is constructive here | **confirmed** |
| H2 | triangularization is close to free | **falsified, decisively** |
| H3 | backed-out seed reproduces the FIS | **confirmed in 1-D, degrades with dimension** |
| H4 | hot start is cheaper in wall clock | **confirmed in 1-D, falsified above it** |
| H5 | the FIS's *placement* is what helps | **falsified everywhere** |
| H6 | warm start survives training | **partly** — comparable on bikeshare, worse on Concrete and WEC |
| H7 | advantage grows with dimension | **falsified — it inverts** |
| H8 | gating no longer reaches the conversion | **confirmed in 1-D, falsified above it** |

Six of eight came back wrong or partly wrong. Several are more useful that way.

---

## H1 — the equivalence, executable (confirmed)

`test_fis2nn.py`, 13 tests, all passing. The load-bearing ones:

* A triangular membership function equals its three-ReLU expansion to `< 1e-12`
  over a 24,001-point grid — interior triangles, asymmetric ones, and both
  shoulder forms. Trapezoids likewise, in four ReLUs.
* A Ruspini partition built by `tribblefis.ruspini.build_triangular_partition`,
  with singleton consequents, converts to a one-hidden-layer ReLU network
  agreeing to `< 1e-10`, with **exactly one hidden unit per apex knot**, using no
  data at all. Twenty-five randomized partitions.
* The piecewise-linear decomposition is exact *between* knots as well as at
  them, which is what separates a genuine PWL identity from an interpolation
  that happens to hit the samples.

## H2 — triangularization collapses, and the mechanism is dimensional (falsified)

Concrete, sweeping TRIBBLE's own `top_n` (`triangularization.md`):

| features kept | Gaussian RMSE | triangular RMSE | triangular dead rows |
|---|---|---|---|
| 1 | 15.19 | 15.66 | 1.0% |
| 3 | 12.89 | 14.13 | 1.6% |
| 5 | 8.80 | 12.81 | 5.3% |
| 6 | 7.88 | 12.50 | 6.0% |
| 7 | 7.32 | 22.23 | 35.4% |
| 8 | 7.14 | **32.07** | **70.7%** |

"Dead rows" are test rows whose total firing strength across every rule is
`<= 1e-6`, which `regression._normalize_firing_strengths` maps to a prediction
of exactly 0. Bikeshare at 12 features: **100% dead**. WEC at 12: 18.3%.

The cause is not a bad triangle fit. A Gaussian is positive everywhere, so every
rule always fires a little and the normalization always has something to divide
by. A triangle is zero outside its feet, and under the product t-norm a rule's
strength is the product across features — so **one** feature landing outside its
triangles zeroes that rule, and the probability that this happens for every rule
compounds with the feature count. It is a dimension effect, visible as a knee
between 6 and 7 features.

This matters beyond this experiment: `triangle_fit.fit_triangles_to_mixture`'s
docstring says it "turns a Gaussian-based FIS into a triangle-based one without
touching the rule base", and at 8 features it turns it into a model that
predicts zero for 71% of inputs. **Worth raising upstream** — either as a
documented precondition or as a coverage-preserving widening rule.

## H3 — the seed reproduces the FIS, in proportion to how additive the FIS is

Fidelity is the analytic seed's RMSE against the FIS it was backed out of,
relative to that FIS output's own standard deviation. Zero means the seed *is*
the FIS.

| rung | features | seed fidelity |
|---|---|---|
| synth1d | 1 | **0.030 ± 0.003** |
| concrete | 8 | 0.294 ± 0.058 |
| bikeshare | 12 | 1.027 ± 0.442 |
| wec | 12 of 301 | 1.170 ± 0.705 |

Exactly the predicted shape. With one input there is nothing to average over,
the partial-dependence profile *is* the FIS's function, and the seed carries it
whole. With more inputs the seed carries the FIS's additive part and the
residual is the FIS's interaction structure, which no axis-aligned first layer
can hold. Fidelity above 1.0 means the seed is worse than predicting the FIS's
own mean — on bikeshare and WEC the FIS is mostly *not* additive.

## H4 — time to quality (the metric that matters)

Wall-clock seconds to first reach a target, FIS fit and conversion charged to
the hot arms, mean over ten seeds. Targets are multiples of the best RMSE any
arm reached on that seed, so every arm faces the same bar. Full tables in
`time_to_quality.md`.

**synth1d** — the rung where the conversion is exact:

| arm | FIS parity | 1.50x best | 1.10x best | 1.05x best |
|---|---|---|---|---|
| `hot-analytic` | 0.06 | 0.06 | 0.06 | 0.07 |
| `hot` | 0.06 | 0.06 | 0.06 | 0.06 |
| `he-all` | never | never | never | never |

The randomly initialized arms do not reach *any* target in 150 epochs; they
plateau at 0.195 RMSE against the hot arms' 0.053. This is the hot start working
exactly as hoped, on the rung where the theorem holds without qualification.

**concrete** — speedup over `he-all` at the same target:

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best |
|---|---|---|---|---|
| `hot` | 0.1x | 0.2x | 0.5x | never |
| `quantile` | inf | inf | **22.3x** | **3.1x** |
| `he` | 1.0x | 1.0x | 0.8x | 0.9x |

**bikeshare** — the largest rung, and the only one where the hot arms win
anything:

| arm | FIS parity | 1.50x best | 1.25x best | 1.10x best | 1.05x best |
|---|---|---|---|---|---|
| `hot-analytic` | 0.8x | **1.5x** | 0.8x | **1.4x** | 1.0x |
| `hot` | 0.7x | **1.4x** | **1.3x** | 0.9x | **1.3x** |
| `quantile` | **149.1x** | **8.2x** | 2.1x | never | never |

The pattern across rungs is a cost-ratio story. The FIS fit is 0.19 s on
Concrete against a 0.56 s training run — 34% overhead for a start that is not
better than quantile's. On bikeshare it is 0.51 s against 12 s — 4% — and the
hot arms move into the black at several targets. **The regime where a warm start
can pay is the one where training is expensive relative to a FIS fit**, and none
of these datasets is far enough into it to make the case decisively. That is the
next experiment, not a conclusion from this one.

## H5 — placement is not what helps (falsified)

`quantile` is the same architecture, the same width, the same closed-form
read-out, and the same optimizer as `hot`, differing only in that its knots come
from per-feature data quantiles instead of from the FIS's membership functions.
It matches or beats `hot` on every rung above 1-D, at every target it reaches.

Checked and excluded as an explanation: knot degeneracy. Merging near-duplicate
FIS knots (tolerance swept from `1e-9` to `1e-2` of the unit-scaled range, which
moves the width from 225 to 169 units) changes the seed's test RMSE from 5.260
to 5.178 — inside the seed-to-seed spread. The FIS's knots are not being wasted
on duplicates; they are simply not better placed than quantiles for an additive
piecewise-linear model.

The honest reading is that **the useful invention here is the architecture**: an
axis-aligned ReLU knot layer with a closed-form read-out is a very strong, very
cheap regressor, and the equivalence is what makes it obvious that such a thing
should exist. Which knots you feed it matters much less than that you feed it
some.

## H6/H7 — the ceiling is the FIS, and WEC finds it

| dataset | FIS test R² | `hot` test R² | `he-all` test R² |
|---|---|---|---|
| synth1d | 0.975 | 0.978 | 0.698 |
| concrete | 0.809 | 0.883 | 0.904 |
| bikeshare | 0.680 | 0.914 | 0.925 |
| wec | **−89.7** | −6.5 | **0.966** |

On WEC, TRIBBLE restricted to 12 of 301 columns produces a model with R² of
−89.7 — far worse than predicting the mean — and the warm start faithfully
inherits it. A plain network on all 301 raw columns reaches 0.966. A warm start
cannot be better than what it is warm from, and H7's premise (that the advantage
grows with dimension because feature selection matters more) inverts: the same
mechanism that should have helped is what breaks.

**But feature selection itself does work.** Within WEC, comparing two arms that
differ *only* in which columns they see:

| arm | features | time to `1.5x best` |
|---|---|---|
| `he` | TRIBBLE's 12 | **0.18 s** (2/10 seeds) |
| `he-all` | all 301 | 1.62 s (10/10) |

9.2× faster to the loose target on the seeds that get there — though `he` then
stalls and never reaches the tighter ones, because 12 columns are not enough for
this problem. TRIBBLE's contribution to a neural network on this data is
*which inputs to look at*, delivered in 1.6 s. That is a real result, and it is
not the one the experiment was designed to find.

## H8 — gating reaches the conversion after all (falsified above 1-D)

Backing the seed out of the FIS's response rather than its gates was supposed to
make the t-norm choice structurally irrelevant. It does — a product t-norm no
longer blocks anything. But it still reaches the seed's *quality*, because the
norm family changes how additive the FIS is (`gating.md`):

| norm family | concrete FIS RMSE | seed fidelity | synth1d seed fidelity |
|---|---|---|---|
| `probability` (default) | 7.138 | **0.313** | 0.030 |
| `einstein` | 7.388 | 0.450 | 0.030 |
| `hamacher` | 6.831 | 0.504 | 0.031 |
| `min/max` | 6.829 | 0.513 | 0.031 |
| `luk` | 34.029 | 1.283 | 0.030 |

In one dimension every family gives fidelity 0.030 — exactly as predicted, since
there is no interaction for the gate to create. In eight dimensions fidelity
spreads by a factor of four across the families that work at all.

A convenient alignment: `probability`, which `experiments/fis-acceleration`
already made the default on accuracy grounds, is also the family whose FIS is
most additive and therefore converts best. The gating choice that is right for
the FIS is right for the conversion — so on current evidence there is nothing to
trade off, and no reason to change the default for this pipeline's sake.

---

## Two things that need a decision from you

**1. Bikeshare's shared loader leaks the target.** `cnt` is exactly
`casual + registered`, and `_fuzzy_models.load_bikeshare` drops only `cnt` and
`instant`, leaving both components in `X`. The first run of this experiment
scored 0.897 RMSE against the FIS's 33.9, on a target whose standard deviation
is ~181 — a model finding a sum, not learning demand.

This experiment uses its own leak-free loader (`load_bikeshare_noleak`) and the
numbers above are all from it. The shared loader is deliberately **not** patched:
proposal Tables 4.1 and 6.1 quote it, and changing what it returns would move
archived numbers with no table announcing the change — the exact failure
`WORKINGDOC.md` catalogues. Fixing it means re-running and re-quoting those
tables, which is your call, not mine.

**2. `triangle_fit.fit_triangles_to_mixture` needs a precondition or a fix.**
See H2. At 8 features it produces a model that predicts zero for 71% of inputs
and its docstring gives no warning. Candidate upstream issue.

## What I would do next

* **Test the regime where a warm start can actually pay.** Every rung here trains
  in under 13 s, so a 0.2–1.6 s FIS fit is 4–34% overhead. The claim needs a
  problem where training costs minutes: more epochs, a wider or deeper network,
  or a genuinely large dataset (PhiUSIIL at 235k rows, RT-IOT2022 at 123k).
* **Take the architecture seriously on its own.** `quantile` — axis-aligned ReLU
  knots plus a closed-form read-out — reaches `1.25x best` on Concrete 22×
  faster than a standard network and beats every other arm's final RMSE
  (4.739 vs 5.105). That is a result about a cheap, interpretable regressor that
  the equivalence pointed at, independent of whether TRIBBLE placed the knots.
* **Close the loop on WEC.** The FIS at `top_n=12` is broken there (R² −89.7).
  Either TRIBBLE's feature selection is picking badly on 301 correlated buoy
  columns, or 12 is simply too few. Worth knowing which, since the FIS's own
  quality is the ceiling on everything downstream.
* **Try the 2025 paper's tetrahedral construction.** Every failure above 1-D
  traces to the same place: an axis-aligned first layer can only carry the FIS's
  additive part. A simplicial partition is exactly the fix the IJCCC paper
  proposes, and would make the conversion exact in n dimensions rather than a
  projection. That is the version of this experiment that could confirm H3 and
  H5 together.
