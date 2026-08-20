# Results — the laboratory record

The narrative write-up is [`paper.md`](paper.md). This file is the record it
draws on: every hypothesis as written *before* its run, and how it scored.

Run of record: `outputs/results.json`, ten seeds, 150 epochs, commit `d830022`,
`tribble-fis` `5b92ec8`. The tetrahedral follow-up is `outputs/simplicial_results.json`,
five seeds, and the Part 3 refinements are `outputs/warped_results.json`, five seeds.
Tables quoted here are generated, not transcribed — `outputs/results_summary.md`,
`outputs/time_to_quality.md`, `outputs/triangularization.md`, `outputs/gating.md`, `outputs/simplicial.md`,
`outputs/warped.md`.

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

**Part 2 took the obvious next step** — the 2025 paper's tetrahedral
construction, which removes the additive restriction in principle. It is exactly
and cheaply representable (`O(n)` ReLU units per rule, `n+1` rules active at any
point, vertices bounded by the data rather than by `K**n` — 5,162 built where a
dense partition would need 43 million). But the binding constraint turns out to
be statistical rather than computational: in eight dimensions there are 6× more
tetrahedral rules than training rows, so the rules have no data behind them. The
usable form is a hybrid — additive main effects, tetrahedral interactions on a
few features — which improves the conversion by 17–30% on every dataset.

**Part 4 settled the open question.** PhiUSIIL at 235,795 rows was chosen as the
regime where a warm start should finally pay, and it does not: the conversion is
excellent — the seeded network starts at 0.0035 error against the FIS's own
0.0060, with no labels, in 0.5 s — but a from-scratch network reaches 2% error in
**0.03 s of training** against the FIS's **2 s** of setup. Scale buys per-epoch
cost, not epochs-to-target, and epochs-to-target is what a warm start saves.

**Part 5 found the regime and the warm start finally paid.** On the damped
double-pendulum time-step operator — 3,444 updates to R2 0.9 against PhiUSIIL's
25 — the converted network reaches R2 0.9 in **2.45 s against 11.62 s** for
random initialization, 4.7x on wall clock and 15.6x in updates, with the FIS fit
charged against it. That confirms the Part 4 diagnosis exactly: the variable is
updates-to-target, not rows. **But the quantile-knot baseline still wins**,
landing at R2 0.9387 in 0.64 s and zero updates — better than any arm reaches
after 20 epochs. Three parts of this experiment now say the same thing from
different directions: the architecture is the contribution, not the knots.

**Part 3 removed the hybrid's two arbitrary choices** — a bounding-box lattice,
and a subspace ranked by main effects. Putting the vertices on the FIS's own
knots does not lower the best fidelity but makes it *robust*: the lattice arm
collapses when asked for a subspace it cannot support (0.597 at k=4 on
Concrete), the warped arm holds flat at 0.255–0.265 across every k. Ranking the
subspace by pair lift instead of importance pays exactly where interactions
dominate. Neither moves the ceiling, and the reason is the same one Part 1
found from the other side: **the ceiling is the FIS**.

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
| T1–T5 | the tetrahedral construction (Part 2) | 2 confirmed, 2 falsified, 1 partial |
| W1–W2 | FIS-aligned vertices, interaction subspaces (Part 3) | both partial |
| P1–P4 | PhiUSIIL at full scale (Part 4) | 2 confirmed, 2 falsified |
| S1–S4 | slow-converging problem, damped and undamped (Part 5) | 2 confirmed, 2 falsified |

Six of eight in Part 1 came back wrong or partly wrong, four of five in Part 2,
and both in Part 3. Several are more useful that way.

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

Concrete, sweeping TRIBBLE's own `top_n` (`outputs/triangularization.md`):

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
`outputs/time_to_quality.md`.

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
norm family changes how additive the FIS is (`outputs/gating.md`):

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

---

# Part 2 — the tetrahedral construction

Prompted by the obvious next step above: every failure past one dimension traced
to an axis-aligned first layer carrying only the FIS's additive part, and the
2025 IJCCC paper's fix for exactly that is to replace triangular membership
functions with **tetrahedral** ones. The paper is now in
`papers/nn-fis-equivalence/`; the implementation is `simplicial.py`, the tests
are `test_simplicial.py` (8/8), the measurements are `outputs/simplicial.md`.

## The construction, and one exactness result worth having

The paper's n-D rule is *one TSK rule per triangulation vertex*: `If M_p(x) then
z = f(p)`, where `M_p` is a **polyhedral pyramid** — the piecewise-linear hat
over the simplices meeting at `p`. Their triangulation is the one induced by a
trained network's own linear regions, so the interpolation is exact by
construction.

Running the map backwards there is no such canonical triangulation, because a
TRIBBLE FIS is not piecewise linear. `simplicial.py` imposes a regular
**Freudenthal/Kuhn** lattice, on which the hat has a closed form:

```
phi_v(x) = relu( 1 - relu(max_i d_i) - relu(max_i (-d_i)) ),    d = (x - v)/h
```

Checked against Kuhn interpolation at **zero** error in dimensions 1, 2, 3, 5
and 8 — not to a tolerance, bit-identical. Since `max(a,b) = a + relu(b-a)`, a
tetrahedral membership function is `2(n-1) + 3` ReLU units at depth
`ceil(log2 n) + 2`: **O(n) units, O(log n) depth**, the n-dimensional analogue
of "a triangle is three ReLUs". Barycentric weights are non-negative and sum to
1 to `1e-15` at n = 32.

## Computational scaling is a non-issue — this part works

Three facts, all measured:

* **Only `n+1` rules fire at any point**, in any dimension, and which ones is
  found by sorting the fractional coordinates — `O(n log n)`, no search over
  simplices.
* **Only vertices the data reaches are ever built.** A dataset of `N` rows
  touches at most `N(n+1)` of them, whatever the lattice.
* The gap that buys is not marginal:

| dataset | features | dense tetrahedral partition at K=8 | vertices actually built |
|---|---|---|---|
| concrete | 8 | 43,000,000 | 5,162 |
| bikeshare | 12 | 282,000,000,000 | 8,192 (capped) |
| wec | 12 | 282,000,000,000 | 1,788 |

The `K**n` rule explosion that `tribblefis.anfis` raises `RuleExplosionError`
over simply never materializes.

## The binding constraint is statistical, not computational

**The paper's own consequent rule is the worst of three in high dimension.**
`c_v = f(p)` is exactly right when `f` is the network being decompiled — it is
being evaluated at vertices of its own linear regions. When `f` is a FIS being
converted on an imposed lattice, those vertices sit *off the data manifold*. On
Concrete the FIS spans 1.4–78.0 on real rows and −3.9–90.2 at lattice vertices,
with the median vertex 1.12 cells from the nearest datum. Fidelity, full
dimensional lattice, five seeds:

| dataset | features | rows/vertex | `c_v = FIS(v)` | support-weighted | projected |
|---|---|---|---|---|---|
| synth1d | 1 | 48.0 | 0.207 | 0.171 | **0.086** |
| concrete | 8 | 0.16 | 1.420 | 0.643 | 0.698 |
| bikeshare | 12 | 1.70 | 1.029 | 0.914 | 1.841 |
| wec | 12 | 1.18 | 5.758 | 1.006 | 0.818 |

(`K=8` row for each; fidelity 0 means the conversion reproduces the FIS exactly,
and >1 means it is worse than predicting the FIS's own mean.)

The `rows/vertex` column is the whole explanation. In one dimension a lattice
vertex has 48 rows behind it and the conversion is excellent. In eight
dimensions it has 0.16 — there are 6× more tetrahedral rules than training rows,
and refining the lattice cannot help, because the occupied-vertex count
saturates near `N(n+1)` while the resolution keeps rising. **You cannot buy
resolution and support at the same time in high dimension**, which is the curse
of dimensionality arriving exactly where the theory says it must.

## The construction that does scale: additive main effects + tetrahedral interactions

Main effects go in the first-order additive seed, where every one of the `N`
rows feeds every 1-D profile. Interactions go in a tetrahedral basis over the
top `k` features the FIS ranked, at a resolution chosen automatically to keep
~10 rows behind every vertex (`simplicial.auto_resolution`; the threshold is
measured, not assumed — fidelity turns erratic below ~5, swinging 0.42 → 1.76 →
2.51 across neighbouring resolutions once vertices outnumber rows).

Cost is `O(n * knots + K**k)` with `k` small and fixed, so it does not grow with
the full feature count. Fidelity, best `k` per dataset, five seeds:

| dataset | additive seed | hybrid | best k | K | vertices | ReLU units | improvement |
|---|---|---|---|---|---|---|---|
| synth1d | 0.030 | **0.021** | 1 | 24 | 26 | 78 | 30% |
| concrete | 0.313 | **0.257** | 2 | 8 | 63 | 317 | 18% |
| wec | 1.471 | **1.169** | 1 | 24 | 16 | 48 | 21% |
| bikeshare | 1.101 | **0.911** | 5 | 4 | 906 | 9,964 | 17% |

It improves the conversion on every dataset and every seed, for a fixed cost of
0.04–2.5 s. And the failure mode is visible in the same table — on Concrete,
`k=4` (9.0 rows/vertex) scores 0.597 and `k=5` (4.0) scores 0.642, both far
worse than `k=2`. The subspace dimension is not a free parameter; it is bounded
by how much data you have.

## Scoring

| | hypothesis | verdict |
|---|---|---|
| T1 | the tetrahedral hat is a compact exact ReLU circuit | **confirmed** — zero error to n=8, O(n) units, O(log n) depth |
| T2 | the construction is computationally scalable | **confirmed** — n+1 active rules, data-bounded vertices, no `K**n` |
| T3 | it closes the fidelity gap the additive seed left | **partly** — 17–30% better via the hybrid, not closed |
| T4 | the paper's `z = f(p)` consequent transfers to this direction | **falsified** — worst of three estimators above 1-D |
| T5 | a full-dimensional tetrahedral basis is usable | **falsified** — support per vertex collapses exponentially |

---

# Part 3 — FIS-aligned vertices, and interaction-chosen subspaces

Part 2 left two arbitrary choices inside the hybrid. `run_warped.py` removes
both, one at a time, five seeds (`outputs/warped.md`).

**Where the vertices sit.** `simplicial.AxisWarp` warps each axis until the
FIS's own knots land on lattice integers, so a unit cell is one inter-knot
interval. Every hat stays exactly the closed form — the warped lattice is still
regular — and the warp is itself piecewise linear, so the composition is still a
ReLU circuit at a cost of one unit per interior knot per axis (195–275 units
across all axes here).

**Which features the correction spans.** The hybrid took the top `k` by
differentiation score, which ranks *main* effects.
`gauss_math.calculate_interaction_scores` ranks feature *pairs* by joint lift,
which is the question actually being asked.

## Neither is a uniform win, and the pattern says why

| dataset | additive | best lattice | best warped | best selector |
|---|---|---|---|---|
| concrete | 0.313 | 0.257 (k=2) | **0.255** (k=4, interaction) | either |
| wec | 1.471 | **1.066** (k=4, interaction) | 1.611 (k=4, importance) | interaction, decisively |
| bikeshare | 1.101 | 0.959 (k=4) | **0.932** (k=2) | no difference |

**Warping's value is robustness, not a lower floor.** On Concrete the best
lattice and best warped fidelities are within noise of each other (0.257 vs
0.255) — but the lattice arm *collapses* as the subspace grows (0.597 and 0.628
at `k=4`, where 9.0 rows per vertex is below the support threshold), while the
warped arm holds flat at 0.255–0.265 across every `k` and both selectors.
Putting vertices on the FIS's knots spends resolution where the FIS says
structure is, so the correction stops falling apart when asked for a subspace
it cannot support. That is worth having even though it does not move the best
case.

**Where the FIS's knots are unreliable, warping is actively harmful.** On WEC it
is much worse than the lattice at every setting (6.442 vs 1.278 at `k=2`). WEC is
the dataset whose FIS has R² of −89.7 — the knots are being placed by a model
that does not work, and aligning the geometry to them concentrates resolution in
the wrong places. The lattice's indifference to the FIS is a liability when the
FIS is good and a safeguard when it is not.

**Interaction selection pays exactly where interactions dominate.** On WEC — the
least additive FIS, fidelity 1.471 — it is the difference between +28% and −7%
at `k=4`. On bikeshare it selects the same features as importance at `k=2` and
`k=3`, so the arms are identical. On Concrete it is neutral.

## One bug found and fixed on the way

The first warped run scored **16.46** on WEC. Not extrapolation — test rows all
landed inside the training box. The cause was knot spacing: WEC's FIS knot gaps
span 4.6×10⁴-to-1 (4.17e-06 to 0.19), so two knots 4 microns apart became a full
unit cell, and the lattice put a boundary across a gap no data can resolve.

Merging knots below `AxisWarp.MIN_GAP` before building the warp brings it to
6.44. Worth recording that the *same* near-duplicate knots are harmless in the
first-order seed — sweeping the merge tolerance from 1e-9 to 1e-2 moved its test
RMSE by under 2% — because there a duplicate knot is just one more nearly
collinear ReLU column. In the warped construction it distorts the geometry
instead. `test_simplicial.py::test_from_knots_merges_near_duplicates` guards it.

## Scoring

| | hypothesis | verdict |
|---|---|---|
| W1 | FIS-aligned vertices beat a bounding-box lattice | **partly** — same floor, far more robust to subspace size; harmful when the FIS is bad |
| W2 | interaction-ranked subspaces beat importance-ranked | **partly** — decisive on the least additive FIS, neutral elsewhere |

Both refinements are worth keeping and neither moves the ceiling much. Across
all three, the best achievable conversion fidelity improved from Part 2's
0.257 / 1.226 / 0.959 to 0.255 / 1.066 / 0.932. **The ceiling is the FIS**, not
the conversion — which is the same conclusion Part 1 reached from the other
direction, and the reason WEC keeps being the dataset that breaks things.

---

# Part 4 — PhiUSIIL at full scale, the regime the warm start needed

Parts 1-3 all ended at the same wall: every dataset trained in under 13 s, so a
0.2-1.6 s TRIBBLE fit was 4-34% overhead and no warm start could amortize.
PhiUSIIL is 235,795 x 50, 17x bikeshare and 229x Concrete. `run_phiusiil.py`,
three seeds; the data is recovered into `data/` from `tribble-fis` history by the
command in `data/.gitignore` (57 MB, not vendored).

First classification rung, so the conversion seeds a **logit** and training
minimizes cross-entropy (`fis2nn.train_adam(loss="bce")`).

## The 99% premise reproduces, and it is preprocessing

TRIBBLE reaches **0.9940 accuracy in 2.1 s** at `top_n=5` — but only on scaled
inputs. On raw features the same fit scores **0.730**. The repository's own log +
min-max treatment is load-bearing, not incidental, and every arm here gets it.

## Two things had to be fixed before the comparison meant anything

**The partial-dependence seed fails on a saturating classifier.** 53.7% of the
FIS's logits are clipped extremes, so the profile averages are dominated by them
and the additive decomposition's `(F-1) * baseline` centering compounds it. The
seed's *ranking* survives — AUC 0.996 — but its level does not: raw error 0.574,
and still 0.032 after a two-parameter Platt rescale.

Projecting the FIS's logit onto the same ReLU basis by one ridge solve fixes it
completely, and still uses no labels:

| conversion route | seeded error | FIS's own |
|---|---|---|
| partial-dependence (Parts 1-3) | 0.5721 | 0.0060 |
| **logit projection** | **0.0035** | 0.0060 |

The converted network starts *better than the FIS it came from*, in 0.5 s, 63-72
hidden units, before a single gradient step. That is the cleanest confirmation
of the hot start in the whole experiment.

**`URLSimilarityIndex` alone scores 0.9914.** With it present every arm lands
within a fraction of a point of every other and the dataset cannot distinguish
initializations at all. `--drop-dominant` removes it; both runs are reported
(`outputs/phiusiil.md`, `outputs/phiusiil_hard.md`).

**The five-feature cap is a confound, not a result.** `hot` trains on the columns
TRIBBLE kept while `he-all` trains on all 49, so they differ in inputs as well as
initialization — and without the dominant feature that difference dominates
everything (5 columns cap the model at 1.16% error; 49 reach 0.01%). `hot-all`
removes it: same knots, same read-out target, embedded in the full feature space
with the linear skip covering the rest at zero initial weight. It starts where
`hot` starts (0.0435) and finishes where `he-all` finishes (0.0002).

## And the answer is still no — decisively, and for a measurable reason

Epoch resolution was too coarse to see anything (every arm crossed every target
inside one epoch), so the curves are recorded every 25 minibatches. Separating
setup cost from training cost, without the dominant feature:

| arm | setup (scale + FIS + convert) | training time to reach 2% error |
|---|---|---|
| `hot` | 3.07 s | **0.025 s** |
| `hot-all` | 2.86 s | 0.031 s |
| `quantile` | 1.15 s | 0.040 s |
| `he` | 1.15 s | 0.088 s |
| `he-all` | 1.15 s | 0.031 s |

**Training from scratch reaches 2% error in 0.03 seconds. The FIS fit that would
warm-start it costs 2 seconds** — roughly 60x the work it saves. The gap is not
close and it does not close by scaling the dataset: at 235k rows a from-scratch
network crosses every target inside the first 25 minibatches, so the setup cost
has nothing to amortize against.

That refutes the hypothesis the whole ladder was built to test, in the regime
chosen to give it the best chance. The reason is now precise: **a warm start pays
only when reaching the target takes longer than building the warm start, and
gradient descent on a well-conditioned problem reaches these targets in
milliseconds regardless of how many rows there are.** Row count buys per-epoch
cost, not epochs-to-target, and epochs-to-target is what a warm start saves.

## Scoring

| | hypothesis | verdict |
|---|---|---|
| P1 | the 99% FIS premise reproduces | **confirmed** — 0.9940, and preprocessing-dependent |
| P2 | the conversion works for classification | **confirmed** — seeds 0.0035 against the FIS's 0.0060, no labels |
| P3 | the partial-dependence route transfers | **falsified** — saturation breaks it; projection is the fix |
| P4 | at 235k rows the warm start finally pays | **falsified** — 2 s setup against 0.03 s of training |

---

# Part 5 — the warm start on a slow-converging problem, and it finally pays

`find_slow_problem.py` located one: the damped n=2 double-pendulum time-step
operator from `AnalyticalDynamics/chaos`, `(theta_1(0), theta_2(0), t) ->
theta_1(t)`, 62,000 real rows, **3,444 updates** for a from-scratch network to
reach R2 0.9 against PhiUSIIL's 25. Friction rather than the frictionless chain
because it is the better-conditioned of the two — the frictionless one stalls at
R2 0.76 however long it trains, which measures network width, not convergence.

`run_pendulum.py`, three seeds, 20 epochs. FIS fit and conversion charged to the
hot arms.

## Against random initialization, the hypothesis is confirmed

| arm | R2 at start | wall clock to R2 0.9 | updates to R2 0.9 |
|---|---|---|---|
| tribble FIS (0.84 s) | — | — | — |
| `hot` | **0.8747** | **2.45 s** | **347** |
| `he` | −2.1208 | 11.62 s (2/3 seeds) | 5,410 |

**4.7x faster in wall clock and 15.6x fewer updates**, with the 0.84 s FIS fit
and 0.8 s conversion both charged against it. And at R2 0.93 only the hot arms
arrive at all inside the budget — `he` never does.

The conversion itself is again near-exact: the projection seed starts at R2
0.8747 against the FIS's own 0.8746, having seen no labels.

**This is the first rung where the warm start does what it was supposed to do**,
and it confirms the Part 4 diagnosis rather than contradicting it: the variable
that matters is updates-to-target, not rows. PhiUSIIL had 235,795 rows and
needed 25 updates; this has 43,400 and needs 5,410.

## Against the cheap baseline, it still loses

| arm | setup | R2 at start | wall clock to R2 0.93 |
|---|---|---|---|
| `quantile` | **0.64 s** | **0.9387** | **0.64 s** |
| `hot` | 1.65 s | 0.8747 | 7.85 s |

`quantile` — the same axis-aligned ReLU knot layer with knots at per-feature
quantiles and one closed-form ridge solve — lands at R2 0.9387 immediately, which
is *better than any arm reaches after 20 epochs of training*, for 0.64 s and zero
updates. It beats `hot` at every target, by 2.6x at R2 0.9 and 12x at R2 0.93.

One asymmetry to state plainly, because it favours quantile: its ridge solve fits
the **labels**, while the hot arms fit the FIS's output and never see `y` during
setup. That is what "no labels" means and it is deliberate, but it means the two
are not doing the same job. The fair reading is that `quantile` is a *model*, not
an initialization — and on this problem it is the best model in the table.

The reason it works here is visible in the data: `theta_1(0)` is held at 120° for
every trajectory, so the operator is effectively a 2-input problem in
`(theta_2(0), t)` and damping makes it close to additive in `t`. An additive
knot basis is nearly the right hypothesis class, which is exactly the condition
under which Parts 1-3 found the same thing.

## The frictionless chain: a different failure, and a cleaner one

The obvious follow-up was the undamped chain, on the theory that it was
capacity-limited rather than slow and a wider network would fix it. **It is
neither.** Widening from 128 to 1024 units moves the from-scratch ceiling only
from R2 0.725 to 0.771, and the FIS plateaus in the same place — 0.558 at 16
buckets, 0.757 at 64. Two methods hitting the same ceiling from opposite
directions is the signature of an irreducible component, not an underpowered
model: without damping the trajectories separate exponentially, so past some
horizon in `t` the map `(theta_2(0), t) -> theta_1(t)` is not a function anything
can learn from a 0.1-degree grid of initial conditions. **R2 0.9 is unreachable
here, not slow**, and asking for it would have measured nothing.

Re-run at targets the problem admits (2 seeds, 32 buckets, 706 hidden units):

| arm | R2 at start | best R2 | to R2 0.6 | updates |
|---|---|---|---|---|
| tribble FIS (3.12 s) | — | 0.6696 | — | — |
| `hot` | 0.5562 | 0.6254 | 7.95 s | 310 |
| `hot-anova` | 0.5580 | 0.6672 | 7.75 s | 570 |
| `quantile` | 0.5682 | 0.6576 | **0.97 s** | 190 |
| `he` | −1.7690 | 0.6086 | 18.27 s | 3,190 |

The warm start still beats random initialization — 2.3x on wall clock at R2 0.6,
10x in updates — but the margin is less than half the damped case's 4.7x, and
`quantile` wins by 8x again.

**The interesting number is 0.5562.** In the damped case the projection seed
started at R2 0.8747 against the FIS's own 0.8746 — an essentially exact
conversion. Here it starts at 0.5562 against the FIS's 0.6696, losing 0.11 R2 in
the conversion alone. That is the additive-projection limit from Parts 1–3
appearing at its worst: damping makes the operator nearly additive in `t`, and
removing it makes the interaction between initial condition and time the whole
problem. **The conversion is exact exactly when the FIS is additive, and this is
the cleanest demonstration of that in the whole experiment** — the same FIS, the
same conversion, the same data generator, with damping as the only difference.

It also explains why `quantile` keeps winning and why the tetrahedral work in
Part 2 mattered: the ceiling on an axis-aligned conversion is the FIS's additive
part, and no amount of training-time advantage changes where it starts.

## Scoring

| | hypothesis | verdict |
|---|---|---|
| S1 | a slow-converging problem exists and is findable | **confirmed** — 3,444 updates, 138x PhiUSIIL |
| S2 | on it, the warm start beats random init on wall clock | **confirmed** — 4.7x at R2 0.9, 15.6x in updates |
| S3 | the FIS's knots beat quantile knots | **falsified again** — quantile wins at every target, damped and undamped |
| S4 | the frictionless chain is capacity-limited and a wider net fixes it | **falsified** — it has an irreducible component; 8x width buys 0.05 R2 |

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
* ~~**Try the 2025 paper's tetrahedral construction.**~~ Done — Part 2 above.
  It confirmed the representation and refuted the hope: the construction is
  exactly and cheaply representable, but a full-dimensional simplicial basis
  cannot be supported by the data at these sizes.

## After Part 2

* **Triangulate on the FIS's structure, not on a lattice.** The paper's
  triangulation is induced by the linear regions of the object being converted,
  which is why its interpolation is exact and why an imposed lattice's is not.
  The analogue here would be a complex built from TRIBBLE's own membership
  geometry — its Gaussian centres are already landmarks, and
  `tribblefis.ruspini` already merges them into shared knots. That puts vertices
  where the data is by construction, which is the one thing the lattice cannot
  do, and would address T4 and T5 together.
* **Choose the interaction subspace properly.** The hybrid currently takes the
  FIS's top-`k` features by differentiation score, which ranks *main* effects.
  `gauss_math.calculate_interaction_scores` already scores feature *pairs* for
  joint lift, and `detect_interactions=True` exposes it. Ranking the subspace by
  interaction rather than by importance is a one-line change with a real chance
  of moving the 17–30%.
* **A slow-converging problem has been found and measured** —
  `find_slow_problem.py`, `outputs/slow_problems.md`. Ranking candidates by *minibatch
  updates* to reach R2 >= 0.9 (updates, not seconds: they are what an
  initialization skips, and they are comparable across dataset sizes):

  | problem | updates | note |
  |---|---|---|
  | PhiUSIIL (Part 4) | **25** | why the warm start could not repay 2 s |
  | `illcond` (cond 1e4) | 36 | Adam absorbs conditioning entirely — dead end |
  | Concrete | 386 | the Part 1 reference |
  | **`pendulum-n2-fric`** | **3,444** | 62k real rows, the chaos time-step operator |
  | `chirp-k4` | 8,018 | synthetic spectral-bias probe |
  | `sine2d-k2` | 8,699 | synthetic, with interactions |

  Two results worth keeping. **Ill-conditioning is refuted as a route** — 36
  updates, because Adam's per-parameter scaling absorbs a condition number of
  1e4. And the oscillatory problems do not degrade gracefully: at 128 hidden
  units they jump from a few thousand updates to *never* (chirp at k>=8,
  sine2d at k>=4, and the frictionless pendulum at 0.76 R2). That is a
  **capacity** wall, not a convergence cost, and it means the benchmark's
  difficulty knob has to be frequency *and* width together.

  The recommended target is `pendulum-n2-fric`: 3,444 updates is 138x PhiUSIIL,
  the data is real rather than synthetic, it is in the author's own domain, and
  `AnalyticalDynamics/chaos` already applies TRIBBLE to exactly this operator —
  where the FIS beats all eight of the reference paper's time-step models in six
  of seven cells. That last point matters most: every part of this experiment
  ended up bounded by the FIS's own quality, so the one problem worth testing is
  one where the FIS is already known to be good.

* **Test the regime where a warm start can pay** (unchanged from Part 1, and
  still the biggest open question). Every rung here trains in under 13 s, so a
  0.2–1.6 s FIS fit is 4–34% overhead. The claim needs a problem where training
  costs minutes: PhiUSIIL at 235k rows, RT-IOT2022 at 123k, or simply a wider
  network and a longer budget.
* **Take the architecture seriously on its own.** `quantile` — axis-aligned
  ReLU knots plus a closed-form read-out — reaches `1.25x best` on Concrete 22×
  faster than a standard network and beats every other arm's final RMSE
  (4.739 vs 5.105). That is a result about a cheap, interpretable regressor that
  the equivalence pointed at, independent of whether TRIBBLE placed the knots.
* **Close the loop on WEC.** The FIS at `top_n=12` is broken there (R² −89.7).
  Either TRIBBLE's feature selection is picking badly on 301 correlated buoy
  columns, or 12 is simply too few. Worth knowing which, since the FIS's own
  quality is the ceiling on everything downstream.
