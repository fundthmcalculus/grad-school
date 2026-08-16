# Experiment: converting a TRIBBLE-built FIS into a ReLU network, and training on from there

**Status:** measured · **Started:** 2026-08-16 · Results in [`RESULTS.md`](RESULTS.md)

## Question

Bede, Kreinovich & Toth (NAFIPS 2023; IJCCC 2025 — see
[`papers/nn-fis-equivalence/`](../../papers/nn-fis-equivalence/)) prove that a
Takagi–Sugeno fuzzy system with triangular membership functions and a neural
network with ReLU activation compute the same class of functions, and give the
map between them. Their direction of travel is network → rules: take a trained
black box, read it as a rule base, call it explainable.

This experiment runs the map backwards.

> TRIBBLE constructs a fuzzy inference system **consequent-first** — it finds the
> structure in the data (which features matter, where each one's modes are, what
> the output looks like in each region) without gradient descent, in a fraction
> of a second. If that FIS converts to a neural network, the network inherits
> that structure as its initial weights. **Does a TRIBBLE-derived initialization
> give a neural network a materially better starting point than random
> initialization — one that survives training rather than being washed out by
> it?**

If it does, the practical claim is a hot start: a network that begins at or
above the FIS's accuracy for the price of a FIS fit, and then improves under
ordinary gradient training instead of spending its first hundred epochs
discovering structure the FIS already had.

## Backing the equivalence out into the weights

The identity the papers turn on is exact and small — a triangular membership
function is a sum of three ReLUs of the input, with the term's knots as the ReLU
biases:

```
T(x; a, b, c) = s_a * relu(x - a) - (s_a + s_c) * relu(x - b) + s_c * relu(x - c)
s_a = 1 / (b - a),   s_c = 1 / (c - b)
```

Taken literally at the level of the FIS's *internal gates*, the system-level
equivalence then needs two more things, and a TRIBBLE regressor supplies
neither:

1. **Partition of unity**, so TSK's firing-strength normalization is a division
   by 1 and vanishes — division is not piecewise linear. TRIBBLE's
   `GaussianMixtureModel` gives every `(feature, class)` pair its own private
   Gaussians, which overlap arbitrarily and "do not sum to anything in
   particular" (`tribblefis.ruspini`'s own words).
2. **Piecewise-linear firing strengths.** TRIBBLE defaults to the
   probabilistic-sum/product pair, and a product of piecewise-linear functions
   is piecewise *multi*linear, which no finite ReLU network reproduces exactly.
   This is exactly why the 2025 sequel replaces triangles with *tetrahedral*
   membership functions — a simplicial partition, whose barycentric coordinates
   are piecewise linear and sum to one by construction — to get past one
   dimension.

**So this experiment does not convert the gates. It converts the function.**

The equivalence says a continuous piecewise-linear function of one variable *is*
a one-hidden-layer ReLU network, with the slope change at each knot as that
knot's output weight. It says nothing about how the piecewise-linear function
arose. So rather than demanding piecewise-linear firing strengths, we take the
FIS's own one-dimensional profiles — which we can evaluate exactly, whatever the
gating — and convert those:

* For each feature, sample the FIS's **partial-dependence profile**
  `g_f(t) = mean_i FIS(x_i with x_i[f] := t)` at that feature's own knots. This
  is the first-order term of the FIS's functional ANOVA decomposition; under
  independent inputs it is the exact projection of the FIS onto functions of
  that feature alone, which is the most any additive seed can carry.
* Convert each profile by second differences: slope changes become hidden-unit
  output weights, leading slopes become the linear skip, constants fold into the
  bias (`fis2nn.pwl_to_relu_weights`, exact — `test_fis2nn.py` pins it at knots
  *and* between them).

No labels are consulted anywhere in that. The seed is a conversion of the FIS,
not a refit against `y`, and `analytic_seed_from_fis` takes no `y` argument so
it cannot become one by accident.

Two consequences worth stating:

* **In one dimension there is nothing to average over, so the profile is the FIS
  itself and the seed reproduces it exactly.** That is the rung where the
  process is *proved* rather than merely measured — `synth1d` exists for it.
* **The gating choice stops being load-bearing.** A product t-norm no longer
  blocks the conversion; it only shapes the function being converted.
  `analysis_gating.py` measures whether it still matters empirically now that it
  no longer matters structurally.

### The two hot arms

`hot-analytic` is the seed exactly as backed out — every weight a function of
the FIS's parameters. `hot` gives it one closed-form ridge polish against the
labels, which is a single linear solve, not an epoch. That polish fits the
*residual* of the seed and adds the correction, so the ridge penalty shrinks
toward the backed-out weights rather than toward zero: a plain least-squares
read-out would solve the seed away and keep only the knots, discarding the
information the seed exists to carry.

Architecture, identical for every arm:

```
y = relu(X @ W1 + b1) @ w2 + X @ v + c
```

The linear skip is not decoration: a ReLU layer whose knots all sit inside the
data range cannot express a slope to the left of its first knot, and both the
exact 1-D conversion and the profile decomposition need one.

## Hypotheses

Each is falsifiable, with the measurement that would falsify it.

**H1 — The 1-D equivalence is constructive on this codebase.**
A Ruspini triangular partition with singleton consequents converts to a
one-hidden-layer ReLU network agreeing to `< 1e-10` on a dense grid, with one
hidden unit per apex knot, using no data.
*Falsified by* any error above tolerance, or by needing more units than knots.

**H2 — Triangularization is close to free.**
Replacing the fitted FIS's Gaussians with the package's MAE-optimal triangles
costs little test accuracy, so the exactly-convertible model is a fair stand-in
for the Gaussian one that TRIBBLE actually builds.
*Falsified by* a large accuracy gap between the Gaussian and triangularized FIS.

**H3 — The backed-out seed reproduces the FIS.**
`hot-analytic`, having seen no labels, matches the FIS it came from: exactly in
one dimension, and closely enough elsewhere to start at the FIS's accuracy.
*Falsified by* a seed materially worse than its own FIS.

**H4 — The hot start is cheaper, not just earlier.**
Counting the FIS fit against it, the converted network reaches the quality a
from-scratch network eventually attains in less wall-clock time than training
from scratch takes to get there.
*Falsified by* the from-scratch arm reaching its own final quality sooner.

**H5 — It is the FIS's *placement* that helps.**
The converted network beats an identically-shaped network whose knots are placed
at per-feature quantiles instead of by the FIS. This is the hypothesis that
distinguishes "TRIBBLE found the right breakpoints" from "axis-aligned ReLU
knots plus a closed-form read-out is a good architecture."
*Falsified by* the quantile ablation matching or beating the converted arm.

**H6 — The warm start survives training.**
After the full epoch budget, with early stopping chosen on validation, the
converted network is no worse than the from-scratch network.
*Falsified by* the from-scratch arm ending materially better.

**H7 — The advantage grows with input dimension.**
On WEC (301 raw features, of which TRIBBLE is asked to keep 12) the margin over
a from-scratch network trained on all raw columns is larger than on Concrete
(8 features, all kept), because feature selection is part of what the FIS
contributes.
*Falsified by* the margin failing to widen with dimension.

**H8 — Gating no longer reaches the conversion.**
Because the seed is backed out of the FIS's response rather than its gates, the
t-norm/t-conorm family changes the FIS's own accuracy but not the seed's
*fidelity* to whichever FIS it converted.
*Falsified by* seed fidelity varying materially across norm families.

### Part 2 — the tetrahedral construction

The 2025 IJCCC paper replaces triangular membership functions with **tetrahedral**
ones, which is exactly the fix for H3/H5's failure mode: an axis-aligned first
layer can only carry the FIS's additive part. `simplicial.py` implements it on a
Freudenthal/Kuhn lattice; `run_simplicial.py` measures it.

**T1 — The tetrahedral membership function is a compact exact ReLU circuit.**
The closed form `phi_v(x) = relu(1 - relu(max_i d_i) - relu(max_i -d_i))`,
`d = (x-v)/h`, equals the Freudenthal hat exactly, and expands to `O(n)` ReLU
units at depth `O(log n)`.
*Falsified by* any disagreement with Kuhn interpolation, or by a unit count that
grows faster than linearly in `n`.

**T2 — The construction is computationally scalable.**
Rule count follows the data, not the lattice: `n+1` rules fire at any point, and
only vertices the data reaches are built, so no `K**n` term ever materializes.
*Falsified by* the built-rule count tracking the dense grid.

**T3 — It closes the fidelity gap the additive seed left.**
Carrying interactions should push fidelity materially below the additive seed's
0.31 / 1.03 / 1.17.
*Falsified by* the tetrahedral conversion failing to beat the additive one.

**T4 — The paper's `z = f(p)` consequent transfers to this direction.**
Setting each rule's singleton to the FIS's value at its vertex is the paper's own
rule and needs no data at all.
*Falsified by* another estimator of the same consequents beating it.

**T5 — A full-dimensional tetrahedral basis is usable.**
*Falsified by* fidelity degrading as the lattice refines, or by the support per
vertex collapsing with dimension.

## Arms

Six, sharing one architecture, one optimizer, one epoch budget, and one hidden
width (the number of knots the FIS produced), so no arm has a capacity edge.

| arm | layer 1 | read-out at epoch 0 | features | what it isolates |
|---|---|---|---|---|
| `hot-analytic` | FIS knots | **backed out of the FIS**, no labels | FIS-selected | the equivalence itself (H3) |
| `hot` | FIS knots | seed + one anchored ridge solve | FIS-selected | the recommended conversion |
| `quantile` | per-feature quantiles | ridge, from zero | FIS-selected | knot *placement* (H5) |
| `elm` | He-random | ridge, from zero | FIS-selected | the closed-form read-out alone |
| `he` | He-random | random | FIS-selected | standard NN training |
| `he-all` | He-random | random | **all raw** | from-scratch, no FIS at all |

`quantile` and `elm` run on the features TRIBBLE kept, so they differ from `hot`
in knot placement and nothing else — which is what H5 needs. But that also hands
them the FIS's feature selection for free, so they are not honest stand-ins for
"what you would have done without a FIS"; on WEC that is 12 columns out of 301.
`he-all` is the arm that gets no FIS output of any kind.

## Protocol

* Ten seeds (`FIS2NN_SEEDS` to override), 80/20 train/test split per seed.
* Feature scaling fitted on the training fold only (`_fuzzy_models.fit_scaler`,
  which exists for exactly this).
* A 15% validation fold is carved out of the *training* fold. It chooses the
  learning rate and the stopping epoch. The test fold is scored and never
  consulted — reading either off the test curve would give each arm a free peek
  in proportion to how jagged its curve is, which is the axis the arms differ on.
* Learning rate swept per arm over `{3e-4, 1e-3, 3e-3, 1e-2}` on the first
  seed's validation fold, then pinned for that dataset's remaining seeds. Per
  arm because a warm-started net and a random one are not at comparable points
  on the loss surface; pinned because re-selecting per seed would let every arm
  draw four samples of its own noise.
* Wall clock for the hot arms is charged the FIS fit *and* the conversion. A
  warm start you cannot afford is not a warm start.

## Datasets — smallest first

The ladder is deliberate: the process is proved where the conversion is an
identity, and only then carried upward to where it is a projection.

| rung | dataset | rows × features | why it is here |
|---|---|---|---|
| 1 | `synth1d` (generated) | 600 × 1 | one input: the seed *is* the FIS, exactly. Proves the process. |
| 2 | Concrete compressive strength | 1,030 × 8 | the proposal's regression workhorse; small, all features informative |
| 3 | WEC Sydney (100 buoys) | 2,319 × 301 | the high-dimensional case, where feature selection is most of the work (H7) |
| 4 | Bike sharing (hourly) | 17,379 × 12 | the scale partner, 17× larger than Concrete |

Two dataset notes the driver encodes rather than works around:

* **Bikeshare leaks.** Its target `cnt` is exactly `casual + registered`, and
  `_fuzzy_models.load_bikeshare` leaves both columns in `X`. The first run of
  this experiment scored 0.897 RMSE against the FIS's 33.9 on a target with a
  standard deviation of ~181 — a linear model finding a sum, not a demand model.
  This experiment uses its own leak-free loader. The shared loader is *not*
  patched: proposal Tables 4.1 and 6.1 quote it, and silently changing what it
  returns would move archived numbers with no table announcing it.
  **This needs a decision upstream** — see `RESULTS.md`.
* **WEC needs `top_n=12`,** and the reason is a finding rather than a
  convenience: the converted network's width is the FIS's membership-function
  count, and nothing bounds it. At the default `top_p=0.95`, TRIBBLE keeps 300
  of 301 columns and builds 3,686 membership functions — an 8,751-unit hidden
  layer, 2.6M parameters in `W1` for 1,854 training rows.

## Running it

```bash
python experiments/fis-to-neural-net/test_fis2nn.py                  # H1, H3, seconds
python experiments/fis-to-neural-net/test_simplicial.py              # T1, T2, seconds
python experiments/fis-to-neural-net/run_experiment.py               # the ladder
python experiments/fis-to-neural-net/run_simplicial.py               # T3, T4, T5
python experiments/fis-to-neural-net/time_to_quality.py              # H4, from results.json
python experiments/fis-to-neural-net/analysis_triangularization.py   # H2's mechanism
python experiments/fis-to-neural-net/analysis_gating.py              # H8

FIS2NN_SEEDS=0,1 python experiments/fis-to-neural-net/run_experiment.py \
    --datasets synth1d concrete --epochs 150                          # a quick look
```

`run_experiment.py` writes `results.json` (every curve, every seed, every arm)
and `results_summary.md` (the tables `RESULTS.md` quotes).
