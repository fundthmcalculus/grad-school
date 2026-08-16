# Experiment: converting a TRIBBLE-built FIS into a ReLU network, and training on from there

**Status:** running · **Started:** 2026-08-16 · Results in [`RESULTS.md`](RESULTS.md)

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

## Why the conversion is not simply "apply the theorem"

The identity the papers turn on is exact and small — a triangular membership
function is a sum of three ReLUs of the input, with the membership function's
knots as the ReLU biases:

```
T(x; a, b, c) = s_a * relu(x - a) - (s_a + s_c) * relu(x - b) + s_c * relu(x - c)
```

but the *system-level* equivalence needs two more things, and a TRIBBLE
regressor supplies neither:

1. **Partition of unity.** The equivalence needs the terms to sum to 1
   everywhere, so that TSK's firing-strength normalization is a division by 1
   and vanishes. Division is not piecewise linear, so nothing survives without
   it. TRIBBLE's `GaussianMixtureModel` gives every `(feature, class)` pair its
   own private Gaussians which overlap arbitrarily and "do not sum to anything
   in particular" (`tribblefis.ruspini`'s own words).
2. **Piecewise-linear firing strengths.** TRIBBLE defaults to the
   probabilistic-sum/product norm pair. A product of piecewise-linear functions
   is piecewise *multi*linear, which no finite ReLU network reproduces exactly.
   This is precisely why the 2025 sequel had to replace triangles with
   *tetrahedral* membership functions — a simplicial partition, whose
   barycentric coordinates are piecewise linear and sum to one by construction —
   to get past one dimension.

So the honest structure of this experiment is a ladder, not a single claim:

* **In one dimension, on a Ruspini partition, the conversion is an identity.**
  No data, no fitting, machine precision. That is H1, and `test_fis2nn.py` pins
  it against `tribblefis.ruspini`'s own partition builder.
* **In n dimensions the conversion is a warm start**, and the experiment's job
  is to measure exactly what that warm start is worth and what it costs.

## What gets converted

The FIS's membership functions are expanded into ReLU knots (Gaussians first
fitted to triangles by the package's own `tribblefis.triangle_fit`), those knots
become hidden layer 1 — one unit per knot, each reading a single feature, so the
initial network is additive across inputs — and the read-out is then solved in
closed form by ridge least squares. That last step is not a training run: for
fixed hidden units the output is linear in the read-out, the same argument
`regression.solve_tsk_consequents` makes for TSK consequents at fixed firing
strengths. The whole conversion is one linear solve.

Architecture, identical for every arm:

```
y = relu(X @ W1 + b1) @ w2 + X @ v + c
```

The linear skip is not decoration: a ReLU layer whose knots all sit inside the
data range cannot express a slope to the left of its first knot, and the exact
1-D conversion needs one.

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

**H3 — The converted network starts hot.**
At epoch 0 the converted network is dramatically better than He-random
initialization, and at least as good as the FIS it came from.
*Falsified by* an epoch-0 test RMSE worse than the FIS's.

**H4 — The hot start is cheaper, not just earlier.**
Counting the FIS fit against it, the converted network reaches the quality a
from-scratch network eventually attains in less wall-clock time than training
from scratch takes to get there.
*Falsified by* the from-scratch arm reaching its own final quality sooner in
seconds.

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
On WEC (301 raw features, of which TRIBBLE keeps a handful) the margin over a
from-scratch network trained on all raw columns is larger than on Concrete
(8 features, all kept), because feature selection is part of what the FIS
contributes.
*Falsified by* the margin failing to widen with dimension.

## Arms

Six, sharing one architecture, one optimizer, one epoch budget, and one hidden
width (the number of knots the FIS produced), so no arm has a capacity edge.

| arm | layer 1 | read-out at epoch 0 | features | what it isolates |
|---|---|---|---|---|
| `hot` | FIS knots | closed-form ridge | FIS-selected | the conversion |
| `quantile` | per-feature quantiles | closed-form ridge | FIS-selected | knot *placement* (H5) |
| `elm` | He-random | closed-form ridge | FIS-selected | the closed-form read-out alone |
| `he` | He-random | random | FIS-selected | standard NN training |
| `quantile-all` | per-feature quantiles | closed-form ridge | **all raw** | from-scratch, no FIS at all |
| `he-all` | He-random | random | **all raw** | from-scratch, no FIS at all |

The `-all` arms exist because of a trap: running the controls on the features
TRIBBLE kept makes them differ from `hot` only in knot placement — which is what
H5 needs — but it also hands them the FIS's feature selection for free, which
would make them dishonest stand-ins for "what you would have done without a
FIS". On WEC that is 8 columns out of 301. Both families are therefore reported.

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
* Wall clock for `hot` is charged the FIS fit *and* the conversion solve. A warm
  start you cannot afford is not a warm start.

## Datasets

All already in `data/`; see `research/proposal-defense/prose/DATASETS.md`.

| dataset | rows × features | why it is here |
|---|---|---|
| Concrete compressive strength | 1,030 × 8 | the proposal's regression workhorse; small, all features informative |
| Bike sharing (hourly) | 17,379 × 14 | the scale partner, 17× larger |
| WEC Sydney (100 buoys) | 2,319 × 301 | the high-dimensional case, where feature selection is most of the work (H7) |

## Running it

```bash
python experiments/fis-to-neural-net/test_fis2nn.py          # H1, seconds
python experiments/fis-to-neural-net/run_experiment.py       # everything else
FIS2NN_SEEDS=0,1 python experiments/fis-to-neural-net/run_experiment.py \
    --datasets concrete --epochs 150                          # a quick look
```

Writes `results.json` (every curve, every seed, every arm) and
`results_summary.md` (the tables `RESULTS.md` quotes).
