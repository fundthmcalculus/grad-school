# Experiment: accelerating fuzzy inference training in `tribble-fis`

**Status:** complete, all five PRs open · **Started:** 2026-08-01

Findings are in [`RESULTS.md`](RESULTS.md). Short version:

*Phase 1 (acceleration):* **3.7–5.5x on training, 3.3–8.7x on inference,
bit-identical output**, plus a further 1.9–4.9x available on a GPU as an explicit
opt-in.

*Phase 2 (operators and search):* once the arithmetic was fast, the bottleneck
moved out of it and the question became a modelling one. `min/max` turned out to
be the **worst** of the four De Morgan families; switching the default to
`probability` and pairing it with SLSQP plus an analytic gradient gives
**+2.6 points of accuracy and another 1.96x** on top.

Five of the seven hypotheses across both phases came back wrong, several in
instructive ways — see the scoring sections there.

## Question

How much faster can zeroth-order TSK fuzzy inference — training and inference —
be made without changing a single number it produces, and at what point (if
any) does a GPU start to pay for itself?

"Training" here means antecedent refinement
(`tribblefis.refine.refine_classifier_antecedents` and its regressor siblings),
which searches the Gaussian membership functions' `(mu, sigma)` against a
classification or regression loss. It is by far the most expensive thing the
library does, and it is expensive for a structural reason: the search evaluates
a full forward pass tens of thousands of times.

## Where the time goes (measured, not assumed)

Profiled at `tribble-fis@b7d25c5`. See `tribble-fis/benchmarks/README.md` for
the full tables and how to reproduce them.

A large forward pass (50k samples × 20 features × 8 labels × 4 MF, 165 ms) is
**93% `GaussianMembership.evaluate`**. Each of the 640 calls allocates about
five 50 000-element temporaries (`x - mu`, `/sigma`, `**2`, `* -0.5`, `exp`) to
produce one membership column. The kernel is bandwidth-bound on temporaries,
not compute-bound on useful arithmetic.

An end-to-end classifier refinement (625 ms, ~1.3k fitness evaluations) splits
roughly:

| cost | share | why it exists |
|---|---|---|
| the forward kernel | ~63% | the same work as above, run once per fitness call |
| `apply_gaussian_params` | ~17% | rebuilds the whole immutable `NamedTuple` model tree on every fitness call (~65k `_replace`) |
| pandas column lookup | ~5% | `_classifier_proba` does not pass the pre-extracted `feature_arrays` mapping |
| SciPy L-BFGS-B machinery | remainder | 96 tiny sub-problems, mostly finite-difference gradients |

Two independent kinds of waste, then. The **kernel** is slow per element. The
**search** is wasteful per evaluation: coordinate descent perturbs one
membership function's two parameters, and then recomputes every membership
function of every feature for every label.

## Hypotheses

**H1 (representation).** Flattening the model into contiguous parameter arrays
once — instead of walking a dict-of-dicts-of-`NamedTuple`s per evaluation —
removes the rebuild cost outright and makes the kernel expressible as batched
array operations. *Predicted: large win on small/wide workloads, modest on
large ones.*

**H2 (fusion).** A Cython kernel that fuses the Gaussian evaluation and the
norm folds into one pass over each sample removes the temporaries entirely.
*Predicted: this is where the large-workload win lives, since the baseline is
bandwidth-bound.*

**H3 (algorithmic).** Because a coordinate-descent step touches exactly one
`(feature, label)` cell, caching the per-cell conorm folds turns an O(features ×
labels × MF) evaluation into O(MF) plus an O(features) recombination.
*Predicted: the largest single training win, and it multiplies with H1/H2
rather than overlapping them.*

**H4 (GPU).** A single forward pass is too small and too memory-bound to beat a
good CPU kernel at PCIe distance. A GPU should win only where the work is
genuinely batched: many candidate parameter vectors evaluated at once
(population optimizers), or very large inference batches.
*Predicted: loss or break-even on `forward-*`, win on batched-candidate search.*

## Method

Every claim is a delta on a fixed, seeded benchmark suite
(`tribble-fis/benchmarks/`) with a per-workload checksum. A change is only
allowed to move the time column; a moved checksum invalidates the result on the
same row, and `--compare` exits non-zero when one moves.

Baseline hardware: Windows 11, 13th-gen mobile i9, RTX 4080 Laptop (12 GB,
compute 8.9), Python 3.12.3, NumPy 2.4.6.

## Delivery

A stack of small PRs on `fundthmcalculus/tribble-fis`, each one rebased on the
last and each carrying its own before/after table:

1. `perf/01-benchmark-harness` — the suite, the baseline, the profiles above.
2. `perf/02-vectorized-kernel` — H1: `CompiledFIS` flat representation + NumPy kernel.
3. `perf/03-cython-kernel` — H2: fused Cython kernel over the same representation.
4. `perf/04-incremental-fitness` — H3: cached per-cell folds for coordinate descent.
5. `perf/05-gpu-backend` — H4: batched-candidate Torch/CUDA backend.

## Results

Filled in as each PR lands — see `RESULTS.md`.
