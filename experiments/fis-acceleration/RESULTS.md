# Results: accelerating fuzzy inference in `tribble-fis`

Reproduce with `python -m benchmarks.bench` in `tribble-fis/`; stored runs are in
`tribble-fis/benchmarks/results/`. Hardware: Windows 11, i9-14900HX (24 cores /
32 threads), RTX 4080 Laptop (12 GB, compute 8.9), Python 3.12.3, NumPy 2.4.6.

Every CPU number below carries an unchanged output checksum: the whole CPU stack
is **bit-identical** to the code it replaced, not merely close.

## Headline

| workload | baseline | final | speedup |
|---|---|---|---|
| forward-small (1k x 8 x 3 x 3) | 472 us | 118 us | **4.00x** |
| forward-wide (2k x 40 x 6 x 4) | 9.91 ms | 2.00 ms | **4.96x** |
| forward-large (50k x 20 x 8 x 4) | 164.98 ms | 24.85 ms | **6.64x** |
| forward-prob (20k, probability norms) | 71.32 ms | 8.22 ms | **8.68x** |
| predict-large (`predict_proba`, 50k) | 99.91 ms | 30.64 ms | **3.26x** |
| refine-classifier (training, small) | 625.08 ms | 170.15 ms | **3.67x** |
| refine-classifier-wide (training, realistic) | 3.701 s* | 675.13 ms | **5.48x** |

\* `refine-classifier-wide` was added mid-stack; its "baseline" is measured at
the Cython commit, so its number understates the total.

Plus, on the GPU (never automatic — see H4):

| | CPU | GPU | |
|---|---|---|---|
| 1M-sample forward pass, float64 | 351.24 ms | 188.72 ms | 1.86x |
| 1M-sample forward pass, float32 | 351.24 ms | 90.40 ms | 3.88x |
| 64 candidate evaluations, float32 | 67.95 ms | 13.85 ms | 4.91x |

## Hypotheses, scored

**H1 (representation) — half right, and the interesting half was the failure.**
Flattening the model into contiguous arrays removed the per-evaluation model
rebuild and the pandas lookups, which was worth 1.16x on training. But the
predicted win on the forward pass *did not materialise*: the NumPy pass over the
flat layout measured within a couple of percent of the reference nested loop. A
large forward pass is ~93% `np.exp`, and `np.exp` on float64 runs at ~280 M
elem/s on this machine — essentially the reference's entire runtime. **NumPy was
already at its floor; rebatching the same arithmetic cannot beat it.** That is
what redirected the rest of the work.

**H2 (fusion) — wrong as stated; threading was the mechanism, not fusion.**
Fusion alone *loses*: a serial compiled loop runs at 0.50x of NumPy on the
50k-sample workload, because libm's `exp` is one scalar call per element while
NumPy's is SIMD-vectorized. What pays is that samples are independent, so the
loop is a `prange` — 9–14x over NumPy across a size sweep from 4 800 to 6.4M
membership evaluations. The prediction ("this is where the large-workload win
lives") was right about the *outcome* and wrong about the *reason*, which
matters because it changed the dispatch rule: without OpenMP the compiled kernel
is only worth taking on small inputs.

**H3 (algorithmic) — confirmed, and the largest single training win.** A
coordinate-descent step touches one `(feature, label)` cell, so caching the
per-cell conorm folds turns an `O(n·F·K·L)` evaluation into `O(n·(K+F))` — 360
membership evaluations per sample down to 23 on the wide model. 5.52x on
realistic training, on top of everything before it. As predicted it multiplied
with H1/H2 rather than overlapping them.

**H4 (GPU) — qualified.** The prediction was "loss or break-even on forward
passes, win on batched candidates". Reality: a *win* on forward passes but a
modest one (1.86x float64, 3.88x float32), and the batched-candidate win is
real (4.91x) but is **not** a batching effect — the same candidates evaluated
one at a time take 15.83 ms against 13.85 ms batched. The device is saturated by
a single candidate at these sizes; batching saves parameter uploads, nothing
more. The prediction's reasoning ("too small and too memory-bound at PCIe
distance") was wrong about single passes and its conclusion about batching was
right for the wrong reason.

## What the profile looked like at each stage

Training (`refine-classifier-wide`, 4k x 20 x 6 x 3, 720 free parameters):

| stage | total | forward pass | model rebuild | proba/CE | SciPy |
|---|---|---|---|---|---|
| at the Cython commit | 3.79 s | 62% | — | ~8% | ~28% |
| after incremental folds | 1.53 s | ~5% | — | 36% | ~50% |
| after fusing the CE | 675 ms | ~10% | — | ~10% | ~70% |

The shape of the remaining cost has completely inverted. Training is now
**dominated by SciPy's L-BFGS-B machinery** — 360 sub-problems, each spending
finite-difference evaluations on a 2-parameter block — rather than by the fuzzy
arithmetic. That is the next thing worth attacking, and it is a different kind
of problem: an analytic gradient for the classifier objective (the regressor
already has one under `probability` norms, from issue #43) would remove most of
it, but the default `min/max` t-norm is only piecewise smooth, so it would be a
subgradient and would change the search trajectory. That is a modelling
decision, not an optimization, which is why this stack stopped here.

## Method notes worth keeping

**Checksums are what made the negative results legible.** Every workload records
a weighted checksum of its output and `--compare` fails on drift. Twice during
this work a change looked like a speedup and was actually a different
computation; twice a "regression" turned out to be measurement noise. Without
the checksum column the first class is invisible and the second is
indistinguishable from the first.

**Benchmarks need enough repeats to be evidence.** Once the compiled kernel took
`forward-small` under 200 us, 20 repeats gave ±25% run-to-run spread on the min
— enough to report a 0.79x regression on a code path nothing had touched. Raised
to 300.

**The CPU is the noisy side of a CPU/GPU comparison.** The 24-thread kernel
ranged 351–435 ms across repeats of an identical input while the GPU held within
1%. Comparisons here use the CPU's best run.

**An unrelated 40x on the dev loop.** The test suite took ~16–26 minutes because
an interactive matplotlib backend stalled on `plt.show()` inside
`tests/test_regression.py` and in library plotting code. Selecting Agg in
`conftest.py` took the full suite to **22 seconds**. This had nothing to do with
fuzzy inference and was worth more per day than several of the optimizations
above.

## Phase 2: the bottleneck moved, and so did the question

With the arithmetic ~5x faster, training stopped being dominated by it, and the
next round turned into a *modelling* investigation rather than a performance one.
Details in `tribble-fis/docs/analytic-gradient-evaluation.md` and
`norm-family-evaluation.md`.

**The default t-norm family was the wrong one.** Across 18 dataset × split
combinations (iris, wine, breast_cancer, digits, two overlapping
`make_classification` problems), `min/max` measured as the *worst* of the four
De Morgan families:

| family | refined accuracy | vs min/max |
|---|---|---|
| min/max | 0.7881 | — |
| hamacher | 0.8029 | +0.0148 ± 0.0078 |
| probability | 0.8135 | +0.0254 ± 0.0063 |
| einstein | 0.8175 | +0.0294 ± 0.0061 |

`luk` is unusable past a handful of features — its bounded sum saturates and
leaves 99–100% of rows with no membership at all (mean accuracy 0.4458).

**A bug made the earlier question unanswerable.** `refine_classifier_antecedents`
hard-coded the default pair and the estimator never passed `norm_conorm` down, so
`norm_conorm="probability", refine=True` tuned against min/max firing strengths
and deployed under the probabilistic pair. The accept/reject guard scored under
the wrong pair too. Fixed before any of the above was measured.

**Old defaults vs new**, same protocol:

| | accuracy | worst | time |
|---|---|---|---|
| min/max + L-BFGS-B + finite differences | 0.7881 | 0.3722 | 906.5 ms |
| probability + SLSQP + analytic gradient | **0.8138** | **0.4028** | **463.2 ms** |

**+0.0257 ± 0.0070 accuracy and 1.96x faster.** Accuracy from the family, speed
from the solver and the gradient; they compose because they act on different
parts of the problem.

### Three more hypotheses that died on measurement

- **"An analytic gradient will just be faster."** Under `min/max` it is a
  *subgradient*, and it turned the search into an accuracy lottery — mean −0.9pp,
  worst −9.7pp — for 1.43x. Under `probability` the objective is smooth, the
  closed form is exact, and the same measurement gives +0.0012 ± 0.0026 at 1.74x.
  The flag now keys on smoothness rather than being on or off.
- **"SLSQP is 1.60x free."** It was, under min/max. Re-measured under the new
  default family it is 1.14x — the smooth surface already suits L-BFGS-B, so
  there is less to win. Worth taking, but the number does not transfer.
- **"cProfile found two easy wins."** It had not. Both survived a first
  measurement at 1.12x and evaporated under a paired isolated A/B (medians 648 ms
  → 652 ms). cProfile's per-call overhead systematically overstates small
  functions called thousands of times, which is every function in that loop.

### The acceptance guard does not do what its name suggests

Worth recording independently. It compares a refined model against the
*heuristic*, which scored 0.005–0.51 on these problems — so it accepts
essentially everything, including a run that lost 9.7 points against a sibling
configuration. It protects against refining being worse than not refining, and
says nothing about one refinement being worse than another.

## The PR stack

Phase 1 — acceleration:

| | PR | receipt |
|---|---|---|
| 1 | [#53](https://github.com/fundthmcalculus/tribble-fis/pull/53) benchmark suite + baseline | the instrument, the baseline, the profiles |
| 2 | [#54](https://github.com/fundthmcalculus/tribble-fis/pull/54) compiled flat model | training 1.16x; the H1 negative result |
| 3 | [#55](https://github.com/fundthmcalculus/tribble-fis/pull/55) Cython kernel, threaded | forward 3.9–9.2x, training 2.13x |
| 4 | [#56](https://github.com/fundthmcalculus/tribble-fis/pull/56) incremental fitness | training 5.52x on a realistic model |
| 5 | [#57](https://github.com/fundthmcalculus/tribble-fis/pull/57) Torch/CUDA backend | 1.86–4.91x, opt-in only |

Merged as the squashed [#58](https://github.com/fundthmcalculus/tribble-fis/pull/58).

Phase 2 — operators and search:

| PR | receipt |
|---|---|
| [#59](https://github.com/fundthmcalculus/tribble-fis/pull/59) analytic gradient | built, evaluated, shipped off — the min/max verdict |
| [#60](https://github.com/fundthmcalculus/tribble-fis/pull/60) norm pass-through | the bug that made the rest measurable; `sub_method` |
| [#61](https://github.com/fundthmcalculus/tribble-fis/pull/61) probability default | +2.5pp accuracy |
| [#64](https://github.com/fundthmcalculus/tribble-fis/pull/64) SLSQP + auto gradient | 1.96x combined, accuracy-positive |
