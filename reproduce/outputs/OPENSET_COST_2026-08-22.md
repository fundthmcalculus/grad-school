# Why `table_4_4_openset` takes 3h38m — and the order dependence found on the way

_Measured 2026-08-22 on the workstation of record, RT-IOT2022 (123,117 × 82, 12
classes). Reproduce with `reproduce/experiments/profile_openset_cost.py`, on a
quiet machine — these are wall-clock numbers._

**Summary.** The runtime is fully explained by one cost model, and the obvious
saving is not available: the antecedent screen ranks all 82 features and then
keeps all 82, which looks discardable and is not. Removing it changes the
prediction on 29% of test rows. The reason is the finding: **the complement rule
is order-dependent on its feature list**, which §4.3.5 describes as combined by a
commutative, associative t-conorm.

---

## 1. Where the time goes

One leave-one-class-out fold — train 92,293 × 82, 11 known classes:

| stage | seconds | share |
|---|---:|---:|
| `calculate_gaussian_correlation` (the antecedent screen) | 25.72 | **46.1%** |
| `to_simple_model` | 24.60 | **44.1%** |
| `create_gaussian_membership_dict` | 4.70 | 8.4% |
| `simple_gaussian_predict` | 0.82 | 1.5% |
| `take_top_features` | 0.00 | 0.0% |
| **total** | **55.84** | |

Two stages are 90% of it.

**The cost model reproduces the observed runtime.** The main table runs
12 held-out classes × 10 seeds = 120 folds; the θ-sweep re-runs the complement
rule alone at 7 θ values (× however many sweep seeds). At 55.8 s/fold:

```
main table    120 folds × 55.8 s  =  1h 52m
θ-sweep        84 folds × 55.8 s  =  1h 18m      (7 θ × 12 classes × 1 seed)
                        + Isolation Forest, one-class SVM (capped at 20k)
                                   ≈  3h 10m + baselines
observed                              3h 38m
```

Nothing anomalous is happening. The table is expensive because
`calculate_gaussian_correlation` is $\mathcal{O}(M \cdot K^2)$ — **82 features ×
55 class pairs = 4,510 pairwise distance computations per fold** — and it runs
once per fold, 204 times.

That quadratic-in-$K$ term is the one §4.3.4 already names as "the one place this
construction is not linear in anything", and RT-IOT2022's twelve classes are
exactly the axis it bites on. §4.3.4 says it "would bite first on a many-class
problem". It has.

---

## 2. The saving that is not available

`complement_rule` computes the ranking and then discards it:

```python
diffs = calculate_gaussian_correlation(X_tr, y_tr)
_, top_vars = take_top_features(diffs, top_n=len(X_tr.columns))   # keeps ALL 82
```

So 46% of the runtime produces an ordering that selects nothing. Skipping it
should be free.

It is not. Same fold, same 82 features, only the order differs:

| | agreement | anomaly-flag rate |
|---|---:|---:|
| ranked vs ranked *(control)* | **1.0000** | 0.4551 |
| column order vs column order *(control)* | **1.0000** | 0.1704 |
| **ranked vs column order** | **0.7063** | — |

**The controls come first and they matter.** Both arms are *exactly*
deterministic — the same input twice gives byte-identical predictions. So the
0.7063 is not run-to-run noise; it is the order.

29% of test rows change classification, and the rate at which the detector fires
at all moves by a factor of 2.7. The screen has to stay.

---

## 3. What that means, which is more than a performance note

The model is **not** what differs. Built in both orders:

| order | rules | antecedent terms |
|---|---:|---:|
| ranked | 11 | 902 |
| column | 11 | 902 |

Identical structure. So the difference is in **evaluation**, not in what was
fitted — the same 902 terms combined in a different sequence.

§4.3.5 presents the complement rule as an algebraic consequence:

$$\mu_{\text{anom}}(x) = 1 - S\big(c_1, \ldots, c_K\big)$$

A t-conorm is commutative and associative, so that expression is
order-independent by construction. **The implementation is not.** Something in
the evaluation chain accumulates in list order in a way the algebra does not
describe.

The leading explanation — and it is an explanation, not a measurement — is
floating-point accumulation order through an 82-term chain, amplified by the
knife-edge §4.3.5 itself derives: at $\theta = 0.99$ the clip makes
$\mu_{\text{anom}} > 0$ only when **every** class firing is below 0.01, so the
decision sits on a threshold where firing strengths are near zero and the last
bits decide. That would make the sensitivity worst exactly at the shipped
operating point, and milder across the swept band of §4.3.5's Table 4.6
(θ = 0.5–0.8), where the clip bites at 0.5 down to 0.2.

**I have not confirmed that mechanism**, and the size of the effect (0.4551 vs
0.1704) is larger than I would expect from float noise alone, so it should not be
written up as settled. What is settled is the observation: deterministic in each
order, materially different between them, with an identical fitted model.

> **Correction (C16, 2026-08-23): the mechanism above is refuted.** It is *not*
> floating-point accumulation order in the t-conorm. Holding one fold's boosted
> matrix fixed and reducing the **same columns** in four associations — the
> shipped column order, reversed, per-row-sorted, and a balanced pairwise tree —
> gives **identical predictions at every θ (agreement 1.0000)**. The t-conorm
> reduction is order-invariant, so §4.3.5's commutative-associative claim holds
> as written and needs no caveat. The order-sensitivity `profile_openset_cost`
> measured lives one stage earlier, in the model **build**: see §6.
> Reproduce with `reproduce/experiments/diagnose_openset_order.py`.

### Why it matters for the document

- **Table 4.7b's rates depend on the feature ordering the screen happens to
  produce.** Not on which features — on their order. Nothing in §4.3.5 says so.
- It gives the $\mathcal{O}(M \cdot K^2)$ screen a second job nobody assigned it:
  it is not only a selector, it is fixing an evaluation order the result depends
  on. That is why it cannot be cut.
- **It is a candidate explanation for Table 4.7's instability**, which §4.4
  already reports as ordering that "has changed three times across runs" with
  spreads "roughly five times the largest gap in the table". Order sensitivity of
  this size would do that.

---

## 4. The saving that *is* available

The θ-sweep re-runs the whole fold for each θ, but **θ enters only at
`to_simple_model(params)`**. The two stages ahead of it —
`calculate_gaussian_correlation` (46%) and `create_gaussian_membership_dict`
(8%) — take no θ and are recomputed identically seven times per fold.

Hoisting them out of the θ loop is result-identical by construction (the same
`memb` object, reused) and should cut the sweep by **~54%**, about **40 minutes**
of the 3h38m.

Not done here, deliberately: it changes the generator's code path, and this pass
is also re-running the suite for a pin bump. Mixing a performance change into a
run whose purpose is attributing numeric drift is how the two become hard to tell
apart. It wants its own change and its own before/after.

Two cheaper knobs already exist and are worth stating in one place:
`REPRO_THETA_SWEEP_SEEDS` bounds the sweep's seed count (this pass used one seed,
which is what made the sweep 1h18m instead of ~13h at ten), and
`REPRO_OCSVM_TRAIN_CAP` bounds the one-class SVM's $\mathcal{O}(n^2)$ fit.

---

## 4b. Deeper profile: neither hot stage is slow because of its algorithm

Both are slow for the same reason — **scalar work done through vectorized-array
APIs, inside a Python loop.** Neither is the mathematics anyone would point at.

### The screen (46%) is ~100% pandas label masking

`cProfile` on one fold's `calculate_gaussian_correlation`, 27.3 s total:

| | tottime | calls |
|---|---:|---:|
| `pandas ... missing.py:_isna_string_dtype` | **16.17 s** | 9,020 |
| `pandas ... string_.py:_cmp_method` (cumulative 23.54 s) | 2.02 s | 9,020 |
| `stats_numba.wasserstein_distance` (cumulative) | **0.58 s** | 4,510 |

`_differentiation_score` masks with `data[y == unique_labels[ij]]` for every
(feature, class-pair), so the same $K$ boolean masks are recomputed $M$ times —
**9,020 comparisons over 92,293 rows** where 11 would do. The distance
computations everyone would suspect total **0.58 s**.

The cost is entirely the label **dtype**. One comparison, timed directly:

| `y` dtype | one comparison | × 9,020 |
|---|---:|---:|
| `str` (what the harness passes) | 2.91 ms | **26.2 s** |
| `object` | 2.59 ms | 23.4 s |
| **`category`** | **0.02 ms** | **0.1 s** |

That predicted 26.2 s against 26.22 s measured end-to-end, so the screen *is* the
masking, to within noise.

**Converting the labels to `category` is bit-identical and 11.5× faster:**

```
str      : 26.33 s   (82 scores)
category :  2.29 s   speedup 11.5x
max |score diff| : 0.000e+00
ranking identical: True
```

The ranking being identical matters for more than correctness: it means this
does **not** disturb the order dependence of §3 above.

End to end on one fold:

| | screen | memb | to_model | predict | total |
|---|---:|---:|---:|---:|---:|
| `str` | 26.09 | 4.39 | 23.65 | 0.84 | **54.97 s** |
| `category` | 2.29 | 1.97 | 23.42 | 0.86 | **28.54 s** |

**1.93× per fold, predictions identical, order identical.** It also halves
`create_gaussian_membership_dict`, which pays the same masking cost.

### `to_simple_model` (44%) is 2.9M scalar `np.isclose` calls

`gauss_data.py:_is_close` is called **2,800,161** times and calls `np.isclose`
**2,921,252** times — the membership-function dedup of §4.3.1 (`rtol=1e-2,
atol=1e-3`), as an $O(T^2)$ scan over 902 antecedent terms with a scalar
`np.isclose` per attribute per pair.

Appendix A.3 records that cProfile's per-call charge once turned a 9.8% speedup
into a published 19%, so this one gets a wall clock rather than a profile:

```
np.isclose(scalar)   8.06 us/call  ->  2.9M calls = 23.4 s
plain-python equiv   0.12 us/call  ->  2.9M calls =  0.3 s   (70x)
```

23.4 s against the 23.65 s actually measured for `to_simple_model`. The profile's
story survives the wall clock: **the stage is its `isclose` calls and nothing
else.** `np.isclose` on two Python floats allocates arrays, enters an errstate
context and runs two ufunc reductions, for a comparison that is one subtraction.

**Not proposed as a drop-in yet.** `np.isclose` is
`|a-b| <= atol + rtol*|b|`, which the plain expression reproduces exactly — but
it also has NaN and inf semantics the plain form does not, and per §3 this code
path is dedup, where §4.3.1 already shows the *order* of comparison decides which
membership function survives. A change here has to be shown bit-identical on real
data before it is worth 70×, not argued from the formula.

### What the two together would be worth

| | per fold | `table_4_4_openset` |
|---|---:|---:|
| today | 55.8 s | 3h 38m |
| screen as `category` | 28.5 s | **~2h 06m** |
| + dedup vectorised *(unverified)* | ~5 s | **~20 m** |


## 5. Owed

1. ~~**Pin down the order mechanism**~~ **Done (C16, §6).** It is the dedup
   representative selection, not the t-conorm.
2. ~~**State the order dependence in §4.3.5**~~ **Withdrawn.** §4.3.5's rule is
   order-invariant as measured; the caveat that belongs anywhere belongs to
   §4.3.1's dedup (§6), and it is latent — the shipped table uses one order.
3. ~~**Hoist the θ-independent work** out of the sweep~~ **Done (C17).** Not just
   the screen+memb the estimate below assumed: the whole model build and the
   class firing are θ-independent (θ enters only at the anomaly step, not at
   `to_simple_model`). `simple_gaussian_predict_sweep` (tribble-fis #176) runs
   the firing once — **5.72× on the sweep, bit-identical**, one RT-IOT2022 fold,
   six θ, two seeds.
4. Re-read Table 4.7's run-to-run instability against this. It may not be noise.


## 6. C16 resolved (2026-08-23): the divergence is in the model build, not the rule

Two measurements, both single-fold, both cheap, run in this order:

**(a) The t-conorm reduction is order-invariant.** Fix one fold's boosted matrix;
reduce the same columns four ways.

| θ | column *(ships)* | reversed | sorted | pairwise |
|---|---:|---:|---:|---:|
| 0.50 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.90 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 0.99 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

Identical predictions at every θ. Float accumulation order (the §3 hypothesis)
is refuted; §4.3.5's algebra holds.

**(b) The divergence is upstream, in the build.** Build the model with the
*ranked* feature order and again with plain *column* order, then compare the
class-firing matrix on the same test rows, label-aligned:

```
ranked  : 11 rules, 2367 terms
column  : 11 rules, 2367 terms
max |class_firing_ranked − class_firing_column| : 8.643e-01
rows whose class firing differs at all         : 98.49%
```

0.86 is not float noise; it is a different fitted model. Same rule count, same
term count, materially different firing. The cause is the $O(T^2)$ cross-feature
dedup in `to_simple_model` (§4.3.1): it merges membership functions within
`rtol=1e-2, atol=1e-3` and **the first occurrence wins**, so the surviving
representative — hence the (μ, σ) actually evaluated on a given column — depends
on the order features enter the list. Reorder the features, and a clause that
was its own representative now points at a near-but-not-equal MF from another
feature.

**What this means for the document.** The order-sensitivity is real but latent:
the shipped table always uses the screen's ranked order, so it is deterministic
and reproducible. §4.3.5 needs no caveat. If order-*invariance* is wanted, the
fix is in §4.3.1's dedup — pick a canonical representative (e.g. smallest
(μ, σ), or the mean of the merged group) instead of first-seen. That changes
results, so it is the author's call, not a reproduction fix, and is not made
here. Reproduce both with `reproduce/experiments/diagnose_openset_order.py`.
