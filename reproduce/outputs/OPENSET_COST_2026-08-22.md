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

## 5. Owed

1. **Pin down the order mechanism** — instrument the t-conorm chain and confirm
   (or refute) that accumulation order explains a 2.7× swing in flag rate. If it
   does not, something less benign is going on.
2. **State the order dependence in §4.3.5**, or remove it. If the rule is meant
   to be the algebraic consequence the section describes, evaluation should be
   order-invariant — sorting the terms, or accumulating in a fixed canonical
   order, would make it so and would cost nothing.
3. **Hoist the θ-independent work** out of the sweep, as its own change.
4. Re-read Table 4.7's run-to-run instability against this. It may not be noise.
