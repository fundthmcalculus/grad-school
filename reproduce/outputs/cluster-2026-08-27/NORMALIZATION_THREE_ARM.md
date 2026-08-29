# The normalization arm: what "log+std" actually measured, and what z-score costs

_Measured 2026-08-03. Ten seeds (`common.SEEDS`), shared splits, UCI Concrete
(N=1030, M=8). `tribble-fis` `a385a1a`. Run of record
`reproduce/outputs/norm-three-arm-a385a1a/`; the two-arm predecessor it must
agree with is `reproduce/outputs/full-14900hx-r2/`._

**This file reports a measurement and states a decision that is now waiting. It
does not take the decision, and no prose label has been changed.**

---

## 1. The concession

`gauss_math.standard_transform` — the function behind **every** "log+std",
"log + standardized" and "normalized" number in this document — applied

```
(X - X.min()) / (X.max() - X.min())
```

That is **min-max scaling to [0,1]**, not z-score standardization. The name was
wrong, and it was wrong in the direction that flatters the text: the prose says
"standardized" and a reader of a fuzzy-systems dissertation will read that as
μ=0/σ=1.

So: **log + z-score had never been measured at all.** `tribble-fis` PR #67
(`a385a1a`) split the helper into two honestly-named transformers —
`UnitScalar` (min-max) and `StandardScalar` (z-score) — and deleted the
originals, which is what forced the question.

`UnitScalar` is the behaviour-preserving successor. That is not an assumption:
the deleted implementations were reimplemented from `d0efefc` and compared to the
new scalers on the real data, giving **`max|diff| = 0.0` exactly** on all four
call shapes in use, with identical log-feature detection (`['Slag', 'Age']`).

---

## 2. The measurement

Ten seeds, shared splits, `log_dynamic_range=2` (matching the old
`min_dynamic_range=2`; note `UnitScalar` defaults to 3.0, which would drop `Slag`).
Values below are read from
`norm-three-arm-a385a1a/table_hyperparam_normalization.csv`.

| Model | Hyperparameters | raw | **log + min-max** *(what "log+std" meant)* | **log + z-score** *(new)* | Δ mm−raw | Δ zs−raw | **Δ zs−mm** |
|---|---|---|---|---|---|---|---|
| CART (reference) | sklearn default | 0.825 ± 0.047 | 0.826 ± 0.047 | 0.826 ± 0.046 | +0.001 | +0.001 | **-0.000** |
| Random Forest (reference) | sklearn default | 0.909 ± 0.018 | 0.909 ± 0.019 | 0.909 ± 0.018 | +0.000 | +0.000 | **-0.000** |
| flat MoG-TSK 1st | pipeline default | 0.646 ± 0.039 | 0.772 ± 0.034 | 0.087 ± 0.089 | +0.126 | -0.559 | **-0.685** |
| flat MoG-TSK 2nd | pipeline default | 0.779 ± 0.036 | 0.824 ± 0.043 | 0.781 ± 0.045 | +0.044 | +0.001 | **-0.043** |
| flat MoG-TSK full-2nd | pipeline default | 0.790 ± 0.054 | 0.859 ± 0.039 | 0.819 ± 0.058 | +0.069 | +0.029 | **-0.040** |
| fuzzy tree | demo-tuned | 0.712 ± 0.030 | 0.740 ± 0.051 | 0.740 ± 0.051 | +0.028 | +0.028 | **-0.000** |
| fuzzy tree | library default | 0.583 ± 0.067 | 0.689 ± 0.056 | 0.691 ± 0.055 | +0.106 | +0.108 | **+0.002** |
| mixture of experts | demo-tuned | 0.768 ± 0.029 | 0.834 ± 0.025 | 0.706 ± 0.024 | +0.066 | -0.063 | **-0.128** |
| mixture of experts | library default | 0.686 ± 0.060 | 0.763 ± 0.057 | 0.730 ± 0.067 | +0.077 | +0.044 | **-0.034** |

RMSE, in MPa, for the rows that move most: flat MoG-TSK 1st goes
**7.803 ± 0.734 → 15.610 ± 0.732**; mixture of experts (demo-tuned)
**6.661 ± 0.570 → 8.873 ± 0.403**.

### 2.1 The control holds — so the movements are real

CART, Random Forest and the fuzzy tree split on **rank**. Min-max and z-score are
both strictly monotone per feature, so these rows are *provably* invariant
between the two normalized arms and must not move. They do not:

| control row | Δ z-score − min-max | own seed spread |
|---|---|---|
| CART | -0.000 | ±0.047 |
| Random Forest | -0.000 | ±0.018 |
| fuzzy tree (demo-tuned) | -0.000 | ±0.051 |
| fuzzy tree (library default) | +0.002 | ±0.056 |

Worst |Δ| = **0.002**, against seed spreads of 0.018–0.056. Independently,
column-wise Spearman correlation between the two scalers' outputs is
**1.000000000000** on every feature. The plumbing is sound; the fuzzy models'
movements are the world, not the wiring.

### 2.2 The pre-existing two arms did not move

All 45 pre-existing cells (9 rows × {raw, log+min-max, Δ, RMSE raw, RMSE
log+std}) are **byte-identical** to `full-14900hx-r2/`, checked programmatically.
Renaming the column changed no number.

---

## 3. The awkward part: z-score is *worse*, and at 1st order it is catastrophic

Min-max is **best or tied in 8 of 9 rows** — uniquely best in 5 (the three flat
MoG orders and both mixture-of-experts settings), tied in 3 (CART, Random Forest,
fuzzy tree demo-tuned, all of which are the monotone-invariant controls and so
*must* tie). The single row where z-score is ahead is `fuzzy tree / library
default`, by **+0.002** against its own ±0.056 seed spread — another invariance
control, i.e. noise, not a win.

The 1st-order flat MoG under z-score scores **0.087**, which is *worse than raw
features* (0.646). The transform the prose has been claiming would not merely
have been a different choice — on the headline model of Chapter 4 it would have
**destroyed** the result that chapter is built on.

Two innocent explanations were ruled out before concluding that:

- **Not a ridge-scale artifact.** `l2_reg` swept 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 0.
  The 1st-order gap moves by 0.001 across the whole sweep (−0.790 → −0.789).
  Flat.
- **Not the BIC structure choice.** `n_gaussians=-1` picks membership counts by
  BIC, which is *not* scale-invariant, and the rule bases do differ slightly
  (`Age`/label 0: 3 → 4 functions; 72 vs 73 total, 1 of 24 (feature, label)
  cells). But pinning `n_gaussians` so both arms get an **identical** rule base
  leaves the collapse intact: **−0.407** at n=2, **−0.524** at n=3, **−0.634** at
  n=4. The transform is the cause.

The mechanism is consistent with the pipeline's own stated premises. Gaussian
membership functions and the `[0,1]`-pinned extreme bucket means (Ch 4 §4.3) both
assume a **bounded, non-negative** input domain; centering features on zero
breaks the 1st-order consequent's affine fit. It underfits on **train** as well
(MSE 0.030 vs 0.009 at seed 0), so this is a fitting failure, not extrapolation.
Raising the consequent order recovers most of it (−0.685 → −0.043 → −0.040),
which is what one expects if the 1st-order affine term is the thing that broke.

**Read plainly: the mislabelling was lucky. The code did the right thing under
the wrong name.**

---

## 4. The decision that is now waiting (author's, not the harness's)

The three-arm table exists to make this choosable. Three directions, with costs:

### Option A — relabel to min-max; keep it the default
Say "log + min-max to [0,1]" wherever the text says "standardized"/"log+std".

- **Costs: no numbers.** Every value in Chapters 4 and 6 stands as printed.
- **Gains coherence.** Two sentences that currently read as loose become exactly
  right: §6.3.2's and §4.3's *"cement ≥ 0.42 after standardization"* is a
  legitimate min-max value and an impossible z-score one (z-scores here run
  −2.80 … +4.35). §4.3's "target standardized to [0,1]" and the pinned bucket
  means at 0.0/1.0 already say min-max explicitly, so the document becomes
  self-consistent rather than half-consistent.
- **Cost: one terminology collision to police.** Chapter 5 uses "the minimax
  transform" throughout in the iVAT bottleneck-ultrametric sense, which is
  unrelated. Prefer "min-max scaling to [0,1]" or "unit scaling", never bare
  "minmax", in Ch 4/6.
- **Honesty cost:** requires saying somewhere that the earlier name was wrong.
  Cheap, and §4.3 is the natural place.

### Option B — switch the default to z-score (what the text has been claiming)
- **Costs: essentially the whole of Chapters 4 and 6.** Table 4.1's headline
  finding falls from +0.126 to −0.559 on the 1st-order model; Table 4.5's and
  Table 6.1's Concrete columns all re-quote; §4.3's "the transform is worth
  nearly thirteen points of R²" inverts sign.
- **Also breaks the argued-for machinery:** the extreme-bucket-mean pin (§4.3),
  which exists so extreme rules read as values the target can actually take,
  assumes a `[0,1]` target and is the fix landed in `fix/pin-extreme-bucket-means`.
- **Recommendation: do not.** There is no measurement supporting it.

### Option C — report both arms side by side
Keep min-max as the default, print the z-score column too.

- **Costs: one wider table and roughly a paragraph** in §4.3.
- **Gains the strongest defensive position.** It converts an embarrassment into a
  result: "normalization helps" becomes the sharper and more interesting
  *"bounded normalization helps; centered normalization does not, and the
  bounded-input assumption is load-bearing rather than incidental."* That
  sentence is a better answer to a committee question than either A or B, and it
  is the one the data actually supports. It also documents the CART/RF invariance
  control across three levels instead of two.
- **Cost: obliges a sentence explaining *why*** (§3's mechanism), which is
  currently evidenced but not proven.

**Suggested (not decided): C for §4.3 and Table 4.1, A everywhere else** — the
one place the distinction is load-bearing carries all three arms; every other
mention just gets the honest name. This is the author's call.

---

## 5. What was *not* found

No prose statement is **outright false** — only mislabelled. A full sweep of
`research/proposal-defense/prose/*.md` for `z-score`, `zero mean`, `unit
variance`, `mu=0`, `sigma=1`, "subtract the mean", "divide by the standard
deviation" and eleven related phrasings returns **nothing**. The prose never
spells out the transform's arithmetic; it only ever calls it "standardized". Every
occurrence of "standard deviation" in the prose refers to seed spread in a results
table, never to a feature transform.

Two places are already *correct*, and they corroborate min-max:
`prose/04-fast-fis-synthesis-mog.md:82` ("With a target standardized to $[0,1]$…")
and `:84` ("the bucket means come back as the intended $0.0$ and $1.0$").

One consequence: `build/proposal-combined.md` is a stale generated
concatenation and will need a rebuild, not a hand-edit, whichever option is taken.

---

## 6. Reproducing this

```bash
export PYTHONIOENCODING=utf-8
bash reproduce/run_all_tables.sh <label> table_hyperparam_normalization
py -3 reproduce/compare_runs.py full-14900hx-r2 <label>
```

The migration's behaviour-preservation evidence is in commit `59390b7`; the
third arm and the two ruled-out explanations are in `dc1fb2c`.
