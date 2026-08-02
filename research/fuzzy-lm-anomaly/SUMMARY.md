# Fuzzy anomaly detection in small language models — what we learned

Short version of `FINDINGS.md` (30 sections). ~69,000 generations across six
frozen models on one 12 GB card. Nothing was fine-tuned; the models were only
read.

---

## The one-line result

**Five apparent successes were each destroyed by a control, and the controls are
the contribution.** Under an equal-budget comparison the fuzzy anomaly rule loses
to every standard alternative on every model (§30); the durable output of the
study is the five confounds, their remedies, and one strong practical finding —
**max-token entropy is a far better hallucination-detection baseline than the
mean-entropy / perplexity defaults** (+0.116 AUROC, better in 61 of 66 cells).

## Five ways to manufacture a hallucination detector

Each of these, alone, produced a result that looked publishable.

| confound | what it alone achieves | remedy |
|---|---|---|
| **answer length** | `n_tokens` alone: **AUROC 0.843** | exact matching on token count |
| **prompt family / style** | ~0.9 for a detector reading style, not fabrication | real-entity twins in identical surface forms |
| **confidence** | mean entropy: **0.964** on a templated set | matching on entropy quartile |
| **unequal search budget** | flips a comparison's sign (+0.014 → −0.019) | equal search for all arms, or fixed configs |
| **baseline under-specification** | a weak default baseline invents a niche: `ent_max` beats `ent_mean` by **+0.116** (61/66 cells) | search the baseline family too |

The last two were found in our own protocol, after the first three had already
taught us to look. The fourth: a 120-candidate configuration search scored against
labelled validation positives, with none for the baselines. The fifth is sharper,
because the baseline was never the object of suspicion — we picked mean entropy,
held it fixed for twenty-odd sections, and it manufactured a "weak-entropy regime"
that vanishes under a 38-candidate search over the statistic family.

**Matching is a cheap, general remedy** for the first three. The last two need only
that the search budget — *and the baseline family* — be stated and equalised.

## What was retracted

* §9 — 0.906 ± 0.017, 10/10 seeds, p = 0.002, pre-registered. Killed by the
  template control: at chance (0.529) against a different fabrication family.
* §19 — the false-premise "niche". Killed by long-form real-subject twins.
* §21/23/24 — the fuzzy rule beating full-covariance Mahalanobis. Killed by the
  search-budget audit (§26).
* §27/28/29 — the complementarity result and everything built on it. Killed by
  re-baselining mean entropy to max entropy (§30).

## What survived

**⚠ §27's complementarity result was itself retracted by §30** — it was measured
against *mean* entropy; re-baselined on `ent_max` the weak-entropy regime shrinks
from 12/66 cells to 1/66 and Gemma3-270m stops being a weak-entropy model. §28's
switching and §29's scaling conclusions inherit that correction. Kept below for
the record.

**The complementarity result (§27, §29) — RETRACTED.** Across 44 cells (models × templates),
fixed configurations for both detectors:

* `corr(entropy AUROC, FIS − entropy)` = **−0.78**, p = 5e-10, verified against
  the shared-term artefact by split-half estimation on disjoint seeds.
* Crossover at **entropy AUROC ≈ 0.61**. Below it the fuzzy rule leads, by up to
  **+0.265** in near-chance-entropy cells.
* The fuzzy rule does *not* win overall — 0.628 vs entropy's 0.743, ahead in 10 of
  44 cells.

(§29's scaling conclusion and §28's switching work inherit the §30 correction:
they were measured against mean entropy. The observation that entropy improves
monotonically within the SmolLM2 family — 0.713 → 0.838 → 0.909, and 0.866 →
0.946 → 0.962 under `ent_max` — is unaffected and stands.)

**The equal-budget result (§30), which does stand.** Every family given a
comparable search, same validation positives, same criterion:

| family | budget | AUROC | FPR@95 |
|---|---|---|---|
| **single statistic** | 38 | **0.870** | **0.492** |
| Mahalanobis | 120 | 0.776 | 0.714 |
| IsolationForest | 96 | 0.768 | 0.694 |
| OneClassSVM | 100 | 0.767 | 0.749 |
| **FIS** | 120 | **0.750** | 0.785 |

The fuzzy rule finishes last against every rival on every model (paired
p ≤ 0.001), despite the largest budget in the comparison.

**Negative results that are worth their cost:**

* Hidden-state geometry never survived the controls; the output-distribution
  statistics did.
* FPR@95TPR is a structural weakness of this rule class — a cascade with
  abstention and an FPR-targeted selection objective both failed to move it.
* Zero-parameter score fusion gains nothing: the detectors are complementary
  *across* regimes, not *within* a cell.

## Findings about the fuzzy machinery itself

These had equal budgets across arms and are unaffected by the §26 audit:

* **The membership family dominates every other knob** — Gaussian beats trapezoid
  by **+0.262**, 8/8 seeds. An order of magnitude larger than the metric (≤0.015)
  or the norm pair (+0.002). Now declared and *asserted* in `fis_config.py`.
* **θ is provably rank-invariant**: `μ_anom = 1 − max(μ_k) − θ` is a constant
  shift, so it sets the operating point and cannot change separability.
* **Nilpotent norm families are unusable as the outer t-conorm** — they saturate
  to 1 under aggregation and drive the complement to a constant 0.
* Five defects found and fixed in `tribble-fis` (#22–#25, #30). The consequential
  one: `norm_conorm` was silently ignored in the anomaly aggregation, so
  `beth-anomaly.py`'s Hamacher setting never applied.

## Where it goes

* **`papers/hallucination-detection-confounds.md`** — the viable write-up. Four
  confounds, each measured, each with a cheap remedy, one caught in the authors'
  own protocol. Blocking: measure length/style imbalance in a public benchmark.
* **`papers/fuzzy-anomaly-rule-slm.md`** — **closed**. The equal-budget
  comparison was run (§30) and the claim does not survive it.
* **Ch 4 material regardless:** the θ result, the nilpotent-norm result, the
  membership-family dominance, and the library fixes.

## The method that actually worked

Matching over modelling; equal budgets stated up front; and checking every
correlation for the artefact that would produce it by construction. Three
apparent findings died to the first, one to the second, and one nearly survived
the third (`corr(X, Y−X)` is negative even for independent variables — the
split-half check is what made §27 trustworthy).
