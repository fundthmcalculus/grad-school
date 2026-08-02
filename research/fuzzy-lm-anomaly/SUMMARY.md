# Fuzzy anomaly detection in small language models — what we learned

Short version of `FINDINGS.md` (29 sections). ~69,000 generations across six
frozen models on one 12 GB card. Nothing was fine-tuned; the models were only
read.

---

## The one-line result

**Four apparent successes were each destroyed by a control, and the controls are
the contribution.** What survived is narrower and more useful than what we set out
to find: the fuzzy anomaly rule does not beat confidence-based detection in
general, but *when* it helps is strongly and predictably governed by how badly
confidence is doing.

## Four ways to manufacture a hallucination detector

Each of these, alone, produced a result that looked publishable.

| confound | what it alone achieves | remedy |
|---|---|---|
| **answer length** | `n_tokens` alone: **AUROC 0.843** | exact matching on token count |
| **prompt family / style** | ~0.9 for a detector reading style, not fabrication | real-entity twins in identical surface forms |
| **confidence** | mean entropy: **0.964** on a templated set | matching on entropy quartile |
| **unequal search budget** | flips a comparison's sign (+0.014 → −0.019) | equal search for all arms, or fixed configs |

The fourth was found in our own protocol — after the first three had already
taught us to look. A 120-candidate configuration search scored against labelled
validation positives, with none for the baselines, produced a stable,
seed-consistent, mechanistically plausible advantage that vanished at fixed
configurations.

**Matching is a cheap, general remedy** for the first three and needs no change to
the detector under test. The fourth needs only that search budget be stated and
equalised.

## What was retracted

* §9 — 0.906 ± 0.017, 10/10 seeds, p = 0.002, pre-registered. Killed by the
  template control: at chance (0.529) against a different fabrication family.
* §19 — the false-premise "niche". Killed by long-form real-subject twins.
* §21/23/24 — the fuzzy rule beating full-covariance Mahalanobis. Killed by the
  search-budget audit.

## What survived

**The complementarity result (§27, §29).** Across 44 cells (models × templates),
fixed configurations for both detectors:

* `corr(entropy AUROC, FIS − entropy)` = **−0.78**, p = 5e-10, verified against
  the shared-term artefact by split-half estimation on disjoint seeds.
* Crossover at **entropy AUROC ≈ 0.61**. Below it the fuzzy rule leads, by up to
  **+0.265** in near-chance-entropy cells.
* The fuzzy rule does *not* win overall — 0.628 vs entropy's 0.743, ahead in 10 of
  44 cells.

**The regime is not set by model size (§29).** Entropy improves monotonically
within the SmolLM2 family (0.713 → 0.838 → 0.909 for 135M/360M/1.7B), but
Gemma3-270m sits at 0.546 while SmolLM2-135M — half its size — reaches 0.713. It
is a **weak-entropy-model** technique, not a small-model one.

**Detector choice is decidable (§28, §29).** Twenty labelled examples buy +0.0104
at 91% agreement with an oracle; 100 buy +0.0134 at 99.5%. Label-free, entropy's
own reliability is predictable from the known-good split alone (held-out
r = **+0.689** across six models) — but not sharply enough at the decision
boundary to act on. Predictable, not yet actionable.

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
* **`papers/fuzzy-anomaly-rule-slm.md`** — ON HOLD pending an equal-budget
  comparison.
* **Ch 4 material regardless:** the θ result, the nilpotent-norm result, the
  membership-family dominance, and the library fixes.

## The method that actually worked

Matching over modelling; equal budgets stated up front; and checking every
correlation for the artefact that would produce it by construction. Three
apparent findings died to the first, one to the second, and one nearly survived
the third (`corr(X, Y−X)` is negative even for independent variables — the
split-half check is what made §27 trustworthy).
