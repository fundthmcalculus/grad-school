# Fuzzy anomaly detection in small language models

**Branch with the full work:** [`exp/fuzzy-lm-2`](https://github.com/fundthmcalculus/grad-school/tree/exp/fuzzy-lm-2)
· code and data in `research/fuzzy-lm-anomaly/`
· full write-up in [`FINDINGS.md`](https://github.com/fundthmcalculus/grad-school/blob/exp/fuzzy-lm-2/research/fuzzy-lm-anomaly/FINDINGS.md) (30 sections)
· one-page version in [`SUMMARY.md`](https://github.com/fundthmcalculus/grad-school/blob/exp/fuzzy-lm-2/research/fuzzy-lm-anomaly/SUMMARY.md)

**Question:** can the tribble "none of the above" anomaly rule (Ch 4.3.5) flag
hallucinated output from a frozen small language model?

**Scale:** ~69,000 generations across six frozen models (SmolLM2 135M/360M/1.7B,
Qwen2.5-0.5B, Gemma3-270m, LFM2.5-350M) on one 12 GB card. No fine-tuning — the
models were only read.

**Answer: no.** Under an equal-budget comparison the fuzzy rule finishes last
against every rival on every model. It also loses at zero budget. There is no
fair comparison in which it wins.

---

## The result that matters

Five separate apparent successes were each destroyed by a control. Each one, on
its own, produced a number that looked publishable.

| confound | what it alone achieves | remedy |
|---|---|---|
| answer length | `n_tokens` alone: **AUROC 0.843** | exact matching on token count |
| prompt family / style | ~0.9 for a detector reading style, not fabrication | real-entity twins in identical surface forms |
| confidence | mean entropy: **0.964** on a templated set | matching on entropy quartile |
| unequal search budget | flips a comparison's sign (+0.014 → −0.019) | equal search for every arm |
| baseline under-specification | `ent_max` beats the default `ent_mean` by **+0.116** (61/66 cells) | search the baseline family too |

The last two were found in our own protocol, after the first three had already
taught us to look for exactly this kind of thing.

## The one practical finding

**Use maximum per-token entropy, not mean.** Over 66 cells, `ent_max` scores
**0.883** against `ent_mean`'s **0.767** — better in 61 of 66 — and it beat every
learned detector tested, at a fraction of the cost:

| detector | search budget | AUROC | FPR@95TPR |
|---|---|---|---|
| **single statistic (`ent_max`)** | 38 | **0.870** | **0.492** |
| Mahalanobis · 19 stats | 120 | 0.776 | 0.714 |
| Isolation forest | 96 | 0.768 | 0.694 |
| One-class SVM | 100 | 0.767 | 0.749 |
| tribble FIS | 120 | 0.750 | 0.785 |

This is a one-line change for anyone working on hallucination detection.

## Findings that stand for the dissertation

Equal budgets across arms; unaffected by the audits above.

* **The membership family dominates every other FIS knob** — Gaussian over
  trapezoid, **+0.262 AUROC, 8/8 seeds** — an order of magnitude beyond the
  ranking metric (≤0.015) or the norm pair (+0.002).
* **θ is provably rank-invariant**: `μ_anom = 1 − max(μ_k) − θ` is a constant
  shift, so it sets the operating point and cannot change separability.
  `plot_anomaly_threshold_sweep` invites the opposite reading.
* **Nilpotent norm families (Łukasiewicz, drastic, nilpotent minimum) are
  unusable as the outer t-conorm** — they saturate to 1 under aggregation and
  drive the complement to a constant 0.
* **Five defects found and fixed in `tribble-fis`** (issues #22–#25, #30). The
  consequential one: `norm_conorm` was silently ignored in the anomaly
  aggregation, so `beth-anomaly.py`'s Hamacher setting never applied — the BETH
  numbers should be re-run.

## Where it goes

* [`papers/hallucination-detection-confounds.md`](https://github.com/fundthmcalculus/grad-school/blob/exp/fuzzy-lm-2/papers/hallucination-detection-confounds.md)
  — the viable write-up. Five confounds, each measured, each with a cheap remedy.
  Blocking: measure length/style imbalance in a public benchmark.
* [`papers/fuzzy-anomaly-rule-slm.md`](https://github.com/fundthmcalculus/grad-school/blob/exp/fuzzy-lm-2/papers/fuzzy-anomaly-rule-slm.md)
  — **closed.** The equal-budget comparison was run and the claim does not
  survive it.

## The method that worked

Matching over modelling. Equal budgets stated up front. And checking every
correlation for the artefact that would produce it by construction — one result
survived only because a split-half check confirmed it wasn't the
`corr(X, Y−X)` bias, and another died because the baseline had never been
questioned.
