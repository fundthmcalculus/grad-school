# Correction: what used TribbleFIS and what did not

A review of the code found that the anomaly-detection work (`FINDINGS_DETECTION.md`
Parts 2–5) did **not** use the `tribblefis` library. It used a reimplementation,
`FISAnomaly` in `fmri_detect.py`, and that reimplementation is **not faithful**.
This corrects the record.

## What actually used the library

| work | real `tribblefis`? |
|---|---|
| Drafting bakeoff (`exp2`), bottleneck (`exp7`), IT2 fix + learned-FoU | **Yes** — `TribbleRegressor`, `IntervalType2FuzzyRegressor`, `apply_tsk_consequents`. PR #103 is a genuine fix to the library. |
| Anomaly / injection detection (Parts 2–5) | **No** — `FISAnomaly`, a hand-written per-feature-Gaussian + product-t-norm score. |

## Why the reimplementation was not a valid stand-in

`FISAnomaly` fits an independent Gaussian per feature and sums the squared
z-scores. On **whitened** features that is, by construction, the Mahalanobis
distance — so the reported "FIS-whitened = Mahalanobis exactly" was near-
tautological and says nothing about `tribblefis`. **That claim is withdrawn.**

More importantly, `FISAnomaly` omits everything that makes TribbleFIS the method:
differentiation-score feature selection, per-class Gaussian *mixtures*, TSK rule
firing under a chosen norm family, and the real `AnomalyParameters` rule
(`1 − max class membership`). Run on the same weak feature set, the two give
different numbers — real `TribbleClassifier` 0.627 vs `FISAnomaly` 0.597 — so
they are not interchangeable.

## The real library, run properly

Given the full 279 per-layer-PCA feature set and a search over its own knobs
(`top_n`, `n_gaussians`, `norm_conorm`), the genuine `TribbleClassifier`, in a
supervised two-class setting (benign + half the injections, held-out test):

| detector | within-length AUROC |
|---|---|
| **real `TribbleClassifier` (best of its knobs)** | **0.852 ± 0.019** |
| Mahalanobis (same features) | 0.835 ± 0.009 |

So the real library is **competitive and slightly ahead** here — a genuinely
positive result, and the honest one. It also yields readable rules, e.g. on
per-layer-norm features it selects layer 1, layer 11, layer 30 and learns:

> injection ⟸ final-layer (30) activation norm HIGH (μ≈31 vs benign ≈25)
> AND early-layer (1) norm LOW (μ≈25 vs benign ≈28)

which matches the Part 4 finding that injections perturb the deep layers.

## The framing consequence (a real decision, not a detail)

`tribblefis`'s anomaly rule needs class structure — its feature selection
requires ≥2 classes — so the library's natural mode here is **supervised /
few-shot** (a handful of known attacks), not the pure **one-class / no-attack-
examples** setting Parts 3–4 advertised. That selling point belonged to the
reimplementation, not the library.

Two honest paths, and they are different contributions:

* **(A) Supervised/few-shot with the real library.** The genuine TribbleFIS,
  competitive with Mahalanobis (0.852 vs 0.835), producing interpretable rules.
  Cost: drop the "no attack examples" claim.
* **(B) One-class, clearly relabelled.** Keep the no-attack detector but call it
  "an FIS-style one-class rule inspired by the tribble anomaly mechanism," not
  the library, and stop reporting the tautological Mahalanobis equivalence.

Parts 3–5's numbers stand as measurements; what changes is the label on the
detector and the framing of the claim. The multi-model, confound-control,
operating-point, and refinement findings are unaffected — they compared a
one-class score against surface/length baselines and would read the same with
the real library's supervised score substituted, which is the next run to do.
