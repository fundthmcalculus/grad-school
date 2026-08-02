# Paper plan (TODO) — Confounds in hallucination-detection benchmarks

**Status:** not started. This is a plan of record, not a draft.
**Source of results:** `research/fuzzy-lm-anomaly/` (see `FINDINGS.md`; all numbers
below are already produced and committed, with seeds and controls).
**Provisional title:** *Five Ways to Manufacture a Hallucination Detector — and
the Controls That Remove Them*

---

## 1. Why this is a paper

While testing whether a fuzzy inference system could flag hallucination from a
frozen 360M language model, **five separate apparent successes were each
destroyed by a control** — and each confound, on its own, produced a result that
would have looked publishable.

| confound | what it alone achieves | how it was removed |
|---|---|---|
| **answer length** | `n_tokens` alone: **AUROC 0.843** | exact matching on token count |
| **prompt family / style** | ~0.9 for a detector reading style, not fabrication | real-entity twins in identical surface forms |
| **confidence** | mean entropy: **0.964** on a templated set | matching on entropy quartile |
| **unequal search budget** | flips a detector comparison's sign (+0.014 → −0.019) | equal search for all, or fixed configs |
| **baseline under-specification** | `ent_max` beats the default `ent_mean` by **+0.116** (61/66 cells), erasing a "regime" we had reported | search the baseline family too |

The fifth is the sharpest example, because the baseline was never the object of
suspicion: we picked mean entropy, held it fixed for twenty-odd sections, and it
manufactured a "weak-entropy regime" that vanishes under a 38-candidate search
over the statistic family (12/66 cells below the crossover → 1/66). It cost three
sections of results.

The fourth was found in our own protocol too: giving one detector a
120-candidate supervised configuration search on labelled validation data while
its rivals get none produces a stable, seed-consistent, mechanistically plausible
advantage that vanishes at fixed configurations (§26). It has the same signature
as the other three, and it is invisible unless comparisons are stated in terms of
search budget as well as features and data.

The paper's claim is methodological and negative-leaning, which is exactly why it
is useful: **a hallucination detector evaluated without these three controls
cannot be distinguished from a length, style, or confidence detector — and a
comparison run without an equal search budget cannot be distinguished from a
tuning artefact.** Matching is a cheap, general remedy for the first three and
requires no change to the detector under test; the fourth needs only that the
search budget be stated and equalised.

This is worth writing because the literature reports hidden-state hallucination
probes with strong AUROC on datasets where truthful and hallucinated examples
differ systematically in length and prompt construction. We do not claim those
results are wrong; we claim the controls needed to establish them are usually
absent, and we show how much can be manufactured without them.

## 2. The evidence already in hand

Every number below is in `research/fuzzy-lm-anomaly/FINDINGS.md` with seeds.

* **§8** — length alone reaches 0.843; exact matching drops perplexity from 0.720
  to 0.550 and entropy from 0.692 to 0.559, while leaving a hidden-state detector
  untouched. *Different detectors depend on the confound to very different
  degrees*, so the confound reorders the leaderboard rather than shifting it.
* **§9** — a detector reaching 0.906 ± 0.017, beating all rivals 10/10 seeds,
  p = 0.002. It looked solid: pre-registered decision rule, 10 seeds, paired test,
  permutation-style rigour.
* **§11–§12** — the same detector is at **chance (0.529 ± 0.004)** against a
  different family of fabrications with the same truthful set and length matched.
  Fit-set size accounts for −0.028; the template accounts for +0.184.
* **§18** — a late-layer signal that *does* survive length, template, entropy
  matching and a 200-permutation multiplicity null (p < 0.005) at n = 12–21 pairs.
  Shows the controls are not so strong that nothing survives them.
* **§19** — a 7×7 transfer matrix: per-class specialists have a **+0.190**
  diagonal-vs-off-diagonal gap and several **below-chance** off-diagonal cells.
  Evidence that "hallucination" is not one phenomenon.
* **§20** — the third dissolution, with stacked controls quantifying each step.
* **§21 (RETRACTED by §26)** — appeared to show a 4-rule fuzzy system beating
  full-covariance Mahalanobis by +0.040, 8/8 seeds. The fuzzy rule had received a
  120-candidate supervised configuration search on labelled validation positives
  and the rivals none; at fixed configurations the sign flips to −0.019 (1/8
  seeds). **This is the fourth confound, and the most useful single anecdote in
  the paper** — it was found in our own protocol, after three others had already
  taught us to look.
* **§26** — the search-budget measurement: selected +0.014 vs fixed −0.019 on
  identical splits, plus a fit-size sweep showing the fixed-configuration gap is
  negative in 23 of 24 cells across four models.

### Reusable artefacts the paper can offer

* **Probe-set construction** with real-entity twins in identical surface forms
  (`build_prompts_v2.py`, `build_prompts_v3.py`) — the template control at source.
* **A matching harness** for exact conditioning on nuisance variables
  (`length_control.py`, `template_control.py`).
* **A worked example of a wrong method**: residualising on a nuisance fitted only
  on the negative class *increased* one variable's AUROC from 0.632 to 0.958,
  because the model extrapolates badly onto positives. Matching avoids this. Worth
  a short subsection — it is an easy mistake and the failure is not obvious.
* **A reporting convention**: AUROC alongside FPR@95TPR, tunable parameters,
  training time and inference speed. Several detectors here look respectable on
  AUROC while flagging essentially everything at 95% recall.

## 3. What is missing before this can be written

Ordered; (1) and (2) are blocking.

1. **A second model.** Every result is `SmolLM2-360M-Instruct`. The confound claim
   is about *benchmark construction*, not about one checkpoint, so it must be shown
   to survive a different architecture. `Qwen2.5-0.5B-Instruct` is ~7 min of
   capture and the harness already parameterises the model id.
2. **At least one public benchmark.** The confounds are demonstrated on probe sets
   we built. The argument becomes much stronger if length and style imbalance are
   *measured in existing datasets* — e.g. report `n_tokens`-alone AUROC on
   HaluEval, TruthfulQA, FEVER, or an existing activation-probing benchmark. If a
   published dataset has a high length-alone AUROC, that is the paper's headline
   figure and it requires no new modelling.
3. **More matched samples.** The tightest analyses run at n = 12–21 pairs per
   class after stacked matching. Cause is known and specific: only 632 of 846
   curated real questions are answered correctly. Growing the curated fact tables
   (capitals and elements are easiest) is the direct fix.
4. **Human validation of the abstention regex.** Currently audited only against a
   second heuristic (κ = 0.968, errors bounded at 0.29% and in the conservative
   direction). A few hundred hand labels would make it citable.
5. **A power/False-discovery treatment.** We have one permutation null (§18);
   the paper should state, as a function of matched n, how large an apparent
   advantage must be before it is distinguishable from selection noise. This is
   the practical guidance a reader wants.

## 4. Shape of the paper

1. Introduction — hidden-state hallucination probes report strong numbers; what
   would have to be true for them to be trusted.
2. Three confounds, each with its mechanism and a measured magnitude.
3. Controls: exact matching on length, template, and confidence. Cheap,
   detector-agnostic, and no retraining.
4. Case study: three apparent successes, each dissolved, with the decomposition
   isolating cause.
5. What survives: the late-layer signal (§18), which passes length, template,
   entropy matching *and* a 200-permutation multiplicity null — so the controls
   are demonstrably not vacuous.
6. Hallucination is not one target: the transfer matrix.
7. Recommendations: a checklist and a reporting convention.

## 5. Venue notes

Methods/evaluation venues fit better than a fuzzy-systems venue — the contribution
is about benchmark validity, and the fuzzy system is the vehicle, not the subject.
Workshop tracks on evaluation or reproducibility are a natural first target.

Note that §26 removes the companion paper's positive result, so this is currently
the **only** viable write-up from the study. That is not a weakness: four
confounds, each measured, each with a cheap remedy, and one of them caught in the
authors' own protocol after the other three had already trained them to look, is a
stronger and more honest paper than a marginal detector win would have been.

## 6. Honesty constraints to carry into the draft

* Do not imply specific published results are confounded without measuring their
  datasets. The claim is about *missing controls*, not about known errors.
* Report that the first version of our own analysis was wrong, and how it was
  caught. The paper's credibility rests on that being visible.
* Single seed set, single GPU, one model until (1) is done. State it.
* AUROC without FPR@95TPR would hide that several detectors are unusable in
  deployment even when their AUROC looks fine.
