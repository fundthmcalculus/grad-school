# Write-up plan — A parsimonious fuzzy anomaly rule for ungrounded LM generation

**Status:** plan of record. Results exist and are committed; prose not written.
**Source:** `research/fuzzy-lm-anomaly/` — see `FINDINGS.md` §§20–23.
**Companion:** `papers/hallucination-detection-confounds.md` (the methodological
negative result). **Keep these separate** — one is an evaluation-methods paper
where the fuzzy system is the vehicle; this one is the positive fuzzy contribution
and belongs in the dissertation.
**Provisional title:** *Four Rules Are Enough: A Parsimonious Fuzzy Anomaly Rule
for Detecting Ungrounded Language-Model Generation*

---

## 1. The claim

On a **template-matched, length-matched** open-set task, a Mixture-of-Gaussians
fuzzy inference system with the Ch 4.3.5 "none of the above" anomaly rule —
**4 rules over 6 antecedents, 95 parameters** — detects ungrounded generation at
**AUROC 0.877 ± 0.022**, beating

* **full-covariance Mahalanobis on identical features** by **+0.036, 8/8 seeds,
  p = 0.008**, with **2.2× fewer parameters** (95 vs 209) and better FPR@95TPR
  (0.551 vs 0.588) — and this **replicates on Qwen2.5-0.5B**: +0.032, 8/8 seeds,
  p = 0.008 (§24);
* **isolation forest** (0.831, 11,649 parameters — 123× more) and **one-class SVM**
  (0.785, 928 parameters).

It does **not** reliably beat a zero-parameter mean-entropy threshold: it wins on
SmolLM2 long-form (+0.036), ties on Qwen long-form (+0.009, p = 0.383), and
**loses on the short-factual task** (−0.014, p = 0.016). The argument therefore
rests on parsimony-and-legibility *versus learned detectors*, not on beating
entropy — see §3.

On (AUROC, parameters) the Pareto front holds exactly two points: the zero-parameter
entropy threshold and this rule. Everything else is dominated.

And it prints:

    RULE 4  IF maxp_first is VERY LOW or LOW  AND ent_first is VERY HIGH
            AND margin_first is LOW  AND ent_max is HIGH
            AND ent_std is VERY HIGH or HIGH  AND n_tokens is VERY LOW
            THEN the model is behaving normally
    ANOMALY IF none of the four rules fires strongly THEN flag as suspect

## 2. Why this is a contribution and not a benchmark number

Three things carry it, in order of strength:

1. **Parsimony with accuracy, on identical features.** The comparison against
   Mahalanobis is the load-bearing one: same 19 inputs, same fit data, same
   open-set protocol, only the density model differs. Winning there is a statement
   about the *model class*, not about feature engineering.
2. **The representation was chosen adversarially, not conveniently.** It is the
   only one of six that survived length, template, and entropy matching
   (§§18–20). Four earlier configurations built on hidden-state geometry were
   each falsified by a control. That history is a strength: the surviving result
   was not selected for looking good.
3. **A legible artefact.** Four rules over named statistics, with membership plots.
   The one-class SVM that scores 0.785 cannot be written down; a 100-tree isolation
   forest cannot either.

A fourth, more interesting observation: the selected antecedents are dominated by
**first-token** statistics (`maxp_first`, `ent_first`, `margin_first`) plus entropy
spread. Grounded generation appears to have a characteristic *confidence profile on
its very first token*, and the rule base carves that into four distinct known-good
modes. Mean entropy collapses all of it into one scalar — which is a mechanistic
explanation for why four rules beat it, and a genuinely interpretable finding worth
its own subsection.

## 3. What must be stated as a limitation, not buried

* **FPR@95TPR = 0.961** in the entropy-matched condition (vs 0.780 for
  Mahalanobis). The rule wins on ranking and loses badly at a high-recall
  operating point. Report both; a paper that quotes only AUROC here would be
  misleading.
* **The entropy-matched advantage is not significant** (+0.029, 6/8 seeds,
  p = 0.078). Only the template+length condition supports a claim.
* **Scope is "ungrounded generation about a non-existent entity"**, not
  hallucination generally. v3 labels are *groundedness*, not correctness — a
  paragraph cannot be graded automatically. Ordinary factual error is **not**
  detected by this mechanism (TriviaQA ≈ chance, §9/§19).
* **The entropy comparison does not generalise** (§24). Two models and two task
  families were tested; only SmolLM2-on-long-form beats entropy. State this in the
  abstract, not the discussion.
* **Two models, two task families, 8 seeds.** Broader than a single checkpoint,
  still narrow.
* **"+entropy matched" is not equally strong across tasks.** On the short-factual
  set entropy scores 0.958 and quartile matching leaves it at 0.839 — when a
  nuisance separates almost perfectly, coarse bins cannot condition it away.
  Report the residual nuisance AUROC rather than assuming the control worked.
* **Hidden-state geometry is a negative result** (§§9, 19, 20). Do not let the
  positive framing imply otherwise.

## 4. What is missing before drafting

1. ~~A second model~~ — **done (§24)**: Qwen2.5-0.5B replicates the Mahalanobis
   result and refutes the entropy result.
2. ~~A second task family~~ — **done (§24)**: v2 short-factual reported alongside.
3. ~~The FPR@95 problem, addressed~~ — **tried and failed (§25)**. Two structurally
   different remedies: Ch 4.3.1's cascade with abstention (neutral on ranking,
   *worse* on the tail, 0/8 seeds improved under entropy matching) and selecting the
   configuration on FPR@95 rather than AUROC (worse on both metrics on the primary
   condition — a single-quantile criterion overfits validation noise). **Report
   FPR@95 as a standing limitation of the method, and say both remedies were tried.**
   The one untried route worth crediting is changing the *score construction* —
   calibrating μ_anom against the fit-split distribution — which is a separate
   contribution, not a tuning pass.
4. **An ablation table** in one place: antecedent count, mode count, membership
   family, metric, norm pair. §22's factorial covers the last three; the first two
   are in `fuzzy_stats_selection.csv` but not tabulated.
5. **Prior-art pass.** Concede one-class SVM, isolation forest, Mahalanobis
   novelty detection, and the LM-uncertainty literature (max-softmax, predictive
   entropy, semantic entropy). The claim is parsimony-plus-legibility at
   competitive accuracy, not novelty of open-set detection.

## 5. Relationship to the dissertation

This closes **Ch 4.3.5's owed experiment** — a head-to-head against one-class SVM
and isolation forest on identical data — in a second domain (host telemetry →
language-model internals), which is what that section asked for. It also exercises
the interpretability claim Ch 4.3.5 makes but never demonstrates.

Two findings from this work belong in Ch 4 regardless of whether this paper is
written:

* **θ is provably rank-invariant** (§3.4): `μ_anom = 1 − max(μ_k) − θ` is a
  constant shift, so it sets the operating point and cannot change separability.
  `plot_anomaly_threshold_sweep` invites the opposite reading.
* **Nilpotent norm families (Łukasiewicz, drastic, nilpotent minimum) are unusable
  as the outer t-conorm** — they saturate to 1 under aggregation and drive the
  complement to a constant 0. Measured at 0 valid runs of 32 (§22) and confirmed
  twice (§3.5).

## 6. Reproduction

Configuration is declared and enforced in `research/fuzzy-lm-anomaly/fis_config.py`
(Gaussian membership, Wasserstein metric, Hamacher/Hamacher norms, θ = 0.5), with
a runtime assertion that the membership family actually built is the one requested
— §22 measured that factor at ±0.262 AUROC, and two committed numbers had already
moved under library default changes before the guard existed.

    python build_prompts_v3.py && python capture.py --prompts prompts_v3.jsonl \
        --prefix capture_v3 --max-new-tokens 96
    python fuzzy_stats.py                 # the headline
    python fuzzy_stats.py --print-rule    # the rule base + membership plots
    python compare_variants.py            # the ablation factorial
