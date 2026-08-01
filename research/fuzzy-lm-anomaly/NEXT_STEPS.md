# Next steps — conditioned on the seed sweep

Written **before** reading the sweep output, so the decision rule is not fitted
to the result. The sweep re-runs the whole §8 comparison over 10 independent
splits and reports the paired advantage of the FIS over its best rival on
length-matched data.

## 0. The decision rule

| verdict | criterion (paired Δ = FIS − best rival, matched, per seed) |
|---|---|
| **STRONG PASS** | mean Δ ≥ +0.05, wins ≥ 9/10 seeds, std ≤ 0.04 |
| **WEAK PASS** | mean Δ > 0 but wins 6–8/10, or std > 0.04 |
| **FAIL** | mean Δ ≤ 0, or the sign flips across seeds |

A weak pass is *not* treated as a pass: it routes to the diagnostics in Branch B
first, then re-enters Branch A only if the instability is explained and fixed.

### Statistical power, independent of verdict

The matched split yields ~170 pairs/class, so the standard error on a single
AUROC is roughly 0.03 — the same order as the effects being compared. **This is
the binding constraint on every claim here**, and it is cheap to fix: capture
scales at ~7,500 generations per 6.7 min at 2.5 GB of a 12 GB card. Going to
~30,000 prompts (~27 min, still single-GPU) roughly doubles the matched-pair
count. Do this in either branch before writing anything up.

---

## VERDICT (recorded after the sweep — the rule above was not changed)

**STRONG PASS.** False-premise, length-matched, 10 seeds:

    FIS · centroid        0.906 ± 0.017
    best rival (OCSVM)    0.789 ± 0.022     <- same 8 features
    paired Δ = +0.117 ± 0.016, wins 10/10, Wilcoxon p = 0.0020

All three thresholds met (Δ ≥ +0.05 ✓, ≥9/10 wins ✓, std ≤ 0.04 ✓). The worst
seed still leads by +0.095 and the sign never flips.

Two results the single split could not have shown, which change the plan's
priorities:

1. **The best rival is now One-class SVM on the *same 8 centroid features*.**
   The comparison is therefore a clean same-representation test, and the +0.117
   is attributable to the fuzzy density model rather than to feature engineering.
   This *strengthens* A5 (the interpretability pitch) — we can now claim the rule
   wins on equal footing and is readable, which the SVM is not.
2. **FIS · PCA collapses to 0.454 ± 0.089 matched** — below chance, and the most
   variable row in the table. The PCA pipeline was almost entirely reading
   length. This retroactively justifies removing PCA and means any future
   representation work should be checked against the length control *before* its
   AUROC is believed.

**Route: Branch A**, in the stated order — A1 (template control) is still
blocking, because a 10/10 seed win on a confound is still a win on a confound.
Seed stability says nothing about template novelty.

Branch B is not triggered. Its B3 list (per-layer deltas, logit-lens trajectory,
token-level features) stays on the backlog as cheap upside rather than as repair
work, and B1's variance decomposition is now partly answered: std of 0.017 across
seeds means KMeans instability is *not* a problem at K=2 on the centroid
representation, despite the low silhouette.

---

## Branch A — the advantage survives

### A1. Kill the template confound (blocking; do this first)

Length is controlled, template novelty is **not**. The false-premise probes are
templated, so the detector may be reading "this prompt looks synthetic" rather
than "the model is fabricating." Until this is closed, the honest scope is
"novel-entity fabrication on templated probes," which is too narrow to be the
headline.

Fix — **template-matched controls.** For every false-premise template, emit a
real-entity twin through the identical surface form:

* fabricated: *"Who won the 1997 Nobel Prize in Interpretive Dance?"*
* real twin: *"Who won the 1997 Nobel Prize in Physics?"*

Now truthful and fabricated answers share the template distribution exactly, and
the same exact-matching machinery from `length_control.py` can match on template
id as well as `n_tokens`. If the advantage survives *template-matched and
length-matched*, the claim is about fabrication. If it collapses, we have learned
the earlier result was prompt-style detection — which we would need to know.

Cost: a few hours; reuses `build_prompts.py` and the existing control harness.
**This is the highest-value experiment in the whole plan**, because it is the one
that can still invalidate the headline.

### A2. A second model — is it transformers, or this checkpoint?

`Qwen2.5-0.5B-Instruct` (494M, 24 layers × 896, different tokenizer, different
training data). The question is whether "late-layer centroid geometry separates
fabrication" is architectural or incidental. Note the winning antecedents were
layers 25–30 of 32 — *relative* depth ~0.8, which is the thing to test for
transfer, not absolute layer index.

Cost: ~7 min capture + re-run stages 4–6. Cheapest strong generalization claim
available.

### A3. Scaling — the SmolLM2 family controls for training data

135M → 360M → 1.7B, same data and recipe, so size is the only variable. 1.7B in
fp16 is ~3.4 GB, comfortable in 12 GB. Two hypotheses worth separating:

* *Detection gets easier with scale* — better-calibrated internals.
* *Detection gets harder* — larger models confabulate more fluently, so the
  fabricated state looks more like the truthful one.

Either answer is publishable, and it is a genuine scaling study rather than a
size anecdote because the family fixes the data.

### A4. More hallucination types — the current two are far apart

Present coverage is "impossible" (non-existent entity) and "ordinary error"
(TriviaQA), and the fuzzy rule solves the first and fails the second. The
interesting cases live in between:

| type | probe | why it matters |
|---|---|---|
| long-tail real entities | real but rare subjects | the hard middle; is 0.9 vs 0.5 a cliff or a gradient? |
| context-conflict | passage contradicting parametric knowledge | the practical RAG failure mode |
| post-cutoff / temporal | events after the training cutoff | fabrication under a *known* knowledge boundary |
| decoding pressure | rising temperature / top-p, forced continuation | graded severity axis rather than a binary label |

The long-tail sweep is the scientifically sharpest: it turns a binary result into
a dose-response curve and would explain *why* TriviaQA fails.

### A5. Convert the win into the interpretability claim

This is what distinguishes the fuzzy approach from a one-class SVM that scores
similarly, and it is Ch 4.3.5's actual selling point — currently unexercised.
Deliverables:

* **Print the rule.** 8 antecedents, 2 rules, readable: *"IF L26 distance is HIGH
  AND L27 cosine is LOW THEN no known rule matched."* Include the membership
  plots per antecedent.
* **Use θ as designed.** §3.4 proved θ moves the operating point without changing
  ranking — so report precision at fixed recall for a deployment setting
  ("warn on 5% of outputs"), which is what θ is actually for.
* **Report cost** (stage 6 measures it): the detector rides a frozen model with no
  second forward pass and no gradient.
* **Close Ch 4.3.5's owed experiment.** It asks for a head-to-head against
  one-class SVM and isolation forest on identical data. We now have exactly that,
  in a second domain (host telemetry → LM internals).

---

## Branch B — the advantage does not survive

**Diagnose before adding capacity.** The temptation is to add rules and features;
the measurements already argue against that being the fix.

### B1. Variance decomposition — locate the instability

Three candidate sources; isolate by freezing one at a time across seeds:

1. **the split** (which truthful answers land in fit)
2. **KMeans init** — the prime suspect. Silhouette was 0.011–0.337, i.e. the
   known-good manifold barely clusters, so the "behavioural modes" the rule base
   is built on may be arbitrary. Freeze the antecedents and labels across seeds
   and see whether variance collapses.
3. **antecedent ranking** — does `calculate_gaussian_correlation` pick the same
   8 of 66 features every seed? Cheap to log and highly diagnostic.

If (2) dominates, mode discovery is the bug, not the rule.

### B2. Replace mode discovery rather than enlarging the rule base

* **K=1 with variance-ranked antecedents** — no clustering at all. The anomaly
  rule does not require multiple classes; §1's finding was only that feature
  *ranking* degenerates at K=1, and variance ranking already solves that.
* **Quantile-conditioned classes** — Ch 4.3.4's `partition_output`/`qcut`
  approach, bucketing on a legitimate behavioural axis instead of KMeans. This is
  the repo's own answer to "where do the classes come from."
* **tribble-cluster's VAT/iVAT + persistence-gated selection (Ch 5)** — the
  pillar built precisely for "no Gaussian assumption, only a dissimilarity
  matrix." If any part of the dissertation should supply modes here, it is this
  one, and wiring it in would connect Ch 5 to Ch 4.3.5.

### B3. On "more rules" and "more variables" — what the data already says

*More rules / antecedents is unlikely to be the fix.* Measured: widening the
antecedent budget 8 → 48 bought ~0.017, and beyond ~12 antecedents the product
t-norm underflows (issue #24 territory — fixed, but the underflow is
mathematical, not a bug). There is a real numerical ceiling on antecedent count.
The productive version of "more rules" is **hierarchical**: Ch 6's fuzzy
trees / HME, or Ch 4.3.1's confusion-driven second-pass specialists with
abstention — depth instead of width.

*More variables is worth trying, in this order* (all cheap — the 33-layer capture
is already on disk):

1. **Per-layer deltas** `h[L+1] − h[L]` — the *update* each layer applies rather
   than the state it holds. Untried, free, and the most likely to carry signal
   the absolute state does not.
2. **Logit-lens trajectory** — how the eventual top token's rank evolves with
   depth; a fabrication may be decided late.
3. **Token-level rather than sequence-level** features — pool-free, one score per
   token, then aggregate. Sequence pooling may be destroying a local spike.
4. **Attention statistics** — per-head entropy and attention-to-prompt mass.
   Listed last because head-level features explode dimensionality against a
   numerical ceiling we already know about.
5. **MLP activation sparsity** per layer.

### B4. If it genuinely fails, report it as a negative result

The repo's stated culture is candid adversarial self-review, and a clean negative
is worth more than a fragile positive: *"the Ch 4.3.5 anomaly rule transfers to
language-model internals and beats confidence baselines on a single split, but
the advantage does not survive resampling; here is the variance decomposition and
the reason."* That also stands as a caution for the BETH result, which is
likewise a single-split number.

---

## Regardless of verdict

* **Scale the capture** (§0) — the power problem is the cheapest thing to fix.
* **Fix the abstention regex properly.** It is hand-built and was already wrong
  once (§5). Hand-label ~200 responses to measure its precision/recall, because
  leakage here inflates results by letting the detector learn "refusal template."
* **Re-verify after any `tribblefis` change** — `norm_sweep.py`'s parity check is
  now a working regression test and caught issue #22 in the first place.
* **Report AUROC, not AUPRC,** and always alongside the base rate (§5).
