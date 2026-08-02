# Four things — the plan of record

**Supersedes `NEXT_STEPS.md`**, which was written before the §11–§12 falsification
and routes to branches that no longer apply.

Current state: one defensible positive result (§23) — a fuzzy rule of 4 rules over
6 antecedents (95 parameters) over the 19 output statistics, **AUROC 0.877 ± 0.022**
on a template- and length-matched task, beating full-covariance Mahalanobis on
identical features by **+0.036, 8/8 seeds, p = 0.0078**. Hidden-state geometry is a
negative result (§§9, 19, 20). Two write-up plans exist in `papers/`, both blocked.

---

## 1. Second model + second task family — ✅ DONE (§24)

**Outcome:** the Mahalanobis comparison replicated across both architectures (+0.036 SmolLM2, +0.032 Qwen, both 8/8, p = 0.008). The *entropy* comparison did **not** — it ties on Qwen and loses on the short-factual task, so that half of the §23 headline is retracted. Claim narrowed accordingly.

<details><summary>original plan</summary>


**Cost:** ~15 min. `Qwen2.5-0.5B-Instruct` is ~7 min of capture and the harness
already parameterises the model id; the v2 short-factual data is **already on
disk**, so reporting the same detector there is nearly free.

**Why together:** they answer one question — does 0.877 / +0.036 / 8-of-8 hold on a
different architecture *and* a different task family?

**Why it matters that this can fail:** the selected antecedents are dominated by
**first-token** statistics (`maxp_first`, `ent_first`, `margin_first`). A different
tokenizer means a different first token, so there is a real mechanism by which this
could be checkpoint-specific. That is precisely the reason to test rather than
assume. It is also the top blocking item for *both* write-ups
(`papers/fuzzy-anomaly-rule-slm.md` §4.1, `papers/hallucination-detection-confounds.md`
§3.1).

**Outcome either way is useful:** it holds → both papers unblock; it does not →
scope narrows to one checkpoint, which we need to know before drafting, not after.

</details>

## 2. The FPR@95TPR problem — ❌ CLOSED AS NEGATIVE (§25)

**Two structurally different remedies tried, both failed.** The Ch 4.3.1 cascade was neutral on ranking and *worse* on the tail (0/8 seeds improved under entropy matching). Selecting the configuration on FPR@95 instead of AUROC was worse on *both* metrics on the primary condition — it is a single-quantile statistic and overfits validation noise. This points at the tail being a property of the rule class, not the fitting procedure. Report FPR@95 as a standing limitation.

The one route still worth crediting: change the **score construction** — calibrate μ_anom against the fit-split distribution (per-mode rank or p-value rather than a raw membership complement) so the tail is shaped by data rather than operator algebra. Scope it as its own contribution, not a tuning pass.

<details><summary>original plan</summary>


Under entropy matching the rule wins on ranking (0.789 vs 0.760) but flags
essentially everything at 95% recall. This is **structural, not a tuning miss**:
§3.4 proved θ is rank-invariant (`μ_anom = 1 − max(μ_k) − θ` is a constant shift),
so no threshold search can fix it.

The natural fuzzy answer is **Ch 4.3.1's confusion-driven second-pass cascade with
abstention** — a specialist that fires only in the region where the first rule base
is uncertain. This both repairs a real defect and adds dissertation-relevant fuzzy
machinery rather than more measurement.

</details>

## 3. Measure length/style imbalance in a public benchmark

Unblocks `papers/hallucination-detection-confounds.md`. Report `n_tokens`-alone
AUROC on HaluEval, TruthfulQA, FEVER, or an existing activation-probing set. **No
modelling required.** If a published dataset scores high on length alone, that is
the paper's headline figure.

This is currently the confounds paper's weakest point: the claim rests entirely on
probe sets we built ourselves. Note the framing constraint already recorded — the
claim is about *missing controls*, not about known errors in specific published
work.

## 4. Grow the curated fact tables — ✅ DONE

Real subjects 165 → 424; with 3 phrasings, real prompts **471 → 1,272 (2.7×)**, fake 1,640 → 4,052, total 5,324 (`prompts_v4.jsonl`). Six new long-form templates (theorem, paradox, reaction, law, constant, experiment). Also fixed a generator defect that capped five templates' fake side at 48.

<details><summary>original plan</summary>


Grunt work, but the matched-sample ceiling (147–186 pairs after stacked matching)
is what leaves the entropy-matched condition non-significant (p = 0.078). Capitals
and chemical elements are the cheapest to extend; `build_prompts_v2.py` /
`build_prompts_v3.py` hold the tables.

</details>

---

### Not on the list, and why

* **Scaling study** (135M → 1.7B): premature until (1) says the result generalises
  at all.
* **Attention / MLP feature families:** hidden-state geometry is a negative result;
  adding families there widens the multiplicity problem without a hypothesis.
* **More norm/metric sweeps:** §22 settled these. The membership family dominates
  (±0.262) and is now locked and asserted in `fis_config.py`; metric and norm pair
  are worth ≤0.015 and +0.002 respectively.

---

# Added after §27 — the complementarity result

§27 established, with equal search budgets and an artefact check, that the fuzzy
rule's advantage over mean entropy is governed by entropy's own performance
(r ≈ −0.78, crossover at entropy AUROC ≈ 0.61). Two experiments follow directly.
They are independent and can run in either order.

## 5. Does model scale set the regime? — ✅ DONE (§29)

**Answer: no.** Entropy improves monotonically within the SmolLM2 family (0.713 → 0.838 → 0.909 for 135M/360M/1.7B), but Gemma3-270m sits at 0.546 while SmolLM2-135M — half its size — reaches 0.713. Between-family variation swamps within-family scaling, so the fuzzy rule is a **weak-entropy-model** technique, not a small-model one, and you cannot predict the regime from parameter count.

<details><summary>original plan</summary>


**The hypothesis.** Gemma3-270m is the smallest model tested and the only one in
the winning regime (entropy 0.546, barely above chance). If entropy calibration
improves with scale, then the fuzzy rule's niche is *small or weakly calibrated
models*, and the crossover is predictable from size alone.

**Design.** The SmolLM2 family holds training data and recipe fixed and varies
only size: **135M / 360M / 1.7B**. The 360M capture already exists
(`capture_v4_smollm2`); the other two are ~4 min and ~30 min of capture on the
same `prompts_v4.jsonl`, in bfloat16 like the rest.

Measure, per (size × template) cell with the §27 protocol — fixed configuration
for both detectors, template constant, length matched:

* entropy AUROC as a function of size;
* (FIS − entropy) as a function of size;
* whether the crossover at ≈0.61 is crossed somewhere in the family.

**What each outcome means.**

* *Entropy improves monotonically with scale and the gap closes* → the method is
  a **small-model** technique. That is a clean, honest scope statement and it is
  useful: sub-500M models are exactly where cheap on-device detection matters.
* *Entropy does not improve with scale* → the regime is set by something else
  (tokenizer, instruction tuning, abstention behaviour), and Gemma's position is
  not about size. That redirects the question but is equally publishable.
* *Non-monotonic* → most interesting, and would need the confound checks run
  again before believing it.

**Cost:** ~35 min capture + ~10 min analysis. **Blocking risk:** none; the 1.7B
model is 3.4 GB in bf16 and fits alongside activations in 12 GB.

</details>

## 6. Can the detector be switched without labels? — ✅ DONE (§28, §29)

**With labels, yes and cheaply:** 20 labelled examples buy +0.0104 (91% oracle agreement), 100 buy +0.0134 (99.5%). **Label-free: predictable but not yet actionable.** At six models the known-good proxy predicts entropy's AUROC out-of-sample at r = +0.689 (it was −0.492 at four models — the n=4 'untestable' call was right), but converts to only +0.0025 of an available +0.0132, because it is not sharp enough near the 0.61 boundary where the decision is contested.

<details><summary>original plan</summary>


**Why switching and not blending.** §27 showed a zero-parameter rank-average of
the two scores reaches 0.735 against entropy's 0.743 and beats both in only 9 of
44 cells. The detectors are complementary **across** regimes but not **within** a
cell, so a blend gains nothing. The value — if any — is in *choosing* the right
detector per deployment.

**Three rungs, in increasing realism.**

1. **Oracle ceiling.** Switch using the true per-cell entropy AUROC. This is
   cheating and is only there to bound what switching could ever be worth. If the
   oracle gain over always-entropy is small, stop: there is nothing to win.
2. **Small labelled calibration set.** Estimate entropy's AUROC from *k* labelled
   examples (k = 20, 50, 100, 200), switch if the estimate is below the crossover.
   Report net gain against always-entropy **and** the cost in labels. This is the
   realistic deployment story, and the honest failure mode is that the AUROC
   estimate at small k is too noisy to switch on — which is testable directly.
3. **Label-free proxy.** The interesting one, and the only one that keeps the
   open-set protocol intact. Ask whether entropy's discriminative power in a cell
   is predicted by statistics computable on the **known-good fit split alone** —
   e.g. the variance, range, skew or bimodality of entropy over grounded
   generations. Intuition: if entropy barely varies on known-good output, it has
   little room to separate anything. Fit that predictor on some models/templates
   and test it on held-out ones, so the proxy itself is validated out-of-sample.

**Reporting rules, carried from §26.** Every arm gets the same search budget, or
none. The oracle arm must be labelled as an upper bound wherever it appears.

**Cost:** no new capture — all four v4 captures are on disk. ~20 min.

**What would make this a result:** rung 3 working at all. Rungs 1–2 are
bookkeeping; a label-free switch that recovers most of the oracle gain would be a
genuinely useful and, as far as I know, unreported observation about when
confidence-based hallucination detection can be trusted.

</details>
