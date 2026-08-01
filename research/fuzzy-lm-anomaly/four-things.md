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
