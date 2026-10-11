# Planning notes for three follow-ups (GPU session, 2026-10-11; not run; the user asked for plans only)

Context: the 96,416-parameter BPE1024 D model, 30M-character reference 1.1095 +/- 0.0048 (5 seeds), 0.9777 at 3B characters. Cap stays at 99,999 parameters. Each item names the registered prediction, the minimum design, the cost and what would change my mind.

## 3. RFLM at 30M characters: is the sign pattern from the 4M screen real?
**Question.** At 4M characters (`v20-rflm`, 2 seeds) a fixed recurrence cost +0.005 BPC (94,880 params) and consequent propagation on both sides gained -0.0045 (98,464 params); neither cleared the 0.010 bar.
**Design.** GPU, 30M characters, ctx 512 / batch 16, lr 6e-2, seeds 0-4, one job at a time (~100 s each): `fuzzydelta_static --decay fixed`, `fuzzydelta_cpboth`, `fuzzydelta_cpread`, against `random` of `v22-tribble` (same seeds, paired). 15 jobs, ~30 min. Driver: `flm.run_variant` through a grid like `flm/run_rflm.py` with `--device cuda` and `--chars 30e6`.
Add one control the 4M screen lacked: `cpboth` with the propagation matrix frozen at zero (`prop.requires_grad_(False)`), to separate "more parameters in the head" from "propagation" (the 2,048 extra parameters could help for any reason).
**Predictions.** fixed: worse than D by 0.003-0.010 (the gate is worth less than expected, and shrinks with data); cpboth: better than D by less than 0.010 and not distinguishable from the frozen-zero control (the effect is parameters, not propagation); cpread: no effect.
**Decision rule.** Same sign in at least 4 of 5 paired seeds and a mean beyond 2 s.e. (~0.005). A real propagation gain must beat its frozen control by that margin; otherwise it is recorded as "no effect". If fixed costs under 0.005 at 30M, the data-dependent gates are not carrying the model, and a faster parallel (convolutional) form of the mixer becomes a real option.
**Cost / risk.** Cheap. Risk: none to the cited recipe (new mixers register on import, `train.py` untouched).

## 4. A repetition-aware training signal (from the CPU session's `rule_story.py`)
**Evidence.** Across 144 rated texts from 9 checkpoints, the share of repeated token trigrams correlates with the rater total at about -0.27 and with plot at -0.33 (3-4 sd over a shuffle null; pooled across models, so model and text effects are not separated). Loops are visible in the rules (8-rule reading agrees with the model's top choice 0.80 inside repeats vs 0.71 elsewhere).
**Candidate signals (cheapest first).**
(a) **Unlikelihood on repeats** (Welleck et al.): add `-log(1 - p(token))` for tokens that already occurred in the previous K = 32 positions of the *context* but are not the target. The model is trained not to put mass on recently seen tokens unless they are correct; cost: one mask per batch.
(b) **Repeat-weighted loss:** weight the cross-entropy at positions whose target token completes a trigram already seen in the window by 1 + lambda (lambda 0.5), so loop-continuations are not learned as cheaply.
(c) **Decoding-side, no training** (already tested: repetition penalty was the worst decoder, `coherence3`), so this is training-side only.
**Design.** `--rep-loss {off,unlikelihood,weight}` and `--rep-lambda` in `train.py` (add only when no sweep is running); 96K model, 30M characters, seeds 0-4; then *generation* from each: repeated-trigram share at T 0.5 / top-k 10 over 200 texts per model, and one blind coherence set (3 models x 8 texts) rated by two raters.
**Predictions.** Validation BPC within +0.005 of the baseline for (a) and (b) (the loss changes only on a small share of positions); trigram-repeat share falls by at least 20% relative for (a); the blind total does not move by more than the set-to-set noise (about 1.5-2 of 15), because repetition is one of several failure modes and plot sense stays near 2/5.
**What it tests.** Whether the one measured correlate of low coherence is causal. A drop in repeats with no coherence change would say repetition is a symptom, not the bottleneck, and points to capacity and long-range structure.
**Cost / risk.** About 1 GPU-hour for training plus rater time; risk is a BPC penalty above 0.005, which would make it a trade-off, not a free gain.

## 5. Gate-matching distillation (uses the exact rule decomposition, `PLAN_RULEDIST.md`)
**Idea.** The model's logits are `emb + mixers + sum_l sum_r g_{l,r}(t) pi_{l,r}`, so a teacher gives, per position and layer, a gate vector g in R^72 and a profile matrix pi. A student can be trained against the teacher's *per-layer logit terms* as well as its output distribution.
**Two student families (equal parameters is the point; both under 99,999).**
(i) **same-size student, richer target:** the 96K architecture trained with `L = CE + lam_KL * KL(teacher || student) + lam_gate * sum_l ||g_l^student - g_l^teacher||^2` (the teacher's gates are available for free by running the 3B checkpoint on the same windows; they are not stored, so compute them online or cache for 30M characters = 9.5M tokens x 2 layers x 72 floats = 5.5 GB in float16 per layer set, which does not fit; compute online on the GPU, the teacher forward is cheap relative to the student step).
(ii) **fewer-rule student:** R = 36 rules per layer (FFN mult 0.75), the freed ~8K parameters spent on width (d 52); trained with the same gate-matching but matching the teacher's gates projected onto the student's rules (a learned 72 -> 36 linear map, discarded after training).
**Controls.** The same students trained on data alone; output-KL-only distillation from the same teacher (earlier test: no effect at 30M for a 280K teacher, `v3-post`); each 5 seeds at 30M characters.
**Predictions.** (i) gate matching improves on output-KL-only by less than 0.01 at 30M characters, because the student already sees the same targets through the logits; (ii) the fewer-rule student is worse than the 72-rule baseline by 0.01-0.03 trained on data alone and recovers at most half of that with gate matching. The earlier null for KL distillation (teacher 0.97 vs student 1.11) is the prior: **expect small effects, and the value is mainly diagnostic** (does the teacher's *rule structure* transfer?), plus a rule-level check from the CPU session's reader: does the 8-rule reading of the student match the teacher's (its `rule_reader.py` hit rate is the metric).
**Design cost.** The largest of the three: a teacher-forward hook in `train.py` (`--teacher` exists for KL; add the per-layer gate term), 1-2 days of careful work; the GPU cost is ~1 h for 30 runs. Do it last and only after items 3-4 and the diagnosis in `v24-diag`.
**What would change my mind.** If gate matching lets the 36-rule student match the 72-rule model, then the 14-direction profile geometry (participation rank ~14 of 72) is exploitable and a much smaller rule bank is possible; that is the distillation result worth having.

### Item 4 outcome (`v25-rep`, GPU, 30M chars, seeds 0-4; run on the user's "Run item 4")
Code: `flm/run_rep.py` (loss patch; train.py untouched), `flm/run_rep_sweep.py`, `flm/rep_share.py`. lambda 0.5, K = 32.

| arm | BPC mean ± sd (5 seeds) | paired vs random (v22 seeds) | repeated-trigram share, seed-0 model, 100 texts (T 0.5, top-k 10) |
|---|---|---|---|
| random (baseline) | 1.1095 ± 0.0107 | – | 0.147 ± 0.056 |
| unlikelihood | 1.1147 ± 0.0067 | +0.0053 (worse on 4/5 seeds, one −0.0008) | 0.117 ± 0.040 (−20% relative) |
| repeat-weighted CE | 1.1091 ± 0.0078 | −0.0004 (3/5 worse) | 0.162 ± 0.059 (+10%) |

Predictions: BPC within +0.005 for both — weight right, unlikelihood marginally wrong (+0.0053, driven by seed 3 at +0.018); repeat share falls >=20% for unlikelihood — met only at the boundary (−20.4%), on ONE checkpoint, so a screen, not a claim; weighting does nothing for generated repeats (+10%, within noise). Blind coherence was NOT rated (needs raters): the question whether fewer repeats move the 6-8/15 score is open. Next if wanted: 3-4 checkpoints per arm for the share, then a blind set of unlikelihood vs baseline (the peer session offered rating).

## 6. Embedding configuration (`v28-emb`; user question 2026-10-11 "have we evaluated messing with the embeddings?")
Already done: factorized tied embedding `--emb-dim` (G35 refuted: 0.040-0.14 worse, monotone in width), vocabulary ladder (wide embedding pays), n-gram-aligned embedding for the char models (`v8`), PPMI successor vectors as embedding *combined with* consequent-first init (`succcf`, worse; confounded with cf, so the embedding alone was never tested). Not done: embedding init alone, init scale, embedding weight decay. At BPE1024 the embedding is 51% of the 96,416 parameters, so it is the biggest single block.
Arms (`flm/run_emb.py`, 30M chars, seeds 0-4, vs random 1.1095): `ppmi`, `ppmip` (PPMI-SVD successor / predecessor vectors as the initial embedding), `std05` / `std20` (init scale x0.5 / x2), `nowd` (no weight decay on the embedding).
Registered before running: (1) `ppmi`/`ppmip` give a head start that is gone by step 300 and a final within +-0.01 of random (the cf head start was; the embedding is trained directly, so unlike the rule init it cannot die) — if either is >= 0.01 better on >= 4/5 seeds it is a real win; (2) `std05`/`std20` within +-0.01 (AdamW at lr 6e-2 rescales the embedding fast); (3) `nowd` within +-0.01, slightly better if anything because 51% of parameters sit in the one decayed block. Prior: all null; this is a screen of untested knobs. Not tested here and why: untied output (+49K params, over the cap at d 48), frequency-tiered embedding widths (needs new model code; next if the knobs are null and the user wants it), embedding dropout.

### Item 4 coherence follow-up (CPU session, set 9, 2026-10-11; one rater, scores before the key)
Baseline (v24-diag/random-save_s0) 7.25/15 (plot 1.50); repeat-weighted 6.62; unlikelihood 5.62 (plot 1.12). Unlikelihood cut repeats ~20% yet scored ~1.6 lower (~2 SE, one checkpoint per arm, one rater). Fewer repeats did not mean better stories here, so repetition looks like a symptom, not the bottleneck; unlikelihood is not adopted. A second rater on report/coherence9_blind.json would firm this up (the key is pushed, so a rater here is less blind). Files: report/coherence9_*, DECISIONS.md D13.

### Item 6 outcome (`v28-emb`, GPU, 30M chars, seeds 0-4; random baseline 1.1095)
| arm | BPC mean ± sd | paired vs random | seeds better |
|---|---|---|---|
| `ppmi` (successor vectors init) | 1.1057 ± 0.0041 | −0.0037 | 3/5 |
| `ppmip` (predecessor vectors init) | 1.1039 ± 0.0064 | −0.0056 | 4/5 |
| `std05` (init x0.5) | 1.1168 ± 0.0050 | +0.0073 | 1/5 |
| `std20` (init x2) | 1.1153 ± 0.0088 | +0.0058 | 1/5 |
| `nowd` (no weight decay on embedding) | 1.1111 ± 0.0102 | +0.0016 | 2/5 |
Predictions: all within ±0.01 — right for every arm; PPMI win rule (>= 0.01 better on >= 4/5 seeds) — not met (best −0.0056, one seed +0.021). `nowd` I expected slightly better; it is slightly worse (noise). Embedding init and scale and decay are null at this size; the best arm (predecessor-vector init) is a ranking of -0.006 at the seed spread (sd ~0.006-0.010), not a claim. Remaining untested embedding ideas: frequency-tiered widths (new model code), untied output (over the cap).

## 7. Frequency-tiered embedding (`v30-tiered`; under the cap, standing goal; code `flm/run_tiered.py`, driver `flm/run_tiered_sweep.py`)
Idea: the factorized embedding (`v16-factemb`) lost 0.040+ because *every* token's logit became rank-limited. Rare tokens carry little loss mass, so give the TIER_NH most frequent tokens a full d_model-wide embedding and put the tail in TIER_E dims with a shared up-projection (tail logits rank TIER_E); spend the saved parameters on the FFN (mult 1.5 -> 2.0, which is over the cap at full embedding: 102,608). Arms (30M chars, seeds 0-2; baseline 1.1095 at 96,416 params): `t256e32m15` (isolates the embedding cost, ~85.7K params), `t256e32m20` (91,856), `t512e24m20`.
Registered before running: (1) `t256e32m15` is worse than baseline by 0.005-0.03 (some rank loss on the tail; far less than factemb's 0.04+ because only the tail is limited); (2) `t256e32m20` lands within 0.01 of baseline either way — the extra FFN roughly pays for the tail's rank loss — and beats it by >= 0.01 on all 3 seeds with probability ~20%; (3) `t512e24m20` worse than `t256e32m20`. Prior: null-to-slightly-negative; a screen, seeds 0-2 only, so a hit would need a 5-seed rerun before any claim.

### Item 7 outcome (`v30-tiered`, GPU, 30M chars, seeds 0-2; baseline 1.1095 at 96,416 params, 5 seeds)
| arm | params | BPC (seeds 0/1/2) | mean | paired vs baseline |
|---|---|---|---|---|
| `t256e32m15` | 85,664 | 1.1424 / 1.1337 / 1.1350 | 1.1370 | +0.026 (3/3 worse) |
| `t256e32m20` | 91,856 | 1.1300 / 1.1214 / 1.1343 | 1.1286 | +0.018 (3/3 worse) |
| `t512e24m20` | 91,472 | 1.1210 / 1.1069 / 1.1167 | 1.1149 | +0.004 (1/3 better) |
Predictions: (1) `t256e32m15` worse by 0.005-0.03 — right (+0.026); (2) `t256e32m20` within 0.01 of baseline — wrong (+0.018); (3) `t512e24m20` worse than `t256e32m20` — wrong (better by 0.014). Reading: the tail's rank limit costs ~0.026 at 256 head tokens; a wider head (512) removes most of it even with a narrower tail (24), and the extra FFN pays back ~0.009. `t512e24m20` reaches baseline quality (+0.004, inside seed spread) with 4.9K fewer parameters, but no arm beats the baseline, so tiering is a parameter-efficiency trade, not a win at the cap. Untested: head 768 / tail 16, and spending the 4.9K saved on d. Screen only (3 seeds).

### Item 7 follow-up, registered before running (`v30-tiered`, arms added to the same driver, seeds 0-2)
`t512e24m25` (512 head, tail 24, FFN mult 2.5; ~97.7K params, under the cap) and `t768e16m25` (768 head, tail 16, mult 2.5). Reasoning: the 512-head arm matched baseline with 4.9K fewer parameters, so spending the savings on more FFN rules should now buy something. Prediction: `t512e24m25` beats the baseline by 0.003-0.010 (a ranking at 3 seeds; ~35% that it beats it by >= 0.005 on all 3 seeds); `t768e16m25` is within 0.005 of `t512e24m25` (the tail matters little by then). Prior: small.

### Item 7 follow-up outcome (seeds 0-2; baseline 1.1095 at 96,416 params)
| arm | params | mean BPC | paired vs baseline |
|---|---|---|---|
| `t512e24m25` | 97,664 | 1.1033 | **−0.0077 (3/3 better)** |
| `t768e16m25` | **101,376 (OVER the 99,999 cap: my parameter estimate was wrong)** | 1.0883 | −0.0227 (3/3) |
Predictions: `t512e24m25` beats baseline by 0.003-0.010 — right (−0.0077, all 3 seeds); `t768e16m25` within 0.005 of it — wrong (−0.015 further), and it is over the cap so it does not count. First under-cap result in this line that is ahead of the baseline on every seed, but 3 seeds with paired sd ~0.006: not yet a claim.
Registered before the next round (seeds 0-4 for all arms; existing seeds 0-2 reused): under-cap arms `t640e16m25` (~97.7K) and `t768e16m225` (FFN mult 2.25, ~98.3K), plus seeds 3-4 for `t512e24m25`. Prediction: both new arms beat `t512e24m25` (the more tokens kept at full width / the smaller the tail, the better, as the over-cap arm suggests), by 0.003-0.015; the 5-seed paired gain of `t512e24m25` stays between -0.004 and -0.010. A claim needs >= 0.005 better on >= 4 of 5 seeds for the final arm, then a 300M-scale confirmation.

### Item 7, five-seed outcome (`v30-tiered`, 30M chars; baseline seeds 0-4 = 1.112 / 1.1129 / 1.108 / 1.0926 / 1.1219, mean 1.1095)
| arm | params | mean ± sd | paired vs baseline | seeds better | per-seed paired |
|---|---|---|---|---|---|
| `t256e32m15` | 85,664 | 1.1350 ± 0.0045 | +0.0256 | 0/5 | +0.030 +0.021 +0.027 +0.038 +0.012 |
| `t256e32m20` | 91,856 | 1.1283 ± 0.0047 | +0.0188 | 0/5 | +0.018 +0.009 +0.026 +0.037 +0.005 |
| `t512e24m20` | 91,472 | 1.1137 ± 0.0100 | +0.0043 | 2/5 | +0.009 −0.006 +0.009 +0.031 −0.022 |
| `t512e24m25` | 97,664 | 1.1059 ± 0.0048 | −0.0036 | 4/5 | −0.014 −0.007 −0.002 +0.018 −0.013 |
| `t640e16m25` | 97,280 | 1.1046 ± 0.0064 | −0.0048 | 3/5 | −0.009 −0.005 +0.001 +0.017 −0.028 |
| **`t768e16m225`** | **98,280** | **1.0965 ± 0.0061** | **−0.0130** | **4/5** | −0.025 −0.010 −0.010 +0.007 −0.026 |
Predictions: both new arms beat `t512e24m25` by 0.003-0.015 — `t768e16m225` yes (by 0.0094), `t640e16m25` no (by 0.0012); `t512e24m25`'s 5-seed gain between −0.004 and −0.010 — narrowly wrong (−0.0036; the 3-seed −0.0077 shrank when seeds 3-4 came in: seed 3 is the baseline's lucky seed, 1.0926). The registered claim rule (>= 0.005 better on >= 4 of 5 seeds) is **met by `t768e16m225` only** (4/5 seeds beat by 0.010-0.026; seed 3 is +0.007). It is the first under-cap (98,280 params) configuration ahead of the 96K baseline at 30M characters. Per the plan it needs a 300M-character confirmation before it is a result: `v31-tiered300m`, seeds 0-1, vs v12-long512 (0.9976 / 1.0053), registered: the 30M gain (−0.013) shrinks at 300M (warm-up's did), prediction −0.003 to −0.010; counts as confirmed if better than the matching baseline seed by >= 0.003 on both seeds; if both seeds are within ±0.003 the 30M gain is a short-run effect and the cited recipe stays.
