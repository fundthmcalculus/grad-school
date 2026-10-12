# PLAN_MATRYOSHKA: a Matryoshka-trained embedding for a smaller, interpretable embedding space (registered 2026-10-12; plan only, nothing run)

**Terms.** Matryoshka Representation Learning (MRL, Kusupati et al., NeurIPS 2022, arXiv:2205.13147): train one embedding so every *prefix* of its dimensions (8, 16, 32, 48) is itself a usable representation, by summing the loss over several prefix lengths. Prefix k is the coarse version; later dimensions add detail.

**Why here.** The model width is d = 48 and each rule antecedent is a Gaussian over d dimensions, so reading a rule means reading 48-dim centres. The tied embedding is also over half the 96K budget (BPE1024 x 48 = 49K parameters), and the tiered embedding (`flm/run_tiered.py`) already showed most tokens need only 16 dimensions. An MRL embedding gives an ordered, nested space where the first 8-16 dimensions carry coarse distinctions, so (a) rules can be inspected in a low-dimensional space, (b) there is a principled place to cut width, and (c) it unifies with the tiered embedding (the tail's 16 dims become the *prefix* of the head's 48).

**The interpretability claim is to be tested, not assumed.** If no readability measure below moves, the claim is dropped.

## Scope
- The embedding is tied input/output (`flm/models.py`). MRL changes the *output* loss: logits from each prefix, h[:k] . E[:, :k], with cross-entropy summed over k in K. Inference uses the full vector, so the deployed model is unchanged in size (96,416 params, no additions in plain MRL).
- New module `flm/run_matryoshka.py` that monkeypatches the loss (same pattern as `run_tiered.py`); `flm/models.py` and `flm/train.py` are not edited while a sweep runs.

## Experiments (30M chars, seeds 0-4; baseline = v22-tribble random, 1.1095 +- 0.0107)
- **M0 (no training, minutes):** post-hoc truncation curve of the existing 96K checkpoint: zero the dimensions beyond k and score val BPC for k in {8, 16, 24, 32, 48}. This is the null MRL must beat.
- **M1 (plain MRL, equal weights):** K = {8, 16, 32, 48}. Report full-width BPC (must stay within +0.01 of baseline to count as "free") and the truncation curve at each k.
- **M2 (full-width heavy):** weights 0.25 / 0.25 / 0.25 / 1.0, to see whether a full-width penalty is the price of nesting.
- **M3 (MRL + tiered):** t768e16m225 with MRL on the head's first {16, 32}; tests unification with the best recipe.
- **M4 (rules in prefix space):** on the M1 checkpoint, restrict each rule's antecedent distance to the first k dims (k = 8, 16) at inference and measure BPC loss. A rule base that survives is readable as a low-dimensional rulebook.
- **M5 (readability, CPU session's tools):** `flm.rule_reader` / `flm.rule_explain_eval` on M1 vs baseline: variance of rule centres by dimension; nearest-token labels of rule centres at k = 8 vs 48; whether explanations from the 8-dim prefix score as well as from the full vector. Rater-style scoring is blind, key withheld until scores are written.

## Registered predictions
1. M0: truncating the baseline to 16 dims raises val BPC by more than 0.5 (85%); to 32 dims by more than 0.1 (80%).
2. M1: full-width BPC within +0.01 of baseline on a paired five-seed comparison (55%); worse by more than 0.02 (25%); better by more than 0.01 (10%). A 16-dim prefix within 0.25 BPC of baseline (45%) would be the headline.
3. M2: the heavier full-width weight recovers the full-width gap but gives up about half of the prefix quality (60%).
4. M3: no BPC gain over t768e16m225 at full width (65%).
5. M4: restricting to 16 dims costs M1 less than 0.15 BPC while the baseline loses more than 0.5 (50%).
6. M5: variance concentration is higher in the early dims for M1 than the baseline, but nearest-token labels at k = 8 are no more interpretable to a blind rater (60%). A nesting that is real in the loss and not in the semantics is the likely failure.
7. Coherence stays at ~5/15 (90%): this is an interpretability test, not a capability lever.

## Decision rules
- If M1 full-width BPC is worse by more than 0.02, nesting costs capability: stop and report the truncation curve as a pure trade-off.
- If M1 passes (within 0.01) and M4 survives, run M3 and a ten-seed round, and add the prefix-space rulebook to the explainability track (`REVIEW.md`).
- Readability is a claim only for the blind test, not for variance statistics.

## Compute and ownership
M0 takes minutes. Each 30M cell is ~3 min on the GPU (MRL adds K output projections, estimated +25-35% per step); M1-M3 over five seeds ~ 45 min total. M4-M5 are CPU-session jobs on `--save-model` checkpoints. Any 300M confirmation goes back to the user first.

## Relation to other work
`PLAN_NEXT.md` §6-7 (embedding null results, tiered embedding), `flm/rule_reader.py`, `PLAN_RULEDIST.md`. The earlier embedding screens (PPMI, scale, weight decay) tested *initialisation*; MRL changes the *objective*, the untested axis.

## M0 outcome (2026-10-12; `flm/matryoshka_m0.py`, 96K 3B-char checkpoint, val BPC on the first 1M chars; manipulation check: k = 48 reproduces 0.9777)
| k | output-side truncation only | embedding and output truncated |
|---|---|---|
| 8 | 2.7595 | 3.7066 |
| 16 | 2.3963 | 3.7749 |
| 24 | 1.7032 | 3.2787 |
| 32 | 1.4223 | 2.6720 |
| 40 | 1.1835 | 1.7120 |
| 48 | 0.9777 | 0.9777 |
Prediction 1: truncating to 16 dims raises BPC by more than 0.5 (**right**, +1.42 output-only, +2.80 both); to 32 dims by more than 0.1 (**right**, +0.44 / +1.69). The plainly trained model has no usable prefix structure, so M1 (the MRL objective) has a real null to beat. Truncation was applied to the final state and tied embedding, not inside the rule layers (that is M4).

## M1/M2 outcome (`v34-mrl`, 30M chars, seeds 0-4, paired vs v22-tribble baseline 1.1095; `flm/run_matryoshka.py`, driver `flm/run_matryoshka_sweep.py`)
| arm | loss weights on k = 8/16/32/48 | full-width BPC | paired | seeds better |
|---|---|---|---|---|
| M1 | 1/1/1/1 | 1.1952 | +0.0857 | 0/5 |
| M2 | 0.25/0.25/0.25/1 | 1.1529 | +0.0434 | 0/5 |
Truncation check, seed 0 (`flm/matryoshka_m0.py`; manipulation check passed: the output-side prefixes now work). Output-side truncation, val BPC at k = 8/16/32/48: M1 1.385 / 1.217 / 1.193 / 1.193; M2 1.449 / 1.244 / 1.165 / 1.160; plain baseline (96K 3B checkpoint, different budget) 2.760 / 2.396 / 1.422 / 0.978. With the *input* embedding also truncated both MRL models still collapse (M1 k = 16: 3.54; k = 32: 2.39), because the objective nests only the output side; the rule layers and input embedding are unconstrained.
Predictions: M1 within +0.01 of baseline (55%): **wrong**; worse by > 0.02 (25%): **right** (+0.086). M2 recovers the full-width gap and gives up about half of the prefix quality (60%): **partly wrong**: the gap shrank by half (+0.043), not to zero, and the prefix quality at k = 16 barely moved (1.22 vs 1.24). 16-dim output prefix within 0.25 of baseline (45%): **right on the output side** (M1 1.217 vs 1.1095, +0.107).
Decision rule fires: full-width BPC worse by more than 0.02 in both arms, so nesting costs capability under the plain output-side loss. M3 (with the tiered recipe) and M4/M5 are not run. Reading: a 16-dim output readout is almost free (+0.02 over its own full width), which is the useful finding; reducing the whole model to 16 dims (input embedding, rule layers) would need the nesting imposed on the input side too (a different experiment: tie the loss to prefix-masked input embeddings), not registered here. Seeds: 5, a ranking screen, not citable.
