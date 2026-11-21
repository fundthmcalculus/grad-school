# PLAN_TIERVOCAB: does the tiered embedding make a larger vocabulary pay? (registered 2026-10-19, before running)

**Why.** `v7-vocab` found BPE1536 d40 (plain embedding, 93,792 params) 0.0086 better than BPE1024 d48 at 30M chars (2 seeds) but each extra token cost 40 numbers. The tiered embedding (`v30-tiered`, `flm/run_tiered.py`) stores tail tokens in E=16 numbers, so a bigger vocabulary costs 16 numbers per extra token. Untested combination. Larger vocabularies also put more characters in each token (3.16 chars/token at 1024), so a 512-token context covers more text.
**Baseline:** t768e16m225 on BPE1024, 98,280 params, 30M chars, seeds 0-4 (`v30-tiered`, mean 1.0965). Frequencies from the token stream (`outputs/freq_bpe<V>.npz`, first 100M tokens). Same recipe (lr 6e-2, warm-up 100, ctx 512, batch 16, 32 rules, d48, 2 layers).
**Arms (`flm/run_tiered_vocab.py`, `v40-tiervocab`):** V1536 head 640 FFN x2.0 (99,280 params); **V1536 head 512 FFN x2.25 (98,280, exactly iso-parameter with the baseline)**; V2048 head 512 FFN x1.5 (97,184). Confound to report: at a fixed character budget a bigger vocabulary means fewer optimiser steps (30M chars is about 1,160 steps at BPE1024).
**Predictions.** The iso-parameter V1536 arm beats the baseline by 0.004-0.012 BPC mean (45%), on >= 4/5 paired seeds (40%); it is worse than the baseline (20%). V1536 head 640 within 0.003 of the iso arm (60%). V2048 worse than the iso V1536 arm by >= 0.005 (60%; its FFN is 1.5x narrower). 
**Decision rule.** A screen: a >= 0.005 paired gain on >= 4/5 seeds earns a 300M-char two-seed confirmation (~35 min each) and a coherence set; otherwise record as closed. BPC is per character, comparable across tokenizers.

## Round 1 outcome (`v40-tiervocab`, 30M chars, seeds 0-4, paired with `v30-tiered` t768e16m225 on BPE1024, mean 1.0965)
| arm | params | chars/token | steps | BPC mean | paired diff (se) | seeds better |
|---|---|---|---|---|---|---|
| V1536 head 640 FFN x2.0 | 99,280 | 3.48 | 1,052 | 1.0769 | -0.0196 (0.0057) | 5/5 |
| V1536 head 512 FFN x2.25 (iso-parameter) | 98,280 | 3.48 | 1,052 | 1.0817 | -0.0148 (0.0048) | 5/5 |
| **V2048 head 512 FFN x1.5** | 97,184 | 3.66 | 1,001 | **1.0704** | **-0.0261** (0.0045) | 5/5 |
Predictions: iso V1536 beats the baseline by 0.004-0.012 (45%): **wrong, better than predicted** (-0.0148; 5/5 seeds, so the >= 4/5 prediction held); worse than baseline (20%): no; V1536 head 640 within 0.003 of the iso arm (60%): **wrong** (0.0048 better); V2048 worse than the iso V1536 arm by >= 0.005 (60%): **wrong, it is the best arm, 0.0113 better than the iso V1536** despite a 1.5x narrower FFN.
Reading: the tiered embedding does make a bigger vocabulary pay, and the gain grows with the vocabulary (1024 -> 1536 -> 2048: 0, -0.015, -0.026) even while the FFN shrinks from 108 to 72 rules per layer. The lever is the tokenizer: more characters per token (3.16 -> 3.66) means the same 512-token context covers 16% more text and each prediction step covers more of it. Confound: fixed 30M characters means fewer optimiser steps for the bigger vocabularies (1,001 vs about 1,160), which works against them. Per the decision rule (>= 0.005 on >= 4/5 seeds) this earns round 2: larger vocabularies (3072, 4096; tokenizers being trained) and other V2048 shapes, then a 300M-char two-seed confirmation and a coherence set.

## Round 2 registered before running (30M chars, seeds 0-4, same baseline; tokenizers BPE3072 / BPE4096 trained by `python -m flm.bpe`: 3.85 / 3.95 chars/token on the 600M-char stream, vs 3.16 / 3.66 for 1024 / 2048)
Arms (`python -m flm.run_tiered_vocab v2048h384e16m20 v2048h512e12m22 v3072h384e12m15 v4096h256e8m20`): V2048 head 384 FFN x2.0 (99,280 params); V2048 head 512 tail-width 12 FFN x2.2 (99,620); **V3072 head 384 tail-width 12 FFN x1.5 (98,528)**; V4096 head 256 tail-width 8 FFN x2.0 (about 97,000; the tail rank must drop to 8 to afford 3,840 tail tokens).
Predictions: V3072 beats the round-1 best (V2048 head 512 FFN x1.5, mean 1.0704) by >= 0.005 (50%); V4096 is worse than V3072 (55%: rank-8 tail embeddings starve the rare tokens); the two extra V2048 shapes land within 0.006 of V2048 head 512 FFN x1.5 (60%) and neither beats it by 0.005 on >= 4/5 seeds (65%). The vocabulary, not the head/FFN split, is the lever.

## Round 2 outcome (30M chars, seeds 0-4, paired with the BPE1024 tiered baseline 1.0965 and with round 1's best)
| arm | params | chars/token | steps | BPC mean | vs baseline (se) | seeds better | vs round-1 best (V2048 h512 m1.5, 1.0704) |
|---|---|---|---|---|---|---|---|
| V2048 head 384 FFN x2.0 | 99,280 | 3.66 | 1,001 | 1.0743 | -0.0223 (0.0027) | 5/5 | +0.0039 (1/5 better) |
| V2048 head 512 tail-width 12 FFN x2.2 | 99,620 | 3.66 | 1,001 | 1.0762 | -0.0203 (0.0028) | 5/5 | +0.0058 (1/5) |
| **V3072 head 384 tail-width 12 FFN x1.5** | 98,528 | 3.85 | 953 | **1.0547** | **-0.0418** (0.0015) | **5/5** | **-0.0157 (5/5)** |
| V4096 head 256 tail-width 8 FFN x2.0 | 96,848 | 3.95 | 927 | 1.0654 | -0.0311 (0.0033) | 5/5 | -0.0050 (4/5) |
Predictions: V3072 beats the round-1 best by >= 0.005 (50%): **right** (0.0157, 5/5); V4096 worse than V3072 (55%): **right** (+0.0107; its tail rank is 8); the extra V2048 shapes within 0.006 of the round-1 best (60%): right (+0.004, +0.006) and neither beats it on >= 4/5 seeds (65%): right (1/5 each).
Reading: the vocabulary is the lever, not the head/FFN split (V2048 shapes differ by <= 0.006). Along the vocabulary axis the tiered recipe gains 0 / 0.015 / 0.026 / 0.042 at 1024 / 1536 / 2048 / 3072, and V4096 falls back only because it had to drop the tail width to 8; the FFN is shrinking at the same time (108 -> 72 rules per layer). Not yet shown: that the gain survives more data (30M is the screen; the 1B BPE1024 number is 0.9646), and whether it moves coherence.

## 300M confirmation registered before running (`flm/run_tiervocab_300m.py`, `v41-tiervocab300m`; V3072 head 384 tail-width 12 FFN x1.5, 98,528 params; seeds 0-1; reference BPE1024 tiered at 300M: 0.98099 / 0.98209)
Predictions: the gain shrinks with data but stays: V3072 300M BPC in 0.955-0.972 (55%), i.e. a paired gain of 0.010-0.026 (the 30M gain was 0.042; the gain of tiering itself fell from 0.017 at 30M to 0.016 at 300M, i.e. little); below 0.9646 (the best BPE1024 tiered result, at 1B chars) (40%); no better than the BPE1024 reference (10%). Round 3 (head/tail/FFN shapes at V3072 and V4096 tail width 10) follows the confirmation.

## 300M confirmation outcome (`v41-tiervocab300m`; V3072 head 384 tail-width 12 FFN x1.5, 98,528 params; seeds 0-1; 9,522 steps, 721 s each)
**0.95758 / 0.96015 (mean 0.95886)** against the BPE1024 tiered recipe at 300M: 0.98099 / 0.98209 (mean 0.98154). Paired gain **-0.0234 / -0.0219 (mean -0.0227)**, both seeds. Predictions: BPC in 0.955-0.972 (55%): **right**; below 0.9646 (40%): **right: 300M characters of BPE3072 beat 1B characters of BPE1024 (0.9646)**; no better than the BPE1024 reference (10%): no. The gain (30M: 0.042, 300M: 0.023) shrinks with data as expected but remains larger than the gain of the tiered embedding itself (0.017). **New best under 100K parameters: 0.9589 BPC (two seeds, 300M characters); the previous best was 0.9646 (BPE1024 tiered, 1B characters).** Not yet done: round-3 shapes (V3072 head/tail/FFN split, V4096 with tail width 10), the 1B-character run of the V3072 recipe (expected ~40 min: the 300M cell took 12 minutes), and a blind coherence set (set 13 found no coherence gain from the BPE1024 tiered likelihood gain).

## 1B-character run registered before running (`flm/run_tiervocab_1b.py`, `v42-tiervocab1b`; V3072 head 384 tail-width 12 FFN x1.5, 98,528 params, seeds 0-1; the _1b stream re-encoded with BPE3072 by `flm/prep_bpe_long.py`: 337.9M tokens for 1.3B characters)
References: BPE1024 tiered at 1B 0.96796 / 0.96129 (mean 0.9646); BPE3072 tiered at 300M 0.95758 / 0.96015.
Predictions: V3072 at 1B in 0.940-0.955 (55%), i.e. a further 0.004-0.019 below the 300M result; below 0.9589 on both seeds (80%); the paired gain over the BPE1024 tiered recipe at 1B is 0.010-0.025 (60%). Data helps this model less per decade than at 30M -> 300M (saturation, as for BPE1024: 300M 0.981 -> 1B 0.9646).

## 1B-character outcome (`v42-tiervocab1b`; V3072 head 384 tail-width 12 FFN x1.5, 98,528 params; 31,738 steps, 2,475 s each)
**0.94274 / 0.94777 (mean 0.94525)** against the BPE1024 tiered recipe at 1B, 0.96796 / 0.96129 (mean 0.9646): paired **-0.0252 / -0.0135 (mean -0.0194)**; and -0.0136 against the same recipe at 300M (0.95886). Predictions: BPC in 0.940-0.955 (55%): **right** (both seeds inside, seed 0 near the low end); below 0.9589 on both seeds (80%): **right**; paired gain over BPE1024 tiered at 1B 0.010-0.025 (60%): **right** on the mean (0.0194; seed 0's 0.0252 is just outside, seed 1's 0.0135 inside). **New best under 100K parameters: 0.9453 BPC (two seeds, 1B characters); the 96K reference at 1B was 0.9820, the BPE1024 tiered recipe 0.9646.** Seed spread (0.005) is wider than at 300M (0.003). A correction to a status message: seed 0's 0.9427 was inside the registered range, not below it.

## Round 3 registered before running (30M chars, seeds 0-4, vs the BPE3072 head-384 tail-12 FFN x1.5 arm, mean 1.0547)
Arms (`python -m flm.run_tiered_vocab v3072h320e12m175 v3072h384e10m20 v4096h256e10m15`): V3072 head 320 FFN x1.75 (99,320 params); V3072 head 384 tail-width 10 FFN x2.0 (about 99,200); V4096 head 256 tail-width 10 FFN x1.5 (98,432).
Predictions: none beats the V3072 winner by >= 0.005 on >= 4/5 paired seeds (65%); V4096 tail-width 10 within 0.005 of the V3072 winner (45%), better than the V4096 tail-width 8 arm (1.0654) (75%); the head/tail/FFN split matters less than the vocabulary (all three within 0.008 of 1.0547, 60%).

## Blind coherence set 16 registered before rating (`report/coherence16_*`; `flm/coherence_vocab.py`; fresh seeds 70-71; 5 models x 8 texts: BPE3072 tiered 1B seed 0 (0.9427) and seed 1 (0.9478), BPE1024 tiered 1B seed 0 (0.9680) and seed 1 (0.9613), and the 1.32M 1B model as the scale anchor (0.636); story prompts, T 0.5 / top-k 10, 600 characters; two Claude sub-agent raters reading only the blind file, rubric G/C/P 1-5, total 3-15; scores before the key)
Predictions: the BPE3072 pair is within 1 point of the BPE1024 tiered pair on both raters (55%); BPE3072 at least 1 point higher on both raters (30%); at least 1 point lower on both (15%); the 1.32M anchor is at least 2 points above every sub-100K model (75%); no sub-100K model reaches 10/15 (85%). The earlier set (13) found no coherence gain from a 0.02 lower BPC at fixed vocabulary; a bigger vocabulary changes the unit of prediction (more characters per piece), so this is not the same question.

### Set 16 scores (two raters; both score files were written before the key was opened; mean total of G+C+P over 8 texts per model, plot in parentheses)
| model (BPC at 1B chars) | rater A | rater B |
|---|---|---|
| BPE3072 tiered seed 0 (0.9427) | 6.25 (1.50) | 6.88 (1.88) |
| BPE3072 tiered seed 1 (0.9478) | 4.00 (1.00) | 5.88 (1.75) |
| BPE1024 tiered seed 0 (0.9680) | 5.50 (1.50) | 6.38 (1.88) |
| BPE1024 tiered seed 1 (0.9613) | 5.75 (1.50) | 6.62 (1.88) |
| 1.32M anchor (0.636) | 10.12 (3.00) | 10.75 (3.50) |
BPE3072 pair mean 5.12 / 6.38 vs BPE1024 tiered pair 5.62 / 6.50: difference -0.50 / -0.12. Predictions: the BPE3072 pair within 1 point of the BPE1024 pair on both raters (55%): **right**; at least 1 point higher on both (30%): no; at least 1 point lower on both (15%): no; the 1.32M anchor at least 2 points above every sub-100K model (75%): **right** (3.4 to 6.1 above the best); no sub-100K model reaches 10/15 (85%): **right**.
Reading: the 0.019 BPC gain from the larger vocabulary buys **no rated coherence**, as with the fixed-vocabulary tiered recipe in set 13; the seed-to-seed spread of the BPE3072 pair (2.25 points on rater A) is larger than the gap between recipes. Likelihood at 98K keeps improving (0.982 -> 0.965 -> 0.945) while rated coherence stays at 4-7/15; only size moved it (1.32M: 10/15). The amended sub-100K target is therefore reported as: best TinyStories BPC 0.9453 (BPE3072 tiered, two seeds, 1B characters), story coherence 4-7/15.

## Round 3 outcome (30M chars, seeds 0-4; reference: BPE3072 head 384 tail-12 FFN x1.5 = 1.0547; BPE1024 tiered baseline 1.0965)
| arm | params | BPC mean | vs the V3072 winner (se) | seeds better | vs BPE1024 baseline |
|---|---|---|---|---|---|
| V3072 head 320 FFN x1.75 | 99,320 | 1.0607 | +0.0060 (0.0022) | 0/5 | -0.0358 |
| V3072 head 384 tail-width 10 FFN x2.0 | 99,248 | 1.0655 | +0.0108 (0.0020) | 0/5 | -0.0310 |
| V4096 head 256 tail-width 10 FFN x1.5 | 98,432 | 1.0546 | -0.0001 (0.0040) | 3/5 | -0.0419 |
Predictions: none beats the V3072 winner by >= 0.005 on >= 4/5 seeds (65%): **right**; V4096 tail-width 10 within 0.005 of the winner (45%): **right** (a tie); better than the V4096 tail-width 8 arm, 1.0654 (75%): **right** (0.0108 better, so the tail width, not the vocabulary size, was what limited V4096 in round 2); all three within 0.008 of 1.0547 (60%): **wrong** (tail width 10 with the larger FFN is 0.0108 worse).
Reading: the vocabulary gain saturates between 3072 and 4096 at this parameter budget (1.0547 vs 1.0546); smaller tail widths cost (tail 10 vs 12 at V3072: +0.011; tail 8 vs 10 at V4096: +0.011) more than the FFN rules they buy back, and trading head tokens for FFN width also loses (+0.006). The shape found in round 2 (head 384, tail 12, FFN x1.5) is a local optimum of this family. Whether V4096 tail-10 pulls ahead with more data (more examples per tail token) is the open question; next: 300M characters, two seeds, V4096 vs the V3072 300M result (0.95886).

## V4096 tail-10 at 300M registered before running (`flm/run_tiervocab_arm.py v4096h256e10m15 300`, `v43-tiervocab-arm`; seeds 0-1; reference V3072 head 384 tail-12 FFN x1.5 at 300M: 0.95758 / 0.96015)
Predictions: V4096 within 0.005 of V3072 at 300M on both seeds (50%); better than V3072 by >= 0.005 on both seeds (20%: tail tokens get ~3,800 x 10 parameters and see more examples); worse by >= 0.005 on both seeds (30%).

## V4096 tail-10 at 300M outcome (`v43-tiervocab-arm`; 98,432 params; seeds 0-1)
**0.96652 / 0.96166 (mean 0.96409)** against V3072 head 384 tail 12 at 300M: 0.95758 / 0.96015 (mean 0.95886): **+0.0089 / +0.0015 (mean +0.0052)**, V4096 worse on both seeds. Predictions: within 0.005 of V3072 on both seeds (50%): **wrong** (seed 0 is 0.0089 behind); better by >= 0.005 on both (20%): no; worse by >= 0.005 on both (30%): no (seed 1 is only 0.0015 behind): the outcome is a mild loss that the registered ranges did not cover. At 30M the two tied (1.0546 vs 1.0547); with more data the larger vocabulary does not pull ahead, so **V3072 head 384 tail 12 FFN x1.5 stays the recipe** (the tail rank 10 vs 12 and the extra 1,000 tail tokens cost more than the extra characters per token buy).
