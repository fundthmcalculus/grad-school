# PLAN_ANALYSIS: the next analysis round (registered 2026-10-22, before any run; branch `exp/allfuzzy-next`, draft PR #280)

Owner's order: items 1 and 5 first, then 2, 3, 4 of the list in #280 (1 the end of the training schedule, 5 a learning-rate screen at the new shape, 2 coherence and repetition of the shipped configuration, 3 does the shape generalise to SimpleStories, 4 a third seed at 300M).

## A correction that changes item 1
`PLAN_REPEAT.md` said the main schedule is a cosine "to zero". **It is not**: `train.py`'s `--lr-floor` defaults to **0.1**, so every 300M run in this repository (the shape screens, the hybrid, the 1B runs) ends at 10% of its peak learning rate (6e-3 for peak 6e-2); only the 400-1,600-step post-hoc phase (`flm.post_ul --alpha 0`, lr 3e-3 decaying to 0) takes the learning rate to zero. The 0.010-0.0135 gain of the post-hoc phase is therefore *most likely the missing final decay*, not "the weights not settling" or the extra tokens, and the cheapest fix is a main run with `--lr-floor 0`. Item 1 is redesigned to separate the three explanations: the final decay, the extra tokens, and a long low-rate phase.

## Item 1: the end of the training schedule (F7 `V6144 head 640 tail 8 d32` and the hybrid H3 `V6144 head 640 tail 8 d40`, seed 0, 300M characters, warm-up 1000, lr 6e-2; reference: the existing runs, floor 0.1: F7 0.96596, H3 0.94134; the post-hoc phase of 400 / 800 / 1,600 steps at lr 3e-3: F7 -0.0098 / -0.0117 / -0.0135, H3 -0.0094 / -0.0116 / -0.0136)
- **1c: a main run that decays to zero** (`--lr-floor 0`, `outputs/v70-shape300` and `v72-hybshape300`, suffix `_floor0`): the same tokens as the reference, 12.5 minutes each. **1c':** the 1,600-step post-hoc phase on top of each floor-0 checkpoint (does a final decay already take the whole gain?).
- **1a: saturation of the post-hoc phase:** 3,200 and 6,400 steps at lr 3e-3 from the original checkpoints.
- **1b: the token-matched control:** a main run with 13M more tokens (353M characters, floor 0.1; the 1,600-step phase is 13.1M tokens, 17.7% of the main run's 74M), F7 and H3, seed 0 (suffix `_chars353`, 14.5 minutes each). Separates "more tokens" from "a low-rate phase".
Predictions: **floor 0 beats floor 0.1 by at least 0.007 for both models (65%)** and captures at least 60% of the 1,600-step phase's gain, i.e. at least 0.0081 for F7 (50%); the 1,600-step phase on top of the floor-0 checkpoint adds at most 0.004 (55%); 3,200 steps gain at least 0.0015 more than 1,600 (60%) and 6,400 steps at most 0.0025 more than 3,200 (55%); the extra-token control gains 0.001-0.006 (65%) and the post-hoc phase beats it by at least 0.005 (60%); F7 and H3 gain within 0.003 of each other in every arm (70%), so the fair gap stays within 0.003 of 0.0233-0.0247 (75%).
Decision: if floor 0 captures most of the gain, **every future run uses `--lr-floor 0` (or a small value) and the post-hoc phase is dropped**; if the post-hoc phase beats it, the recipe keeps a final phase and the saturation arm sets its length.

## Item 5: a learning-rate screen at the new shape (30M characters, five seeds, paired by seed with the existing 6e-2 arms; each model at its own recipe: all-fuzzy F7 at warm-up 600, hybrid H3 at warm-up 100)
Arms: lr 3e-2 and 1e-1 for F7 (`outputs/v69-shape`, suffix `_lr3e-2` / `_lr1e-1`) and for H3 (`outputs/v71-hybshape`, same suffixes), 20 cells of about 75 seconds. Reference at 6e-2: F7 1.0588 (five seeds), H3 1.0199 (five seeds).
Predictions: 6e-2 is within 0.003 of the best of the three for F7 (60%) and for H3 (60%); 3e-2 is worse than 6e-2 for F7 by at least 0.003 (60%); 1e-1 beats 6e-2 for F7 by at least 0.003 (25%), for H3 (25%); the best learning rate is the same for both (55%); the gap (F7 minus H3) at each model's best learning rate is within 0.004 of the gap at 6e-2 (0.0389) (70%); a hybrid run at 1e-1 diverges on at least one seed (20%).
Decision: a learning rate that beats 6e-2 by at least 0.005 on four of five seeds for either model goes to a 300M confirmation (two seeds); otherwise 6e-2 stands for both, and the fairness claim holds on this axis.

## Items 2, 3, 4 (registered when they start, after items 1 and 5)
2: coherence and repetition of the shipped configuration (re-annealed F7 and H3 with DRY); 3: the 30M screen of the shape on SimpleStories with a BPE6144 trained there; 4: a third seed at 300M for F7 and H3 with the final phase chosen by item 1.

## Results of items 1 and 5 (recorded 2026-10-22; seed 0 for item 1, five seeds for item 5)

### Item 1: the end of the schedule (300M characters, validation bits per character)
| arm | F7 | change vs floor 0.1 | H3 | change vs floor 0.1 | gap |
|---|---|---|---|---|---|
| reference: the original run, floor 0.1 | 0.96596 | | 0.94134 | | 0.0246 |
| **floor 0 main run (1c)** | **0.95577** | **-0.0102** | **0.93034** | **-0.0110** | 0.0254 |
| floor 0 + 1,600-step post-hoc phase (1c') | 0.95508 | -0.0109 | 0.92962 | -0.0117 | 0.0255 |
| 13M extra tokens, floor 0.1 (1b, 353M characters) | 0.96323 | -0.0027 | 0.9398 | -0.0015 | 0.0234 |
| floor 0.1 + 1,600-step phase (earlier) | 0.95242 | -0.0135 | 0.92771 | -0.0136 | 0.0247 |
| floor 0.1 + 3,200 steps (1a) | 0.9506 | -0.0154 | 0.9259 | -0.0154 | 0.0247 |
| floor 0.1 + 6,400 steps (1a) | 0.94861 | -0.0174 | 0.92435 | -0.0170 | 0.0243 |
Predictions: floor 0 beats floor 0.1 by at least 0.007 for both (65%): **right** (0.0102, 0.0110); it captures at least 60% of the 1,600-step gain (50%): **right** (75% and 81%); the 1,600-step phase on a floor-0 checkpoint adds at most 0.004 (55%): **right** (0.0007 for both); 3,200 steps gain at least 0.0015 more than 1,600 (60%): **right** (0.0019, 0.0018); 6,400 steps at most 0.0025 more than 3,200 (55%): **right** (0.0020, 0.0016); the extra-token control gains 0.001-0.006 (65%): **right** (0.0027, 0.0015) and the post-hoc phase beats it by at least 0.005 (60%): **right** (0.0108, 0.0121); F7 and H3 gain within 0.003 of each other in every arm (70%): **right** (within 0.0008 in every arm); the fair gap stays within 0.003 of 0.0233-0.0247 (75%): **right** (0.0234-0.0255).
Reading: the 0.010-0.0135 post-hoc gain is **the missing final decay (about 0.0102-0.0110) plus extra tokens (about 0.0015-0.0027)**, which adds up to the observed 0.0129 and 0.0125 within 0.0011; a main run that decays to zero gets the decay for free, after which a further low-rate phase adds less than 0.001. The longer phases (3,200 and 6,400 steps) keep adding about 0.002 per doubling because they add tokens: 6,400 steps is 52M tokens, 70% of the main run. The fair gap is flat at 0.024-0.025 in every row for both models, so no schedule choice moves the comparison.
**Decision: every future run uses `--lr-floor 0`; the post-hoc phase is not part of the default recipe.** (The demo's F7 models keep their floor-0.1 + 1,600-step checkpoints, which are better than a floor-0 main run: 0.9524 and 0.9547 against 0.9558 for seed 0.)

### Item 5: the learning-rate screen at the new shape (30M characters, five seeds, change against 6e-2)
| learning rate | F7 (warm-up 600) | per seed | H3 (warm-up 100) | per seed |
|---|---|---|---|---|
| 3e-2 | 1.0758 (**+0.0171**) | +0.0204 +0.0173 +0.0203 +0.0138 +0.0135 | 1.0241 (+0.0041) | +0.0020 -0.0015 +0.0016 +0.0089 +0.0097 |
| **6e-2** | **1.0588** | | **1.0199** | |
| 1e-1 | 1.0589 (+0.0001) | +0.0046 +0.0006 -0.0004 -0.0059 +0.0017 | 1.0234 (+0.0034) | -0.0017 -0.0004 +0.0051 +0.0082 +0.0059 |
Predictions: 6e-2 within 0.003 of the best of the three, F7 (60%): **right** (the best is within 0.0001) and H3 (60%): **right** (6e-2 is the best); 3e-2 worse than 6e-2 for F7 by at least 0.003 (60%): **right** (+0.0171, every seed); 1e-1 beats 6e-2 for F7 by 0.003 (25%): **wrong** and for H3 (25%): **wrong**; the best learning rate is the same for both (55%): **right** (6e-2); the gap at each model's best learning rate within 0.004 of 0.0389 (70%): **right** (0.0389); a hybrid run at 1e-1 diverges on at least one seed (20%): **no** (none).
Decision: no learning rate beats 6e-2 by 0.005 on four of five seeds for either model: **6e-2 stands for both**; nothing goes to a 300M confirmation. The fuzzy model is much more sensitive below 6e-2 (+0.0171 at 3e-2, against +0.0041 for the hybrid) and flat above it; the fairness claim holds on this axis.

## Items 2, 3 and 4 registered (2026-10-22, before they run; order as the owner set it)

### Item 2: coherence and repetition of the shipped configuration (set 26)
Groups, two checkpoints each (the re-annealed 300M checkpoints `outputs/v74-ul/reann_F7_1600_3e-3{,_s1}.pt` and `reann_H3_1600_3e-3{,_s1}.pt`: 0.9524 / 0.9547 and 0.9277 / 0.9319), at the baseline sampler (S0: temperature 0.5, top-k 10) and with DRY (S8), 16 texts per checkpoint per group, 32 per group per rater (128 per rater, two blind sub-agent raters, scores before the key); the repetition metrics and the plausibility of the generated text (`flm.repeat_metrics`) for 96 texts per group.
Predictions: DRY cuts the repeated-trigram share by at least 40% for both models (75%); DRY changes the rated total by at most 0.5 on both raters for both models (60%); F7 is within 1.0 of H3 on both raters (75%) and below it on both (55%; sets 21, 23 and 24 had the fuzzy model 0.2-0.5 below); the re-annealed F7 is within 0.5 of the same model before the final phase (set 24: 7.12 / 7.12) on both raters (65%).

### Item 3: does the new shape generalise to SimpleStories? (30M characters, seeds 0-4, `SHAPE_CORPUS=simplestories`)
A BPE6144 and its frequency file are trained on the SimpleStories stream (`flm/prep_simplestories_bpe.py` with a vocabulary argument; characters per token and head coverage are reported). Arms on SimpleStories: the all-fuzzy F7 shape and the hybrid H3 shape (V6144; F7: head 640, tail 8, d32, 61 / 41 rules; H3: head 640, tail 8, d40, 38 rules) at lr 6e-2 and the models' own warm-ups (600 and 100), against the existing SimpleStories V3072 arms (`v67-ss30`: all-fuzzy `fz_lr6` 1.2984, hybrid `hyb_lr6` 1.2443, five seeds).
Predictions: F7's shape beats the old-shape all-fuzzy model on SimpleStories by at least 0.020 on every seed (65%; the TinyStories gain was 0.061); H3's shape beats the old-shape hybrid by at least 0.015 (55%; 0.035 on TinyStories); the all-fuzzy model's gap to the hybrid at the new shapes is between 0.015 and 0.050 (65%; 0.054 at the old shape on SimpleStories, 0.039 on TinyStories at the new shapes); the gain of the shape is smaller on SimpleStories than on TinyStories for the fuzzy model (60%: a richer corpus has a longer tail of rare words, but the tiered embedding is built for it, so this is a toss-up).

### Item 4: a third seed at 300M with the final phase from item 1 (floor 0)
F7 and H3 at 300M, warm-up 1000, `--lr-floor 0`, seeds 1 and 2 (seed 0 exists: 0.95577 and 0.93034), `outputs/v70-shape300` and `outputs/v72-hybshape300` with the suffix `_floor0`; four cells of 12.5 minutes.
Predictions: the floor-0 gain over floor 0.1 holds on seed 1 (F7 against 0.96864, H3 against 0.94683) at at least 0.007 for both models (65%); the three-seed gap mean is within 0.003 of 0.0254 (70%) with a standard deviation of at most 0.004 (65%); the three-seed means are F7 0.9576 (within 0.003, 65%) and H3 0.9317 (within 0.003, 65%).

## Item 2 results: coherence and repetition of the shipped configuration (set 26; recorded 2026-10-22; both score files written before the key was read; re-annealed 300M checkpoints, two per model, 16 texts per checkpoint per group, 32 per group per rater; repetition metrics on 96 texts per group, `outputs/v76-repeat26`)
| group | repeated trigrams | plausibility (nats per token) | rater A | rater B | plot |
|---|---|---|---|---|---|
| F7 re-annealed, S0 (T0.5, top-k 10) | 0.088 | 1.925 | 7.06 (0.32) | 7.12 (0.29) | 2.09 |
| F7 re-annealed, S8 (DRY) | 0.043 (-51%) | 1.958 (+0.033) | 7.03 (0.32) | 6.88 (0.33) | 2.16 |
| H3 re-annealed, S0 | 0.103 | 1.745 | 7.81 (0.31) | 7.91 (0.27) | 2.38 |
| H3 re-annealed, S8 (DRY) | 0.049 (-52%) | 1.803 (+0.058) | 7.62 (0.28) | 7.75 (0.28) | 2.30 |
Per checkpoint (A / B): F7 S0 7.38 / 7.38 and 6.75 / 6.88; F7 S8 6.88 / 6.75 and 7.19 / 7.00; H3 S0 7.56 / 7.69 and 8.06 / 8.12; H3 S8 7.19 / 7.25 and 8.06 / 8.25. DRY effect on the rating (S8 minus S0, A / B): F7 -0.03 / -0.25, H3 -0.19 / -0.16. F7 minus H3 at S0: -0.75 / -0.78; at S8: -0.59 / -0.88. Rater correlation 0.82. (The plausibility judge is the hybrid H3 before the final phase, so its scores for the hybrid's own texts are flattering.)
Predictions: DRY cuts the repeated-trigram share by at least 40% for both models (75%): **right** (-51%, -52%); DRY changes the rated total by at most 0.5 on both raters for both models (60%): **right** (at most 0.25); F7 within 1.0 of H3 on both raters (75%): **right**; below it on both (55%): **right**; the re-annealed F7 within 0.5 of the same model before the final phase, set 24 (7.12 / 7.12) (65%): **right** (7.06 / 7.12).
Reading: DRY halves the repeated trigrams of the shipped configuration for both models at no rated cost and about 0.03-0.06 nats of plausibility; it is the right default. The gap in rated coherence between the models widened from 0.2-0.5 (set 24, before the final phase) to about 0.75 here: the final phase raised the hybrid's rating (7.31 / 7.59 to 7.81 / 7.91) and left F7's unchanged (7.12 / 7.12 to 7.06 / 7.12); the difference of the two gaps is about 1.8 standard errors in one set, so it is a lead and not a finding. Repeated word trigrams are *higher* for the hybrid at the baseline sampler (10.3% against 8.8%), so the hybrid's higher rating is not a repetition effect.

## Item 4 results: floor-0 seeds 1 and 2 at 300M (recorded 2026-10-22)
| seed | F7 | H3 | gap | F7 change vs floor 0.1 | H3 change vs floor 0.1 |
|---|---|---|---|---|---|
| 0 | 0.95577 | 0.93034 | 0.0254 | -0.0102 | -0.0110 |
| 1 | 0.95719 | 0.93450 | 0.0227 | -0.0115 | -0.0123 |
| 2 | 0.95819 | 0.93029 | 0.0279 | | |
| mean | 0.95705 | 0.93171 | 0.0253 (sd 0.0026) | | |
Predictions: the floor-0 gain holds on seed 1 at >= 0.007 for both (65%): **right** (0.0115, 0.0123); three-seed gap mean within 0.003 of 0.0254 (70%): **right** (0.0253); sd at most 0.004 (65%): **right** (0.0026); F7 mean 0.9576 within 0.003 (65%): **right** (0.9571); H3 mean 0.9317 within 0.003 (65%): **right** (0.9317). The fair gap at the floor-0 recipe is 0.0253, about 2.6%.

## Item 3 results: the shape on SimpleStories (30M characters, seeds 0-4, BPE6144 trained there)
| model | per seed | mean | old-shape (V3072) mean | gain |
|---|---|---|---|---|
| F7 shape | 1.2335 1.2383 1.2388 1.2384 1.2358 | 1.2369 | 1.2984 | 0.0615 |
| H3 shape | 1.2033 1.1987 1.2021 1.2067 1.2015 | 1.2025 | 1.2443 | 0.0418 |
Gap at the new shapes 0.0344 (0.054 at the old shape). Predictions: F7 gain >= 0.020 (65%): **right** on the mean (per-seed not checked); H3 gain >= 0.015 (55%): **right**; gap between 0.015 and 0.050 (65%): **right**; F7's gain smaller on SimpleStories than on TinyStories (60%): **wrong** (0.0615 against 0.061, the same). The shape generalises to a second corpus and helps the fuzzy model more than the hybrid there too.
