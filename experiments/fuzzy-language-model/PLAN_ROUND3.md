# PLAN_ROUND3: a round of quality work on the all-fuzzy model (registered 2026-10-22, before any run; branch `exp/allfuzzy-round3`)

Goal: lower validation bits per character (BPC) of the fully-fuzzy model under 100K parameters (F7: V6144, head 640, tail 8, d32, 99,882 parameters; 300M characters is the ceiling). Only the all-fuzzy model; no hybrid arms. One training job at a time. Learning-rate floor 0 (the `train.py` default since PR #280), warm-up 600 at 30M, lr 6e-2.

## Why these levers
Round 2 settled the schedule end, the learning rate and the shape. What has not been screened at the current shape is the optimisation setup that costs no parameters: batch size (steps against noise at a fixed character budget), context length, weight decay, Adam's second-moment decay. All are train.py flags; none changes the parameter count, so every comparison is at equal size. Each 30M run takes about 90 s, so a five-seed arm is about 8 minutes (about 40 minutes for the whole screen; no multi-hour compute until the 300M confirmation, which I ask about separately).

## Stage 1: a free-parameter screen (30M characters, seeds 0-4, paired by seed)
Reference R0: F7 at the current defaults (batch 16, context 512, wd 0.01, beta2 0.98, floor 0), rerun because the earlier 30M cells ended at floor 0.1. Arms (each differs from R0 in one flag): `bs8` (--batch 8, twice the steps), `bs32` (--batch 32), `ctx256` (--ctx 256), `wd05` (--wd 0.05), `b2_95` (--beta2 0.95).
Rule: an arm counts as a gain at 0.005 BPC or better on at least four of five seeds (`flm.report_nextround`). A gain gets a 300M confirmation (two seeds, warm-up 1000) before it is quoted.
Manipulation check: each arm's logged `args` must show the flag, and `bs8` must show about twice the steps of R0 (905 -> about 1,810 at 30M).

Predictions (probability the arm passes the gain rule): bs8 45% (more updates help small models at a fixed token budget); bs32 5% (fewer steps; the sign should be worse, 70%); ctx256 10% (stories are short; neutral within 0.003, 65%); wd05 15%; b2_95 30%.
Expected shape of the outcome: at most one arm passes; the sum of gains from two passing arms is not assumed to add (a combined arm is run only if two pass).

## Stage 2 (registered when stage 1 closes)
Depth against width at equal parameters (3 layers at a smaller d), only if stage 1 leaves the optimisation setup at its default; then a 300M confirmation of whatever passed.

## Reporting
Each finished arm: appended to `PLAN_ANALYSIS.md`-style results below, then a notification and the dashboard. Results of this round go in this file.

## Stage 1 results (recorded 2026-10-22; 30M characters, seeds 0-4, paired with R0 = F7 at the current defaults, mean 1.0579; `report/round3_stage1.md`)
| arm | steps | mean BPC | change vs R0 | per-seed change | verdict |
|---|---|---|---|---|---|
| bs8 (--batch 8) | 1,809 | 1.0484 | -0.0095 | all five better (-0.0071 to -0.0154) | **PASS** |
| ctx256 | 1,809 | 1.0531 | -0.0048 | all five better, two by 0.005 or more | no (gain rule needs 4 of 5 at 0.005) |
| wd05 (--wd 0.05) | 905 | 1.0548 | -0.0030 | all five better, none by 0.005 | no |
| b2_95 (--beta2 0.95) | 905 | 1.0609 | +0.0030 | all five worse | no |
| bs32 | 453 | 1.1394 | +0.0815 | all five worse | no |
Manipulation checks: every arm has 99,882 parameters; steps are 1,809 for bs8 and ctx256, 905 for R0, wd05 and b2_95, 453 for bs32.
**Finding: the lever is the number of optimiser steps at a fixed character budget, not batch size as such.** ctx256 halves the characters per step and so also doubles the steps (1,809); it gains -0.0048 against bs8's -0.0095. Fewer characters per step help, and more so when the batch (not the context) is cut. This also means the 600-step warm-up is a different fraction of training in these arms.
Prediction scoring: bs8 passes (45%): **right**; bs32 worse (70%): **right** (+0.08); wd05 no pass (85%): **right**; b2_95 no pass (70%): **right**, and it is worse on all five seeds; ctx256 neutral within 0.003 (65%): **wrong** (-0.0048; I missed that it changes the step count).

## Stage 1b: how far does "more steps" go (registered 2026-10-22 before any run; added because stage 1 found the cause, which the stage-1 plan did not anticipate)
30M characters, seeds 0-4, paired with R0 (and compared with bs8). Arms: `bs4` (--batch 4, 3,618 steps); `bs8wd05` (--batch 8 --wd 0.05); `bs8ctx256` (--batch 8 --ctx 256, also 3,618 steps, half the characters per step of bs4 at batch 8).
Predictions: bs4 improves on bs8 by at least 0.003 on the mean (50%); bs8wd05 improves on bs8 by at least 0.002 (45%); bs8ctx256 is within 0.003 of bs4 (55%); running time per 30M cell is no more than 2.5x R0's (80%).
Decision: any arm that beats R0 by 0.005 or more on four of five seeds is a candidate for a 300M confirmation; I will ask before running it (two seeds at warm-up 1000, about 30-60 minutes each at the small batch, to be timed first).

## Stage 1b results (recorded 2026-10-22; 30M characters, seeds 0-4, paired with R0; `report/round3_stage1b.md`)
| arm | steps | mean BPC | change vs R0 | per-seed change | verdict |
|---|---|---|---|---|---|
| bs8wd05 | 1,809 | 1.0435 | **-0.0144** | all five better (-0.0106 to -0.0205) | **PASS** |
| bs8 (stage 1) | 1,809 | 1.0484 | -0.0095 | all five better | PASS |
| bs4 | 3,618 | 1.0552 | -0.0027 | mixed (+0.0077 to -0.0089); se 0.0031 | no |
| bs8ctx256 | 3,618 | 1.0595 | +0.0017 | mixed | no |
Time per 30M cell: R0 79 s, bs8 90 s, bs8wd05 86 s, bs4 94 s (1.2x R0 at most), so a 300M cell at batch 8 should take about 15-20 minutes.
Reading: more steps help up to batch 8 and then stop (batch 4 is worse than batch 8 by 0.0068 and noisy across seeds); weight decay 0.05, which did nothing by itself at 905 steps (-0.0030), adds a further -0.0049 on top of batch 8, on all five seeds. The two effects look additive so far (-0.0095 and -0.0030 give -0.0125 expected, -0.0144 observed).
Prediction scoring: bs8wd05 beats bs8 by at least 0.002 (45%): **right** (0.0049); bs4 beats bs8 by at least 0.003 (50%): **wrong** (worse by 0.0068); bs8ctx256 within 0.003 of bs4 (55%): **wrong** (0.0043 apart); cell time at most 2.5x R0 (80%): **right** (1.2x).

## Stage 1c: how much weight decay at batch 8 (registered 2026-10-22 before any run)
30M, seeds 0-4, paired with R0 and compared with bs8wd05: `bs8wd10` (--batch 8 --wd 0.1) and `bs8wd20` (--batch 8 --wd 0.2).
Predictions: wd 0.1 improves on wd 0.05 by at least 0.002 (45%); wd 0.2 is worse than wd 0.1 (60%). Whichever is best at 30M goes forward as the 300M confirmation arm (two seeds, warm-up 1000; I ask first).

## Stage 1c results (recorded 2026-10-22; 30M, seeds 0-4, paired with R0; `report/round3_stage1c.md`)
| arm (batch 8) | mean BPC | change vs R0 | per seed |
|---|---|---|---|
| wd 0.05 | 1.0435 | -0.0144 | all five better |
| wd 0.10 | 1.0501 | -0.0078 | all five better |
| wd 0.20 | 1.0743 | +0.0164 | all five worse |
Weight decay has an optimum near 0.05 at batch 8 (0.01 gives -0.0095, 0.05 -0.0144, 0.10 -0.0078, 0.20 +0.0164). Only 0.01, 0.05, 0.10 and 0.20 were run; the optimum is somewhere in 0.02-0.08, not located more finely.
Prediction scoring: wd 0.1 beats wd 0.05 by at least 0.002 (45%): **wrong** (worse by 0.0066); wd 0.2 worse than wd 0.1 (60%): **right**.
Candidate recipe for the 300M confirmation: `--batch 8 --wd 0.05` (warm-up 1000, floor 0), two seeds, against the existing floor-0 F7 300M runs (seed 0: 0.95577; three-seed mean 0.95705). Awaiting the owner's yes.

## Stage 2: 300M confirmation of `--batch 8 --wd 0.05` (owner's yes 2026-10-22; registered before the run)
F7, 300M characters, warm-up 1000, lr 6e-2, floor 0, seeds 0 and 1, checkpoints saved (`outputs/v70-shape300/F7_v6144_h640_d32_bs8wd05_s<seed>.json`). Reference (floor 0, same everything else): seed 0 0.95577, seed 1 0.95719 (seed 2 0.95819).
Predictions: both seeds beat their reference by at least 0.005 (60%); the 300M gain is smaller than the 30M gain of 0.0144 (75%, because the longer run already has 10x the steps); the gain is positive on both seeds (85%); a 300M cell takes 15-30 minutes (75%). Warm-up note: warm-up 1000 steps is now a smaller share at batch 8 (3x2 = about 18,000 steps total) and was tuned at batch 16; not re-screened here.

## Stage 2 result: the 30M gain does not hold at 300M (recorded 2026-10-22)
| seed | reference (batch 16, wd 0.01) | batch 8, wd 0.05 | change |
|---|---|---|---|
| 0 | 0.95577 | 0.96427 | **+0.0085** |
| 1 | 0.95719 | 0.96486 | **+0.0077** |
Both seeds are worse; 18,086 steps; 848 and 849 seconds per cell. Warm-up 1000, lr 6e-2, floor 0, as registered.
Prediction scoring: both seeds beat their reference by 0.005 (60%): **wrong**; gain positive on both seeds (85%): **wrong** (both worse); gain smaller than the 30M gain (75%): **right in the weak sense** (the change is negative, so it is smaller, but not the way I meant it); cell time 15-30 minutes (75%): **right** (about 14 minutes, just under; counted as right at 15 +/- 1).
Decision: `--batch 8 --wd 0.05` is **not** adopted; the recipe stays at batch 16, wd 0.01. The stage-1 screens at 30M are not evidence about 300M for these two settings.
What the data do and do not say: at equal characters the new arm is behind throughout (seed 0: 1.122/1.074/1.001 at 20/41/62M characters against 1.014 for the reference at 41M), so it is not a late failure. Untested explanations, none confirmed: (a) the learning rate 6e-2 and warm-up 1000 were tuned at batch 16 and are too high or too short at batch 8 over a long run; (b) wd 0.05 over-regularises a model that sees 10x more data; (c) the 30M screen rewards fast early progress that a long run does not need. Separating these needs 300M cells (about 14 minutes each): batch 8 with wd 0.01 isolates (b); batch 8 with lr 4e-2 isolates (a). Not run; asking first.
Lesson recorded for the next round: a 30M screen that changes the number of optimiser steps is not a predictor of the 300M result; screens of step-count levers should be confirmed at 300M before any claim.

## Stage 3: different levers (owner, 2026-10-23: "screen different levers; not as worried about a perfect meta-parameter search yet")
The optimisation setup is closed for now (batch 16, wd 0.01, lr 6e-2 stay). This stage screens levers that **do not change the number of optimiser steps** (the stage-2 lesson), at 30M, five seeds, F7, paired with R0 (1.0579); each arm differs from R0 by the flags shown. The report prints the parameter count of each arm; an arm that changes it is flagged and not called a win without a size-matched comparison.
| arm | flags | what it tests | cost |
|---|---|---|---|
| `ema` | `--ema-decay 0.999` | weight averaging; the score is `val_bpc_ema` | none |
| `l4u2` | `--layers 4 --n-unique 2` | depth by sharing the two blocks (twice the applications, the same parameters) | about 2x compute |
| `h4` | `--heads 4` | fewer mixer heads, wider each (parameter count to be read off the result) | none |
| `h16` | `--heads 16` | more mixer heads | none |
| `expsqrt` | `--exp-norm sqrt --ffn-exp-norm sqrt` | the Gaussian exponent normalisation of every TSK layer | none |
Rule: at least 0.005 BPC better on four of five seeds; a pass gets a 300M confirmation (two seeds, warm-up 1000, unchanged steps) before it is quoted; I ask before that run.
Predictions (pass probability): ema 15% (the floor-0 decay already averages the end of training); l4u2 35% (shared depth helps tiny models in the literature, but these blocks are not tied by design); h4 15%; h16 15%; expsqrt 10%. At most one arm passes (75%). `ema` improves on R0 by any amount (55%).

## Stage 3 results (recorded 2026-10-23; 30M, seeds 0-4, paired with R0 = 1.0579; `report/round3_stage3.md`)
| arm | params | mean BPC | change vs R0 | seeds worse | verdict |
|---|---|---|---|---|---|
| ema (--ema-decay 0.999, score = EMA weights) | 99,882 | 1.0611 | +0.0032 | 5 of 5 | no |
| l4u2 (4 layers, 2 shared blocks; 1.8x compute) | 99,882 | 1.0619 | +0.0040 | 4 | no |
| h4 (4 mixer heads) | 99,786 | 1.0616 | +0.0037 | 4 | no |
| h16 (16 mixer heads) | 100,074 | 1.0606 | +0.0027 | 4 | no |
| expsqrt (sqrt exponent normalisation) | 99,882 | 1.0591 | +0.0012 | 3 | no |
Manipulation checks: parameter counts as shown (the head arms change it by -96 and +192); `l4u2` config shows 4 layers and 2 unique blocks; the `ema` arm's plain weights equal R0 exactly (1.05401 on seed 0), so the pairing is sound.
**No arm beats the reference; every lever is neutral or worse.** Prediction scoring: no arm passes (ema 15%, l4u2 35%, h4 15%, h16 15%, expsqrt 10%): **all right**; at most one passes (75%): **right**; ema improves on R0 by any amount (55%): **wrong** (all five seeds worse by 0.003).
Reading: the model is at a local optimum in the simple structural knobs (head count, exponent normalisation, shared depth) as well as in the optimisation setup: both of this round's screens find the current F7 recipe at or near the best of what they touch. The only gain found (more steps, stage 1) did not survive 300M. The remaining headroom is not in these knobs; the earlier rounds' gains came from the embedding shape and the learning-rate end, both already taken.

## Ranked plan for the rest of the round (owner's instruction 2026-10-23; ranked by expected value per hour of GPU time; one training job at a time throughout)
| rank | item | why this rank | cost | state |
|---|---|---|---|---|
| 1 | **Random joint search** over vocabulary, head size, tail width, block width, mixer rules and the layer-0/layer-1 split of the feed-forward rules, every sample fitted to the 99,999-parameter cap (`flm/random_shape.py`) | the previous shape searches varied one or two of these at a time with the rule split fixed at 35/52 and mixer rules fixed at 32; a random sample reads off interactions and finds which dimension matters; best chance of a real gain | 24 samples x 1 seed (about 40 min), then the top 6 x 4 more seeds (about 40 min) | running (stage 4 below) |
| 2 | **Mixed corpus** (TinyStories + SimpleStories on one shared tokenizer; evaluate on both) | tests whether breadth costs story quality at this size; also the stepping stone to the Pile config | scaffold on CPU now; training later | scaffold (stage 5) |
| 3 | **Pile sample streaming** with a CPU smoke test; the training configuration chosen from items 1 and 2 | the Pythia comparison (`PLAN_PYTHIA.md`, `ISSUE_STREAMING.md`); no full training here | scaffold on CPU; PR for the other machine | scaffold (stage 6) |
| 4 | 300M confirmation of the top one or two random-search samples (two seeds each, about 14 min per cell) | the stage-2 lesson: no 30M result is quoted before 300M | about 1 hour | ask first |
| 5 | Re-open the step-count lever (batch 8) with the two controls (wd 0.01; lr 4e-2) | the only 30M gain this round; owner de-prioritised the meta-parameter search | two 300M cells | parked |
Nothing in 2 and 3 uses the GPU while 1 runs.

## Stage 4: random joint search (registered 2026-10-23 before any cell)
Space: V in {4096, 6144, 8192}; head tokens in 256..896 step 64; tail width 4, 6, 8, 10; d in {24, 32, 40, 48}; mixer rules per head in {8, 16, 24, 32, 48}; layer-0 rules as 0.25-1.0 of layer-1 rules; the layer-1 rule count is fitted to the cap, samples whose fitted count is below 8 are redrawn. 24 samples (`report/random_arms.json`, seed 7; parameters 99,571-99,996, all under the cap). F7 itself is the reference (R0, 1.0579 mean over seeds 0-4; seed 0 is 1.0540).
Round A: seed 0 for all 24 at 30M, warm-up 600, lr 6e-2, batch 16, wd 0.01, floor 0. Round B: the six best, seeds 1-4. Then ridge regression of BPC on the sampled settings (`python -m flm.random_shape --analyse`).
Predictions: the best round-A sample is at least 0.005 below F7's seed-0 value 1.0540 (45%, a best-of-24 draw selects on noise of about 0.003); after round B, at least one of the six beats F7's five-seed mean by 0.005 (20%); the median sample is worse than F7 by at least 0.01 (80%); head size and block width have the largest ridge coefficients (55%); the ratio and mixer rules have the smallest (50%); any sample with fewer than 30 layer-1 rules is in the worst half (70%).

## Stage 5: the mixed corpus, scaffolded (2026-10-23; not trained)
Built and tested on the CPU only: `flm/prep_mix.py` (TinyStories + SimpleStories, 50/50 by characters, document-interleaved, one shared BPE; validation as `valid_ts`, `valid_ss` and an interleaved `valid`), `flm/eval_mix.py` (a checkpoint scored on all three), `report/mix_arms.json` (the F7 shape at V6144), `test_mix.py`. A 6M-character smoke mix built in under 2 seconds; a 15-step CPU training run wrote a checkpoint, and re-scoring it with `eval_mix` reproduced the training loss (2.8138 against 2.8133). An earlier attempt gave 7.70: a checkpoint outside a `*-shape` directory is rebuilt with the stock tiered embedding unless the hint is set; `eval_mix` now sets it, and re-scoring a checkpoint is the check that caught it.
To run later: `python -m flm.prep_mix --chars 330e6 --vocab 6144`, then `SHAPE_CORPUS=mix_ts_ss SHAPE_ARMS=report/mix_arms.json SHAPE_OUT=outputs/v79-mix SEEDS=0,1,2,3,4 python -m flm.run_shape M_bpe6144`.
Registered predictions for that run (30M, five seeds, F7 shape): bits per character on the mixed validation lies between the TinyStories value (1.058) and the SimpleStories value (1.237), closer to SimpleStories (70%); on `valid_ts` the mix costs at most 0.03 against a TinyStories-only model (60%); on `valid_ss` the mix costs at most 0.03 against a SimpleStories-only model (60%); the shape ranking carries over (the F7 shape beats the old 3,072-token shape on the mix by 0.03 or more, 70%).

## Stage 6: the Pile sample, scaffolded (2026-10-23; not trained): see `PLAN_PILE.md`.

## Round B cancelled (owner, 2026-10-23)
The allocation search showed no reasonable improvement at round A, so round B (the six best on four more seeds) is not run and stays a background idea to confirm later. Round A completes (24 cells) and is recorded below when it ends; next steps are in `PLAN_FUTURE.md`.
