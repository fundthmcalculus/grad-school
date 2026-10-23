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
