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
