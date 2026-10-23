# PLAN_M14_EXTEND: 300M more characters for the TinyStories 14M model, in bf16 (owner request 2026-10-23; registered before the runs)

Owner: "go with the TinyStories one, 300M more characters in bf16, I want to see how good we can get it." The 300M-character ceiling is per run, so this is a second 300M-character run warm-started from the first (600M characters of training in total, two schedules), not a 600M run.

## Setup
- Start: the weights of `outputs/v82-m14-300/M14_v6144_h1024_d384_l4_lr1e-2_m14_300M_s0.pt` (val 0.6683 bits per character, lr 1e-2, warm-up 1000, floor 0, fp32). That file holds weights only (no optimiser state), so this is a **warm start** with a fresh Adam and a fresh warm-up/decay schedule (`--init-from`, existing in `flm/train.py`), not a resume.
- Same data (TinyStories train stream, BPE6144), shape, seed 0, batch 16, context 512, wd 0.01, floor 0. The second pass over the same stream sees the same text again (the stream is about 470M characters, so 300M more characters is about 0.6 epoch past the first run's end; the data is not exhausted but it is a repeat of what the model has seen). Validation is held out, so the number is honest; the risk is memorisation, watched through train-vs-val.
- bf16 autocast (`--precision bf16`), measured 1.16-1.22x in `PLAN_PRECISION.md`; fp32 val is logged beside it (`val_bpc_fp32`). Checkpoint every 1000 steps (`--ckpt-dir`, resume safe), eval batch 16 (shared-VRAM rule), one GPU job at a time.
- Output: `outputs/v90-m14-ext/`.

## Step 1: learning-rate probe from the warm start (3M characters each, warm-up 20, about 1 minute each)
lr 1e-3, 3e-3, 1e-2, from the same weights. Reading: the lowest final val; a rate that lands above 0.6683 after 91 steps only says it perturbs the model, so I also look at which rate is least damaged, not just the lowest. A guard against a wasted 35 minutes, not a tuning claim.

## Step 2: the 300M-character run
At the probe's best rate (ties go lower), warm-up 1000, floor 0, bf16, about 9,000 steps.

## Predictions
- The probe's best rate is 3e-3 (45%), 1e-3 (30%), 1e-2 (25%); no probe diverges (90%).
- The extension ends below 0.6683 (95%), below 0.65 (80%), below 0.62 (50%), below 0.60 (20%). Basis: 30M -> 300M took 0.945 -> 0.668; the next doubling of total characters should give less.
- The gap between train (EMA) and val stays under 0.15 bits per character in the same units as the first run's end (65%); val never rises for two consecutive evaluations (75%).
- bf16 steady speed at least 1.15x the fp32 run's 31,500 tokens per second (65%), so the run takes at most 36 minutes (65%). bf16 val and fp32 val of the final model differ by less than 0.005 (85%).

## Not in scope
No third pass, no other seeds, no learning-rate tuning beyond the probe; whether the demo model is replaced by the result is for the owner after reading it.

## Probe result (recorded 2026-10-23; `outputs/v90-probe/`; 3M characters, 91 steps, warm-up 20, bf16, from the 0.6683 weights)
lr 1e-3 **0.6725**, 3e-3 0.6823, 1e-2 0.7103 (fp32 re-evaluation within 0.0002 of the bf16 number each time); none diverged. All three are above the starting 0.6683, as expected for a restart at a nonzero rate with no time to decay; the order says the lower the rate the less the weights are disturbed. Prediction scoring: best is 1e-3 (30%): **happened**; no divergence (90%): **right**. The rule written above (lowest probe val) picks **1e-3** for the run; caveat registered here: 91 steps favour the smallest rate, so if the run plateaus early, a rerun at 3e-3 is the obvious next screen.
Run: `outputs/v90-m14-ext/`, lr 1e-3, warm-up 1000, floor 0, bf16, batch 16, seed 0, checkpoint every 1000 steps, `--resume auto`.

## Result of the lr 1e-3 extension (recorded 2026-10-23; `outputs/v90-m14-ext/`)
300M more characters, bf16, lr 1e-3, warm-up 1000, floor 0, seed 0, 9,043 steps: validation **0.6576** bits per character (fp32 re-evaluation 0.6574), from 0.6683: **-0.0107**. Curve: 0.6636 at step 5,000, 0.6576 at 9,043. No divergence; the train EMA ends at 2.66 (the first run ended at 2.71, same units), so no sign of overfitting. Speed: 39,600 tokens per second, **1,871 s (31 minutes)** against 2,350 s in fp32: **1.26x** from bf16; peak memory 5.3 GB.
Prediction scoring: ends below 0.6683 (95%): **right**; below 0.65 (80%): **wrong** (0.6576); below 0.62 (50%) and 0.60 (20%): **wrong**; train-val gap and no rising val (65%, 75%): **right**; bf16 at least 1.15x and at most 36 minutes (65%, 65%): **right** (1.26x, 31 min); bf16 vs fp32 val within 0.005 (85%): **right** (0.0002).
Reading: doubling the total characters gave 0.011, far below the 0.28 that 10x gave at the first step, so at this learning rate the model is on the flat part of its data curve, or the restart rate was too low to leave the basin it is in (the probe's bias, registered above). The two readings predict different things for the next run.
Next, registered before it runs (`outputs/v91-m14-ext3/`): the same extension from the same weights at **lr 3e-3**. Predictions: ends below 0.6576 (55%), below 0.650 (30%), above 0.6683 (10%); the curve is above the 1e-3 run's at step 5,000 (65%) and below it at the end only if the rate matters. If 3e-3 does not beat 1e-3 by 0.005, the lever is not the rate, and more characters of this corpus are not worth buying; capacity or data is next.
