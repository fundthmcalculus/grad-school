# Plan: tuning the ~96K-parameter model on the CPU (GPU session, 2026-10-11)

The user's request: build plans and tests that can run on the CPU, focused on the ~96K models, and tune them.
Scope: **nothing above 99,999 parameters.** The reference cell is BPE1024 D, d 48, 2 layers, 8 heads, 32 rules per head,
FFN mult 1.5, tsk-dim 16, shortconv 8, ctx 512 / batch 16, lr 6e-2, warm-up 100, 96,416 parameters
(1.1095 +/- 0.0048 BPC at 30M characters, 5 seeds; 0.9820 at 1B). The GPU is busy with the 3B-character run (`v18-3b`),
so everything here is on the CPU, one thread budget of 4 per job.

## What has and has not been tuned

Tuned (RESULTS.md): learning rate (interior optimum 6e-2 at BPE1024), context (512), rules per head (32-256 flat within
noise), vocabulary (best 1024-1536), width and depth taper, hybrids, weight sharing, factorised embedding, decoding.
**Never varied on the BPE model** (all sat at defaults chosen on the early character-level runs):
weight decay (0.01), AdamW beta2 (0.98), the lr floor of the cosine schedule (10% of peak), gradient clipping (1.0),
warm-up (100 steps, chosen ad hoc), and the layer/width split at fixed parameters (d 48 x 2 layers vs d 40 x 3 etc.),
heads, tsk-dim, FFN mult and shortconv width at the final vocabulary and context.
Weight decay and regularisation should matter little (one pass over the data, no overfitting, train ~ val),
which is itself a prediction below.

## What a CPU can and cannot do

Measured 2026-10-11 (4 threads, GPU job running alongside): **5.4K characters/s**, i.e. 4M characters = 12 min per job,
30M = 92 min per job; the GPU does 116K characters/s (21x). So the CPU can run *screens* (4M characters), not confirmations
that carry a claim, unless it is given many hours. Screens at 10M overstated vocabulary gains before (RESULTS, v7), so screens
here are used to **rank settings that act on the optimiser**, where small-budget and long-budget behaviour agree better than
they do for capacity changes; capacity changes (stage B) are re-checked at a larger budget before being believed.
Noise: set by the pilot (`v19-pilot`: the baseline, 3 seeds, 4M characters). A setting is called better only if it beats the
baseline by more than twice the pilot's seed standard deviation on the 3-seed mean.

## Pilot result (`v19-pilot`, 2026-10-11): the CPU is too slow to tune this way

Baseline, 4M characters, seeds 0/1/2, three jobs in parallel at 4 threads each: **1.4220 / 1.4399 / 1.4222 BPC, mean 1.4280, sd 0.0103**
(960-1,000 s per job). Two findings. (1) Parallel jobs do not add throughput: three jobs ran at ~1.8K chars/s each, the same 5.4K total as one job
alone, so the CPU is bandwidth- or thread-bound and the 12-minute-per-job figure above becomes ~16 minutes at any concurrency.
(2) The seed sd at 4M characters is 0.010, so a 3-seed screen can only see effects larger than ~0.02 BPC, bigger than every optimiser effect predicted below.
**Stage A as written (39 jobs) would take ~8 h of CPU and could not resolve its own predictions.** Recommended instead: run stage A on the GPU after
`v18-3b` finishes (39 jobs x ~45 s = ~30 min, at 30M characters 5 seeds ~2 min per job), and use the CPU only for the cheap parts (parameter counts, smoke checks).
Not launched; waiting on the user's choice.

## Stages (each registered before it runs; one knob changes at a time)

**Stage A: optimiser and schedule** (`v19-optA`, 13 cells x 3 seeds, 4M characters, 39 jobs)
lr floor {0.0, 0.1*, 0.3}; beta2 {0.95, 0.98*, 0.999}; grad clip {0.3, 1.0*, off}; warm-up {30, 100*, 300};
lr {4e-2, 6e-2*, 9e-2}; weight decay {0, 0.01*, 0.1}.
Predictions (written before any result): weight decay no effect (|d| < 0.005); lr floor 0.0 slightly better than 0.1
(at a short budget the tail of the schedule is a larger share of training); beta2 0.95 better than 0.999 at 4M characters
(few steps, so a faster second moment helps); clip and warm-up no effect; lr 9e-2 worse, 4e-2 worse (the interior optimum
holds at this budget).

**Stage B: capacity layout under 100K** (to be registered after stage A; needs a parameter count per cell, all <= 99,999)
layers x width {(L2, d48)*, (L3, d40), (L1, d64)}; heads {4, 8*, 16}; tsk-dim {8, 16*, 24}; FFN mult {1.0, 1.5*, 2.0} with d
nudged so the total stays under the cap; shortconv {4, 8*, 16}. Parameter counts are printed in the registration.

**Stage C: confirmation** of anything that wins stages A or B: 30M characters, 5 seeds, against the 5-seed baseline
(1.1095 +/- 0.0048), on the GPU once it is free (about 2 min per job) or on the CPU at 92 min per job. Only then does it
change the cited recipe; if it also holds at a longer budget it is run on the 1B stream.

## Rules for this track

* Nothing above 99,999 parameters; no change to the model structure in stage A (the three new flags `--lr-floor`,
  `--beta2`, `--clip` default to the old behaviour, so earlier runs are reproduced exactly).
* A screen result is a ranking, never a citable number (AGENTS.md: ten seeds for claims).
* Do not run more than three CPU jobs at once with four threads each while the GPU job is running (it needs one core).
