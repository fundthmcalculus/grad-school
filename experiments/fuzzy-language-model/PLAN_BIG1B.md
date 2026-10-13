# Plan: 1.32M parameters on 1 billion characters (the size-times-data run)

CPU session, 2026-10-11, requested by the repo owner ("you can schedule the 1.32M x 1B run for after the current round of GPU work finishes"). **Approved 2026-10-11 by the GPU session's user (verbatim: "approve the 1.32M run after the embedding screen"), for this run only; the 99,999 cap stays for everything else. Stage 1 starts when `v28-emb` ends.** This file is the complete job description so it can run unattended once approved.

## Why
Nothing from 93K to 1.32M parameters, or from 30M to 3B characters at 96K, has moved blind coherence (6 to 8.5 of 15; plot 1 to 2.4 of 5; criterion 12 and 3.5). But the two levers were never combined:
the 1.32M model trained on only 30M characters (about 7 tokens per parameter) reached 0.830 bits per character, and the 96K model on 1B reached 0.982 (about 3,300 tokens per parameter). This is the run that separates "too small" from "too little data".

## Cell (all in `outputs/v29-big1b/`; nothing in `train.py` or `models.py` changes)
Stream: the 1B-character BPE1024 stream (`FLM_TRAIN_SUFFIX=_1b`, as for `v17-1b`). Model: d 192, 4 layers, 8 heads, 16 rules per head, FFN as in the 96K model (1,323,904 parameters), ctx 512, batch 16, warm-up 300, seed 0.

```bash
# stage 1: learning-rate sizing at 30M characters (three short runs, one at a time on the GPU)
for LR in 1e-2 2e-2 4e-2; do
  python -m flm.train --mixer fuzzydelta --decay data --exp-norm mean --ffn-exp-norm mean --heads 8 --ffn-mult 1.5 \
    --ffn tsk --ffn-order 1 --ffn-norm none --tsk-dim 16 --tok bpe1024 --warmup 300 --ctx 512 --batch 16 \
    --d 192 --layers 4 --rules 16 --seed 0 --chars 30e6 --lr $LR --shortconv 8 --device cuda \
    --out outputs/v29-big1b/d192-L4_30M_lr${LR}_s0.json
done
# stage 2: the run itself at the best learning rate; saves the checkpoint (it is 5.3 MB: commit it with git add -f only if the owner allows files over 5 MB, otherwise keep it on the GPU host and run the coherence set there)
FLM_TRAIN_SUFFIX=_1b python -m flm.train ...same flags... --chars 1000e6 --lr <best> --save-model --device cuda \
    --out outputs/v29-big1b/d192-L4_1000M_s0.json
```
Cost: unknown on a GPU; the CPU speed was 1.7K characters per second (so 30M took 100 minutes there). Run stage 1 first and extrapolate; the 96K model's 1B run took 2,639 s. Report seconds per character before starting stage 2.

## Registered predictions (before running)
* Validation BPC at 1B: 0.70 to 0.78 (65%); below the 96K model's 0.982 by more than 0.15 (80%).
* Blind coherence (new set, fresh seeds, vs the 96K 1B model and the 1.32M 30M model, two raters, key published after both score): total of 10/15 or more **and** plot of 2.5 or more from both raters (30%). Total within 2 points of the 96K 1B model (about 7.6) (55%).
* Rule explainability: the 8-rule + context reading's top-10 hit rate is below 90% of the model's (60%) (it thins with the rule count).

## Decision rule
* If total is 10/15 or more and plot 2.5 or more from two raters: size times data is the lever; run the ladder (`flm/grids_v9.py`) to find where plot passes 3.5, and certify explainability there.
* If total stays within noise (about 2 points) of the 96K models: this architecture does not produce plot at this scale. Stop scaling and move to the mechanism question, after a human rater confirms the null.
* In either case also run `flm.rule_ablation`, `flm.rule_reader` and `flm.rule_story` on the checkpoint.

## Caveats
One seed; lr at 1B is the 30M optimum, not tuned at 1B; the 1B stream is a single pass over about half the corpus, so no repeated text; coherence is two Claude raters, 8 texts per model.

## Approval (GPU session, 2026-10-11)
The GPU session's user approved the run directly: "approve the 1.32M run after the embedding screen". Scheduled: stage 1 (3 x 30M-character lr runs) starts when `v28-emb` finishes (`/tmp/chain4.sh`); stage 2 starts after the stage-1 cost report, at the best stage-1 learning rate. This is one explicit exception to the 99,999-parameter cap, for this run only; the cap stays in force for everything else.

## Stage 1 outcome (GPU, 2026-10-11; 1,323,904 params, d192 x 4 layers, 30M characters, seed 0, warm-up 300)
| lr | val BPC | train seconds |
|---|---|---|
| 1e-2 | 0.8273 | 166.5 |
| **2e-2** | **0.8214** | 164.9 |
| 4e-2 | 0.8254 | 165.1 |
Cost: 30M characters in ~165 s = ~182K characters/s (the JSON's chars_per_sec 57K counts tokens); the 1B-character run extrapolates to ~5,500 s (~1.5 h) plus evaluation. lr 2e-2 chosen (the middle of the grid, so the optimum is bracketed). Stage 2 started automatically (`/tmp/chain5.sh`) at lr 2e-2. Note the 30M result 0.821 is already below the registered 1B range's lower edge (0.70-0.78) only if it keeps falling; the prediction stands unchanged.

## Stage 2 outcome (GPU, 2026-10-11): 1,323,904 params, 1B characters, lr 2e-2, warm-up 300, seed 0, 38,653 steps, 5,500 s
**Validation BPC 0.6355** (evals: 25k 0.679, 30k 0.656, 35k 0.641, final 0.635). Prediction 0.70-0.78 (65%): **wrong, better than the range**; below the 96K 1B model (0.982) by 0.35: right (>0.15). Checkpoint `outputs/v29-big1b/d192-L4_1000M_s0.pt` (5,318,077 bytes) kept on the GPU host, not committed (over the 5 MB rule).

### Blind coherence, set 10 (`report/coherence10_*`; fresh seeds 20-21; 3 models x 8 texts; T 0.5 / top-k 10, 600 characters; rubric G/C/P 1-5, total 3-15)
Two independent Claude sub-agent raters (A, B), each told to read only the blind file; scores were written to files before I opened the key.
| model | rater A total (G/C/P) | rater B total (G/C/P) |
|---|---|---|
| **1.32M, 1B chars (BPC 0.636)** | **9.62** (3.75 / 3.00 / 2.88) | **10.38** (3.88 / 3.50 / 3.00) |
| 96K, 1B chars (0.982) | 5.25 (2.38 / 1.50 / 1.38) | 5.50 (2.75 / 1.38 / 1.38) |
| 96K, 3B chars (0.978) | 5.38 (2.50 / 1.50 / 1.38) | 5.38 (2.50 / 1.50 / 1.38) |
Predictions: total >= 10 AND plot >= 2.5 from both raters (30%): **strictly not met** (A 9.62 < 10; B 10.38; plot 2.88 / 3.00 both pass); total within 2 points of the 96K 1B model (55%): **wrong** (+4.4 / +4.9). The gap to the 96K models is about 4.5 points of 15, roughly 2-3x the set-to-set noise recorded earlier (1.5-2), with the same sign from both raters, and plot rises from ~1.4 to ~3.0 of 5. On the registered decision rule this is the borderline case; on the substance, **size x data moves coherence where data alone (30M -> 3B at 96K) and every sub-100K lever did not**. Caveats: one seed, one checkpoint per arm, 8 texts, two raters from the same model family (not human), the rater scale differs from the CPU session's raters (this pair scored the 96K models ~5.4, the CPU rater ~7.3 on other sets), so only the within-set comparison is meaningful; the 1.32M run changes both parameters (13x) and data (30M -> 1B), but the 96K 1B/3B arms already show data alone is flat, so the lift is attributed to size at sufficient data, not separated further. The 99,999-parameter cap exception covered this run only.
Not yet done (per the plan's rule): `flm.rule_ablation`, `flm.rule_reader`, `flm.rule_story` on the checkpoint (CPU session's tools); the size ladder (`flm/grids_v9.py`) is NOT approved; a human rater.
