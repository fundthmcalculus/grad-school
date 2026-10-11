# Plan: 1.32M parameters on 1 billion characters (the size-times-data run)

CPU session, 2026-10-11, requested by the repo owner ("you can schedule the 1.32M x 1B run for after the current round of GPU work finishes"). **Needs the GPU session's user to approve exceeding the 99,999-parameter cap; the GPU session has said it will not run it without that.** This file is the complete job description so it can run unattended once approved.

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
