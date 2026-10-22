# PLAN_SCALEUP: the scaffold for a larger all-fuzzy model on a larger dataset (note for the future, written 2026-10-22; NOT STARTED, nothing here has been run)

Owner's instruction (2026-10-22): "Coherence we know only improves with larger model sizes. The objective now is to lay the scaffold for training a larger model on a larger dataset, eventually going to as large as a 50M parameter model like Pythia. Don't do that _yet_, just make a note for the future in a markdown document."

This is a note, not a registration: no predictions are registered, nothing is scheduled, and every number marked *(estimate)* or *(to verify)* is a planning figure, not a measurement. Measured facts cite their source file.

## 1. Why, and what the sub-100K work says about it

- **The coherence ceiling is size.** Across sets 12-24 the rated total for sub-100K models stays at 6-8.5 of 15 whatever the likelihood gain (sets 20-24: a likelihood difference of 0.025 to 0.048 bits per character is invisible to two raters on 32 texts); the larger models rated higher on TinyStories (the 1.32M anchor 10.0 in set 14 and 11.9-12.0 in set 19; the 2.22M model's own rating is in `PLAN_BIG2M.md`); on the broader Cosmopedia text even the 2.22M model rated only about 7 (`PLAN_CORPUS.md`). The only lever that moved coherence was parameters.
- **Where the all-fuzzy model stands at 99K (TinyStories, 300M characters):** 0.9673 bits per character (F7: vocabulary 6144, 640 crisp head tokens, tail rank 8, width 32), 0.0233 (2.5%) behind the hybrid recipe at its own best shape and warm-up (0.94405), `PLAN_SHAPE.md` sections 19-23. That gap, and whether it persists, shrinks or grows with size, is the first scientific question of a scale-up.
- **What transfers, as hypotheses to test, not facts:** (a) shape matters as much as size: the embedding is 51-66% of a 99K model, and shifting parameters from feed-forward rules to a larger vocabulary and head gained 5.5%; at 50M the embedding share falls (section 3) and the optimum will move. (b) A long warm-up (about 10% of the steps) helped both models by 0.009-0.016. (c) lr 6e-2 with AdamW and cosine decay, batch 16 x 512 tokens, worked at every scale we tried up to 1B characters; it will not transfer to 50M parameters unchanged (section 5). (d) 30M-character screens mislead: gains from head repairs and larger vocabularies shrank 55-60% at 300M.

## 2. Targets and the Pythia reference (figures from memory, **to verify against the Pythia paper and model cards before use**)

| model | parameters | non-embedding | layers x width | vocabulary | training tokens |
|---|---|---|---|---|---|
| Pythia-70M | about 70M | about 19M | 6 x 512 | 50,304 | about 300B |
| Pythia-160M | about 162M | about 85M | 12 x 768 | 50,304 | about 300B |
Our "50M" is between the two: the target is the *shape of the experiment* (a fixed parameter count, a standard vocabulary and a public benchmark set), not a Pythia clone; with an untied 50k vocabulary Pythia-70M spends about three quarters of its parameters on the embedding and unembedding, which is exactly the regime where our tied, tiered, fuzzy embedding (a head of full-width rows and a tail as fuzzy sets over a few prototypes) saves the most. We train on 0.5-10B tokens, not 300B: the data budget is a few percent of Pythia's.

## 3. A first parameter budget (all estimates; run `flm/shape_budget.py`-style fits before trusting them)

Embedding cost with the tiered fuzzy table (`NH*d + (V - NH)*E + E*d`), tied output:
| vocabulary V | width d | head NH | tail rank E | embedding parameters *(estimate)* | share of 50M |
|---|---|---|---|---|---|
| 32,768 | 512 | 4,096 | 32 | 2.1M + 0.9M + 16K = about 3.0M | 6% |
| 50,304 | 512 | 4,096 | 32 | 2.1M + 1.5M = about 3.6M | 7% |
| 50,304 | 512 | 16,384 | 32 | 8.4M + 1.1M = about 9.5M | 19% |
Untied and crisp at V 50,304 and d 512 the embedding alone would be 25.8M (51.5M untied, as Pythia-70M): the tiered fuzzy embedding frees 15-20M parameters for blocks at this size, a larger *fraction* than at 99K. The open question is the head size at 50M: our search found that a large crisp head helped (640 of 6,144 tokens, 10%); the equivalent here is a head of 5-30% of the vocabulary.
Blocks: with about 40M parameters left, six blocks at d 512 with the all-fuzzy components (fuzzy delta mixer, TSK feed-forward rules, TSK projections, fuzzy norm and gates) cost per block *(estimate, to be fitted)* about 6M; the rule counts per layer are the free knob exactly as in `PLAN_SHAPE.md`.

## 4. Data

| source | size on disk now | notes |
|---|---|---|
| TinyStories (V2, GPT-4) | 2.2B characters (about 0.55B tokens at BPE3072) | too narrow for 50M parameters: the 99K model already saturates it; use only as a regression check |
| SimpleStories | 2.6B characters (prepared, `data/simplestories`, MIT) | a step broader; about 0.65B tokens at BPE3072 |
| Cosmopedia v2 | 3.0B characters (prepared, `data/cosmopedia`, ODC-By) | synthetic textbooks, stories, how-to: 1.13 BPC at 2.2M parameters and 1B characters (`PLAN_BIG2M.md`) |
| FineWeb-Edu `sample-10BT` | not downloaded; about 28 GB parquet *(to verify)*, 10B tokens | the standard comparison corpus for small models; would need an explicit download decision |
| BabyLM, Simple English Wikipedia | not downloaded | register-mixed; for diversity checks |
A 50M-parameter model should see on the order of 20 tokens per parameter *(Chinchilla heuristic, to verify)*: about 1B tokens (about 4B characters). The prepared corpora total about 7.8B characters, so a mixture of SimpleStories and Cosmopedia (with a TinyStories share for the regression check) is enough for the first full run without any download; FineWeb-Edu is the next step for external comparability.

## 5. What has to be built before the first run (the scaffold)

1. **A scaling ladder before the target:** the same recipe at about 1M, 5M, 15M and 50M parameters on one fixed data mixture (30-100M-token screens at each rung), reading how the best embedding shape, learning rate and warm-up move with size (the 99K search found three non-obvious optima). Budget the ladder so that no rung costs more than a couple of GPU hours.
2. **Hyperparameter transfer:** lr 6e-2 is far above the usual rate for a 50M transformer (about 1e-3 to 6e-4 with AdamW); our value works because the blocks are fuzzy rule banks with small gradients. Treat lr, weight decay, batch size and warm-up as unknown at each rung; consider muP-style width scaling only if the rungs show a trend worth the engineering.
3. **Data loader:** sharded memory-mapped token streams (the existing `train_bpe*.bin` files are one flat uint16 array: enough to 4B tokens), a held-out validation split by document, a deterministic mixing recipe with the mix recorded in the result file, and no repeated data within an epoch (the 1B-character stream was a single pass; 4B characters of mix is about one epoch).
4. **Tokenizer:** BPE at 32k-50k vocabulary trained on the mixture, with the frequency file the tiered embedding needs; characters per token saturated at 4.05-4.07 between V 6,144 and 8,192 on TinyStories, so a larger vocabulary buys little compression on this register and the choice is about the head size and the tail rank, not compression.
5. **Throughput and hardware** *(measured at 99K; the rest estimated)*: a 99K step is 75-77 ms for 8,192 tokens (about 109K tokens per second), GPU-compute-bound on elementwise kernels, with `torch.compile` unavailable (no Triton on this Windows host); CUDA graphs save at most 6%. A 50M model is matmul-heavy: at 6N FLOPs per token (0.3 GFLOP) and an RTX 4080 at an effective 10-15 TFLOPs *(estimate)* the speed is 30-50K tokens per second, so 1B tokens is about 6-10 hours and 4B tokens 1-1.5 days: one run at a time on this machine, or a rented GPU. Decide early between Windows (no `torch.compile`, so the elementwise fuzzy kernels stay unfused; a hand-written fused kernel for the Gaussian rules is the likely 2x) and a Linux environment with Triton.
6. **Precision and memory:** bf16 autocast for the dense parts and fp32 for the rule firing strengths (Gaussian exponentials), a gradient-accumulation path for batches above memory, and activation checkpointing if 12 GB is binding at 50M with context 1024.
7. **Checkpointing and evaluation cadence like Pythia:** log-spaced checkpoints (so learning curves can be read), validation BPC on each corpus every few thousand steps, and a **fixed evaluation suite** (validation BPC per corpus; the blind coherence protocol with at least 32 texts per model and two raters; a few small public benchmarks such as LAMBADA, HellaSwag and PIQA subsets at zero-shot for comparability with Pythia *(to verify that the models are large enough to leave chance level)*).
8. **Fuzzy blocks at width 512:** the FFN rule banks use `R = ffn_mult x d` rules per layer with a 2d-wide antecedent window; at d 512 the antecedent distance computations dominate and the rule count must be fitted, not scaled linearly. Re-run the budget helper at each rung (`flm/shape_budget.py` is written for d 32-64) and re-check the hybrid at the same shape (the fairness lesson of `PLAN_SHAPE.md`: compare each model at its best shape and warm-up).
9. **Repetition and decoding:** see `PLAN_REPEAT.md` (decode-time and sequence-level post-training experiments on the 99K model); larger models repeat less at the same sampler, but a sampler setting chosen at 99K will not carry over.

## 6. Proposed order (when the owner says go)

1. A throughput probe: a 50M-parameter all-fuzzy model for 100 steps at context 512 and 1024, to settle the hardware question (Windows against Linux, bf16, compile) before any science.
2. The scaling ladder (item 1) on the SimpleStories + Cosmopedia mixture at 30-100M-token screens: shape, lr and warm-up per rung.
3. The 50M model for 0.5-1B tokens, two seeds if the budget allows, with the all-fuzzy model and the hybrid recipe both at their rung-tuned shapes.
4. The evaluation suite and a blind coherence set (item 7), then the write-up.
Gate: the owner's go, an explicit decision on the data download for FineWeb-Edu if wanted, and a compute estimate (section 5, item 5) approved in advance; no run over about one hour starts without a yes.
