# PLAN_FUTURE: work beyond the allocation search (drafted 2026-10-23 at the owner's request; nothing here has been run)

Owner's decision (2026-10-23): the random allocation search shows no reasonable improvement, so further validation of what does not work stops (round B is cancelled and stays a background idea to confirm later). This plan looks for levers of a different kind.

## What the evidence says is closed (do not re-screen)
| lever | result | where |
|---|---|---|
| embedding shape (vocabulary, head, tail, block width) | taken: F7, V6144 / head 640 / tail 8 / d32 | `PLAN_SHAPE.md`, `PLAN_ANALYSIS.md` |
| end of the schedule, learning rate | taken: floor 0, lr 6e-2 | `PLAN_ANALYSIS.md` items 1, 5 |
| batch size, weight decay, beta2, context, EMA | a 30M gain at batch 8 did not survive 300M (+0.008); the rest neutral or worse | `PLAN_ROUND3.md` stages 1-2 |
| head count, shared depth, exponent normalisation | neutral or worse | stage 3 |
| random joint allocation (24 samples) | none beats F7 (best 0.0076 worse, seed 0); the mixer rule count and the layer-0/layer-1 rule split are about neutral | stage 4 |
| repetition | DRY sampling halves repeated trigrams; rated coherence unchanged | `PLAN_REPEAT.md` |
Two facts shape what is left. (1) The embedding holds 65% of the parameters, and in the random search tail width and vocabulary size had the largest effects (ridge: tail -0.017, vocabulary -0.013 per standard deviation) while the mixer rule count and the rule split did nothing: the model is limited by how well rare tokens are represented, not by the blocks. (2) The remaining fair gap to the hybrid is 2.5-2.6% and does not move with schedule, learning rate or corpus.

## Candidates, ranked by expected gain per unit of effort
| rank | idea | why it might work here | first cheap test | cost | P(real gain) |
|---|---|---|---|---|---|
| 1 | **Compose the tail from the BPE merge tree.** A tail token's embedding (and its output scores) is a fuzzy combination of its two merge parents' embeddings, recursively down to head tokens and bytes, plus a small learned per-token residual of rank 1-2; so the 5,504 tail tokens stop costing 8 parameters each (44K) | tail capacity is what the search rewards; a composed tail gives rare tokens structure that a free table cannot learn from few examples, and frees most of 44K parameters for blocks or a bigger head. `literature/04-fuzzy-embeddings.md` idea 4 proposed a byte-composed tail (about 35% chance of a net gain, never run); the merge-tree version uses the tokenizer's own structure and is not in that note | implement in `flm/fuzzy_tokens.py` (one new tail mode), check the parameter count and parity on the CPU, then a 30M five-seed arm | new code, about a day; 30M screens cheap | 30% |
| 2 | **SuperBPE tokenizer** (`flm/prep_superbpe.py`, untracked since 2026-10-19; E6 of `PLAN_SHAPE.md`, conditional and never finished) | the tokenizer is where the biggest earlier gain came from; phrase tokens ("once upon a time") raise characters per token, cutting the number of sequence steps for the same text | rebuild at V6144 with a few hundred superword merges; the stream must reach at least 4.6 characters per token (4.05 now); 30M five seeds at the F7 shape, bits per character is tokenizer-independent | tokenizer prep about an hour; screens cheap | 30% |
| 3 | **A different optimiser for the weight matrices** (Muon-style orthogonalised updates on the 2-D matrices, Adam on the fuzzy centres, widths and biases) | the tiny-model literature note in this repo reports Muon on the 2-D matrices in a small recurrent model; this changes the geometry of the update, not a learning-rate or batch meta-parameter | 30M, five seeds, one arm; same recipe otherwise; must be confirmed at 300M (the stage-2 lesson) | about 80 lines; cheap | 25% |
| 4 | **Train wide, prune to budget** (over-provision rules by 1.5-2x, learn a firing-importance score, prune the weakest rules to the parameter cap, fine-tune) | overparameterise-then-prune beats training small at equal final size in several small-model results; fuzzy rules have a natural importance (their firing mass); `PLAN_RULEDIST.md` found that rules cannot be removed by low-rank or distillation tricks, so this is a different mechanism (sparsity during training) | one 30M arm: 1.5x rules for 70% of training, prune, 30% fine-tune | new code, about two days | 20% |
| 5 | **Inference-time adaptation** (dynamic evaluation and an in-context n-gram or copy rule: a cache of the last 512 tokens as one extra fuzzy rule set) | known large wins for small models on repetitive text; stories repeat names and phrases; no extra parameters | evaluate the existing checkpoints on the CPU with a cache mixed in; costs nothing to train | changes the evaluation protocol, so it needs its own label and is not comparable with the numbers above; `RESULTS.md` already says "would need its own label" | 50% of a gain, but of a different kind |
| 6 | **Multi-token prediction as a training-only objective** (extra heads dropped at inference) | training signal for free, parameters not counted at inference | 30M, five seeds | extra heads need a parameter-accounting decision (training-only parameters) | 15% |
| 7 | **The mixed corpus (TinyStories + SimpleStories)** | scaffolded (`PLAN_ROUND3.md` stage 5); tests whether breadth costs story quality, and the Pile configuration needs it | `python -m flm.prep_mix`, then 30M five seeds | about an hour of prep, cheap runs | answers a question rather than gaining |
| background | round B of the allocation search; the batch-8 controls (wd 0.01; lr 4e-2) | confirms that nothing is hiding; not pursued now | -- | -- | -- |

## Suggested order and decision rules
1. **#2 SuperBPE first**: the code exists, the test is cheap, and a failure is informative (tells us whether sequence length is a bottleneck). Gate: 4.6 characters per token and 0.01 BPC better at 30M on at least 4 of 5 seeds, then a 300M confirmation (I ask first).
2. **#1 merge-tree tail** next (the new-code bet with the best rationale), after SuperBPE since the two interact (SuperBPE tokens have merge trees too).
3. **#3 Muon** in the gaps: small code, independent of the others.
4. #5 inference-time adaptation is worth doing as a separately labelled result for the demo and the write-up; it must not be mixed into the parameter-matched tables.
5. #4 only if #1 and #2 leave the embedding as the limit.
Predictions (registered now): at least one of #1-#4 yields a 30M gain of 0.01 BPC or more on four of five seeds (45%); that gain survives at 300M (60% of the cases where it appears); the SuperBPE stream reaches 4.6 characters per token at V6144 (60%); a merge-tree tail loses at least 0.02 BPC before the freed parameters are reinvested (70%).

## Constraints that stay
All-fuzzy model only; 300M characters is the ceiling; one training job at a time; predictions before runs; paired seeds; any 30M result is confirmed at 300M before it is quoted; I ask before runs of an hour or more.

## Decisions needed from the owner
1. Start with #2 (SuperBPE), or go straight to the new-code bet #1?
2. Is a separately labelled inference-time adaptation result (#5) wanted, given that it is not a parameter-matched improvement?
3. Is a training-only head (#6) acceptable under the parameter cap, or must every trained parameter count?

## SuperBPE experiment (owner: "let's do SuperBPE", 2026-10-23; registered before the tokenizer is built)
Tokenizer: BPE6144 cut back to its first 5,632 merges, then 512 superword merges (no word-boundary restriction) learned on 60M stage-1 tokens (`flm/prep_superbpe_eq.py`): the vocabulary stays 6,144, so the embedding costs the same and F7 (V6144 / head 640 / tail 8 / d32, 61/41 rules, 99,882 parameters) runs unchanged. Corpus `sb6144`. Arm: F7 at 30M characters, seeds 0-4, same recipe as R0 (warm-up 600, lr 6e-2, batch 16, floor 0), paired by seed with R0 (mean 1.0579). Bits per character is per character, so it is comparable across tokenizers; the same 1,000,000 validation characters are scored.
Known confound: a token covers more characters, so the same 30M characters is fewer optimiser steps (905 at 4.05 characters per token, about 780 at 4.7); the stage-1 result suggests that costs a little (about +0.002), so a small gain could be hidden by it and a loss is not caused by it alone. Gate (PLAN_FUTURE.md): at least 4.6 characters per token and at least 0.01 bits per character better on four of five seeds; then a 300M confirmation, asked for first.
Predictions: characters per token on the training stream at least 4.6 (60%), at least 5.0 (25%); the 30M mean beats R0 by at least 0.01 (30%); worse than R0 (45%); the arm needs at least 10% fewer steps than R0 (85%); the new tokens are mostly common story phrases ("once upon a time", " in the ", " he was ") (90%).
