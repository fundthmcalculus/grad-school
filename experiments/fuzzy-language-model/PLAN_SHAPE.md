# PLAN_SHAPE: which shape of the all-fuzzy model works best under 100K parameters? (drafted 2026-10-21; plan only, nothing run)

Owner's request: "design a set of experiments varying the shape of the model while staying under 100K parameters. Let's figure out what works the best! The experiments on embedding are more important." Embedding experiments therefore come first and carry most of the budget; block-shape experiments follow and are conditioned on the embedding result.

## 1. Why the embedding is the main lever

The all-fuzzy model (`e2e_tail`, 99,019 parameters: V 3072, 384 head tokens at full width, 2,688 tail tokens as fuzzy sets over 12 prototypes, d 48, 2 blocks) spends **51,264 parameters (51%) on the embedding** (`NH*d + (V-NH)*E + E*d`), about 29,200 on a fixed block floor (mixers, norms, gates, projections at d 48) and the remaining 18,500 or so on the feed-forward rules (54 / 36 rules per layer at the cap). Every embedding choice therefore trades embedding parameters against feed-forward rules, which is the trade the recipe has never been scanned along.

What the earlier work established (hybrid recipe on TinyStories, 30M characters, five seeds, `PLAN_TIERVOCAB.md`, `PLAN_TIERED_D.md`):
| axis | measured | note |
|---|---|---|
| vocabulary | BPE1024 / 1536 / 2048 / 3072 gain 0 / 0.015 / 0.026 / **0.042**; V4096 with the tail rank cut to 8 gains 0.031 (0.011 worse than 3072) | the lever is characters per token (3.16 / 3.48 / 3.66 / 3.85 / 3.95) |
| head / FFN split at V2048 | three shapes within 0.006 | little sensitivity there |
| width d at BPE1024 | d56 head 448 gains 0.007 (4/5 seeds), d56 head 384 loses | not repeated at V3072 |
| **never run** | **round 3 of `PLAN_TIERVOCAB.md`: head, tail width and FFN split at V3072; V4096 with tail width 10**; any V above 4096 | the recipe's one point (head 384, tail 12) was chosen from three arms |
| head repairs | fuzzy head costs 0.011 at 300M (signed memberships fix most of it; prior and untied prototypes lose their gain at 300M) | already closed; shapes below use the crisp-head / fuzzy-tail recipe |

## 2. Rules for every arm

- **Iso-parameter**: the free knob is the feed-forward rule count (layer 0 at 35/52 of layer 1), fitted by `flm/shape_budget.py` so each arm sits just under 99,999 (the helper reproduces the recipe's 99,019 exactly). The recipe refitted to the cap is **54 / 36 rules, 99,694 parameters**: the reference arm is that refit (a new run), not the 52 / 35 recipe, so no arm gains from 700 extra parameters. Each result states its rule counts.
- **Screen**: 30M characters, lr 6e-2, warm-up 600, TinyStories, paired by seed against the refit reference. Successive halving: stage 1 three seeds per arm; an arm that is not better than the reference on all three seeds by at least 0.003 on average is dropped; the survivors get seeds 3-4 (stage 2); the gain rule is 0.005 on four of five seeds. Stage 3: the top two arms at 300M characters, two seeds, warm-up 1000, against the hybrid-fitted reference at 300M. Stage 4: the winner on SimpleStories at 30M (five seeds) for a second corpus.
- **Confounds recorded per arm**: characters per token and steps (a larger vocabulary means fewer optimiser steps at 30M characters), embedding share, FFN rules, and the gap to the hybrid recipe at the same shape where it matters.
- Reporting: `python -m flm.report_nextround <dir> --ref <reference prefix>` (paired table with the decision rule), PLAN entries with predictions scored, dashboard section after each series.

## 3. Embedding experiments (the main series)

Parameter counts below are fitted (`flm/shape_budget.py`); "emb" is the embedding parameters, "share" the embedding's fraction of the total.

**E1: head size at V3072, tail width 12.**
| arm | head | emb | FFN rules L1/L0 | params | share |
|---|---|---|---|---|---|
| E1a | 192 | 44,352 | 72 / 48 | 99,724 | 0.44 |
| E1b | 288 | 47,808 | 63 / 42 | 99,709 | 0.48 |
| **ref** | **384** | 51,264 | 54 / 36 | 99,694 | 0.51 |
| E1c | 512 | 55,872 | 42 / 28 | 99,674 | 0.56 |
| E1d | 640 | 60,480 | 30 / 20 | 99,654 | 0.61 |
Question: how much of the budget should go to the crisp head tokens against FFN rules? (Head 384 covers 73% of SimpleStories tokens and about 80% of TinyStories tokens.)

**E2: tail width (prototype count) at V3072.**
| arm | head | tail | emb | rules | params |
|---|---|---|---|---|---|
| E2a | 384 | 8 | 40,320 | 80 / 54 | 98,970 |
| E2b | 192 | 16 | 56,064 | 42 / 28 | 99,866 |
| E2c | 256 | 16 | 58,112 | 36 / 24 | 99,600 |
Question: is rank 12 for the 2,700 rare tokens right? E2b against E1a (same head, tail 16 against 12) isolates the tail width from the head size.

**E3: vocabulary (tokenizers to be trained from the TinyStories stream; BPE training takes minutes).**
| arm | V | head | tail | emb | rules | params |
|---|---|---|---|---|---|---|
| E3a | 2560 | 384 | 12 | 45,120 | 70 / 47 | 99,817 |
| E3b | 4096 | 384 | 8 | 48,512 | 61 / 41 | 99,738 |
| E3c | 4096 | 384 | 10 | 56,032 | 42 / 28 | 99,834 |
| E3d | 6144 | 256 | 6 | 47,904 | 63 / 42 | 99,805 |
| E3e | 6144 | 192 | 6 | 45,216 | 70 / 47 | 99,913 |
| E3f | 8192 | 256 | 6 | 60,192 | 31 / 21 | 99,848 |
Question: where does the vocabulary lever stop (3.85 characters per token at 3072, 3.95 at 4096; the 4096 arm of the earlier round lost only because its tail rank fell to 8)? The tail rank is held at 6-10 for the large vocabularies, so the arms test both vocabulary and tail width; E3b against E3c separates them.

**E4: block width d (all with the head resized to keep the embedding share near 50%).**
| arm | d | head | tail | emb | rules | params |
|---|---|---|---|---|---|---|
| E4a | 32 | 512 | 12 | 47,488 | 117 / 79 | 97,160 (rules to be refitted upward) |
| E4b | 40 | 448 | 12 | 49,888 | 80 / 54 | 99,250 |
| E4c | 56 | 320 | 12 | 51,616 | 36 / 24 | 99,544 |
| E4d | 64 | 192 | 10 | 41,728 | 43 / 29 | 99,820 |
Question: is d 48 the right width for the fuzzy blocks (a narrower model buys many more rules; a wider one buys richer mixers)?

**E5: three-tier embedding (new code: `flm/fuzzy_tokens.py`, a second tail group).** Head tokens at full width d, a middle tier at width E1 with its own prototypes, a tail at width E2 < E1, each with its own up-projection. Rationale: the E3 and E2 arms are forced to cut the tail rank for rare tokens to afford a larger vocabulary; a middle tier keeps rank 16 where tokens still carry loss mass and rank 6 only where they do not. Implemented (`FUZZ_TOK_MID` in `flm/fuzzy_tokens.py`, verified: tier shapes, embedding parameter count equal to the formula, finite logits). Arms (fitted): **E5a** V4096, head 256 / middle 768 at 16 / tail 3,072 at 6, 73 / 49 rules, 99,918 parameters; **E5b** V4096, head 192 / middle 1,024 at 16 / tail 2,880 at 6, 73 / 49 rules, 99,790; **E5c** V3072, head 384 / middle 768 at 16 / tail 1,920 at 8, 64 / 43 rules, 99,615. Hypothesis: this beats the best two-tier arm at V4096.

**E6: tokenizer algorithm at V3072 (conditional).** `flm/prep_superbpe.py` (superword merges that cross word boundaries; untracked, used locally) raises characters per token without changing the vocabulary size; one arm if the stream reaches at least 4.3 characters per token with BPE3072-sized tables, otherwise dropped.

## 4. Block-shape experiments (second series; after the E series picks an embedding)

All at the winning embedding shape, iso-parameter:
- **B1 mixer heads** 2 / 4 / **8** / 16 (the mixer state per head).
- **B2 FFN layer split**: layer-0 / layer-1 rule ratio 0.5 / **0.67** / 0.85 at fixed total rules.
- **B3 context length** 256 / **512** / 1024 tokens (at 30M characters the number of optimiser steps changes with batch; batch rescaled to keep tokens per step constant).
- **B4 feed-forward taps** (the temporal window of the layer-0 rules) 1 / **2** / 3.
- **B5 antecedent width** (`--tsk-dim` 12 / **16** / 24).
Each at three seeds first; the best of each axis combined in one joint arm (stage 3).

## 5. Compute

One RTX 4080; a 30M cell is about 75 s (5 seeds of one arm 6.5 minutes).
| step | cells | time |
|---|---|---|
| tokenizers for V 2560 / 4096 / 6144 / 8192 and their frequency files | CPU | 20-40 minutes, parallel to GPU work |
| reference refit (5 seeds) | 5 | 6 minutes |
| stage 1, E series: about 22 arms x 3 seeds | 66 | 85 minutes |
| stage 2: about 6 survivors x 2 more seeds + reference | 15 | 20 minutes |
| stage 3: top two at 300M, two seeds + a reference at 300M (3 cells, 12 min each, plus the refit's 2) | 8 | 95 minutes |
| stage 4: the winner on SimpleStories, five seeds + reference | 10 | 13 minutes |
| B series (after E): about 12 arms x 3 seeds + survivors + a joint arm at 300M | about 60 | 80 minutes |
Total about **5 GPU hours**, none of it 1B characters; the E series with its confirmation is about 3.5 hours. I will ask before starting the 300M cells.

## 6. Registered predictions (before any run)

1. At least one E-series arm beats the refit reference by 0.005 on four of five seeds (50%); the best arm gains at least 0.015 at 30M (25%).
2. E1: the best head size lies between 288 and 512 (65%); both 192 and 640 are worse than the reference by at least 0.005 (45%).
3. E2: E2a (tail 8) is worse than the reference (60%); E2b (tail 16, head 192) beats E1a (head 192, tail 12) by at least 0.005 (40%).
4. E3: V4096 with tail 8 or 10 beats V3072 (30%); V6144 and V8192 are worse than V3072 (65%); V2560 is worse (65%). The vocabulary lever has stopped near 3072-4096.
5. E4: d 48 is within 0.005 of the best width (70%); no width beats it on four of five seeds (75%).
6. E5: the best three-tier arm beats the best two-tier V4096 arm by at least 0.005 (35%) and beats the V3072 reference (25%).
7. E6 (if run): at least 4.3 characters per token and at least 0.01 BPC better than BPE3072 (40%).
8. B series: no block-shape axis moves BPC by 0.005 at the winning embedding (60%): the blocks are already near their optimum, as the 1- and 3-block and projection results suggest.
9. Whatever wins at 30M keeps at least half its gain at 300M (40%; the head repairs of stages 6-9 mostly did not).
10. The all-fuzzy model's gap to the hybrid does not close by more than 0.01 at 300M under any shape (75%).

## 7. Decision rules and risks
- An arm advances only on the stated rules; nothing is called a win on fewer than five seeds (30M) or two seeds (300M), and **no 30M gain is quoted without the 300M confirmation**.
- If no embedding arm beats the refit reference, the recipe's point (V3072, head 384, tail 12) is a local optimum along every embedding axis and the write-up says so.
- Iso-parameter fits move the FFN rule count by large factors (30 to 117 rules per layer); a result may then reflect the FFN rule count more than the embedding shape, so the E1 series (head size against rules) is read together with the FFN-only controls from stage 10 (trimming layer-1 rules to 41 costs 0.0046).
- Larger vocabularies reduce the optimiser steps at 30M characters (about 900 steps): an effect against them that the 300M stage corrects.
- `flm/fuzzy_tokens.py` is touched for E5 only; the screens E1-E4 need no model change (`TIER_NH`, `TIER_E`, `FUZZ_FFN_R`, `FUZZ_FFN_R0`, `--d`, `--tok`).

## 8. E7: a common, essentially reversible embedding / de-embedding layer (added 2026-10-21 at the owner's request: "evaluate the idea of a common embedding / de-embedding layer, so that we save parameters. The embedding layer should be essentially reversible.")

**What exists.** The model already has one shared layer: the table that embeds a token (`E[t]`) also de-embeds the output (`score_s = <h, E[s]>`; head rows crisp, tail rows `tier_up(mu(emb_t))`), so the V x d output matrix is not paid twice (the untied variants of stages 8-9 cost +0.002 to +0.014 at 30M and gain nothing at 300M). The new questions are (a) how reversible the shared layer is (decode(embed(t)) = t), (b) whether making it reversible by construction helps, and (c) whether a smaller shared layer saves parameters without losing what the recipe has.

**Baseline measurement** (`flm/embed_reversibility.py`, `outputs/embed_reversibility.log`; trained 300M checkpoints; round trip with the model's own tied scoring, h = E[t], share of tokens whose top-1 decode is the token itself):
| model | head tokens (384) | tail tokens (2,688) | all tokens, frequency-weighted | effective rank of E (of 48) |
|---|---|---|---|---|
| hybrid recipe, 300M | 0.893 | 0.182 | 0.818 | 5.6 |
| all-fuzzy `e2e_tail`, warm-up 1000, 300M | 0.992 | 0.190 | 0.865 | 6.9 |
| all-fuzzy `e2e_tail`, 1B | 0.995 | 0.137 | 0.844 | 7.4 |
| fully fuzzy embedding (`e2e_all_tanh`), 300M | 0.997 | 0.494 | 0.935 | 10.9 |
Reading: the shared table is **already essentially reversible where it matters**: head tokens round-trip almost perfectly and the tokens that carry the probability mass round-trip 82-94% of the time. The tail does not (14-19% top-1; 2,688 tokens in 12 dimensions cannot be told apart by a dot product), and the fully fuzzy table is the most reversible tail (49%). The effective rank of the whole table is **5.6-10.9 of 48**: the embedding uses a small subspace, which suggests a rank-limited head may cost little (the earlier factorised embedding, `v16-factemb`, lost 0.040 or more by limiting *every* token's rank; here only the head is limited and the tail keeps its own prototypes).

**Arms** (`E7*` in `flm/shape_budget.py`, fitted; V3072, head 384, tail 12, d 48; `flm/run_embed_rev.py` adds the loss terms, `FUZZ_TOK_HEADRANK` the rank limit):
| arm | change | params | FFN rules L1/L0 |
|---|---|---|---|
| E7a ortho1 | orthonormality penalty (weight 1) on the tail prototypes (`tier_up`): the membership vector is exactly recoverable from the embedding | 99,694 | 54 / 36 |
| E7b ortho0.1 | the same, weight 0.1 | 99,694 | 54 / 36 |
| E7c cycle0.05 | cycle loss: the shared table is trained to decode its own embedding (CE of `embed(t) . E^T` against t, 512 tokens per step), weight 0.05 | 99,694 | 54 / 36 |
| E7d cycle0.2 | the same, weight 0.2 | 99,694 | 54 / 36 |
| E7e hr32 | head tokens through a rank-32 bottleneck with a shared up-projection (saves 4,608 parameters, spent on feed-forward rules) | 99,714 | 66 / 44 |
| E7f hr24 | rank 24 (saves 8,064) | 99,729 | 75 / 50 |
| E7g hr16 | rank 16 (saves 11,520) | 99,551 | 83 / 56 |
| E7h hr24 + ortho | rank 24 with orthonormal head and tail prototypes: a small, reversible shared layer | 99,729 | 75 / 50 |
Re-run the reversibility measurement on the winning checkpoints (baseline above); the loss terms leave the plain cross-entropy as the logged training loss.

**Registered predictions (E7).**
1. E7a / E7b (orthonormal prototypes, no parameter change) are within 0.005 of the reference (55%); neither beats it by 0.005 on four of five seeds (80%).
2. E7c / E7d (cycle loss) are no better than the reference by 0.005 (75%); the 0.2 weight is worse than the reference (50%): the output scoring is a language-model task, not a retrieval task, and the table is already reversible for the head.
3. A rank-limited head that saves parameters: E7e (rank 32) is within 0.005 of the reference (45%); E7f (rank 24) beats the reference by 0.005 on four of five seeds (20%); E7g (rank 16) is worse than the reference by 0.005 (65%). The effective rank of 6-11 says the head table is compressible; the earlier factorised-embedding loss says the cost of limiting rank is real.
4. E7h (rank 24, orthonormal) beats E7f (rank 24, free) (35%).
5. Round-trip reversibility of the tail (tokens recovered) rises with the orthonormal penalty by at least 0.1 absolute (60%) without a likelihood gain: reversibility by construction is a property, not a lever.
