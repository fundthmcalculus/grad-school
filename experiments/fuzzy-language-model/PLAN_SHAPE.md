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

## 9. E8: combinations of the stage 1 signals (registered 2026-10-21 18:10, after seed 0 of the 18 stage 1 arms and before any combination arm ran; seeds 1-2 of stage 1 were still running)
Seed 0 of stage 1 (provisional, one seed): a larger vocabulary (V4096 tail 10 -0.028, V4096 tail 8 -0.023, V6144 -0.023, V8192 -0.019), a larger head (h512 -0.012, h640 -0.009) and a slightly narrower block (d40 -0.007) each beat the refit reference; head 192-288, tail width 8, d56-64 and V2560 lose. These axes have not been combined. Arms (fitted, `flm/shape_budget.py`):
| arm | V | head | tail | d | FFN rules L1/L0 | params | embedding share |
|---|---|---|---|---|---|---|---|
| C1 | 4096 | 512 | 10 | 48 | 29 / 20 | 99,877 | 0.61 |
| C2 | 4096 | 384 | 10 | 40 | 73 / 49 | 99,910 | 0.53 |
| C3 | 4096 | 512 | 10 | 40 | 61 / 41 | 99,890 | 0.57 |
| C4 | 6144 | 384 | 8 | 48 | 19 / 13 | 99,924 | 0.65 |
| C5 | 4096 | 640 | 8 | 48 | 34 / 23 | 99,565 | 0.59 |
| C6 | 6144 | 512 | 8 | 40 | 33 / 22 | 99,939 | 0.66 |
The large embedding shares push the feed-forward banks to 19-34 rules per layer (the reference has 54), so these arms also test how far rules can be traded for embedding; the stage 10 control (41 rules costs 0.0046) says the trade is not free.
Predictions: the best combination beats the best single-axis arm of stage 1 by at least 0.005 on the three-seed mean (45%); some combination reaches at least 0.035 better than the reference (30%); C4 (19 rules) is worse than C2 (73 rules) (55%: the feed-forward floor); C3 beats C1 (50%).

## 10. Stage 1 results (recorded 2026-10-21 18:45-19:10; `v69-shape`, 30M characters, three seeds, iso-parameter, refit reference 1.1213 on seeds 0-2 (54 / 36 rules, 99,694 parameters); `report/shape_stage1.md`; change against the reference, negative is better)
| series | arm | BPC | change (se) | per seed | FFN rules |
|---|---|---|---|---|---|
| head size | h192 | 1.1451 | +0.0242 (0.0015) | +0.0249 +0.0214 +0.0265 | 72 / 48 |
| | h288 | 1.1324 | +0.0115 (0.0039) | +0.0113 +0.0049 +0.0184 | 63 / 42 |
| | **h512** | 1.1109 | **-0.0099** (0.0015) | -0.0124 -0.0101 -0.0071 | 42 / 28 |
| | **h640** | 1.1122 | **-0.0086** (0.0022) | -0.0089 -0.0123 -0.0046 | 30 / 20 |
| tail width | e8 (head 384) | 1.1387 | +0.0179 (0.0039) | +0.0203 +0.0102 +0.0232 | 82 / 55 |
| | h192 e16 | 1.1378 | +0.0170 (0.0017) | +0.0168 +0.0141 +0.0201 | 42 / 28 |
| | h256 e16 | 1.1303 | +0.0095 (0.0021) | +0.0102 +0.0056 +0.0127 | 36 / 24 |
| vocabulary | V2560 | 1.1403 | +0.0195 (0.0024) | +0.0148 +0.0210 +0.0226 | 70 / 47 |
| | **V4096 h384 e8** | 1.0999 | **-0.0209** (0.0021) | -0.0232 -0.0230 -0.0167 | 61 / 41 |
| | **V4096 h384 e10** | **1.0935** | **-0.0273** (0.0028) | -0.0276 -0.0320 -0.0222 | 42 / 28 |
| | **V6144 h256 e6** | 1.0991 | **-0.0218** (0.0007) | -0.0230 -0.0207 -0.0216 | 63 / 42 |
| | V6144 h192 e6 | 1.1199 | -0.0009 (0.0041) | -0.0025 -0.0071 +0.0069 | 70 / 47 |
| | **V8192 h256 e6** | 1.1045 | **-0.0163** (0.0015) | -0.0190 -0.0138 -0.0162 | 31 / 21 |
| block width | d32 (head 512) | 1.1215 | +0.0007 (0.0014) | +0.0032 -0.0016 +0.0006 | 128 / 86 |
| | d40 (head 448) | 1.1160 | -0.0048 (0.0023) | -0.0067 -0.0076 -0.0002 | 82 / 55 |
| | d56 (head 320) | 1.1337 | +0.0128 (0.0029) | +0.0154 +0.0071 +0.0161 | 36 / 24 |
| | d64 (head 192, e10) | 1.1589 | +0.0381 (0.0004) | +0.0375 +0.0379 +0.0389 | 43 / 29 |
Survivors to stage 2 (better on all three seeds by at least 0.003 on average): **E1c h512, E1d h640, E3b V4096 e8, E3c V4096 e10, E3d V6144 h256, E3f V8192, E4b d40** (seven arms; seeds 3 and 4 follow).
Predictions (section 6): at least one arm beats the reference on four of five seeds (50%): **right on three seeds, five-seed confirmation pending**; best arm gains at least 0.015 (25%): **right** (V4096 e10, -0.0273); best head between 288 and 512 (65%): **right at the edge** (512 beats 640 by 0.001); both 192 and 640 worse than the reference by 0.005 (45%): **wrong** (640 gains 0.0086); E2a worse than the reference (60%): **right**; E2b beats E1a by 0.005 (40%): **right** (0.0073, every seed); V4096 beats V3072 (30%): **wrong** (-0.021 and -0.027); V6144 and V8192 worse than V3072 (65%): **wrong** (-0.022 and -0.016); V2560 worse (65%): **right**; d48 within 0.005 of the best width (70%): **right, just** (d40, -0.0048); no width beats d48 on four of five seeds (75%): pending.
Reading: **the recipe's embedding shape was not optimal**: a vocabulary of 4096 with the 384 head and a rank-10 tail beats the vocabulary-3072 recipe by 0.027 bits per character (2.4%) on every seed, through the same mechanism as the earlier vocabulary lever (characters per token 3.85 to 3.95), and a larger head (512-640) gains 0.009-0.010 at vocabulary 3072 even though it cuts the feed-forward rules from 54 to 30-42. Both trade feed-forward rules for embedding and win, so the feed-forward banks are the cheaper component to shrink. The tail wants at least rank 10 (rank 8 loses 0.006 at vocabulary 4096 and 0.018 at 3072), the vocabulary lever peaks near 4096-6144 (8192 loses 0.011 against 4096 e10 once the tail is 6 wide), and d 40-48 is the right block width. Combinations (E8) are queued.

## 11. E5 and E7 results (recorded 2026-10-21 20:10; `v69-shape`, 30M characters, three seeds, iso-parameter; change against the refit reference, negative is better)
| arm | BPC | change | per seed | FFN rules |
|---|---|---|---|---|
| E5a three-tier V4096 (256 / 768 at 16 / 3,072 at 6) | 1.1032 | **-0.0177** | -0.0232 -0.0136 -0.0162 | 73 / 49 |
| E5b three-tier V4096 (192 / 1,024 at 16 / 2,880 at 6) | 1.1014 | **-0.0194** | -0.0192 -0.0231 -0.0159 | 73 / 49 |
| E5c three-tier V3072 (384 / 768 at 16 / 1,920 at 8) | 1.1195 | -0.0013 | +0.0012 -0.0063 +0.0012 | 64 / 43 |
| E7a orthonormal prototypes, weight 1 | 1.1677 | +0.0469 | +0.0464 +0.0425 +0.0517 | 54 / 36 |
| E7b orthonormal prototypes, weight 0.1 | 1.1664 | +0.0455 | +0.0432 +0.0423 +0.0511 | 54 / 36 |
| E7c cycle loss 0.05 | 1.1289 | +0.0081 | +0.0077 +0.0071 +0.0095 | 54 / 36 |
| E7d cycle loss 0.2 | 1.1583 | +0.0375 | +0.0402 +0.0341 +0.0381 | 54 / 36 |
| E7e head rank 32 | 1.1393 | +0.0184 | +0.0180 +0.0216 +0.0157 | 66 / 44 |
| E7f head rank 24 | 1.1476 | +0.0267 | +0.0276 +0.0230 +0.0296 | 75 / 50 |
| E7g head rank 16 | 1.1780 | +0.0572 | +0.0522 +0.0574 +0.0619 | 83 / 56 |
| E7h head rank 24 + orthonormal | 1.1684 | +0.0476 | +0.0476 +0.0452 +0.0500 | 75 / 50 |
Predictions. E5: the best three-tier arm beats the best two-tier V4096 arm by 0.005 (35%): **wrong** (E5b -0.0194 against E3c -0.0273: 0.008 worse); beats the V3072 reference (25%): **right for the V4096 arms, wrong for E5c**. E7: orthonormal prototypes within 0.005 of the reference (55%): **wrong** (+0.046); neither beats it by 0.005 (80%): right; cycle loss no better than the reference (75%): right; the 0.2 weight worse (50%): right; rank 32 within 0.005 (45%): **wrong** (+0.0184); rank 24 beats the reference on four of five seeds (20%): **wrong** (+0.0267, every seed worse); rank 16 worse (65%): right; E7h beats E7f (35%): **wrong** (+0.0476 against +0.0267); round-trip reversibility of the tail rising with orthonormality (60%): **not measured** (the screen saves no checkpoints).
Reading. **The reversibility ideas do not pay.** (1) A rank-limited shared head loses monotonically (+0.018, +0.027, +0.057 for ranks 32, 24, 16), even though the saved parameters buy 12-29 more feed-forward rules per layer: the head table's low effective rank (5.6-10.9) is a property of its spectrum, not permission to cut its width, because the top-1 round trip and the output scores depend on the small singular directions as well. (2) Orthonormal prototypes cost 0.045 at either weight (so the constraint, not its strength, is the cost; with softmax memberships, orthonormal columns give every tail embedding a norm of at most one, which may cap the tail logits: **a hypothesis, tested by the scaled-orthonormal arm E7i below**). (3) A cycle loss on the table is neutral at 0.05 and harmful at 0.2. (4) Three tiers help at V4096 (-0.018 to -0.019) but a plain two-tier rank-10 tail does better (-0.027): the middle tier's extra prototypes do not pay for the smaller head and larger bank cost.
**E7i (registered before it runs):** orthonormal prototypes up to one learned global scale (`FUZZ_ORTHO_SCALED=1`, one extra parameter: the penalty is ||W^T W - c I||^2 with c learned), weight 1, 54 / 36 rules. Prediction: it removes most of E7a's cost (within 0.015 of the reference, 55%) and does not beat the reference (80%); if the cost stays near +0.045 the norm-cap explanation is wrong.

## 12. E8 results so far and E9, a second round of combinations (E9 registered 2026-10-21 20:45, after C1-C5 (three seeds) and before C6 and any E9 arm)
E8, three seeds, change against the refit reference: **C4 V6144 head 384 tail 8 d48 -0.0430** (-0.0441 -0.0433 -0.0415; 19 / 13 rules); **C3 V4096 head 512 tail 10 d40 -0.0362** (-0.0363 -0.0433 -0.0291; 61 / 41); C1 V4096 head 512 tail 10 d48 -0.0348 (-0.0345 -0.0363 -0.0337; 29 / 20); C5 V4096 head 640 tail 8 d48 -0.0345 (-0.0333 -0.0391 -0.0310; 34 / 23); C2 V4096 head 384 tail 10 d40 -0.0287 (-0.0263 -0.0373 -0.0225; 73 / 49); C6 pending.
Predictions scored so far: the best combination beats the best single-axis arm (E3c, -0.0273) by 0.005 (45%): **right** (C4, 0.0157 better); some combination reaches 0.035 (30%): **right** (C4, C3); C4 (19 rules) worse than C2 (55%): **wrong** (C4 is 0.014 better); C3 beats C1 (50%): **right, narrowly** (0.0014).
Reading: the optimum moved: **the vocabulary lever reappears at 6144 once the head is kept at 384** (stage 1's 6144 arms had a 192-256 head and trailed 4096 e10), and the feed-forward bank can be cut to 19 / 13 rules at a 65% embedding share while the arm keeps improving. Several arms cannot be fitted at all (V6144 head 512 tail 8 at d48 needs a negative rule count): the next arms stay inside the feasible region.
E9 arms (fitted): **D4** V6144 head 384 tail 8 **d40** (45 / 30 rules, 99,703; C4 with a narrower block); **D6** V6144 head 448 tail 6 (42 / 28, 99,770); **D7** V6144 head 384 tail 6 (49 / 33, 99,878; C4 with a rank-6 tail and 30 more rules); **D8** V8192 head 384 tail 6 d40 (43 / 29, 99,828); **D11** V6144 head 512 tail 6 d40 (67 / 45, 99,612).
Predictions: the best E9 arm beats C4 (-0.0430) by 0.005 (35%); D7 beats C4 (45%: 30 more rules against a narrower tail); D4 beats C4 (40%: d40 helped by 0.0014 at V4096); D8 (V8192) is worse than D4 (V6144) (60%); the best arm of all five-seed results is a V6144 arm (55%).

## 13. E8 complete and E10, a third round (E10 registered 2026-10-21 21:00, after C6 (three seeds) and before any E10 arm)
**C6 V6144 head 512 tail 8 d40: BPC 1.0621, -0.0587** (-0.0622 -0.0642 -0.0497; 33 / 22 rules; embedding 66% of the parameters): the best arm by 0.016 over C4 (-0.0430). Prediction scored: C6 had no registered individual prediction; the E8 predictions stand as in section 12. The optimum is still moving: a large head (512) with vocabulary 6144 and a narrow block (40) beats every earlier arm, and the feed-forward bank is down to 33 / 22 rules.
E10 arms (fitted; blocks narrower than 48 need d divisible by the 8 mixer heads: 40, 32, 24): **F1** V6144 head 640 tail 8 d40 (20 / 13 rules, 99,773; embedding share 0.70); **F4** V6144 head 512 tail 8 d32 (73 / 49, 99,902); **F6** V6144 head 768 tail 8 d32 (49 / 33, 99,862); **F7** V6144 head 640 tail 8 d32 (61 / 41, 99,882); **F9** V8192 head 768 tail 8 d24 (34 / 23, 99,845; share 0.78); **F11** V8192 head 512 tail 8 d24 (55 / 37, 99,816); **F13** V6144 head 512 tail 10 d32 (29 / 20, 99,957); **F14** V8192 head 640 tail 8 d24 (45 / 30, 99,879).
Predictions: the best E10 arm beats C6 (-0.0587) by 0.005 (50%); the best E10 arm is a d32 or d24 arm (55%); some d24 arm is worse than C6 (60%: the block floor shrinks but the mixer state and the output scores narrow with d); F1 (20 rules) is worse than F7 (61 rules) (55%: the feed-forward floor again, as C4 was not); a V8192 arm beats every V6144 arm of the same head size (35%).

## 14. Stage 2: five seeds for the seven stage 1 survivors (recorded 2026-10-21; `v69-shape`; reference mean 1.1200 over seeds 0-4; `report/shape_stage2.md`; rule: 0.005 better on at least four of five seeds)
| arm | BPC | change (se) | per seed (0-4) | seeds past 0.005 | verdict |
|---|---|---|---|---|---|
| V4096 head 384 tail 10 (E3c) | 1.0941 | **-0.0259** (0.0019) | -0.0276 -0.0320 -0.0222 -0.0263 -0.0214 | 5 | PASS |
| V4096 head 384 tail 8 (E3b) | 1.0998 | -0.0203 (0.0013) | -0.0232 -0.0230 -0.0167 -0.0185 -0.0200 | 5 | PASS |
| V6144 head 256 tail 6 (E3d) | 1.1000 | -0.0200 (0.0012) | -0.0230 -0.0207 -0.0216 -0.0177 -0.0169 | 5 | PASS |
| V8192 head 256 tail 6 (E3f) | 1.1051 | -0.0150 (0.0012) | -0.0190 -0.0138 -0.0162 -0.0124 -0.0135 | 5 | PASS |
| head 640, V3072 (E1d) | 1.1107 | -0.0093 (0.0014) | -0.0089 -0.0123 -0.0046 -0.0121 -0.0085 | 4 | PASS |
| head 512, V3072 (E1c) | 1.1114 | -0.0086 (0.0015) | -0.0124 -0.0101 -0.0071 -0.0038 -0.0097 | 4 | PASS |
| d40, head 448 (E4b) | 1.1168 | -0.0032 (0.0019) | -0.0067 -0.0076 -0.0002 -0.0039 +0.0023 | 2 | no |
Six of the seven pass; the three-seed estimates held (changes of at most 0.002 between three and five seeds). Predictions: at least one embedding arm beats the reference on four of five seeds (50%): **right** (six arms); no block width beats d48 on four of five seeds (75%): **right** (d40 passes on two); d48 within 0.005 of the best width (70%): **right** (d40 gains 0.0032).
The combination arms of E8-E10 (C1-C6, D, F) have three seeds each and are the candidates for the 300M stage: C6 (V6144 head 512 tail 8 d40, -0.0587 on three seeds) leads; seeds 3-4 for the best of them follow the end of the E10 round.

## 15. E9 results (recorded 2026-10-21; `v69-shape`, three seeds; change against the refit reference, negative is better)
| arm | V / head / tail / d | rules | BPC | change | per seed |
|---|---|---|---|---|---|
| C6 (E8) | 6144 / 512 / 8 / 40 | 33 / 22 | 1.0621 | **-0.0587** | -0.0622 -0.0642 -0.0497 |
| D11 | 6144 / 512 / 6 / 40 | 67 / 45 | 1.0649 | -0.0559 | -0.0563 -0.0601 -0.0513 |
| D6 | 6144 / 448 / 6 / 48 | 42 / 28 | 1.0673 | -0.0536 | -0.0529 -0.0575 -0.0503 |
| D7 | 6144 / 384 / 6 / 48 | 49 / 33 | 1.0740 | -0.0469 | -0.0450 -0.0499 -0.0457 |
| D4 | 6144 / 384 / 8 / 40 | 45 / 30 | 1.0734 | -0.0474 | -0.0439 -0.0504 -0.0479 |
| C4 (E8) | 6144 / 384 / 8 / 48 | 19 / 13 | 1.0779 | -0.0430 | -0.0441 -0.0433 -0.0415 |
| D8 | 8192 / 384 / 6 / 40 | 43 / 29 | 1.0818 | -0.0390 | -0.0409 -0.0431 -0.0330 |
Predictions: the best E9 arm beats C4 by 0.005 (35%): **right** (D6, 0.0106 better; none beats C6); D7 beats C4 (45%): **right** (0.0039); D4 beats C4 (40%): **right** (0.0044); D8 (V8192) worse than D4 (V6144) (60%): **right** (0.0084 worse); the best five-seed arm is a V6144 arm (55%): V6144 arms hold every top-six place so far, to be confirmed on five seeds.
Reading: at vocabulary 6144 the head is the dominant axis (384 -> 448 -> 512 gains 0.0536 - 0.0469 = 0.007 and 0.0587 - 0.0536 = 0.005 per step), the block width 40 beats 48 by about 0.004, and the tail rank trades almost one for one against feed-forward rules (D11 tail 6 / 67 rules against C6 tail 8 / 33 rules: 0.003 apart). V8192 loses to V6144 at equal head and width.
