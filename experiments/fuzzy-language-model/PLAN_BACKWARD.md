# Plan: consequent-first ("backwards") construction for the FLM and the RFLM (GPU session, 2026-10-11)

The user's request: while the other runs train, build the TRIBBLE approach for the FLM and RFLM.
**My reading, stated so it can be corrected:** TRIBBLE builds a fuzzy system *consequent-first* (`research/proposal-defense/prose/04-fast-fis-synthesis-mog.md`:
"commit to the answers first ... and only then work backward to the antecedents that select them"; `experiments/fis-to-neural-net/README.md`:
a gradient-free construction in a fraction of a second, then trained on as a hot start). The language-model analogue: the answers are the next tokens,
the rules are built from them, and gradient descent only refines. The existing `--rule-init data` is the opposite order (antecedent-first: centres on random inputs).
The FLM is the TSK FFN; the RFLM is the fuzzy-delta recurrent mixer (D). Stage 1 is the FFN; the mixer is stage 2.

## Stage 1 (built): `flm/conseq_first.py`, no gradients, layer by layer
1. answers: frequency-weighted spherical k-means of the next-token embeddings into R groups; every position joins the group of its own next token;
2. backward: rule r's antecedent = the Gaussian (centre = mean, width = std) of the FFN inputs of group r's positions;
3. consequents: group centroid direction (`cf`) or ridge least-squares of the target embeddings on the rules' firing (`cf-ls`), times one gain per layer chosen by cross-entropy.
Check: `python -m flm.conseq_first_check` (forward passes only, BPE1024 D d 48, 2 layers, 64 construction windows, 300K validation characters).

### First measurement (seed 0, no training; uniform over the vocabulary = 3.16 BPC, unigram = 2.617 BPC)
| init | val BPC |
|---|---|
| random | 3.2928 |
| antecedent-first (`--rule-init data`) | 3.2922 |
| consequent-first, centroid (`cf`) | 3.1588 |
| consequent-first, least squares (`cf-ls`) | 3.0024 |

**Reading.** The construction beats random and antecedent-first, but **it does not reach even the unigram (2.617)**, let alone the trained 1.11. Two reasons seen while building it:
the tied embedding is still random at construction time, so clusters of next-token embeddings are arbitrary token groups, not meaningful "answers"; and the best gain hit the top of the grid
(1024), i.e. the output RMSNorm swallows scale and the construction cannot express a sharp readout. The earlier n-gram init (RESULTS.md, `screen`) did start much lower and still washed out within
~100 steps with no final gain. **Prior for the training test below: the head start will not survive, and the final BPC will not move beyond seed noise (0.005 at 30M).**

## Stage 1b (registered before it runs; GPU, after `v18-3b`): does the head start matter at all?
Cells: random / data / cf / cf-ls on the 96K model, seeds 0-4, 30M characters. Needs a `--rule-init conseq` hook in `train.py` (not yet added: `train.py` is imported by the CPU screen `v19-small` that is
running now). Predictions: (a) steps to reach 1.5 BPC at least 25% fewer for cf-ls than random; (b) final BPC within 0.005 of random for every variant. If (b) fails in the good direction it is the first
sub-100K gain from structure-first, and goes to 10 seeds.

## Stage 2 (not built): the RFLM mixer, consequent-first
The mixer's rules select what to write into the recurrent state; the "answers" are the value vectors written. Candidate construction: cluster the values that a trained layer-0 writes (or targets
of the next token through the output map) and place each head's key rules on the keys that preceded them. Not started; it needs a design decision (what is the mixer's answer?) and stage 1b's result first.

## Other uses of the same idea (not run)
* a gradient-free *fit* of only the readout/consequents given fixed antecedents (closed-form, seconds): what a fully TRIBBLE-style model would be, with a quality number to compare to a trained one;
* backward target propagation across layers instead of the forward layer-by-layer pass used here.

## Clustered answers: by frequency or by association (user request, 2026-10-11; init-only, seed 0, same check; `flm/assoc.py`)
Answer groups can now come from (`freq`) equal-probability-mass buckets of unigram frequency, (`assoc`) a spherical k-means of the successor vectors of a PPMI-SVD of the bigram counts
(60M training tokens, k = 48, context smoothing 0.75), or (`emb`) the model's own random embedding (the first measurement). The tied embedding is either random or **set to the PPMI-SVD successor vectors** (`emb=succ`, rms 0.1).
Init validation BPC (uniform 3.16; **unigram 2.617**; random init 3.2928; antecedent-first 3.2922):

| answer groups | consequents | embedding random | embedding = PPMI-SVD successor |
|---|---|---|---|
| model embedding (`emb`) | centroid | 3.1588 | **2.8123** |
| model embedding (`emb`) | least squares | 3.0024 | 3.0240 |
| frequency (`freq`) | centroid | 3.1210 | 2.9295 |
| frequency (`freq`) | least squares | 3.0018 | 3.0181 |
| association (`assoc`) | centroid | 3.1709 | 2.8123 (same clusters as `emb` here) |
| association (`assoc`) | least squares | **2.9789** | 3.0240 |

**Reading.** (1) The choice of how to cluster the answers moves the starting loss by only 0.02-0.15 BPC: frequency and association are no better than the random-embedding k-means once the consequents are fit by least squares (3.00 / 3.00 / 2.98).
(2) What helps is a *meaningful embedding* with centroid consequents (2.81, the best), not the clustering rule. (3) **Nothing reaches the unigram (2.617)**, and the PPMI-SVD embedding explains part of why: PPMI divides out the
token marginals, so the embedding carries no frequency direction and the bias-free tied readout cannot express a unigram prior. Next (not run): add a log-frequency dimension to the aligned embedding.
(4) Init-only: the n-gram init (RESULTS.md) showed head starts like this wash out within ~100 steps. The test that matters is the training curve (stage 1b), which needs the GPU and the `--rule-init conseq` hook.
A different reading of "clustered embeddings" would be to *share* embedding rows within frequency/association clusters (the embedding is 49K of the 96K parameters); that is a parameter-budget test, not built.

## Registered 2026-10-11, before `v22-tribble` (stage 1b; user: "generate the results for the TRIBBLE-form FLM")
GPU, one job at a time: 96K BPE1024 D, 30M characters, seeds 0-4, evaluated at step 0 and every 100 steps. Variants (`flm/run_tribble.py`, `flm/run_conseq.py`, `--rule-init data` is the switch, train.py untouched):
`random` (no rule init), `data` (antecedent-first, existing), `cf` (answers = k-means of the model embedding, centroid consequents), `cfls` (least-squares consequents), `assocls` (answers by PPMI-SVD association, least squares),
`succcf` (embedding set to the PPMI-SVD successor vectors, centroid consequents). 30 jobs, ~1 h.
Predictions: (a) **the head start is gone by step 300**: every construction variant is within 0.02 BPC of `random` at step 300; (b) **final BPC at 30M is within 0.005 of `random` for every variant** (random's 5-seed mean was 1.1095 +/- 0.0048 at the 30M reference);
(c) `succcf` reaches 1.5 BPC in at least 25% fewer steps than `random` (the init BPC advantage is the largest, 2.81 vs 3.29). (b) failing in the good direction would be the first sub-100K gain from structure-first; it would go to ten seeds.

## Outcome `v22-tribble` (GPU, 96,416 params, 30M characters, seeds 0-4, 30 jobs; evaluated every 100 steps; paired by seed against `random`)
| init | step 0 | step 100 | step 300 | step 600 | final BPC (mean +/- sd) | paired delta vs random | seeds worse |
|---|---|---|---|---|---|---|---|
| random (no rule init) | 3.267 | 1.5836 | 1.3571 | 1.2263 | **1.1095 +/- 0.0107** | 0 | - |
| antecedent-first (`--rule-init data`, existing) | 3.267 | 1.5776 | 1.3531 | 1.2206 | 1.1029 +/- 0.0070 | **-0.0066** | 1 of 5 |
| consequent-first, centroid (`cf`) | 3.170 | 1.6024 | 1.3857 | 1.2717 | 1.1626 +/- 0.0319 | **+0.0531** | **5 of 5** |
| consequent-first, least squares (`cfls`) | 3.013 | 1.6181 | 1.3960 | 1.2753 | 1.1605 +/- 0.0156 | **+0.0510** | **5 of 5** |
| association groups + least squares (`assocls`) | 2.991 | 1.6147 | 1.3948 | 1.2744 | 1.1572 +/- 0.0059 | **+0.0477** | **5 of 5** |
| aligned PPMI embedding + centroid (`succcf`) | 2.865 | 1.6411 | 1.4078 | 1.2845 | 1.1737 +/- 0.0072 | **+0.0642** | **5 of 5** |

**Registered predictions scored.** (a) head start gone by step 300: **gone by step 100, and it reverses**: every consequent-first variant is *behind* random from step 100 on (+0.02 to +0.06 at step 100). (b) final within 0.005 of random: **wrong**: all four consequent-first variants are worse by 0.048-0.064 in 5 of 5 seeds,
far outside the seed spread (0.006-0.032); this is an actively harmful init, not a washed-out one. (c) `succcf` reaches 1.5 BPC in 25% fewer steps: **wrong** (evaluation grid is 100 steps; random, data and the cf variants all reach 1.5 at step 200, `succcf` at 200-300).
The antecedent-first init is marginally *better* than random (-0.0066, 4 of 5 seeds), which is the opposite ordering to the TRIBBLE-order claim, but it is a small effect on five seeds and not claimed.
**Reading.** In this model, committing to the answers first and building the antecedents backward from them is worse than random placement, by about 0.05 BPC at 30M characters, and the better the starting loss (2.87 for `succcf`), the worse the final one. Seen while building: the construction chose its largest gain (1,024) in every layer,
i.e. it starts with a hot output scale that the RMSNorm then rescales. Whether the scale, the rule placement (collapse onto frequent-token groups) or the loss of rule diversity is the cause is what the diagnosis below tests (`cfg1`: gain capped at 1; `cfw300`: warm-up 300; rule usage after training).
