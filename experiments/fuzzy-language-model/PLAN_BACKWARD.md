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

## Registered 2026-10-11, before `v24-diag` (user: "do 1 and 2"): why is the consequent-first init harmful?
Three candidate causes: (H1) the hot output scale (the construction picked gain 1,024 in every layer); (H2) an unstable start at lr 6e-2 that a longer warm-up would tame; (H3) rule collapse: rules placed on frequent-token groups end up used by fewer positions (lost diversity).
Cells (30M characters, 96K model; `flm/run_tribble2.py`): `cfg1` (centroid init, gain capped at 1; seeds 0-2); `cfw300` (centroid init, warm-up 300; seeds 0-2; the warm-up-alone reference is v19-optA30's wu300, mean 1.0997); `random-save` and `cf-save` (seed 0, checkpoints saved).
Predictions: H1: `cfg1` improves on `cf` (1.1626) by at least 0.02, but stays worse than random (1.1095) by at least 0.01. H2: `cfw300` improves on `cf` by at least 0.02 and stays worse than the random wu300 reference. H3: after training the effective number of rules used (exp of the entropy of the winning-rule distribution, layer 0 and layer 1)
is at least 10% lower for `cf-save` than `random-save`. If H1/H2 close the whole gap (cfg1 or cfw300 within 0.005 of random) the construction becomes a candidate again; otherwise the TRIBBLE-order construction is dropped for this model.

### Registered 2026-10-11 (after seeing the `v24-diag` first pass, before the follow-up cell): `cfwide`
The first pass found layer 1's FFN dead after training in the consequent-first model (effective rules 3.2 vs 43.4, mean |gate| 0.0 vs 0.18; see Outcome below). A raw-firing Gaussian rule that stops firing gets no gradient, and the construction sets each rule's width from its *own group's* spread, which is narrower than the global spread.
`cfwide` = the same construction with every rule's width set to the global input std (centres still at the group means). Cell: 30M characters, seeds 0-2 (`flm/run_tribble2.py`, queued behind the chain). Prediction: layer-1 effective rules back above 20 and the final BPC within 0.01 of random; if BPC is still worse than random by 0.02 or more, the construction is dropped for this model.

## Outcome `v24-diag` first pass (GPU, 96K, 30M characters; seeds 0-2 unless noted)
| cell | BPC (s0 / s1 / s2) | mean | note |
|---|---|---|---|
| random | | 1.1110 | v22-tribble, same seeds |
| random, warm-up 300 | | 1.0997 | v19-optA30 |
| `cf` (centroid, gain chosen = 1,024) | | 1.1402 | v22-tribble, seeds 0-2 |
| **`cfg1`** (gain capped at 1) | 1.1403 / 1.1525 / 1.1455 | **1.1461** | no better than `cf` (+0.006) |
| **`cfw300`** (warm-up 300) | 1.1303 / 1.1190 / 1.1365 | **1.1286** | -0.0116 vs `cf`; the same size as the warm-up effect on random (-0.0113): no interaction |

**Rule usage after training (seed 0, 128 validation windows; winner = argmax of |gate| x ||consequent||):**
| model | layer 0: effective rules (of 72) / rules with >1% share / mean abs gate | layer 1: same |
|---|---|---|
| random | 56.1 / 38 / 0.406 | 43.4 / 24 / 0.181 |
| `cf` (consequent-first) | 46.4 / 29 / 0.500 | **3.2 / 6 / 0.000** |

**Predictions scored.** H1 (gain cap closes >= 0.02): **wrong**: capping the gain changed nothing (+0.006), so the hot output scale is not the cause. H2 (warm-up 300 closes >= 0.02): **wrong**: it closes 0.012, exactly what it gives a random init, so the harm is not an instability that a slower start cures.
H3 (>= 10% fewer effective rules): **right, and much stronger than registered**: layer 0 uses 17% fewer effective rules, and **layer 1's FFN is essentially dead after training (3.2 effective rules, mean |gate| 0.000)**: the consequent-first model is running with a one-layer FFN, which accounts for a loss of ~0.05 BPC.
Mechanism (hypothesis, not yet tested; `cfwide` below tests it): raw-firing Gaussian rules (`ffn_norm none`) get no gradient once they stop firing, and the construction gives each rule the width of its own group, narrower than the global spread, so in layer 1 (where the input moves as layer 0 trains) the rules are pushed off the data and die.

### `cfwide` outcome (30M chars, seeds 0-2)

Final BPC 1.1544 / 1.1575 / 1.1508 (mean 1.1542); random-init reference on the same stream 1.1095 (5 seeds), so +0.045 worse, 3/3 seeds, vs. +0.053 for plain `cf`. Prediction "within 0.01 of random" — **wrong**; the 0.02 drop rule triggers. Widening the rules recovers only ~0.008 of the ~0.053 gap, so narrow widths are at most a minor part of the cause. Layer-1 effective-rule count was not measured (no checkpoint saved for this cell), so the "rules die from narrow widths" mechanism is neither confirmed nor excluded; the dead layer-1 FFN observed for `cf` stays a measured fact with an untested cause. Decision: the consequent-first init is dropped for this model.

## Hybrid construction (`v26-hybrid`, CPU, 4M chars, seeds 0-1 paired with `v19-pilot` base; user request 2026-10-11)

Idea (user's): build only a few *key* rules consequent-first (`keep` = 4, 12 of the 72 per FFN layer) and leave the rest at their random init, so the live random rules keep layer 1 from collapsing while the clustered rules supply the head start. Code: `keep` in `flm/conseq_first.py`; variants `hyb4`, `hyb12` (`hyb24` registered, not yet run) in `flm/run_conseq.py`; the output gain applies only to the built rules. Reference cells at the same size: pilot base (1.4220 / 1.4399, sd 0.010) and plain `cf`.
Registered before running (screen, ranking only; 2 seeds cannot support a claim): (1) plain `cf` is worse than base at 4M by > 0.02 (30M gap was +0.053); (2) `hyb4`/`hyb12` land between base and `cf`, none better than base by more than 0.010; (3) post-training layer-1 effective rules for the hybrids stay > 20 (checked from the saved seed-0 checkpoints). A hybrid that beats base by >= 0.010 on both seeds earns a 30M/5-seed GPU run; otherwise the backward line stays closed.

### `v26-hybrid` first pass (CPU, 4M chars, seeds 0-1; base 1.4220 / 1.4399)
cf 1.4377 / 1.4524 (+0.016 / +0.012); hyb12 1.4423 / 1.5632 (+0.020 / +0.123); hyb4 1.4573 / 1.6724 (+0.035 / +0.232). Predictions: (1) cf worse than base by >0.02 at 4M — wrong (+0.012 to +0.016; the gap grows with run length, +0.053 at 30M); (2) hybrids between base and cf — wrong: seed 0 sits at or above cf and seed 1 is far worse; (3) layer-1 effective rules not checked. Both seed-1 hybrids picked the top output gain (1024) in both layers, and every hybrid's init CE is ~7.0 nats (uniform = 6.93): with random rules mixed in, the gain search has no signal and can peg at 1024, which diverges early. So the first pass is confounded by the gain rule, not a clean test of the hybrid idea. Follow-up registered before running: `hyb4g1`, `hyb12g1` (same, gain capped at 1), seeds 0-1; prediction: the seed-1 blow-up disappears (BPC within 0.03 of base) but neither beats base by 0.010 on both seeds.

## Coverage hybrid (`v27-cover`, GPU, 30M chars, seeds 0-4; user request 2026-10-11)

Idea (user's): cluster 25% of the rules (18 of 72 per FFN layer, consequent-first), then place the other 75% "intelligently" rather than randomly. Implemented as a *coverage* rule: the remaining centres are chosen by D^2 (k-means++) sampling over the layer's whitened FFN inputs, measured against the centres already placed (so they land where the built rules do not reach); width = the global input spread; consequents stay at their small random init; the built rules' output gain is capped at 1 (the unconstrained search pegged at 1024 in `v26-hybrid`). Arms: `cov25` (the idea), `cov0` (all rules coverage-placed, no clustering: isolates the cover from the clustering), `hyb18g1` (25% built, rest random: isolates intelligent vs random placement). References at the same size and seeds (v22-tribble): random 1.1095, antecedent-first `data` 1.1029.
Registered before running (5 seeds each, so these can support a ranking and, with paired differences, a claim): (1) `hyb18g1` is within +0.02 of random (the gain cap removes the blow-up) but not better; (2) `cov0` is about equal to `data` (within 0.01), because both place centres on real inputs; (3) `cov25` is no better than `cov0` by more than 0.005 — the clustered quarter adds nothing over placing every centre by coverage; (4) nothing beats `data` by more than 0.01. Expect the init effect to wash out by step 100-300 as in `v22-tribble`; a coverage arm that is still ahead of `data` at the final step by >= 0.01 on >= 4 of 5 seeds is the only result that would reopen this line. Post-training layer-1 effective rules are measured from the seed-0 checkpoints (`rule_dist` usage) to see whether any arm kills layer 1.
