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
