# PLAN_DUST: can an FLM be trained with a DUST-like forward-only gradient estimate, and are other cheap optimisers worth anything? (CPU exploration, 2026-10-23)

Question from the owner: explore, on the CPU, a simple SGD approach like DUST (qlabs-eng/dust, https://qlabs.sh/research/dust) for the 100K models on a small dataset: **can** an FLM be trained this way, and is it faster in any meaningful way? Mid-run the owner judged the DUST results unpromising and asked for other inspiration instead; the second half records that.

## What DUST is (from the paper page; the repo README gives no update rule)
A zeroth-order method: Gaussian noise is added to a layer's *output* at every token, each token's noise is scored by the change in loss (its own, plus later tokens' with a decay gamma), and the scored noise is averaged over K draws to estimate the layer's output error; the weight gradient is then the usual outer product with the layer input, fed to SGD with momentum. The paper reports it matching or beating backprop SGD at 1M-10M tokens for 2M-243M parameter transformers in tokens seen, and says wall-clock is not reported and that it is not yet compute-efficient enough to replace backprop.

## What was built (`flm/dust_cpu.py`; our adaptation, not the authors' code)
The 96,416-parameter hybrid-style model (BPE1024, d48, 2 blocks, fuzzy delta-rule mixer, TSK rule bases, `flm.run_tribble` defaults) is cut into six units (embedding; mixer and rule-base sub-layers of both blocks; head). The output of each unit is jittered (relative noise scale 0.1, per-token credit, gamma 0), only the later stages are recomputed, and the unit's own parameter gradient is the local vector-Jacobian product of its estimated output error through that unit alone; autograd never crosses a unit boundary. The head jitters the logits. Not done: DUST's special handling of attention internals, the vocabulary slab, the residual scalars, and per-layer tuning of noise scale and credit decay.

## Results (one seed; batch 8 x 128 tokens; 300 steps; validation bits per character on 200 fixed windows of 128 tokens; 8 CPU threads, other jobs running, so seconds are rough)
**Estimator quality** (cosine with the true gradient, one batch, untrained model; per unit: mixers and rule bases):
| draws K | all parameters | per unit |
|---|---|---|
| 8 | 0.25 | 0.48-0.68 |
| 32 | 0.49 | 0.75-0.83 |
| 128 | 0.75 | 0.92-0.97 |
| 512 | 0.91 | 0.97-0.99 |
At a briefly trained point (150 Adam steps) the same: 0.19 / 0.36 / 0.63 / 0.85. So the estimator works on the fuzzy units (the rule bases and the delta-rule mixer), but needs hundreds of draws for a usable gradient; the all-parameter cosine is held back by the embedding and head.
**Training:**
| method | final val bits per char (300 steps) | seconds for 300 steps |
|---|---|---|
| DUST-like, K=8, SGD momentum lr 0.3 | 2.192 | 174 |
| DUST-like, K=32, lr 0.3 | 1.949 | 725 |
| backprop + SGD momentum, lr 0.03 / 0.1 / 0.3 / 1.0 | 2.079 / 1.813 / 1.706 / 1.681 | 11-16 |
| backprop + Adam, lr 3e-3 / 1e-2 | 1.649 / 1.585 | 16-26 |
Verdict: **an FLM can be trained this way (the loss falls steadily; the gradient estimate aligns with the true one), but it is neither faster nor better.** Per step it is 11x (K=8) to 45x (K=32) slower than backprop on this model, and at equal steps it is worse than backprop SGD (1.95 against 1.68 at K=32); K=512 would be about 28 s per step. A forward-only method has no advantage here: the model is small, so backprop is cheap, and the fuzzy rule bases make each forward evaluation (rule firing, delta-rule recurrence) the costly part. Not run (stopped when the owner redirected): K=32 with lr 1.0, credit decay 0.9, K=128.

## Other inspiration (owner's redirect): cheap first-order optimisers, same model, same 300 steps (`flm/cheap_opts.py`)
| optimiser | lr | val bits per char | seconds |
|---|---|---|---|
| **Adam** | 1e-2 | **1.585** | 25 |
| Lion | 3e-3 | 1.599 | 33 |
| Muon (2-D matrices) + Adam (the rest) | 0.01 | 1.620 | 20 |
| Muon | 0.03 / 0.1 | 1.623 / 1.698 | 30 / 49 |
| Adam | 3e-3 | 1.649 | 16-20 |
| Lion | 1e-3 / 3e-4 | 1.722 / 2.188 | 24 / 22 |
| SGD + momentum | 1.0 | 1.681 | 16 |
Reading: nothing beats Adam at lr 1e-2 in 300 steps; Lion is within 0.015 and Muon within 0.035 (one seed; noise perhaps 0.01-0.02, so these are ties at best); plain SGD with momentum works but needs a large rate (1.0) and trails Adam by 0.1. Adam's rate is not bracketed (1e-2 is the best tried). Wall-clock per step is not meaningfully lower for any alternative on this CPU (the sign and orthogonalisation steps cost as much as they save).

## What is left that is specific to FLMs (not tried)
TSK rule bases are **linear in their consequent parameters** (`conseq`, the first-order slopes): ANFIS-style hybrid learning (solve those by recursive least squares, train antecedents by gradient) and Gauss-Newton or natural-gradient steps that exploit that linearity are the fuzzy-native alternatives to Adam. The cost and the benefit depend on the downstream layers, which are not linear in the consequents, so a layerwise local least-squares target would be needed; that is a design question for the owner, not a quick test.

## Files
`flm/dust_cpu.py` (estimator, training and cosine modes; `python -m flm.dust_cpu cosine|train --method dust|backprop-sgd|backprop-adam|backprop-lion|backprop-muon`), `flm/cheap_opts.py` (Lion, Muon). Raw results in `outputs/dust/` (untracked).
