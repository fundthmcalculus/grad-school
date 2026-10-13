# Deconstructing the output distribution into TSK rules (GPU session, 2026-10-11; for identification now, distillation later)

User: use the output probability distribution to deconstruct the TSK rules, as a way to identify rules and, later, to distill from existing models.
Tool: `flm/rule_dist.py` (CPU, forward passes only; `python -m flm.rule_dist <ckpt>`; results `outputs/v21-ruledist/*.json`).

## The decomposition is exact (not an approximation)
`logits = RMSNorm(x_L) E^T`, and RMSNorm is a per-position scalar s_t times a fixed per-dimension weight, so the logits are a sum of per-component terms.
Each first-order FFN rule contributes **g_r(t) * pi_r**, where `pi_r = (conseq_r * norm_weight) E^T` is a fixed vector over the vocabulary and `g_r(t) = firing_r * slope_r(x)` a scalar gate
(times s_t). So the output distribution is `softmax( emb term + mixer terms + sum_r g_r(t) pi_r )`: a rule is a fixed *direction in logit space* that is turned up or down (and flipped) by its gate.
Manipulation check A: the reconstruction matches the model's logits to **1.7e-5** (max |error|) on both checkpoints. Layer-0 rules also act through layer 1, which the direct decomposition does not see;
experiments C and F are true forward passes and so include that.

## Results (128 validation windows x 256 tokens; BPC on those windows; two checkpoints, both 96,416 params)
| | 3B-char model (0.9777 on the full val) | 1B-char model (0.9796) |
|---|---|---|
| full model, these windows | 1.0359 | 1.0417 |
| B: drop layer-0 rules / drop layer-1 rules / drop mixer 1 / drop mixer 0 / drop embedding (direct path) | 1.778 / 1.721 / 1.427 / 1.043 / 1.045 | (3B file) |
| C: only the top-k rules fire per position per layer, k = 1, 2, 4, 8, 16, 32 (of 72) | 4.25 3.13 2.23 1.59 1.22 **1.062** | 3.78 3.00 2.22 1.64 1.24 **1.072** |
| F: gate vectors projected to rank r (r = 4, 8, 16, 24, 36 of 72) | 2.91 2.54 2.15 1.83 1.49 | 3.06 2.54 2.04 1.76 1.40 |
| D: participation rank of the rule profiles (of 72), near-duplicate pairs (cos > 0.9) | 13.7 / 13.9, none | (3B file) |
| E: dominant rule's stated top-10 tokens contain the true next token | **0.187** | **0.152** |
| E references: model's own top-10 / the 10 most frequent tokens | 0.814 / 0.245 | 0.812 / 0.245 |

## Reading
1. The output is a **superposition**, not a lookup. A short rulebook does not reproduce it: 8 rules per position cost +0.55 BPC, 16 cost +0.18, it takes ~32 of 72 active rules per position and layer to be within 0.03.
   Low-rank gates are worse (rank 36 costs +0.36 to +0.46): the gate code is not low-dimensional, so a student with far fewer rules than 72 would have to be trained differently, not merely pruned.
2. A rule's stated consequent is **not a faithful prediction on its own**: when a rule dominates a position, its signed top-10 tokens contain the true token 15-19% of the time, *below* the 24.5% from naming the ten most frequent tokens.
   The earlier readable-rule examples (rules 30, 49) are real but describe the direction a rule pushes, not what the model then predicts. Faithful rule reading needs the full gate vector, not the single winner.
3. The profiles are distinct (no near-duplicates) but live in a low-dimensional space (participation rank ~14 of 72), which is the geometry a distillation target could use: the teacher's rule outputs span ~14 independent vocabulary directions.
4. Layer-0 and layer-1 rules each carry ~0.7 BPC of direct-path effect, and mixer 1 ~0.4; the embedding and mixer 0 add almost nothing directly (+0.01): the model's logits are built almost entirely by the FFN rules.

## What this makes possible (not run)
* **Rule distillation:** train a student's gates to match the teacher's gate vectors (or the teacher's per-layer logit terms `g pi`) layer by layer, in addition to the output KL, since the decomposition gives per-rule, per-layer targets.
* **Fewer-rule student:** because top-k and low-rank both degrade quickly, test a student with R = 36 trained by gate-matching distillation against a student trained on data alone, at equal parameters.
* **A calibrated rule reader:** replace "winner top-10" by the set of rules needed to reach most of the logit mass, then check hit rate again (E should rise toward 0.8).
