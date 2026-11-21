# Plan: two short CPU experiments on the RFLM (fuzzy-delta mixer), GPU session, 2026-10-11

User: "a couple short CPU experiments on the RFLM: treat it as a fixed recurrence, or an expansion to consequent propagation."
Base: BPE1024 D (d 48, 2 layers, 8 heads, 32 rules/head, ctx 512, lr 6e-2, 4M characters, 96,416 params); paired by seed with `v19-pilot` base_s0 / base_s1 (1.4220 / 1.4399).
Code: `flm/rflm_variants.py` (registers the mixers; `test_rflm_variants.py`: starts equal to D when the propagation matrix is zero, parallel form equals the stepwise recurrent form), driver `flm/run_rflm.py`. Nothing in `models.py` / `train.py` was touched.

* **fixed recurrence** (`fuzzydelta_static --decay fixed`): the per-head write strength beta and forgetting alpha are learned constants, not functions of the input. The delta rule then runs as the same linear system at every step.
* **consequent propagation** (`fuzzydelta_cpread`, `fuzzydelta_cpboth`): the state's consequents theta_r are spread to related rules by a learned rule-relation matrix A (R x R, shared by the 8 heads, zero-initialised = D):
  read side `phi' = phi + phi A`, or both read and write side. +1,024 parameters per layer (98,464 total, under the cap).

Predictions (written before any run; Delta = variant - D, paired by seed; counts only if both seeds have the same sign and the mean exceeds 0.010 BPC):
1. **fixed is worse than D by more than 0.01** (input-dependent write/forget gates are what lets a recurrent mixer choose what to keep; ~65%).
2. **cpread is within 0.010 of D** (a zero-initialised relation learned in ~1,200 steps; ~65%); P(better by more than 0.010) about 25%.
3. **cpboth is no better than cpread**.

## Outcome `v20-rflm` (CPU, 4M characters, seeds 0-1, paired with the pilot's base_s0 / base_s1 = 1.4220 / 1.4399; 6 jobs, 740-1,000 s each)
| variant | params | BPC s0 / s1 | paired delta s0 / s1 | mean | rule (same sign and mean > 0.010) |
|---|---|---|---|---|---|
| fixed recurrence (`fuzzydelta_static --decay fixed`) | 94,880 | 1.4270 / 1.4447 | +0.0050 / +0.0048 | +0.0049 | not resolved (consistently slightly worse) |
| consequent propagation, read side | 98,464 | 1.4192 / 1.4480 | -0.0028 / +0.0081 | +0.0027 | not resolved (inconsistent) |
| consequent propagation, both sides | 98,464 | 1.4174 / 1.4356 | -0.0046 / -0.0043 | -0.0045 | not resolved (consistently slightly better) |

**Predictions scored:** (1) fixed worse than D by more than 0.01: **wrong**: it costs only ~0.005 BPC at 4M characters while removing the data-dependent write and forget gates (1,536 fewer parameters), so the input-dependent gating of the RFLM is worth about 0.005 BPC at this budget, not the large effect I expected.
(2) read-side propagation within 0.010 of D: **right**. (3) both-sides no better than read-side: **wrong in the mean** (-0.0045 vs +0.0027) but neither clears the 0.010 bar.
**Reading.** Nothing here is resolved at 4M characters and two seeds, so there is no claim either way for consequent propagation; the one robust finding is the *small* cost of a fixed recurrence. The sign pattern (cpboth consistently better by ~0.004, fixed consistently worse by ~0.005) is worth one 30M-character, 5-seed GPU check (cpboth and fixed vs D) before anything is believed; not yet run.
