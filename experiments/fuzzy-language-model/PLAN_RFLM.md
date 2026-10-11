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
