# PLAN_BIJECTIVE: a three-layer all-fuzzy language model with an invertible meaning layer (design, 2026-10-26; nothing trained yet)

Owner's layout: (1) an N-dimensional logical/semantic embedding layer that is a **bijective transform**, so the same fuzzy layer is also the de-embedding layer; (2) the logic space is routed back through that layer to generate text; (3) sampling is dropped, output is the highest-likelihood text; (4) every layer is an **order-1 TSK system**.

## What "bijective" can mean here
Tokens are V discrete items; R^N is continuous, so no map between the two is a bijection. What can be exact is a bijection **F: R^N -> R^N** with the tokens as V points of its domain (a codebook c_v, the crisp embedding), their images z_v = F(c_v) being the "meaning vectors" and F^-1 the de-embedding. Decoding is then a likelihood over the V codewords, which is the only place discreteness enters. F is exactly invertible by construction (not trained to be approximately invertible): coupling layers whose scale and shift are order-1 TSK systems of the untouched half of the coordinates (`flm/bijective_tsk.py`), inverse = the same two fuzzy systems evaluated on that half, no solve, log-determinant = sum of the scales.

## Architecture
1. **Meaning layer F** (invertible order-1 TSK coupling stack): token id -> codebook vector c_v (the existing tiered embedding can supply it: crisp head rows, fuzzy tail) -> z = F(c_v).
2. **Logic layer**: the sequence model that works in z space and predicts the next meaning vector z_hat from the context: fuzzy delta-rule memory mixer plus order-1 TSK feed-forward rules, as in the current all-fuzzy blocks.
3. **Routing back**: u = F^-1(z_hat); token likelihood p(v | z_hat) proportional to exp(-|u - c_v|^2 / 2 tau^2) with one learned temperature tau; output = argmax_v (greedy, no sampling). Loss = cross-entropy of that likelihood, so bits per character stay comparable with every earlier number.
Two readings of the output, as an ablation: distances measured in c space after F^-1 (the routed-back reading, the one the owner described; the gradient flows through F^-1) against distances in z space to the cached z_v (needs no F^-1 at inference; if it wins, the inverse is conceptual only).

## Checks done (`python -m flm.bijective_tsk`, CPU, seconds)
N=32, R=16 rules per system, 4 couplings, parameters perturbed hard so the map is far from the identity: float64 round trip on 6,144 codewords max error 8e-9; the log-determinant equals the autograd Jacobian's to 6 digits on three points; F(F^-1(z)) on random logic-space points 4e-8. **Float32 is where it can fail**: with the scale bound s_max = 2 the stacked exp(s) gains reach 100x and the float32 round trip lost 0.2% of the codewords at the nearest-codeword decode; with s_max <= 1 it recovered 100%, and with s_max 0.5 and a mild perturbation the error was 3e-6. Design rule: s_max in 0.5-1, and the training log reports the round-trip recovery rate in float32 every evaluation.

## What the experiment has to separate
The current model already ties its output head to the embedding (scores are dot products with the same rows). A bijection adds a *geometry*: an invertible warp between the space where tokens are stored and the space where the logic layer works. So the comparison that means something is **the same model with F = identity** (distance-likelihood head on the plain codebook), not only the F7 baseline. If F does not beat identity-F, the bijection costs parameters for nothing at this scale.

## Plan (registered before any run)
Scale first: the 100K budget (F7 trains 30M characters in 90 s, so five seeds per arm cost about 10 minutes). F costs parameters (N=32, R=16, 4 couplings is 39K, too much at a 100K budget): arms use R=6-8 and 2 couplings (about 8-12K) and take the parameters from the feed-forward rules, so the total stays near 99.9K. Arms, five paired seeds, 30M characters, the F7 recipe: (A) F7 as is; (B) identity-F with the distance head; (C) bijective F, c-space likelihood; (D) bijective F, z-space likelihood. Then a 300M confirmation of the best arm before it is quoted.
Predictions: (B) is within 0.01 of (A) (60%); (C) beats (B) by at least 0.005 on 4 of 5 seeds (30%); (C) beats (A) (20%); (D) within 0.005 of (C) (55%); float32 round-trip recovery stays 100% through training (75%); the learned tau settles in a range where the likelihood is neither one-hot nor flat (65%).

## Decisions I need (my recommendation first)
1. **Context in the logic layer.** A TSK system maps one vector to one vector, so a layer made only of TSK systems cannot see earlier tokens. I propose keeping the fuzzy delta-rule memory mixer (the current all-fuzzy mixer, itself a fuzzy rule system but not strictly order-1 TSK) as the logic layer's routing, with the order-1 TSK feed-forward rules after it. The strict alternative is an order-1 TSK over a window of the last few meaning vectors; it is smaller but sees little history.
2. **Scale of the first test:** 100K budget (recommended), or straight to the 14M model.
3. **Go for the 100K screen** (about 40 minutes of GPU in total for four arms and five seeds).

## 10K-parameter first test (owner: "start as small as possible"; registered 2026-10-26 before any run)
**Setting.** Character-level TinyStories (98 symbols, so the codebook is 98 x 16 = 1,568 parameters; BPE vocabularies would eat the whole budget), d = 16, two blocks, two mixer heads, the all-fuzzy blocks of `run_allfuzzy` (fuzzy norm and gates, rule-gated low-rank projections, raw-input first-order TSK feed-forward rules), context 256, batch 32. Runner `flm/run_bijective.py`, arms by env `BIJ`; F = 2 couplings, 4 rules per TSK system, s_max 0.75 (1,408 parameters at d = 16).
**Arms, sized to the same total:** `tied` 9,963 parameters (the usual tied dot-product head; feed-forward rules 42 / 29); `ident` 9,964 (codebook with the distance likelihood and F = identity, same rules); `z` and `c` 9,946 each (F in, rules 32 / 21 pay for it). Parameter counts are measured, not estimated; the F arms carry 10 fewer layer-1 rules and 8 fewer layer-0 rules than the control.
**Protocol.** One learning-rate screen per arm (10M characters, seed 0, lr 1.5e-2 / 4e-2 / 1e-1, warm-up 100, final-lr floor 0.1), then the best rate per arm at 30M characters, seeds 0-4, paired by seed; a gain counts when it is at least 0.005 bits per character on 4 of 5 seeds. If the best rate is an end of the screen's range it is reported as unbracketed. Every log line carries tau and the float32 round-trip recovery of the codebook.
**Predictions.**
- `tied` ends between 1.7 and 2.1 bits per character at 30M (70%).
- `ident` within 0.03 of `tied` (60%).
- `z` beats `ident` by the gain rule (35%); `c` beats `ident` by the gain rule (30%); `c` is worse than `z` (60%).
- Round-trip recovery stays 1.0000 on every evaluation of every F arm (80%); tau ends in 0.3-3 (60%).
- The best lr is the middle value for at least three of the four arms (50%).
**Reading rule.** If neither F arm beats `ident`, the invertible layer does not pay for its 1,408 parameters at this size, and I say so; if `z` beats `ident` but `c` does not, the geometry helps but the routed-back inverse does not.
