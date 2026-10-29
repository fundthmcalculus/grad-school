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

## Owner's constraint (2026-10-26): the invertible layer is non-negotiable
The meaning layer F stays in every arm of the design whatever the 10K numbers say. The reading rule above is therefore not a go/no-go: if F does not beat `ident` at this size, the finding is "this F, at this size and with this training, costs more than it returns", and the next steps vary *how F is built* (more couplings or rules, a different s_max, initialisation as the identity, the codebook's own fuzzy structure, an auxiliary loss on the round trip) rather than whether to have it. The `ident` and `tied` arms stay as controls so any gain or loss is attributable, not as candidates to replace F.

## Sampling run across sizes (owner: "a sampling run of sizes, then work on full-sample results"; registered 2026-10-26)
Sizes 5K, 10K, 20K, 40K, 80K parameters, four arms each, 30M characters, learning rate per arm from the 10K screen (tied and c 0.04, ident and z 0.1; not re-screened at the other sizes, a limitation: larger models usually want lower rates, which can flatter the small sizes). Width d = 12 / 16 / 24 / 32 / 48 with the same F (2 couplings, 4 rules) and the rest of the shape fixed (two blocks, two heads, tsk-dim 8, projection ranks 8,6); feed-forward rule counts per arm chosen to equal the total (measured: 5K 5,019 / 5,019 / 5,005 / 5,005; 40K 40,064 / 40,064 / 39,969 / 39,969; 80K 80,007 / 80,007 / 80,156 / 80,156; 10K and 20K as above). Sampling run = seed 0 at 5K, 40K, 80K (the 10K and 20K full runs supply seed 0 for those sizes); full-sample results = seeds 0-4 at the sizes where the sampling run shows something worth the cost. Results and error bars are on the "Invertible meaning layer" artifact (`flm/bij_status.py` collects them).
Predictions (written after seeing one number, the 40K `tied` seed 0 at about 1.53, so not blind for that arm only): bits per character fall with size for every arm (95%); c's gap to ident shrinks from 10K to 80K (55%); at 80K c is within 0.02 of ident (45%); z is worse than c at every size (75%); the order of the arms at 10K (tied, ident, c, z) is the same at 80K (50%).

## Identity-gain arm and the full schedule (owner: "build the entire schedule and let's rip it", 2026-10-27)
New arm **ci** (`BIJ=ci` in `flm/run_bijective.py`): arm c with the output stage's lossy norm replaced by the exactly invertible gain G of `flm/invertible_gain.py` (the fork's layer): embedding G(F(C)), output u = F^-1(G^-1(h)), the lossy `self.norm` deleted; parameter count identical to c at every size (G has 4R + d parameters, the deleted norm had the same). The float32 round-trip recovery logged for ci covers F then G then G^-1 then F^-1.
Schedule (`flm/bij_schedule/schedule.py`, two parallel streams, cells skipped if present): phase 1, learning-rate sweep (0.01, 0.02, 0.04, 0.08, 0.16) at 10K, 40K, 100K for five arms (tied, ident, z, c, ci), seed 0, 30M characters; phase 2, 20K and 70K at each arm's best swept rate (geometric neighbour rule snapped to the grid; 70K takes 100K's); phase 3, seeds 1-2 at every size and arm at its chosen rate. About 130 runs, roughly 4.5 hours. The collector (`flm/bij_status.py`) uses, for every (size, arm), the rate with the lowest seed-0 validation and drops runs at other rates from the error-bar table; every rate is kept in the sweep panel of the artifact.
Predictions (registered before the sweep): the best rate falls as size grows for at least four of the five arms (60%); ci beats c at 100K (60%) and at 10K (50%); the best rate for ident and z at 10K is higher than for tied and c (as the 10K screen found; 65%); at 100K the order of the arms is not the one seen at lr 0.04 (tied worst of the four old arms) once each arm has its best rate (55%); no run diverges (90%).

## Paused for another GPU job (2026-10-27)
The sweep was paused after 58 sweep points (10K and 40K complete, 100K tied/ident partly done; no phase 2 or 3 yet) because a `kumo_tabular_eval` job from another project was holding most of the GPU (and the owner's priority is that job). To resume from `C:\personal\grad-school-bij\experiments\fuzzy-language-model` with the GPU free: `WORKERS=2 /c/personal/grad-school/.venv/Scripts/python outputs/bij10k/schedule.py > outputs/bij10k/schedule.out 2>&1 &` (skips finished cells; the script is in `flm/bij_schedule/schedule.py`, results under `outputs/bij10k/`, git-ignored). The artifact's half-hourly refresh job was removed; refresh with `python -m flm.bij_status outputs/bij10k/runs.json` and re-upload runs.json, curves.json, sweep.json.
Observed so far (seed 0, 30M characters; see the artifact): at 40K the routed-back arm with the invertible gain (1.520) ties the tied head (1.519) and beats the identity control (1.528); at 10K the identity control is best (1.882); at 100K the routed-back arm at lr 0.04 (1.383) already beats the tuned tied head (1.408, lr 0.01), with the other F arms' rates not yet swept.

## Results of the full schedule (2026-10-27; 78 runs at 30M characters, seeds 0-2 at each size and arm, each arm at its chosen lr; mean (sample std); bits per character)
| size | tied | ident | z (meaning-space) | c (routed back) | ci (routed back + invertible gain) |
|---|---|---|---|---|---|
| 10K | **1.882** (.003) | 1.891 (.010) | 1.951 (.004) | 1.916 (.003) | 1.903 (.010) |
| 20K | **1.673** (.003) | 1.706 (.007) | 1.746 (.028) | 1.709 (.015) | 1.693 (.008) |
| 40K | **1.522** (.005) | 1.571 (.048) | 1.549 (.011) | 1.551 (.024) | 1.540 (.019) |
| 70K | 1.449 (.003) | 1.473 (.060) | 1.474 (.019) | 1.482 (.051) | **1.444** (.005) |
| 100K | 1.409 (.001) | 1.510 (.030) | 1.453 (.066) | 1.419 (.052) | **1.396** (.003) |
(5K: tied 2.178, ident 2.189, one seed.) Chosen rates: tied 0.04 (0.01 from 70K), ident 0.05-0.1, z 0.04-0.05, c 0.04, ci 0.05 (0.01 from 70K). Collector rule: `chosen_lr.json` from the schedule decides which rate's seeds are counted.
Reading: (1) the invertible gain is a consistent improvement over the lossy norm on the routed-back arm at every size (ci beats c by 0.013-0.038 at 10K-70K, 0.023 at 100K), with much smaller seed spread at 70K-100K (std 0.005 and 0.003 against 0.051 and 0.052): the lossy norm made training erratic, the exact inverse did not. (2) ci is the best arm at 70K and 100K (1.444 and 1.396), ahead of the tuned tied head by 0.005 and 0.013 (larger than the seed spread at 100K, about equal at 70K); at 10K-40K the tied head is best, ci second or third, 0.02-0.03 behind at 10K-20K and 0.018 at 40K. (3) Against the identity control, which has the same distance head and no F, every F arm is better from 70K up (ci by 0.03 at 70K and 0.11 at 100K) and ci is better from 20K up: the invertible layer pays for its parameters once the model is big enough. (4) The meaning-space readout z is never best. (5) Caveats: three seeds; the chosen rates are the grid edges for several arms (tied and ci at 0.01 from 70K), so those results may improve at lower rates; one learning rate was not swept at 20K and 70K (neighbour rule); at 40K and above the identity control and c are erratic at some seeds. Predictions scored: the best rate falls with size for tied and ci (4 of 5 arms: right, 60%); ci beats c at 100K and at 10K (right on both, 60% / 50%); ident and z want higher rates than tied and c at 10K (right, 65%); the 100K order changes once each arm has its best rate (right, 55%); no run diverged (right, 90%).

## 200K parameters (2026-10-28; d 72, arms matched to 199.9-200.1K; lr sweep 0.0025 / 0.005 / 0.01 / 0.04 at seed 0, then seeds 1-2; 30M characters)
Mean (sample std), best rate 0.01 for every arm: ci **1.314** (.006), tied 1.329 (.001), ident 1.384 (.075), z 1.373 (.037), c 1.359 (.045). The invertible-gain arm beats the tied head by 0.015, about 2.5 times the gain arm's own seed spread and larger than the tied head's, and beats the identity control by 0.07 (with the control and the plain routed-back arm erratic across seeds again). The lower rates the owner asked about do not help: at seed 0 the validation at lr 0.0025 / 0.005 / 0.01 / 0.04 is, for ci, 1.472 / 1.367 / 1.308 / 1.330 and, for tied, 1.506 / 1.385 / 1.329 / 1.468, so 0.01 is bracketed on both sides for every arm. The trend over sizes now reads: the tied head wins up to 40K, the invertible-gain arm wins from 70K up with a margin that grows (0.005, 0.013, 0.015 at 70K, 100K, 200K).

## Stage 1: full-data runs at 100K parameters (owner, 2026-10-28: "a training run at 300M characters, then continue to 2.2B characters and see the comparison"; registered before the runs)
Arms ci (invertible gain, lr 0.01) and tied (lr 0.01), seed 0, d 56 (the 100K shapes), full TinyStories character stream (train.bin 2.19e9 characters, windows drawn at random so a 2.2B budget is about one epoch in expectation, not a strict single pass), batch 32, context 256, warm-up 1000, final-lr floor 0.02, checkpoint every 2000 steps (`--resume auto`), evaluation on the first 1M characters of valid. Part A: 300M characters (36,622 steps). Part B: warm start from part A's weights (`--init-from`) for 1.9B more characters (about 232K steps), fresh optimiser and schedule; the 300M and 2.2B numbers are then the two points of the comparison (the continuation is a second schedule, not one 2.2B-long cosine; a single schedule would be cleaner and is the alternative if the continuation looks odd). Files and logs under `outputs/full/` in the worktree (git-ignored).
Predictions: at 300M ci beats tied by at least 0.01 (55%); both models beat their 30M numbers by at least 0.05 (85%: 30M ci 1.399 / tied 1.407); at 2.2B both improve on their 300M result by at least 0.02 (70%) and ci's margin over tied at 2.2B is within 0.01 of its 300M margin (55%); ci stays ahead of tied at 2.2B (65%); neither run diverges (92%). Time: part A about 25 minutes for tied and 40 for ci running together; part B about 2.5 hours for tied and 4 for ci.

### Stage 1, part A result (2026-10-28; 300M characters, 36,622 steps, seed 0, lr 0.01, floor 0.02; `outputs/full/`)
ci (100,022 parameters) **1.2455** bits per character in 1,798 s (169K characters per second); tied (99,960) 1.2577 in 1,336 s (229K per second). Margin 0.0122 for ci. Both beat their 30M numbers (1.399, 1.407) by 0.15, far more than the 0.05 predicted. Prediction scoring: ci beats tied by at least 0.01 (55%): **right** (0.0122, just over); both improve by at least 0.05 (85%): **right**; no divergence (92%): **right**. Part B (1.9B more characters from these weights, `outputs/full/*_2200M/`) started 2026-10-28 16:20 with the same lr 0.01, warm-up 1000, floor 0.02.

### Stage 1, part B result (2026-10-28; +1.9B characters, 231,934 steps, warm start from part A, lr 0.01, floor 0.02; `outputs/full/*_2200M/`)
At 2.2B characters in total: ci **1.1962** (11,352 s for part B), tied **1.2065** (8,168 s). Gains over the 300M points: ci 0.049, tied 0.051. Margin of ci over tied 0.0103 (0.0122 at 300M). Float32 round-trip recovery through F and G stayed 1.0000 to the end (tau 1.27). One seed per cell, so the 0.01 margin is consistent across both budgets but not tightly measured.
Prediction scoring: both improve by at least 0.02 over 300M (70%): **right**; ci's margin within 0.01 of its 300M margin (55%): **right** (differs by 0.0019); ci stays ahead of tied at 2.2B (65%): **right**; neither run diverges (92%): **right**. Time: part B took 2.3 h for tied and 3.2 h for ci, under the 2.5 h and 4 h predicted.

### Pile, character level, 100K parameters, 300M characters (2026-10-28; `outputs/pile/`, seed 0, lr 0.01, same settings as stage 1)
ci **2.3560** bits per character (1,819 s), tied 2.3783 (1,323 s); margin 0.0223 (0.0122 on TinyStories at the same budget). Data: 332M characters of the deduplicated Pile sample folded to the 98 symbols (`flm/prep_pile_chars.py`). One seed.

## Explainability tests (owner, 2026-10-28; `flm/explain_bijective.py`; registered before running, on the Pile 300M checkpoints of ci and tied)
Questions: what each block's state says when read through the model's own output stage (the exact F^-1 for ci), how many rules each feed-forward system really uses, whether the winning rule says anything about the next character beyond the previous character, whether the rulebook is causal (silence the 8 most-used rules against 8 random ones), and what greedy continuations look like.
Predictions: (P1) for ci the embedding state decodes to the current character at least 99% of the time and next-character accuracy rises with depth, embedding < block 0 < block 1 (85%); (P2) in the last block the winning rule carries more information about the next character than the previous character alone does (40%); (P3) silencing the 8 most-used rules costs more than the worst of 10 random 8-rule draws in both blocks, for both models (70%); (P4) the effective number of rules per position (exp of firing entropy) is below 10% of the rules (60%); (P5) ci's greedy text repeats less (higher share of distinct 8-grams, averaged over the five prompts) than tied's (45%).

### Explainability results (2026-10-28; 256 validation windows of 256 characters; Pile 300M checkpoints, and the TinyStories 2.2B checkpoints for comparison; `outputs/explain/*.json`)
**Lens.** For ci the embedding state decodes to the current character 100% of the time (the exact F^-1 and G^-1), and reading the state after each block gives a calibrated next-character prediction: Pile accuracy 3.9% (embedding), 7.3% (block 0), 53.4% (block 1), bits per character 6.42, 5.77, 2.40; TinyStories 1.8%, 12.6%, 73.8% and 5.82, 5.44, 1.20. The tied head's early-layer reading is not calibrated (100 bits per character at the embedding on TinyStories; the state is not in the head's coordinates), so only the invertible model can be read this way at every depth. The state moves from "the current character" (still the current character at 55% of positions after block 0 on the Pile) to "the next character" (4.7% still the current one after block 1).
**Rules.** About half the rules are active at a position, not a few: the effective number of rules is 55 of 109 and 84 of 156 (ci, Pile); 52 of 109 and 80 of 156 (ci, TinyStories); tied is similar (74/130, 99/186). Every rule wins somewhere; the 8 most-used rules win 15-32% of positions. The winning rule alone says less about the next character than the previous character does (information 0.66 and 0.85 bits against 1.00 for ci on the Pile; 0.92 and 0.99 against 1.13 on TinyStories), so no single rule is a stand-alone explanation; they work as a distributed code. The readable ones are local character patterns, for example block 0 rule 52 wins after "t" (43%), "s" (19%) and "a" (12%) and the next character is "h" 31% of the time; block 1 rule 35 wins after a space 95% of the time and the next character is a word-initial "t", "a", "o" or "i".
**Causality.** Silencing the 8 most-used rules costs more than the worst of 10 random 8-rule draws in 7 of the 8 (model, block) cells on the Pile and in all 8 on TinyStories (ci Pile block 0: +0.45 against at most +0.36 bits; block 1: +0.18 against at most +0.14; the one miss is the tied head, Pile, block 0: +0.384 against +0.387).
**Greedy text.** Pile: both models fall into a loop ("the state of the ..."), distinct 8-gram share 0.04 for ci and 0.10 for tied; this is the greedy-decoding failure of a 100K model, not an explanation. TinyStories: coherent short stories with loops after the first sentence or two ("Lily. She was very happy and said, "Thank you, Tim! I want to play with your toys ..."); distinct 8-gram share 0.54 for ci and 0.76 for tied, so the invertible model repeats more under greedy decoding.
**Prediction scoring.** P1 (ci embedding reads the current character at least 99%, accuracy rises with depth, 85%): **right**. P2 (the last block's winning rule beats the previous character in information, 40%): **wrong**. P3 (the top-8 silencing beats every random draw in both blocks for both models, 70%): **wrong on the Pile by one cell** (tied block 0, by 0.003 bits), right on TinyStories. P4 (effective rules below 10% of the rules, 60%): **wrong** (about half). P5 (ci's greedy text repeats less than tied's, 45%): **wrong** on both corpora.
Reading: the invertible model is more readable in one specific way, the exact lens through F^-1, but the rules themselves are as diffuse as the tied head's, and a single rule is not an explanation. A sparser rule system (a sparsity penalty, or a winner-take-most firing) is the lever if rule-level explanations are the goal; it would be a new arm with its own cost in bits per character.

## Precision study (owner, 2026-10-28: "evaluate BF16/F16 training options for performance and quality"; `flm/bij_schedule/precision.py`, registered before running)
100K-parameter ci and tied (the stage-1 shapes), 30M characters of TinyStories, lr 0.01, warm-up 1000, floor 0.02, batch 32, context 256; `--precision` fp32 (control), tf32, bf16, fp16 (autocast; fp16 with the loss scaler); seeds 0 and 1; validation also re-scored in fp32 (`val_bpc_fp32`). Measured: validation, characters per second, peak memory, and (`flm/precision_roundtrip.py`) whether F then G then G^-1 then F^-1 still recovers all 98 codewords when run in each precision.
Predictions: (P1) bf16 is at least 1.15 times as fast as fp32 for tied (35%) and for ci (30%): a 100K model is mostly launch-bound; (P2) bf16 validation within 0.01 of fp32 for tied (75%) and for ci (55%); (P3) fp16 ends more than 0.02 worse than fp32 or fails (NaN or loss-scaler collapse) for ci (40%) and for tied (20%); (P4) tf32 within 0.005 of fp32 (85%) and at least 1.1 times as fast (40%); (P5) after bf16 training the float32 round trip still recovers 100% of the codewords (85%), and the round trip run in bf16 arithmetic recovers less than 99% (70%).

## A fixed, shared character dictionary (owner, 2026-10-28: "a fixed character fuzzy-embedding space across all models ... any model can be related by fixing this vector dictionary space"; evaluation, CPU only; `flm/embedding_space.py`, `flm/run_fixed_embedding.py`, output `outputs/explain/embspace_compare.txt`)
**What the trained dictionaries look like** (98 symbols, d 56, spaces compared up to rotation and scale; distance-matrix correlation "rsa", Spearman over the 4,753 pairs; a random matrix scores +0.05; all of these runs are seed 0, so they share an initial codebook, which flatters every comparison):
- Same run, more training (ci 300M vs 2.2B, warm start): rsa 0.90, Procrustes unexplained 0.19. Same corpus, different budget, separate run (ci 300M vs the 30M demo model): rsa 0.91. Tied head 300M vs 2.2B: rsa 0.94, linear CKA 0.98. So on one corpus the dictionary settles into a stable geometry.
- Same arm, other corpus (ci TinyStories 300M vs ci Pile 300M, the same initial codebook): rsa 0.57, Procrustes unexplained 0.31. Tied head: 0.40. So the geometry depends on the text, and a single dictionary is not what training finds for two corpora.
- Other arm, same corpus (ci vs tied, TinyStories): rsa 0.54-0.65, CKA 0.66-0.71. Pile: 0.79 (ci vs tied, rsa), the closest cross-arm pair.
- The meaning-space state G(F(C)) is less shared than the raw codebook: ci TinyStories vs ci Pile has CKA 0.00-0.03 and no common rotation (Procrustes unexplained 0.84-0.95), against 0.53 for the codebooks. A fixed C alone would not fix the space the blocks work in, because F is trained per model.
- Character classes (lower case, upper case, digits, punctuation, space): the ratio of between-class to within-class distance is 1.1-1.2 for the invertible codebooks and 1.6-1.9 for the tied head's TinyStories embedding, so the invertible codebook carries little class structure by itself; F carries it.
- A data-derived dictionary (positive PMI of characters within 2 positions, SVD to 56) correlates 0.5-0.7 with the learned spaces on the Pile and has no use on TinyStories: most printable symbols never occur there, so their rows are zero and the distances are degenerate (negative rsa is that artefact). Not a usable reference there.
**Options.** (A) Freeze only the codebook C at a common width N: the dictionary is 98 x N parameters (5,488 at N 56, 5.5% of a 100K model; free per model once shared). Relates models only through their inputs; the state space after F, and every block, is still model-specific. (B) Freeze C, F and G together (a universal invertible meaning layer trained once, then shared): every model at width N works in the same coordinates, so hidden states can be compared across models, blocks can be swapped between models (stitching), and the exact lens of the explainability section reads any model. This is the version that answers "any model can be related". The cost is whatever the frozen map loses against one trained with the model, and a fixed width N. (C) One dictionary for several widths: models of width d below N read the first d coordinates (needs a nested construction: a random orthogonal dictionary keeps prefix distances approximately, and F is not nested); this is a research item, not a switch.
**Recommendation.** Do (B) at N 56 first, because (A) alone does not relate the models, and test the cost with `flm/run_fixed_embedding.py` (written, not run; it freezes C from an `.npz`, the F and G freeze is the next small change): the learned dictionary (control), a frozen random Gaussian dictionary, a frozen dictionary taken from the other corpus's model, then F and G frozen from a trained model. Ci, 100K, 30M characters of TinyStories, seeds 0 and 1, lr 0.01 (10 runs, about an hour on the GPU once the precision study has finished; held back so the precision timings stay clean). Checkpoints with an independent seed (the precision study's seed 1) give the first honest cross-seed dictionary comparison without any training of its own.
Registered predictions (not yet run): frozen random Gaussian dictionary ends within 0.02 of the learned one (65%); a dictionary taken from the Pile model ends within 0.02 of the learned one on TinyStories (55%); freezing C, F and G from a trained TinyStories model ends within 0.03 of the learned one (45%) and, with F and G frozen from the Pile model, within 0.05 (40%); across seeds the learned codebooks' rsa with each other is below 0.9 (60%).
Environment note: importing `scipy` hangs in this environment, so the comparison uses a numpy rank correlation.

### Precision study results (2026-10-28; 16 runs, 30M characters, seeds 0 and 1, `outputs/precision/`)
Mean validation in bits per character / characters per second / peak memory:
| arm | fp32 | tf32 | bf16 | fp16 |
|---|---|---|---|---|
| tied | 1.4093 / 230.8K / 373 MB | 1.4103 / 252.6K / 373 MB | 1.4093 / 228.2K / 318 MB | 1.4092 / 222.6K / 318 MB |
| ci | 1.3933 / 166.9K / 430 MB | 1.3924 / 176.9K / 430 MB | 1.3944 / 160.7K / 367 MB | 1.3931 / 159.7K / 367 MB |
Quality is the same in every precision: the largest gap to fp32 is 0.0011 (ci, bf16), far below the seed spread (0.002 to 0.007) and the 0.015 gap between the arms; the ordering ci < tied holds in all four precisions and both seeds. fp16 trained without any loss-scaler trouble. Speed does not improve: bf16 and fp16 run at 0.96 to 0.99 times fp32 (a 100K model is launch-bound, not matrix-bound); tf32 is the only faster setting, 1.06 times (ci) and 1.09 times (tied). Low precision saves about 15% of peak memory (a few tens of MB here). The exact round trip survives: the nearest-codeword decode recovers all 98 codewords in float64, float32, bfloat16 and float16 arithmetic, for models trained in fp32, bf16 and fp16, although the raw reconstruction error is 0.2 in bfloat16 and 0.02 in float16 (against 3e-6 in float32), so the margin between codewords, not the layer, is what keeps it exact.
Prediction scoring: P1 (bf16 at least 1.15 times as fast, 35% tied, 30% ci): **wrong** (0.99, 0.96). P2 (bf16 within 0.01 of fp32, 75% tied, 55% ci): **right** (0.000, 0.0011). P3 (fp16 more than 0.02 worse or fails, 20% tied, 40% ci): **wrong** (it matches fp32). P4 (tf32 within 0.005: 85%): **right**; (at least 1.1 times as fast: 40%): **wrong**, just under (1.06, 1.09). P5 (float32 round trip exact after bf16 training, 85%): **right**; (round trip in bf16 arithmetic below 99%, 70%): **wrong** (still 100%).
Reading: at this size precision is a free choice for quality; use tf32 for the small speed gain, or fp32. bf16 and fp16 matter for memory and for larger models, not here. Whether bf16 pays off for the 14M and larger shapes is the earlier finding (1.2 times at 14M), so the break-even is between 100K and 14M parameters.

### Shared-dictionary test results (2026-10-28; `flm/bij_schedule/shared_dict.py`, `flm/run_fixed_embedding.py`; ci, 100K, TinyStories 30M characters, fp32, lr 0.01, seeds 0 and 1, `outputs/shared/`)
Mean validation (seeds 0, 1), difference from the learned-dictionary control (1.3933, the precision study's fp32 ci runs: same settings):
| dictionary | validation | difference |
|---|---|---|
| learned with the model (control) | 1.3933 | 0 |
| frozen random Gaussian C | 1.4383 | +0.045 |
| frozen C from the Pile model | 1.4076 | +0.014 |
| frozen C, F and G from the Pile model | 1.4363 | +0.043 |
| frozen C, F and G from the TinyStories 2.2B model | 1.4399 | +0.047 |
Reading. (1) A frozen dictionary that came from trained data (the Pile model's) costs only 0.014 bits (1%), even on a different corpus; a random one costs 0.045, so the dictionary geometry matters and carries over between corpora better than the cosine-style comparison (rsa 0.57) suggested. (2) Freezing F and G on top costs about 0.03 more than freezing C alone: a frozen universal meaning layer ends 0.043 to 0.047 worse than the learned model whether it came from the other corpus or the same one (the TinyStories-trained layer is not better on TinyStories than the Pile-trained one, so the cost is the mismatch with blocks trained against it, not the corpus). (3) Every frozen variant is worse than the tied head at the same settings (1.4093), so a frozen universal layer gives up the gain arm's advantage; the price of relating models this way is about 0.04 to 0.05 bits per character at 100K. (4) Independent seeds learn nearly the same dictionary: the codebooks of the precision study's fp32 seeds 0 and 1 have rank correlation 0.898 (CKA 0.61, Procrustes unexplained 0.28), against 0.05 for random matrices, so the shape is set by the data and training, not the initialisation.
Prediction scoring: random C within 0.02 of learned (65%): **wrong** (0.045). Pile C within 0.02 (55%): **right** (0.014). Frozen C, F, G from the TinyStories model within 0.03 (45%): **wrong** (0.047). Same from the Pile model within 0.05 (40%): **right** (0.043). Cross-seed codebook rsa below 0.9 (60%): **right**, by 0.002 (0.898).
Next steps if the owner wants a shared space: freeze C only (cheapest: 1% cost, and models then share their input coordinates), or train F and G once on a pooled corpus together with a small model and then freeze, rather than freezing them from a model trained with its own blocks; the stitching test (swap blocks between two models that share a frozen layer) is the real measure of relatability and has not been run.

**Decision (owner, 2026-10-28):** freeze only the codebook C as the shared dictionary; F and G stay trained per model.

## A canonical dictionary by construction (owner, 2026-10-28: "build a canonical dictionary by construction instead ... the goal is clarity and similarity ... fix the width, make it a power of 2"; `flm/canonical_dictionary.py`, `flm/bij_schedule/canon.py`; registered before running)
The dictionary is 98 x 64, built without training: coordinates for the character class, the letter (shared by the two cases), the digit value as an ordered thermometer, vowels, punctuation and symbol groups, a 6-bit symbol code and a kind flag (layout in the file's docstring). Readable by construction: a is nearest to A (0.71), then to the other vowels (1.41), then the other letters (1.5); 3 is nearest to 2 and 4 (0.5) and 3 to 9 is 1.22; . is nearest to ! : ? ; every character is distinct. Width 64: the feed-forward rule counts are refitted to keep the total at 100K (R 125, R0 88: about 100,010 parameters against 100,022 at width 56).
Test: ci, 30M characters of TinyStories, fp32, lr 0.01, seeds 0 and 1, width 64: learned dictionary (control), constructed dictionary frozen, constructed dictionary as the initial value only (it then trains), random Gaussian dictionary frozen.
Predictions: constructed frozen ends within 0.02 of the learned control (50%); constructed frozen beats random frozen by at least 0.015 (70%); constructed as initial value ends within 0.01 of the learned control (75%); the learned control at width 64 is within 0.005 of the width-56 control, 1.3933 (60%).

### Canonical dictionary results (2026-10-28; ci, width 64 (100,202 parameters), TinyStories 30M characters, fp32, lr 0.01, seeds 0 and 1, `outputs/canon/`)
| dictionary | validation (seeds 0, 1) | difference from learned |
|---|---|---|
| learned with the model (control) | 1.3978 (1.3998, 1.3958) | 0 |
| constructed, frozen | 1.4149 (1.4132, 1.4167) | +0.017 |
| constructed, as initial value (then trains) | **1.3844** (1.3831, 1.3858) | **-0.013** |
| random Gaussian, frozen | 1.4417 (1.4409, 1.4424) | +0.044 |
The constructed dictionary frozen costs 0.017 bits per character against a learned one (1.2%), 0.027 less than a frozen random dictionary (0.044), and the float32 round trip stays exact (1.0000). Used as the starting point instead of frozen, it is the best arm: 0.013 better than a learned-from-random start, more than three times the seed spread (0.002 to 0.004), and better than the tied head (1.4093 at width 56). So structure in the dictionary helps even when it is allowed to move away: the gain is not from fixing the vectors but from where they begin.
Prediction scoring: constructed frozen within 0.02 of learned (50%): **right** (0.017). Constructed frozen beats random frozen by at least 0.015 (70%): **right** (0.027). Constructed as initial value within 0.01 of learned (75%): **wrong**, it is better by 0.013. Width-64 control within 0.005 of the width-56 control (60%): **right** (0.0045).
Reading: a shared canonical dictionary by construction costs about 1.2% frozen (comparable with the 1% of a frozen trained dictionary, and fully readable), and the same construction is a better initialisation than random. If models must share the dictionary exactly, freeze it; if closeness is enough, initialise from it and let it train, then compare the dictionaries afterwards. One seed pair at 30M characters; the initial-value advantage needs a longer run and a third seed before it is trusted.

## Constructed dictionary: relative separations and width (registered 2026-10-28, before the runs)

Owner's questions: do different relative separations (especially digits that map mathematically) help, and how few dimensions can the constructed dictionary use?

Context: the 300M-character confirmation (`flm/bij_schedule/canon300.py`, learned vs constructed initial value, seeds 0-2) is still running.  Its predictions, written after seeing seed 0 at step 26,000 (1.2847 constructed against 1.2938 learned): the constructed start wins on every seed; margin 0.006-0.012 bits; the trained dictionary stays close to the constructed one (pairwise-distance correlation above 0.8).  Only seeds 1 and 2 are a clean test.

Design (`flm/bij_schedule/make_variants.py`, `flm/bij_schedule/width.py`; ci, about 100K parameters, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1; dictionary is the initial value and trains, the best arm from the last test).  The codebook is standardised to std 0.5 on load, so only relative separations matter.
- Separations at width 64: `dlin64` (digit = one linear coordinate, step 0.25, so distance = 0.25 x gap; the thermometer gives 0.5 x sqrt(gap)), `dlin64b` (step 0.5), `kind64` (letter identity weight 2 instead of 1), `soft64` (class, flag, code and digit weights 0.25 instead of 0.5).  Control: `canon64` (seed mean 1.3844 at width 64 from the earlier run).
- Width: the constructed 64-dimensional dictionary projected to k = 56, 32, 16, 8 (top 3k/4 principal components plus a fixed random projection of the residual, which keeps every symbol distinct; pairwise-distance correlation with the original 1.000, 0.975, 0.858, 0.774 for 56, 32, 16, 8).  Each width is run with the projection as the initial value and with a learned dictionary (control), at the shape that gives about 100K parameters: rules/per-system 156/109 (56), 328/230 (32), 706/494 (16), 1443/1010 (8).

Predictions:
1. Digit coding makes no measurable difference to total bits per character (within 0.005 of `canon64`) because digits are a tiny share of TinyStories; a difference would show only at digit positions, which this run does not measure (a follow-up on the Pile, which has more digits).
2. `kind64` and `soft64` land within 0.01 of `canon64`.
3. Width 56 is within 0.01 of width 64 for both arms.
4. Width 32 is within 0.015 of width 64 (projected start).
5. Width 16 costs 0.02-0.06 bits against width 64, and the constructed start stays ahead of learned.
6. Width 8 costs more than 0.06 bits.
7. The constructed start beats learned at every width (the margin is not a width-64 accident).

Scope change (owner, 2026-10-28): drop the digit coding and the other separation variants; run only the width test.  Predictions 1 and 2 are withdrawn; 3 to 7 stand.  The job list is now 16 runs (widths 56, 32, 16, 8; projected start and learned; seeds 0 and 1).

## 300M confirmation results (2026-10-28)

ci, width 64, 100,202 parameters, 300M TinyStories characters, fp32, lr 0.01, seeds 0-2.  Learned 1.2544 / 1.2577 / 1.2602; constructed as initial value 1.2484 / 1.2513 / 1.2537; gain 0.0060 / 0.0064 / 0.0065 (mean 0.0063, 0.5%).  Prediction "wins on every seed, margin 0.006-0.012": right, at the low edge (the 30M margin was 0.013, so it shrinks with more training).

Drift of the trained codebook from the constructed one (correlation of pairwise distances / share of each symbol's 5 nearest neighbours kept): constructed start 0.57, 0.53, 0.56 / 0.42, 0.38, 0.41; learned-from-random 0.29, 0.31, 0.34 / 0.28, 0.23, 0.25.  Prediction "above 0.8": wrong.  Training moves the dictionary a long way from the constructed geometry, though it stays about twice as close as an unrelated start.  The explainability value therefore needs the dictionary frozen (cost about 1.2%) or a drift penalty; the 0.5% gain from the start alone does not keep the geometry.

## Width test results (2026-10-28)

ci, about 100K parameters at each width, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1, mean validation bits per character.  Dictionary: the constructed 64-wide one projected to the width and used as the initial value, against a learned one.  Rules per system rise as the width falls (124, 156, 328, 706, 1443 for 64, 56, 32, 16, 8).

| width | learned | projected start | start vs width 64 (1.3844) |
|---|---|---|---|
| 64 (earlier run) | 1.3978 | 1.3844 | 0 |
| 56 | 1.3934 | 1.3905 | +0.0060 |
| 32 | 1.3700 | 1.3699 | -0.0146 |
| 16 | 1.4251 | 1.4233 | +0.0389 |
| 8 | 1.6297 | 1.6340 | +0.2495 |

Scoring (3 to 7; 1 and 2 withdrawn): 3 right (56 within 0.01).  4 right, but only just and in the other direction from the worry: 32 is 0.0146 better, not worse.  5 right on the cost (0.039, inside 0.02-0.06); the constructed start is ahead of learned by only 0.002, inside the seed spread (about 0.01 at this width), so "stays ahead" is not established.  6 right (0.25, a cliff between 16 and 8).  7 wrong: the constructed start beats learned by 0.003 at 56, ties at 32, is 0.002 ahead at 16 and 0.004 behind at 8; none of these gaps is outside the seed spread, so the width-64 advantage does not carry to the projected dictionaries.

Reading: width 32 is the best width tested, but it also has 2.6 times the rules of width 64, so the gain may come from the rule count rather than the dictionary width; the two are not separated here.  The projected dictionary has unnamed principal-component coordinates, so it is not the clear dictionary the owner wants; the next step is a hand-built 32-wide dictionary with named coordinates (owner's preference: more rules, fewer dictionary entries).

## Rules against width grid (registered 2026-10-28, before the runs)

Owner's request: test rule count as well as dictionary width, and add width 24.  So far width and rules moved together to hold 100K parameters, so the width-32 win (1.3700 against 1.3978 at width 64) may be a rule-count effect.

Design (`flm/bij_schedule/grid.py`; ci, learned dictionary, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1): widths 64, 32, 24, 16 crossed with 124, 328 and 706 rules per system (the per-block second system keeps the ratio 0.7), plus width 24 at its own 100K shape (456 rules).  Parameters: width 64: 99.6K / 207K / 406K; 32: 46K / 99.9K / 200K; 24: 33.9K / 74.4K / 149K (456 rules: 99.8K); 16: 22.3K / 49.5K / 99.6K.  Reused from earlier runs (learned dictionary, same settings): (64, 124), (32, 328), (16, 706); the (64, 124) cell was 125 rules in the earlier run.

Predictions:
1. At every width more rules gives lower bits per character, with smaller gains from 328 to 706 than from 124 to 328.
2. At 706 rules the four widths are within 0.03 bits of each other (rule count, not width, explains the width-32 result).
3. Width 24 at 456 rules (99.8K parameters) is within 0.02 of width 32 at 328 (1.3700).
4. At about 100K parameters (the diagonal: 64/124, 32/328, 24/456, 16/706) the best width is 32 or 24, not 64.
5. At 124 rules, width 64 beats width 32 (the extra parameters are in the dictionary and gain), by more than 0.02.

## Rules against width grid results (2026-10-28)

ci, learned dictionary, 30M TinyStories characters, fp32, lr 0.01, mean of seeds 0 and 1, validation bits per character (parameters in brackets).  Seed spread is about 0.002-0.02.

| width | 124 rules | 328 rules | 706 rules |
|---|---|---|---|
| 64 | 1.3978 (99.6K) | 1.3028 (207K) | 1.2511 (406K) |
| 32 | 1.5242 (46K) | 1.3700 (99.9K) | 1.3062 (200K) |
| 24 | 1.5723 (33.9K) | 1.4259 (74.4K) | 1.3386 (149K) |
| 16 | 1.6932 (22.3K) | 1.5274 (49.5K) | 1.4251 (99.6K) |
| 24, 456 rules | 1.3815 (99.8K) | | |

Scoring: 1 right (more rules always lower; 124 to 328 gains 0.095-0.166, 328 to 706 gains 0.052-0.102, so diminishing).  2 wrong: at 706 rules the widths are 1.2511, 1.3062, 1.3386, 1.4251, a spread of 0.17, not 0.03 (but parameters differ from 406K to 100K).  3 right (width 24 at 456 rules is 0.012 worse than width 32 at 328).  4 right (at about 100K: 64/124 1.3978, 32/328 1.3700, 24/456 1.3815, 16/706 1.4251; best is 32).  5 right (64 beats 32 at 124 rules by 0.126, but with 99.6K against 46K parameters).

Reading: both levers matter and parameters are the common currency.  Rules are the strongest lever; wider is better at fixed rules, but mostly by adding parameters.  At matched parameters width 32 is the efficient shape at 100K (0.028 better than width 64); at about 200K width 32 with 706 rules (1.3062, 200K) and width 64 with 328 rules (1.3028, 207K) are equal within the seed spread; at about 50K width 32 with 124 rules (1.5242, 46K) and width 16 with 328 (1.5274, 49.5K) are equal.  So a narrow dictionary with many rules costs nothing at equal size, which supports the owner's preference, and the width-32 win at 100K is a rule-count effect plus a modest width-efficiency edge.  Width 16 loses at 100K (1.4251).  Larger models (width 32, 706 rules, 200K) improve a lot (1.3062); size, not width, is the remaining lever.

## Hand-built 32-wide dictionary (registered 2026-10-28, before the runs)

Owner's preference: more rules, fewer dictionary entries, for clarity.  `flm/canonical_dictionary32.py` builds a 98 x 32 dictionary with every coordinate named (kind, unknown, newline, upper case, vowel, 5-bit letter code, linear digit value, mark group flags, 6-bit mark code at half weight, four alphabet bands).  Nearest neighbours read sensibly (a: A e E b c; 3: 2 4 1 5 0; ".": ! ? & @ $; "(": [ < { $ &).

Design (`flm/bij_schedule/hand32.py`; ci, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1): width 32 at 328 rules (99.9K parameters) and 706 rules (200K), the dictionary frozen (`hand32Frozen`) and as the initial value (`hand32Init`).  Controls from the grid, same shapes: learned 1.3700 (328) and 1.3062 (706); projected start 1.3699 (328).

Predictions:
1. Frozen costs 0.01-0.04 bits against learned at 328 rules (the width-64 constructed dictionary cost 0.017 frozen).
2. As the initial value it is within 0.01 of learned at both sizes (the projected start was; the width-64 advantage did not carry).
3. The frozen cost does not grow at 706 rules (within 0.01 of the 328 cost).
4. The trained init arm drifts as far as before (distance correlation with the constructed dictionary 0.4-0.7).

First hand32 results (2026-10-28, partial): the trainable arm matches learned (328 rules: 1.3758 and 1.3680, mean 1.3719 against 1.3700), but the frozen arm is far worse than predicted (328 rules: 1.6342 and 1.6235, about +0.26 against 0.01-0.04 predicted; prediction 1 wrong).  The run is sound (same 99,939 parameters, float32 round trip 1.0000).  Hypothesis: the hand-built dictionary separates the most frequent symbols too little (letters are a 5-bit code at weight 0.5, so two letters can differ by one bit; after standardisation the closest pair is 0.55 apart, against 1.39 for the 64-wide dictionary), and a frozen dictionary cannot fix that while a trainable one can.
Test (`hand32L`, letter code weight 1.0, `flm/bij_schedule/make_hand32l.py`): frozen, 328 rules, seeds 0 and 1.  Prediction: the frozen cost falls below 0.05 bits against learned (1.3700), which would confirm that letter separation is the cause; if it stays above 0.1 the cause is something else (the dictionary's low rank or the readout) and I will not guess further without a diagnostic.

## Hand-built 32-wide dictionary results (2026-10-28)

ci, width 32, 30M TinyStories characters, fp32, lr 0.01, mean of seeds 0 and 1, validation bits per character.

| rules (parameters) | learned | hand-built, frozen | hand-built, as start | letter weight 1.0, frozen |
|---|---|---|---|---|
| 328 (99.9K) | 1.3700 | 1.6289 (+0.259) | 1.3719 (+0.002) | 1.6250 (+0.255) |
| 706 (200K) | 1.3062 | 1.5300 (+0.224) | 1.2998 (-0.006) | not run |

Scoring: 1 wrong (frozen cost +0.26, not 0.01-0.04).  2 right (as a start it is within 0.01 of learned at both sizes).  3 wrong as written (the frozen cost 0.259 to 0.224 differs by 0.035, not within 0.01; it does not grow).  4 not yet measured (drift of the trained start arms).  The letter-weight test refutes the hypothesis: doubling the letter code weight left the frozen cost at +0.255, above the 0.1 line I set, so letter separation is not the cause and I will not guess further without a diagnostic.
Note that the frozen cost with this dictionary (about 0.23-0.26) is an order of magnitude above the 64-wide constructed dictionary frozen (+0.017) and the 64-wide trained codebook frozen (+0.014); something in the hand-built layout, not frozenness itself, hurts.
Next: a diagnostic on the saved checkpoints (bits per character by symbol class, frozen against trained start) to see where the loss is, before changing the dictionary again.

## Diagnostic: where the frozen hand-built dictionary loses (2026-10-28)

`flm/diagnose_dictionary.py`, 512 validation windows of 256 characters, checkpoints at width 32, 328 rules, seed 0 (`outputs/hand32/diagnose.json`).  Bits per character by the class of the next character: learned 1.3720; hand-built as start 1.3742; hand-built frozen 1.6326 (letter weight doubled, frozen: 1.6154).  Frozen against learned: lower-case letters (73% of positions) 1.493 to 1.792 bits, adding +0.218 of the +0.26; upper-case +0.027; marks +0.011; space, newline and special about +0.003 together.  Top-1 accuracy falls only from 0.677 to 0.653 on lower-case letters, so the frozen model still picks the right letter about as often; the loss is in how the probability is spread over letters.
Hypothesis (fits the failed letter-weight test, which scaled distances but kept the structure): the 5-bit letter code makes some letters look similar for no reason (a and b differ in one bit), the likelihood is a softmax over distances, and a frozen dictionary cannot undo the false similarity while a trained one can.  The 64-wide constructed dictionary, with one flag per letter (all letters equally far apart), cost only +0.017 frozen.
Test (`canon64H`: the 64-wide constructed dictionary with letters as a 5-bit code in dims 8-12, nothing else changed; frozen, 124 rules, seeds 0 and 1; controls: the frozen one-flag version 1.4149, learned 1.3978): prediction: the frozen cost jumps above 0.1 bits against learned.  If it stays below 0.05 the hypothesis is refuted and the cause is elsewhere in the layout (the other features, or the width).

Letter-code test result (2026-10-28): `canon64H` frozen, 124 rules, seeds 0 and 1: 1.9263 and 1.9215 (mean 1.924), against 1.4149 for the one-flag-per-letter dictionary frozen and 1.3978 learned.  Cost +0.51 with only the letter coding changed; prediction (above 0.1) right.  The cause of the hand-built dictionary's frozen cost is the false similarity between letters, not the width or the other features.  A frozen dictionary needs letters that are all equally far apart.

## Minimum dimension with equal-distance letters, three options (registered 2026-10-28, before the runs)

Owner: try all three options and find the minimum dimension space.
1. Wider frozen dictionary with a flag per letter: `oh48` (48 named dimensions; 196/137 rules, 100.1K parameters).
2. Narrow and trainable: the dictionary as an initial value that trains (`Init` arms); the learned-dictionary ladder from the grid already answers the narrow-trainable question (100K: width 32 best 1.3700, 24 1.3815, 16 1.4251, 8 1.6297 with the larger rule counts).
3. Equal-distance letters squeezed into the least room: `squeeze32` (26 letter flags, lower case 1.0 and upper case 0.6, plus a 6-bit code for the 47 non-letters; 328/230 rules, 99.9K parameters).  26 letters plus a 6-bit code is the floor for one flag per letter.
Builder: `flm/canonical_dictionary_oh.py`; driver `flm/bij_schedule/ohsq.py` (ci, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1; each dictionary frozen and as the initial value; new control: learned dictionary at width 48, 196 rules).

Predictions:
1. `oh48` frozen costs at most 0.04 bits against the learned width-48 control (the width-64 constructed dictionary cost 0.017).
2. `squeeze32` frozen costs more than `oh48` frozen (its non-letter code puts false similarity between space, digits and marks, 27% of positions), but less than 0.1 against learned width 32 (1.3700), and at least 0.15 less than the hand-built 5-bit-letter dictionary (1.6289).
3. As initial values both are within 0.01 of their learned controls.

## Equal-distance letters, results (2026-10-28)

ci, about 100K parameters, 30M TinyStories characters, fp32, lr 0.01, mean of seeds 0 and 1, validation bits per character.

| shape | learned | frozen | as initial value |
|---|---|---|---|
| width 48, 196/137 rules (`oh48`, a flag per letter, every coordinate named) | 1.3800 | 1.3849 (+0.005) | 1.3628 (-0.017) |
| width 32, 328/230 rules (`squeeze32`, 26 letter flags + 6-bit non-letter code) | 1.3700 | 1.5659 (+0.196) | 1.3739 (+0.004) |

Scoring: 1 right (oh48 frozen +0.005, inside 0.04).  2 mostly wrong: squeeze32 frozen does cost more than oh48 (right), but +0.196 is above 0.1 and only 0.063 better than the 5-bit-letter dictionary (1.6289), not 0.15.  3 half right: squeeze32 as a start is +0.004 (within 0.01), but oh48 as a start is 0.017 better than learned, outside the 0.01 band (the same direction as the width-64 constructed start, -0.013).

Reading: a frozen, fully named dictionary of 48 dimensions costs about nothing (+0.005, inside the seed spread) and as a starting value beats learned.  The 32-wide squeeze costs +0.196 frozen: with letters fixed as equal-distance flags, the loss moved to the 6-bit non-letter code, plausibly the same false-similarity mechanism acting on space, digits and marks (space is 19% of positions); that mechanism is not tested here.  The minimum for a frozen readable dictionary is therefore somewhere between 32 and 48 and depends on giving the non-letters equal-distance structure too; trainable dictionaries go much lower (32 at no cost, 24 +0.012, 16 +0.055; the learned ladder).
Open: a 40-wide dictionary with a flag per letter and equal-distance flags for the frequent non-letters (space, newline, the common marks) would locate the frozen minimum; the drift of the trained oh48 start has not been measured.

## oh48 at 300M characters (registered 2026-10-28, before the runs)

Confirmation at scale of the best named dictionary: `oh48` (width 48, 196/137 rules, 100.1K parameters), learned against frozen against as initial value, 300M TinyStories characters, fp32, lr 0.01, seeds 0-2 (`flm/bij_schedule/oh48_300.py`, `outputs/oh48_300/`).  Controls from the 30M test: frozen +0.005, as start -0.017 against learned.
Predictions: 1. frozen stays within 0.02 of learned on every seed (a frozen dictionary's cost fell with more data at width 56, +0.014 shared, so it should not grow).  2. As the initial value it beats learned on every seed, by 0.004-0.012 (the width-64 gain halved from 30M to 300M: 0.013 to 0.006).  3. Frozen is not better than learned on any seed.

## GPU queue after oh48 at 300M (registered 2026-10-28, before the runs)

`flm/bij_schedule/queue2.py` runs three studies back to back, two jobs at a time.
1. **Frozen minimum, `oh40`** (40 dimensions: a flag per letter, equal-distance flags for space and newline, a 6-bit code only for the marks; 250/175 rules, 100.1K parameters; 30M TinyStories characters, seeds 0 and 1): learned width-40 control, frozen, as initial value.  Prediction: frozen costs at most 0.03 against the control (squeeze32 cost +0.196 because space and digits shared the binary code; here only the marks, 4% of text, do), so the frozen floor is 40 or below.  If the cost is above 0.1, equal-distance structure is needed for the marks too and the floor is nearer 48.
2. **Drift penalty on `oh48`** (30M, seeds 0 and 1; `FIXED_PULL=k` in `flm/run_fixed_embedding.py`, FIXED_TRAIN=1: after each optimiser step the dictionary moves the fraction k back toward its constructed value; k = 0.001 and 0.01).  Controls from the earlier run: frozen 1.3849, learned 1.3800, as initial value 1.3628.  Predictions: both penalised arms are within 0.01 of the unpenalised initial-value arm or better than the frozen arm (1.3849); the stronger pull keeps the dictionary closer to the constructed one (distance correlation above 0.9) than the weaker (0.7-0.9); an unpenalised trained start would be about 0.55.
3. **`oh48` on the Pile** (300M characters, seed 0): learned, frozen, as initial value (`outputs/oh48_pile/`).  Prediction: frozen within 0.03 of learned (harder text, more symbols: digits, code, marks), as initial value within 0.02 of learned in either direction.

## Goal: the full invertible fuzzy model at 100K parameters on a larger corpus (owner, 2026-10-28)

"Build out the full invertible fuzzy model at the 100K parameter space, training on a larger dataset if possible."

Definition of the full model: ci (exactly invertible coupling embedding F that is also the de-embedding, invertible gain, every layer an order-1 TSK system, greedy output) with the 48-wide named dictionary `oh48` (frozen, or as the initial value), 196/137 rules per system, 100.1K parameters (width 56 learned is the earlier 100K shape; the width ladder showed width 48 sits between 56 and 32 at equal size).
Corpus: Cosmopedia v2 (synthetic textbooks, stories and web-style explainers; `data/cosmopedia/tinystories`, 3.0B characters of the same 98-symbol alphabet, 5.0M validation).  The budget stays at 300M characters per run (the owner's rule: no 1B+ runs), so a run sees a tenth of the corpus and no character repeats: more diverse text without a bigger budget.  Pile (332M characters) is the second corpus used for the cross-corpus evaluation.
Runs (`flm/bij_schedule/final100k.py`, `outputs/final100k/`, fp32, lr 0.01, 300M characters): learned dictionary, `oh48` frozen, `oh48` as initial value, seeds 0 and 1; plus the tied head at its 100K shape (width 56, 186/130 rules), seed 0, as the control for the architecture.  Then, on the chosen model: validation bits per character on Cosmopedia, TinyStories and the Pile; the explainability suite (`flm/explain_bijective.py`); the exact round trip (`flm/verify_invertible.py`); greedy continuations of fixed prompts as a blind coherence set.
Predictions:
1. Learned `oh48` reaches 1.6-2.2 bits per character on Cosmopedia (between TinyStories 1.25 and the Pile 2.36: textbook prose is more regular than the Pile and richer than children's stories).
2. Frozen costs at most 0.03 against learned; as initial value it is better than learned by 0.003-0.02 (the width-64 gain was 0.0063 at 300M).
3. ci beats the tied head on Cosmopedia by 0.01-0.03 (0.012 on TinyStories, 0.022 on the Pile at 300M).
4. The Cosmopedia-trained model scores worse on the Pile than the Pile-trained model by more than 0.3 bits, and worse on TinyStories than the TinyStories-trained model by more than 0.2.

## oh48 at 300M characters, results (2026-10-28)

ci, width 48, 196/137 rules, 100,114 parameters, 300M TinyStories characters, fp32, lr 0.01, seeds 0-2, validation bits per character.

| arm | seed 0 | seed 1 | seed 2 | mean | against learned |
|---|---|---|---|---|---|
| learned | 1.2455 | 1.2431 | 1.2418 | 1.2435 | 0 |
| frozen (`oh48`) | 1.2465 | 1.2505 | 1.2506 | 1.2492 | +0.0057 |
| as initial value | 1.2341 | 1.2332 | 1.2393 | 1.2355 | -0.0080 |

Scoring: 1 right (frozen within 0.02 of learned on every seed: +0.0010, +0.0074, +0.0088).  2 mostly right: the initial value beats learned on every seed (-0.0114, -0.0099, -0.0025), but the seed-2 gain (0.0025) is under the 0.004 floor of the registered band.  3 right (frozen is worse than learned on every seed).
Reading: at scale a fully named, frozen 48-wide dictionary costs 0.0057 bits (0.5%), inside what the seed spread can hide only on seed 0; as a starting value it gains 0.008 on average (0.6%), with a seed spread of 0.009 in the gain, so the starting-value advantage is real in direction on three seeds but small and variable in size.  For the explainability goal the frozen dictionary is the cheaper-to-defend choice: half a percent for a space that every coordinate of which means something.

## oh40 results (2026-10-28)

ci, width 40, 250/175 rules, 100,134 parameters, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1, validation bits per character: learned 1.3676 / 1.3803 (mean 1.3740); frozen 1.4119 / 1.4258 (mean 1.4189, +0.0449); as initial value 1.3583 / 1.3666 (mean 1.3625, -0.0115).
Scoring: prediction 1 wrong (frozen +0.045, not at most 0.03), but well under the 0.1 line, so the frozen floor is not forced up to 48.  Compared with `oh48` frozen (+0.005, 30M), dropping the named mark-group flags and keeping only a 6-bit mark code costs about 0.04 bits; the learned-dictionary control at width 40 is as expected (between 32 at 1.3700 and 48 at 1.3800, within seed spread).  As an initial value `oh40` again beats learned (-0.0115, the same size as `oh48` at 30M: -0.017).
Reading: the frozen floor is between 40 and 48 dimensions at a cost of 0.005 (48) to 0.045 (40) bits; below that the binary mark code is what the frozen model pays for, since letters and space are now equal-distance flags.  The named-dictionary advantage as a starting value holds at 40, 48 and 64.

## Drift penalty on oh48, results (2026-10-28)

ci, width 48, 196/137 rules, 30M TinyStories characters, fp32, lr 0.01, seeds 0 and 1, `FIXED_TRAIN=1` with `FIXED_PULL=k` (after every optimiser step the dictionary moves the fraction k of the way back to its constructed value).  Validation bits per character, with the dictionary's drift measured on the final checkpoints (correlation of pairwise distances with the constructed dictionary; relative move = norm of the change over norm of the dictionary):

| arm | seed 0 | seed 1 | mean | against learned (1.3800) | distance correlation | relative move |
|---|---|---|---|---|---|---|
| learned | 1.3837 | 1.3762 | 1.3800 | 0 | - | - |
| frozen | 1.3889 | 1.3809 | 1.3849 | +0.0049 | 1.000 | 0 |
| initial value, no penalty | 1.3665 | 1.3590 | 1.3628 | -0.0172 | 0.496 | 1.28 |
| pull 0.001 | 1.3773 | 1.3720 | 1.3747 | -0.0054 | **0.957** | 0.36 |
| pull 0.01 | 1.3848 | 1.3787 | 1.3818 | +0.0018 | 1.000 | 0.03 |

Scoring: both penalised arms are better than the frozen arm (right, by 0.0103 and 0.0032); the stronger pull keeps the distance correlation above 0.9 (right, 1.000); the weaker pull's correlation was predicted 0.7-0.9 and came out 0.957 (wrong, it keeps more than expected); an unpenalised start about 0.55 (right in range, 0.496).
Reading: a weak pull (0.001) keeps the dictionary's geometry almost intact (distance correlation 0.957, against 0.496 without a penalty) and still gains 0.0054 over learned, about a third of the unpenalised start's 0.0172 gain and 0.0103 better than frozen.  So "fix the dictionary, but maybe revise it" works: the dictionary can adapt by a third of its own size and stay recognisable.  The 5-nearest-neighbour overlap is not a useful drift measure for these dictionaries (0.66 even at correlation 1.000) because a flag-per-letter layout has many exactly equal distances and any tiny move reorders the ties.
