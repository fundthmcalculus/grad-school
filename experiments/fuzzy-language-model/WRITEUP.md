# Tiny fuzzy language models: linear vs quadratic attention, and a recurrent TSK language model

**Status:** CPU phase stopped 2026-10-10; GPU phase summarised in §10–§13 (§10–§12 transcribed from the GPU session's records; §13 is the final state, 2026-10-12) ·
**Experiment:** `experiments/fuzzy-language-model/` · **PR:** #272 · **Dashboard:**
https://claude.ai/artifact/XGJzPWz5T5i2wQvVE6C3Y4 (private until shared)

§1–§9 summarise 365 completed CPU training runs (≈235 CPU-hours on a 4-core laptop, 2026-10-08 → 10)
and the d = 16 n-gram screen (§5.6). §10–§12 summarise the GPU session's goal-track and subword work,
**transcribed by the CPU session from `RESULTS.md` and not re-run; the GPU session should check every
number** (the open fairness gaps are in §12). The laboratory record behind it is [`RESULTS.md`](RESULTS.md): every
hypothesis as registered before its run, every protocol amendment, every correction. Where
this document and `RESULTS.md` differ, `RESULTS.md` is the record.

> **Citability.** AGENTS.md non-negotiable 2 makes ten seeds the protocol for quotable numbers.
> The subword comparison of §11 (`v6-final10`: fuzzy D against two tuned neural references, ten
> seeds each, 30M characters) has them and may be quoted with its caveats: one tokenizer, one
> size, one budget; the references were chosen at 10M characters on seed 0; D runs about 1.5
> times slower per character. Every other cell here is 3 seeds or fewer and **may not be quoted
> in the proposal** until repeated at ten seeds (the character-level `v4-final10` grid has not run).

## 1. Summary

1. **Quadratic attention buys nothing per parameter at this scale.** On character-level
   TinyStories, softmax attention needs about as many parameters as linear-cost mixers to
   reach a given quality: 26.7K vs 26.0K (linear) and 23.7K (DeltaNet) for 1.6 bits/char.
   The linear-cost mixers are slightly ahead below ~20K parameters. Softmax's KV cache also
   outgrows the whole model after 91 characters of context at width 16.
2. **A language model whose every sublayer is a TSK fuzzy system trains, and its recurrent
   form scales smoothly** (FRLM: 1.961 → 1.368 bits/char from 8.6K to 220K params). With
   HTSK rule firing it reaches 2.0 bits/char at ≤ 8.6K parameters. At matched size it trails
   the best neural linear-cost model by **0.07–0.08 bits/char**, i.e. it needs ≈ 1.5× the
   parameters at 1.6 bits/char.
3. **Fuzzy attention is free; the fuzzy feed-forward layer is the cost.** Writing attention
   as a Gaussian-TSK rule base (one rule per past token) matches softmax and is marginally
   better (−0.013 / −0.008 at d = 32 / 64). The classic TSK feed-forward costs +0.12 / +0.32
   at the same width, because its rule softmax saturates with dimension.
4. **HTSK (Cui, Wu & Xu 2021) removes most of that cost**, about −0.10 bits/char. A direct
   2×2 test places the gain entirely in the feed-forward layer. An earlier cross-arm
   inference said the opposite and is corrected.
5. **Negative results:** data-driven rule initialisation hurts or does nothing; the
   learning rate tuned at width 32 transfers to width 96 for attention mixers (checked) and
   is suspected not to for GRU / FLM / classic FRLM (not yet checked). Rule initialisation
   from n-gram statistics is a null at d = 16 on that old FFN (§5.6).
6. **GPU phase (§10–§12, transcribed, not re-run):** redesigning the TSK feed-forward
   (first-order consequents, un-normalised firing, HTSK) and adding heads gives a 97.5K-parameter
   fuzzy model that scores **1.274** vs **1.293** for the size-near neural reference at 30M
   characters (3 seeds; a 2.85-standard-error lead that is **not yet citable**: the reference
   is 4.6% smaller and less tuned). A BPE1024 version (93K params) reaches **1.129** at 30M
   (2 seeds), **1.055** at 100M and **1.021** at 300M (1 seed each), and, at 10M characters on the same
   tokenizer, tuned fuzzy D beats both tuned neural references by **0.027 and 0.030** bits per character at ten seeds, 30M characters (§11, `v6-final10`).
   Post-training adds nothing beyond compute. The open fairness gaps are listed in §12.

## 2. Questions

* **Size.** How small can a CPU-trained causal LM on TinyStories be, and does the answer
  depend on quadratic (softmax) vs linear-cost sequence mixing?
* **Fuzzy.** Can every sublayer be a Takagi–Sugeno–Kang system?
  * **FLM:** TSK attention + TSK feed-forward.
  * **FRLM:** a fixed TSK rule base whose consequents are recurrent state.
  * What does that cost per parameter?
* **Weights ↔ rules.** What is the exact correspondence between these mixers' weights and
  fuzzy rules?

The prompt for the size question was a ~19K-parameter "TinyStories" model on Reddit. It is
**MacroStories** (19,969 params, one shared block applied 4×). It was trained on its own
Gemma-generated stories, not TinyStories, on a GPU, and reports no loss, so it is a size
reference, not a comparable result (`literature/01`).

## 3. The correspondences ([`theory.md`](theory.md); each pinned by a test)

* **Softmax attention is a weighted zero-order TSK system with a dynamic rule base.**
  Expanding q·k = ½(‖q‖² + ‖k‖² − ‖q−k‖²), softmax(q·k/τ) equals Gaussian memberships
  exp(−‖q−k_s‖²/2τ) times a rule weight exp(‖k_s‖²/2τ). Each context token is one rule,
  "IF query ≈ k_s THEN v_s". The quadratic cost is a rule base growing by one rule per token.
  *Not new:* Jiang … Lin (IEEE TFS 2025) derive softmax attention as TSK firing; Tsai et al.
  (2019) read it as a kernel smoother. The rule-weight and RoPE-commuting-width bookkeeping
  is ours.
* **Normalised linear attention with a non-negative feature map is a TSK system with a
  fixed rule base and recurrent consequents.** Rule r's consequent is the decayed fuzzy mean
  of the values written into it. Katharopoulos' elu+1 map reads as one monotone,
  single-antecedent "rule" per feature. The FRLM uses a fuzzy partition of R Gaussian rules
  instead, which compresses the per-token rules into R prototype rules (fuzzy-clustering
  compression of attention).
* **The delta rule on fuzzy-partition features is in-context normalised-LMS training of
  zero-order TSK consequents.** That is ANFIS's consequent update, run within the context
  window. It is stable for β ∈ (0,1) since ‖φ‖₂ ≤ 1. One targeted search found no prior
  statement of this fuzzy reading; it is not yet checked for priority.
* **HTSK** replaces the product t-norm's summed exponent by its mean, so each rule fires by
  the *geometric mean* of its per-dimension memberships. That is an averaging operator, not
  a t-norm, and the rule reading must say so.

**Novelty, stated narrowly** (`literature/03`, re-check before the defense): "first fuzzy
language model" is **not** defensible. FuzzyS2S/GenFS (IEEE TNNLS 2026), NC-FFN (arXiv
2606.31845) and fuzzy class n-gram LMs exist; recurrent TSK networks date from 1999–2002;
"neural weights ≡ fuzzy rules" goes back to Jang & Sun (1993), Benítez et al. (1997) and
Mantas & Puche (2008). What appears open is *a causal LM in which every sublayer is a TSK
system, including a recurrent variant whose rule consequents are fit in context by LMS*.

## 4. Setup

| | |
|---|---|
| Data | TinyStories V2 (GPT-4 split), HF `roneneldan/TinyStories` @ `f54c09f`; 2.19B train chars |
| Tokens | 98 symbols: printable ASCII, `\n`, EOT, UNK. UNK is 0.06% of bytes (non-ASCII) |
| Metric | Validation bits per character, first 1M chars of V2 valid, non-overlapping 256-char windows |
| Model | Pre-norm residual stack, 2 layers, 2 heads, tied embeddings; widths 16–96 (≈5K–220K params) |
| Budget | 30M chars per scaling run (10M for tuning), batch 32 × 256, AdamW, warmup + cosine to 10% |
| Hardware | Intel i7-1185G7, CPU only, 1 thread per run, 4–8 runs in parallel |
| Tuning | Identical grid for every arm: lr ∈ {1e-3 … 1e-1} (6) × short-conv ∈ {off, 4}, d = 32, seed 0 |
| Seeds | 3 per scaling cell (0–2) |

**Arms.** Neural: `softmax-mlp`, `linear-mlp` (elu+1, fixed decay), `gla-mlp` (data-dependent
decay), `delta-mlp` (DeltaNet), `gru`. Fuzzy: `flm` (Gaussian-TSK attention + TSK FFN),
`frlm-acc` / `frlm-delta` (16 Gaussian rules per head, accumulating or delta-rule
consequents, TSK FFN), and their `-htsk` versions.

**Protocol amendments**, all logged in `RESULTS.md` before the affected results were seen:
1. The lr grid was extended twice, for every arm, when optima hit the grid edge.
2. A gated edge check blocked scaling until every optimum was interior.
3. The ten-seed headline was moved to the GPU host as complete single-platform grids.

**The HTSK arms got extra search** (a 6-variant design iteration) that the neural arms did
not. They are labelled as post-hoc wherever they appear.

## 5. Results

### 5.1 Smallest model reaching each quality level (3 seeds, CPU, 30M chars)

Total parameters, log-interpolated at the first crossing. `≤` means the smallest size trained
already reached the level, so the number is an upper bound.

| model | ≤ 2.0 | ≤ 1.8 | ≤ 1.6 | ≤ 1.4 |
|---|---|---|---|---|
| softmax attention | 7,435 | 12,496 | 26,701 | 70,210 |
| linear attention | 6,547 | 11,714 | 26,013 | 70,048 |
| gated linear attention | 6,369 | 11,493 | 25,211 | 66,560 |
| DeltaNet | ≤ 6,008 | 10,961 | 23,677 | 68,432 |
| GRU | 5,850 | 10,268 | not reached | not reached |
| FLM | 11,413 | 22,027 | 125,627 | not reached |
| FRLM delta (classic) | ≤ 8,568 | ≈ 18,300 | not reached | not reached |
| FRLM delta + HTSK † | ≤ 8,568 | 14,931 | 34,846 | 140,947 |

† post-hoc arm with extra search. Full per-width table: `outputs/scaling-all/summary.md`.

**Linear vs quadratic (H1 refuted).** At the same width, softmax is never reliably ahead:

| width | softmax | linear | gated | DeltaNet |
|---|---|---|---|---|
| 32 | 1.6726 ± .0022 | 1.6599 ± .0018 | 1.6507 ± .0045 | 1.6330 ± .0027 |
| 48 | 1.4863 ± .0041 | 1.4911 ± .0029 | 1.4846 ± .0040 | 1.4780 ± .0045 |
| 64 | 1.3942 ± .0034 | 1.3935 ± .0027 | 1.3829 ± .0048 | 1.3890 ± .0164 |
| 96 | 1.2916 ± .0074 | 1.2886 ± .0031 | 1.2898 ± .0035 | 1.2943 ± .0090 |

Below ~20K parameters the order is DeltaNet < gated < linear < softmax. From width 48 up,
all four are within ~0.01. What linear-cost mixing does buy is memory. At width 16, softmax
carries a KV cache larger than its own 5,872 parameters after **91 characters** of context
(155 at width 32), while DeltaNet and the FRLM carry a fixed 352–608 floats.

**Training cost (caveat).** At T = 256 the linear mixers train in their parallel T×T form.
On CPU that is *not* cheaper than PyTorch's fused softmax kernel, and the delta-rule arms
cost ≈ 3–5× softmax per character because of the triangular solve. Timings come from runs
sharing 4 cores and are relative only.

### 5.2 The fuzzy components, one at a time (`ablate`, 3 seeds; 24 of 36 cells complete)

| change against softmax-mlp | d = 32 | d = 64 | |
|---|---|---|---|
| attention → Gaussian-TSK attention | **−0.013** | **−0.008** | H2 supported: fuzzy attention is free |
| MLP → classic TSK FFN | **+0.118** | **+0.315** | H3 supported: the FFN is the cost, growing with width |
| + data-driven rule init (TSK FFN) | +0.132 | −0.017 (within noise) | H4 refuted |
| FLM + data-driven rule init | +0.078 | +0.023 | H4 refuted |

The FLM scales poorly (1.553 at 213K vs softmax 1.292 at 158K) because of its feed-forward
layer, not its attention.

### 5.3 HTSK, and where its gain lives (`fuzzyfix`, `h9`)

The classic TSK FFN uses 64 rules over a 32-dimensional antecedent. Its normalised firing is
a softmax over exponents that grow linearly with dimension, so one rule takes all the mass:
6–12 of 64 FFN rules ever won a position in early rulebooks. HTSK averages the exponent over
dimensions instead of summing it.

| frlm-acc, d = 32, 10M chars | mixer sum | mixer HTSK |
|---|---|---|
| **FFN sum** | 1.858 ± 0.006 | 1.860 ± 0.023 |
| **FFN HTSK** | **1.752 ± 0.014** | 1.759 ± 0.003 |

(3 seeds; frlm-delta at 1 seed shows the same pattern: FFN-only recovers 114% of the gain,
mixer-only 20%.)

* **The gain lives entirely in the TSK feed-forward layer.**
* **Correction:** the first attribution inferred the opposite *across arms* (the FLM did not
  improve) and was published to the PR before the direct test overturned it. Both are kept
  in `RESULTS.md`.
* Determinism was checked: independent reruns of identical configurations reproduce val
  BPC exactly.
* FFN-only HTSK is marginally better than HTSK on both layers. The `-htsk` arms in the
  scaling sweep use both layers and may be slightly off their best configuration.

### 5.4 The FRLM against neural models at matched size (H7)

Gap to the best of linear / gated / DeltaNet / GRU at the same parameter count, interpolated
on the neural curve, with no extrapolation beyond 158K:

| params ≈ | 17K | 28K | 59K | 102K |
|---|---|---|---|---|
| FRLM delta (as registered) | +0.14 | +0.16 | +0.25 | +0.30 |
| FRLM delta + HTSK † | +0.08 | +0.08 | +0.07 | +0.08 |

H7 ("within 0.10 at d ≥ 32") **fails for the arms as registered**. The post-hoc HTSK arm
sits inside the margin across the range; that is reported as a post-hoc observation, not
as a pass.

### 5.5 Learning-rate transfer across width (`lrwidth`, 6 of 20 cells)

At d = 96, seed 0, with the d = 32 tuned lr ×1 / ÷3 / ÷10:

| arm | ×1 | ÷3 | ÷10 |
|---|---|---|---|
| softmax | 1.3000 | +0.007 | +0.033 |
| linear | 1.2921 | −0.009 | +0.016 |
| gated | 1.2925 | −0.008 | +0.023 |

For attention mixers the tuned lr transfers to 3× the width within 0.01 (H10b so far).
**Not yet measured:** GRU, FLM and the classic FRLMs, which get *worse* with width
(GRU 1.68 → 1.81; frlm-acc 1.665 → 1.756). Until their cells run, their large-width points
are treated as LR-limited and say nothing about capacity.

### 5.6 Initialising the fuzzy rules from n-gram statistics (CPU screen, d = 16)

A count-based n-gram language model is a crisp zero-order TSK system ("IF the last k characters are g THEN
p(next | g)"), so the fuzzy rules can start from it. `flm/ngram.py` places one rule per frequent context
(centres found empirically through the forward pass), decodes the consequents through the tied embedding and
calibrates rule width, consequent scale and output gain on training windows.

* **Reference points on this data** (1M validation chars): unigram 4.42, **bigram 3.30, trigram 2.39,
  four-gram 1.82** bits/char. The 8.6K-parameter FRLM (1.96) sits between trigram and four-gram quality.
* **A random d = 16 embedding cannot represent n-gram distributions:** its rank-16 readout of the bigram
  log-probs scores 7.5 BPC (worse than uniform, 6.6); an embedding aligned to the top-16 principal
  directions of the conditional log-probs scores 3.57. The first init attempt failed for exactly this reason.
* **A 32-rule bank covers 98% of positions with bigram rules but only 43% with trigram rules** (ceilings 3.59
  and 4.04 BPC); clustering contexts into rules does not beat top-k.
* **The init works as an init:** held-out starting BPC 3.70 (bigram) and 4.16 (trigram, 2-tap antecedent)
  against 6.62 for a random start; zeroing the layer-0 consequents sends BPC to 12+, so the rules carry it.
* **It does not survive training.** Every variant (d = 16, 10M chars, 4 baseline seeds, mean 2.077, std 0.026):
  n-gram init **+0.027**, 2-tap antecedent **−0.030**, FFN-only HTSK **−0.025**, none beyond 2 standard
  errors. The head start is gone within ~100 of 1221 steps. A lesson from this screen: with 2 baseline seeds the
  noise estimate was off by 35%, and a one-seed variant ranking is not usable at this design's noise.
* **Scope:** a null for the *old zero-order FFN* at 8.6K parameters. The GPU session's FFN (§10) has seed noise
  ~10× smaller, so the same ideas are testable there (`v4-taps`); the init now runs on that FFN (starting BPC
  3.94 bigram / 3.47 trigram + 2 taps on D) but has not been trained.

## 6. Hypothesis scorecard

| | hypothesis (abridged; registered text in `RESULTS.md`) | outcome |
|---|---|---|
| H1 | softmax beats linear at every width ≥ 32 | **refuted** |
| H2 | Gaussian-TSK mixer ≈ softmax (\|Δ\| < 0.02) | **supported** (slightly better) |
| H3 | TSK FFN worse than MLP by > 0.03 | **supported** (+0.12 / +0.32) |
| H4 | data-driven rule init helps | **refuted** (frlm cells unfinished) |
| H5 | FRLM delta beats accumulating at every width | **refuted as stated** (delta wins from d ≈ 32–48 up) |
| H6 | softmax reaches 2.0 below 15K params; linear-cost needs ≥ as many | **half**: 7.4K ✓, linear-cost needs fewer ✗ |
| H7 | best FRLM within 0.10 of best neural linear-cost at d ≥ 32 | **refuted** as registered; post-hoc HTSK arm inside |
| H8 | an HTSK / data-init variant beats classic | **supported** for FRLMs (predicted winner wrong); not for FLM |
| H9 / H9b | where the HTSK gain lives | **gain is in the FFN** (H9 supported; H9b refuted) |
| H10 | tuned lr transfers to d = 96 | **H10b supported so far**; H10a open |
| H11 / H12 | true minimum size; leaner rules / shared depth | **not run** (stopped); shared depth and shared widths screened at d = 16: worse, in line with their parameter counts |
| P1–P12 | n-gram init screen (`RESULTS.md`) | init is a null (§5.6); GPU-phase hypotheses H13–H23 are scored in §10–§11 |

## 7. Limitations

* **Three seeds, one tune seed, one tune width.** The tune picks are single-seed at d = 32,
  and gaps under ~0.03 between neural arms at that stage were never established.
* **Learning rate fixed across width.** Confirmed harmless for attention arms; unverified for
  the arms that degrade.
* **Extra search for the HTSK arms** (disclosed); the neural arms got no analogous iteration.
* **Character vocabulary with UNK** for 0.06% of bytes. BPC is exact for comparisons between
  arms, but it is not bits-per-byte of the raw file and is not directly comparable to
  subword-tokenised literature numbers.
* **Single platform** (one CPU). The GPU repeat is a different platform and must not be
  pooled with these cells (the analysis keys cells on device).
* **Unfinished grids:** `ablate` (12 cells), `lrwidth` (14), `minsize` (24), `lean` (12).

## 8. Repeating and extending on the GPU host

> **Partly superseded.** The GPU session ran its own track (§10–§11) rather than this repeat. The
> unfinished CPU grids (`ablate`, `lrwidth`, `minsize`, `lean`) and the ten-seed `headline10*` grids
> concern the old zero-order FFN and have not run. What to run next, in priority order, is in
> [`PLAN_GPU.md`](PLAN_GPU.md); who owns what is in [`COORDINATION.md`](COORDINATION.md).

Everything is resumable. Grids skip finished records, hyperparameters come from the
committed `outputs/tune/*.json`, and every record carries `device` / `host`.

```bash
cd experiments/fuzzy-language-model
# 1. the citable numbers: ten seeds at d=32, complete single-platform grids
.venv/bin/python -m flm.sweep headline10   --device cuda --workers 8
.venv/bin/python -m flm.sweep headline10v2 --device cuda --workers 8
# 2. the unfinished CPU grids (records land in their own dirs, keyed device=cuda)
.venv/bin/python -m flm.sweep lrwidth  --device cuda --workers 8   # H10a: is gru/flm/frlm LR-limited?
.venv/bin/python -m flm.sweep ablate   --device cuda --workers 8   # frlm data-init cells
.venv/bin/python -m flm.sweep minsize  --device cuda --workers 8   # H11: true size at 2.0 bits/char
.venv/bin/python -m flm.sweep lean     --device cuda --workers 8   # H12: shared widths, shared depth
git add -f outputs/*/*.json && git commit -m "GPU records" && git push
```

**Before mixing results:** the CPU records for `ablate` and `lrwidth` are committed in the
same directories. `flm.analyze` keys cells on device, so CPU and GPU runs stay separate, but
a seed run on the CPU is *skipped* on the GPU (a record exists). To give the GPU grid every
cell, run it into a fresh checkout's `outputs/`, or move the CPU records aside first.

**Next arms, in order of expected value:**
1. **FLM with an HTSK feed-forward** (`--exp-norm sum --ffn-exp-norm mean` on `flm`). Fuzzy
   attention is already free, and the FFN fix carried the whole FRLM gain, so this is the
   most likely route to an all-TSK model at parity.
2. **FRLM with FFN-only HTSK**, the marginally better configuration from `h9`.
3. **The rulebook of the ~2 bits/char FRLM** (`flm/rules.py` on a saved checkpoint): about
   130 rules, small enough to read in full. This is the dissertation artifact.

Each new arm gets the standard 12-cell tune before it enters a comparison, as the
`ARMS_V2` pattern in `flm/grids.py` shows. Never add arms to `ARMS` itself: the queued grids
would silently go empty.

## 9. Reproduce

`README.md` → Reproduce. Every number in §5 is regenerated from the committed run records
by `flm/analyze.py` (tables) and `flm/dashboard_export.py` (the dashboard's six datasets,
verified byte-identical to the published ones). Tests (32) pin the identities in §3:
`.venv/bin/python -m pytest -q`.

## 10. GPU phase: a fuzzy LM that leads the size-near neural reference (transcribed, not re-run)

*Source: `RESULTS.md`, "Goal track" and the `v3-*` records, by the GPU session. Objective: best val BPC at < 100K
total parameters, embedding included, on CUDA, 30M chars unless stated. Nothing here was pre-registered before its
screen; each stage was chosen after seeing the last, so none of it is a hypothesis test.*

**10.1 Where the fuzzy deficit was** (`v3-attrib`, d = 64, seed 0). Every mixer × FFN pairing:

| mixer | MLP FFN | TSK FFN (the old design) |
|---|---|---|
| softmax | 1.512 (72.6K) | 1.826 (97.2K) |
| Gaussian-TSK attention | 1.492 | 1.781 |
| linear | 1.503 | 1.720 |
| FRLM, accumulating | 1.507 | ≈ 1.715 (101.6K) |

The fuzzy *mixers* cost nothing; the whole ≈ 0.3 BPC fuzzy deficit and ≈ 25K of the parameters were the
zero-order, softmax-normalised TSK FFN. This independently confirms the CPU finding (§5.2–5.3) at a larger size.

**10.2 The FFN redesign** (`v3-ffn*`, fuzzy-accumulating mixer, d = 64). In order of effect: tied per-rule Gaussian
widths (fewer parameters); fewer antecedent dimensions (16); **first-order consequents** (rule output
(vᵣ·x)·oᵣ, a GLU whose gate is the fuzzy membership); **un-normalised firing** (no softmax across rules); the
HTSK exponent. Raw firing with the summed exponent is unstable (3 seeds: 1.517 / 1.588 / 1.557); with HTSK it is
stable (seed std 0.0003–0.002). At 3 seeds, 10M chars: the fuzzy FFN **1.4675 ± 0.0003** (87.5K) vs the same mixer
with a GELU MLP **1.4957 ± 0.0086** (77.0K). *Rejected:* raw firing with 32-dim antecedents (1.685), lr 6e-2.

**10.3 Mixer and shape** (2 seeds). The gain is mostly **head count** (1 / 2 / 4 / 8 heads: 1.534 / 1.464 / 1.441 /
1.420); the delta form overtakes accumulation from 4 heads up; 16 rules per head beat 8, 32 is no better;
short-conv 8 > 4; depth (3 layers at d = 48, 4 at d = 40) is no better than 2 at d = 64; lr 3e-2 ≈ 4e-2 > 2e-2.
The filled configuration **D** (d = 72, 8 heads, 16 rules per head, FFN width 1.5× rules, short-conv 8) has
**97,520 parameters**.

**10.4 The comparison** (CUDA, 30M chars, 3 seeds):

| model | params | val BPC |
|---|---|---|
| **fuzzy D** | 97,520 | **1.2735 ± 0.009** |
| delta-mlp, d 72, 8 heads | 93,272 | 1.2929 ± 0.004 |
| delta-mlp, d 72, 4 heads | 92,104 | 1.3108 ± 0.006 |
| delta-mlp, d 74, 2 heads | 96,430 | 1.3426 ± 0.007 |
| original FLM, d 64 (the CPU-phase design) | 97,280 | 1.6231 ± 0.008 |

At 100M chars (seed 0, one run each): D **1.2080** vs a *smaller* (73K) delta-mlp 1.3270. D trained at 34K chars/s. The size-near reference's three runs recorded 50K, 56K and 85K
chars/s on the shared GPU (the 114K figure quoted in `RESULTS.md` is the *smaller* 73K delta-mlp at
100M chars), so the speed gap is roughly 1.5–2.5×, not a precise ratio. **The lead over the closest reference is 0.019 BPC = 2.85 standard errors at
3 seeds (CPU-session calculation) and is not yet a citable claim:** the reference has 4.6% fewer parameters
(≈ 0.006 BPC of the lead from the `scaling2` slope), D's search covered heads, rules, antecedent size, short-conv,
depth, FFN shape and lr while the reference got heads and one lr, and ten seeds are the protocol (§12).

**10.5 Bigger models and post-training** (`v3-big`, `v3-post`; registered before running; 2 seeds unless noted).

| D at | params | 30M val BPC | +10M continuation | + EMA | + distillation |
|---|---|---|---|---|---|
| d = 72 | 97,520 | 1.2753 | **1.2486** | 1.2481 | 1.2496 |
| d = 80 | 117,312 | 1.2485 | | | |
| d = 88 | 138,896 | 1.2351 | | | |
| d = 96 | 162,272 | 1.2170 | 1.1951 | 1.1948 | |

BPC falls monotonically with size (H13 supported), but the gain per parameter is tapering (−0.018 for the last
+23K). **Post-training is a null:** +10M chars at lr 3e-3 gives −0.027 (this is just more training and equals
what the 30M → 100M curve predicts), weight EMA adds −0.0005, and distilling from a 280K-parameter neural teacher
that is 0.15 BPC better adds +0.001. Everything that "helped" was compute.

## 11. Subwords: the best model under 100K is a BPE1024 fuzzy model (transcribed, not re-run)

*Source: `RESULTS.md`, `v5-*`. The user's goal is coherent text; at ≈ 1.2 BPC every character model writes real
words but loses characters and plot. A byte-level BPE (`flm/bpe.py`) lets a 256-token window cover ≈ 600 characters.
BPC is still bits per character of the same validation text (token character lengths sum exactly to the stream
length). CPU-session note: the model is scored on the canonical tokenisation only, which is a lower bound on
the text's probability, so the reported BPC is, if anything, conservative for the subword models.*

| experiment | model | params | val BPC |
|---|---|---|---|
| `v5-bpe-probe` (10M, seed 0) | char D d = 96 | 162,272 | 1.3380 |
| | BPE512 D d = 72 | 127,328 | 1.2671 |
| | BPE512 D d = 80 | 150,432 | 1.2479 |
| `v5-bpe30` (30M, 2 seeds) | **BPE512 D d = 80** | 150,432 | **1.1028 ± 0.0054** |
| | char D d = 96 | 162,272 | 1.2170 |
| | neural delta-mlp char d = 128 (60M chars) | 280,480 | 1.1153 |
| `v5-sub100k` (30M, 2 seeds) | **BPE512 D d = 56** | 86,496 | **1.2054 ± 0.0090** |
| | BPE384 D d = 64 | 97,824 | 1.2260 ± 0.0024 |
| | char D d = 72 | 97,520 | 1.2753 |
| `v5-spend` (10M, 2 seeds, warm-up 100) | **BPE1024 D d = 48** | 93,344 | **1.2604 ± 0.0050** |
| | BPE768 D d = 48 | 81,056 | 1.3080 |
| | BPE512 D d = 56, FFN 2.0× | 94,616 | 1.3271 |
| | BPE512 D d = 56 (baseline) | 86,496 | 1.3335 |
| | BPE512 D d = 56, 24-dim antecedent | 90,080 | 1.3413 |
| `v5-final100k` | **BPE1024 D d = 48**, 30M, 2 seeds | 93,344 | **1.1292 ± 0.0031** |
| | BPE1024 D d = 48, 100M, 1 seed | 93,344 | **1.0548** |
| `v5-long` (300M, 1 seed) | BPE1024 D d = 48, lr 3e-2 / lr 2e-2 | 93,344 | **1.0212** / 1.0227 |

* **Caveats the GPU session asked to be stated:** the subword result is **2 seeds at 30M and 1 seed at 100M**;
  the neural control on the same tokenizer is at **ten seeds and 30M characters** (below), one tokenizer and size; the 30M, 100M and 300M runs used
  lr 3e-2, which is **below D's optimum of 6e-2 at 10M**, so they understate D; and the 150K-vs-162K comparisons are at equal
  *characters*, not equal steps (BPE sees ≈ 2.4× more characters per step, so it takes fewer optimizer steps and
  its warm-up is a larger share of training).
* A larger vocabulary buys the most at roughly constant parameters (512 < 768 < 1024, monotone); more FFN rules
  or a wider antecedent buy ≈ 0 (inside the 0.01 noise). The trend is untested above 1024.
* The 93K subword model at 30M chars (1.129) beats the 97.5K character model at 100M (1.208).
* **Returns on data shrink:** 30M → 100M gave −0.074 and 100M → 300M gave −0.034 (single seeds), so 1B characters would be expected to give about −0.02. The two learning rates tried at 300M (3e-2 and 2e-2) differ by 0.0015, inside seed noise.
* **The subword control (`v6-*`, 10M characters, BPE1024, ~93K parameters each; the GPU session ran it, the CPU
  session built the grids and verified the numbers from the records).** Fuzzy D over six learning rates (2 seeds each):
  2e-2 → 1.2809, 3e-2 → 1.2604, 4e-2 → 1.2542, **6e-2 → 1.2502 ± 0.0018**, 8e-2 → 1.2519, 1e-1 → 1.2628, so the
  optimum is interior. The neural reference (delta rule or gated linear attention, MLP padded to 93,376–93,392
  parameters; 24 cells over heads, short convolution and lr, one seed each): best **1.2666** (delta, 8 heads, short-conv 4,
  lr 4e-2), then 1.2694 and 1.2710. **Tuned fuzzy D leads the best tuned neural cell by 0.016 bits per character**
  at 10M characters (one reference seed). `v6-final10` then tested it at ten seeds and 30M characters:
  D 1.1291 ± 0.0039 against 1.1564 ± 0.0044 and 1.1593 ± 0.0038, leads of 0.027 and 0.030 (2 se 0.0037
  and 0.0034); D's worst seed beats the references' best. Citable with the caveats in the banner.
* **Text is better but not coherent.** At T 0.7 / top-k 20 the BPE1024 model invents more non-words (2.2%) than
  the character models (0.4–0.9%); sampling at T 0.5 / top-k 10 drops that to 0.65% at the price of looping
  ("They did not like the box. They did not like the box.", repeated-sentence share 2.9%). Samples show
  dialogue and a story arc of sorts, then semantic errors (a girl named Lily "wants to play with Lily"). BPC is
  not coherence, and no coherence grader exists.

## 12. What stands between these results and a citable claim

| gap | why it matters | how it closes |
|---|---|---|
| **Char-level D vs the reference at equal effort** | 4.6% parameter mismatch; D searched far more than the reference; 3 seeds | `v4-ref-tune`, `v3-lr`, `v4-final10` (`PLAN_GPU.md`) |
| **Neural control on the same tokenizer**: *closed at ten seeds, 30M characters* | D beats both tuned references by 0.027 and 0.030 (`v6-final10`); other tokenizers, sizes and budgets, each with its own tuned reference, are untested | the generality grids |
| **D's lr on subwords**: *closed at 30M* | D at 6e-2 (ten seeds) 1.1291 vs 3e-2 (two seeds) 1.1292; not checked at 100M and 300M, so quote those only as lower bounds | an lr check at 100M, 300M |
| **Ten seeds** | done for the subword comparison (`v6-final10`); the character-level comparison has 3 | `v4-final10` |
| **LR transfer (H10a) and `v3-lr`** | incomplete: the 1e-2 cells were lost to a crash from a stale `train.py`, the 2e-2 cells were killed for VRAM contention | resume `v3-lr`, `lrwidth` |
| **Coherence** | BPC is not coherence; text loops at low temperature | a coherence grader, or longer context and larger models |
| **One GPU, one tokenizer family** | unreplicated | `COORDINATION.md` |

Vocabulary ladder (`v7-vocab`, `v7-confirm`, two seeds each, lr 3e-2): at 10M characters BPE1536 d40 (93,792 parameters) scored 1.2336 and BPE2048 d32 1.2378 against 1.2604 for BPE1024 d48, but at 30M characters the 1536 gain shrank from 0.027 to 0.009 (1.1205 against 1.1291) and 2048 lost its edge (1.1316), so the 10M screens overstated larger vocabularies. The headline stays BPE1024 d48; BPE1536 d40 needs ten seeds and its own neural control before any fuzzy-vs-neural claim at that vocabulary.

Only the `v6-final10` comparison of §11 may be quoted in the proposal, with its caveats; the character-level claims wait for the first row of this table.

## 13. GPU phase, final state (2026-10-12): sub-100K is capped, coherence is a size lever

Measured by the GPU session; every row is in `RESULTS.md` / `PLAN_*.md` with the prediction registered first. Seeds are stated; only the 30M five-seed screens and the 300M two-seed confirmations below carry more than one seed, so **none of this is a ten-seed claim**.

**Best sub-100K recipes (val BPC, 96K reference = BPE1024, d48, 2 layers, 32 rules/head):**
| recipe | params | 30M chars | 300M | 1B | 3B |
|---|---|---|---|---|---|
| reference (warm-up 100) | 96,416 | 1.1095 +- 0.0107 (5 seeds) | 0.9976 / 1.0053 | 0.982 | 0.97767 |
| warm-up 300 | 96,416 | 1.0984 (-0.0111, 5/5) | ~0 gain (2 seeds) | | |
| frequency-tiered tied embedding (head 768 x d48, tail 16-dim, FFN mult 2.25) | 98,280 | 1.0965 (-0.0130, 4/5) | **0.981 / 0.982** (-0.017 / -0.023) | **0.96796 / 0.96129** (seeds 0 / 1, mean 0.9646 vs 0.9820 for the reference; new sub-100K best) | not run |

The tiered embedding reaches the 96K model's 1B-char BPC with a third of the data. Warm-up 300 helps at 30M and vanishes at 300M, so the cited recipe keeps warm-up 100.

**Closed levers (all at 30M, five seeds unless noted, null or worse than the random-init reference):** consequent-first / backwards rule construction (+0.045); data-driven, clustered, hybrid and D^2-cover rule placement (+0.025 to +0.042); PPMI-initialised and re-scaled embeddings (-0.006 to +0.006); repetition-weighted loss (-0.0004) and unlikelihood loss (+0.005, rated *less* coherent blind); tiered embedding with the saved parameters spent on width d56 (-0.007 / +0.001, worse than spending them on the FFN).

**Coherence is a size lever, not a data or recipe lever.** Blind sets (two same-family Claude raters, text only, scores before the key; rubric G/C/P 1-5, total 3-15; one seed per model, 8 texts per model, rater scale shifts between sets so compare within a set):
| model | params | 1B-char BPC | set 12 total (plot) A / B |
|---|---|---|---|
| 96K | 96,416 | 0.982 | 4.75 (1.25) / 4.62 (1.12) |
| 251K | 251,168 | 0.817 | 6.38 (1.88) / 6.25 (1.62) |
| 653K | 653,248 | 0.694 | 8.88 (2.38) / 9.00 (2.38) |
| 1.32M | 1,323,904 | 0.636 | 10.00 (3.12) / 10.12 (2.88) |
Set 10 gave the 1.32M model 9.6 / 10.4 against 5.3-5.5 for the 96K models at 1B and 3B chars; set 11 gave the tiered 98K model at 300M the same ~5.5 as the 96K models despite 0.017-0.023 lower BPC. The 10/15 bar (total >= 10, plot >= 2.5) is reached at about 1.3M parameters x 1B chars and not below ~650K. The 12/15 bar is not reached by any model. Rule readability of the 1.32M checkpoint (`rule_ablation`, `rule_reader`, `rule_story`) has not been run.

**Cap status.** The 99,999-parameter cap was exceeded for the 1.32M run (explicit user approval) and for the 251K / 653K ladder rungs (user: "raise the cap and keep moving", understood as covering up to 1.32M). Larger rungs (2.2M-7M) were not run.

**Larger general corpus (Cosmopedia v2, 3.0B characters, BPE1024 retrained, `PLAN_CORPUS.md`).** Same-size models are far weaker per character on broad synthetic text: 96K reference 1.873 BPC at 30M (five seeds) and 1.761 at 300M (one seed), against a 300M-token trigram table at 1.622; the tiered embedding helps by -0.017 at both scales (5/5 seeds at 30M). The TinyStories result that 96K saturates by ~1B characters does not transfer: on this corpus the model is capacity-bound. No 3B-char run and no coherence rating yet.

**Next (planned, not run):** a larger general-text corpus at the same sizes (`PLAN_CORPUS.md`) and a Matryoshka-trained embedding for a smaller interpretable embedding space (`PLAN_MATRYOSHKA.md`).

## 14. What the size ladder and the Cosmopedia runs taught us (2026-10-13; one seed per size unless stated)

**Option 4 noted (a BPC-only sub-100K target).** The goal "as capable as possible under 100K" has two measurable halves, and they separate:

* **Likelihood (BPC) at sub-100K is nearly saturated on TinyStories.** The tiered 98K recipe reached 0.9646 (two seeds, 1B chars) against 0.9820 for the 96K reference; each remaining lever (data 1B -> 3B, width/head variants, tiered widths, Matryoshka training) moved it by < 0.02 or made it worse.
* **Readable coherence is a size lever, and it saturates too.** Blind totals (G+C+P, 3-15; two raters; compare within a set only): 96K ~5, 251K ~6.3, 653K ~8.9, 1.32M ~10, 2.22M ~9.7 (set 14: 9.5/9.9 vs 9.9/10.5 for the 1.32M model). BPC fell 0.980 -> 0.817 -> 0.694 -> 0.636 -> 0.599 along the ladder, but the last step bought no rated coherence. The 12/15 bar was reached by no model of any size; no sub-100K model passes ~8 on any rater, and the best sub-100K BPC (tiered 1B) did not raise coherence beyond seed noise (set 13).
* **Broad text is capacity-bound, not data-bound, at 96K** (Cosmopedia, `PLAN_CORPUS.md`): 96K at 300M chars 1.761 vs a 300M-token trigram 1.622; the 2.22M model gets 1.473 / 1.201 / 1.134 at 30M / 300M / 1B chars (`PLAN_BIG2M.md`), beating the trigram everywhere. A mixed TinyStories + Cosmopedia corpus is worse than either specialist at 96K. So a sub-100K model should be judged on its narrow-domain BPC and with a narrow-domain coherence bar.

**Amended target (accepted by the user 2026-10-19: "The modified target is fine"):** report sub-100K capability as BPC on TinyStories (0.9646, two seeds) and rated coherence ~5-8/15, and state that ~10/15 needs >= ~1.3M parameters at 1B chars. That is a finding about this architecture and corpus, not a claim about all tiny LMs. The Cosmopedia 3B-char tiered 98K run reached 1.718 BPC (-0.026 vs 300M; no data lever left) and the expository coherence set 15 rated every sub-100K model at the rubric floor (3-4/15; the 2.22M model 7/15; `PLAN_CORPUS.md`): on broad text sub-100K is not a coherence candidate at any data budget tried.

## References

Full verified bibliographies are in [`literature/`](literature/README.md) (three reviews, 86
BibTeX entries). Cited directly above:

* Cui, Wu & Xu, *Curse of dimensionality for TSK fuzzy neural networks*, IJCNN 2021,
  [arXiv:2102.04271](https://arxiv.org/abs/2102.04271)
* Eldan & Li, *TinyStories*, 2023, [arXiv:2305.07759](https://arxiv.org/abs/2305.07759)
* Katharopoulos et al., *Transformers are RNNs*, ICML 2020, [arXiv:2006.16236](https://arxiv.org/abs/2006.16236)
* Schlag, Irie & Schmidhuber, *Linear transformers are secretly fast weight programmers*, ICML 2021, [arXiv:2102.11174](https://arxiv.org/abs/2102.11174)
* Yang et al., *Parallelizing linear transformers with the delta rule*, NeurIPS 2024, [arXiv:2406.06484](https://arxiv.org/abs/2406.06484)
* Tsai et al., *Transformer dissection*, EMNLP 2019, [arXiv:1908.11775](https://arxiv.org/abs/1908.11775)
* Jiang … Lin, IEEE TFS 2025, [10.1109/TFUZZ.2025.3528376](https://doi.org/10.1109/TFUZZ.2025.3528376)
* Yang, Deng et al. (FuzzyS2S / GenFS), IEEE TNNLS 2026, [arXiv:2411.13867](https://arxiv.org/abs/2411.13867)
* Oskin (NC-FFN), [arXiv:2606.31845](https://arxiv.org/abs/2606.31845)
* Jang & Sun, IEEE TNN 1993, [10.1109/72.182710](https://doi.org/10.1109/72.182710); Jang (ANFIS), IEEE TSMC 1993, [10.1109/21.256541](https://doi.org/10.1109/21.256541)
* Mantas & Puche, IEEE TFS 2008, [10.1109/TFUZZ.2007.902016](https://doi.org/10.1109/TFUZZ.2007.902016)
* Vyas, Katharopoulos & Fleuret, *Clustered attention*, 2020, [arXiv:2007.04825](https://arxiv.org/abs/2007.04825)
