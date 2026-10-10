# Tiny fuzzy language models: linear vs quadratic attention, and a recurrent TSK language model

**Status:** CPU phase stopped 2026-10-10, to be repeated and extended on a GPU host ·
**Experiment:** `experiments/fuzzy-language-model/` · **PR:** #272 · **Dashboard:**
https://claude.ai/artifact/XGJzPWz5T5i2wQvVE6C3Y4 (private until shared)

This write-up summarises 365 completed training runs (≈235 CPU-hours on a 4-core laptop,
2026-10-08 → 10). The laboratory record behind it is [`RESULTS.md`](RESULTS.md): every
hypothesis as registered before its run, every protocol amendment, every correction. Where
this document and `RESULTS.md` differ, `RESULTS.md` is the record.

> **Citability.** Every cell here is 3 seeds or fewer. AGENTS.md non-negotiable 2 makes ten
> seeds the protocol for quotable numbers, and the ten-seed `headline10` grids were offloaded
> to the GPU host and have not run. **Nothing below may be quoted in the proposal** until
> it is repeated at ten seeds.

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
   is suspected not to for GRU / FLM / classic FRLM (not yet checked).

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
| H11 / H12 | true minimum size; leaner rules / shared depth | **not run** (stopped) |

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
