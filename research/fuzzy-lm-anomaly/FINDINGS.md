# Fuzzy Anomaly Detection in a Frozen Small Language Model

Exploratory study: can the tribble "none of the above" anomaly rule (Ch 4.3.5)
flag hallucinated output from a small language model it was never trained on?

**Status:** first pass, single model, single seed, no error bars. Several results
are negative. Read the caveats before quoting any number.

---

## 1. Setup

| item | value |
|---|---|
| model | `HuggingFaceTB/SmolLM2-360M-Instruct`, 361.8M params, **frozen** |
| architecture | 32 layers (33 hidden states incl. embeddings), hidden 960, vocab 49152 |
| precision / hardware | fp16 on RTX 4080 Laptop 12 GB — **2.46 GB peak** |
| decoding | greedy, `max_new_tokens=48` |
| probes | 7,500 = 6,000 TriviaQA closed-book + 1,500 false-premise |
| capture cost | 6 min 40 s for all 7,500 generations |

The model is never modified: weights load in `inference_mode` with
`requires_grad_(False)`, and we only read activations and logits back out.

Per generation we record 19 output-distribution statistics (entropy, max-prob,
margin, chosen log-prob — each as mean/min/max/std plus first-token, and
perplexity) and the residual stream at **all 33 layers** at three pooling sites:

* `prompt` — state after reading the prompt, *before any token is emitted*
* `first` — representation of the first generated token
* `mean` — mean over generated tokens, masked to the real answer

### Protocol

Fit on **truthful answers only**; no hallucination is ever seen during fitting.
This is the open-set setup of the BETH experiment, transplanted.

Truthful answers split 60/20/20 into fit/val/test; hallucinations split 50/50
into val/test. **Every representation choice is selected on val and reported on
the disjoint test split**, so headline numbers are not tuned on the data they are
quoted against.

Evaluation is *within-family*: truthful and hallucinated TriviaQA answers share
one prompt distribution, so nothing can be separated on prompt topic. The
false-premise family is reported separately as a novel, easier open-set.

---

## 2. Adversarial elicitation works

|  family | correct | hallucination | abstain |
|---|---|---|---|
| TriviaQA | 1,761 (29.4%) | 3,948 (65.8%) | 291 (4.9%) |
| false-premise | 0 | **1,431 (95.4%)** | 69 (4.6%) |

The model fabricated an answer to **95.4%** of questions about entities that do
not exist, pushing back on only 4.6%. The very first probe invented an Alvin
Ailey Nobel Prize in Interpretive Dance — Ailey died in 1989, and the category
does not exist.

Grading needs no human: false-premise subjects are assembled from invented
syllables, so any substantive answer is necessarily fabricated; TriviaQA uses
alias substring exact-match.

---

## 3. Results

### 3.1 The anomaly rule detects, but does not beat simple baselines

AUROC on the untouched test split, hallucination vs truthful:

| detector | TriviaQA | false-premise |
|---|---|---|
| perplexity | **0.666** | 0.720 |
| mean entropy | 0.661 | 0.692 |
| Mahalanobis (19 stats) | 0.645 | **0.825** |
| OneClassSVM (hidden, 64 PC) | 0.536 | 0.818 |
| Mahalanobis (hidden, 64 PC) | 0.580 | 0.763 |
| **tribble FIS anomaly rule** | **0.643** | **0.819** |

The fuzzy rule is *competitive but consistently behind* the best simple baseline
on both families — by 0.023 on TriviaQA and 0.006 on false-premise. It is not a
win as configured.

Numbers are post-fix (submodule at `f779a42`; see §4). The false-premise 0.819
uses K=4 modes, 48 antecedents, Hamacher — a configuration that was **not
reachable before the fix**, since `norm_conorm` was silently overridden to
min/max and Hamacher NaN'd 42% of runs. Hamacher is now the best-performing norm
on both families. Widening the antecedent budget from 8 to 48 buys ~0.017, so
the feature budget is a minor constraint at most.

This is consistent with the caveat Ch 4.3.5 already concedes: per-feature
factorized Gaussians are a diagonal-covariance density estimate, strictly weaker
than the full-covariance Mahalanobis it is being compared against.

### 3.2 The pre-generation state is the best internal signal

Validation AUROC of the best configuration at each pooling site:

| pooling site | TriviaQA | false-premise |
|---|---|---|
| **`prompt`** (before any token emitted) | **0.595** | **0.769** |
| `mean` (over generated tokens) | 0.521 | 0.141 |
| `first` (first generated token) | 0.487 | 0.507 |

Selected: `prompt`, **layer 20 of 32**, 64 components, no L2.

The model's state *before it emits a single token* is the most diagnostic place
to look — better than anything measured from the text it went on to produce.
This is the practically useful direction: it supports a **pre-emptive** warning
rather than a post-hoc one.

(`mean` pooling at 0.141 on false-premise is far *below* chance — a strong
inverse signal. Confabulated answers sit systematically closer to the truthful
centroid than held-out truthful answers do. Worth chasing; likely an
answer-length/norm artifact, since false-premise answers are long and fluent.)

### 3.3 PCA/SVD: the truthful manifold is not low-rank, and does not cluster

SVD of the truthful residual stream (960 dims):

| layer | comps for 90% var | for 99% | effective rank |
|---|---|---|---|
| 0 | 320 | 629 | 194 |
| 8 | 310 | 624 | 248 |
| 16 | 265 | 581 | 197 |
| 24 | 219 | 532 | 174 |
| 32 | 252 | 575 | 212 |

There is **no compact low-rank truthful subspace**. Effective rank stays at
174–248 of 960, and 90% of variance needs 219–320 components. Truncating to 32
components — the natural choice for an interpretable rule base — discards real
signal, which is why the sweep selected the largest budget offered (k=64).

Clustering the known-good manifold for "behavioural modes" also came up empty
(best silhouette by feature set): stats **0.337**, fused **0.091**, hidden
**0.011**. Only the distribution statistics show even weak structure; the
residual stream shows essentially none. K=2 won everywhere, and larger K did not
help. **The hypothesis that truthful behaviour decomposes into discoverable
modes is not supported by this data.**

### 3.4 θ is an operating point, not a discriminative parameter

AUROC is *identical* for every θ ∈ {0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}. The
algebra explains it: for a min/max conorm,

    μ_anom = 1 − max_k(μ_k + θ) = 1 − max_k(μ_k) − θ

a constant shift, which cannot change a ranking. θ moves the precision/recall
operating point; it cannot improve separability. Sweeping θ for detection
quality is wasted effort — only the antecedents and the operators matter.

### 3.5 Norm/conorm pair sweep

11 axiom-checked families (`norms.py`), evaluated as both matched De Morgan dual
pairs and **mismatched** (T, S) combinations — 726 evaluations, **413 valid**.

Best pairs, and matched vs mismatched:

| family | TriviaQA | false-premise |
|---|---|---|
| best matched pair | 0.641 | 0.802 |
| best mismatched pair | **0.643** (T=Dombi₂, S=Hamacher) | **0.809** (T=Dombi₂, S=Einstein) |

Decoupling T from S gives a small but consistent gain — an option the library's
single `norm_conorm` field cannot express.

**Degeneration by operator** (fraction of runs yielding no usable score):

| family | as T | as S |
|---|---|---|
| drastic | 1.00 | 1.00 |
| Łukasiewicz | 0.36 | 1.00 |
| nilpotent min | 0.36 | 1.00 |
| Schweizer–Sklar₂ | 0.47 | 1.00 |
| product / Hamacher / Einstein / Dombi₂ / minimum / Yager | 0.36 | 0.11 |

Clean split: **nilpotent (non-strict) families are unusable as the outer
t-conorm.** They saturate to exactly 1 once several class firings are
aggregated, making the complement identically 0. Only strict Archimedean
families survive. This is a structural constraint on the Ch 4.3.5 construction,
not a tuning detail.

---

## 4. Four bugs found in `tribblefis` — all fixed

Filed as `fundthmcalculus/tribble-fis` issues **#22–#25**; fixed by **PR #26**
with regression tests. This repo's submodule now pins **`f779a42`** (was
`c32e896`). Independently verified after the bump — see §4.5.

These affected the existing Ch 4.3.5 / BETH results, not just this study.

**(a) `norm_conorm` is silently ignored when aggregating class firings.**
The array-reduction branches of both operators recurse without forwarding the
selected norm (`gauss_math.py` ~lines 308–312 and 330–335):

```python
if y is None:
    z = np.zeros(x.shape[0])
    for ij in range(0, x.shape[1]):
        z = t_conorm(z, x[:, ij])      # <-- selected_norm not passed
    return z
```

It therefore falls back to `DefaultNormCornorm = "min/max"`. The anomaly column
in `tsk_firing_strengths` is computed through exactly that branch, so
**`norm_conorm="hamacher"` in `beth-anomaly.py` never applied at the anomaly
step** — it silently used min/max. `simple_gaussian_predict` has the same path.
Verified: our independent re-aggregation matches the library *exactly*
(0.00e+00) for min/max and diverges for `probability`, which is the signature of
this fallback.

**(b) The Hamacher t-conorm is out of range.** It is implemented as
`(x + y) / (1 - x*y)`, which returns **1.333 at x=y=0.5** — outside [0,1], so it
is not a t-conorm. The standard Hamacher sum (λ=0) is `(x + y - 2xy) / (1 - xy)`.

**(c) The Hamacher t-norm is 0/0 when both memberships underflow.**
`xy/(x+y-xy)` → NaN, which is unavoidable once the antecedent count is large
enough that some Gaussian evaluates to exactly 0. With 83 antecedents it NaN'd
immediately; at 12 antecedents 42% of runs still degenerated. Relatedly, `μ+θ`
is never clipped back into [0,1], so θ can push memberships outside the
operators' domain.

`norms.py` provides verified replacements; all 11 families pass boundary,
range, commutativity, identity, and De Morgan checks.

### 4.5 Post-fix verification

Re-ran the original reproductions against `f779a42`:

| # | check | before | after |
|---|---|---|---|
| 22 | `t_conorm(x, None, "probability")` | 0.500 (min/max) | **0.750** ✓ |
| 22 | `t_norm(x, None, "probability")` | 0.500 (min/max) | **0.250** ✓ |
| 23 | `S_hamacher(0.5, 0.5)` | 1.333 (out of range) | **0.667** ✓ |
| 24 | `T_hamacher(0, 0)` | `nan` | **0.0** ✓ |
| 25 | `μ_anom` at θ=0.5, firing 0.8 | −0.300 | **clipped, in [0,1]** ✓ |

The decisive test is the independent-reimplementation parity check in
`norm_sweep.py`, which compares our re-aggregation against the library on the
library's own norms:

| θ | min/max | probability |
|---|---|---|
| 0.0 — before fix | 0.00e+00 | up to 2.26e-01 |
| 0.0 — after fix | 0.00e+00 | **0.00e+00** |
| 0.5 — before fix | 4.08e-01 | 2.69e-01 |
| 0.5 — after fix | **0.00e+00** | **0.00e+00** |

Exact agreement across all 12 variant × family × norm combinations at both θ
values. The θ=0 column confirms #22 (norm now forwarded); the θ=0.5 row confirms
#25 (clipping now applied on both sides).

Measured effect on results: Hamacher's degeneration rate through the library
dropped **0.42 → 0.00**, and Hamacher became the best-performing norm on both
families — an outcome that was unreachable before, because the requested norm
was being discarded. Conclusions in §3 are unchanged in direction; the
false-premise gap to the best baseline narrowed from 0.013 to 0.006.

---

## 5. Caveats — read before quoting

* **AUPRC is flattered by class imbalance.** The test split is 1,974 positives
  vs 353 negatives (base rate 0.848) for TriviaQA, and 716 vs 353 (0.670) for
  false-premise. An AUPRC of 0.913 against a 0.848 base rate is a *modest* lift,
  not the strong result it looks like. AUROC is the number to read.
* **Single model, single seed, no error bars.** Nothing here is repeated across
  seeds, and per the board-wide TODO (Goal G4) that is required before any of
  these numbers are cited.
* **TriviaQA grading is lenient substring exact-match** — the field-standard
  metric, but it credits an answer that merely echoes a title containing the
  gold string. Some label noise in the `correct` class is expected.
* **Refusals are separated from fabrications** by a hand-built regex. It was
  broadened once after it mislabeled "as a text-based AI, I don't have the
  ability to access…" as a hallucination; residual leakage is possible, and it
  would inflate results by letting the detector learn "refusal template".
* **False-premise prompts are templated**, so that family partly measures
  template novelty, not just fabrication. This is why TriviaQA (identical prompt
  distribution on both sides) is the primary result and false-premise the
  secondary one.

---

## 6. Where this leaves the hypothesis

*Can tribble flag when the LM is behaving unlike itself?* **Weakly yes, but it
is not yet better than perplexity.** The anomaly rule produces a usable,
interpretable open-set score with no hallucination examples and no second model
— but it trails a one-line perplexity baseline on the honest within-distribution
task.

The two genuinely promising threads:

1. **Layer-20 pre-generation state.** The best internal signal precedes the
   output, which is what a real warning system needs. Worth isolating properly:
   sweep pooling × layer with length/norm controls to kill the artifact
   hypothesis behind §3.2's sub-chance `mean` result.
2. **Mismatched (T, S) pairs.** A small but consistent gain, and a knob the
   current library cannot express.

The main obstacle is §3.3: with no low-rank structure and no clusterable modes,
the MoG antecedents have little to grip. Any real improvement likely has to come
from a better representation, not a better fuzzy operator.

### Reproduce

```bash
python build_prompts.py --n-trivia 6000 --n-false 1500
python capture.py --batch-size 24          # ~7 min, 2.5 GB VRAM
python analyze.py                          # SVD, representation sweep, baselines
python detect_fis.py                       # FIS grid: modes x antecedents x norm
python norm_sweep.py                       # 726 (T,S) pairs + library parity check
python norms.py                            # operator axiom certification
```
