# Fuzzy Anomaly Detection in a Frozen Small Language Model

Exploratory study: can the tribble "none of the above" anomaly rule (Ch 4.3.5)
flag hallucinated output from a small language model it was never trained on?

**Status:** single model, single seed, no error bars. Read the caveats before
quoting any number.

> ## RETRACTION — read §11 and §12 before anything else
>
> The §9 headline (**AUROC 0.906 ± 0.017**, beating every baseline 10/10 seeds)
> **does not survive the template control** and is retracted. On a
> template-matched probe set the fuzzy rule falls to **0.671 ± 0.068** while mean
> entropy reaches **0.964 ± 0.009**; and against a *different* family of
> fabrications with the same broad truthful set — the configuration that produced
> 0.906 — it sits at **chance (0.529 ± 0.004)**. The §9 advantage was reading
> prompt-family style, not fabrication. §§7–9 are kept for the record; §§10–17
> are the current state.
>
> **Headline (SUPERSEDED — was revised after §7–§8).** On the raw split the fuzzy rule *trailed*
> a perplexity baseline, and that was the original conclusion. It was wrong,
> because the baselines were partly reading **answer length** — `n_tokens` alone
> scores 0.853 on the raw false-premise split. Dropping PCA/SVD and controlling
> for length reverses the ranking: the tribble anomaly rule reaches
> **AUROC 0.896** on length-matched false-premise data while perplexity collapses
> to **0.550** (near chance). The in-distribution TriviaQA task remains
> unsolved by the fuzzy rule (~0.62).

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

> ⚠️ **Superseded for the false-premise family by §7–§8.** These are raw-split
> numbers, and `n_tokens` alone scores 0.853 on the raw false-premise split, so
> this ranking is partly a length measurement. §8 removes the confound and the
> order inverts. The TriviaQA column stands.

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

> ⚠️ **This section was written before §7–§8 and its verdict is retracted for
> the false-premise family.** Kept for the record; read §8's summary instead.

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

**What §7–§8 changed.** That last sentence was right, and acting on it is what
produced the result: the improvement came from the *representation*
(PCA-free per-layer centroid geometry), not from a better operator. Two of the
predictions above also resolved — the §3.2 `mean`-pooling artifact hypothesis was
correct (it was length), and the length control that killed it also invalidated
the baseline comparison this section rested on.

### Reproduce

```bash
python build_prompts.py --n-trivia 6000 --n-false 1500
python capture.py --batch-size 24          # ~7 min, 2.5 GB VRAM
python analyze.py                          # SVD, representation sweep, baselines
python detect_fis.py                       # FIS grid: modes x antecedents x norm
python norm_sweep.py                       # 726 (T,S) pairs + library parity check
python norms.py                            # operator axiom certification
python nopca.py                            # PCA/SVD-free representations (s7)
python length_control.py                   # length-matched control (s8)
python plot_results.py                     # figures/comparison.* and layers.*
```

### Figures

* `figures/comparison.png` — panel A: detector comparison on the raw split;
  panel B: the length control that inverts the ranking.
* `figures/layers.png` — validation AUROC by layer and pooling site.

---

## 7. Removing PCA/SVD entirely

§3.3 showed there is no low-rank truthful subspace, so truncating to 32–64
components was discarding signal on principle. PCA is also fit *unsupervised* on
the truthful split: it keeps the highest-**variance** directions, which need not
be the ones that separate truthful from confabulated. `nopca.py` drops PCA and
SVD completely.

Protocol unchanged — fit on accurate (question, answer) pairs only, flag the
wrong pairs. Four projection-free families:

| family | features | idea |
|---|---|---|
| `stats` | 19 | output-distribution statistics (already projection-free) |
| `rawdim` | 960 | raw residual-stream coordinates at layer 20 |
| `layerstat` | 165 | per-layer L2 norm, mean, std, max&#124;a&#124;, kurtosis over all 33 layers |
| `centroid` | 66 | per-layer cosine similarity + L2 distance to the truthful centroid |

Antecedents are chosen by tribble's own discriminant ranking over the discovered
modes. Each representation is scored **twice** — Mahalanobis (full covariance)
and the FIS anomaly rule — so the two failure modes can be told apart: if
Mahalanobis improves but the FIS does not, the limit is the diagonal-Gaussian
antecedent structure rather than the features.

Best AUROC per representation (raw test split):

| representation | TriviaQA FIS | TriviaQA Mahal. | False-prem. FIS | False-prem. Mahal. |
|---|---|---|---|---|
| `stats` | 0.621 | **0.696** | 0.762 | 0.865 |
| `rawdim` | 0.559 | 0.588 | 0.841 | 0.758 |
| `layerstat` | 0.557 | 0.555 | 0.670 | 0.536 |
| `centroid` | 0.521 | 0.538 | **0.881** | 0.699 |
| *reference:* best FIS **with** PCA | 0.643 | — | 0.819 | — |

Two things fall out:

* **`centroid` + FIS is the best configuration found anywhere** — 0.881 on
  false-premise with only **8 antecedents and K=2 modes**, beating the 0.825
  best baseline and the 0.819 PCA-based FIS.
* **On `centroid`, the FIS beats Mahalanobis by 0.18** (0.881 vs 0.699) — a
  reversal of every other row. Per-layer distances are strongly skewed, and the
  FIS fits a *mixture* of Gaussians per feature, whereas Mahalanobis assumes one
  joint Gaussian. This is the case where the fuzzy machinery genuinely earns its
  place rather than merely keeping up.
* On TriviaQA, raw `stats` under Mahalanobis (0.696) beats everything including
  the PCA pipeline — so PCA was hurting there too, just less interestingly.

The selected antecedents are all **late layers**: `L26_dist`, `L26_cos`,
`L25_dist`, `L27_dist`, `L25_cos`, `L28_dist`, `L27_cos`, `L30_dist`. Note this
is a *different* signal from §3.2's layer-20 finding, which came from PCA of the
prompt state.

## 8. The length confound — and why it flips the conclusion

False-premise answers are long and fluent; truthful ones are short. Measured:

| family | truthful median | hallucinated median | **AUROC of `n_tokens` alone** |
|---|---|---|---|
| TriviaQA | 21 tok | 25 tok | 0.583 |
| false-premise | 21 tok | 48 tok (the generation cap) | **0.853** |

**0.853 from answer length alone** — higher than every baseline in §3.1. So the
raw false-premise numbers are partly a length measurement, and §3.1's ranking
cannot be trusted for that family.

`length_control.py` applies **exact matching on the confounder**: for each
distinct `n_tokens` value keep k = min(#pos, #neg) from each side, so the two
classes have identical length distributions and `n_tokens` becomes exactly
uninformative. The probe returning 0.500 confirms the control works.

False-premise, 170 vs 170 matched pairs:

| detector | raw | **matched** | Δ |
|---|---|---|---|
| **FIS · centroid (PCA-free)** | 0.881 | **0.896** | **+0.015** |
| Mahalanobis · stats | 0.825 | 0.779 | −0.046 |
| perplexity | 0.720 | 0.550 | −0.170 |
| mean entropy | 0.692 | 0.523 | −0.169 |
| `n_tokens` (control) | 0.853 | 0.500 | −0.353 |

**The ranking inverts.** The confidence baselines were riding the confound and
collapse to near chance once it is removed; the fuzzy rule does not depend on it
at all and actually improves slightly. On length-matched data the tribble anomaly
rule leads the best baseline by **+0.117** and beats perplexity by **+0.346**.

This retracts the §3.1/§6 conclusion that the rule "does not beat trivial
baselines" *for the false-premise family*. The comparison was unfair to the FIS
in a direction that was not obvious until the confound was measured.

TriviaQA, 353 vs 353 matched: nothing changes much (`n_tokens` is only 0.583 raw
there), and the fuzzy rule stays at chance (0.508) while entropy/perplexity hold
~0.65. **The in-distribution task is genuinely unsolved**, and there the
distribution statistics remain the only usable signal.

### What this does and does not establish

* It **does** establish that a length-independent, interpretable, open-set
  detector built from tribble's anomaly rule beats standard confidence baselines
  at catching fabrication about non-existent entities, using 8 antecedents and
  2 rules, with no hallucination examples and no second model.
* It **does not** establish detection of ordinary factual error
  (TriviaQA ≈ chance for the fuzzy rule).
* Remaining confounds are not all controlled. Length is the one that was
  measured and removed; prompt-template novelty in the false-premise family
  (§5) is *not* controlled by length matching and remains open. The honest
  scope is "novel-entity fabrication," not "hallucination" in general.
* Single split, single seed. §8's headline rests on 170 matched pairs per class —
  small enough that a seed sweep is required before publication.

---

## 9. Seed sweep and cost (10 splits)

§8 rested on one split. `seed_sweep.py` re-draws the fit/val/test split 10 times,
re-fitting *everything* split-dependent each time — the truthful centroid, the PCA
basis, the KMeans modes, the antecedent ranking, the rule base, and the
length-matched subsample. Nothing is carried across seeds. The decision rule was
fixed in `NEXT_STEPS.md` §0 **before** the results were read.

### 9.1 False-premise — the advantage holds

AUROC, mean ± std over 10 seeds:

| detector | raw | **length-matched** |
|---|---|---|
| **FIS · centroid (PCA-free)** | 0.883 ± 0.015 | **0.906 ± 0.017** |
| OneClassSVM · centroid | 0.802 ± 0.020 | 0.789 ± 0.022 |
| Mahalanobis · stats | 0.815 ± 0.014 | 0.760 ± 0.023 |
| IsolationForest · stats | 0.767 ± 0.015 | 0.718 ± 0.022 |
| perplexity | 0.728 ± 0.013 | 0.580 ± 0.045 |
| mean entropy | 0.703 ± 0.014 | 0.559 ± 0.045 |
| `n_tokens` (control) | 0.843 ± 0.011 | 0.500 ± 0.000 |
| FIS · PCA (64 comp) | 0.615 ± 0.072 | 0.454 ± 0.089 |

Paired advantage over the best rival, per seed:

    mean Δ = +0.117 ± 0.016   (min +0.095, max +0.143)
    wins 10/10 seeds · Wilcoxon p = 0.0020

**STRONG PASS** against the pre-registered rule (Δ ≥ +0.05, ≥9/10 wins,
std ≤ 0.04). The sign never flips and the worst seed still leads by +0.095.

Three things the sweep settled that the single split could not:

1. **The best rival is now OneClassSVM on the *same* centroid features** (0.789),
   not a distribution-statistic baseline. That makes the comparison a clean
   same-representation test: identical 8 features, different density model. The
   +0.117 is therefore attributable to **the fuzzy rule itself**, not to the
   representation — the strongest form this claim can take.
2. **The PCA pipeline was almost entirely the length confound.** FIS · PCA drops
   from 0.615 raw to **0.454 ± 0.089** matched — below chance and by far the most
   variable row. §3.1's 0.819 was a length measurement wearing a fuzzy hat.
   Dropping PCA was not a marginal improvement; it was the difference between a
   real detector and an artifact.
3. **`n_tokens` = 0.500 ± 0.000 on every seed**, so the control is exact by
   construction rather than on average.

### 9.2 TriviaQA — the negative result is equally firm

| detector | raw | length-matched |
|---|---|---|
| mean entropy | 0.670 ± 0.012 | **0.673 ± 0.014** |
| perplexity | 0.670 ± 0.012 | 0.669 ± 0.016 |
| Mahalanobis · stats | 0.641 ± 0.012 | 0.640 ± 0.019 |
| FIS · PCA (64 comp) | 0.530 ± 0.020 | 0.517 ± 0.027 |
| **FIS · centroid (PCA-free)** | 0.497 ± 0.014 | **0.499 ± 0.013** |

    mean Δ = -0.174 ± 0.024, wins 0/10 seeds

Exactly chance, on every seed. **Ordinary factual error is not detectable by this
mechanism** — the scope limit in §8 is now measured rather than suspected. Note
also that entropy/perplexity are *not* length-confounded here (raw ≈ matched), so
the distribution statistics remain the honest choice for in-distribution error.

### 9.3 Training and scoring cost

Mean over seeds. `feat_ms` is one-time feature construction over all 7,500 rows;
`fit_ms` is split-dependent fitting; scoring is normalised per 1,000 samples.

| detector | feat_ms | fit_ms | **total train** | score / 1k | rules | MFs |
|---|---|---|---|---|---|---|
| perplexity / entropy / `n_tokens` | 0 | 0 | **0** | 0.3 ms | — | — |
| Mahalanobis · stats | 1.1 | 1.6 | **2.7 ms** | 0.7 ms | — | — |
| IsolationForest · stats | 1.1 | 94.3 | **95 ms** | 7.0 ms | — | — |
| OneClassSVM · centroid | 1152 | 6.0 | **1,158 ms** | 4.7 ms | — | — |
| FIS · PCA (64 comp) | 94 | 955 | **1,049 ms** | 1.4 ms | 2 | 24 |
| **FIS · centroid (PCA-free)** | 1152 | 1066 | **2,218 ms** | **1.8 ms** | **2** | **30** |

Reading:

* The fuzzy detector is the **most expensive to train** — ~2.2 s, roughly 800×
  Mahalanobis. In absolute terms this is irrelevant: it is two seconds, once,
  on CPU, against a model that took 6.7 min just to generate the probe set.
* It is **cheap to score** — 1.8 ms per 1,000 samples, ~4× faster than
  IsolationForest and ~2.6× faster than OneClassSVM, the two detectors closest to
  it in accuracy. For a warning system that runs per generation, scoring cost is
  the one that matters, and the fuzzy rule wins it.
* Half its training cost (1,152 ms of 2,218) is the centroid feature build, which
  is **shared** with OneClassSVM · centroid and is a fixed preprocessing cost over
  all 7,500 rows — ~0.15 ms per generation.
* The whole detector is **2 rules over 30 membership functions on 8 antecedents**.
  That is small enough to print, which is the interpretability claim Ch 4.3.5
  makes and §A5 of `NEXT_STEPS.md` proposes to cash in.

### 9.4 What is now established

* A length-independent, interpretable, open-set detector built from the Ch 4.3.5
  anomaly rule detects novel-entity fabrication at **AUROC 0.906 ± 0.017**,
  beating the best rival on identical features by **+0.117, 10/10 seeds**, with
  no hallucination examples, no second model, no gradient, and ~2 s of training.
* It does **not** detect ordinary factual error (0.499 ± 0.013).
* Prompt-template novelty remains uncontrolled and is now the single largest
  threat to validity — see `NEXT_STEPS.md` §A1, which is blocking.

---

## 10. Template-matched probe set (v2)

`build_prompts_v2.py`. The v1 design had fabricated questions templated and the
truthful comparison set untemplated, so a detector could separate them on surface
form. v2 removes that at the source: every fabricated question has a **real-entity
twin in the identical surface form**, from a curated table of checkable facts.

| family | n | role |
|---|---|---|
| `triviaqa` | 17,944 | in-distribution control (5,286 correct) |
| `template_real` | 846 | known-good, gradeable (632 correct) |
| `template_fake` | 5,000 | necessarily fabricated (4,114 fabrications) |
| `falsepremise` | 1,500 | the v1 probes, kept for continuity |

Five templates (capital / chemical symbol / novel author / currency / film
director) x 3 phrasings, applied identically to both sides, over ~200 curated
facts and index-addressed invented entities. 25,290 generations, 22 min, 4.8 GB
of activations.

Two construction bugs worth recording: rejection sampling from a hand-written
entity pool silently delivered 411 of 5,000 fakes (fixed by combinatorial
index-addressing), and short syllable joins produced **real** place names --
"Braz"+"or"+"ia" = Brazoria, an actual Texas county -- which would break the
"necessarily fabricated" label. Infixes were lengthened until unmistakably
synthetic.

## 11. The template control falsifies section 9

`template_control.py`. Fit on `template_real` correct; test truthful = held-out
`template_real` correct; positives = `template_fake`. Ten seeds:

| detector | raw | length-matched | **length+template** |
|---|---|---|---|
| mean entropy | 0.957 +/- 0.004 | 0.938 +/- 0.012 | **0.965 +/- 0.007** |
| Mahalanobis - stats | 0.952 +/- 0.004 | 0.924 +/- 0.013 | 0.952 +/- 0.012 |
| perplexity | 0.944 +/- 0.006 | 0.901 +/- 0.016 | 0.938 +/- 0.010 |
| **FIS - centroid** | 0.714 +/- 0.080 | 0.696 +/- 0.056 | **0.668 +/- 0.078** |
| OneClassSVM - centroid | 0.625 +/- 0.095 | 0.611 +/- 0.090 | 0.600 +/- 0.095 |
| `n_tokens` (control) | 0.884 +/- 0.010 | 0.500 +/- 0.000 | 0.500 +/- 0.000 |

    paired delta (FIS - mean entropy) = -0.297 +/- 0.079, wins 0/10, p = 0.0020

**The ranking inverts back, decisively.** Where truthful and fabricated share a
template, the confidence baselines are near-ceiling and the fuzzy rule is far
behind.

## 12. Decomposition -- the template, or the fit set?

Section 11 changed two things at once: the template *and* the truthful
distribution. `decompose_confound.py` separates them, holding the fabrications
fixed and length-matching throughout:

| detector | TriviaQA truthful (n=5,286) | subsampled to 632 | template-matched (n=632) |
|---|---|---|---|
| mean entropy | 0.872 +/- 0.004 | 0.862 +/- 0.011 | 0.940 +/- 0.012 |
| Mahalanobis - stats | 0.806 +/- 0.008 | 0.789 +/- 0.022 | 0.923 +/- 0.010 |
| **FIS - centroid** | **0.529 +/- 0.004** | **0.500 +/- 0.053** | 0.685 +/- 0.063 |
| `n_tokens` | 0.500 | 0.500 | 0.500 |

    cost of shrinking the fit set : -0.028
    cost of matching the template : +0.184

The verdict is worse than "the template explains it". **Against `template_fake`
fabrications with the broad TriviaQA truthful set -- the exact configuration that
produced 0.906 in section 9 -- the fuzzy rule is at chance (0.529).** Fit-set size
is not the cause.

The section 9 result was therefore specific to the v1 `falsepremise` family and
did not transfer to a different family of fabrications, even with the same
truthful set and with length matched. What it had learned was the prompt/answer
*family* -- long, discursive probes -- not fabrication. That is exactly the
confound `NEXT_STEPS.md` A1 flagged as blocking.

## 13. Which statistic should rank the antecedents?

`ranker_compare.py`. Each of the four statistics inside
`calculate_gaussian_correlation` reimplemented separately, plus four
non-parametric alternatives, scored by downstream AUROC (6 seeds, top-8):

| ranker | AUROC | note |
|---|---|---|
| bhattacharyya | **0.673 +/- 0.081** | best |
| overlap | 0.673 +/- 0.081 | |
| jensen_shannon | 0.672 +/- 0.083 | |
| wasserstein | 0.667 +/- 0.089 | not currently computed |
| **blend (library)** | 0.658 +/- 0.088 | worse than 3 of its own 4 terms |
| variance / ks | 0.653 | |
| auc | 0.647 | |
| mutual_info | 0.644 | |
| **hist_corr** | 0.632 +/- 0.042 | worst -- and it is 1/4 of the blend |

The spread is small against a seed std of ~0.08, so **the ranker is not the
bottleneck**. But the blend is beaten by three of its own components, and its
weakest term is also the one that crashes on constant features and is scaled to
[0,2] while the others are [0,1]. Filed as tribble-fis **#30** with a `method=`
proposal.

Separately: layer 0 of the `prompt` pooling site is **constant by construction**
-- the last prompt token is always the same chat-template token, so `L00_dist` is
identically 0. Harmless as signal, but it crashes the post-`f779a42`
`calculate_gaussian_correlation`; `drop_constant()` now guards it.

## 14. Standing report -- accuracy, parameters, train time, inference

`pareto.py`. Every comparison from here reports all four. Template-matched task,
6 seeds:

| detector | AUROC | FPR@95 | params | train | inference | structure |
|---|---|---|---|---|---|---|
| mean entropy | **0.964 +/- 0.009** | 0.203 | **0** | **0 ms** | 1,473,552/s | threshold only |
| Mahalanobis - stats | 0.948 +/- 0.009 | 0.317 | 209 | 4 ms | 544,022/s | 19 feat, full cov |
| perplexity | 0.934 +/- 0.009 | 0.291 | 0 | 0 ms | **1,738,387/s** | threshold only |
| IsolationForest - stats | 0.923 +/- 0.016 | 0.439 | 12,092 | 101 ms | 73,484/s | 100 trees |
| **FIS - centroid** | 0.671 +/- 0.068 | 1.000 | **53** | 4,880 ms | 241,733/s | **2 rules, 29 MFs** |
| OneClassSVM - centroid | 0.619 +/- 0.125 | 0.776 | 398 | 3,894 ms | 266,620/s | 44 SV x 8 dims |

### Pareto verdict -- negative, as measured

**Mean entropy dominates every other detector on every cost axis**: highest
AUROC, zero fitted parameters, zero training time. The fuzzy rule is dominated on
parameters, training time and inference speed simultaneously. On this task **the
FIS is not on the Pareto front**, and no arrangement of these numbers makes it so.

What is true, and is the most that can be claimed:

* Among **learned** detectors the FIS is the most parsimonious -- 53 continuous
  parameters against 209 for Mahalanobis and 12,092 for IsolationForest, i.e.
  4x and 228x fewer.
* Within the **hidden-state-geometry** sub-family it beats OneClassSVM on
  identical features (0.671 vs 0.619) with 7.5x fewer parameters, so it is on the
  front *of that sub-family*.
* It is the only detector here that yields a readable rule base (section 15).
* But a zero-parameter output-distribution threshold beats all of them, so that
  sub-family is the wrong place to be for this task.

Training cost is dominated by the centroid feature build (~3.9 s of 4.9 s) over
all 25,290 rows -- about 0.19 ms per generation amortised.

## 15. The rule base is readable (the one surviving differentiator)

`print_rule.py` prints the fitted FIS and plots its membership functions
(`figures/membership_functions.png`). On the v1 configuration it recovered two
interpretable known-good modes:

    RULE 1 (mode0)  IF L26_dist is HIGH AND L26_cos is LOW ...         THEN normal
    RULE 2 (mode1)  IF L26_dist is LOW/MED AND L26_cos is HIGH/MED ... THEN normal
    ANOMALY         IF neither fires                                    THEN flag

That is a real finding independent of the accuracy result: known-good behaviour
has **two** modes, one diffuse and far from the centroid, which is why single-blob
detectors underperform the FIS *within the centroid representation*. It does not
rescue the headline, because the representation itself is beaten by entropy.

`operating_points_*.csv` reports precision/recall at fixed warning rates, the
correct use of theta given section 3.4.

## 16. Abstention regex audit

`audit_abstain.py` cross-checks the refusal regex against an independent
structural heuristic and surfaces only the disagreements.

    agreement 99.71%   kappa 0.968   22 of 7,500 disagree (0.29%)

Reading all 22: the regex is right in the large majority, and the identified
errors are fabrications *mislabelled as abstentions* (an invented researcher's
"research was abandoned ... they were not able to find a suitable collaborator"
matches "not able to"). That is the **conservative** direction -- it removes true
positives rather than adding them -- so label leakage is not inflating anything.
Error budget bounded at 0.29%.

## 17. Where this actually stands

Established:

* The adversarial elicitation works: 95.4% fabrication rate on non-existent
  entities, from a frozen 360M model.
* Output-distribution statistics are a strong, cheap, length-robust and
  template-robust signal for this task: **0.964 AUROC, zero parameters, zero
  training time**.
* The fuzzy anomaly rule as constructed does **not** beat them, is not on the
  Pareto front, and is at chance once the prompt-family confound is removed.
* Four tribblefis defects found and fixed (#22-#25); a fifth filed (#30).
* **The methodological result stands on its own and is the durable contribution:
  two confounds -- answer length and prompt family -- each independently produce
  a large, entirely spurious "hallucination detection" result (0.843 and ~0.9
  AUROC).** Any work in this area that does not control both is untrustworthy,
  and this holds regardless of the detector used.

Not established, and not worth asserting: any advantage for the FIS on this task.

### TODO -- proposal defense

Not yet written into `research/proposal-defense/`. When it is, the honest framing
is a **methods/negative-results contribution** (the two confounds and the controls
that expose them), not a detection win. It also still closes Ch 4.3.5's owed
head-to-head against one-class SVM and isolation forest, in a second domain --
that part is unaffected by the retraction.

---

## 18. Expanded variable families -- and a signal that survives every control

`features_ext.py` + `correlate.py`. Everything is derived from activations
already on disk; no recapture. Six new families beyond the two used so far:

| family | n | what it measures |
|---|---|---|
| `centroid` | 61 | per-layer distance + cosine to the truthful centroid (as before) |
| `delta` | 64 | the **update** each layer applies: ‖h[L+1]−h[L]‖, cos(h[L+1],h[L]) |
| `deltaref` | 63 | how unusual that update is vs the **mean truthful update** |
| `geom` | 64 | per-layer norm and consecutive norm ratio |
| `curve` | 31 | curvature (2nd difference) of the depth profile |
| `agg` | 29 | **aggregate shape** of a profile: slope, argmax, early/late contrast, monotonicity, roughness |
| `stats` | 19 | the output-distribution statistics |

`agg` is the conceptually interesting one: it collapses 33 per-layer numbers into
a handful of shape descriptors ("the distance profile rises late and peaks near
the output"), which is the kind of variable a fuzzy rule can actually talk about.

### Screening method -- and one method rejected

Entropy is at 0.968 on this task, so raw correlation is uninformative; the
question is what separates fabrication **among generations entropy scores the
same**. Conditioning is done by *matching* (as for length and template), adding
entropy quartile to the match keys.

> **A rejected method, recorded because the failure mode is instructive.** The
> first version residualised each variable on (entropy, n_tokens) with a linear
> model fit on the truthful split, then scored the residual. That is invalid:
> the nuisance model never sees high-entropy fabrications, so it extrapolates
> badly on them and the residual *re-encodes* entropy through its own prediction
> error. The tell was unmissable — residualising **raised** `N03_ratio` from
> 0.632 to 0.958. Matching conditions without extrapolating.

Entropy overlap is thin by construction: truthful entropy median 0.697,
fabricated 1.585, with only 36 of 135 fabrications inside the shared range. After
matching on (template, n_tokens, entropy quartile) just **12–21 pairs per class**
remain, so a permutation null is mandatory.

### Result: late-layer update geometry, p < 0.005

| family | best raw AUROC | best entropy-matched |
|---|---|---|
| **`deltaref`** | 0.900 | **0.942** |
| `delta` | 0.923 | 0.934 |
| `centroid` | 0.857 | 0.918 |
| `geom` | 0.867 | 0.886 |
| `curve` | 0.846 | 0.876 |
| `agg` | 0.834 | 0.873 |
| `stats` | **0.968** | 0.846 |

Top variables, entropy-matched: `R30_cos` 0.942, `D31_cos` 0.934, `R30_dist`
0.928, `L32_cos` 0.918, `L32_dist` 0.916, `L31_cos` 0.902. **All from layers
29–32 of 32** — the last few layers.

**Permutation null** (200 label shuffles, max over all 331 variables, same
matched n):

    median 0.751   95th percentile 0.827   maximum 0.879
    observed best 0.942  ->  p < 0.005

The observed maximum exceeds all 200 permutations. So this is **not** selection
noise, and it survives template matching, length matching, entropy matching, and
multiplicity correction simultaneously — every control that killed the §9 result.

The null also fixes a read-off threshold: **entropy-matched values below ~0.83
are indistinguishable from selection noise** at this sample size. About nine
variables clear it, all late-layer, all from `deltaref` / `delta` / `centroid`.

### What this does and does not mean

* It **does** mean there is genuine information about fabrication in the
  late-layer *update* geometry that mean entropy does not already carry. The
  best family — `deltaref`, how far each layer's update deviates from the mean
  truthful update — is new and was not examined before §18.
* It **does not** resurrect §9, and it is not yet a detector. n = 12–21 matched
  pairs per class is far too small to fit anything, and the supervised ceiling
  (nested selection, 2-fold) moves entropy 0.974 → 0.987 with n = 270 against
  331 candidates, i.e. n < p. That number is an optimistic bound, not a result.
* The binding constraint is now **matched sample size**, and its cause is
  specific: only 632 of 846 `template_real` questions are answered correctly, so
  the truthful side of the matched comparison is small. Expanding the curated
  fact tables (capitals and elements are the cheap ones to grow) is the direct
  fix and is worth more than any modelling change.

### Next, in priority order

1. **Grow the curated fact tables** to ~1,000 real instances so the
   entropy-matched cells hold hundreds rather than tens of pairs. Everything
   below is underpowered until this is done.
2. **Re-test the late-layer `deltaref` hypothesis** pre-registered on the larger
   set, with the permutation null as the acceptance criterion.
3. **Then, and only then**, ask whether a fuzzy rule over the `agg` shape
   descriptors of `deltaref` beats an entropy threshold — that is the version of
   the original hypothesis that the current evidence actually supports testing,
   and `agg` is small and legible enough to keep the interpretability claim.
4. Attention-head and MLP-sparsity families still require a fresh capture with
   hooks; deferred until (1) makes any result from them interpretable.
