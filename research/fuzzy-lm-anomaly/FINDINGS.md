# Fuzzy Anomaly Detection in a Frozen Small Language Model

Exploratory study: can the tribble "none of the above" anomaly rule (Ch 4.3.5)
flag hallucinated output from a small language model it was never trained on?

**Status:** single model, single seed, no error bars. Read the caveats before
quoting any number.

> ## RETRACTION 2 — the fuzzy positive result does not stand (§26)
>
> §§21/23/24 claimed a 4-rule fuzzy system beats full-covariance Mahalanobis on
> identical features. **It does not.** That comparison gave the fuzzy rule a
> 120-candidate supervised configuration search on labelled validation positives
> and gave its rivals none. With a fixed configuration the sign flips
> (−0.019, 1/8 seeds) and the fuzzy rule is beaten by both Mahalanobis and a
> zero-parameter entropy threshold. The confound/controls work (§§8, 11, 12, 20)
> is unaffected and remains the durable contribution.
>
> ## RETRACTION 1 — read §11 and §12 before anything else
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

---

## 19. One detector per class of error -- and the transfer matrix

`per_class.py`. Under the open-set protocol nothing about the rule base is
class-specific, so a "specialist" is made by letting each error class choose
*which variables the rule watches*: antecedents are selected on a validation
slice of that class (supervised selection) while the rule base is still fit on
truthful data only (open-set fitting). Seven classes, a pooled generalist, and
mean entropy as reference. Common truthful set (TriviaQA-correct), length matched
throughout, 4 seeds.

Transfer matrix, specialist (row) evaluated on class (column):

| specialist | capital | symbol | novel | currency | film | falseprem | tqa_err |
|---|---|---|---|---|---|---|---|
| **fake:capital** | **0.835** | 0.326 | 0.910 | 0.656 | 0.963 | 0.805 | 0.556 |
| **fake:symbol** | 0.515 | **0.908** | 0.430 | 0.724 | 0.582 | 0.662 | 0.529 |
| **fake:novel** | 0.403 | 0.366 | **0.949** | 0.410 | 0.799 | 0.765 | 0.506 |
| **fake:currency** | 0.708 | 0.632 | 0.802 | **0.851** | 0.830 | 0.645 | 0.533 |
| **fake:film** | 0.717 | 0.489 | 0.926 | 0.644 | **0.972** | 0.710 | 0.570 |
| **falsepremise** | 0.602 | 0.503 | 0.823 | 0.602 | 0.933 | **0.786** | 0.567 |
| **triviaqa_error** | 0.635 | 0.566 | 0.830 | 0.564 | 0.949 | 0.763 | **0.606** |
| *GENERALIST* | 0.637 | 0.560 | 0.827 | 0.570 | 0.948 | 0.762 | 0.605 |
| *mean entropy* | 0.886 | 0.753 | 0.925 | 0.822 | 0.966 | **0.561** | 0.675 |

    FIS specialists: diagonal mean 0.844, off-diagonal mean 0.654
                     transfer gap +0.190
    FIS generalist  : 0.701 mean over classes
    mean entropy    : 0.798 mean over classes

### Four things this settles

**1. Per-class detectors are empirically justified.** The +0.190 transfer gap is
large, and the variables each specialist chose differ *systematically by class*:

| specialist | top variables chosen |
|---|---|
| fake:capital | `L32_dist`, `C21_curv`, `L32_cos` -- geometry |
| fake:symbol | `N08_ratio`, `N26_ratio`, `N26_norm` -- norm ratios |
| fake:novel | `L31_cos`, `R31_cos`, `L30_cos` -- late-layer geometry |
| fake:currency | `D25_norm`, `N26_ratio`, `R30_cos` -- update geometry |
| fake:film | `ent_std`, `logp_std`, `maxp_min` -- **distribution stats** |
| falsepremise | `ent_max`, `ent_std`, `maxp_min` -- **distribution stats** |
| triviaqa_error | `ent_std`, `ent_max`, `logp_std` -- **distribution stats** |

The four short-factual classes select *hidden-state geometry*; the three
long-form classes select *output-distribution statistics*. Different kinds of
error really are visible in different variable families. That is a positive
result and it directly answers the question.

**2. Specialists can be worse than useless off-class.** `fake:capital` scores
0.326 on `fake:symbol`; `fake:novel` scores 0.366 on `fake:symbol` and 0.403 on
`fake:capital`. Below chance means the anomaly score is *inverted* -- a detector
tuned for one kind of fabrication would systematically rank the wrong outputs as
suspect on another. This is a practical hazard, not just a weak result.

**3. The generalist is not a compromise, it is a collapse.** Pooling all classes
for selection produced the same variables as the `triviaqa_error` specialist
(`ent_std`, `ent_max`, ...) and the same row of scores (0.637/0.560/0.827/... vs
0.635/0.566/0.830/...). The largest class simply dominates selection, so the
"generalist" is the TriviaQA specialist wearing a different name -- and at 0.701
it is beaten by plain entropy at 0.798.

**4. There is exactly one class where the fuzzy rule beats entropy.**
`falsepremise`: FIS specialist **0.786** vs mean entropy **0.561**. Entropy is
*weakest* precisely on the long-form false-premise family, which is the family
the original §9 result was built on. So the §9 intuition was not baseless -- it
was over-generalised. Properly scoped, the surviving claim is narrow: *for
long-form false-premise fabrication, where the model is fluent and confident and
entropy therefore fails, late-layer geometry carries signal that entropy does
not.* That is a real niche, and it is the one worth pursuing.

Figures: `figures/negative_results.png` (the falsification) and
`figures/transfer_matrix.png` (this matrix, diverging palette centred on chance
so the below-chance cells are visible as such).

---

## 20. The false-premise niche, properly controlled -- it also dissolves

Section 19's one apparent win (FIS 0.786 vs entropy 0.561 on `falsepremise`) used
TriviaQA-correct as the truthful set, so it carried the same confound that
falsified section 9: long discursive fabrications against short factual answers,
with only length matched. `build_prompts_v3.py` builds the missing control --
every long-form template gets a **real-subject twin in the identical surface
form**, so both sides are fluent and discursive.

    real : "Explain how the quicksort algorithm works."
    fake : "Explain how the Drennick-partition algorithm works."

Seven templates x 3 phrasings, 471 real / 1,640 invented subjects, 96-token
generations, 2,111 prompts in 3 min 46 s at 3.41 GB. Labels are **groundedness**,
not correctness: a paragraph cannot be graded reliably, but a real subject anchors
the answer and an invented one cannot. The model fabricated on **1,627 of 1,640
(99.2%)** invented long-form subjects -- only 13 pushbacks.

### Stacked controls, 8 seeds

| detector | raw | + length | **+ template** | + entropy |
|---|---|---|---|---|
| Mahalanobis - stats | 0.851 | 0.852 | **0.847** | **0.799** |
| mean entropy | 0.875 | 0.855 | 0.841 | 0.642 |
| perplexity | 0.852 | 0.830 | 0.826 | 0.624 |
| **FIS - agg** (best fuzzy) | 0.608 | 0.611 | **0.666** | 0.657 |
| FIS - centroid | 0.571 | 0.579 | 0.626 | 0.625 |
| FIS - deltaref | 0.578 | 0.575 | 0.614 | 0.617 |
| OneClassSVM - deltaref | 0.615 | 0.623 | 0.580 | 0.612 |
| `n_tokens` (control) | 0.464 | 0.500 | 0.500 | 0.500 |

Under template matching the best fuzzy family reaches **0.666** against **0.847**
for the distribution statistics. Under the hardest condition the gap narrows only
because entropy is being matched away, and the paired test is **not significant**:

    FIS - agg vs mean entropy, entropy-matched:
      mean delta = +0.015 +/- 0.037, wins 6/8 seeds, Wilcoxon p = 0.3125

Per subtype (template-matched), entropy beats the best fuzzy family on **every
one**: algorithm -0.280, researcher -0.288, conjecture -0.198, effect -0.099,
element -0.018.

**So the section 19 niche was the same confound a third time.** The pattern is now
the most reliable finding in this study: *prompt-family style is a very strong
confound in hallucination detection, and any apparent advantage for a
hidden-state detector should be assumed to be style until a template-matched
control says otherwise.* Three separate claimed advantages (section 9, section 19,
and the raw column above) each dissolved under exactly that control.

### One genuinely new positive result

**Mahalanobis over the 19 output-distribution statistics holds 0.799 even with
entropy matched**, where mean entropy itself falls to 0.642 by construction. So
there *is* substantial signal beyond mean entropy -- it lives in the **other
output statistics** (margin, max-prob, log-prob spread, their min/max/std), not in
hidden-state geometry. That is directly actionable and cheap: 19 features, 209
parameters, 1 ms to fit, 749k samples/s.

### Standing report -- condition `length+template+entropy`

| detector | AUROC | FPR@95 | params | train | inference | structure |
|---|---|---|---|---|---|---|
| **Mahalanobis - stats** | **0.799 +/- 0.023** | 0.653 | 209 | **1 ms** | 749,343/s | 19 features |
| FIS - agg | 0.657 +/- 0.028 | 1.000 | **42** | 720 ms | 418,569/s | 2 rules / 24 MFs |
| mean entropy | 0.642 +/- 0.014 | 0.738 | **0** | **0 ms** | **2,025,765/s** | threshold |
| FIS - centroid | 0.625 +/- 0.032 | 1.000 | 53 | 923 ms | 317,059/s | 2 rules / 27 MFs |
| FIS - deltaref | 0.617 +/- 0.019 | 1.000 | 62 | 767 ms | 353,437/s | 2 rules / 31 MFs |
| OneClassSVM - deltaref | 0.612 +/- 0.026 | 0.788 | 352 | 2 ms | 330,748/s | 37 SV |

Note **FPR@95TPR = 1.000 for every fuzzy variant**: to catch 95% of fabrications
they flag essentially everything. Even where AUROC looks respectable the operating
characteristics are unusable, which the AUROC column alone hides -- which is why
the standing report carries FPR@95.

`figures/falsepremise_control.png`.

### Consequence for the plan

`NEXT_STEPS.md` A2/A3 (second model, scaling) are **not worth running**: they
would test whether a chance-to-mediocre result generalises. The honest remaining
directions are:

1. **Follow the output statistics, not the activations.** Mahalanobis over the 19
   stats is the only thing that has survived every control. Ask which of the 19
   carry the entropy-independent signal, and whether a *small fuzzy rule over
   those* is competitive -- that keeps the interpretability claim while betting on
   the representation that actually works.
2. **Stop treating "hallucination" as one target.** Section 19's transfer matrix
   (+0.190 gap, below-chance off-diagonals) says these are different phenomena;
   pick one narrow, well-controlled class and characterise it properly.
3. The **methodological contribution is the paper here**: three confounds --
   length, prompt family, and entropy -- each independently manufacture large
   spurious results, and matching is a cheap general remedy. That is worth
   writing up on its own, and it is what this study actually established.

---

## 21. The fuzzy result that stands: a small rule over the OUTPUT statistics

Section 20 established that the one representation surviving every control is the
19 output-distribution statistics -- Mahalanobis over them held 0.799 with entropy
matched, while every hidden-state family failed. So the question changed from
"can fuzzy beat entropy on activations" (no, repeatedly) to the sharper one:

> Can a small, readable fuzzy rule over the output statistics match the
> full-covariance detector on that same representation?

**Yes, and it beats it.** `fuzzy_stats.py`. Hyperparameters (antecedent count,
mode count, norm pair, whitening) selected on a **validation half of the
positives** and reported on the disjoint test half, so the headline is not the
maximum of a grid. Rule bases still fit on grounded data only. 8 seeds.

### Condition `length + template` matched

| detector | AUROC | FPR@95 | params | train | inference | structure |
|---|---|---|---|---|---|---|
| **FIS - stats** | **0.881 +/- 0.015** | 0.588 | **80** | 1,102 ms | 375,066/s | **4 rules / 39 MFs / 6 antecedents** |
| mean entropy | 0.841 +/- 0.014 | 0.593 | 0 | 0 ms | 1,962,600/s | threshold |
| Mahalanobis - stats | 0.841 +/- 0.024 | 0.588 | 209 | 1 ms | 722,354/s | 19f full covariance |
| IsolationForest - stats | 0.831 +/- 0.027 | 0.637 | 11,649 | 94 ms | 89,297/s | 100 trees |
| perplexity | 0.828 +/- 0.014 | 0.587 | 0 | 0 ms | 2,266,532/s | threshold |
| OneClassSVM - stats | 0.785 +/- 0.040 | 0.818 | 928 | 1 ms | 293,906/s | 49 SV |
| `n_tokens` (control) | 0.500 +/- 0.000 | 0.950 | 0 | 0 ms | 2,508,771/s | threshold |

    paired FIS vs Mahalanobis (identical 19 features):
      mean delta = +0.040 +/- 0.017 (min +0.014, max +0.068)
      wins 8/8 seeds, Wilcoxon p = 0.0078, 2.6x fewer parameters

### Condition `length + template + entropy` matched (hardest)

| detector | AUROC | FPR@95 | params |
|---|---|---|---|
| **FIS - stats** | **0.794 +/- 0.022** | 0.955 | **80** |
| Mahalanobis - stats | 0.760 +/- 0.032 | 0.780 | 209 |
| IsolationForest - stats | 0.691 +/- 0.030 | 0.892 | 11,649 |
| OneClassSVM - stats | 0.671 +/- 0.039 | 0.913 | 928 |
| mean entropy | 0.601 +/- 0.016 | 0.751 | 0 |
| perplexity | 0.580 +/- 0.018 | 0.750 | 0 |

    paired delta = +0.034 +/- 0.033, wins 7/8 seeds, p = 0.0234

### Pareto position

On (AUROC, parameters) the front has exactly **two** points: mean entropy (0 params,
0.841) and **FIS - stats** (80 params, 0.881). Mahalanobis, IsolationForest and
OneClassSVM are all dominated -- Mahalanobis by entropy (equal AUROC, fewer
parameters) and by the FIS (higher AUROC, fewer parameters). Under entropy
matching the FIS is the single best detector outright. `figures/fuzzy_stats_pareto.png`.

### The rule base

Four rules over six antecedents, all readable:

    RULE 2 (mode1)  IF maxp_first is MEDIUM or LOW  AND ent_first is MEDIUM
                    AND margin_first is MEDIUM or LOW  AND ent_max is HIGH
                    AND ent_std is HIGH  AND n_tokens is LOW or VERY LOW
                    THEN the model is behaving normally

    RULE 4 (mode3)  IF maxp_first is VERY LOW or LOW  AND ent_first is VERY HIGH
                    AND margin_first is LOW  AND ent_max is HIGH
                    AND ent_std is VERY HIGH or HIGH  AND n_tokens is VERY LOW
                    THEN the model is behaving normally

    ANOMALY         IF none of the four fires strongly THEN flag as suspect

The selected antecedents are dominated by **first-token** statistics
(`maxp_first`, `ent_first`, `margin_first`) plus entropy spread (`ent_max`,
`ent_std`) and length. Interpretation: grounded generation has a characteristic
*confidence profile on its very first token*, and the rule base carves that into
four distinct known-good modes. Mean entropy collapses all of that into one
number, which is precisely why a 4-rule system beats it by +0.040 and beats it by
+0.193 once mean entropy is matched away.

Whitening was offered as a variant and **never selected** on any seed, so no
decorrelation is needed -- the raw statistics work directly. Selected
configurations were stable: 4-8 antecedents, K = 3-4 modes, min/max or
Dombi-Einstein pairs. `figures/membership_functions_stats.png`.

### Honest limits

* **FPR@95TPR = 0.955 in the entropy-matched condition** -- worse than
  Mahalanobis's 0.780. The FIS wins on ranking but its high-recall tail is poor,
  so at a 95%-recall operating point it is not the right choice. In the
  template-matched condition the two are tied at 0.588.
* Training is ~1.1 s against 1 ms for Mahalanobis. Irrelevant in absolute terms,
  but it is not free.
* One model, one probe family (long-form groundedness), 8 seeds. The claim is
  scoped to *detecting ungrounded generation about non-existent entities*, which
  is what the v3 labels support.
* This does not revive sections 9, 19 or 20. Hidden-state geometry remains a
  negative result; the win is on the output statistics.

### Why this is the version worth putting in the dissertation

It is the same Ch 4.3.5 construction -- MoG antecedents, the complement-of-conorm
anomaly rule -- applied to a representation chosen because it survived adversarial
controls rather than because it was convenient. It beats every learned rival on
identical features, uses 2.6x-146x fewer parameters, and prints as four rules a
human can read. That is a defensible interpretability-plus-parsimony claim, and it
took four failed configurations and three dissolved confounds to find the place
where it is true.

---

## 22. Against tribble-fis main: metric, membership function, and norm pair

Submodule moved to **`3e49376`** (was `f779a42`), picking up five commits, two of
them direct responses to what this study filed:

* **#34** (fixes our #30) -- `calculate_gaussian_correlation(X, y, method=...)`.
  The hard-coded blend is **gone**; `bhattacharyya` is the new default and
  `wasserstein` was added, both chosen on the numbers section 13 reported.
  `hist_corr` was removed outright.
* **#32** -- an `einstein` family plus `resolve_norm_pair` /
  `AnomalyParameters(t_norm=, t_conorm=, allow_mixed_norms=)`. Mixed-family pairs
  are now **gated behind an explicit opt-in**, on the correct grounds that they are
  not De Morgan duals and the anomaly rule's complement `1 - S(...)` depends on
  that duality for its meaning.
* `create_trapz_membership_dict` -- a trapezoid analogue of the Gaussian builder
  returning the same container, so the anomaly rule accepts it directly.
* Also `ruspini.py` (triangular Ruspini partitions) and `#28`/`#29` on
  `pin_extremes`.

`norms.py` and the hand-rolled ranker comparison in `ranker_compare.py` are now
partly redundant -- the library covers Bhattacharyya, Wasserstein, Einstein, and
decoupled pairs natively. They are kept because they cover families the library
still lacks (Dombi, Yager, Schweizer-Sklar, drastic, nilpotent) and because
`norm_sweep.py`'s parity check remains the regression test that caught #22.

### First: section 21 had to be re-verified, and it moved

Section 21 was produced when the default ranker was the blend. The default is now
`bhattacharyya`, so those numbers do not carry over. Re-run, 8 seeds, same
protocol (validation-selected configuration, disjoint test half):

| condition | detector | AUROC | FPR@95 | params |
|---|---|---|---|---|
| length+template | **FIS - stats** | **0.872 +/- 0.022** | **0.554** | 94 |
| | mean entropy | 0.841 +/- 0.014 | 0.593 | 0 |
| | Mahalanobis - stats | 0.841 +/- 0.024 | 0.588 | 209 |
| +entropy | **FIS - stats** | **0.784 +/- 0.039** | 0.903 | 94 |
| | Mahalanobis - stats | 0.760 +/- 0.032 | 0.780 | 209 |

    paired FIS vs Mahalanobis, length+template:
      was  +0.040 +/- 0.017, 8/8 seeds, p = 0.0078   (blend)
      now  +0.030 +/- 0.023, 7/8 seeds, p = 0.0156   (bhattacharyya)

    paired FIS vs Mahalanobis, +entropy:
      was  +0.034 +/- 0.043, 7/8 seeds, p = 0.0234
      now  +0.024 +/- 0.043, 6/8 seeds, p = 0.1484   <- NO LONGER SIGNIFICANT

**The template-matched win survives; the hardest-condition win does not.** Under
entropy matching the FIS is still directionally ahead (+0.024) but 6/8 seeds and
p = 0.148 is not a claim. Section 21's text is superseded by these numbers. This is
the second time a committed number moved under a library change, which is a
argument for keeping the parity/regression checks wired in.

### The factorial

`compare_variants.py`. Honest task (v3 long-form, template matched), 19 output
statistics, 6 antecedents, K=4, no whitening -- so only the three factors vary.
2 metrics x 2 membership functions x 8 norm pairs (5 De Morgan + 3 explicitly
mixed), 8 seeds.

**Main effects, condition `length+template`:**

| factor | levels | mean AUROC | paired test |
|---|---|---|---|
| **membership function** | gaussian **0.850** vs trapezoid **0.588** | | **+0.262 +/- 0.097, 8/8, p = 0.0078** |
| coefficient metric | wasserstein 0.726 vs bhattacharyya 0.711 | | -0.015 +/- 0.048, 2/8, p = 0.875 |
| norm pair | mixed 0.720 vs De Morgan 0.718 | | +0.002 +/- 0.001, 8/8, p = 0.0078 |

Three things follow, and they are not the ones I would have predicted:

**1. The membership function dominates everything else, by an order of
magnitude.** Gaussian beats trapezoid by **+0.262** on every seed. Nothing else in
this study -- metric, norm, antecedent count, mode count -- has come close to that
effect size. Trapezoids collapse to 0.556-0.588, barely above chance in the
entropy-matched condition. The flat top is the likely cause: a trapezoid assigns
membership exactly 1.0 across an interval, so the conjunction cannot distinguish
"comfortably inside the known-good region" from "at its edge", and the anomaly
complement loses precisely the gradation it needs. If any single knob deserves
attention in the dissertation, it is this one.

**2. The coefficient metric barely matters on average -- but Wasserstein is much
more stable.** The means are within noise (p = 0.875), yet in the hardest
condition Wasserstein + Gaussian gives **0.797 +/- 0.025** against Bhattacharyya's
**0.776 +/- 0.067** -- a 2.7x smaller standard deviation. For a dissertation
claim, the non-parametric metric is the safer default even though its mean
advantage is not significant. Worth noting the library's new default is
Bhattacharyya, chosen on section 13's evidence, which measured means and not
variances.

**3. Gating mixed norm pairs costs almost nothing.** Mixed pairs beat De Morgan
pairs by **+0.002** -- consistent across seeds (8/8, p = 0.0078) but practically
irrelevant. So #32's decision to require `allow_mixed_norms=True` buys back the
complement's semantic interpretation for a rounding error, which is the right
trade. It also retires section 3.5's "mismatched pairs are slightly better" as a
finding worth acting on.

**Łukasiewicz produces zero valid runs** (0 of 32 cells), reconfirming section 3.5:
nilpotent families saturate to 1 under aggregation and drive the complement to a
constant 0.

**Best configurations, condition `length+template+entropy`** (Mahalanobis 0.782):

| metric | MF | T | S | AUROC | FPR@95 |
|---|---|---|---|---|---|
| wasserstein | gaussian | min/max | hamacher | **0.811 +/- 0.031** | 0.795 |
| wasserstein | gaussian | min/max | min/max | 0.810 +/- 0.027 | 0.931 |
| **wasserstein** | **gaussian** | **hamacher** | **hamacher** | **0.802 +/- 0.022** | **0.786** |
| wasserstein | gaussian | hamacher | einstein | 0.801 +/- 0.022 | 0.783 |

8 of 32 configurations beat Mahalanobis; 24 do not. The win is configuration-
dependent, and choosing badly (trapezoid) loses badly.

### Recommended configuration

**Wasserstein metric + Gaussian membership functions + Hamacher/Hamacher norms.**
Not the top AUROC (0.802 vs 0.811) but the best overall: within noise of the best,
the **lowest variance** of the leaders (+/-0.022), a materially better **FPR@95
(0.786 vs 0.931** for min/max -- the min/max leader is unusable at high recall),
and a De Morgan-consistent pair, so the anomaly rule keeps its interpretation as a
genuine complement. Ranking on AUROC alone would have picked min/max and shipped a
detector that flags 93% of clean output at 95% recall.

---

## 23. The configuration is now declared and enforced

Section 22 showed the membership family is worth **+/-0.262 AUROC** -- an order of
magnitude more than any other knob -- and that two committed numbers had already
moved when a library default changed. Leaving the family implicit was therefore
the largest unguarded risk in the pipeline. `fis_config.py` fixes that.

### Why a keyword was not enough

`AnomalyParameters.member_function` does **not** decide the membership family.
It is read only by `simple_gaussian_predict` (`gauss_math.py:504`);
`tsk_firing_strengths` calls `mf.evaluate(...)` on whatever objects the model
happens to hold. The family is determined entirely by **which builder was
called** -- `create_gaussian_membership_dict` vs `create_trapz_membership_dict`.
Setting `member_function="gaussian"` while calling the trapezoid builder would
produce trapezoid behaviour and report itself as Gaussian.

So `fis_config.build_memberships()` chooses the builder explicitly *and asserts
the objects it produced are the family requested*:

    membership family mismatch: asked for 'gaussian' (GaussianMembership) but
    feature 'a' / label 'm0' produced TrapezoidMembership. Section 22: the
    membership family is worth +/-0.262 AUROC, so this is never a cosmetic
    difference.

Verified by negative test: monkeypatching the `"gaussian"` slot to the trapezoid
builder -- simulating a library change or a miswiring -- raises immediately
instead of silently shifting the headline by a quarter of an AUROC point.
`anomaly_params()` likewise states every field rather than inheriting any default.

### The declared configuration

| setting | value | evidence |
|---|---|---|
| membership | **gaussian** | +0.262 over trapezoid, 8/8 seeds, p = 0.0078 (section 22) |
| metric | **wasserstein** | mean within noise of Bhattacharyya, but 2.7x lower variance |
| norm pair | **hamacher / hamacher** | best variance and FPR@95 of the leaders; De Morgan dual |
| theta | 0.5 | rank-invariant; sets the operating point only (section 3.4) |

`fuzzy_stats.py` and `compare_variants.py` now build through it. Historical
scripts are left as they were, so previously committed numbers stay reproducible.

### Final numbers under the declared configuration

8 seeds, validation-selected configuration reported on a disjoint test half:

| condition | detector | AUROC | FPR@95 | params |
|---|---|---|---|---|
| **length+template** | **FIS - stats** | **0.877 +/- 0.022** | **0.551** | **95** |
| | mean entropy | 0.841 +/- 0.014 | 0.593 | 0 |
| | Mahalanobis - stats | 0.841 +/- 0.024 | 0.588 | 209 |
| **+ entropy** | **FIS - stats** | **0.789 +/- 0.027** | 0.961 | 95 |
| | Mahalanobis - stats | 0.760 +/- 0.032 | 0.780 | 209 |
| | mean entropy | 0.601 +/- 0.016 | 0.751 | 0 |

    paired FIS vs Mahalanobis (identical 19 statistics):
      length+template  +0.036 +/- 0.020, wins 8/8, p = 0.0078
      + entropy        +0.029 +/- 0.042, wins 6/8, p = 0.0781  (not significant)

Switching the metric to Wasserstein **restored the 8/8 result** on the primary
condition that the Bhattacharyya default had reduced to 7/8 (p = 0.0156), and it
improved the hardest condition from p = 0.148 to p = 0.078. The recommendation
section 22 derived from variance rather than mean therefore paid off, which is a
small point in favour of reporting variance alongside means.

**The claim that stands:** on a template-matched, length-matched task, a fuzzy
rule of **4 rules over 6 antecedents (95 parameters)** beats full-covariance
Mahalanobis on identical features by **+0.036 on 8/8 seeds (p = 0.0078)** with
**2.2x fewer parameters** and a better FPR@95 (0.551 vs 0.588), and beats a mean
entropy threshold by the same margin. Under additional entropy matching it remains
ahead but not significantly (p = 0.078), and its FPR@95 degrades badly to 0.961 --
so the high-recall operating point is a genuine weakness, not a rounding detail.

---

## 24. Second model and second task family — the claim narrows

`four-things.md` item 1. Captured `prompts_v3.jsonl` with **Qwen2.5-0.5B-Instruct**
(24 layers x 896, different tokenizer, different training data; 2,111 generations,
3 min 16 s, 8.23 GB peak) and re-ran the analysis on the **v2 short-factual** set
already on disk. Both use only the 19 output statistics, so the comparison is
architecture-independent by construction.

One behavioural difference worth recording: **Qwen pushed back on 420 of 1,640
(25.6%)** invented long-form subjects, against SmolLM2's 13 (0.8%). The surviving
1,220 fabrications are therefore the ones Qwen was *confident* about, which makes
its task strictly harder — and explains the lower absolute AUROC throughout.

### Paired comparisons, 8 seeds each

| task / model | condition | rival | FIS | rival | Δ | wins | p |
|---|---|---|---|---|---|---|---|
| v3 long-form / SmolLM2 | L+T | Mahalanobis | 0.877 | 0.841 | **+0.036** | 8/8 | **0.008** |
| v3 long-form / SmolLM2 | L+T | mean entropy | 0.877 | 0.841 | **+0.036** | 8/8 | **0.008** |
| v3 long-form / **Qwen2.5** | L+T | Mahalanobis | 0.781 | 0.749 | **+0.032** | 8/8 | **0.008** |
| v3 long-form / **Qwen2.5** | L+T | mean entropy | 0.781 | 0.772 | +0.009 | 5/8 | 0.383 |
| v2 short-factual / SmolLM2 | L+T | Mahalanobis | 0.945 | 0.936 | +0.009 | 6/8 | 0.109 |
| v2 short-factual / SmolLM2 | L+T | mean entropy | 0.945 | **0.958** | **−0.014** | 1/8 | **0.016** |
| v3 long-form / SmolLM2 | +entropy | Mahalanobis | 0.789 | 0.760 | +0.029 | 6/8 | 0.078 |
| v3 long-form / Qwen2.5 | +entropy | Mahalanobis | 0.718 | 0.693 | +0.025 | 6/8 | 0.078 |
| v2 short-factual / SmolLM2 | +entropy | Mahalanobis | 0.817 | 0.748 | **+0.069** | 8/8 | **0.008** |

### What replicates, and what does not

**Replicates — the fuzzy rule beats full-covariance Mahalanobis on identical
features.** The sign is positive in **all six** cells, significant in three
(p = 0.008), marginal in two (p = 0.078), and non-significant in one (p = 0.109).
Critically it replicates **across architectures**: SmolLM2 +0.036 and Qwen +0.032,
both 8/8 seeds, both p = 0.008, on the same task. That is a statement about the
*density model* -- same 19 inputs, same fit data, same open-set protocol, only the
model class differs -- and it now has two-architecture support.

**Does not replicate — the fuzzy rule beats a mean-entropy threshold.** This was
part of the §21/§23 headline and it does not survive:

* SmolLM2 long-form: +0.036, 8/8, p = 0.008 ✓
* Qwen long-form: +0.009, 5/8, p = 0.383 -- a tie
* short-factual: **−0.014, 1/8, p = 0.016** -- entropy wins, significantly

So "beats entropy" is **specific to SmolLM2 on the long-form task**, not a property
of the method. The §23 sentence claiming it "beats a mean entropy threshold by the
same margin" is retracted; only the Mahalanobis comparison generalises.

The mechanism is consistent with §21's observation that the selected antecedents
are first-token statistics: a different tokenizer produces a different first token,
and Qwen's much higher abstention rate changes which fabrications remain. The
first-token confidence profile is evidently real but not equally exploitable
across models.

### A limitation of the matching method, exposed here

On the v2 task mean entropy scores **0.958**, and after matching on entropy
quartiles it still scores **0.839** — matching barely dented it. On v3 the same
procedure took entropy from 0.841 to 0.601. The reason is mechanical: when a
nuisance separates the classes almost perfectly, four quartile bins are too coarse
to neutralise it, because within-bin variation still carries most of the signal.

So "+entropy matched" is **not equally strong across tasks**, and the v2
entropy-matched row should not be read as conditioning entropy away. Finer bins
would fix it in principle but there is not enough overlap to support them (§18 hit
the same wall). Any future use of this control should report the residual
nuisance AUROC, as here, rather than assume matching worked.

### Revised claim

> On a template- and length-matched open-set task, a fuzzy rule of 4 rules over
> 6 antecedents (95 parameters) over output-distribution statistics detects
> ungrounded generation better than full-covariance Mahalanobis on identical
> features — **+0.036 (SmolLM2) and +0.032 (Qwen2.5), both 8/8 seeds, p = 0.008** —
> with roughly half the parameters. Against a zero-parameter mean-entropy
> threshold it wins on one model/task pairing, ties on another, and loses on a
> third; the parsimony-and-legibility argument therefore rests on the comparison
> with learned detectors, not on beating entropy.

That is a narrower claim than §23's, and it is the one the evidence supports.
`four-things.md` item 1 is complete; items 2–4 remain.

---

## 25. Item 2 — two attempts at the FPR@95 problem, both negative

The defect (§23): under entropy matching the rule base wins on ranking but its
**FPR@95TPR is 0.903–0.961** — to catch 95% of fabrications it flags nearly
everything. §3.4 proved θ is rank-invariant, so no threshold search can reach a
point the ranking does not already offer; only a different *ranking* helps.

### Attempt 1 — Ch 4.3.1's cascade of specialists with abstention

`cascade.py`. Pass 1 defers rows landing in the uncertain band of its own
known-good score distribution (the band is set from **fit-split quantiles**, so no
label is consulted and the protocol stays open-set). A pass-2 rule base fitted
**only on known-good rows inside that band** re-ranks the deferred rows. Ranks are
composed rather than averaged, so the head of the ranking — where pass 1 is already
confident — cannot be harmed by construction.

`capture_v3`, 8 seeds, cascade minus single pass:

| condition | ΔAUROC | improves | ΔFPR@95 | improves |
|---|---|---|---|---|
| length+template | −0.000 ± 0.010 | 3/8 (p = 0.641) | **+0.050 ± 0.082** | 1/8 |
| + entropy | −0.007 ± 0.013 | 2/8 (p = 0.195) | +0.005 ± 0.010 | **0/8** |

Neutral on ranking and **worse on the tail it was built to fix**. The extra rule
base costs ~29 parameters and buys nothing.

### Attempt 2 — select the configuration on FPR@95 instead of AUROC

The cascade run surfaced a cleaner hypothesis: single-pass FPR@95 with a *fixed*
configuration is 0.694, against 0.961 for the configuration §23 selected on
validation **AUROC**. Selecting on AUROC optimises the whole ranking and can pick a
configuration whose high-recall tail is poor — so target the tail directly.
`fuzzy_stats.py --select-on {auroc,fpr95}`.

| condition | selected on | AUROC | FPR@95 |
|---|---|---|---|
| length+template | auroc | **0.877 ± 0.022** | **0.551** |
| length+template | fpr95 | 0.863 ± 0.022 | 0.572 |
| + entropy | auroc | **0.789 ± 0.039** | 0.903 |
| + entropy | fpr95 | 0.774 ± 0.037 | **0.863** |

**Also negative, and instructively so.** On the primary condition, selecting on
FPR@95 is worse on *both* metrics — including the one it optimises. FPR@95 depends
on a single quantile of the validation positives, so it is a far noisier statistic
than AUROC; selecting on it fits validation noise and generalises worse. On the
harder condition it does trade 0.015 AUROC for 0.040 FPR@95, which is a real but
small effect and does not survive as a recommendation.

### Reading

Two structurally different interventions — a second rule base, and a different
selection objective — both fail to move the high-recall tail. That points at the
tail being a property of the **rule class** rather than of the fitting procedure:
a handful of axis-aligned Gaussian modes with a complement-of-conorm anomaly score
produces a score distribution whose extreme quantiles are simply not well
separated, and adding capacity or re-targeting selection does not change its shape.

Item 2 is therefore **closed as a negative result**. Honest consequence for the
write-up: FPR@95 should be reported as a **standing limitation of the method**, not
as an implementation detail awaiting a fix. `papers/fuzzy-anomaly-rule-slm.md` §3
already lists it; it should now say two remedies were tried and neither worked.

What remains untried, and is the only route I would still credit: change the
**score construction** rather than the rule base — e.g. calibrate μ_anom against
the fit-split score distribution (a per-mode rank or p-value rather than a raw
membership complement), so the tail is shaped by data rather than by the operator
algebra. That is a different contribution from "tune the FIS", and it should be
scoped as such rather than bolted on.

---

## 26. RETRACTION — the fuzzy advantage was configuration selection, not model class

The §21/§23/§24 claim ("a 4-rule fuzzy system beats full-covariance Mahalanobis on
identical features") **does not survive**, for a reason that is my own
methodological error rather than a property of the data.

### The flaw

`fuzzy_stats.py` selects the FIS configuration — whitening × antecedent count ×
mode count × norm pair, **120 candidates** — by scoring each against **labelled
validation positives**, then reports the winner on a disjoint test half. That part
is fine in isolation.

What is not fine: **Mahalanobis, isolation forest, one-class SVM and entropy
received no equivalent search.** They were fitted once, with no hyperparameter
selection against labelled data at all. So the comparison awarded the fuzzy rule a
120-way supervised search and its rivals nothing, and then attributed the
difference to the model class.

### The measurement

Identical splits, identical data, the *only* difference being whether the
configuration is searched or fixed at `fis_config`'s declared defaults
(`--no-select`), v4 SmolLM2, 8 seeds:

| configuration | FIS − Mahalanobis | wins | p |
|---|---|---|---|
| **selected** on validation positives | +0.014 ± 0.017 | 6/8 | 0.078 |
| **fixed** (no search) | **−0.019 ± 0.033** | **1/8** | 0.109 |

The search is worth ~0.033 and **flips the sign**.

`sample_size.py` says the same thing independently. Sweeping the fit-set size on
v4 with a fixed configuration, across all four models and 6 sizes (8 seeds each),
FIS − Mahalanobis is **negative in 23 of 24 cells**, from −0.013 to −0.098:

| n_fit | smollm2 | qwen | gemma | lfm |
|---|---|---|---|---|
| 200 | −0.016 | −0.052* | −0.013 | −0.093* |
| 280 (≈ v3 size) | −0.016 | −0.064* | −0.013 | −0.098* |
| 550 | −0.017 | −0.083* | −0.019 | −0.084* |
| 760 | −0.021* | −0.066* | −0.026* | −0.088* |

\* p < 0.05

### Two hypotheses tested and rejected on the way

**Not a small-sample effect.** The natural explanation for the v3 → v4 shrinkage
was that Mahalanobis's 209 covariance parameters were under-determined at v3's ~280
fit rows (n/p ≈ 1.3) and better estimated with more data. The size sweep refutes
this as the *cause*: the fixed-configuration gap is already negative at n=120 and
stays roughly flat, so the sign was never positive without the search. Mahalanobis
does improve with data (0.700 → 0.709 absolute) while the FIS is flat (~0.658), so
the effect is real but second-order — it is not what produced the v3 result.

**Not the representation.** Absolute AUROCs under a fixed configuration, averaged
over models and seeds: entropy **0.735**, Mahalanobis **0.700–0.709**, FIS
**0.657–0.664**. The fuzzy rule is last.

### What is retracted, and what stands

**Retracted:** §21, §23, §24's positive claim, and the corresponding sections of
`papers/fuzzy-anomaly-rule-slm.md`. There is no evidence here that the fuzzy rule
beats full-covariance Mahalanobis, and with a fixed configuration it is beaten by
both Mahalanobis and a zero-parameter entropy threshold.

**Stands, and is unaffected:**

* Everything in §§8, 11, 12, 20 — the three confounds (length 0.843, prompt family
  ~0.9, entropy 0.964) and the matching controls. That work never depended on the
  fuzzy detector winning, and it is the durable contribution.
* §22's factorial: the membership family dominates (±0.262), the metric is worth
  ≤0.015, mixed norm pairs +0.002. Those are *within-FIS* comparisons where every
  arm had the same search budget, so the fairness flaw does not apply.
* §25's negative results on the FPR@95 tail.
* §19's transfer matrix.

### What a fair comparison would require

Either give every detector the same 120-way supervised search on the same
validation positives (isolation forest and one-class SVM have obvious
hyperparameters; Mahalanobis has shrinkage and feature subsets), **or** compare all
of them at fixed, pre-declared configurations. The second is cheaper and is what
`--no-select` now does. Any future claim in this repo should use one of those, and
report which.

The general lesson is worth carrying into
`papers/hallucination-detection-confounds.md`: an unequal hyperparameter-search
budget is a **fourth confound**, and it is the one I fell for. It has the same
signature as the other three — a real-looking effect, stable across seeds, with a
plausible mechanistic story attached — and it is invisible unless the comparison
is stated in terms of search budget as well as features and data.

---

## 27. When does the fuzzy rule beat entropy? A measured answer

§26 removed the headline but left one live pattern: across four models the fuzzy
rule's advantage over mean entropy tracked how badly entropy was doing. Four
points is an anecdote; this makes it a measurement.

**Design, with both §26 lessons applied.** 4 models x 13 templates = **44 cells**,
each a self-contained detection problem. Both detectors run at **fixed
configuration** — `fis_config` defaults, no search — so neither receives a
supervised budget the other lacks. Template is constant within a cell (so §11's
confound cannot operate) and length is matched inside each cell. 6 seeds.

### The relationship

    corr(entropy AUROC in a cell, FIS − entropy in that cell)
      Pearson  r = -0.779   p = 4.8e-10
      Spearman r = -0.764   p = 1.6e-09

**Checked for the obvious artefact.** `corr(X, Y−X)` is biased negative even for
independent X and Y, because X appears on both sides. Estimating entropy from one
half of the seeds and the difference from the disjoint other half removes the
shared noise:

| estimate | r | p |
|---|---|---|
| naive (same seeds both sides) | −0.779 | 4.8e-10 |
| **split-half A → B** | **−0.776** | 6.2e-10 |
| **split-half B → A** | **−0.723** | 3.0e-08 |
| split-half Spearman | −0.792 | 1.5e-10 |

The effect is essentially unchanged, so it is not the artefact. And
`corr(entropy, FIS) = +0.220 (p = 0.15)` — the two detectors are not simply both
tracking cell difficulty, which would have produced a strong positive.

### What it says

    linear fit:  (FIS − entropy) = −0.846 · entropy + 0.514
    crossover:   the fuzzy rule is ahead when entropy AUROC < 0.608

| regime | cells | FIS − entropy |
|---|---|---|
| entropy weak (≤ 0.783) | 22 | −0.012 |
| entropy strong (> 0.783) | 22 | **−0.217** |

Mann–Whitney p = 0.0002. Per model, ordered by entropy strength:

| model | entropy | FIS | FIS − entropy |
|---|---|---|---|
| **gemma3-270m** | **0.546** | 0.588 | **+0.042** |
| qwen2.5-0.5b | 0.774 | 0.575 | −0.199 |
| lfm2.5-350m | 0.812 | 0.603 | −0.210 |
| smollm2-360m | 0.838 | 0.747 | −0.091 |

The largest single gains are all in near-chance-entropy cells:
`gemma/effect` entropy 0.436 → FIS 0.702 (**+0.265**), `gemma/conjecture` 0.492 →
0.630 (+0.138), `gemma/constant` 0.438 → 0.568 (+0.131).

### The honest claim

> The fuzzy anomaly rule does **not** beat a mean-entropy threshold in general —
> across 44 cells entropy averages 0.743 against the rule's 0.628, and the rule
> wins in only 10 of 44. But **its relative advantage is strongly and predictably
> governed by entropy's own performance** (r ≈ −0.75, p < 1e-7, artefact-checked),
> crossing over at entropy AUROC ≈ 0.61. Where the output distribution is close to
> uninformative, a fixed 4-rule fuzzy system over the *same* statistics recovers
> signal the entropy threshold misses, by up to +0.265.

This is a **complementarity** result, not a superiority one, and it is the first
claim in this study built with equal search budgets from the start.

### Two things it implies

**Model size may set the regime.** Gemma3-270m is the smallest model here and the
only one in the winning regime; its entropy is barely above chance (0.546). If
entropy calibration improves with scale, the fuzzy rule's niche is small or weakly
calibrated models — which is testable directly with the SmolLM2 family
(135M / 360M / 1.7B), and is a far better-motivated scaling study than the one
`four-things.md` shelved.

**Fusion does not work.** A zero-parameter rank-average of the two scores (chosen
because it fits nothing and so stays budget-fair) reaches 0.735 against entropy's
0.743 — it beats *both* detectors in 9 of 44 cells and is not significantly
different from entropy alone (p = 0.51). The two detectors are complementary
*across* regimes but not *within* a cell, so a per-cell blend gains nothing. If
this is to be exploited it must be by **switching** on an estimate of entropy's
reliability, not by blending.

`figures/entropy_regime.png`.

---

## 28. Switching between detectors — works with labels, unproven without

`switching.py`, plan item 6. §27 showed blending gains nothing because the two
detectors are complementary *across* regimes rather than *within* a cell, so the
value is in **choosing** per deployment. Three rungs, all at fixed configuration
(§26) so no arm gets a search budget.

44 cells, 264 cell-seeds, four models.

| rule | AUROC | gain vs always-entropy | |
|---|---|---|---|
| always FIS | 0.6434 | −0.110 | |
| **always entropy** | **0.7538** | — | the incumbent |
| oracle threshold (true AUROC, switch below 0.608) | 0.7694 | **+0.0157** | cheating |
| oracle per-cell best | 0.7828 | +0.0290 | absolute ceiling |

**Rung 1 sets a modest ceiling.** Even a perfect oracle picking the better
detector per cell buys **+0.029**, and the realisable threshold rule +0.016. Most
of the time entropy is already the right choice, so this was never going to be a
large effect — worth knowing before investing in it.

**Rung 2 works, and cheaply.** Estimating entropy's AUROC from *k* labelled
examples and switching on the estimate:

| k labelled | net AUROC | gain | agrees with oracle |
|---|---|---|---|
| 20 | 0.7675 | +0.0137 | 89.0% |
| 50 | 0.7663 | +0.0125 | 94.3% |
| 100 | 0.7697 | +0.0160 | 99.2% |
| 200 | 0.7698 | +0.0160 | 99.6% |

**20 labelled examples recover 87% of the oracle threshold gain**, and 100 recover
essentially all of it. The switching decision is a coarse one — above or below
0.61 — so it tolerates a noisy AUROC estimate far better than a ranking task
would. That is a genuinely practical result: a deployment can decide which
detector to trust from a calibration set small enough to label by hand.

**Rung 3 is unproven, and the reason matters.** Predicting entropy's AUROC from
statistics of the known-good split alone, with the predictor fitted on three
models and tested on the fourth: r = **−0.492**, net gain **+0.0000**. The
prediction is *anti*-correlated with truth on held-out models.

But the individual statistics are strongly predictive when cells are pooled:

| known-good statistic | r with entropy AUROC | p |
|---|---|---|
| `kg_std` (entropy spread on grounded output) | **−0.725** | <0.001 |
| `kg_range` | −0.722 | <0.001 |
| `kg_iqr` | −0.719 | <0.001 |
| `kg_cv` | −0.695 | <0.001 |
| `kg_bimod` | −0.663 | <0.001 |
| `kg_kurt` | +0.616 | <0.001 |

The sign is mechanistically sensible: **the more entropy varies on known-good
output, the worse it separates** — a wide known-good distribution overlaps the
fabricated one. Within-model z-scoring does not rescue the transfer (r = −0.470).

So the honest reading is *not* "the label-free proxy fails". It is that **with
four models, leave-one-model-out cannot test transfer at all**: the fit has three
models to learn from, between-model offsets dominate the eight features, and the
held-out prediction inverts. The signal is real in-sample; whether it generalises
is currently untestable, not tested-and-refuted.

**This is directly unblocked by plan item 5.** The SmolLM2 scaling captures
(135M, 1.7B) take the model count from four to six, which is the minimum at which
leave-one-model-out is worth interpreting. Rung 3 should be re-run then, and the
result reported either way.

### Per model

| model | entropy | FIS | oracle | oracle gain |
|---|---|---|---|---|
| gemma3-270m | 0.5719 | 0.5868 | 0.6376 | **+0.0657** |
| smollm2-360m | 0.8464 | 0.8064 | 0.8826 | +0.0362 |
| lfm2.5-350m | 0.8262 | 0.6059 | 0.8347 | +0.0085 |
| qwen2.5-0.5b | 0.7705 | 0.5743 | 0.7762 | +0.0057 |

Gemma has by far the most to gain, consistent with §27 — it is the model whose
entropy is weakest. SmolLM2's +0.0362 is more interesting: its entropy is the
*strongest* of the four, yet per-cell switching still finds a real gain, which
means the regime is set at the level of individual templates and not only at the
level of the model.

---

## 29. Scale does not set the regime — and the label-free proxy works after all

Plan items 5 and 6's blocked rung, both resolved by the SmolLM2 family captures
(135M / 360M / 1.7B on `prompts_v4.jsonl`, bfloat16, same protocol as §27).

### Item 5 — entropy improves with scale, but family matters more

SmolLM2 holds training data and recipe fixed and varies only size:

| model | params | entropy | FIS | FIS − entropy |
|---|---|---|---|---|
| SmolLM2-135M | 135M | 0.713 | 0.627 | −0.086 |
| SmolLM2-360M | 360M | 0.838 | 0.747 | −0.091 |
| SmolLM2-1.7B | 1.7B | **0.909** | 0.841 | −0.068 |

**Entropy improves monotonically with scale — 0.713 → 0.838 → 0.909.** The
hypothesis that entropy calibration gets better with size is confirmed *within a
family*. The fuzzy rule improves in step (0.627 → 0.747 → 0.841) and never closes
the gap, which stays roughly flat.

**But size is not what sets the regime.** Gemma3-270m has **entropy 0.546** — well
below the crossover — while SmolLM2-**135M**, half its size, reaches **0.713**,
comfortably above it. A model half the size is in the opposite regime.

So the answer to item 5 is the third option in the plan's outcome table, not the
first: within a family entropy scales with size, but **between-family variation is
much larger than the within-family scaling effect**. Whatever puts Gemma in the
winning regime — instruction tuning, calibration, tokenizer, its near-total
absence of pushback (0.1% vs Qwen's 13%) — it is not parameter count. The
practical consequence is that you cannot predict from a model's size whether the
fuzzy rule will help; you have to measure it. Which is what item 6 is for.

Across the SmolLM2 family alone the §27 relationship still holds but weaker
(r = −0.502, p = 0.003, crossover 0.650, FIS ahead in 6/33 cells) — expected,
since restricting to one family removes most of the range in entropy AUROC.

### Item 6, rung 3 — the label-free proxy works with six models

§28 reported the label-free predictor as *untestable* rather than refuted: with
four models, leave-one-model-out has three to learn from, between-model offsets
dominate eight features, and the held-out prediction inverted (r = −0.492). Six
models was the stated minimum for the test to mean anything.

| models | held-out r (predicted vs true entropy AUROC) | p |
|---|---|---|
| 4 | −0.492 | 0.001 |
| **6** | **+0.689** | <0.001 |

**The sign flips and the prediction becomes genuinely useful.** Entropy's
discriminative power in a cell is predictable, out of sample and across model
families, from statistics of the **known-good split alone** — no labels, no
fabricated examples. That is a real and, as far as I know, unreported observation:
*you can tell how much to trust a confidence-based hallucination detector by
looking only at how its confidence behaves on output you already believe.*

The n=4 diagnosis was therefore correct, which is worth recording because the
tempting move at the time was to call the proxy dead.

**But it does not convert into a switching gain.** Switching on the prediction
nets **+0.0025** against the oracle threshold rule's **+0.0132** — about 19% of
what is available. The prediction correlates well over the full range but is not
accurate enough *near the 0.61 boundary*, which is the only place the switching
decision is actually contested. Rung 2 remains the practical route: **20 labelled
examples buy +0.0104 (91% agreement with the oracle), 100 buy +0.0134 (99.5%)**.

### Six-model summary

| model | entropy | FIS | oracle | oracle gain |
|---|---|---|---|---|
| gemma3-270m | 0.5719 | 0.5868 | 0.6376 | **+0.0657** |
| SmolLM2-135M | 0.7245 | 0.6076 | 0.7480 | +0.0235 |
| qwen2.5-0.5b | 0.7705 | 0.5743 | 0.7762 | +0.0057 |
| lfm2.5-350m | 0.8262 | 0.6059 | 0.8347 | +0.0085 |
| SmolLM2-360M | 0.8464 | 0.8064 | 0.8826 | +0.0362 |
| SmolLM2-1.7B | 0.9103 | 0.8630 | 0.9238 | +0.0136 |

### Where this leaves the two questions

* **"Is the fuzzy rule a small-model technique?"** No — that framing is wrong.
  It is a *weak-entropy-model* technique, and weak entropy is not a function of
  size. Gemma3-270m and SmolLM2-135M are the same order of magnitude and sit on
  opposite sides of the crossover.
* **"Can you know in advance whether to use it?"** Partly. With ~20 labels, yes
  and cheaply. Label-free, you can predict entropy's reliability well (r = +0.69
  held-out) but not sharply enough at the decision boundary to act on — so the
  honest statement is *predictable, not yet actionable*.

The remaining gap is precision near the crossover, not the existence of the
signal. A classifier trained directly on "is this cell below the crossover"
— rather than regressing AUROC and thresholding the prediction — is the obvious
next attempt, and would need more models still to validate.
