# An FIS as a token drafter: what the idea has to survive

Working design for `exp/fis-drafter`. Written before any results, so that the
predictions in it can be scored later rather than reconstructed favourably.

---

## 1. The claim, restated precisely

> An FIS can generate the *shape* of the output logit probabilities, and we can
> thus use it as a token drafter.

Split this into two claims, because they have wildly different difficulty and
only one of them is about shape.

**C1 (shape).** From cheap features, an FIS can predict a low-dimensional
description of the next-token distribution — its entropy, peakedness, effective
support size, tail decay.

**C2 (drafting).** That shape is enough to build a draft distribution `q` that
accelerates speculative decoding.

C1 is plausible and is what the rest of this document takes seriously. **C2 does
not follow from C1**, and the gap between them is not a detail. It is the
project.

## 2. Why shape alone cannot draft

Speculative decoding (Leviathan et al. 2023; Chen et al. 2023) samples `x ~ q`
and accepts with probability `min(1, p(x)/q(x))`, resampling rejections from the
normalised residual `(p − q)₊`. The output is then distributed *exactly* as `p`
— the target model's own distribution — regardless of how bad `q` is. A bad
drafter costs speed, never correctness.

The expected acceptance rate has a closed form:

```
α  =  Σ_x min(p(x), q(x))  =  1 − TV(p, q)
```

and the expected tokens per verification pass at draft length γ is
`(1 − α^{γ+1}) / (1 − α)`.

This is good news and bad news at once.

**Good:** the figure of merit is a *distributional distance*, not argmax
accuracy. A drafter is not required to be right about the next token. It is
required to be close in total variation. That is genuinely a statement about
shape, and it is the reason the idea is worth testing at all.

**Bad:** total variation is computed over the *labelled* simplex. Suppose the
FIS predicts the sorted shape perfectly — exact entropy, exact top-1 mass, exact
tail — but assigns those probabilities to the wrong token identities. Then
`min(p, q)` is near zero almost everywhere and `α ≈ 0`. A perfectly-shaped `q`
with a permuted support is not a mediocre drafter; it is worse than useless,
because it costs a draft pass and accepts nothing.

So a distribution factorises into **shape × identity**, and an FIS with ~10
inputs and ~10 outputs can supply shape but cannot supply identity over a
49,152-token vocabulary. Any honest version of this project needs identity to
come from somewhere else.

### 2.1 Exactness is preservable, and the usual objection does not bite

The standard objection — and the one a committee member who has read Leviathan
carefully *will* raise — is that the acceptance rule needs `q(x)` as an exact
normalised probability, and the rejection branch needs `q` over the **entire**
vocabulary to form `(p − q)₊`. So a system emitting "coarse membership grades"
or "an interval" cannot be dropped into the formula without silently corrupting
the output distribution.

That objection is correct as stated, and it does not apply to what is proposed
here, provided one design rule is followed: **the FIS's output must be
materialised into a genuine normalised categorical before it touches the
acceptance test.**

Concretely, in V2 the draft distribution is constructed as

```
q(x) = s_j            if x is the j-th candidate from the cheap ranker
q(x) = (1 − Σ s_j) · u(x) / Σ_{x ∉ cand} u(x)     otherwise
```

with `s` the FIS-predicted head shape and `u` a fixed tail model (uniform, or
the unigram distribution). This is a fully specified, normalised distribution
over all 49,152 tokens, evaluable at any `x` in O(1) and materialisable in O(V)
by a single vectorised operation — the same order as the softmax already being
computed. `(p − q)₊` is therefore computable exactly, and **speculative decoding
remains lossless.** The output is distributed exactly as the target model's.

This matters for positioning as much as for correctness. It means the work does
*not* have to enter the relaxed/lossy-acceptance sub-literature (Medusa's
typical acceptance; BiLD's divergence thresholds; the 2025–26 lossy-verification
papers) and compete there. The FIS supplies a *component* of an exact scheme.
Fuzziness lives in how `q` is arrived at, not in the acceptance test.

The genuine cost is different and should be stated plainly: materialising `q`
over the vocabulary is an O(V) operation per draft step. That is cheap relative
to a transformer forward pass but not free, and Experiment 4 must charge it.

### 2.2 Why "just make the FIS output the vocabulary" is not available

`tribble-fis`'s MIMO predictor trains one regressor per output. 49,152
regressors is not a model. The library's own rule-explosion guard exists for
exactly this reason.

There is one non-obvious escape, and it is worth testing rather than dismissing:
the log-probability matrix of a language model is **approximately low rank**.
This is the softmax bottleneck (Yang et al. 2018) — the matrix of log-probs is
`H Wᵀ` with `H` of width 576 for SmolLM2-135M, so its rank is at most 576 before
the log-normaliser. If a rank-`k` basis with small `k` reconstructs these
distributions well, then an FIS predicting `k` coordinates *does* emit a full
distribution, identities included.

That is a measurable quantity, not a hope, and measuring it is Experiment 1a.
It bounds every model in this family before we build any of them.

## 3. Three variants, in increasing ambition

**V1 — FIS as speculation controller.** Do not draft at all. Predict, from
tier-A/B features, how confident the target model is about to be, and use it to
set the draft length γ per step. Fixed γ is the standard and is known to be
wasteful. Requires only C1.

**V2 — FIS as calibrator of a cheap drafter.** Identity from an n-gram or
prompt-lookup drafter; *probabilities* from the FIS. Cheap drafters have
candidates but no principled confidence, so they are systematically
miscalibrated — and since `α = 1 − TV`, miscalibration is a direct, quantified
loss of acceptance. The FIS's job is to predict how much to trust the ranker at
this step. This is the real "FIS as drafter", made sound.

**V3 — FIS predicts coordinates in a low-rank basis.** Full distributions from
an FIS, contingent on Experiment 1a returning a favourable rank–fidelity curve.

V1 and V2 are the experiments. V3 is gated on a measurement.

## 4. Where the features come from, and the one that makes this work

A drafter that needs the target model's hidden state has already paid for the
forward pass it exists to avoid. So the feature tiers in `capture.py` are a
*deployability* classification, not bookkeeping:

* **Tier A (free).** Statistics of the distribution at steps t−1, t−2, t−3.
  In speculative decoding the verification pass returns the target model's true
  distribution at the last accepted position **at zero marginal cost**. A
  drafter may legitimately condition on it. This is the observation that makes
  the whole scheme coherent, and it turns the problem into one of *shape
  dynamics along a trajectory*: given the shape now, predict the shape a few
  steps ahead.
* **Tier B (cheap).** Token surface form, position, n-gram context. No forward
  pass.
* **Tier C (expensive, not deployable).** Hidden states, per-layer norms.
  Recorded only to establish a ceiling.

Shape-dynamics-along-a-trajectory is, structurally, the problem
`MimoGaussianPredictorMemory` was written for — short/long memory windows and
iterative rollout, built for the double pendulum. The library already has the
right tool pointed at the wrong domain.

## 5. Experiments, and what each one actually proves

Each entry states the claim it can *settle*, separately from what it might
suggest. The distinction is the point.

### Experiment 0 — Characterisation

Distribution of entropy across 20k+ steps; its autocorrelation along a
trajectory; whether the regime labels in the synthetic probe set
(`low_entropy`, `mid_word`, …) actually separate.

*Proves:* whether tier-A features carry any signal at all. If entropy at step t
is uncorrelated with entropy at t−1, V1 is dead immediately and no FIS is
needed to find that out.
*Does not prove:* that an FIS can exploit the signal.

### Experiment 1a — Representational ceiling (no FIS)

SVD the collected log-probability matrix. For k = 1, 2, 4, …, 512, reconstruct
and measure mean `TV(p, p̂_k)` and the implied acceptance `1 − TV`.

*Proves:* an upper bound on **every** model that emits k numbers, this one
included. A perfect FIS with k outputs cannot beat this curve.
*Kills V3 outright* if rank-16 reconstruction leaves TV ≈ 0.7.

### Experiment 1b — Predictive ceiling (no FIS)

Gradient boosting on tier A+B features predicting each shape parameter. Also on
tier A+B+C, for the gap.

*Proves:* the best achievable R² on these features. The FIS is being asked to be
interpretable, not to beat GBM — but if GBM only reaches 0.15, there is nothing
for the FIS to be interpretable *about*.

### Experiment 2 — FIS against the controls that killed the last study

Arms: single best feature · linear · GBM · T1-FIS · IT2-FIS. **Equal
hyperparameter search budget for every arm, stated before running.**

This is not boilerplate. In `experiments/fuzzy-lm-anomaly.md` the tribble FIS
finished *last* against every rival on every model, and a single scalar
(`ent_max`) beat every learned detector. Two of the five confounds that study
found were unequal search budget and an under-specified baseline. The prior
probability that the FIS loses here is high, and the design has to be able to
report that cleanly.

*Proves:* whether the FIS adds anything over a scalar and a line, on identical
budget.

### Experiment 3 — Does calibrated shape buy acceptance?

A real speculative-decoding loop. Ranker held fixed (n-gram / prompt-lookup);
shape from {oracle, FIS, fixed temperature, ranker's own scores}. Measure
acceptance rate α and tokens per verification pass.

*Proves:* C2, or refutes it. The oracle arm is the essential one — it separates
"the FIS's shape is bad" from "shape does not help even when perfect".

### Experiment 4 — Adaptive draft length

γ chosen per step from the FIS's predicted acceptance. Wall-clock tokens/second
against best fixed γ, with the FIS's own cost charged against it.

*Proves:* V1 end to end, in the only unit that matters.

## 6. Type-2: where it is real and where it is decoration

`IntervalType2FuzzyRegressor` builds its footprint of uncertainty by scaling σ
by a **fixed hyperparameter** (`uncertainty_width`), symmetric and identical for
every rule. The FoU is *derived*, not learned. So "IT2 models the uncertainty in
the distribution" is not supported by this implementation, and claiming it would
be the kind of thing §7 of `WORKINGDOC.md` warns about.

There is, however, one use that is not decoration. Acceptance is symmetric in TV
— being flatter than `p` costs as much as being peakier — so the FoU does not
help the point estimate. But **the cost of a rejection is not symmetric across
draft length**: one rejection discards every drafted token after it. So the
quantity worth knowing is not just the predicted shape but the *confidence* in
that prediction, and that is exactly what an interval output is.

**Type-2's defensible role here is as the input to the draft-length controller,
not as a better point estimate.** Wide FoU → draft fewer tokens. That is a
falsifiable claim (Experiment 4 with and without the interval), and it is the
only Type-2 claim this design will make.

## 7. Pre-registered predictions

Recorded now so they can be scored, not adjusted.

| # | prediction | confidence |
|---|---|---|
| P1 | Entropy at step t correlates with t−1 at r > 0.3 | 0.85 |
| P2 | Rank-16 reconstruction leaves mean TV > 0.35 (V3 impaired) | 0.7 |
| P3 | GBM on tier A+B predicts entropy at R² > 0.35 | 0.6 |
| P4 | Tier C adds > 0.15 R² over tier A+B | 0.75 |
| P5 | FIS does **not** beat GBM on accuracy at equal budget | 0.85 |
| P6 | FIS beats the single-best-feature baseline | 0.5 |
| P7 | Oracle shape + n-gram ranker beats fixed-temperature by > 0.05 α | 0.6 |
| P8 | The whole pipeline yields **no** wall-clock speedup at first attempt | 0.8 |

P5 and P8 being high is deliberate. If the FIS's contribution is
interpretability rather than accuracy, the design should say so up front rather
than discover it and re-frame.

## 8. Prior art, and what it does to the novelty claim

A literature sweep was run before building. Three results change the design.

**The gap for FIS-as-drafter is real and confirmed empty.** No Mamdani or TSK
system has been used as a speculative-decoding drafter. The nearest name
collision is a trap and must be cited and distinguished explicitly: Holsman,
Huang & Dhingra, *Fuzzy Speculative Decoding for a Tunable Accuracy–Runtime
Tradeoff* (arXiv:2502.20704) uses "fuzzy" as an English adjective for a
divergence-bounded acceptance test. No membership functions, no rule base, no
fuzzy sets. A committee member who searches the obvious phrase will find this
paper first, so the proposal has to get there before they do.

Existing FIS + LLM work (fuzzy reasoning chains, fuzzy prompting, fuzzy
membership features) sits at other layers of the stack — input features or
post-hoc reasoning control — never on the output distribution.

**V1's mechanism is not novel, and this is the finding that matters most.**
Entropy-gated adaptive draft length is an active 2024–2026 sub-literature:

| work | mechanism | reported |
|---|---|---|
| AdaEDL (arXiv:2410.18351) | draft-model entropy → early draft stopping | 10–57% over static γ |
| Confidence-Modulated SD (arXiv:2508.15371) | entropy + margin → γ and strictness | speedup, BLEU/ROUGE held |
| SpecKV (arXiv:2605.02888) | small MLP on confidence/entropy → per-step γ | 56% over fixed γ=4 |
| EntMTP (arXiv:2606.27550) | running local entropy → switch draft trees | 1.09–1.36× |

So "predict a cheap statistic of distribution shape and gate a decoding
decision on it" is **taken**, executed with small neural probes. V1 cannot be
proposed as a new mechanism. It can only be proposed as *an interpretable
rule base substituted into a known mechanism*, with the accuracy cost against
an MLP measured and stated. That is a narrower claim, and it is the honest one.

It also raises the bar for Experiment 2: the baseline set must now include a
small MLP, because that is what the literature actually uses.

**V2 is the less-occupied slot.** Calibrating a cheap non-neural drafter's
probabilities — as opposed to gating on entropy — is not something the
entropy-gating papers do. Combined with §2.1's result that this can be done
without giving up losslessness, V2 is where the defensible contribution is.

**Useful corroboration.** SAGE reports strong temporal correlation of entropy
across decoding steps, which is P1. *Sequences of Logits Reveal the Low Rank
Structure of Language Models* (arXiv:2510.24966) and SlimSpec (arXiv:2605.10453,
a low-rank draft LM-head) are direct support for Experiment 1a's premise — and
SlimSpec is also the precedent that would make V3 an FIS-flavoured variant of
an existing neural idea rather than a new one.

**Hazard to test for, not just cite.** Cui, Wu & Xu (arXiv:2102.04271) show TSK
defuzzification saturates in high input dimension. Our feature count is small
enough to be safe, but the failure is silent, so Experiment 2 should check
firing strengths are not collapsing.

Citations marked unverified by the sweep (2606.30265, 2607.26627, 2605.02888,
2606.27550, 2512.23765) must be confirmed from primary sources before any of
this reaches a document with a committee's name on it.

## 9. The fallback that is still a result

If C2 fails — and P8 says the first attempt will — the salvageable contribution
is not nothing:

*An interpretable rule base over cheap features that predicts when a language
model is about to be uncertain*, validated against a stated ceiling and honest
baselines. That is a readable-rules result of the same kind as the thesis's
Chapter 4 argument that interpretability is a property of the feature ranking.
And a negative C2 with a measured ceiling is publishable in a way that a vague
positive is not.
