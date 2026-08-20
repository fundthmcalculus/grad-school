# Round 1 results: the ceilings

Run of record: `runs/main/` — SmolLM2-135M-Instruct, 1,222 prompts (1,200 dolly
stratified across 8 categories + 22 synthetic probes), 71,359 generation steps,
temperature 1.0, seed 0, 51 s on one GB10. Full log-probs retained for 20,000
steps. Every number below is from that single run.

Predictions were registered in `DESIGN.md` §7 before any of this was measured.

---

## 0. What round 1 settles

| | verdict |
|---|---|
| **V3** — FIS predicts a low-rank code, expands to the full vocabulary | **dead** |
| **V2** — cheap ranker supplies identity, FIS supplies shape | **alive**, ceiling α = 0.85 |
| **V1** — FIS predicts shape to control draft length | signal is real but modest; mechanism not novel |

The single most useful number: at a budget of ~10–16 numbers per step, spending
them on **latent shape coordinates** buys an acceptance ceiling of **0.18**;
spending them on **token identities** buys **0.85**.

---

## 1. Experiment 1a — the representational ceiling

Rank-`k` basis learned on a training half of the log-probability matrix,
held-out half projected onto it (oracle coefficients — no predictor could do
better), re-softmaxed, compared to the true distribution. Acceptance in
speculative decoding is exactly `α = 1 − TV(p,q)`, so the TV column converts
directly.

| rank k | TV | **α ceiling** | explained var |
|---|---|---|---|
| 0 (unconditional mean) | 0.918 | 0.082 | 0.000 |
| 1 | 0.912 | 0.088 | 0.550 |
| 4 | 0.883 | 0.117 | 0.658 |
| 8 | 0.850 | 0.150 | 0.697 |
| **16** | **0.818** | **0.182** | 0.741 |
| 32 | 0.767 | 0.233 | 0.782 |
| 64 | 0.704 | 0.296 | 0.826 |
| 128 | 0.599 | 0.401 | 0.880 |
| 256 | 0.427 | 0.573 | 0.938 |
| 512 | 0.144 | 0.856 | 0.993 |

Rank needed for α = 0.5: **256**. For α = 0.7: **512**. There is no rank below
512 that reaches the acceptance rates published cheap drafters report
(REST 1.62–2.36×, BiLD ~2×).

**P2 confirmed** (predicted rank-16 TV > 0.35; measured 0.818).

An FIS in this scheme would need hundreds of outputs, and `tribble-fis`'s MIMO
predictor trains one regressor per output. V3 is not a tuning problem.

### 1a.1 The methodological finding, which may outlast the rest

At rank 16 the basis explains **74% of the variance** in log-probability space
and yields an acceptance ceiling of **0.18**.

Explained variance in logit space is close to uninformative about drafting
utility. The two metrics disagree because L2 error is spread over 49,152
coordinates while total variation is dominated by the handful carrying the mass
— and the head is exactly what a low-rank projection smooths away. Any
low-rank drafter reporting reconstruction error or explained variance is
reporting a quantity that does not translate into speedup. This is worth
stating in its own right; it is cheap to verify and easy to get wrong.

## 2. The same budget spent on identities

`α_max` for an oracle top-k candidate set (true probabilities on the true top-k
tokens) is just the expected mass those tokens carry:

| k | **α ceiling** | steps with >0.9 of mass |
|---|---|---|
| 1 | 0.561 | 22.8% |
| 5 | 0.789 | 44.5% |
| **10** | **0.848** | 54.9% |
| 50 | 0.929 | 76.2% |
| 128 | 0.957 | 86.3% |

Median effective support is **8 tokens** for 90% of the mass (mean 80.6 — the
mean is dragged by a heavy tail; the median is the operative number).

Ten identities beat sixteen latent dimensions by 4.7×. This is the quantitative
form of the argument in `DESIGN.md` §2: a distribution factorises into shape ×
identity, and essentially all of the usable information is in identity. It is
also why V2 is the surviving variant — it takes identity from a ranker and asks
the FIS only for the part that is genuinely low-dimensional.

## 3. Experiment 1b — the predictive ceiling

Gradient boosting, ridge, and a single-best-feature arm on each tier. **Split by
prompt, not by step** — steps within one generation are dependent and a row
split inflates everything. Step 0 dropped (no tier-A history by construction).

R² on held-out prompts:

| target | tier A | tier B | **A+B (deployable)** | A+B+C (ceiling) | best single feature |
|---|---|---|---|---|---|
| entropy | 0.220 | 0.153 | **0.381** | 0.606 | 0.173 |
| top1_prob | 0.165 | 0.125 | **0.285** | 0.480 | 0.140 |
| log_margin_12 | 0.174 | 0.158 | **0.308** | 0.545 | 0.144 |
| log nucleus_90 | 0.233 | 0.154 | **0.398** | 0.620 | 0.179 |

The best single feature is `ent_ema_short` (running mean of the last three
entropies) for every target; the best tier-B feature is `tok_logfreq`.

**P3 confirmed** (predicted A+B entropy R² > 0.35; measured 0.381).
**P4 confirmed** (predicted tier C adds > 0.15; measured +0.225).

Two things follow.

**There is something for a multivariate model to do.** GBM on A+B reaches 0.381
against the single feature's 0.173 — it more than doubles it. This is a
different outcome from `experiments/fuzzy-lm-anomaly.md`, where a single scalar
beat every learned detector, and it is the main reason to keep going.

**The deployable set recovers about 63% of the reachable signal** (0.381 of
0.606). The missing 0.225 is in the hidden state, which a drafter cannot afford.
That gap is a hard budget on V1, and it should be quoted whenever the FIS's R²
is quoted.

## 4. Experiment 0 — characterisation

**Entropy** mean 2.031 nats (sd 1.581), median 1.857, against a 10.803 maximum.
12.2% of steps are near-deterministic (< 0.1 nats) and 22.8% put > 0.9 on one
token — those are free for any drafter and they inflate every average, so
aggregate acceptance figures need this denominator stated.

**Persistence, pooled vs within-prompt.** The control matters:

| statistic | pooled lag-1 | within-prompt lag-1 | between-prompt var frac |
|---|---|---|---|
| entropy | 0.343 | 0.202 | 0.175 |
| varentropy | 0.360 | 0.203 | 0.194 |
| top1_prob | 0.300 | 0.190 | 0.134 |
| log_margin_12 | 0.318 | 0.222 | 0.121 |

**P1 confirmed pooled (0.343 > 0.3), not confirmed within-prompt (0.202).**
Both numbers are legitimate and they answer different questions. A drafter does
observe the running trajectory, so it can exploit prompt-level difficulty — the
pooled number is available in deployment. The within-prompt number is what is
left after knowing the prompt is a hard one, and it is weak. Reporting only the
pooled figure would have overstated the step-to-step dynamics by 70%.

**The tail exponent is nearly invariant.** Slope −1.651 ± 0.296 (IQR
[−1.849, −1.437]), mean fit residual 0.090. Zipfian structure holds, and the
exponent barely moves across regimes. Consequence: `tail_slope` is close to a
constant and carries little predictive signal — it is a feature to drop, and
the invariance is a small finding in its own right.

## 5. A defect this run found in its own protocol

The first smoke run applied the chat template to every prompt, including the
synthetic completion probes. Wrapping `"The capital of France is"` as a user
turn makes the model answer conversationally instead of completing the string,
and the designed regime separation vanished: `low_entropy` measured **2.508**
nats against `high_entropy`'s **2.477** — no separation at all. The template is
now a per-prompt property (`Prompt.use_chat`), set for dolly rows and unset for
the probes.

Recorded because it is the failure mode `WORKINGDOC.md` §7 names: the run exited
zero and produced a plausible table. Nothing but the *expected* contrast between
two categories revealed it.

## 6. Scorecard

| # | prediction | outcome |
|---|---|---|
| P1 | entropy lag-1 correlation > 0.3 | **split** — 0.343 pooled, 0.202 within-prompt |
| P2 | rank-16 leaves TV > 0.35 | **confirmed** — 0.818 |
| P3 | GBM tier A+B entropy R² > 0.35 | **confirmed** — 0.381 |
| P4 | tier C adds > 0.15 R² | **confirmed** — +0.225 |
| P5–P8 | (require an FIS / a decoding loop) | not yet tested |

## 7. Type-2: a library defect, then a structural wall

### 7.1 `IntervalType2FuzzyRegressor.predict` discarded the TSK consequents

Found by the invariant that should have been a test: **IT2 must reduce to
type-1 as the footprint of uncertainty vanishes.** It did not. At
`uncertainty_width=0.01`, type-1 scored R² **+0.192** and IT2 scored **−0.551**
— worse than predicting the mean.

The cause, in `it2_regressor.py`:

```python
y_normalized = np.mean(firing_crisp, axis=1)
y_pred = self.y_min_ + y_normalized * (self.y_max_ - self.y_min_)
```

This is not TSK inference. It averages raw firing strengths across output
buckets and rescales that into the target range — never touching
`y_bucket_mean_` or `corr_terms_`, and never normalising by the total firing
strength. The output was therefore driven by the *magnitude* of the firing
strengths rather than by their distribution across buckets. Three symptoms, all
observed and all explained by it:

* predictions collapsed toward `y_min` (mean **0.91** against a true **2.03**);
* the bias shrank monotonically as `uncertainty_width` grew (0.91 → 2.52),
  because a wider footprint raises the lower membership's firing strengths —
  backwards, since width should set the interval, not the location;
* the output range was compressed (max 3.68 against a true 8.20).

**Fixed** by routing the type-reduced firing strengths through the same
consequent evaluation type-1 uses (`regression.apply_tsk_consequents`, extracted
for the purpose). `predict_intervals` carried the identical bug and was fixed
with it. After the fix IT2 matches type-1 to **3e-05** at `uncertainty_width=1e-6`.

Effect on this study: IT2 on entropy moved from **−0.036 to +0.308**. The
library's test suite goes 481 → 483 passing with the two new invariant tests;
the 5 remaining failures are pre-existing (`optimizers` backend) and were
confirmed against a stash of the change.

**Every IT2 regression number this library has produced is affected.** The
thesis has IT2 content and it needs re-checking.

### 7.2 Learning the footprint does not rescue it — the mechanism saturates

With the estimator fixed, `LearnedFoUIT2Regressor` (in `fisdraft/learned_fou.py`)
replaces the global `uncertainty_width` with a fitted vector — one width per
feature, or per (feature, bucket) cell — optimised against the **Winkler
interval score**, a proper scoring rule for the (1−α) interval. Fitted on a
held-out slice of train so the widths cannot shrink onto in-sample residuals.

It does not work, and the reason is structural rather than numerical.

| model | R² | interval score ↓ | coverage (nominal 0.90) | mean width |
|---|---|---|---|---|
| stock `uw=0.1` | 0.195 | 22.66 | 0.012 | 0.05 |
| stock `uw=0.5` | 0.202 | 19.95 | 0.140 | 0.50 |
| stock `uw=0.9` | 0.200 | 13.09 | 0.555 | 2.09 |
| learned, global | 0.191 | 12.79 | 0.568 | 2.14 |
| learned, per-feature | 0.191 | 12.79 | 0.568 | 2.14 |
| learned, per-cell | 0.152 | 18.30 | 0.153 | 0.66 |

Two things to read off. **Stock IT2 intervals are drastically miscalibrated** —
a nominal 90% interval achieving 1.2% coverage at the guide's low settings, 14%
at the recommended 0.5. And **the fitted widths all saturate at the optimiser's
upper bound**, which is why per-feature is bit-identical to global: the extra
freedom is never used, because every width wants to be larger.

Pushing the footprint far past any plausible setting shows why:

| footprint w | mean interval width | coverage |
|---|---|---|
| 0.5 | 0.294 | 0.080 |
| 1 | 2.053 | 0.545 |
| 2 | 2.065 | 0.549 |
| 10 | 2.059 | 0.546 |
| 200 | 2.058 | 0.546 |

**The output interval width saturates at ≈2.06 and never moves again.** A
calibrated 90% interval on this target needs ≈5.19 (2 × 1.645 × sd). The
mechanism cannot reach it at any footprint.

The reason is that this design puts the footprint on the **antecedents** while
the **consequents stay crisp**. The two interval bounds are two different
firing-weighted averages of the *same fixed* consequent values, so their
difference is bounded by the spread of those consequents — a quantity fixed at
fit time and unrelated to how uncertain the target actually is at a given
input. As `w → ∞` the upper membership tends to uniform weighting and the lower
to a hard max, both limits fixed, and the width stops responding.

Raising `n_output_buckets` lifts the ceiling (max observed width 4.93 at 5
buckets, 5.66 at 10) but by widening the consequent spread — at 20 buckets that
spread reaches 242 against a target sd of 1.58, which is the bucket means
becoming unstable, not the uncertainty being better modelled.

**Conclusion for the Type-2 question.** Making the footprint learned was the
right thing to try and it is not sufficient, because the footprint is attached
to the wrong half of the model. Interval-valued *consequents* — the Liang &
Mendel IT2-TSK formulation — are what would be needed for calibrated prediction
intervals. That is a substantially larger change to `tribble-fis` than a fitted
`uncertainty_width`, and it should be decided on its own merits rather than as
a dependency of the drafter work.

This is a stronger result than a working learned FoU would have been: it says
*why* the current design cannot quantify uncertainty, with a saturation curve
that anyone can reproduce in a minute.

## 8. Experiment 3 — does a cheap ranker plus a predicted shape accept?

Computed **exactly, offline**. Acceptance is `α = Σ min(p,q)` and the capture
holds the true `p` for 20,000 steps, so given a candidate set and a shape rule
`α` follows in closed form — no generation, no Monte-Carlo error. `q` is
materialised as a full normalised categorical (§2.1 of `DESIGN.md`), so the
acceptance test is the exact one and decoding stays lossless.

Rankers fitted on 60% of prompts, scored on the disjoint 40%; each step sees
only the prefix it actually had. k=8 candidates, 7,988 scored steps.

| ranker | argmax hit rate | fix .30 | fix .56 | fix .80 | **oracle shape** | shape headroom |
|---|---|---|---|---|---|---|
| oracle ranker | 1.000 | 0.443 | 0.653 | 0.766 | **0.831** | 0.065 |
| prompt_lookup | 0.491 | 0.143 | 0.204 | 0.256 | **0.416** | 0.161 |
| bigram | 0.411 | 0.118 | 0.157 | 0.191 | **0.347** | 0.155 |
| unigram | 0.233 | 0.053 | 0.060 | 0.074 | **0.200** | 0.126 |

### 8.1 The decomposition

```
oracle ranker + oracle shape   0.831   <- ceiling at k=8
oracle ranker + best fixed     0.766   <- shape is worth  0.065
prompt_lookup + oracle shape   0.416   <- identity costs  0.414
```

**Identity costs 6.4× what shape is worth.** This is the §2 argument confirmed
on the acceptance metric itself rather than by analogy.

### 8.2 The one result that goes the FIS's way

Shape headroom is *larger* when the ranker is worse — 0.161 for prompt_lookup
against 0.065 for a perfect candidate set. That is the mechanism `DESIGN.md` §3
predicted: when the candidate set is unreliable, how much mass you put on it
matters more, and an overconfident `q` on wrong candidates is expensive. The
effect is real and measurable.

It is also not enough. With a real ranker and a **perfect** shape the ceiling is
α = 0.416, i.e. 1.69 tokens per verification pass at γ=4, against 1.34 for the
best fixed shape. An FIS predicting `top1_prob` at R² ≈ 0.28 would capture some
fraction of that 0.161 — optimistically +0.04 α, moving 1.34 → ~1.40 tokens per
pass. Published cheap drafters report 1.62–2.36× (REST) and the EAGLE family
2.7–6.5×.

### 8.3 The rescue hypothesis, tested

n-gram and retrieval drafters are known to depend on lexical overlap, so the
obvious defence is that dolly is the wrong benchmark. Tested by category:

| category | prompt_lookup hit rate | n |
|---|---|---|
| closed_qa | 0.612 | 580 |
| summarization | 0.596 | 586 |
| information_extraction | 0.547 | 525 |
| classification | 0.474 | 1392 |
| open_qa | 0.472 | 1206 |
| creative_writing | 0.433 | 1018 |

The effect is real and in the predicted direction — context-bearing categories
beat creative writing by 0.18 — but the *best* category reaches 0.612, not the
0.9 that would change the conclusion. Lexical overlap moves the number; it does
not rescue the approach.

**Limitation, stated rather than buried.** This is a reimplementation of
prompt-lookup, and dolly contexts were capped at 600 characters — not the
long-document editing and summarization regime where the published 2–4× figures
come from. The identity/shape decomposition is robust to ranker quality (the
0.065 headroom is measured under a *perfect* ranker), but the absolute α for
prompt_lookup here should be read as a lower bound on what a well-tuned
retrieval drafter would achieve.

## 9. Experiment 4 — embedding space, which is where the problem actually lives

SmolLM2 ties its embeddings and its LM head has no bias, so

```
logits = h @ E.T        h in R^576,  E in R^{49152 x 576}
```

Verified against the stored log-probs: max abs error **0.0157**, mean
**0.0038**, which is the float16 storage floor. Rank-576 reconstruction returns
α = **0.9998**. The identity holds exactly.

This means the 49,152-dimensional logit vector is an exact linear image of 576
numbers, and §1's rank-512-for-α=0.86 result was rediscovering `hidden_size`
the hard way — through a softmax, in the wrong coordinate system.

It also means `hidden_last.npy` (180 MB) regenerates every distribution
exactly and `full_logprob.npy` (2 GB) is redundant.

### 9.1 Identity stops being a separate problem

Which tokens carry the mass is decided by which rows of `E` have the largest
inner product with `h`. A model that predicts `h` gets candidate identities
from a top-k against `E` — it never has to name a token. The shape/identity
split that killed V3 and closed V2 does not apply here: both come out of the
same 576 numbers. This is the right frame, and it is why §1 and §8 were asking
the question in a form that could not be answered well.

### 9.2 But low-rank compression is no better here

| rank k | α, log-prob space (§1) | α, **embedding space** | argmax agreement |
|---|---|---|---|
| 16 | 0.182 | **0.178** | 0.117 |
| 32 | 0.233 | **0.228** | 0.166 |
| 64 | 0.296 | **0.299** | 0.233 |
| 128 | 0.401 | **0.446** | 0.399 |
| 256 | 0.573 | **0.645** | 0.602 |
| 512 / 576 | 0.856 | **0.9998** | 1.000 |

Embedding space wins decisively at high rank and is **indistinguishable at low
rank**. At k=16 both give α ≈ 0.18. So an FIS emitting 8–32 numbers caps at
α ≈ 0.14–0.23 whichever space it emits them in. Changing coordinates does not
change the low-rank verdict.

### 9.3 The precision budget — the number worth keeping

α as a function of relative L2 error in `h`:

| rel. error in h | α | argmax agreement |
|---|---|---|
| 0.005 | 0.995 | 0.994 |
| 0.01 | 0.989 | 0.985 |
| 0.02 | 0.979 | 0.970 |
| 0.05 | 0.947 | 0.933 |
| **0.10** | **0.894** | 0.867 |
| **0.20** | **0.788** | 0.747 |
| 0.40 | 0.598 | 0.554 |

In absolute terms this is *generous*: a 20% error on a 576-dimensional vector
still yields α = 0.79, which would beat every cheap drafter in the literature.
Restated as a fit quality, hitting α ≈ 0.79 needs **R² ≈ 0.95 on the centred
hidden state**, and α ≈ 0.89 needs R² ≈ 0.99.

### 9.4 The kill: consecutive hidden states are nowhere near that close

The natural cheap predictor is persistence — reuse the previous step's `h`,
which speculative decoding's verification pass returns at zero marginal cost.

| predictor | α |
|---|---|
| previous step's `h` | **0.0614** |
| training-set mean `h` | **0.0813** |

**Persistence is worse than the unconditional mean.** Measured over 70,137
consecutive pairs:

| | median | mean |
|---|---|---|
| relative L2 distance ‖h_t − h_{t−1}‖ / ‖h_t‖ | **0.842** | 0.857 |
| cosine similarity | 0.648 | 0.614 |
| relative L2, mean-centred | **1.000** | 1.007 |
| cosine similarity, mean-centred | 0.503 | 0.487 |

The budget is 0.10–0.20. The observed step-to-step movement is **0.84** — a
4–8× gap. Centred, the relative distance is exactly 1.0, i.e. **R² = 0**:
knowing `h_{t−1}` tells you nothing about `h_t` in the geometry that decides
the output.

(Half of ‖h‖ is the constant mean vector — ‖mean h‖ = 22.3 against a median
‖h‖ = 44.7 — the usual rogue-dimension effect. Removing it is what takes
cosine from 0.65 to 0.50 and is why the raw numbers flatter persistence.)

### 9.5 What this settles

The embedding-space frame is correct and it makes the target concrete: predict
a 576-dimensional vector to within 10–20% relative error and you have a
state-of-the-art drafter. It also shows why no cheap predictor is going to:

* **compression does not help** — 16 latent dimensions cap at α = 0.18;
* **the tolerance is tight** — R² ≈ 0.95 on a 576-dimensional target;
* **the free signal is empty** — the previous hidden state carries R² = 0.

Against tier A+B predicting a *scalar* (entropy) at R² = 0.38, the gap is not a
matter of a better model class. An FIS is not the limiting factor here, and
neither is an MLP: the features do not contain the information.

## 10. Experiment 5 — predicting `h` from the context embeddings

Experiment 4 closed persistence. This asks the structurally different question:
not "where was the state one step ago" but "what is in the context now",
using only lookups into the same tied `E` the LM head already is, plus one
linear map. Scored in α, since experiment 4 gives the conversion.

| arm | α | argmax | rel. L2 (centred) | implied R² |
|---|---|---|---|---|
| mean `h` (floor) | 0.083 | 0.055 | 1.000 | 0.00 |
| previous `h` | 0.063 | 0.030 | 1.000 | 0.00 |
| `E[h ǀ last token]` lookup | 0.224 | 0.190 | 0.892 | 0.20 |
| `E[h ǀ last 2 tokens]` lookup | 0.247 | 0.213 | 0.899 | 0.19 |
| ridge on last embedding | 0.215 | 0.179 | 0.866 | 0.25 |
| ridge on bag-of-embeddings | 0.239 | 0.193 | 0.805 | 0.35 |
| **ridge on bag + previous `h`** | **0.300** | 0.243 | **0.666** | **0.56** |

**Context embeddings carry real signal.** α goes 0.083 → 0.300, a 3.6× lift over
the floor, from a lookup table and a single matmul. This is the first arm in the
whole study that moves the needle.

**Persistence is informative conditionally, not marginally.** `h_{t−1}` alone
scores *below* the floor (0.063 vs 0.083), but adding it to the bag takes
0.239 → 0.300. Its useful component only becomes visible after conditioning on
the context; the marginal test in §9.4 understated it.

### 10.1 The linear route saturates at α ≈ 0.30

Scaling the context window, with and without `h_{t−1}`:

| context tokens | α (alone) | α (+ prev `h`) | R² (+ prev `h`) |
|---|---|---|---|
| 1 | 0.215 | 0.299 | 0.559 |
| **2** | 0.232 | **0.303** | **0.559** |
| 4 | 0.239 | 0.303 | 0.549 |
| 8 | 0.239 | 0.298 | 0.528 |
| 16 | 0.231 | 0.285 | 0.477 |

It peaks at two context tokens and then *degrades*.

**Checked that this is a real ceiling and not ridge under-regularising.** Re-run
with λ selected per feature set on a held-out validation split (λ ∈ 10⁻¹…10⁶,
train/val/test all split by prompt):

| features | selected λ | val R² | α | test R² |
|---|---|---|---|---|
| concat2 + prev | 1e+01 | 0.544 | **0.3008** | 0.552 |
| concat8 + prev | 1e+02 | 0.523 | 0.2899 | 0.534 |
| concat16 + prev | 1e+02 | 0.503 | 0.2836 | 0.509 |

The selection does pick heavier regularisation for the wider feature sets, and
the ordering is unchanged. A linear map over context embeddings tops out at
**α ≈ 0.30, R² ≈ 0.55**, and more context genuinely does not help it — which is
the expected shape of the result, since what a transformer contributes over a
bag of embeddings is nonlinear mixing.

### 10.2 Where that leaves the gap

```
floor (mean h)                    0.083
best cheap linear predictor       0.303   <- saturated
precision budget for α = 0.79     needs rel L2 <= 0.20, i.e. R^2 >= 0.96
exact h                           1.000
```

The linear route closes roughly a quarter of the floor-to-exact distance. The
remainder is not an information problem — `h_t` is a *deterministic* function
of the context tokens, so R² = 1 is attainable in principle, by running the
model. It is a capacity problem, and closing it needs nonlinearity.

### 10.3 This frame is EAGLE's, and that is the finding

Predicting the next *hidden state* rather than the next token is precisely
EAGLE's premise (arXiv:2401.15077, "Speculative Sampling Requires Rethinking
Feature Uncertainty"), which reports 2.7–6.5× across its versions. Arriving at
the same frame independently is a check that the frame is right. It also says
the slot is occupied, by small neural networks doing exactly the nonlinear
regression §10.2 identifies as the remaining gap.

For the FIS specifically this is terminal: the target is 576 correlated
outputs, `tribble-fis` MIMO trains one regressor per output, and the bar is
R² ≥ 0.96 where an unconstrained linear map saturates at 0.56. There is no
version of this where a rule base is the right tool.

What the study can still contribute here is the **measurement apparatus**: the
precision budget (§9.3), the α-vs-relative-error conversion, and the cheap
baselines (§10) are what an EAGLE-style drafter should be reported against, and
they were not available before. A drafter paper that reports R² on hidden
states is reporting a number whose relationship to speedup nobody had measured.

## 11. Replication on an untied head — pythia-410m

The embedding-space argument leaned on SmolLM2's tied embeddings, so it needed
checking on a model where the LM head is a separate matrix. `pythia-410m`:
untied, vocab 50,304, hidden 1,024, 24 layers, base (not instruction-tuned).
Same battery, same protocol — 75,659 steps in 141 s.

`logits = h @ W.T` holds here too: max abs log-prob error **0.0156** against the
stored float16, i.e. the storage floor. **Tying was not load-bearing** — what
matters is only that the head is linear and unbiased.

Absolute α is higher for pythia everywhere, because a base model's
distributions are flatter than an instruction-tuned one's — its *floor* is
0.143 against SmolLM2's 0.083. So the comparison has to be **relative to each
model's own floor**, otherwise the instruct/base difference is mistaken for a
structural one.

| arm | SmolLM2-135M | ×floor | pythia-410m | ×floor |
|---|---|---|---|---|
| mean `h` (floor) | 0.083 | 1.00 | 0.140 | 1.00 |
| **previous `h`** | **0.063** | **0.76** | **0.118** | **0.85** |
| `E[h ǀ last 2 tokens]` | 0.247 | 2.97 | 0.292 | 2.09 |
| ridge on bag | 0.239 | 2.87 | 0.294 | 2.10 |
| **ridge on bag + prev `h`** | **0.300** | **3.61** | **0.366** | **2.62** |
| implied R² of that arm | 0.56 | | 0.50 | |

Every qualitative finding replicates:

* **Persistence is below the floor on both models** (0.76× and 0.85×). The most
  counter-intuitive result in the study is not an artifact of one architecture.
  On pythia the centred relative error of `h_{t−1}` is **1.040** — literally
  worse than using the mean.
* **`h_{t−1}` helps conditionally on both** — pythia 0.294 → 0.366 when it is
  added to the bag, the same pattern as SmolLM2's 0.239 → 0.300.
* **The linear ceiling replicates** at R² ≈ 0.50–0.56, α ≈ 0.30–0.37, against a
  budget of R² ≥ 0.96. Pythia's precision budget is somewhat more forgiving
  (rel. error 0.20 → α 0.856 vs 0.788) but its achieved error is 0.705, still
  3.5× too large.
* **Rank-16 buys about 2× the floor on both** — 0.178/0.083 = 2.2× for SmolLM2,
  0.302/0.143 = 2.1× for pythia. Low-rank compression is equally limited in
  both, once the floor is accounted for.

Two variables differ between these models (tying *and* instruction-tuning), so
this is not a clean isolation of tying. It does not need to be: the claim being
tested is that the findings survive changing the architecture, and they survive
changing two things at once, which is the stronger version.

(One bookkeeping note: the rank sweep tops out at 576, which is full rank for
SmolLM2 but not for pythia's 1,024 — hence pythia's rank-576 α of 0.760 rather
than 1.0. The comparison rows above are all at ranks below 576 and unaffected.)

## 12. Experiments 6 and 7 — the objective was wrong, and §9.2 is retracted

Every arm up to here was fitted by least squares: PCA minimises L2
reconstruction of `h`, ridge minimises L2 prediction error. But α is what
decides a drafter, and L2 is only a surrogate for it — a fixed L2 budget can be
spent well or badly, since error aligned with the embedding directions carrying
the top tokens costs a great deal of α and error in unoccupied directions costs
almost none. §1.1 already showed the two diverging in the other direction.

### 12.1 The rank-k ceiling is a property of the basis, not the problem

Rank-k reconstruction with oracle coefficients, three ways of choosing the code:

| rank k | PCA (L2-optimal) | α-optimised linear encoder | **per-sample α-optimal code** |
|---|---|---|---|
| 8 | 0.137 | 0.342 | — |
| **16** | **0.179** | **0.394** | **0.574** |
| 32 | 0.227 | 0.426 | — |

**§9.2 is retracted.** It reported that "an FIS emitting 8–32 numbers caps at
α ≈ 0.14–0.23 whichever space it emits them in", and that conclusion was an
artifact of measuring an L2-optimal basis. With the code chosen against α, 16
numbers reach **0.574** — 3.2× the reported figure, and above the acceptance of
every cheap drafter measured in §8.

An α-optimised rank-8 basis (0.342) beats a PCA rank-32 basis (0.227). The
information needed for a competitive drafter fits in a handful of numbers. That
was never the obstacle.

(The α-optimised *linear encoder* row is itself an intermediate: it constrains
the code to be linear in `h`. Freeing that gives the 0.574. Both are ceilings —
they use the true `h` — and both are far above PCA.)

### 12.2 But refining the *predictor* against α does not help

The same treatment applied to §10's context predictor:

| | α |
|---|---|
| ridge (L2) | 0.2962 |
| refined against α | 0.2767 |
| refined against KL(p‖q) | 0.3048 |

Nothing moves. Direct α refinement is slightly *worse* — its subgradient is
sparse, since `min(p,q)` contributes no gradient wherever `q > p`. KL is the
better surrogate and buys +0.009.

**This is a clean dissociation.** The objective matters enormously for
*representation* (0.18 → 0.57) and not at all for *prediction* (0.296 → 0.305).
Which locates the bottleneck precisely: it is not the code, not the basis, and
not the fitting objective. It is that cheap context features do not determine
`h`.

### 12.3 The FIS in the architecture that suits it best

The bottleneck architecture the corrected ceiling implies:

```
cheap features --[predictor]--> z in R^16 --[decoder]--> h_hat --> logits
```

with decoder and code space learned end-to-end against α, and the predictor
being the part an FIS could be. All arms share the identical 63 features (PCA-32
of the context bag, PCA-16 of the previous hidden state, the tier-A scalars),
the identical decoder, and the identical per-sample α-optimal code targets — so
the only thing varying is the function class. The feature set is deliberately
modest: handing an FIS 2,304 raw embedding dimensions would destroy the
readability that is its reason for being here, and walk into Cui, Wu & Xu
(arXiv:2102.04271).

| arm | α |
|---|---|
| oracle code, same decoder | **0.574** |
| end-to-end linear (α-trained) | 0.212 |
| GBM, 16 models | 0.197 |
| linear, same targets | 0.181 |
| **`TribbleRegressor`, 16 models** | **0.130** |

The FIS is the weakest arm, below a plain least-squares linear map on identical
targets. This is the most favourable framing the study could construct for it —
few outputs, a modest interpretable feature set, the right code targets, a
decoder tuned to the objective — and it still loses.

The gap that matters is 0.574 against 0.212: a **2.7× ceiling-to-achieved gap**
that no function class in the comparison closes. Representation is solved;
prediction from cheap features is not.

## 13. Experiment 8 — the gap is information, not capacity

§12 left a 2.7× gap: oracle rank-16 code 0.574, best predictor 0.212. Two
candidate causes — the features determine `h` and every predictor tried was too
weak (function-class limit), or the information is not in the features at all
(feature limit). Every predictor so far was linear in the features, additive in
them, or a rule base over eight of them, so the nonlinear arm was missing.

Trained against KL(p‖q), which §12.2 measured as the better surrogate. Two
feature sets so the two questions do not confound: **reduced** is §12.3's
identical 63 features (isolates function class), **full** is the raw
bag-of-embeddings plus the previous hidden state (says whether the PCA
reduction was itself the bottleneck).

| arm | features | params | α | µs/step |
|---|---|---|---|---|
| oracle code, same decoder | — | — | **0.574** | — |
| MLP 512×512 | full | 1,746k | **0.2340** | 74 |
| MLP 256 | full | 742k | 0.2200 | 32 |
| MLP 512×512 | reduced (63) | 304k | 0.2161 | 108 |
| end-to-end linear (§12.3) | reduced (63) | 1k | 0.2118 | — |
| MLP 256 | reduced (63) | 20k | 0.2005 | 106 |
| GBM, 16 models | reduced (63) | — | 0.1966 | — |
| `TribbleRegressor`, 16 models | reduced (63) | — | 0.1300 | — |

**It is a feature limit.** On identical features, 304k parameters of
nonlinearity buy **+0.004** over a linear map (0.2161 vs 0.2118). Unreducing the
features and going to 1.75M parameters — 87× the capacity — buys **+0.022**
total (0.2340). The gap to the oracle stays at **2.5×**.

The scaling curve settles it: 20k → 1,746k parameters moves α by 0.033. Closing
0.234 → 0.574 on that trend needs orders of magnitude more, at which point the
predictor is a transformer, which is EAGLE.

Two secondary readings. §12.3's PCA reduction was *not* the bottleneck — it cost
0.02 of α, not the missing 0.34. And cost was never the binding constraint:
these run at 32–108 µs against a ~5–10 ms target forward pass, so an arm that
worked would comfortably have paid for itself. It is accuracy that fails, not
speed.

### 13.1 Why the features cannot contain it

`h_t` is a deterministic function of the full token context, so an oracle of
the context reaches α = 1. But the features here are a **bag** of embeddings —
they discard order and composition. Recovering what the transformer computes
from an order-free summary is the thing that cannot be done, and adding
capacity to a function of the wrong statistic does not fix it. Sequence
modelling is not an implementation detail of the drafter; it is the drafter.

## 14. What round 2 has to do

1. **Experiment 2** — FIS against ridge, GBM, a small MLP, and the single-feature
   arm on tier A+B, at equal search budget. The MLP is in the set because
   that is what the entropy-gating literature actually uses.
2. **Experiment 3** — build the V2 loop: a real cheap ranker (n-gram /
   prompt-lookup), `q` materialised per `DESIGN.md` §2.1 so exactness is kept,
   shape from {oracle, FIS, fixed temperature}. The oracle arm separates "the
   FIS's shape is bad" from "shape does not help even when perfect".
3. **Replace the oracle ranker with a real one.** Every α in §2 assumes the
   candidate set contains the true top-k. A real n-gram ranker will not, and the
   gap between 0.848 and what it actually delivers is the honest headline for V2.
