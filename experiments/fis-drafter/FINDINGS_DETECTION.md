# A sixth confound: the grader manufactures the result

Hallucination detection, three small models, 5,400 graded generations. The
headline is not a detector. It is that **the choice of automatic grader moves
the reported number by more than the gap between competing detectors**, and
that one popular grader family *creates* the confound that
`experiments/fuzzy-lm-anomaly.md` lists first.

This is upstream of the five confounds in
`papers/hallucination-detection-confounds.md`: those contaminate the *features*,
this one contaminates the *labels*.

---

## Setup

Each dolly `closed_qa` item is asked twice — once with its context paragraph
(the answer is readable off the page), once without (recall or fabricate).
Identical question wording, identical answer format, only the evidence differs.
Greedy decoding, 40-token cap.

**The label is measured correctness, never the condition.** Labelling by
condition would let any detector win by reading prompt length. Wrong
`with_context` answers are positives exactly like wrong `no_context` ones.
AUROC is reported *within condition* throughout, so no arm can score by
detecting which condition a generation came from.

| model | n | wrong % | mean answer length |
|---|---|---|---|
| SmolLM2-360M-Instruct | 1,800 | 51.9 | 30.5 tok |
| SmolLM2-135M-Instruct | 1,800 | 55.8 | 35.5 tok |
| gemma-3-270m-it | 1,800 | 64.5 | 40.0 tok (never emits EOS) |

---

## Finding 1 — five defensible graders disagree, and the detector's score moves with them

All five are choices a competent person would make. Agreement between pairs
runs from **0.543 to 0.902**, and the prevalence of "wrong" swings **0.274 to
0.731** — a 2.7× range on the same 1,800 generations.

The same detector (`entropy__mean`), scored under each:

| grader | AUROC (within-condition) |
|---|---|
| `f1 < 0.5` | **0.686** |
| `content_word` | 0.647 |
| `ref_recall < 0.5` | 0.631 |
| `f1 < 0.3` | 0.621 |
| `ref_recall < 0.3` | **0.597** |

**Spread: 0.089 from the grading choice alone.** For scale, the detector gaps
this literature argues over are smaller than that — in
`experiments/fuzzy-lm-anomaly.md`, isolation forest vs one-class SVM is 0.001,
FIS vs isolation forest 0.018, Mahalanobis vs `ent_max` 0.094.

A paper reporting a single AUROC without validating its grader is reporting a
number with roughly ±0.05 of uncontrolled slack.

## Finding 2 — F1-style graders *manufacture* the length confound

The top of the detector ranking is stable across graders. The bottom is not,
and the arm that moves is the one that matters most:

| detector | rank under `ref_recall` graders | rank under `f1` graders |
|---|---|---|
| `renyi2_mean` | 1 | 1–2 |
| `ent_mean` | 2 | 1–2 |
| **`n_tokens`** | **6 (last)** | **3** |

Answer length goes from useless to third-best purely by changing the grader:

| model | `n_tokens` AUROC, F1 grader | `n_tokens` AUROC, recall grader | Δ |
|---|---|---|---|
| SmolLM2-360M | 0.648 | 0.545 | **0.103** |
| SmolLM2-135M | 0.669 | 0.558 | **0.111** |
| gemma-3-270m | 0.500 | 0.500 | — (degenerate, see below) |

**The mechanism is the precision term.** Token-F1 penalises a long answer even
when it fully contains the reference; recall does not. So an F1 grader writes
answer length into its own labels:

| grader | corr(length, label) | `n_tokens` AUROC |
|---|---|---|
| `ref_recall < 0.3` | 0.016 | 0.518 |
| `ref_recall < 0.5` | 0.077 | 0.545 |
| `content_word` | 0.115 | 0.566 |
| `f1 < 0.3` | 0.201 | 0.619 |
| `f1 < 0.5` | 0.229 | 0.648 |

corr(`token_f1`, length) = **−0.264** against corr(`ref_recall`, length) =
**−0.126**: the F1 metric is twice as length-entangled, exactly as its
definition predicts.

**Consequence for the existing paper.** Confound #1 there is "answer length:
`n_tokens` alone reaches AUROC 0.843, remedy exact matching on token count."
Part of that may not be a property of hallucination at all — it may be
manufactured by a length-sensitive grading metric. The remedy is then not only
matching on length downstream, but choosing a length-neutral grader upstream.
**That is worth re-checking before submission**, and it is cheap to check: score
the same detectors under a recall-based grader and see whether the 0.843 moves.

## Finding 3 — `ent_max` over `ent_mean` does not generalise

That study's one practical recommendation — "use maximum per-token entropy, not
mean", `ent_max` 0.883 vs `ent_mean` 0.767, better in 61/66 cells, written up
as "a one-line change for anyone working on hallucination detection" —
**reverses here**, and does so robustly:

* across all **five graders** (`ent_max − ent_mean` = −0.023, −0.031, −0.032,
  −0.040, −0.043 — negative every time);
* across all **four answer-length bins** (−0.015 to −0.065), so it is not a
  length effect;
* across models: −0.031 (360M), −0.030 (135M), −0.004 (gemma, neutral).

This is not a refutation of the original result *in its own setting* — different
task, different label definition, one dataset against six models. It is evidence
the recommendation is **setting-dependent**, while the write-up states it
without qualification.

## What did not work (recorded so it is not retried)

Three attempts to beat a single entropy scalar, all failed:

| arm | within-condition AUROC |
|---|---|
| `entropy__mean` / `renyi2__mean` (single scalar) | **0.631–0.633** |
| logistic regression on entropy family | 0.608 |
| supervised probe on raw hidden state (960-d) | 0.634 |
| probe + entropy | 0.652 |
| hand-crafted geometry (manifold dist., Mahalanobis, layer-norm shape) | 0.560 |
| entropy + geometry | 0.557 |

The geometric family — the new idea this round was built on — is *worse* than
entropy, and adding it to entropy makes entropy worse. A supervised probe on the
raw hidden state, which is what the semantic-entropy-probe literature uses, only
ties. **The prior study's core lesson replicates: a single scalar is at or near
the ceiling of what is extractable here.**

## Limits

* One task (dolly `closed_qa`, context present/absent), three small models.
* Automatic graders only. This work shows the graders *disagree* and that one
  family is length-entangled; it cannot say which is correct. Establishing that
  needs human labels, and that is the obvious next step.
* gemma-3-270m never emits EOS inside the 40-token cap, so its `n_tokens` has
  zero variance and its 0.500 is degenerate rather than a counterexample. It is
  uninformative for Finding 2 and reported as such.
* AUROCs here (0.60–0.69) are well below the 0.87 of the prior study. Some of
  that is label noise, which is precisely Finding 1's point.

---

# Part 2 — "fMRI of an SLM": one-class anomaly detection on the activation atlas

A different regime, and the first one in this project that suits the FIS. Not
regression (predict a value, where the FIS kept losing) but **one-class anomaly
detection** — learn "normal" activity from innocuous prompts, flag deviation —
which is exactly what tribble's "none of the above" rule was built for. And
unlike hallucination, a malformed prompt is a *large* distribution shift.

## Method

One forward pass per prompt over SmolLM2-135M, capturing at every layer
l = 0..30 the masked-mean hidden state over the prompt's tokens — an
`(n_layers+1, hidden)` activation atlas per prompt. Detectors are fitted on
**innocuous activations only** (true one-class): the FIS anomaly rule
(per-feature Gaussian memberships, product t-norm, complement), Mahalanobis in
whitened activation space, and isolation forest. Features are per-layer PCA
coordinates plus per-layer norms.

The battery is 175 innocuous questions (dolly + hand-written) against nine
malformation types, split into **surface** (random characters, random vocab
ids, degenerate repetition — abnormal tokens) and **structural** (a real
question with its words shuffled, a grammatical-but-absurd sentence, an
instruction-injection — near-normal tokens, broken arrangement or meaning). The
structural types are generated from a held-out slice of the innocuous pool, so
their token content is matched to the normal class.

## Result 1 — it works, and activations add real value over reading the input

One-class AUROC, held-out normal vs each malformation:

| detector | all | structural | surface |
|---|---|---|---|
| Mahalanobis (activations) | 0.857 | **0.809** | 0.988 |
| FIS anomaly (activations, raw) | 0.799 | 0.671 | 0.986 |
| Mahalanobis (surface-only control) | 0.792 | 0.682 | 0.984 |

Activations beat the surface-only control on **structural** malformation by
**+0.127** (0.809 vs 0.682). Surface types are easy for everything (~0.99) and
are a sanity floor, not the result. Per type, the activation-over-surface gain
concentrates exactly where it should — where the tokens are normal:

| type | activation | surface | gain |
|---|---|---|---|
| nonsense (grammatical, absurd) | 0.722 | 0.325 | **+0.397** |
| char_scramble | 0.904 | 0.677 | +0.227 |
| word_salad | 0.742 | 0.604 | +0.138 |
| injection | 0.946 | 0.930 | +0.016 |
| char_gibberish | 1.000 | 1.000 | 0.000 |

`nonsense` is the sharpest: grammatical sentences with normal words that mean
nothing score *below* chance on surface features (0.325 — they look more normal
than normal) yet the activation atlas flags them at 0.722.

## Result 2 — the micro-dose: a clean dose-response, surface held at chance

The definitive control the project's history demands. Eighty base questions,
each emitted at increasing word-shuffle fraction 0.0 → 1.0. Every dose of a base
is a **permutation of the same words**, so surface statistics are identical by
construction. Detector trained on clean prompts only:

| shuffle dose | Mahalanobis AUROC | FIS-whitened AUROC | surface AUROC |
|---|---|---|---|
| 0.15 | 0.593 | 0.593 | 0.414 |
| 0.30 | 0.577 | 0.577 | 0.413 |
| 0.50 | 0.612 | 0.612 | 0.414 |
| 0.70 | 0.648 | 0.648 | 0.413 |
| 1.00 | **0.685** | **0.684** | 0.418 |

The anomaly score **rises monotonically with the degree of disorder** while the
surface control stays pinned at chance. This is the concept — watch, learn
normal, flag deviation — demonstrated on perfectly matched content: the internal
activity encodes syntactic well-formedness that the input tokens cannot reveal,
and the deviation is graded, not binary.

## Result 3 — the FIS is finally competitive, with one caveat

On raw activation features the FIS anomaly rule loses to Mahalanobis (structural
0.671 vs 0.809) — its per-feature independence assumption breaks on correlated
activations, the same weakness seen throughout. Given **decorrelated
(whitened)** features it matches Mahalanobis exactly (0.809, and identical at
every dose above).

This is the first regime in the whole investigation where the FIS mechanism is
competitive with the best baseline. The honest caveat: whitening is itself most
of what Mahalanobis does, so "the FIS matches Mahalanobis once you whiten" is
partly circular. What it establishes is narrower but real — the "none of the
above" rule is a *sound* one-class detector here, not a broken one, provided its
inputs are decorrelated first.

## Honest limits and one confound found

* **Effect sizes for pure structure are modest** (word-order AUROC 0.59–0.69).
  The large numbers (gibberish ~1.0, injection ~0.95) are surface-detectable and
  do not need the atlas.
* **Word-shuffling is not perfectly token-matched.** Per-layer localization of
  the word-salad signal peaks at layer 0 (0.688) and the final layer (0.812),
  not the middle. A layer-0 (embedding) effect on *mean-pooled* activations
  should be impossible for a pure reordering — mean-pooling is order-invariant —
  which means shuffling perturbs the BPE token *boundaries* slightly and leaks a
  small surface signal. The final-layer signal (0.812) substantially exceeds it,
  so genuine deep detection is present, but the clean "middle-layer semantic"
  story from the first small run did **not** survive scaling and is withdrawn.
* One model, automatic battery, one-class Ns of ~100 train / 70 test.

## Why this direction is worth continuing

Unlike drafting (closed) and hallucination (entropy is hard to beat), this
regime has three things going for it at once: the task suits the FIS's actual
strength, the signal is real and survives the matched control, and the strongest
detected class — **instruction-injection at 0.946** — is a security-relevant
application where an interpretable, cheap, one-class monitor that needs no
attack examples is genuinely useful. The plan below pursues that.

---

# Part 3 — Prompt-injection detection on a real corpus (Phase 1)

The application from `PLAN_ANOMALY.md`, run on `deepset/prompt-injections` (343
benign, 203 injection, real prompts). One-class throughout: the detector is
fitted on benign activations only and never sees an injection in training.

## The confound, and why every number is reported within-length

Public injection corpora are severely length-confounded (here injection median
65 tokens vs benign 42; jailbreak sets are far worse). A detector that only
reads length would look excellent and mean nothing. So the headline metric is
**within-length AUROC** — computed inside token-count deciles and pooled — with
length-alone and a surface-only baseline reported alongside.

## Result — activations beat length, and beat surface, after controlling length

Eight seeds, one-class (train on benign only), within-length AUROC:

| detector | within-length AUROC |
|---|---|
| **activation atlas** (Mahalanobis = FIS-whitened) | **0.874 ± 0.014** |
| surface-only (length + token stats) | 0.799 ± 0.024 |
| length alone | 0.560 ± 0.023 |

* **Length alone collapses from 0.830 pooled to 0.560 within-length** — direct
  confirmation that the pooled score was mostly the length confound.
* **Activation beats surface by +0.076 ± 0.033, winning 8 of 8 seeds.** This is
  the confound-controlled value of watching activations over reading the input.
* **The FIS "none of the above" rule equals Mahalanobis exactly** (0.936 pooled,
  0.892 within-length at the reference seed) once features are whitened.

## Length-matched confirmation

A benign/injection subset matched on token count (benign median 58 vs injection
57; length AUROC 0.470, i.e. at chance by construction):

| detector | AUROC |
|---|---|
| activation | **0.894** |
| surface | 0.865 |
| length | 0.470 |

With length neutralised to chance, activations still lead. The margin narrows
(the matched set's residual signal is token content, which surface also reads),
but the ordering holds.

## The selling point, delivered

A supervised detector that *has* seen attacks (5-fold logistic regression on
activations) reaches 0.925 within-length. The one-class monitor reaches 0.874–
0.892 **with no attack examples at all** — within ~0.03 of the supervised upper
bound. That is the case for the method: unsupervised, needs no attack corpus,
and within a whisker of the supervised ceiling.

## Where injection lives in the network

Per-layer one-class within-length AUROC rises monotonically from the embedding
layer to the output:

| layer | 0 | 6 | 12 | 21 | 30 |
|---|---|---|---|---|---|
| within-len AUROC | 0.417 | 0.510 | 0.634 | 0.636 | **0.770** |

Layer 0 sits *below* chance, so — unlike the word-salad case in Part 2 — there is
no surface/BPE artifact inflating the input layer. The discriminative signal is
genuinely built up through the network and is strongest at the final layer,
which is what "the model recognises the injection as it reads it" should look
like.

## Status against the plan

Phase 1 is substantially done: real corpus, realistic benign traffic, the length
confound measured and controlled three ways (surface baseline, within-length
stratification, length-matched subset), FIS competitive with the best
unsupervised baseline, and the no-attack-examples claim delivered. Remaining
Phase 1 items: multi-model replication, and per-attack-family breakdown once a
family-labelled corpus is in. Phase 2 (per-layer FIS *attribution* as the
distinctive contribution, since AUROC ties Mahalanobis) is the next build.

---

# Part 4 — Phase 2: multi-model, operating points, and the FIS's distinctive contribution

## Replication across models and a second attack corpus

Within-length AUROC, one-class (benign only), activation vs surface vs length:

| model / corpus | activation | surface | length alone |
|---|---|---|---|
| SmolLM2-135M · deepset | 0.874 ± 0.014 | 0.799 | 0.560 |
| SmolLM2-360M · deepset | 0.863 | 0.805 | 0.562 |
| SmolLM2-135M · jailbreak | 0.708 | 0.573 | 0.541 |
| pythia-410m · deepset | see below | 0.62 | 0.56 |

The activation-over-surface gap holds on both instruct models and both attack
corpora (deepset's terse overrides, jailbreak's long role-play), after
controlling length. Length alone collapses to ~0.55 within-length everywhere,
confirming the confound was real and is removed.

## The instruct-vs-base result (a finding and a limit)

Operating points — detection rate at a fixed benign false-positive rate:

| model | det @ 1% FPR | det @ 5% FPR | FPR @ 95% TPR |
|---|---|---|---|
| SmolLM2-135M (instruct) | **0.71** | 0.80 | 0.43 |
| SmolLM2-360M (instruct) | **0.61** | 0.73 | 0.37 |
| pythia-410m (base) | 0.16 | 0.33 | 0.71 |
| surface baseline (any) | ~0.13 | ~0.36 | ~0.63 |

On instruction-tuned models the monitor catches **71% of injections while
flagging only 1% of benign traffic**. On the **base** model it collapses to the
surface baseline. This is mechanistically sensible: an instruction-tuned model
has learned to treat instructions as special, so "ignore previous instructions"
disturbs its representation; a base model just continues text. **The method
needs an instruction-following target** — which is also the only kind you would
deploy an injection monitor in front of, so the limit is tolerable, but it must
be stated.

It also sets the honest ceiling: within-length AUROC 0.87 is a *useful* signal,
not a solved problem. FPR@95%TPR of 0.43 is too high for "catch everything"; the
usable operating point is det@5%FP = 0.80 — flag 5% of benign to catch 80% of
injections, on an instruct model, with no attack examples.

## The FIS's distinctive contribution: faithful per-layer attribution

AUROC ties Mahalanobis, so the FIS must justify itself another way, and its
mechanism does: the "none of the above" score is a **sum of per-feature
contributions**, so it decomposes by layer into a per-prompt anomaly signature
that a single Mahalanobis quadratic does not provide.

1. **The attribution is faithful.** Correlation between a layer's attribution
   gap (injection minus benign) and that layer's own one-class AUROC is **0.90**
   (deepset) and **0.96** (jailbreak). The rule weights the layers that actually
   discriminate, not arbitrary ones.
2. **Injections peak deep.** The most-anomalous layer is in the deep half of the
   network for **83%** of deepset injections and **94%** of jailbreaks, against
   ~53% of benign prompts. Layer 0 sits below chance — no surface/BPE artifact,
   unlike the word-order probe in Part 2.
3. **Attack styles have distinct signatures.** deepset's terse
   instruction-overrides light up a broad mid-to-deep band; jailbreak's long
   role-play prompts are near-silent early and concentrated in the last few
   layers. Two attack families, two visibly different layer profiles — an
   interpretability output Mahalanobis' scalar cannot produce, and the kind of
   readable result the thesis argues for elsewhere.

## Status

Phase 1 and most of Phase 2 are done and hold up under the project's confound
discipline. The result is real and honest: a cheap, unsupervised,
self-explaining injection monitor that works on instruction-tuned models, needs
no attack examples, lands within ~0.03 AUROC of a supervised detector, and
whose FIS form adds faithful per-layer attribution for free. Its limits are
stated: it needs an instruction-tuned target, and its operating points make it a
strong triage signal rather than a standalone gate. That is a defensible thesis
chapter.

---

# Part 5 — Refinement, optimizers, and the metric that actually matters

Restricting to instruction-tuned targets (justified in Part 4). Question: can an
optimizer refinement stage, applied after the FIS is constructed, beat the
Part 3/4 detector? Answer: **no on AUROC by refinement, yes on AUROC by
representation — but that "yes" makes deployment worse**, which is the finding.

## Construction is free, so refinement is affordable

FIS anomaly construction on 200 benign prompts, 279 features: fit (moment
match) **0.04 ms**, score **0.3 us/prompt**. The per-layer PCA feature build
(48 ms) dominates and is shared by every detector. Refinement can cost thousands
of times more than construction and still be negligible against a forward pass —
the constraint is whether it helps, not whether it fits.

## Optimizer refinement does not beat the unrefined detector

Baseline to beat: the Part 3 detector — single-Gaussian Mahalanobis in the
joint whitened activation space, equivalently the whitened FIS "none of the
above" rule — at **0.871 ± 0.015** within-length AUROC (6 seeds), no attacks.

Everything tried lands at or below it:

| refinement | within-len AUROC | note |
|---|---|---|
| **single Gaussian (current)** | **0.871 ± 0.015** | the baseline |
| joint 2-comp GMM | 0.866 ± 0.035 | no gain |
| joint 3-comp GMM | 0.859 | overfitting begins |
| joint 5-comp GMM | 0.780 | overfits |
| robust covariance (MCD) | 0.815 | worse |
| per-layer 5-GMM + Powell layer weights (25 attacks) | 0.851 | below baseline |
| stacked few-shot [joint + per-layer], Powell (10 attacks) | 0.819 | below baseline |

The mechanism: in the joint whitened space the benign activation distribution is
effectively **unimodal Gaussian**, so a single Gaussian is the maximum-likelihood
one-class model. Mixtures add parameters that fit the benign sample's noise;
few-shot weighting fits a small attack set that does not generalise — and adding
*more* attack examples made it worse, not better (n_val 10 → 50: Powell stacked
0.819 → 0.792). This is the same lesson as the acceleration study's guard
experiment: a construction that looks improvable often is not.

**One place refinement does pay:** the *per-layer* (interpretable) FIS, whose
single-Gaussian uniform-weight form is only 0.62, is lifted to **0.85** by an
unsupervised per-layer 5-component GMM (no attacks). That does not beat the joint
detector, but it nearly closes the gap while keeping the per-layer attribution
of Part 4 — so if you want the interpretability *and* competitive accuracy, GMM
refinement of the per-layer detector is worth its ~130 ms.

## The optimizer comparison itself (when refinement is applied)

Learning per-layer weights on a small attack set, same objective (soft-AUROC),
same data:

| optimizer | within-len AUROC (n_val=50) | time | verdict |
|---|---|---|---|
| Powell | **0.771** | 26 ms | best derivative-free |
| logistic (convex proxy) | 0.711 | 1 ms | fast, close |
| differential evolution | 0.703 | 685 ms | global, no better, 25× slower |
| LDA (closed form) | 0.618 | 0.1 ms | barely moves |
| **L-BFGS-B** | 0.614 | 1 ms | **fails — stuck at init** |
| **Nelder-Mead** | 0.614 | 1 ms | **fails — stuck at init** |
| coordinate descent | 0.614 | 2 ms | fails |

The gradient methods (L-BFGS-B) and Nelder-Mead **fail outright**, stuck at the
flat uniform initialisation because the soft-AUROC gradient there is near-zero;
Powell's coordinate line-searches escape it and win. This mirrors the
acceleration study's "an analytic gradient will just be faster" — false under a
badly-conditioned objective. If a refinement stage is used, use Powell or a
convex logistic proxy, not off-the-shelf gradient descent.

## The real finding: AUROC is the wrong objective

Chasing AUROC through the *representation* (last-token readout instead of
mean-pool, more joint-PCA components) does raise it substantially — and makes
the deployable operating point worse:

| model | representation | within-AUROC | det@1%FP | det@5%FP |
|---|---|---|---|---|
| 135M | **mean-pool, jpca=32 (current)** | 0.871 | **0.53** | **0.71** |
| 135M | last-token, jpca=128 | **0.934** | 0.25 | 0.47 |
| 360M | **mean-pool, jpca=32 (current)** | 0.872 | **0.55** | 0.70 |
| 360M | last-token, jpca=128 | **0.965** | 0.36 | 0.68 |

The high-dimensional last-token Mahalanobis ranks better on average (higher
AUROC) but its benign scores are heavy-tailed (chi-squared with many dof), so
the strict 1%-FPR threshold sits high and catches *fewer* injections. Sweeping
the dimension budget confirms no last-token setting reaches the mean-pool
det@1%FP of 0.53 (last-token peaks at 0.28 around jpca=48).

**So the +0.06 AUROC "improvement" is a regression on the metric that matters.**
The unrefined mean-pool detector is already at its best deployment operating
point. For an injection monitor the objective is detection at a fixed low
false-positive rate, not AUROC, and optimising AUROC actively hurt it.

## Multi-model confirmation (four instruct architectures)

Activation vs length, within-length AUROC:

| model | activation | length alone |
|---|---|---|
| SmolLM2-135M-Instruct | 0.874 | 0.560 |
| SmolLM2-360M-Instruct | 0.863 | 0.562 |
| gemma-3-270m-it | 0.741 | 0.582 |
| TinyLlama-1.1B-Chat | **0.935** | 0.630 |

The activation-over-length signal holds across four instruct architectures, and
is strongest on the largest (TinyLlama-1.1B) — consistent with the Part 4
instruct-vs-base result that the signal tracks instruction-following capability.

## Bottom line for the user's question

* FIS construction: **~0.04 ms**, negligible.
* Optimizer refinement (GMM, robust cov, few-shot layer weights across Powell /
  logistic / L-BFGS / Nelder-Mead / diff-evo / coordinate): **does not beat the
  unrefined single-Gaussian detector** — the benign density is unimodal.
* Optimizer choice, when refining, matters: Powell/logistic work, gradient and
  simplex methods fail from the flat init, global search is slow for no gain.
* The one useful refinement is unsupervised per-layer GMM to make the
  *interpretable* per-layer FIS competitive (0.62 → 0.85).
* And the load-bearing finding: **AUROC is the wrong target**; the current
  detector is already at the best deployment operating point, and the metric to
  refine against is det@low-FPR, not AUROC.

---

# Part 6 — Parts 3–5 re-run through the genuine tribblefis library

`CORRECTION.md` established that Parts 2–5 used a reimplementation. This re-runs
the core injection result with the real
`tribblefis.one_class.TribbleOneClassDetector` (PR #105/#106, `whiten=True`) as
the FIS arm — a true one-class fit on benign activations only, no attack
examples. Same confound controls as Part 3 (surface baseline, within-length
stratification, operating points). These numbers supersede the reimplementation's
as the authoritative ones.

## Within-length AUROC, one-class, four instruct models (6 seeds)

| model | **tribble one-class** | Mahalanobis | surface | length |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | **0.853 ± 0.018** | 0.871 | 0.801 | 0.564 |
| SmolLM2-360M-Instruct | **0.844 ± 0.034** | 0.872 | 0.801 | 0.564 |
| gemma-3-270m-it | 0.746 ± 0.029 | 0.746 | **0.856** | 0.609 |
| TinyLlama-1.1B-Chat | **0.912 ± 0.016** | 0.924 | 0.832 | 0.612 |

**The core claim holds with the real library on 3 of 4 models.** The genuine
one-class FIS detector beats the surface-only and length baselines after
controlling length (by +0.05 to +0.08), with no attack examples — which is the
Part 3 result, now honestly attributable to `tribblefis`.

It is **consistently a little below Mahalanobis** (0.84–0.91 vs 0.87–0.92) — the
fuzzy per-component Gaussians are a slightly weaker density model than the exact
joint quadratic, even after whitening. That gap is the honest price of the
rule-based form.

## Two caveats the re-run surfaced, both real

**1. gemma-3-270m-it is a counterexample.** On the smallest model, *both*
activation detectors (tribble 0.746, Mahalanobis 0.746) **lose to surface token
statistics** (0.856). The earlier Part 4 write-up compared gemma's activations
only to *length* (which it beats, 0.746 vs 0.609) and never to the full surface
baseline. It should have. "Watch the activations" is **model-dependent**, and the
weakest instruct model breaks it — surface features are the better injection
detector there. This qualifies the whole approach: it needs a model whose
activations actually encode the injection, and capability seems to matter
(TinyLlama-1.1B strongest, gemma-270m the failure).

**2. The FIS detector is worse at the strict operating point.** Detection at a
fixed benign false-positive rate:

| model | detector | det@1%FP | det@5%FP |
|---|---|---|---|
| SmolLM2-135M | tribble one-class | 0.12 | 0.71 |
| SmolLM2-135M | Mahalanobis | **0.53** | 0.71 |
| SmolLM2-360M | tribble one-class | 0.00 | 0.59 |
| SmolLM2-360M | Mahalanobis | **0.55** | 0.70 |
| TinyLlama-1.1B | tribble one-class | 0.10 | 0.75 |
| TinyLlama-1.1B | Mahalanobis | **0.58** | 0.75 |

At **5% FPR the two match** (both ~0.71 catch), but at **1% FPR the fuzzy
detector collapses** (0.00–0.12 vs Mahalanobis 0.53–0.58). The `1 − max firing`
score gives benign prompts a heavy upper tail, so the strict 1%-FPR threshold
sits high and catches almost no injections. For a deployment that needs very low
false positives, Mahalanobis is the better score; the fuzzy detector is usable
only at the more permissive 5% operating point. This is the mirror image of the
Part 5 finding (there a high-dim *Mahalanobis* had the heavy tail) and it
reinforces the same lesson: rank the detector by the operating point you will
deploy at, not by AUROC.

## Net, stated honestly

Through the real library, the one-class FIS injection monitor is a **genuine but
modest** result: it beats surface/length on the larger instruct models with no
attack examples, trails Mahalanobis slightly on AUROC and badly at 1% FPR, and
fails outright on the smallest model where surface wins. Its live advantages
remain the ones AUROC does not show — it is unsupervised, and (Part 4) it yields
faithful per-layer attribution. The honest headline is not "the FIS detects
injections best" but "an interpretable, unsupervised FIS monitor is competitive
at a 5%-FPR operating point on capable instruct models, and says which layers
fired."

---

# Part 7 — Optimizer refinement, timing, and the ROC sweep

## Construction and inference performance

The genuine `TribbleOneClassDetector`, one-class, whitened, 32 components:

| model | layers | hidden | features | train (benign) | construct (ms) | infer (µs/prompt) |
|---|---|---|---|---|---|---|
| SmolLM2-135M | 30 | 576 | 32 | 205 | 110 | 11.1 |
| SmolLM2-360M | 32 | 960 | 32 | 205 | 110 | 10.9 |
| gemma-3-270m-it | 18 | 640 | 32 | 205 | 116 | 10.9 |
| TinyLlama-1.1B | 22 | 2048 | 32 | 205 | 115 | 10.8 |

Construction is ~110 ms, dominated by the shared per-layer PCA feature build;
the fuzzy fit itself is ~0.04 ms (Part 5). Inference is ~11 µs/prompt. Both are
negligible against a model forward pass — the detector is effectively free.

## Refining TribbleFIS with the `optimizers` package — it does not help

The `refine_method="optimizers"` idea (population + local-polish GA/PSO/ACO from
the `optimizers` package), applied to the one-class detector: extract the
model's Gaussian antecedent parameters and search them against a validation
objective (AUROC, and a low-FPR-weighted objective aimed at the det@1%FP
weakness), with L2 shrinkage toward the heuristic and a disjoint test split.

Two implementation notes worth recording:
* The installed `optimizers` build uses a **keyword-only** constructor, so
  tribblefis's own `_run_optimizer_search` (positional) does not drive it — the
  same incompatibility behind that repo's 5 pre-existing `optimizers_backend`
  test failures. The package had to be called directly against its current API.
* The low-FPR objective is **degenerate at the start** (pAUC@1%FP = 0 for the
  heuristic), so the derivative-free search has no gradient to follow there and
  wanders; only an AUROC-anchored objective is searchable.

The result, across seeds and shrinkage settings:

| l2 shrink | unrefined (AUROC / det@1%FP) | refined (AUROC / det@1%FP) |
|---|---|---|
| 0.00 | 0.847 / 0.224 | 0.847 / 0.224 |
| 0.02–0.15 | 0.847 / 0.224 | 0.847 / 0.224 |

**Refinement is a no-op with the guard, harmful without it.** Removing the
"never worse than heuristic" fallback, the GA moves *away* from the
moment-matched start and its validation AUROC gets **worse** (0.883 → 0.792 at
l2=0, → 0.493 with shrinkage), dragging test with it (0.942 → 0.768). The
moment-matched Gaussians on whitened components are already a strong local
optimum; population search around them finds worse points, and the guard
correctly falls back to the heuristic.

This is the third independent confirmation of the project's recurring result —
Part 5 (few-shot layer weights), tribblefis's own refinement-guard evaluation,
and now the `optimizers`-package antecedent search — that **good construction is
not improved by a refinement stage here**. The lever that raised performance was
never the optimizer; it was the representation (Part 5) and, for the one-class
detector specifically, whitening (Part 6 / PR #106).

## The sweep

A published figure — ROC curves (detection quality vs false-positive rate) for
all four models, with the 5%-FPR operating point marked, plus the operating-
point and timing tables — is at the artifact link. It makes the Part 6 caveats
visual: the curves are strong on TinyLlama-1.1B and SmolLM2, gemma sits lowest,
and every curve rises steeply only after the 5%-FPR line, which is why 1%-FPR
detection is weak.
