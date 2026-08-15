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
