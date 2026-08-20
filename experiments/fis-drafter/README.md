# FIS for LLM inference — drafting, hallucination, and an injection monitor

A sequence of demonstrations probing where a Fuzzy Inference System (TribbleFIS)
can and cannot help inside LLM inference. Three investigations, each carried to a
falsifiable end, with the negative results kept because they are the load-bearing
ones. Full write-ups are in the findings docs; this README is the runnable index.

> **⚠️ Submodule-pin caveat (2026-08-20) — regenerate before citing.**
> Every committed run summary under `runs/` was generated against
> `tribble-fis` at `1f7bb0d` (the `feat/one-class-scoring-fewshot` branch,
> where the non-saturating `surprisal`/`trimmed` scores, Ledoit-Wolf
> whitening, and few-shot mode were developed). That commit was never merged
> to `tribble-fis` `main` and is divergent from it, so when this branch landed
> on `grad-school` `main` the submodule pin was **left at `main`'s `141596e`**
> rather than advancing it. `141596e` carries the same one-class work merged
> upstream — `TribbleOneClassDetector` with `score ∈ {complement, surprisal,
> trimmed}` and `cov ∈ {pca, ledoit_wolf}` — so it is a functional superset,
> **but the numbers here were not re-run against it and may shift.** Re-generate
> the affected `runs/` and reconcile any moved figures before quoting a number
> in the proposal. One known behavioural difference: `few_shot` now defaults to
> `"none"` (was `"logistic"`); harmless for the benign-only fits here, which
> pass no labels, but pass `few_shot=`/`score=`/`cov=` explicitly to be safe.
> See grad-school PR #101 for the full pin discussion.

## What was investigated, and how it ended

| # | Question | Verdict | Write-up |
|---|---|---|---|
| 1 | Can an FIS **draft tokens** by predicting logit *shape*? | **Closed.** The 16-number code ceiling is fine (α=0.57) but cheap features cannot reach it — a feature limit, not a model-class or objective limit, across linear/GBM/FIS/MLP and two architectures. | `RESULTS.md` |
| 2 | Can an FIS flag **hallucinations** better than a scalar? | **No**, and a *sixth confound* found: the automatic grader manufactures the result; F1 graders create the length confound. A single entropy scalar is at the ceiling. | `FINDINGS_DETECTION.md` Part 1 |
| 3 | Can an FIS **monitor activations** to flag prompt injection? | **Yes, narrowly.** A one-class, no-attack-examples, interpretable monitor; deployable ~0.66–0.89 recall at 1% FPR on capable instruct models against plain injections. | `FINDINGS_DETECTION.md` Parts 2–12 |
| 4 | Is the monitor robust to **how the model is run**? | **Calibrate in place.** Sampling/precision/attention are null; quantization and system-prompt shifts move the benign manifold and must be recalibrated; last-token pooling is robust where mean-pool is fragile; generation/logit-shape add cost not signal. | `FINDINGS_DETECTION.md` Parts 16–19 |
| 5 | Does an **exogenous shift within a sequence** localise (and clamp) the response? | **Yes — a context-change detector + a causal ~8-dim subspace.** Within-sequence baseline needs no cross-prompt calibration (AUROC 0.80–0.93); it tracks semantic context-change (not surprise/attack); the response lives in a specific low-rank subspace that clamping causally suppresses. | `FINDINGS_DETECTION.md` Parts 20–21 |

Investigation 3 is the surviving contribution; 4–5 stress and extend it. They are
the focus of the demos below.

## Setup

```bash
cd experiments/fis-drafter
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch --index-url https://download.pytorch.org/whl/cu130
VIRTUAL_ENV=.venv uv pip install numpy scipy pandas pyarrow scikit-learn transformers datasets
VIRTUAL_ENV=.venv uv pip install -e ../../tribble-fis
```

Captured tensors (`runs/**/*.npy`, `*.parquet`) are gitignored and regenerable;
the JSON result summaries and the published-figure HTML are tracked.

## Demonstrations

Each is a standalone module under `fisdraft/`. Capture writes an activation atlas
once (`fmri_capture.py`); the analyses read it. Model set spans 50M–14B across
five architecture families; corpora are deepset / SPML / jailbreak / safeguard.

| Demo | Command | Shows |
|---|---|---|
| **Capture** | `python -m fisdraft.fmri_capture --mode injection --model-id <id> --out runs/<tag>` | Per-layer activation "atlas" for benign+injection prompts (one forward pass each). |
| **Detect** | `python -m fisdraft.injection_detect_v2 --runs runs/<tag>` | One-class FIS vs Mahalanobis vs surface vs length, within-length + operating points. |
| **1%-FPR fix** | `python -m fisdraft.improve_lowfpr --runs runs/<tag>` | The three ideas; the log-domain **trimmed** score recovers det@1%FP from ~0.1 to 0.5–0.9. |
| **Attribution** | `python -m fisdraft.fmri_attribution --run runs/<tag>` | Per-layer anomaly signature (faithful, corr 0.9+); the FIS's distinctive output. |
| **Comprehensive** | `python -m fisdraft.comprehensive` | Master table: every model × corpus, activation-vs-surface. |
| **Scaling** | `python -m fisdraft.sweep_and_time` then open `scaling_report.html` | det@1%FP and act−surf margin vs model size (50M–14B). |
| **Classifier sweep** | `python -m fisdraft.classifier_sweep --run runs/<tag>` | TribbleClassifier knob sweep; `top_n` and `refine=True` are the levers (det@1%FP → 0.77). |
| **Benign scaling** | `python -m fisdraft.benign_scaling --run runs/<tag>` | More baseline data → +AUROC (saturates ~200–800), does not fix the hard-corpus tail. |
| **Whitening ablation** | `python -m fisdraft.whiten_ablation` | Decorrelation is essential; PCA rank-32 beats full-rank; ZCA ranks but doesn't gate. |
| **Refinement** | `python -m fisdraft.optimizer_refine` | Optimizer refinement of the one-class antecedents is a no-op (no discriminative objective). |

## Headline results (injection monitor)

* **The 1%-FPR fix.** The library's `1 − max firing` score saturates over many
  whitened components; scoring in the log domain (**trimmed** surprisal sum)
  recovers det@1%FP from ~0.1 to **0.66 (deepset)** / **0.89 (SPML)** — filed as
  tribblefis #108.
* **Scale widens applicability but the strict gate plateaus.** det@1%FP rises
  0.07 (50M) → 0.67 (3B) then flat to 14B; ranking on the *hard* corpus keeps
  climbing (activation−surface margin +0.02 at 3B → +0.12 at 14B). Ranking and
  operating point dissociate.
* **It is dataset- and model-dependent.** Activations beat surface features on
  deepset and SPML and (at ≥3B) safeguard, but not jailbreak; base models rank
  fine but fail at 1% FPR — the instruct advantage is a low-FPR effect.

## Library contributions (in `tribble-fis`)

| | What | Status |
|---|---|---|
| PR #103 | IT2 regressor discarded its TSK consequents — fix + invariant tests | open |
| PR #105 | `TribbleOneClassDetector` (one-class novelty detection) + built-in whitening | open (closes #104, #106) |
| Issue #108 | `1 − max firing` saturates; add `surprisal`/`trimmed` scoring | filed |

## Published figures

* `roc_report.html` — ROC sweep + operating-point + timing tables.
* `scaling_report.html` — detection vs model size, and the activation-vs-surface crossover.

## Findings index

`RESULTS.md` (drafting) · `FINDINGS_DETECTION.md` (Parts 1–12: hallucination and
the injection monitor; 13–15: strict-gate/few-shot/covariance; 16–19:
inference-sensitivity; 20–21: within-sequence shift and the causal clamp) ·
`PLAN_INFERENCE_SENSITIVITY.md` (all axes resolved) · `CORRECTION.md` (what used
the real library vs a reimplementation, and how that was fixed) · `PRIOR_ART.md` ·
`DESIGN.md` · `PLAN_ANOMALY.md`.

## Method notes worth keeping

* **Within-length AUROC, not pooled.** Every injection corpus carries a length
  confound (deepset injections longer, safeguard/SPML much longer); pooled
  numbers are not comparable across corpora. All cross-corpus claims use
  within-length stratification, with surface-only and length-only baselines.
* **The grader can manufacture the result** (Part 1). Choose a length-neutral
  grader before trusting any hallucination-detection AUROC.
* **A HF XET-transfer bug** left model shards `.incomplete` despite exit 0;
  `HF_HUB_DISABLE_XET=1` is the workaround for the larger downloads.
