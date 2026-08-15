# FIS for LLM inference — drafting, hallucination, and an injection monitor

A sequence of demonstrations probing where a Fuzzy Inference System (TribbleFIS)
can and cannot help inside LLM inference. Three investigations, each carried to a
falsifiable end, with the negative results kept because they are the load-bearing
ones. Full write-ups are in the findings docs; this README is the runnable index.

## What was investigated, and how it ended

| # | Question | Verdict | Write-up |
|---|---|---|---|
| 1 | Can an FIS **draft tokens** by predicting logit *shape*? | **Closed.** The 16-number code ceiling is fine (α=0.57) but cheap features cannot reach it — a feature limit, not a model-class or objective limit, across linear/GBM/FIS/MLP and two architectures. | `RESULTS.md` |
| 2 | Can an FIS flag **hallucinations** better than a scalar? | **No**, and a *sixth confound* found: the automatic grader manufactures the result; F1 graders create the length confound. A single entropy scalar is at the ceiling. | `FINDINGS_DETECTION.md` Part 1 |
| 3 | Can an FIS **monitor activations** to flag prompt injection? | **Yes, narrowly.** A one-class, no-attack-examples, interpretable monitor; deployable ~0.66–0.89 recall at 1% FPR on capable instruct models against plain injections. | `FINDINGS_DETECTION.md` Parts 2–12 |

Investigation 3 is the surviving contribution and the focus of the demos below.

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

`RESULTS.md` (drafting) · `FINDINGS_DETECTION.md` (Parts 1–12: hallucination,
injection monitor, all sweeps) · `CORRECTION.md` (what used the real library vs a
reimplementation, and how that was fixed) · `PRIOR_ART.md` · `DESIGN.md` ·
`PLAN_ANOMALY.md`.

## Method notes worth keeping

* **Within-length AUROC, not pooled.** Every injection corpus carries a length
  confound (deepset injections longer, safeguard/SPML much longer); pooled
  numbers are not comparable across corpora. All cross-corpus claims use
  within-length stratification, with surface-only and length-only baselines.
* **The grader can manufacture the result** (Part 1). Choose a length-neutral
  grader before trusting any hallucination-detection AUROC.
* **A HF XET-transfer bug** left model shards `.incomplete` despite exit 0;
  `HF_HUB_DISABLE_XET=1` is the workaround for the larger downloads.
