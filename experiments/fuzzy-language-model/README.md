# Experiment: how small can a CPU-trained language model be — and can it be a fuzzy system?

**Status:** in progress: tune complete, scaling running · **Started:** 2026-10-08

## Questions

1. **Size.** How small can a causal language model trained on TinyStories, on a
   laptop CPU, be and still model the text? How does the answer differ between
   quadratic (softmax) attention and linear-cost (linear-attention / recurrent)
   sequence mixers?
2. **Fuzzy.** Can every sublayer of such a model be a Takagi–Sugeno–Kang fuzzy
   inference system? That means a **Fuzzy Language Model (FLM)**, with
   quadratic-cost TSK attention and TSK feed-forward, and a **Fuzzy Recurrent
   Language Model (FRLM)**, with a fixed TSK rule base whose consequents are
   recurrent state. What does it cost in quality per parameter?
3. **Weights ↔ rules.** What is the exact relationship between the neural mixers'
   weights and fuzzy rules, and can a trained FLM/FRLM be read back as a rulebook?

## Read in this order

1. [`theory.md`](theory.md): the identities. Softmax attention is a weighted
   zero-order TSK system with one rule per token. Linear attention is a TSK system
   with fixed rules and recurrent consequents. The delta rule on fuzzy features is
   in-context LMS training of TSK consequents. Each is pinned by a test.
2. [`literature/`](literature/README.md): three reviews with verified citations,
   including the prior-art check. **"First fuzzy LM" is not a defensible claim**;
   the defensible claim is narrower (see `literature/03`).
3. `RESULTS.md`: hypotheses as registered before each run, and how they scored.

## Design

| choice | value | why |
|---|---|---|
| data | TinyStories V2 (GPT-4 split), HF `roneneldan/TinyStories` @ `f54c09f` | the standard tiny-LM corpus; pinned revision |
| tokens | 98 characters (printable ASCII + `\n` + EOT + UNK) | at 20K params a subword vocabulary would *be* the model; characters make the embedding ≈3K params |
| metric | validation bits per character, first 1M chars of the V2 valid file, non-overlapping 256-char windows | tokenizer-free, identical for every arm |
| context | 256 characters | |
| training cost | linear mixers train in their parallel (T×T) form; at T = 256 that is *not* cheaper than fused softmax attention on CPU. "Linear" buys O(1) inference state (`flm/state.py`), not cheaper training at this length | |
| hardware | Intel i7-1185G7 (4 cores / 8 threads), CPU only, **1 thread per run**, 4 runs in parallel | the question is about CPU training |
| budget | equal **characters** per run (10M for tuning, 30M for scaling) | same data for every arm; wall-clock reported separately |
| tuning | every arm gets the *same* search: lr ∈ {1e-3, 3e-3, 1e-2, 3e-2, 6e-2, 1e-1} × short-conv ∈ {off, 4}, d = 32, seed 0, 10M chars, picked on val BPC. The lr grid was extended twice, for every arm, when optima hit the edge (`RESULTS.md`) | an unequal search budget can flip a comparison |
| seeds | stated per table; never pooled across unequal seed sets | |

### Arms

| arm | sequence mixer | channel mixer | cost in context |
|---|---|---|---|
| `softmax-mlp` | softmax attention + RoPE | GELU MLP | quadratic |
| `linear-mlp` | linear attention, elu+1, fixed per-head decay | MLP | linear |
| `gla-mlp` | linear attention, data-dependent decay | MLP | linear |
| `delta-mlp` | DeltaNet (delta rule), data-dependent decay | MLP | linear |
| `gru` | 2-layer GRU (no attention, no FFN) | — | linear |
| **`flm`** | Gaussian-TSK attention (1 rule per token) | TSK | quadratic |
| **`frlm-acc`** | fixed Gaussian rule base, accumulating consequents | TSK | linear |
| **`frlm-delta`** | fixed Gaussian rule base, LMS-fit consequents | TSK | linear |
| **`frlm-acc-htsk`**, **`frlm-delta-htsk`** | as above, with HTSK firing (exponent averaged over dims; Cui, Wu & Xu 2021) | TSK | linear |

The two `-htsk` arms were added after a design iteration (`fuzzyfix`, see `RESULTS.md`).
They got extra search the neural arms did not, which is disclosed wherever they are
compared, and then the same tune grid as everyone else.

## Layout

| path | what |
|---|---|
| `flm/data.py` | fetch + convert TinyStories to the 98-symbol stream (`data/tinystories/`, gitignored) |
| `flm/models.py` | all mixers, FFNs, the model, data-driven rule init |
| `flm/train.py` | one run → one JSON record (config, params, curve, val BPC, wall-clock, sample) |
| `flm/sweep.py`, `flm/grids.py` | resumable grid runner; grid definitions |
| `flm/analyze.py` | sweep → `summary.{md,csv}` + BPC-vs-params plot |
| `flm/rules.py` | trained checkpoint → Markdown rulebook |
| `flm/state.py` | inference-state size per arm vs context length |
| `scripts/chain*.sh` | the unattended sweep chains as run (tune → edge check → scaling → ablate → headline) |
| `requirements.lock` | `uv pip freeze` of the environment the runs used |
| `test_models.py` | the identities in `theory.md`, parallel ≡ recurrent forms, causality |

## Reproduce

```bash
cd experiments/fuzzy-language-model
uv venv --python 3.12 .venv
VIRTUAL_ENV=.venv uv pip install torch --index-url https://download.pytorch.org/whl/cpu
VIRTUAL_ENV=.venv uv pip install numpy pandas pyarrow huggingface_hub tokenizers matplotlib pytest
.venv/bin/python -m flm.data                 # fetch + convert (≈2.2 GB, ~10 s to convert)
.venv/bin/python -m pytest -q                # identities
.venv/bin/python -m flm.sweep tune --workers 4
.venv/bin/python -m flm.sweep scaling --workers 4   # uses each arm's best tune cell
.venv/bin/python -m flm.analyze scaling --threshold 2.0
```

### On a GPU host (the ten-seed headline grids)

```bash
VIRTUAL_ENV=.venv uv pip install torch --index-url https://download.pytorch.org/whl/cu130   # CUDA wheel instead of CPU
.venv/bin/python -m flm.data
.venv/bin/python -m flm.sweep headline10   --device cuda --workers 8   # 80 runs, seeds 0-9 at d=32
.venv/bin/python -m flm.sweep headline10v2 --device cuda --workers 8   # 20 runs, the HTSK FRLM arms
git add -f outputs/headline10/*.json outputs/headline10v2/*.json        # records only, not logs
```

The models are tiny, so one GPU holds many workers; raise `--workers` until the GPU is
busy. Hyperparameters come from the committed CPU `tune` records. Every run records
`device`/`host`, and `flm.analyze` never pools devices into one cell.

Outputs land in `outputs/` (gitignored, regenerable). Run records worth keeping
are force-added (`git add -f`), as AGENTS.md describes.
