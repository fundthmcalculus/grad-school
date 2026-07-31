# `reproduce/` — reproduction harness

Goal: make reproducing the grad-school experiments trivial for someone who just
cloned the repo. Some experiments need hardware you may not have (a CUDA GPU, a
big-memory node) — those are marked and skipped cleanly rather than failing.

## Right now: experiment tables

The current focus is the proposal's **experiment tables**. See
[`tables/README.md`](tables/README.md). In short, from the repo root:

```bash
uv run --project tribble-fis     python reproduce/tables/table_4_1_mog_baselines.py
uv run --project tribble-fis     python reproduce/tables/table_6_1_model_family.py
uv run --project tribble-cluster python reproduce/tables/table_3_1_pvat_scaling.py
```

Each writes `reproduce/outputs/<table>.{md,csv}` — Markdown ready to drop into the
proposal, CSV for further processing. Numbers are mean ± std over fixed seeds;
`N/A` marks anything genuinely unavailable on this machine.

## Layout

```
reproduce/
  common.py        shared metrics, seed list, Markdown/CSV emitters
  manifest.py      registry of ALL experiments (for the full orchestrator; WIP)
  tables/          the experiment-table generators (current focus)
  outputs/         generated .md / .csv tables and run logs
```

## Later: the full pipeline

`manifest.py` already enumerates every experiment across the four repositories
(pVAT/clustering, MoG/tree/HME, gated-minimax, optimizers) with its command,
submodule, datasets, and hardware tier. A `run.py` orchestrator that walks that
manifest — running everything, or a chapter/tag subset, skipping hardware-gated
runs — is the planned next step but is **not built yet** (intentionally parked
while we finish the tables).
