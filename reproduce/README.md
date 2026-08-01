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

To run every table generator at once and archive the result under a label, with
the submodule SHAs and seed set recorded alongside it:

```bash
reproduce/run_all_tables.sh my-label
```

## Which table comes from which script

[`PROVENANCE_MAP.md`](PROVENANCE_MAP.md) is the answer to "where did this number
come from". It has one row per numbered table in the proposal, naming the
generator, the output file, and — importantly — whether the prose currently
matches that output. Several tables predate the harness and do not yet agree with
it; the map says which, and why, rather than leaving the reader to diff by hand.

Check it before citing any table, and update it when a generator changes.

## Everything writes into this repository

Reproducing a proposal result must never dirty a pinned submodule. The Chapter 3
experiments live in `tribble-cluster` and, left alone, save their figures next to
their own source — so regenerating a proposal figure would modify a library and
file the evidence for a grad-school table somewhere the proposal cannot see.

`experiments/run_cluster_experiment.py` inverts that. The experiment code stays
in the submodule; only the destination moves here:

```bash
# one experiment, or --all
uv run --project tribble-cluster --with scipy \
    python reproduce/experiments/run_cluster_experiment.py --all
```

Figures land in `outputs/figures/cluster/`, and `git -C tribble-cluster status`
stays clean afterwards. Override the destination with `REPRO_FIG_DIR`.

The runner also fixes the invocation: these scripts do `from
experiments.blockwise_vat import ...`, which needs the submodule *root* on
`sys.path`. Run by path they raise `ModuleNotFoundError` before doing any work.

Generated figures are **gitignored for now** — regenerate rather than commit
them. The labelled run archives under `outputs/<label>/` are tracked, because
they are the evidence a later diff is taken against.

## Layout

```
reproduce/
  common.py          shared metrics, seed list, Markdown/CSV emitters
  manifest.py        registry of ALL experiments (for the full orchestrator; WIP)
  PROVENANCE_MAP.md  proposal table -> generator -> output file, with drift status
  run_all_tables.sh  run every generator, archive outputs + provenance under a label
  tables/            the experiment-table generators (current focus)
  experiments/       runners for experiments that live in submodules
  outputs/           generated .md / .csv tables and run logs
    <label>/         tracked run archives (the evidence)
    figures/         regenerated figures (ignored)
```

## Later: the full pipeline

`manifest.py` already enumerates every experiment across the four repositories
(pVAT/clustering, MoG/tree/HME, gated-minimax, optimizers) with its command,
submodule, datasets, and hardware tier. A `run.py` orchestrator that walks that
manifest — running everything, or a chapter/tag subset, skipping hardware-gated
runs — is the planned next step but is **not built yet** (intentionally parked
while we finish the tables).
