# Issue: stream the training text instead of materialising it (and make the work portable to another machine)

Status: implemented as `flm/prep_pile.py` (2026-10-23; see `PLAN_PILE.md` for what changed from this draft: one token file per vocabulary instead of rolling shards, and a BPE trained on the sample instead of the Pythia vocabulary, which cannot fit under 100K parameters). The draft below is kept as written. Original status: draft, decisions of 2026-10-22 applied (the stream is the Pile mirror only; the other system will probably have a bigger GPU). Nothing here has been built or run. Related: `PLAN_PYTHIA.md` (the comparison this feeds), `PLAN_CORPUS.md` (corpus choice), `flm/data.py`, `flm/bpe.py`, `flm/prep_*.py`.

## Problem
Every corpus today is fully materialised: raw text (2-4 GB), a character stream and one BPE stream per vocabulary (3-5 GB each). TinyStories, SimpleStories and Cosmopedia already take about 13 GB under `data/`, all gitignored and recoverable only by re-running the prep scripts on the machine that has the raw files. A Pythia-scale comparison needs web-scale text (the Pile is about 825 GB; one epoch of Pythia is about 300B tokens), which can never be stored here. Our training budget is small (300M characters per run, about 75M tokens), so what we need from a big corpus is a **sample we can regenerate exactly**, not the corpus.

## Goal
A streaming data path that, given a corpus id, a pinned revision, a tokenizer and a seed, produces the same token stream on any machine, using bounded disk (a few GB of shard cache at most), and feeds `flm.train` unchanged.

## Design
1. **Source (decided: the Pile, only)**: `EleutherAI/the_pile_deduplicated`, pinned at revision `fcbfcfde4222cbb1acd1d33bad0be250ee14b1bb` (verified 2026-10-22: public, not gated, 1,650 parquet shards `data/train-NNNNN-of-01650-<hash>.parquet`, about 250 MB each, about 824.5 GB and 134M documents in all). It is the deduplicated Pile that the Pythia "deduped" models trained on, so it is the closest available source for the comparison; the non-deduplicated Pythia models saw the plain Pile, which we would state as a caveat. Streamed with `datasets` (`streaming=True`) or by fetching shards one at a time through `huggingface_hub`; never cloned. Alternatives checked and not used: `monology/pile-uncopyrighted` (a filtered derivative, licence "other"), `EleutherAI/pile` (a loading script only). Caveats to record in the manifest: the Pile includes copyrighted material (Books3 was withdrawn from the original distribution; confirm what this mirror contains before using any text in a published figure), and its licence is mixed by component, so the sample is for internal comparison only unless you decide otherwise.
2. **Determinism**: take a fixed, evenly spaced list of shards (for example every 40th of the 1,650, so the sample spans the corpus; whether the shards are internally shuffled is checked in S1 by reading the `pile_set_name` mix of a few shards); iterate them in that order; select documents by `hash(doc_id or shard:row) mod 1000 < k` so a sample is identical regardless of download speed; shuffle only inside a bounded buffer with a seeded generator (record the seed). The validation split is chosen by the same hash on a different residue class, **by document**, never by byte offset (`PLAN_CORPUS.md`).
3. **Tokenizer**: Pythia's own 50,277-token tokenizer (`EleutherAI/pythia-14m`, revision `cf967c0`) for the comparison, so token ids and bits per character line up with the published models; a corpus-specific BPE (stored with its sha256) is only a later ablation. Tokenisation happens in the stream; only token ids are written.
4. **Output**: rolling `uint16` shards (`train_000123.bin`, 64M tokens each, so about 128 MB), a `manifest.json` (source revision, shard list, document-selection rule, seed, tokenizer sha256, token and character counts, per-shard sha256), and a small `valid.bin`. Shards past the training budget are never written; shards already trained on can be deleted (the manifest regenerates them).
5. **Loader**: `flm/data_stream.py` exposes the same `load(split)` memmap contract as `flm/data.py` over the shard list, so `train.py` does not change; a run records the manifest hash in its result JSON.
6. **Resume**: the stream is a pure function of the manifest, so an interrupted prep restarts at the first missing shard (verified by sha256), and a training run restarts from its step counter.

## Sequence
1. **S1: contract and parity.** Implement `flm/data_stream.py` and a prep writer; reproduce the existing Cosmopedia `valid.bin`/`train.bin` token counts from the stream (same documents, same tokenizer) as the parity test. Exit: the stream's first 10M characters equal the materialised ones, bit for bit.
2. **S2: bounded-disk check.** Stream 300M characters of a new corpus with at most 2 GB of cache; measure throughput (tokens per second) against the GPU's consumption so data never starves the one training job.
3. **S3: a sample for the comparison.** Write the manifest for the Pile sample (shard list, selection rule, tokenizer pin), stream the 30M-character screen set and the 300M-character set.
4. **S4: score the baselines on the same slice.** Pythia-14M (done: `flm/score_pythia.py`) and later 70M, on this validation set.
5. **S5: portability.** Everything in the manifest plus the repo is enough to rebuild on another machine (see below).
6. **S6 (only on request):** a larger fuzzy model on the stream, per `PLAN_PYTHIA.md` P3.

## Moving to a different system
What travels: the git branch (code, plans, manifests under `report/streams/`), `requirements.lock`, and the HF revision pins. What does not travel and must be rebuilt: `data/` (13 GB, gitignored), checkpoints (`outputs/`, gitignored). Checklist on the new machine:
- clone the repo at the branch; `uv sync` (root) and the FLM environment from `requirements.lock`;
- set `GRAD_SCHOOL_DATA` to a disk with 20 GB free; `HF_HOME` to a cache with the same;
- verify the GPU: the recipe here was tuned on a 12 GB laptop card (RTX 4080 Laptop). A bigger card tempts a bigger batch, which changes the optimisation and the learning rate that was screened at batch 16; so first rerun the reference cell at the same settings and compare against 1.054 (F7, seed 0, 30M, floor 0), and treat any larger batch as its own arm with a fresh learning-rate screen, not as a free speed-up. The bigger GPU also changes what stream throughput must reach (S2 is measured again there);
- rebuild the stream from the manifest and compare shard sha256 against the manifest; a mismatch means a different `tokenizers`/`datasets` version or a changed upstream, not a bug to be worked around;
- **one training job at a time** stays in force; and nothing over 5 MB is committed (`data/.gitignore` records recovery).

## Risks
- Upstream datasets change under a name; pins and per-shard hashes are the defence.
- Tokenizer drift between library versions changes token ids silently; the manifest stores the tokenizer sha256 and the `tokenizers` version.
- Streaming throughput over the network can be below the GPU's appetite; S2 measures it and, if needed, prefetches one shard ahead.
- Licences differ by corpus; record each in the manifest and in `data/.gitignore`.

## Out of scope
Training on more than 300M characters (standing limit, unless you lift it for the Pythia-scale run), and any new model changes.

## Decisions recorded (2026-10-22)
- The stream is the Pile mirror only (`EleutherAI/the_pile_deduplicated`).
- The other system will probably have a bigger GPU.

## Open questions for the owner
1. Is internal-comparison use enough, given the Pile's mixed licences and the withdrawn components, or will any figure from it be published?
2. Which GPU (memory) on the other system, so S2 and the batch arm can be sized?
3. Should this be opened as a GitHub issue now, drafted from this file?
