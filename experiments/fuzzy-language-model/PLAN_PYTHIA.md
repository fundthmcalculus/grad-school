# PLAN_PYTHIA: an all-fuzzy model at Pythia scale, for comparison and planning (registered 2026-10-22; plan only, nothing to be run until the owner says so)

## What we are comparing to
Pythia (EleutherAI) trained on the Pile (about 300B tokens; one epoch, and a deduplicated variant). The smallest published sizes are 14M and 70M non-embedding-inclusive "70M" (the "50M" in conversation: the nearest published checkpoints are Pythia-14M and Pythia-70M; the plan targets the 70M row and reports 14M as the floor). GPT-NeoX architecture, 2K context, a 50,277-token BPE. Every Pythia checkpoint is public at 154 steps, so loss-against-tokens curves are available for comparison at any budget.

## Do we have the training data? No, not for the Pythia recipe
Held locally under `data/` (all ignored by git, rule 8):
| corpus | local state | use here |
|---|---|---|
| TinyStories V2 (GPT-4) | 2.2 GB raw, character and BPE streams (V1024-V8192) | narrow; Pythia has never been scored on it as a reference |
| SimpleStories | 1.6 GB raw, 3.2 GB derived (BPE3072/6144) | second narrow corpus |
| Cosmopedia v2 | 4.4 GB raw, 5.0 GB derived (BPE1024) | the only broad, expository text here; synthetic |
| mixed (TinyStories + Cosmopedia) | 31 MB slice | a recipe for mixing, not a corpus |
Not held: the Pile, or any large natural web text. The Pile is about 825 GB (a 300B-token Pythia epoch is far outside the machine and outside the 300M-character ceiling).
So a "Pythia-like" run cannot reuse "all that data"; it needs a new download, which needs your yes (size, source and revision stated first).

## What can be compared honestly
1. **Loss per token on a shared validation slice.** Pythia's published numbers are on the Pile. The comparison needs both models scored on the same held-out text with the same tokenizer's bits per character (not per token). Pythia-14M/70M are scoreable on any text we hold (no training needed): C0-style scoring of the public checkpoints on TinyStories, SimpleStories and Cosmopedia validation, in bits per character. This is cheap, needs only the model download (14M about 30 MB, 70M about 160 MB), and puts our 100K models on the same axis as a real baseline.
2. **Parameter efficiency.** Our model is 99,882 parameters; Pythia-14M is about 14M (140x), Pythia-70M about 70M (700x). The honest framing is bits per character against parameter count, with Pythia's published training budget stated, not assumed.
3. **Budget.** Pythia saw about 300B tokens; we see 300M characters (about 75M tokens). A 70M-parameter fuzzy model on the 300M-character ceiling is data-starved by design; any "match" claim must say so.

## Proposed stages (none started)
- **P0 (cheap, no training):** download Pythia-14M and Pythia-70M at pinned revisions (needs your yes: files, sizes), score both on the TinyStories, SimpleStories and Cosmopedia validation slices in bits per character with a sliding 512 context (our protocol's context). Also score our F7 at 300M on Cosmopedia for the gap. Registered prediction: Pythia-70M scores better than our F7 on every held slice (95%); Pythia-14M is within 0.10 BPC of F7 on TinyStories (35%).
- **P1 (scale plan, no training):** the parameter ladder for the all-fuzzy model above 100K (1M, 10M, 50M, 70M): width, depth, rules, tail and head sizes by the shape rules found at 100K (V6144/head 640/tail 8 at 100K; the head fraction does not scale linearly, so we would screen it again), and a memory/time estimate on the 12 GB GPU. Output is a table, no run.
- **P2 (a decision, not a run):** which corpus and budget make a fair comparison: (a) Cosmopedia only (held, synthetic, expository), (b) a new natural-text sample (FineWeb-Edu 1-3 GB, as in `PLAN_CORPUS.md`), (c) the Pile subset Pythia used (too large; a pinned sample of one shard at most). Needs the owner.
- **P3 (later, only on request):** a 10M-parameter all-fuzzy model at 300M characters on the chosen corpus, against Pythia-14M scored on the same slice; then 70M.

## Standing constraints carried over
Only the fully-fuzzy model; 300M characters is the maximum budget (no 1B/3B runs unless explicitly requested); one training job at a time; floor 0; paired seeds; predictions before runs; nothing over 5 MB committed; downloads need explicit permission with filename, source and size stated.

## Open questions for the owner
1. "Pythia 50M": is the target the published 70M row (or 14M), or a custom 50M model sized like Pythia?
2. May P0 download the two small Pythia checkpoints (about 30 MB and 160 MB)?
3. Is a 300M-character budget acceptable for a model this large, knowing it is far below Pythia's data?
