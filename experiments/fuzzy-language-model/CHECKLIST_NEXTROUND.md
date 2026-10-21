# CHECKLIST_NEXTROUND: testing checklist for the round in `PLAN_NEXTROUND.md` (opened 2026-10-21)

Branch `exp/allfuzzy-nextround`. Item IDs are stable (never renumber). Every item names its evidence file; an item is ticked only when that file exists and was read, not when a process exited zero. Predictions are in `PLAN_NEXTROUND.md` (A5, 3.4) and are scored in `PLAN_FUZZYLOCAL.md` when the results land.

## Reporting rules (apply to every item)
- **Register before running**: the prediction and the decision rule are committed before the first cell starts.
- **Paired by seed** against the stated reference; `python -m flm.report_nextround <dir> --ref <prefix>` writes the table (`report/nextround_<id>.md` and `.json`) with per-seed deltas and the verdict. A partial seed set is reported as partial, never as a pass.
- **Gain rule**: at least 0.005 bits per character (BPC) better than the reference on at least four of five seeds; then one 300M confirmation (two seeds, warm-up 1000). No 30M gain is quoted without it.
- **Provisional results**: each finished cell is reported when it lands; the artifact dashboard and a push notification follow each result.
- **Record**: results, prediction scoring and a one-paragraph reading are appended to `PLAN_FUZZYLOCAL.md` and committed; nothing is called a win on fewer than five seeds (30M) or two seeds (300M).
- **Manipulation check** printed or stored for every variable that is supposed to change something (group sizes for A2, launch counts for A1, token counts and dropped fraction for new data).

## 0. Setup
- [x] S1 Branch `exp/allfuzzy-nextround` off `exp/allfuzzy-1block` (PR #275 unmerged). Evidence: `git branch`.
- [x] S2 Plan with datasets and registered predictions: `PLAN_NEXTROUND.md`.
- [x] S3 Reporting tool `flm/report_nextround.py`; checked against stage 10 (reproduces `proj_r16` +0.0009, `ffn_r41` +0.0046, `proj_bias` +0.0028). Evidence: command output in this session; rerun reproduces it.
- [x] S4 Group-learning-rate runner `flm/run_fuzzy_groups.py`; smoke run of `g_c3` and `g_wd0` printed the parameter groups (learning-rate multiplier and weight decay per group, with sizes). Evidence: `outputs/smoke-groups/logs/`.
- [ ] S5 Pinned `black==26.5.1` on every new `.py` (done for the two files above; repeat for each new file).

## A. Training efficiency (TinyStories)
- [x] A2.1 Run `python -m flm.run_fuzzy_groups` (4 arms x 5 seeds, 30M, `v64-groups`). Evidence: 20 cells in `outputs/v64-groups/`.
- [x] A2.2 Report against `lr6_wu600`: `python -m flm.report_nextround outputs/v64-groups --ref outputs/v53-fuzzylr/lr6_wu600 --out report/nextround_a2.md`.
- [x] A2.3 (not triggered: no arm passed) Any arm with verdict PASS: 300M confirmation, two seeds, warm-up 1000, against 0.98929. Evidence: `outputs/v65-groups300/`.
- [x] A2.4 Score predictions A5.3-A5.5 in `PLAN_FUZZYLOCAL.md`; commit.
- [x] A3.1 (not triggered) (only if A2 finds a lever) warm-up shape: centres and widths frozen for the first 300 steps; five seeds; same report.
- [x] A1.1 `flm/run_fuzzy_graph.py`: whole step as a CUDA graph (eager evaluation); fallback: block stack only.
- [x] A1.2 Numerics check: the first 200 logged losses of the graphed and the eager run agree within 1e-3; print launches per step. Evidence: `outputs/v66-graph/check.json`.
- [x] A1.3 (replaced by the 60-step timing check; 1.01x, so no 30M run) 30M graphed run (seeds 0-1) within 0.002 BPC of eager; measured seconds per 300M-equivalent; hybrid timed the same way.
- [x] A1.4 Score predictions A5.1-A5.2; commit; note the saving in `PLAN_FUZZYLOCAL.md`.

## B. Larger dataset (SimpleStories first; see `PLAN_NEXTROUND.md` 3.1)
- [x] B0.1 **Owner yes** (given 2026-10-21: "yes, download SimpleStories and start preparation, please complete this without my input") for the download: SimpleStories, `SimpleStories/SimpleStories`, revision `e63b8adc3b1a1bdc7cac5b500d150b71346b0628`, MIT, 7 train shards of about 238 MB plus a 17 MB test file = **1.68 GB** (the 431 MB `processed.parquet` is skipped). Requested in the 2026-10-21 hand-off; not yet given.
- [x] B0.2 Fetch to `$GRAD_SCHOOL_DATA/simplestories/` at the pinned revision; `data/.gitignore` recovery note; nothing over 5 MB committed. Evidence: file list with sizes and checksums.
- [x] B0.3 `flm/prep_simplestories.py`: validation split by document; ASCII filter; characters kept and dropped fraction reported; train stream of at least 1B characters if the corpus has them. Evidence: `outputs/prep_simplestories.log`.
- [x] B0.4 BPE3072 trained on the new corpus; frequency file for the tiered head (384 head tokens, 12 prototypes); characters per token reported. Evidence: tokenizer and `outputs/freq_simplestories_bpe3072.npz`.
- [x] B1.1 C0 baselines: unigram, bigram and trigram BPC on the validation slice; existing TinyStories checkpoints (hybrid and all-fuzzy at 300M and 1B) scored on it. Evidence: `report/nextround_c0.md`.
- [x] B2.1 Hybrid reference, 30M, five seeds, lr 3e-2 and 6e-2.
- [x] B2.2 All-fuzzy `e2e_tail`, 30M, five seeds, lr 3e-2 and 6e-2, warm-up 600.
- [x] B2.3 Report the gap to the hybrid at 30M (both models given the same learning-rate freedom).
- [ ] B3.1 300M confirmations: hybrid and all-fuzzy, two seeds, warm-up 1000 for the fuzzy model, checkpoints saved.
- [ ] B3.2 Decision rule (below 0.015: a TinyStories effect; above 0.04: grows with breadth; between: unchanged); scored predictions 3.4.1-3.4.5.
- [ ] B4.1 Blind coherence set on story prompts, at least 16 texts per model, two raters, scores before the key.
- [ ] B5.1 (second arm, after B3) TinyStories + SimpleStories mix, evaluated on both validation sets.
- [ ] B5.2 (far end, optional) Cosmopedia mix from the existing `data/mixed`.

## R. Reporting and housekeeping
- [ ] R1 After each result: PLAN entry, commit, dashboard section updated, push notification.
- [ ] R2 After each model change: `allfuzzy/export_weights.py`, parity check (numpy against torch, browser against numpy), demo rebuilt, gh-pages republished.
- [ ] R3 Open a PR for `exp/allfuzzy-nextround` once track A closes (stacked on #275 or on main after #275 merges); body states inherited CI failures separately from this change.
- [ ] R4 Memory: record any new standing decision (the 1B exception, the dataset choice).
