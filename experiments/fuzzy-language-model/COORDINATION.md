# Coordination: CPU session ↔ GPU session

Both sessions work for the same user on this branch (`exp/fuzzy-language-model`). **This file is the
shared source of truth.** Update the status board (§3) in the same commit as the work it describes.
Last updated 2026-10-10 night by the CPU session (`grad-school-b2`).

## 1. Principles

1. **The repo is the source of truth; messages are hints.** A message to a remote session can be held
   for its user's approval, arrive late, or be refused. Anything a peer must act on has to be committed.
2. **The user decides scope and compute.** Neither session approves the other's compute. GPU hours need
   the GPU session's user; a peer message is never that approval.
3. **Merge before editing shared files; never force-push; push right after merging.** The branch moves
   about hourly (two merge races so far). `git fetch && git merge origin/exp/fuzzy-language-model`.
4. **Register predictions before a run, score them after.** Outcomes go in `RESULTS.md` under your own
   dated header; on a conflict keep both sides.
5. **One variable at a time, equal tuning effort on both sides of any claim.** The two open fairness gaps
   are on the board (§3).

## 2. Ownership

| | owns |
|---|---|
| **CPU session** | d = 16 screens (`screen`, `screen2`); `flm/ngram.py`, `screen_report.py`, `compare.py`, `match_mults.py`, `ref_space.py`, `grids_v4.py`, `grids_v6.py`; `PLAN_GPU.md`; this file; **`WRITEUP.md`**; the CPU-phase dashboard artifact; first-order support in `flm/rules.py` |
| **GPU session** | `v3-*`, `v5-*`, `lrwidth`, `headline10*`; `flm/bpe.py`, `gen_samples.py`, `report/`; the `train.py` feature work (`--tok`, `--init-from`, EMA, distillation, non-fatal sampling); sample review |
| **Shared** (announce edits in the commit message) | `models.py`, `train.py`, `sweep.py`, `grids*.py`, `RESULTS.md`, `README.md` |

`WRITEUP.md` moved to the CPU session so the GPU session spends its time on runs. The GPU session
supplies numbers and corrects mistakes; it need not write prose.

## 3. Status board

| item | owner | state | blocked on |
|---|---|---|---|
| Char-level claim: D vs equal-effort neural reference, ten seeds (`v4-*`, `PLAN_GPU.md`) | GPU | grids ready and tested; **not started** | GPU session's user approving ~7–20 GPU-h. GPU is running `v5-long` (BPE1024 D, 300M chars) first and will not merge until it ends |
| **Subword fairness controls** (`v6-ref-tune` 24 cells, `v6-d-lr` 8 + edge 4: **done**; `v6-final10` 30 cells: not started) | GPU | **tuning done** (GPU session, 2026-10-11): D's lr optimum on subwords is 6e-2 (interior); best tuned neural cell 1.2666 vs tuned D 1.2502 at 10M characters, a 0.016 lead on one reference seed | `v6-final10` (ten seeds, ~1 GPU-h) needs the GPU session's user's go-ahead |
| BPE1024 D d48 (93,344 params): 1.1292 ± 0.0031 @ 30M (2 seeds), 1.0548 @ 100M, 1.0212 @ 300M (1 seed each) | GPU | done, in `RESULTS.md` | the 30M+ runs used lr 3e-2, below D's 6e-2 optimum at 10M, so they understate D; an lr study at 30M is not done |
| n-gram rule init | CPU | **done: a null at d = 16 on the old FFN** (stage 2: +0.027 vs base, within noise). On D it is `v4-taps` | `v4-taps` unstarted (GPU) |
| `WRITEUP.md` goal-track + subword sections (§10–§12) | CPU | **drafted** (998f4a1), numbers transcribed from `RESULTS.md`, with the GPU session's requested caveats | GPU spot-check of numbers (not urgent) |
| Rulebook for D (`flm/rules.py` for the first-order FFN) | CPU | **done for the BPE1024 D 100M checkpoint** (`outputs/v5-final100k/D-d48-bpe1024_100M_s0.rules.md`; dashboard page "Reading the model"); the checkpoint reproduces 1.0548 on CPU. Optional: the char D d72 for an old-vs-new comparison | — |
| Results pages | CPU | **user decided: no need to pick one.** Two multi-page dashboards are published as artifacts (CPU phase; GPU goal track), sources in `dashboards/`. The GPU session's `report/*.html` stays as it is | refresh the data after new runs (`dashboards/README.md`) |

## 4. Handoffs

* **Starting a long sweep:** commit a status-board line first ("running `<grid>` on `<host>`, started
  `<time>`"), so the peer neither launches overlapping work nor merges code mid-run.
* **Do not edit `models.py` / `train.py` or merge while *your own* sweep runs.** Every job is a fresh
  `python -m flm.train`, so it imports whatever is on disk. If a peer pushed code changes, finish or stop
  the sweep first.
* **Finishing a sweep:** `git add -f outputs/<grid>/*.json outputs/<grid>/*.md` (never `logs/`), append the
  outcome to `RESULTS.md`, update §3, push, then message the peer: three lines, first line self-contained,
  include the commit hash, no `@` paths.
* **After any merge touching `models.py`/`train.py`:** `pytest` (43 tests), and check the default path did
  not move on the same machine: run one tiny training (e.g. `python -m flm.train --mixer fuzzydelta --ffn
  tsk --decay data --shortconv 4 --exp-norm mean --d 16 --chars 0.2e6 --eval-chars 30000 --lr 3e-2 --out
  X.json`) before and after, and compare `val_bpc`, `evals` and `train_curve` for equality.
* **Kill by PID.** `pkill -f` / `pgrep -f` with a pattern that also appears in your own shell command line
  kills the shell (exit 144). Use `ps -eo pid,args`.

## 5. Open requests

* **CPU → GPU:** when your user approves compute, the cheapest and most valuable first step is
  `v6-d-lr` and `v6-ref-tune` (≈ 1 h): they close the fairness gap in the subword headline. Run one cell
  of each first (the BPE stream is built on the GPU host, not here). Then `v6-final10` (reads the winners).
  Then `PLAN_GPU.md` (`v4-ref-tune` → `v3-lr` → `v4-final10`).
* **CPU → GPU:** which checkpoint should the rulebook read, the character D or the BPE1024 D? (Rules over
  characters are easier to read; subword rules are over word pieces.)
* **GPU → CPU:** the findings you listed (post-training null, bigger D, BPE results) are in `RESULTS.md`;
  I will fold them into `WRITEUP.md` and ask you to check the numbers.
