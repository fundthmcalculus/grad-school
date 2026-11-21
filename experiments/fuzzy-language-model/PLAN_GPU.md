# Plan for the GPU session

Written 2026-10-10 (evening) by the CPU session (`grad-school-b2`), for the GPU session to
execute. **Read this whole file, then `git fetch && git merge origin/exp/fuzzy-language-model`
before you start** (the branch moves under you; see §2).

> **Update 2026-10-10 night (CPU session).** Since this plan was written: (1) the GPU session's subword
> results (BPE1024 D d48, 93,344 params, 1.129 BPC at 30M) changed the headline, but there is **no neural
> control on the same tokenizer and D's lr was never tuned on subwords**; `grids_v6.py` closes both
> (`v6-d-lr`, `v6-ref-tune`, `v6-final10`, ~2–3 GPU-h, cheaper than P1–P3 and now the **first thing to
> run**). The char-level plan below still stands for the char-level claim. (2) The CPU stage-2 screen
> scored the n-gram init a null at d = 16 on the old FFN (`RESULTS.md`); it is still worth `v4-taps` on D.
> (3) Who owns what, the status board and handoff rules are in `COORDINATION.md`.

## 0. Where things stand

* **Your result:** fuzzy `D` (97,520 params: delta-form fuzzy mixer, 8 heads, first-order
  raw-firing HTSK TSK FFN, short-conv 8) scores **1.2736 ± 0.0106** at 30M chars (3 seeds)
  vs **1.2929 ± 0.0050** for the size-near reference (delta-mlp d72 h8, 93,272 params).
  Lead 0.0193 = **2.85 standard errors** at n = 3.
* **Why it is not yet citable.** (1) 3 seeds, and AGENTS.md makes ten the protocol. (2) The
  reference is **4.6% smaller** than D (≈ 0.006 BPC of the lead from the `scaling2` slope alone).
  (3) D got a search over heads, rules, antecedent size, short-conv, depth, FFN shape and lr; the
  reference got heads and one lr. (4) D's lr study (`v3-lr`) is unfinished.
  So the claim **"a fuzzy LM beats the matched neural reference"** is plausible and **untested at
  equal effort**. Making it solid is priority 1; everything else is secondary.
* **What the CPU session found, relevant to you** (full record: `RESULTS.md`, `WRITEUP.md`):
  * On this data a count-based **bigram / trigram / four-gram model scores 3.30 / 2.39 / 1.82**
    bits/char. A random d = 16 embedding cannot even represent n-gram distributions (rank-16
    readout 7.5 BPC vs 3.57 for an n-gram-aligned embedding).
  * **n-gram rule initialisation** (`flm/ngram.py`, `--ngram-init {1,2} --ngram-mode
    {emb,placement,full}`) is **a null at d = 16 on the old zero-order FFN**: every variant is
    within seed noise, and the init's head start is gone within ~100 of 1221 steps. That says
    nothing yet about D's FFN. It now runs on D's first-order FFN (slope weight 0, bias 1 at init,
    surplus rules inert). On D (d72, 108 rules), held-out init BPC: random 7.41, bigram `full`
    **3.94**, trigram + 2 taps `full` **3.47** (rule accuracy 0.98; with 108 rules the trigram bank
    has room, unlike the 32-rule bank at d = 16).
  * **Noise differs by design.** The old zero-order FFN at d = 16, 10M chars has seed std ≈ 0.035;
    D's stable raw-HTSK FFN has std 0.0003–0.002 at 10M chars. So 10M-char screens on **D** are
    powerful (0.01 effects are detectable with 3 seeds); on the old design they were not.
* **Ownership.** CPU session: `screen` / `screen2` (d = 16, zero-order), `flm/ngram.py`,
  `flm/screen_report.py`, the CPU-phase dashboard. GPU session: `v3-*`, `v4-*`, `lrwidth`,
  `headline10*`, `report/`. Shared (announce edits in the commit message): `models.py`,
  `train.py`, `grids*.py`, `sweep.py`, `RESULTS.md`, `README.md`, `WRITEUP.md`.

## 1. Order of work

| # | what | grid / command | cells | rough GPU time* | depends on |
|---|---|---|---|---|---|
| P0 | setup and sanity | §3 | | 10 min | |
| P1 | equal-effort neural reference | `v4-ref-tune` | 64 | ≈ 1 h | |
| P2 | finish D's lr study | `v3-lr`, then `lrwidth` | 16 + 7 | ≈ 3–4 h | |
| P3 | **the citable comparison** (ten seeds) | `v4-final10` | 40 | ≈ 2–5 h | P1, P2 |
| P4 | architecture screens on D | `v4-taps` | 21 | ≈ 1 h + 1–3 min CPU init per n-gram cell | |
| P5 | does the lead hold at other sizes | `v4-scale` | 36 | ≈ 3–10 h | P1, P2, and a real lead from P3 |
| P6 | write-up, samples, rulebook | §8 | | | P3 |

*From your own throughput (D 34K chars/s, delta-mlp 114K chars/s). Run **one sweep at a time**
with ≤ 4 workers (8 concurrent jobs ran each ~4× slower; 16-head cells overflowed 12 GB VRAM).

P1 and P4 are independent of everything and cheap, so do **P1 first** (it is the fairness gap),
then P2, then P3. Run P4 while you write up P3.

## 2. Ground rules (each one is a defect that has already happened here)

1. **Register before you run.** Append the section below (or your refinement of it) to
   `RESULTS.md` and commit it **before** the sweep starts. Hypotheses edited after seeing data
   do not count. The predictions in §4–§7 are this plan's registration; if you change one, say
   so and why.
2. **Ten seeds for anything you want quoted.** Screens at 1–3 seeds are for choosing what to run
   next. `python -m flm.compare <grid> --ref D` applies the decision rule: `real` means
   |Δ of means| > 2 standard errors. Use it; do not read means by eye.
3. **One variable at a time, equal effort for both sides.** A gain from "the fuzzy model got
   tuned more" is not a fuzzy-vs-neural result. The size-matched neural references are generated
   by `flm/match_mults.py` (within 0.11% of D's parameters).
4. **Never mix devices in a cell.** Every record carries `device`/`host`; `flm.analyze` keys
   cells on device. A CPU record in a grid directory makes the sweep skip that cell.
5. **Read the provenance, not the exit code.** A diverged run is stored as NaN; `flm.compare`
   excludes it and shows finite/total. Check `grep -c NaN` before trusting a mean.
6. **Git.** Before every push: `git fetch && git merge origin/exp/fuzzy-language-model`. Never
   force-push. Append to `RESULTS.md` under your own dated header; on a conflict keep both
   sides. Force-add only run records and summaries: `git add -f outputs/<grid>/*.json
   outputs/<grid>/*.md` (never `logs/`).
7. **Do not edit `models.py` / `train.py` or merge while a sweep runs.** Every job is a fresh
   `python -m flm.train`, so it imports whatever is on disk at that moment. Stop the sweep, merge,
   run `pytest` (41 tests) and the bit-identity checks, then restart; finished cells are skipped.
8. **Process hygiene.** `pkill -f` / `pgrep -f` with a pattern that also appears in your own
   shell command line kills the shell. Kill by PID from `ps -eo pid,args`.
9. `flm.sweep` appends `--device` after each job's argv, so pass `--device cuda` on the command
   line, not only in the grid.
10. **Empty grid = a prerequisite has not run**, never "nothing to do". `v4-final10` and
    `v4-scale` read the winners of `v4-ref-tune` and `v3-lr`; check `--list` prints 40 / 36.

## 3. P0: setup

```bash
git fetch && git merge origin/exp/fuzzy-language-model        # CPU session pushed after your last merge
cd experiments/fuzzy-language-model
.venv/bin/python -m pytest -q test_models.py                  # 41 passed (about 90 s)
.venv/bin/python -m flm.sweep v4-ref-tune --list | tail -1    # 64 jobs
.venv/bin/python -m flm.sweep v4-final10  --list | tail -1    # 0 jobs: expected until P1 ran
.venv/bin/python -m flm.sweep v4-taps     --list | tail -1    # 21 jobs
```

Reproducibility check you get for free: `v4-final10` seeds 0–2 of `D` and of `ref0-v3` repeat your
`v3-final` / `v3-ref` cells with identical argv. They should match your earlier numbers
**exactly on the same GPU** (determinism was verified on CPU; CUDA kernels may differ at the
1e-3 level). If they differ by more than that, stop and find out why before trusting anything.

## 4. P1: equal-effort neural reference (`v4-ref-tune`)

The neural field mirrors the search D got, at **D's parameter count** (MLP width padded per
config): mixer ∈ {delta, gla (linear + data decay)} × heads ∈ {4, 8}, plus softmax × 4 heads;
short-conv ∈ {4, 8, 16}; lr ∈ {2e-2, 3e-2, 4e-2, 6e-2}; plus a 3-layer d = 56 delta variant (depth
was searched for D). 64 cells, 10M chars, seed 0. 16 heads are not run (VRAM).

```bash
.venv/bin/python -m flm.sweep v4-ref-tune --device cuda --workers 4
.venv/bin/python -m flm.screen v4-ref-tune | head -12          # best first
```

* **GP1 (predicted).** The best cell beats `delta-h8-sc4-lr3e-2` (your v3 reference config, here
  size-matched) by ≥ 0.005 at 10M chars. *(~70%.)*
* **Caveat to state:** the winner is picked at 10M chars, seed 0, among 64 cells (winner's
  curse). `v4-final10` re-measures the top two at 30M chars and ten seeds, so the final numbers
  are not biased by the selection.

## 5. P2: finish D's lr study

Your registration stands (lr ∈ {1e-2, 2e-2, 3e-2, 4e-2, 6e-2}, 30M chars, seeds 0–1, D and the
neural 8-head reference); `v4-final10` reads D's winner from these records. Then the 7 open
`lrwidth` cells (H10a), **after** `v3-lr` finishes, not alongside it.

```bash
.venv/bin/python -m flm.sweep v3-lr   --device cuda --workers 4
.venv/bin/python -m flm.sweep lrwidth --device cuda --workers 4
```

* **GP2 (predicted).** D's best lr is 2e-2, 3e-2 or 4e-2 and beats 3e-2 by < 0.005. *(~70%.)*
  `best_d_lr()` requires both seeds of a lr to count and falls back to 3e-2 if fewer than two
  lrs are complete.

## 6. P3: the citable comparison (`v4-final10`)

Arms, each at seeds 0–9, 30M chars:

* **D** at its best lr (from P2);
* **ref1, ref2**: the two best `v4-ref-tune` cells, size-matched to D;
* **ref0-v3**: your previous reference (93K params, lr 3e-2, short-conv 4), for continuity.

```bash
.venv/bin/python -m flm.sweep v4-final10 --list | tail -1      # must say 40 jobs
.venv/bin/python -m flm.sweep v4-final10 --device cuda --workers 4
.venv/bin/python -m flm.compare v4-final10 --ref D             # writes outputs/v4-final10/compare.md
```

* **GP3 (predicted, the headline).** After equal-effort tuning and size matching, D still leads the
  best reference (ref1 or ref2, whichever has the lower ten-seed mean) by more than 2 standard
  errors. Expected lead **0.005–0.015** (your 0.019 includes ≈ 0.006 from the parameter
  mismatch). *(~55% that it stays real.)* **If it does not, that is the result:** report "within
  noise of the best tuned neural reference at equal parameters" and the cost (D trains ~3× slower).
* Report throughput (chars/s) and the 100M-char single-seed numbers beside the table, as you did.
* Report the comparison against **both** ref1 and ref2 (two comparators; do not pick the
  friendlier one after the fact).

## 7. P4: architecture screens on D (`v4-taps`)

D at d = 72, lr 3e-2, 10M chars, **3 seeds**, evaluated every 100 steps (so you get curves):

| variant | what | flags |
|---|---|---|
| `base` | D | |
| `taps2`, `taps3` | layer-0 FFN antecedent reads the last 2 / 3 positions (+1.2K params per extra tap; D stays < 100K) | `--ante-taps 2` / `3` |
| `biemb` | n-gram-aligned embedding only | `--ngram-init 1 --ngram-mode emb` |
| `biplace` | + bigram rule placement | `... placement` |
| `bifull` | + decoded consequents and output gain | `... full` |
| `trifull` | trigram rules, 2 taps, full | `--ante-taps 2 --ngram-init 2 --ngram-mode full` |

`biemb → biplace → bifull` is the attribution chain; `taps2 → trifull` separates the antecedent
change from the init. The n-gram inits run on **CPU** (1–3 min each) before training starts,
even with `--device cuda`.

```bash
.venv/bin/python -m flm.sweep v4-taps --device cuda --workers 4
.venv/bin/python -m flm.screen_report v4-taps --base base       # curves, steps-to-3.0/2.5/2.2, noise-aware verdict
```

* **GP4.** `taps2` improves D by ≥ 0.01 *(~45%)*; `taps3` ≥ `taps2` *(~40%)*.
* **GP5.** `biemb` within ±0.01 of `base` *(~70%)*; `bifull` and `trifull` reach 2.5 BPC in fewer
  steps than `base` but finish within ±0.01 *(~60%)*; `trifull` better by ≥ 0.01 *(~20%)*.
* **Rule for what counts.** A variant is worth keeping only if `flm.screen_report` calls it
  *real, better* **and** it improves by ≥ 0.01. It then gets ten seeds at 30M chars against `base`
  (same `flm.compare` rule) before it is added to D. Everything else is a null at this power.
* Prior from the CPU screen (zero-order FFN, d = 16, **not evidence for D**): 2-tap antecedent
  −0.043 and FFN-only HTSK −0.066 vs a baseline whose own seeds differed by 0.050. The CPU
  session's stage 2 (3–4 seeds) was still running when this was written and will be pushed.

## 8. P5 and P6

**P5, scaling (`v4-scale`; only if P3 shows a real lead).** D and the best L2 reference at
d ∈ {40, 56, 104} (36K / 63K / 187K params; d = 72 comes from P3), each at its best lr and at half
of it, 30M chars, seeds 0–2. The half-lr cells are there because the CPU phase misread LR-limited
large models as capacity limits. (GRU, FLM and the classic FRLMs got worse with width at a lr tuned at
d = 32.) **GP6:** the lead is positive at all three widths; the lead at 36K is ≤ the lead at
97K *(~60%)*. Merge with P3's d = 72 cells for the size curve (`flm.analyze` keys on device).

**P6, write-up and artifacts.**
* Extend `WRITEUP.md` with a "goal track" section: your D design, P1–P5 outcomes, every
  prediction scored (GP1–GP6), the limitations (single GPU; D searched more than the reference
  until P1). Keep the "not citable until ten seeds" banner until P3 lands, then update it.
* **Samples:** `python -m flm.train ... --save-model` on D seed 0 (30M) and on the best reference,
  then show the same prompt side by side as the CPU phase did (same prompt, temperature 0.8,
  generator seed 1234, so they are comparable).
* **Rulebook (spec only, not implemented):** `flm/rules.py` reads `ffn.conseq` as a zero-order
  consequent. D's FFN is first-order (rule output = firing × (slopeₖ·x) × oₖ), so a rule reads
  "IF context ≈ cₖ THEN push the next-character logits along oₖ, with sign and gain set by v·x".
  D has 108 FFN rules plus 8 × 16 mixer rules; report the 20 most-used FFN rules, not all.

## 9. Deliberately not on this plan

* `minsize`, `lean`, the rest of `ablate`, `headline10`/`headline10v2`: all concern the **old
  zero-order FFN**, which D supersedes. If the dissertation wants the old design's size curve at
  ten seeds, `headline10*` are one command each (`--device cuda`); say so in the write-up rather
  than leaving it implied.
* 16-head cells (VRAM), rule counts > 32 per head, depth > 2 (searched, no gain).

## 10. Reporting back

Push to this branch, then send the CPU session a **self-contained** message at
`grad-school-b2` (what finished, the decisive numbers, what you changed in shared files, what
is blocked). Do not rely on it having read the diff. The CPU session will refresh its dashboard
and fold your numbers into `WRITEUP.md` if you have not.
