# Working doc — making the proposal's tables and scripts reproducible

_Session of 2026-08-01. Branch `feat/proposal-def-1`, pushed through `7f399bc`._

The task was "ensure all tables and scripts are reproducible." The short version
is that three scripts did not run at all, one submodule was on the wrong commit,
and most of the prose predated the pipeline meant to reproduce it. Fixing that
changed numbers in every results chapter, retracted one published conclusion,
refuted another, and surfaced a model that diverges on one split in ten.

Everything below is committed. Start with `reproduce/PROVENANCE_MAP.md`, which is
the per-table index this session produced.

---

## 0. State in one screen

| | |
|---|---|
| Branch | `feat/proposal-def-1`, pushed, in sync with `origin` |
| Commits | `16f614d` audit + prose · `b008bb3` output redirect · `59f79ca` 10-seed re-quote · `7f399bc` jobs=4 |
| `tribble-fis` pin | `23bfdbc` (restored — it was checked out at the wrong SHA) |
| `tribble-cluster` pin | `c71171e` (only `uv.lock` dirty, and it was before this session) |
| Seed standard | **10** (`common.SEEDS`), up from 5. This is now protocol, not a knob. |
| Full sweep | 11 generators, all green, ~37 min → `reproduce/outputs/seeds10-2026-08-01/` |

Two decisions are waiting on you, in §4.

---

## 1. Scripts that did not run

These are the reproducibility defects proper. Each produced plausible-looking
output or exited zero, which is why none had been noticed.

**`tribble-fis` was checked out at `d0d6714`** — the *pre*-`pin_extremes`
baseline — while the parent repo pins `23bfdbc`. Any run in that state silently
reproduces the superseded numbers. `run_all_tables.sh` records the submodule SHA
in `PROVENANCE.txt` for exactly this reason; check it before trusting a run.

**`gated-minimax-selection/run_all.py` crashed and discarded its own results.**
It died in `fig_membership` with `NameError: name 'row' is not defined`. A commit
on 2026-07-20 changed that figure from `subplots(2, 2)` to `subplots(2, 1)` while
the body still addressed `axes[row, 0]`, `axes[0, 1]`, and `axes[1, 1]`. Because
`results.json` is written *after* every figure, each invocation threw away its
entire numeric phase — the JSON on disk was last written 2026-07-20 and nothing
since could regenerate it. Chapter 5's claim that the driver "writes the results
and every figure referenced below" was not true when written.

Fixed by restoring the 2×2 grid and the `enumerate` that supplies `row`, matching
the pattern `fig_transform` already uses twenty lines earlier. The driver now
completes, regenerates 16 of 17 figures (`fig11_scaling` is behind an opt-in
`--scaling` flag), and rewrites `results.json` **byte-identical** to the
2026-07-20 file. Chapter 5's numbers were never wrong — only unreproducible.

**All four registered `tribble-cluster` experiments were unrunnable as invoked.**
They do `from experiments.blockwise_vat import ...`, which needs the submodule
*root* on `sys.path`. Run by path — `python experiments/adversarial_eval.py`, the
form the manifest used — Python puts `experiments/` there instead and every one
dies with `ModuleNotFoundError` before doing any work. Now run through
`reproduce/experiments/run_cluster_experiment.py`.

**`PROVENANCE.txt` was recording the wrong seed count.** It hardcoded
`"0,1,2,3,4 (default)"` whenever `REPRO_SEEDS` was unset, so when the default
moved to ten it kept reporting five — while every table footer in the same
directory said ten. Now derived from `common.SEEDS`.

---

## 2. Where the output goes now

Reproducing a Chapter 3 figure used to dirty a pinned submodule.
`reproduce/experiments/run_cluster_experiment.py` inverts that: the experiment
code stays in `tribble-cluster`, and only the destination moves up into this
repo. It rebinds `FIG_DIR` to `reproduce/outputs/figures/cluster/` and puts the
submodule root on `sys.path`, so it fixes the invocation and the output location
together, with no edit to submodule source.

```bash
uv run --project tribble-cluster --with scipy \
    python reproduce/experiments/run_cluster_experiment.py --all
```

`git -C tribble-cluster status` stays clean afterwards. Generated figures are
**gitignored for now** (`reproduce/outputs/.gitignore`); labelled run archives
under `outputs/<label>/` stay tracked, because they are the evidence a later diff
is taken against.

Two notes for the experiments-and-notes migration:

- The `findings/*.md` files are **not generated**. All three scripts write only
  PNGs; the one `.md` mention is a docstring cross-reference. They are
  hand-authored notes, nothing regenerates them, and the manifest still lists
  them at their submodule paths — those strings need updating when they move.
- `gated-minimax-selection/outputs/` is covered by a top-level `.gitignore` entry,
  so none of Chapter 5's figures or `results.json` are tracked. A `git mv`-based
  migration will silently leave all of them behind.

---

## 3. Running it

```bash
reproduce/run_all_tables.sh my-label            # the real thing, ~37 min
reproduce/run_all_tables.sh --fast smoke-check  # minutes, NOT citable
```

`--fast` cuts seeds (`REPRO_FAST_SEEDS`, default `0,1,2`) on the four tables that
dominate runtime — `concrete_reconciliation`, `pvat_scaling`,
`norm_conorm_matrix`, `hyperparam_normalization` — and leaves the other seven
alone, since they total under two minutes and reducing them would only cost
credibility. A fast archive is stamped **NOT CITABLE** in `PROVENANCE.txt` and
records the seed set *per table*, so a thin cell cannot be mistaken for a full
one later.

**`REPRO_THETA_SWEEP` is a comma-separated θ list, not a boolean.** I ran two
full sweeps with `REPRO_THETA_SWEEP=1`, which is a valid list of one and emits a
single row at θ=1.0 where the boost saturates and every cell is legitimately
zero — a mis-set knob that reads exactly like a null result. Use
`REPRO_THETA_SWEEP=0.5,0.6,0.7,0.8,0.9,0.99,1.1`.

The Concrete table went from **1301s to 652s** (measured at 8 workers,
byte-identical output) by hoisting seed-independent preprocessing that was being
recomputed 60 times and running the arms concurrently. Results are reassembled in
job order, never completion order, so the table cannot depend on scheduling.
`REPRO_JOBS` defaults to 4 to leave cores free; `REPRO_JOBS=1` restores the
serial path. Worker stdout now interleaves and arrives late — cosmetic, and
documented in the script, because this project has been bitten by misread logs.

---

## 4. Two things that need your decision

**The mixture of experts diverges on seed 9.** Under normalized features at
library defaults it predicts up to **10,536 MPa** on a Concrete target that never
exceeds ~82. The other nine seeds give 0.805 ± 0.059 with nothing anomalous about
them; a five-seed run did not contain the split and reported a clean
0.813 ± 0.039. Including it, the cell reads R² = −220.9 ± 665.0.

Table 6.1 quotes the nine stable seeds with the failure disclosed in a footnote,
on the grounds that a mean of −220.9 describes the failure rather than the model
and suppressing it would be worse than either. **This is a real defect, not a
reporting artifact** — the gating solve needs a numerical guard before the
hierarchy can be recommended for use. Written into Ch6 and Goal G4.

**Goal G5 has moved from "settled (complete)" to reopened.** The headline
"quantile's advantage grows monotonically with skew (+0.003 → +0.201)" was a
three-seed artifact. At ten seeds Q−U is negative in every row past symmetry,
down to −11.8. The real finding is in the spreads: quantile *destabilises*
(±0.99, ±4.45, ±21.2) rather than becoming inaccurate, while uniform decays
smoothly. Neither scheme is safe on a heavily skewed target.

The recommendation is withdrawn and the goal reopened; the starvation diagnosis
and the tail correction both survive. **If G5's "complete" status was ever
communicated to the committee, that needs correcting before the next
conversation with them.**

---

## 5. What changed in the prose

Every numbered table is now quoted at ten seeds against
`reproduce/outputs/seeds10-2026-08-01/`. Beyond the two items above:

| Table | Change |
|---|---|
| 3.4 | Two cells corrected (circles naive-block 0.10 → 0.00, bridged 0.07 → 0.08) |
| 3.5 | Re-quoted: light 0.51 → 0.47, top-m 0.74 → 0.61, fps 0.37. Conclusion *strengthened* — the gap to each single ingredient is wider |
| 4.1 | Re-quoted, plus CART/RF control rows showing the transform is worth +0.001/+0.000 to rank-based models and +0.125 to the fuzzy one |
| 4.2 | Four-bucket crossover **retracted** — largest gap in all 18 configs is 0.012 against σ ≈ 0.02–0.03. The starvation mechanism survives |
| 4.6/4.7 | Re-quoted twice. Best point moved θ=0.80/J=+0.155 → θ=0.70/+0.261 → a flat band of +0.222…+0.239 peaking at θ=0.60. The complement-vs-isolation-forest ordering **flipped twice**, and at ten seeds they are level to 0.002 — every previous ordering was noise |
| 5.1 | `0.00 (chaining)` was filed under *NERFCM on D\**; 0.00 is single-linkage. Three cells were dashed "not run" that the driver had run |
| 6.1 | Re-quoted; caption said "3 seeds" |
| 6.2 | Re-quoted; PhiUSIIL column filled and shows the dataset is saturated (CART and RF both 1.000), so it should carry no weight |
| 6.4 §6.3.5 | "Refinement hurts at high capacity" softened to diminishing returns — it stays positive at every order in the uniform sweep |
| Ch1, Ch7 | Intro figures updated; G4 widened to cover accuracy claims, with ten seeds as the floor |

Chapter 6 also asserted flat 0.658 / tree 0.746 / mixture 0.791 as "what is
measured today" fourteen lines above the table that explicitly retires them. And
the `Reproduction` blockquotes in Ch3/Ch4/Ch6 all named the wrong scripts — Ch4
claimed Tables 4.4–4.7 come from `table_4_1_mog_baselines.py` when 4.6 and 4.7
come from `table_4_4_openset.py`.

---

## 6. Still outstanding

- **Table 3.1's headline 4,096-point pair** (124 s vs 2.56 s, ~48×) is reproduced
  by neither generator and appears to be an external measurement from the NAFIPS
  work. It needs a citation or a harness run. This is the one headline number in
  Ch3 with no in-repo provenance.
- **Tables 3.2 and 3.3** have no generator; 3.3 needs a GPU host.
- **Table 6.3** is structural by design; **6.4**'s entry point is unconfirmed.
- **ANFIS and GA-FIS adapters** absent, so those cells stay `N/A`.
- **BETH** — unchanged from `research/proposal-defense/HANDOFF_LOCAL_SESSION.md`
  §1, which is still the right brief. The LOCO harness needs ≥3 classes and BETH
  is binary, so it needs its own one-class code path; that is a research decision
  before it is a coding one.
- **The HME numerical guard** and **the G5 decision**, from §4.

---

## 7. Traps worth not repeating

**Silence is not success.** Every defect in §1 produced plausible output or
exited zero. A submodule on the wrong commit, a driver crashing before it wrote
its results, experiments dying on import, a provenance file misstating its own
seed count — none of them announced themselves. Read the provenance, not the exit
status.

**A conclusion can be reproducible and still wrong.** The retracted crossover and
the refuted skew hypothesis came from generators that ran correctly and
deterministically every time. The harness was never broken; the sample was too
small to support the story built on it. Determinism is not evidence.

**A five-seed mean does not establish stability.** The HME case is the clean
example: a model that is excellent nine times in ten and catastrophic the tenth
reads as a solid 0.813 ± 0.039 if the tenth split is not in the sample. Ten seeds
is the floor now, and `PROVENANCE.txt` records the count per table.

**My own errors this session, for calibration.** I ran both sweeps with
`REPRO_THETA_SWEEP=1` and got a table of zeros that reads like a null result. My
first Table 4.1 edit left orphaned rows behind that I caught only on a later
sweep. And my background waiter shells used
`until ! pgrep -f run_all_tables; do sleep 20; done`, where `pgrep -f` matched the
waiter's *own* command line — three of them spun for up to 71 minutes and their
completion notifications were never going to arrive. Nothing reported here
depended on them, but the pattern is the same one that caused the drift in the
first place: output that looks right is not the same as output that is right.
