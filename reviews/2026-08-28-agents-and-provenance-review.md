# AGENTS.md / provenance review — 2026-08-28

Requested review of `AGENTS.md` for drift against the repository, plus a
provenance re-verification of the CHECKLIST's load-bearing claims.
Every finding below was checked against source on this host; severities
are **high** (a rule that now misleads), **med** (stale or missing
record), **low** (cosmetic or defensive). What was changed is in the
"Changes made" section; open decisions are at the end.

---

## A. AGENTS.md findings (verified 2026-08-28)

| # | location | sev | finding | status |
|---|---|---|---|---|
| A1 | "Submodule conventions" | **high** | Pin snapshot was stale: doc said `tribble-fis 141596e · tribble-cluster 635ed6e · tribble-opt 8049b94` (2026-08-19); actual is `353162c · 1dcf331 · 4d81121`, advanced 2026-08-25 (`a37b85a`, #171). The doc self-instructs to run `git submodule status`, but the stale values would have been read as current for nine days, and the record run cited by the latest PROVENANCE_MAP whole-document section (`full-2026-08-22`) was at the *previous* fis pin — numbers are only comparable to that pin. | fixed: snapshot updated + "pins are a snapshot, not a guarantee" caveat added |
| A2 | Environment → "Style" | med | `make format` claimed to be "black + flake8 + mypy over the whole repo". Two errors: (a) it was *not* whole-repo-safe — `black ./` descends into the submodules' own `.venv` directories (≈900 MB of vendored code) and would rewrite submodule checkouts on any black-version difference (non-negotiable 1); (b) the three tools did not agree on gating: black is the only CI-gated formatter, flake8/mypy are informational, but the makefile failed the target on either. | fixed: submodules excluded from all three (black `--extend-exclude`, flake8/mypy `--exclude`); flake8/mypy run for reporting without failing; `make format` now exits 0 in ~3 s |
| A3 | Environment → "Style" | med | CI pins `black==26.5.1` deliberately (workflow comment records the 2026-08-22 disagreement incident) but the root `pyproject.toml` dev extra declared unpinned `black` — the documented install could pull a black that disagrees with CI. | fixed: `black==26.5.1` in the root dev extra; `uv.lock` synced |
| A4 | lock | med | `uv.lock` had drifted: it was missing the `pyyaml` package entry even though `pyyaml` is a declared root dependency (the dependency the CI `test` job's `build_pdf.py --validate-only` step exists to protect). `uv lock` added it with no other re-resolution. | fixed |
| A5 | Environment | med | `build_pdf.py` note was missing two operational facts: a stale `.venv` that predates a declared dependency dies with `[fatal] could not read reproduce/dataset_specs.yaml` (`uv sync` fixes it), and `--validate-only` is **not read-only** — it copies figures into `research/proposal-defense/prose/fig/` and can dirty tracked files. | fixed: both noted |
| A6 | non-negotiable 2 | med | "Ten seeds is the protocol" stated without its one documented exception: the Chapter 5 driver (`gated-minimax-selection/run_all.py`) runs seeds `0..4`. `common.SEEDS` records this in a comment (lines 343–346) — the exception was in the code, not the rule. Whether the driver moves to ten seeds or the exception gets its own justification is an **open methods decision** (see D2 below). | fixed: exception documented in non-negotiable 2 |
| A7 | non-negotiable 5 | **high** | The pin-bump rule said "diff the CSVs" but not *how much* of them. CHECKLIST B13's bump check diffed three R² values of `table_4_1` and concluded "byte-identical" while the two classification-accuracy columns beside them had already collapsed inside the same 22-commit window (`0.997 ± 0.001 → 0.729`, `0.927 ± 0.002 → 0.500` — B14's defect). | fixed: rule now says diff **every column**, with the B13/B14 lesson recorded in-line; cross-referenced from CHECKLIST B13 |
| A8 | non-negotiable 5 | **high** | Rule covered the submodule's own environment only. But `tribble-fis/uv.lock` pins `tribble-clustering` at a git SHA, so the `tribble-fis` environment — which runs the Chapter 3 FCM rows (`table_3_1`, `table_3_2`, `table_3_4` import `tribbleclustering`) — can run a `tribbleclustering` *older than the grad-school `tribble-cluster` submodule pin*. Found live 2026-08-28: lock at `635ed6e`, pin at `1dcf331`; the installed build lacks the FCM zero-distance crisp-membership fix (cluster #75/#72) that `1dcf331` contains. Preflight's `INSTALL-FRESH` check (added for exactly this class of bug) **failed** on this host as a result; the cluster-side preflight (submodule env) passed. The archive produced under this state would record the new SHA while computing with the old code — the 2026-08-22 "cached wheel after a pin move" failure mode, one environment over. | fixed: rule extended to *every* environment the harness imports from, with the mechanism (uv.lock's SHA pin) and the live finding recorded; **resolution left to the owner** (see D1) |
| A9 | repo map | med | `fis-tsp-strategy/`, `tools/` (ccshim — the B15 C-toolchain bootstrap), `reviews/`, `reproduce/preflight.py`, and `reproduce/dataset_specs.yaml` were absent from the map; the traps section referenced `fis-tsp-strategy` without the map naming it. `gated-minimax-selection`'s `results.json` was described as seeded + tracked; it is **gitignored** (regenerated artifact, written after every figure). | fixed: all rows added/corrected |
| A10 | read-order list | low | Did not name `preflight.py` (the guard `run_all_tables.sh` runs before every numeric phase — a failure stamps the archive NOT CITABLE rather than aborting) and did not warn that PROVENANCE_MAP's dated sections state the world as of that run. | fixed |
| A11 | knobs intro | low | "Documented in `reproduce/tables/README.md`" was inaccurate for four of ten knobs: `REPRO_FAST_SEEDS`, `REPRO_OUTPUT_DIR`, `REPRO_FIG_DIR` live in `reproduce/README.md`; `GRAD_SCHOOL_DATA` in the proposal's `DATASETS.md`. | fixed: per-knob locations listed |
| A12 | traps | low | Wrong path: `experiments/run_note12_threading.py` → `reproduce/experiments/run_note12_threading.py`. | fixed |
| A13 | "Running everything" | low | `run_all_tables.sh` timing "~30–40 min" was the run-of-record host's number (i9-14900HX); this host runs the preflight + a single table visibly slower. | fixed: host qualified |
| A14 | CI section | low | Under-described: the `test` job also runs `build_pdf.py --validate-only` (assembly-time checks; no TeX engine) and the `FuzzySystemsExperiments`/`experiments`/`gated-minimax-selection` pytest suites inside a `uv sync --project tribble-fis --extra dev` environment — deliberately, because a plain `pip install ./tribble-fis` would fetch the SIGILL-prone PyPI tribble-clustering wheel instead of the git-sourced lock override (tribble-fis#124). black's CI pin noted (A3). | fixed |

## B. CHECKLIST / provenance findings

| # | location | sev | finding | status |
|---|---|---|---|---|
| B14-banner | `research/proposal-defense/CHECKLIST.md` top banner | **high** | Banner said the wasserstein defect "BLOCKS every Chapter 4 and Chapter 6 accuracy number at the current pin" and that B14's upstream filing was "not done here; outward-facing, left to the author". Both are now false *in the direction that misleads*: the fix landed as `5253aa0` (tribble-fis #171, CI green) and is inside the current pin `353162c` — so the block is **resolved at the pin**, and what remains is the *document's* re-verification (no ten-seed sweep has run at `353162c`). A reader on 2026-08-28 would have spent time filing a bug that is already merged, or wrongly believed the prose was safe to quote. D8's attribution (`_kmeans_labels_1d` k-means++ restoration, the same #95 commit, different function) independently covers the remaining Table 4.1 regression-row drift, which the banner still called "still untraced". | fixed: banner re-baselined with the two named causes, the environment note (A8), and the remaining owed work (the sweep + re-quote) |
| B14-body | CHECKLIST item B14 | med | Item status was `[ ] 🔒 BLOCKS`; "Owed: (a) file it upstream — not done here". | fixed: item now `[x] ✅ RESOLVED AT THE PIN`; "Owed" re-baselined — (a) done retroactively (5253aa0/#171), (b) the re-quote is what's owed (run `run_all_tables.sh` at the current pin, diff against `full-2026-08-22` and the prose, re-quote; D8 predicts the outcome), (c) the every-column lesson cross-referenced to B13 and AGENTS.md non-negotiable 5 |
| B13 | CHECKLIST item B13 | med | The "Verified before the bump was committed" paragraph records the check that missed B14's collapse (three R² columns eyeballed, two accuracy columns not looked at) without saying so — a future reader would take it as evidence the check was sufficient. | fixed: "Lesson recorded 2026-08-28" paragraph added directly under it, with the exact magnitudes and the every-column rule |
| D8 | CHECKLIST item D8 | low | Correct as written; no change. Its table-by-table attribution is what re-baselined the banner and B14. (The "Owed" list in D8 remains valid — it names the re-quote, not the fix.) | untouched |

## C. Verified-and-unchanged (checked, no action)

- **Repo identity**: top-level is `/home/scott/PycharmProjects/grad-school`,
  origin `fundthmcalculus/grad-school` (push), working on branch
  `docs/agents-review-2026-08-28` created from `main`.
- **Working tree**: clean except this review's own edits; all three
  submodule working trees clean after every command (non-negotiable 1).
- **5 MB rule** (non-negotiable 8): no tracked file over 5 MB
  (`git ls-files` × `du`). `heart_2020_cleaned.csv` (24 MB) and
  `turbine-data.csv` present at the root are **untracked**. `data/.gitignore`
  carries the recovery notes.
- **Preflight, cluster side**: `uv run --project tribble-cluster
  python reproduce/preflight.py --suite cluster` → 3 passed, 0 skipped,
  incl. `INSTALL-FRESH` — the cluster environment is self-consistent.
- **Preflight, fis side**: `INSTALL-FRESH` **fails** (A8) — expected given
  the lock/pin divergence; the run continues and would stamp NOT CITABLE,
  per `run_all_tables.sh`'s preflight handler (verified in the script:
  a preflight failure stamps `## NOT CITABLE` and does not abort).
- **`run_all_tables.sh` knobs**: `REPRO_FAST_SEEDS` default `0,1,2`
  (line 71), `--archive-only` early-exit (lines 404–412), preflight
  NOT CITABLE stamping (lines 418–437), `--fast` seed stamping
  (lines 509–517) — all as AGENTS.md describes.
- **GPU skip** (non-negotiable 6): `table_3_4`'s probe returns
  `(False, reason)` when CuPy finds no CUDA device; the script comments
  say a missing runtime "is a result" (line 137). Confirmed.
- **`common.SEEDS`**: default `"0,1,2,3,4,5,6,7,8,9"` (line 27); the
  five-seed Chapter 5 exception is recorded in its comment (lines 343–346).
- **`hostenv.sh` / `tools/ccshim`**: sourced by `run_all_tables.sh`
  (lines 91–96); `tools/ccshim/cc_shim.c` present (B15).
- **`fis-tsp-strategy/test_invariants.py`**: present; `.tsp` instances are
  fetched by `ClusteringExperiments/tsplib/download.py` (none vendored) —
  the trap's explanation holds.
- **CI workflow**: matches the corrected AGENTS.md CI section (A14),
  including the black-pin comment (lines 27–35) and the SIGILL rationale
  (lines 104–113).
- **Table generators named in AGENTS.md's run examples**
  (`table_4_1_mog_baselines.py`, `table_6_1_model_family.py`,
  `table_3_1_pvat_scaling.py`): all exist and route through
  `uv run --project tribble-fis` / `tribble-cluster` as documented.
- **`black --check`** on the pinned submodule *source* dirs with the local
  black 26.5.1: clean (1.8 s) — the exclusion is future-proofing against
  version drift and the `.venv` descent, not a fix for a live red.
- **`notes/Theoretical Advances…pdf`** (25 MB) is tracked — non-negotiable 8
  is scoped to *data* files; a supporting-paper PDF is out of its scope.
  Flagging here so the scoping decision is visible, not silently relied on.

## D. Open decisions (owner)

**D1 — the `tribble-fis` environment's stale `tribble-clustering` (A8).**
Two consistent resolutions, both touching the *fis project* — not this
repo's submodule pin:
1. Inside `tribble-fis` (upstream project): `uv lock --upgrade-package
   tribble-clustering` to move its lock from `635ed6e` toward the
   cluster's current line, commit the lock there, then bump the
   grad-school `tribble-fis` pin to include it — a full pin-bump event:
   re-sync the fis environment, preflight both sides, run the affected
   Chapter 3 tables before/after and diff **every column** (the A7 rule),
   record it (B13 procedure). Outward-facing: a tribble-fis commit +
   possibly a PR.
2. Document-and-monitor: accept the lock's pinned SHA as the fis
   project's own choice (it tracks upstream clustering's main, a
   different consumer), note in `reproduce/PROVENANCE_MAP.md` that
   Chapter 3 FCM rows run under lock-pinned `tribbleclustering` `635ed6e`,
   and let `INSTALL-FRESH` keep failing until the two converge — but
   then every archive on this host is NOT CITABLE, which is not a
   workable steady state.
Recommendation: (1), at the owner's convenience. Until it lands, this
host's `run_all_tables.sh` runs are NOT CITABLE by the harness's own
stamp, and that is the correct, self-enforcing behavior — nothing here
changed the stamp to make it pass.

**D2 — Chapter 5's five-seed driver (A6).** `run_all.py` runs seeds `0..4`
and its tables honestly footer the actual seed set (the five-seed stamp
was the past defect; the footer fix holds). Either move the driver to ten
seeds (runtime ×2, Chapter 5 table re-quotes) or justify the exception in
the methods chapter (NERFCM restart semantics make five restarts the
deliberate protocol per `common.py`'s comment). The rule now documents
the exception either way; the *decision* is a methods call.

**D3 — flake8/mypy debt.** Root scope: 474 flake8 findings (~140 files,
dominated by E402 module-level import-not-at-top in legacy scripts) and a
mypy whole-tree run blocked by duplicate module names across the
coursework dirs (`AEEM6022/hw4.py` vs `AEEM6097/hw4.py`). Treated as
informational, matching CI. If the owner wants a green `make format`, the
scoped move is per-directory mypy invocations plus a targeted flake8 pass
over `reproduce/` and `research/` (the proposal-critical trees), leaving
graded coursework untouched (its rule: don't refactor as production).

## E. Changes made in this review

| file | change |
|---|---|
| `AGENTS.md` | A1, A2, A3 (ref), A5, A6, A7, A8, A9, A10, A11, A12, A13, A14 |
| `research/proposal-defense/CHECKLIST.md` | banner re-baseline; B13 lesson; B14 status + "Owed" re-baseline; B14 "still untraced" line corrected to D8's attribution |
| `makefile` | submodule exclusions (black `--extend-exclude`, flake8/mypy `--exclude`); flake8/mypy informational |
| `pyproject.toml` | `black==26.5.1` in dev extra with the CI-pin rationale |
| `uv.lock` | `uv lock`: black specifier + missing `pyyaml` entry |
| `reviews/2026-08-28-agents-and-provenance-review.md` | this file |

Nothing inside a submodule was touched. No table was re-run at the new
pin — that is the owed sweep (B14/D8/D1), deliberately not done here:
it is a 30–40 min run-of-record event with a documented procedure, and
running it silently from a review pass would blur the line between
"verified the docs" and "produced new citable numbers".
