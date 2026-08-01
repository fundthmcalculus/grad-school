# Handoff — continue on a local machine

_Written 2026-08-01 at the end of a cloud session (PR
[grad-school#25](https://github.com/fundthmcalculus/grad-school/pull/25), branch
`claude/tribble-fis-evaluation-s97850`). Everything below is committed and pushed._

This session ran the full t-norm/t-conorm re-evaluation. The one thing it could
not do is **BETH**, because the data is too large to commit and this environment
has no route to fetch it. That is the headline task for a local session, and it
is **not** a drop-the-files-in-and-run job — see §1.

---

## 0. State in one screen

| | |
|---|---|
| Branch | `claude/tribble-fis-evaluation-s97850` → PR grad-school#25 (open, clean, targets `feat/proposal-def-1`) |
| `tribble-fis` pin | `23bfdbc` (= `main`) |
| `tribble-cluster` pin | `c71171e` (unchanged all session) |
| `tribble-opt` pin | `94547ff` (unchanged) |
| Merged upstream this session | tribble-fis #29 (pin_extremes edge cases), #32 (Einstein + decoupled norms), #33 (Concrete CSV) |
| Open upstream issue | tribble-fis #31 — `build_tree` accepts `t_norm_name` and never uses it |

**Findings already written up** — do not redo these:

- `research/proposal-defense/TNORM_REEVALUATION_PLAN.md` — the plan, and the two blockers as they stood
- `research/proposal-defense/TNORM_REEVALUATION_RESULTS.md` — all results, incl. two addenda. **Read the superseded banner at the top**: the original "no measurable change" headline was wrong and is corrected in Addendum 2.
- `reproduce/outputs/FIX_IMPACT.md` — cell-by-cell diff of the nine tables

**Archived runs** (each with `PROVENANCE.txt` recording submodule SHAs and seeds):

```
reproduce/outputs/baseline-d0d6714/     nine tables, pre-fix
reproduce/outputs/postfix-pr29/         nine tables, post-fix
reproduce/outputs/openset-prefix/       Table 4.4 + θ sweep, pre-fix
reproduce/outputs/openset-postfix/      Table 4.4 + θ sweep, post-fix
reproduce/outputs/norm-matrix-ba87f5a/  the five De Morgan pairs
```

---

## 1. BETH — the actual task, and why it is not plug-and-play

### The trap

`table_4_4_openset.py` already has a BETH branch. **It is dead code.** Dropping
the files in produces a table of `N/A`, in about one second, with exit status 0
and no error. I verified this with a synthetic BETH-shaped file:

```
[data] BETH found -- using it
BETH: N=3000 M=6 classes=['anomaly', 'regular']
wrote reproduce/outputs/table_4_4_openset.csv     <- every cell N/A
```

**Why.** The harness runs *leave-one-class-out*: hold out class `c`, train on the
rest, treat `c` as unseen. BETH's labels are binary (`evil` → `anomaly` /
`regular`), so holding either one out leaves a single-class training set, and
this guard skips every iteration:

```python
# reproduce/tables/table_4_4_openset.py:170
if len(np.unique(yk)) < 2 or len(Xk) < 40:
    continue
```

LOCO needs ≥3 classes. Glass has 6, which is why it works. BETH has 2.

### What Chapter 4 actually describes

A **one-class / novelty** protocol, not LOCO: *train on benign traffic only, then
show the model novel attacks.* That is a different experiment and needs its own
code path. Sketch:

```python
def run_one_class(X, y, benign_label="regular"):
    """Ch4's BETH protocol. Train on benign ONLY; test = held-out benign + all evil."""
    benign = (y == benign_label).values
    for seed in C.SEEDS:
        Xtr, Xte_b = train_test_split(X[benign], test_size=0.3, random_state=seed)
        Xte = pd.concat([Xte_b, X[~benign]], ignore_index=True)
        is_unknown = np.r_[np.zeros(len(Xte_b), bool), np.ones(int((~benign).sum()), bool)]
        # NOTE: the complement rule needs >=2 known classes to build a rule base,
        # so a single benign class will not fit as-is. Options, in preference order:
        #   (a) partition benign into k pseudo-classes (quantile bins on a strong
        #       feature, or k-means) so the rule base has structure to complement;
        #   (b) use BETH's `sus` column as a second known class -- check whether
        #       that leaks, since sus and evil are correlated;
        #   (c) keep LOCO but derive >=3 classes from the attack taxonomy if the
        #       full BETH release has attack subtypes rather than a binary flag.
```

**Option (a) is the one to try first**, and it is worth stating in the chapter
either way: the complement rule is defined as the complement of the aggregate of
the *known-class* rules, so it structurally requires more than one known class.
That is a real property of the method, not an implementation detail, and Ch 4
§4.3.5 currently glosses over it.

Decide this before writing code — it changes what the experiment means.

### Where the files go

```
tribble-fis/gaussian_mixture/beth_data/labelled_training_data.csv
tribble-fis/gaussian_mixture/beth_data/labelled_testing_data.csv
```

Add `/gaussian_mixture/beth_data/` to `tribble-fis/.gitignore` — do not commit it.
`gaussian_mixture/beth.py` and `beth-anomaly.py` already read these paths.

### What to compare against

The Glass numbers are the reference. From `openset-postfix/` (5 seeds, 6 classes,
hamacher, `23bfdbc`):

| θ | detection | false alarm | J |
|---|---:|---:|---:|
| 0.70 *(best)* | 0.612 | 0.352 | **+0.261** |
| 0.99 *(default)* | 0.360 | 0.190 | +0.170 |

and the arms at θ=0.99: complement **+0.170**, one-class SVM +0.062, isolation
forest **+0.208**.

BETH is the dataset Ch 4 actually claims, and it is ~1000× larger than Glass, so
it is the first chance to test whether the complement rule's poor showing is the
method or the 214-sample testbed. **Expect the answer to be "the testbed" only if
the numbers say so** — Glass's spread (±0.331 detection) is wide enough that the
three detectors are statistically tied there.

---

## 2. Everything else outstanding

### Datasets still missing

Tier 1 is now just BETH. Full inventory, with the exact path each loader expects:

| Dataset | Path expected | Used by |
|---|---|---|
| **BETH** | `tribble-fis/gaussian_mixture/beth_data/labelled_{training,testing}_data.csv` | Table 4.4 (needs §1 rework), `beth.py`, `beth-anomaly.py` |
| RT-IOT2022 | `tribble-fis/gaussian_mixture/rt-iot2022/RT_IOT2022.csv` | `ch4-mog-iot` (manifest, `big-mem`) |
| N-BaIoT | `tribble-fis/gaussian_mixture/iot-botnet/Danmini_Doorbell/{benign_traffic.csv,gafgyt_attacks/combo.csv}` | `iot-botnet.py` |
| Statlog Shuttle | `fetch_ucirepo(id=148)` — needs network | `nasa.py` |
| DARWIN | `tribble-fis/gaussian_mixture/darwin.csv` | 5 scripts |
| Wine Quality | `tribble-fis/gaussian_mixture/winequality-{white,red}.csv` | `wine_red.py` |
| Tetouan power | `tribble-fis/gaussian_mixture/powerconsumption.csv` | `powerconsumption.py` |
| Gas turbine | `tribble-fis/gaussian_mixture/gas_turbine/test/ex_4.csv` | `turbine.py` |
| Wave energy | `tribble-fis/gaussian_mixture/WEC_Perth_49.csv` | `wec.py`, `wec-p1.py` |
| Turbine | `turbine-data.csv` (repo root) | `turbine.py` |
| IRIS / heart_2020_cleaned | repo root, or `$GRAD_SCHOOL_DATA` | `verify_beta_*.py` |
| UCR/UEA (G2) | downloaded by `aeon` — needs network | G2, not yet started |

**Already present:** Concrete (committed to `tribble-fis` by #33), PhiUSIIL
(committed by #19), Glass (committed this session).

A local machine with network makes most of these one `ucimlrepo` call away — the
loaders already have that fallback. This environment blocks
`archive.ics.uci.edu` and `api.openml.org` at the egress gateway.

### Code tasks

1. **BETH protocol** (§1) — the main one.
2. **tribble-fis#31** — `build_tree` accepts `t_norm_name` and never uses it. Either
   wire it into split selection or delete the parameter. Right now it is a knob
   that silently does nothing, which is how it was found.
3. **Mixed norm/conorm pairs.** `tribble-fis#32` landed the ability
   (`allow_mixed_norms=True`); the sweep was deliberately *not* run. The right
   testbed is the open-set harness, because breaking De Morgan duality should
   bite exactly where the anomaly rule takes a complement of a conorm. Run it once
   BETH works. See `TNORM_REEVALUATION_RESULTS.md` Addendum 1 for why the
   coordinate sweep did not justify the 25-cell version.
4. **`IRIS.csv` / `heart_2020_cleaned.csv`** — the `verify_beta_*.py` scripts now
   resolve paths properly but the data is still absent. They will raise a clear
   `FileNotFoundError` naming the search path and the `GRAD_SCHOOL_DATA` override.
5. **De-duplicate `verify_beta_*.py`** — three files exist twice, byte-identical,
   at the repo root and under `gated-minimax-selection/`. Left alone pending the
   experiments-out-of-submodules reorganisation.

### Text corrections queued (not yet applied to the chapters)

- **Ch 4 §4.3.5** — re-quote the complement rule from `openset-postfix`. Best
  operating point is **J = +0.261 at θ = 0.70**, not the θ = 0.99 default
  inherited from `beth-anomaly.py`, which is past the useful range. Drop the claim
  that the rule leads the dedicated detectors: it trails isolation forest at a
  matched operating point, and the spread makes all three a tie.
- **Ch 4 / Ch 6** — refined 0th-order Concrete figures predate the refinement fix.
  Re-quote R² **0.461**, RMSE **12.01** from `postfix-pr29`.
- **`ACTION_ITEMS.md` line 93–94** — "best J = +0.155 at θ = 0.80" is pre-fix; retire.
- **`ACTION_ITEMS.md` line 65** — says six experiments verified end-to-end.
  `run_all_tables.sh` now drives **ten**, and each has produced output at least
  once. Note no single pass has run all ten together since Glass and the
  norm/conorm matrix were added, so that is worth doing early as a smoke test:
  `reproduce/run_all_tables.sh full-sweep`.
- Consider **`probability` as the default norm** for the TSK regression path:
  `min/max` won no Concrete row in the sweep, and probability is already the
  default for the Ruspini models and fuzzy trees.

---

## 3. Running things locally

```bash
git clone --recurse-submodules https://github.com/fundthmcalculus/grad-school
cd grad-school && git checkout claude/tribble-fis-evaluation-s97850
```

One table:

```bash
uv run --project tribble-fis python reproduce/tables/table_4_4_openset.py
```

All ten, archived under a label with provenance:

```bash
reproduce/run_all_tables.sh my-label            # everything
reproduce/run_all_tables.sh my-label table_4_4_openset   # just one, appends provenance
```

Diff two archived runs into `reproduce/outputs/FIX_IMPACT.md`:

```bash
python3 reproduce/compare_runs.py baseline-d0d6714 postfix-pr29
```

Useful env knobs: `REPRO_SEEDS`, `REPRO_THETA_SWEEP` (emits Fig 4.2),
`REPRO_ANOM_CONORM`, `REPRO_NORM_FAMILIES`, `REPRO_PHIUSIIL_N`, `GRAD_SCHOOL_DATA`.

### Two traps this session hit — worth not repeating

1. **Always run table scripts from the repo root.** A `cd` into a submodule makes
   the relative script path unresolvable; python exits instantly, and if a
   previous run left outputs on disk the archive looks populated. This produced a
   confident, completely wrong "the fix changed nothing" result that survived
   until a single-split probe contradicted it. `run_all_tables.sh` detects a
   script that writes nothing, but **not** one whose outputs were left by an
   earlier invocation. Read the log, not the exit status.
2. **`tribble-fis` is installed editable**, so a `git checkout` inside the
   submodule takes effect immediately for the next `uv run`. That makes
   before/after comparisons easy — check out the old SHA, run, check out `main`,
   run — but it also means an accidentally-left checkout silently changes results.
   `PROVENANCE.txt` records the SHA for exactly this reason; check it.

---

## 4. What I would do first

1. **Decide the BETH protocol** (§1) — one-class with partitioned benign, or a
   multi-class attack taxonomy if the full release has one. This is a research
   decision, not a coding one, and everything else about BETH depends on it.
2. Implement it as a **separate arm** in `table_4_4_openset.py` rather than
   bending LOCO, so the Glass result stays reproducible and the two protocols are
   comparable side by side.
3. Re-run and compare against the Glass reference above. If the complement rule
   is competitive at BETH scale, that is the Ch 4 result the chapter has been
   claiming without evidence.
4. Then the mixed-pair sweep, which finally has a testbed worth running it on.
