# Working doc — making the proposal's tables and scripts reproducible

_Session of 2026-08-01/02. Branch `feat/proposal-def-1`, pushed. Run of record and
submodule pins are in the table below; those are the versions to check, not this
doc's own commit._

The task was "ensure all tables and scripts are reproducible." What that turned
into: three scripts that did not run at all, a submodule on a commit that did not
exist, two datasets silently substituted, a library regression that cost 7 points
of accuracy, and one published conclusion retracted. Every table is now quoted
from a single coherent run, and the failures are documented because most of them
recur in the same shape.

Start with `reproduce/PROVENANCE_MAP.md` — the per-table index this work produced.

---

## 0. State in one screen

| | |
|---|---|
| Branch | `feat/proposal-def-1`, pushed, in sync |
| Submodules | `tribble-fis` `4371a9d` · `tribble-cluster` `5d44dfa` · `tribble-opt` `94547ff` |
| Seed standard | **10** (`common.SEEDS`) — protocol, not a knob |
| Run of record | `reproduce/outputs/full-2026-08-02/` — 11 tables, all green, 24 min |
| Concrete runtime | **369 s**, from 1301 s serial (analytic gradient + parallelism) |

One thing needs your judgement, in §5 (the submodule item below was resolved 2026-08-19).

---

## 1. What was broken

Each of these produced plausible output or exited zero, which is why none had
been noticed.

**`tribble-fis` was on the wrong commit** — `d0d6714`, the pre-`pin_extremes`
baseline, while the repo pinned `23bfdbc`. Anything run in that state silently
reproduced superseded numbers.

**Chapter 5's driver crashed and discarded its own results.**
`gated-minimax-selection/run_all.py` died in `fig_membership`
(`NameError: name 'row'`) because a commit changed `subplots(2,2)` to
`subplots(2,1)` while the body still addressed `axes[row,0]`. `results.json` is
written *after* every figure, so each run threw away its whole numeric phase.
Fixed; it now completes and reproduces the JSON byte-identically — the numbers
were never wrong, only unreproducible.

**All four `tribble-cluster` experiments were unrunnable**, then unrunnable again
for a different reason. Originally they imported `from experiments.foo import
...` and died with `ModuleNotFoundError` when invoked by path. Then grad-school
#26 moved them to `ClusteringExperiments/` **without updating those imports**, so
all 37 files were broken on arrival. Rewritten to sibling imports; verified
reproducing Tables 3.4/3.5/3.6.

**Two datasets were silently substituted.** Caught by one tell: CART and Random
Forest moved between runs, and no solver change can touch sklearn models.

- *Concrete* fell back to `fetch_ucirepo(id=165)`, which is **rounded to two
  decimals** (79.99 vs 79.98611076). Every row shifted slightly.
- *PhiUSIIL* raised `FileNotFoundError` and fell back to ucimlrepo, which returns
  a **different feature set**. That was the entire 0.997 → 0.913 "regression".

Both restored to `data/`; the fallback now announces that its results are not
comparable.

**`PROVENANCE.txt` misstated its own seed count** — hardcoded `"0,1,2,3,4"`, so
when the default moved to ten it kept reporting five while every table footer in
the same directory said ten. Now derived from `common.SEEDS`.

**`REPRO_THETA_SWEEP` is a θ *list*, not a boolean.** I ran two sweeps with `=1`,
a valid list of one, which emits a single row at θ=1.0 where the boost saturates
and every cell is legitimately zero — output indistinguishable from a null
result.

---

## 2. Upstream: what this surfaced in `tribble-fis`

Nine issues filed. **All nine are now closed**, across six upstream PRs.

| | | |
|---|---|---|
| #36 | `solve_tsk_consequents` returned 1e24 coefficients on a near-singular design | fixed (#37) |
| #39 | `trapz_pdf` degenerate trapezoid | fixed (#45) |
| #40 | pytest collection failed from either directory | fixed (#44) |
| #41 | `l2_reg` defaulted to 0.0 with unpenalised intercepts | fixed (#47) |
| #42 | pandas in the refinement hot path | fixed (#48) |
| #43 | analytic gradients for antecedent refinement | fixed (#48) |
| #49 | **PhiUSIIL accuracy 0.997 → 0.927, bisected to #34** | fixed (#52) |
| #50 | `top_p` documented as cumulative, implemented as a threshold | fixed (#52) |
| #51 | default scorer → `wasserstein` | fixed |
| #38 | (PR) firing-strength column hoist, 9.8%, bit-identical | merged |

The consequent-solver bug is the one worth remembering: `np.linalg.solve` only
raises on an *exactly* singular matrix, which floating point almost never
produces, so the `except LinAlgError` guard caught the case that never happens
and missed the one that does — returning finite coefficients of order 10²⁴.

---

## 3. Findings that changed the proposal

**The mixture of experts diverged on one split in ten.** Seed 9 predicted up to
10,536 MPa on an ~82 MPa target. Nine seeds looked unremarkable, and a five-seed
protocol reported a clean 0.813 ± 0.039. Now fixed upstream and reading
**0.810 ± 0.064** across all ten. Chapter 6 keeps the episode as the concrete
evidence behind G4's seed floor: a five-seed mean did not give a slightly wrong
number, it *certified as stable a model that fails one time in ten*.

**Goal G5 is refuted and reopened.** "Quantile's advantage grows monotonically
with skew (+0.003 → +0.201)" was a three-seed artifact. At ten seeds Q−U is
negative in every row past symmetry, to −11.8. The real finding is in the
spreads: quantile *destabilises* (±0.99, ±4.45, ±21.2) rather than becoming
inaccurate. Ch7 had this marked "settled (complete)" with a recommendation; both
are withdrawn. Unaffected by any of the library fixes — it is synthetic.

**Table 4.2's four-bucket crossover is retracted.** Largest gap across all 18
configurations is 0.012 against σ ≈ 0.02–0.03. The starvation *mechanism*
survives; the accuracy story built on it does not.

**Chapter 5 gained the `IVATMeans` motivation.** `tribble-cluster` ships a
wrapper that finds clusters with iVAT — minimax, ultrametric, good at non-convex
structure — and then represents each one by a Euclidean mean and refines with
Euclidean FCM. The two halves are in incompatible geometries: the mean of a ring
is in the hole. On exactly the data where iVAT beats k-means, the back end throws
the advantage away. The library documented this in `docs/novel-niche.md`; the
proposal had never mentioned it, which was a real omission, because Ch5's method
*is* the resolution — stay in minimax geometry and use a relational method that
consumes a dissimilarity matrix. It also positions the work: the VAT lineage
stayed crisp and visual, the fuzzy relational lineage never adopted the minimax
ordering, and the gap between them is unoccupied rather than contested.

**Appendix A.4 is new: feature scoring.** The Chapter 4 construction ranks
features before building anything, so rule count and readability follow from that
ranking. On PhiUSIIL the most informative feature is ranked 1st by wasserstein
and outside the top 20 by bhattacharyya — 0.9967 versus 0.4267 at one feature.
The argument: **interpretability is a property of the feature ranking, not the
architecture**. A change to a step the pipeline treats as preprocessing damaged
readability far more than accuracy, which an accuracy-only evaluation would never
have surfaced.

**What did *not* move, checked rather than assumed.** Tables 4.2, 4.3 and 4.6/4.7
are byte-identical between the pre- and post-fix sweeps — the first two are
synthetic and the third runs on Glass, so no Concrete or PhiUSIIL change can
reach them. Table 4.1 is unchanged too: its rows are the library-default arms,
and only the *demo-tuned* tree and mixture rows of the underlying table moved.
Worth stating because "unchanged" is a claim like any other.

**Tables 4.6/4.7 re-quoted twice, and the ordering flipped both times.**
Complement-rule-leads → isolation-forest-by-0.038 → level to 0.002. Three
orderings from one experiment is the tell that all three were noise; the table
now says so rather than naming a winner.

---

## 4. Running it

```bash
reproduce/run_all_tables.sh my-label            # 24 min, all 11 tables
reproduce/run_all_tables.sh --fast smoke-check  # minutes, stamped NOT CITABLE
uv run --project tribble-cluster --with scipy \
    python reproduce/experiments/run_cluster_experiment.py --all   # Ch3 figures
```

`--fast` reduces seeds only on the four slow tables and records the seed set
*per table*, so a thin cell can never be mistaken for a full one.
`REPRO_THETA_SWEEP=0.5,0.6,0.7,0.8,0.9,0.99,1.1` for the operating curve.
Datasets resolve under `data/` (`GRAD_SCHOOL_DATA` overrides); nothing is ever
written inside a submodule.

---

## 5. Needs your judgement

**Goal G5.** Not "corroborate on a real dataset" any more. Either characterise
and guard quantile's instability, accept that heavy skew needs a target transform
rather than a better partition, or default to uniform for predictability. The
skew sweep is one of the cheapest tables in the harness (39 s), so this is
minutes of compute, not hours. **If G5's "complete" status ever reached the
committee, that needs correcting.**

**`origin/main`'s submodule gitlinks were broken — resolved 2026-08-19.**
PR #28 had added `branch = main` to `.gitmodules`, but the recorded SHAs
`56ac26e` and `de699c5` did not exist on their remotes. `branch = main` only
affects `git submodule update --remote`; ordinary clone and CI checkout use the
recorded SHA, so a fresh clone of main failed. main has since re-pinned:
`deb4a78` (bump submodules to latest main) and `ed0f1c8` (tribble-fis →
`141596e`) left it recording `tribble-fis 141596e`, `tribble-cluster 635ed6e`,
`tribble-opt 8049b94` — each the tip of its remote `main`, verified resolvable
on all three remotes 2026-08-19 via `git ls-remote`. A fresh clone of main now
resolves all three; the submodules' working trees in this checkout were stale
commits behind those pins and were fast-forwarded the same day.

---

## 6. Still outstanding

- **Table 3.1's 4,096-point pair** is cited to NAFIPS by decision, not
  harness-reproduced. Raising `REPRO_NAIVE_CAP` would cost hours to re-derive a
  constant factor the chapter does not rest on.
- **Tables 3.2 and 3.3** have no generator; 3.3 needs a GPU host.
- **Table 6.3** is structural; **6.4**'s entry point is unconfirmed (moved to
  `AnalyticalDynamics/`).
- **ANFIS and GA-FIS adapters** absent, so those cells stay `N/A`.
- **BETH** — unchanged from `research/proposal-defense/HANDOFF_LOCAL_SESSION.md`
  §1. LOCO needs ≥3 classes and BETH is binary, so it needs its own one-class
  path; a research decision before a coding one.
- **Appendix A.4's composite column** is measured against the *restored*
  `composite`, which is not the pre-#34 blend — the blend ranked
  `URLSimilarityIndex` first, the restored one ranks it second. The section says
  which is which; worth a footnote at submission.

---

## 7. Traps worth not repeating

**Silence is not success.** Every defect in §1 produced plausible output or
exited zero. A submodule on the wrong commit, a driver crashing before writing
its results, experiments dying on import, a provenance file misstating its own
seeds, two datasets quietly swapped. Read the provenance, not the exit status.

**A conclusion can be reproducible and still wrong.** The retracted crossover and
the refuted skew hypothesis came from generators that ran correctly and
deterministically every time. Determinism is not evidence; the sample was too
small to support the story built on it.

**Attribute changes to one variable at a time.** One sweep had three: a solver
change, a rounded dataset, and a swapped feature set. I would have credited all
of it to the solver. The tell was CART and Random Forest moving — sklearn models
that the solver cannot touch. When a number moves, find something that *should
not* have moved and check it.

**Do not size an optimisation from a profiler.** cProfile charges per-call
overhead, so pandas `__getitem__` — deep internal call chain — showed as 57% of
runtime and inflated a 130 s seed to 257 s. I published a 19% speedup that was
really 9.8%. Profiles find hotspots; wall clocks size them.

**My own errors this session, for calibration.** `REPRO_THETA_SWEEP=1` producing
a table of zeros. An orphaned-row edit to Table 4.1 caught only on a later sweep.
Background waiters using `pgrep -f run_all_tables` that matched their *own*
command line and spun for over an hour. And a `git commit --amend` run in the
wrong repository, which rewrote the message of an unrelated commit. Two of the
four were directory or self-reference confusion; check which repo you are in.
