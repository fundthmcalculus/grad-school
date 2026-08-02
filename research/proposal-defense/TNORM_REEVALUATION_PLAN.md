# Re-evaluation plan — the `tribble-fis` t-norm/t-conorm fix

_Written 2026-08-01. Scope: update the submodule, re-run every reproduction script,
report what changed, and add a norm/conorm comparison table across datasets._

This is a plan, not results. Nothing below has been re-run yet. Items are ordered so
each one is independently completable and reviewable.

---

## 1. What the fix actually is

Upstream `tribble-fis/main` is now **f779a42 — "Fix t-norm/t-conorm issues and add
regression tests (#26)"**, which lands four distinct defects (issues #22–#25) in
`src/tribblefis/gauss_math.py`, plus a 319-line regression suite.

| # | Defect | Fix |
|---|---|---|
| 22 | `t_norm(x, None, norm)` / `t_conorm(x, None, norm)` — the **array-reduction branch dropped `selected_norm`** on the recursive call, silently falling back to `DefaultNormCornorm` (`min/max`) | forward `selected_norm` |
| 23 | Hamacher t-conorm implemented as `(x+y)/(1-xy)`, which **is not a t-conorm** — it leaves `[0,1]` and diverges as `xy → 1` | `(x+y-2xy)/(1-xy)`, the De Morgan dual, with the `xy=1` limit set to 1 |
| 24 | Hamacher t-norm `xy/(x+y-xy)` → **NaN at `x=y=0`** (removable singularity, limit 0) | guarded divide, returns 0 |
| 25 | Anomaly rule added `threshold` to class firings **before** clipping, feeding values > 1 into operators only defined on `[0,1]` — producing negative anomaly memberships | `np.clip(..., 0, 1)` before aggregation |

**Measured damage of the old code** (50×50 grid over `[0,1]²`, run here today):

```
hamacher t-norm   (OLD): 1 NaN cell out of 2500
hamacher t-conorm (OLD): 1521 / 2500 cells outside [0,1]   (61%),  max value 96.999
hamacher t-conorm (NEW): 0 cells outside [0,1]
anomaly boost θ=0.99 (OLD, unclipped): firings up to 1.39 fed to the conorm
```

So the Hamacher conorm was wrong on 61% of its domain, and #22 means that in the two
places the reduction form is called, **the `hamacher` setting was never honoured at
all** — every "hamacher" run in the record was silently a `min/max` run.

## 2. Blast radius — what can and cannot have changed

I traced every `t_norm`/`t_conorm` call site in the library. This bounds the re-run.

| Call site | Form | Affected by |
|---|---|---|
| `gauss_math.py:397` (`tsk_firing_strengths`, anomaly label) | reduction | #22, #23, #25 |
| `gauss_math.py:494` (`simple_gaussian_predict`, anomaly label) | reduction | #22, #23, #25 |
| `gauss_math.py:418/425`, `481/482` (normal class rules) | pairwise | #23/#24 **only if** `norm_conorm="hamacher"` |
| `ruspini.py:165/166` | pairwise | #23/#24 only at `hamacher` (default is `probability`) |
| `fuzzytree/firing.py:44` | pairwise | #23/#24 only at `hamacher` (default `probability`) |

The only non-default norm consumer in the whole tree is
`gaussian_mixture/beth-anomaly.py` (`norm_conorm="hamacher"`), mirrored by
`reproduce/tables/table_4_4_openset.py` (`REPRO_ANOM_CONORM` default `hamacher`).

**Therefore the honest prior is:**

- **Table 4.4 + the θ sweep (Fig 4.2) must change** — they are the only harness
  experiments that touch the anomaly path, and they ran at `hamacher`, which the
  library was ignoring.
- **Tables 3.1, 4.1, 4.2/4.3 (G5, G5b), 6.1, the Concrete reconciliation, and the
  hyperparameter×normalization matrix should be bit-identical.** They are
  regression/classification without an anomaly label, at the `min/max` and
  `probability` defaults — paths the diff does not touch.

I will **verify** that rather than assume it (item 3 below runs everything at both
pins and diffs), but you should not expect the fix to move the Concrete or PhiUSIIL
numbers.

### A caution on "the results are stronger"

I do not think the direction is predictable, and I would rather say so now than
walk it back later. Two effects push opposite ways at the Table 4.4 operating point:

- Clipping (#25) mostly does **not** bite where it matters. For a genuinely unseen
  point the Gaussian MFs underflow to ≈0, so `0 + 0.99 = 0.99` is already in range
  and the anomaly membership is unchanged. Clipping only changes points whose known
  class already fires above 0.01 — which previously got a *negative* anomaly
  membership and now get exactly 0. Neither wins the argmax. So on `min/max` the
  θ=0.99 numbers may barely move.
- Honouring `hamacher` (#22) is the real change, and Hamacher is a **strong**
  disjunction — the aggregate of the known-class rules gets larger, so its
  complement gets smaller, so **fewer** points get flagged. That could lower
  detection as easily as raise it.

The recorded pre-fix baseline to beat (`ACTION_ITEMS.md` line 93–94, Glass, 2 seeds):
**best J = +0.155 at θ = 0.80**, J = +0.075 at the θ = 0.99 default, best ≈ 31%
detection at 13% false alarm, and the complement rule statistically indistinguishable
from isolation forest / one-class SVM. If the new numbers beat that, good. If they do
not, the finding is that the fix corrects the *mechanism* — a hamacher run is now
actually a hamacher run — and the comparison table in item 4 becomes the deliverable
rather than a single improved cell.

---

## 3. Two blockers that need your call before I can re-run

### 3.1 The submodule pin conflicts with your unmerged PR

The submodule is pinned at **d0d6714**, `fix(regression): hold the extreme bucket
means fixed during the consequent solve`. That commit is **not on `main`** — it lives
only on `origin/fix/pin-extreme-bucket-means`, which is Tier 0.1 in `NEXT_STEPS.md`,
still unmerged. Upstream `main` (f779a42) does not contain it.

So "update the submodule to main" would silently revert `pin_extremes`, and every
Concrete number in Ch 4 — which commit `3f4c1da` just wrote into the text — would
stop describing the shipped code.

Options, in the order I'd recommend them:

1. **Merge `fix/pin-extreme-bucket-means` into `tribble-fis` main upstream, then pin
   to that merge.** Clears Tier 0.1 at the same time. Needs you (I do not have push
   rights to `tribble-fis`, and it is outside this session's repo scope).
2. **Pin to a local merge of `main` + the fix branch**, pushed to a branch of
   `tribble-fis`. Works, but the submodule then points at a non-`main` commit again.
3. **Pin to plain `main` and drop `pin_extremes`.** Cheapest, but Ch 4 §4.x would
   need its pinning paragraph struck, and the rule-base defect that commit fixed
   (consequents reading `THEN output is -0.81` for a non-negative target) comes back.

**I recommend (1).** Until it happens I will run the comparison with a local merge so
the work is not blocked, and mark the pin as provisional.

### 3.2 Two of the datasets are unreachable from this environment

The egress policy allows PyPI and GitHub and **blocks `archive.ics.uci.edu` and
`api.openml.org`** (403 at the CONNECT, confirmed in the proxy's failure log).

| Dataset | Status | Consequence |
|---|---|---|
| **Concrete** | ✅ available — built from `AEEM6097/project-data/Concrete_Data.xls` (needs `uv run --with xlrd` once; it then caches the CSV) | Ch 4 / Ch 6 tables run |
| **Glass** | ❌ **absent** — gitignored at `/glass.csv`, never committed, and the UCI fetch is blocked | **Table 4.4 cannot run at all** — the one table the fix actually changes |
| **PhiUSIIL** | ❌ `ucimlrepo` id 967 blocked | those columns → `N/A` in Tables 4.1 / 6.1 |
| **BETH** | ❌ not in the repo (known) | unchanged from before |

Glass is the sharp one. Options:

1. **You commit `glass.csv`** (214×10, ~12 KB) and un-ignore it — it is a public UCI
   dataset and the harness already claims it is "public, in-repo". Cleanest, and it
   makes Table 4.4 reproducible for anyone who clones.
2. **Allow-list `archive.ics.uci.edu`** in the environment's network policy.
3. **Substitute** an sklearn-bundled multiclass set (`wine`, 178×13×3, or
   `digits`) for the leave-one-class-out protocol. Runs today with no network, but
   changes the dataset the chapter cites, so it is a supplement, not a replacement.
4. Pull Glass from a GitHub mirror — allowed by the proxy, but I would rather not
   introduce a dataset of unverified provenance into a reproducibility harness
   without you agreeing to it.

**I recommend (1), with (3) added regardless** — a second dataset for the open-set
protocol is worth having anyway, since `ACTION_ITEMS.md` already flags Glass's 214
samples as "a stress test, not a demonstration."

---

## 4. The norm/conorm comparison table

You asked for a permutation table across datasets. There is an API constraint to
settle first.

### 4.1 Norm and conorm are currently a single coupled knob

`NormConorm = Literal["min/max", "probability", "luk", "hamacher"]` selects the
t-norm **and** its conorm together — you cannot ask for "product AND, Łukasiewicz
OR". So "permutations of norm/conorm" means one of:

- **(a) 4 coupled pairs** — `min/max`, `probability`, `luk`, `hamacher`. No library
  change; runnable immediately.
- **(b) 16 decoupled combinations** — split `norm_conorm` into `t_norm` /
  `t_conorm` (keeping the coupled name as a back-compatible alias). A small,
  self-contained change to `gauss_data.py` + the call sites, and it is the more
  interesting table: De Morgan-dual pairs versus mismatched pairs is a real
  question, and mismatched pairs break the complement identity the anomaly rule
  depends on, which is worth showing.

**I recommend (b)**, done as its own upstream PR so the harness change and the
library change stay separable. (a) is the fallback if you want the table today.

### 4.2 The plumbing is incomplete, and that is itself a result

Which models can even accept a norm right now:

| Model | Knob | Default | Sweepable? |
|---|---|---|---|
| MoG classifier | `norm_conorm` | `min/max` | ✅ |
| Anomaly rule | `AnomalyParameters.norm_conorm` | `min/max` | ✅ |
| Ruspini models | `norm_conorm` | `probability` | ✅ |
| Fuzzy tree (clf + reg) | `t_norm` | `probability` | ✅ t-norm only |
| **MoG regressor** | — | hard-wired `min/max` | ❌ **not exposed** |
| **HME gate** | — | hard-wired `probability` (`hme.py:211`) | ❌ **not exposed** |

So a norm sweep on the **regression** tables (Concrete, the whole of Ch 6) is
impossible without a library change: `tsk_firing_strengths` reads the norm off
`anomaly_details`, and with no anomaly parameters it silently gets `None → min/max`.
Two more small PRs (`norm_conorm` on `MixtureOfGaussiansFuzzyRegressor`, `t_norm` on
the HME gate) unlock it. I'd fold these into the same upstream branch as 4.1(b).

### 4.3 Proposed table shape

```
reproduce/tables/table_norm_conorm_matrix.py
  → reproduce/outputs/table_norm_conorm_matrix.{md,csv}
```

Axes, held to what is actually runnable:

- **t-norm** × **t-conorm** ∈ {min/max, probability, luk, hamacher}² (16, or 4
  coupled under 4.1(a))
- **task/dataset**: Concrete (regression, R²) · Glass or wine (classification,
  accuracy) · Glass/wine leave-one-class-out (open-set, detection / false-alarm /
  Youden's J) · PhiUSIIL if the network opens
- **model**: flat MoG · fuzzy tree · HME, per `_fuzzy_models.py`
- mean ± std over `common.SEEDS`, `N/A` where the knob is not plumbed — the
  gaps in the table are the honest record of 4.2

Plus a **θ × conorm** panel for the anomaly rule, since the θ sweep and the conorm
choice interact and neither is interpretable alone.

---

## 5. Work items, in order

Each is a separate commit on `claude/tribble-fis-evaluation-s97850`.

| # | Item | Depends on | Est. |
|---|---|---|---|
| **1** | **Capture the pre-fix baseline.** Run every harness script at the current pin (d0d6714) and archive `reproduce/outputs/` as `outputs/baseline-d0d6714/`. Without this there is nothing to diff against — the outputs directory is gitignored and no prior run is stored. | 3.2 (Concrete only; Glass blocked) | 1 h |
| **2** | **Update the submodule** to the chosen commit (per 3.1) and re-run everything. Archive as `outputs/postfix-<sha>/`. | 3.1 decision | 1 h |
| **3** | **Diff report** — `reproduce/outputs/FIX_IMPACT.md`: every table, old vs new, per cell, with "unchanged" stated explicitly where it is unchanged. This is the "what works / what doesn't" deliverable. | 1, 2 | 2 h |
| **4** | **Make Table 4.4 runnable** — commit `glass.csv` (your call) and/or add the sklearn-`wine` arm; re-run with the θ sweep on. | 3.2 decision | 1–2 h |
| **5** | **Upstream PR: decouple `t_norm` / `t_conorm`** + expose the norm on the MoG regressor and the HME gate. With tests. | — | 3 h |
| **6** | **Build `table_norm_conorm_matrix.py`** and generate it across every runnable dataset. | 4, 5 | 3 h |
| **7** | **Register** the new table in `manifest.py`, document it in `reproduce/tables/README.md`. | 6 | 30 m |
| **8** | **Write the findings into the text** — Ch 4 §4.3.5 (the complement rule, corrected), Ch 6 if anything moved, and a note in `ACTION_ITEMS.md` retiring the stale line 93–94 numbers. | 3, 6 | 2 h |
| **9** | **Re-run the non-`tribble-fis` scripts** (Table 3.1 mergeVAT under `tribble-cluster`, the gated-minimax pipeline, the `tribble-opt` TSP sample) to confirm they are untouched, and record that they are. | — | 1 h |

Items 1, 2, 3 and 9 are the "full evaluation" you asked for. Items 4–7 are the
comparison table. Item 8 is what makes it count.

---

## 6. Decisions I need from you

1. **Submodule pin** — merge the `pin-extreme-bucket-means` PR upstream (recommended),
   or pin to a local merge, or drop the pin fix?
2. **Glass** — commit `glass.csv`, allow-list UCI, add the sklearn-`wine` arm, or
   some combination? Table 4.4 is blocked until one of these.
3. **Norm/conorm** — decoupled 4×4 with a library change (recommended), or the 4
   coupled pairs runnable today?
4. **PhiUSIIL** — leave those columns `N/A`, or allow-list UCI so Tables 4.1/6.1 fill in?

Nothing in items 1, 3 (partially) and 9 is blocked on any of these, so I can start
there on your word.
