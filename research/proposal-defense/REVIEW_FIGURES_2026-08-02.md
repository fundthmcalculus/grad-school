# Figure production review — proposal-defense/, 2026-08-02

Scope: produce the fifteen undrawn figures called out in the prose, and review the
prose against the code while doing it. Producing a figure is a stronger check on a
paragraph than reading it, because a figure has to run: a sentence describing an
operator can be approximately right and survive, while a plot of that operator
either matches the shipped code or does not.

Companion to `REVIEW_2026-08-02.md`, which reviewed the text. This one reviews the
text *by drawing it*.

---

## At a glance

| | |
|---|---|
| Figures cited in the prose | 16 |
| Produced before this pass | 1 (Fig 3.2, by the Table 3.1 sweep) |
| Produced in this pass | 13 |
| Deliberately still placeholders | 2 (Fig 4.3, Fig 6.3), both with recorded reasons |
| Prose/code disagreements found | 3, all reconciled in the prose |
| Silently-wrong exported API found | 1 (`predict_trajectory`) |
| New generators | 15 modules under `reproduce/figures/` |

Nine of the fourteen produced figures are **computed, not drawn**: they run the
chapters' own code or read the archive of record's CSVs. Five are schematics, and
each says so in its own docstring and caption.

---

## The three places the prose and the code disagreed

Each of these was found by drawing the thing the paragraph described and getting
something the paragraph did not predict. All three are now fixed in the prose,
with the correction stated rather than quietly applied.

### F1. §4.3.1 had the t-norm and the t-conorm in the wrong places

The paragraph said the per-feature memberships for a class are combined "with a
fuzzy OR — a t-conorm — and that combination *is* the rule for that class", and
called each rule "a disjunction over at most $M \times p$ Gaussian terms".

`simple_gaussian_predict` does something else. Within a feature, the mixture
components are combined by the **t-conorm** (`local_vals = t_conorm(local_vals,
mf.evaluate(...))`). Across features, those per-feature results are combined by
the **t-norm** (`rule_firing[:, i] = t_norm(local_vals, rule_firing[:, i])`). A
rule is a conjunction of disjunctions, not a disjunction.

This matters more than a terminology slip. Under the prose's reading a class fires
if *any* feature matches; under the code a class fires only if *every* retained
feature matches to some degree. Those are different models with different failure
modes, and a committee member who read §4.3.1 and then read the code would find
the discrepancy in about a minute.

Chapter 2 §2.1 and Chapter 5 §5.3.5 were both already consistent with the code —
§5.3.5's "a class is recognized if *this* Gaussian fires or *that* one does" is
exactly the within-feature disjunction. Only §4.3.1 was wrong. Figure 4.1 draws
both levels explicitly, and calls the library's own `t_norm`/`t_conorm` with the
library's own `resolve_norm_pair()` default rather than reimplementing either, so
a change to the default norm family shows up in the figure.

### F2. §5.3.4's persistence ramp is crisp on an ultrametric

The section presents

$$\mu_B(x) = \mathrm{clip}\!\left(\frac{\text{death}_B - d_B(x)}{\text{death}_B - \text{birth}_B}, 0, 1\right)$$

as the membership function, and closes by noting that "a long-persistence block is
both more likely to be admitted and gentler in its membership falloff".

There is no falloff. $d_B(x)$ is the bottleneck height at which $x$ joins $B$: at
most $\text{birth}_B$ for a member, at least $\text{death}_B$ for a non-member,
since $\text{death}_B$ is by definition the height at which $B$ first absorbs
anything outside itself. The interval the ramp slopes across is empty by
construction, and $\mu_B$ takes only 0 and 1.

The code already knows this. `multiscale_persistence.block_membership` documents
the ramp as "CRISP by construction ... Kept for the record" and defaults to a
Gaussian in minimax distance with half-maximum at the death height — which grades
the non-member skirt and is genuinely fuzzy. So the chapter describes the variant
that is kept for the record and not the one that runs.

The claim §5.3.4 actually needs — every parameter is a merge height the hierarchy
already supplies, nothing is fitted — is true of both, and survives untouched.
Figure 5.3 draws both curves with every sample plotted at its own $d_B$, so the
empty interval is visible rather than asserted, and §5.3.4 gains a paragraph
stating the collapse and naming the shipped kernel.

### F3. §6.3.2's 28-day split is a property of the tuned tree, not the method

"On the Concrete dataset the tree splits first on cement content and then on age,
right at the standard 28-day curing mark — which is to say it recovers domain
knowledge nobody told it."

It does, and the recovery is exact: under `tribble-tree/demo_concrete.py`'s
settings (`max_depth=3, n_terms=2, top_n=4, min_soft_count=20`) the second split
is `Age ≤ 28`, to the unit. Under `FuzzyRegressionTree`'s **library defaults**
(`n_terms=3, top_n=-1`) the second split is Superplasticizer, and the Age
boundaries that do appear deeper in the tree are at 35.3 and 67.7 — nowhere near
28.

Both facts are fine; the sentence needs the qualifier, because Table 6.1's fuzzy
arms are quoted at library defaults elsewhere in the same document, and a reader
who fits the default tree to check the 28-day claim will not find it. Figure 6.1
prints the tuned tree, states the configuration, and names what the default does
instead.

---

## The defect

### F4. `MimoGaussianPredictorMemory.predict_trajectory` never advances a step

Found while attempting Figure 6.3, which asks for exactly this rollout.

`predict_trajectory` slices `trajectory.iloc[-(self.window_size):]` — exactly
`window_size` rows of history. `MemoryWindowFeatureExtractor.prepare_sequences`
then computes each row's long-term average over
`[i - window_size - memory_size + 1, i - window_size + 1)`. At the last row,
$i = \text{window\_size} - 1$, that interval is `[max(0, -memory_size), 0)` —
empty — so the value is NaN. The method's own guard

```python
if X_mem.isna().any(axis=1).iloc[-1]:
    break
```

then fires at step 0, and the function returns the initial window unchanged. The
caller gets a DataFrame of the right type and the wrong length, with no error.

Reproduced at `(window_size, memory_size)` = (3, 1), (4, 2), (10, 4) and (2, 1).
It is unconditional: for a window of exactly `window_size` rows the last row's
long-term interval is always empty.

The one-step `predict` path is unaffected, so §6.3.6's one-step-ahead claim
stands. Its *iterated* claim — "predicts either one step ahead or, iterated, a
whole trajectory" — does not, and neither does the R² 0.92 → 0.96 figure if that
was measured through this path. The fix is a one-line slice:
`window_size + memory_size` rows of history.

This is the **second** exported-and-wrong API this project has surfaced, after
`vat_prim_mst_seq`. Both share a shape worth naming: a function that returns a
plausible object instead of raising, so nothing downstream notices. A test that
asserts `len(predict_trajectory(w, n)) == len(w) + n` would have caught this one
on the day it was written.

Figure 6.3 is therefore skipped rather than approximated, with the reason recorded
in `reproduce/figures/registry.py`, in its caption, and in `ACTION_ITEMS.md`.

---

## Smaller findings

**F5. The pinned `tribble-cluster` has moved since the archive of record.**
`main-d0efefc/PROVENANCE.txt` records `tribble-cluster 5d44dfa`; the submodule
pinned in `HEAD` is `e3c27e6`. The two commits between them are
`fix: remove broken vat_prim_mst_seq from public API` and a lockfile sync, so no
measured number changes — but this is the third instance of the pattern that
`REVIEW_2026-08-02.md` Tier 1.1 asks to automate, and it is worth noting that the
guard would have flagged it correctly and harmlessly.

**F6. §5.3.1's "indistinguishable" is imprecise, and the figure had to be more
careful.** On the raw dissimilarity matrix the *inner* ring reads as a perfectly
clear block; it is the outer ring that does not. The measurable reason is that the
outer ring's own within-ring distances reach 8.4 while the nearest inner-to-outer
distance is 2.6 — which is precisely the configuration a Euclidean prototype
cannot resolve, and a sharper statement of the point the section is making. Figure
5.1 computes both numbers and the caption now states them.

**F7. Figure 4.1 shows degenerate mixture components, and that is worth keeping.**
Several of the automatically-selected Gaussians on Glass have $\sigma = 0$ — a
zero-width spike on a single observation, which is what the automatic component
count yields on a 214-sample dataset with six classes. The figure does not hide
them. It is a fair thing for a committee to ask about, and better asked from a
figure the author drew deliberately than discovered later.

---

## What was produced

| Fig | File | Kind | Source of what it shows |
|---|---|---|---|
| 1.1 | `01-structure-before-search` | schematic + measured | rule counts exact; time from Table 4.1; the conventional side prints "no measured baseline" |
| 1.2 | `01-pipeline-roadmap` | schematic + measured | every stage claim read from its owning table's CSV |
| 2.1 | `02-fis-components` | schematic | — |
| 2.2 | `02-vat-rdi` | computed | `circle_random_clusters` + `compute_vat` |
| 2.3 | `02-persistence` | computed | scipy single linkage; persistence over hierarchy nodes |
| 3.1 | `03-pvat-reorder` | schematic | — |
| 3.2 | `03-complexity-fit` | computed | *pre-existing*, by the Table 3.1 sweep |
| 4.1 | `04-mog-classification` | computed | the harness's own three-call MoG fit on Glass |
| 4.2 | `04-anomaly-sweep` | measured | `table_4_4b_theta_sweep.csv` |
| 4.3 | `04-rtiot-confusion` | **skipped** | the experiment does not exist (§4.3.1) |
| 5.1 | `05-minimax-transform` | computed + measured | matrices computed; ARIs from `table_5_1_battery.csv` |
| 5.2 | `05-band-discovery` | computed | `select_multiscale`, same call as Table 5.2 |
| 5.3 | `05-persistence-ramp` | computed | `block_membership`, both kernels |
| 6.1 | `06-fuzzy-tree` | computed | `render_tree_text` on two demo-configured fits |
| 6.2 | `06-hme-structure` | schematic | — |
| 6.3 | `06-mimo-rollout` | **skipped** | `predict_trajectory` is broken (F4) |

Everything runs from one entry point:

```bash
reproduce/figures/make_figures.py --list     # inventory, including why each skip is a skip
reproduce/figures/make_figures.py            # draw all, install into prose/fig/
```

---

## What this changes about what the document can claim

Three things, none of them cosmetic.

**The two load-bearing figures exist.** The review's Tier 1.4 named Fig 1.2 and
Fig 5.2 specifically. Fig 1.2 now orients the reader with every stage's claim read
from the table that owns it, so it cannot drift from the tables the way four
rounds of prose did. Fig 5.2 shows Chapter 5's mechanism and its result together,
running the real selector — the granularities `[8, 4, 2]` in the panel titles are
`sel.granularities()`, not typed in.

**Nine figures are now a second, independent check on the tables.** A figure that
reads `table_4_4b_theta_sweep.csv` cannot disagree with Table 4.6. A figure that
calls `select_multiscale` cannot disagree with Table 5.2. This is the mechanism
D8 asked for — chapters citing a named artifact instead of restating values —
applied to figures first because figures are where it was cheapest to install.

**Two placeholders now argue for themselves.** A reader reaching Figure 4.3 or 6.3
finds a sentence saying what is missing and why, not a grey box. For 6.3 that
sentence is a defect report, which is more useful to the committee than a plot
would have been.

---

## Remaining

1. **Look at all fourteen on paper.** They were composed on screen at 200 dpi;
   type sizes were chosen for a printed text block but not yet verified on one.
2. **Fix the `predict_trajectory` slice** and then draw Figure 6.3. It also
   unblocks a generator for Table 6.4, which `PROVENANCE_MAP` still marks
   ungenerated while Chapter 6 calls it the clearest result.
3. **Quantify the correction-rule pass** and then draw Figure 4.3. Already Tier 1
   item 5 in `NEXT_STEPS.md`; the figure is waiting on it, not the reverse.
4. **Re-run `make_figures.py` after any harness re-run**, since nine figures read
   the archive of record. `REPRO_ARCHIVE=<label>` pins a specific one, and every
   data figure prints the label it drew from.

---

*Review dated 2026-08-02. Figures drawn against `reproduce/outputs/main-d0efefc/`,
`tribble-fis` `d0efefc`, `tribble-cluster` `e3c27e6`.*
