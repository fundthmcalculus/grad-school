# Membership-Function Progress Log

Running journal of the branch chain implementing `MEMBERSHIP_ROADMAP.md`. Each
phase is its own branch built on the previous. **Honest record — negative
results are kept, not deleted.** Newest phase at the bottom.

Branch chain (each builds on the one above):
```
feat/option-d-multiscale          (base: multi-scale selection + scaling)
  └─ feat/mf-phase1-ramp          Phase 1 — direct ramp MF        [DONE, negative result]
     └─ feat/mf-phase2-kernel     Phase 2 — gaussian-kernel MF    [DONE, works + caveat]
        └─ feat/mf-phase3-hierarchy  Phase 3 — POU + fuzzy model  [DONE, works]
           └─ feat/mf-phase4-bands   Phase 4 — band merge/cleanup [DONE, partial + finding]
              └─ feat/mf-phase6-validation  Phase 6 — soft metrics [DONE, negative result]
```
(Phase 5, the one-pass refactor, was **not attempted** — it is plumbing, and the
Phase 6 result below changes what should be built next; see the TL;DR.)

---

## TL;DR (read this first)

**What works.** Membership functions can be generated *directly* from the
multi-scale persistence structure with **no separate fitting stage** and **no
accuracy cost**: argmax of the generated fuzzy partition reproduces the hard
Option-D labels exactly at every scale, and each scale is a valid Ruspini
partition of unity (error ~1e-16). On genuinely **nested** data (`many_scale`
8/4/2) the whole fuzzy hierarchy is recovered perfectly. This is a real, working
capability (Phases 1–3).

**What does NOT work — the two load-bearing negative results.**
1. **The persistence ramp is crisp by construction** (Phase 1). A block's members
   are fixed at birth and don't grow until death, so nothing attaches in the
   (birth, death) interval — μ ∈ {0,1} always. Gradation had to come from a
   distance *kernel* instead (Phase 2).
2. **The resulting fuzzy memberships are ultrametric, not calibrated
   posteriors** (Phases 2 & 6). Minimax distance is constant across a foreign
   cluster, so membership is a **step function** (own cluster ~0.67, others
   ~0.33) with no boundary resolution. Measured against true Gaussian posteriors
   they score **worse than crisp 0/1 labels** (Brier 0.12–0.21 vs 0.02–0.10).
   → For calibrated *spatial* soft memberships the minimax route is the wrong
   tool; the feature-space surrogate (Option C, Gaussian/Mahalanobis) is. The
   minimax MFs are *hierarchy-aware*, good for hard multi-scale partitions, not
   for smooth uncertainty.

**Also found (Phase 4):** birth-height banding only works when each cluster
occupies a narrow birth range. A diffuse/chained cluster produces many nested
sub-blocks (one cluster → 18 blocks spanning birth 25→180), which shreds across
bands and fools a containment test. For single-level, widely-varying-spread data
the flat set-cover is correct and multi-scale banding is the wrong tool.

**Net recommendation for tomorrow.** Multi-scale persistence is strong for
recovering *nested hard* structure (its intended use). Direct MF generation is
viable and free for that. But "fuzzy" here means *hierarchy-aware ultrametric*,
not *calibrated spatial* — if the thesis needs the latter, pair the multi-scale
selector (for structure) with feature-space kernels (for the soft MFs), rather
than pushing the minimax ramp/kernel further. Next concrete build: a
single-vs-multi-level gate (defer to flat cover when the data is one antichain),
then Option-C-style spatial MFs per selected block.

---

## Phase 1 — direct persistence-ramp membership (`feat/mf-phase1-ramp`)

**Goal:** turn each discovered scale band into a *fuzzy* partition by reading the
Mapping-2 persistence ramp off every selected block, no separate fit.

**Implemented** (`multiscale_persistence.py`): `block_membership`,
`band_memberships`, `defuzzify_memberships` (argmax + proximity tie-break),
`multiscale_memberships`. Ramp: `mu_B(x) = clip((death−d_B)/(death−birth),0,1)`,
core (`d_B ≤ birth`) → 1, with `d_B(x)=min_{y∈B} D*(x,y)`.

**What works:**
- Argmax of the fuzzy partition reproduces the hard `assign_band` **exactly**
  (agreement 1.000) on all hierarchical datasets; per-level ARI identical
  (nested [1.0/…], three-level [8,4,2] all 1.0, density [1.0/…]). So as a
  selection→labels path it is faithful.

**What does NOT work — the ramp is crisp by construction (key finding):**
- Measured `graded fraction = 0.000` on **every** dataset tried, including
  `bridged_gaussians` and two Gaussians overlapped down to sep=1.5σ. No
  membership value is ever strictly in (0,1).
- Verified directly: across all 159 blocks of a maximally-overlapping mixture,
  the number of (block, point) pairs with `birth < d_B(x) < death` is **0**.
  Members sit at `d=0` (→ μ=1); every non-member sits at `d ≥ death` (→ μ=0).
- **Why (proof):** a dendrogram block's member set is fixed at its birth height
  and does not grow until it *dies* (merges with its sibling). So no point ever
  attaches in the open interval (birth, death) — the ramp's graded zone is
  provably empty. `d_B(x) ∈ {0} ∪ [death, ∞)` for every x. The linear ramp
  therefore collapses to a 0/1 indicator regardless of data.
- Consequence: the original `ivat_mf.mapping2_persistence` "graded" MF is, for a
  single fixed block, also crisp for the same reason.

**Takeaway / pivot for Phase 2:** genuine gradation cannot come from the
birth/death ramp. But the *raw* minimax distances `d_B(x)` to a block **do** vary
across non-members (e.g. 0.369, 0.371, 0.383, 0.445, …) — the crispness is only
in how the ramp thresholds them. So Phase 2 replaces the ramp with a **distance
kernel** `mu_B(x) = f(d_B(x)/scale_B)` (smooth, e.g. half-max at a
block-characteristic scale), which turns the surviving graded distance
information into graded membership. This is the same idea as Option A's auto-tuned
kernel, but driven by the block's own minimax geometry.

**Status:** Phase 1 committed as a faithful-but-crisp baseline + this negative
result. Proceed to Phase 2 (kernelized membership).

---

## Phase 2 — gaussian-kernel membership (`feat/mf-phase2-kernel`)

**Goal:** recover genuine gradation that Phase 1's ramp cannot, by kerneling the
minimax distance instead of thresholding it.

**Implemented:** `block_membership(..., kernel=)` now supports `'ramp'` (Phase 1,
kept) and `'gaussian'` (new default): `mu_B(x) = 2**(-(d_B(x)/death_B)**2)` — a
Gaussian in minimax distance with **half-max at the block's death (escape)
height**, the principled scale at which the block dissolves into its parent.
Threaded through `band_memberships` / `multiscale_memberships` via `kernel=`.

**Scale choice (measured):** half-max at `death` gives graded_frac 0.29–0.42 with
ARI preserved; at `birth` it stays crisp (≈0, non-members sit at d≫birth); `mid`
≈ `death`. `death` chosen — principled and best.

**What works:**
- Genuine gradation: graded_frac = **0.417 / 0.292 / 0.375** on nested /
  three-level / density (was 0.000 for the ramp).
- **Labels unchanged:** argmax of the gaussian partition still equals the hard
  `assign_band` exactly on every band (ARI at every level unchanged from
  Option D). So we gained fuzziness for free — no accuracy cost.

**Important caveat — the gradation is ULTRAMETRIC, not spatial (key finding):**
- Minimax distance is piecewise-constant across clusters: every point in a
  *foreign* cluster has the *same* bottleneck distance to block B (their common
  merge height). So `mu_B` is **constant within each cluster** and graded only
  *between* clusters.
- Concretely (nested, fine block 0): 20 points read μ≈1 (its own sub-cluster),
  40 read μ≈0.5 (the two *sibling* sub-clusters in the same super-cluster),
  60 read μ≈0 (the far super-cluster). The 0.5 is the hierarchical-proximity
  signal — siblings are "half in" — which is meaningful and correct for a
  hierarchy, but it is NOT a smooth within-cluster spatial gradient.
- Implication: this route gives **hierarchy-aware (ultrametric) fuzzy
  memberships**, not RBF-style spatial ones. Smooth within-cluster gradients
  would require the feature-space surrogate (Option C), which trades away
  non-convex support. Documented so we don't over-claim "smooth fuzzy MFs".

**Status:** Phase 2 committed. Gradation achieved and characterized honestly.
Next: Phase 3 — partition-of-unity across a band's blocks + the cross-scale fuzzy
model (and decide how normalization interacts with the 0.5 sibling mass).

---

## Phase 3 — partition-of-unity + fuzzy hierarchy (`feat/mf-phase3-hierarchy`)

**Goal:** normalize each band to a Ruspini partition of unity and package the
per-scale partitions into one multi-scale fuzzy model.

**Implemented:** `normalize_partition` (column-sum-to-1; uncovered points left
all-zero, deliberately possibilistic), `FuzzyHierarchy` dataclass (per-scale `U`,
`.level`, `.defuzzify`, `.partition_of_unity_error`, `.coverage`), and
`build_fuzzy_hierarchy(Dstar, kernel=, normalize=)`.

**What works:**
- **Partition-of-unity error = ~1e-16** (machine zero) on covered points at every
  scale of nested / three-level / density; coverage 1.00.
- Normalization is **argmax-invariant** (verified `argmax_unchanged=True` every
  band) — positive per-column scaling — so ARI at every level is unchanged from
  Phase 2 / Option D. We get a valid fuzzy partition for free.
- The model is now a clean object: `fh.level(i)` = fuzzy partition at scale i,
  `fh.defuzzify(i)` = its hard labels, fine→coarse.

**Design note (the 0.5-sibling mass):** normalization redistributes the
ultrametric sibling mass from Phase 2. A fine point that read [1.0, 0.5, 0.5, 0,
0, 0] becomes [0.5, 0.25, 0.25, 0, 0, 0] — its own cluster now 0.5, each sibling
0.25, summing to 1. Argmax still its own cluster (correct), but the absolute
"confidence" is diluted by how many siblings exist. This is inherent to
partition-of-unity over ultrametric memberships; the possibilistic (unnormalized)
view keeps own=1.0 and is the better readout of "how core is this point".
Both are available via `normalize=`.

**Not done here (deferred):** a single cross-scale flat blend (t-conorm over all
scales) — the per-scale hierarchy is the more honest object and the levels are
what the evaluation needs, so a flat blend is left as optional.

**Status:** Phase 3 committed. Valid fuzzy partition-of-unity per scale, no
accuracy cost. Next: Phase 4 — soft band membership (the research-interesting
piece; also targets the `log_separated` small-n over-segmentation from the
scaling study).

---

## Phase 4 — band discovery fix (`feat/mf-phase4-bands`)

**Goal:** fix the `log_separated` small-n over-segmentation (SCALING_STUDY §4).
The roadmap guessed "soft bands"; the evidence pointed elsewhere, so I followed
the evidence (documented pivot).

**Evidence-driven pivot — containment, not soft weighting.** Probed what actually
distinguishes a genuine scale hierarchy from a spurious birth-gap split: in
`many_scale`, 8/8 and 4/4 finer blocks are **nested inside** coarser blocks
(containment). In over-split `log_separated`, adjacent "bands" are **disjoint
siblings** (antichain), 0/1 nested. So the fix is to **merge adjacent bands not
related by containment**, plus **drop single-block bands** (a 1-cluster partition
carries no information — usually the near-root scale).

**Implemented** (`select_multiscale`): `merge_antichain=True`, `nest_frac_thresh`
— a containment-aware merge pass over the raw birth-gap bands; and a post-filter
dropping k=1 bands (keeping the finest if that would empty the result).

**What works:**
- **No regression on genuine hierarchies** (the critical property): `many_scale`
  stays [8,4,2] @ ARI 1.0 for all n=100..2000; nested/three-level/density all
  unchanged @ 1.0.
- **single_scale cleaned up**: n=500 now [5] (was [5,2]); the spurious near-root
  band is gone at that size. Large-n keeps a defensible k=2 coarser view.
- **log_separated improved but NOT fixed**: small-n ARI 0.0 → ~0.57 (no more
  degenerate all-singleton [1,1,1]).

**What does NOT work — birth-banding shreds a chained diffuse cluster (key finding):**
- Flat `coverage_cover` gets `log_separated` **[3] @ ARI 1.0 even at n=100** — so
  the 3 clusters DO form clean blocks; this is a **band-logic failure, not
  sampling**.
- Root cause: single-linkage **chains through the sparse diffuse cluster**
  (σ=175), so that ONE cluster produces **18 significant blocks spanning birth
  25→180** (vs 1 and 2 for the tight/medium clusters). Birth-height banding
  therefore fragments a single cluster across many bands.
- Worse, those 18 blocks are internally **nested** (small fragment ⊆ big
  fragment), so the containment test is **fooled**: it reads one cluster's
  caterpillar as a genuine multi-level hierarchy and refuses to merge it. The
  antichain-merge can't undo the shredding.
- **Lesson:** birth height is a clean band coordinate only when each cluster
  occupies a *narrow* birth range. A diffuse/chained cluster violates this, and
  containment cannot distinguish "nested distinct clusters" from "nested
  fragments of one chained cluster." For single-level data with widely varying
  spreads, the **flat global set-cover is the right tool and multi-scale banding
  is the wrong one** — the value of multi-scale is specifically NESTED structure
  (`many_scale`), which it handles perfectly.

**Net:** Phase 4 is a real improvement (no-regression + cleaner single_scale +
antichain detection) with an honestly-bounded scope. `log_separated` at small n
remains a known failure of the birth-banding premise, not closed. A principled
future fix would gate multi-scale banding on a *global* single-vs-multi-level
test (e.g. "does flat cover already explain the data as one antichain?") and
defer to the flat cover when it does.

**Status:** Phase 4 committed with the honest partial result + the birth-banding
limitation pinned to its mechanism.

---

## Phase 6 — soft-metric validation (`feat/mf-phase6-validation`)

**Goal:** every earlier test scored only argmax (hard labels). Ask the question
that actually exercises the *graded* MFs: do they match true soft posteriors?

**Experiment:** two equal-variance 2-D Gaussians, separation swept 2→6. Build the
two (ground-truth) blocks, generate gaussian-kernel memberships, normalize, and
score the **Brier distance to the analytic posterior** `P(0|x) =
N0/(N0+N1)`. Baseline: crisp 0/1 labels as a soft prediction.

| sep | Brier fuzzy | Brier hard (0/1) | boundary pts |
|----:|------------:|-----------------:|-------------:|
| 2.0 | 0.136 | 0.096 | 0.30 |
| 3.0 | 0.200 | 0.042 | 0.09 |
| 4.0 | 0.208 | 0.016 | 0.04 |
| 6.0 | 0.122 | 0.000 | 0.00 |

**Result — negative, and clear:** the fuzzy MF is a **worse** soft predictor than
crisp labels at every separation. Mechanism (confirming Phase 2): with two blocks
the ultrametric distances make the normalized membership a **constant step** —
cluster-0 points all read ~0.67, cluster-1 points ~0.33 — *independent of
distance to the boundary*. So it is simultaneously (a) under-confident on easy,
far-from-boundary points (0.67 where the truth is ~1.0) and (b) devoid of
boundary resolution (no smooth 0→1 transition). Crisp labels at least nail the
confident points, hence the lower Brier.

**Interpretation:** `graded_frac = 0.5` earlier was misleading — the "gradation"
is one constant value per cluster, not resolution of uncertainty. The minimax
fuzzy MFs encode *hierarchical membership* (which cluster, and its siblings), not
*spatial confidence*. Calibrated spatial soft memberships require an actual
spatial kernel in feature space (Option C), which the minimax route deliberately
avoids (that is what buys it non-convex support).

**Status:** Phase 6 committed as a documented negative result. It redirects the
roadmap: stop trying to make the minimax ramp/kernel "fuzzy" in the calibrated
sense; use multi-scale for structure and feature-space kernels for soft MFs.
Reproduce: `python notes/phase6_soft_validation.py`.
