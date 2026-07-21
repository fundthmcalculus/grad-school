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
```

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
