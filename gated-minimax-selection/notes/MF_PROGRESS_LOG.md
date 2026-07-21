# Membership-Function Progress Log

Running journal of the branch chain implementing `MEMBERSHIP_ROADMAP.md`. Each
phase is its own branch built on the previous. **Honest record — negative
results are kept, not deleted.** Newest phase at the bottom.

Branch chain (each builds on the one above):
```
feat/option-d-multiscale          (base: multi-scale selection + scaling)
  └─ feat/mf-phase1-ramp          Phase 1 — direct ramp MF        [DONE, negative result]
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
