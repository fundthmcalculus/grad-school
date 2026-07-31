# Roadmap — Generating Membership Functions Directly

*Next steps from the current state to a native fuzzy multi-scale model. Design
sketch, not yet implemented.*

---

## Where we are

The pipeline today is **two-stage**: select blocks, then (optionally) fit
membership functions onto the selected blocks.

```
D ──minimax──▶ D* ──single-linkage──▶ dendrogram
      │                                    │
      │                        enumerate blocks (birth, death, persistence)
      │                                    │
      │                     significance gate (MAD-outlier persistence)
      │                                    │
      ├── flat:  coverage_cover ──▶ hard set-cover partition
      └── multi-scale: select_multiscale ──▶ per-band hard partitions
                                             │
                        THEN a separate fitter turns blocks into MFs:
                        · Mapping 2 (ivat_mf): persistence-ramp μ per block
                        · Option B (ruspini_mf): partition-of-unity
                        · Option A (auto_select_mf_v2): auto-tuned Ruspini
                        · Option C (feature_space_mf): interpretable rules
```

`select_multiscale` currently ends at a **hard** assignment (`assign_band`,
argmin minimax distance). The MF machinery exists but is bolted on afterward and
re-derives shapes (medoid Gaussians, Mahalanobis surrogates) rather than reading
them off the structure that selection already computed.

**Goal:** emit graded membership functions *directly* from the persistence
structure, in one pass, with the multi-scale bands as first-class fuzzy
partitions — no separate surrogate-fitting stage.

## The key observation

Every dendrogram block already carries a native membership function. Mapping 2
(`ivat_mf.mapping2_persistence`) is exactly this:

```
d_B(x) = min_{y in B} D*(x, y)            # minimax distance from x to block B
μ_B(x) = clip( (death_B − d_B(x)) / (death_B − birth_B), 0, 1 )
       = 1                     for x in the core (d_B ≤ birth_B)
       ramps 0 → 1             as x attaches between death and birth
```

This is a **direct** MF: no medoid, no Gaussian, no fit — just the block's own
birth/death heights, which selection already has. The persistence gate that
decides *whether* a block is real and the ramp that decides *how graded* its
membership is are the same two numbers (birth, death). So MF generation and
selection can be the **same computation**.

---

## Proposed phases

### Phase 1 — per-band fuzzy partitions (direct ramp MFs)
Add `band_memberships(band, D*) → U_band` (shape `k_band × n`): apply the
Mapping-2 ramp to each *selected* block in the band, using that block's own
birth/death. Output becomes a **fuzzy partition per scale** instead of a hard
label vector. Reuses the existing ramp; no new shape model.
*Deliverable:* `multiscale_persistence` returns `U_band` per band; `assign_band`
becomes `argmax(U_band)` for scoring only.

### Phase 2 — partition-of-unity per scale
Optionally Ruspini-normalize each band so `Σ_k μ_k(x) = 1` (drive
`ruspini_mf`'s normalization from the ramp MFs rather than from medoid Gaussians).
Keep an unnormalized (possibilistic) mode too — a point far from every block in a
band *should* read low everywhere, which is information, not an error.
*Deliverable:* `normalize={"ruspini","none"}`; report partition-of-unity error
(already 0.0 for Option B) and coverage.

### Phase 3 — the multi-scale fuzzy model
Represent the output as a **fuzzy hierarchy**: `{fine: U_f, medium: U_m,
coarse: U_c}`. Two consumers:
- keep the stack (the honest object — memberships at every scale); or
- blend to one flat fuzzy summary via the fixed **t-conorm** already in
  `disjunct._tconorm` (over-segmentation is cheap under the conorm — the whole
  rationale behind `coverage_cover`).

### Phase 4 — soft band membership (fixes the small-n artifact)
Replace the hard log-birth cut with **kernel-weighted band membership**: a block
near a band edge contributes to both adjacent scales with weight decaying in
log-birth distance. This directly targets the `log_separated` over-segmentation
found in `SCALING_STUDY.md` (§4) — a block on a boundary no longer forces a
degenerate single-cluster band. Also add the cheap guard: drop single-block bands
(uninformative 1-cluster partitions).

### Phase 5 — one-pass generation (the end state)
Collapse enumerate → gate → cover → fit into a **single walk of the dendrogram**:
every block whose (scale-local) persistence clears the gate emits its ramp MF;
the t-conorm recombines overlapping/redundant ramps; the surviving envelope *is*
the fuzzy model. Selection becomes implicit in the gate; MFs fall out with no
separate stage. This is "generating the membership functions directly."

### Phase 6 — validation
- **Hard proxy:** ARI(argmax) — must not regress on the battery or the scaling
  families (`many_scale` should stay [8,4,2] @ 1.0).
- **Soft metrics:** on synthetic Gaussian mixtures the *true* soft memberships
  are known → score fuzzy-ARI / cross-entropy of `U` vs ground-truth posteriors,
  not just the argmax. This is the first test that actually exercises the
  *graded* part of the MFs.
- **Partition-of-unity, coverage, convexity** (reuse `battery` metrics).

---

## Open design questions
1. **Death height for a fuzzy block at scale s** — the block's own parent-merge
   height (current), or the band's upper edge (makes ramps comparable within a
   band)? Affects cross-block calibration.
2. **Normalized vs possibilistic** as the default output — partition-of-unity is
   clean but hides "belongs to nothing here"; possibilistic keeps tendency
   awareness (matters for the `uniform_noise` / decline-to-assert story).
3. **Cross-scale blending weights** — equal per scale, or weighted by band
   persistence mass / coverage?
4. **Non-convex blocks** — the ramp is minimax-distance based, so it already
   follows non-convex block geometry (unlike Option C's Mahalanobis surrogate,
   which fails on rings). Confirm this holds for the fuzzy output.

## Sequencing
Phases 1–3 are the direct-MF core and are low-risk (they reuse existing ramp +
normalization + t-conorm). Phase 4 is the research-interesting piece (soft bands)
and also fixes a known failure mode. Phase 5 is the refactor that makes it "one
pass." Phase 6 runs throughout. Scaling refactor (sparse/kNN graph, see
`SCALING_STUDY.md` §5) is orthogonal and only needed for n ≫ 5000.
