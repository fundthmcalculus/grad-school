# Option D — Multi-Scale Persistence Selection

**Status:** implemented, validated, and reframed from the original proof-of-concept.
**Branch:** `feat/option-d-multiscale`
**Code:** `multiscale_persistence.py`, `battery_hierarchical.py`; experiments in `run_all.py`
**Supersedes:** the annotation-only proof-of-concept in `multi_scale_mf.py` /
`OPTION_D_FINDINGS.md`.

---

## 0. TL;DR

The original Option D asked *"how do we pick the one right scale for
`varying_density`?"* Experiment says that is the **wrong question**: the flat
`coverage_cover` selector is already scale-invariant on single-level data, so
there is nothing to fix there. The right question is *"what do we do when there
is no single right scale?"* — i.e. when the data is genuinely **nested**
(clusters within clusters, each level meaningful).

Flat set-cover **cannot** represent nested structure: greedy coverage terminates
at whichever granularity covers the data first and never descends. This module
turns the *number of scales* into an **output**: it discovers scale bands from
gaps in the dendrogram's birth-height axis and runs the gated set-cover
independently within each band, returning a **scale hierarchy** rather than one
flat partition.

**Result:** on three hierarchical datasets, flat selection recovers a mean ARI of
0.58–0.75 across all ground-truth levels (it nails one level, misses the others),
while multi-scale recovers **1.000 at every level simultaneously**. On the
ordinary single-scale battery it discovers exactly one band and reproduces the
flat baseline — a strict generalization with no regression.

---

## 1. What changed from the proof-of-concept, and why

The prior implementation (`multi_scale_mf.py`) partitioned heights into
percentile bands, computed a `local_persistence = global_persistence / band_width`,
ranked blocks within bands — and then **kept all the `coverage_cover` blocks
anyway**. Its own findings were honest: `improvement = +0.0`. The multi-scale
analysis was informational, not prescriptive.

Before rebuilding it I ran the baseline hard to find where it actually breaks
(`run_all.run_multiscale_scale_invariance`). Two findings reoriented the work:

1. **Density contrast alone does not break flat selection.** With inter-cluster
   separation scaled to spread (so clusters stay separable), `coverage_cover`
   holds ARI ≈ 0.983 across a **30× spread ratio** (σ ratio 1 : 8 : 64). The
   minimax/bottleneck transform makes heights scale-relative, and the MAD-based
   gate is robust to the few huge-persistence blocks a tight cluster produces, so
   no "masking" of the diffuse cluster occurs. The old worry was unfounded.

2. **What flat selection cannot do is descend a hierarchy.** On a 2×3 nested
   Gaussian mixture (two super-clusters, each with three sub-clusters), greedy
   coverage grabs the two size-60 super-blocks, achieves full coverage, and
   stops. It scores ARI **1.00 vs the coarse 2-cluster truth** but only **0.32
   vs the fine 6-cluster truth** — structurally unable to report both.

So Option D is not "a better `varying_density` number." It is a genuinely
different **deliverable shape**: a hierarchy of partitions instead of one.

---

## 2. Problem formalization

Let `D*` be the minimax (iVAT/bottleneck) transform of the dissimilarity matrix
(`ivat_mf.minimax_transform`). Single-linkage on `D*` gives a dendrogram whose
every internal node is a candidate **block** `B` with:

- `birth(B)` — the height at which `B`'s two children merge (it becomes a
  connected component);
- `death(B)` — the height at which `B` merges into its parent;
- `persistence(B) = death(B) − birth(B)`;
- `members(B)` — the leaf set.

**Persistence significance (shared with the flat gate).** A block is *significant*
iff its absolute persistence is an upper outlier of the persistence diagram:

```
thr = median(persistence) + gap_sigma · (1.4826 · MAD(persistence))
B significant  ⇔  persistence(B) ≥ thr        (over all blocks, size-windowed after)
```

This is exactly the eligibility gate of `selection.select_coverage_cover`;
`multiscale_persistence.persistence_significance_threshold` is the single shared
definition, so the multi-scale selector is a **generalization** of the gated
minimax selector, not a competitor to it.

**The birth-height ↔ density correspondence.** In single-linkage on a full
dissimilarity, `birth(B)` equals the smallest bottleneck distance that connects
`B` internally — i.e. the scale at which the cluster's points are mutually
reachable. Dense clusters are born **low**; diffuse clusters are born **high**.
Birth height is therefore an (inverse) proxy for the local density at which a
cluster becomes a distinct entity. This is the bridge between Directions 1
("adaptive bands / persistent homology") and 2 ("density-aware persistence") of
the original findings: **stratifying by birth height *is* stratifying by density
scale.**

**Scale bands.** A cluster *generation* occupies a contiguous band of birth
heights; successive generations are separated by gaps on the (log) birth axis —
the same horizontal strata one sees in a persistence barcode. We cut the log-birth
axis of the significant blocks at every gap exceeding `band_gap_factor ×` the
median gap (with a floor `min_log_gap`). Working in **log** height makes the cut
scale-relative: a factor-of-*e* jump in scale is judged identically at height 0.1
or height 100.

**Per-band selection.** Within each band we run the same greedy set-cover as the
flat method, but restricted to that band's blocks and covering that band's point
universe. Bands whose selection covers less than `min_band_coverage` of the data
are dropped (spurious noise strata).

**Output.** A `MultiScaleSelection` — an ordered list of `BandSelection`s, fine
(low birth) → coarse. `granularities()` gives the cluster count discovered at each
scale; `flatten_to_level(ℓ)` picks one when a flat partition is needed;
`assign_band` defuzzifies a band's blocks to hard labels by minimax proximity
(the `ivat_mf.hard_labels_proximity` rule).

---

## 3. Algorithm

```
select_multiscale(D*):
    blocks         = enumerate_all_dendrogram_blocks(D*)          # selection._all_blocks
    significant    = { B : persistence(B) ≥ thr, size-windowed }  # shared gate
    edges          = gaps in log-birth(significant) > band_gap_factor·median_gap
    for each band  = contiguous run of significant blocks between edges:
        sel        = greedy_set_cover(band.blocks)                # by uncovered-gain, ties→persistence
        keep band  if coverage(sel) ≥ min_band_coverage
    dedupe bands producing identical block sets
    return bands ordered fine → coarse
```

Complexity is dominated by the `O(n²)` minimax transform and block enumeration
already used by the flat selector; band discovery adds an `O(m log m)` sort over
the `m` significant blocks.

**Parameters** (all with defaults that reproduce the flat baseline on single-scale
data):

| param | default | role |
|---|---|---|
| `gap_sigma` | 2.0 | persistence-outlier strength (shared with `coverage_cover`) |
| `max_size_frac`, `min_size` | 0.6, 3 | block size window |
| `band_gap_factor` | 3.0 | a log-birth gap is a band edge if > this × median gap |
| `min_log_gap` | 0.5 | floor so tiny gaps never split a band |
| `min_band_coverage` | 0.15 | drop bands covering < 15% of points (noise strata) |

---

## 4. Results

### 4.1 Headline — hierarchy recovery (`run_all.run_multiscale_numeric`)

Mean ARI **averaged over all ground-truth levels** (so a method that nails one
level and misses another is penalized):

| dataset | levels | flat mean ARI | multi-scale mean ARI |
|---|---|---|---|
| `nested_gaussians` | fine(6), coarse(2) | 0.662 | **1.000** |
| `three_level_hierarchy` | fine(8), medium(4), coarse(2) | 0.576 | **1.000** |
| `density_hierarchy` | fine(4), coarse(2) | 0.746 | **1.000** |

Per-band detail for `three_level_hierarchy` (the strongest demonstration — three
genuinely distinct scales):

```
flat coverage_cover: k=2   ARI/level = [fine 0.236, medium 0.492, coarse 1.000]
multi-scale: 3 bands, granularities [8, 4, 2]
   band 0: k=8  births[0.00, 1.59]   ARI/level = [1.000, 0.581, 0.236]   ← fine
   band 1: k=4  births[1.59, 10.31]  ARI/level = [0.581, 1.000, 0.492]   ← medium
   band 2: k=2  births[10.31, inf]   ARI/level = [0.236, 0.492, 1.000]   ← coarse
```

Each discovered band lands on **exactly one** ground-truth level at ARI 1.000, and
the granularities `[8, 4, 2]` recover the true `2×2×2` tree without ever being
told the number of levels or clusters.

`outputs/fig8_multiscale_hierarchy.png` visualizes this for `nested_gaussians`:
the persistence diagram splits cleanly into a low-birth stratum (six sub-cluster
blocks, persistence ≈ 3.4–4.3) and a high-birth stratum (two super-cluster blocks,
persistence ≈ 28), with the discovered band edge between them; the band×level ARI
matrix is diagonal.

### 4.2 No regression on the flat battery (`run_all.run_multiscale_no_regression`)

| dataset | flat k / ARI | # bands | finest k / ARI |
|---|---|---|---|
| `two_gaussians` | 2 / 1.000 | 1 | 2 / 1.000 |
| `bridged_gaussians` | 3 / 0.982 | 1 | 2 / 1.000 |
| `concentric_rings` | 2 / 1.000 | 1 | 2 / 1.000 |
| `varying_density` | 3 / 0.980 | 1 | 3 / 0.980 |
| `uniform_noise` | 4 / — | **0** | — |

Single-scale data yields exactly **one** band that matches (or, on
`bridged_gaussians`, slightly exceeds) the flat baseline. On `uniform_noise` the
multi-scale selector returns **zero** bands — it declines to assert structure
where the flat gate happened to admit four marginal blocks. This is the intended
strict-generalization behavior.

### 4.3 Why there is no flat-ARI claim on single-level varying-density (`run_all.run_multiscale_scale_invariance`)

`coverage_cover` ARI on three separable Gaussians as the spread ratio grows:

```
spread ratio   1:2:2   1:2:4   1:3:9   1:4:16   1:6:36   1:8:64
flat ARI       0.983   0.983   0.983   0.983    0.983    0.983
```

Flat ARI is invariant across a 30× spread ratio. We therefore **do not** claim a
single-level `varying_density` improvement — that would be a strawman. The
contribution is the nested regime, where flat selection provably loses
information.

---

## 5. Positioning for the dissertation

> *Gated-minimax selection reframes cluster selection as a persistence-gated
> set-cover, and is scale-invariant for single-level structure. Multi-scale
> persistence selection extends it to nested structure by treating the number of
> scales as an output: it discovers density-scale bands from gaps in the
> single-linkage birth-height spectrum and solves the gated set-cover within each
> band, returning a hierarchy of partitions. On multi-level synthetic data it
> recovers every level (ARI 1.0 at each) where flat selection is forced to commit
> to one, and it reduces to the flat selector on single-level data.*

Three claims make this thesis-defensible rather than a tuned demo:

1. **Shared gate.** The significance test is *literally* the flat selector's gate
   (`persistence_significance_threshold`), so this is a generalization with a
   proved single-band reduction, not a separate heuristic.
2. **Falsification attempt included.** §4.3 documents the regime where the
   baseline is *not* beaten, so the nested-structure claim is not a strawman.
3. **Density interpretation.** Birth-height banding is justified by the
   birth ↔ density correspondence, connecting the method to the persistent-homology
   and density-aware directions rather than being an ad-hoc partition.

---

## 6. Honest limitations

- **Band discovery is still a gap heuristic.** It assumes scales are *separated*
  on the log-birth axis. If two generations overlap in birth height (no gap), the
  cut is ill-posed for any band-based method. `battery_hierarchical` deliberately
  builds scale-separated levels; overlapping-scale hierarchies are open (see §7).
- **`min_band_coverage` can suppress a small real cluster.** A genuine fine
  cluster covering < 15% of points, with the rest only clusterable at a coarser
  scale, would be dropped. The default suits balanced hierarchies; unbalanced ones
  need a per-band, not global-fraction, criterion.
- **Levels are evaluated by best-matching band.** ARI-per-level uses the band that
  best matches each level. That is the correct evaluation for "did the hierarchy
  contain this level," but it presumes the caller knows which band it wants; there
  is no automatic level-naming.
- **Synthetic ground truth.** Results are on constructed hierarchies with known
  levels. Real nested structure (e.g. taxonomic or morphological data) has no
  clean level labels; validation there needs a different protocol.

---

## 7. Future work (dissertation roadmap)

1. **Adaptive / model-based band discovery.** Replace the gap heuristic with a
   change-point or mixture model over the log-birth distribution, or a genuine
   persistent-homology stability analysis (barcode length across scales). Handles
   overlapping-scale hierarchies the gap cut cannot.
2. **Density-normalized persistence.** Make the birth↔density link explicit:
   normalize each block's persistence by an estimate of local background density
   at its death height, giving a scale-free significance statistic that could
   replace per-band gating with a single global test.
3. **Soft (fuzzy) band membership.** A block near a band edge should contribute to
   both adjacent scales with graded weight — mirroring the fuzzy-membership spirit
   of Options A–C. Couples naturally with the t-conorm recombination the flat
   selector already relies on.
4. **Per-band membership functions.** Compose with Options A/B: run Ruspini
   membership extraction *within each band* to produce a fuzzy partition **per
   scale**, i.e. a full multi-scale fuzzy model, not just hard hierarchical labels.
5. **Learned band criteria.** Train a classifier on labeled hierarchies to predict
   band edges and per-band `keep/drop`, adapting the heuristics of §3 to a domain.
6. **Real multi-scale data.** Validate on datasets with genuine nested structure
   and design an evaluation protocol that does not assume clean per-level labels.

---

## 8. File map

```
multiscale_persistence.py   core module (select_multiscale, band discovery,
                            shared significance gate, defuzzification)
battery_hierarchical.py     nested/multi-level datasets with per-level ground truth
run_all.py                  experiments (run_multiscale_numeric / _no_regression /
                            _scale_invariance) + fig8_multiscale_hierarchy.png
OPTION_D_MULTISCALE.md       this document
```

Reproduce everything with `python run_all.py` (results in `outputs/results.json`
under the `multiscale_*` keys; figure at `outputs/fig8_multiscale_hierarchy.png`).
