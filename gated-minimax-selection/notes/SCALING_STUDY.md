# Scaling Study — Multi-Scale Selection at n = 100 … 5000

**Reproduce:** `python run_all.py --scaling`
→ `outputs/scaling_results.json`, `outputs/fig11_scaling.png`
**Datasets:** `battery_hierarchical.SCALABLE` (`single_scale`, `many_scale`, `log_separated`)

---

## 1. Why this study, and the enabling change

The reference `ivat_mf.minimax_transform` is a pure-Python Prim loop — **O(n³)** —
so it cannot reach the sizes asked for (n=500 already takes ~4.6 s; n≥1000 is
minutes). I added **`ivat_mf.minimax_transform_fast`**: build the MST once
(scipy) and fill all-pairs bottleneck distances by Kruskal-style union-find
(each pair set exactly once → **O(n²)**). It is **numerically identical** to the
reference (validated `max|Δ| = 0.0` for n = 5, 20, 60, 120), and 19–130× faster
on n = 100–500. This is what makes n = 5000 tractable.

Three fixed-structure families, each grown to size n (cluster counts fixed, only
the sample size changes — the right design for "does the same structure survive
as n grows, and how does runtime scale?"):

- **`single_scale`** — 5 well-separated blobs, one spread. Correct answer: **one**
  scale, 5 clusters.
- **`many_scale`** — balanced 2×2×2 nested tree, three genuine scales. Correct
  answer: **three** scales with granularities **[8, 4, 2]**.
- **`log_separated`** — 3 clusters whose spreads differ by orders of magnitude
  (σ = 0.35, 7, 175; >400× ratio), separation scaled to spread so they stay
  linearly separable. This is a **single-level** problem (3 clusters, no
  nesting); correct answer: **one** scale, 3 clusters.

---

## 2. Results

Wall-clock is `t_transform + t_select` (seconds); ARI is the best-matching band
per ground-truth level.

| dataset | n | t_transform | t_select | scales | granularities | ARI/level |
|---|---|---|---|---|---|---|
| single_scale | 100 | 0.002 | 0.001 | 2 | [5, 2] | [1.00] |
| single_scale | 500 | 0.032 | 0.002 | 2 | [5, 1] | [1.00] |
| single_scale | 1000 | 0.135 | 0.007 | 2 | [5, 2] | [1.00] |
| single_scale | 2000 | 0.578 | 0.021 | 2 | [5, 2] | [1.00] |
| single_scale | 5000 | 4.72 | 0.113 | 2 | [5, 2] | [1.00] |
| **many_scale** | 100 | 0.013 | 0.001 | 3 | **[8, 4, 2]** | **[1.0, 1.0, 1.0]** |
| **many_scale** | 500 | 0.026 | 0.002 | 3 | **[8, 4, 2]** | **[1.0, 1.0, 1.0]** |
| **many_scale** | 1000 | 0.113 | 0.005 | 3 | **[8, 4, 2]** | **[1.0, 1.0, 1.0]** |
| **many_scale** | 2000 | 0.515 | 0.014 | 3 | **[8, 4, 2]** | **[1.0, 1.0, 1.0]** |
| **many_scale** | 5000 | 4.37 | 0.095 | 3 | **[8, 4, 2]** | **[1.0, 1.0, 1.0]** |
| log_separated | 100 | 0.013 | 0.001 | 3 | [1, 1, 1] | [0.00] |
| log_separated | 250 | 0.006 | 0.001 | 3 | [1, 2, 1] | [0.57] |
| log_separated | 500 | 0.024 | 0.002 | 3 | [1, 1, 1] | [0.00] |
| log_separated | 1000 | 0.101 | 0.006 | 1 | [3] | [0.99] |
| log_separated | 2000 | 0.529 | 0.019 | 1 | [3] | [0.99] |
| log_separated | 5000 | 4.34 | 0.124 | 1 | [3] | [1.00] |

(Full per-n rows in `scaling_results.json`.)

## 3. What it confirms

**Runtime is clean O(n²)** and dominated entirely by the transform: n×50 (100→5000)
costs ~n²×2400 in time (0.002 s → 4.7 s), tracking the O(n²) guide in
`fig11_scaling.png`. Selection is negligible at every size (≤0.13 s at n=5000).
The dense n×n matrix — not Prim's speed — is the wall past this range (§5).

**`many_scale` is the headline: perfect at every size.** Three scales,
granularities [8, 4, 2], ARI 1.0 at all three levels, unchanged from n=100 to
n=5000. The multi-scale contribution recovers a full nested hierarchy robustly
across a 50× sample-size range without ever being told the number of levels.

**`single_scale` always recovers its 5 clusters** (finest band = 5, ARI 1.0 at
every n) but reports a spurious **extra coarse band** (n_scales = 2). The
coarsest merges always form their own band, which here over-groups the 5 blobs
into ~2 super-groups. Harmless to the correct fine partition, but it means
"n_scales" over-counts on single-scale data.

## 4. The honest finding: `log_separated` is n-sensitive

On log-magnitude-separated **single-level** data, band discovery **over-segments
at small n** (n ≤ 500): each cluster lands in its own scale band as a lone block,
so every band is a degenerate 1-cluster partition (ARI 0). It **self-corrects at
n ≥ 1000**, collapsing to the correct single band of 3 clusters (ARI ≈ 0.99).

Mechanism: birth-height banding cannot, from the dendrogram alone, distinguish
"3 clusters at very different **spreads**" (one level) from "3 clusters at
different **nesting scales**" (three levels). At small n the log-birth gaps
between the widely-separated clusters look exactly like scale-band boundaries; at
larger n each cluster grows internal significant sub-blocks that fill the
log-birth axis, the gaps close, and the three clusters merge into one band. This
is the documented "band discovery is a gap heuristic assuming scale-*separated*
levels" limitation, now pinned down empirically with an n-threshold.

Two concrete fixes (deferred — see `MEMBERSHIP_ROADMAP.md` §soft bands):
1. **Drop single-block bands.** A band with one block is an uninformative
   1-cluster partition; dropping it (unless it is the only band) would remove the
   log_separated small-n artifact and the single_scale spurious coarse band.
2. **Soft band membership.** Kernel-weight each block's contribution across
   adjacent bands by log-birth distance, replacing the hard cut that
   over-reacts to sampling density.

## 5. Scaling wall (for the record)

`select_multiscale`, `_all_blocks`, and `assign` all consume a **dense n×n
`Dstar`**: 200 MB at n=5000, ~800 MB at 10k, ~3.2 GB at 20k. That memory (and the
O(n²) fill) — not the MST/Prim step — is the limit past this range. Going
substantially larger means refactoring the downstream onto a **sparse / kNN
graph** so the dense matrix never materializes; a fast sparse-graph Prim's/MST
(e.g. the one in `github.com/fundthmcalculus/clustering`) would slot in at that
point. Not needed for the n ≤ 5000 targets here.
