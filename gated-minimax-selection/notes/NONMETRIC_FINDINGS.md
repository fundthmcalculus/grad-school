# Non-Euclidean / Non-Metric Dissimilarities: What the Minimax Pipeline Actually Guarantees

**Date**: 2026-08-24
**Status**: Complete first pass — four experiments, all reproducible via `python run_nonmetric.py`
**Code**: `nonmetric_data.py` (generators + diagnostics), `run_nonmetric.py` (driver), `test_nonmetric.py` (invariants)
**Outputs**: `outputs/nonmetric_results.json`, `outputs/fig12..fig15_*.png`

---

## Executive Summary

Four findings, in decreasing order of importance:

1. **The minimax transform is a canonical Euclideanizer.** D* = minimax(D) is an
   ultrametric for *any* symmetric nonnegative dissimilarity — metric or not —
   and finite ultrametrics embed isometrically in Euclidean space. Empirically:
   every family tested (DTW, edit, Hamming, graph, cosine, and matrices with up
   to 82% of pairs corrupted) has raw-D negative-eigenvalue ratios of 0.02–0.90
   and D* ratios at machine zero (≤1e-16), with ultrametricity checked exactly.
   **D* statically subsumes NERFCM's beta-spread repair**: beta(D*) = 0 always,
   by construction rather than by luck.

2. **Stretch and shortcut violations hurt different methods — and the split is
   structural, not empirical happenstance.** Inflating distances (stretch: what
   DTW warping and cosine saturation do) leaves every minimax method untouched
   (ARI ≥ 0.98 at up to 66% of pairs corrupted) because bottleneck paths route
   around large edges. Deflating distances (shortcut: what sparse inter-community
   graph edges and bridge points do) is exactly single-linkage chaining: one deep
   corrupted pair fuses two blocks, collapsing D*-based methods (ARI 1.0 → 0.15)
   while relational averaging (NERFCM on raw D) holds at 1.00 through *every*
   tested corruption level, because a fuzzy mean integrates over all n distances
   per point and random deflations cancel. Min/max aggregation is worst-case
   sensitive; mean aggregation is average-case robust. The same property that
   makes D* topology-adaptive (elongated/nested clusters) makes it bridge-fragile.

3. **Beta-spread activation tracks violation *depth*, not violation fraction or
   the Gram spectrum.** Every synthetic non-Euclidean family here is formally
   inadmissible (negative Gram eigenvalues up to 0.35), yet NERFCM's relational
   update never goes negative on any of them, at any (c, m) probed — resolving
   the open question `verify_beta_nonmetric.py` left ("the test matrices may not
   have been sufficiently non-Euclidean"). What fires it is *deep* violations:
   synthetic stretch caps the depth at ~1.0 (a pair at most 2x its two-hop
   bound) and never activates; **real flight-profile DTW reaches depth 1.98 and
   activates on every restart seed at every c** (beta 0.04–0.22); the spiked
   control (depth >> 1) fires at beta = 11.6. The activation boundary sits
   between depth ~1 and ~2 — and in every case, beta(D*) = 0.

4. **The standing `multi_scale_hierarchy` "open problem" (NERFCM ARI 0.29 in
   RELATIONDATA.md) is solved — and was partly an artifact.** Its declared
   labels are ~18% noise: the generator's leaf-expansion loop assigns
   `rng.integers(0, 4)` labels regardless of where a leaf attaches
   (`relationdata.py:271`). Scored against the *structural* labels the distances
   actually encode, multi-scale selection (Option D) recovers the dataset
   perfectly at both levels (ARI 1.0 / 1.0, bands [6, 3]); the flat cover gets
   coarse right (1.0) and fine wrong (0.55) because it must commit to one
   granularity. The 0.29 decomposes as: granularity mismatch (c=3 scored against
   6 fine labels) + label noise. Multi-scale also recovers the new clean
   two-level relational hierarchy at 1.0/1.0 and **keeps doing so under both
   stretch (r=.2, s=.8) and shortcut (r=.05, s=.8) corruption**.

---

## E1 — Diagnostics: making "non-Euclidean" quantitative

Two instruments, both in `nonmetric_data.py`:

- `triangle_violation_stats(D)`: fraction of pairs strictly above their
  tightest two-hop bound (metricity).
- `euclidean_embeddability(D)`: most negative eigenvalue of the classical-MDS
  Gram matrix, relative to the largest (embeddability). This is the exact
  quantity the beta-spread exists to repair.

| dataset | TI-violated pairs | neg-eig(D) | neg-eig(D*) | ultrametric(D*) | beta(D) | beta(D*) |
|---|---|---|---|---|---|---|
| dtw_traces | 0.115 | 0.045 | 0 | yes | 0 | 0 |
| edit_strings | 0 | 0.038 | 0 | yes | 0 | 0 |
| hamming_categorical | 0 | 0.023 | 0 | yes | 0 | 0 |
| graph_communities | 0 | 0.121 | 0 | yes | 0 | 0 |
| cosine_topics | 0.012 | 0.073 | 0 | yes | 0 | 0 |
| blobs_stretch (r=.2 s=.8) | 0.185 | 0.253 | 0 | yes | 0 | 0 |
| blobs_shortcut (r=.2 s=.8) | 0.816 | 0.184 | 0 | yes | 0 | 0 |
| **spiked_random (control)** | 0.005 | **0.897** | 0 | yes | **11.6** | 0 |

Notes:
- "Metric" and "Euclidean-embeddable" separate cleanly: edit/Hamming/graph
  violate zero triangles yet all carry negative Gram eigenvalues.
- The spiked control row is what makes the zeros elsewhere *measurements*: the
  harness demonstrably detects activation when it occurs.
- Beta probed across c ∈ {2,3,4} × 5 restart seeds per cell (and, offline,
  m ∈ {1.5, 2, 3} and c up to 8 on the worst stretched matrix — still 0).

**Theory note** (for the chapter): D*_ij ≤ max(D*_ik, D*_kj) holds for any
symmetric input because concatenating the bottleneck paths i→k→j gives a path
whose bottleneck is the max of the two; hence D* is an ultrametric. Finite
ultrametric spaces embed isometrically into Euclidean space (Lemin's theorem;
equivalently, ultrametrics are of negative type), so the double-centered Gram
matrix of D* is PSD and the RFCM relational update can never produce a negative
distance. `test_nonmetric.py` pins the ultrametricity + admissibility +
idempotence (minimax(D*) = D*) invariants across all families, including
corrupted ones.

## E2 — Method battery on five non-Euclidean families

ARI vs planted clusters (NERFCM: mean over 5 restart seeds; selection methods
discover k themselves):

| dataset | NERFCM(D) | NERFCM(D*) | SL@k | gap-cover | beta-plateau | bootstrap(rel.) |
|---|---|---|---|---|---|---|
| dtw_traces | 1.00 | 1.00 | 1.00 | 1.00 (k=3) | 1.00 (k=3) | 1.00 (k=3) |
| edit_strings | 1.00 | 1.00 | 1.00 | 1.00 (k=3) | 1.00 (k=3) | 1.00 (k=3) |
| hamming_categorical | 1.00 | 1.00 | 1.00 | 1.00 (k=3) | 1.00 (k=3) | 1.00 (k=3) |
| graph_communities | **0.61** | 0.39 | 0.00 | 0.33 (k=4) | 0.26 (k=4) | 0.26 (k=4) |
| cosine_topics | 0.43 | 0.56 | 0.00 | **0.80 (k=3)** | 0.56 (k=2) | **0.80 (k=3)** |

- The whole selection stack **extends unchanged to relational non-Euclidean
  data** on the three well-structured families, discovering the correct k.
  (`select_bottleneck_bootstrap` needed a relational mirror — resample matrix
  indices instead of points; added as
  `run_nonmetric.select_bottleneck_bootstrap_relational`.)
- `cosine_topics` is the pro-selection case: plain SL at the true k scores 0.00
  (outlier chaining), yet the persistence gate rescues it to 0.80 — better than
  either NERFCM arm. The gate, not the hierarchy cut, is doing the work.
- `graph_communities` is the honest limit case and it is *predicted* by finding
  2: sparse inter-community edges are natural shortcuts, so every D*-based
  method chains (SL 0.00; NERFCM(D*) 0.39 < NERFCM(D) 0.61). Note the
  probe-driven parameter choice documented in the generator: denser graphs
  concentrate shortest paths until *nothing* works, which is less informative.

## E3 — Controlled violation sweep (fig14)

Base: three 2-D Gaussian blobs (sep = 6.5σ, calibrated so the clean baseline is
1.0 for every method across replicate seeds). Corruption per
`violate_pairs(mode, rate, strength)`; cells average 3 dataset seeds × 5 NERFCM
restarts.

- **Stretch**: flat. NERFCM(D) = 1.00 and NERFCM(D*)/gap-cover ≥ 0.98
  everywhere up to rate 0.4 @ strength 0.8 and rate 0.2 @ strength 1.0; cover
  dips to 0.85 only at rate 0.8 (two-thirds of all pairs corrupted). Formal
  inadmissibility (neg-eig up to 0.33) never matters operationally.
- **Shortcut, by strength (r=0.2)**: threshold onset, geometrically where
  bridge formation first becomes possible. Deflation must push a cross-cluster
  pair (≈4.6σ after blob spread) below the intra-cluster merge scale (≈1.2σ),
  i.e. factor < ~0.26, so with factor = 1 − s·u, bridges require s > ~0.7:
  observed 1.00/1.00/1.00/0.98/**0.72/0.15** at s = 0/…/0.8/1.0. NERFCM(D):
  1.00 at every strength.
- **Shortcut, by rate (s=0.8)**: non-monotone, and the non-monotonicity is
  informative — damage is worst when shortcuts are *sparse* (isolated bridges,
  ARI ≈ 0.65–0.72 at rates 0.05–0.4) and partially heals at rate 0.8
  (ARI 0.97) because near-uniform deflation rescales the geometry instead of
  rewiring it.
- NERFCM(D) never leaves 1.00 anywhere in the sweep, including rate 0.8 at
  strength 1.0 (probed separately). Random corruption is the averaging regime.

**Scope limit this hands the thesis**: choose the representation by the noise
model. If dissimilarity errors are inflationary (alignment slack, saturation),
the minimax pipeline is safe and adds k-discovery + topology-adaptivity. If
errors can be deflationary (false matches, hub edges, data-entry shortcuts),
raw-D relational averaging is the robust choice — or bridge-pruning (ConiVAT)
must precede the transform. This is the same bridge sensitivity documented in
SELECTION_METHODS_COMPARISON.md, now with a controlled dose-response curve and
a mechanism.

## E4 — Relational multi-scale (fig15)

| case | truth | flat cover | multi-scale (bands) |
|---|---|---|---|
| multi_scale_hierarchy (declared labels) | fine6 / coarse3 | 0.29 / 0.55 | 0.61 / 0.55 |
| multi_scale_hierarchy (**structural labels**) | fine / coarse | 0.55 / 1.00 | **1.00 / 1.00** ([6, 3]) |
| relational_nested (clean) | fine6 / coarse3 | 0.54 / 1.00 | **1.00 / 1.00** ([6, 3]) |
| relational_nested (stretch r=.2 s=.8) | fine6 / coarse3 | 0.54 / 1.00 | **1.00 / 1.00** ([6, 3]) |
| relational_nested (shortcut r=.05 s=.8) | fine6 / coarse3 | 0.54 / 1.00 | **1.00 / 1.00** ([6, 3]) |

- Option D's headline result (flat commits to one granularity; band-wise
  selection recovers every level) **transfers to distance-matrix-only data**,
  and survives both corruption modes at the tested doses.
- The `multi_scale_hierarchy` label-noise bug (structural components 4 and 5
  contain leaves declared as labels {1,2,3} and {0} respectively — random
  assignments from the expansion loop, 7/39 points ≈ 18%) should be fixed or
  the dataset retired in favor of `relational_nested_hierarchy`, which has
  exact two-level truth by construction. Tracked as a repo issue; not fixed
  here because `run_all.py`'s relational table/figures currently depend on the
  generator's exact output. **[FIXED 2026-08-25, issue #160]** — declared
  labels now match structure exactly (pinned in `test_relationdata.py`),
  `run_all.py`'s relational row moved 0.285 → 0.551 (the c=3-vs-6-labels
  ceiling; the only numbers in results.json that changed), and E4's
  declared-vs-structural split now coincides: multi-scale recovers the dataset
  at 1.0/1.0 against its own declared labels.

## E5 — Real-data DTW: N-CMAPSS DS01 flight altitude profiles

75 flights (25 per flight class) from DS01-005 dev (DS02's dev units are all
class 3, so DS01 is the multi-class set), altitude subsampled at a fixed
~5-minute rate — NOT length-normalized, so sequence length carries flight
duration, which is what the class labels bin. DTW dissimilarity.

Diagnostics: **69.6% of pairs violate the triangle inequality** with max depth
1.98 (real DTW is far more non-metric than any synthetic family), neg-eig 0.29,
and **beta-spread activates on every restart at every c** (0.04–0.22) — the
only naturally occurring activation in the whole study. D* is, as everywhere,
ultrametric and admissible.

Clustering (the class bins genuinely overlap — class 2 spans 1.3–3.3h against
class 1's 0.9–1.6h and class 3's 2.6–5.2h — so the reference ceiling is
**duration-only 3-means at ARI 0.60**, not 1.0):

| method | k | ARI |
|---|---|---|
| duration-only k-means (ceiling) | 3 (given) | 0.60 |
| NERFCM(D) | 3 (given) | 0.45 |
| NERFCM(D*) | 3 (given) | 0.51 |
| SL @ k=3 | 3 (given) | 0.45 |
| gap-cover / beta-plateau / bootstrap | 2 (discovered) | 0.46 |

Every method lands within ~0.15 of the duration ceiling; the k-discovering
selectors all choose k=2, which is defensible — the class-2/class-3 duration
overlap makes the 3-way split weakly supported in this sample. The value of E5
is less the clustering score than the diagnostics: it is the one dataset where
the beta-spread mechanism is *actually needed* on raw D, and D* still
eliminates it.

## E7 — Real fault-mode clustering (run_fault_modes.py)

Closes the gap E5 named: fault modes ARE cluster-shaped truth. 30 units pooled
from DS01 (HPT efficiency), DS04 (fan), DS07 (LPT); per-unit degradation
trajectories = condition-corrected sensor residuals (per-unit ridge on the
operating conditions over the first 10 healthy cycles — without this step the
trajectories are flight regime, not degradation: raw-trajectory ARI ≈ 0.03,
the same lesson as the CMAPSS cluster-tendency memo), 4 mode-informative
channels (T48, T50, P15, P21), median-smoothed, final 40% of life,
multivariate DTW.

| comparison | ARI |
|---|---|
| NERFCM(D), 3 fault modes | **0.47** |
| NERFCM(D), fan vs turbine (2-way) | **0.74** |
| endpoint-fingerprint reference (no trajectory), 3-way | 0.26 |
| NERFCM(D*), 3-way | 0.12 |
| gap-cover | abstains (k=1, cov 0.10) |

- **Trajectory shape carries real fault information**: DTW at 0.47 beats the
  static endpoint fingerprint at 0.26, and the physically meaningful
  fan-vs-turbine split reaches 0.74 from shape alone (fig19: turbine faults
  raise turbine temps with flat fan pressures; the fan fault does the
  opposite).
- **The honest limit**: HPT vs LPT are near-indistinguishable in this
  representation (their residual signatures are visually identical), which is
  where the 3-way score goes to die. Separating them likely needs
  station-ratio features, not more clustering.
- The method taxonomy holds on real data once again: NERFCM on raw D is the
  right tool for this (deeply non-metric, 7.6% TI-violated) dissimilarity;
  D*-based methods chain or abstain, as E5 and the real-DTW battery predicted.

## What was NOT established (honest gaps)

- ~~E5 uses flight class as truth, which is a duration *bin*, not a cluster; a
  real dataset with genuinely cluster-shaped truth under DTW (e.g. fault-mode
  families of degradation trajectories) would be a stronger test.~~ **Done
  (E7, `run_fault_modes.py`)**: 30 N-CMAPSS units pooled across DS01/DS04/DS07,
  each dataset planting a distinct failure mode, clustered on multivariate DTW
  over condition-corrected degradation trajectories — NERFCM(D) reaches ARI
  0.47 three-way and 0.74 fan-vs-turbine, against 0.26 for a no-trajectory
  endpoint fingerprint. Residual gap: the HPT and LPT residual signatures are
  near-identical, so the three-way score is bounded by that pair, not by the
  clustering; separating them likely needs station-ratio features.
- The battery's three clean families are *easy* for everything; they establish
  transfer, not superiority. The two hard families split the methods — but a
  wider hard set (hub-dominated kNN graphs, heavy-tailed noise) would map the
  boundary better.
- ~~Shortcut fragility was characterized, not repaired.~~ **Done** — see
  `notes/BRIDGE_REPAIR.md`: one-sided reverse-TI metric repair
  (`metric_repair.py`) restores the collapse (0.11 → 0.98) and the broken
  multi-scale cells (→ 1.0/1.0), is provably identity on metric inputs, and
  its scope boundary (inert on geometric bridges, out-of-scope on densely
  non-metric data like real DTW) is measured and pinned in tests. ConiVAT
  remains the route for *geometric* bridges.
- ~~`bottleneck_bootstrap_relational` resamples with replacement like the
  coordinate original... A jackknife variant may behave differently;
  untested.~~ **Tested (E6, `run_bootstrap_variants`)**: the two schemes make
  the SAME final selection on 6/7 cases (the full-matrix threshold step
  absorbs the vote differences), but the jackknife (`replace=False`) turns
  the gap-consensus frequency into a one-sided confidence signal — ≥ 0.47 on
  every clean structured family (dtw 0.21 → 0.75) and ≤ 0.27 on every
  hard/ambiguous case, where with-replacement reads an uninformative
  0.15–0.34 everywhere. Duplicate rows (distance 0) inject spurious
  tiny-persistence blocks into every boot, scattering the vote; true
  submatrices don't. High jackknife frequency ⇒ trustworthy discovery; low
  frequency is inconclusive, never a false alarm in the tested battery.

## Reproduction

```bash
cd gated-minimax-selection
python run_nonmetric.py        # ~2 min; writes JSON before figures
python -m pytest test_nonmetric.py -q
```

Seeds: NERFCM restarts [0..4] (matching run_all.py), sweep dataset seeds
[0..2], generator seeds fixed per generator (see `nonmetric_data.py`).
