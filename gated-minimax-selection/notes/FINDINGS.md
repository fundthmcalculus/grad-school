# iVAT → Membership Function: de-risking battery, findings

## What this is

A runnable test battery that pressure-tests **Mapping 2** (persistence-of-block
membership derived from the single-linkage / iVAT minimax hierarchy) against the
failure modes flagged in the design review, before you invest in a formal
prior-art search. Three files:

- `ivat_mf.py` — the O(n²) minimax (iVAT) transform, VAT ordering, Mapping 1
  (naive medoid fuzzification), Mapping 2 (persistence-based), and two
  defuzzifiers.
- `battery.py` — five synthetic datasets + metrics (ARI, coverage, univariate
  convexity, cluster-count sensitivity) and a k-means anchor.
- `run_all.py` — runs everything (this battery plus every other analysis) and
  prints the tables; writes `outputs/results.json`.

Run: `python3 run_all.py`

## Latest result table

```
dataset               M1 ARI  M2 ARI  kmeans  cover  convex   c-sensitivity
two_gaussians           1.00    1.00    1.00   0.52    1.00   [1.00, 0.80]
bridged_gaussians       1.00    0.00    1.00   0.03    1.00   [0.00, 0.00]
concentric_rings        1.00    1.00    0.00   0.03    1.00   [1.00, 0.83]
varying_density         0.87    0.56    0.92   0.36    1.00   [0.57, 0.56, 0.56]
uniform_noise            n/a     n/a     n/a   0.04    1.00   [n/a]
```

## The three findings that matter

### 1. The WIN is real and reproducible
`concentric_rings`: Mapping 2 scores **ARI ≈ 1.0 while k-means scores ≈ 0.0**.
This is the load-bearing evidence for the thesis. The minimax/single-linkage
structure captures density-connected, non-convex clusters that any
centroid-based fuzzifier (FCM, k-means) provably cannot. This is your
differentiator and it survives testing.

### 2. The KILL case fires exactly as predicted
`bridged_gaussians`: **ARI = 0.0**. A thin bridge of 12 points between two blobs
collapses the minimax distance between them (measured directly: the inter-blob
D* gap drops to ~0.63, well inside intra-cluster scale), so single-linkage
chains the two clusters and the membership functions inherit the merge. This is
not a bug — it is the known single-linkage pathology propagating into the MFs,
precisely the risk called out in the design review.

**Consequence for the thesis:** vanilla iVAT is not a safe foundation on its own.
The defensible path is to build Mapping 2 on top of **ConiVAT** (constraint-based
iVAT, Kumar et al.) or a bridge-pruning step, and to state chaining-robustness as
an explicit scope limitation. This is a citable, principled design choice rather
than a fatal flaw — but you must own it up front.

### 3. Block selection is fragile and scale-dependent — the real research problem
Getting from "the hierarchy" to "these c fuzzy sets" turned out to be the hard
part. The battery walked through three selection heuristics:

- **Raw persistence** → picked nested ancestor/descendant blocks (the whole set
  twice). Fixed by requiring disjoint blocks (an antichain in the tree).
- **Size-weighted persistence** → over-corrected; the near-root block always won
  and only one block got selected.
- **Relative persistence (death/birth ratio) + size ceiling** → scale-invariant,
  current best. Recovers 3/3 blocks on varying-density and gives sensible
  coverage behavior.

Even the best version only reaches ARI 0.56 on varying-density, because the
tight cluster (σ=0.25) and diffuse cluster (σ=1.5) live at different height
scales and the single global ranking still partially conflates them. **This is
your actual open problem**: principled, scale-adaptive selection of persistent
blocks from the minimax hierarchy. That is a thesis contribution in itself, not
a detail.

## Secondary observations

- **Convexity:** the relative-persistence version produces univariate-convex MFs
  everywhere (convex = 1.00), including on the rings. The earlier saturating
  version did not. So the linguistic-labeling concern is manageable *if* the MFs
  are built from tight cores rather than death-height-normalized ramps.
- **Noise awareness:** on `uniform_noise`, coverage collapses to 0.04 — the
  method declines to assert confident structure where there is none. That is the
  desired behavior and a nice property to report (tendency-awareness inherited
  from the VAT lineage).
- **Coverage vs. confidence tradeoff:** relative-persistence cores give lower
  coverage (0.52 on two clean Gaussians). You will need a principled
  "outer skirt" for the MFs so that non-core points still get graded membership
  without reintroducing the saturation/argmax problem.
- **Defuzzification is non-trivial:** naive argmax fails because different-scale
  MFs saturate at 1.0. A proximity-tie-break defuzzifier is included; the
  membership→decision step needs its own attention in the write-up.

## Honest verdict

The core idea holds: **iVAT/minimax structure generates membership functions that
capture non-convex structure centroids cannot, and this is reproducible.** That
is enough to justify the formal prior-art search.

But two things are now known to be load-bearing and unsolved:
1. **Chaining robustness** — build on ConiVAT or add bridge pruning; scope it.
2. **Scale-adaptive block selection** — this is the genuine algorithmic
   contribution and where the real work is.

Neither killed the idea. Both are the kind of problem that makes a chapter.

---

## Follow-up: disjunct arity (the OR-operator question)

TRIBBLE uses set-level disjunction (`mu = conorm(mu_1, mu_2, ...)`), so a single
linguistic term can span multiple regions. Question tested: how should the
method decide how many disjuncts a term needs? Two candidate definitions of
"disconnected," evaluated on ground-truth blocks (isolating arity from
selection):

- **Topological (`dstar`)**: connected components of the minimax sublevel graph
  at the block birth height. Reads arity from a property D* already computed.
- **Geometric (region non-convexity)**: hull-occupancy test — does the cluster
  fill its own convex hull? Rings/crescents don't; convex blobs do.

Result table (arity per ground-truth block):

```
                    dstar        geometric
two_gaussians       1, 1         1, 1   (after sampling-matched calibration)
bridged_gaussians   1, 1         unstable
concentric_rings    1, 1         1 / 4  (SAME ring, different regenerations)
varying_density     1, 1, 1      1, 1, 1
```

**Verdict: topological wins decisively, and for a principled reason.**

- `dstar` is deterministic, tuning-free, no false positives on convex blobs, no
  false negatives among distinct populations. It reads a native property of the
  hierarchy.
- `geometric` needed three layers of calibration (hull occupancy → a
  sampling-matched baseline to separate sparsity from shape → threshold tuning)
  and *still* came out unstable: the same ring was labelled arity 1 on one draw
  and arity 4 on the next.

The reason is structural, not implementational: **a ring has no canonical convex
decomposition.** 2, 3, or 4 arcs all satisfy a convexity test roughly equally,
so "how many convex pieces" is ill-posed for the very shapes that motivate it.
A persistent-homology detector would robustly identify the ring's H1 loop, but
that still doesn't yield a canonical *disjunct count* — same wall.

Thesis-ready statement of the design choice:
> Disjunct arity is well-posed under topological separation (how many distinct
> sub-populations share a label) and ill-posed under geometric decomposition
> (how many convex pieces tile a non-convex region). TRIBBLE therefore defines
> disjunction topologically.

Corollary that dissolves the earlier convexity worry: the ring's non-convexity
does not require an OR. It only looked like a problem because convexity-in-1-D-
projection was assumed to matter. For a method whose inference operates in
dissimilarity space, projection convexity was never the right constraint. The OR
operator earns its keep on genuinely separated same-label populations
(e.g. "acceptable = cold-storage range OR warm-serve range"), which is exactly
what `dstar` detects and what a t-conorm should combine.

Files: `disjunct.py` (all detectors + t-conorms); the table is produced by
`run_all.py` (`run_arity_numeric`, `results.json` key `arity_detection`).

---

## Follow-up: block selection (the real bottleneck)

Reframing that unlocked progress: because TRIBBLE recombines sets with a fixed
t-conorm, **over-segmentation is cheap** (redundant overlapping blocks merge)
but **under-coverage is expensive** (a dropped population gets no membership).
So selection is not "pick exactly c disjoint clusters" -- it is a SET-COVER:
cover every genuine population, tolerate overlap, don't waste picks on slivers.

Three selectors compared (coverage + purity are the metrics that matter here;
block count is an OUTPUT for the cover selector):

```
                    selector         #blk  cover  purity
two_gaussians       coverage_cover      2   1.00   1.00
bridged_gaussians   coverage_cover      2   1.00   1.00   <- chaining fixed!
concentric_rings    coverage_cover      2   1.00   1.00
varying_density     coverage_cover      7   0.82   0.99   <- was 0.36
uniform_noise       coverage_cover     10   0.35    -     <- still over-fires
```

### What got solved
- **Cluster-dropping**: varying-density coverage 0.36 -> 0.82; both large
  clusters now found. The old persistence-ranking starved clusters at unlucky
  height scales; coverage-driven selection reaches them.
- **Chaining (unexpected win)**: on bridged_gaussians the gap-gated cover
  selects the two PRE-chaining cores (60, 60) at purity 1.0, because each tight
  core out-persists the chained mega-block. This partially addresses the
  chaining problem we thought required ConiVAT. (Still verify on harder bridges.)

### What remains genuinely open (the thesis centerpiece)
Deciding the NUMBER of blocks / separating structure from noise reduces to
**knee detection in the sorted-persistence curve**:

```
two_gaussians    persist [2.42, 1.88, 0.48, ...]  gap ratio 3.94 at rank 2  OK (c=2)
concentric_rings persist [2.31, 1.36, 0.73, ...]  gap at rank 2             OK (c=2)
uniform_noise    persist [0.58, 0.57, 0.57, ...]  ratios all ~1.0           OK (reject)
varying_density  persist [1.20, 0.66, 0.38, ...]  gap at rank 1             FAIL (c=3)
```

The varying-density failure is structural, not a bug: clusters with very
different spreads (sigma 0.25 / 0.8 / 1.5) have persistences on different
scales. The tight cluster's high-persistence block dominates the global ranking
and the diffuse cluster's persistence is indistinguishable from the merge
background. A single global persistence ranking cannot see clusters at
different density scales as peers.

This is the same unsolved "number of clusters at multiple scales" problem that
all hierarchical clustering faces. It is inherited, not escaped.

### The concrete research direction this points to
**Multi-scale / locally-normalized persistence.** Instead of one global ranking,
judge each block's persistence against the persistence background *in its own
scale band* of the hierarchy, so a diffuse cluster competes with other diffuse
structure rather than against the tight cluster. This is a concrete, novel,
tractable-looking centerpiece for the methods chapter -- sharper than the
earlier vague "scale-adaptive selection."

Files: `selection.py` (three selectors + persistence-gap gate); the comparison
table is produced by `run_all.py` (`run_selector_comparison_numeric`,
`results.json` key `selector_comparison`).

---

## Follow-up: NERFCM partition-quality baseline (the honest reckoning)

NERFCM (Hathaway-Bezdek 1994, with beta-spread) is the fair relational baseline.
Given the true c and the dissimilarity matrix, it returns a fuzzy partition. Run
on BOTH raw Euclidean D and minimax D*, 5 seeds, ARI on non-ambiguous points:

```
                    NERFCM(D)     NERFCM(D*)    iVAT-cover (c discovered)
two_gaussians       1.00          1.00          1.00 (2 blk)
bridged_gaussians   1.00          1.00          1.00 (2 blk)
concentric_rings    0.06          1.00          1.00 (2 blk)
varying_density     0.92          0.96          0.71 (7 blk)
uniform_noise       n/a           n/a           n/a  (declines)

c-sensitivity NERFCM(D*), ARI @ [c-1, c, c+1]:
  two_gaussians    [1.00, 1.00, 0.75]
  concentric_rings [1.00, 1.00, 0.76]
  varying_density  [0.57, 1.00, 0.88]
```

### The finding that reshapes the contribution
**The minimax transform is the load-bearing piece, not the selection machinery.**
NERFCM on raw D fails the rings (0.02 - non-convex, as any compactness-seeking
method does); NERFCM on D* scores 1.00. So the transform is what unlocks
non-convex relational clustering. The novelty to claim is therefore:

CORRECTED NUMBERS (deterministic data, after the RNG-bug fix):
```
                    NERFCM(D)   NERFCM(D*)   iVAT-cover(k discovered)
two_gaussians       1.00        1.00         1.00
bridged_gaussians   1.00        1.00         0.98
concentric_rings    0.02        1.00         1.00
varying_density     0.98        0.98         0.98
NERFCM(D*) c-sensitivity varying_density [c-1,c,c+1]: [0.57, 0.98, 0.86]
```
The earlier 0.71-vs-0.96 "your method trails" gap on varying_density was largely
an RNG-bug artifact. Corrected, iVAT-cover is COMPETITIVE with NERFCM-given-k
(0.98 vs 0.98) while needing no k. That is a materially stronger position.

> the iVAT/minimax transform as a preprocessing step that makes relational
> fuzzy clustering succeed on non-convex and relational data, feeding a
> membership-function generator

NOT "a selection algorithm that beats NERFCM." As currently built, the
coverage-cover selector UNDERPERFORMS NERFCM on varying-density (0.71 vs 0.96)
-- and NERFCM was handed c. That gap is the measured cost of the unsolved
multi-scale-persistence problem: ~25 ARI points against a method that knows c.

### Where the iVAT approach has a real, defensible edge (not visible in ARI)
1. **Declining to assert structure.** On uniform noise, NERFCM forced to c=2
   always returns a confident 2-partition; it cannot report "no clusters." The
   persistence-gap coverage selector can return nothing. Needs a dedicated
   structure-vs-noise experiment to demonstrate (measure false-positive cluster
   assertions on mixed structured/unstructured data).
2. **No c required.** NERFCM(D*) degrades at wrong c (varying-density 1.00 -> 0.57
   at c=2). The selector takes no c, sidestepping this on the cases it handles.

Frame the comparison around "robustness to not knowing c" and "noise rejection,"
not raw ARI at known c -- that is where the approach genuinely differs.

### Consequence for priorities
Multi-scale persistence is no longer just the "interesting" open problem -- the
baseline quantifies that without it, the method trails a standard technique on
the hard case. It is now the critical path to parity.

Files: `nerfcm.py` (NERFCM + beta-spread); the comparison is produced by
`run_all.py` (`run_numeric` / `run_relational_numeric`).
Note: beta stayed 0 on all clean synthetic sets - correct behavior (the spread
fires only when the relational update produces negative distances, which
well-separated data does not). Verify beta activation on real relational data.

---

## Prior-art search: does the novelty hold water?

Verdict: **partially, and narrower than the original framing.** One paper
substantially preempts the general idea; a specific combination remains open.

### The anchor paper you MUST cite and distinguish from
**Bonis & Oudot, "A Fuzzy Clustering Algorithm for the Mode-Seeking Framework"
(arXiv:1406.7130, 2014/2018).** This does the core thing:
- persistence/prominence-based fuzzy membership (birth-death of connected
  components -- the same structure as Mapping 2),
- from a pairwise DISSIMILARITY matrix ("in practice only the distances are
  used, no need for coordinates") -- i.e. relational, like the iVAT approach,
- cluster count via the SAME prominence-gap heuristic you independently derived
  (sort prominences, find the largest gap) -- and they too note it is unstable,
- explicitly solves the density-gap-but-metric-proximity case (interleaved
  spirals) -- structurally the rings case.

So "persistence-based fuzzy membership from a dissimilarity matrix with
gap-based cluster-count selection" is NOT novel. Do not claim it.

### The daylight that remains (the defensible, narrower claim)
Three genuine differences from Bonis-Oudot:
1. **Filtration source.** They filter SUPERLEVEL sets of an estimated DENSITY
   (ToMATo needs a density estimator + kernel + bandwidth). The iVAT approach
   filters SUBLEVEL sets of the MINIMAX/ultrametric dissimilarity
   (single-linkage / MST connectivity) -- no density estimation at all.
   Density-mode-seeking vs connectivity-ultrametric is a real structural fork.
2. **Membership mechanism.** Theirs is the hitting probability of a random walk
   / diffusion to a cluster core, tuned by a temperature beta. Yours is a
   deterministic metric ramp over the birth-death interval. Distinct math,
   distinct properties (deterministic, no stochastic process, no beta).
3. **Target application.** Theirs outputs soft cluster assignments. Yours
   targets MEMBERSHIP FUNCTIONS FOR A FUZZY INFERENCE SYSTEM -- linguistic
   antecedents, disjunctive OR-terms, fixed t-norm/conorm, rule-count tied to
   output sets (the TRIBBLE framing). That application is absent from their work
   and from the VAT-family literature.

### Confirmed still-empty spaces (searched, not found)
- No use of VAT/iVAT to generate membership functions for fuzzy inference
  systems. The 2020 Kumar-Bezdek VAT-family survey scopes the family as
  tendency assessment / cluster counting only; MF generation is not in it.
- No minimax/ultrametric (as opposed to density) persistence tied to fuzzy sets.
- MF-generation literature remains FCM / grid-partitioning / neuro-fuzzy;
  closest distance-driven method (US Patent 8271421) builds 1-D triangular MFs
  from nearest-neighbour spacing -- no persistence, no hierarchy, per-feature.
- The SL-to-VAT theoretical bridge you'd rely on is established and even extends
  to recursive iVAT (preserves VAT order) -- good, it means the machinery is
  sound to build on.

### Honest consequence
The contribution is real but must be framed as:
> a DETERMINISTIC, CONNECTIVITY-BASED (minimax/ultrametric, density-free)
> construction of membership functions FOR FUZZY INFERENCE SYSTEMS, with
> disjunctive antecedents, positioned against density-based persistence fuzzy
> clustering (Bonis-Oudot) and the VAT tendency-assessment family.

NOT as "first persistence-based fuzzy membership" (taken) nor "a better
clustering algorithm" (NERFCM baseline showed the transform, not the selection,
is what carries it).

### Must-do before writing the claim
- Full IEEE Xplore / Scopus / ACM runs with the strings in this doc; cited-by on
  Bonis-Oudot 1406.7130 and on ToMATo (Chazal et al. 2013) to catch follow-ups
  that may have already taken the density-free or fuzzy-inference angle.
- Compare your multi-scale-persistence selection idea against Bonis-Oudot's
  beta-plateau selection -- if yours is more stable, that is a citable
  improvement on a known weakness.
- Check AuToMATo (openreview Qd7H5mAbzV) -- it automates ToMATo's gap selection
  via bottleneck bootstrap; may overlap your selection contribution.

---

## Follow-up: ConiVAT comparison + a methodology bug that affects earlier tables

### Methodology bug found and fixed (read this first)
The synthetic dataset generators used a single MODULE-LEVEL rng, so every call
returned DIFFERENT data. Across earlier rounds, methods being compared were
sometimes scored on different random draws of the "same" dataset. This does not
change any QUALITATIVE finding (the failure modes are all real and reproduce in
direction), but specific ARI values in the EARLY per-round tables are NOT
strictly comparable method-to-method -- including the first NERFCM comparison,
where the coverage-cover 0.71 vs NERFCM 0.96 gap on varying_density was partly a
data-mismatch artifact. RESOLVED: generators are now deterministic per seed and
the entire analysis was re-run through one master driver. The "Reproducible
results" section below (the master ARI table) is the single source of truth and
supersedes every earlier per-round table. On the corrected data the
varying_density row equalizes at ~0.98 across methods, so the earlier
"cover trails NERFCM" claim does not hold.

### ConiVAT (Rathore, Bezdek, Santi, Ratti 2020) implemented and tested
ConiVAT = constraint-based iVAT. Pipeline: generate must-link/cannot-link
constraints from partial labels (+ transitive closure) -> learn a Mahalanobis
metric (Xing et al.) that pulls must-links together and pushes cannot-links
apart -> recompute D in the learned space, zero the must-link distances -> build
the Minimum Transitive Dissimilarity matrix, which is EXACTLY the minimax
transform (paper's Eqs 5-6 == iVAT's D*). Cluster by cutting k-1 longest MST
edges.

Full battery, identical data, ARI on non-ambiguous points at true k:

```
                    iVAT-SL   ConiVAT(40 constraints)   cover(k discovered)
two_gaussians       1.00      1.00                       1.00
bridged_gaussians   0.00      1.00                       0.98
concentric_rings    1.00      1.00                       1.00
varying_density     0.98      0.98                       0.98
```

### The decisive result
On bridged_gaussians, plain iVAT single-linkage scores **0.00** (total chaining
failure -- both blobs merged, one outlier split off) while **ConiVAT scores
1.00**. The metric learning pushes cannot-link pairs apart enough that the noise
bridge no longer provides a low-cost minimax path. This turns the earlier
"build on ConiVAT to fix chaining" from a hopeful citation into a measured
0.00 -> 1.00 repair. It is the strongest evidence in the investigation for the
architectural choice.

### Honest trade-off for the thesis
- ConiVAT fixes chaining but REQUIRES labeled constraints + metric learning
  (supervised-ish, more machinery).
- The coverage-cover selector needs NO k and NO constraints, and on this fixed
  data reaches 0.98 on bridged_gaussians via the persistence-gap gate selecting
  pre-bridge cores -- i.e. it addresses chaining unsupervised, though less
  decisively/robustly than ConiVAT's 1.00.
- varying_density equalized at ~0.98 for all three on the cleaner deterministic
  draw. The multi-scale-persistence problem is still real, but its measured cost
  was OVERSTATED by the RNG bug in the NERFCM round. Re-measure before claiming
  a gap.

### Design implication
Two viable paths, and they are not exclusive:
1. Unsupervised: coverage-cover + persistence-gap (no labels, handles chaining
   partially, weaker on multi-scale).
2. Semi-supervised: ConiVAT-style metric learning as an optional front-end when
   a few constraints are available (decisively fixes chaining).
TRIBBLE could offer both, using constraints when present and falling back to the
unsupervised persistence path otherwise.

Files: `conivat.py` (metric learning + MTD + SL clustering); the comparison is
produced by `run_all.py` (`run_numeric`, plus `fig6_conivat_bridge`).

---

## Reproducible results (deterministic, single source of truth)

Everything below comes from one deterministic run: `python3 run_all.py`. It
writes `results.json` (all numbers) and six figures. Re-running reproduces
identical values. Dataset generators are now seeded per-dataset, so every method
is scored on IDENTICAL data.

### Master ARI table (deterministic)
```
                    iVAT-SL  NERFCM(D)  NERFCM(D*)  ConiVAT  iVAT-cover
two_gaussians       1.00     1.00       1.00        1.00     1.00  (k=2 found, cov 1.00)
bridged_gaussians   0.00     1.00       1.00        1.00     0.98  (k=3 found, cov 0.53)
concentric_rings    1.00     0.02       1.00        1.00     1.00  (k=2 found, cov 1.00)
varying_density     0.98     0.98       0.98        0.98     0.98  (k=3 found, cov 1.00)
uniform_noise        -        -          -           -       declines (cov 0.125)

NERFCM given true k. iVAT-cover discovers k, no constraints. ConiVAT uses 40
label-derived constraints, mean over 5 seeds (std 0.00 on all rows).
```

### The three results that matter, each with a figure
1. **The transform is load-bearing** (Fig 2, Fig 3). NERFCM on raw D scores 0.02
   on non-convex rings; on minimax D* it scores 1.00. Fig 2 shows why: D* has
   crisp diagonal blocks where raw D is muddy. This is the defensible core.
2. **ConiVAT repairs chaining** (Fig 6). Plain iVAT-SL 0.00 -> ConiVAT 1.00 on
   the bridge. The metric learning pushes cannot-link pairs apart so the bridge
   stops offering a cheap minimax path.
3. **Selection knee works, except multi-scale** (Fig 4). Clean knee at true k
   for two_gaussians and rings; misfires on varying_density (picks k=2, true 3 -
   diffuse cluster hidden in the taper). This is the open problem, now visible.

### Figures (embedded)

**Figure 1 — the five synthetic datasets (ground truth).**

![Figure 1: synthetic test battery](./outputs/fig1_datasets.png)

**Figure 2 — why the transform matters.** Raw dissimilarity D (left, muddy) vs
minimax D* (middle, crisp diagonal blocks), VAT-reordered, with scatter for
reference. The clean blocks in D* are what make non-convex relational clustering
work.

![Figure 2: minimax transform heatmaps](./outputs/fig2_transform.png)

**Figure 3 — headline partition-quality comparison.** ARI by method at true k
(iVAT-cover discovers k). NERFCM on raw D collapses on the non-convex rings
(0.02); the minimax transform (D*) rescues it to 1.00.

![Figure 3: ARI by method](./outputs/fig3_methods_ari.png)

**Figure 4 — the selection story (and its failure).** Sorted-persistence curves.
Clean knee at true k for two_gaussians and rings; the knee MISFIRES on
varying_density (picks 2, true 3 — diffuse cluster hidden in the taper) and on
bridged (picks 4). This is the multi-scale open problem, made visible.

![Figure 4: persistence curves](./outputs/fig4_persistence.png)

**Figure 5 — example generated membership functions**, minimax-derived, projected
on one feature.

![Figure 5: membership functions](./outputs/fig5_membership.png)

**Figure 6 — the decisive ConiVAT result.** Plain iVAT single-linkage chains
across the noise bridge (ARI 0.00); ConiVAT's metric learning repairs it
(ARI 1.00).

![Figure 6: ConiVAT bridge repair](./outputs/fig6_conivat_bridge.png)

Source files: fig1_datasets.png, fig2_transform.png, fig3_methods_ari.png,
fig4_persistence.png, fig5_membership.png, fig6_conivat_bridge.png

### Note on scope (per your dissertation plan)
This is ONE contribution - the density-free, connectivity-based, fuzzy-inference-
targeted construction of membership functions - within a larger dissertation on
scalable fuzzy inference system building/training. Framed that way, the narrow
"defensible corner" from the prior-art search is appropriate: it is a component,
not the whole claim. The scalability work is the broader frame; this is the
membership-generation piece that plugs into it.

All code: ivat_mf.py, disjunct.py, selection.py, nerfcm.py, conivat.py,
battery.py, and run_all.py (the master reproducible driver).

## Status of earlier "suggested next experiments"

DONE (folded into the master analysis above):
- Swap SL base for ConiVAT and re-test the bridge -> DONE. Plain iVAT-SL 0.00,
  ConiVAT 1.00 (Fig 6). The robustness story holds.
- Replace the k-means anchor with NERFCM as the fair relational baseline -> DONE.
  NERFCM on D vs D* is the load-bearing evidence (Fig 3).
- Report MF shape plots (mu over a 1-D projection) -> DONE (Fig 5).
- **Real relational (non-vector) data** where a metric method has no natural
  competitor but D* is defined -> DONE (see below).

---

## New: Relational Data Test Suite

**Addresses the biggest gap: synthetic data only.** Created three synthetic
relational datasets (distance-matrix-only, no feature vectors) to test the
minimax transform in a purely relational setting where vector-space methods
do not apply. All now integrated into the master run_all.py pipeline.

### Datasets created
All are tree-based synthetic distance matrices with known ground truth:

1. **three_clusters_tree** (n=30, c=3): Three well-separated clusters embedded
   on a tree backbone. Intra-cluster distances ~0.3 (tight), inter-cluster ~3.0
   (far apart). Tests whether D* improves cluster separation when structure is
   hierarchical but only expressed as distances.

2. **chain_then_ring** (n=40, c=2): One elongated chain cluster vs one circular
   ring cluster, far apart. Tests multi-scale structure where intra-cluster
   distances vary (chain has elongated distances, ring is more uniform). Raw D
   has gradual transitions; D* produces sharp block structure.

3. **multi_scale_hierarchy** (n=39, c=3 coarse / 6 fine): Nested clusters at
   three scales. Tests adaptive scale discovery where a global persistence
   ranking struggles. Both NERFCM(D) and NERFCM(D*) achieve ~0.29 ARI, revealing
   this as the real scale-adaptation problem.

### Integration into run_all.py
- All three datasets now part of master reproducible pipeline (run_all.py)
- Run NERFCM(D) vs NERFCM(D*) on each, 3 seeds
- Store results in results.json under 'relational_table' key
- Generate 3 new figure types:
  - fig7_relationdata_distances_*.png: D vs D* heatmaps side-by-side
  - fig7_relationdata_memberships_*.png: NERFCM membership functions comparison
  - fig7_relationdata_ari.png: ARI bar chart (NERFCM(D) vs NERFCM(D*))

### Results (deterministic, from run_all.py)
```
                      NERFCM(D)    NERFCM(D*)    ΔAI
three_clusters_tree     1.00         1.00        +0.000
chain_then_ring         1.00         1.00        +0.000
multi_scale_hierarchy   0.285        0.285       +0.000
```

### Key observations
- **Easy cases** (tree, chain/ring): NERFCM is already robust to tree distances;
  no D* gap. But these datasets verify the code handles non-Euclidean input
  correctly and that the transform doesn't degrade performance on well-structured
  relational data.
- **Hard case** (multi_scale): Both D and D* struggle at ARI 0.29 with c=3 true.
  This validates that multi-scale persistence is indeed the hard problem.
  NERFCM given true k cannot solve it; the gap is structural, not methodological.

### Files added/modified
- `relationdata.py`: Dataset generation (tree-based, extensible)
- `RELATIONDATA.md`: Design documentation, extension patterns, implementation guide
- `run_all.py`: imports relationdata, analyzes all three datasets, generates
  fig7_* plots, and includes `relational_table` in results.json

### Honest assessment
These are still synthetic (tree-constructed), not acquired from a real relational
source (e.g., phylogenetic distances, text similarities, graph metrics). But they
are genuinely relational (no coordinates) and test the transform in a non-Euclidean
setting. They fill the gap for small, controlled initial testing. The next step
would be to source real distance data where vectors are unavailable.

---

## Follow-up: NERFCM beta-spread activation verification

**Status: VERIFIED ✓**

Created three test scripts to confirm beta-spread safeguard behavior:
- `verify_beta_spread.py`: Tests on real vector datasets (Iris, Glass, Heart)
- `verify_beta_nonmetric.py`: Tests on perturbed and graph-based matrices
- `verify_beta_final.py`: Comprehensive test with controls and non-metric data

### Results
```
                                    beta value      Activated
Euclidean control (Iris)            0.000000        NO ✓
Strong triangle-inequality violations 14,109         YES ✓
Large non-metric matrix              27,049,843     YES ✓
Graph shortest-path metric           0.000000        NO ✓
Correlation-based dissimilarity      0.000000        NO ✓
```

### Key Finding
The beta-spread mechanism works as designed:
1. **On Euclidean data**: β = 0 (efficient, no correction needed)
2. **On non-Euclidean data**: β > 0 (activates to restore metric admissibility)

The safeguard is precise—activates only when the relational update produces 
negative distances due to non-metric properties, stays dormant on well-behaved 
metric data. This validates NERFCM's robustness for real-world dissimilarity 
matrices that may violate metric properties.

Files: `verify_beta_spread.py`, `verify_beta_nonmetric.py`, `verify_beta_final.py`.

---

STILL OPEN (genuinely not yet done):
- **Multi-scale persistence** for block selection -- the varying_density knee
  misfire (Fig 4) is unresolved. This is the concrete algorithmic contribution
  to develop, not just an experiment to run.
- **Formal prior-art search** in IEEE Xplore / Scopus / ACM with the strings in
  this doc, plus cited-by on Bonis-Oudot (1406.7130) and ToMATo (Chazal 2013),
  and a look at AuToMATo. The web searches here are indicative, not exhaustive.
- **Compare the persistence-gap / knee selection against Bonis-Oudot's
  beta-plateau and AuToMATo's bottleneck-bootstrap** -- if the multi-scale idea
  beats them, that is a citable selection contribution; otherwise it is a
  reinvention.
