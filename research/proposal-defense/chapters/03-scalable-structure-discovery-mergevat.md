# Chapter 3 — Scalable Structure Discovery: mergeVAT

**Status:** Outline · Part II (COMPLETED work — NAFIPS papers 1 & 2)
**Repo:** `tribble-cluster`
**Mirrors:** Pickering Ch 4/Ch 6 (a completed methods paper with intro→prior-art→method→results→discussion).
**One-line claim:** an exact, parallel, memory-lean VAT/iVAT engine that operates on arbitrary (incl. non-metric) dissimilarity matrices, lifting feasible problem size from ~5K to 130K+ and occupying regimes existing fast-VAT methods cannot.

---

## 3.1 Introduction

- Motivation: preliminary data review / structure discovery is step 1 of the FIS pipeline; must scale to 10⁵–10⁶ rows and to non-Euclidean dissimilarities. VAT is the ideal tool but was O(N³).
- Contributions of the chapter (enumerate): (1) O(N²log N) exact VAT/iVAT; (2) in-place memory engine; (3) GPU-Borůvka exact VAT; (4) divide-&-conquer with principled stitch; (5) correctness on arbitrary/non-metric dissimilarity; (6) VAT↔TSP hot-start (secondary).

## 3.2 Background & Prior Art

- Recap VAT/iVAT and the **MST-only dependence** insight (→ "fast VAT ≡ fast MST"): serial Prim, parallel Borůvka, or GPU kernel all reproduce bit-identical VAT.
- **Competitors to differentiate (the reviewer's first demands):**
  - *clusiVAT* (Kumar 2013/2016) — sampling, only approximate.
  - *eVAT* (Meng–Yuan 2018) — exact GPU VAT; pre-empts a naive "first GPU VAT" headline — cite and differentiate (arbitrary-dissimilarity, in-place, D&C).
  - *Fast-VAT* (Avinash–Lachheb 2025) — concurrent Cython+Numba CPU, VAT-only, up to 50×.
  - *BB-VAT / kdT-VAT* (Information Sciences 2024) — sub-quadratic memory but **Euclidean-only** (needs coordinates) — cannot run on precomputed non-metric D.
- **The unoccupied regime (the niche):** exact VAT/iVAT on a large, arbitrary, precomputed, non-metric dissimilarity matrix, CPU-parallel + memory-lean — no competitor covers this.

## 3.3 Methodology

### 3.3.1 The O(N²log N) reorder ("mergeVAT")
- Historical inner loop was BubbleSort-style argmin over the column remainder → O(N³).
- Replace with a **priority queue / binary(Fibonacci) heap** extract-min → O(1) amortized; overall O(N²log N). (Kreinovich's note: closer to HeapSort; name is historical from a failed 2D-mergesort experiment — include the anecdote, correct the O-notation.)
- **Fix to make (defensibility):** retire the "priority-queue MST speedup" framing for *dense* graphs — heap-Prim is O(N²log N) vs O(N²) dense-Prim; defend empirically or restate as the argmin/sort speedup it actually is.

### 3.3.2 In-place memory engine
- VAT caches full D and D′ (2N²). Compute D_{i,j} on demand (one copy + workspace); lower-triangular in-place iVAT recursion.
- **Loop-walking / bit-masked cycle-following permutation** (Cate–Twigg 1977; Catanzaro 2014): the VAT sequence + original sequence form directed cyclic loops; follow loops to permute in place → 2 buffers → 1.
- **Correctness note (must include):** the first shipped in-place permutation was silently wrong (coupled a cell with its mirror); tests only checked permutation-invariant quantities. Fixed & verified bit-identical (PR #18). Add a regression test. Honesty here is a strength.

### 3.3.3 GPU acceleration
- On-device Borůvka MST + device-resident VAT front-end (distances→MST→order), bit-identical to serial.
- Data-resident GPU Fuzzy-C-Means.

### 3.3.4 Divide-&-conquer with a principled stitch
- Naive block concat → seam / pseudo-cluster artifacts.
- **Principled stitch:** farthest-point-sampling boundary representatives + top-m cross-edges → partition-robust at bounded O(N²r²).
- Reverse-delete (Kruskal twin) with min-degree knob: m=1→MST (dual of VAT's additive Prim, bit-identical), m=2→Hamiltonian tour.

### 3.3.5 VAT↔TSP hot-start (secondary; standalone-paper candidate)
- MST ordering as seriation/TSP Hamiltonian-path warm start (Lenstra 1974; Climer–Zhang 2006); MST-seeded ACO pheromone hot-start (Dai–Ji–Liu 2009 nearest competitor); VAT-cluster-blocking D&C TSP (CTSP; Guttmann-Beck 2000 endpoint stitch).
- **Report the negatives honestly:** VAT's raw *closed-tour* init is the worst start; LKH is start-insensitive; shorter tour ≠ better clustering. Verdict: compositional/engineering, not algorithmically novel. (This is why it's secondary.)

## 3.4 Results

*Tables/figures to port from `SUMMARY_REPORT.md`, `white-paper.md`, `ADVERSARIAL_EVAL_FINDINGS.md`, quals paper1/paper2 slides.*

- **Scaling (headline):** 4096-element dataset 124 s → 2.56 s; feasible size 5K → 130K+; "58K×58K in 60 s." Table: N vs time (BubbleSort-VAT vs mergeVAT), the ~8000× at 135K.
- **In-place memory:** max feasible N at 64 GB 52K → 89K; N=64000 float64 iVAT 98 GB (infeasible) → 32.85 GB / 25 s.
- **GPU:** FCM 30–56× (n=50K–500K, >99% identical labels); Borůvka MST ~5× at n=32000 (order match 1.0); on-device VAT front-end 4.8–6.6×. **Honest negative:** GPU pairwise distances 1.3–2.5× only at high-d+float32, <1× at low-d/float64.
- **Clustering quality (adversarial eval, ARI):** two_moons/circles VAT & stitched = 1.00 where k-means = 0.27/0.00; inherits single-linkage failures on bridged/aniso (0.00). Stitch ablation: principled (fps+top-m=8) mean ARI 1.00 across all partitions vs light 0.51.
- **Non-metric robustness:** stitch preserves exact single-linkage on fractional p=0.5 Minkowski (triangle-inequality violated 14%), cosine, kNN-geodesic — agreement 1.0.
- **Figures:** VAT/iVAT RDI images (shuttle 58K, psych 135K), Prim-MST block diagram (`img/vat_prim_mst_block_diagram_v2.svg`), scaling curves, adversarial ARI table, stitch ablation grid.

## 3.5 Discussion & Contributions

- What the composition buys: exact + parallel + non-metric + bounded-cost D&C in one engine; error confined to where single-linkage itself is unreliable.
- Limits: inherits single-linkage bridge/aniso failures (→ ConiVAT / Ch 5 metric learning addresses this); consumer-GPU FP64 penalty; single-seed timings.
- **Before-submission checklist:** thermally-stable re-timing + error bars; datacenter GPU full-rate FP64; head-to-head vs eVAT & clusiVAT on identical data; drop the ungrounded "pVAT six-orders-of-magnitude" web claim (self-citation at most); verify DOIs.

---

### Open items
- Confirm NAFIPS paper numbers / publication status for the Publications chapter.
- Decide whether VAT↔TSP gets a full section or a short subsection (recommend short; spin the paper separately).
