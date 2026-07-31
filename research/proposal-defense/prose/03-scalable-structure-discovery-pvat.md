# Chapter 3 — Scalable Structure Discovery: pVAT

## 3.1 Introduction

The first step in the pipeline is to look at the data before modeling it, and to do that at the scale the data actually arrives in. I introduced VAT in Chapter 2 as the tool I use for this: it takes a dissimilarity matrix, reorders it so that similar points sit together, and shows me the cluster structure as dark blocks along the diagonal. It is exactly the right tool, with one problem — in its textbook form it is far too slow and far too memory-hungry to run on the datasets I care about. A 58,000-point matrix is already out of reach for the classical algorithm, and I want to go well past that.

This chapter presents pVAT, an accelerated VAT and iVAT engine, and the collection of implementation ideas that make it work. The contributions are: an $O(N^2 \log N)$ reordering that replaces the classical cubic inner loop; an in-place memory scheme that holds the whole computation in a single matrix; a GPU implementation of the underlying minimum spanning tree that is bit-identical to the serial result; a divide-and-conquer scheme that splits a large problem and stitches the pieces back together at bounded cost; and, importantly, correctness on an *arbitrary* dissimilarity matrix, including non-metric ones. Together these move the feasible problem size from a few thousand points to well over a hundred thousand. A secondary result, which fell out of the same machinery, is that the VAT ordering makes a good hot-start for the Traveling Salesman Problem; I treat that briefly at the end.

## 3.2 Background and Prior Art

Recall the one fact that everything here rests on: the VAT ordering depends only on the minimum spanning tree of the points. VAT grows an MST with a modified Prim's algorithm, and the reordering is just the order in which points are added. Any method that builds the same MST — serial Prim, parallel Borůvka, or a GPU kernel — produces the same ordering, bit for bit. So "make VAT fast" is really "make the MST fast," and I am free to choose whichever MST construction suits the hardware.

There is already work on fast VAT, and I want to place pVAT against it plainly, because these are the comparisons a reviewer asks for first. **clusiVAT** [Kumar et al. 2013] samples the data and is therefore approximate — it is fast, but it is not the exact VAT ordering. **eVAT** [Meng and Yuan 2018] is an exact GPU VAT, so I cannot claim to be the first to put VAT on a GPU, and I do not. The kd-tree memory methods [Information Sciences 2024] cut memory below quadratic, but they require Euclidean coordinates and cannot run on a precomputed dissimilarity matrix at all. The regime none of them occupy is the one I target: *exact* VAT and iVAT, on a *large, arbitrary, possibly non-metric* dissimilarity matrix, computed with modest memory on ordinary hardware. That is not a corner case — sequence data under dynamic time warping, strings under edit distance, and graphs under a kernel dissimilarity all land there, and none of them have coordinates.

## 3.3 Methodology

### 3.3.1 The reorder, and a note on the name

The classical VAT reorder is slow for a mundane reason. At each step it needs the smallest remaining dissimilarity, and the usual implementation finds it by scanning the remainder of the current column — an $O(N)$ linear search, repeated $N$ times across $N$ columns, which is where the cubic behavior comes from. Replacing that linear scan with a priority queue turns each extraction into an $O(\log N)$ operation, and the whole reorder into $O(N^2 \log N)$. This is the single change that lets the method scale, and it is worth being precise about what it buys: the win is in the argmin, not in some new MST theory, and against a maximally tuned dense-Prim implementation the asymptotic gap is smaller than the raw numbers suggest. The measured speedups below are real regardless, and they come from the priority queue plus the memory scheme in the next section.

A word on the name, because it is a small story I am fond of. I originally called this *mergeVAT*, after a separate idea I was chasing at the time — an attempt to structure the reordering as something like a two-dimensional merge sort. That idea did not pan out, but the name stuck to the code. Dr. Vladik Kreinovich pointed out that what the working method actually does is far closer to a heap sort than a merge sort: it is a priority-queue algorithm. He is right, and so I have renamed it **pVAT**, for priority-queue VAT. The rename is a deliberate nod to his observation, and it has the side benefit of describing the method honestly — the priority queue on the CPU, and the Borůvka construction I use on parallel hardware, are the two things the name should point at.

**[FIGURE 3.1 — placeholder]** *Block diagram of the pVAT reorder: the priority-queue extraction of the minimum remaining dissimilarity, replacing the classical linear-scan argmin. (Adapt `vat_prim_mst_block_diagram_v2.svg`.)*
`![pvat-reorder](fig/03-pvat-reorder.png)`

### 3.3.2 Holding it in one matrix

Speed is only half the problem; memory is the other half, and at these sizes it is the wall you hit first. A 135,000-row psychiatric-evaluation dataset I worked with — 165 mostly-binary features whose names were anonymized, so their meaning is moot here and the exercise is purely one of scale — is 73 gigabytes as a single-precision distance matrix, and the classical VAT keeps *two* such matrices, the original and the reordered copy, to trade memory for compute. Since pVAT no longer needs that trade, I would rather spend the memory budget on a bigger problem.

Two ideas do this. The first is to never materialize the full distance matrix at all: compute each dissimilarity $D_{i,j}$ on demand as the algorithm asks for it, keeping only one copy plus working space. The second is to do the reordering permutation *in place*. The VAT ordering, paired with the original index order, forms a set of directed cycles — permutation loops — and I can walk each loop, moving elements into their final positions, masking entries as I visit them and incrementing past the ones already placed. This is a known trick for in-place permutation [Cate and Twigg 1977], and it collapses the two permutation buffers down to one. The net effect is that iVAT, which for a 64,000-point float64 problem would need about 98 gigabytes and simply not run, instead runs in about 33 gigabytes in 25 seconds, and the largest feasible problem on a 64-gigabyte machine rises from roughly 52,000 to 89,000 points.

I want to flag one thing here, because it is the kind of thing that is easy to hide and I would rather not. The first version of my in-place permutation was silently wrong — it coupled each cell with its mirror image across the diagonal, and produced a plausible-looking result that was not the correct ordering. My tests missed it for an embarrassing stretch because they only checked quantities that happen to be invariant under that particular error. I found it, fixed it, and added a test that checks the ordering itself against the serial reference bit for bit. The current implementation is verified identical to serial VAT. The lesson I took from it is that "the picture looks right" is not a test, and it now informs how I validate everything in this work.

### 3.3.3 Onto the GPU

Because the ordering depends only on the MST, I can build the MST however I like. On a GPU the natural choice is Borůvka's algorithm, which contracts many MST edges in parallel each round, and I run the whole front end — distances, MST, ordering — on the device so the data does not shuttle back and forth. The result is bit-identical to serial VAT. On my hardware (a 32-core CPU and a laptop-class RTX 4080) the on-device MST is about five times faster than serial Prim at 32,000 points, and the gap widens with size; the full on-device VAT front end runs about five to seven times faster end to end. I also run Fuzzy C-Means on the device, where the win is larger — thirty to fifty times over the 32-core CPU across 50,000 to 500,000 points, converging to the same fixed point with over 99% identical labels.

The honest caveat is about precision. The pairwise distance computation on the GPU only wins at high dimension in single precision; at low dimension or in double precision it actually *loses* to the CPU, because a consumer card's double-precision throughput is a small fraction of its single-precision throughput. I report this because it matters for anyone trying to reproduce the numbers on a datacenter card, where the double-precision penalty disappears and the picture would change.

### 3.3.4 Splitting the problem

For problems past what fits on one machine, I split the data into blocks, run pVAT on each, and stitch the orderings together. The naive version of this — order each block and concatenate — is fast but wrong: it creates seams at the block boundaries that show up as spurious clusters. The fix is a principled stitch. I pick boundary representatives from each block by farthest-point sampling, add the strongest handful of cross-block edges between them, and reconcile the orderings across those edges. This keeps the reconstruction faithful to the true single-linkage structure at a bounded cost that grows only with the number of representatives, not with the block size. As I show below, both ingredients — the farthest-point representatives and the top cross-edges — are necessary; either one alone is not enough.

### 3.3.5 A hot-start for the Traveling Salesman Problem

One more result came out of this, and I include it because it is a neat consequence rather than a central claim. The MST gives a well-known bound on the optimal tour, $T_{best} \le 2\,T_{MST}$, and the VAT ordering is essentially a walk of that tree. So visiting the VAT-ordered points in sequence gives a tour that is provably within a factor of two of optimal, computed almost for free. It makes a reasonable warm start for a stochastic TSP solver. I will be candid about the limits, though: VAT's raw closed *tour* is actually a poor starting point for the strongest solvers — Lin–Kernighan and LKH are largely insensitive to where they start — and a shorter tour does not imply a better clustering. The honest verdict is that this is a useful engineering connection, not a new optimization result, which is why it sits at the end of the chapter and not the front.

## 3.4 Results

*Hardware: 32-core Intel CPU, 64 GB RAM, laptop RTX 4080 (12 GB). Every result labeled "exact" is bit-identical to the serial VAT reference.*

> **TODO — repeatable performance (board-wide standard):** the numbers in this section are single-machine, some taken on a thermally throttled laptop. Before any of them are cited as scalability *or* stability results they must be reproduced under a fixed protocol — pinned clocks/thermals, multiple seeds, reported error bars, and a datacenter GPU with full-rate FP64. This same standard applies to every performance/scaling claim in the dissertation (Ch 5, Ch 6). Tracked as Goal G4 in Chapter 7.

**Scaling.** The reorder speedup is the headline. On a 4,096-point problem the classical method takes 124 seconds and pVAT takes 2.56 seconds; at 135,000 points the improvement is roughly eight thousand-fold, which is the difference between "run it over lunch" and "run it interactively." The feasible problem size moves from about 5,000 points to over 130,000, and the NASA shuttle set — 58,000 points — orders in about a minute, which is where the paper title comes from.

**Table 3.1 — Reorder time, classical VAT vs. pVAT.** *(Confirmed points below; intermediate rows and the classical 135K figure are extrapolations/estimates to be filled in and error-barred under the G4 protocol.)*

| N (points) | classical VAT | pVAT | speedup |
|---:|---:|---:|---:|
| 4,096 | 124 s | 2.56 s | ~48× |
| 58,000 | infeasible on this hardware | ~60 s | — |
| 135,000 | infeasible (extrapolated) | — | ~8,000× (reported) |

**Memory.** The in-place scheme takes the 64,000-point float64 iVAT from infeasible (≈98 GB) to 33 GB in 25 seconds, and raises the largest feasible problem on 64 GB from ≈52,000 to ≈89,000 points.

**GPU.** On-device Borůvka MST is ≈5× serial Prim at 32,000 points and growing; the full on-device VAT front end is ≈4.8–6.6× end to end; GPU Fuzzy C-Means is 30–56× across 50,000–500,000 points at >99% label agreement. Pairwise distances win (1.3–2.5×) only at high dimension in float32, and lose below 1× at low dimension or in float64.

**Clustering quality.** Because pVAT is exact single-linkage, it inherits single-linkage's strengths and weaknesses honestly. On non-convex data where k-means fails — two moons, concentric circles — pVAT and the stitched version both reach an adjusted Rand index of 1.00, against 0.27 and 0.00 for k-means. On bridged or touching-anisotropic clusters, where a single chain of points connects two real groups, pVAT scores 0.00, exactly as single-linkage does; I do not paper over this, and it is precisely the failure mode that Chapter 5's metric-learning and persistence work is meant to repair.

**Table 3.2 — Adversarial clustering quality (adjusted Rand index).** pVAT is exact single-linkage, so it wins where k-means fails and inherits single-linkage's failures honestly.

| Dataset | k-means | single-linkage | exact pVAT | naive block | principled stitch |
|---|---:|---:|---:|---:|---:|
| two_moons | 0.27 | 1.00 | 1.00 | 0.39 | **1.00** |
| circles | 0.00 | 1.00 | 1.00 | 0.10 | **1.00** |
| aniso | 0.61 | 0.00 | 0.00 | 0.30 | 0.00 |
| bridged | **1.00** | 0.00 | 0.00 | 0.07 | 0.00 |

**The stitch.** Ablating the divide-and-conquer stitch on two moons across a grid of partitions and sizes: the light stitch (random representatives, one cross-edge) averages ARI 0.51; farthest-point representatives alone or top cross-edges alone are similar or worse; the principled combination of both reaches a mean ARI of 1.00 across every partition tested, at bounded cost. Both ingredients are required together.

**Table 3.3 — Stitch ablation on two moons, over a grid of partitions and sizes.**

| Stitch variant | mean ARI | min ARI | fraction ≥ 0.9 |
|---|---:|---:|---:|
| light (random rep, 1 cross-edge) | 0.51 | 0.00 | 0.44 |
| top-m cross-edges only (m = 8) | 0.74 | 0.00 | 0.72 |
| farthest-point reps only | 0.39 | 0.00 | 0.32 |
| **principled (fps + top-m = 8)** | **1.00** | **1.00** | **1.00** |

**Non-metric robustness.** This is the point of the whole exercise, so I test it directly. On a fractional Minkowski dissimilarity with $p = 0.5$ — which violates the triangle inequality about 14% of the time — and on cosine and on a k-nearest-neighbor geodesic dissimilarity, the stitched pVAT agrees with exact VAT to within a rounding error (agreement 1.0). The method does not quietly assume a metric, which is what lets it run on the data that has no coordinates.

## 3.5 Discussion and Contributions

What the composition buys is an engine that is exact, parallel, memory-lean, and correct on arbitrary dissimilarities, all at once, with its error confined to exactly the places where single-linkage itself is known to be unreliable. None of the individual pieces — the priority queue, in-place permutation, Borůvka on the GPU, the divide-and-conquer stitch — is new on its own. Bringing them together into one engine that reaches the exact-non-metric-at-scale regime is the contribution, and that regime is unoccupied by the existing fast-VAT literature.

There is work left before this is airtight as a journal result, and I would rather name it than let a committee find it. The timings were taken on a single machine, some of them on a thermally throttled laptop, and they need to be re-run with fixed clocks and error bars across multiple seeds. The GPU story needs a datacenter card with full double-precision throughput to separate the algorithm from the consumer card's penalty. And the non-metric claim, which is the heart of the niche, is so far demonstrated on synthetic non-metric dissimilarities; it needs a real non-coordinate domain — time series under dynamic time warping, or strings under edit distance — to be fully convincing. That last item is a goal for completion in Chapter 7. Finally, I owe the reader a direct head-to-head against eVAT and clusiVAT on identical datasets, which I have not yet run and which is the first comparison a reviewer will want.

---

*Draft — Chapter 3 prose, in the author's voice. Citations in bracketed shorthand pending the consolidated `references.bib`. One figure and two table placeholders marked inline. Source outline in `../chapters/03-scalable-structure-discovery-pvat.md`.*
