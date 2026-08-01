# Chapter 3 — Scalable Structure Discovery: pVAT

## 3.1 Introduction

The first step in the pipeline is to look at the data before modeling it, and to do that at the scale the data actually arrives in. I introduced VAT in Chapter 2 as the tool I use for this: it takes a dissimilarity matrix, reorders it so that similar points sit together, and shows me the cluster structure as dark blocks along the diagonal. It is exactly the right tool, with one problem — in its textbook form it is far too slow and far too memory-hungry to run on the datasets I care about. A 58,000-point matrix is already out of reach for the classical algorithm, and I want to go well past that.

This chapter presents pVAT, an accelerated VAT and iVAT engine, and the collection of implementation ideas that make it work. The contributions are: a reordering that replaces the classical cubic inner loop, improved in two stages — first to $O(N^2 \log N)$ with a priority queue (published), then to $O(N^2)$ with a compact-active-set formulation that removes the heap entirely (unpublished), both detailed in §3.3.1; an in-place memory scheme that holds the whole computation in a single matrix; a GPU implementation of the underlying minimum spanning tree that is bit-identical to the serial result; a divide-and-conquer scheme that splits a large problem and stitches the pieces back together at bounded cost; and, importantly, correctness on an *arbitrary* dissimilarity matrix, including non-metric ones. Together these move the feasible problem size from a few thousand points to well over a hundred thousand. A secondary result, which fell out of the same machinery, is that the VAT ordering makes a good hot-start for the Traveling Salesman Problem; I treat that briefly at the end.

## 3.2 Background and Prior Art

Recall the one fact that everything here rests on: the VAT ordering depends only on the minimum spanning tree of the points. VAT grows an MST with a modified Prim's algorithm, and the reordering is just the order in which points are added. Any method that builds the same MST — serial Prim, parallel Borůvka, or a GPU kernel — produces the same ordering, bit for bit. So "make VAT fast" is really "make the MST fast," and I am free to choose whichever MST construction suits the hardware.

There is already work on fast VAT, and I want to place pVAT against it plainly, because these are the comparisons a reviewer asks for first. **clusiVAT** [Kumar et al. 2016] samples the data and is therefore approximate — it is fast, but it is not the exact VAT ordering. The **parallel edge-based GPU VAT** of Meng and Yuan (2018), sometimes called eVAT, already puts an exact VAT on a GPU, so I cannot claim to be the first to do that, and I do not. **Fast-VAT** [Avinash and Lachheb 2025] is concurrent CPU work in the same spirit, accelerating VAT with compiled kernels and reporting speedups of up to 50×; it addresses VAT only, not iVAT, and it appeared alongside this work rather than before it. **Deshpande and Kumar (2024)** is the closest prior art and deserves more than a passing mention. They attack the ordering step itself — BB-VAT, kdT-VAT and TkdT-VAT use bounding-box and k-d-tree nearest-neighbour search to build the VAT spanning tree, aiming *below* $O(N^2)$ by exploiting geometry — and their MST-iVAT computes the iVAT reordered index without materializing the full distance matrix. That is a stronger route to both goals than anything I do: geometry beats a better constant factor. Their constraint is the one that matters for my purposes: those methods need Euclidean coordinates, so they cannot run on a precomputed or non-metric dissimilarity matrix at all. That is the seam I occupy, and it is a narrower seam than I would have claimed before reading them.

The regime none of them occupy is the one I target: *exact* VAT **and iVAT**, on a *large, arbitrary, possibly non-metric* dissimilarity matrix, computed with modest memory on ordinary hardware. That is not a corner case — sequence data under dynamic time warping, strings under edit distance, and graphs under a kernel dissimilarity all land there, and none of them have coordinates. I should be candid that the argument for this niche is currently structural rather than empirical: I can show that the competing methods cannot run in this regime, and that mine does, but §3.4 reports that on synthetic non-metric matrices rather than on genuinely coordinate-free data. Closing that is Goal G2.

## 3.3 Methodology

### 3.3.1 The reorder, in two stages

The reorder went through two distinct improvements, and I present them as two stages because that is how the work actually happened and because they are separate results. The first replaced the implementation I found in the literature and is the published contribution. The second improved on my own first stage and is in preparation as a separate paper.

**Where the literature starts.** The classical VAT reorder is slow for a mundane reason. At each step it needs the smallest dissimilarity between the tree and any unchosen point, and the usual implementation finds it by re-scanning, for every unchosen point, its distance to every point already in the tree. That is an $O(N)$ search per candidate, across $N$ candidates, across $N$ steps: $O(N^3)$. This is not a strawman — it is what the reference implementations do, and it is the reason VAT is treated as a method for a few thousand points.

**Stage one: the priority queue.** $O(N^3) \rightarrow O(N^2 \log N)$. The wasted work in the classical version is that it recomputes, from scratch, distances it already knew. Instead I keep a priority queue of candidate edges: when a vertex joins the tree I relax its row once, pushing any improved candidate onto the heap, and the next vertex to add is whatever the heap hands back. Each accepted vertex costs one $O(N)$ relaxation plus heap operations, and the $\log$ factor comes from the heap — with lazy deletion the queue can hold $O(N^2)$ entries, so each push and pop is $O(\log N)$. This is the result published at NAFIPS, and it is what took VAT from a few thousand points to the tens of thousands.

**Stage two: the compact active set.** $O(N^2 \log N) \rightarrow O(N^2)$. Having removed the redundant scanning, the remaining overhead is the heap itself, and the insight is that the heap is not needed at all. The reorder does not require a fully ordered queue of candidates — it only ever asks for the current minimum, once per round. So instead of maintaining a heap, I keep the unvisited vertices packed into the first $m$ slots of a small set of parallel arrays holding each candidate's current best key and parent. Two things then fall out:

- Removing a vertex is an $O(1)$ swap with the last active slot, so round $r$ scans only $m = N - r$ entries rather than all $N$. The work shrinks as the tree grows.
- The relaxation and the minimum-selection can be *fused into a single pass*. Walking the active slots once, I update each candidate's key against the newly added row and track the running minimum in the same loop. The round costs one pass, not two, and the sequence of operations is what makes this possible — the argmin for the next round is available as a by-product of the relaxation for this one.

Summing $N - r$ over all rounds gives $N^2/2$ comparisons, with $O(N)$ workspace beyond the matrix itself and no heap allocation at all. The log factor is gone.

I want to be precise about what is and is not novel in stage two, because the ingredients are old and I would rather scope the claim myself than have a reviewer do it for me. Compact active-set dense Prim is classical. Maintaining best-distances-to-tree, removing by swap-with-last, and fusing the relaxation with the selection are what a competently written dense Prim looks like, and dense Prim has been $O(N^2)$ since 1957. Stage two did not discover a new bound; it reached a bound that was available all along and that this literature had not been using.

So what remains is a claim about the literature rather than about the algorithm, and a prior-art search has since narrowed it considerably — usefully, because my first version of this argument was too broad.

What I cannot claim is that the VAT literature is confused about *time*. The Kumar–Bezdek survey states $O(N^2)$ for VAT in four separate places, and correctly credits Havens and Bezdek with taking iVAT from $O(N^3)$ to $O(N^2)$. The stated complexity is right. What is wrong is the *implementations*: the widely used ones — the R **seriation** package, Python's **pyclustertend**, and the reference code accompanying Fast-VAT (2025) — do the cubic re-scan while citing the $O(N^2)$ figure. That gap between the stated bound and the shipped code is real, and it is what stage one actually improved on.

What I can claim, and what survives the search, concerns *space*. The literature repeatedly asserts that the $O(N^2)$ memory footprint is inherent to using Prim: Fast-VAT states an "$O(n^2)$ space complexity for storing $R$"; Kumar's thesis states that constructing the spanning tree "requires $O(n^2)$ time and space to store all the edges"; Deshpande and Kumar's own motivation is that existing methods have "$O(N^2)$ time and space complexity as they use Prim's algorithm." That is incorrect. Array-based dense Prim needs $O(N)$ working memory and never materializes an edge list, which is exactly what stage two exploits and what lets it compose with the on-demand distance computation of §3.3.2. A misconception that motivated a 2024 *Information Sciences* paper is worth correcting in print.

I should be equally clear about how thin that leaves the contribution. Bezdek's own group published *"Is VAT really single linkage in disguise?"* in 2009, printing Prim's algorithm and the VAT ordering side by side; Müllner (2011) then gives the $O(N)$-memory argument for single-linkage. So a reviewer can assemble my correction from two existing papers without much effort, and Deshpande and Kumar have already *solved* the memory problem by a different and more powerful route. The honest position is that stage two is a simpler fix to a problem someone else has already fixed, plus a correction to a stated-but-wrong space bound. That is a modest contribution, and §9.3 scopes the planned note accordingly rather than pretending otherwise.

**Both stages remain in the code, deliberately.** The priority-queue version is the portable path, in pure Python and Numba; the compact-active-set version is the compiled Cython kernel and is preferred whenever the extension is built. They produce bit-identical orderings, which is also how each validates the other. Both are cited by permalink in §3.4 so the complexity claims can be checked against the source.

**A note on the name.** I originally called this *mergeVAT*, after a separate idea I was chasing at the time — an attempt to structure the reordering as something like a two-dimensional merge sort. That idea did not pan out, but the name stuck to the code. Dr. Vladik Kreinovich pointed out that what the method actually did was far closer to a heap sort: it was a priority-queue algorithm. He was right, and I renamed it **pVAT** accordingly. With stage two the acronym outgrew its original reading, since the fast path contains no priority queue at all.

That renaming problem is now worse than a matter of taste, because **the name is taken**. Parveen and Sreevalsan-Nair published a method called **pVAT** in 2013 — *parallel* VAT, on a GPU, which likewise swaps the MST algorithm used for the ordering (Prim for Borůvka). So my "pVAT" collides with a prior method that is close enough in subject to cause genuine confusion, and reading the *p* as *parallel* or *performant* collides with it harder rather than less. I need a different name, and I would rather find that out now than in a review. Resolving it is an open item; the method is referred to as pVAT throughout this draft only because renaming mid-proposal would be more confusing than flagging it.

> **Publication status.** Stage two is not yet published. What a write-up could and could not claim is set out in §9.3, after a prior-art search substantially narrowed it.

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

One more result came out of this, and I include it because it is a neat consequence rather than a central claim. The MST gives a well-known bound on the optimal tour, $T_{best} \leq 2\,T_{MST}$, and the VAT ordering is essentially a walk of that tree. So visiting the VAT-ordered points in sequence gives a tour that is provably within a factor of two of optimal, computed almost for free. It makes a reasonable warm start for a stochastic TSP solver. I will be candid about the limits, though: VAT's raw closed *tour* is actually a poor starting point for the strongest solvers — Lin–Kernighan and LKH are largely insensitive to where they start — and a shorter tour does not imply a better clustering. The honest verdict is that this is a useful engineering connection, not a new optimization result, which is why it sits at the end of the chapter and not the front.

## 3.4 Results

*Hardware: 32-core Intel CPU, 64 GB RAM, laptop RTX 4080 (12 GB). Every result labeled "exact" is bit-identical to the serial VAT reference.*

> **Reproduction.** Table 3.1 regenerates from two generators under `reproduce/tables/`: `table_3_1_pvat_scaling.py` times the exact pVAT reorder against a self-contained classical $O(N^3)$ reference across a configurable grid of $N$, and `table_3_1_reorder_three_arm.py` separates the three complexity regimes and verifies every arm's ordering is bit-identical. Both are multi-seed and emit Markdown and CSV. Cells marked *pending* are sizes not yet swept. The remaining tables run in the `tribble-cluster` submodule rather than the table harness: Table 3.4 from `experiments/adversarial_eval.py`, Table 3.5 from `experiments/principled_stitch.py`, and Table 3.6 from `experiments/hardening_eval.py`, each writing a findings file under `experiments/findings/`. Tables 3.2 and 3.3 have no generator yet — the GPU rows need a device host. Per-cell provenance is tracked in `reproduce/PROVENANCE_MAP.md`, which also records that the 4,096-point pair below is an external measurement not yet reproduced in-repo.
>
> **TODO — repeatable performance (board-wide standard):** the numbers in this section are single-machine, some taken on a thermally throttled laptop. Before any of them are cited as scalability *or* stability results they must be reproduced under a fixed protocol — pinned clocks/thermals, multiple seeds, reported error bars, and a datacenter GPU with full-rate FP64. This same standard applies to every performance/scaling claim in the dissertation (Ch 5, Ch 6). Tracked as Goal G4 in Chapter 7.

**Scaling.** On a 4,096-point problem the classical cubic implementation takes 124 seconds and pVAT takes 2.56 seconds — a measured factor of about 48. The advantage grows with $N$, since the two differ by a factor of roughly $N/\log_2 N$; at $N = 135{,}000$ that ratio is about $135{,}000 / 17 \approx 8{,}000$, which is the difference between running the analysis over lunch and running it interactively.

Two cautions about that pair of numbers, both of which I would rather raise than have raised for me. First, the 48× is a measurement and the 8,000× is not: it is an *asymptotic projection*, because the classical method cannot be run at 135,000 points on this hardware to be timed against, and a projection carries the assumption that constant factors stay put. Second, as §3.3.1 explained, both figures are against the cubic implementations that exist in this literature, not against a tuned $O(N^2)$ dense Prim, which would narrow the gap considerably.

So the claim I actually want to rest on is neither ratio. It is that the feasible problem size moves from roughly 5,000 points to over 130,000 on ordinary hardware — the NASA shuttle set at 58,000 points orders in about a minute, which is where the paper's title comes from — and that this comes from the memory scheme as much as from the reorder. A speedup ratio is a claim about a baseline; a feasible problem size is a claim about what you can actually study.

**Table 3.1 — Reorder time, classical VAT vs. pVAT.** Measured rows are marked; the largest sizes are pVAT-only because the classical method is infeasible there, and the speedup at those sizes is an asymptotic projection rather than a measurement.

| N (points) | classical VAT (cubic) | pVAT | speedup | basis |
|---:|---:|---:|---:|---|
| 4,096 | 124 s | 2.56 s | ~48× | **measured** |
| 58,000 | infeasible on this hardware | ~60 s | — | pVAT **measured** |
| 135,000 | infeasible | *(runs)* | ~8,000× | *projection* from $N/\log_2 N$ |
| stage two, $O(N^2)$ compact active set | — | *(is the fast path)* | ~$N$ vs cubic | **shipped kernel** |

The intermediate grid of $N$ is swept by the harness rather than tabulated here. Every speedup above is measured against the cubic implementation found in the literature. Per §3.3.1, stage one (priority queue) gains a factor of about $N/\log_2 N$ over it and stage two (compact active set) a factor of about $N$; the table's measured rows are from the compiled stage-two path.

**Implementation, for verification.** The two stages are checkable against the source rather than taken on trust. The stage-two $O(N^2)$ compact-active-set kernel is `_prim_mst_kernel_64` in [`src/tribbleclustering/pcvat.pyx`](https://github.com/fundthmcalculus/clustering/blob/c71171e/src/tribbleclustering/pcvat.pyx#L22-L113) (lines 22–113; the `float32` twin is `_prim_mst_kernel_32` at line 392), whose fused relax-and-select inner loop is the `for i in range(m)` block. The stage-one $O(N^2 \log N)$ priority-queue implementation is `vat_prim_mst` in [`src/tribbleclustering/pvat.py`](https://github.com/fundthmcalculus/clustering/blob/c71171e/src/tribbleclustering/pvat.py#L141-L211) (lines 141–211). The compiled stage-two path is preferred at import time, with stage one as the portable fallback; the two agree bit-for-bit, which is how each validates the other.

**Memory.** The in-place scheme changes what is possible rather than merely what is fast.

**Table 3.2 — Memory footprint and the largest feasible problem (64 GB host).**

| Quantity | Classical (2–3 matrices) | pVAT (in place) |
|---|---:|---:|
| iVAT at N = 64,000, float64 | ≈ 98 GB — does not run | **33 GB, 25 s** |
| Largest feasible N at 64 GB | ≈ 52,000 | **≈ 89,000** |
| Permutation buffers | 2 | **1** |

**GPU.** The device results are mixed in an instructive way, and the losses are as informative as the wins.

**Table 3.3 — GPU speedups over the 32-core CPU** (laptop RTX 4080, 12 GB — consumer double precision at a fraction of single-precision throughput).

| Kernel | Speedup | Conditions | Exactness |
|---|---:|---|---|
| Borůvka MST (device-resident) | ≈ 5× at N = 32,000, growing with N | any | VAT order match 1.0 |
| Full VAT front end (device-resident) | ≈ 4.8–6.6× end to end | any | bit-identical |
| Fuzzy C-Means | **30–56×** at N = 50,000–500,000 | any | same fixed point, > 99% label agreement |
| Pairwise distances | 1.3–2.5× | **only** high dimension + float32 | exact |
| Pairwise distances | **< 1× (loses)** | low dimension or float64 | exact |

The pairwise-distance row is the one worth dwelling on. On this hardware the GPU is slower than the CPU for low-dimensional or double-precision distance computation, because a consumer card's double-precision throughput is a small fraction of its single-precision throughput. That is a property of the card and not of the algorithm, which is exactly why the repeatability protocol calls for a datacenter GPU: on a card with full-rate FP64 this row would likely flip, and I would rather flag that as an untested prediction than quietly present laptop numbers as the method's ceiling.

**Clustering quality.** Because pVAT is exact single-linkage, it inherits single-linkage's strengths and weaknesses honestly. On non-convex data where k-means fails — two moons, concentric circles — pVAT and the stitched version both reach an adjusted Rand index of 1.00, against 0.27 and 0.00 for k-means. On bridged or touching-anisotropic clusters, where a single chain of points connects two real groups, pVAT scores 0.00, exactly as single-linkage does; I do not paper over this, and it is precisely the failure mode that Chapter 5's metric-learning and persistence work is meant to repair.

**Table 3.4 — Adversarial clustering quality (adjusted Rand index).** pVAT is exact single-linkage, so it wins where k-means fails and inherits single-linkage's failures honestly.

| Dataset | k-means | single-linkage | exact pVAT | naive block | principled stitch |
|---|---:|---:|---:|---:|---:|
| two_moons | 0.27 | 1.00 | 1.00 | 0.39 | **1.00** |
| circles | 0.00 | 1.00 | 1.00 | 0.00 | **1.00** |
| aniso | 0.61 | 0.00 | 0.00 | 0.30 | 0.00 |
| bridged | **1.00** | 0.00 | 0.00 | 0.08 | 0.00 |

**The stitch.** Ablating the divide-and-conquer stitch on two moons across a grid of partitions and sizes: the light stitch (random representatives, one cross-edge) averages ARI 0.47; farthest-point representatives alone are worse still at 0.37, and top cross-edges alone reach only 0.61; the principled combination of both reaches a mean ARI of 1.00 across every partition tested, at bounded cost. Both ingredients are required together, and neither alone gets close.

**Table 3.5 — Stitch ablation on two moons, over a grid of partitions and sizes.**

| Stitch variant | mean ARI | min ARI | fraction ≥ 0.9 |
|---|---:|---:|---:|
| light (random rep, 1 cross-edge) | 0.47 | 0.00 | 0.44 |
| top-m cross-edges only (m = 8) | 0.61 | 0.00 | 0.60 |
| farthest-point reps only | 0.37 | 0.00 | 0.32 |
| **principled (fps + top-m = 8)** | **1.00** | **1.00** | **1.00** |

**Non-metric robustness.** This is the point of the whole exercise, so I test it directly rather than asserting it.

**Table 3.6 — Agreement with exact single-linkage under non-metric dissimilarities.** Agreement is the fraction of the ordering reproduced identically; 1.0 means the divide-and-conquer result is indistinguishable from running exact VAT on the whole matrix.

| Dissimilarity | Metric? | Triangle-inequality violations | Agreement with exact |
|---|:--:|---:|---:|
| Euclidean (control) | yes | 0% | 1.0 |
| Fractional Minkowski, $p = 0.5$ | **no** | ≈ 14% of triples | **1.0** |
| Cosine | no (not a metric) | — | **1.0** |
| $k$-nearest-neighbour geodesic | no | — | **1.0** |
| Real non-coordinate domains (DTW, edit distance, graph kernel) | no | — | *pending* |

The fractional-Minkowski row is the sharp test: it violates the triangle inequality on roughly one triple in seven, and the method still reproduces exact single-linkage. That is the evidence that pVAT does not quietly assume a metric somewhere in its internals, which is the precondition for the regime I claimed in §3.2.

The last row is the honest gap, and it is the one that matters most for the claim. Every dissimilarity above is a synthetic non-metric constructed from coordinate data — I made the matrix non-metric on purpose. What I have not yet done is run the method on data that has *no coordinates in the first place*: time series under dynamic time warping, strings under edit distance, graphs under a kernel. Those are the domains the niche argument is built on, and until the method is demonstrated there, §3.2's regime claim rests on a proxy. This is Goal G2 in Chapter 7 and it is shared with Chapter 5.

## 3.5 Discussion and Contributions

What the composition buys is an engine that is exact, parallel, memory-lean, and correct on arbitrary dissimilarities, all at once, with its error confined to exactly the places where single-linkage itself is known to be unreliable. None of the individual pieces — the priority queue, in-place permutation, Borůvka on the GPU, the divide-and-conquer stitch — is new on its own. Bringing them together into one engine that reaches the exact-non-metric-at-scale regime is the contribution, and that regime is unoccupied by the existing fast-VAT literature.

There is work left before this is airtight as a journal result, and I would rather name it than let a committee find it. The timings were taken on a single machine, some of them on a thermally throttled laptop, and they need to be re-run with fixed clocks and error bars across multiple seeds. The GPU story needs a datacenter card with full double-precision throughput to separate the algorithm from the consumer card's penalty. And the non-metric claim, which is the heart of the niche, is so far demonstrated on synthetic non-metric dissimilarities; it needs a real non-coordinate domain — time series under dynamic time warping, or strings under edit distance — to be fully convincing. That last item is a goal for completion in Chapter 7. Finally, I owe the reader a direct head-to-head against eVAT and clusiVAT on identical datasets, which I have not yet run and which is the first comparison a reviewer will want.

---

*Draft — Chapter 3 prose, in the author's voice. Citations in bracketed shorthand pending the consolidated `references.bib`. Six tables (3.1–3.6) and one figure placeholder inline. Source outline in `../chapters/03-scalable-structure-discovery-pvat.md`; open items in `../ACTION_ITEMS.md`.*
