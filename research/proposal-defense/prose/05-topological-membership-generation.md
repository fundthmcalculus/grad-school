# Chapter 5 — Membership Functions from Topological Structure (Proposed)

*This is proposed work. The construction and the preliminary results below are real and reproducible today; the extensions in Section 5.5 are what I propose to complete for the dissertation.*

## 5.1 Introduction

Chapter 4 builds a fuzzy model fast, but it makes two assumptions that are not always available. It assumes the data has coordinates, so that a Gaussian can be fit to it, and it assumes the structure is roughly blob-shaped, so that a Gaussian is the right thing to fit. Plenty of real data satisfies neither. Sequences compared by dynamic time warping, strings compared by edit distance, and graphs compared by a kernel come to me as a dissimilarity matrix and nothing else — no coordinates to average, no natural notion of a Gaussian. And plenty of structure is not blob-shaped: two concentric rings are a textbook case where every centroid method fails, because the mean of a ring sits in the middle of the hole.

The tool I already trust for this setting is VAT, from Chapter 3, which works on exactly a dissimilarity matrix and finds structure of any shape because it is built on single-linkage connectivity rather than on centroids. But VAT, and the whole VAT family, stops at *assessment* — it tells me how many clusters there are and roughly where, and then hands off. No one in that literature turns the VAT structure into a fuzzy model. That gap is what this chapter is about: taking the minimax hierarchy that VAT and iVAT produce and reading fuzzy membership functions directly off it — the number of clusters, the number of *scales*, and the membership functions themselves — with no coordinates and no Gaussian assumption. This is the bridge between the clustering half of the dissertation and the fuzzy-modeling half, and it is the piece I consider the most novel.

The contributions I propose are: the minimax transform as the preprocessing step that makes relational fuzzy clustering succeed on non-convex data; a selection rule that treats the number of clusters as an *output* rather than an input, by gating persistence and covering the data; a multi-scale extension that recovers a whole hierarchy of partitions and discovers how many scales are present; and membership functions extracted natively from each piece of the hierarchy, ready to serve as the antecedents of a fuzzy inference system.

## 5.2 Background and Prior Art

I have to be careful and honest here, because the nearest prior work is close. Persistence-based clustering already exists: ToMATo [Chazal et al. 2013] and, most pointedly, the beta-plateau method of Bonis and Oudot (2014, 2018) already derive a cluster count from gaps in a persistence diagram, and there is recent automated work in the same vein [AuToMATo]. I am not going to claim persistence-for-clustering as new. What I claim is different on three specific axes, and I state them plainly so the distinction survives scrutiny. First, my filtration is the *connectivity* (single-linkage, ultrametric) structure, which is density-free, rather than a density mode-seeking filtration. Second, my membership function is a *deterministic* ramp read off the merge heights, not a random-walk hitting probability. Third, and most importantly, my target is not a clustering — it is a set of *membership functions for a fuzzy inference system*: linguistic antecedents, combined by disjunction, that a downstream FIS consumes. Nobody in the persistence literature, and nobody in the VAT family, is producing that object. If a reviewer collapses the first two axes, the third still stands.

The other prior art I lean on is relational: NERFCM [Hathaway and Bezdek 1994], which runs Fuzzy C-Means directly on a dissimilarity matrix, with a beta-spread safeguard for when the matrix is badly non-metric; and ConiVAT [Rathore et al. 2020], which repairs single-linkage's chaining failure with constraint-based metric learning.

## 5.3 Methodology

### 5.3.1 The minimax transform does the heavy lifting

I want to be honest about where the result actually comes from, because it would be easy to over-credit the clever part. The single most important step is the minimax (iVAT) transform from Chapter 2 — replacing each dissimilarity with the bottleneck value along the best path between two points. On concentric rings, ordinary relational Fuzzy C-Means run on the raw dissimilarity matrix scores an adjusted Rand index of about 0.02, which is to say it fails completely. Run the *same* algorithm on the minimax-transformed matrix and it scores 1.00. The transform, not the selection machinery I describe next, is what carries that result, and I say so up front. What the rest of this chapter adds is how to choose the clusters and turn them into membership functions without being told how many there are.

**[FIGURE 5.1 — placeholder]** *Why the transform is the load-bearing step: concentric rings shown as (a) the raw dissimilarity matrix, where the two rings are indistinguishable, and (b) the minimax-transformed matrix $D^*$, where they separate into clean blocks — with the relational-FCM ARI (0.02 → 1.00) annotated on each.*
`![minimax-transform](fig/05-minimax-transform.png)`

### 5.3.2 Selection as gated set-cover, with k as an output

Given the minimax hierarchy, the usual question is "where do I cut it?" — which presupposes I know how many clusters to look for. I reframe it.

A word on terminology first. By a **block** I mean a candidate cluster in the hierarchy: a set of points that merges into one connected group at some threshold and later merges into something larger. It is a node of the single-linkage dendrogram, and it is also, not coincidentally, one of the dark squares along the diagonal of the reordered VAT image from Chapter 2 — which is where the name comes from. Every block has a birth height (where it forms) and a death height (where it is absorbed), and the hierarchy contains many more blocks than there are real clusters; the job of selection is to decide which ones to believe.

Each block has a persistence — its death height minus its birth height, from Chapter 2 — and I admit only the blocks whose persistence is a statistical outlier, above the median by a robust multiple of the median absolute deviation. Among the admitted blocks I then greedily cover the data, taking at each step the block that covers the most still-uncovered points. The number of clusters falls out of this process; it is an output, not an input. The reason to frame it as coverage rather than as "pick exactly k" is asymmetry: under a fixed disjunction, admitting one too many blocks is cheap, while failing to cover part of the data is expensive, so a set-cover is the right shape for the objective.

I should say now, rather than let it surface later as a surprise, that this gate is not uniformly the best selector available. It is deliberately conservative: it declines to assert structure it cannot see clearly, which is the behavior I want on noise and the wrong behavior on a chained "bridge" between two real clusters, where a more aggressive selector wins outright. §5.4 reports that comparison in full, including the case my gate loses. The conservatism is a defensible choice for this pipeline — a fuzzy model built on spurious antecedents is worse than one that abstains — but it is a choice, not a free lunch.

### 5.3.3 Multi-scale persistence: recovering the whole hierarchy

The most interesting case, and the one I want to be the headline of this chapter, is nested structure — clusters within clusters, where each level is real. A single flat cut cannot represent that; it stops at whichever granularity covers the data first. The idea that unlocks it is the density observation from Chapter 2: birth height in single-linkage is an inverse proxy for local density, so structures at different densities are born at different heights. I look for gaps along the log-birth axis, use them to separate the hierarchy into density *bands*, and run the same gated set-cover *within each band*. The result is not a single partition but a stack of them — a fuzzy hierarchy — and the number of scales is itself discovered rather than specified.

**[FIGURE 5.2 — placeholder]** *Band discovery on the three-level hierarchy: the log-birth-height spectrum with the gaps that separate the density bands marked, and beside it the three recovered partitions at granularities 8, 4, and 2 — the mechanism and its result in one figure. This is the chapter's key figure.*
`![band-discovery](fig/05-band-discovery.png)`

I want to frame this honestly, because there is a strawman available that I refuse to set up. It would be tempting to claim that this beats a flat method on varying-density data, but that is not true and I have the experiment to show it: a flat cover already holds up across a very wide range of densities (see the falsification result below). The genuine open problem is not varying density at a single level; it is *nested* structure across levels, which a flat method structurally cannot recover. The multi-scale method is a strict generalization — on single-scale data the band discovery finds one band and the whole thing reduces exactly to the flat selector, and on pure noise it finds no bands at all. That reduction is what makes it defensible rather than a rival heuristic.

### 5.3.4 Membership functions, read off the hierarchy

Each selected block carries its own membership function, with no medoid and no Gaussian fit. The natural one is a persistence ramp. Let $d_B(x)$ be the minimax distance from the point $x$ to the block $B$ — that is, the bottleneck height at which $x$ would join $B$ in the hierarchy, which is the same quantity the transform of §5.3.1 already computes. Membership then falls linearly from one, for a point that joins as early as the block itself was born, to zero, for a point that only joins when the block dies:

$$ \mu_B(x) = \mathrm{clip}\!\left(\frac{\text{death}_B - d_B(x)}{\text{death}_B - \text{birth}_B},\, 0,\, 1\right). $$

Every term is a merge height the hierarchy already gives me, so the membership function is *read off* the structure rather than fitted to it. That is the property that makes this work without coordinates: there is no mean to compute and no shape to assume. The denominator is exactly the block's persistence, which also means the gate of §5.3.2 and the membership function are two uses of the same quantity — a long-persistence block is both more likely to be admitted and gentler in its membership falloff.

**[FIGURE 5.3 — placeholder]** *The persistence ramp as a membership function: one selected block's $\mu_B(x)$ plotted against minimax distance, annotated with the block's birth and death heights, next to the resulting fuzzy partition over the dataset. Shows that the membership function is read off the hierarchy, not fitted.*
`![persistence-ramp](fig/05-persistence-ramp.png)`

I have several variants of this — a Ruspini partition-of-unity form where the memberships sum to one exactly, a spread-aware auto-tuned form, and an interpretable feature-space form that works when coordinates *are* available (and, honestly, fails on rings, which is the point of having the others). The blocks are combined into a rule by disjunction, a t-conorm, exactly as the OR of antecedents in Chapter 4. The upshot is that the same minimax hierarchy that told me how many clusters there are also hands me the linguistic antecedents of a fuzzy model, directly.

### 5.3.5 Counting disjuncts topologically

There is one more consequence worth drawing out, because it answers a question Chapter 4 left open. The rules in Chapter 4 are disjunctions — a class is recognized if *this* Gaussian fires or *that* one does — and the number of terms in that OR is a parameter I had to choose. Here it is determined by the data. A class occupying two separated regions of the input space appears in the minimax structure as two connected components at the relevant threshold, so the arity of the disjunction is simply a component count.

The reason this matters is that the same question is ill-posed geometrically. Asking how many convex pieces a ring consists of has no good answer — a ring is one connected object that no finite set of blobs covers naturally, which is exactly why the feature-space variant of §5.3.4 fails on rings. Asking how many connected components it has is well-posed and returns one. So the topological formulation both answers the question and explains why the geometric one could not.

## 5.4 Preliminary Results

> **Reproduction.** The whole pipeline behind this section regenerates from a single deterministic driver, `gated-minimax-selection/run_all.py`, which writes the results and every figure referenced below; it is registered in `reproduce/manifest.py`.
>
> **TODO — repeatable performance (board-wide standard):** the scaling numbers below are single-machine point estimates and must be reproduced under the fixed protocol (see `ACTION_ITEMS.md` §A and Ch 7 Goal G4) before citation.

The results here are on synthetic data with known ground truth, which is both their strength (I know the right answer) and their limitation (Section 5.5).

**The transform.** As above: concentric rings, relational FCM on the raw matrix scores ARI ≈ 0.02; on the minimax-transformed matrix, 1.00. On bridged Gaussians, plain single-linkage scores 0.00 (the chaining failure), and ConiVAT's metric learning repairs it to 1.00. Across a battery of five synthetic sets, the gated set-cover — discovering $k$, with no constraints — matches NERFCM-given-$k$ and ConiVAT at roughly 0.98–1.00, and declines only on uniform noise, which is the correct behavior: there is no structure there to find.

**Table 5.1 — The battery (adjusted Rand index).** The comparison that matters is the last column against the first two: the baselines are *given* $k$, and mine discovers it. Matching a method that was handed the answer is the result.

| Dataset | NERFCM on raw $D$ | NERFCM on $D^*$ (given $k$) | ConiVAT (constrained) | Gated set-cover (**$k$ discovered**) |
|---|---:|---:|---:|---:|
| concentric_rings | 0.02 | **1.00** | 1.00 | **1.00** |
| bridged_gaussians | — | 0.00 (chaining) | **1.00** | 0.00 |
| well_separated | ≈ 1.00 | 1.00 | 1.00 | **≈ 0.98–1.00** |
| varying_density | — | 1.00 | 1.00 | **≈ 0.98** |
| uniform_noise | — | — | — | **abstains (correct)** |

Two cells deserve comment rather than a dash. The bridged-Gaussians row is the failure I flagged in §5.3.2 — my conservative gate declines where ConiVAT's metric learning succeeds — and the uniform-noise row is the one where declining *is* the right answer, which no accuracy-style metric rewards. Cells shown as "—" were not run in that configuration rather than being zero.

**Multi-scale, the headline.** Averaged over *all* ground-truth levels, the multi-scale method lifts nested Gaussians from 0.66 to 1.00, a three-level hierarchy from 0.58 to 1.00, and a density hierarchy from 0.75 to 1.00. On the three-level set it recovers granularities of 8, then 4, then 2 clusters — each band landing on exactly one true level at ARI 1.0 — without ever being told there were three levels.

**Table 5.2 — Multi-scale recovery (adjusted Rand index, averaged over all ground-truth levels).**

| Dataset | flat cover | multi-scale | granularities recovered |
|---|---:|---:|:--:|
| nested_gaussians | 0.66 | **1.00** | *not recorded* |
| three_level_hierarchy | 0.58 | **1.00** | [8, 4, 2] |
| density_hierarchy | 0.75 | **1.00** | *not recorded* |

**The falsification experiment.** To keep myself honest, a flat cover holds ARI ≈ 0.983 across a thirty-fold spread in cluster width. This is the result that says the multi-scale method is not solving a single-level varying-density problem, because there is no such problem to solve; it is solving nesting.

**Scaling.** With the exact $O(N^2)$ minimax transform, the full pipeline runs to 5,000 points in about five seconds, and the multi-scale recovery of [8, 4, 2] is unchanged from 100 points up to 5,000.

**The selection bake-off.** Comparing my persistence-gap gate against beta-plateau and bottleneck-bootstrap, there is no universal winner, and I report that as a finding rather than hide it. My gate fails a deliberately adversarial "bridge" case (ARI 0.001) but is conservative on noise; beta-plateau and bottleneck-bootstrap fix the bridge (0.927 and 0.891) but over-fire on noise, reporting seven clusters where there are none. Bridge-robustness and noise-conservatism turn out to be incompatible for any single fixed threshold, which is itself worth stating.

**Table 5.3 — Selection-method comparison.** No universal winner; the two objectives (bridge-robust vs. noise-conservative) are incompatible for a single fixed threshold.

| Selection method | bridge case (ARI) | noise behavior |
|---|---:|---|
| persistence-gap gate (ours) | 0.001 | conservative — reports no clusters (correct) |
| beta-plateau [Bonis–Oudot] | 0.927 | over-fires — k = 7 where there are none |
| bottleneck-bootstrap [AuToMATo] | 0.891 | over-fires |

**Relational-only data.** On dissimilarity-matrix-only datasets built from trees, the simple cases are already solved by NERFCM, but a genuinely multi-scale relational case leaves both the raw and transformed matrices stuck at ARI ≈ 0.29 — confirming that multi-scale relational structure is the honestly hard, still-open problem.

**What these results do not yet show.** Every number above is a *clustering* score, and the chapter's stated purpose is not clustering — it is generating membership functions for a fuzzy inference system. So the result that would most directly support the claim is missing: I have not yet built a FIS from these membership functions and measured how it predicts. Adjusted Rand index against a known partition is a reasonable proxy, since antecedents that carve up the space correctly are the precondition for a good model, but it is a proxy and I should not let it stand in for the real thing. Closing that gap — wiring the generated membership functions into the Chapter 6 inference system and reporting end-to-end accuracy against the Chapter 4 Gaussian construction on the same data — is the deliverable I consider most important in §5.5, and it is what would turn this chapter from a promising method into a demonstrated one.

## 5.5 Proposed Work

What turns this from a strong preliminary result into a dissertation chapter is the following. The goal labels refer to Chapter 7, where these are scheduled.

First, and most important, the **end-to-end integration** — the deliverable the previous section named as the missing evidence. The whole point is to feed a fuzzy model, so the closing result must be a FIS built from these membership functions, evaluated end to end from a bare dissimilarity matrix to a working, readable rule base, and compared against the Chapter 4 Gaussian construction on data where both can run. Until that exists, the chapter's central claim rests on a proxy. This feeds the capstone of Chapter 7.

Second, **direct, one-pass membership generation (G1).** Right now the pipeline selects clusters and then fits membership functions in two stages; I propose to collapse that into a single pass, in which every block emits its native ramp membership, the disjunction recombines them, and the surviving envelope simply *is* the fuzzy model. The research-interesting piece within that is a soft, kernel-weighted band membership, which I expect to fix the over-segmentation that shows up at small sample sizes.

Third, **real non-coordinate data (G2).** Everything above is synthetic, with known ground truth. The core claim — that this works where there are no coordinates — is only fully convincing on genuine non-metric domains: time series under dynamic time warping, sequences under edit distance, graphs under a kernel. This is the same gap I named in Chapter 3, and it is a shared goal for completion. It is also the setting where the honestly hard case above (multi-scale relational structure at ARI 0.29) has to be confronted rather than reported.

Fourth, the **prior-art head-to-head.** I owe a direct comparison against Bonis–Oudot beta-plateau and AuToMATo on identical data, and a formal literature search to bound the novelty claim. Given how close that prior work is, this is a defensive necessity as much as a scientific one.

Fifth, **band discovery for overlapping scales (G7, stretch).** The current gap heuristic on the log-birth axis assumes the density scales are well separated; when they overlap it is ill-posed. I propose a model-based discovery — a change-point or barcode-stability criterion — to handle that case, and I flag it honestly as the piece most likely to be cut if time runs short.

## 5.6 Discussion and Contributions

The position I am staking out is narrow and, I think, defensible: a density-free, connectivity-based generator of fuzzy membership functions for problems that arrive as a dissimilarity matrix — the missing bridge from the VAT/iVAT structure I can already compute at scale to the fuzzy antecedents I need. Everything the method needs is a merge height the hierarchy already computed, which is what lets it work with no coordinates and no assumed shape.

I have tried to be equally clear about the boundaries. The result is carried by the transform, not by the selection machinery. The method does not apply to single-level varying density, which flat methods already handle, and I ran the falsification experiment rather than let that go untested. My selection gate is deliberately conservative and loses outright on the bridge case to a more aggressive competitor. The band discovery assumes separated scales and is ill-posed when they overlap. The coverage floor can drop a small real cluster. Every result so far is on synthetic data, and every metric so far is a clustering proxy for the fuzzy-model quality I actually care about. And the overlap with Bonis and Oudot is close enough that it has to be actively managed rather than waved away — which is why the distinction rests on three separate axes, the third of which is the one I would defend if pressed.

None of those limitations is fatal, and stating them is not hedging. This is proposed work; the honest measure of it is whether the open problems are the interesting ones, and I think nested structure in relational data and one-pass membership generation are exactly that.

---

*Draft — Chapter 5 prose, in the author's voice; proposed work with preliminary results front-loaded. Citations in bracketed shorthand pending the consolidated `references.bib`. Three tables (5.1–5.3) and three figure placeholders (5.1–5.3) inline. Source outline in `../chapters/05-topological-membership-generation.md`; open items tracked in `../ACTION_ITEMS.md`.*
