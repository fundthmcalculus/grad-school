# Prior art, and what is actually novel here

Assembled from a literature search plus direct verification of the papers that matter. Where a
claim could not be verified from a primary source it says so — the point of this file is to be
trusted later, and an unmarked guess would destroy that.

---

## The headline

**No fuzzy inference system has been placed inside the inner loop of a Lin-Kernighan search.**
An arXiv full-text query for `all:"fuzzy" AND all:"Lin-Kernighan"` returns **zero results**, and
no phrasing tried surfaced one. That gap is real.

**But the conceptual slot — a learned, per-decision controller replacing LK's fixed traversal
rules — is occupied, by strong work.** Novelty of *substrate* is not novelty of *idea*, and the
empirical bar in that slot is high.

---

## 1. The papers this work lives or dies against

| paper | what it does | why it matters here |
|---|---|---|
| **VSR-LKH** — Zheng, He, Zhou, Jin & Li, AAAI 2021, [arXiv:2012.04461](https://arxiv.org/abs/2012.04461) | Q-learning / Sarsa / Monte Carlo "replaces the inflexible traversal operation in LKH and lets the program learn to make choices at each search step". 111 TSPLIB instances to n = 85 900, 10 runs each. | The identical conceptual slot. First reviewer question will be "why fuzzy rather than Q-learning". |
| **NeuroLKH** — Xin, Song, Cao & Zhang, NeurIPS 2021, [arXiv:2110.07983](https://arxiv.org/abs/2110.07983) | Sparse Graph Network produces LKH's edge candidate set and node penalties. Beats LKH at every matched time limit. | Sets the empirical bar: 0.05–0.09% where we are at 0.3–1.3%. See BENCHMARKS.md. |
| **Joshi, Cappart, Rousseau, Laurent & Bresson**, CP 2021, [arXiv:2006.07054](https://arxiv.org/abs/2006.07054) | Neural TSP solvers match classical methods only at trivial sizes and fail to generalize. | The project's **best friend** — it is the motivation for scale-free antecedents. Cite as motivation, not as related work. |
| **Marques & Gomide (2011)**, "Parameter control of metaheuristics with genetic fuzzy systems", *Evolutionary Intelligence* 4:183–202, [doi](https://doi.org/10.1007/s12065-011-0059-y) | A **genetic fuzzy system controlling tabu search memory** and GA rates. | The architectural precedent. Not citing it looks like not knowing the field. |
| **Herrera & Lozano (2003)**, "Fuzzy adaptive genetic algorithms: design, taxonomy and future directions", *Soft Computing* 7(8):545–562 | The canonical taxonomy of fuzzy control of metaheuristic parameters. | Establishes that fuzzy parameter control is itself not new. |
| **Applegate, Cook & Rohe (2003)**, "Chained Lin-Kernighan for Large TSPs", *INFORMS JoC* 15(1):82–92 | Don't-look bits; kicks restricted to the region around breakpoints. | The crisp, unlearned version of *both* `EFFORT` and the aimed kick. The baseline both must beat. |
| **Hottung & Tierney (2022)**, "Neural large neighborhood search for routing problems", *Artificial Intelligence* | Learned *destroy* operators choose which region to tear up. | The modern learned rival to score-directed kick placement. |

**Reframing that follows from Applegate et al.:** `EFFORT` is best described as *don't-look bits
generalized from a boolean to a graded, learned, multi-output allocation*. That is honest, and
immediately legible to the TSP community. Claiming per-city effort allocation as new is not.

---

## 2. XKICK — run down, and it does not block us

Flagged by the search as the closest possible prior art for aimed perturbation, since FINDINGS §6.3's
result is the one thing that survived. Chased to a conclusion:

**Provenance.** "Xkick, an intelligent kick perturbation for the traveling salesman problem",
Alfredo Garcia W, May/June 2017. Self-posted on
[ResearchGate](https://www.researchgate.net/publication/317370678_Xkick_an_intelligent_kick_perturbation_for_the_traveling_salesman_problem)
and [Academia.edu](https://www.academia.edu/33405441/XKICK_AN_INTELLIGENT_KICK_PERTURBATION_FOR_THE_TRAVELING_SALESMAN_PROBLEM).
Implementation at [sourceforge.net/projects/jalicanto](https://sourceforge.net/projects/jalicanto/)
(Java, "time-based TSP solver, iterated local search, implements xkick perturbation").

* **No DBLP entry**; no journal or conference venue found by any search.
* ResearchGate **resolves no citations** for it.
* So: grey literature, unrefereed, uncited. Real, and worth citing — but not a blocking
  publication.

**Mechanism, and why it is not ours.** From the abstract (retrieved consistently three times):

> "a new kick perturbation based on the extraction of information from **the comparison of
> existent tours** that leads to the regions in which the kick is more probable to be
> effective… the new kick is **not performed by a double-bridge** (4-opt), but a more complex
> change in the tour."

Two substantive differences:

| | XKICK | this project |
|---|---|---|
| where the signal comes from | comparing **multiple tours** — backbone-style disagreement between solutions | a **single current tour**, scored per city from local geometry by a fitted rule base |
| what it needs | a population of tours to compare | one tour |
| the move | explicitly **not** a double-bridge; "a more complex change" | the standard double-bridge, unchanged; only its *location* is chosen |

The overlap is the **goal** — kick where it is more likely to pay — not the method. That goal
is in any case older than XKICK: Lourenço, Martin & Stützle's ILS chapter already documents
biasing double-bridge selection, and Applegate/Cook/Rohe restrict kicks geometrically.

**Verification caveat.** ResearchGate, Academia.edu and SourceForge all return HTTP 403 to
automated fetching, so the above rests on the abstract as returned by search and on the
SourceForge project description, **not on the full text**. Before any novelty claim goes into a
paper, someone should open the PDF in a browser and confirm the mechanism. One search result
asserted the work appears in "peer-reviewed papers" — that is **unsupported** and contradicted
by the absence of any venue or citation; do not repeat it.

---

## 3. Verdict, component by component

**Genuinely new — nothing found:**

* A fuzzy inference system anywhere in the LK/LKH inner loop.
* **`CHAIN`** — a learned per-level continue/cut decision from the gain trajectory. Variable-depth
  termination in LK is governed by the fixed positive-gain rule plus a depth cap; VSR-LKH learns
  *which edge*, not *whether to go deeper*. The most defensible single contribution.
* Reusing **one** effort model both to allocate inner-loop parameters and to aim the kick. The
  coupling, not either half.

**Well-trodden — no novelty available:**

* Fuzzy control of metaheuristic parameters (a subfield with surveys).
* GA-fitted membership functions *and* consequents — textbook genetic fuzzy systems (Cordón,
  Herrera, Hoffmann & Magdalena, *Genetic Fuzzy Systems*, World Scientific 2001).
* ML-guided LKH.
* Per-instance algorithm configuration (ParamILS, SMAC, irace) — though note this project
  *sidesteps* rather than competes with it: scale-free antecedents mean the rule base is fitted
  once and needs no per-instance tuning.
* Surrogate models of runtime standing in for noisy wall clock (Hutter, Xu, Hoos &
  Leyton-Brown, "Algorithm Runtime Prediction", *AIJ* 206, [arXiv:1211.0906](https://arxiv.org/abs/1211.0906)).

**New in combination, or minor:**

* The NNLS work-counter cost proxy as a *deterministic* GA fitness — the idea is standard, this
  instantiation was not found published. One paragraph with a "we are not aware of" hedge.
* The AUC screen reframing "how much effort does this city deserve" (no ground truth) as "will
  searching this city pay off" (observable by instrumentation). Methodological, not scientific.
* Lookup-table membership banks making membership *shape* fitted data. **Under-researched** —
  check the embedded/hardware fuzzy control literature before claiming anything.

---

## 4. The one claim the measurements actually support

Not "the FIS beats LKH". The measured, bounded claim is:

> On TSPLIB instances that LKH itself fails to solve to optimality in 10/10 runs, an
> `EFFORT`-aimed double-bridge perturbation reaches tours that uniform kicking cannot, on the
> instances where uniform kicking has plateaued — up to 5.3x better at matched budget — and on
> `d2103` reaches a shorter tour than any measured LKH configuration below 39 s.

Its supports are: no GPU, no labelled training set (NeuroLKH needs ≈780 000 Concorde-solved
instances and ~4 GPU-days), no per-instance configuration, and no warm-up — VSR-LKH's own
appendix concedes that "in the beginning of the iterations, LKH can yield better solutions than
the reinforced algorithms" because of RL's trial-and-error. A rule base fitted offline has no
such cost.

That is a narrow, defensible paper. It is not the paper the earlier drafts of FINDINGS.md were
writing.
