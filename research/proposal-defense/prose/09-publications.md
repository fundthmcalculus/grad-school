# Chapter 9 — Publications

This chapter documents the completed and proposed publication record, mapping published, submitted, and in-preparation work to the substantive chapters of this dissertation. It serves two purposes: to establish the completed scholarly contribution, and to lay out the plan for future publication of the remaining chapters.

## 9.1 Published Work — Venues Confirmed

Two peer-reviewed meetings have accepted work derived from Chapter 3 (VAT and TSP initialization):

- **NAFIPS 2025** (Banff, Alberta; July 2025)
- **NAFIPS 2026** (El Paso, TX; March 2026)

Both venues are confirmed. However, the paper metadata—titles, page numbers, DOI assignments, and co-author lists—has not yet been supplied. Additionally, the current state does not resolve whether this work was published as two separate papers or as one combined paper across both venues.

Two manuscripts are cited in the qualification work:
- A paper on VAT utilization for hot-starting the traveling salesman problem, prepared for submission to one of the two NAFIPS venues.
- A paper on the mergeVAT method, demonstrating the consolidation of 58K×58K matrices in 60 seconds, also prepared for one of the two NAFIPS venues.

Until the exact titles, venues, page numbers, DOI assignments, and co-author attributions are provided, this section cannot be finalized. This missing information (checklist item **D2: Author Records**) is also blocking the second pass on the acknowledgements chapter, which currently thanks the committee and key collaborators but cannot accurately cite co-authors until their contributions are recorded here.

## 9.2 Submitted and Under Review

There is currently no work under active review. The next planned submission is the Chapter 5 membership paper to **EUSFLAT 2027** (deadline: February 2027). This section is retained rather than omitted to make clear that the current absence of under-review work is a state statement, not an oversight.

## 9.3 In Preparation

The primary target venue for new submissions is **EUSFLAT 2027** (September 2027), with secondary options including FUZZ-IEEE, Fuzzy Sets & Systems, and Information Sciences for journal editions.

### Fast Interpretable Fuzzy Inference Systems via Mixture-of-Gaussians

Chapter 4 is planned as a full methods paper ("draft paper 3"), with in-preparation status and targeting submission to a primary or secondary venue in the EUSFLAT family or FUZZ-IEEE.

### Topological Membership Generation for Fuzzy Inference Systems

Chapter 5 introduces a differentiator method for topological membership generation, positioned as a lead contribution and primary target for EUSFLAT 2027.

### Hierarchical Fuzzy Trees and HME with a Shared Ridge-TSK Primitive

Chapter 6 proposes a methods paper combining hierarchical fuzzy trees, hierarchical mixture of experts, and a shared ridge-TSK primitive, prepared for submission after EUSFLAT 2027.

## 9.4 Standalone Paper Opportunities

Several research directions have been flagged for collaborative discussion with Dr. Cohen, emerging from the dissertation's broader methodological contributions:

- **Performance-Engineering Study (tribble-opt).** A systems or methods paper targeting optimization and performance venues.

- **Quality-Diversity Over Legacy Solvers.** A comparative study combining CVT-MAP-Elites with Iso+LineDD algorithms, targeting optimization-focused venues.

- **Exact GPU/Parallel VAT Engine as a Systems Paper.** Positioned against Fast-VAT, eVAT, and clusiVAT, this paper is on hold pending re-measurement and re-quotation of Table 3.4 (checklist **E2b**). The underlying exactness result reproduces; however, a Fuzzy C-Means row in that table drifted due to comparing formulation styles (NumPy broadcasting vs. GEMM-based implementation) rather than device differences, and the speed envelope is being re-measured to restore its fidelity.

- **Lin-Kernighan Dual-Backend with VAT-Blocked TSP.** A hybrid approach combining dual-backend Lin-Kernighan with VAT-informed TSP blocking strategies.

---

## Notes on Reconciliation with Other Chapters

The publication record in this chapter is the authoritative source for all claims about publication status across the dissertation. Two sections of existing prose currently overstate the publication status and should be reconciled against this chapter:

- **§7 (Goals for Completion), opening sentence:** Currently states "Chapters 3 and 4 are done and published or nearly so." Chapter 4's paper appears only in §9.3 (In Preparation); only Chapter 3's work is published.

- **§1.3 (Dissertation Outline):** Currently states "This work was published across NAFIPS 2025 (Banff) and NAFIPS 2026 (El Paso)." The phrase "across" presumes the exact split between the two venues that this chapter marks as unresolved. This sentence should be softened to acknowledge that the distribution of papers across venues is not yet determined.