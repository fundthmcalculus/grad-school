# Chapter 9 — Publications

**Status:** Outline · Part III — **blocked on author records** (paper metadata; checklist **D2**). The venues are confirmed; nothing about the papers themselves is.
**Mirrors:** Pickering Ch 10 "My Publications".
**Purpose:** list published / submitted / in-preparation work mapped to chapters, so the committee sees the completed record and the plan.

---

## 9.1 Published — venues confirmed, paper metadata not yet supplied

Two NAFIPS meetings, both feeding Chapter 3, both confirmed by the author:
- **NAFIPS 2025 — Banff, Alberta (July 2025)**
- **NAFIPS 2026 — El Paso, TX (March 2026)**

Everything about the *papers* is unresolved, and this section cannot be written until it is supplied from the author's own records. Nothing below is a citation; each line is a placeholder that names what is missing, and none of it should be typeset as a reference in this state.

- 🔒 **Paper A** *(working title "Utilization of VAT for Hot-start of TSP")* → Ch 3. **Blocking:** exact title; which of the two meetings; co-author list; pages; DOI.
- 🔒 **Paper B** *(working title "mergeVAT: 58K×58K in 60 seconds")* → Ch 3. **Blocking:** exact title; which of the two meetings; co-author list; pages; DOI.
- 🔒 **The cardinality is itself unresolved.** The quals slides present a "paper_combined", so it is not currently known whether these went to the two meetings as two separate papers or as one combined paper. Until that is settled, "two published papers" is an assumption rather than a record, and this chapter does not assert it.

All three are checklist **D2** — author records, not research. They also gate the second pass on the acknowledgements (**A6**), which currently thanks the committee and Jon Salisbury but no co-authors, precisely because the co-author lists are what is missing here.

> ⚠️ **Other chapters currently overstate this, and the sentences should be reconciled against this section rather than the other way round.** Two in particular:
> - `prose/07-goals-for-completion.md` §7 opening — *"Chapters 3 and 4 are done and published or nearly so"*. Chapter 4's paper appears only under **In Preparation** in §9.3; only Chapter 3's work is published.
> - `prose/01-introduction.md` §1.3 (Dissertation Outline) — *"This work was published across NAFIPS 2025 (Banff) and NAFIPS 2026 (El Paso)."* The word "across" presumes exactly the split this section records as unknown; it should be softened until the cardinality question above is answered.
>
> Chapter 8's version of the same claim has been narrowed to "Chapter 3's work is published, Chapter 4's paper is in preparation". Chapter 10 §10.1 refers to the two meetings at venue level only and asserts nothing about papers, so it needs no change.

## 9.2 Submitted / Under Review — nothing currently under review

There is no work under review as of this draft. The next submission is the Chapter 5 membership paper to **EUSFLAT 2027** (February 2027 deadline), listed under §9.3 as in preparation. This heading is kept rather than deleted so that its emptiness is a statement about the current state rather than an omission a reader has to infer.

## 9.3 In Preparation (mapped to chapters)

Primary target venue: **EUSFLAT 2027 (September 2027)** (also consider FUZZ-IEEE / Fuzzy Sets & Systems / Information Sciences for journal versions).

- **A correction note on VAT ordering complexity and memory** → Ch 3 §3.3.1. An audit and correspondence piece rather than a methods paper, scoped down after the 2026-07-31 prior-art search (Ch 3 §3.3.1 gives the reasoning and the honest assessment of what it is worth). Its claim is that the VAT literature repeatedly states an O(N²) *space* bound as inherent to using Prim for the ordering — Fast-VAT, Kumar's thesis, and Deshpande & Kumar's own motivation all say so — when array-based dense Prim needs only O(N) working memory, and that the widely used implementations (R **seriation**, Python **pyclustertend**, Fast-VAT's own code) are cubic in the ordering while citing O(N²); the supporting evidence is the measured three-arm envelope of cubic, heap and dense. It claims **no algorithmic novelty**, cites Müllner (2011) and Deshpande & Kumar (2024) prominently and early, and targets a short correspondence or software-note venue. It is contingent on two full-text reads that are still outstanding — checklist **E8**: **Deshpande & Kumar 2024** and **Wang et al. 2010 (PAKDD)** — and if the former already states the O(N)-workspace result for VAT itself rather than only for MST-iVAT, the note should be dropped rather than narrowed further.

- **Fast interpretable FIS via Mixture-of-Gaussians** ("draft paper 3") → Ch 4.
- **Topological membership generation for fuzzy inference systems** → Ch 5 (lead differentiator; EUSFLAT 2027 target).
- **Hierarchical fuzzy trees & HME with a shared ridge-TSK primitive** → Ch 6.

## 9.4 Standalone-paper opportunities (flagged for Dr. Cohen — from master outline)

- Performance-engineering study (tribble-opt) — systems/methods venue.
- Quality-Diversity over legacy solvers (CVT-MAP-Elites + Iso+LineDD) — optimization venue.
- Exact GPU/parallel VAT engine as a systems paper (vs eVAT / Fast-VAT / clusiVAT). **On hold until Table 3.4 is re-quoted:** `reproduce/PROVENANCE_MAP.md` note 15 marks that table **drifted**, and its Fuzzy C-Means row overstates the GPU by roughly an order of magnitude because the quoted ratio compares a NumPy broadcasting implementation against a GEMM-based one — a difference of formulation, not of device. The exactness result the paper would rest on does reproduce; the speed envelope is being re-measured (checklist **E2b**).
- Lin-Kernighan dual-backend + VAT-blocked TSP.

---

### Open items — NEED FROM AUTHOR
- ~~Exact venues/years~~ → **RESOLVED: NAFIPS 2025 Banff (July 2025) + NAFIPS 2026 El Paso (March 2026), both published; EUSFLAT 2027 (Sept 2027) is the in-prep target.** Still need exact paper titles, page numbers/DOIs, which paper went to which NAFIPS meeting, and whether the two are separate or combined.
- Author lists / co-authors.
- Any awards or invited talks to note.
