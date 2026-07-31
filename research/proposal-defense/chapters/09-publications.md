# Chapter 9 — Publications

**Status:** Outline · Part III (needs author input — publication metadata)
**Mirrors:** Pickering Ch 10 "My Publications".
**Purpose:** list published / submitted / in-preparation work mapped to chapters, so the committee sees the completed record and the plan.

---

## 9.1 Published / Presented

Two published conference venues, both feeding Chapter 3:
- **NAFIPS 2025 — Banff, Alberta (July 2025)**
- **NAFIPS 2026 — El Paso, TX (March 2026)**

Papers:
- **"Utilization of VAT for Hot-start of TSP"** → Ch 3. *(confirm venue: Banff 2025 or El Paso 2026; exact title, co-authors, page/DOI)*
- **"mergeVAT: 58K×58K in 60 seconds"** → Ch 3. *(confirm venue; exact title, co-authors, page/DOI)*
- *(Note: quals slides also present a "paper_combined" — confirm whether the two papers published separately or as one combined paper, and which paper went to which of the two NAFIPS meetings.)*

## 9.2 Submitted / Under Review

- *(fill as applicable)*

## 9.3 In Preparation (mapped to chapters)

Primary target venue: **EUSFLAT 2027 (September 2027)** (also consider FUZZ-IEEE / Fuzzy Sets & Systems / Information Sciences for journal versions).

- **An $O(N^2)$ formulation of VAT/iVAT sequencing** — *with Dr. Vladik Kreinovich* → Ch 3 §3.3.1 (stage two). The follow-on to the published priority-queue result: removing the heap entirely via a compact active set with fused relax-and-select, taking the reorder from $O(N^2 \log N)$ to $O(N^2)$ with $O(N)$ workspace. Kreinovich's observation that the stage-one method was really a heap algorithm is what prompted the line of thinking; co-authorship is intended. Implementation already exists and is the shipped fast path (`pcvat.pyx::_prim_mst_kernel_64`), so what the paper needs is the write-up plus the timing study against both stage one and the classical implementation. **Scope the novelty tightly** — dense Prim is textbook; the claim is the VAT-sequencing formulation, the fused single pass, the O(N) workspace, and bit-identical verification.
- **Fast interpretable FIS via Mixture-of-Gaussians** ("draft paper 3") → Ch 4.
- **Topological membership generation for fuzzy inference systems** → Ch 5 (lead differentiator; EUSFLAT 2027 target).
- **Hierarchical fuzzy trees & HME with a shared ridge-TSK primitive** → Ch 6.

## 9.4 Standalone-paper opportunities (flagged for Dr. Cohen — from master outline)

- Performance-engineering study (tribble-opt) — systems/methods venue.
- Quality-Diversity over legacy solvers (CVT-MAP-Elites + Iso+LineDD) — optimization venue.
- Exact GPU/parallel VAT engine as a systems paper (vs eVAT / Fast-VAT / clusiVAT).
- Lin-Kernighan dual-backend + VAT-blocked TSP.

---

### Open items — NEED FROM AUTHOR
- ~~Exact venues/years~~ → **RESOLVED: NAFIPS 2025 Banff (July 2025) + NAFIPS 2026 El Paso (March 2026), both published; EUSFLAT 2027 (Sept 2027) is the in-prep target.** Still need exact paper titles, page numbers/DOIs, which paper went to which NAFIPS meeting, and whether the two are separate or combined.
- Author lists / co-authors.
- Any awards or invited talks to note.
