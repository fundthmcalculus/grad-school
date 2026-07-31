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

- **The right complexity for VAT/iVAT sequencing** — *with Dr. Vladik Kreinovich* → Ch 3 §3.3.1 (stage two). Kreinovich's observation that the stage-one method was really a heap algorithm prompted the line of thinking; co-authorship is intended.

  **What the contribution is NOT.** It is not a new MST algorithm. Compact active-set dense Prim — swap-with-last removal, fused relax-and-select — is classical and has been $O(N^2)$ since Prim 1957. The repo's own novelty review says this plainly (`tribble-cluster/docs/performance-novelty.md` §4.4: *"The individual techniques are classical … Claim the composition + the regime + the measured envelope, not the parts."*). Any framing that reads as "we invented a faster MST" will and should be rejected.

  **What the contribution is.** A correction to the VAT literature plus the measurement that settles it:
  1. **The correction.** VAT operates on a *complete* graph, so heap Prim is $O(E \log V) = O(N^2 \log N)$ — asymptotically *worse* than plain dense Prim's $O(N^2)$. The VAT family has been shipping cubic re-scans and heap variants (mine included, at stage one) when the dense formulation was strictly better all along. That observation does not appear to be in print in this literature; confirming that is the open prior-art question below.
  2. **The measured envelope.** Heap versus compact-dense across $N$, precision, and cache regime — where does each actually win, and by how much? The repo's own review flags this comparison as *"itself a publishable result."* Three arms: classical cubic, stage-one heap, stage-two dense.
  3. **iVAT, not just VAT.** Fast-VAT (2025) is the nearest concurrent work and covers VAT only. The $O(N)$-workspace formulation carrying through the iVAT minimax recursion is the part with no direct competitor.
  4. **The regime.** Exact, arbitrary (non-metric) dissimilarities, $O(N)$ working memory — stated as the constant-factor memory win it is (≈2× peak), not an asymptotic one, since the kd-tree line achieves sub-quadratic memory for Euclidean data.

  Implementation already exists and is the shipped fast path (`pcvat.pyx::_prim_mst_kernel_64/_32`), so the work is the write-up plus the three-arm timing study. **Venue fit:** a short, sharp complexity-correction note suits NAFIPS or a similar short-communication venue; this is not an algorithms-conference paper and should not be aimed at one.

  ⚠️ **Open prior-art question (search in progress):** whether anyone has already published an explicitly $O(N^2)$ VAT/iVAT sequencing bound, or already noted the heap-versus-dense point for VAT. The whole note hinges on that being unsaid. See `ACTION_ITEMS.md`.
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
