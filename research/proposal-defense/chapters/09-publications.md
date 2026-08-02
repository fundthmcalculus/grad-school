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

- **A correction note on VAT ordering complexity and memory** → Ch 3 §3.3.1. **⚠️ SCOPE CUT after the 2026-07-31 prior-art search; do not write this until the two blocking reads below are done.**

  **What the search killed.** (a) The claim that the literature is confused about *time* — the Kumar–Bezdek 2020 survey states O(N²) for VAT in four places and correctly credits Havens–Bezdek for iVAT O(N³)→O(N²). (b) The claim that "which MST algorithm should VAT use" is untouched — **Parveen & Sreevalsan-Nair 2013 already published a method called mergeVAT** that swaps Prim for Borůvka on GPU. (c) The O(N)-workspace/no-full-matrix angle for iVAT — **Deshpande & Kumar 2024** (*Information Sciences* 664:120324) already do this via MST-iVAT, and attack the ordering *sub-quadratically* with k-d trees. (d) Any pretence of distance from the single-linkage literature — Bezdek's own group published *"Is VAT really single linkage in disguise?"* (Havens et al. 2009) printing Prim and the VAT ordering side by side, after which Müllner (2011) supplies the O(N)-memory argument directly.

  **What survives.** One genuine, cited error: the VAT literature repeatedly asserts that O(N²) *space* is inherent to using Prim — Fast-VAT ("O(n²) space complexity for storing R"), Kumar's thesis ("O(n²) time and space to store all the edges"), and Deshpande & Kumar's own motivation ("O(N²) time and space complexity as they use Prim's algorithm"). That is false; array-based dense Prim needs O(N) working memory. The misconception is load-bearing enough to have motivated a 2024 *Information Sciences* paper. Alongside it: the widely used implementations (R **seriation**, Python **pyclustertend**, Fast-VAT's own code) are cubic in the ordering while citing O(N²).

  **Therefore: an audit/correspondence piece, not a methods paper.** Structure: the stated-vs-actual space bound; the O(N)-workspace result following from Rohlf (1973)/Müllner (2011) via the equivalence Havens et al. (2009) already established; the survey of shipped implementations that are cubic in practice; the measured three-arm envelope (cubic / heap / dense). Claim **no** algorithmic novelty. Cite Müllner and Deshpande & Kumar prominently and early. Venue: short correspondence or software note. Framed as a new algorithm it will be rejected, and correctly.

  **Honest assessment of worth:** a simpler fix to a problem Deshpande & Kumar already solved by a better route, plus a real correction to a real error. Modest. Worth writing only if the audit of shipped implementations is done properly, since that is the part with no precedent.

  🚫 **BLOCKING READS.** (1) **Deshpande & Kumar 2024** — full text unobtainable in the search; if it already states the O(N)-workspace result for VAT itself rather than only for MST-iVAT, even this narrow framing collapses. (2) **Wang et al. 2010 (PAKDD)** — genuinely unverifiable, no OA copy; its complexity content is unknown.

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
