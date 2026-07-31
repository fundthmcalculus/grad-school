# Proposal Defense — Action Items & Open TODOs

Consolidated tracker for every open item scattered across the chapter files. Update status here; the chapters point back to this doc. Legend: ⬜ open · 🟨 in progress · ✅ resolved.

_Last updated: 2026-07-31 (post consistency pass across Ch 1–8)_

---

## A. Board-wide standards (apply to multiple chapters)

- ⬜ **Repeatable performance results** (scalability AND stability). One fixed protocol for every performance/scaling number: pinned clocks/thermals, multiple seeds, reported error bars, datacenter GPU with full-rate FP64. Mirrored in Ch 3/5/6; consolidated as **Goal G4 (Ch 7)**. Current numbers are single-machine point estimates, some thermally throttled.
- 🟨 **Consolidate the bibliography** — DONE: `references.bib` built (hfis_review.bib folded in + VAT/clustering/persistence/optimization/XAI entries). REMAINING: verify every `[?]`-marked DOI/pages/venue; fill the two placeholder entries (`meng2018evat`, `avinash2025fastvat` — author names); confirm `bonis2018fuzzy` PRL details and `automato2024` authors.
- ⬜ **Regenerate all figures** at consistent publication style/size (see per-chapter figure placeholders).
- ✅ **Reference PDFs are structural templates only** — never cite Pickering/Arnett as an intellectual source; author's work is independent.
- ⬜ **Reconcile the Concrete flat-model R²** across chapters: Ch 4 reports flat MoG-TSK at 0.44/0.77/0.87 (orders 0/1/2); Ch 6 reports flat baseline at 0.658 (from the tree/HME experiment). Different split/preprocessing/order. Run one consistent Concrete benchmark so the flat baseline reads identically in both chapters.
- 🟨 **Tables built from real data (consistency pass):** Table 3.2 (adversarial ARI), 3.3 (stitch ablation), 5.1 (multi-scale), 5.2 (selection bake-off), 6.1 (model family), 7.1 (goals map). Partial/placeholder-with-real-data: 3.1 (reorder time — needs intermediate rows + classical 135K), 4.1 (MoG filled; ANFIS/GA/ref columns owed).
- ⬜ **Figure placeholders** still to produce: Ch 1 (×2), Ch 2 (×3), Ch 3 (×1), Ch 4 (×2), Ch 6 (×3).

## B. Needed from author / advisor

- ⬜ **NAFIPS paper details** (Ch 9): both published at **NAFIPS 2025 Banff (July 2025)** and **NAFIPS 2026 El Paso (March 2026)**. Need exact titles, page numbers/DOIs, co-author lists, which paper went to which meeting, and whether they're separate or combined.
- ⬜ **Confirm EUSFLAT 2027 (Sept 2027) submission deadline** — anchors the Ch 5 (and possibly Ch 4) paper schedule and the Ch 10 timeline.
- ⬜ **Confirm exact proposal-defense month** (assumed ~Dec 2026). Final defense = March 2028 (✅).
- ⬜ **Teaching/RA load per semester** (affects timeline throughput, Ch 10).
- ⬜ **Flagship end-to-end dataset** — author-preferred IoT (RT-IOT2022 / IoT-botnet) or UCI-58 Shuttle; left flexible, confirm later (Ch 7).
- ✅ Title, committee, tribble-opt→appendix, final-defense date — all resolved.

## C. Experiments / results owed (the "make it airtight" list)

- ⬜ **Ch 4 — ANFIS + GA-tuned-FIS baseline table** (train time + accuracy on identical splits). First thing owed to Ch 4's speed claim.
- ⬜ **Ch 3 — head-to-head vs eVAT & clusiVAT** on identical datasets. First comparison a reviewer will demand.
- ⬜ **Ch 3 / Ch 5 — real non-metric domains** (DTW time-series, edit distance, graph/kernel dissimilarity). The core niche is so far only synthetic. (Goal G2.)
- ⬜ **Ch 6 — HME EM refinement implemented** + full baseline suite (ANFIS, CART/C4.5, M5, flat TSK, Fumanal-Idocin 2025, D-TSK-FC). (Goal G3.)
- ⬜ **Ch 5 — head-to-head vs Bonis–Oudot beta-plateau & AuToMATo** on identical data; formal prior-art search (IEEE Xplore/Scopus/ACM, cited-by on 1406.7130 / ToMATo).
- ⬜ **Ch 6 — interpretability evaluation** (rule count, path length, expert/audience study or established metric); empirical Magdalena-2018 rebuttal. (Goal G5.)

## D. Proposed builds (Part III deliverables)

- ⬜ **Ch 5 / G1 — direct one-pass MF generation** (`MEMBERSHIP_ROADMAP.md` phases 1–6); phase 4 soft/kernel-weighted band membership is the research-interesting piece (fixes small-n over-segmentation).
- ⬜ **Ch 5 / G6 (stretch) — adaptive/model-based band discovery** for overlapping scales (change-point / barcode stability), beyond the gap heuristic. Designated first cut if timeline slips.
- ⬜ **Ch 5 — wire output MFs into the tribble-fis FIS** (integration that ties Ch 5 → Ch 6 → capstone).
- ⬜ **Ch 7 — integrated end-to-end pipeline** capstone + flagship case study.

## E. Defensibility fixes before submission

- ⬜ **Retire the "priority-queue MST speedup" O-notation framing** for dense graphs (heap-Prim O(N²log N) vs O(N²) dense-Prim) — restate as the argmin/sort speedup it is; defend empirically. (Ch 3 §3.3.1 already softened in prose.)
- ⬜ **Drop the ungrounded "pVAT six-orders-of-magnitude" web claim** (AI-search confabulation; at most a self-citation).
- ⬜ **Fix Zhang-2023 author attribution** in the HFIS references (README misattributes to "H. Wang et al.").
- ⬜ Close open literature searches: knot/breakpoint optimization precedent (Ch 6); dedicated fuzzy-MoE/HHFNN search to narrow the HME nesting claim (Ch 6).

## F. Structural / editorial decisions (lower stakes)

- ⬜ Ch 1 — how heavily to invoke XAI/regulation framing (secondary per author).
- ⬜ Ch 2 — whether to include a formal-methods/verification subsection (possible Kreinovich nod).
- ⬜ Ch 5 — consolidate Options A–D presentation (recommend: lead with D + persistence-ramp; A/B/C supporting).
- ⬜ Ch 6 — MIMO temporal-memory as its own short chapter vs a section (recommend section; nice aerospace hook).
- ⬜ Engineering debt: de-duplicate the six caller scripts' predict loops (tribble-fis).

---

### How this doc is maintained
Each chapter file carries inline TODO notes; this doc is the roll-up. When a chapter TODO is added or resolved, reflect it here. Goal labels (G1–G6) map to Chapter 7 §7.2.
