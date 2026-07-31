# Chapter 10 — Timeline

**Status:** Outline · Part III (placeholder milestones — refine with author/advisor)
**Mirrors:** Pickering Ch 11; Arnett Ch 7.
**Anchors:** proposal defense ~Dec 2026; **final (terminal) defense March 2028** (~15 months of research runway). Aggressive scope.
**Deliverable:** a Gantt-style chart (create as figure) + the milestone table below.

---

## 10.1 Milestone table (draft — quarters relative to proposal)

| Quarter | Focus | Goal (Ch 7) | Deliverable |
|---|---|---|---|
| **Q0 — 2026 Q4** | Proposal defense (~Dec 2026) | — | Defended proposal; committee feedback incorporated |
| **Q1 — 2027 Q1** | Close credibility gaps | G4 (part) | Thermally-stable re-timing + error bars; eVAT/clusiVAT head-to-head (Ch 3 → journal) |
| **Q2 — 2027 Q2** | Membership generation | G1 | One-pass MF (MEMBERSHIP_ROADMAP phases 1–4); Ch 5 paper → EUSFLAT 2027 |
| **Q3 — 2027 Q3** | Real non-metric data + hierarchy | G2 + G3 (start) | DTW / edit-distance / graph benchmarks; begin HME EM implementation |
| **Q4 — 2027 Q4** | Hierarchy + baselines + integration | G3 + capstone start | HME EM done; ANFIS/CART/M5/flat-TSK suite (Ch 6); integrated pipeline stood up |
| **Q5 — 2028 Q1** | Capstone + write-up + **defense (March 2028)** | G1–G4 capstone, G5 | End-to-end flagship case study; interpretability eval; dissertation written; **final defense March 2028** |

*Note: this is deliberately aggressive — G3 (EM) now shares Q3–Q4 with G2, and the capstone overlaps write-up. G6 (adaptive multi-scale, overlapping scales) is an explicit stretch/cut if anything slips (see §10.3).*

## 10.2 Dependency notes

- G2 (real non-metric) depends on nothing — start early (Q3); it's the top credibility item.
- G1 (one-pass MF) feeds the integrated pipeline (Q5) — must precede capstone; do it Q2.
- G3 EM is the largest single build — spans Q3–Q4; de-scope path = keep HME one-shot if it slips.
- Papers pipeline: Ch 3 journal (Q1) → Ch 5 → EUSFLAT 2027 (submit ~spring 2027, present Sept 2027) → Ch 6 (Q4) → capstone/journal (Q5).

## 10.3 Buffer / risk

- The 15-month runway to March 2028 has little slack — this is intentional per the aggressive-scope directive. **G6 (adaptive multi-scale / overlapping scales) is the designated first cut.** If G3 (EM) or G2 (real non-metric) overruns, fall back to HME one-shot and synthetic-plus-one-real-domain respectively; the completed Part II work (Ch 3, Ch 4) already constitutes a defensible dissertation floor.

---

### Open items — NEED FROM AUTHOR/ADVISOR
- ~~Hard graduation deadline~~ → **RESOLVED: final defense March 2028** (2028 Q1). Still confirm the exact proposal-defense month (assumed ~Dec 2026).
- Confirm teaching/RA load per semester (affects realistic throughput).
- Confirm which conferences drive the paper schedule. **Known anchor: EUSFLAT 2027 (September 2027)** — the target venue for the Ch 5 (topological membership) and Ch 4 (MoG) submissions; back-plan the Q1–Q2 2027 writing against its submission deadline once announced. (NAFIPS 2025 Banff / NAFIPS 2026 El Paso are already published.)
