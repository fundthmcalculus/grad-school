# Chapter 10 — Timeline

The plan runs from the proposal defense at the end of 2026 to the final defense in **March 2028** — a little over a year of research runway. It is deliberately aggressive, per my own intent to defend a substantial body of results rather than a minimal one, and the risk of that is managed in Chapter 7: the completed work (Chapters 3 and 4) is the floor, and the stretch goal (G7) is the designated first cut. The schedule below is organized around the eight goals of Chapter 7 and the papers they feed. Quarters are calendar quarters; the goal labels match Table 7.1 and `ACTION_ITEMS.md`.

## 10.1 Gantt

```mermaid
gantt
    title Dissertation timeline — proposal (Dec 2026) to final defense (Mar 2028)
    dateFormat  YYYY-MM-DD
    axisFormat  %b %Y
    todayMarker off

    section Milestones
    Proposal defense                          :milestone, crit, 2026-12-15, 0d
    Final defense                             :milestone, crit, 2028-03-15, 0d

    section Research goals
    G4 repeatable perf + eVAT/clusiVAT        :crit, g4, 2027-01-01, 90d
    G1 one-pass membership generation         :g1, 2027-04-01, 91d
    G5 output partitioning study              :g5a, 2027-04-01, 60d
    G2 real non-coordinate benchmarks         :crit, g2, 2027-07-01, 92d
    G3 HME EM + baseline suite                :crit, g3, 2027-07-01, 183d
    Integrated pipeline (capstone)            :crit, cap, 2028-01-01, 45d
    G6 interpretability, measured             :g6, 2028-01-01, 45d
    G7 adaptive multi-scale (stretch)         :g7, 2028-01-15, 30d

    section Papers & conferences
    Ch3 pVAT journal                          :p3, 2027-01-01, 90d
    VAT complexity correction note            :pk, 2027-02-01, 75d
    Ch5 membership paper (write/submit)       :p5, 2027-01-15, 105d
    EUSFLAT 2027 (present)                    :milestone, e27, 2027-09-15, 0d
    Ch6 hierarchy paper                       :p6, 2027-10-01, 90d

    section Writing
    Dissertation writing                      :w, 2027-10-01, 150d
    Defense preparation                       :wp, 2028-02-01, 43d
```

The Chapter 5 membership paper targets **EUSFLAT 2027** (September 2027); it is written and submitted in the first half of 2027 against the EUSFLAT deadline and presented in September. NAFIPS 2025 (Banff, July 2025) and NAFIPS 2026 (El Paso, March 2026) are already published and therefore predate this timeline.

## 10.2 Quarter grid (renderer-independent fallback)

```
                                    2026    2027    2027    2027    2027    2028
Goal / activity                       Q4      Q1      Q2      Q3      Q4      Q1
                                   (prop.)                                (defense)
------------------------------------------------------------------------------------
G4  repeatable perf + eVAT/clusiVAT    .     ####      .       .       .       .
G1  one-pass membership generation     .       .     ####      .       .       .
G5  output partitioning study          .       .     ###       .       .       .
G2  real non-coordinate benchmarks     .       .       .     ####      .       .
G3  HME EM + baseline suite            .       .       .     ####    ####      .
    integrated pipeline (capstone)     .       .       .       .       .     ####
G6  interpretability, measured         .       .       .       .       .     ###
G7  adaptive multi-scale (stretch)     .       .       .       .       .     ~~~
------------------------------------------------------------------------------------
    Ch3 pVAT journal                   .     ####      .       .       .       .
    VAT complexity note (if pursued)   .     ####      .       .       .       .
    Ch5 paper -> EUSFLAT 2027          .     ####    ####    PRES      .       .
    Ch6 hierarchy paper                .       .       .       .     ####      .
    dissertation writing               .       .       .       .     ....    ####
------------------------------------------------------------------------------------
    milestones                       PROP.                    EUSFLAT           DEFENSE
```

Legend: `####` scheduled work · `~~~` stretch (first to cut) · `....` ramp-up · `.` idle.

## 10.3 Milestone table

| Quarter | Focus | Goals | Deliverable |
|---|---|---|---|
| **2026 Q4** | Proposal defense (~Dec) | — | Defended proposal; committee feedback folded in |
| **2027 Q1** | Close credibility gaps | G4 | Fixed-protocol re-timing (error bars); eVAT/clusiVAT head-to-head; one consistent Concrete benchmark so Ch 4 and Ch 6 numbers are comparable; Ch 3 → pVAT journal; VAT complexity correction note (if the blocking reads support it); begin Ch 5 paper |
| **2027 Q2** | Membership generation | G1, G5 | One-pass MF (roadmap phases 1–4); output-partitioning study; **Ch 5 paper submitted to EUSFLAT 2027** |
| **2027 Q3** | Real data + hierarchy | G2, G3 (start) | DTW/edit/graph benchmarks; begin HME EM; **present at EUSFLAT 2027 (Sept)** |
| **2027 Q4** | Hierarchy + baselines | G3 | HME EM done; ANFIS/CART/M5/flat-TSK suite; Ch 6 paper; writing begins |
| **2028 Q1** | Capstone + write-up + **defense (Mar)** | capstone, G6, G7\* | End-to-end flagship case study (Ch 5 → Ch 6 integration); interpretability eval; dissertation complete; **final defense March 2028** |

## 10.4 Dependencies and critical path

- **G4 is front-loaded** (Q1) because every later number is reported under its protocol; the credibility fixes should land before the new results pile up.
- **G1 precedes the capstone** — the one-pass membership generator is what the integrated pipeline consumes, so it must be done (Q2) well before the 2028 Q1 capstone. The capstone is also the first time Chapter 5's membership functions are fed to Chapter 6's models at all, which makes it a genuine experiment rather than an integration chore.
- **G2 has no upstream dependency** — start it as soon as G4's protocol exists; it is the top credibility item and the riskiest to leave late.
- **G3's EM is the largest single build**, spanning Q3–Q4; its de-scope path is to keep the one-shot mixture if it slips.
- **Papers track the work**: Ch 3 journal and, if the prior-art blockers clear, the VAT complexity correction note (Q1 — both draw on work already implemented, so they are write-ups rather than new builds) → Ch 5 → EUSFLAT 2027 (written Q1–Q2, presented Q3) → Ch 6 (Q4) → the capstone/journal version alongside write-up (2028 Q1).

## 10.5 Buffer and risk

The 15-month runway has little slack, which is intentional. **G7 (adaptive multi-scale) is the designated first cut.** If G3 (EM) or G2 (real non-metric) overruns, the fallbacks from Chapter 7 apply — the one-shot mixture and a synthetic-plus-one-real-domain result, respectively — and the completed Chapters 3 and 4 remain a defensible floor regardless. Defense preparation is carved out explicitly in Feb 2028 so the final month is not a scramble.

---

### Open items — NEED FROM AUTHOR/ADVISOR
- Confirm the exact proposal-defense month (assumed ~Dec 2026); final defense is fixed at March 2028.
- **Confirm the EUSFLAT 2027 submission deadline** (conference September 2027) — it anchors the Ch 5 (and possibly Ch 4) paper schedule. NAFIPS 2025 Banff and NAFIPS 2026 El Paso are already published.
- Confirm teaching/RA load per semester (affects realistic throughput).

*Draft — Chapter 10 prose + Gantt, in the author's voice. Source outline in `../chapters/10-timeline.md`; goals map to Table 7.1 and `../ACTION_ITEMS.md`.*
