# Proposal Defense — Action Items & Open TODOs

Consolidated tracker for every open item scattered across the chapter files. Update status here; the chapters point back to this doc. Legend: ⬜ open · 🟨 in progress · ✅ resolved.

_Last updated: 2026-07-31 (post consistency pass across Ch 1–8)_

---

## A. Board-wide standards (apply to multiple chapters)

- ⬜ **Repeatable performance results** (scalability AND stability). One fixed protocol for every performance/scaling number: pinned clocks/thermals, multiple seeds, reported error bars, datacenter GPU with full-rate FP64. Mirrored in Ch 3/5/6; consolidated as **Goal G4 (Ch 7)**. Current numbers are single-machine point estimates, some thermally throttled.
- ✅ **Consolidate + verify the bibliography** — `references.bib` built and fully verified (Crossref/arXiv/DBLP pass 2026-07-31): 41 `[V]` + 23 `[S]`, zero unresolved. Placeholders resolved (eVAT = Meng & Yuan 2018 IJDSA; Fast-VAT = Avinash & Lachheb 2025 arXiv:2507.15904); Bonis–Oudot = PRL 102:37–43 (2018); AuToMATo title/authors corrected (TMLR 2025); ConiVAT confirmed arXiv-only. **Only proof-stage item:** confirm the "Kališnik" accent survives BibTeX/LaTeX encoding.
- ✅ **Broken citation resolved.** "[*Information Sciences* 2024]" = **Deshpande & Kumar, "Time and Memory Scalable Algorithms for Clustering Tendency Assessment of Big Data", Information Sciences 664:120324 (2024), DOI 10.1016/j.ins.2024.120324.** Now cited properly in Ch2 §2.2 and Ch3 §3.2 as `deshpande2024scalable`. It is also the closest prior art to the planned Ch9 note — see the blocking reads below.
- 🚫 **BLOCKING READS before writing the Ch9 complexity note** (and before approaching anyone about it). (1) **Deshpande & Kumar 2024** (Information Sciences 664:120324) — full text was unobtainable in the prior-art search; it already does MST-iVAT "without computing the full distance matrix" and attacks the ordering sub-quadratically with k-d trees. If it also states the O(N)-workspace result for VAT itself, the note has no contribution left. (2) **Wang et al. 2010** (PAKDD, iVAT) — no OA copy reachable; complexity content unknown. Institutional access needed for both.
- ⚠️ **NAME COLLISION — "pVAT" is taken.** Parveen & Sreevalsan-Nair, *"pVAT: Parallel VAT on the GPU"*, BDA 2013 (LNCS 8302:151–170) is a published method of that name that also swaps the MST algorithm for the VAT ordering (Prim → Borůvka). Reading our *p* as parallel/performant collides harder, not less. **A new name is needed**; the draft still says pVAT only because renaming mid-proposal is more confusing than flagging it. Affects the dissertation title-level nomenclature, Ch 3 throughout, and Ch 9.
- ⬜ **Regenerate all figures** at consistent publication style/size (see per-chapter figure placeholders).
- ✅ **Reference PDFs are structural templates only** — never cite Pickering/Arnett as an intellectual source; author's work is independent.
- 🔴 **CRITICAL — the Concrete reconciliation RAN and inverts Ch6's central claim.** First execution of `reproduce/tables/table_concrete_reconciliation.py` (2 seeds, 80/20, default hyperparameters, no target transform):

  | Model | R² | vs. proposal |
  |---|---:|---|
  | flat MoG-TSK order 0 | 0.279 | proposal says 0.44 |
  | flat MoG-TSK order 1 | 0.644 | proposal says 0.77 |
  | **flat MoG-TSK order 2** | **0.774** | proposal says 0.87 |
  | fuzzy tree | 0.579 ± 0.117 | proposal says 0.746 |
  | mixture of experts (HME) | 0.689 | proposal says 0.791 |
  | CART | 0.810 | was *pending* |
  | Random Forest | 0.909 | was *pending* |

  **Two problems, the second worse than the first.** (1) Every fuzzy number comes out lower than the proposal quotes — consistent with the proposal's figures being measured on a transformed target (the Ch4 pipeline applies `standard_transform`/log transforms), which is exactly the incomparability this experiment exists to expose. (2) **The ordering inverts.** Under one protocol the flat order-2 model (0.774) BEATS both the tree (0.579) and the HME (0.689), and CART/RF beat every fuzzy model. Ch6's narrative — hierarchy improves on flat — does not survive this protocol, and Ch6 already concedes the accuracy trade but not this strongly.

  **Do not rewrite Ch6 from this yet.** Confounds to eliminate first: only 2 seeds; default tree/HME hyperparameters with no tuning (the tree's ±0.117 suggests instability at defaults); no target transform, whereas the proposal's pipeline uses one; one dataset. **Next:** re-run at 5+ seeds, with and without the Ch4 target transform, and with the tree/HME configured as the demos configure them. If the inversion survives that, Ch6's empirical claim must change and the honest framing becomes "the hierarchy buys readability at a real accuracy cost, and does not close the gap to a random forest."
- ⬜ **Reconcile the Concrete numbers (original note, HIGH PRIORITY — three incomparable figures for "the flat model").** Ch 4: flat MoG-TSK 0.44/0.77/0.87 (orders 0/1/2). Ch 6 Table 6.1: flat baseline 0.658 (tree/HME experiment). Ch 6 §6.3.5: antecedent refinement 0.88→0.92. All real, all different configs (split/preprocessing/order/objective), none comparable. Worst symptom: refinement's 0.92 *appears* to beat the HME's 0.791, which would make Ch 6 pointless — it doesn't, they're different configurations. Both chapters now warn the reader explicitly, but the fix is ONE consistent Concrete benchmark so every model is measured identically.
- 🟨 **Tables (12 total, all carry real data; `*pending*` cells await the harness).** Fully measured: 3.2 (memory), 3.3 (adversarial ARI), 3.4 (stitch ablation), 4.1 (MoG results), 5.1 (multi-scale), 5.2 (selection bake-off), 6.1 (model family), 7.1 (goals map). Structure-fixed with pending cells: 3.1 (reorder time — intermediate N grid), 4.2 (ANFIS/GA-FIS/RF baselines), 6.2 (CART/M5/RF/ANFIS/flat-TSK baselines), 6.3 (interpretability counts at matched accuracy).
- 🟨 **Reproduction harness** (`reproduce/`): generators for Tables 3.1, 4.1/4.2, 6.1–6.3 written and compile-clean; emit Markdown+CSV with mean ± std over fixed seeds. REMAINING: execute them under the submodule envs to fill `*pending*` cells (expect minor API-name fixes on first run); add ANFIS/GA-FIS adapters; build the `run.py` orchestrator over `manifest.py`.
- ⬜ **Figure placeholders** still to produce (17): Ch 1 (×2), Ch 2 (×3), Ch 3 (×1), Ch 4 (×3), Ch 5 (×3), Ch 6 (×3). Ch 5's Fig 5.2 (band discovery) is flagged as that chapter's key figure.

## B. Needed from author / advisor

- ⬜ **NAFIPS paper details** (Ch 9): both published at **NAFIPS 2025 Banff (July 2025)** and **NAFIPS 2026 El Paso (March 2026)**. Need exact titles, page numbers/DOIs, co-author lists, which paper went to which meeting, and whether they're separate or combined.
- ⬜ **Confirm EUSFLAT 2027 (Sept 2027) submission deadline** — anchors the Ch 5 (and possibly Ch 4) paper schedule and the Ch 10 timeline.
- ⬜ **Confirm exact proposal-defense month** (assumed ~Dec 2026). Final defense = March 2028 (✅).
- ⬜ **Teaching/RA load per semester** (affects timeline throughput, Ch 10).
- ⬜ **Flagship end-to-end dataset** — author-preferred IoT (RT-IOT2022 / IoT-botnet) or UCI-58 Shuttle; left flexible, confirm later (Ch 7).
- ✅ Title, committee, tribble-opt→appendix, final-defense date — all resolved.

## C. Experiments / results owed (the "make it airtight" list)

- ⬜ **Ch 4 — ANFIS + GA-tuned-FIS baseline table** (train time + accuracy on identical splits). First thing owed to Ch 4's speed claim.
- ⬜ **Ch 4 — OUTPUT PARTITIONING STUDY: quantile vs uniform (open question, author has gone back and forth). (Goal G5.)** Settle empirically rather than by assertion; §4.3.2 currently presents both without a verdict.

  **The trade-off being tested.** *Uniform* (equal-width buckets across the output range) gives a more natural function approximation — each rule owns an equal span of the output, so TSK consequents interpolate evenly and the extremes stay covered — but on skewed targets some buckets get very few samples, so their antecedents and consequents are poorly estimated (or starve entirely). *Quantile* (equal-frequency buckets) guarantees every rule is statistically well-supported, but bucket centers crowd where the data is dense, under-resolving sparse regions — which for regression are often the extremes, i.e. exactly the values that matter.

  **Hypotheses.** (H1) The two coincide when the output is near-symmetric, so any difference must be driven by output skew. (H2) Quantile wins on aggregate error (R²/RMSE) as skew grows. (H3) Uniform wins on *tail* accuracy — error in the top/bottom deciles — and on max error. (H4) Quantile's advantage grows with bucket count, since starved buckets get likelier under uniform as `n_output_buckets` rises. (H5) A **hybrid** — equal-frequency interior buckets with centroids pinned at both range extremes — dominates both, and is the arm most likely to become the recommended default.

  **Design.** Three arms (uniform / quantile / hybrid-pinned-extremes) × `n_output_buckets` ∈ {2,3,4,5,6,8} × TSK order ∈ {0,1,2} × multi-seed, on (a) real regression sets spanning a range of output skewness — Concrete, turbine, WEC, wine-red, power consumption — and (b) a **synthetic sweep** with a controlled skewness parameter (e.g. lognormal-transformed target) to locate the crossover point directly rather than inferring it.

  **Metrics — aggregate error alone will not answer this.** Report: global R² and RMSE; **per-decile error** (exposes tail failure); error on the extreme deciles specifically; 95th-percentile and max absolute error; **min samples per bucket** and count of starved buckets (the stability diagnostic that explains *why* uniform fails when it does); and rules-to-reach-a-target-R² (the interpretability cost).

  **Deliverable.** A recommendation table — which scheme to use as a function of output skew and bucket count — plus a defensible default with the evidence behind it, and a plot of the crossover from the synthetic sweep. Feeds §4.3.2 and Table 4.1.

- ⬜ **Ch 4 — quantify the correction-rule pass** (§4.3.1): accuracy before vs after the confusion-matrix-driven second pass, with the paired confusion matrices (Fig 4.3). Currently claimed but unmeasured.
- ⬜ **Ch 4 — semi-supervised / incremental benchmark** (§4.3.3): the per-class-independence → incremental-update property is stated as a structural consequence, not a measured result. Needs a controlled streaming/partial-label experiment to promote it to a claim.
- ⬜ **Ch 4 — anomaly/open-set head-to-head (Table 4.3):** complement rule vs **one-class SVM** and **isolation forest** on BETH (train on benign only, detect unseen `evil==1`). Report detection rate + false-alarm rate at a matched operating point, plus the θ sweep curve (Fig 4.2, from `plot_anomaly_threshold_sweep`). This is the experiment owed to the §4.3.5 claim.
- ✅ **Ch 3 — dense-Prim question RESOLVED by code review (2026-07-31).** An earlier note claimed a tuned O(N²) dense Prim was a missing baseline. It is not missing: `_prim_mst_kernel_64/_32` in `tribble-cluster/src/tribbleclustering/pcvat.pyx` (lines 22–113 / 392+) IS a compact-active-set dense Prim — no heap, fused relax+next-min in one pass over the m active slots, O(N) workspace, O(N²) total — and it is the preferred import path. The O(N² log N) heap version (`pvat.py::vat_prim_mst`, lines 141–211) is the portable fallback. §3.3.1 rewritten to state all three regimes (cubic / heap / dense).
- ✅ **Ch 3 — TWO-STAGE FRAMING (resolved with author, 2026-07-31).** The complexity story is a progression, not a caveat: **stage one** = priority queue, O(N³)→O(N² log N), the *published* NAFIPS result; **stage two** = compact active set with fused relax-and-select, O(N² log N)→O(N²), heap removed, O(N) workspace — the shipped Cython fast path, *not yet published*. §3.3.1 rewritten around this. Name: *p* now read as **performant** VAT (also covers Borůvka-on-GPU), with "priority-queue VAT" retained for stage one specifically.
- ⬜ **POSSIBLE NOTE — VAT/iVAT sequencing complexity & memory** (Ch 9, tentatively 2027 Q1; authorship undecided). Implementation exists (`pcvat.pyx::_prim_mst_kernel_64/_32`); the work is the write-up plus a THREE-ARM timing study (classical cubic vs stage-one heap vs stage-two dense).
  **Novelty scoped per the repo's own review** (`tribble-cluster/docs/performance-novelty.md` §4.1, §4.4): the techniques are classical, so the claim is (a) the *correction* — heap Prim is asymptotically worse than dense Prim on VAT's complete graph, apparently unremarked in this literature; (b) the measured heap-vs-dense crossover, which that doc calls "itself a publishable result"; (c) iVAT coverage, which Fast-VAT 2025 lacks; (d) the O(N)-workspace regime, stated as the ≈2× constant-factor memory win it is. NOT "a faster MST."
  **Venue:** short-communication/NAFIPS style, not an algorithms conference.
- 🟨 **BLOCKING PRIOR-ART CHECK for the above (search dispatched 2026-07-31).** Does anyone already publish (i) an explicitly O(N²) VAT/iVAT *sequencing* bound, or (ii) the heap-vs-dense observation for VAT? The note's entire contribution hinges on both being unsaid. Also confirm what complexity Bezdek-Hathaway 2002 / Havens-Bezdek 2012 / Kumar-Bezdek 2020 actually claim, and whether Fast-VAT 2025 is constant-factor only.
- ⬜ **Ch 3 — head-to-head vs eVAT (Meng & Yuan 2018) & clusiVAT** on identical datasets. First comparison a reviewer will demand.
- ⬜ **Ch 3 — datacenter GPU re-run:** the pairwise-distance kernel currently LOSES (<1×) at low dimension/float64 on a consumer card. Prediction is that full-rate FP64 flips this; flagged in Table 3.3 as untested.
- ⬜ **Ch 3 / Ch 5 — real non-metric domains** (DTW time-series, edit distance, graph/kernel dissimilarity). The core niche is so far only synthetic. (Goal G2.)
- ⬜ **Ch 6 — HME EM refinement implemented** + full baseline suite (ANFIS, CART/C4.5, M5, flat TSK, Fumanal-Idocin 2025, D-TSK-FC). (Goal G3.)
- ⬜ **Ch 5 — END-TO-END FIS RESULT (the chapter's missing evidence).** Every Ch5 number is a *clustering* score (ARI), but the chapter's purpose is generating FIS antecedents. Build a FIS from the generated membership functions, measure prediction accuracy end-to-end from a bare dissimilarity matrix, and compare against the Ch4 Gaussian construction on data where both run. Until this exists the central claim rests on a proxy. Feeds the Ch7 capstone.
- ⬜ **Ch 5 — head-to-head vs Bonis–Oudot beta-plateau & AuToMATo** on identical data; formal prior-art search (IEEE Xplore/Scopus/ACM, cited-by on 1406.7130 / ToMATo).
- ⬜ **Ch 6 — interpretability evaluation** (rule count, path length, expert/audience study or established metric); empirical Magdalena-2018 rebuttal. (Goal G6.) NOTE: this is what fills Table 6.3's pending row; until then Ch 6 must say the interpretability payoff is *described*, not quantified.
- ⬜ **Ch 6 — Atwood machine memory result** (Table 6.4 pending row) + re-verify the double-pendulum numbers under the repeatability protocol.

## D. Proposed builds (Part III deliverables)

- ⬜ **Ch 5 / G1 — direct one-pass MF generation** (`MEMBERSHIP_ROADMAP.md` phases 1–6); phase 4 soft/kernel-weighted band membership is the research-interesting piece (fixes small-n over-segmentation).
- ⬜ **Ch 5 / G7 (stretch) — adaptive/model-based band discovery** for overlapping scales (change-point / barcode stability), beyond the gap heuristic. Designated first cut if timeline slips.
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
Each chapter file carries inline TODO notes; this doc is the roll-up. When a chapter TODO is added or resolved, reflect it here. Goal labels (G1–G7) map to Chapter 7 §7.2.
