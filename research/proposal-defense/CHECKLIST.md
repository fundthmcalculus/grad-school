# Burn-down checklist

Shared working list. Tick items as they land; each has enough context to start cold.
Companion docs: [`REVIEW_2026-08-02.md`](REVIEW_2026-08-02.md) (what was found and why),
[`ACTION_ITEMS.md`](ACTION_ITEMS.md) (full backlog), [`NEXT_STEPS.md`](NEXT_STEPS.md) (plan of record).

_Opened 2026-08-02. Legend: ⬜ open · 🟨 in progress · ✅ done · 🔒 blocked on you._

---

## A. Blocked on the author

- [ ] 🔒 **A1 — Rename the method.** `pVAT` is taken (Parveen & Sreevalsan-Nair, BDA 2013 — also
      a GPU VAT that swaps the MST algorithm). Reading the *p* as parallel or performant
      collides harder. Touches the dissertation's title-level nomenclature, Ch 3 throughout,
      Ch 9, and two published NAFIPS papers. **Decide before the defense** — §3.3.1 currently
      walks the committee through three names in sequence and will eat time.
- [ ] 🔒 **A2 — NAFIPS paper metadata** (Ch 9): exact titles, page numbers/DOIs, co-authors,
      which paper went to Banff 2025 vs. El Paso 2026, and whether they published separately
      or combined. Ch 9 cannot be finished without it.
- [ ] 🔒 **A3 — Confirm the flagship end-to-end dataset**: RT-IOT2022 / IoT-botnet, or UCI-58
      Shuttle. Defines the Ch 7 capstone.
- [ ] 🔒 **A4 — Confirm the EUSFLAT 2027 submission deadline.** Anchors the Ch 5 paper schedule
      and the Ch 10 timeline.
- [ ] 🔒 **A5 — Confirm the proposal-defense month** (assumed ~Dec 2026) and teaching/RA load
      per semester (affects Ch 10 throughput).

## B. Reproducibility infrastructure

- [x] ✅ **B1 — Machine specs captured in every archive.** `run_all_tables.sh` writes host, OS,
      kernel, CPU, cores, RAM, governor, boost, GPU, Python into `PROVENANCE.txt`, on backfills too.
- [x] ✅ **B2 — Machine block on every emitted table.** `common.machine_block()` appends it to
      every generated Markdown.
- [x] ✅ **B7 — Figures export PNG + EPS.** `common.save_figure()` writes both from one
      matplotlib figure — PNG for the Markdown, EPS (vector) for the LaTeX build — framed
      identically via `bbox_inches="tight"`. Every future figure should go through it. Keep
      figures opaque and fully vector: EPS has no alpha channel.
- [x] ✅ **B3 — Timings reported as ratios, seconds kept in CSV.** `common.normalized_worst()`
      normalizes each row against its slowest arm; `emit(md_header=, md_rows=)` lets Markdown
      and CSV diverge. Applied to `table_3_1_pvat_scaling` and `table_3_1_reorder_three_arm`.
- [ ] ⬜ **B8 — Automate the harness → document figure hop.** `save_figure()` writes to
      `reproduce/outputs/figures/fig_03_complexity_fit.{png,eps}`; the document references
      `prose/fig/03-complexity-fit.png`. That copy is **manual today**. `build_pdf.py` now
      emits an image line when the target exists and still strips it when it does not, so a
      figure that is not copied across silently reverts to a placeholder — which is the
      failure mode worth automating away. A name map plus a copy step in the build closes it.
- [ ] ⬜ **B4 — Submodule SHA guard.** The harness should refuse to emit, or loudly stamp, when
      a submodule SHA differs from the last archive's. **This failure has happened twice** —
      once with `fix/pin-extreme-bucket-means`, once with `resolve-flm-pr`. Highest-value
      remaining infra item.
- [ ] ⬜ **B5 — Re-take Chapter 3's timing grid on the workstation.** §3.4 currently spans two
      hosts: the swept grid is laptop data, the memory ceilings and large reorders are
      workstation. Cheap, and it also settles whether the ~45% swing was thermal or
      cross-machine. *(Author: earlier results were a faster machine — flagged, not a blocker.)*
- [ ] ⬜ **B6 — Fix or remove `pvat.vat_prim_mst_seq`.** Exported public API that silently
      returns a wrong ordering (seed vertex, then ascending index order). Cause is a vectorized
      call to a scalar-typed `_get_dist`. Nothing calls it. See `REVIEW` ★2.

## C. Experiments owed

- [ ] ⬜ **C1 — ANFIS and GA-tuned-FIS baselines** (Ch 4, Table 4.5). **The single most
      important experiment in the backlog**: the title, Ch 1, Ch 7 and Ch 8 all claim *orders
      of magnitude faster*, and there is currently no fuzzy baseline to say faster *than what*.
      Adapters go at `reproduce/tables/_baseline_anfis.py` and `_baseline_gafis.py`; the table
      auto-detects them.
- [x] ✅ **C2 — Complexity fit against reference curves.** Table 3.2 + Figure 3.2 now sweep a
      small grid (100–1,000, sized so the cubic arm runs at every point) with both axes
      normalized, and fit a log-log exponent per arm. Classical **3.11** (theory 3) and stage
      one **1.87** (theory ≈2.1, the log factor invisible over one decade) both confirm.
- [ ] ⬜ **C2b — Diagnose the stage-two cliff.** *(New, and the sharpest open item in Ch 3.)*
      Stage two tracks $N^2$ almost exactly to N = 500 — 1.00 / 4.03 / 8.66 / 25.45 against a
      theoretical 1.00 / 4.00 / 9.00 / 25.00 — then jumps more than an order of magnitude
      between 500 and 750 and stays there. The **existence and location** reproduce across
      runs and independently in the larger-grid study; the **magnitude** does not (563× then
      789× at N = 1,000). Candidates, none tested: a cache boundary (the matrix is ~4.5 MB at
      N = 750), a threading threshold, or an allocation path past a footprint. Until this is
      understood, §3.3.1's quadratic claim is confirmed only to N = 500.
- [ ] ⬜ **C3 — Ch 5 end-to-end FIS result.** Every Ch 5 number is a *clustering* score; the
      chapter exists to produce FIS antecedents. Until a model is built from them and measured,
      the central claim rests on a proxy. **Recommend pulling a minimal version into 2027 Q2**
      rather than leaving it all in the 2028 Q1 capstone alongside G6/G7/G8/write-up/defense.
- [ ] ⬜ **C4 — Quantify the correction-rule pass** (Ch 4 §4.3.1). Claimed, never measured.
      Paired confusion matrices, before and after. Fills Fig 4.3.
- [ ] ⬜ **C5 — Ch 3 head-to-head vs. eVAT (Meng & Yuan 2018) and clusiVAT** on identical
      datasets. First comparison a reviewer will demand.
- [ ] ⬜ **C6 — Ch 5 head-to-head vs. Bonis–Oudot beta-plateau and AuToMATo** on identical data.
      Defensive as much as scientific, given how close that work is.
- [ ] ⬜ **C7 — Ch 6 Atwood machine result** (Table 6.4 pending row), and reconcile Table 6.4's
      R²/RMSE pair — 0.92/0.045 implies target σ ≈ 0.159, 0.96/0.028 implies ≈ 0.140. It is also
      the one table `PROVENANCE_MAP` marks ungenerated while Ch 6 calls it the clearest result.
- [ ] ⬜ **C8 — Ch 3 datacenter GPU re-run.** The pairwise-distance kernel loses (<1×) at low
      dimension / float64 on a consumer card; the prediction that full-rate FP64 flips it is
      untested and labeled as such.
- [ ] ⬜ **C9 — Ch 6 interpretability, measured** (G6): rule counts, path lengths, and either an
      established metric or a small expert study. Fills Table 6.3's pending row. Until then Ch 6
      must keep saying the payoff is *described*, not quantified.

## D. Writing and figures

- [ ] ⬜ **D1 — Produce the remaining figures.** One of fifteen now exists (Figure 3.2, the
      complexity fit, generated by the harness in PNG + EPS). The rest are unstarted. Two are load-bearing: **Fig 1.2** (pipeline
      roadmap, orients the document) and **Fig 5.2** (band discovery, carries Ch 5's
      contribution). Largest visible gap in the draft.
- [ ] ⬜ **D2 — Write Chapter 9.** Still an outline; §3.3.1, §3.4, Ch 1 and Appendix A.3 all
      forward-reference §9.3. Blocked in part on A2.
- [ ] ⬜ **D3 — Regenerate `chapters/00-README-master-outline.md` from the prose.** It still reads
      "Status: Scaffold," describes pillar 1 without stage two or the name collision, and lists
      two already-completed fixes as pending. It is the document a committee member is most
      likely to open first.
- [ ] ⬜ **D4 — Fill the remaining 22 `*pending*` table cells**, or mark them honestly.
- [ ] ⬜ **D5 — Install a LaTeX engine** so display math typesets:
      `sudo zypper install texlive-xetex texlive-latex texlive-collection-fontsrecommended`.
      `build_pdf.py` auto-detects and switches.
- [ ] ⬜ **D6 — Rebuild the PDF.** `build/` is from 07-31; Ch 1, 3, 4, 5, 6, 7 and the appendix
      have all changed since.

## E. Decisions and framing

- [ ] ⬜ **E1 — t-norm: present min/max as the default.** *(Author decision recorded 2026-08-02:
      keep tables at factory/library defaults, show the better configuration alongside, treat as
      future work.)* Upstream `53e89ab` made *probability* the library default. Data: min/max is
      nominally best for the flat MoG (0.651 vs 0.650) but by 0.001 against σ ≈ 0.05, so the case
      is simplicity rather than accuracy. **The reportable finding is that Łukasiewicz collapses
      the regression models** (−3.761 flat, −3.626 HME) while the other four families sit within
      0.03. Also: the whole norm/conorm study appears in *no chapter*, while Ch 2 §2.1 promises
      "Chapter 4 shows" something Chapter 4 does not show — harvest it or drop the reference.
- [ ] ⬜ **E2 — Capture the Borůvka / GPU work.** Table 3.3 has no generator; fp16 was
      deliberately scoped out of the CPU memory table on the grounds it belongs here; the
      datacenter-FP64 prediction is untested; and the exact-GPU-engine standalone paper is
      flagged in §9.4. None of it is in the harness.
- [ ] ⬜ **E3 — Schedule or explicitly defer G8.** Ch 7 assigns it 2028 Q1 and one quarter of
      effort; Ch 10's Gantt and quarter grid omit it entirely. 2028 Q1 already carries the
      capstone, G6, G7, writing and the defense.
- [ ] ⬜ **E4 — Bound the Magdalena/G8 tension.** Ch 5 §5.5 and Ch 6 §6.2 both concede joint 2-D
      memberships approach what Magdalena's condition forbids, and both defer to stretch goals
      scheduled after the defense. Consider stating in §6.2 that the chapter's claim holds
      *without* G8, so the hole is bounded rather than pending.
- [ ] ⬜ **E5 — Decide the SHAP question** (§2.6). The position is argued and explicitly
      untested, in the section that justifies the dissertation. Either scope a minimal
      comparison into G6, or reframe from "post-hoc is worse" to "post-hoc answers a different
      question," which needs no experiment.
- [ ] ⬜ **E6 — Give the Concrete benchmark one canonical citation.** Ch 4 and Ch 6 now agree
      cell-for-cell and all 35 values trace to harness CSVs, but no chapter yet *cites* a named
      table instead of restating values — which is the mechanism that let the numbers drift in
      the first place.
- [ ] ⬜ **E7 — Two literature searches**: knot/breakpoint optimization precedent (Ch 6), and a
      dedicated fuzzy-MoE search to bound the HME nesting claim. Plus the Zhang-2023 attribution
      fix in the HFIS references (misattributed to "H. Wang et al.").
- [ ] ⬜ **E8 — Two blocking reads** before writing the Ch 9 complexity note: **Deshpande & Kumar
      2024** full text and **Wang et al. 2010** (PAKDD). If the former already states the
      O(N)-workspace result for VAT itself, drop the note.

---

## Done this session

<details><summary>26 findings fixed — expand</summary>

**Chapter 3.** Scaling paragraph reworked into the two-stage history (the 2.56 s ↔ 0.265 s gap
is stage one vs stage two, 9.7×, matching the three-arm's own 7.5–9×); 8,000× projection
deleted; three baselines separated; Table 3.1 re-quoted, then converted to worst-case ratios;
§3.3.2 split into in-place (built) vs matrix-free (not built); **Table 3.2 replaced with a new
generator** pairing exact memory arithmetic with a measured cross-precision ordering check;
hardware corrected to 96 GB with a 64 GB working cap; both machines now labeled.

**Chapter 4.** Retracted quantile recommendation deleted and section reordered; Tables 4.1–4.5
and §4.4 re-quoted against `main`; Table 4.2's retraction re-argued (largest gap 0.012 → 0.022,
and the sign pattern reversed); tail claim withdrawn; Table 4.5's MoG row split so accuracy and
training time come from the same configuration; `Slag`+`Age`; "second order barely helps"
corrected.

**Chapter 5.** Granularities filled from the harness ([6,2], [4,2]); 0.00 → 0.001; Table 5.3
gains a coverage column and the chapter concedes bottleneck-bootstrap is the real competitor.

**Chapter 6.** Table 6.1 rebuilt as an architecture × configuration grid; phantom 0.805/0.875
removed; PhiUSIIL reversal fixed in two places; refinement table re-quoted; configuration
paragraph re-quoted and re-attributed; Table 6.2 re-quoted and its purpose narrowed.

**Cross-cutting.** Bibliography false alarm removed and six missing keys added; six count fixes;
G5 marked reopened in two documents; estimates-vs-demonstrations standard written into G4 and
Appendix A.5; four notes logged in `ACTION_ITEMS.md`.

**New results.** float32 in-place reaches 126,491 points under the cap and 154,919 on the full
machine, with an ordering elementwise identical to float64 across ten seeds. Chapter 6's
conclusion moved to *level* (tuned mixture 0.833 ± 0.024 vs flat 0.824 ± 0.043 at matched
capacity), which is what the chapter always wanted to argue. Refinement's decay sharpened to a
factor of twenty-five across consequent orders.

</details>
