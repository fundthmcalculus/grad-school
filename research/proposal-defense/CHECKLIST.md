# Action items & burn-down checklist

Shared working list — merged 2026-08-04 from the former `ACTION_ITEMS.md` and `CHECKLIST.md`
into one document, since the two had grown into near-duplicates of each other with
inconsistent bookkeeping (miscounted table totals, a stale bibliography tally) between them.
Consolidated with timeline information from `NEXT_STEPS.md` on 2026-08-08.

Tick items as they land; each has enough context to start cold. IDs (A1, B6, C1, D2, E7, …)
are load-bearing — several are cited by ID from the prose chapters (e.g. Ch 7 §7 cites
**C1**, Ch 6 §6.4 cites **C9**, Ch 9 cites **D2**/**E8**/**E2b**) — so numbering is stable
across edits; a retired item keeps its ID rather than freeing it for reuse.

**Proposal defense: December 2026 · Final defense: March 2028.**

Companion doc: [`REVIEW_2026-08-02.md`](REVIEW_2026-08-02.md) (what was found and why,
dated snapshot — not updated after the fact). Chapter 7 §7.5's Table 7.1 is the canonical
tracker for the research goals themselves (G1–G9, C1, C3, M5); this file does not duplicate
that table — items below that correspond to a Chapter 7 goal say so and point at it rather
than re-describing it.

_Opened 2026-08-02, merged 2026-08-04, consolidated with timeline 2026-08-08._
_Overnight reproduction pass against latest `main` and latest submodules, 2026-08-22._
_Second overnight pass, 2026-08-27: submodules synced to their upstream heads, **B14** closed,
**B18** opened and fixed, **E9** answered, **E11**'s citation half closed, and five items found
to have been done for days without being ticked (**B16(a)**, **B16(b)**, **B16(d)**, **D7**,
**E10(d)**). Outstanding items **28 → 18**: open 25 → 12, the one 🔒 cleared, and in-progress
2 → 6 because four items (**C1**, **D8**, **E1**, **E11**) moved from *not started* to *half
done* rather than closing. The recurring lesson is the one
**B13**'s standing procedure already names — nobody reads the upstream log for what a bump
**fixes** — so this pass added the missing half of it: read the log for the revisions that will
actually be **imported**, which is not the same set as the submodule SHAs._

_⚠️ **Two passes ran over this file on the same day and did not know about each other**, which is
its own finding. Merging them produced three genuine conflicts, not textual ones: both had
independently concluded **D7** was done (the accounts are complementary and both are kept), both
had worked **E10**, and both had repaired the black gate #183 left red. It also produced the
better outcome twice over — the other pass landed **C1**'s ANFIS and GA-FIS adapters, the item
this file calls the single most important experiment in the backlog, and reorganized §5.3 for
**E10(c)**. A shared file with two writers wants a lock as much as `reproduce/outputs/` does._

> ✅ **B14 is closed: the pin carries the fix** (2026-08-27). `stats_numba.wasserstein_distance`
> weights its CDF gaps by $dx$ again, so it is once more a distance in the data's own units.
> Settled by ancestry and by behaviour rather than by reading a diff — `5253aa0`
> (`tribble-fis` #171) is an ancestor of the current pin, and preflight's `W1-SCALE` check
> multiplies both samples by 1,000 and gets 1,000× the distance back. Chapter 4 and Chapter 6
> accuracy numbers are quotable from this pin; the ordinary re-quoting that remains is **D8**.
>
> ⚠️ **B18 — `PROVENANCE.txt` was one archive away from naming code that had not run.**
> `tribble-fis` resolves `tribble-clustering` and `optimizers` from its own lockfile, and the
> archives stamp the submodule SHAs; for nine days those were different revisions. Every
> existing archive checks out — the audit is in B18 — so nothing already written is in doubt.
> The next one would have been the first that was. Fixed upstream, and preflight's new
> `PIN-MATCH` check now fails the run rather than trusting the two to agree.

**Legend: ⬜ open · 🟨 in progress · ✅ done · 🚫 descoped · 🔒 blocked on you.**

---

## Suggested order (from Tier 0 through Tier 4)

**Where this actually stands, 2026-08-27.** Twelve items open and six in progress, and only
three are things a committee would notice: **C3** (Chapter 5's claim still rests on a clustering
proxy), **C9** (interpretability described, not measured) and **E2b** (Table 3.4's FCM row
overstates the GPU by roughly an order of magnitude). The rest are research scheduled after the
defense, or editorial.

**C1 changed shape today rather than closing.** The ANFIS and GA-FIS adapters landed, so the
title's *orders of magnitude faster* finally has something to be faster *than* — but the run is
still owed, and it is a scheduled job rather than a background one: both baselines build a
256-rule grid partition on Concrete's eight features, which costs 5–6 minutes per seed on the
smallest of the five datasets. Build a slot for it; do not shortcut it to three seeds.

Infrastructure is no longer the constraint it was at the start of August: the B-section closed
seven items this month and has one substantive gap left, **B10**'s fp16/GPU capture.

**This week:** the three above, in that order.

**Before the proposal defense (December 2026):** finish Tier 1 items. Get G4's protocol defined even if not fully executed — every number in the document is reported under it, so the committee will ask.

**After:** Tier 2 (real research, 2027 Q1–Q4, 2028 Q1) in the scheduled order, with **G2 started early** since it has no upstream dependency and is the riskiest to leave late. See G2 appendix below for verified non-coordinate datasets (Crop 24K, ElectricDevices, etc.).

**Tier 3 (defensibility, before submission)** and **Tier 4 (editorial, low stakes)** round it out.

---

## A. Author decisions — settled 2026-08-02, one reopened since, one new
**[Tier 0: all cleared by 2026-08-02]**

_Kept as the record of what was decided and why, since several of these changed the document
materially and a committee may ask. **A9** is the reopened item (narrowed 2026-08-03); **A10**
is new, folded in from the former `ACTION_ITEMS.md`'s "needed from author" section._

- [x] ✅ **A9 — SETTLED IN THE PROSE (2026-08-22): option C in §4.3/Table 4.1, option A
      everywhere else — the reasonable default this item itself recommended.** Table 4.1 already
      carried all three arms (raw / log+min-max / log+z-score), which is C. What was missing was
      A: Chapter 6 still said *"cement ≥ 0.42 after standardization"* and *"the
      log-and-standardize preprocessing"* four times, in direct contradiction of §4.3's own rule
      that the one thing this transform is not is standardization. Chapter 6 now follows it.
      ⚠️ **Two figures this item quotes are stale against the prose it describes.** §4.3 now
      records the first-order z-score arm at $0.713 \pm 0.035$, not the $0.087 \pm 0.089$ below,
      and *withdraws* the −0.407/−0.524/−0.634 component sweep rather than re-taking it — the
      effect those numbers characterised was the output partition, not the transform. Read the
      body below as the record of what was decided, not as current measurements; **E9** carries
      the same correction. Original entry:

  ↳ _Record of the decision, as written while it was open._ **A9-orig.**
      *(Opened 2026-08-03; narrowed the same day. **Author confirmed min-max/`UnitScalar` is
      correct**, which rules out option B below — the transform the samples now use is the one
      Ch 4 and Ch 6 have measured all along, so no number in the document is in question and
      nothing here is blocking. What remains is purely how the prose labels it. The empirical
      follow-up on *why* bounded beats centred is tracked separately and at low priority as
      **E9**. Full write-up with costs: `reproduce/outputs/NORMALIZATION_THREE_ARM.md`; measured
      facts: `PROVENANCE_MAP.md` note 16; data: `reproduce/outputs/norm-three-arm-a385a1a/`.)*
      **The concession.** `gauss_math.standard_transform` — behind every "log+std",
      "standardized" and "normalized" number in Chapters 4 and 6 — computed
      `(X − min)/(max − min)`, i.e. **min-max to [0,1]**, never z-score, despite the name. So
      **log + z-score had never been measured.** It has now been, at ten seeds, as Table 4.1's
      third arm.
      **The result is favourable, which is what makes this cheap.** Min-max is best-or-tied in
      8 of 9 rows. Under genuine z-score the 1st-order flat MoG drops to **0.087 ± 0.089**,
      *below raw features* (0.646 ± 0.039); the demo-tuned mixture of experts falls 0.834 →
      0.706. So the mislabel was lucky: the code did the right thing under the wrong name, and
      relabelling costs **no numbers at all**. Ruled out as explanations: ridge scale (sweeping
      `l2_reg` 1e-2 → 0 moves the gap 0.001) and the scale-dependent BIC membership-count choice
      (identical pinned rule bases still give −0.407/−0.524/−0.634).
      **The control that licenses believing it:** CART, Random Forest and both fuzzy-tree rows
      move ≤ **0.002** between the two normalized arms, against ±0.018–0.056 seed spreads —
      exactly as required, since both transforms are monotone and those models split on rank.
      **The migration underneath it is also done and re-verified** (2026-08-03): `tribble-fis`
      is pinned at `a385a1a`, and the deleted `gauss_math.detect_and_apply_log_transform` /
      `standard_transform` helpers are replaced by `UnitScalar` (min-max) and `StandardScalar`
      (z-score) in `tribblefis.scaling`. The migration moved no number —
      `UnitScalar(log_dynamic_range=2)` is bit-for-bit identical to the deleted pair
      (`max|diff| = 0.0`; 256 cells across four tables byte-identical at ten seeds against
      `outputs/full-14900hx-r2/`). Thirteen files depend on the deleted helpers, not seven as
      first counted — the missed six import `log_transform`, deleted in the same upstream PR
      (`tribble-fis` [PR #67](https://github.com/fundthmcalculus/tribble-fis/pull/67)) — and
      `_fuzzy_models.py` was not "comment only": its `normalize()` called both deleted
      functions directly.
      **What you are choosing between** (full costs in §4 of the findings file):
      **(A)** relabel to "log + min-max to [0,1]" and keep it the default — no numbers change,
      and it *fixes* two sentences, since §6.3.2's and §4.3's `"cement ≥ 0.42 after
      standardization"` is a valid min-max value and an impossible z-score one;
      **(B)** switch the default to z-score — re-quotes most of Ch 4 and Ch 6, inverts §4.3's
      headline, and breaks the `[0,1]`-target assumption behind the extreme-bucket-mean pin
      (**not recommended; no measurement supports it**);
      **(C)** report both arms side by side in §4.3 / Table 4.1 — costs one wider table and a
      paragraph, and upgrades the finding from "normalization helps" to the sharper *"bounded
      normalization helps, centred normalization does not, and the bounded-input assumption is
      load-bearing"*.
      A reasonable default if you don't want to think about it: **C for §4.3 and Table 4.1, A
      everywhere else.** One terminology trap either way: Ch 5 uses "the minimax transform" in
      the unrelated iVAT bottleneck-ultrametric sense, so write "min-max scaling to [0,1]" or
      "unit scaling", never bare "minmax", in Ch 4/6.
      **Nothing in the prose is false**, only mislabelled — a sweep for `z-score`, `zero mean`,
      `unit variance`, `μ=0`, `σ=1`, "divide by the standard deviation" and related phrasings
      across `prose/*.md` returns nothing; the text never states the arithmetic. **No prose
      label has been changed**, deliberately. When you pick, `build/proposal-combined.md` needs
      a rebuild rather than a hand-edit.
- [x] ✅ **A10 — Teaching/RA load: none. RESOLVED (author, 2026-08-21).** The author carries **no
      teaching or RA load** over the runway, so the throughput assumption behind Chapter 10's
      timeline is full research effort — the one open item that could have moved every bar in the
      Gantt resolves in the favourable direction, and no bar needs to move. `10-timeline.md`'s
      intro now states the effort assumption explicitly, which is what a committee asks of a
      schedule described as "deliberately aggressive," rather than leaving it implicit.
- [x] ✅ **A1 — Method name settled: `mergeVAT`** (author decision, 2026-08-02).
      The name went round-trip: mergeVAT → `pVAT` (on Dr. Kreinovich's observation that stage one
      is a priority-queue algorithm) → collision → back to **mergeVAT**. `pVAT` is taken by
      Parveen & Sreevalsan-Nair, *"pVAT: Parallel VAT on the GPU"*, BDA 2013 (LNCS 8302:151–170),
      a GPU VAT that also swaps the MST algorithm, so reading our *p* as parallel/performant
      collided harder rather than less. Acknowledged by citation in §3.3.1.
      ✅ **The pVAT collision is genuine; the citation had the wrong title, now fixed (2026-08-21;
      deep research, see E7).** "pVAT" is not a paper title — it is a method named inside Parveen &
      Sreevalsan-Nair's *"Visualization of Small World Networks Using Similarity Matrices"* (BDA 2013,
      LNCS 8302:151–170, DOI 10.1007/978-3-319-03689-2_10), whose **Algorithm 1 is captioned "pVAT:
      Parallel implementation of VAT"** — a GPU/CUDA parallel VAT built on **Borůvka's MST**
      (contribution 1, confirmed against the authors' preprint). So the prior use of the name is real,
      and it collides *harder* than first stated: their pVAT and this work's GPU path both swap Prim
      for Borůvka. The `.bib` had lifted the algorithm caption as the paper title and carried the wrong
      given name ("Sherin"); corrected to the real title and "Saima", restored to `[V]`, DOI added.
      §3.3.1's prose already describes it correctly (it never used the bogus title). No journal version
      exists — the BDA chapter is the only publication. **The rename to mergeVAT is fully supported.**
      **The name is imperfect and §3.3.1 says so**: it describes neither stage one (priority
      queue) nor stage two (compact active set). What it does describe is §3.3.4's
      divide-and-conquer stitch — which is a merge, is measured, and is the least finished part
      of the method. Kept because a stable imperfect name beats a third rename while the work
      underneath settles. See **C10** for the open merge questions.
- [x] ✅ **A3 — Flagship dataset: the UCI shuttle set (58K)** (author, 2026-08-02).
      *Decision: "UCI-shuttle (58K) for the sake of convenience and history. We might change it later."*
      Chosen for continuity — it is the set Ch 3 already uses to demonstrate scale, so one dataset
      runs from reorder through membership generation to the final rule base. Also public (the
      135K psychiatric set is not), so the capstone is third-party reproducible, and its ~80%/7-class
      imbalance exercises the complement rule of §4.3.5 rather than leaving it described.
      ⚠️ **Caveat written into §7.3:** the shuttle set *has coordinates*, so the capstone does **not**
      exercise the coordinate-free regime Ch 5's premise rests on. The capstone and **G2** answer
      different questions and neither substitutes for the other — "we ran the pipeline on shuttle"
      must not be mistaken for having closed G2. Revisitable: switching to the IoT sets later costs
      a re-run, not a redesign.
- [x] ✅ **A4 — EUSFLAT 2027 deadline confirmed: February 2027** (author, 2026-08-02);
      conference September 2027. *Decision: "2027-02".*
      ⚠️ **This broke the Ch 10 schedule and the fix is a decision, now taken.** The grid had the Ch 5 paper in 2027 Q2 and Goal **G1** — the differentiator §5.5
      names — also in Q2. A February deadline is Q1, so the paper was scheduled a quarter late
      *and* its headline contribution would not have existed when it was due. Resolution: submit
      what §5.4 already supports (multi-scale recovery at ARI 1.00 vs 0.58–0.75 flat, the
      selection bake-off, the falsification experiment) and make G1 the journal/next-year
      extension. Cost: the EUSFLAT paper reports clustering scores rather than
      end-to-end accuracy — the same proxy limitation §5.4 already concedes.
- [x] ✅ **A6 — Acknowledgements written** (author, 2026-08-02). Template replaced with the real page; it renders correctly ahead of Chapter 1 in the build. One residual dependency: it thanks the committee and Jon Salisbury by name but not the NAFIPS co-authors, so give it a second pass once **D2** supplies those author lists.
- [x] ✅ **A5 — Proposal defense confirmed: December 2026.** *Decision: "Dec 2026. Let's GO!"*
      Hedged wording ("assumed ~Dec 2026") removed from Ch 7 Table 7.1, Ch 10, and this file.
      Final defense stays March 2028, so the runway is 15 months as planned.

## B. Reproducibility infrastructure
**[Tier 0–2: mix of completed and ongoing infrastructure]**

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
- [x] ✅ **B8 — Automate the harness → document figure hop. DONE.** `save_figure()` writes to
      `reproduce/outputs/figures/fig_03_complexity_fit.{png,eps}`; the document references
      `prose/fig/03-complexity-fit.png`. `build_pdf.py` now closes this automatically: `main()`
      calls `copy_figures()` (unconditionally, as its first step), which walks the `FIGURE_COPIES`
      name map — sourced from `reproduce/figures/registry.py`, so adding a figure stays a one-file
      edit — and copies each `fig_*.{png,eps}` to its prose name at build time. It still emits an
      image line only when the target exists and strips it to a placeholder when it does not, so
      the "figure silently reverts to a placeholder" failure mode is now governed by whether the
      harness generated the figure, not by a forgotten manual copy.

      ⚠️ **One gap, found 2026-08-22.** The copy is *unconditional*: `build_pdf.py` re-installs
      `03-complexity-fit.png` from `reproduce/outputs/figures/` on every build, so any local run
      of `table_3_1_reorder_three_arm` silently swaps that figure into the document whether or not
      Table 3.2 beneath it has been re-quoted from the same run. That bit this pass: the figure
      regenerated from gcc-built kernels (fitted exponent 1.77) would have sat directly above a
      table quoting 1.97 from an MSVC build. Reverted by hand, twice, because the build re-copies
      it. The automation is right; what it needs is to refuse, or at least warn, when the figure
      it is about to install came from a different archive than the table it illustrates.

- [x] ✅ **B4 — Submodule SHA guard. DONE.** The harness now loudly stamps when a submodule SHA
      differs from the last archive's, taking the "loudly stamp" branch rather than refusing to
      emit (an intentional submodule change should not block a run). `run_all_tables.sh`'s
      `check_submodule_shas()` compares the most recent archive's recorded `tribble-fis` /
      `tribble-cluster` / `grad-school` SHAs against the current `git rev-parse HEAD` and prints a
      boxed divergence banner on any mismatch; it is called before the run. **This failure had
      happened twice** — once with `fix/pin-extreme-bucket-means`, once with `resolve-flm-pr` —
      which is what the guard now catches.
- [x] ✅ **B5 — Chapter 3's timing grid re-taken on the workstation.** Run of record
      `reproduce/outputs/full-14900hx-r2/` (i9-14900HX, 32 logical cores, 96 GB, RTX 4080
      Laptop), 10 seeds, all 13 generators green in one pass; §3.4 no longer spans two hosts.
      See `reproduce/PROVENANCE_MAP.md` note 11. Three outcomes, two of which cost the chapter
      something:
      **(a) The ~45–50% swing was thermal and laptop-specific.** Three runs here put the
      1,024-point classical arm at 13.7 / 14.2 / 14.5 s — a 6% spread against the laptop's
      22.2 / 31.7 / 21.3 s.
      **(b) The ratio is NOT machine-invariant, which §3.4 asserts it is.** The 1,024-point
      speedup reads 1,129× on the laptop and 660–700× here — a 40% move — because the
      classical arm is interpreted Python and mergeVAT is compiled, so a change of host does
      not scale the two arms by the same factor. The report-ratios-not-seconds standard
      stands; its justification needs weakening from "ratios survive a change of machine" to
      "stable within a host, and far more portable than seconds". **§3.4's paragraph beginning
      "Why this table reports ratios and not seconds" needs that edit.**
      **(c) Table 3.1's swept ratios must be re-quoted**: 28.8× → 28.6×, 398× → 304×,
      1,129× → 660× at 10 seeds on one host.
      This resolves what an earlier note called "two machines mixed in Chapter 3": the
      `main-d0efefc` suite (including Table 3.1's swept timing grid) had run on the
      development laptop rather than the workstation, while Table 3.2's memory ceilings, the
      58,000- and 135,000-point reorders and the GPU rows were already workstation results.
      §3.4 now labels both machines explicitly wherever a number could be read as coming from
      either.
- [x] ✅ **B5b — §3.4's swept rows and Table 3.2 re-quoted from the workstation run.** Table 3.1
      reads 25× / 311× / 673× (was 28.8× / 398× / 1,129×); the 4,096-point stage-two figure is
      0.229 ± 0.006 s, so the comparison against the published stage-one measurement reads ~11×.
      Table 3.2's grid and exponents are re-quoted at classical **3.15**, stage one **1.86**,
      stage two **1.97**. Every cell was checked programmatically against the archive CSVs. The
      plateau and parity-band paragraphs are rewritten as an explicit retraction rather than
      replaced silently, and Ch 7's G4 is rebuilt around the lesson — repeatability cannot
      distinguish a property of the code from a property of the host. The appendix's hardware
      bullet and §3.4's two-hosts note are updated to match. (The 64 GB → 96 GB hardware
      correction — the host is a 32-core i9 with 96 GB RAM, and 64 GB is a self-imposed working
      cap, not a hardware limit — is folded into the same appendix passage and into Table 3.3.)
- [x] ✅ **B5c — Install a PDF renderer on this host.** ✅ DONE (2026-08-08). `build_pdf.py` now renders the PDF; LaTeX engine installed.
- [x] ✅ **B6 — `pvat.vat_prim_mst_seq` fixed and implemented. RESOLVED (author, 2026-08-21).**
      Resolved by *fix*, not removal: the function now computes the correct VAT/MST ordering and
      `tribble-cluster` carries a `test_vat_prim_mst_seq.py` regression test plus a compiled fast
      path, so the silent-wrong-answer hazard this item opened against is closed.
      ⚠️ **The fix landed *after* the pinned commit (`tribble-cluster e3c27e6`).** So the code is
      resolved but the dissertation still pins the earlier commit where the function is the removed,
      chance-level negative result — which means Appendix A.6, Table 3.3's negative-result row, and
      **Goal G4d** (the matrix-free reorder, whose whole premise is "no working matrix-free
      implementation exists") describe the *pinned* state accurately but are now overtaken by the
      code. **Downstream decision, tracked at G4d:** bump the pin to include the fix and re-run —
      Table 3.3's negative row becomes a positive result, the reachable-N ceiling likely moves past
      155k, and G4d's decision-rule (§7 line 51: elementwise ordering check at N∈{1k,2k,5k},
      reachable N under the memory cap, the wall-clock threshold) can actually be run, turning G4d
      from a *build* into an *integrate-and-measure* — **or** keep the current pin and add a "since
      fixed upstream at `<commit>`" note to A.6/G4d. Author to choose; the fix itself is done.

      ✅ **RESOLVED 2026-08-22: the choice is moot and the measurement is taken.** The pin had
      already moved. Settled by ancestry, not assumption: `c9be437` (the fix, 2026-08-10) is an
      ancestor of the current submodule pin `635ed6e` and is *not* an ancestor of `e3c27e6` or of
      the old pin `85b68a8`. So `e3c27e6` — which §3.4's verification permalinks still point at —
      genuinely lacks the fix, and it is genuinely not what the harness runs.
      **G4d's decision rule was then run in full** (`reproduce/experiments/check_matrix_free_reorder.py`,
      outcomes registered in its docstring before the run): ordering $1.000 \pm 0.000$ at
      $N \in \{1{,}000, 2{,}000, 5{,}000\}$ over ten seeds each against chance levels of
      0.0010/0.0005/0.0002, with no run showing the old ascending-index signature; peak working
      set flat at **64.7–65.2 MB** from $N = 2{,}000$ to $12{,}000$ while the implied matrix grows
      36× (materialising arm: 193.6 MB → 4.67 GB); wall clock **0.14–0.22×** the materialising
      arm, i.e. passing the "more than an order of magnitude slower" threshold in the opposite
      direction. At float32 the ordering is $0.9996 \pm 0.0012$ — tie-breaking, per §3.2, not error.
      **Two residual items, both small:** the 155,000-point figure is an *extrapolation* from a
      ratio stable to 1.62× across an 8× change in $N$, since the in-place arm's matrix is 96 GB
      at float32 and cannot be run here — one at-scale run on a bigger host closes it. §3.4's source
      permalinks are **re-pinned** to `635ed6e` (done 2026-08-22): `pcvat.pyx`'s two kernels sit at
      identical line numbers in both commits, but `vat_prim_mst` moved 141 → 159, so the cited
      range is now L159–L226.
      Prose updated: §3.1, §3.3.2, Table 3.3, §3.4, A.6 and Ch 7's G4d entry.
      **Original defect (kept as the record):** the exported API silently returned a wrong
      ordering — the seed vertex followed by every other vertex in ascending index order,
      chance-level agreement (0.001 ± 0.001) with the true ordering at both float64 and float32.
      Cause: `_get_dist(samples, u, vertices[mask])` was typed for scalar indices, so
      `np.sum(np.square(diff))` reduced over *all* candidates and returned one scalar;
      `key[mask] = <scalar>` gave every candidate the same key and the heap popped in index order.
      See `REVIEW` ★2.
- [x] ✅ **B9 — Backfill `log_features` into the sample scripts.** ✅ DONE (tribble-fis #73 merged, 2026-08-08).
      The samples were converted onto `UnitFuzzyScalar` (PR #55 here), which auto-detects log
      columns by dynamic range, whereas each sample previously named its own columns. Upstream
      #73 adds `log_features=[...]`, and the lists restore each sample's original logged set — turning behaviour-*changing*
      conversions into behaviour-*preserving* ones.
      **This is not a mechanical sweep. Two files are deliberate exceptions.**
      **`concrete_trapz.py` — do NOT backfill.** *(Author decision, 2026-08-03: "I like the
      improvement.")* Under auto-detection it **improved in 9 of 10 rows**, +0.023 to +0.123 R²
      (Gaussian 2-full 0.854 → 0.915; Trapz 1st 0.692 → 0.815; only 0th order regresses, −0.099,
      and that row has no consequent to fit). Its original list was `['Slag','FlyAsh','Age']`,
      which is exactly the set `concrete.py` logs and exactly the set no dynamic-range threshold
      can select — so `log_features` is what would make it expressible, and expressing it here
      means choosing the worse configuration. It keeps auto-detection. Single split, so re-measure
      at ten seeds before quoting the gain anywhere.
      **`nasa.py` — cannot be fully restored.** Its column set is recoverable (`Rad Flow`,
      `Fpv Close`, `Fpv Open`, `High`, `Bypass`, `Bpv Close`, `Bpv Open`) but its *order* was
      normalize→log, which the scaler cannot express. `log_features` restores the set, not the
      order, so this stays a deliberate behaviour change either way. Do not let the list imply
      otherwise. Unverifiable in principle — `load_data()` fetches Statlog Shuttle over the wire.
      **Lists for the rest** (recovered from `feat/normalization-migration:FuzzySystemsExperiments/`):
      `beth.py` → `["timestamp","processId","mountNamespace","eventId","userId"]`;
      `beth-anomaly.py` → `["processId","mountNamespace","eventId","userId"]`;
      `iot.py` → the 12 rate/inter-arrival columns (`fwd_pkts_per_sec` … `flow_duration`);
      `wine_red.py` → `["total sulfur dioxide","free sulfur dioxide","chlorides"]`;
      `phiusiil.py` → two calls, 17 count/length columns then 4 ratio columns.
      `iot-botnet.py` has no recoverable explicit list — re-read its pre-conversion source rather
      than assuming one.
      **Only `phiusiil.py` is verifiable here** (its dataset is in `data/`), and it came out
      result-neutral under auto-detection — accuracy identical to four decimals despite a
      completely different logged set. Check rather than assume that restoring its list is also
      neutral. The remaining four have no local data and cannot be verified either way, which is
      the honest reason to prefer behaviour-preserving lists for them.
- [ ] ⬜ **B10 — Capture the Borůvka / GPU work properly.** Ch 3 §3.3.3 and Table 3.4 are the
      thinnest part of the chapter: the GPU rows have no generator of their own history
      (`PROVENANCE_MAP` marks them ungenerated pre-**E2**, needing a device host), and reduced
      precision below float32 was deliberately scoped *out* of the CPU memory table (Table 3.3)
      on the grounds that half precision belongs with the Borůvka/GPU path, where it would
      actually pay. That makes the GPU side the natural home for: the fp16 distance/MST
      question, the datacenter-FP64 re-run **C8** already tracks, and the exact-GPU-engine
      standalone paper flagged in §9.4. Currently none of the fp16 question is captured in the
      harness.
- [x] ✅ **B11 — Reproduction harness in place.** `reproduce/` is the single entry point:
      `run.py` orchestrator plus a growing set of registered experiments across the four
      submodules (command, environment, datasets, hardware tier); `run_all_tables.sh` drives
      the table generators, each of which emits Markdown + CSV with mean ± std over a fixed
      seed set, and reports what it cannot run rather than substituting a guess. Remaining
      work is the ANFIS/GA-FIS adapters (**C1**) and re-quoting drifted tables as they're found
      (`reproduce/PROVENANCE_MAP.md` is the place that tracks drift), not harness plumbing.
- [x] ✅ **B12 — Two upstream/harness defects caught and fixed during the 2026-08-11/12 full
      evaluation pass.** (a) `table_a1_feature_scoring.py` and `_mf_dedup.py` /
      `table_4_8_mf_dedup.py` both called `tribblefis.gaussian_classifier`'s
      `MixtureOfGaussiansFuzzyClassifier` / `SequenceClassifier` and
      `gaussian_regressor.MixtureOfGaussiansFuzzyRegressor`, all three renamed upstream to
      `TribbleClassifier`, `TribbleSequenceClassifier`, `TribbleRegressor` at the currently
      pinned `tribble-fis` SHA (`80e98d7`) — both tables were silently failing every run under
      the current pin until this pass. Fixed, signature-compatible, backfilled at ten seeds.
      (b) `run_all_tables.sh`'s own archive step hit the exact mid-run-edit failure mode its
      header already warns about (a concurrent change to the script while it was running
      shifted byte offsets bash reads incrementally, crashing the PROVENANCE-writing heredoc
      with a syntax error) — recovered via the script's own documented `--archive-only` path;
      the numeric phase itself was unaffected. Full account:
      `reproduce/outputs/SESSION_FINDINGS_2026-08-12.md`.

      **(c) That rename sweep was incomplete, and the gap was invisible for the same reason the
      original was.** Swept every generator and study script on 2026-08-22 for upstream APIs that
      no longer exist. Three more files carried them, all confirmed GONE at the pinned SHA rather
      than assumed:
      - `reproduce/tables/table_norm_conorm_matrix.py` → both flat-MoG rows silently `N/A` since
        at least the 2026-08-11 archive. **This one matters most**, because it is the sole
        evidence for **E1**, and the failure is *graceful*: the skip path catches the ImportError,
        prints its reason, emits `N/A`, and the generator exits 0 — so the orchestrator reported
        *ok* and the run reported green with a third of the table empty. A graceful degradation
        nobody reads is indistinguishable from a result.
      - `reproduce/optimizers/structure.py` → imported `gauss_math.standard_transform` and
        `.detect_and_apply_log_transform`, deleted in tribble-fis #67, with **no fallback**, so
        `StructureProblem.__init__` raised before doing any work and took the optimizer structure
        study — §6.3.5's evidence — down with it.
      - `reproduce/regression_scale/mog_top_p_sweep.py` → same rename. Superseded by C13's
        generator, but a superseded script that raises `ImportError` is not superseded, it is
        broken.

      All three fixed 2026-08-22, reusing `_fuzzy_models`' documented successors rather than
      adding second copies that could drift, and keeping the old names as fallbacks so the
      generators still run against an older pin. **The lesson is the same one B13 and Ch 8's tally
      name:** each of these was found by grepping for the *class* of defect rather than by
      re-checking the instances already known. A fix applied to the files you happened to be
      looking at is not a sweep.

- [ ] ⬜ **B13 — Upstream trapezoid-fitter fix: pin bumped to `141596e`, no proposal table
      moved, two sample scripts still owed.** `tribble-fis`
      [#170](https://github.com/fundthmcalculus/tribble-fis/pull/170) fixes a defect in
      `trapz_math_fast.fit_trapezoids_fast` and lowers its default `n_bins` from 50 to 10.
      Merged, and the submodule pin moved `058501f` → `141596e` in this same change.

      **Verified before the bump was committed, not assumed.**
      `reproduce/tables/table_4_1_mog_baselines.py` at ten seeds is byte-identical across the
      bump — R² 0.808 ± 0.030, 0.867 ± 0.031, 0.965 ± 0.001 and both reference columns
      unchanged. Only wall-clock moved (0.14 → 0.15 s), which is machine noise. That is the
      check this item predicted, and it passed.

      **The bump spans 22 upstream commits, not just #170** — the pin was already behind
      `c27e586`. Two that looked risky and are not: #123 renamed the scaling classes but kept
      all four old names as aliases bound to the same class objects (`UnitScalar is
      MinMaxScaler` confirmed), so no import breaks; #138 unified the zero-firing threshold by
      moving IT2/GT2 onto Type-1's existing `1e-6`, leaving the Type-1 path this repo uses
      untouched. The table diff is the empirical confirmation of both.

      **The defect.** The fitter set each region's `a` to `bin_edges[start]`, i.e. the minimum
      of the data it was fitted to, and inset the plateau from there.
      `TrapezoidMembership.evaluate` rises with a strict inequality (`x > a`), so membership is
      exactly **0 at `x == a`** — correct for an open trapezoid, and wrong in that combination:
      the smallest observed value, and everything tied with it, got zero membership from the
      term fitted to describe it. Under the `min` t-norm one such feature zeroes a rule across
      every input. On unit-scaled Concrete 55% of rows sit at FlyAsh's minimum, 47% at Slag's
      and 38% at Superplasticizer's, and **78.6% of held-out rows were covered by no rule at
      all** — answered with exactly `0.0`, a finite value that passes every non-finite check in
      the pipeline, so it read as a bad model rather than a broken one. `partition_output`
      already guards the same hazard on the output side with `edges[0] -= 1e-9`.

      **Scope, checked at the call sites rather than assumed.** The change only reaches callers
      that take the `trapz_method="fast"` **default**:
      - **Still owed — the reason this item stays unticked:**
        `FuzzySystemsExperiments/darwin_comparison.py` (line 113, `member_function="trap"` with
        no `trapz_method`) and the default-method configs in `darwin_quick_comparison.py`. Both
        now run against the fixed fitter and should be re-run, with any quoted numbers
        refreshed. Not done here: neither is wired into `reproduce/`, so neither has an archived
        output to diff against, and the DARWIN data is not on this host.
      - **Not affected — no action:** `concrete_trapz.py`, whose `main()` runs only `"gaussian"`
        and `"trapz"` (the EM fitter); its `"trapz-fast"` branch in `run_model` is never
        invoked. So the `Trapz 1st 0.692 → 0.815` figures under **B9** above come from the EM
        path and stand. `darwin_trapz.py` passes `trapz_method="em"` explicitly.
      - **Not affected — no proposal table touches this path at all:** every table in
        `reproduce/` uses `member_function="gaussian"`. Verified by grep over
        `reproduce/tables/`; there is no trapezoid arm in any of them.

      **Expected direction when the pin does move** (Concrete, ten seeds, 5 buckets, 2nd-order
      consequents, through `solve_tsk_consequents`/`predict_tsk`): the fast trapezoid arm goes
      from test R² 0.121 with 78.6% of rows uncovered to **0.812** with none, and to **0.839**
      at the new 10-bin default — from unusable to slightly ahead of the Gaussian arm's 0.831,
      with roughly half the membership functions. Gaussian numbers are unchanged, so a pin bump
      should move the trapezoid rows and nothing else; if a Gaussian row moves, something else
      changed too and the bump is not the explanation.

      ---

      ### 📌 Standing procedure — every pin bump runs BOTH checks, not just the first

      The check above asks *what did this pin **move**?* That is the half we do. The half we
      keep missing is *what did this pin **fix**?* — and it has now cost us three times:

      | recorded blocker | retired by | how long it sat open after the fix |
      |---|---|---|
      | BETH open-set "needs a one-class protocol, a research decision" | `TribbleOneClassDetector` landing upstream | months; found only when the experiment was attempted anyway (§7.3, note 22) |
      | **D8** — `wasserstein_distance` CDF-gap defect (B14) | tribble-fis #171 `5253aa0` | in the pin, unnoticed |
      | **D8** — `_kmeans_labels_1d` single-start init, which blocked re-quoting Table 4.7b | tribble-fis #191 `353162c` — **the pin itself** | 2 days, and note 22's own lesson had already been written (note 26) |

      `check_prose.py` compares prose numbers against harness output, so it sees a number
      drift. **No check looks in the other direction**: a capability or a defect fix arriving
      in a submodule, silently retiring a blocker the document still records as open. That is
      the audit direction with nothing behind it, and a stale blocker is worse than a stale
      number — it suppresses work that is already unblocked.

      **On every pin bump, before running anything:**

      - [ ] Read the upstream log across the bump for *fixes*, not just changes:
            `git -C tribble-fis log --oneline <old>..<new>` and flag every commit whose subject
            is a fix, a restoration, or a new capability.
      - [ ] Grep this file and `PROVENANCE_MAP.md` for blockers those commits could retire —
            by function name, class name and defect: e.g. `grep -n "_kmeans_labels_1d\|wasserstein" CHECKLIST.md reproduce/PROVENANCE_MAP.md`.
      - [ ] For each hit, **verify in the running environment, not from the diff** — a stale
            wheel will happily serve the old code while the source tree shows the fix:
            ```
            uv run --project tribble-fis python -c "import inspect, tribblefis; from tribblefis import gauss_math as G; print(tribblefis.__file__); print(inspect.getsource(G._kmeans_labels_1d))"
            ```
            (See the reinstall trap: the dist name is hyphenated and `--no-cache` is required,
            or `uv` serves the cached wheel and the sweep re-measures the old code.)
      - [ ] Tick, or annotate with the retiring commit, every blocker the bump has settled —
            **in the same change as the bump**, not later. An untouched blocker after a bump
            should be a deliberate decision, not an omission.

      Worth automating rather than remembering: a script that takes the two pins, lists
      upstream fix-shaped commits, and greps the checklist for each touched symbol would have
      caught all three of the rows above. Until it exists, this list is the procedure.

      ✅ **Run in full for the 2026-08-27 bump (`353162c` → `2047fa1`), and it paid for itself.**
      Five commits, of which #194 retired **E10(d)** — a blocker this file had carried as open
      while the work sat done upstream. The other four are additive (`trapz_math_smooth.py`, the
      deconstructed-tree classes, an `h5py` extra); the one behaviour change, in `em.py`, is
      confined to the **trapezoid** gate M-step, and every generator in `reproduce/` runs
      `member_function="gaussian"`, so it cannot reach a proposal table.
      Two things the procedure did **not** have, both now added:
      - **Read the fix half against the environment, not only the source tree.** **B14**'s fix
        turned out to be an ancestor of the pin already, so the 🔒 that headed this file was
        stale. Ancestry alone would not have settled it — the wheel can lag the tree — which is
        why `preflight.py`'s `W1-SCALE` is the check that closed it.
      - **Diff the whole table, which is the concrete change B13's own ⚠️ asked for.** Now the
        default: the run of record is compared column by column, not on the three values that
        are easiest to eyeball.

      ⚠️ **One more trap this bump exposed, now covered by preflight's PIN-MATCH:** reading the
      log between two *submodule* SHAs is not the same as reading the log between the revisions
      that will actually be imported. For nine days those were different libraries. See **B18**.

      Found in `experiments/overlap-modeling` (stage 4);
      `experiments/overlap-modeling/diagnose_trapz_defect.py` regenerates the three
      measurements above, and `experiments/overlap-modeling/RESULTS.md` §"Stage 4" is the full
      account.

      ⚠️ **The verification above is sound and its coverage was narrower than the conclusion drawn
      from it — see B14.** "Byte-identical across the bump" was established on `table_4_1`'s three
      R² values, which do match, then and now. The same table's two *accuracy* columns were not
      checked, and they had already collapsed (PhiUSIIL $0.997 \rightarrow 0.729$, RT-IOT2022
      $0.927 \rightarrow 0.500$) at tribble-fis #95, inside the same 22-commit window this bump
      spanned. **The concrete change owed: diff every column of the table, not the columns that are
      easiest to eyeball.** This item's own rule was right — *"if a Gaussian row moves, something
      else changed too and the bump is not the explanation"* — and would have caught it applied
      whole-table.


- [x] ✅ **B14 — `stats_numba.wasserstein_distance` was not the Wasserstein distance.
      Fixed upstream and confirmed in the running environment (2026-08-27).**
      `tribble-fis` #171 (`5253aa0`, 2026-08-22) adds the missing $dx$ weighting, and that
      commit is an ancestor of the current pin — checked with `git merge-base --is-ancestor`,
      not inferred from a date. The behaviour is confirmed separately, because an ancestor in
      the source tree and a wheel in the venv are different claims: preflight's `W1-SCALE`
      multiplies both samples by 1,000 and reads back `x1000.0 distance`. The suite is
      6/6 green on this pin. **Chapter 4 and Chapter 6 accuracy numbers are quotable again.**
      Two of the three "owed" items below are therefore discharged: (a) it was filed and fixed
      upstream, and (b) the hold on quoting is lifted. **(c) is the one that outlived the
      defect** — B13's pin-bump check still has to read every column of a table, and that
      change is now written into the standing procedure rather than left as advice.
      _The account below is kept as the diagnostic record; its warnings are historical._

      Found 2026-08-22 by re-running the suite on latest `main` and latest submodules.
      Full account: [`reproduce/outputs/WASSERSTEIN_REGRESSION.md`](../../reproduce/outputs/WASSERSTEIN_REGRESSION.md);
      one-command reproduction: `reproduce/experiments/diagnose_wasserstein_regression.py`.

      **The symptom.** `table_4_1_mog_baselines.py` at ten seeds against the archived run of
      record moves the two *classification* rows by margins no seed spread covers, while the
      three *regression* rows move slightly the other way and every training time falls 5–7×:
      PhiUSIIL $0.997 \pm 0.001 \rightarrow 0.729 \pm 0.023$; RT-IOT2022
      $0.927 \pm 0.002 \rightarrow 0.500 \pm 0.244$. Those are Ch 1 §1.2's and Ch 4 §4.4's
      headline numbers.

      **Attributed, not guessed.** Data frozen to one `.npz` before either library is imported
      (and the loader did not move across the bump anyway); old library $0.9952 \pm 0.0014$
      against new $0.7405 \pm 0.0092$ in the same shell; bisected over the 48 commits in
      `80e98d7..141596e` to first-bad `5237ebe` (tribble-fis #95, *"Replace scipy/sklearn stats
      functions with numba-accelerated implementations"*), parent `ce4a0fc` good; then each
      replaced function restored one at a time at the current pin. Exactly one recovers the
      accuracy: `wasserstein_distance` (0.9947 ± 0.0017). `norm_fit`, `norm_pdf`,
      `jensenshannon_distance`, `silhouette_score` and `_kmeans_labels_1d` are all inert.

      **The defect.** $W_1 = \int |F_u(x) - F_v(x)|\,dx$. The implementation returns the *mean*
      of the CDF gaps over the union support, with no $dx$ weighting — dimensionless, bounded in
      $[0,1]$, and **completely scale-invariant**: multiply both samples by 1000 and scipy's
      answer scales by 1000 while this one returns the identical 0.245960. Against the analytic
      values it is off by 3× to 30,000× depending on scale. Fix is one line.

      **Blast radius.** It feeds `gauss_math._pairwise_label_distance`'s `"composite"` score,
      which *is* the feature-differentiation screen; `mog_classifier` runs `top_n=5`, so a wrong
      score picks the wrong five features. Same metric behind **A.4** and **Tables A.1/A.2**.
      That call site's comment says it "squash[es] the *unbounded* pooled-std-normalized
      wasserstein distance" — it is already bounded, so the squash and the composite's
      three-term balance operate on a quantity they were not designed for.

      **Table 4.1 is not wrong; the pin is.** Re-running the generator at the current pin with
      only this function corrected returns PhiUSIIL to $0.997 \pm 0.001$ *exactly* and
      RT-IOT2022 to $0.923 \pm 0.011$, and takes the table from 2 cells beyond noise to **0**
      against the archive. ⚠️ That is *not* the same as restoring the document: over the whole
      proposal the fix moves `check_prose` from 59 matching pairs to 70, against 156 under the
      old pin. **D8** carries the remainder. `reproduce/experiments/run_with_reference_stats.py`
      re-runs any generator that way, so the question can be settled table by table without
      waiting on upstream. The three regression rows move identically with and without the fix,
      so their drift has a separate and smaller cause, still untraced.

      **Owed:** (a) file it upstream — *not done here; outward-facing, left to the author*;
      (b) do not quote the current pin for any Ch 4/Ch 6 accuracy number until it lands;
      (c) extend B13's pin-bump check to **every column of a table**, which is the concrete
      change that would have caught this — B13 verified three R² values and concluded
      "byte-identical", and the two accuracy columns beside them were not looked at. Ch 8's
      tally already names the lesson: *repetition is not the same thing as coverage.*

- [x] ✅ **B15 — The host lost its C toolchain; the harness now bootstraps one.** `reproduce/`
      could not run at all: the `Microsoft Visual Studio/2022` directory is present but empty,
      and all three submodules carry compiled extensions, so every `uv run --project …` failed
      during resolution before a generator could import numpy. Fixed by
      `reproduce/hostenv.sh` + `tools/ccshim/`, sourced from `run_all_tables.sh`; a no-op on
      Linux/macOS and on any Windows host that still has MSVC. Three upstream/tooling defects had
      to be worked around, each reproduced first:
      (a) **`optional=True` does not survive `cythonize()`** — tribble-opt declares both
      extensions optional so a missing compiler degrades to numba, and Cython rebuilds the
      Extension without carrying the flag (measured: True in, False out). Since Cython is in
      `build-system.requires`, that documented degradation has never been reachable.
      (b) **tribble-opt picks MSVC flags by `platform.system()`, not by compiler**, so gcc is
      handed `/O2 /openmp` and reads them as filenames. (tribble-cluster gets this right.)
      (c) **`DIST_EXTRA_CONFIG` is honoured by `build_wheel` and ignored by `build_editable`** —
      tribble-clustering built fine as a git dependency and failed as the project itself, same
      shell, seconds apart. `UV_NO_EDITABLE=1` forces the wheel path.
      ⚠️ **Consequence for Chapter 3:** the compiled kernels are now built by **gcc**, not MSVC.
      Timings taken here are not comparable to archives taken before, and any Ch 3 timing
      re-quote from this host must say so. This is B5b/§3.4's host hazard, extended to compilers.

- [x] ✅ **B16 — All six sub-items closed (2026-08-27).** Four had already been fixed and were
      simply never ticked, which is its own small lesson: this file was carrying four open
      blockers that were not blocking anything. The other two are fixed upstream now.
      (a) ✅ **Fixed upstream by `clustering` #87 (`1dcf331`, 2026-08-24), and by the better
      route than the one proposed here.** The original diagnosis was right about the symptom and
      wrong about the cause: `numba-progress>=1.2.0` was declared in `[project].dependencies`
      and absent from the lock, so `uv run --project tribble-cluster` re-resolved and *dirtied a
      pinned submodule* on every run — reproduced against the pre-fix tree, where a plain
      `uv run` rewrites `uv.lock`. But the dependency was **dead**: nothing imports
      `numba_progress` anywhere in that repository. Upstream deleted the declaration rather than
      locking a package it never used. Verified at the current pin: `uv lock --check` exits 0,
      and `uv run --project tribble-cluster` leaves `uv.lock` byte-identical.
      **The desync was also wider than recorded** — `c4efd84` added `matplotlib>=3.10.9` and
      tightened `numpy>=2.4.6` without re-locking either, so three declarations were stale, not
      one. **What is still owed is the gate, not the fix**: nothing in `clustering`'s CI runs
      `uv lock --check`, which is why a desync introduced 2026-08-15 survived four commits until
      an unrelated audit tripped over it. One residual crumb: `pyproject.toml:116`'s mypy
      override still names `numba_progress`, a package the project no longer depends on.
      (b) ✅ **Done 2026-08-22, unticked until now.** `SLOW_TABLES` carries
      `table_4_1_mog_baselines`, `table_4_4_openset` and `table_3_4_gpu_speedups`, with the
      measured runtimes (530 s, 13,084 s) in a comment beside them, so `--fast` bounds the
      suite's runtime again.
      (c) ✅ **Fixed upstream: [`optimizers` #126](https://github.com/fundthmcalculus/optimizers/issues/126),
      [PR #127](https://github.com/fundthmcalculus/optimizers/pull/127) (2026-08-27).**
      `requires-python = ">=3.10,<3.15"` against `numpy>=2.4.6` has no solution, because
      **numpy 2.3.0** is the first release to require $\geq$ 3.11 (checked against PyPI release
      metadata, not assumed from the error). Attribution to the numpy bump (`8049b94`) confirmed
      by restoring `numpy>=1.21` against the same floor in a scratch copy, which resolves.
      The floor is now `>=3.11`, matching `tribble-fis`, `tribble-cluster` and the repo root.
      Verified: `uv run --project tribble-opt` builds the Cython extensions and imports the
      package, where before it failed during resolution.
      **Why nobody noticed for twelve days:** `pr.yml` installs with pip on a single 3.14
      interpreter, and pip resolves for the running interpreter only — it never evaluates the
      `python_full_version == '3.10.*'` split that a forking resolver forks on. The `>=3.10`
      claim was metadata no gate in the project ever read.
      ⚠️ **That "related pin drift, no defect" note was the whole finding, and calling it
      benign was the mistake. It is now B18.** What was recorded here as a curiosity —
      tribble-fis's lockfile pinning `optimizers` at one revision while the submodule sat at
      another — is the mechanism by which every FIS-suite archive recorded a `tribble-clustering`
      SHA that had not run. Written up in full at **B18**. The lesson is narrow and worth
      keeping: *two revisions of one library in a single run* is never a curiosity, because the
      provenance record can only name one of them.
      (d) ✅ **Done 2026-08-22, unticked until now.** `table_3_7_g2_downstream` sits in
      `CLUSTER_TABLES` with `--with aeon` and a comment recording why it matters — it is the
      evidence for Ch 3's G2 downstream claim and for §5.4's corrected coordinate-free claim, so
      a sweep reporting green while silently carrying it forward would be asserting a result
      nobody re-measured.
      (e) **Tables 4.6 and 4.7 can no longer be regenerated at all.** `load_openset_data()` prefers
      RT-IOT2022 whenever it is present, and it has been present since 2026-08-12, so
      `table_4_4_openset.py` now emits the RT-IOT2022 table (the prose's **4.7b**) under the same
      output filenames. The Glass-based 4.6 and 4.7 the prose quotes are frozen at
      `uniform-2026-08-03` with no way to re-derive them short of hiding the dataset. A
      ✅ **Fixed 2026-08-22:** `REPRO_OPENSET_DATASET` pins the choice
      (`glass` / `rt-iot2022` / `beth`) instead of taking the priority order, and a pinned-but-
      absent dataset is an error rather than a reason to quietly use another — substituting one
      silently is the whole reason the knob exists. Tables 4.6 and 4.7 regenerate again:
      `REPRO_OPENSET_DATASET=glass`. At the current pin the shape holds and the values move
      (4.6 peaks at $\theta = 0.90$, +0.162, against the prose's $\theta = 0.80$, +0.154;
      4.7 keeps its ordering, Isolation Forest +0.157 > complement +0.141 > SVM +0.085, with the
      gap narrowed from 0.047 to 0.016). Not a re-quote — **D8** applies — but they can be
      re-derived at all now, which they could not be.
      (f) **Glass moved into `data/` and three call sites never followed it** — fixed 2026-08-22,
      see B12(c). Worth repeating here for the consequence: Table 4.8's Glass row and *the whole of
      Table 4.9* were missing from the run of record, so **C4's headline correction-pass
      measurement had no regenerable output**. Restored and re-verified; C4's conclusion holds
      (gated cascade $+0.043 \pm 0.050$ against the prose's $+0.031 \pm 0.027$) once B14 is
      corrected in the same run.


- [x] ✅ **B17 — The harness now fails loudly where it failed quietly.** Every defect the
      2026-08-22 pass found shared one shape: an upstream function kept its name and changed
      its behaviour, the generator caught it, printed a reason, emitted `N/A` or a plausible
      wrong number, **exited 0**, and `run_all_tables.sh` printed *ok*. Graceful degradation is
      the right design; a graceful degradation nobody reads is indistinguishable from a result.
      Three changes, aimed at the class rather than the instances:

      **(a) `reproduce/preflight.py` — check the upstream contracts before spending hours.**
      Five invariants a caller cannot see from a signature: `W1-SCALE` (scaling the data must
      scale the distance — catches **B14**), `VAT-MATRIXFREE` (**B6**), `SCALER-ALIAS`
      (`UnitScalar is MinMaxScaler`, which **B13** verified by hand once), `MODEL-NAMES`
      (**B12**'s renames), `DATASETS` (**B16(f)**'s moved Glass). *An unavailable import is
      reported SKIP, never PASS* — a check that passes because it could not run is the bug this
      file exists to catch. `run_all_tables.sh` runs it per submodule environment before the
      numeric phase; a failure does **not** abort the run but stamps the archive **NOT CITABLE**,
      the same treatment `--fast` gets and for the same reason. It fails today, on W1-SCALE,
      which is the point.

      **(b) N/A accounting in the run summary and in `PROVENANCE.txt`.** A table now reports
      `ok  27s  (10 N/A)` instead of `ok  27s`. Counted from the CSVs that table actually wrote,
      via the same before/after snapshot the no-output check uses, so it needs no timestamp
      window. This is the change that would have surfaced **E1**'s evidence base going missing
      weeks earlier: `table_norm_conorm_matrix` emitted a third of its cells as `N/A` and
      reported `ok` every run.

      **(c) `build_pdf.py` warns when a figure copy changes the document** — **B8**'s one gap.
      The copy is unconditional, so any local run of the generator behind a figure swaps it into
      the document whether or not the table beside it was re-quoted from the same run. The build
      now names each figure it changed and says a figure and its table must come from one run.
      Verified by making it fire on `03-complexity-fit.png`, which is exactly the case that
      caught this: a gcc-built figure (exponent 1.77) above an MSVC-built table quoting 1.97.

- [x] ✅ **B18 — `PROVENANCE.txt` records a SHA the FIS suite does not necessarily run.
      Found and fixed 2026-08-27, one archive before it would have mattered.** The gap is
      between the submodule a SHA is read from and the wheel an import actually resolves to —
      the one place the `reproduce/` apparatus was not looking.

      ⚠️ **Scope, corrected after review: no archive was actually misnamed.** A first write-up of
      this item said every FIS-suite archive named a revision that had not run. That is wrong,
      and the audit is one command:

      ```
      $ grep -rh "^tribble-cluster:" reproduce/outputs/*/PROVENANCE.txt | sort | uniq -c
           23 tribble-cluster: e3c27e67...      7 tribble-cluster: 85b68a8a...
            7 tribble-cluster: c71171e7...      6 tribble-cluster: 635ed6ed...
            2 tribble-cluster: 5d44dfa4...
      ```

      **No archive records `1dcf331b`.** The six from the `635ed6ed` era record exactly what ran.
      The submodule advanced on 2026-08-24 and the lock did not, and the next archive would have
      been the first written in the drifted window — this one. So the exposure was real and
      prospective, not retrospective, and the item is worth keeping at that strength rather than
      the one it was first written at.
      **The five `opt-*` archives are correct too, for a reason worth recording.** They record
      `tribble-opt: 94547ff7` while the lock pinned a later `7b5958a1`, which looks like the same
      defect and is not: those studies run `--project tribble-fis --with-editable tribble-opt`,
      and `--with-editable` overrides the lock with the submodule checkout. For them the
      submodule SHA *is* the code that ran. Which record is right depends on how the environment
      was built, and neither "the submodule SHA" nor "the lock's pin" is universally the truth —
      which is exactly why the fix reads the installed distribution's `direct_url.json` instead
      of picking one of the two by convention.

      **The mechanism.** `tribble-fis/pyproject.toml` sources two dependencies from a git URL
      with no revision:

      ```toml
      [tool.uv.sources]
      optimizers         = { git = ".../optimizers.git" }
      tribble-clustering = { git = ".../clustering.git" }
      ```

      so `uv.lock` alone decides which commit installs. Every FIS table runs
      `uv run --project tribble-fis`, which resolves `tribble-clustering` **from that lockfile**
      and never consults `../tribble-cluster` at all. `run_all_tables.sh` meanwhile stamps
      `git -C tribble-cluster rev-parse HEAD` into `PROVENANCE.txt`. The two are unrelated
      quantities, and they had come apart:

      | dependency | lock | submodule | behind |
      |---|---|---|---:|
      | `optimizers` | `7b5958a1` | `4d811214` | 11 commits, 9 days |
      | `tribble-clustering` | `635ed6ed` | `1dcf331b` | 5 commits, 8 days |

      Not routine churn: that window holds a GA crossover identity bug fix and three
      determinism/seeding fixes in `optimizers` (#115, #118, #119), and an FCM
      coincident-point membership fix, an FCM crash fix and an `IVATMeans` `n_clusters` fix in
      `clustering` (#82, #84, #85). Being silently behind on *seeding* fixes is the worst case,
      because the symptom is irreproducibility rather than a crash.

      **Why nothing caught it.** `uv lock --check` passes in both states — measured, exit 0
      before and after — because a git source with no `rev` in `pyproject.toml` is "in sync"
      with any locked commit, however old. **B4**'s submodule SHA guard compares
      `git rev-parse HEAD` against the last archive, so it sees the submodule move and never the
      import. There was no check anywhere on the axis that mattered.

      **What limits the damage, established rather than hoped for.** `tribble-clustering` is
      referenced exactly once in all of `tribble-fis` —
      `src/tribblefis/gauss_math.py:11: from tribbleclustering import IVATMeans, FuzzyCMeans` —
      and neither name is used anywhere. A full-tree grep returns that one line. So the
      clustering half of the drift is numerically inert for every Chapter 4 and Chapter 6 table,
      and the recorded SHAs were wrong without the numbers being wrong. The `optimizers` half
      is the one that could bite, since `refine.py`, `optimizer_utils.py`, `regression.py` and
      the two `*_refine.py` modules do import it; that is under review. Filed upstream as
      [`tribble-fis` #203](https://github.com/fundthmcalculus/tribble-fis/issues/203), because a
      compiled package should not be a hard import dependency for an unused line.

      **Fixed three ways, because one was not enough.**
      1. Upstream: [`tribble-fis` #201](https://github.com/fundthmcalculus/tribble-fis/issues/201)
         and [PR #202](https://github.com/fundthmcalculus/tribble-fis/pull/202) refresh both pins
         to the submodule heads.
      2. Locally: `preflight.py` gains **PIN-MATCH**, which reads the *installed* distribution's
         `direct_url.json` and compares its `commit_id` against the submodule HEAD. It earned
         its keep within the hour, catching a submodule left on a feature branch that would
         otherwise have produced a quietly mislabelled archive.
      3. `INSTALL-FRESH`'s remedy text was wrong for this failure — it said to reinstall, which
         cannot fix a lockfile pinning an older revision. It now names both causes and points at
         PIN-MATCH to tell them apart.

      **Still owed upstream:** the durable gate. `uv lock --check` cannot see this class of
      drift, so #201 proposes either a scheduled job that runs `uv lock --upgrade-package` and
      reports when a pin moves, or pinning `rev=` in `pyproject.toml` so the commit is visible
      in a diff where a reviewer already looks. The second is smaller and brings
      `uv lock --check` back into play as a real gate.


## C. Experiments owed
**[Tier 1: critical before defense (C1, C4). Tier 2: real research (C2–C3, C5–C6, C8–C11–C13). Tier 3: defensive (C5–C6, C8). Tier 1.5: reduced scope (C4 done). Tier 4 (C7 descoped)]**

- [ ] 🟨 **C1 — the adapters have landed; what is owed now is a run, and it is not a cheap one.**
      `reproduce/tables/_baseline_anfis.py` and `_baseline_gafis.py` arrived on `main` (#191), so
      the claim that this work is *orders of magnitude faster* finally has something to be faster
      *than*. The ANFIS adapter is a Jang-style hybrid — least squares for the consequents,
      Adam on the premises — on numpy and scikit-learn only, so it adds no dependency.

      ⚠️ **A ten-seed run of Table 4.1 is now a scheduled job, not a background one.** Attempted
      2026-08-27 and abandoned deliberately. The cost is structural rather than incidental: both
      baselines build a **grid partition**, which on Concrete's 8 features is $2^8 = 256$ rules,
      and GA-FIS then searches a population of 20 over 15 generations on top of that. Measured
      here: roughly **5–6 minutes per seed on Concrete alone**, the smallest of the five datasets,
      against 217 s for the whole MoG-only table at ten seeds. Bike Sharing (17k rows),
      PhiUSIIL (235k) and RT-IOT2022 (123k × 82 features) follow, and a grid partition cannot be
      what runs on 82 features — check what the adapter falls back to there before quoting a
      timing, because a fallback that silently changes the baseline's *structure* would make the
      speed comparison meaningless in the flattering direction.

      **Do not shortcut it to three seeds.** This file's own protocol note says why: going from
      three seeds to ten retracted one conclusion and refuted another. A cheap C1 number is worth
      less than no C1 number, because the headline claim of the title rests on it.
- [x] ✅ **C2 — Complexity fit against reference curves.** Table 3.2 + Figure 3.2 now sweep a
      small grid (100–1,000, sized so the cubic arm runs at every point) with both axes
      normalized, and fit a log-log exponent per arm. Classical **3.15** against a theory of 3
      confirms; stage two fits **1.93–1.97**, confirming the quadratic claim it makes.
      **Stage one is the arm this item over-claimed on.** Five runs on the workstation fit it at
      **1.86–1.88** against a theoretical ≈2.1 for $O(N^2 \log N)$, and the number to notice is
      that it sits *below* the pure quadratic reference of **2.00** — so "the log factor is
      invisible over one decade" is asserted rather than shown, and an exponent under 2 is no
      evidence for a bound above it. What the sweep establishes is the cubic-to-quadratic
      *separation*, which both arms agree on; stage one's own exponent is
      **bounded, not confirmed**. Reporting a constrained fit at
      $t = c \cdot N^2 \log N$ beside the free exponent would settle it, and is the remaining
      work — tracked in Chapter 7 under G4a.
- [x] 🚫 **C2b — DESCOPED.** The ~10 ms fixed cost was a property of the laptop's power-saving governor
      and thermal throttling, not the kernel. It does not reproduce on the workstation. Across
      **five** independent measurements, stage two is monotone in N and beats stage one by
      **8.1–17.7× at every size**, confirming the quadratic claim cleanly. **Ignore the development
      laptop; rely on workstation results.** Chapter 3's claim that the compiled kernel "buys nothing
      across a band of problem sizes" is not supported and has been removed from the text.
- [ ] ⬜ **C10 — Generalize the merge operator** *(the method is named after this, and it is
      the most open item in Ch 3).* §3.3.4's stitch works and is measured — Table 3.6 has the
      principled version at ARI 1.00 across every partition tested against 0.47 for naive
      concatenation — but it is a two-way stitch over farthest-point-sampled blocks, not a general
      operator. Three unknowns, all of which a distributed implementation needs:
      **(a) does it compose?** Merging four blocks pairwise should give the same ordering as
      merging them at once; untested. **(b) How does the reconstruction-error bound grow** under
      repeated or hierarchical application rather than a single pass? **(c) How are block
      boundaries chosen** when the data does not partition cleanly — the ablation shows
      farthest-point sampling is *necessary* but not that it is *sufficient*.
      Until these are settled, G4's half-million-point target rests on a single-level result.
      Noted briefly in Ch 7 G4e.
- [ ] 🟨 **C3 — Ch 5 end-to-end FIS result.** Every Ch 5 number is a *clustering* score; the
      chapter exists to produce FIS antecedents. Until a model is built from those memberships and measured,
      the central claim rests on a proxy. **Author has FIS results on PhishingURII — needs clarification:**
      Are these clustering-only (still proxy), or end-to-end (Ch 5 → model → regression/classification → measured)?
      If end-to-end, C3 is done. Ch 7 §7.2 tracks this as **C3**.
- [x] ✅ **C4 — Quantify the correction-rule pass** (Ch 4 §4.3.1, Table 4.9, Fig 4.3;
      2026-08-05). Measured on Glass, ten paired seeds — not RT-IOT2022. **RT-IOT2022 is now in
      the repository (2026-08-12), and two of its three related-but-distinct scale claims are now
      measured.** Table 4.4's plain classification/timing claim (twelve classes, eighty-two
      features): MoG trains in $37.42 \pm 0.64$ s at $0.927 \pm 0.002$ accuracy against Random
      Forest's $0.999 \pm 0.000$ (ten seeds, `table_4_1_mog_baselines.py`). The open-set scale
      claim (§4.3.5, Table 4.7b): the complement rule loses to Isolation Forest at scale
      (+0.394 vs +0.537 Youden's $J$, five seeds).
      ✅ **Re-run at the document's own ten-seed floor, 2026-08-22** (3 h 38 m; the five-seed
      figure was a session-length compromise the prose names). The direction holds and the
      margin narrows sharply:

      | method | archive, 5 seeds | 2026-08-22, 10 seeds | **2026-08-27, 10 seeds, de-leaked** |
      |---|---:|---:|---:|
      | Complement rule (this work) | +0.394 | +0.515 | **+0.366** |
      | One-class SVM | +0.408 | +0.271 | **+0.410** |
      | Isolation Forest | **+0.537** | +0.579 | **+0.535** |

      The middle column is superseded, and the correction ran against the more interesting
      reading — it had been taken to mean the margin narrowed sharply to 0.064 and the
      complement rule had overtaken the one-class SVM. Two things changed before the re-run:
      `load_rt_iot2022` stopped passing the CSV's unnamed index column as a feature, which
      encodes the class because the per-class capture counter restarts at zero; and the pin
      restored k-means++ in `_kmeans_labels_1d` (#191 — see **D8** and note 26). At ten seeds on
      2026-08-27 the margin to Isolation Forest is **0.169**, *wider* than the archive's 0.143,
      and the complement rule sits behind the one-class SVM, where the five-seed archive had
      also put it. **This column is now Table 4.7b.**

      **Which of the two moved it is not settled.** A matched single-seed control puts the leaky
      column at only **0.019 $J$**, so most of the −0.149 is most plausibly the init
      restoration — but that is one seed read against a ±0.27 spread, an inference rather than a
      measurement. Separating them cleanly needs a matched ten-seed run at the old pin, which
      has not been done and is not planned. The question the document actually depends on is
      settled either way: this column is measured on correct data at a coherent pin.

      The spreads remain ±0.15–±0.27, so **no separation in the table clears its own error bar** —
      the same point §4.4 already makes about Table 4.7 — and that caveat is unchanged by the
      re-run. What *has* changed is that the D8 hold is lifted: both of its causes (#171, #191)
      are ancestors of the current pin, verified in the running environment (note 26), so this is
      a run of record rather than a diagnostic. **The *correction-rule cascade's own* scale
      claim is the one still open** — that specific experiment (the gated cascade below, on
      RT-IOT2022 rather than Glass) has not been run — but "RT-IOT2022 is absent" is no longer
      the reason for any of the three.
      The gated cascade gains +0.031 ± 0.027 accuracy over the flat base at a cost of raising
      raw membership functions from 81.4 to 109.0; collapsing it into one deployable FIS
      (union every layer, dedup at exact tolerance, predict by plain argmax) keeps +0.014 ±
      0.061 of that gain at 83.5 MF. This is not the paired confusion matrices the original
      wording asked for — that per-class detail is still unmeasured — but it settles the
      coarser question the wording was standing in for: does the pass help, and what does
      deploying it cost. Same pass also produced **Table 4.8** (MF-deduplication tolerance
      sweep across six datasets: Glass, Wine, Breast Cancer, Digits, Concrete, Diabetes — see
      §4.3.1) and filed [`tribble-fis` #85](https://github.com/fundthmcalculus/tribble-fis/issues/85)
      upstream, asking the library to expose the dedup tolerance, extend it to the cascade
      classifier and the regressor, and add a unit test for the exact-tolerance path this
      table's flattened arm depends on.
- [ ] ⬜ **C12 — Semi-supervised / incremental benchmark** (Ch 4 §4.3.3). The per-class
      independence → incremental-update property (new labeled data for one class updates only
      that class's rules) is stated as a structural consequence, not a measured result. Needs a
      controlled streaming or partial-label experiment before it can be promoted to a claim.
- [x] ✅ **C13 — Large-scale regression benchmark, promoted to a generator and measured at
      ten seeds** (2026-08-12; Appendix A.7.1). `reproduce/tables/table_a7_regression_scale.py`
      supersedes the single-seed pilot (`reproduce/regression_scale/RESULTS_2026-08-05.md`).
      Both datasets are now canonically sourced: California Housing via
      `sklearn.fetch_california_housing()` (no mirror needed), Superconductivity via UCI id 464
      direct download (the mirror-vs-canonical question the pilot left open is resolved for
      both). Ten-seed results: California Housing RF $R^2 = 0.809 \pm 0.008$, flat MoG
      $0.631 \pm 0.020$; Superconductivity (decorrelated) RF $R^2 = 0.923 \pm 0.004$, flat MoG
      $-0.261 \pm 1.431$. Random Forest wins both cleanly, confirming the single-seed pilot's
      finding at the document's own protocol. **New finding the pilot's one seed could not
      show:** flat MoG and HME are wildly unstable on Superconductivity even after
      decorrelation — occasionally catastrophically negative R², echoing the seed-9 HME
      divergence `table_concrete_reconciliation` already documents on Concrete. This is now the
      dataset/model-family decision the item was waiting on: both datasets are promoted, and the
      instability is itself worth a sentence in Chapter 6, not merely a caveat here.
- [ ] ⬜ **C14 — Train-subsample variance study for the turbofan-RUL case study** *(future PR;
      Ch 4 §4.4.1, Table 4.10, Appendix A.7.1).* N-CMAPSS DS02 RUL is currently *demonstrated*
      (one run on the dataset's own fixed split, `FuzzySystemsExperiments/cmapss_all_datasets.py`
      + `cmapss_all_datasets_report.md`), not *measured*. The reproducibility axis here is **not**
      a ten-seed random split — the train/test split is fixed by the dataset (the held-out engine
      units, the same split the published baselines use), so re-seeding it would measure the wrong
      thing and break the baseline comparison. What *should* be seeded is the **training-set
      subsample** (the pooled fit draws 30k of ~221k rows at a fixed seed) plus the model's
      `random_state`: re-draw both across ten seeds on the fixed split and report mean ± s.d., ideally
      via a seeded generator under `reproduce/tables/`. Blocker is redistribution, not compute: the
      10 `.h5` files total ~28 GB and are gitignored, so the generator must document the manual
      `data/nasa-cmapps2/` download the way `DATASETS.md` does for RT-IOT2022. Until then §4.4.1 is labelled
      *demonstrated* and the figure a single fixed-split run.
- [ ] ⬜ **C15 — Verify the DS02 CNN/MLP baseline figures from the source** *(Ch 4 §4.4.1).* The
      7.22 / 8.34 public-file re-runs are attributed to `custode2022evolutionary` and corroborated
      from search snippets and co-author code, but not read from the paper's own table (MDPI blocks
      automated fetch). Confirm via institutional access before the comparison is cited as settled;
      the `.bib` entry and §4.4.1 both flag this. Metadata for the entry is `[V]`; the *figures* are
      not content-verified, the same "`[V]` is metadata, not content" distinction the bibliography
      draws for `deshpande2024scalable`.
- [x] ✅ **C16 — The open-set order dependence is dedup, not the t-conorm** (2026-08-23;
      `reproduce/outputs/OPENSET_COST_2026-08-22.md` §6, `reproduce/experiments/diagnose_openset_order.py`).
      `profile_openset_cost.py` had established that feeding the antecedent screen's features in
      ranked vs column order changes 29% of open-set predictions deterministically, and
      OPENSET_COST *speculated* the cause was floating-point accumulation order in the complement
      rule's t-conorm chain, flagged as unconfirmed. **Refuted:** holding one fold's boosted
      matrix fixed and reducing the same columns four ways (column, reversed, sorted, pairwise)
      gives identical predictions at every θ (agreement 1.0000), so the t-conorm reduction is
      order-invariant and §4.3.5's commutative-associative claim needs no caveat. The divergence
      is one stage earlier, in the model **build**: ranked vs column order gives the same 11 rules
      / 2367 terms but max |firing difference| = 0.86 on 98.5% of rows — a different fitted model,
      because the O(T²) cross-feature dedup in `to_simple_model` (§4.3.1) is first-occurrence-wins,
      so the surviving representative MF depends on feature order. **Real but latent:** the shipped
      table always uses the screen's ranked order, so it stays deterministic and reproducible. A
      fix (canonical dedup representative instead of first-seen) would change results and is left
      as the author's call, not made here.
- [x] ✅ **C17 — Hoist the θ-independent work out of the open-set sweep** (2026-08-23;
      tribble-fis #176 pin `1435811`, `reproduce/tables/table_4_4_openset.py`). Table 4.4b's
      operating-curve sweep rebuilt the whole complement-rule model for every θ — screen,
      membership dict, dedup — though θ enters only at the anomaly step of prediction.
      (OPENSET_COST had this wrong too, corrected in place: it said θ enters at `to_simple_model`
      and estimated a 54% saving from hoisting only screen+memb; θ enters at *predict*, so the
      entire model build **and** the class rule firing are θ-independent.) `simple_gaussian_predict_sweep`
      computes the class firing once and sweeps θ over the anomaly step; `simple_gaussian_predict`
      is unchanged bit-for-bit (it composes the same helpers), and `theta_sweep` now builds once
      per (held-out class, seed). **~6× on the sweep, result-identical:** the restructured
      `theta_sweep` emits rows bit-identical to the old per-θ path (2 classes × 2 seeds, 6.13×),
      flags bit-identical across 2 seeds × 6 θ on a full RT-IOT2022 fold, preflight fis 5/5 at the
      new pin. The headline Table 4.7b (single θ) is untouched. Verified on single folds by
      construction, so no 120-fold rerun was needed.
- [x] ✅ **C18 — Table 4.10's 6.48/9.20/85 were never reproduced by the scripts that claim to
      produce them** (2026-08-23; Ch 4 §4.4.1, Table 4.10). Two independent causes, found by
      direct re-run rather than reasoning about it:
      (1) **Real bug.** The design-of-experiments' winning DS02 config (deleted `cmapss_rul_best.py`)
      used 20 channels — the 18 real sensors *plus* two "virtual" ones, T40/P30 — to reach RMSE 6.48.
      The 2026-08-17 consolidation into `cmapss_ds02_rul.py`/`cmapss_all_datasets.py` asserted,
      without re-testing, that dropping T40/P30 "does not change the result"; its loader
      (`tribble_predictive_health/data.py::load_ncmapss`) does not read the HDF5 `X_v` group at
      all, so the claim was never actually checkable from that script. Re-running the original DOE
      script's `best` pipeline with T40/P30 (same-era `tribble-fis` pin) gives RMSE 6.65 — close to
      6.48 — versus the current real-sensors-only pipeline's 7.23. Decision: kept real-sensors-only
      deliberately (simplicity over the last ~0.7 RMSE); the false "costs nothing" claim is removed
      from both scripts' docstrings.
      (2) **Residual, unexplained by any pinnable dependency.** Re-running the *exact* commit
      (`0ab0ad6`) and *exact* `tribble-fis` pin (`058501f`) that produced the committed
      `cmapss_all_datasets_report.md`, on this same host, with `numba`/`llvmlite`/`scipy`/
      `tribble-clustering`/`optimizers` pinned to that pin's own lockfile versions and `pandas`
      pinned to both 2.2.3 and 3.0.5, reproduces neither run-to-run randomness nor the committed
      `whole_cycle` 9.20/85 — it deterministically gives 11.11/142 instead, every time, regardless
      of thread count. `whole_cycle` never used T40/P30, so cause (1) does not apply here. Likely
      culprit: this host is openSUSE Tumbleweed, a rolling-release distro whose system snapshot
      moved from the report's commit date (2026-08-17) to `20260818` the very next day — i.e. "same
      host" does not mean "same glibc/libm state" days apart. This is the same fragile
      near-degenerate-arm sensitivity `tribble-fis`#179 already documents, previously attributed to
      Windows-vs-Linux; it is better described as sensitivity to *any* environment perturbation,
      including routine OS package updates on a single machine. Not fixed (no PR to point to); the
      committed report and Table 4.10 are regenerated with what this host reproducibly measures now
      (`whole_cycle` 15.44/12.07/320, `raw_memory` 15.58/17.00/655, blend 11.14/216, DS02 solo 7.23),
      and both scripts' docstrings and §4.4.1 say so instead of citing the unreproduced numbers.
- [x] ✅ **C19 — BETH one-class anomaly detection landed (grad-school #95 / #180), two trackers
      hadn't caught up** (2026-08-27; Ch 4 §4.4, Table 4.11 and companions 4.11(b)-(e)). The
      blocker WORKINGDOC.md §6 and Ch 7 §7.3 named — BETH needs a purpose-built one-class
      training path, not the leave-one-class-out protocol Glass/RT-IOT2022 use — is resolved:
      `tribblefis.one_class.TribbleOneClassDetector` already existed in the library, and
      `table_4_11_beth_anomaly.py` plus three companion sweeps (`table_4_11c/d/e_*.py`) now run
      it on the full 1.14M-row BETH capture. On the configuration §4.3.5 was written for, the
      complement rule reaches parity with a one-class SVM ($J$ +0.843 vs +0.841), and the
      "conorm is inert with a single class" algebra §4.3 argues traces correctly into
      `gauss_math.py`. Landing this updated WORKINGDOC §6 and Ch 7's timeline row, but left two
      trackers stale: `reproduce/PROVENANCE_MAP.md`'s four Table 4.11 rows still read *"no prose
      slot yet"* after the prose section existed, and `data/.gitignore` still named the
      pre-rename `table_4_x_beth_anomaly.py`. Both fixed here. Separately,
      `table_4_11e_beth_boost_sweep.py`'s θ-sweep "control" model built its Gaussian memberships
      without threading the loop's `seed` into `create_gaussian_membership_dict`'s own
      `random_state` (default 42), so for seeds other than 42 it was not provably "the same
      fitted memberships" its own comment claimed against the seeded `TribbleOneClassDetector`
      control. Fixed by passing `random_state=seed` through; unlikely to move the table's
      verdict (the delta-detection metric it reports is the point of the algebraic argument, not
      dependent on this), but the seeding is now actually as described.
- [ ] ⬜ **C11 — Benchmark `IVATMeans` against FCM and k-means** *(Ch 7 **G9**,
      Ch 3 §3.3.5).* §3.3.5 now presents `IVATMeans` as a contribution, and every property it
      claims is provable from `ivatmeans.py` rather than measured: initialization-free because
      the iVAT ordering is deterministic, verifiable against the reordered image, assignment
      and membership from one fit. Chapter 3 measures the **engine** — the reorder, the
      footprint, the device MST — and never the estimator. **Nothing in this repository times
      `IVATMeans` against either baseline or scores its partitions against theirs.** Both
      halves owed: wall clock across §3.4's existing size ladder with the CPU and
      whole-pipeline-on-device paths reported separately, and ARI on Table 3.5's four
      constructions plus one blob set where a prototype is the right model.
      Two protocol points that are the reason this is a real experiment rather than a
      formality. **The suite must include the sets where §3.3.5's envelope predicts a loss** —
      two moons and circles — because a benchmark run only where the method wins is not one,
      and that predicted loss is the refutation condition: if `IVATMeans` reaches ARI 1.00
      there, the Euclidean-prototype bound is not a bound and §5.2's argument for the
      relational method loses its motivating case. And the **determinism asymmetry belongs in
      the protocol**: FCM and k-means are reported as a spread over restarts, `IVATMeans` has
      none over seeds, so the protocol verifies the labelling elementwise identical across ten
      seeds and prints the zero rather than leaving a blank column.
      Three weeks on existing machinery; estimator, both baselines and the timing harness all
      exist. Related: clustering#61, which notes the estimator could also surface the
      hierarchy it already computes.
- [ ] ⬜ **C5 — Ch 3 head-to-head vs. eVAT (Meng & Yuan 2018) and clusiVAT** on identical
      datasets. First comparison a reviewer will demand.
- [ ] ⬜ **C6 — Ch 5 head-to-head vs. Bonis–Oudot beta-plateau and AuToMATo** on identical data.
      Defensive as much as scientific, given how close that work is.
- [x] 🚫 **C7 — DESCOPED, not done.** **DESCOPED from the proposal 2026-08-04.** The temporal-data
      subsection, Table 6.4, Figure 6.3 and Goal C7 are removed from the document. The `MimoGaussian` /
      `AnalyticalDynamics` work continues separately and the proposal no longer rests on it, so
      nothing here is owed *to the proposal*. Kept as a record of what was found, not as an open
      item. Original item: **Ch 6 Atwood machine result** (Table 6.4 pending row), and
      reconcile Table 6.4's R²/RMSE pair — 0.92/0.045 implies target σ ≈ 0.159, 0.96/0.028
      implies ≈ 0.140. It was also the one table `PROVENANCE_MAP` marked ungenerated while Ch 6
      called it the clearest result.
      **The diagnosis behind the descope, kept because it was expensive to get and the fix is
      one line.** `MimoGaussianPredictorMemory.predict_trajectory` never advances a step: it
      slices exactly `window_size` rows of history; `prepare_sequences` computes the last row's
      long-term average over an interval that is empty at exactly that row and returns NaN; the
      method's own NaN guard then breaks at step 0 and returns the initial window unchanged.
      Reproduced at `(window_size, memory_size)` = (3,1), (4,2), (10,4), (2,1) — unconditional.
      The one-step `predict` path is unaffected. Fix is a one-line slice
      (`window_size + memory_size` rows of history) in the pinned `tribble-fis`; second
      silent-wrong-answer defect found in an exported API by this project, after `B6`'s
      `vat_prim_mst_seq`.
- [ ] ⬜ **C8 — Ch 3 datacenter GPU re-run.** The pairwise-distance kernel loses (<1×) at low
      dimension / float64 on a consumer card; the prediction that full-rate FP64 flips it is
      untested and labeled as such.
- [ ] ⬜ **C9 — Ch 6 interpretability, measured** (G6): rule counts, path lengths, and either an
      established metric or a small expert study. Fills Table 6.3's pending row. Until then Ch 6
      must keep saying the payoff is *described*, not quantified.

  ↳ _Record of the filing, superseded by the resolved **C16** above (2026-08-23). Kept because
      its leading explanation was refuted rather than confirmed, and a wrong hypothesis that was
      correctly labelled as one is worth keeping. This is not open work; it was a second
      checkbox carrying the same ID._ **C16-orig.**
      **The complement rule is order-dependent on its feature list, and §4.3.5 says it
      cannot be.** Found 2026-08-22 while profiling why `table_4_4_openset` takes 3h38m; full
      account in [`reproduce/outputs/OPENSET_COST_2026-08-22.md`](../../reproduce/outputs/OPENSET_COST_2026-08-22.md),
      reproduced by `reproduce/experiments/profile_openset_cost.py`.

      Same RT-IOT2022 fold, same 82 features, **only the order differs** — the screen's ranking
      against plain column order:

      | | agreement | anomaly-flag rate |
      |---|---:|---:|
      | ranked vs ranked *(control)* | **1.0000** | 0.4551 |
      | column vs column *(control)* | **1.0000** | 0.1704 |
      | **ranked vs column order** | **0.7063** | — |

      The controls are the point: each order is *exactly* deterministic, so 0.7063 is not
      run-to-run noise. 29% of test rows change classification and the rate at which the detector
      fires moves by 2.7×. The fitted model is **identical** in both orders (11 rules, 902
      antecedent terms), so this is *evaluation* order, not what was fitted.

      §4.3.5 derives the rule as $\mu_{\text{anom}} = 1 - S(c_1, \ldots, c_K)$. A t-conorm is
      commutative and associative, so that is order-independent **by construction**. The
      implementation is not.

      **Leading explanation, not yet confirmed:** floating-point accumulation order through an
      82-term chain, amplified by the knife-edge §4.3.5 itself derives — at $\theta = 0.99$ the
      clip makes $\mu_{\text{anom}} > 0$ only when *every* class firing is below 0.01, so the
      decision sits where the last bits decide. That predicts the sensitivity is worst at the
      shipped operating point and milder across Table 4.6's swept band. **A 2.7× swing is larger
      than I would expect from float noise alone, so this must not be written up as settled.**

      **Three consequences.** (a) Table 4.7b's rates depend on an ordering nothing in the prose
      mentions. (b) The $\mathcal{O}(M \cdot K^2)$ screen has a second, unassigned job — it is
      fixing an evaluation order the result depends on — which is why it cannot be cut even though
      `top_n` keeps all 82 features. (c) It is a **candidate explanation for Table 4.7's
      instability**, which §4.4 already reports as an ordering that "has changed three times
      across runs" with spreads "roughly five times the largest gap in the table".

      **Owed at the time:** instrument the t-conorm chain to confirm or refute the mechanism;
      then either state the order dependence in §4.3.5 or remove it — accumulating in a fixed
      canonical order would make the implementation match the algebra and costs nothing.
      **Done, and it came back the other way:** the t-conorm reduction is order-invariant
      (agreement 1.0000 across four reduction orders), so §4.3.5's algebra needs no caveat. The
      divergence is one stage earlier, in the model build. See the resolved **C16**.

  ↳ _Record of the filing, superseded by the resolved **C17** above (2026-08-23), which found
      the saving was larger than estimated here and in a different place. Not open work._
      **C17-orig.** **Hoist the θ-independent work out of the θ-sweep (≈ 40 min of 3h38m).** θ enters
      only at `to_simple_model(params)`, but the sweep re-runs the whole fold per θ — including
      `calculate_gaussian_correlation` (46% of a fold) and `create_gaussian_membership_dict` (8%),
      neither of which takes θ. Hoisting them is result-identical by construction (the same
      `memb` object reused) and should cut the sweep by ~54%. Deliberately **not** done in the
      2026-08-22 pass: it changes the generator's code path, and folding a performance change into
      a run whose purpose was attributing numeric drift is how the two become impossible to tell
      apart. Wants its own change and its own before/after. Related knobs that already exist:
      `REPRO_THETA_SWEEP_SEEDS` (bounding the sweep to one seed is what made it 1h18m rather than
      ~13h) and `REPRO_OCSVM_TRAIN_CAP`.


## D. Writing and figures
**[Tier 1: D1, D4 done. Tier 1.4 open (D5). Tier 0.3 (D2, your records). Tier 1+ (D3, D6)]**

- [x] ✅ **D1 — Produce the remaining figures.** All fifteen exist, generated by
      `reproduce/figures/` in PNG + EPS against one shared style module. Both load-bearing
      figures are done: **Fig 1.2** (pipeline roadmap) and **Fig 5.2** (band discovery).
      **Fig 4.3** was the last holdout — retargeted to the Glass correction-pass measurement
      (**C4**) rather than left waiting on RT-IOT2022, which still is not a dataset the harness
      can load; the reasoning is recorded in `registry.py`. Fig 6.3 was descoped with the temporal-data subsection
      (see **C7**). What remains is a style pass on printed pages, not production.
- [x] ✅ **D2 — Acknowledgements audited, mostly done.** Acknowledgements have been audited (2026-08-08).
      Jon Salisbury is the author's boss, not a co-author. Ch 9 (publications outline) remains
      pending author records for the NAFIPS papers (exact titles, pages/DOIs, which conference/year,
      separate or combined). This is a writing task awaiting those records, not a decision item.
- [x] ✅ **D3 — `chapters/00-README-master-outline.md` removed (2026-08-04)**, rather than
      regenerated. It had fallen a generation behind the prose ("Status: Scaffold," pillar 1
      missing stage two and the name collision, MIMO still listed after that work was
      descoped). A committee member skimming first now gets that orientation from Chapter 1's
      outline and Chapter 7's goal table, which stay in sync with the rest of the prose.
- [x] ✅ **D4 — The `*pending*` table cells are all marked or resolved** (2026-08-02).
      **The count was wrong: 20, not 23.** Three different documents (this one, `NEXT_STEPS`,
      and the review that opened it) had said 23, 23, and 22 respectively; the actual inventory
      across the chapter tables was 20 — Ch 3: 1, Ch 4: 10, Ch 6: 9. (The three stray counts
      came from counting rows and prose mentions along with cells.) None is now a bare
      `*pending*`: each names what blocks it and the item that tracks it, so the gaps can be
      triaged rather than re-derived.
      ANFIS / GA-FIS, 11 cells across Tables 4.5 and 6.2 → `N/A (C1)`. Table 3.7's
      non-coordinate row → Goal G2. Table 6.3's counts → C9 / G6.
      Table 4.4's RT-IOT2022 accuracy → dataset absent from the repository *(as of 2026-08-02;
      the dataset is present since 2026-08-12 and the cell now marks a different blocker — see
      C4 above and Table 4.7b — rather than resolving this historical note)*. Table 6.2's M5 row
      → **a dependency fault, not an unrun experiment**: the generator already imports `m5py`
      optionally and would fill it unattended, but `m5py` does not load against scikit-learn
      1.9.0. Table 4.5's full-2nd training time → deliberately left empty; see note 14 in
      `PROVENANCE_MAP.md`, because that row's R² and its available timing come from two
      different code paths and pairing them would repeat the mismatch the table's caption
      exists to prevent. **What this item did not do is fill any cell with a number**, and the
      three follow-ups it exposed — C1, C7 and the Table 4.5 timing split — are the real work.
      (C7 is since descoped; see its entry above.) The total *table* count separately dropped
      from 22 to 21 on 2026-08-04 when Table 6.4 was descoped along with C7 — a different count
      than this item's pending-*cell* inventory, which already excluded it.
- [x] ✅ **D5 — Install a LaTeX engine.** ✅ DONE (2026-08-08). `texlive-xetex texlive-latex texlive-collection-fontsrecommended` installed; `build_pdf.py` auto-detects and renders.
- [x] ✅ **D6 — PDF build.** ✅ DONE (2026-08-08). Auto-rebuilds on every `python build_pdf.py` run; appends CHECKLIST at the end.

- [x] ✅ **D7 — Done by the prose tightening pass, `50456a1` (#134); ticked 2026-08-27 after
      checking rather than assuming.** All seven passages were consolidated and every marker
      removed, so this item's own instruction now returns nothing:
      `grep -rn "CONSOLIDATE" research/proposal-defense/prose/` exits 1. (The string still
      appears in `build/proposal-combined.md`, which is not a stale build — it is line 3027,
      inside the copy of this checklist that `build_pdf.py` appends, quoting the command above.)
      Spot-checked against the seven the item names: the −0.434 zeroth-order caveat, the
      ±0.210 spread and the §5.4 `NameError` note are gone from `prose/` entirely; the rest were
      compressed in place rather than deleted, which is what the item asked for.
      _Original entry, kept for the reasoning behind which seven were chosen:_

      Marked in place, not
      rewritten, because which to compress is an authorial call. Find them with:

      ```bash
      grep -rn "CONSOLIDATE" research/proposal-defense/prose/
      ```

      **Per-passage audit from the parallel pass on main, kept because it
      enumerates what this entry only spot-checked:**
      **Audit complete: 6 of 7 already consolidated into flowing text; 1 remains archival.**
      All seven passages located and reviewed:
      (1) §3.3.3 device win (30-56× vs 1.2-3.7×) — integrated into algorithm explanation
      (2) §4.3 z-score collapse — framed as caveat on normalization choice
      (3) §4.4 zeroth-order failure — mechanism explained inline, numbers given once
      (4) §5.4 NameError reproduction — kept in A.5 Reproducibility section (archival)
      (5) §5.4 many_scale paragraph — integrated as single statement of measurement
      (6) §6.4 optimizer comparison — old 2-optimizer vs new 8-optimizer, naturally framed
      (7) A.6 z-score withdrawal — one sentence suffices; full investigation in A.6.
      **No HTML `<!-- CONSOLIDATE -->` markers added;** passages flow without archaeology.

      `research/proposal-defense/mark_consolidations.py` inserts and re-checks them; the markers
      are HTML comments, which `build_pdf.py` strips, so they are invisible in the PDF and
      greppable in the source. The pattern in each: a number is reported, then an earlier pass of
      this same work is reported to have found a different one, then that one is withdrawn. Every
      retracted value appears nowhere else in the document and none was ever published, so for a
      first-time reader the retraction is the only place the wrong number lives at all. The seven:
      §3.3.3/Table 3.4's 30–56× parenthetical; §4.3's withdrawn $0.014 \pm 0.195$ z-score paragraph;
      §4.4's −0.434 zeroth-order caveat (the same finding as the previous one, told twice);
      §5.4's Reproduction note on the two-week `NameError` that changed no number; §5.4's
      `many_scale` paragraph, which states its result three times against its own history; §6.4's
      superseded ±0.241/±0.210 spreads; and A.6's withdrawal of a belief asserted nowhere else.
      **Two deliberately NOT marked**, because they are live methodology rather than archaeology:
      §3.4's retraction of ratio-invariance across machines, and §6.3.5's superseded two-optimizer
      comparison. Both earn their history.


- [ ] 🟨 **D8 — The accounting is done; what is left is a re-quote, which is ordinary work.**
      `check_prose.py` over the whole proposal, same prose, four archives:

      | archive | ok | drifted | untraceable |
      |---|---:|---:|---:|
      | `goal-8h-2026-08-11-fullsuite` (old pin) | **156** | 10 | 44 |
      | `full-2026-08-22` (current pin, 16/16 green at ten seeds) | **59** | 23 | 128 |
      | `wasserstein-fixed-2026-08-22` (current pin, B14 corrected in-process) | **70** | 23 | 117 |
      | **`pins-synced-2026-08-27`** (B14 landed, pins coherent, 9/9 green) | **61** | 109 | **44** |

      ✅ **The column that mattered is fixed. `untraceable` is back to 44** — identical to the
      run of record — where the two 2026-08-22 archives sat at 128 and 117. That number counts
      prose values that appear in *no CSV of the archive*, so it was never measuring drift at
      all; it was measuring **missing tables**. Those archives were incomplete, and reading
      their `ok` counts as "the document stopped reproducing" was reading a coverage failure as
      a numerical one. With a complete archive, every value the harness can produce, it produced.

      **`drifted` at 109 is the honest remainder, and it is a re-quote rather than a defect.**
      The prose is still quoted from the old pin, and the pin has since moved across four real
      changes — B14's `wasserstein_distance` fix (#171), the k-means++ restoration (#191),
      `pin_extremes` (#102), and a data leak removed in this repository (`c0ec380`). Numbers are
      *supposed* to move across those. What the count does not distinguish is a value that moved
      because the code got better from one that moved because it got worse, which is why
      **Table 4.1 is the one to read**, not the total:

      | row | run of record | 2026-08-27 | |
      |---|---:|---:|---|
      | PhiUSIIL, accuracy | 0.997 ± 0.001 | **0.997 ± 0.001** | exact |
      | RT-IOT2022, accuracy | 0.927 ± 0.002 | **0.927 ± 0.002** | exact |
      | Concrete, full 2nd order | 0.852 ± 0.030 | **0.852 ± 0.030** | exact |
      | Concrete, 1st order | 0.795 ± 0.025 | 0.799 ± 0.027 | within noise |
      | Bike Sharing | 0.939 ± 0.004 | 0.620 ± 0.014 | **moved — see below** |

      **The two accuracy rows returning *exactly* is the end of B14.** Those are the cells that
      read 0.729 and 0.500 under the defect; they are back on the archive to three decimals in a
      full ten-seed run, not a corrected-in-process diagnostic.

      **Bike Sharing moved because the data was wrong, and the reference column proves it.**
      The Random Forest reference on that row fell from a perfect **1.000 ± 0.000** to
      0.944 ± 0.004 in the same run — and Random Forest is scikit-learn, so a reference model
      moving means the *features* changed, not the library. `c0ec380` dropped `casual` and
      `registered`, which sum to `cnt` exactly on all 17,379 rows: the model was being handed
      the target's two addends. An RF at 1.000 was the tell, and it had been sitting in the run
      of record. **So this row is a correction, not a regression** — the same shape as C4's
      RT-IOT2022 index column, now twice in two months.
      ⚠️ **This supersedes the attribution below.** The Bike Sharing move was traced here to
      #102's `pin_extremes` default, measured on *frozen features* — correct as far as it went,
      and much smaller than the leak it could not see, because freezing the features is exactly
      what hides a feature defect. A probe designed to isolate the model from the data cannot
      find a fault in the data; worth remembering next time that design looks like rigour.

      Four causes are identified and three of them are benign:
      1. **B14** — decisive where its mechanism applies and irrelevant elsewhere. `table_4_1`
         goes 2 cells beyond noise → **0**; `table_6_1` 5 → 1. Worth 11 pairs document-wide (59 → 70).
      2. **The compiler (B15).** Chapter 3's kernels are gcc-built on this host, not MSVC, so
         ⚠️ **it moves the fitted exponents, not only the seconds.** Stage two:
         **1.97** (prose) → **1.89** (archive, MSVC) → **1.77** (this pass, gcc). §3.4 calls the
         exponent "the most portable quantity… invariant to any constant factor", and concedes
         one host-to-host move (2.12 laptop against 1.97 workstation); a *compiler* change on
         one host moves it about as far. Two consequences: the archive's own 1.89 already sits
         **below** the 1.93–1.97 stability band §3.4 claims across five runs, which is a
         pre-existing drift nobody had noticed; and the quadratic claim should rest on the
         cubic-versus-quadratic *separation*, which survives every variation, rather than on a
         decimal that does not. §3.4 updated to say so. **Figure 3.2 was deliberately NOT
         re-committed from this pass** — a gcc-built figure beside MSVC-built exponents in the
         table beneath it would be worse than a stale one.
         every Chapter 3 timing moved. Exactness columns did not. Not a defect — but it means
         **no Chapter 3 timing can be re-quoted from `full-2026-08-22`** without a host that can
         rebuild with MSVC, or an explicit note that the compiler changed.
      3. **The landed FCM fix (E2c).** Table 3.4 moved again, exactly as E2c predicted it would.
      4. **Restored rows.** The norm/conorm flat-MoG rows, Table 4.8's Glass row and the whole of
         Table 4.9 were *missing* from the archive and now exist, so they register as differences
         against nothing. See B12(c).

      **What is left over is the actual item.** After correcting B14, `table_a1_feature_ranking`
      still moves 8 cells beyond noise, `table_a2_feature_count` 21, and
      `table_g5_output_partitioning` 11 — none of which the compiler, the FCM fix or a restored
      row explains. The three regression rows of `table_4_1` are the cleanest instance: they moved
      $0.795 \rightarrow 0.808$, $0.852 \rightarrow 0.867$, $0.939 \rightarrow 0.960$ and move
      *identically with and without* the B14 correction, so they have a separate cause.

      ✅ **That instance is now traced** (`reproduce/experiments/diagnose_regression_drift.py`,
      `--freeze` / `--probe` / `--isolate`). Features are frozen once, already normalized, so the
      probe isolates the model from the scaler. Bisected over the same 48 commits:

      | tribble-fis commit | 1st order | full-2nd |
      |---|---:|---:|
      | `80e98d7` (archive pin) | 0.7950 ± 0.0249 | 0.8517 ± 0.0297 |
      | `ce4a0fc` (#87) | 0.7950 ± 0.0249 | 0.8517 ± 0.0297 |
      | **`5237ebe` (#95)** | **0.8041 ± 0.0297** | **0.8680 ± 0.0277** |
      | `141596e` (current) | 0.8078 ± 0.0297 | 0.8666 ± 0.0311 |

      **It is the same commit as B14** — #95, the scipy/sklearn → numba replacement — acting
      through a *different* function, which is why restoring `wasserstein_distance` never moved
      these rows. Restoring each replacement one at a time at the current pin, exactly one moves
      them: **`_kmeans_labels_1d`**, back to 0.7986 ± 0.0255 / 0.8521 ± 0.0287 (the full-2nd row
      lands on the archive to within 0.0004). The others are inert.

      The mechanism: sklearn's `KMeans` seeds with **k-means++**; the replacement `kmeans_1d` takes
      a **single uniform-random start** and no restarts. That changes the 1-D mixture
      initialization, hence the fitted memberships, hence $R^2$. **Unlike B14 this is not a wrong
      answer** — the values went *up* — but it is an unexplained change to a headline number
      arriving inside a commit described as a performance optimization, and a single random start
      is a worse guarantee than k-means++ even where it happens to score better. Worth one
      sentence upstream and a decision here about which to quote.

      ✅ **The residue is now fully attributed (2026-08-22). Two upstream commits and host
      noise account for every drifting cell; nothing is left unexplained.**
      `reproduce/experiments/run_with_reference_stats.py` restores any subset of the six functions
      #95 replaced (`REPRO_RESTORE=wasserstein,kmeans` / `all` / `none`) and re-runs a generator
      unmodified. Restoring **all six** at the current pin, against the archive:

      | table | result | cause |
      |---|---|---|
      | `table_a1_feature_ranking` | **byte-identical** | #95 |
      | `table_g5_output_partitioning` | **byte-identical** | #95 |
      | `table_g5b_skew_sweep` | **byte-identical** | #95 |
      | `table_a2_feature_count` | every accuracy identical; only wall-clock differs | #95 |
      | `table_4_1`, classification rows | **exact** ($0.997 \pm 0.001$, $0.927 \pm 0.002$) | #95 (`wasserstein_distance`) |
      | `table_4_1`, Concrete rows | within 0.004 | #95 (`_kmeans_labels_1d`) |
      | `table_4_1`, **Bike Sharing** | 0.962 vs 0.939 — **not** recovered | **#102**, below |
      | `table_6_1`, `table_concrete_reconciliation` | 3rd–4th decimal (9.122 → 9.117 RMSE) | host / BLAS |
      | `table_norm_conorm_matrix`, `table_4_8_mf_dedup` | extra rows the archive lacks | rows this pass **restored** (B12(c), B16(f)) |

      **Bike Sharing is a second commit, and a pleasing one.** Probed on frozen features: the
      archive's pin `80e98d7` gives $0.9394 \pm 0.0043$ — the archive's own 0.939 ± 0.004 — and the
      current pin gives $0.9646 \pm 0.0013$. Bisected in one probe to **`69e0bab`** (#102), whose
      parent `5b92ec8` reads $0.9348 \pm 0.0035$. The diff is a single default:
      `pin_extremes=True` → `pin_extremes=False` in `TribbleRegressor`.

      That is **the same parameter §4.3.2's G5 study is about**. G5 measured pinning the extreme
      bucket centroids to the observed min and max as costing 0.676 $R^2$ at zeroth order on
      Concrete, and recommended against it; upstream independently flipped the default the same
      way. So this drift is not a regression at all — it is the library adopting the proposal's own
      recommendation, worth +0.025 on a heavily skewed count target. **§4.3.2 and G5 should say
      that the default now matches the recommendation**, which is a stronger position than
      recommending against a shipped default.

      **Consequence for the archive question:** `full-2026-08-22` remains a *diagnostic* archive
      and the run of record stays `goal-8h-2026-08-11-fullsuite` — but for a different reason than
      before. Not "we do not know what moved" any more; rather, one of the two causes (B14) is a
      defect awaiting an upstream fix (`tribble-fis` PR #171, CI green on both kernels), and the
      other (#102) is an improvement the prose has not yet absorbed. Once #171 lands, a pin bump
      plus a re-run should reproduce the archive everywhere except Bike Sharing and the
      restored rows, both of which are then *expected* moves with a stated reason.


## E. Decisions and framing
**[Tier 0–4: mix of settled defaults (E1, E3), verification paths (E2, E2b, E2c), and low-stakes editorial (E10). E1.6–E1.7 Tier 1 (normalization + FCM). E9 low-priority investigation.]**

- [ ] 🟨 **E1 — the study is now measurable and says something worth a chapter; the framing
      decision is what is still open.** Re-run 2026-08-27 at a pin where the feature screen
      works (**B14** closed), ten seeds, no `N/A` cells — the first time this table has been
      complete. `reproduce/outputs/pins-synced-2026-08-27/table_norm_conorm_matrix.md`.

      | model | min/max | probability | **Łukasiewicz** | hamacher | einstein |
      |---|---:|---:|---:|---:|---:|
      | flat MoG-TSK, $R^2$ | 0.705 ± 0.053 | 0.687 ± 0.051 | **−3.774 ± 0.569** | 0.711 ± 0.053 | 0.678 ± 0.048 |
      | HME (experts only), $R^2$ | 0.788 ± 0.027 | 0.774 ± 0.020 | **−3.592 ± 0.500** | 0.796 ± 0.025 | 0.761 ± 0.024 |
      | fuzzy tree (t-norm only), $R^2$ | 0.708 ± 0.031 | 0.712 ± 0.032 | 0.713 ± 0.035 | 0.712 ± 0.031 | 0.713 ± 0.033 |

      **The Łukasiewicz collapse is confirmed at full magnitude**, and the values land back on
      the *original* 2026-08-02 figures (−3.761 flat, −3.626 HME) rather than on the −0.507 /
      −1.084 this item recorded from the 2026-08-22 diagnostic run. That diagnostic run was
      taken under **B14**, which this item flagged at the time — *"do not re-quote the absolute
      values yet"* — and the flag was right. The correction this item made to itself, that
      **min/max is the worst of the four non-Łukasiewicz families**, was a B14 artifact and
      does not survive: min/max is second of four here, and the four sit within **0.033** of
      each other against ±0.05 seed spreads, so none of that ordering clears its error bar.

      **The new finding, and it is the one worth harvesting: the collapse is a *conorm* effect.**
      The fuzzy tree row uses the **t-norm only** — path weights are a pure AND with no OR to
      apply a conorm to — and it does not collapse. All five families land within **0.005** of
      each other there, Łukasiewicz included. So the two models that use both operators lose
      four points of $R^2$ under Łukasiewicz while the model that uses one operator is
      indifferent to the choice. That localizes a dramatic effect to a single operator, on
      evidence already in hand, and it is a much better reason to give this study a home in a
      chapter than the family ordering ever was.

      **What is still open is unchanged and is a decision, not a measurement:** whether to
      present min/max as the default, and whether to harvest this study into a chapter — it
      still appears in none. _Original entry, kept as the record:_ `table_norm_conorm_matrix.py` imported
      `MixtureOfGaussiansFuzzyRegressor` and `MixtureOfGaussiansFuzzyClassifier`, renamed upstream
      to `TribbleRegressor`/`TribbleClassifier` — the **same rename B12(a) swept for, in a file
      that sweep missed**. Both flat-MoG rows of this table have therefore been silently `N/A`
      since at least the 2026-08-11 archive, which is why nothing failed: the skip path works
      exactly as designed, prints its reason, and emits `N/A`, so the table reported *ok* and the
      run reported green with a third of its rows empty. Fixed 2026-08-22 (new name first, old
      name as a fallback so the generator still runs against an older pin).
      **What the restored rows show**, ten seeds on Concrete and PhiUSIIL:

      | model | min/max | probability | **Łukasiewicz** | hamacher | einstein |
      |---|---:|---:|---:|---:|---:|
      | flat MoG-TSK, $R^2$ | 0.576 ± 0.037 | 0.605 ± 0.042 | **−0.507 ± 0.254** | 0.588 ± 0.041 | 0.607 ± 0.041 |
      | HME (experts only), $R^2$ | 0.735 ± 0.040 | 0.745 ± 0.035 | **−1.084 ± 0.397** | 0.741 ± 0.038 | 0.744 ± 0.035 |

      So **the reportable finding survives and is now demonstrable on both models** rather than
      one: Łukasiewicz collapses the regression models while the other four sit within 0.03 of
      each other. Two corrections to this item's own text: the magnitudes are not the −3.761 /
      −3.626 recorded here, and **"min/max is nominally best for the flat MoG (0.651 vs 0.650)"
      does not reproduce** — min/max is the *worst* of the four non-Łukasiewicz families here
      (0.576 against einstein's 0.607). ⚠️ **Do not re-quote the absolute values yet**: this table
      runs through the feature screen and is therefore affected by **B14** (its PhiUSIIL
      accuracies all sit in the broken ≈0.73 band). The Łukasiewicz collapse is far too large to
      be a B14 artifact; the finer family ordering is not, and must be re-taken after B14 lands.
      The rest of this item stands unchanged: the case for min/max is simplicity rather than
      accuracy, and **the whole study still appears in no chapter** — §4.3.5 says only that the
      conorm family is "a parameter" and that the family sweep is on accuracy only. Harvest it or
      drop the reference. Original entry:

  ↳ _As recorded 2026-08-02._ *(Author decision recorded 2026-08-02:
      keep tables at factory/library defaults, show the better configuration alongside, treat as
      future work.)* Upstream `53e89ab` made *probability* the library default. Data: min/max is
      nominally best for the flat MoG (0.651 vs 0.650) but by 0.001 against σ ≈ 0.05, so the case
      is simplicity rather than accuracy. **The reportable finding is that Łukasiewicz collapses
      the regression models** (−3.761 flat, −3.626 HME) while the other four families sit within
      0.03. **Dangling-reference sub-item fixed (2026-08-21):** §2.1's "Chapter 4 shows that this
      choice changes how readily a model declares something familiar" overclaimed exactly what
      §4.3.2 disclaims (`table_norm_conorm_matrix.py` sweeps the five De Morgan families on
      *accuracy* only; the open-set comparison across families is untested). The §2.1 sentence now
      points to §4.3's actual use of the Hamacher conorm and carries §4.3.2's own "untested" hedge.
      **What remains open in E1:** the min/max-as-default framing decision, and whether to *harvest*
      the norm/conorm study into a chapter (it still appears in none) rather than only reference it.
- [x] ✅ **E2 — Table 3.4 now has a generator, and it runs on this host.**
      `reproduce/tables/table_3_4_gpu_speedups.py`, 31 rows, ten seeds, each row one CPU arm
      against one GPU arm timed in the same pass, device timings stream-synchronised and all
      JIT warmed first (the first `boruvka_mst_device` call spends ~0.4 s compiling — 13× the
      N=16,000 kernel time, so cold timings would have been fiction). In the sweep with its
      CuPy dep isolated and a no-device fallback. Run of record
      `reproduce/outputs/gpu-table34-2026-08-02/`. Environment: CuPy 14.1.1, CUDA 12.9,
      RTX 4080 Laptop 12 GB, driver 610.74, compute 8.9.
      **The exactness claim holds where it matters**: the device ordering is elementwise
      identical to serial at float64 for every N and seed, and at float32 to N = 32,000. The
      48,000-point float32 *demonstration* reads 0.99992 — about 4 positions in 48,000 — which
      is a benign tie-break, not an error: the Prim totals agree to every digit printed, so the
      device found a different member of an equal-weight MST set. Worth one sentence in §3.3.3
      rather than the unqualified "bit-identical".
      **The negative result reproduced**: pairwise distances lose at float64, 0.30× at d=10.
      The datacenter-FP64 prediction remains untested and untestable here; no cell estimates it.
      fp16 and the §9.4 standalone-paper item are still open and unaffected (see **B10**).
- [ ] ⬜ **E2b — Re-quote Table 3.4 and §3.3.3 from the generator; one row overstates the GPU
      by roughly an order of magnitude.** `PROVENANCE_MAP` now marks Table 3.4 **drifted**.
      **The FCM row is the problem.** The chapter's "thirty to fifty times over the 32-core CPU"
      compares `fcm.fuzzy_c_means` — NumPy broadcasting with (n,k,d) and (n,k,k) temporaries —
      against a GPU path using the gram identity and two GEMMs. Those are different
      *algorithms*, not the same algorithm on different hardware. Measured against the GPU's own
      formulation written in NumPy/BLAS, the same three sizes give **1.3× / 2.1× / 3.7×**, and
      the library CPU arm is ~11× slower than a matched CPU arm at every size. The generator now
      emits both, and the chapter must quote the matched one or say plainly which comparison it
      is making. **This is note 11's hazard in a second place** — a ratio between arms that
      differ in implementation is not a property of the hardware.
      Three smaller mismatches: the MST is *faster* than quoted (5.4–7.7×) but does **not** grow
      with N — it peaks mid-grid and falls to 6.3× at 32,000, as expected from an O(n²) dense
      Prim CPU arm against O(n² log n) Borůvka rounds; the front end's since-corrected 4.8–6.6× matched
      matched-work only at the top of the grid (4.9×) and is reproducible as a band only if the
      CPU arm also materialises the reordered n×n matrix the GPU never builds (5.6–11.8×); and
      "exact" is wrong for the two fastest pairwise cells (2.06×, 4.18×), which run
      `high_precision=False` and deviate ~1e-4.
      **Do not quote a single FCM cell without the CSV**: with identical initial centres and
      convergence test, iterations to the fixed point range 11→100 across seeds, so the spread
      rivals the mean (29.16 ± 26.21 s). And the N=48,000 demonstration moved 3.3× between runs
      at the VRAM edge (9.2 of 11.6 GB), cause unknown — likely WDDM memory management; it is
      labelled volatile.
- [x] ✅ **E2c — LANDED UPSTREAM (verified 2026-08-22). The CPU FCM formulation is fixed;
      Table 3.4 and §3.3.3 must now be re-measured, which is **E2b**.** `clustering` #75
      (`bb61851`) replaces the broadcasting distance computation with the gram identity and a
      GEMM, and #72 (`5f1bb1d`) adds the `n_iter_` and `converged` fields plus a `max_iter`
      parameter — both halves of what the issue asked for. Confirmed by reading
      `src/tribbleclustering/fcm.py` at the pinned SHA, not from the commit messages: the
      distance block now carries the comment *"gram identity"* and `FuzzyCMeansResult`
      declares `n_iter_: int` and `converged: bool`. The pin (`635ed6e`) contains both.
      **Consequence:** the warning this item carried — *"Table 3.4's device row moves again
      once it lands"* — has come due, so the device row is provisional in the same direction
      twice, exactly as predicted. The ~10× CPU-side win is now real and available with no
      GPU involved, and still appears in no chapter. Original account, kept as the record:

  ↳ _Record of the filing, kept for the diagnosis; superseded by the line above._ **E2c-orig**,
      [clustering#62](https://github.com/fundthmcalculus/clustering/issues/62) (2026-08-04).
      This is the root cause **E2b** re-quotes around, not a separate finding: `_get_weights`
      computes distances by NumPy broadcasting, allocating an `(n,k,d)` temporary, then forms an
      `(n,k,k)` ratio tensor to normalise the weights — at n = 500,000, k = 10, d = 20 that is
      800 MB plus 400 MB *per iteration*, for up to 100 iterations, and `_get_v_ij` allocates
      another `(n,k,d)`. Neither is needed: the gram identity gives distances in one GEMM as
      `(n,k)`, and the ratio tensor collapses algebraically to `d^(-2/(m-1))` row-normalised —
      already what `gpu.fuzzy_c_means_gpu` does, which is the entire reason the device looked
      13–39× faster instead of 1.24–3.71×. Landing the fix (measured ~10× CPU-side speedup, no
      GPU required: 2315 ms broadcasting → 217 ms matched → 176 ms GPU at n = 50,000) means
      Table 3.4's device row moves *again* once it lands — provisional in the same direction
      twice — and the ~10× CPU-side win is a better, GPU-free result than the number it was
      hiding inside, and currently appears in no chapter. The issue also asks for `n_iter_` and
      a `converged` flag; until it lands, no single-run FCM timing from this library is
      quotable, since 11-to-100-iteration variance is exactly why every FCM cell in Table 3.4
      carries a spread as large as its mean.
- [x] ✅ **E3 — Schedule or explicitly defer G8 (2026-08-27: verified done).** Table 7.1 
      already records G8 as "retargeted post-defense" and "not in the runway"; §7.2 confirms 
      "construction is retargeted post-defense, to a journal extension." Decision is locked in 
      across all three documentation points (Ch 7 table, Ch 7 prose, Ch 10 Gantt).
- [x] ✅ **E4 — Bound the Magdalena/G8 tension (2026-08-27: done).** Strengthened §6.2 to 
      explicitly state "This chapter's interpretability argument holds *without* G8." Expanded 
      passage to list what is complete (solver, trees, mixture, structure spec, Ruspini export) 
      and clarified that G8 is proposed as post-defense extension, with decision rule clear: 
      measurement of disjunct-counter frequency decides if G8 is worth building.
- [x] ✅ **E5 — Decide the SHAP question** (§2.6). **DECIDED: Reframe from "post-hoc is worse" 
      to "post-hoc answers a different question", no experiment needed (2026-08-27).** The V&V 
      review (2026-08-24, finding V2) added `rudin2019stop` to the SHAP paragraph, positioning 
      the choice as: post-hoc attribution answers *"how did this model decide?"* while 
      interpretable-by-construction answers *"can I understand and edit this model?"* These are 
      different questions. For high-stakes domains where editability and verifiability matter, 
      the latter is more valuable — which is what this dissertation chooses. Rudin makes the 
      same case. No SHAP comparison is owed to G6; §2.6 stands as reframed.
- [x] ✅ **E6 — Give the Concrete benchmark one canonical citation (2026-08-27).** 
      Added explicit table citations in both chapters where Concrete Compressive Strength 
      values are quoted: §4.4 line 195 → "(Table 4.1, full 2nd order)" and §6.3.5 line 103 → 
      "(Table 6.1)". Both cite the canonical Concrete baseline measurement rather than merely 
      restating numbers, reducing drift risk by making the provenance explicit on first mention.
- [x] ✅ **E7 — Two literature searches done; attribution pass done (2026-08-21).** Both searches
      ran and their findings are folded into Ch 6:
      **(1) Knot/breakpoint optimization.** A strong triangular partition of unity *is* the order-2
      (linear) B-spline basis, so apex-knot refinement is free-knot linear-spline fitting and the
      sum-to-one property is intrinsic to the form, not an enforced constraint. Nearest fuzzy
      precedent is de Oliveira 1999 (semantic constraints incl. sum-to-one during MF tuning).
      §6.3.4 now credits `deboor2001splines` (added) + `deoliveira1999semantic`; the novelty is
      framed as a positioning/setting claim (apex-only refinement of an *exported* rule base), not
      a new mechanism.
      **(2) Fuzzy mixtures-of-experts.** Wu et al.'s TSK≡MoE equivalence is stated for the *flat*
      layer only; "hierarchical TSK" in the literature is stacking/widening (Zhang 2024 survey),
      not recursive gating. The surviving novelty — one shared closed-form ridge-TSK primitive
      reused across flat FIS, soft-tree leaves and HME experts — is unprecedented as a *composition*
      only; §6.2/§6.5 already frame it that way. **Fixed a real miscitation:** §6.2 attributed the
      TSK≡MoE result to `wu2020optimize` (the MBGD-RDA gradient paper) instead of `wu2020functional`
      (the functional-equivalence paper). Corrected.
      **Attribution/metadata fixes applied to `references.bib`:** Zhang-2023 already handled (see
      E1 note); `kumar2016incvat` real title/authors corrected against Crossref ("Adaptive Cluster
      Tendency Visualization and Anomaly Detection for Streaming Data", 7 authors); `deshpande2024scalable`
      first author corrected (Kartik Vishal, not Ojas); Kališnik accent confirmed already correctly
      encoded (`Kali{\v{s}}nik`). **One item, now resolved by deep research (2026-08-21; see A1):** the pVAT
      collision is **genuine** — "pVAT" is a method named inside Parveen & Sreevalsan-Nair's
      small-world-networks paper (Algorithm 1: "pVAT: Parallel implementation of VAT"; GPU/CUDA,
      Borůvka MST), not a separate paper title. The `.bib` had lifted the algorithm caption as the
      title and carried the wrong given name; corrected to the real title + "Saima", restored to
      `[V]`, DOI added. No journal version exists. The mergeVAT rename is fully supported.
- [x] ✅ **E8 — Two blocking reads done (2026-08-21); the note-scoping decision is now teed up for
      the author.** The reads that blocked the Ch 9 complexity note (short-communication /
      NAFIPS-style venue; novelty scoped to (a) heap-vs-dense correction, (b) measured crossover,
      (c) iVAT coverage Fast-VAT 2025 lacks, (d) O(N)-workspace as a ≈2× constant-factor win, *not*
      "a faster MST") are complete:
      **Deshpande & Kumar 2024 — the decisive read.** The paper states, verbatim, that its ordering
      methods (BB-VAT, kdT-VAT, TkdT-VAT) "do not even calculate the n × n distance matrix for the
      input data X" — i.e. the no-full-matrix result is claimed for the **VAT ordering step itself**,
      not only for MST-iVAT. So claim (d) in its bare form ("we avoid the full matrix for VAT") is
      **pre-empted**. The escape is that D&K achieve it by a **coordinate-based** kd-tree/bounding-box
      route that needs Euclidean coordinates, whereas this work's O(N)-working-memory reorder is
      **coordinate-free** (arbitrary/non-metric dissimilarity) — which is exactly the distinction Ch 3
      §3.2 already draws ("solved... by a coordinate-based route mine does not require"). D&K's *exact*
      per-method space bounds are in the paywalled Section 5 and remain unread, so whether they also
      claim a strict O(N) bound is unconfirmed.
      **Wang et al. 2010 (PAKDD) + Fast-VAT 2025.** Confirmed: claims (a), (b), (c) survive. Fast-VAT
      is a Cython/Numba implementation speedup, exact, **VAT-only (no iVAT)**, stores the full N×N
      matrix and names O(N²) memory as an unsolved bottleneck — so "iVAT coverage Fast-VAT lacks" (c)
      and the workspace contrast are defensible. Attribution point banked: the O(N²) iVAT *recurrence*
      is Havens & Bezdek 2012, not Wang 2010; Ch 3 already credits this correctly.
      **Author decision, taken (2026-08-21):** the complexity note is **not** a standalone
      short-communication or a novelty claim — it is folded into the **Ch 3 mergeVAT methods paper**
      as an *observation*: what the literature claims (O(N²) time/space) versus what public libraries
      actually implement (the cubic re-scan; the full-matrix footprint). This is what D&K's pre-emption
      leaves standing anyway. Prose reconciled to the decision: the standalone-note framing is removed
      from §3.2 (now "an audit the methods paper carries, not a novelty claim of its own"), from the
      Ch 10 Gantt/quarter-grid (the "VAT complexity note" deliverable is dropped and folded into the
      Ch 3 journal row; G4d moves from fifth to fourth in the cut order), from Appendix A.2.4/A.6, and
      from `bibliography.md` — this also clears a latent dangling reference, since Ch 9 §9.3 never
      actually contained a note subsection for the many cross-references that pointed at it.
      `deshpande2024scalable` first-author metadata fixed in the `.bib` as part of this read.
- [x] ✅ **E9 — Answered 2026-08-27, and the answer is sharper than the question.** All three
      experiments this item proposed were run, on shared splits, ten seeds, plus a fourth axis
      the first pass forced: `reproduce/experiments/diagnose_bounded_normalization.py`,
      output `reproduce/outputs/diagnose_bounded_normalization.md`.

      **The first pass found nothing, and that was the finding.** At the current pin every
      normalized arm — min-max `[0,1]`, min-max `[-1,1]`, z-score, z-score clipped at 2σ and
      3σ, and both half-and-half arms — lands on **0.797–0.799**, against raw features at
      0.687. The transform does not matter. That is not what this item recorded, so something
      had changed underneath it, and the obvious candidate was the mechanism the item's own
      working account names: `pin_extremes`, which fixes the first and last output-bucket means
      to the observed min and max. `tribble-fis` #102 (`69e0bab`) flipped its default
      **True → False** — the change **D8** already traced for a different reason.

      **Sweeping that as a second axis makes the mechanism switchable, and it settles the item:**

      | transform | bounded | centred | `pin=False` | `pin=True` | Δ |
      |---|:--:|:--:|---:|---:|---:|
      | raw | – | – | 0.687 ± 0.049 | 0.687 ± 0.049 | 0.000 |
      | log + min-max `[0,1]` *(shipped)* | ✓ | ✗ | 0.799 ± 0.026 | 0.795 ± 0.025 | −0.004 |
      | log + min-max `[-1,1]` **(a)** | ✓ | ✓ | 0.798 ± 0.025 | 0.751 ± 0.027 | **−0.047** |
      | log + z-score | ✗ | ✓ | 0.798 ± 0.025 | 0.714 ± 0.042 | **−0.084** |
      | z-score, clip ±3 **(b)** | ✓ | ✓ | 0.797 ± 0.022 | 0.712 ± 0.039 | **−0.085** |
      | z-score, clip ±2 **(b)** | ✓ | ✓ | 0.799 ± 0.015 | 0.666 ± 0.060 | **−0.133** |
      | z-score on logged cols only **(c)** | – | ✓ | 0.798 ± 0.025 | 0.793 ± 0.025 | −0.005 |
      | z-score on unlogged cols only **(c)** | – | ✓ | 0.799 ± 0.026 | 0.753 ± 0.033 | **−0.046** |

      **The mechanism is centring, not unboundedness — which is the opposite of the phrasing
      this item and §4.3 were reaching for.** The one arm that survives the pin is the one that
      is *not centred*. Every centred arm loses, and **(b)** decides it: clipping the z-score
      *increases* boundedness and makes the loss **worse** (−0.133 at ±2σ against −0.084
      unclipped), so bounding gives nothing back and the tails own none of the effect. **(a)**
      then reads exactly as this item's own decision rule said it would — *"if it degrades, the
      pin on 0.0/1.0 bucket means is the real culprit"* — and it degrades.
      **(c) inverts the natural guess:** z-scoring only `Slag` and `Age`, the two log-detected
      columns, costs −0.005; z-scoring only the other six costs −0.046. The damage does not
      follow the log step at all.
      **The underfitting prediction holds**, which is the falsification condition the script
      registers before running: train MSE/var rises with the loss (0.176 → 0.247 z-score,
      → 0.272 clipped ±2), so the model fits *train* worse too rather than overfitting.

      **What this changes in the prose.** The sharper claim this item hoped for — *"bounded
      normalization helps, centred normalization actively hurts, and the bounded-input
      assumption is load-bearing"* — is **half right and no longer current**. Centred
      normalization hurts, but the assumption that is load-bearing is *non-negative and
      uncentred*, not *bounded*; and since #102 it is not load-bearing at all, because nothing
      pins the extremes any more. At the shipped default §4.3 can say only that normalization
      helps (0.687 → 0.799) and that the choice among transforms is free — which is a weaker
      headline and a much easier one to defend. **G5's story improves in the same move**: G5
      recommended against pinning the extremes, upstream adopted that recommendation, and this
      is the measurement showing what the recommendation was worth.
      _Original entry, kept because its reasoning was right and its conclusion was overtaken:_
      *(Low priority — author 2026-08-03: "I don't need it but it's worth addressing." Nothing in
      the document depends on it; the choice itself is already settled. Data in hand:
      `reproduce/outputs/norm-three-arm-a385a1a/`, write-up in
      `reproduce/outputs/NORMALIZATION_THREE_ARM.md`, measured facts in `PROVENANCE_MAP.md`
      note 16.)*
      **Settled, so this is not a decision:** min-max (`UnitScalar`) is correct and is what the
      samples and the harness use. Confirmed by measurement — best-or-tied in 8 of 9 rows — and by
      author decision. z-score is *not* a candidate: it takes the 1st-order flat MoG to
      R² 0.087 ± 0.089, below raw features at 0.646 (RMSE 7.8 → 15.6 MPa), and drops the
      demo-tuned mixture 0.834 → 0.706.
      **What is left is the explanation**, which the chapter currently asserts rather than shows.
      The working account is that Gaussian membership functions and the `[0,1]`-pinned extreme
      bucket means assume a **bounded, non-negative** domain, so an unbounded centred transform
      breaks an assumption the construction relies on — supported by the model underfitting on
      *train* as well as test (MSE 0.030 vs 0.009), and by two innocent explanations already ruled
      out: ridge scale (sweeping `l2_reg` 1e-2 → 0 moves the gap 0.001) and the scale-dependent
      BIC membership count (pinning `n_gaussians` for an identical rule base still gives
      −0.407/−0.524/−0.634).
      **Cheap experiments that would settle it**, none needing new data: (a) `UnitScalar` with
      `feature_range=(-1, 1)` — if centring alone is harmless but unboundedness is not, this
      should behave like `[0,1]`, and if it degrades, the pin on 0.0/1.0 bucket means is the real
      culprit; (b) clip a z-score arm to a fixed range and see how much of the loss comes back;
      (c) check whether the damage concentrates on the log-detected features (`Slag`, `Age`),
      which are the ones whose post-transform distribution changes most.
      **Why it is worth the hour eventually:** §4.3's finding is currently "normalization helps",
      which is weak and slightly lucky — the code did the right thing under the wrong name. The
      sharper claim, *"bounded normalization helps, centred normalization actively hurts, and the
      bounded-input assumption is load-bearing"*, is a better answer to the obvious committee
      question and is already 90% measured. Pairs naturally with **A9** option C.
- [ ] 🟨 **E10 — Low-stakes editorial decisions**, bundled since none is blocking and none
      needs a research answer: (a) Ch 1 — how heavily to invoke the XAI/regulation framing
      (secondary per author); (b) Ch 2 — whether to include a formal-methods/verification
      subsection (possible Kreinovich nod); (c) Ch 5 — consolidate the Options A–D membership
      presentation (recommend leading with D + the persistence ramp, A/B/C supporting);
      (d) engineering debt — de-duplicate the six caller scripts' predict loops in `tribble-fis`.
      **(a), (b), and (c) are decided/done as of 2026-08-27** (V&V framing pass; review in
      `reviews/2026-08-24-prose-review-vv.md`). (a) *Lightly in Ch 1, substantively in Ch 2.*
      §1.1 keeps two clauses and a pointer; the argument itself lives in §2.6's new
      "Verification, validation, and certifiable AI" subsection, so the framing is stated once
      where the interpretability definition already is, and Ch 1 does not carry an argument it
      cannot finish. (b) *No separate formal-methods subsection; folded into §2.6.* The
      DO-333 nod is one sentence inside that subsection — formal methods enter as the reason an
      analysable rule base is worth more to a certification argument than a weight matrix, which
      is the only load the document needs them to carry. **The Kreinovich nod is in, and it is
      load-bearing rather than decorative** (a first pass dropped it after a bad search; the
      author overruled that, correctly). The citation is Cohen, Bokati, Ceberio, Kosheleva &
      Kreinovich, *"Why Fuzzy Techniques in Explainable AI? Which Fuzzy Techniques in Explainable
      AI?"* (NAFIPS 2021; LNNS 258:74–78) — `cohen2022whyfuzzy`. It earns its place twice: it is
      the general argument that fuzzy technique is the route to explainable AI, and its second
      half — *which* fuzzy operations are right is problem-dependent — lands directly on §4.3.5's
      own untested conorm-family question, so §2.6 cites it as a caveat this work owes rather
      than as an endorsement it collects. Two further finds came out of the same search and are
      worth more than the nod: **`arnett2021formal`** (Arnett, Ernest, Kunkel & Boronat, NAFIPS
      2020; AISC 1337:361–372) is a fuzzy UAV navigation controller *actually taken through formal
      verification* against a behavioural safety specification — an existence proof for the model
      family, in aerospace, on the constraint set §2.1 shares — and the volume is one Kreinovich
      co-edited. And **`arnett2019iteratively`** resolves what this file's lone `[?]`,
      `arnett2018proposal`, was a proposal *for*: the completed UC dissertation
      *"Iteratively Increasing Complexity During Optimization for Formally Verifiable Fuzzy
      Systems"* (2019). Both are now cited in §2.1, where the `.bib` header had claimed
      `arnett2018proposal` was cited all along and it was not.
      **(c) ✅ §5.3 methodology sections reorganized (2026-08-27)**: Chapter 5's §5.3 now leads with
      multi-scale approach (band discovery + persistence ramp memberships), supporting with flat
      set-cover as an alternative. Reordered §5.3.2–5.3.4 with all cross-references updated.
      
      **(d) landed upstream, unnoticed here until 2026-08-27** — `tribble-fis` #194 (`3daf0f4`)
      extracts `regressor_report` / `classifier_report` / `evaluate_model` into
      `tribble-tree/demo_utils.py` and moves `demo_concrete.py`, `demo_phishing.py`,
      `demo_deconstruct_synthetic.py` and `cmapss_deconstruct_eval.py` onto it, with no
      behavioural change. Verified at the pin by reading the module and its four callers, not
      from the commit message. Nothing in `reproduce/` touches those scripts, so no table moves.
      This is the fourth blocker in this file retired by an upstream commit nobody read —
      the pattern **B13**'s standing procedure exists to catch.
      **All four sub-items are now closed** — (a) and (b) on 2026-08-24, (c) and (d) on
      2026-08-27, the last two in parallel passes that did not know about each other. This item
      can be retired at the next tidy.
- [ ] 🟨 **E11 — The citation half is done (2026-08-27); the framing half is still open.**
      Every site this item was blocked on turned out to be reachable, so five entries were
      re-checked against the source rather than against a listing about the source, and the
      bibliography's unverified count went **4 `[?]` → 1**.
      - `easa2023airoadmap` → `[V]`. The printed title **leads with "EASA"**, which this file
        had dropped. 10 May 2023 confirmed. No document number is shown on the library page.
      - `easa2024mlconcept` → `[V]`, and this is the one the item flagged as genuinely
        unestablished. Exact date **6 March 2024**, where the entry had a bare year. EASA's own
        suggested citation also differs from what was here, and writes **"Issue 2"** in the
        citation against "Issue 02" in the announcement headline; the entry follows the citation
        and records the discrepancy.
      - `arnett2019iteratively` → `[V]`, read from the OhioLINK ETD record itself rather than
        the accession number that surfaced in search. Title, author, degree, department, year
        and committee all confirm — chaired by Kelly Cohen, which is a pleasing loop back to
        `cohen2022whyfuzzy`. **This is the outcome the item predicted:** it resolves what
        `arnett2018proposal`, the file's oldest `[?]`, was a proposal *for*, so that entry is no
        longer a mystery — only its own title and date are unconfirmed.
      - `cohen2022whyfuzzy` and `arnett2021formal` were `[V]` on SpringerLink *index* records;
        the chapter pages are still unreachable (both now sit behind an `idp.springer.com` auth
        redirect), so they were re-checked against **Crossref**, the publisher's own deposit.
        `cohen2022whyfuzzy` is exact. `arnett2021formal` is exact **apart from one given name**:
        this file had "Boronat, Hector", Crossref has **"Hugo"**. Changed, on the same standard
        **E7** used for "Sherin" → "Saima" — a hand-entered name with no recorded source does not
        outrank the publisher's deposit — but it rests on a single source, since search shows
        only "Boronat, H.", the NAFIPS 2020 program page returns 401, and the chapter page is
        behind auth. Left flagged in the entry rather than treated as settled.
      **Still open: the framing half**, which is the part that was never about egress. Original
      entry: *The citations:* `easa2023airoadmap` and
      `easa2024mlconcept` are `[?]` because `easa.europa.eu` is unreachable from the drafting
      environment, so both rest on index and news listings rather than a title page. The
      Roadmap 2.0 date (10 May 2023) is corroborated twice; the Concept Paper Issue 02's exact
      issue date is **not** established, and its entry carries a year with no month for that
      reason. Both are cited only for what a standard or roadmap *asks for*, never for a number,
      so a metadata correction cannot move a result — which is why this is proof-stage work and
      not blocking. Three more entries from the same pass need the same proof-stage pass, for the
      same egress reason: `cohen2022whyfuzzy` and `arnett2021formal` are `[V]` on their
      SpringerLink *indexed records* (title, authors, series, volume, pages, DOI, two independent
      listings agreeing) with the pages themselves unreachable, and `arnett2019iteratively` is
      `[?]` — its title and year come from index listings plus an OhioLINK accession number
      (`ucin156387481300899`) that surfaced in search but whose ETD record was blocked. That last
      one is the interesting item: if it confirms, this file's lone long-standing `[?]`
      (`arnett2018proposal`, placeholder title *"Dissertation Proposal"*) is resolvable to a real
      document rather than merely re-checked, so **verify these two together, not separately**. *The framing:* the V&V and certifiable-AI material added 2026-08-24 spans
      §1.1, §2.6, §4.3.5, §6.3.4, §7.4 and Ch 8, and every one of those passages states the same
      boundary — no certification artifact, no DO-178C or DO-333 objective claimed, no assurance
      case. A committee will test that boundary, so any later edit to one of those six passages
      needs the other five re-read. §7.4 is the load-bearing one: it is where the framing is
      recorded as an exposure with nothing in the runway behind it.


## Appendix — G2 datasets, verified 2026-07-31

**Primary: UCR/UEA time series under DTW.** Verified working in this environment.

Access (note the gotcha): `uv pip install` does **not** persist, because `uv run --project` re-syncs from the lockfile and reverts it. Use `--with` instead:

```bash
uv run --project tribble-cluster --with aeon python your_script.py
```

```python
from aeon.datasets import load_classification
X, y = load_classification("Crop")     # downloads on first call
```

| Dataset | N | length | classes | note |
|---|---:|---:|---:|---|
| **Crop** | **24,000** | 46 | 24 | the scale target; 24k² float64 ≈ **4.6 GB** — squarely in the memory regime Ch 3 exists for |
| **ElectricDevices** | 16,637 | 96 | 7 | second scale point |
| **StarLightCurves** | 9,236 | 1,024 | 3 | long series — DTW cost grows with length too |
| ECG5000 | 5,000 | 140 | 5 | mid-size |
| FordA | 4,921 | 500 | 2 | mid-size |

128 univariate datasets in the archive, all with ground-truth labels, so ARI is directly scorable.

**DTW is more non-metric than the synthetic proxy, measured here:**

| data | triangle-inequality violations |
|---|---:|
| GunPoint (DTW) | **29.3%** of sampled triples |
| ItalyPowerDemand (DTW) | 16.3% |
| *fractional Minkowski p=0.5 (the current synthetic stand-in)* | *14%* |

That is the sentence Ch 3 §3.4 wants: the real domain is *harder* than the synthetic case already reported, not a softer substitute for it.

**Why this family fits G2 better than the alternatives.** Warped time series have no fixed vector embedding — that is the entire premise of DTW — so the coordinate requirement is not merely inconvenient for the kd-tree/bounding-box methods, it is unsatisfiable. It therefore demonstrates both halves of the claim at once: non-metric correctness *and* the scaling regime, on the same data, with labels. Crop at 24,000 objects is also the natural place to exercise the on-demand distance computation, since materialising 288M DTW pairs is exactly what one wants to avoid.

**Second family (to confirm):** graph datasets under graph edit distance or a graph kernel — TUDataset (MUTAG, PROTEINS, ENZYMES, NCI1) — and the Duin & Pękalska dissimilarity collection, which is distributed *as distance matrices* and so matches the claim most literally. A verification pass on these is in progress.
