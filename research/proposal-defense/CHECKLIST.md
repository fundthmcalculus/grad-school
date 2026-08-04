# Burn-down checklist

Shared working list. Tick items as they land; each has enough context to start cold.
Companion docs: [`REVIEW_2026-08-02.md`](REVIEW_2026-08-02.md) (what was found and why),
[`ACTION_ITEMS.md`](ACTION_ITEMS.md) (full backlog), [`NEXT_STEPS.md`](NEXT_STEPS.md) (plan of record).

_Opened 2026-08-02. Legend: ⬜ open · 🟨 in progress · ✅ done · 🔒 blocked on you._

---

## A. Author decisions — settled 2026-08-02, **one reopened 2026-08-03 (A9)**

_Kept as the record of what was decided and why, since several of these changed the document
materially and a committee may ask. The one item that used to live here and is still
outstanding — NAFIPS paper metadata — moved to **D2**, because it needs your records rather
than a decision. **A9 is new and narrowed**: measuring the normalization axis properly turned up
a naming decision the measurement cannot make for you, and the author has since ruled out its one
costly branch. It is not blocking, and it costs no numbers in either remaining direction. The
empirical follow-up it spun off is **E9**, deliberately low priority._

- [ ] ⬜ **A9 — Decide what the document *calls* its normalization. Narrowed to A or C.**
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

- [x] ✅ **A1 — Method name settled: `mergeVAT`** (author decision, 2026-08-02).
      The name went round-trip: mergeVAT → `pVAT` (on Dr. Kreinovich's observation that stage one
      is a priority-queue algorithm) → collision → back to **mergeVAT**. `pVAT` is taken by
      Saima Parveen & Jaya Sreevalsan-Nair, BDA 2013 (LNCS 8302:151–170), whose *method* is named
      pVAT — the paper itself is titled *"Visualization of Small World Networks Using Similarity
      Matrices"* and pVAT is contribution 1 of 3 inside it. It is a GPU VAT that also swaps the MST
      algorithm, so reading our *p* as parallel/performant collided harder rather than less.
      Acknowledged by citation in §3.3.1. Citation fully verified 2026-08-04 against Crossref and
      the full text (`refs/2013_ParveenSreevalsanNair_pvat_preprint.pdf`); the bib entry had been
      carrying the method name as the title, which is why a title search never surfaced the clash.
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
      ⚠️ **This broke the Ch 10 schedule and the fix is a decision, now taken and written into
      §10.5.** The grid had the Ch 5 paper in 2027 Q2 and Goal **G1** — the differentiator §5.5
      names — also in Q2. A February deadline is Q1, so the paper was scheduled a quarter late
      *and* its headline contribution would not have existed when it was due. Resolution: submit
      what §5.4 already supports (multi-scale recovery at ARI 1.00 vs 0.58–0.75 flat, the
      selection bake-off, the falsification experiment) and make G1 the journal/next-year
      extension. Cost, stated in §10.5: the EUSFLAT paper reports clustering scores rather than
      end-to-end accuracy — the same proxy limitation §5.4 already concedes.
- [x] ✅ **A6 — Acknowledgements written** (author, 2026-08-02). Template replaced with the real page; it renders correctly ahead of Chapter 1 in the build. One residual dependency: it thanks the committee and Jon Salisbury by name but not the NAFIPS co-authors, so give it a second pass once **D2** supplies those author lists.
- [x] ✅ **A5 — Proposal defense confirmed: December 2026.** *Decision: "Dec 2026. Let's GO!"*
      Hedged wording ("assumed ~Dec 2026") removed from Ch 7 Table 7.1, Ch 10, ACTION_ITEMS and
      NEXT_STEPS. Final defense stays March 2028, so the runway is 15 months as planned.

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
- [x] ⬜ **B8 — Automate the harness → document figure hop.** `save_figure()` writes to
      `reproduce/outputs/figures/fig_03_complexity_fit.{png,eps}`; the document references
      `prose/fig/03-complexity-fit.png`. That copy is **manual today**. `build_pdf.py` now
      emits an image line when the target exists and still strips it when it does not, so a
      figure that is not copied across silently reverts to a placeholder — which is the
      failure mode worth automating away. A name map plus a copy step in the build closes it.
- [x] ⬜ **B4 — Submodule SHA guard.** The harness should refuse to emit, or loudly stamp, when
      a submodule SHA differs from the last archive's. **This failure has happened twice** —
      once with `fix/pin-extreme-bucket-means`, once with `resolve-flm-pr`. Highest-value
      remaining infra item.
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
- [x] ✅ **B5b — §3.4's swept rows and Table 3.2 re-quoted from the workstation run.** Table 3.1
      reads 25× / 311× / 673× (was 28.8× / 398× / 1,129×); the 4,096-point stage-two figure is
      0.229 ± 0.006 s, so the comparison against the published stage-one measurement reads ~11×.
      Table 3.2's grid and exponents are re-quoted at classical **3.15**, stage one **1.86**,
      stage two **1.97**. Every cell was checked programmatically against the archive CSVs. The
      plateau and parity-band paragraphs are rewritten as an explicit retraction rather than
      replaced silently, and Ch 7's G4 is rebuilt around the lesson — repeatability cannot
      distinguish a property of the code from a property of the host. The appendix's hardware
      bullet and §3.4's two-hosts note are updated to match.
- [ ] ⬜ **B5c — Install a PDF renderer on this host.** `build_pdf.py` now assembles all
      thirteen sections and injects all fifteen figures on Windows (it previously died reading
      `chapters/09-publications.md` under cp1252), but rendering needs pandoc plus either a
      LaTeX engine or WeasyPrint's GTK runtime, none of which are present. The combined
      Markdown builds; the PDF does not. Not a code defect — a machine setup item.
- [x] ⬜ **B6 — Remove `pvat.vat_prim_mst_seq`.** Exported public API that silently
      returns a wrong ordering (seed vertex, then ascending index order). Cause is a vectorized
      call to a scalar-typed `_get_dist`. Nothing calls it. See `REVIEW` ★2.
- [ ] 🔒 **B9 — Backfill `log_features` into the sample scripts. BLOCKED on `tribble-fis` #73.**
      The samples were converted onto `UnitFuzzyScalar` (PR #55 here), which auto-detects log
      columns by dynamic range, whereas each sample previously named its own columns. Upstream
      #73 adds `log_features=[...]`, and once it merges and the gitlink is bumped past it, the
      lists below restore each sample's original logged set — turning behaviour-*changing*
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

## C. Experiments owed

- [ ] ⬜ **C1 — ANFIS and GA-tuned-FIS baselines** (Ch 4, Table 4.5). **The single most
      important experiment in the backlog**: the title, Ch 1, Ch 7 and Ch 8 all claim *orders
      of magnitude faster*, and there is currently no fuzzy baseline to say faster *than what*.
      Adapters go at `reproduce/tables/_baseline_anfis.py` and `_baseline_gafis.py`; the table
      auto-detects them.
- [x] ✅ **C2 — Complexity fit against reference curves.** Table 3.2 + Figure 3.2 now sweep a
      small grid (100–1,000, sized so the cubic arm runs at every point) with both axes
      normalized, and fit a log-log exponent per arm. Classical **3.15** (theory 3) confirms.
      **Stage one does not, and this item was ticked claiming it did.** Five runs on the
      workstation fit stage one at **1.86–1.88** against a theoretical ≈2.1 for
      $O(N^2 \log N)$ — but the number to notice is that it sits *below* the pure quadratic
      reference of **2.00**, so "the log factor is invisible over one decade" is asserted
      rather than shown, and an exponent under 2 is not evidence for a bound above it. What
      the sweep establishes is the cubic-to-quadratic *separation*, which both arms agree on;
      stage one's own exponent is **bounded, not confirmed**. Reporting a constrained fit at
      $t = c \cdot N^2 \log N$ beside the free exponent would settle it, and is the remaining
      work — tracked in Chapter 7 under G4a. Stage two fits **1.93–1.97**, which does confirm
      the quadratic claim it is making.
- [ ] ⬜ **C2b — RESCOPED: the ~10 ms fixed cost is a property of the laptop, not the kernel.**
      It does not reproduce on the workstation. Across **five** independent measurements there
      (`full-14900hx-2026-08-02`, its backfill, three manual repeats, `full-14900hx-r2`) stage
      two is monotone in N — 0.5 ms at N=750 rising smoothly to 8.4 ms at N=3,000, against the
      8–15 ms *flat* band described below — and it beats stage one by **8.1–17.7× at every
      size**, including 17.3× in the 750–1,250 band said to collapse to parity. The fitted
      exponent is **1.93–1.97** here against the laptop's 2.12/2.13, i.e. a cleaner
      confirmation of the quadratic claim, since the plateau was contaminating that fit — the
      chapter already calls 2.12 "right for the wrong reason".
      **The question to answer is now "why did the laptop have it", not "what is it".** The
      OpenMP-parallel-region hypothesis is *weakened*: thread-startup cost should be at least
      as visible on 32 cores as on 4. A 4-core `powersave` governor ramping clocks on thread
      spawn fits the evidence better, and is testable by pinning the laptop's governor.
      **The chapter's claim that the compiled kernel "buys nothing across a band of problem
      sizes" is not supported on the workstation and must be qualified by host.**
      *(Original characterization, which remains accurate for the development laptop:)* With the grid extended to 3,000 the picture is
      no longer "noise": stage two tracks $N^2$ cleanly to N = 500, **acquires a fixed cost of
      roughly 10 ms at N ≈ 750**, and then runs flat — 8 to 15 ms everywhere from 750 to 3,000
      — until the quadratic work catches up near 3,000. Exponents are stable across runs
      (classical 3.07/3.07, stage 1 1.80/1.81, stage 2 2.13/2.12), so this is a real effect and
      not the timer. *(It is real and repeatable — on that host. Reading repeatability as
      "a property of the kernel" is the step the workstation run invalidates: stable across
      runs is not the same as stable across machines, and only one machine had been tried.)*
      **Practical consequence:** below 750 stage two beats stage one by 5–8×; between ~750 and
      ~1,250 the advantage **collapses to parity** and which arm wins varies between runs;
      above 1,500 stage two recovers to 6.7× by N = 3,000. So the compiled kernel that
      §3.3.1 says is "preferred at import time" buys nothing across a band of sizes.
      **Leading candidate:** OpenMP parallel-region setup in the Cython `nogil` path — a
      threading threshold above a size cutoff produces exactly this step-then-plateau
      signature. A cache boundary or an allocation path would also fit. All testable; none
      tested. If it is thread startup, a size-gated serial path below ~1,500 is the fix.
- [ ] ⬜ **C10 — Generalize the merge operator** *(new; the method is named after this, and it is
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
      Noted briefly in Ch 7 G4.
- [ ] ⬜ **C3 — Ch 5 end-to-end FIS result.** Every Ch 5 number is a *clustering* score; the
      chapter exists to produce FIS antecedents. Until a model is built from them and measured,
      the central claim rests on a proxy. **Recommend pulling a minimal version into 2027 Q2**
      rather than leaving it all in the 2028 Q1 capstone alongside G6/G7/G8/write-up/defense.
- [ ] ⬜ **C4 — Quantify the correction-rule pass** (Ch 4 §4.3.1). Claimed, never measured.
      Paired confusion matrices, before and after. Fills Fig 4.3.
- [ ] ⬜ **C11 — Benchmark `IVATMeans` against FCM and k-means** *(new 2026-08-03; Ch 7 **G9**,
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
- [x] 🚫 **C7 — DESCOPED, not done.** **DESCOPED from the proposal 2026-08-04.** §6.3.6, Table 6.4, Figure 6.3 and Goal C7 are removed from the document. The `MimoGaussian` / `AnalyticalDynamics` work continues separately and the proposal no longer rests on it, so nothing here is owed *to the proposal*. Kept as a record of what was found, not as an open item. Original item: **Ch 6 Atwood machine result** (Table 6.4 pending row), and reconcile Table 6.4's
      R²/RMSE pair — 0.92/0.045 implies target σ ≈ 0.159, 0.96/0.028 implies ≈ 0.140. It is also
      the one table `PROVENANCE_MAP` marks ungenerated while Ch 6 calls it the clearest result.
- [ ] ⬜ **C8 — Ch 3 datacenter GPU re-run.** The pairwise-distance kernel loses (<1×) at low
      dimension / float64 on a consumer card; the prediction that full-rate FP64 flips it is
      untested and labeled as such.
- [ ] ⬜ **C9 — Ch 6 interpretability, measured** (G6): rule counts, path lengths, and either an
      established metric or a small expert study. Fills Table 6.3's pending row. Until then Ch 6
      must keep saying the payoff is *described*, not quantified.

## D. Writing and figures

- [x] ✅ **D1 — Produce the remaining figures.** Fourteen of fifteen exist, all generated by
      `reproduce/figures/` in PNG + EPS against one shared style module. Both load-bearing
      figures are done: **Fig 1.2** (pipeline roadmap) and **Fig 5.2** (band discovery). One
      is skipped on purpose and recorded with a reason in `registry.py`: **Fig 4.3**, the
      correction-pass experiment does not exist. Fig 6.3 was descoped with §6.3.6. What remains is a style pass on printed pages, not production.
- [ ] ⬜ **D2 — Write Chapter 9, and gather the NAFIPS metadata it needs.** Ch 9 is still an
      outline, and §3.3.1, §3.4, Ch 1 and Appendix A.3 all forward-reference §9.3. What is missing is
      author records rather than research: exact titles, page numbers/DOIs, co-author lists, which
      paper went to Banff 2025 vs. El Paso 2026, and whether the two published separately or
      combined. *(Was tracked as A2; it is a writing task, not a decision, so it belongs here.)*
      Also gates the second pass on the acknowledgements (**A6**), which currently names the
      committee and Jon Salisbury but no co-authors.
- [ ] ⬜ **D3 — Regenerate `chapters/00-README-master-outline.md` from the prose.** It still reads
      "Status: Scaffold," describes pillar 1 without stage two or the name collision, and lists
      two already-completed fixes as pending. It is the document a committee member is most
      likely to open first.
- [x] ✅ **D4 — The `*pending*` table cells are all marked or resolved** (2026-08-02).
      **The count was wrong: 20, not 23.** This item, `NEXT_STEPS` and `ACTION_ITEMS` all
      said 23 and `REVIEW_2026-08-02` said 22; the actual inventory across the chapter tables
      was 20 — Ch 3: 1, Ch 4: 10, Ch 6: 9. (The three stray counts came from counting rows and
      prose mentions along with cells.) None is now a bare `*pending*`: each names what blocks
      it and the item that tracks it, so the gaps can be triaged rather than re-derived.
      ANFIS / GA-FIS, 11 cells across Tables 4.5 and 6.2 → `N/A (C1)`. Table 3.7's
      non-coordinate row → Goal G2. Table 6.3's counts → C9 / G6.
      Table 4.4's RT-IOT2022 accuracy → dataset absent from the repository. Table 6.2's M5 row
      → **a dependency fault, not an unrun experiment**: the generator already imports `m5py`
      optionally and would fill it unattended, but `m5py` does not load against scikit-learn
      1.9.0. Table 4.5's full-2nd training time → deliberately left empty; see note 14 in
      `PROVENANCE_MAP.md`, because that row's R² and its available timing come from two
      different code paths and pairing them would repeat the mismatch the table's caption
      exists to prevent. **What this item did not do is fill any cell with a number**, and the
      three follow-ups it exposed — C1, C7 and the Table 4.5 timing split — are the real work. (C7 is since descoped; see its entry above.)
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
      fp16 and the §9.4 standalone-paper item are still open and unaffected.
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
- [ ] ⬜ **E9 — `UnitScalar` vs `StandardScalar`: characterize *why* bounded normalization wins.**
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
G5 marked settled in two documents; estimates-vs-demonstrations standard written into G4 and
Appendix A.5; four notes logged in `ACTION_ITEMS.md`.

**New results.** float32 in-place reaches 126,491 points under the cap and 154,919 on the full
machine, with an ordering elementwise identical to float64 across ten seeds. Chapter 6's
conclusion moved to *level* (tuned mixture 0.833 ± 0.024 vs flat 0.824 ± 0.043 at matched
capacity), which is what the chapter always wanted to argue. Refinement's decay sharpened to a
factor of twenty-five across consequent orders.

</details>
