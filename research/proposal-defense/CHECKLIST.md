# Action items & burn-down checklist

Shared working list — merged 2026-08-04 from the former `ACTION_ITEMS.md` and `CHECKLIST.md`
into one document, since the two had grown into near-duplicates of each other with
inconsistent bookkeeping (miscounted table totals, a stale bibliography tally) between them.
Tick items as they land; each has enough context to start cold. IDs (A1, B6, C1, D2, E7, …)
are load-bearing — several are cited by ID from the prose chapters (e.g. Ch 7 §7 cites
**C1**, Ch 6 §6.4 cites **C9**, Ch 9 cites **D2**/**E8**/**E2b**) — so numbering is stable
across edits; a retired item keeps its ID rather than freeing it for reuse.

Companion docs: [`REVIEW_2026-08-02.md`](REVIEW_2026-08-02.md) (what was found and why,
dated snapshot — not updated after the fact), [`NEXT_STEPS.md`](NEXT_STEPS.md) (the
prioritized plan of record, i.e. *what to do next and in what order*; this file is
*everything*, in one flat list by category). Chapter 7 §7.5's Table 7.1 is the canonical
tracker for the research goals themselves (G1–G9, C1, C3, M5); this file does not duplicate
that table — items below that correspond to a Chapter 7 goal say so and point at it rather
than re-describing it.

_Opened 2026-08-02, merged 2026-08-04. Legend: ⬜ open · 🟨 in progress · ✅ done ·
🚫 descoped · 🔒 blocked on you._

---

## A. Author decisions — settled 2026-08-02, one reopened since, one new

_Kept as the record of what was decided and why, since several of these changed the document
materially and a committee may ask. **A9** is the reopened item (narrowed 2026-08-03); **A10**
is new, folded in from the former `ACTION_ITEMS.md`'s "needed from author" section._

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
- [ ] ⬜ **A10 — Teaching/RA load per semester** *(needed from author)*. Sets realistic
      throughput for Chapter 10's timeline; currently unconfirmed, which is the one open item
      that could move every bar in the Gantt. `10-timeline.md`'s "Open items" section already
      asks for this; recorded here so it has an ID like everything else waiting on you.
- [x] ✅ **A1 — Method name settled: `mergeVAT`** (author decision, 2026-08-02).
      The name went round-trip: mergeVAT → `pVAT` (on Dr. Kreinovich's observation that stage one
      is a priority-queue algorithm) → collision → back to **mergeVAT**. `pVAT` is taken by
      Parveen & Sreevalsan-Nair, *"pVAT: Parallel VAT on the GPU"*, BDA 2013 (LNCS 8302:151–170),
      a GPU VAT that also swaps the MST algorithm, so reading our *p* as parallel/performant
      collided harder rather than less. Acknowledged by citation in §3.3.1.
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
      Hedged wording ("assumed ~Dec 2026") removed from Ch 7 Table 7.1, Ch 10, and this file.
      Final defense stays March 2028, so the runway is 15 months as planned.

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
- [ ] ⬜ **B5c — Install a PDF renderer on this host.** `build_pdf.py` now assembles all
      thirteen sections and injects all fifteen figures on Windows (it previously died reading
      `chapters/09-publications.md` under cp1252), but rendering needs pandoc plus either a
      LaTeX engine or WeasyPrint's GTK runtime, none of which are present. The combined
      Markdown builds; the PDF does not. Not a code defect — a machine setup item.
- [x] ⬜ **B6 — Remove `pvat.vat_prim_mst_seq`.** Exported public API that silently
      returns a wrong ordering: it returns the seed vertex followed by every other vertex in
      ascending index order — chance-level agreement (0.001 ± 0.001) with the true ordering at
      both float64 and float32. Cause: `_get_dist(samples, u, vertices[mask])` is typed for
      scalar indices, so `np.sum(np.square(diff))` reduces over *all* candidates and returns
      one scalar; `key[mask] = <scalar>` gives every candidate the same key and the heap pops
      in index order. Nothing in the package calls it. See `REVIEW` ★2.
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
- [ ] ⬜ **C3 — Ch 5 end-to-end FIS result.** Every Ch 5 number is a *clustering* score; the
      chapter exists to produce FIS antecedents. Until a model is built from them and measured,
      the central claim rests on a proxy. Ch 7 §7.2 tracks this as **C3**, pulled forward into
      2027 Q3 rather than left inside the 2028 Q1 capstone alongside G6/G7/G8/write-up/defense.
- [x] ✅ **C4 — Quantify the correction-rule pass** (Ch 4 §4.3.1, Table 4.9, Fig 4.3;
      2026-08-05). Measured on Glass, ten paired seeds — not RT-IOT2022, which is still not in
      the repository, so the *scale* claim (twelve classes, eighty-three features) stays open.
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
- [ ] ⬜ **C13 — Large-scale regression benchmark, pilot started** (Appendix A.7's regression
      gap: no large dataset exists in any form). `reproduce/regression_scale/RESULTS_2026-08-05.md`
      piloted California Housing (20,433 × 8) and Superconductivity (21,263 × 81), single seed,
      not yet canonically sourced (both come from a GitHub mirror; UCI/figshare are unreachable
      from the session that ran this). Findings so far: California Housing works out of the box
      (R² = 0.660); Superconductivity's raw fit is badly broken (R² = −0.644) from feature
      collinearity that the library's own `top_p` selector cannot see, fixed by
      `sklearn.cluster.FeatureAgglomeration` decorrelation first (R² = 0.685 at the tuned peak).
      Table 6.1's model family run on both shows Random Forest beating every FIS-family arm by a
      wide margin on both datasets, and fuzzy tree beating tuned MoG on Superconductivity with no
      tuning at all. No decision yet on which dataset or model family, if any, is worth promoting
      to a `reproduce/tables/` generator — this item stays open until one is made and the chosen
      dataset is re-sourced from its canonical location.
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
- [x] 🚫 **C7 — DESCOPED, not done.** **DESCOPED from the proposal 2026-08-04.** §6.3.6,
      Table 6.4, Figure 6.3 and Goal C7 are removed from the document. The `MimoGaussian` /
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

## D. Writing and figures

- [x] ✅ **D1 — Produce the remaining figures.** All fifteen exist, generated by
      `reproduce/figures/` in PNG + EPS against one shared style module. Both load-bearing
      figures are done: **Fig 1.2** (pipeline roadmap) and **Fig 5.2** (band discovery).
      **Fig 4.3** was the last holdout — retargeted to the Glass correction-pass measurement
      (**C4**) rather than left waiting on RT-IOT2022, which still is not a dataset the harness
      can load; the reasoning is recorded in `registry.py`. Fig 6.3 was descoped with §6.3.6
      (see **C7**). What remains is a style pass on printed pages, not production.
- [ ] ⬜ **D2 — Write Chapter 9, and gather the NAFIPS metadata it needs.** Ch 9 is still an
      outline, and §3.3.1, §3.4, Ch 1 and Appendix A.3 all forward-reference §9.3. What is missing is
      author records rather than research: exact titles, page numbers/DOIs, co-author lists, which
      paper went to Banff 2025 vs. El Paso 2026, and whether the two published separately or
      combined. It is a writing task, not a decision, which is why it sits here rather than in
      §A above. Also gates the second pass on the acknowledgements (**A6**), which currently
      names the committee and Jon Salisbury but no co-authors. (Teaching/RA load, the other
      author-records item outstanding, is tracked separately as **A10**, since it is a
      scheduling input rather than writing.)
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
      Table 4.4's RT-IOT2022 accuracy → dataset absent from the repository. Table 6.2's M5 row
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
- [ ] ⬜ **E2c — Upstream fix for the CPU FCM formulation, filed as
      [clustering#62](https://github.com/fundthmcalculus/clustering/issues/62) (2026-08-04).**
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
      fix in the HFIS references (misattributed to "H. Wang et al."; see `bibliography.md` for
      the full accounting of this and four other reference-level gaps).
- [ ] ⬜ **E8 — Two blocking reads** before writing the Ch 9 complexity note (short-communication
      or NAFIPS-style venue, not an algorithms conference; novelty scoped to (a) the
      heap-vs-dense correction, (b) the measured crossover, (c) iVAT coverage Fast-VAT 2025
      lacks, (d) the O(N)-workspace regime as a ≈2× constant-factor win — explicitly *not* "a
      faster MST"): **Deshpande & Kumar 2024** full text and **Wang et al. 2010** (PAKDD). If
      the former already states the O(N)-workspace result for VAT itself, drop the note.
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
- [ ] ⬜ **E10 — Low-stakes editorial decisions**, bundled since none is blocking and none
      needs a research answer: (a) Ch 1 — how heavily to invoke the XAI/regulation framing
      (secondary per author); (b) Ch 2 — whether to include a formal-methods/verification
      subsection (possible Kreinovich nod); (c) Ch 5 — consolidate the Options A–D membership
      presentation (recommend leading with D + the persistence ramp, A/B/C supporting);
      (d) engineering debt — de-duplicate the six caller scripts' predict loops in `tribble-fis`.

## F. History — reconciliation sagas and resolved findings

_Carried over from the former `ACTION_ITEMS.md` because the sequence is instructive on its own,
not because the numbers below are still current — where a number changed later, the chapter
prose and the items above are the current source, not this section._

<details><summary>Five sagas — expand</summary>

**The bibliography.** Consolidated and verified against Crossref/arXiv/DBLP (2026-07-31 pass):
70 entries, **45 `[V]` + 23 `[S]` + 2 `[?]`** — see `bibliography.md` for the full accounting,
including the two open `[?]` entries and five entry-level gaps, four load-bearing. (An earlier
version of this line, in this file, read "47 `[V]` + 24 `[S]`, zero unresolved"; that was wrong
on every figure, as `bibliography.md` itself notes, and has been corrected above rather than
repeated here.) The one previously-broken citation is resolved: "[*Information Sciences*
2024]" is Deshpande & Kumar, *"Time and Memory Scalable Algorithms for Clustering Tendency
Assessment of Big Data"*, Information Sciences 664:120324 (2024), now cited by name in Ch 2
§2.2 and Ch 3 §3.2 as `deshpande2024scalable` — though its full text is still one of **E8**'s
two blocking reads.

**Ch 3 — three items resolved by code review and author decision (2026-07-31).** *Dense-Prim:*
an earlier note claimed a tuned O(N²) dense Prim was a missing baseline; it was not missing —
`_prim_mst_kernel_64/_32` in `pcvat.pyx` already *is* a compact-active-set dense Prim (no heap,
fused relax+next-min in one pass, O(N) workspace, O(N²) total), and it is the preferred import
path; the O(N² log N) heap version (`pvat.py::vat_prim_mst`) is the portable fallback. *Naming:*
the complexity story was reframed as a progression rather than a caveat — stage one (priority
queue, O(N³)→O(N² log N), published) then stage two (compact active set, O(N² log N)→O(N²),
unpublished) — which is now §3.3.1 as written. *Prior-art search:* complete, findings folded
into Ch 3/Ch 9, scope cut to the two blocking reads in **E8**.

**Two defensibility fixes, both retired into the prose.** The "priority-queue MST speedup"
framing for dense graphs was retired in favour of the two-stage story above, with the
three-arm timing harness as evidence. The ungrounded "six-orders-of-magnitude" web claim was
dropped outright — and the prior-art search that prompted the drop showed the opposite problem
existed too: a real prior **pVAT** (Parveen & Sreevalsan-Nair 2013) had to be conceded, driving
the rename saga in **A1**.

**Ch 4 §4.3.5 — the anomaly/open-set head-to-head ran; mechanism validated, absolute
performance not.** `table_4_4_openset.py`, leave-one-class-out on Glass (BETH is not in the
repository). The θ knob is monotone exactly as designed — raising θ shrinks the complement and
cuts both detection and false alarms, saturating past θ ≈ 1.1 — and there is no sharp optimum,
which argues for reporting the curve rather than tuning to a point. Absolute performance is a
real but noisy signal, not a deployable detector, and the complement rule vs. isolation forest
ordering **flipped three times across successive re-runs** (five seeds, then ten, then a
component-selection fix) — the tell that every one of those readings was noise rather than
signal. Table 4.6 and Table 4.7 in the current prose are the numbers to quote; nothing here
should be re-derived from an older run.

**Ch 4/Ch 7 — output partitioning (Goal G5), the full saga (hypotheses labelled H2–H5 in the
working record this section preserves).** Three studies at first and
second order found no usable difference between uniform and quantile partitioning (largest gap
0.012 in R² against σ ≈ 0.02–0.03) and concluded, wrongly, that partitioning didn't matter
(**H4**, later retracted at ten seeds). A
fourth study added **zeroth order** — where a rule's constant *is* its output — and the arms
separated by 0.828: uniform 0.394 ± 0.065, pure quantile 0.242 ± 0.070, the pinned-hybrid
default −0.434 ± 0.241. **H5** is that finding: the "hybrid" is a real third scheme and the
worst of the three, not bit-identical to pure quantile as first (wrongly) recorded — it differs
in all eighteen configurations tested, by noise-sized amounts no accuracy metric would flag.
Reading the solved coefficients showed why: the shipped default pinned
the extreme rules' constants to the target's global min/max, so the bottom rule emitted the
minimum for a bucket of 344 points whose mean was 0.195. A synthetic skew sweep (**H2**, then
**H3** on the tails) then confirmed
the mechanism directly and reversed an earlier 3-seed reading that had quantile *improving*
with skew (a genuine artifact — quantile in fact becomes *unstable* under skew, not less
accurate, with deviations exploding to ±24 while uniform degrades smoothly toward zero).
**Recommendation, now shipped as `partition_output`'s default: uniform, equal-width cuts, plus
a monotone target transform on badly skewed targets.** Ch 4 §4.3.2 and Ch 7 §7.2 (marked
"settled") carry the current, correct version; this paragraph is kept only as the record of how
many passes it took to ask the right question.

**Ch 4/Ch 6 — the Concrete reconciliation.** Three incomparable figures for "the flat model"
existed at once (Ch 4's 0.44/0.77/0.87, Ch 6 Table 6.1's 0.658, Ch 6 §6.3.5's refinement
0.88→0.92), each from a different configuration. Reconciling them under one protocol showed
the gap was mostly preprocessing and hyperparameters, not a real model difference — the
"Ch 6 inversion" (hierarchy beating the flat model) was largely an artifact of running the
hierarchy at library defaults, and tuning the mixture per `demo_concrete.py` closed most of the
gap. The refinement claim of 0.88→0.92 did **not** reproduce and was struck; the best model on
record became the unrefined closed-form full-2nd fit. All of this has since been superseded
again by the `main`-vs-superseded-branch discovery (★1 in `REVIEW_2026-08-02.md`) and re-quoted
against current code; Ch 4 Tables 4.1–4.5 and Ch 6 Table 6.1 are the current numbers. The
lasting lesson, not the numbers, is why this stays here: **a mean without a spread, over a
sample too small to contain the failure modes, is not evidence** — the ten-seed floor in
Goal G4a exists because of exactly this saga.

</details>

---

## Done in the 2026-08-02 session

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
Appendix A.5; four notes logged in this file.

**New results.** float32 in-place reaches 126,491 points under the cap and 154,919 on the full
machine, with an ordering elementwise identical to float64 across ten seeds. Chapter 6's
conclusion moved to *level* (tuned mixture 0.833 ± 0.024 vs flat 0.824 ± 0.043 at matched
capacity), which is what the chapter always wanted to argue. Refinement's decay sharpened to a
factor of twenty-five across consequent orders.

</details>
