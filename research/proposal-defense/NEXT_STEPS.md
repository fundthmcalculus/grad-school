# Next Steps

Prioritized plan of record. `ACTION_ITEMS.md` is the full backlog with findings and history; this file is what to do next and in what order.

**For the live burn-down list, see [`CHECKLIST.md`](CHECKLIST.md)** — tickable items with enough context to start cold, and the record of what the 2026-08-02 review pass closed. [`REVIEW_2026-08-02.md`](REVIEW_2026-08-02.md) is the findings report behind it.

_Last updated: 2026-07-31. Proposal defense ~Dec 2026 · final defense March 2028._

**Where things stand.** Chapters 1–8 and 10 are drafted and quality-passed (92 pages). The bibliography is consolidated and verified (70 entries, 0 unresolved). A reproduction harness runs 19 registered experiments, 6 of them verified end-to-end, and it has already produced four corrections to the text and one upstream bug fix. What is left divides cleanly into things only you can decide, cheap wins, and real research.

---

## Tier 0 — Blocking, and only you can do these

| # | Item | Why it blocks | Effort |
|---|---|---|---|
| 0.1 | **Merge `fix/pin-extreme-bucket-means`** ([PR](https://github.com/fundthmcalculus/tribble-fis/pull/new/fix/pin-extreme-bucket-means)), then re-run `reproduce/run.py --chapter Ch4 --chapter Ch6` | Every Concrete number in the text was measured under the *old* solve. Small deltas (≤0.01), but the tables should describe shipped code. | 1 h |
| 0.2 | **Rename the method — `pVAT` is taken.** Parveen & Sreevalsan-Nair (BDA 2013) published a *"pVAT: Parallel VAT on the GPU"* that also swaps the MST algorithm for the ordering. Reading the *p* as parallel/performant collides harder, not less. | Touches Ch 3 throughout, the method's identity, and how you introduce it to the committee. Independent of everything else, so do it early. | decision + 1 h |
| 0.3 | **NAFIPS paper metadata** (Ch 9): exact titles, page numbers/DOIs, co-authors, and which paper went to Banff 2025 vs. El Paso 2026 — and whether they published separately or combined. | Ch 9 cannot be finished without it. | your records |
| 0.4 | **Confirm the flagship end-to-end dataset** — RT-IOT2022 / IoT-botnet, or UCI-58 Shuttle. | Defines the Ch 7 capstone. | decision |

---

## Tier 1 — Cheap, high-value, do before the defense

1. **Produce the 15 figures.** None exist yet. Two are load-bearing: **Fig 1.2** (pipeline roadmap — orients the whole document) and **Fig 5.2** (band discovery on the log-birth spectrum — carries Ch 5's contribution). The rest are supporting. This is the single largest visible gap in the draft.
2. **Fill the 28 remaining `*pending*` table cells** by running the harness where the adapters exist, and marking the rest honestly.
3. **ANFIS and GA-tuned-FIS baselines** (Ch 4, Table 4.5). The speed claim is only as strong as what it is measured against, and these are the two methods the construction displaces. Needs adapters at `reproduce/tables/_baseline_anfis.py` and `_baseline_gafis.py`; the table auto-detects them.
4. **Install a LaTeX engine** so the PDF gets real typeset math: `sudo zypper install texlive-xetex texlive-latex texlive-collection-fontsrecommended`. `build_pdf.py` auto-detects and switches; display equations currently render as flattened glyphs.
5. **Quantify the correction-rule pass** (Ch 4 §4.3.1) — claimed but never measured. Paired confusion matrices, before and after.

---

## Tier 2 — Real research, scheduled in the Ch 10 timeline

| Goal | What | Target |
|---|---|---|
| **G4** | Repeatable-performance protocol: pinned clocks/thermals, ≥5 seeds, error bars, datacenter GPU with full-rate FP64. Consolidation point for every performance number in the document. Plus the eVAT/clusiVAT head-to-head Ch 3 owes. | 2027 Q1 |
| **G1** | One-pass membership generation (`MEMBERSHIP_ROADMAP.md` phases 1–6); phase 4 (soft kernel-weighted band membership) is the research-interesting piece. | 2027 Q2 |
| **G2** | **Real non-coordinate domains.** The single most important credibility gap: Ch 3's niche and Ch 5's premise both rest on *synthetic* non-metrics built from coordinate data. **Datasets identified and verified — see the G2 appendix below.** | 2027 Q3 |
| **G3** | HME EM refinement implemented, plus the full baseline suite (ANFIS, CART/C4.5, M5, flat TSK, Fumanal-Idocin 2025, D-TSK-FC). | 2027 Q3–Q4 |
| **capstone** | The integrated pipeline end to end. **Ch 5's membership functions have never been fed to Ch 6's models** — that link is specified but undemonstrated, which is why the capstone is a real experiment and not an integration chore. | 2028 Q1 |
| **G6** | Interpretability measured, not asserted: rule counts, path lengths, and either an established metric or a small expert study. Fills Table 6.3's pending row. | 2028 Q1 |
| **G7** | *(stretch, first to cut)* Adaptive band discovery for overlapping density scales. | 2028 Q1 |

**G5 (output partitioning) is reopened.** The three-seed recommendation ("quantile by default") did not survive ten seeds — the sign reverses in every row past symmetry, because quantile becomes *unstable* under skew rather than less accurate. See Ch 4 §4.3.2 and Ch 7 §7.2. What survives is the diagnosis; the remaining work is a decision, not corroboration: either characterize and guard quantile's instability, or accept that a heavily skewed target needs a transform rather than a better partition.

---

## Tier 3 — Defensibility, before submission

- **Ch 5 end-to-end result.** Every Ch 5 number is a *clustering* score, but the chapter exists to produce FIS antecedents. Until a model is built from them and measured, the central claim rests on a proxy. (Same work as the capstone.)
- **Ch 5 prior-art head-to-head** vs. Bonis–Oudot beta-plateau and AuToMATo on identical data. Given how close that work is, this is defensive as much as scientific.
- **Ch 3 datacenter GPU re-run.** The pairwise-distance kernel currently *loses* (<1×) at low dimension / float64 on a consumer card. The prediction that full-rate FP64 flips this is untested and labelled as such.
- **Ch 6 Atwood machine** result (Table 6.4 pending row) + re-verify the double-pendulum numbers.
- **Two literature searches**: knot/breakpoint optimization precedent (Ch 6), and a dedicated fuzzy-MoE search to bound the HME claim.
- **Fix the Zhang-2023 attribution** in the HFIS references (misattributed to "H. Wang et al.").
- **Optional — the VAT complexity note.** Scope was cut hard by the prior-art search; what survives is an audit piece (the literature asserts O(N²) *space* is inherent to Prim, which is false). Two blocking reads first: **Deshpande & Kumar 2024** full text and **Wang et al. 2010**. If the former already states the O(N)-workspace result for VAT itself, drop the note.

---

## Tier 4 — Editorial, low stakes

- Ch 1: how heavily to invoke the XAI/regulation framing (secondary per your call).
- Ch 2: whether to add a formal-methods/verification subsection.
- Ch 5: consolidate the Options A–D presentation (recommend leading with D + the persistence ramp).
- Ch 6: MIMO temporal memory as its own short chapter vs. a section (recommend section — it is a good aerospace hook either way).
- Engineering debt: de-duplicate the six caller scripts' predict loops in `tribble-fis`.
- Build a `run.py --all` pass and fix whatever else it surfaces.

---

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

---

## Suggested order

**This week:** 0.1 → 0.2 → 0.4, then start figures (Tier 1.1). Those unblock everything downstream and none of them need new research.

**Before the proposal defense:** finish Tier 1, and get G4's protocol defined even if not fully executed — every number in the document is reported under it, so the committee will ask.

**After:** Tier 2 in the scheduled order, with G2 started early since it has no upstream dependency and is the riskiest to leave late.
