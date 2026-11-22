# 2026-11-22 — Full prose recheck against `full-03d327a-2026-11-21`

Every `mean ± std` pair and every table-cell-like bare number in `research/proposal-defense/prose/` was traced to the
cell it quotes (file / row / column), then read from the run of record. Method: a cell tracer over every archive under
`reproduce/outputs/`, then a per-section read for the cells it could not place unambiguously. Nothing in the prose was
edited by this pass. `check_prose.py full-03d327a-2026-11-21 --against full-2026-08-22`: 59 match / 96 drifted / 159
untraceable. Most of the gap is that the prose cites older archives by name, or quotes cells this run does not produce.

Classes: SAME, DDOF (#143, population → sample std), ROUND, MOVED_EXPECTED (pin bump, B14 fix, or the deferred
Concrete host note), MOVED_UNEXPLAINED, HOST (timing), NOT_RUN, CITED_ARCHIVE (the sentence names its source),
UNTRACEABLE, DERIVED.

**The controls held.** CART and Random Forest keep their means in every table in scope; they differ only by the ddof
factor. Chapter 5 matches the regenerated `results.json` everywhere.

## Priority 1 — the prose quotes a retracted number

- **04:594–596**: the sentence says every cell comes from `phiusiil-leakfree-2026-08-30/table_4_1.csv`, then quotes
  Concrete 0.795 ± 0.025 at 0.43 s and **PhiUSIIL acc 0.997 ± 0.001** at 0.28 s. The named file holds Concrete
  0.799 ± 0.027 at 0.53 s and PhiUSIIL **0.440 ± 0.181** at 0.13 s. 0.997 is the leak-era value. The warm-up paragraph
  (04:600) and the 0.44 s at 04:661 come from the same wrong archive.
- **06:308–309**: "flat 0.997 ± 0.001, tree 0.970" on PhiUSIIL. Table 6.1 (new run): flat 0.440 ± 0.181, tree
  0.958 ± 0.003. The chapter contradicts its own Table 6.2.
- **A:1602** (Table A.6): HME on PhiUSIIL 1.000 is leak-era. The new run gives 0.914 ± 0.039, so "parity with CART/RF" is false.

## Priority 2 — a claim changes under the run of record

| Where | Claim | Run of record |
|---|---|---|
| 04:47–48, 163–170, 1005–1007 (Table 4.9) | correction pass helps; gated cascade +0.031 ± 0.027 | +0.0077 ± 0.0109 (CI spans 0); the flattened FIS has 80.9 MF, fewer than the base's 81.4. Documented in map note 29, but the prose was never updated |
| 04:283–308 (Table 4.3 skew sweep) | quantile trails uniform in every row; uniform's spread stays bounded; starvation past skew 14 | Q−U +0.020 at skew 5.32; uniform 0.025 ± 0.225 at skew 17.7; min bucket is 0 from skew 5.32 (tribble-fis #95 `kmeans_1d`; already in `goal-8h-2026-08-11`). Also 07:353 |
| 04:85–86 (Table 4.1 mixture / tree rows) | z-score trails min-max (Δ −0.028 / −0.012) | Δ +0.000 / +0.003. Moved between `goal-8h-2026-08-11` and `all-restored-2026-08-22`. **Cause undocumented** (not #95) |
| 01:114, 04:520, speedup range | 14×–194×; "one to two orders of magnitude" | 4×–72× on this host (HOST). Bike Sharing at 4× breaks the range |
| 04 (RF vs MoG on Bike Sharing) | RF not slower than MoG | false in old and new runs (7.98 vs 0.59 s; 4.26 vs 0.65 s) |
| 04:177–180, Fig 4.7 caption (Table 4.8 summary) | 2×–10×, 0.0%–44.2% | 3×–50×, 2.4%–66.5% (inverts the Breast Cancer parenthetical) |
| 06:336 | refined 2nd 0.862 ≈ full-2nd 0.861 | 0.875 ± 0.029 vs 0.859 ± 0.027: refinement now wins by more than one spread. The refined arms moved while `refine.py` changed (#134/#136/#178/#209); **no map entry records it** |
| 06:325, Fig 6.5 | 2nd-order refinement gain | now +0.035 (> one spread); the "factor of nine" becomes about 10 and still holds |
| A:567 | 0.914 / 0.072 / 0.037, factor of 25 | stale (`main-d0efefc`); now +0.348 / +0.037 / +0.035, about 10× |
| 03:399–405 | pVAT 4,096 = 0.231 s, ~11× under the published 2.56 s, bracketed by the three-arm 7.4–17.5× | 0.145 s (17.7×); three-arm s1/s2 now 5.4–11.6×, **so it no longer brackets and the reconciliation argument fails** (HOST, but the argument depends on it) |
| 03:702–707, DATASETS:163–164, 281 | Crop gap within 0.05; ElectricDevices/StarLightCurves not measured | stale against `g2-downstream-2026-08-25`: Crop 0.084; both were measured |
| 04 Table 4.4 Concrete order-1/2 rows | 0.787 / 0.832 | disagree with 04:547 and the new run (0.795 / 0.840) |

## Priority 3 — wrong in both runs (arithmetic and labels)

- 04:233 "twelve times the widest spread" is about 3×. 04:245 "0.005" computes to 0.002/0.004. 04:308 "falls 0.051→0.060" is a rise.
- 04 Table 4.10: `raw_memory` is marked best per-sample at 15.58, but `whole_cycle` is 15.44.
- A:1608 "nearly a factor of three" is about 2.2× against the table above it. A:241 "0.96 takes fifteen features under
  every scorer": composite reaches 0.9761 at 10. Table A.2 omits the new `higher_order` scorer (0.9261 vs 0.9186 at 5).
- 02:259 "+0.017" (archive and Ch 6: +0.016). DATASETS:277 "4.2 s" is in no archive and contradicts DATASETS:44's
  3.64 s. DATASETS:43–44, 277 label the RT-IOT2022 figures "Table 4.4", but they belong to 4.1/4.1b.
- Ch 4 sentences say "ANFIS/GA not run", "adapters absent", and "RT-IOT2022 θ-sweep is future work", while the chapter's own tables show otherwise (Table 4.7c exists).
- Table 4.12's spreads are population std at n = 5, against the #143 convention.

## Unexplained moves

- **Table 4.8**: Diabetes Δ at 1× and at max-lossless (2× → 3×). Map note 29 already lists this as "cause not
  identified". Concrete at max-lossless −0.0118 ± 0.0305 → −0.0122 ± 0.0332 is not a pure ddof change, and note 29
  doesn't cover it.
- The Table 4.1 mixture/tree z-score change, and the refined arms (above).

## Mechanical re-quotes (no claim changes)

The DDOF rows (spread ×√(10/9), or ×√(60/59) for open-set Table 4.4 at 04:726–728), ROUND, and the small
MOVED_EXPECTED Concrete cells (0.861 → 0.859, 0.394 → 0.399, −0.434 → −0.425, 0.796 → 0.795). The Concrete ones are
the deferred same-pin host note: the Xeon archive at the same pin still gives the old means.

## Not checkable against this run

A.8 HDBSCAN study (about 65 cells; its script is not in the runner), A.7, RT-IOT2022 / BETH rows, FordA/Crop, Table 3.4
GPU, and Table 3.3's float32 on-demand cell (`check_matrix_free_reorder.py` archives nothing).

---

Per-cell ledgers follow, one per section of the document.


---

## Ledger: Ch 1, 2, 3, 5, 7, 8, DATASETS vs `full-03d327a-2026-11-21`

Scope: every `mean ± std` pair and every 3+-decimal table-cell number in 01, 02, 03, 05, 07, 08 and DATASETS.md.
"NEW" means `reproduce/outputs/full-03d327a-2026-11-21/` unless another path is given. Chapter 5 synthetic values were
checked against both the archived `table_5_{1,2,3,4}` files and `gated-minimax-selection/outputs/results.json`
(regenerated 2026-11-21 21:18 on the run-of-record host, seeds 0..4).

Two things to know before reading the rows:
- **Concrete MoG cells moved about 0.002 to 0.009 at the same pin.** `bump-03d327a-2026-11-21` (Xeon, same pin
  03d327a) still gives the old means, e.g. 0th order 0.394 ± 0.068 and hybrid −0.434 ± 0.254. The new i9 archive gives
  0.399 ± 0.063 and −0.425 ± 0.270. That is the BLAS / summation-order note in the READ FIRST section of PROVENANCE.txt,
  which was "deferred by decision". It is not a pin bump. It is the only documented explanation, so I class these
  cells MOVED_EXPECTED.
- **Table 5.4 runs ten seeds, not five** (`table_5_4_ch5_g1_scaling.py` runs its own seeds 0..9). Its ddof factor is
  therefore √(10/9), not √(5/4). Tables 5.1 to 5.3 have no spreads at all (single deterministic run per cell), so the
  √(5/4) factor never comes into play in my lines.

| file:line | prose value | quantity (table / row / col) | new-archive value (file) | class | note |
|---|---|---|---|---|---|
| 01:22 | R² 0.861 ± 0.026 | Concrete flat MoG-TSK full-2nd, log + min-max (Hyperparam/normalization table) | 0.859 ± 0.027 (table_hyperparam_normalization.csv) | MOVED_EXPECTED | pins-synced-2026-08-27 had 0.861 ± 0.028, so the spread change is ddof. The mean −0.002 is the same-pin host/BLAS Concrete note. "Level with the tree" still holds against CART 0.826 ± 0.049 |
| 01:23 | 0.826 ± 0.047 | CART, sklearn default, log + min-max | 0.826 ± 0.049 (hyperparam_normalization; concrete_reconciliation log+std is the same) | DDOF | 0.047 × 1.054 = 0.0495 |
| 01:23 | 0.909 ± 0.019 | RF, log + min-max column (same protocol as the 0.861) | 0.909 ± 0.020 (hyperparam_normalization) | DDOF | The raw column and Table 6.1 show 0.909 ± 0.019, a coincidental hit. Under "one protocol" the right cell is 0.020 |
| 01:24 | 0.394 ± 0.065 | Concrete 0th order, uniform, 3 buckets, closed form (G.5 / concrete_reconciliation) | 0.399 ± 0.063 (table_g5_output_partitioning.csv, table_concrete_reconciliation.csv) | MOVED_EXPECTED | Same pin on the Xeon: 0.394 ± 0.068, which is the prose's 0.065 × √(10/9). The +0.005 is host/BLAS. "Less than half as good" still holds (0.399/0.859 = 0.46) |
| 01:107 | 0.13 ± 0.02 s | Table 4.1 PhiUSIIL, MoG train time | 0.17 ± 0.01 s (table_4_1.csv) | HOST | Source phiusiil-leakfree-2026-08-30 |
| 01:108 | 0.440 ± 0.181 | Table 4.1 PhiUSIIL, MoG accuracy | 0.440 ± 0.181 | SAME | |
| 01:112 | ANFIS 0.999 ± 0.001 | Table 4.1 PhiUSIIL, ANFIS | 0.998 ± 0.002 | MOVED_EXPECTED | ANFIS is documented as nondeterministic run to run and imports nothing from tribblefis. PROVENANCE_MAP (bump-03d327a section) records 0.998–0.999 throughout. ANFIS now ties the GA-FIS instead of leading it; the sentence's point is unaffected |
| 01:112 | GA-FIS 0.998 ± 0.001 | Table 4.1 PhiUSIIL, GA-tuned FIS | 0.998 ± 0.001 | SAME | |
| 01:112 | CART 0.997 ± 0.001 | Table 6.1 PhiUSIIL, accuracy, CART | 0.997 ± 0.001 (table_6_1.csv) | SAME | |
| 01:113 | RF 1.000 ± 0.000 | Table 4.1 PhiUSIIL, tree/RF ref | 1.000 ± 0.000 | SAME | |
| 01:114 | 194× | Table 4.1b PhiUSIIL, MoG speedup vs slowest fuzzy | **72×** (table_4_1b_baseline_timing.csv) | HOST (DERIVED) | 194× came from phiusiil-leakfree (ANFIS 22.5 s, GA 25.3 s). On the i9 the baselines take 11.6 / 11.9 s, so the claimed magnitude shrinks by 2.7×. **Claim affected** |
| 01:117 | 3.64 ± 0.25 s | Table 4.1/4.1b RT-IOT2022, MoG train | N/A | NOT_RUN | RT-IOT2022 is absent. Verified present in phiusiil-leakfree-2026-08-30 |
| 01:117 | 0.927 ± 0.002 | Table 4.1 RT-IOT2022, MoG accuracy | N/A | NOT_RUN | Same; verified in phiusiil-leakfree-2026-08-30 |
| 01:118 | 0.998 ± 0.000 | Table 4.1 RT-IOT2022, RF | N/A | NOT_RUN | Same; verified in phiusiil-leakfree-2026-08-30 |
| 01:128 | ARI 1.000 on the coarse level, all three sets | Table 5.2 flat_ari_per_level, coarse | 1.0 / 1.0 / 1.0 (results.json; table_5_2) | SAME | |
| 01:129 | 0.24 to 0.49 | Table 5.2 flat per-level ARI on the missed levels | 0.236 / 0.324 / 0.492 (results.json) | SAME | |
| 01:130 | 1.000 on each | ms_best_ari_per_level | all 1.0 | SAME | |
| 02:24 | 0.861 ± 0.026 | as 01:22 | 0.859 ± 0.027 | MOVED_EXPECTED | as 01:22 |
| 02:24 | 0.826 ± 0.047 | as 01:23 | 0.826 ± 0.049 | DDOF | |
| 02:24 | 0.909 ± 0.019 | as 01:23 | 0.909 ± 0.020 | DDOF | |
| 02:25 | 0.394 ± 0.065 | as 01:24 | 0.399 ± 0.063 | MOVED_EXPECTED | as 01:24 |
| 02:259 | +0.017 to +0.025 | Refinement study, hot arms ΔR² vs construction (opt-hotcold-kmbic-2026-08-03/table_opt_hotstart.csv) | not in run of record | NOT_RUN | The optimizer study is not in run_all_tables. **The archive range is +0.016 to +0.025** (PSO and ACO +0.016), and Ch 6:232 says +0.016. Ch 2 has a transcription mismatch |
| 03:215 | 1.000 ± 0.000 | Matrix-free ordering agreement at N = 1k / 2k / 5k | not archived | NOT_RUN | Comes from `reproduce/experiments/check_matrix_free_reorder.py` (stdout JSON only, not in the harness). Partial corroboration: table_3_2 float64 on-demand reads 1.000 (exact) at N = 2,000 |
| 03:216 | 0.001 / 0.0005 / 0.0002 | Chance levels | 1/N | DERIVED | Exact |
| 03:401 | 0.231 ± 0.011 s | Table 3.1 pVAT at N = 4,096 | **0.145 ± 0.002 s** (table_3_1.csv) | HOST | Source uniform-2026-08-03. **Claim affected:** "about 11× faster than 2.56 s" becomes 2.56/0.145 ≈ 17.7× |
| 03:403 | 7.4–17.5× | Three-arm stage-1/stage-2 ratio over the grid | **5.4–11.6×** (table_3_1_three_arm.csv, s1/s2) | HOST (DERIVED) | **Claim affected:** the new 17.7× is no longer bracketed by the three-arm range. The paragraph's reconciliation ("differ by very nearly the amount stage two predicts") fails on this host |
| 03:586–590 | bytes/entry, 98.3 / 32.8 / 49.2 / 16.4 GB, 51,639 … 154,919, 1.000 (exact) | Table 3.3, 5 dense/on-demand rows | identical (table_3_2_memory_precision.csv) | SAME | Exact arithmetic plus measured exact orderings |
| 03:591 | 0.9996 ± 0.0012 (ties) | Table 3.3 float32 on-demand, ordering column | **0.999 ± 0.002** in the table's own generator | UNTRACEABLE | The prose cell is not the generator's cell. 0.9996 would round to 1.000, so it is a different measurement (pooled N ∈ {1k, 2k, 5k} from check_matrix_free_reorder.py). That script has no archived output; the only other occurrence is CHECKLIST.md:346, a transcription |
| 03:604 | 0.9996 ± 0.0012 | as above | — | UNTRACEABLE | as above |
| 03:621 | 0.9996 ± 0.0012 | as above, attributed to check_matrix_free_reorder.py | — | UNTRACEABLE | Script named, but nothing archived to verify against |
| 03:624 | float32 0.999 ± 0.002 | table_3_2 run of record, float32 on-demand | 0.999 ± 0.002 | SAME | Stable since full-2026-08-22 |
| 03:691 | 1.000 (ECG5000) | Table 3.7 last row, exactness | 1.000 (table_3_7_g2_dtw_nonmetric.csv) | SAME | |
| 03:691 | 1.000 (FordA, Crop) | same | absent | NOT_RUN | The run of record's 3.7 generators ran ECG5000 only. Verified 1.000 in goal-8h-2026-08-11-fullsuite |
| 03:692 | Crop reorder 4.7 s | 3.7 timing | absent | NOT_RUN | goal-8h-2026-08-11-fullsuite: "1597s matrix + 4.7s reorder" |
| 03:696 | 20.9% / 23.6% / 0.4% | Triangle-inequality violations | ECG5000 20.9% SAME; Crop and FordA absent | SAME / NOT_RUN | |
| 03:704 | 0.122 ARI | ECG5000 set-cover minus NERFCM-given-k | gap = 0.122 (table_3_7_g2_downstream.csv) | SAME | The surrounding claim is stale; see Claims affected |
| 05:192, 05:804 | ARI 1.000, seven partitions (swept flat cut) | §5.4 baseline sweep | not in results.json or any archive table | NOT_RUN | Not a harness output |
| 05:232 | k = 2 ARI 1.000 vs flat k = 3 at 0.982 | multiscale_no_regression.bridged | 1.0 / 0.982 | SAME | |
| 05:398 | 0.715 vs 0.593 | ECG5000 set-cover vs NERFCM-given-k | 0.715 / 0.593 ± 0.047 (table_3_7_g2_downstream.csv) | SAME | |
| 05:417–421 | Table 5.1, all cells | main_table / persistence_methods | identical (table_5_1_battery.csv) | SAME | The prose's well_separated row is results.json's `two_gaussians` |
| 05:425 | k-means 1.00 / 1.00 / 0.98 / 0.017 | main_table.kmeans_ari | 1.0 / 1.0 / 0.98 / 0.017 | SAME | |
| 05:445 | sizes 58, 3, 3; purity 1.000; coverage 0.533 | selector_comparison.bridged.coverage_cover | same | SAME | |
| 05:452 | cover_nblocks 4, coverage 0.125 | main_table.uniform_noise | same | SAME | |
| 05:455–466 | 0.982, 0.001, ms_finest_ari 1.000 | main_table / multiscale_no_regression | same | SAME | |
| 05:479–485 | 0.66/0.58/0.75 → 1.00; 0.662/0.576/0.746; 1.000 | Table 5.2 | same | SAME | |
| 05:493–496 | band vectors [1.000, 0.581, 0.236] … | three_level bands ari_per_level | same | SAME | |
| 05:506–508 | Table 5.2, all cells | table_5_2_multiscale | same | SAME | |
| 05:514 | birth edges 1.593 and 10.311 | three_level bands birth_hi | 1.593 / 10.311 | SAME | |
| 05:523, 05:539 | flat_k 3, flat_ari 0.983 at all six settings | multiscale_scale_invariance | 0.983 × 6 | SAME | |
| 05:557–558 | 5–7 of 10 seeds; ≈0.98–1.00 | Table 5.4 single_scale | (6, 6, 6, 5, 7, 6)/10; 0.98, 1.00 | SAME | |
| 05:564 | 0.73 ± 0.20 | Table 5.4 log_separated n = 100, two-stage ARI | 0.73 ± 0.21 (table_5_4_ch5_g1_scaling.csv) | DDOF | n = 10 seeds, so ×√(10/9). The n = 250 and 500 cells are 0.72 ± 0.20 and 0.73 ± 0.21 |
| 05:564 | 0.99 ± 0.01 | Table 5.4 log_separated n ≥ 2,000 | 0.99 ± 0.01 (n = 2000); 0.99 ± 0.00 (n = 5000) | SAME | |
| 05:566 | 0.85–0.99 | Table 5.4 log_separated flat ARI | 0.85 … 0.99 | SAME | |
| 05:588–590 | 0.001, 0.533, 0.927, 0.891, 98.3%, 97.5% | Table 5.3 | same | SAME | |
| 05:605–607 | Table 5.3, all cells | table_5_3_selection | same | SAME | |
| 05:626 | ARI 1.000 on D and D* (two cases) | relational_table | 1.0 / 1.0 | SAME | |
| 05:627, 05:657 | original ARI 0.285 | multi_scale_hierarchy before #160 | not in results.json (superseded by design) | CITED_ARCHIVE | Explicitly historical (pre-issue-#160). Not verified against any archive |
| 05:629 | 0.551 on D and D* | relational_table.multi_scale_hierarchy | 0.551 / 0.551 | SAME | |
| 05:632, 05:658, 05:789 | bands [6, 3], ARI 1.000 on both levels | run_nonmetric.py E4 | not in results.json or archive | NOT_RUN | Traced to notes/NONMETRIC_FINDINGS.md, not the harness |
| 05:721–724 | 1.0000/0.0000/1.0000; 0.8352/0.0864/0.7488; 0.9799/0.9799/0.0000; 1/1/0 | feature_space | same | SAME | |
| 05:731 | surrogate_l2_mean 0.2915 / 0.1041 / 0.0777 | feature_space | same | SAME | |
| 05:782, 05:820 | up to 1.000 ARI swing; rings 1.000 with zero variance over ten seeds | HDBSCAN baselines | not archived | NOT_RUN | gated-minimax-selection/notes/HDBSCAN_BASELINES.md (run_hdbscan_baselines.py, outputs gitignored) |
| 05:812 | 0.423 ± 0.387 | cosine_topics, gate | not archived | NOT_RUN | HDBSCAN_BASELINES.md:89, matching appendix:829. ddof of that driver not checked |
| 05:813 | 0.682 ± 0.183 | cosine_topics, tuned HDBSCAN* | not archived | NOT_RUN | same |
| 05:814–815 | [8, 4, 2] on 9/10 seeds; density 7/10 | seed study (Appendix A.8) | not archived | NOT_RUN | Different generator from Table 5.4's many_scale, which reads 10/10 at every n. See Open questions |
| 07:139 | 1.000 ± 0.000 | as 03:215 | — | NOT_RUN | as 03:215 |
| 07:140 | 0.001/0.0005/0.0002 | chance levels | 1/N | DERIVED | |
| 07:144 | 0.9996 ± 0.0012 | as 03:591 | — | UNTRACEABLE | as 03:591 |
| 07:239–242 | [8, 4, 2] at ARI 1.00 every seed and n; 5–7 of 10 | Table 5.4 many_scale / single_scale | same | SAME | |
| 07:243 | 0.73 ± 0.20 | as 05:564 | 0.73 ± 0.21 | DDOF | |
| 07:267, 07:512 | exactness 1.000 on three DTW sets | 3.7 last row | ECG5000 SAME; FordA and Crop absent | SAME / NOT_RUN | as 03:691 |
| 07:279–282 | 0.002/0.000, 0.147/0.175, 0.715/0.593 (+0.122), 0.664/0.482 (+0.182), 0.114/0.030 (+0.084) | G2 downstream, five sets | Archive named in the text (07:292): g2-downstream-2026-08-25, all verified. The ECG5000 row also reproduces in the new archive | CITED_ARCHIVE | Verified |
| 07:290 | Crop 0.064 → 0.114 | Crop set-cover before and after the transform fix | 0.064 in goal-8h-2026-08-11-fullsuite; 0.114 in g2-downstream-2026-08-25 | CITED_ARCHIVE | Verified |
| 07:296–297 | 1,597 s / 4.7 s | Crop build / reorder | absent | NOT_RUN | goal-8h verified |
| 07:341 | span 0.009; spreads ±0.018 to ±0.027 | G.5 first/second-order arms, 3 buckets | span 0.008 (1st) / 0.009 (2nd); spreads ±0.019 to ±0.030 | DDOF | Source uniform-2026-08-03 (ddof 0). The 2nd-order hybrid 0.028 → 0.030 is the same-pin host shift. Claim unaffected |
| 07:343 | span 0.828 | uniform minus hybrid, 0th order, 3 buckets | 0.399 − (−0.425) = **0.824** | DERIVED | |
| 07:343 | 0.394 ± 0.065 | as 01:24 | 0.399 ± 0.063 | MOVED_EXPECTED | |
| 07:344 | −0.434 ± 0.241 | G.5 3-bucket 0th, hybrid (shipped) | −0.425 ± 0.270 | MOVED_EXPECTED | Same pin on the Xeon: −0.434 ± 0.254 (= 0.241 × √(10/9)). Host/BLAS |
| 07:344 | ordering holds at 3/4/6 buckets; each separation clears both arms' bars | G.5 0th rows | 3: 0.399 > 0.242 > −0.425; 4: 0.412 > 0.314 > 0.033; 6: 0.543 > 0.437 > 0.319 | SAME (claim) | Holds if each separation is compared against each arm's bar on its own. The tightest is 6-bucket quantile − hybrid, 0.118 against the hybrid's ±0.107. Compared against the sum of both bars it failed in the old archive too, so that was never the reading |
| 07:345 | 0.152 / 0.676 | boundary (uniform − quantile) / pinning (quantile − hybrid) | **0.157 / 0.667** | DERIVED | |
| 07:357 | quantile deviations ±0.99, ±4.45, ±21.2 | G.5b quantile R² std at skew 10.44 / 14.71 / 17.71 | ±0.857 / ±4.505 / ±25.298 (table_g5b_skew_sweep.csv) | MOVED_EXPECTED | The prose quotes seeds10-2026-08-01 / full-2026-08-02. The means moved at uniform-2026-08-03 (a code change in that window), and ddof applied later. The qualitative claim stands |
| 07:353–354 | min bucket falls to 1 by skew 1.8, to 0 past skew 14 | G.5b uniform min bucket n | 1 at skew 1.84; **0 from skew 5.32 on** | MOVED_EXPECTED | Uniform starved at 5.32 starting with bumped-0764bc5-2026-08-22. **Claim wording affected**: "past skew 14" should be "past ≈5.3" |
| 07:448 | J +0.843 vs +0.841 | Table 4.11 BETH, one-class | absent | NOT_RUN | BETH is absent. Verified in cluster-2026-08-27/table_4_11_beth_anomaly.csv |
| 07:505 | R² = 0.6475 | WEC_Perth (#97) | — | UNTRACEABLE | The only hits are per-fit cells in table_4_1c_topn_runs.csv (coincidental). No generator for this claim |
| 08:53 | 0.99992 | Table 3.4 GPU, 48,000-point float32 cell, ordering | absent | NOT_RUN | GPU table deferred. Verified in gpu-table34-2026-08-02/table_3_4_gpu_speedups.csv and PROVENANCE_MAP:540 |
| DAT:42–43, DAT:277 | +0.366 vs +0.535 | Table 4.7b RT-IOT2022 open-set J | not in new archive (4.4 ran on Glass) | CITED_ARCHIVE | Text says "re-measured 2026-08-27 on the de-leaked loader". Verified in rtiot-deleaked-2026-08-27/table_4_4_openset.csv |
| DAT:44 | 3.64 ± 0.25 s | RT-IOT2022 MoG train | N/A | NOT_RUN | Verified in phiusiil-leakfree. Label says "Table 4.4"; it is Table 4.1/4.1b |
| DAT:44 | 92.7 ± 0.2% | RT-IOT2022 MoG accuracy (0.927 ± 0.002) | N/A | NOT_RUN | |
| DAT:44 | 99.8 ± 0.0% | RT-IOT2022 RF (0.998 ± 0.000) | N/A | NOT_RUN | |
| DAT:77 | 0.620 / 0.944 | Table 4.1 Bike Sharing, MoG R² / RF | 0.620 ± 0.014 / 0.944 ± 0.004 | SAME | |
| DAT:158–159 | 1.000, 20.9%, 0.122 | ECG5000 | same | SAME | |
| DAT:160 | 1.000, 0.4% | FordA | absent | NOT_RUN | goal-8h verified |
| DAT:163 | 1.000, 23.6%, 1,597 s, 4.7 s | Crop | absent | NOT_RUN | goal-8h verified |
| DAT:163–164 | NERFCM 0.029, set-cover 0.064 | Crop downstream | g2-downstream-2026-08-25: **0.030 ± 0.012 / 0.114**, gap 0.084 | CITED_ARCHIVE (stale) | The dated 2026-08-12 value is verified in goal-8h-2026-08-11-fullsuite, but it was superseded by the corrected-transform run Ch 7 cites. **Claim affected** |
| DAT:277 | MoG 92.7% / 4.2 s vs RF 99.8% | RT-IOT2022 classification/timing | N/A | NOT_RUN; 4.2 s UNTRACEABLE | No archive has 4.2 s (RT-IOT2022 MoG times on record: 2.82, 2.88, 3.64, 3.74, 33–37 s). It contradicts DAT:44's 3.64 s |
| DAT:278 | +0.843 vs +0.841 | BETH | absent | NOT_RUN | as 07:448 |
| DAT:281 | exactness 1.000 on 3 of 5; downstream "2 of 3 tested" | G2 | ECG5000 SAME, rest NOT_RUN | NOT_RUN (stale text) | g2-downstream-2026-08-25 measured all five (parity 2 of 5) |

Not scored (not table cells): 01:109 AUC 0.996 and 01:111 majority baseline 0.5755 come from the PhiUSIIL leak
investigation (PROVENANCE_MAP:898, URLSimilarityIndex 0.9961). 03:216–219 peak working sets (64.7–65.0 MB, 193.6 MB →
4.67 GB, 0.14–0.22×) come from check_matrix_free_reorder.py and are not archived. 05:306 1.4826 is a constant.

### Counts (rows above)

| Class | Rows |
|---|---|
| SAME | 41 (all of Table 5.1, 5.2 and 5.3 and every results.json-derived Ch 5 number; 3 of the rows also carry a NOT_RUN half) |
| DDOF | 8 |
| ROUND | 0 |
| MOVED_EXPECTED | 9 (Concrete same-pin host/BLAS ×6: 01:22, 01:24, 02:24, 02:25, 07:343, 07:344; PhiUSIIL ANFIS nondeterminism ×1; G.5b skew sweep ×2) |
| MOVED_UNEXPLAINED | **0** (no sklearn reference moved: CART 0.826 and RF 0.909 Concrete, CART 0.997 and RF 1.000 PhiUSIIL, Bike RF 0.944 are all unchanged) |
| HOST | 4 (01:107, 01:114, 03:401, 03:403) |
| NOT_RUN | 27 (RT-IOT2022, BETH, GPU 3.4, FordA/Crop DTW, optimizer study, check_matrix_free_reorder, HDBSCAN notes, run_nonmetric) |
| CITED_ARCHIVE | 6 (all verified except 05:627's historical 0.285; DAT:163–164 verified but stale) |
| UNTRACEABLE | 6 (0.9996 ± 0.0012 ×4, 07:505's 0.6475, DAT:277's 4.2 s) |
| DERIVED | 4 (chance levels ×2; 0.828 → 0.824; 0.152/0.676 → 0.157/0.667) |

Approximate: several rows hold multiple cells, and three rows are split SAME / NOT_RUN.

### Claims affected

1. **01:114 "194× the slowest fuzzy baseline (Table 4.1b)" is now 72×** on the run-of-record host. The baselines
   (ANFIS, GA-FIS) run 2× faster on the i9 while MoG did not speed up. The claim survives in kind but the quoted
   magnitude does not. Re-quote, or attribute it to phiusiil-leakfree-2026-08-30.
2. **03:401–405, the pVAT scaling reconciliation, breaks.** N = 4,096 stage two is now 0.145 s, so the published
   2.56 s stage-one figure is 17.7× slower, not about 11×. The three-arm s1/s2 range is now 5.4–11.6× (it was
   7.4–17.5×). The paragraph's argument, that the gap is bracketed by the three-arm decomposition, no longer holds on
   this host. Both endpoints moved because of host; it needs rewording or a ratio-only argument.
3. **03:702–707 is stale against its own Chapter 7.** It says Crop and FordA "land within 0.05 of each other" and
   "ElectricDevices and StarLightCurves were not attempted". On the corrected transform (g2-downstream-2026-08-25,
   which Ch 7 cites) Crop's gap is 0.084, which fails the band in the favourable direction. All five sets were
   measured. The parity passes are FordA and ElectricDevices (2 of 5), not Crop and FordA. DATASETS:163–166 and
   DAT:281 carry the same stale picture (Crop 0.029 / 0.064, "ED not yet attempted", "2 of 3 tested").
4. **07:353 starvation threshold.** "Falls to zero past skew 14" is now "zero from skew 5.32". Quantile's deviation
   numbers (±0.99 / ±4.45 / ±21.2) are pre-uniform-2026-08-03 values; they are now ±0.857 / ±4.505 / ±25.298. The
   qualitative comparison stands.
5. **07:343–345 decomposition numbers shift slightly:** span 0.828 → 0.824, boundary 0.152 → 0.157, pinning
   0.676 → 0.667. Ordering and error-bar claims hold.
6. Minor: in 01:112 ANFIS now ties the GA-FIS (0.998) instead of leading. 02:259's "+0.017" disagrees with its own
   archive and with Ch 6 ("+0.016").

### Open questions

- **Table 3.3's float32 on-demand cell (0.9996 ± 0.0012) is not what the table's generator produces (0.999 ± 0.002).**
  It comes from check_matrix_free_reorder.py, which archives nothing. That makes 03:591/604/621 and 07:144
  untraceable to an archive. Options: archive that script's output, or quote the generator's 0.999 ± 0.002 in the
  table (03:624 already does). The same applies to 03:215 and 07:139's 1.000 ± 0.000 at N = 1k/2k/5k.
- The run of record ran the 3.7 DTW generators on ECG5000 only. FordA/Crop exactness, violations and timings rest on
  goal-8h-2026-08-11-fullsuite, and downstream rests on g2-downstream-2026-08-25. Should Ch 3:691 and DATASETS cite
  those archives explicitly?
- **Concrete host/BLAS drift is larger than "3rd decimal" on the zeroth-order arm.** Hybrid moved −0.434 → −0.425,
  and its spread went from 0.254 to 0.270 at the same pin. Every Concrete MoG cell in Ch 1/2/7 depends on resolving
  the deferred BLAS note before re-quoting.
- 05:814–815 says "[8, 4, 2] on nine seeds of ten", while Table 5.4's many_scale gets [8, 4, 2] on 10/10 at every n.
  They are different generators (battery_hierarchical vs many_scale), but a reader will compare them. Which seed
  study produced 9/10? It is not in any archive I found.
- DATASETS:43–44 and :277 label RT-IOT2022 classification/timing as "Table 4.4"; it is Table 4.1/4.1b. DAT:277's
  "4.2 s" contradicts DAT:44's 3.64 s and matches no archive.
- The ddof of the HDBSCAN-baseline figures (05:812–813, 0.423 ± 0.387 / 0.682 ± 0.183) is unknown. Their driver is
  outside the harness and its outputs are gitignored.

---

## Ledger: ch4a — `prose/04-fast-fis-synthesis-mog.md` lines 1–560 vs `full-03d327a-2026-11-21`

New archive dir = `reproduce/outputs/full-03d327a-2026-11-21/`. Abbreviations: HN = `table_hyperparam_normalization.csv`,
G5 = `table_g5_output_partitioning.csv`, G5b = `table_g5b_skew_sweep.csv`, T49 = `table_4_9_correction_pass.csv`,
T41 = `table_4_1.csv`, T41b = `table_4_1b_baseline_timing.csv`, T61 = `table_6_1.csv`.

Where the prose came from: Tables 4.1/4.2/4.3 transcribe `uniform-2026-08-03` (identical to `goal-8h-2026-08-11-fullsuite`
for HN and G5; for G5b, goal-8h already differed, see below). Table 4.9 transcribes `mf-dedup-2026-08-05`. The captions name
`uniform-2026-08-03` as "the run of record". That is a stale run-of-record claim, not a deliberate citation of a different
archive, so these cells are classified against the new archive. Every one of them was checked and is present verbatim in
`uniform-2026-08-03`.

Attribution sources used for MOVED_EXPECTED:
- (H) READ FIRST note / PROVENANCE_MAP 2026-11-21: Concrete MoG differs in the 3rd decimal between hosts at the same pin (BLAS/summation order).
- (K) CHECKLIST B13/B14 residue: tribble-fis #95 replaced sklearn KMeans (k-means++) with a single-start `kmeans_1d`. Restoring it returns G5b byte-identical.
- (W) B14 / #95 era churn of tree/HME rows in HN, recorded in `outputs/FIX_IMPACT.md` and `WASSERSTEIN_REGRESSION.md` (18 of 84 HN cells).
- (218) PROVENANCE_MAP note 29: tribble-fis #218 `correlation_threshold=0.85` moved Table 4.9's expert arm.

### Pairs

| file:line | prose value | quantity (table/row/col) | new-archive value (file) | class | note |
|---|---|---|---|---|---|
| 04:47 | +0.031 ± 0.027 | T4.9 gated cascade, paired Δ vs base | +0.0077 ± 0.0109 (T49) | MOVED_EXPECTED | (218), note 29. The 95% CI now spans zero. **Claim affected** |
| 04:48 | +0.014 ± 0.061 | T4.9 flattened FIS, paired Δ | +0.0046 ± 0.0384 (T49) | MOVED_EXPECTED | (218). **Claim affected** |
| 04:80 | 0.695 ± 0.030 | T4.1 MoG 1st, raw | 0.695 ± 0.031 (HN) | DDOF | |
| 04:80 | 0.796 ± 0.018 | T4.1 MoG 1st, log+min-max | 0.795 ± 0.019 (HN) | MOVED_EXPECTED | (H). −0.001 |
| 04:80 | 0.713 ± 0.035 | T4.1 MoG 1st, log+z-score | 0.710 ± 0.033 (HN) | MOVED_EXPECTED | (H) / pins, −0.003 |
| 04:80 | +0.101 / −0.083 | Δ columns, 1st | +0.100 / −0.085 (HN) | DERIVED | |
| 04:81 | 0.804 ± 0.030 | MoG 2nd, raw | 0.804 ± 0.032 (HN) | DDOF | |
| 04:81 | 0.841 ± 0.021 | MoG 2nd, min-max | 0.840 ± 0.022 (HN) | MOVED_EXPECTED | (H) |
| 04:81 | 0.827 ± 0.028 | MoG 2nd, z-score | 0.825 ± 0.028 (HN) | MOVED_EXPECTED | (H) |
| 04:81 | +0.037 / −0.015 | Δ, 2nd | +0.036 / −0.015 (HN) | DERIVED | |
| 04:82 | 0.830 ± 0.025 | MoG full-2nd, raw | 0.830 ± 0.027 (HN) | DDOF | 0.025×1.054 = 0.026; 0.027 is a rounding edge |
| 04:82 | 0.861 ± 0.026 | MoG full-2nd, min-max | 0.859 ± 0.027 (HN) | MOVED_EXPECTED | (H) |
| 04:82 | 0.809 ± 0.115 | MoG full-2nd, z-score | 0.807 ± 0.120 (HN) | MOVED_EXPECTED | (H). Spread ≈ ddof (0.121) |
| 04:82 | +0.030 / −0.051 | Δ, full-2nd | +0.028 / −0.052 (HN) | DERIVED | |
| 04:83 | 0.712 ± 0.030 | fuzzy tree demo, raw | 0.713 ± 0.032 (HN) | MOVED_EXPECTED | Unchanged through 08-30. Moved only in this run (H/pins) |
| 04:83 | 0.740 ± 0.051 (×2) | fuzzy tree demo, min-max and z-score | 0.740 ± 0.054 (both) (HN) | DDOF | |
| 04:83 | +0.028 / −0.000 | Δ | +0.027 / +0.000 | DERIVED | |
| 04:84 | 0.583 ± 0.067 | fuzzy tree lib-default, raw | 0.586 ± 0.075 (HN) | MOVED_EXPECTED | Moved only in this run. Spread ×1.12, more than ddof (pins/host) |
| 04:84 | 0.689 ± 0.056 | fuzzy tree lib, min-max | 0.689 ± 0.059 (HN) | DDOF | |
| 04:84 | 0.691 ± 0.055 | fuzzy tree lib, z-score | 0.689 ± 0.059 (HN) | MOVED_EXPECTED | Now exactly equal to min-max |
| 04:84 | +0.106 / +0.002 | Δ | +0.103 / +0.000 | DERIVED | |
| 04:85 | 0.774 ± 0.025 | MoE demo, raw | 0.775 ± 0.024 (HN) | MOVED_EXPECTED | (W). Value stable since 08-27 |
| 04:85 | 0.834 ± 0.027 | MoE demo, min-max | 0.836 ± 0.028 (HN) | MOVED_EXPECTED | (W) |
| 04:85 | 0.806 ± 0.031 | MoE demo, z-score | **0.837 ± 0.028** (HN) | MOVED_EXPECTED | (W), +0.031. z-score now equals min-max. No single documented cause. See Open questions |
| 04:85 | +0.060 / −0.028 | Δ | +0.062 / **+0.000** | DERIVED | **Claim-relevant**: the MoE z-score penalty is gone |
| 04:86 | 0.648 ± 0.093 | MoE lib, raw | 0.631 ± 0.121 (HN) | MOVED_EXPECTED | (W) |
| 04:86 | 0.756 ± 0.059 | MoE lib, min-max | 0.758 ± 0.062 (HN) | MOVED_EXPECTED | (W) |
| 04:86 | 0.744 ± 0.066 | MoE lib, z-score | 0.761 ± 0.063 (HN) | MOVED_EXPECTED | (W). z-score now *above* min-max |
| 04:86 | +0.108 / −0.012 | Δ | +0.127 / **+0.003** | DERIVED | Sign flip |
| 04:87 | 0.825 ± 0.047 / 0.826 ± 0.047 / 0.826 ± 0.046 | CART raw / mm / z | 0.825 ± 0.049 / 0.826 ± 0.049 / 0.826 ± 0.049 (HN) | DDOF | sklearn control. Means unmoved, as they should be |
| 04:87 | +0.001 / −0.000 | CART Δ | +0.001 / −0.000 | SAME | |
| 04:88 | 0.909 ± 0.018 / 0.909 ± 0.019 / 0.909 ± 0.018 | RF raw / mm / z | 0.909 ± 0.019 / 0.909 ± 0.020 / 0.909 ± 0.019 (HN) | DDOF | sklearn control unmoved |
| 04:88 | +0.000 / −0.000 | RF Δ | same | SAME | |
| 04:92 | 0.695 → 0.796 | 1st raw → min-max | 0.695 → 0.795 | MOVED_EXPECTED | (H) |
| 04:93 | 0.106 and 0.108 | tree-lib / MoE-lib Δ min-max − raw | 0.103 and **0.127** | DERIVED | Gain claim holds (larger) |
| 04:96 | +0.001 and +0.000 | CART/RF Δ | same | SAME | |
| 04:101 | 0.002 | max \|mm − z\| over CART, RF, both tree rows | 0.000 | DERIVED | Claim holds, more strongly |
| 04:102 | ±0.018 to ±0.056 | those rows' spreads | ±0.019 to ±0.059 | DDOF | |
| 04:103 | 1.000000000000 | Spearman between scalers | not in archive | UNTRACEABLE | From the `norm-three-arm` investigation (`NORMALIZATION_THREE_ARM.md` / note 16), not a generator column |
| 04:108 | 0.713 ± 0.035 vs 0.796 ± 0.018 | 1st z vs mm | 0.710 ± 0.033 vs 0.795 ± 0.019 | MOVED_EXPECTED | (H) |
| 04:109 | 0.083 / "0.018 above raw" | Δ z − mm / Δ z − raw | 0.085 / 0.015 | DERIVED | "roughly two seed spreads": 0.085 is 2.6× the z arm's ±0.033 and 4.5× min-max's ±0.019. Still "about two to four" |
| 04:110 | 0.015 | 2nd Δ z − mm | 0.015 | SAME | Still inside ±0.022/±0.028 |
| 04:114 | 0.809, ±0.115 vs ±0.026 | full-2nd z mean/sd vs mm sd | 0.807, ±0.120 vs ±0.027 | MOVED_EXPECTED | Ratio still 4.4×. "Same mean to within 0.05": the gap is 0.052 in both old and new, so already slightly false |
| 04:115 | ±1.718 vs ±0.512 MPa | RMSE spread, full-2nd z vs mm | ±1.799 vs ±0.529 (HN) | MOVED_EXPECTED | Means moved too (6.905 → 6.960, 6.072 → 6.118) (H) |
| 04:163 | +0.031 ± 0.027 | T4.9 gated Δ | +0.0077 ± 0.0109 | MOVED_EXPECTED | (218). **Claim affected** |
| 04:164 | 81.4 → 109.0 raw MF | T4.9 base / cascade MF | 81.4 → 107.4 | MOVED_EXPECTED | (218). Base byte-identical |
| 04:166 | +0.014 ± 0.061 at 83.5 MF | T4.9 flattened | +0.0046 ± 0.0384 at 80.9 MF | MOVED_EXPECTED | (218). **Claim affected**: 80.9 is *below* base's 81.4 |
| 04:221 | 0.394 ± 0.065 | G5 3b 0th uniform R² | 0.399 ± 0.063 (G5) | MOVED_EXPECTED | Stable through 08-30, moved only in this run (H/pins) |
| 04:221 | 18.55 / 38.2 / 132 | tail / max err / min bucket | 18.50 / 38.0 / 132 | MOVED_EXPECTED | 2-dp cells. Min bucket SAME |
| 04:222 | 0.242 ± 0.070 | 0th quantile | 0.242 ± 0.079 | MOVED_EXPECTED | Mean same. Spread ×1.13, beyond ddof (old ddof-1 value 0.073) |
| 04:222 | 22.15 / 40.9 / 343 | | 22.19 / 40.8 / 343 | MOVED_EXPECTED | |
| 04:223 | −0.434 ± 0.241 | 0th quantile+pinned | −0.425 ± 0.270 | MOVED_EXPECTED | (H/pins) |
| 04:223 | 20.70 / 56.3 / 343 | | 20.71 / 56.3 / 343 | ROUND | |
| 04:224 | 0.796 ± 0.018 | 1st uniform | 0.795 ± 0.019 | MOVED_EXPECTED | (H) |
| 04:225 | 0.789 ± 0.026 | 1st quantile | 0.789 ± 0.027 | DDOF | |
| 04:226 | 0.787 ± 0.026 | 1st pinned | 0.787 ± 0.027 | DDOF | |
| 04:224-226 | 8.10/29.2, 8.08/28.5, 7.93/28.6 | tails/max err | 8.09/29.1, 8.11/28.5, 7.96/28.5 | MOVED_EXPECTED | Tiny (H) |
| 04:227 | 0.841 ± 0.021 | 2nd uniform | 0.840 ± 0.022 | MOVED_EXPECTED | (H) |
| 04:228 | 0.836 ± 0.025 | 2nd quantile | 0.835 ± 0.028 | MOVED_EXPECTED | (H) |
| 04:229 | 0.832 ± 0.027 | 2nd pinned | 0.831 ± 0.030 | MOVED_EXPECTED | (H) |
| 04:227-229 | 6.50/28.5, 6.59/28.4, 6.68/29.1 | | 6.47/28.5, 6.63/28.2, 6.62/28.9 | MOVED_EXPECTED | 2nd-order tail ordering now uniform < pinned < quantile |
| 04:231 | span 0.009 and 0.009; ±0.018 to ±0.027 | 1st/2nd arm spans | 0.008 and 0.009; ±0.019 to ±0.030 | DERIVED | Null result holds |
| 04:233 | 0.828, "twelve times the widest spread" | 0th span | 0.824. Widest 0th spread is now ±0.270 | DERIVED | "Twelve times" did not hold in the old data either (0.828/0.241 = 3.4; 12.7× only against uniform's ±0.065). New: 3.1× the widest, 13× uniform's. Wording issue |
| 04:234 | 0.394 / 0.416 / 0.541 | 0th uniform at 3/4/6 buckets | 0.399 / 0.412 / 0.543 | MOVED_EXPECTED | (H/pins) |
| 04:235 | −0.434 / 0.048 / 0.326 | 0th pinned at 3/4/6 | −0.425 / 0.033 / 0.319 | MOVED_EXPECTED | Separations still clear both bars (0.379 > 0.220; 0.224 > 0.158) |
| 04:236 | 56.3 vs 38.2 | max err | 56.3 vs 38.0 | MOVED_EXPECTED | |
| 04:241-242 | 344 pts / 0.195; 343 / 0.653 | bucket means in pinned diagnostic | not in archive | UNTRACEABLE | Diagnostic, not a table cell (A.10.11) |
| 04:244 | −0.38, −1.19 | interior constants | not in archive | UNTRACEABLE | Diagnostic |
| 04:245 | 0.005 vs 0.676 | pinning penalty at 1st+ vs 0th | 1st 0.002, 2nd 0.004 vs 0.667 | DERIVED | 0.005 is not reproducible from old cells either (old: 0.002 / 0.004) |
| 04:251 | 0.152 / 0.676 | boundary vs pinning decomposition | 0.157 / 0.667 | DERIVED | "More than four times" holds (4.25×) |
| 04:258 | 0.206, 0.788 | uniform end-bucket means | not in archive | UNTRACEABLE | Diagnostic |
| 04:262 | 18.55 vs 22.15 / 20.70 | 0th tails | 18.50 vs 22.19 / 20.71 | MOVED_EXPECTED | Also holds at 4 and 6 buckets (19.05 vs 21.45/20.15; 14.95 vs 17.70/16.45) |
| 04:266-268 | 132/75/39; 343/257/171 | min bucket | same | SAME | |
| 04:281 | 0.912 ± 0.009 | G5b skew +0.05, uniform | 0.912 ± 0.009 | SAME | |
| 04:281 | 0.911 ± 0.010 | quantile | 0.911 ± 0.011 | DDOF | |
| 04:281 | −0.001, 0.052, 0.051, 11 | | same | SAME | |
| 04:282 | 0.884 ± 0.016 / 0.876 ± 0.035 | +1.84 U / Q | 0.884 ± 0.017 / 0.876 ± 0.037 | DDOF | |
| 04:282 | −0.008, 0.066, 0.062, 1 | | same | SAME | |
| 04:283 | **0.731 ± 0.083** | +5.32 uniform | **0.708 ± 0.092** | MOVED_EXPECTED | (K) #95 kmeans_1d. Already 0.708 in goal-8h-2026-08-11 |
| 04:283 | 0.728 ± 0.169 | +5.32 quantile | 0.728 ± 0.178 | DDOF | |
| 04:283 | −0.003 | Q − U | **+0.020** | DERIVED | **Sign flip: quantile now ahead.** Claim affected |
| 04:283 | 0.092 / 0.066 / 1 | U tail / Q tail / U min bucket | 0.084 / 0.066 / **0** | MOVED_EXPECTED | (K) |
| 04:284 | 0.297 ± 0.126 | +10.44 uniform | 0.337 ± 0.128 | MOVED_EXPECTED | (K) |
| 04:284 | 0.183 ± 0.813 | quantile | 0.183 ± 0.857 | DDOF | |
| 04:284 | −0.114, 0.118, 0.064, 1 | | −0.154, 0.097, 0.064, **0** | MOVED_EXPECTED / DERIVED | (K) |
| 04:285 | 0.168 ± 0.073 | +14.71 uniform | 0.184 ± 0.100 | MOVED_EXPECTED | (K) |
| 04:285 | −2.106 ± 4.274 | quantile | −2.106 ± 4.505 | DDOF | |
| 04:285 | −2.274, 0.111, 0.062, 0 | | −2.290, 0.078, 0.062, 0 | MOVED_EXPECTED / DERIVED | (K) |
| 04:286 | **0.084 ± 0.016** | +17.71 uniform | **0.025 ± 0.225** | MOVED_EXPECTED | (K). Spread ×14. **Claim affected** |
| 04:286 | −12.562 ± 24.000 | quantile | −12.562 ± 25.298 | DDOF | |
| 04:286 | −12.646, 0.132, 0.060, 0 | | −12.587, **0.066**, 0.060, 0 | MOVED_EXPECTED / DERIVED | (K) |
| 04:288 | −0.008, −0.003, −0.114, −2.274, −12.646 | Q − U series | −0.008, **+0.020**, −0.154, −2.290, −12.587 | DERIVED | **Claim affected** |
| 04:293 | ±0.169 / ±0.813 / ±4.274 / ±24.000 | quantile sds | ±0.178 / ±0.857 / ±4.505 / ±25.298 | DDOF | |
| 04:294 | −12.6 ± 24.0 | | −12.6 ± 25.3 | DDOF | |
| 04:302 | 0.912 → 0.084 | uniform R² end to end | 0.912 → 0.025 | MOVED_EXPECTED | (K) |
| 04:307-308 | U tail 0.052 → 0.132; Q 0.051 → 0.060 | | U 0.052 → 0.066 (peak 0.097 at skew 10); Q 0.051 → 0.060 | MOVED_EXPECTED | (K). **Claim affected**. The prose's "quantile's actually *falls*, 0.051 to 0.060" is a rise. That was already wrong in the old data |
| 04:310 | ±4.3 and ±24.0 | | ±4.5 and ±25.3 | DDOF | |
| 04:335 | −12.6, ±24 | Fig 4.2 caption | −12.6, ±25.3 | DDOF | |
| 04:440 | 0.0012 / 0.0006 | Table 4.11(e) BETH θ vs threshold | not in archive | NOT_RUN | BETH absent (Kaggle). PROVENANCE_MAP :728 has these values |
| 04:506 | 0.13 ± 0.02 s | PhiUSIIL MoG train time | 0.17 ± 0.01 s (T41) | HOST | phiusiil-leakfree-2026-08-30 has 0.13 ± 0.02 |
| 04:507 | 0.440 ± 0.181 | PhiUSIIL MoG acc | 0.440 ± 0.181 (T41) | SAME | |
| 04:510 | AUC 0.996 | URLSimilarityIndex single-feature AUC | not in archive | UNTRACEABLE | PROVENANCE_MAP note 30/31 table (:898, 0.9961). Investigation, not a generator |
| 04:511 | 0.5755 | majority baseline | not in archive | UNTRACEABLE | PROVENANCE_MAP note 31(a) (:999), seed-0 measurement. Full-data majority 134,850/235,795 = 0.5719. Presumably a test-split figure |
| 04:513 | 0.999 ± 0.001 | PhiUSIIL ANFIS | 0.998 ± 0.002 (T41) | MOVED_EXPECTED | ANFIS is nondeterministic run to run (documented: 0.998–0.999). Not a defect |
| 04:513 | 0.998 ± 0.001 | GA-FIS | 0.998 ± 0.001 (T41) | SAME | |
| 04:513 | 0.997 ± 0.001 | CART | 0.997 ± 0.001 (T61 PhiUSIIL) | SAME | |
| 04:513 | 1.000 ± 0.000 | RF | 1.000 ± 0.000 (T41) | SAME | |
| 04:517 | ±0.181 | | ±0.181 | SAME | |
| 04:520 | 194× | PhiUSIIL MoG speedup vs slowest fuzzy (T4.1b) | **72×** (T41b) | HOST | 08-30 archive 194×; Xeon bump 132×. Still the largest ratio in the table (Concrete 44×, Bike 4×) |
| 04:524 | 3.64 ± 0.25 s, 92.7 ± 0.2%, RF 99.8 ± 0.0% | RT-IOT2022 row | N/A | NOT_RUN | Dataset absent. Verified present in phiusiil-leakfree-2026-08-30 (T41b 3.64 ± 0.25 s; T41 acc 0.927 ± 0.002, RF 0.998 ± 0.000) |
| 04:525-526 | "seven points", "3.6 seconds" | derived from above | — | NOT_RUN | |
| 04:548 | 0.394, 0.796, 0.841 | order ladder 0/1/2 (G5 3b uniform) | 0.399, 0.795, 0.840 | MOVED_EXPECTED | (H/pins) |
| 04:548 | 0.861 ± 0.026 | full-2nd (HN min-max) | 0.859 ± 0.027 | MOVED_EXPECTED | (H). Note T41's own full-2nd row is 0.848 ± 0.034 (different code path, note 14). The "(Table 4.1, full 2nd order)" pointer is ambiguous |
| 04:550 | 0.045 and 0.065 | 2nd / full-2nd minus 1st | 0.045 and 0.064 | DERIVED | Claim holds |

Lines 177–180 (Table 4.8 percentages, not ± pairs, listed for completeness): "@1× 0.0% (Wine, BC) to 13.0% (Diabetes)" is SAME.
"Max-lossless 2× (Diabetes) to 10× (Wine, Concrete)" is now **3× to 50×**. "0.0% (BC) to 44.2% (Digits)" is now **2.4% (Wine) to 66.5% (BC)**.
Both are MOVED_EXPECTED per note 29 (#143 ddof/t-CI for Digits, #218 for BC/Wine; Diabetes unattributed). The BC
parenthetical ("almost no redundancy left") inverts.

### Counts (rows above, ± pairs and table cells, excluding the 4.8 percentages)

SAME 17 · DDOF 22 · ROUND 1 · MOVED_EXPECTED 51 · MOVED_UNEXPLAINED 0 · HOST 2 · NOT_RUN 3 · CITED_ARCHIVE 0 ·
UNTRACEABLE 6 · DERIVED 19 (approximate. Some rows bundle several cells.)

### Claims affected

1. **§4.2 :47–48 and §4.3.1 :163–170 (Table 4.9), the biggest change.** The gain is +0.0077 ± 0.0109 (95% CI spans zero), not +0.031 ± 0.027.
   The flattened FIS keeps +0.0046 ± 0.0384, also spanning zero. The flattened FIS has 80.9 MF, *fewer* than the base's 81.4 (the prose says 83.5).
   "The pass helps", "the mechanism claim holds", "under half the gain survives" (it is now ~60% of a non-significant gain) and
   "gating logic … is doing real work, not decoration" are all unsupported now. Documented (note 29, #218), but the prose was never updated.
2. **Table 4.3 / :288–308 (G5b), uniform arm (#95 kmeans_1d).**
   - "Quantile's mean R² trails uniform's in every row" is false. At skew +5.32 quantile leads by +0.020.
   - "Uniform's [spread] stays bounded" fails at skew 17.7, where it is ±0.225 (prose ±0.016).
   - "Min bucket: one by skew 1.8, zero past skew 14" is wrong. It is now zero from skew 5.32.
   - "Uniform R² falls 0.912 → 0.084" is now → 0.025.
   - "Uniform's tail error grows from 0.052 to 0.132 … consistently, with small spreads" is wrong. It rises to 0.097 at skew 10, then falls back to 0.066, with spreads up to ±0.064. The direction (quantile ≤ uniform in tails) holds in every row, but the gap at skew 17.7 is only 0.006.
   - This drift is already present in `goal-8h-2026-08-11-fullsuite`, the archive the prose claims to reproduce against.
3. **Table 4.1 MoE / tree z-score column.** The MoE z-score penalty disappears: demo −0.028 → +0.000, lib −0.012 → +0.003.
   The fuzzy-tree z-score now equals min-max exactly. "Min-max best-or-tied in 8 of 9 rows" (note 16) and the MoE row's contribution to
   "centred normalization costs …" need rechecking. The headline (flat MoG 1st order: −0.085, full-2nd variance blow-up ±0.120 vs ±0.027) holds.
   The CART/RF control is intact (means unmoved, ddof only).
4. **:233 "twelve times the widest spread"** does not hold in either archive. The widest 0th-order spread is ±0.241 (now ±0.270), which gives ≈3×. **:245 "0.005"** is not reproducible: old and new data give 0.002 / 0.004. **:308 "quantile's actually *falls*, 0.051 to 0.060"** is a rise. These three are pre-existing prose errors, not drift.
5. **:114 "same mean to within 0.05"**: the gap is 0.052 in both archives. Marginal.
6. **:520 "194×"** is 72× on the new host. It is still the strongest ratio in the table, so the qualitative claim survives, but the number is host-specific.
7. **:177–180 Table 4.8 ranges** (note 29): the max-lossless range and the BC parenthetical invert.

### Open questions

- Table 4.1 (HN) MoE and fuzzy-tree rows: *why* does z-score now give exactly the min-max result for the tree (and nearly for HME)? The change appeared between `goal-8h-2026-08-11` and the 08-22 archives. It survives `all-restored-2026-08-22` (all six #95 functions restored), so it is **not** #95. `FIX_IMPACT.md` records the churn without a cause. Plausibly a tribble-fis tree/HME change (e.g. #86/#183) making those models affine-invariant. I have classified it MOVED_EXPECTED (pin bump) but it lacks a stated attribution. Worth one line in PROVENANCE_MAP.
- The fuzzy-tree lib-default raw cell (0.583 → 0.586, spread 0.071 → 0.075) and the G5 table (e.g. 0th uniform 0.394 → 0.399, pinned −0.434 → −0.425) were byte-stable through `phiusiil-leakfree-2026-08-30` and moved only in this run. The same-pin cross-host comparison (`bump-03d327a`) did not run these generators, so host-BLAS (READ FIRST) and the intervening pin bumps are not separated for these tables.
- 0.5755 majority baseline vs the full-data 0.5719: confirm it is a test-split figure.
- The Table 4.1/4.2/4.3 captions still name `uniform-2026-08-03` as "the run of record".

---

## Ledger: ch4b — `prose/04-fast-fis-synthesis-mog.md` lines 561–1160

New run of record: `reproduce/outputs/full-03d327a-2026-11-21/` (written NEW below). Other archives are named in full.
Where the prose names its own source archive, the primary class is CITED_ARCHIVE and the note gives what NEW says.

### ± pairs

| file:line | prose value | quantity (table/row/col) | new-archive value (file) | class | note |
|---|---|---|---|---|---|
| 04:561 | 0.862 ± 0.033 | concrete_reconciliation, flat MoG-TSK 2nd, refined, R² | 0.875 ± 0.029 (NEW table_concrete_reconciliation.csv) | CITED_ARCHIVE | Checked: in `uniform-2026-08-03`. NEW moved +0.013 (pin bumps). The "configuration, not disagreement" claim still holds |
| 04:564 | 0.687 ± 0.049 | Table 6.2 / table_6_1, Concrete, flat, R² | 0.687 ± 0.051 (NEW table_6_1.csv) | DDOF | |
| 04:572 | 0.13 ± 0.02 s | T4.4 PhiUSIIL, MoG train time | 0.17 ± 0.01 s (NEW table_4_1.csv) | HOST | |
| 04:572 | 0.440 ± 0.181 | T4.4 PhiUSIIL, MoG acc | 0.440 ± 0.181 (NEW table_4_1.csv) | SAME | |
| 04:573 | 3.64 ± 0.25 s | T4.4 RT-IOT2022, MoG time | N/A | NOT_RUN | RT-IOT2022 absent. Checked: in `phiusiil-leakfree-2026-08-30/table_4_1.csv` |
| 04:573 | 0.927 ± 0.002 | T4.4 RT-IOT2022, MoG acc | N/A | NOT_RUN | same; checked in leakfree |
| 04:573 | 0.998 ± 0.000 | T4.4 RT-IOT2022, RF ref | N/A | NOT_RUN | same; checked in leakfree |
| 04:574 | 0.787 ± 0.026 | T4.4 Concrete, TSK order 1 | 0.795 ± 0.019 (NEW hyperparam, 1st, log+min-max) / 0.787 ± 0.027 (NEW g5, 3-bucket hybrid, 1st) | MOVED_EXPECTED | Old hybrid-partition number from `full-2026-08-03`. The same chapter (l.547) quotes uniform 0.796, so the row disagrees with the chapter. Under the uniform default (G5, note 19) the value is 0.795 ± 0.019. Read as hybrid it is DDOF |
| 04:575 | 0.832 ± 0.027 | T4.4 Concrete, TSK order 2 | 0.840 ± 0.022 (hyperparam) / 0.831 ± 0.030 (g5 hybrid) | MOVED_EXPECTED | same as l.574 |
| 04:576 | 0.861 ± 0.026 | T4.4 Concrete, full 2nd (hyperparam, full-2nd, log+min-max) | 0.859 ± 0.027 (NEW table_hyperparam_normalization.csv) | MOVED_EXPECTED | leakfree had 0.861 ± 0.028 (ddof). Mean moved 0.002 after the pin bumps; also the Concrete BLAS open note |
| 04:595 | 0.795 ± 0.025 | T4.5 row 1, Concrete R², "from one file `phiusiil-leakfree-2026-08-30/table_4_1.csv`" | 0.797 ± 0.028 | CITED_ARCHIVE — **FAILS** | **Not in the named archive**, which has 0.799 ± 0.027. The value is `uniform-2026-08-03/table_4_1.csv` |
| 04:595 | 0.43 ± 0.02 s | same, Concrete time | 0.28 ± 0.05 s | CITED_ARCHIVE — **FAILS** | From uniform-2026-08-03. Leakfree has 0.53 ± 0.05 |
| 04:596 | 0.997 ± 0.001 | same, PhiUSIIL acc | 0.440 ± 0.181 | CITED_ARCHIVE — **FAILS (LOUD)** | **The retracted leaky PhiUSIIL accuracy** (uniform-2026-08-03). Leakfree and NEW both give 0.440 ± 0.181. The sentence contradicts the table directly below it |
| 04:596 | 0.28 ± 0.02 s | same, PhiUSIIL time | 0.17 ± 0.01 s | CITED_ARCHIVE — **FAILS** | From uniform-2026-08-03. Leakfree has 0.13 ± 0.02 |
| 04:600 | 0.43 ± 0.02 s | Concrete clock after warm-up discard | 0.28 ± 0.05 s | HOST | Same uniform-2026-08-03 value as l.595. "2% spread" is really 4.7% (0.02/0.43). NEW's ±0.05 is an 18% spread, so the "warm-up explains the spread" paragraph should be re-checked |
| 04:612 | 0.799 ± 0.027 | T4.5 MoG 1st, Concrete R² | 0.797 ± 0.028 | CITED_ARCHIVE | Checked in leakfree. NEW would be MOVED_EXPECTED: the deferred Concrete BLAS/summation note in PROVENANCE.txt; bump-03d327a has 0.799 |
| 04:612 | 0.53 ± 0.05 s | T4.5 MoG 1st, Concrete time | 0.28 ± 0.05 s | CITED_ARCHIVE | Checked. NEW: HOST |
| 04:612 | 0.440 ± 0.181 | T4.5 MoG 1st, PhiUSIIL acc | 0.440 ± 0.181 | CITED_ARCHIVE | Checked. NEW: SAME |
| 04:612 | 0.13 ± 0.02 s | T4.5 MoG 1st, PhiUSIIL time | 0.17 ± 0.01 s | CITED_ARCHIVE | Checked. NEW: HOST |
| 04:613 | 0.852 ± 0.030 | T4.5 MoG full 2nd (estimator), Concrete R² | 0.848 ± 0.034 (NEW table_4_1.csv) | MOVED_EXPECTED | Deferred Concrete BLAS note (same pin on the Xeon gives 0.852) |
| 04:613 | 0.53 ± 0.04 s | T4.5 full 2nd, time | 0.30 ± 0.02 s | HOST | |
| 04:614 | 0.799 ± 0.112 | ANFIS, Concrete R² | 0.799 ± 0.112 | SAME | |
| 04:614 | 7.28 ± 1.12 s | ANFIS, Concrete time | 2.06 ± 0.04 s (NEW table_4_1b) | HOST | 3.5× faster |
| 04:614 | 0.999 ± 0.001 | ANFIS, PhiUSIIL acc | 0.998 ± 0.002 | MOVED_EXPECTED | ANFIS is nondeterministic from run to run (documented: 0.998–0.999 across runs). Its "beats MoG" ordering still holds |
| 04:614 | 22.53 ± 2.30 s | ANFIS, PhiUSIIL time | 11.62 ± 0.44 s | HOST | |
| 04:615 | 0.896 ± 0.038 | GA-FIS, Concrete R² | 0.896 ± 0.038 | SAME | |
| 04:615 | 41.11 ± 3.81 s | GA-FIS, Concrete time | 12.32 ± 0.34 s | HOST | |
| 04:615 | 0.998 ± 0.001 | GA-FIS, PhiUSIIL acc | 0.998 ± 0.001 | SAME | |
| 04:615 | 25.29 ± 1.55 s | GA-FIS, PhiUSIIL time | 11.89 ± 0.16 s | HOST | |
| 04:617 | 0.825 ± 0.049 | CART, Concrete R² (raw) | 0.825 ± 0.049 (NEW hyperparam / reconciliation, raw) | SAME | log+min-max 0.826 ± 0.049 |
| 04:617 | 0.997 ± 0.001 | CART, PhiUSIIL acc | 0.997 ± 0.001 (NEW table_6_1.csv) | SAME | |
| 04:618 | 0.909 ± 0.020 | RF, Concrete R² | 0.909 ± 0.020 (NEW table_4_1.csv) | SAME | |
| 04:618 | 0.63 ± 0.02 s | RF, Concrete time | 0.30 ± 0.00 s | HOST | |
| 04:618 | 1.000 ± 0.000 | RF, PhiUSIIL acc | 1.000 ± 0.000 | SAME | |
| 04:618 | 1.31 ± 0.23 s | RF, PhiUSIIL time | 1.00 ± 0.01 s | HOST | |
| 04:625 | 0.896 ± 0.038 | GA-FIS, Concrete (text) | 0.896 ± 0.038 | SAME | Compared there with full 2nd 0.852 (NEW 0.848). Claim holds |
| 04:626 | 0.620 ± 0.014 | MoG, Bike R² | 0.620 ± 0.014 | SAME | |
| 04:626 | 0.577 ± 0.075 | ANFIS, Bike R² | 0.561 ± 0.086 (NEW table_4_1); 0.556 ± 0.084 (NEW 4.1c) | MOVED_EXPECTED | ANFIS nondeterminism (map 2026-11-21 records 0.533–0.539 across runs). MoG still wins the row |
| 04:626 | 0.545 ± 0.047 | GA-FIS, Bike R² | 0.545 ± 0.047 | SAME | |
| 04:635 | 0.53 ± 0.05 / 7.28 ± 1.12 / 41.11 ± 3.81 / 0.63 ± 0.02 | T4.1b Concrete: MoG / ANFIS / GA / RF time | 0.28 ± 0.05 / 2.06 ± 0.04 / 12.32 ± 0.34 / 0.30 ± 0.00 (NEW table_4_1b) | CITED_ARCHIVE ×4 | Fig 4.5 text says T4.1b is quoted from phiusiil-leakfree. All checked there. NEW: HOST |
| 04:636 | 0.53 ± 0.04 / 7.79 ± 0.53 / 43.89 ± 0.91 / 0.61 ± 0.02 | T4.1b Concrete full 2nd | 0.30 ± 0.02 / 2.07 ± 0.03 / 12.42 ± 0.26 / 0.29 ± 0.00 | CITED_ARCHIVE ×4 | checked; NEW HOST |
| 04:637 | 0.59 ± 0.07 / 4.80 ± 0.68 / 8.30 ± 1.72 / 7.98 ± 1.41 | T4.1b Bike | 0.65 ± 0.01 / 1.28 ± 0.05 / 2.49 ± 0.08 / 4.26 ± 0.01 | CITED_ARCHIVE ×4 | checked; NEW HOST |
| 04:638 | 0.13 ± 0.02 / 22.53 ± 2.30 / 25.29 ± 1.55 / 1.31 ± 0.23 | T4.1b PhiUSIIL | 0.17 ± 0.01 / 11.62 ± 0.44 / 11.89 ± 0.16 / 1.00 ± 0.01 | CITED_ARCHIVE ×4 | checked; NEW HOST |
| 04:639 | 3.64 ± 0.25 / 264.53 ± 25.30 / 51.58 ± 1.25 / 6.67 ± 0.39 | T4.1b RT-IOT2022 | N/A | NOT_RUN ×4 | dataset absent; checked in leakfree |
| 04:658 | 0.861 ± 0.026 | hyperparam full-2nd (text) | 0.859 ± 0.027 | MOVED_EXPECTED | as l.576 |
| 04:661 | 0.852 ± 0.030 | estimator full-2nd R² (text) | 0.848 ± 0.034 | MOVED_EXPECTED | as l.613 |
| 04:661 | 0.44 ± 0.04 s | estimator full-2nd time (text) | 0.30 ± 0.02 s | HOST | **Comes from uniform-2026-08-03**, while T4.5 row 2 (leakfree) says 0.53 ± 0.04. The paragraph mixes two archives |
| 04:726 | 0.491 ± 0.359 | T4.7 (Glass) complement, det | 0.491 ± 0.362 (NEW table_4_4_openset.csv) | CITED_ARCHIVE | Checked: in uniform-2026-08-03. NEW: DDOF (√(60/59)) |
| 04:726 | 0.362 ± 0.262 | complement, FA | 0.362 ± 0.264 | CITED_ARCHIVE | checked; NEW DDOF |
| 04:727 | 0.493 ± 0.341 | iForest, det | 0.493 ± 0.343 | CITED_ARCHIVE | checked; NEW DDOF |
| 04:727 | 0.317 ± 0.160 | iForest, FA | 0.317 ± 0.162 | CITED_ARCHIVE | checked; NEW DDOF |
| 04:728 | 0.399 ± 0.300 | OC-SVM, det | 0.399 ± 0.303 | CITED_ARCHIVE | checked; NEW DDOF |
| 04:728 | 0.288 ± 0.166 | OC-SVM, FA | 0.288 ± 0.167 | CITED_ARCHIVE | checked; NEW DDOF |
| 04:752–754 | 0.804 ± 0.270, 0.438 ± 0.085, 0.845 ± 0.225, 0.435 ± 0.061, 0.966 ± 0.145, 0.431 ± 0.063 | T4.7b RT-IOT2022 open-set | NEW 4.4 ran on Glass | CITED_ARCHIVE ×6 | All checked in `rtiot-deleaked-2026-08-27/table_4_4_openset.csv`. Not runnable in NEW (dataset absent) |
| 04:806 | 0.152 ± 0.012 | T4.11 OC-SVM FA | — | NOT_RUN | BETH absent. The caption names only the CSV. Checked in `cluster-2026-08-27/table_4_11_beth_anomaly.csv` (already post-#143) |
| 04:807 | 0.001 ± 0.003 / 0.021 ± 0.005 / 0.898 ± 0.005 | T4.11 iForest det / FA / AUC | — | NOT_RUN ×3 | checked in cluster-2026-08-27 |
| 04:904–909 | 24 pairs (0.9992 ± 0.0001 … 0.7026 ± 0.0121) | T4.12 PhiUSIIL one-class | outside the harness | CITED_ARCHIVE ×24 | All 24 checked in `experiments/phishing-oneclass/RESULTS.md`. **run.py:165 uses `np.std` (ddof 0) at n=5**, which breaks the repo's #143 convention (×√(5/4)=1.118, e.g. 0.0180→0.0201). Prose says pin `ef15d6a`; current pin is 03d327a, so nothing re-ran |
| 04:973 | +0.0015 ± 0.0046 | T4.8 Glass, @1× Δ | +0.0015 ± 0.0049 (NEW table_4_8) | CITED_ARCHIVE | Checked in mf-dedup-2026-08-05. NEW: DDOF |
| 04:973 | +0.0000 ± 0.0069 | Glass, @max-lossless (5×) Δ | +0.0000 ± 0.0073 | CITED_ARCHIVE | NEW: DDOF |
| 04:974 | +0.0000 ± 0.0000 | Wine @1× | same | CITED_ARCHIVE | NEW: SAME |
| 04:974 | −0.0037 ± 0.0161 | Wine @max-lossless 10× | −0.0074 ± 0.0234 at **7×** (16.2 MF, 2.4%) | CITED_ARCHIVE | NEW: MOVED_EXPECTED (#218 correlation_threshold, note 29) |
| 04:975 | +0.0000 ± 0.0000 | Breast Cancer @1× | same (but raw MF 11.1→16.4) | CITED_ARCHIVE | NEW: SAME delta; raw count moved (#218) |
| 04:975 | +0.0000 ± 0.0000 | Breast Cancer @max-lossless 5× | −0.0550 ± 0.0788 at **50×** (5.5 MF, 66.5%) | CITED_ARCHIVE | NEW: MOVED_EXPECTED (#218 plus #143 CI) |
| 04:976 | +0.0009 ± 0.0028 | Digits @1× | +0.0009 ± 0.0029 | CITED_ARCHIVE | NEW: DDOF |
| 04:976 | −0.0117 ± 0.0211 | Digits @max-lossless 7× | −0.0189 ± 0.0285 at **10×** (75.0 MF, 56.4%) | CITED_ARCHIVE | NEW: MOVED_EXPECTED (#143 t-CI moved the boundary; note 29 cause A) |
| 04:977 | +0.0000 ± 0.0000 | Concrete @1× | same | CITED_ARCHIVE | NEW: SAME |
| 04:977 | −0.0118 ± 0.0305 | Concrete @max-lossless 10× | −0.0122 ± 0.0332 | CITED_ARCHIVE | **NEW: MOVED_UNEXPLAINED (small).** Spread ×1.089 is not pure ddof, and the mean moved 4e-4. First seen in all-restored-2026-08-22 (−0.0122 ± 0.0315). Regression path, the same pattern as Diabetes. Note 29 only says "Concrete moves in the fifth decimal" |
| 04:978 | +0.0059 ± 0.0167 | Diabetes @1× | +0.0057 ± 0.0178 | CITED_ARCHIVE | **NEW: MOVED_UNEXPLAINED** (note 29: Diabetes regression drift, "cause not identified") |
| 04:978 | +0.0011 ± 0.0168 | Diabetes @max-lossless 2× | −0.0057 ± 0.0244 at **3×** (26.8 MF, 35.6%) | CITED_ARCHIVE | **NEW: MOVED_UNEXPLAINED** (note 29) |
| 04:1005 | 81.4 ± 2.3 | T4.9 base, MF count | 81.4 ± 2.5 | CITED_ARCHIVE | NEW: DDOF (within rounding) |
| 04:1005 | 0.5323 ± 0.0408 | base, acc | 0.5323 ± 0.0430 | CITED_ARCHIVE | NEW: DDOF |
| 04:1006 | 109.0 ± 9.5 | gated cascade, MF | 107.4 ± 9.9 | CITED_ARCHIVE | NEW: MOVED_EXPECTED (#218; note 29 localises it to the expert arm) |
| 04:1006 | 0.5631 ± 0.0568 | gated cascade, acc | 0.5400 ± 0.0413 | CITED_ARCHIVE | NEW: MOVED_EXPECTED (#218) |
| 04:1006 | +0.0308 ± 0.0266 | gated cascade, paired Δ | +0.0077 ± 0.0109 | CITED_ARCHIVE | NEW: MOVED_EXPECTED (#218). **95% CI now spans zero** |
| 04:1007 | 83.5 ± 5.4 | flat union, MF | 80.9 ± 7.2 | CITED_ARCHIVE | NEW: MOVED_EXPECTED (#218) |
| 04:1007 | 0.5462 ± 0.0511 | flat union, acc | 0.5369 ± 0.0380 | CITED_ARCHIVE | NEW: MOVED_EXPECTED (#218) |
| 04:1007 | +0.0138 ± 0.0609 | flat union, paired Δ | +0.0046 ± 0.0384 | CITED_ARCHIVE | NEW: MOVED_EXPECTED (#218) |

### Bare table cells and derived numbers

| file:line | prose value | quantity | new-archive value | class | note |
|---|---|---|---|---|---|
| 04:635–638 | 78× / 83× / 14× / 194× | T4.1b speedup (slowest fuzzy / MoG) | **44× / 42× / 4× / 72×** (NEW table_4_1b) | DERIVED | The prose cells are checked in leakfree. On the new host the ratios shrink 1.8–3.5×, because the baselines sped up more than MoG |
| 04:642 | "14× to 194×", "one to two orders of magnitude" | range | **4× to 72×** | DERIVED | **Claim affected**, see below |
| 04:650 | "Concrete's ratio is 78×" | | 44× | DERIVED | |
| 04:661 | "a 0.009 gap" (0.861 vs 0.852) | | 0.859 − 0.848 = **0.011** | DERIVED | Still inside both spreads |
| 04:663 | "+0.06 R² over first order for one hundredth of a second" | | 0.848 − 0.797 = **+0.051**; 0.30 − 0.28 = 0.02 s | DERIVED | |
| 04:680–688 | T4.6 θ-sweep, 21 cells (0.838/0.719/+0.119 … 0.000) | Glass operating curve | identical in NEW table_4_4b_theta_sweep.csv | CITED_ARCHIVE ×21 | Checked in uniform-2026-08-03. NEW: **SAME** all 21. PROVENANCE_MAP's T4.6/4.7 "stale under #72" status looks out of date: NEW reproduces them exactly |
| 04:695–712 | +0.154 @0.80, floor +0.119, "five-sixths", 70% / 55% | text from T4.6 | same | SAME | 0.129 / 0.154 = 0.84 |
| 04:726–728 | J +0.129 / +0.176 / +0.111 | T4.7 J column | same | SAME | |
| 04:730 | iForest leads by 0.047 | derived | 0.047 | DERIVED | unchanged |
| 04:752–754 | J +0.366 / +0.410 / +0.535 | T4.7b | — | CITED_ARCHIVE ×3 | checked in rtiot-deleaked |
| 04:756, 787–795 | 0.169, 0.030, 0.139; "peak +0.391, gap 0.143" | derived / earlier sweep | — | CITED_ARCHIVE | 0.169, 0.030 and 0.139 recompute exactly. +0.391 and the iForest +0.534 (gap 0.143) are in verify-fis1435811-2026-08-23 |
| 04:775–783 | T4.7c, 21 cells | RT-IOT2022 θ-sweep | — | CITED_ARCHIVE ×21 | All checked in rtiot-deleaked-2026-08-27/table_4_4b_theta_sweep.csv. NOT_RUN in NEW |
| 04:803–807 | T4.11 bare cells (0.993, 0.150, +0.843, 0.990, 0.928, +0.841, 0.996, −0.020) | BETH | — | NOT_RUN | All checked in cluster-2026-08-27 |
| 04:818–867 | BETH text: 1.000 / 0.427, 0.062 AUC, 3,997 / 1,508, +0.864, +0.879, +0.914, +0.870 / +0.843 / +0.755, 0.9905, 0.069, 0.933→0.993, 0.019→0.137, 0.081 s vs ~0.15 s, 0.9903, 11× (1.280 / 0.117), ~0.08 AUC, +0.160 → +0.769 | 4.11 (b–e) | — | NOT_RUN | All match cluster-2026-08-27 except **"0.1500" (l.853), where the CSV has 0.1502** (ROUND-level misquote; "15×" is fine) |
| 04:964–978 | T4.8 counts / percentages / max-lossless × | | NEW: BC raw 11.1→16.4, @1× 16.4; Wine 10×→7×, 12.0%→2.4%; BC 5×→50×, 0.0%→66.5%; Digits 7×→10×, 44.2%→56.4%; Diabetes 2×→3×, 24.5%→35.6% (31.4→26.8 MF) | CITED_ARCHIVE | All checked in mf-dedup-2026-08-05. Glass and Concrete counts unchanged in NEW |
| 04:983–985 | "2×–10× and 0.0%–44.2%" | derived from T4.8 | **3×–50× and 2.4%–66.5%** | DERIVED | Claim affected (map note 29 already lists this) |
| 04:990–993 | Fig 4.7: "3× on Diabetes, 15× on Wine and Concrete" | | T4.8 (cited) says 2× / 10× / 10×; NEW says 3× / 7× / 10× | UNTRACEABLE (as stated) | The figure caption matches neither the table it sits under nor NEW. The figure is drawn from an unnamed sweep CSV. "First break" ≠ "max-lossless", which may explain it, but the caption calls the two the same thing |
| 04:1099–1112 | T4.10 RMSE 15.58 / 17.00 / 15.44 / 12.07, NASA 320 | CMAPSS pooled | outside the harness | CITED_ARCHIVE | Checked in FuzzySystemsExperiments/cmapss_all_datasets_report.md. 7.23 / 7.22 / 8.34 / ~6.5 not checked (DS02 script / paper) |

### Claims affected

1. **l.594–596: the Table 4.5 row-1 sentence is misattributed and states a retracted number.** It says every row-1 cell comes from `phiusiil-leakfree-2026-08-30/table_4_1.csv`, then quotes Concrete 0.795 ± 0.025 @ 0.43 s and **PhiUSIIL acc 0.997 ± 0.001** @ 0.28 s. Those are `uniform-2026-08-03` values, and 0.997 is the leaky PhiUSIIL accuracy the leak-free rerun retracted to 0.440 ± 0.181. The table directly below says 0.799 / 0.53 / 0.440 / 0.13. The "warm-up" paragraph (l.598–602) quotes the same uniform-archive 0.43 ± 0.02.
2. **l.642 and Fig 4.5: the speedup range "14× to 194×, one to two orders of magnitude".** NEW gives **4×–72×**. Bike Sharing at 4× is under one order of magnitude, and PhiUSIIL falls from 194× to 72×. The generator's own footer says "ratios are not machine-dependent", and this run contradicts that: the baselines (ANFIS 3.5×, GA-FIS 3.3× faster) gained far more on the i9 than MoG did. The headline ratio is host-sensitive.
3. **l.646–647 and Fig 4.5 text (l.652–653): "the random forest is not slower than the construction on Concrete or Bike Sharing".** This is false for Bike Sharing even in the cited archive (RF 7.98 s vs MoG 0.59 s), and in NEW (RF 4.26 s vs 0.65 s). On Concrete in NEW, RF and MoG tie (0.30 vs 0.28 s; full 2nd 0.29 vs 0.30).
4. **l.574–575: Table 4.4's Concrete order-1 and order-2 rows (0.787 / 0.832)** are old hybrid-partition values. They disagree with the chapter's own l.547 (0.796 / 0.841) and with NEW (0.795 / 0.840). The order-1 → full-2nd ladder shifts slightly but keeps its shape.
5. **l.661: the full-2nd estimator time 0.44 ± 0.04 s** comes from uniform-2026-08-03, while Table 4.5 row 2 (leakfree) says 0.53 ± 0.04. NEW gives 0.30 ± 0.02. Gap 0.011, not 0.009; still "inside both spreads".
6. **Table 4.9 (cited archive, l.1006):** in NEW the gated-cascade gain is **+0.0077 ± 0.0109, and its CI spans zero**, down from +0.0308 ± 0.0266. Fig 4.8's closing sentence ("the correction pass's accuracy contribution… is closed: it was measured") would now report no measurable gain. The cause is tribble-fis #218 (note 29).
7. **Table 4.8 (cited archive):** the max-lossless column moves for 4 of 6 problems, and the summary "2×–10×, 0.0%–44.2%" becomes 3×–50×, 2.4%–66.5%. The claims "the shipped 1× never costs anything" and "a lossless tolerance strictly past 1× exists for all six" still hold in NEW. The Breast Cancer 66.5% sits at Δ −0.055 ± 0.079, which is lossless only in the CI sense. The Diabetes and Concrete regression drift is unattributed.
8. **Table 4.10 (l.1105–1106, l.1093):** `raw_memory` is bolded and labelled "best per-sample" at 15.58, but `whole_cycle` has the lower per-sample RMSE (15.44) in the same table and in the source report. Lower is better. "The raw-memory model leads only on the per-sample density metric" is contradicted by its own table.
9. ANFIS cells (PhiUSIIL 0.999→0.998, Bike 0.577→0.561) moved because ANFIS is nondeterministic (documented). The ordering claims (ANFIS/GA beat MoG on PhiUSIIL; MoG wins Bike) hold.

### Open questions

- Several prose statements contradict the chapter's own tables. They are not numeric drift, but readers will trip on them:
  - l.663–668 says ANFIS/GA cells are `N/A (C1)` and the adapters are absent, yet Table 4.5 fills them.
  - l.1062–1064 says "I have not run the head-to-head against ANFIS and a GA-tuned FIS".
  - l.739–741 and Fig 4.6 (l.1027–1030) say the RT-IOT2022 θ-sweep is future work / "not given the chance", yet Table 4.7c exists.
  - l.902–903 says tribble-fis is "at the pinned `ef15d6a`".
- T4.12 spreads are ddof 0 at n=5 (`experiments/phishing-oneclass/run.py:165`). Should they be re-run or rescaled ×1.118 to match #143?
- PROVENANCE_MAP marks T4.6/4.7 "stale under #72", but NEW reproduces uniform-2026-08-03 exactly (same means; only ddof moves the spreads). Is "stale" still right?
- Concrete max-lossless Δ (−0.0118 → −0.0122, spread ×1.089) is a regression-path move in the same window as the unattributed Diabetes drift. Should note 29 cover it too?
- l.732: "SDs … seven times the smallest gap". The smallest gap (complement vs OC-SVM, 0.018) gives ≈17×, not 7×. Unclear which gap is meant.
- The NEW Concrete MoG clock is 0.28 ± 0.05 s (an 18% spread). Did the warm-up discard (note 14) work on this run?

---

## Ledger: Chapter 6 + Appendix vs `full-03d327a-2026-11-21`

Archive shorthand: **N** = `reproduce/outputs/full-03d327a-2026-11-21/`, **U** = `uniform-2026-08-03`,
**LF** = `phiusiil-leakfree-2026-08-30`, **F03** = `full-2026-08-03`. CR = `table_concrete_reconciliation.csv`,
HN = `table_hyperparam_normalization.csv`, T61 = `table_6_1.csv`, G5 = `table_g5_output_partitioning.csv`.
"BLAS note" = the documented, deferred same-pin cross-host Concrete divergence in N's PROVENANCE.txt READ FIRST
(Concrete MoG R² differs in the 3rd decimal between Xeon and i9 at the same pin).

Ch6 Table 6.1 and the refinement table **explicitly name `uniform-2026-08-03` as their run of record** (06:266, 06:327),
so they are classed CITED_ARCHIVE (verified there); the new-archive value is given in the note anyway, because those
cells are what a re-quote would print.

### Chapter 6 — `research/proposal-defense/prose/06-hierarchical-refined-fis.md`

| file:line | prose value | quantity | new-archive value | class | note |
|---|---|---|---|---|---|
| 06:106 | R²=0.6475 | WEC_Perth, Issue #97 tuned Tribble | not in harness | CITED_ARCHIVE | Verified in `ISSUE_97_WEC_PERTH_EVALUATION.md` (l.15,128). Single split, not ten seeds |
| 06:108 | R²=0.80 | WEC_Perth RF, same split | not in harness | CITED_ARCHIVE | Verified, same memo l.147 |
| 06:232 | +0.016 to +0.025, 8–9/10 seeds | opt study, hot arms Δ R² vs heuristic | not in N (not in runner) | CITED_ARCHIVE | Verified in `opt-hotcold-kmbic-2026-08-03/table_opt_hotstart.md`: hot arms +0.016…+0.025, wins 8/10–9/10 |
| 06:237 | 3 ± 4 | cold DE, evals to reach heuristic | — | CITED_ARCHIVE | Verified, same archive |
| 06:237 | 10 ± 18 | cold GA/PSO/ACO | — | CITED_ARCHIVE | Verified |
| 06:238 | 188 ± 440 | cold Powell | — | CITED_ARCHIVE | Verified |
| 06:239 | 398 ± 594 | cold L-BFGS-B / GD | — | CITED_ARCHIVE | Verified |
| 06:266 | 0.043 | note 12 cross-host move | — | CITED_ARCHIVE | PROVENANCE_MAP note 12 (map l.1284/1319) |
| 06:294 | +0.005, +0.005, +0.018 | split-first Δ, closed-form 1st/2nd/0th | — | CITED_ARCHIVE | Verified, `uniform-2026-08-03/SPLIT_FIRST_LEAK.md` l.59–61 (these are pinned-hybrid-era F03 vs splitfirst) |
| 06:296 | −0.032; 8 controls 0.000 | split-first, 1st refined; controls | — | CITED_ARCHIVE | Verified, same file l.63–72 |
| 06:309 | 0.997 ± 0.001 | PhiUSIIL flat MoG accuracy (Table 6.2) | **0.440 ± 0.181** (N T61) | MOVED_EXPECTED | #215 leak-free (LF). **Claim affected**: contradicts Table 6.2 on the same page (0.440) |
| 06:309 | 0.970 | PhiUSIIL fuzzy tree | **0.958 ± 0.003** (N T61) | MOVED_EXPECTED | #215 leak-free. Same sentence, same problem |
| 06:317 | 0.394 ± 0.065 | refinement table, 0th closed-form R² | 0.399 ± 0.063 (N CR) | CITED_ARCHIVE | U verified. New mean +0.005: BLAS note |
| 06:317 | 0.720 ± 0.037 | 0th refined | **0.747 ± 0.060** (N CR) | CITED_ARCHIVE | U verified. New value moved +0.027: refine.py changed in the pin window (tribble-fis #134/#136/#178/#209); closed-form arms did not move in that window (U/goal-8h 0.394 → pins-synced 0.394) |
| 06:317 | +0.326 | Δ 0th | +0.348 | DERIVED | from N CR |
| 06:318 | 0.796 ± 0.018 | 1st closed-form | 0.795 ± 0.019 | CITED_ARCHIVE | U verified. New: DDOF + BLAS rounding |
| 06:318 | 0.834 ± 0.045 | 1st refined | 0.832 ± 0.041 | CITED_ARCHIVE | U verified. New: small move, refine path |
| 06:318 | +0.038 | Δ 1st | +0.037 | DERIVED | |
| 06:319 | 0.841 ± 0.021 | 2nd closed-form | 0.840 ± 0.022 | CITED_ARCHIVE | U verified. New: BLAS note + DDOF |
| 06:319 | 0.862 ± 0.033 | 2nd refined | **0.875 ± 0.029** | CITED_ARCHIVE | U verified. New mean +0.013 (refine.py pin window; was already 0.878 ± 0.024 at pins-synced-2026-08-27) |
| 06:319 | +0.021 | Δ 2nd | **+0.035** | DERIVED | **Claim affected** (see below) |
| 06:322 | "a factor of nine" | 0.326/0.038 | 0.348/0.037 = 9.4; 0.348/0.035 = 9.9 | DERIVED | Still ≈ an order of magnitude; holds |
| 06:336 | 0.862 ± 0.033 | refined 2nd (text) | 0.875 ± 0.029 | CITED_ARCHIVE | **Claim affected**: "the same number to within a thousandth" as full-2nd |
| 06:337 | 0.861 ± 0.026 | full 2nd, log+min-max (HN) | 0.859 ± 0.027 (N HN) | CITED_ARCHIVE | U verified. New: −0.002 BLAS note (bump Xeon vs i9 0.852/0.848 class of move) |
| 06:345 | 0.027 | "full-second-order arm losing 0.027", outside the uniform sweep | — | UNTRACEABLE | No archive named. Not found in the CR/HN files I scanned |
| 06:356 | 0.841 ± 0.021 / 6.50 ± 0.43 | T6.1 flat 2nd | 0.840 ± 0.022 / 6.53 ± 0.47 (N CR) | CITED_ARCHIVE | U verified. New: BLAS note |
| 06:357 | 0.861 ± 0.026 / 6.07 ± 0.51 | T6.1 full 2nd | 0.859 ± 0.027 / 6.12 ± 0.53 (N HN) | CITED_ARCHIVE | U verified. New: BLAS note |
| 06:358 | 0.862 ± 0.033 / 6.00 ± 0.52 | T6.1 2nd + refinement | **0.875 ± 0.029 / 5.76 ± 0.55** | CITED_ARCHIVE | U verified. New: refine path moved (MOVED_EXPECTED) |
| 06:359 | 0.689 ± 0.056 / 9.07 ± 0.57 | T6.1 fuzzy tree default (log+min-max) | 0.689 ± 0.059 / 9.07 ± 0.61 | CITED_ARCHIVE | U verified. New = DDOF |
| 06:359 | 0.740 ± 0.051 / 8.29 ± 0.63 | T6.1 fuzzy tree demo-tuned (HN log+min-max) | 0.740 ± 0.054 / 8.29 ± 0.66 | CITED_ARCHIVE | U verified. New = DDOF |
| 06:360 | 0.826 ± 0.047 / 6.73 ± 0.74 | T6.1 CART | 0.826 ± 0.049 / 6.73 ± 0.78 | CITED_ARCHIVE | U verified. New = DDOF. sklearn mean unchanged ✓ |
| 06:361 | 0.909 ± 0.019 / 4.90 ± 0.31 | T6.1 RF | 0.909 ± 0.020 / 4.90 ± 0.33 | CITED_ARCHIVE | U verified. New = DDOF ✓ |
| 06:378 | 0.583 ± 0.071 | T6.2 fuzzy tree Concrete R² | **0.586 ± 0.075** (N T61) | MOVED_EXPECTED | Prose = LF (ddof 1). Same pin, other host: Xeon bump-03d327a gives 0.580 ± 0.071, i9 gives 0.586: BLAS note (Concrete-only, deferred). Not a pin effect |
| 06:378 | 10.53 | T6.2 tree RMSE | 10.48 (10.483) | MOVED_EXPECTED | same |
| 06:378 | 0.958 ± 0.003 | T6.2 tree PhiUSIIL | 0.958 ± 0.003 | SAME | |
| 06:379 | 0.825 ± 0.049 / 6.74 / 0.997 ± 0.001 | T6.2 CART | same | SAME | ×3 |
| 06:380 | 0.909 ± 0.019 / 4.90 / 1.000 ± 0.000 | T6.2 RF | same | SAME | ×3 |
| 06:383 | 0.687 ± 0.051 / 9.12 / 0.440 ± 0.181 | T6.2 flat | 0.687 ± 0.051 / 9.117 / 0.440 ± 0.181 | SAME | ×3 |
| 06:386 | AUC 0.996 | URLSimilarityIndex | — | CITED_ARCHIVE | PROVENANCE_MAP note 31 (0.9961), verified |
| 06:387 | 0.440 ± 0.181 | flat MoG PhiUSIIL | 0.440 ± 0.181 | SAME | |
| 06:387 | 0.5755 | PhiUSIIL majority baseline | — | CITED_ARCHIVE | Map l.999, verified |
| 06:388 | 0.997, 1.000 | CART, RF PhiUSIIL | same | SAME | ×2 |
| 06:392 | 0.999 ± 0.001 | Table 4.5 ANFIS PhiUSIIL | 0.998 ± 0.002 (N table_4_1) | MOVED_EXPECTED | ANFIS is documented as nondeterministic run to run, range 0.998–0.999 (map, 2026-11-21 bump sections). Not a claim change. check_prose falsely matched this to table_3_2 |
| 06:392 | 0.998 ± 0.001 | Table 4.5 GA-FIS PhiUSIIL | 0.998 ± 0.001 | SAME | |
| 06:393 | 0.997, 1.000 | CART, RF | same | SAME | ×2 |
| 06:394 | "0.56 behind" | 0.997 − 0.440 | 0.557 | DERIVED | holds |
| 06:397 | ±0.181 | flat MoG PhiUSIIL spread | ±0.181 | SAME | |

### Appendix — `research/proposal-defense/prose/appendix.md`

| file:line | prose value | quantity | new-archive value | class | note |
|---|---|---|---|---|---|
| A:45 | R²=0.6475, RMSE 58,688, 3.93 s, RF 0.80 | WEC_Perth Issue #97 | not in harness | CITED_ARCHIVE | Verified, `ISSUE_97_WEC_PERTH_EVALUATION.md`. One split, one run |
| A:156–160 | 15 scores (Wasserstein / Bhattacharyya / Composite top-5) | Table A.1 | identical (N table_a1_feature_ranking.csv) | SAME | ×15. N also has a 4th `higher_order` column the prose does not show |
| A:164 | AUC 0.996 | URLSimilarityIndex | — | CITED_ARCHIVE | note 31 |
| A:180–188 | 27 accuracies | Table A.2 | identical (N table_a2_feature_count.csv) | SAME | ×27 |
| A:180,188 | 0.26/0.09/0.10 s; 0.71/0.76/0.77 s | A.2 fit times | 0.06/0.05/0.06; 0.87/0.96/0.95 | HOST | ×6. Prose times = LF |
| A:190–200 | 0.7924, 0.5733, 0.4384, 0.6372, 0.9827 | re-quotes of A.2 | same | SAME | ×5 |
| A:191,195,243 | 0.9967 / 0.4267 | leak-era one-feature row | — | CITED_ARCHIVE | Verified in `main-d0efefc` (and U, F03): wasserstein 0.9967, bhattacharyya 0.4267. See open question 3 |
| A:244–245 | gap 0.57; 0.22 | 0.9967−0.4267; 0.7924−0.5733 | 0.570; 0.219 | DERIVED | holds |
| A:206 | 0.043 | note 12 | — | CITED_ARCHIVE | note 12 / NOTE12_THREADING.md |
| A:241 | "Reaching 0.96 takes fifteen features under every scorer" | A.2 | composite 0.9761 at 10 features | DERIVED | **Claim affected (already false in the cited table)**: composite clears 0.96 at 10 features, in LF and N alike |
| A:254 | 0.4267, 0.4527 | leak-era bhattacharyya 1 and 2 features | — | CITED_ARCHIVE | Verified `main-d0efefc` |
| A:443 | 1.000 ± 0.000; chance 0.001/0.0005/0.0002; 64.7–65.0 MB; 193.6 MB→4.67 GB; 0.14–0.22× | G4d matrix-free reorder | not in N | NOT_RUN | `reproduce/experiments/check_matrix_free_reorder.py` is not in `run_all_tables.sh`. N Table 3.2 is consistent: f64 matrix-free 1.000 (exact) |
| A:452 | 0.9996 ± 0.0012 | float32 matrix-free ordering | N Table 3.2: 0.999 ± 0.002 | NOT_RUN | Different generator. Consistent at 3 dp |
| A:466 | 0.004 | max hybrid−quantile R² gap, 1st/2nd | 0.004 (3-bucket 2nd: 0.835 vs 0.831) | DERIVED | holds |
| A:466 | ±0.018 to ±0.027 | seed spreads, 1st/2nd | ±0.019 to ±0.030 | DDOF | |
| A:473 | span 0.828 | uniform − pinned, 0th | 0.824 | DERIVED | |
| A:473 | 0.394 ± 0.065 | G5 3-bucket 0th uniform | 0.399 ± 0.063 (N G5) | MOVED_EXPECTED | BLAS note (+0.005) |
| A:474 | 0.242 ± 0.070 | G5 0th quantile | 0.242 ± 0.079 | MOVED_EXPECTED | Mean SAME. Spread is ×1.13, more than ddof's ×1.054 (0.074 expected): host/BLAS |
| A:474 | −0.434 ± 0.241 | G5 0th pinned (hybrid) | −0.425 ± 0.270 | MOVED_EXPECTED | BLAS note. Ordering holds at 3/4/6 buckets in N ✓ |
| A:475 | [0.0, 0.4038, 1.0]→[0.0, 0.411, 1.0]; 344 pts; mean 0.195; −0.38, −1.19 | solved coefficients | not archived | UNTRACEABLE | A manual coefficient reading (CHECKLIST §F H5); no generator emits it |
| A:493 | 0.014 ± 0.195; 0.666; 7.5→16.1 MPa | pinned-arm z-score collapse | not produced (partition default now uniform) | CITED_ARCHIVE | **Archive not named in prose.** Verified F03 HN: 0.014 ± 0.195, raw 0.666, RMSE 7.537→16.138. Prose marks them withdrawn |
| A:498–501 | 0.001; −0.407/−0.524/−0.634; MSE 0.030 vs 0.009 | l2 sweep / pinned n_gaussians | — | NOT_RUN | `reproduce/experiments/diagnose_bounded_normalization.py`, not in runner (its outputs were removed from N) |
| A:505 | 0.713 ± 0.035 | HN flat 1st, log+z-score (uniform cuts) | 0.710 ± 0.033 (N HN) | MOVED_EXPECTED | −0.003: BLAS note |
| A:506 | −0.651 → +0.018 | Δ z-score − raw, pinned → uniform | +0.015 | MOVED_EXPECTED | −0.651 is F03 (verified); +0.018 is U, now +0.015 |
| A:511 | 0.083 | bounding worth at 1st (|Δ z − mm|) | 0.085 | DERIVED | |
| A:511 | 0.015 | same at 2nd | 0.015 | SAME | |
| A:512 | ±0.115 vs ±0.026 | full-2nd z-score vs min-max spread | ±0.120 vs ±0.027 | DDOF | (0.121 / 0.027 expected). Claim holds |
| A:519,530 | 0.043; 0.000000; 2.4×; 1.6% | note 12 threading/BLAS | — | CITED_ARCHIVE | NOTE12_THREADING.md |
| A:567 | 0.914 / 0.072 / 0.037, "a factor of twenty-five" | refinement gain by order | **0.348 / 0.037 / 0.035, ≈10×** (N CR) | MOVED_EXPECTED | Stale numbers. They match `main-d0efefc` exactly, a pinned-hybrid-era archive superseded by the uniform-partition switch. **Claim affected**: contradicts Ch6's own "+0.326/+0.038/+0.021, a factor of nine" |
| A:616 | RF 0.809 ± 0.008; flat MoG 0.631 ± 0.020 | A.7 California Housing | not in N | NOT_RUN | table_a7 is not in the runner. FYI, the non-citable Xeon `bump-03d327a`: RF 0.809 ± 0.008, **flat 0.664 ± 0.016** |
| A:617 | RF 0.923 ± 0.004; flat −0.261 ± 1.431; HME −0.766 ± 2.411 | A.7 Superconductivity | not in N | NOT_RUN | bump-03d327a: 0.923 ± 0.005, −0.229 ± 1.391, −1.737 ± 5.730. "Unstable" still holds |
| A:626,628 | same four values repeated | A.7 prose | — | NOT_RUN | ×4 |
| A:651 | 0.440 ± 0.181 | Table 4.5 PhiUSIIL MoG acc | 0.440 ± 0.181 | SAME | |
| A:651 | 0.13 ± 0.02 s | PhiUSIIL MoG train time | 0.17 ± 0.01 s | HOST | |
| A:651 | 0.997 ± 0.001 | leak-era value ("it read") | — | CITED_ARCHIVE | historic, e.g. U |
| A:652 | +0.366 vs +0.535 | RT-IOT2022 open-set J | — | NOT_RUN | RT-IOT2022 absent. Verified in `rtiot-deleaked-2026-08-27/table_4_4_openset.csv` |
| A:652 | 3.64 ± 0.25 s; 0.927 ± 0.002; 0.998 ± 0.000 | RT-IOT2022 Table 4.1 row | N/A in N | NOT_RUN | Verified in LF table_4_1.csv |
| A:668 | +0.366 vs +0.535 | duplicate | — | NOT_RUN | |
| A:717 | exactness 1.000; 20.9%; 0.715 vs 0.593 (0.122) | ECG5000 G2 | 1.000; 20.9%; 0.715 vs 0.593 ± 0.047; gap 0.122 | SAME | ×4 |
| A:718 | 1.000; 0.4% | FordA | not in N | NOT_RUN | N's 3.7 tables ran only ECG5000 |
| A:719 | 1.000; 23.6%; 1,597 s + 4.7 s; 0.029 / 0.064 | Crop | not in N | NOT_RUN | |
| A:798 | 0.061 / 0.863 | HDBSCAN eom on rings, mcs 3/10 | — | NOT_RUN | `run_hdbscan_baselines.py` (manifest `table-5-5-7`) is not in `run_all_tables.sh` |
| A:809–813 | 20 cells (Table A.3) | gate vs HDBSCAN battery | — | NOT_RUN | ×20 |
| A:815–817 | +0.018; 0.887 / 0.820 / +0.067 | A.3 derived | — | NOT_RUN | |
| A:828–834 | 14 ± pairs + 7 deltas (Table A.4) | per-dataset | — | NOT_RUN | |
| A:837–849 | 0.873 ± 0.175; −0.448; 0.423 ± 0.387; 0.803; 0.713–0.817; 0.014–0.037; 0.802 | A.4 text | — | NOT_RUN | |
| A:858–878 | Table A.5 + eps-sweep table | nested recovery | — | NOT_RUN | default-seed per-level ARIs |
| A:893–894 | 0.253→0.011; 0.903→0.458; 0.863→1.000 | m_pts=5 effect | — | NOT_RUN | |
| A:917 | 0.99992 | GPU Borůvka float32 | — | NOT_RUN | GPU, descoped record (A.9) |
| A:1167 | 0.9996 ± 0.0012 | Table 3.3 ordering | N Table 3.2 0.999 ± 0.002 | NOT_RUN | as A:452 |
| A:1284 | 0.0012 | Table 4.11(e) θ-sweep advantage | — | NOT_RUN | BETH absent. table_4_11* is not in N |
| A:1317–1318 | 0.195, 0.653 → bias 0.038, 0.120; 687 rows | outer-bucket means | — | DERIVED | Internally consistent (0.195² = 0.038, 0.347² = 0.120). Bucket means not archived |
| A:1319 | R² −0.434 vs 0.394 | G5 0th pinned vs uniform | −0.425 vs 0.399 | MOVED_EXPECTED | BLAS note |
| A:1327 | 0.676 → 0.005 | pinned penalty, 0th → higher order | 0.667 (quantile − hybrid, 0th); 0.002 (1st) / 0.004 (2nd) | DERIVED | 0.676 = 0.242 − (−0.434) in U. The exact source of 0.005 could not be pinned down |
| A:1343 | +0.001, +0.000 | CART/RF Δ min-max − raw (HN) | +0.001, +0.000 | SAME | ×2 |
| A:1601 | 0.861 ± 0.026 | A.6 flat full-2nd, Concrete | 0.859 ± 0.027 (N HN) | MOVED_EXPECTED | BLAS note |
| A:1601 | 0.440 ± 0.181 | A.6 flat PhiUSIIL | 0.440 ± 0.181 | SAME | |
| A:1602 | 0.756 ± 0.059 | A.6 HME library default, log+min-max | 0.758 ± 0.062 | MOVED_EXPECTED | +0.002, plus DDOF. U value. HME is tribblefis code, so pins/BLAS apply |
| A:1602 | 0.834 ± 0.027 | A.6 HME demo-tuned | 0.836 ± 0.028 | MOVED_EXPECTED | +0.002, plus DDOF |
| A:1602 | 1.000 ± 0.001 | A.6 HME PhiUSIIL | **0.914 ± 0.039** (N T61, = LF) | MOVED_EXPECTED | #215 leak-free. **Claim affected**: "reaches parity with CART and RF" (0.997/1.000) is false. The table also mixes a leak-free flat cell with a leak-era HME cell |
| A:1603 | 0.689 ± 0.056; 0.740 ± 0.051; 0.958 ± 0.003 | A.6 fuzzy tree | 0.689 ± 0.059; 0.740 ± 0.054; 0.958 ± 0.003 | DDOF | DDOF ×2, SAME ×1 |
| A:1604 | 0.826 ± 0.047; 0.997 ± 0.001 | A.6 CART | 0.826 ± 0.049; 0.997 ± 0.001 | DDOF | DDOF ×1, SAME ×1 |
| A:1605 | 0.909 ± 0.019; 1.000 ± 0.000 | A.6 RF | 0.909 ± 0.020; 1.000 ± 0.000 | DDOF | DDOF ×1, SAME ×1 |
| A:1607 | 0.834 ± 0.027 vs 0.841 ± 0.021 | HME tuned vs flat 2nd | 0.836 ± 0.028 vs 0.840 ± 0.022 | MOVED_EXPECTED | small. "Indistinguishable within spread" holds |
| A:1608 | sd 0.060→0.022; mean 0.760→0.829 | HME default→tuned | 0.062→0.028; 0.758→0.836 | CITED_ARCHIVE | **Archive unnamed.** Verified in F03 HN (log+"standardized" = min-max, pinned era). **Claim affected**: "nearly a factor of three" is ≈2.2 in N and in the prose's own Table A.6 (0.059→0.027) |
| A:1617 | 0.813 ± 0.039; 10,536 MPa | five-seed HME, seed-9 divergence | — | CITED_ARCHIVE | PROVENANCE_MAP note 7 |

### Approximate counts (rows with a ×n multiplier counted n times; grouped NOT_RUN blocks counted as cells)
- SAME: Ch6 17, Appendix ~60 (A.1 15, A.2 27 + 5 re-quotes, ECG 4, others)
- DDOF: Ch6 0 (the Table 6.1 DDOF cells are filed under CITED_ARCHIVE), Appendix 8
- ROUND: 0
- MOVED_EXPECTED: Ch6 5, Appendix 17
- MOVED_UNEXPLAINED: **0**
- HOST: Appendix 7
- NOT_RUN: Appendix ~95 (A.8 HDBSCAN ≈ 65 cells, A.7 California/Superconductivity/RT-IOT/FordA/Crop, matrix-free, BETH, GPU)
- CITED_ARCHIVE: Ch6 ~33 (incl. 20 Table 6.1/refinement cells pinned to U), Appendix ~16
- UNTRACEABLE: Ch6 1, Appendix 1
- DERIVED: Ch6 6, Appendix 9

### Claims affected

1. **06:309**: "flat model reaches 0.997 ± 0.001 and the tree 0.970" on PhiUSIIL. These are leak-era values. The new run (and Table 6.2 two paragraphs later) says 0.440 ± 0.181 and 0.958 ± 0.003. The chapter contradicts itself.
2. **06:336–338**: "refined 2nd 0.862 and full-2nd 0.861 … the same number to within a thousandth … two routes to the same ceiling." New: refined 2nd **0.875 ± 0.029** vs full-2nd **0.859 ± 0.027**, a gap of 0.016. They still overlap within spread, but "within a thousandth" fails, and refined 2nd becomes the single best fuzzy number.
3. **06:325 / Fig 6.5 caption (06:331)**: "the higher-order gains are within a spread" and "at second order the two arms sit inside each other's spread." New 2nd-order gain is **+0.035** against spreads of 0.022 and 0.029, so it exceeds one sd. 1st order (+0.037 vs 0.041) still holds. The "factor of nine" is now ≈10 and survives. "Shrinks toward zero" is weaker: 1st and 2nd gains are now equal (0.037 vs 0.035).
4. **A:567**: "refinement worth 0.914 at zeroth order, 0.072 at first and 0.037 at second, a factor of twenty-five." These are stale pinned-hybrid values (`main-d0efefc`). They contradict Ch6's +0.326/+0.038/+0.021 (factor 9). The new run gives +0.348/+0.037/+0.035 (≈10×).
5. **A:1602 + A:1610**: Table A.6 HME PhiUSIIL **1.000 ± 0.001** is leak-era. The new run (and LF) gives **0.914 ± 0.039**. "On PhiUSIIL the mixture reaches parity with CART and the random forest" is false leak-free. The table mixes a leak-free flat cell (0.440) with a leak-era HME cell.
6. **A:1608**: "Tuning cuts the mixture's standard deviation from 0.060 to 0.022, nearly a factor of three." This comes from F03 (pinned era, unnamed). Against the table directly above it (0.059→0.027) and the new run (0.062→0.028) the ratio is ≈2.2. The direction holds; "nearly three" does not.
7. **A:241**: "Reaching 0.96 takes fifteen features under every scorer." Already false in the cited leak-free table: composite is 0.9761 at 10 features (and bhattacharyya 0.9575 at 10 just misses). Unchanged in N.
8. **A.4 (A:195–200) vs N**: N's Table A.2 has a fourth scorer column, **higher_order**, that the prose omits. It is best of all four at 3–5 features (0.8603 / 0.9011 / **0.9261 > composite 0.9186** at 5). "The composite is the best of the three from five features on" stays literally true of three, but stops being true of the table as archived.

No sklearn reference cell (CART, RF) moved in mean anywhere in my scope. Only ddof.

### Open questions

1. **Refined-arm move** (0th 0.720→0.747, 2nd 0.862→0.875): it happened between goal-8h (`80e98d7`, 0.862) and pins-synced/cluster-2026-08-27 (0.878), while every closed-form arm stayed byte-identical. `tribble-fis/src/tribblefis/refine.py` has commits in that window (#134/#136 route sub-solves through `optimizers`; #178 reuses the normal-equation matrix; #181 determinism; #209 constant-feature bound), so the pin-bump explanation is plausible. But I found no PROVENANCE_MAP entry that records this specific move. Worth a sentence in the map before re-quoting Table 6.1.
2. **Table 6.1's declared run of record is still `uniform-2026-08-03`** (06:266, 06:327). Does the author want it re-pointed at N? If so, Table 6.1 cells change as listed (mostly DDOF; refinement row moves).
3. **`full-2026-08-22/table_a2_feature_count.csv`** has 0.9967 under *bhattacharyya* and 0.4267 under *composite* at one feature. `main-d0efefc`, U and F03 have 0.9967 under *wasserstein* and 0.4267 under *bhattacharyya* (the version the prose quotes). This is leak-era and withdrawn, but either a scorer default changed or columns were relabelled in that window.
4. A:1327 "0.005": I could not identify the exact cell (quantile − hybrid gives 0.002/0.004 at 1st/2nd; uniform − hybrid at 2nd gives 0.009).
5. 06:345 "a full-second-order arm losing 0.027": no source archive named, and not found in the CR/HN files I scanned.
6. A.7 California Housing flat MoG moved 0.631 → 0.664 in the (non-citable) Xeon bump archive. Once A.7 is wired into the runner, expect that cell to change.
