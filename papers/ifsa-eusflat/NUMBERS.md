# Numbers in the draft: provenance and status (v0.3)

Status vocabulary follows `reproduce/PROVENANCE_MAP.md`. **Do not cite** a row marked provisional.

| Row | Paper location | Values | Source | Status and blocker |
|---|---|---|---|---|
| T1 | §4 Table 1 | Concrete R² by transform | `reproduce/outputs/pins-synced-2026-08-27/table_hyperparam_normalization.md` | **Provisional.** Backfilled, not run in that pass. Not in the `bump-2bdf875` archive. Re-run at `2bdf875` and diff every column. |
| T2 | §5.1 Table 2 — accuracy by scorer and k | Four scorers, k = 1–20 | `reproduce/outputs/bump-2bdf875-2026-11-21/table_a2_feature_count.md` | **Verified at `2bdf875`.** Passing preflight (6/6) in `bump-2bdf875-2026-11-21/preflight.txt`. The three original scorers are identical to the `phiusiil-leakfree` values. Uses a 20,000-row sample. |
| T3 | §5.1 Table 3 — top five by scorer | Feature names and scores | `reproduce/outputs/bump-2bdf875-2026-11-21/table_a1_feature_ranking.md` | **Verified at `2bdf875`.** Same archive and preflight as T2. |
| T4 | §5.2 Table 4 — shipped vs sweep vs baselines | Concrete, Bike, PhiUSIIL | `reproduce/outputs/bump-2bdf875-2026-11-21/table_4_1.md` and `reproduce/outputs/topn-sweep-2026-11-21/table_4_1c_topn_sweep.md` | **Verified at `2bdf875`.** Passing preflight in both archives. Sweep best cells are a **ceiling**: chosen on the test splits. |
| T4a | §5.2 Table 4 — RT-IOT2022 row | MoG 0.927; ANFIS 0.983; GA-FIS 0.988; RF 0.998 | `reproduce/outputs/phiusiil-leakfree-2026-08-30/table_4_1.md` | **Provisional.** That archive failed preflight. The dataset is absent from the `2bdf875` run (`N/A`). |
| T4b | §5.2 Bike ANFIS 0.538 ± 0.092 | `bump-2bdf875` table_4_1 | — | **Verified, but run-to-run.** ANFIS is nondeterministic. The earlier 0.577 ± 0.075 is superseded and the paper says so. |
| T5 | §5.2 Table 5 — PhiUSIIL accuracy by k and scorer | Full-data sweep, k = 1–30 | `reproduce/outputs/topn-sweep-2026-11-21/table_4_1c_topn_sweep.md` | **Verified at `2bdf875`.** The sweep's sample size is not stated in the archive, so Table 5 is not directly comparable to Table 2's 20,000-row sample beyond k = 20. |
| T6 | §5.4 Table 6 — training time | Four datasets; MoG, ANFIS, GA-FIS | `reproduce/outputs/phiusiil-leakfree-2026-08-30/table_4_1b_baseline_timing.md` | **Provisional.** Same failed archive. The current `2bdf875` run reports MoG times only, on a 4-core VM (Concrete 0.24 s, Bike 0.44 s, PhiUSIIL 0.18 s). Re-run ANFIS and GA-FIS timings on one host. |
| T7 | §5.5 Glass correction pass +0.008 ± 0.011 | `table_4_9_correction_pass` | `reproduce/outputs/*/table_4_8_mf_dedup*` | **Stale.** Earlier gain was +0.031 ± 0.027 under tribble-fis #218. Not re-derived at ten seeds on the current pin. |
| T8 | §5.3 leak: 0.997 retracted; 0.7924 leak-free | Appendix A.4; PROVENANCE_MAP note 31 | `bump-2bdf875` table_a2 (one-feature row) | 0.997 is cited only as the retracted value. 0.7924 is T2. |

## Changes from v0.2

- PhiUSIIL shipped 0.440 is now labelled as a feature-count and scorer effect. The abstract and §5.2 say so. The earlier claim that the MoG trails the baselines on PhiUSIIL is removed.
- Higher-order scorer added as a fourth column (T2, T3, T5). Its definition is not restated.
- Bike Sharing ANFIS updated from 0.577 to 0.538 (T4b).
- Timings moved to provisional (T6), since the current run reports MoG times only.
- Sweep ceiling caveat added (§5.2 and §6).

## Still open

1. **Composite identity.** `composite` at `2bdf875` matches the leak-free values, but it is not verified identical to the original four-metric blend.
2. **Nested selection.** Any sweep-vs-baseline comparison needs a held-out selection step before it can be cited.
3. **Clean timing re-run** (T6): ANFIS and GA-FIS on one host, with the MoG timings from the same run.
4. **RT-IOT2022 re-run** (T4a): dataset absent from current runs.
5. **Transform re-run** (T1) at `2bdf875`.
6. **Whole-document sweep** at `2bdf875` if the paper will cite it; the run of record is still `full-2026-08-22` at `141596e`.
7. **DOIs** for `hastie2009elements`, `kaufman2012leakage`, `lin1991divergence`, `villani2008optimal` (from memory; `bhattacharyya1943` has no DOI); histogram-term reference.
8. **Template and page limit** for IFSA-EUSFLAT.
