# Numbers in the draft: provenance and status

Status vocabulary follows `reproduce/PROVENANCE_MAP.md`. **Do not cite** any row not marked *reproduced*.

| Row | Paper location | Value(s) | Source archive | Status / blocker |
|---|---|---|---|---|
| T1 | §4 Table 1 | Concrete R² by transform, all rows | `reproduce/outputs/pins-synced-2026-08-27/table_hyperparam_normalization.md` | **Provisional.** Its status line says the table was *backfilled* (2026-08-28), not run in that pass. Re-run `table_hyperparam_normalization.py` at the current pin and diff every column (AGENTS.md non-negotiable 5). The raw and min–max columns were earlier verified byte-identical to `full-14900hx-r2`, which predates the B14 fix. |
| T2 | §5.1 Table 2 — MoG, ANFIS, GA-FIS, RF scores | Concrete, Bike, PhiUSIIL, RT-IOT2022 | `reproduce/outputs/phiusiil-leakfree-2026-08-30/table_4_1.md` | **Provisional, not citable.** `PREFLIGHT FAILED`: `INSTALL-FRESH` (stale installed `tribbleclustering` build) and `PIN-MATCH` (archive names `1dcf331`/`4d81121`, submodules at `71dbcc3`/`7ba4fc0`). Per the archive's own banner, every number in it is suspect. |
| T3 | §5.1 Table 2 — train times | Same four datasets | `reproduce/outputs/phiusiil-leakfree-2026-08-30/table_4_1b_baseline_timing.md` | **Provisional.** Same archive and preflight failure as T2. The "14×–194×" headline depends on it. |
| T4 | §5.1 prose: "matches ANFIS on Concrete", "beats both on Bike" | Derived from T2 | — | **Provisional.** Restate only after a clean T2 run. |
| T5 | §5.3 PhiUSIIL leak; 0.997 → 0.440 ± 0.181 | Proposal §4.4 and PROVENANCE_MAP (Table 4.4/4.5 rows) | `phiusiil-leakfree-2026-08-30` | 0.440 is from T2's archive: **provisional**. The 0.997 figure is a *retracted* value (leaked screen), cited only as the retracted number. |
| T6 | §5.2 Glass correction pass +0.008 ± 0.011 | `table_4_9_correction_pass` (PROVENANCE_MAP row 4.9) | `reproduce/outputs/*/table_4_8_mf_dedup*` | **Stale.** The gain fell from +0.031 ± 0.027 under tribble-fis #218; the paper reports the later figure as current but it has not been re-derived at ten seeds on the current pin. Treat as provisional. |
| T7 | §5.3 top-five features: four binary flags, one bounded score | Proposal Ch. 4 prose (lines 513–516) | `reproduce/outputs/phiusiil-leakfree-2026-08-30/table_a1_feature_scoring.md` (status *ok* per table) | Qualitative. Confirm against the table_a1 CSV before quoting. |

## Open items before submission

1. **Clean re-run** of `table_4_1_mog_baselines.py` and `table_hyperparam_normalization.py` (T1–T3). Requires:
   - `git submodule update --init` (submodules are uninitialised in this container; `git submodule status` shows `-`);
   - `uv lock --upgrade-package tribble-clustering` in `tribble-fis`, reinstall, then `PIN-MATCH` and `INSTALL-FRESH` both green (`preflight.py`);
   - data in `data/` (Concrete is present; PhiUSIIL is recovered from `tribble-fis` history per `data/.gitignore`);
   - roughly 80 minutes on the run-of-record host, per the 2026-08-30 run.
2. Archive the clean run under `reproduce/outputs/<label>/` and add a row to `reproduce/PROVENANCE_MAP.md`.
3. Confirm the GA-FIS baseline reference and add the Gaussian naive Bayes and feature-screen references (`@TODO-*` keys in `paper.md`).
4. Venue template: the draft is Markdown. Transfer to the IFSA-EUSFLAT LaTeX/Word template and check the page limit.
