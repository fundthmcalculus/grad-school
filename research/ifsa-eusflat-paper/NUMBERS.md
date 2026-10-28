# Numbers in the draft: provenance and status

Status vocabulary follows `reproduce/PROVENANCE_MAP.md`. **Do not cite** a row until its status is *reproduced*.

| Row | Paper location | Values | Source | Status and blocker |
|---|---|---|---|---|
| T1 | §4 Table 1 | Concrete R² by transform | `reproduce/outputs/pins-synced-2026-08-27/table_hyperparam_normalization.md` | **Provisional.** Backfilled (2026-08-28), not run in that pass. Re-run at the current pin and diff every column. Earlier raw and min–max columns were byte-identical to `full-14900hx-r2`, which predates the B14 fix. |
| T2 | §5.1 Table 2 — scorer comparison | Accuracy, 1–20 features, three scorers | `reproduce/outputs/phiusiil-leakfree-2026-08-30/table_a2_feature_count.md` (appendix A.2) | **Provisional.** Same archive as T4, which `PREFLIGHT FAILED`. The failed checks are `INSTALL-FRESH` (stale `tribbleclustering/fcm.py`) and `PIN-MATCH`. The scorer path is in `tribble-fis`, not FCM, but nothing here verifies that. Re-run under a passing preflight. |
| T3 | §5.1 Table 3 — top five by scorer | Feature names and normalised scores | `reproduce/outputs/phiusiil-leakfree-2026-08-30/table_a1_feature_ranking.md` (appendix A.1) | **Provisional.** Same status as T2. The composite column is the restored method, not verified identical to the original four-metric blend. |
| T4 | §5.4 Table 4 — time and score, MoG vs ANFIS, GA-FIS, RF | Four datasets | `reproduce/outputs/phiusiil-leakfree-2026-08-30/table_4_1.md` and `table_4_1b_baseline_timing.md` | **Provisional, not citable.** `PREFLIGHT FAILED`, and the archive's own banner says every number in it is suspect. Archive pins `1dcf331` and `4d81121` differ from the submodules (`71dbcc3`, `7ba4fc0`). |
| T5 | §5.4 prose: "matches ANFIS", "best of the three fuzzy methods on Bike Sharing", "14×–194×" | Derived from T4 | — | **Provisional.** Restate only after a clean T4 run. |
| T6 | §5.5 Glass correction pass +0.008 ± 0.011 | `table_4_9_correction_pass` | `reproduce/outputs/*/table_4_8_mf_dedup*` | **Stale.** Gain fell from +0.031 ± 0.027 under tribble-fis #218. Not re-derived at ten seeds on the current pin. |
| T7 | §5.3 leak: one-feature 0.997 (retracted), 0.7924 leak-free | Appendix A.4, PROVENANCE_MAP note 31 | `phiusiil-leakfree-2026-08-30` | 0.997 is cited only as the retracted value. 0.7924 is T2. |

## Resolved in v0.2

- Scorer description corrected. v0.1 said the screen used a Wasserstein distance. The leak-free evidence shows Wasserstein is the weakest scorer at small budgets, and the composite is the best from five features on.
- Shipped-default claim removed. The appendix says the shipped scorer default was chosen on a leaked result, and the paper does not defend it. The exact current default is not verified here.
- Composite identity flagged. The restored `method="composite"` is not verified to match the original four-metric blend.
- Cross-host Bhattacharyya variation (up to 0.043) is withdrawn. It came from the leaked archive and has not been re-measured leak-free.

## References: TODOs closed

| Key | Source | Status |
|---|---|---|
| `cordon2001genetic`, `herrera2008genetic` | Copied from `research/proposal-defense/references.bib` (GA-FIS baseline) | Closed |
| `hastie2009elements` (naive Bayes) | New entry | Closed. **DOI `10.1007/978-0-387-84858-7` to verify.** |
| `kaufman2012leakage` (leakage) | New entry | Closed. **DOI `10.1145/2382577.2382579` to verify.** |
| `bhattacharyya1943`, `lin1991divergence`, `villani2008optimal` (scorers) | New entries | `lin1991divergence` DOI `10.1109/18.61115` and `villani2008optimal` DOI `10.1007/978-3-540-71050-9` to verify. `bhattacharyya1943` has no DOI. |
| Histogram overlap and histogram correlation | Composite terms, no reference | **Open.** Cite the `tribble-fis` implementation, or add a reference if there is a standard one. |

## Open items before submission

1. **Clean re-run** of the scorer tables (T2–T3) and the baseline tables (T4). Requires:
   - `git submodule update --init` (submodules are uninitialised here: `git submodule status` shows `-`);
   - `uv lock --upgrade-package tribble-clustering` in `tribble-fis`, reinstall, and `PIN-MATCH` and `INSTALL-FRESH` both green;
   - PhiUSIIL recovered from `tribble-fis` history (`data/.gitignore`);
   - roughly 80 minutes on the run-of-record host for T4.
   Archive under `reproduce/outputs/<label>/` and add a `PROVENANCE_MAP.md` row.
2. Verify the three DOIs above and the histogram-term reference.
3. Transfer to the IFSA-EUSFLAT template and check the page limit.
