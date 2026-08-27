# G2 downstream — all five named DTW datasets, corrected transform (2026-08-25/26)

**What this archive is.** The first complete measurement of Goal G2's
decision-rule item 3 ("downstream usefulness") on **all five** named UCR/UEA
DTW datasets, at **full N for every dataset** — including StarLightCurves'
9,236 × length-1,024 build, which was infeasible (~30 h) before this run.

**Command** (from repo root, on `research/g2-remaining-datasets`):

```
REPRO_G2_DOWNSTREAM_DATASETS="ECG5000,FordA,ElectricDevices,StarLightCurves,Crop" \
REPRO_G2_DTW_IMPL=simd \
REPRO_G2_DTW_CACHE=reproduce/outputs/dtw-cache \
  uv run --project tribble-cluster --with aeon --with scipy --with scikit-learn \
  python reproduce/tables/table_3_7_g2_downstream.py
```

Seeds: `common.SEEDS` (ten, 0–9) for NERFCM restarts. Machine stamp is embedded
in the table's own footer.

**What is different from the 2026-08-12 three-dataset run, and why it is safe:**

1. **DTW kernel**: `REPRO_G2_DTW_IMPL=simd` uses `experiments/dtw-simd`'s
   OpenMP + AVX-512 Cython kernel, **equality-verified against aeon on a
   seeded 300-point subsample of the actual data at every call** (max |diff|
   ≤ 5.7e-13 across all five datasets; the script refuses to substitute on any
   disagreement).

   **Speedup, decomposed (corrected 2026-08-26).** At *equal core budget* the
   kernel is **~3.3–4.8×** faster (`fair_bench.py`: 600×96 → 4.8×, 150×1024 →
   3.3× single-thread-vs-single-thread). The wall-clock gains below are larger,
   ~10–12×, because every call site in this harness had been invoking
   `dtw_pairwise_distance` at its default `n_jobs=1`; aeon parallelises when
   asked, and ~3× of the observed gain is that parallelism rather than this
   kernel. An earlier version of this file reported the 10–12× figure without
   that decomposition — it conflated two variables, and the corrected reading
   is stated here rather than silently replaced.

   Observed build times in this run (vs aeon at n_jobs=1, as previously run):
   ECG5000 59 s (was ~630 s), FordA ~15–20 min (was ~2 h), ElectricDevices
   347–457 s, StarLightCurves ~4.6–4.7 h (was ~30 h, never attempted), Crop
   202–298 s (was ~1,600 s). Against a properly parallelised aeon the same
   builds would have been roughly 3× the "was" figures divided by three —
   e.g. StarLightCurves ~10 h, so this kernel's own contribution there is
   ~10 h → 4.6 h.
2. **Minimax transform**: the lowmem split-phase dense-Prim implementation in
   `table_3_7_g2_downstream.py`, **equality-verified per call against the
   O(n³) reference `ivat_mf.minimax_transform`** (max |diff| = 0.0 on every
   dataset). Deliberately NOT verified against `minimax_transform_fast`: that
   gate caught a latent bug in it — `csr_matrix` drops exact-zero entries
   (duplicate points), so scipy's MST cannot use those edges and `_fast`
   inflates D* for duplicate pairs. ElectricDevices has 40 such pairs in its
   verification subsample and Crop has 2; ECG5000/FordA/StarLightCurves have
   none. Consequence: **ECG5000 and FordA reproduce their 2026-08-12 rows
   exactly**, while **Crop is a corrected re-measurement** (its
   `select_coverage_cover` moved 0.064 → 0.114; NERFCM 0.029 → 0.030).
3. **nerfcm memory fix** (`gated-minimax-selection/nerfcm.py`): the upfront
   `Dbeta = D.copy()` became a reference — provably behavior-identical (Dbeta
   is never written in place; the beta-activation path builds a new array),
   verified by a byte-diff of `run_all.py`'s results.json. Needed to fit
   Crop's N=24,000 in 15 GB RAM.
4. **Bonus metrics cap**: single-linkage-given-k and beta-plateau (context
   columns, not on the decision rule's hard threshold) are skipped with a
   stamped n/a above `REPRO_G2_BONUS_NMAX` (default 20,000) — at Crop's
   N=24,000 their extra memory OOM-killed a run after both decision-rule
   quantities had already printed.
5. The table is re-emitted after every dataset, so no finished row can be
   lost to a crash (this archive's table survived a machine reboot that way),
   and DTW matrices are cached under `reproduce/outputs/dtw-cache/`
   (gitignored) so a re-run costs only downstream time.

**Decision-rule outcome** (gated set-cover within 0.05 ARI of NERFCM-given-k
on ≥3 of the 5 sets):

| dataset | NERFCM given k | set-cover (discovers k) | gap | within 0.05? |
|---|---|---|---|---|
| ECG5000 | 0.593 ± 0.047 | **0.715** (k=25) | 0.122 | no — set-cover ABOVE |
| FordA | 0.000 | 0.002 (k=18) | 0.001 | yes (degenerate tie) |
| ElectricDevices | 0.175 | 0.147 (k=743) | 0.028 | yes |
| StarLightCurves | 0.482 ± 0.060 | **0.664** (k=7) | 0.182 | no — set-cover ABOVE |
| Crop | 0.030 ± 0.012 | **0.114** (k=1142) | 0.084 | no — set-cover ABOVE |

**2 of 5 within the band: the item does not close on the letter of the
threshold.** The refutation condition (missing 0.05 on *every* set) also
plainly does not hold. What the complete measurement actually shows is
stronger than the parity the rule asked for: on every dataset where either
method finds real structure (ECG5000, StarLightCurves, Crop), the set-cover
**beats** NERFCM-given-k by 0.08–0.18 ARI while discovering k itself; the only
within-band cases are the two degenerate near-zero ties. Whether to amend the
decision rule to a one-sided criterion is a call for the author, not this run.
