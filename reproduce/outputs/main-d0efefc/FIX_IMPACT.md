# Fix impact — `baseline-d0d6714` → `postfix-pr29`

Cell-by-cell diff of the archived table runs, produced by `reproduce/compare_runs.py`. Every table is listed, including the unchanged ones: confining a fix's blast radius is a claim, and it is only supported by showing the tables that did *not* move.

<details><summary>Provenance — <code>baseline-d0d6714</code></summary>

```
label:       baseline-d0d6714
generated:   2026-08-01T17:06:52Z
tribble-fis: d0d67144f4d084a905f16031c13b6a0301ca85f8
tribble-cluster: c71171e76fade0bfc3ac9859478dbf45fa9a463e
grad-school: eabb10587aacb34404eea677468ae49b68b94f97
seeds:       0,1,2,3,4 (default)

status:
  table_concrete_reconciliation          ok
  table_hyperparam_normalization         ok
  table_g5_output_partitioning           ok
  table_g5b_skew_sweep                   ok
  table_4_1_mog_baselines                ok
  table_6_1_model_family                 ok
  table_4_4_openset                      no-output (corrected: no glass.csv; script exited 0 having written nothing)
  table_3_1_pvat_scaling                 FAILED
  table_3_1_reorder_three_arm            ok

--- backfill 2026-08-01T17:12:04Z: table_3_1_pvat_scaling ---
tribble-fis: d0d67144f4d084a905f16031c13b6a0301ca85f8
tribble-cluster: c71171e76fade0bfc3ac9859478dbf45fa9a463e
grad-school: dceef9daf8e97cc557cd95026f32cb4435808d5a
seeds:       0,1,2,3,4 (default)

status:
  table_concrete_reconciliation          not-run-this-pass
  table_hyperparam_normalization         not-run-this-pass
  table_g5_output_partitioning           not-run-this-pass
  table_g5b_skew_sweep                   not-run-this-pass
  table_4_1_mog_baselines                not-run-this-pass
  table_6_1_model_family                 not-run-this-pass
  table_4_4_openset                      not-run-this-pass
  table_3_1_pvat_scaling                 ok
  table_3_1_reorder_three_arm            not-run-this-pass
```

</details>

<details><summary>Provenance — <code>postfix-pr29</code></summary>

```
label:       postfix-pr29
generated:   2026-08-01T17:35:58Z
tribble-fis: d4dd392a278c4a89d865a6c30a82270cfc2581a9
tribble-cluster: c71171e76fade0bfc3ac9859478dbf45fa9a463e
grad-school: e8a00b776ae95cb93f8338dc685068c125fc2c32
seeds:       0,1,2,3,4 (default)

status:
  table_concrete_reconciliation          ok
  table_hyperparam_normalization         ok
  table_g5_output_partitioning           ok
  table_g5b_skew_sweep                   ok
  table_4_1_mog_baselines                ok
  table_6_1_model_family                 ok
  table_4_4_openset                      no-output (corrected: reported ok by a
                                         faulty mtime-window check; the script
                                         wrote nothing -- no glass.csv)
  table_3_1_pvat_scaling                 ok
  table_3_1_reorder_three_arm            ok

--- backfill 2026-08-01T17:36:30Z: table_4_4_openset ---
tribble-fis: d4dd392a278c4a89d865a6c30a82270cfc2581a9
tribble-cluster: c71171e76fade0bfc3ac9859478dbf45fa9a463e
grad-school: e8a00b776ae95cb93f8338dc685068c125fc2c32
seeds:       0,1,2,3,4 (default)

status:
  table_concrete_reconciliation          not-run-this-pass
  table_hyperparam_normalization         not-run-this-pass
  table_g5_output_partitioning           not-run-this-pass
  table_g5b_skew_sweep                   not-run-this-pass
  table_4_1_mog_baselines                not-run-this-pass
  table_6_1_model_family                 not-run-this-pass
  table_4_4_openset                      no-output
  table_3_1_pvat_scaling                 not-run-this-pass
  table_3_1_reorder_three_arm            not-run-this-pass
```

</details>

## Summary

| Table | Cells | Verdict |
|---|---:|---|
| `table_3_1` | 16 | 11 timing |
| `table_3_1_three_arm` | 20 | 16 timing |
| `table_4_1` | 6 | **2 changed**, 2 timing |
| `table_6_1` | 16 | **5 changed** |
| `table_concrete_reconciliation` | 34 | **2 changed**, 4 within noise |
| `table_g5_output_partitioning` | 126 | identical |
| `table_g5b_skew_sweep` | 48 | identical |
| `table_hyperparam_normalization` | 30 | identical |

## What moved

### `table_3_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,024 | classical VAT | 22.988 ± 0.405 s | 23.183 ± 0.531 s | +0.1950 | timing |
| 1,024 | pVAT | 0.033 ± 0.005 s | 0.033 ± 0.008 s | +0.0000 | timing |
| 1,024 | speedup | 703x | 692x | -11.0000 | timing |
| 2,048 / infeasible (>cap) | pVAT | 0.139 ± 0.007 s | 0.110 ± 0.004 s | -0.0290 | timing |
| 256 | classical VAT | 0.390 ± 0.034 s | 0.377 ± 0.007 s | -0.0130 | timing |
| 256 | pVAT | 1.271 ± 2.536 s | 0.031 ± 0.058 s | -1.2400 | timing |
| 256 | speedup | 0x | 12x | +12.0000 | timing |
| 4,096 / infeasible (>cap) | pVAT | 0.515 ± 0.094 s | 0.513 ± 0.061 s | -0.0020 | timing |
| 512 | classical VAT | 3.046 ± 0.163 s | 2.913 ± 0.086 s | -0.1330 | timing |
| 512 | pVAT | 0.011 ± 0.003 s | 0.008 ± 0.001 s | -0.0030 | timing |
| 512 | speedup | 274x | 356x | +82.0000 | timing |

### `table_3_1_three_arm`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,000 | classical O(N³) | 0.2704 ± 0.0071 s | 0.2768 ± 0.0061 s | +0.0064 | timing |
| 1,000 | stage 1 O(N²logN) | 0.0157 ± 0.0010 s | 0.0151 ± 0.0005 s | -0.0006 | timing |
| 1,000 | stage 2 O(N²) | 0.0084 ± 0.0003 s | 0.0056 ± 0.0035 s | -0.0028 | timing |
| 1,000 | cls/s2 | 32.3× | 49.4× | +17.1000 | timing |
| 1,000 | s1/s2 | 1.9× | 2.7× | +0.8000 | timing |
| 2,000 / not run (> cap) | stage 1 O(N²logN) | 0.0570 ± 0.0018 s | 0.0527 ± 0.0013 s | -0.0043 | timing |
| 2,000 / not run (> cap) | stage 2 O(N²) | 0.0071 ± 0.0025 s | 0.0048 ± 0.0002 s | -0.0023 | timing |
| 2,000 / not run (> cap) | s1/s2 | 8.1× | 11.0× | +2.9000 | timing |
| 4,000 / not run (> cap) | stage 1 O(N²logN) | 0.2744 ± 0.0078 s | 0.2640 ± 0.0051 s | -0.0104 | timing |
| 4,000 / not run (> cap) | stage 2 O(N²) | 0.0265 ± 0.0019 s | 0.0233 ± 0.0020 s | -0.0032 | timing |
| 4,000 / not run (> cap) | s1/s2 | 10.3× | 11.3× | +1.0000 | timing |
| 500 | classical O(N³) | 0.0270 ± 0.0005 s | 0.0313 ± 0.0049 s | +0.0043 | timing |
| 500 | stage 1 O(N²logN) | 0.0047 ± 0.0003 s | 0.0046 ± 0.0002 s | -0.0001 | timing |
| 500 | stage 2 O(N²) | 0.0006 ± 0.0000 s | 0.0007 ± 0.0000 s | +0.0001 | timing |
| 500 | cls/s2 | 42.5× | 47.8× | +5.3000 | timing |
| 500 | s1/s2 | 7.4× | 7.1× | -0.3000 | timing |

### `table_4_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| PhiUSIIL (classification) | MoG accuracy / R2 | N/A | acc=0.997 ± 0.001 |  | **changed** |
| PhiUSIIL (classification) | tree / RF ref | N/A | 1.000 ± 0.000 |  | **changed** |
| Concrete (regression) | MoG train time | 0.60 ± 0.04 s | 0.63 ± 0.07 s | +0.0300 | timing |
| PhiUSIIL (classification) | MoG train time | N/A | 0.99 ± 0.09 s |  | timing |

### `table_6_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| PhiUSIIL / accuracy | flat | N/A | 0.997 ± 0.001 |  | **changed** |
| PhiUSIIL / accuracy | fuzzy tree | N/A | 0.969 ± 0.001 |  | **changed** |
| PhiUSIIL / accuracy | mixture (HME) | N/A | 0.997 ± 0.001 |  | **changed** |
| PhiUSIIL / accuracy | CART | N/A | 1.000 ± 0.000 |  | **changed** |
| PhiUSIIL / accuracy | Random Forest | N/A | 1.000 ± 0.000 |  | **changed** |

### `table_concrete_reconciliation`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| flat MoG-TSK 0th / log+standardized / refined | R² | 0.135 ± 0.325 | 0.461 ± 0.116 | +0.3260 | **changed** |
| flat MoG-TSK 0th / log+standardized / refined | RMSE | 15.04 ± 2.64 | 12.01 ± 0.98 | -3.0300 | **changed** |
| flat MoG-TSK 1st / log+standardized / refined | R² | 0.822 ± 0.056 | 0.822 ± 0.032 | +0.0000 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | RMSE | 6.85 ± 0.84 | 6.92 ± 0.54 | +0.0700 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | R² | 0.853 ± 0.020 | 0.868 ± 0.020 | +0.0150 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | RMSE | 6.32 ± 0.53 | 5.96 ± 0.32 | -0.3600 | within noise |

## Bit-identical

These tables produced exactly the same numbers on both sides:

- `table_g5_output_partitioning` (126 cells)
- `table_g5b_skew_sweep` (48 cells)
- `table_hyperparam_normalization` (30 cells)

---

> A cell counts as **changed** only if it moved by more than the larger of the two runs' reported standard deviations; smaller moves are labelled *within noise*. Wall-clock columns are always reported separately and never called a regression — this harness does not control clocks or thermals (see G4 in `NEXT_STEPS.md`).
