# Fix impact — `main-d0efefc` → `full-14900hx-r2`

Cell-by-cell diff of the archived table runs, produced by `reproduce/compare_runs.py`. Every table is listed, including the unchanged ones: confining a fix's blast radius is a claim, and it is only supported by showing the tables that did *not* move.

<details><summary>Provenance — <code>main-d0efefc</code></summary>

```
label:       main-d0efefc
generated:   2026-08-02T13:16:08Z
tribble-fis: d0efefc409009e772e1478f35430134051b0fa0b
tribble-cluster: 5d44dfa4c9b501f264192e299a413155fbb3709f
grad-school: e08382d9ddc7290302e5950768134c9ba42b9421
seeds:       0,1,2,3,4,5,6,7,8,9

status:
  table_concrete_reconciliation          ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_hyperparam_normalization         ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_g5_output_partitioning           ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_g5b_skew_sweep                   ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_4_1_mog_baselines                ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_6_1_model_family                 ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_norm_conorm_matrix               ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_4_4_openset                      ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_1_pvat_scaling                 ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_1_reorder_three_arm            ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_5_x_ch5_selection                ok           seeds=0,1,2,3,4,5,6,7,8,9
```

</details>

<details><summary>Provenance — <code>full-14900hx-r2</code></summary>

```
label:       full-14900hx-r2
generated:   2026-08-02T23:29:14Z
tribble-fis: d0efefc409009e772e1478f35430134051b0fa0b
tribble-cluster: e3c27e67ae2a41d636dfb472110ae2ded2e4ef82
grad-school: 4c4fdbcbcd4a5b89fa0a0b42b362e967aac51fe8
seeds:       0,1,2,3,4,5,6,7,8,9
thetas:      0.5,0.6,0.7,0.8,0.9,0.99,1.1 (table_4_4b operating curve)

machine:
  host             NEX-210200
  os               MINGW64_NT-10.0-26200
  kernel           MINGW64_NT-10.0-26200 3.4.10-2e2ef940.x86_64 x86_64
  cpu              Intel(R) Core(TM) i9-14900HX
  cores            32 logical
  ram              95.6 GiB (102.7 GB decimal)
  governor         n/a
  boost            n/a
  gpu              NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB, 610.74
  python           Python 3.13.7

status:
  table_concrete_reconciliation          ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_hyperparam_normalization         ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_g5_output_partitioning           ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_g5b_skew_sweep                   ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_4_1_mog_baselines                ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_6_1_model_family                 ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_norm_conorm_matrix               ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_4_4_openset                      ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_a1_feature_scoring               ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_1_pvat_scaling                 ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_1_reorder_three_arm            ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_2_memory_precision             ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_5_x_ch5_selection                ok           seeds=0,1,2,3,4,5,6,7,8,9

--- backfill 2026-08-02T23:30:52Z: table_5_x_ch5_selection ---
tribble-fis: d0efefc409009e772e1478f35430134051b0fa0b
tribble-cluster: e3c27e67ae2a41d636dfb472110ae2ded2e4ef82
grad-school: 4c4fdbcbcd4a5b89fa0a0b42b362e967aac51fe8
seeds:       0,1,2,3,4,5,6,7,8,9
thetas:      0.5,0.6,0.7,0.8,0.9,0.99,1.1 (table_4_4b operating curve)

machine:
  host             NEX-210200
  os               MINGW64_NT-10.0-26200
  kernel           MINGW64_NT-10.0-26200 3.4.10-2e2ef940.x86_64 x86_64
  cpu              Intel(R) Core(TM) i9-14900HX
  cores            32 logical
  ram              95.6 GiB (102.7 GB decimal)
  governor         n/a
  boost            n/a
  gpu              NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB, 610.74
  python           Python 3.13.7
  numpy            2.4.6  scipy 1.17.1  sklearn 1.9.0  python 3.13.7
  blas             scipy-openblas 0.3.31.188.0

status:
  table_concrete_reconciliation          not-run-this-pass seeds=—
  table_hyperparam_normalization         not-run-this-pass seeds=—
  table_g5_output_partitioning           not-run-this-pass seeds=—
  table_g5b_skew_sweep                   not-run-this-pass seeds=—
  table_4_1_mog_baselines                not-run-this-pass seeds=—
  table_6_1_model_family                 not-run-this-pass seeds=—
  table_norm_conorm_matrix               not-run-this-pass seeds=—
  table_4_4_openset                      not-run-this-pass seeds=—
  table_a1_feature_scoring               not-run-this-pass seeds=—
  table_3_1_pvat_scaling                 not-run-this-pass seeds=—
  table_3_1_reorder_three_arm            not-run-this-pass seeds=—
  table_3_2_memory_precision             not-run-this-pass seeds=—
  table_5_x_ch5_selection                ok           seeds=0,1,2,3,4,5,6,7,8,9
```

</details>

## Summary

| Table | Cells | Verdict |
|---|---:|---|
| `table_3_1` | — | **header-changed** |
| `table_3_1_complexity_fit` | — | **new-only** |
| `table_3_1_three_arm` | — | **header-changed** |
| `table_3_2_memory_precision` | 32 | identical |
| `table_4_1` | 6 | 2 within noise, 2 timing |
| `table_4_4_openset` | 9 | identical |
| `table_4_4b_theta_sweep` | 28 | identical |
| `table_5_1_battery` | 34 | identical |
| `table_5_2_multiscale` | 15 | identical |
| `table_5_3_selection` | 15 | identical |
| `table_6_1` | 16 | 4 within noise |
| `table_a1_feature_ranking` | 20 | identical |
| `table_a2_feature_count` | 36 | **13 changed**, 13 within noise |
| `table_concrete_reconciliation` | 34 | 14 within noise |
| `table_g5_output_partitioning` | 126 | identical |
| `table_g5b_skew_sweep` | 48 | identical |
| `table_hyperparam_normalization` | 48 | **4 changed**, 14 within noise |
| `table_norm_conorm_matrix` | 57 | **1 changed**, 9 within noise |

## Tables that could not be compared

- `table_3_1` — **header-changed**
  - baseline header: `['N (points)', 'classical VAT', 'pVAT', 'speedup']`
  - new header: `['N (points)', 'classical VAT (s)', 'pVAT (s)', 'speedup']`
- `table_3_1_complexity_fit` — **new-only**
- `table_3_1_three_arm` — **header-changed**
  - baseline header: `['N', 'classical O(N³)', 'stage 1 O(N²logN)', 'stage 2 O(N²)', 'cls/s2', 's1/s2', 'orders identical']`
  - new header: `['N', 'classical O(N³) (s)', 'stage 1 O(N²logN) (s)', 'stage 2 O(N²) (s)', 'cls/s2', 's1/s2', 'orders identical']`

## What moved

### `table_4_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete (regression) | MoG train time | 0.61 ± 0.03 s | 1.04 ± 0.62 s | +0.4300 | timing |
| Concrete (regression) | MoG accuracy / R2 | R2=0.650 ± 0.056 | R2=0.780 ± 0.029 | +0.0000 | within noise |
| Concrete (regression) | tree / RF ref | 0.909 ± 0.018 | 0.909 ± 0.019 | +0.0000 | within noise |
| PhiUSIIL (classification) | MoG train time | 0.92 ± 0.03 s | 0.64 ± 0.02 s | -0.2800 | timing |

### `table_6_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete / R2 / 0.650 ± 0.056 | fuzzy tree | 0.580 ± 0.067 | 0.583 ± 0.067 | +0.0030 | within noise |
| Concrete / R2 / 0.650 ± 0.056 | mixture (HME) | 0.682 ± 0.064 | 0.686 ± 0.060 | +0.0040 | within noise |
| Concrete / RMSE (MPa) / 9.633 ± 0.536 | fuzzy tree | 10.575 ± 0.863 | 10.531 ± 0.889 | -0.0440 | within noise |
| Concrete / RMSE (MPa) / 9.633 ± 0.536 | mixture (HME) | 9.167 ± 0.792 | 9.114 ± 0.738 | -0.0530 | within noise |

### `table_a2_feature_count`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 10 | wasserstein (acc / fit s) | 0.9996 / 0.98 | 0.9995 / 1.23 | -0.0001 | **changed** |
| 10 | bhattacharyya (acc / fit s) | 0.9274 / 1.23 | 0.9676 / 1.83 | +0.0402 | **changed** |
| 15 | wasserstein (acc / fit s) | 0.9957 / 1.45 | 0.9974 / 1.96 | +0.0017 | **changed** |
| 15 | bhattacharyya (acc / fit s) | 0.9477 / 1.96 | 0.9765 / 2.91 | +0.0288 | **changed** |
| 20 | wasserstein (acc / fit s) | 0.9989 / 1.92 | 0.9990 / 2.70 | +0.0001 | **changed** |
| 20 | bhattacharyya (acc / fit s) | 0.9477 / 2.40 | 0.9777 / 3.54 | +0.0300 | **changed** |
| 3 | bhattacharyya (acc / fit s) | 0.8457 / 0.34 | 0.8455 / 0.53 | -0.0002 | **changed** |
| 4 / 0.9967 / 0.59 | bhattacharyya (acc / fit s) | 0.8986 / 0.46 | 0.9160 / 0.72 | +0.0174 | **changed** |
| 5 | wasserstein (acc / fit s) | 0.9965 / 0.66 | 0.9966 / 0.70 | +0.0001 | **changed** |
| 5 | bhattacharyya (acc / fit s) | 0.9131 / 0.62 | 0.9456 / 0.95 | +0.0325 | **changed** |
| 7 | wasserstein (acc / fit s) | 0.9997 / 0.77 | 0.9998 / 0.89 | +0.0001 | **changed** |
| 7 | bhattacharyya (acc / fit s) | 0.9183 / 0.85 | 0.9610 / 1.32 | +0.0427 | **changed** |
| 7 | composite (acc / fit s) | 0.9966 / 1.02 | 0.9967 / 1.22 | +0.0001 | **changed** |
| 1 | wasserstein (acc / fit s) | 0.9967 / 0.41 | 0.9967 / 0.55 | +0.0000 | within noise |
| 1 | bhattacharyya (acc / fit s) | 0.4267 / 0.16 | 0.4267 / 0.25 | +0.0000 | within noise |
| 1 | composite (acc / fit s) | 0.4267 / 0.41 | 0.4267 / 0.30 | +0.0000 | within noise |
| 10 | composite (acc / fit s) | 0.9980 / 1.36 | 0.9980 / 1.69 | +0.0000 | within noise |
| 15 | composite (acc / fit s) | 0.9999 / 1.90 | 0.9999 / 2.54 | +0.0000 | within noise |
| 2 | wasserstein (acc / fit s) | 0.9967 / 0.45 | 0.9967 / 0.43 | +0.0000 | within noise |
| 2 | bhattacharyya (acc / fit s) | 0.4527 / 0.22 | 0.4527 / 0.34 | +0.0000 | within noise |
| 2 | composite (acc / fit s) | 0.9967 / 0.48 | 0.9967 / 0.45 | +0.0000 | within noise |
| 20 | composite (acc / fit s) | 0.9995 / 2.58 | 0.9995 / 3.52 | +0.0000 | within noise |
| 3 | wasserstein (acc / fit s) | 0.9967 / 0.55 | 0.9967 / 0.51 | +0.0000 | within noise |
| 3 | composite (acc / fit s) | 0.9967 / 0.54 | 0.9967 / 0.51 | +0.0000 | within noise |
| 4 / 0.9967 / 0.59 | composite (acc / fit s) | 0.9966 / 0.67 | 0.9966 / 0.71 | +0.0000 | within noise |
| 5 | composite (acc / fit s) | 0.9966 / 0.79 | 0.9966 / 0.89 | +0.0000 | within noise |

### `table_concrete_reconciliation`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| flat MoG-TSK 0th / log+standardized / refined | R² | 0.580 ± 0.083 | 0.582 ± 0.072 | +0.0020 | within noise |
| flat MoG-TSK 0th / log+standardized / refined | RMSE | 10.55 ± 1.02 | 10.54 ± 0.95 | -0.0100 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | R² | 0.844 ± 0.059 | 0.836 ± 0.054 | -0.0080 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | RMSE | 6.31 ± 1.00 | 6.52 ± 0.93 | +0.2100 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | R² | 0.861 ± 0.044 | 0.864 ± 0.046 | +0.0030 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | RMSE | 6.01 ± 0.79 | 5.94 ± 0.82 | -0.0700 | within noise |
| fuzzy tree / log+standardized / n/a | R² | 0.688 ± 0.056 | 0.689 ± 0.056 | +0.0010 | within noise |
| fuzzy tree / log+standardized / n/a | RMSE | 9.09 ± 0.58 | 9.07 ± 0.57 | -0.0200 | within noise |
| fuzzy tree / raw / n/a | R² | 0.580 ± 0.067 | 0.583 ± 0.067 | +0.0030 | within noise |
| fuzzy tree / raw / n/a | RMSE | 10.57 ± 0.86 | 10.53 ± 0.89 | -0.0400 | within noise |
| mixture of experts (HME) / log+standardized / n/a | R² | 0.781 ± 0.068 | 0.763 ± 0.057 | -0.0180 | within noise |
| mixture of experts (HME) / log+standardized / n/a | RMSE | 7.54 ± 0.98 | 7.90 ± 0.93 | +0.3600 | within noise |
| mixture of experts (HME) / raw / n/a | R² | 0.682 ± 0.064 | 0.686 ± 0.060 | +0.0040 | within noise |
| mixture of experts (HME) / raw / n/a | RMSE | 9.17 ± 0.79 | 9.11 ± 0.74 | -0.0600 | within noise |

### `table_hyperparam_normalization`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| fuzzy tree / demo-tuned | Δ from normalizing | +0.027 | +0.028 | +0.0010 | **changed** |
| fuzzy tree / library default | Δ from normalizing | +0.108 | +0.106 | -0.0020 | **changed** |
| mixture of experts / demo-tuned / 0.768 ± 0.029 | Δ from normalizing | +0.065 | +0.066 | +0.0010 | **changed** |
| mixture of experts / library default | Δ from normalizing | +0.099 | +0.077 | -0.0220 | **changed** |
| fuzzy tree / demo-tuned | raw features | 0.714 ± 0.029 | 0.712 ± 0.030 | -0.0020 | within noise |
| fuzzy tree / demo-tuned | log + standardized | 0.741 ± 0.051 | 0.740 ± 0.051 | -0.0010 | within noise |
| fuzzy tree / demo-tuned | RMSE raw (MPa) | 8.749 ± 0.513 | 8.764 ± 0.521 | +0.0150 | within noise |
| fuzzy tree / demo-tuned | RMSE log+std (MPa) | 8.279 ± 0.632 | 8.294 ± 0.626 | +0.0150 | within noise |
| fuzzy tree / library default | raw features | 0.580 ± 0.067 | 0.583 ± 0.067 | +0.0030 | within noise |
| fuzzy tree / library default | log + standardized | 0.688 ± 0.056 | 0.689 ± 0.056 | +0.0010 | within noise |
| fuzzy tree / library default | RMSE raw (MPa) | 10.575 ± 0.863 | 10.531 ± 0.889 | -0.0440 | within noise |
| fuzzy tree / library default | RMSE log+std (MPa) | 9.085 ± 0.578 | 9.066 ± 0.575 | -0.0190 | within noise |
| mixture of experts / demo-tuned / 0.768 ± 0.029 | log + standardized | 0.833 ± 0.024 | 0.834 ± 0.025 | +0.0010 | within noise |
| mixture of experts / demo-tuned / 0.768 ± 0.029 | RMSE log+std (MPa) | 6.669 ± 0.562 | 6.661 ± 0.570 | -0.0080 | within noise |
| mixture of experts / library default | raw features | 0.682 ± 0.064 | 0.686 ± 0.060 | +0.0040 | within noise |
| mixture of experts / library default | log + standardized | 0.781 ± 0.068 | 0.763 ± 0.057 | -0.0180 | within noise |
| mixture of experts / library default | RMSE raw (MPa) | 9.167 ± 0.792 | 9.114 ± 0.738 | -0.0530 | within noise |
| mixture of experts / library default | RMSE log+std (MPa) | 7.542 ± 0.976 | 7.905 ± 0.932 | +0.3630 | within noise |

### `table_norm_conorm_matrix`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete / fuzzy tree (t-norm only) / RMSE (MPa) | Best (mean spread) | **luk** (spread 0.078) | **luk** (spread 0.074) | -0.0040 | **changed** |
| Concrete / fuzzy tree (t-norm only) / R2 | min/max | 0.709 ± 0.029 | 0.708 ± 0.030 | -0.0010 | within noise |
| Concrete / fuzzy tree (t-norm only) / R2 | probability | 0.714 ± 0.029 | 0.712 ± 0.030 | -0.0020 | within noise |
| Concrete / fuzzy tree (t-norm only) / R2 | luk | 0.714 ± 0.032 | 0.713 ± 0.033 | -0.0010 | within noise |
| Concrete / fuzzy tree (t-norm only) / R2 | hamacher | 0.713 ± 0.029 | 0.712 ± 0.030 | -0.0010 | within noise |
| Concrete / fuzzy tree (t-norm only) / RMSE (MPa) | min/max | 8.818 ± 0.464 | 8.829 ± 0.469 | +0.0110 | within noise |
| Concrete / fuzzy tree (t-norm only) / RMSE (MPa) | probability | 8.749 ± 0.513 | 8.764 ± 0.521 | +0.0150 | within noise |
| Concrete / fuzzy tree (t-norm only) / RMSE (MPa) | luk | 8.740 ± 0.565 | 8.755 ± 0.572 | +0.0150 | within noise |
| Concrete / fuzzy tree (t-norm only) / RMSE (MPa) | hamacher | 8.751 ± 0.482 | 8.770 ± 0.492 | +0.0190 | within noise |
| Concrete / fuzzy tree (t-norm only) / RMSE (MPa) | einstein | 8.748 ± 0.533 | 8.760 ± 0.539 | +0.0120 | within noise |

## Bit-identical

These tables produced exactly the same numbers on both sides:

- `table_3_2_memory_precision` (32 cells)
- `table_4_4_openset` (9 cells)
- `table_4_4b_theta_sweep` (28 cells)
- `table_5_1_battery` (34 cells)
- `table_5_2_multiscale` (15 cells)
- `table_5_3_selection` (15 cells)
- `table_a1_feature_ranking` (20 cells)
- `table_g5_output_partitioning` (126 cells)
- `table_g5b_skew_sweep` (48 cells)

---

> A cell counts as **changed** only if it moved by more than the larger of the two runs' reported standard deviations; smaller moves are labelled *within noise*. Wall-clock columns are always reported separately and never called a regression — this harness does not control clocks or thermals (see G4 in `NEXT_STEPS.md`).
