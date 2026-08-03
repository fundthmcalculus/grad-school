# Fix impact — `full-14900hx-r2` → `norm-migration-a385a1a`

Cell-by-cell diff of the archived table runs, produced by `reproduce/compare_runs.py`. Every table is listed, including the unchanged ones: confining a fix's blast radius is a claim, and it is only supported by showing the tables that did *not* move.

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

<details><summary>Provenance — <code>norm-migration-a385a1a</code></summary>

```

--- backfill 2026-08-03T04:17:03Z: table_concrete_reconciliation table_hyperparam_normalization table_g5_output_partitioning table_g5b_skew_sweep table_4_1_mog_baselines ---
tribble-fis: a385a1ab3df606af23e46bae75cc56d41c8bb744
tribble-cluster: e3c27e67ae2a41d636dfb472110ae2ded2e4ef82
grad-school: 9a7e4bd7c2fe6653ed32f548c78c2917cd399a23
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
  table_concrete_reconciliation          ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_hyperparam_normalization         ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_g5_output_partitioning           ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_g5b_skew_sweep                   ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_4_1_mog_baselines                ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_6_1_model_family                 not-run-this-pass seeds=—
  table_norm_conorm_matrix               not-run-this-pass seeds=—
  table_4_4_openset                      not-run-this-pass seeds=—
  table_a1_feature_scoring               not-run-this-pass seeds=—
  table_3_1_pvat_scaling                 not-run-this-pass seeds=—
  table_3_1_reorder_three_arm            not-run-this-pass seeds=—
  table_3_2_memory_precision             not-run-this-pass seeds=—
  table_3_4_gpu_speedups                 not-run-this-pass seeds=—
  table_5_x_ch5_selection                not-run-this-pass seeds=—
```

</details>

## Summary

| Table | Cells | Verdict |
|---|---:|---|
| `table_3_1` | — | **baseline-only** |
| `table_3_1_complexity_fit` | — | **baseline-only** |
| `table_3_1_three_arm` | — | **baseline-only** |
| `table_3_2_memory_precision` | — | **baseline-only** |
| `table_4_1` | 7 | **3 changed**, 2 timing, 1 rows added |
| `table_4_4_openset` | — | **baseline-only** |
| `table_4_4b_theta_sweep` | — | **baseline-only** |
| `table_5_1_battery` | — | **baseline-only** |
| `table_5_2_multiscale` | — | **baseline-only** |
| `table_5_3_selection` | — | **baseline-only** |
| `table_6_1` | — | **baseline-only** |
| `table_a1_feature_ranking` | — | **baseline-only** |
| `table_a2_feature_count` | — | **baseline-only** |
| `table_concrete_reconciliation` | 34 | identical |
| `table_g5_output_partitioning` | 126 | identical |
| `table_g5b_skew_sweep` | 48 | identical |
| `table_hyperparam_normalization` | 48 | identical |
| `table_norm_conorm_matrix` | — | **baseline-only** |

## Tables that could not be compared

- `table_3_1` — **baseline-only**
- `table_3_1_complexity_fit` — **baseline-only**
- `table_3_1_three_arm` — **baseline-only**
- `table_3_2_memory_precision` — **baseline-only**
- `table_4_4_openset` — **baseline-only**
- `table_4_4b_theta_sweep` — **baseline-only**
- `table_5_1_battery` — **baseline-only**
- `table_5_2_multiscale` — **baseline-only**
- `table_5_3_selection` — **baseline-only**
- `table_6_1` — **baseline-only**
- `table_a1_feature_ranking` — **baseline-only**
- `table_a2_feature_count` — **baseline-only**
- `table_norm_conorm_matrix` — **baseline-only**

## What moved

### `table_4_1`

Rows only in `norm-migration-a385a1a`: `PhiUSIIL (classification) / 0.64 ± 0.02 s / acc=0.997 ± 0.001`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| PhiUSIIL (classification) | Dataset (task) | PhiUSIIL (classification) | Concrete (regression, full 2nd order) |  | **changed** |
| PhiUSIIL (classification) | MoG accuracy / R2 | acc=0.997 ± 0.001 | R2=0.840 ± 0.049 | +1.0030 | **changed** |
| PhiUSIIL (classification) | tree / RF ref | 1.000 ± 0.000 | 0.909 ± 0.019 | -0.0910 | **changed** |
| Concrete (regression) | MoG train time | 1.04 ± 0.62 s | 0.83 ± 0.02 s | -0.2100 | timing |
| PhiUSIIL (classification) | MoG train time | 0.64 ± 0.02 s | 0.84 ± 0.01 s | +0.2000 | timing |

## Bit-identical

These tables produced exactly the same numbers on both sides:

- `table_concrete_reconciliation` (34 cells)
- `table_g5_output_partitioning` (126 cells)
- `table_g5b_skew_sweep` (48 cells)
- `table_hyperparam_normalization` (48 cells)

---

> A cell counts as **changed** only if it moved by more than the larger of the two runs' reported standard deviations; smaller moves are labelled *within noise*. Wall-clock columns are always reported separately and never called a regression — this harness does not control clocks or thermals (see G4 in `NEXT_STEPS.md`).
