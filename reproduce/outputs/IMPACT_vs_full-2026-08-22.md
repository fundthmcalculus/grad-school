# Fix impact — `full-2026-08-22` → `bumped-0764bc5-2026-08-22`

Cell-by-cell diff of the archived table runs, produced by `reproduce/compare_runs.py`. Every table is listed, including the unchanged ones: confining a fix's blast radius is a claim, and it is only supported by showing the tables that did *not* move.

<details><summary>Provenance — <code>full-2026-08-22</code></summary>

```

--- backfill 2026-08-22T07:56:10Z: table_concrete_reconciliation table_hyperparam_normalization table_g5_output_partitioning table_g5b_skew_sweep table_4_1_mog_baselines table_6_1_model_family table_norm_conorm_matrix table_a1_feature_scoring table_4_8_mf_dedup table_3_1_pvat_scaling table_3_1_reorder_three_arm table_3_2_memory_precision table_3_4_gpu_speedups table_3_7_g2_dtw_nonmetric table_5_x_ch5_selection ---
tribble-fis: 141596e9c88710f78f8eb8b55d073573535f5f0e
tribble-cluster: 635ed6ed713298b9823ff226de533e68a8917c1b
grad-school: 851b88b33e7b0091f76b66cf3a83f7fadc0fc8a6
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
  table_6_1_model_family                 ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_norm_conorm_matrix               ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_4_4_openset                      not-run-this-pass seeds=—
  table_a1_feature_scoring               ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_4_8_mf_dedup                     ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_1_pvat_scaling                 ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_1_reorder_three_arm            ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_2_memory_precision             ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_4_gpu_speedups                 ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_7_g2_dtw_nonmetric             ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_5_x_ch5_selection                ok           seeds=0,1,2,3,4,5,6,7,8,9

--- backfill 2026-08-22T11:44:05Z: table_4_4_openset ---
tribble-fis: 141596e9c88710f78f8eb8b55d073573535f5f0e
tribble-cluster: 635ed6ed713298b9823ff226de533e68a8917c1b
grad-school: d235e28dacb0335944100a7dc6d41491145e3e88
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
  table_4_4_openset                      ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_a1_feature_scoring               not-run-this-pass seeds=—
  table_4_8_mf_dedup                     not-run-this-pass seeds=—
  table_3_1_pvat_scaling                 not-run-this-pass seeds=—
  table_3_1_reorder_three_arm            not-run-this-pass seeds=—
  table_3_2_memory_precision             not-run-this-pass seeds=—
  table_3_4_gpu_speedups                 not-run-this-pass seeds=—
  table_3_7_g2_dtw_nonmetric             not-run-this-pass seeds=—
  table_5_x_ch5_selection                not-run-this-pass seeds=—

--- backfill 2026-08-22T11:55:32Z: table_3_7_g2_downstream ---
tribble-fis: 141596e9c88710f78f8eb8b55d073573535f5f0e
tribble-cluster: 635ed6ed713298b9823ff226de533e68a8917c1b
grad-school: 4749697c7d2442b80f57049c96d207571fcd5d2a
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
  table_4_8_mf_dedup                     not-run-this-pass seeds=—
  table_3_1_pvat_scaling                 not-run-this-pass seeds=—
  table_3_1_reorder_three_arm            not-run-this-pass seeds=—
  table_3_2_memory_precision             not-run-this-pass seeds=—
  table_3_4_gpu_speedups                 not-run-this-pass seeds=—
  table_3_7_g2_dtw_nonmetric             not-run-this-pass seeds=—
  table_3_7_g2_downstream                ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_5_x_ch5_selection                not-run-this-pass seeds=—
```

</details>

<details><summary>Provenance — <code>bumped-0764bc5-2026-08-22</code></summary>

```
label:       bumped-0764bc5-2026-08-22
generated:   2026-08-22T23:56:01Z
tribble-fis: 0764bc5f0485aeac401ccc80379700ea9a38e491
tribble-cluster: 635ed6ed713298b9823ff226de533e68a8917c1b
grad-school: c3f52499fc33f214f91b3ddb6b1144a5db184ea2
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
  table_concrete_reconciliation          ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_hyperparam_normalization         ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_g5_output_partitioning           ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_g5b_skew_sweep                   ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_4_1_mog_baselines                ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=10
  table_6_1_model_family                 ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=3
  table_norm_conorm_matrix               ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_4_4_openset                      ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_a1_feature_scoring               ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_4_8_mf_dedup                     ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_3_1_pvat_scaling                 ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=2
  table_3_1_reorder_three_arm            ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=10
  table_3_2_memory_precision             ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_3_4_gpu_speedups                 ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_3_7_g2_dtw_nonmetric             ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_3_7_g2_downstream                ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=0
  table_5_x_ch5_selection                ok           seeds=0,1,2,3,4,5,6,7,8,9  N/A cells=6
```

</details>

## Summary

| Table | Cells | Verdict |
|---|---:|---|
| `table_3_1` | 16 | 10 timing |
| `table_3_1_complexity_fit` | 89 | **3 changed**, 25 timing |
| `table_3_1_three_arm` | 56 | 32 timing |
| `table_3_2_memory_precision` | 26 | identical |
| `table_3_4_gpu_speedups` | 195 | **4 changed**, 58 within noise, 27 timing |
| `table_3_7_g2_downstream` | 7 | identical |
| `table_3_7_g2_dtw_nonmetric` | 5 | **1 changed** |
| `table_4_1` | 17 | **3 changed**, 4 timing |
| `table_4_4_openset` | 9 | **4 changed**, 5 within noise |
| `table_4_4b_theta_sweep` | 28 | **17 changed** |
| `table_4_8_mf_dedup` | 36 | **24 changed**, 1 within noise, 5 timing |
| `table_4_8_mf_dedup_sweep` | 336 | **123 changed**, 77 within noise |
| `table_4_9_correction_pass` | 8 | **1 changed**, 7 within noise |
| `table_5_1_battery` | 34 | identical |
| `table_5_2_multiscale` | 15 | identical |
| `table_5_3_selection` | 15 | identical |
| `table_5_4_ch5_g1_scaling` | 126 | identical |
| `table_5_4_ch5_g1_scaling_raw` | 1800 | identical |
| `table_6_1` | 15 | **5 changed**, 4 within noise |
| `table_a1_feature_ranking` | 20 | **8 changed**, 2 within noise |
| `table_a2_feature_count` | 36 | **18 changed**, 9 within noise |
| `table_a7_regression_scale` | 30 | identical |
| `table_concrete_reconciliation` | 34 | 12 within noise |
| `table_g5_output_partitioning` | 189 | identical |
| `table_g5b_skew_sweep` | 48 | identical |
| `table_hyperparam_normalization` | 84 | **9 changed**, 8 within noise |
| `table_norm_conorm_matrix` | 54 | **36 changed**, 6 within noise |

## What moved

### `table_3_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,024 | classical VAT (s) | 13.387 ± 0.234 s | 13.459 ± 0.286 s | +0.0720 | timing |
| 1,024 | pVAT (s) | 0.022 ± 0.003 s | 0.021 ± 0.002 s | -0.0010 | timing |
| 1,024 | speedup | 604x | 632x | +28.0000 | timing |
| 2,048 / infeasible (>cap) | pVAT (s) | 0.076 ± 0.007 s | 0.076 ± 0.008 s | +0.0000 | timing |
| 256 | classical VAT (s) | 0.264 ± 0.012 s | 0.262 ± 0.015 s | -0.0020 | timing |
| 256 | pVAT (s) | 0.012 ± 0.031 s | 0.013 ± 0.031 s | +0.0010 | timing |
| 4,096 / infeasible (>cap) | pVAT (s) | 0.229 ± 0.007 s | 0.234 ± 0.009 s | +0.0050 | timing |
| 512 | classical VAT (s) | 1.824 ± 0.048 s | 1.796 ± 0.042 s | -0.0280 | timing |
| 512 | pVAT (s) | 0.007 ± 0.001 s | 0.006 ± 0.001 s | -0.0010 | timing |
| 512 | speedup | 275x | 291x | +16.0000 | timing |

### `table_3_1_complexity_fit`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| fitted exponent /  | classical | **3.11** (6 pts) | **3.05** (6 pts) | -0.0600 | **changed** |
| fitted exponent /  | stage 1 | **1.82** (11 pts) | **1.77** (11 pts) | -0.0500 | **changed** |
| fitted exponent /  | stage 2 | **1.77** (11 pts) | **1.73** (11 pts) | -0.0400 | **changed** |
| 1,000 / 10.0× | classical | 1269.20× | 1130.26× | -138.9400 | timing |
| 1,000 / 10.0× | stage 1 | 57.76× | 50.58× | -7.1800 | timing |
| 1,000 / 10.0× | stage 2 | 61.41× | 53.55× | -7.8600 | timing |
| 1,250 / 12.5× / N/A | stage 1 | 98.24× | 78.53× | -19.7100 | timing |
| 1,250 / 12.5× / N/A | stage 2 | 96.94× | 77.15× | -19.7900 | timing |
| 1,500 / 15.0× / N/A | stage 1 | 128.30× | 110.52× | -17.7800 | timing |
| 1,500 / 15.0× / N/A | stage 2 | 108.12× | 107.12× | -1.0000 | timing |
| 2,000 / 20.0× / N/A | stage 1 | 215.90× | 191.27× | -24.6300 | timing |
| 2,000 / 20.0× / N/A | stage 2 | 191.03× | 176.34× | -14.6900 | timing |
| 2,500 / 25.0× / N/A | stage 1 | 328.23× | 293.12× | -35.1100 | timing |
| 2,500 / 25.0× / N/A | stage 2 | 316.37× | 287.42× | -28.9500 | timing |
| 200 / 2.0× | classical | 7.51× | 7.53× | +0.0200 | timing |
| 200 / 2.0× | stage 1 | 3.23× | 3.17× | -0.0600 | timing |
| 200 / 2.0× | stage 2 | 3.61× | 3.62× | +0.0100 | timing |
| 3,000 / 30.0× / N/A | stage 1 | 459.00× | 409.92× | -49.0800 | timing |
| 3,000 / 30.0× / N/A | stage 2 | 462.68× | 410.72× | -51.9600 | timing |
| 300 / 3.0× | classical | 26.45× | 26.21× | -0.2400 | timing |
| 300 / 3.0× | stage 1 | 6.17× | 6.18× | +0.0100 | timing |
| 300 / 3.0× | stage 2 | 7.42× | 7.50× | +0.0800 | timing |
| 500 / 5.0× | classical | 132.82× | 124.93× | -7.8900 | timing |
| 500 / 5.0× | stage 1 | 16.81× | 15.84× | -0.9700 | timing |
| 500 / 5.0× | stage 2 | 20.33× | 19.49× | -0.8400 | timing |
| 750 / 7.5× | classical | 484.06× | 435.47× | -48.5900 | timing |
| 750 / 7.5× | stage 1 | 34.53× | 29.90× | -4.6300 | timing |
| 750 / 7.5× | stage 2 | 42.67× | 38.57× | -4.1000 | timing |

### `table_3_1_three_arm`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,000 | classical O(N³) (s) | 0.1317 ± 0.0035 s | 0.1300 ± 0.0016 s | -0.0017 | timing |
| 1,000 | stage 1 O(N²logN) (s) | 0.0089 ± 0.0003 s | 0.0087 ± 0.0001 s | -0.0002 | timing |
| 1,000 | cls/s2 | 131.0× | 136.0× | +5.0000 | timing |
| 1,000 | s1/s2 | 8.9× | 9.1× | +0.2000 | timing |
| 1,250 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0152 ± 0.0026 s | 0.0135 ± 0.0005 s | -0.0017 | timing |
| 1,250 / not run (> cap) | stage 2 O(N²) (s) | 0.0016 ± 0.0002 s | 0.0014 ± 0.0002 s | -0.0002 | timing |
| 1,250 / not run (> cap) | s1/s2 | 9.5× | 9.8× | +0.3000 | timing |
| 1,500 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0198 ± 0.0006 s | 0.0190 ± 0.0006 s | -0.0008 | timing |
| 1,500 / not run (> cap) | stage 2 O(N²) (s) | 0.0018 ± 0.0002 s | 0.0019 ± 0.0002 s | +0.0001 | timing |
| 1,500 / not run (> cap) | s1/s2 | 11.2× | 9.9× | -1.3000 | timing |
| 100 / 0.0001 ± 0.0000 s / 0.0002 ± 0.0000 s | cls/s2 | 6.3× | 6.4× | +0.1000 | timing |
| 100 / 0.0001 ± 0.0000 s / 0.0002 ± 0.0000 s | s1/s2 | 9.4× | 9.6× | +0.2000 | timing |
| 2,000 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0333 ± 0.0007 s | 0.0329 ± 0.0008 s | -0.0004 | timing |
| 2,000 / not run (> cap) | s1/s2 | 10.6× | 10.4× | -0.2000 | timing |
| 2,500 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0506 ± 0.0010 s | 0.0503 ± 0.0012 s | -0.0003 | timing |
| 2,500 / not run (> cap) | stage 2 O(N²) (s) | 0.0052 ± 0.0003 s | 0.0051 ± 0.0002 s | -0.0001 | timing |
| 200 | classical O(N³) (s) | 0.0008 ± 0.0000 s | 0.0009 ± 0.0000 s | +0.0001 | timing |
| 200 | cls/s2 | 13.2× | 13.4× | +0.2000 | timing |
| 3,000 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0708 ± 0.0015 s | 0.0704 ± 0.0016 s | -0.0004 | timing |
| 3,000 / not run (> cap) | stage 2 O(N²) (s) | 0.0076 ± 0.0004 s | 0.0073 ± 0.0003 s | -0.0003 | timing |
| 3,000 / not run (> cap) | s1/s2 | 9.3× | 9.6× | +0.3000 | timing |
| 300 | classical O(N³) (s) | 0.0027 ± 0.0001 s | 0.0030 ± 0.0001 s | +0.0003 | timing |
| 300 | stage 1 O(N²logN) (s) | 0.0010 ± 0.0001 s | 0.0011 ± 0.0000 s | +0.0001 | timing |
| 300 | cls/s2 | 22.6× | 22.5× | -0.1000 | timing |
| 300 | s1/s2 | 7.8× | 7.9× | +0.1000 | timing |
| 500 | classical O(N³) (s) | 0.0138 ± 0.0003 s | 0.0144 ± 0.0005 s | +0.0006 | timing |
| 500 | stage 1 O(N²logN) (s) | 0.0026 ± 0.0001 s | 0.0027 ± 0.0001 s | +0.0001 | timing |
| 500 | cls/s2 | 41.4× | 41.3× | -0.1000 | timing |
| 750 | classical O(N³) (s) | 0.0502 ± 0.0012 s | 0.0501 ± 0.0011 s | -0.0001 | timing |
| 750 | stage 1 O(N²logN) (s) | 0.0053 ± 0.0002 s | 0.0051 ± 0.0001 s | -0.0002 | timing |
| 750 | cls/s2 | 71.9× | 72.7× | +0.8000 | timing |
| 750 | s1/s2 | 7.6× | 7.5× | -0.1000 | timing |

### `table_3_4_gpu_speedups`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Pairwise distances / N=16,000, d=784, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 2.39694 ± 0.04507 | 2.48170 ± 0.03023 | +0.0848 | **changed** |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 5.09463 ± 0.04649 | 5.39620 ± 0.04986 | +0.3016 | **changed** |
| VAT front end / N=48,000, float32, 9.22 GB resident / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 4.39892 | 4.11908 | -0.2798 | **changed** |
| VAT front end / N=48,000, float32, 9.22 GB resident / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 1.36464 | 1.28861 | -0.0760 | **changed** |
| Boruvka MST (device) / N=16,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.21282 ± 0.01261 | 0.21748 ± 0.01485 | +0.0047 | within noise |
| Boruvka MST (device) / N=16,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.04792 ± 0.00386 | 0.04786 ± 0.00384 | -0.0001 | within noise |
| Boruvka MST (device) / N=16,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 4.44x | 4.54x | +0.1000 | timing |
| Boruvka MST (device) / N=32,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.82615 ± 0.04086 | 0.81405 ± 0.04405 | -0.0121 | within noise |
| Boruvka MST (device) / N=32,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.19872 ± 0.01434 | 0.19789 ± 0.01252 | -0.0008 | within noise |
| Boruvka MST (device) / N=32,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 4.16x | 4.11x | -0.0500 | timing |
| Boruvka MST (device) / N=4,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.01453 ± 0.00123 | 0.01551 ± 0.00361 | +0.0010 | within noise |
| Boruvka MST (device) / N=4,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.00351 ± 0.00019 | 0.00354 ± 0.00038 | +0.0000 | within noise |
| Boruvka MST (device) / N=4,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 4.14x | 4.38x | +0.2400 | timing |
| Boruvka MST (device) / N=8,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.06229 ± 0.01309 | 0.05906 ± 0.01366 | -0.0032 | within noise |
| Boruvka MST (device) / N=8,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.01191 ± 0.00157 | 0.01146 ± 0.00182 | -0.0005 | within noise |
| Boruvka MST (device) / N=8,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 5.23x | 5.15x | -0.0800 | timing |
| Fuzzy C-Means / N=200,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | CPU (s) | 1.93163 ± 1.94163 | 1.92467 ± 1.89777 | -0.0070 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU (s) | 0.32031 ± 0.22633 | 0.33345 ± 0.36299 | +0.0131 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU speedup (CPU/GPU) | 6.03x | 5.77x | -0.2600 | timing |
| Fuzzy C-Means / N=200,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | CPU (s) | 0.96767 ± 0.94794 | 0.99717 ± 1.01518 | +0.0295 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU (s) | 0.32031 ± 0.22633 | 0.33345 ± 0.36299 | +0.0131 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU speedup (CPU/GPU) | 3.02x | 2.99x | -0.0300 | timing |
| Fuzzy C-Means / N=50,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | CPU (s) | 0.40936 ± 0.35122 | 0.40631 ± 0.35625 | -0.0030 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU (s) | 0.10211 ± 0.10776 | 0.09651 ± 0.09645 | -0.0056 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU speedup (CPU/GPU) | 4.01x | 4.21x | +0.2000 | timing |
| Fuzzy C-Means / N=50,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | CPU (s) | 0.20753 ± 0.17465 | 0.22833 ± 0.23212 | +0.0208 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU (s) | 0.10211 ± 0.10776 | 0.09651 ± 0.09645 | -0.0056 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU speedup (CPU/GPU) | 2.03x | 2.37x | +0.3400 | timing |
| Fuzzy C-Means / N=500,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | CPU (s) | 5.38160 ± 4.82153 | 5.41501 ± 4.82429 | +0.0334 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU (s) | 0.96259 ± 0.91236 | 0.80872 ± 0.41778 | -0.1539 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU speedup (CPU/GPU) | 5.59x | 6.70x | +1.1100 | timing |
| Fuzzy C-Means / N=500,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | CPU (s) | 2.74843 ± 2.35473 | 2.79264 ± 2.56297 | +0.0442 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU (s) | 0.96259 ± 0.91236 | 0.80872 ± 0.41778 | -0.1539 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU speedup (CPU/GPU) | 2.86x | 3.45x | +0.5900 | timing |
| Pairwise distances / N=16,000, d=10, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.18541 ± 0.01037 | 0.18365 ± 0.00861 | -0.0018 | within noise |
| Pairwise distances / N=16,000, d=10, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.53520 ± 0.04155 | 0.54584 ± 0.04957 | +0.0106 | within noise |
| Pairwise distances / N=16,000, d=10, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.35x | 0.34x | -0.0100 | timing |
| Pairwise distances / N=16,000, d=10, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.18467 ± 0.01001 | 0.18337 ± 0.01104 | -0.0013 | within noise |
| Pairwise distances / N=16,000, d=10, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.64196 ± 0.07358 | 0.61367 ± 0.05869 | -0.0283 | within noise |
| Pairwise distances / N=16,000, d=10, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.29x | 0.30x | +0.0100 | timing |
| Pairwise distances / N=16,000, d=10, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.27990 ± 0.01107 | 0.28500 ± 0.01531 | +0.0051 | within noise |
| Pairwise distances / N=16,000, d=10, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.95681 ± 0.08161 | 0.98285 ± 0.04709 | +0.0260 | within noise |
| Pairwise distances / N=16,000, d=200, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.65316 ± 0.03561 | 0.65888 ± 0.02547 | +0.0057 | within noise |
| Pairwise distances / N=16,000, d=200, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.88349 ± 0.03599 | 0.86260 ± 0.03872 | -0.0209 | within noise |
| Pairwise distances / N=16,000, d=200, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.74x | 0.76x | +0.0200 | timing |
| Pairwise distances / N=16,000, d=200, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.65686 ± 0.01877 | 0.65777 ± 0.03646 | +0.0009 | within noise |
| Pairwise distances / N=16,000, d=200, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.64433 ± 0.04044 | 1.62519 ± 0.05426 | -0.0191 | within noise |
| Pairwise distances / N=16,000, d=200, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.84519 ± 0.04621 | 0.85081 ± 0.06486 | +0.0056 | within noise |
| Pairwise distances / N=16,000, d=200, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.66698 ± 0.03414 | 1.65933 ± 0.04905 | -0.0076 | within noise |
| Pairwise distances / N=16,000, d=50, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.25823 ± 0.01437 | 0.25630 ± 0.01276 | -0.0019 | within noise |
| Pairwise distances / N=16,000, d=50, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.62895 ± 0.02592 | 0.64119 ± 0.02813 | +0.0122 | within noise |
| Pairwise distances / N=16,000, d=50, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.41x | 0.40x | -0.0100 | timing |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.26561 ± 0.00892 | 0.26124 ± 0.00616 | -0.0044 | within noise |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.88194 ± 0.02942 | 0.89946 ± 0.02067 | +0.0175 | within noise |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.30x | 0.29x | -0.0100 | timing |
| Pairwise distances / N=16,000, d=50, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.34170 ± 0.01724 | 0.34326 ± 0.01206 | +0.0016 | within noise |
| Pairwise distances / N=16,000, d=50, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.14066 ± 0.03591 | 1.14775 ± 0.02017 | +0.0071 | within noise |
| Pairwise distances / N=16,000, d=784, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 2.77787 ± 0.17178 | 2.77995 ± 0.13526 | +0.0021 | within noise |
| Pairwise distances / N=16,000, d=784, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 1.16x | 1.12x | -0.0400 | timing |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 2.68372 ± 0.07645 | 2.80409 ± 0.17184 | +0.1204 | within noise |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.53x | 0.52x | -0.0100 | timing |
| Pairwise distances / N=16,000, d=784, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 4.81571 ± 0.15468 | 4.74440 ± 0.16858 | -0.0713 | within noise |
| Pairwise distances / N=16,000, d=784, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 13.19158 ± 0.09859 | 13.11990 ± 0.07710 | -0.0717 | within noise |
| Pairwise distances / N=16,000, d=784, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.37x | 0.36x | -0.0100 | timing |
| VAT front end / N=16,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 1.24742 ± 0.03071 | 1.25109 ± 0.02691 | +0.0037 | within noise |
| VAT front end / N=16,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.16774 ± 0.01204 | 0.16483 ± 0.00904 | -0.0029 | within noise |
| VAT front end / N=16,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 7.44x | 7.59x | +0.1500 | timing |
| VAT front end / N=16,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 0.50816 ± 0.02025 | 0.50395 ± 0.02492 | -0.0042 | within noise |
| VAT front end / N=16,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.16774 ± 0.01204 | 0.16483 ± 0.00904 | -0.0029 | within noise |
| VAT front end / N=16,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 3.03x | 3.06x | +0.0300 | timing |
| VAT front end / N=32,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 5.80313 ± 0.13967 | 5.95395 ± 0.48169 | +0.1508 | within noise |
| VAT front end / N=32,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.62908 ± 0.03835 | 0.61254 ± 0.01343 | -0.0165 | within noise |
| VAT front end / N=32,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 9.22x | 9.72x | +0.5000 | timing |
| VAT front end / N=32,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 2.43062 ± 0.05290 | 2.40970 ± 0.05357 | -0.0209 | within noise |
| VAT front end / N=32,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.62908 ± 0.03835 | 0.61254 ± 0.01343 | -0.0165 | within noise |
| VAT front end / N=32,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 3.86x | 3.93x | +0.0700 | timing |
| VAT front end / N=4,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 0.08745 ± 0.00560 | 0.08562 ± 0.00776 | -0.0018 | within noise |
| VAT front end / N=4,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.01573 ± 0.00425 | 0.01547 ± 0.00366 | -0.0003 | within noise |
| VAT front end / N=4,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 5.56x | 5.54x | -0.0200 | timing |
| VAT front end / N=4,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 0.03532 ± 0.00199 | 0.03678 ± 0.00461 | +0.0015 | within noise |
| VAT front end / N=4,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.01573 ± 0.00425 | 0.01547 ± 0.00366 | -0.0003 | within noise |
| VAT front end / N=4,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 2.25x | 2.38x | +0.1300 | timing |
| VAT front end / N=48,000, float32, 9.22 GB resident / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 3.22x | 3.20x | -0.0200 | timing |
| VAT front end / N=8,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 0.30113 ± 0.01017 | 0.31029 ± 0.02254 | +0.0092 | within noise |
| VAT front end / N=8,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.04601 ± 0.00911 | 0.04194 ± 0.00435 | -0.0041 | within noise |
| VAT front end / N=8,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 6.54x | 7.40x | +0.8600 | timing |
| VAT front end / N=8,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 0.13099 ± 0.01397 | 0.12761 ± 0.01310 | -0.0034 | within noise |
| VAT front end / N=8,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.04601 ± 0.00911 | 0.04194 ± 0.00435 | -0.0041 | within noise |
| VAT front end / N=8,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 2.85x | 3.04x | +0.1900 | timing |

### `table_3_7_g2_dtw_nonmetric`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| ECG5000 (DTW, N=5000) / no / 20.9% | Timing | 600s matrix + 0.2s reorder | 593s matrix + 0.2s reorder | -7.0000 | **changed** |

### `table_4_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Bike Sharing (regression) | MoG accuracy / R2 | R2=0.965 ± 0.001 | R2=0.960 ± 0.003 | -0.0050 | **changed** |
| PhiUSIIL (classification) | MoG accuracy / R2 | acc=0.729 ± 0.023 | acc=0.997 ± 0.001 | +0.2680 | **changed** |
| RT-IOT2022 (12-class) | MoG accuracy / R2 | acc=0.500 ± 0.244 | acc=0.923 ± 0.011 | +0.4230 | **changed** |
| Bike Sharing (regression) | MoG train time | 0.11 ± 0.00 s | 0.17 ± 0.01 s | +0.0600 | timing |
| Concrete (regression, full 2nd order) | MoG train time | 0.07 ± 0.00 s | 0.07 ± 0.01 s | +0.0000 | timing |
| PhiUSIIL (classification) | MoG train time | 0.17 ± 0.01 s | 0.07 ± 0.00 s | -0.1000 | timing |
| RT-IOT2022 (12-class) | MoG train time | 33.22 ± 0.12 s | 2.88 ± 0.10 s | -30.3400 | timing |

### `table_4_4_openset`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Complement rule (this work) | Detection − false alarm | +0.515 | +0.367 | -0.1480 | **changed** |
| Isolation Forest | False-alarm rate | 0.279 ± 0.146 | 0.429 ± 0.070 | +0.1500 | **changed** |
| Isolation Forest | Detection − false alarm | +0.579 | +0.534 | -0.0450 | **changed** |
| One-class SVM | Detection − false alarm | +0.271 | +0.416 | +0.1450 | **changed** |
| Complement rule (this work) | Detection rate | 0.803 ± 0.272 | 0.798 ± 0.273 | -0.0050 | within noise |
| Complement rule (this work) | False-alarm rate | 0.287 ± 0.171 | 0.431 ± 0.074 | +0.1440 | within noise |
| Isolation Forest | Detection rate | 0.858 ± 0.306 | 0.962 ± 0.145 | +0.1040 | within noise |
| One-class SVM | Detection rate | 0.584 ± 0.393 | 0.848 ± 0.231 | +0.2640 | within noise |
| One-class SVM | False-alarm rate | 0.313 ± 0.127 | 0.432 ± 0.065 | +0.1190 | within noise |

### `table_4_4b_theta_sweep`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 0.500 | detection rate | 0.995 | 0.959 | -0.0360 | **changed** |
| 0.500 | false-alarm rate | 0.800 | 0.672 | -0.1280 | **changed** |
| 0.500 | detection − false alarm | +0.195 | +0.287 | +0.0920 | **changed** |
| 0.600 | detection rate | 0.955 | 0.945 | -0.0100 | **changed** |
| 0.600 | false-alarm rate | 0.739 | 0.609 | -0.1300 | **changed** |
| 0.600 | detection − false alarm | +0.216 | +0.336 | +0.1200 | **changed** |
| 0.700 | detection rate | 0.947 | 0.935 | -0.0120 | **changed** |
| 0.700 | false-alarm rate | 0.670 | 0.569 | -0.1010 | **changed** |
| 0.700 | detection − false alarm | +0.277 | +0.366 | +0.0890 | **changed** |
| 0.800 | detection rate | 0.923 | 0.908 | -0.0150 | **changed** |
| 0.800 | false-alarm rate | 0.603 | 0.515 | -0.0880 | **changed** |
| 0.800 | detection − false alarm | +0.320 | +0.393 | +0.0730 | **changed** |
| 0.900 | detection rate | 0.851 | 0.836 | -0.0150 | **changed** |
| 0.900 | false-alarm rate | 0.552 | 0.465 | -0.0870 | **changed** |
| 0.900 | detection − false alarm | +0.298 | +0.370 | +0.0720 | **changed** |
| 0.990 / 0.777 | false-alarm rate | 0.305 | 0.431 | +0.1260 | **changed** |
| 0.990 / 0.777 | detection − false alarm | +0.472 | +0.346 | -0.1260 | **changed** |

### `table_4_8_mf_dedup`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| BreastCancer / classification | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification | MF @ 1x (Δ) | 15.3 (-0.0088 ± 0.0316 acc) | 11.0 (+0.0000 ± 0.0000 acc) | -4.3000 | **changed** |
| BreastCancer / classification | Reduction @ 1x | 23.1% | 0.0% | -23.1000 | **changed** |
| BreastCancer / classification | MF @ max-lossless (Δ) | 10.5 (-0.0018 ± 0.0217 acc) | 11.0 (+0.0000 ± 0.0000 acc) | +0.5000 | **changed** |
| BreastCancer / classification | Reduction @ max-lossless | 47.2% | 0.0% | -47.2000 | **changed** |
| Concrete / regression | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression | MF @ 1x (Δ) | 33.1 (-0.0000 ± 0.0000 R²) | 64.5 (+0.0002 ± 0.0005 R²) | +31.4000 | **changed** |
| Concrete / regression | Reduction @ 1x | 2.4% | 3.9% | +1.5000 | **changed** |
| Concrete / regression | MF @ max-lossless (Δ) | 32.0 (-0.0002 ± 0.0022 R²) | 61.9 (-0.0056 ± 0.0116 R²) | +29.9000 | **changed** |
| Concrete / regression | Reduction @ max-lossless | 5.6% | 7.7% | +2.1000 | **changed** |
| Diabetes / regression / 40.6 | MF @ max-lossless (Δ) | 14.4 (-0.3197 ± 0.7944 R²) | 14.1 (-0.3171 ± 0.7967 R²) | -0.3000 | **changed** |
| Diabetes / regression / 40.6 | Reduction @ max-lossless | 64.5% | 65.3% | +0.8000 | **changed** |
| Digits / classification | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification | MF @ 1x (Δ) | 135.6 (-0.0002 ± 0.0010 acc) | 143.8 (-0.0015 ± 0.0016 acc) | +8.2000 | **changed** |
| Digits / classification | Reduction @ 1x | 18.9% | 17.4% | -1.5000 | **changed** |
| Digits / classification | MF @ max-lossless (Δ) | 129.1 (+0.0011 ± 0.0034 acc) | 149.2 (+0.0000 ± 0.0000 acc) | +20.1000 | **changed** |
| Digits / classification | Reduction @ max-lossless | 22.8% | 14.3% | -8.5000 | **changed** |
| Glass / classification | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification | MF @ 1x (Δ) | 47.7 (-0.1231 ± 0.0928 acc) | 65.1 (+0.0000 ± 0.0000 acc) | +17.4000 | **changed** |
| Glass / classification | Reduction @ 1x | 37.2% | 12.1% | -25.1000 | **changed** |
| Glass / classification | MF @ max-lossless (Δ) | 59.1 (-0.0308 ± 0.0413 acc) | 56.3 (-0.0046 ± 0.0249 acc) | -2.8000 | **changed** |
| Glass / classification | Reduction @ max-lossless | 22.2% | 24.0% | +1.8000 | **changed** |
| Wine / classification / 16.2 | MF @ max-lossless (Δ) | 15.8 (+0.0000 ± 0.0000 acc) | 14.5 (-0.0111 ± 0.0206 acc) | -1.3000 | **changed** |
| Wine / classification / 16.2 | Reduction @ max-lossless | 2.5% | 10.5% | +8.0000 | **changed** |
| BreastCancer / classification | Max-lossless × | 3× | 5× | +2.0000 | timing |
| Concrete / regression | Max-lossless × | 10× | 7× | -3.0000 | timing |
| Diabetes / regression / 40.6 | MF @ 1x (Δ) | 35.2 (+0.0060 ± 0.0177 R²) | 35.2 (+0.0058 ± 0.0178 R²) | +0.0000 | within noise |
| Digits / classification | Max-lossless × | 2× | 0.1× | -1.9000 | timing |
| Glass / classification | Max-lossless × | 0.1× | 7× | +6.9000 | timing |
| Wine / classification / 16.2 | Max-lossless × | 7× | 10× | +3.0000 | timing |

### `table_4_8_mf_dedup_sweep`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| BreastCancer / classification / 0.1 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 0.1 | Dedup MF (mean±std) | 19.90 ± 0.54 | 11.00 ± 0.00 | -8.9000 | **changed** |
| BreastCancer / classification / 0.3 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 0.3 | Dedup MF (mean±std) | 19.60 ± 0.66 | 11.00 ± 0.00 | -8.6000 | **changed** |
| BreastCancer / classification / 1 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 1 | Dedup MF (mean±std) | 15.30 ± 1.19 | 11.00 ± 0.00 | -4.3000 | **changed** |
| BreastCancer / classification / 10 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 10 | Dedup MF (mean±std) | 4.10 ± 0.30 | 10.00 ± 0.00 | +5.9000 | **changed** |
| BreastCancer / classification / 10 | Delta (mean±std) | -0.27485 ± 0.11693 | -0.00526 ± 0.00409 | +0.2696 | **changed** |
| BreastCancer / classification / 100 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 100 | Delta (mean±std) | -0.50351 ± 0.01600 | -0.56257 ± 0.01354 | -0.0591 | **changed** |
| BreastCancer / classification / 15 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 15 | Dedup MF (mean±std) | 3.90 ± 0.30 | 9.10 ± 0.30 | +5.2000 | **changed** |
| BreastCancer / classification / 15 | Delta (mean±std) | -0.25380 ± 0.09765 | +0.00175 ± 0.00742 | +0.2555 | **changed** |
| BreastCancer / classification / 2 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 20 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 20 | Dedup MF (mean±std) | 1.90 ± 0.30 | 8.60 ± 0.49 | +6.7000 | **changed** |
| BreastCancer / classification / 20 | Delta (mean±std) | -0.31404 ± 0.14547 | -0.01813 ± 0.03394 | +0.2959 | **changed** |
| BreastCancer / classification / 3 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 30 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 30 | Dedup MF (mean±std) | 1.60 ± 0.49 | 8.00 ± 0.00 | +6.4000 | **changed** |
| BreastCancer / classification / 30 | Delta (mean±std) | -0.33801 ± 0.16305 | +0.00468 ± 0.00510 | +0.3427 | **changed** |
| BreastCancer / classification / 5 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 5 | Dedup MF (mean±std) | 6.80 ± 1.17 | 11.00 ± 0.00 | +4.2000 | **changed** |
| BreastCancer / classification / 5 | Delta (mean±std) | -0.04912 ± 0.04211 | +0.00000 ± 0.00000 | +0.0491 | **changed** |
| BreastCancer / classification / 50 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 50 | Dedup MF (mean±std) | 1.00 ± 0.00 | 5.40 ± 0.49 | +4.4000 | **changed** |
| BreastCancer / classification / 50 | Delta (mean±std) | -0.50351 ± 0.01600 | -0.02456 ± 0.02277 | +0.4789 | **changed** |
| BreastCancer / classification / 7 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 7 | Dedup MF (mean±std) | 5.20 ± 0.60 | 10.50 ± 0.50 | +5.3000 | **changed** |
| BreastCancer / classification / 7 | Delta (mean±std) | -0.16433 ± 0.11377 | -0.00175 ± 0.00268 | +0.1626 | **changed** |
| BreastCancer / classification / 70 | Raw MF | 19.9 | 11.0 | -8.9000 | **changed** |
| BreastCancer / classification / 70 | Dedup MF (mean±std) | 1.00 ± 0.00 | 3.80 ± 0.98 | +2.8000 | **changed** |
| Concrete / regression / 0.1 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 0.1 | Dedup MF (mean±std) | 33.20 ± 2.71 | 64.70 ± 3.29 | +31.5000 | **changed** |
| Concrete / regression / 0.3 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 0.3 | Dedup MF (mean±std) | 33.20 ± 2.71 | 64.70 ± 3.29 | +31.5000 | **changed** |
| Concrete / regression / 1 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 1 | Dedup MF (mean±std) | 33.10 ± 2.77 | 64.50 ± 3.29 | +31.4000 | **changed** |
| Concrete / regression / 10 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 10 | Dedup MF (mean±std) | 32.00 ± 3.22 | 59.00 ± 2.90 | +27.0000 | **changed** |
| Concrete / regression / 100 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 100 | Dedup MF (mean±std) | 3.00 ± 0.63 | 5.70 ± 1.42 | +2.7000 | **changed** |
| Concrete / regression / 15 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 15 | Dedup MF (mean±std) | 29.60 ± 3.14 | 51.10 ± 3.24 | +21.5000 | **changed** |
| Concrete / regression / 2 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 2 | Dedup MF (mean±std) | 33.00 ± 2.76 | 64.40 ± 3.07 | +31.4000 | **changed** |
| Concrete / regression / 20 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 20 | Dedup MF (mean±std) | 26.80 ± 3.37 | 44.50 ± 3.29 | +17.7000 | **changed** |
| Concrete / regression / 3 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 3 | Dedup MF (mean±std) | 33.00 ± 2.76 | 63.80 ± 3.06 | +30.8000 | **changed** |
| Concrete / regression / 30 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 30 | Dedup MF (mean±std) | 21.20 ± 4.21 | 33.70 ± 3.35 | +12.5000 | **changed** |
| Concrete / regression / 30 | Delta (mean±std) | -0.34892 ± 0.26860 | -0.97588 ± 0.54239 | -0.6270 | **changed** |
| Concrete / regression / 5 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 5 | Dedup MF (mean±std) | 33.00 ± 2.76 | 62.70 ± 3.10 | +29.7000 | **changed** |
| Concrete / regression / 50 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 50 | Dedup MF (mean±std) | 12.60 ± 2.06 | 18.90 ± 1.81 | +6.3000 | **changed** |
| Concrete / regression / 50 | Delta (mean±std) | -0.78334 ± 0.59160 | -4.32890 ± 0.99015 | -3.5456 | **changed** |
| Concrete / regression / 7 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 7 | Dedup MF (mean±std) | 32.90 ± 2.77 | 61.90 ± 2.98 | +29.0000 | **changed** |
| Concrete / regression / 70 | Raw MF | 33.9 | 67.1 | +33.2000 | **changed** |
| Concrete / regression / 70 | Dedup MF (mean±std) | 8.60 ± 1.28 | 12.70 ± 1.68 | +4.1000 | **changed** |
| Concrete / regression / 70 | Delta (mean±std) | -2.69400 ± 1.38956 | -5.37694 ± 0.35474 | -2.6829 | **changed** |
| Digits / classification / 0.1 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 0.1 | Dedup MF (mean±std) | 138.80 ± 6.10 | 149.20 ± 4.35 | +10.4000 | **changed** |
| Digits / classification / 0.3 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 0.3 | Dedup MF (mean±std) | 137.80 ± 6.46 | 148.00 ± 4.31 | +10.2000 | **changed** |
| Digits / classification / 1 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 1 | Dedup MF (mean±std) | 135.60 ± 6.58 | 143.80 ± 4.21 | +8.2000 | **changed** |
| Digits / classification / 10 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 100 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 15 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 15 | Dedup MF (mean±std) | 47.70 ± 2.69 | 44.00 ± 2.83 | -3.7000 | **changed** |
| Digits / classification / 2 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 20 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 20 | Dedup MF (mean±std) | 36.50 ± 2.29 | 32.20 ± 1.78 | -4.3000 | **changed** |
| Digits / classification / 3 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 30 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 30 | Dedup MF (mean±std) | 22.90 ± 2.55 | 19.40 ± 1.36 | -3.5000 | **changed** |
| Digits / classification / 5 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 50 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 50 | Delta (mean±std) | -0.08944 ± 0.07553 | +0.03074 ± 0.06749 | +0.1202 | **changed** |
| Digits / classification / 7 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 70 | Raw MF | 167.3 | 174.1 | +6.8000 | **changed** |
| Digits / classification / 70 | Delta (mean±std) | -0.09000 ± 0.05658 | +0.02556 ± 0.05127 | +0.1156 | **changed** |
| Glass / classification / 0.1 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 0.1 | Dedup MF (mean±std) | 59.10 ± 2.47 | 66.10 ± 4.16 | +7.0000 | **changed** |
| Glass / classification / 0.3 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 0.3 | Dedup MF (mean±std) | 50.90 ± 2.74 | 66.10 ± 4.16 | +15.2000 | **changed** |
| Glass / classification / 1 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 1 | Dedup MF (mean±std) | 47.70 ± 2.83 | 65.10 ± 4.16 | +17.4000 | **changed** |
| Glass / classification / 1 | Delta (mean±std) | -0.12308 ± 0.09282 | +0.00000 ± 0.00000 | +0.1231 | **changed** |
| Glass / classification / 10 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 10 | Dedup MF (mean±std) | 34.30 ± 2.33 | 51.10 ± 3.56 | +16.8000 | **changed** |
| Glass / classification / 100 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 15 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 15 | Dedup MF (mean±std) | 28.30 ± 1.79 | 42.80 ± 2.44 | +14.5000 | **changed** |
| Glass / classification / 2 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 2 | Dedup MF (mean±std) | 45.90 ± 2.62 | 63.70 ± 4.15 | +17.8000 | **changed** |
| Glass / classification / 2 | Delta (mean±std) | -0.11385 ± 0.08319 | +0.00000 ± 0.00000 | +0.1139 | **changed** |
| Glass / classification / 20 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 20 | Dedup MF (mean±std) | 22.10 ± 2.17 | 34.80 ± 2.75 | +12.7000 | **changed** |
| Glass / classification / 3 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 3 | Dedup MF (mean±std) | 45.10 ± 2.47 | 62.20 ± 4.24 | +17.1000 | **changed** |
| Glass / classification / 3 | Delta (mean±std) | -0.12462 ± 0.06896 | -0.00308 ± 0.00923 | +0.1215 | **changed** |
| Glass / classification / 30 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 30 | Dedup MF (mean±std) | 14.70 ± 1.35 | 23.90 ± 2.47 | +9.2000 | **changed** |
| Glass / classification / 5 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 5 | Dedup MF (mean±std) | 43.40 ± 2.29 | 58.50 ± 3.98 | +15.1000 | **changed** |
| Glass / classification / 5 | Delta (mean±std) | -0.12000 ± 0.07043 | -0.00615 ± 0.02303 | +0.1138 | **changed** |
| Glass / classification / 50 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 50 | Dedup MF (mean±std) | 5.30 ± 0.78 | 10.80 ± 2.14 | +5.5000 | **changed** |
| Glass / classification / 7 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 7 | Dedup MF (mean±std) | 40.60 ± 2.06 | 56.30 ± 3.74 | +15.7000 | **changed** |
| Glass / classification / 7 | Delta (mean±std) | -0.11538 ± 0.07257 | -0.00462 ± 0.02485 | +0.1108 | **changed** |
| Glass / classification / 70 | Raw MF | 76.0 | 74.1 | -1.9000 | **changed** |
| Glass / classification / 70 | Dedup MF (mean±std) | 3.40 ± 0.80 | 5.70 ± 1.27 | +2.3000 | **changed** |
| Wine / classification / 100 | Dedup MF (mean±std) | 1.20 ± 0.40 | 1.80 ± 0.40 | +0.6000 | **changed** |
| Wine / classification / 100 | Delta (mean±std) | -0.45556 ± 0.05251 | -0.61111 ± 0.02869 | -0.1556 | **changed** |
| Wine / classification / 30 | Dedup MF (mean±std) | 7.10 ± 1.22 | 8.40 ± 1.20 | +1.3000 | **changed** |
| Wine / classification / 50 | Dedup MF (mean±std) | 3.10 ± 0.94 | 5.00 ± 0.89 | +1.9000 | **changed** |
| Wine / classification / 70 | Dedup MF (mean±std) | 1.70 ± 0.46 | 4.10 ± 1.04 | +2.4000 | **changed** |
| BreastCancer / classification / 1 | Delta (mean±std) | -0.00877 ± 0.03163 | +0.00000 ± 0.00000 | +0.0088 | within noise |
| BreastCancer / classification / 2 | Dedup MF (mean±std) | 11.10 ± 1.04 | 11.00 ± 0.00 | -0.1000 | within noise |
| BreastCancer / classification / 2 | Delta (mean±std) | -0.00409 ± 0.03339 | +0.00000 ± 0.00000 | +0.0041 | within noise |
| BreastCancer / classification / 3 | Dedup MF (mean±std) | 10.50 ± 1.20 | 11.00 ± 0.00 | +0.5000 | within noise |
| BreastCancer / classification / 3 | Delta (mean±std) | -0.00175 ± 0.02173 | +0.00000 ± 0.00000 | +0.0018 | within noise |
| BreastCancer / classification / 70 | Delta (mean±std) | -0.50351 ± 0.01600 | -0.34795 ± 0.26280 | +0.1556 | within noise |
| Concrete / regression / 0.1 | Delta (mean±std) | +0.00000 ± 0.00000 | +0.00001 ± 0.00002 | +0.0000 | within noise |
| Concrete / regression / 0.3 | Delta (mean±std) | +0.00000 ± 0.00000 | +0.00001 ± 0.00002 | +0.0000 | within noise |
| Concrete / regression / 1 | Delta (mean±std) | -0.00001 ± 0.00003 | +0.00018 ± 0.00051 | +0.0002 | within noise |
| Concrete / regression / 10 | Delta (mean±std) | -0.00017 ± 0.00224 | -0.02648 ± 0.04062 | -0.0263 | within noise |
| Concrete / regression / 100 | Delta (mean±std) | -5.33977 ± 0.34757 | -5.43376 ± 0.33668 | -0.0940 | within noise |
| Concrete / regression / 15 | Delta (mean±std) | -0.08594 ± 0.08957 | -0.16418 ± 0.17517 | -0.0782 | within noise |
| Concrete / regression / 2 | Delta (mean±std) | -0.00015 ± 0.00041 | +0.00016 ± 0.00052 | +0.0003 | within noise |
| Concrete / regression / 20 | Delta (mean±std) | -0.15185 ± 0.11322 | -0.38670 ± 0.65892 | -0.2348 | within noise |
| Concrete / regression / 3 | Delta (mean±std) | -0.00015 ± 0.00041 | +0.00011 ± 0.00120 | +0.0003 | within noise |
| Concrete / regression / 5 | Delta (mean±std) | -0.00015 ± 0.00041 | -0.00487 ± 0.01161 | -0.0047 | within noise |
| Concrete / regression / 7 | Delta (mean±std) | -0.00013 ± 0.00034 | -0.00560 ± 0.01156 | -0.0055 | within noise |
| Diabetes / regression / 1 | Delta (mean±std) | +0.00604 ± 0.01771 | +0.00582 ± 0.01778 | -0.0002 | within noise |
| Diabetes / regression / 10 | Dedup MF (mean±std) | 11.70 ± 1.42 | 11.60 ± 1.28 | -0.1000 | within noise |
| Diabetes / regression / 10 | Delta (mean±std) | -0.08113 ± 0.07802 | -0.08168 ± 0.07737 | -0.0006 | within noise |
| Diabetes / regression / 100 | Delta (mean±std) | -0.70208 ± 0.55624 | -0.78112 ± 0.52618 | -0.0790 | within noise |
| Diabetes / regression / 15 | Delta (mean±std) | -1.09653 ± 1.02229 | -1.09763 ± 1.02361 | -0.0011 | within noise |
| Diabetes / regression / 2 | Delta (mean±std) | +0.00447 ± 0.01690 | +0.00432 ± 0.01700 | -0.0001 | within noise |
| Diabetes / regression / 20 | Dedup MF (mean±std) | 6.40 ± 1.11 | 6.20 ± 0.98 | -0.2000 | within noise |
| Diabetes / regression / 20 | Delta (mean±std) | -0.33312 ± 0.69760 | -0.34408 ± 0.69453 | -0.0110 | within noise |
| Diabetes / regression / 3 | Dedup MF (mean±std) | 26.00 ± 1.84 | 26.20 ± 1.83 | +0.2000 | within noise |
| Diabetes / regression / 3 | Delta (mean±std) | +0.00144 ± 0.01861 | +0.00133 ± 0.01805 | -0.0001 | within noise |
| Diabetes / regression / 30 | Dedup MF (mean±std) | 2.80 ± 0.98 | 2.50 ± 0.67 | -0.3000 | within noise |
| Diabetes / regression / 30 | Delta (mean±std) | -0.48339 ± 0.60275 | -0.65692 ± 0.57000 | -0.1735 | within noise |
| Diabetes / regression / 5 | Dedup MF (mean±std) | 19.10 ± 2.17 | 18.90 ± 2.17 | -0.2000 | within noise |
| Diabetes / regression / 5 | Delta (mean±std) | -0.30375 ± 0.79826 | -0.30238 ± 0.79803 | +0.0014 | within noise |
| Diabetes / regression / 50 | Delta (mean±std) | -0.65855 ± 0.55875 | -0.71901 ± 0.55396 | -0.0605 | within noise |
| Diabetes / regression / 7 | Dedup MF (mean±std) | 14.40 ± 1.36 | 14.10 ± 1.37 | -0.3000 | within noise |
| Diabetes / regression / 7 | Delta (mean±std) | -0.31967 ± 0.79441 | -0.31708 ± 0.79672 | +0.0026 | within noise |
| Diabetes / regression / 70 | Delta (mean±std) | -0.70208 ± 0.55624 | -0.78112 ± 0.52618 | -0.0790 | within noise |
| Digits / classification / 0.3 | Delta (mean±std) | +0.00000 ± 0.00000 | -0.00056 ± 0.00085 | -0.0006 | within noise |
| Digits / classification / 1 | Delta (mean±std) | -0.00019 ± 0.00100 | -0.00148 ± 0.00161 | -0.0013 | within noise |
| Digits / classification / 10 | Dedup MF (mean±std) | 69.60 ± 5.00 | 66.10 ± 4.61 | -3.5000 | within noise |
| Digits / classification / 10 | Delta (mean±std) | -0.00185 ± 0.02297 | -0.01500 ± 0.02754 | -0.0131 | within noise |
| Digits / classification / 100 | Dedup MF (mean±std) | 2.90 ± 0.54 | 4.00 ± 1.26 | +1.1000 | within noise |
| Digits / classification / 100 | Delta (mean±std) | -0.15333 ± 0.03352 | -0.07296 ± 0.09531 | +0.0804 | within noise |
| Digits / classification / 15 | Delta (mean±std) | -0.01685 ± 0.06467 | -0.02889 ± 0.05270 | -0.0120 | within noise |
| Digits / classification / 2 | Dedup MF (mean±std) | 129.10 ± 7.31 | 135.10 ± 5.49 | +6.0000 | within noise |
| Digits / classification / 2 | Delta (mean±std) | +0.00111 ± 0.00343 | +0.00019 ± 0.00527 | -0.0009 | within noise |
| Digits / classification / 20 | Delta (mean±std) | -0.03093 ± 0.06540 | -0.03426 ± 0.03770 | -0.0033 | within noise |
| Digits / classification / 3 | Dedup MF (mean±std) | 121.20 ± 7.81 | 124.80 ± 5.60 | +3.6000 | within noise |
| Digits / classification / 3 | Delta (mean±std) | +0.00407 ± 0.00474 | +0.00111 ± 0.00941 | -0.0030 | within noise |
| Digits / classification / 30 | Delta (mean±std) | -0.03389 ± 0.07083 | -0.02204 ± 0.03918 | +0.0119 | within noise |
| Digits / classification / 5 | Dedup MF (mean±std) | 104.70 ± 6.33 | 104.40 ± 5.97 | -0.3000 | within noise |
| Digits / classification / 5 | Delta (mean±std) | +0.00704 ± 0.00872 | +0.00296 ± 0.01552 | -0.0041 | within noise |
| Digits / classification / 50 | Dedup MF (mean±std) | 10.10 ± 1.22 | 10.30 ± 1.00 | +0.2000 | within noise |
| Digits / classification / 7 | Dedup MF (mean±std) | 88.40 ± 4.18 | 87.10 ± 4.97 | -1.3000 | within noise |
| Digits / classification / 7 | Delta (mean±std) | -0.00167 ± 0.02693 | -0.00593 ± 0.02390 | -0.0043 | within noise |
| Digits / classification / 70 | Dedup MF (mean±std) | 6.60 ± 1.11 | 7.40 ± 1.11 | +0.8000 | within noise |
| Glass / classification / 0.1 | Delta (mean±std) | -0.03077 ± 0.04128 | +0.00000 ± 0.00000 | +0.0308 | within noise |
| Glass / classification / 0.3 | Delta (mean±std) | -0.04923 ± 0.09122 | +0.00000 ± 0.00000 | +0.0492 | within noise |
| Glass / classification / 10 | Delta (mean±std) | -0.09692 ± 0.07348 | -0.02615 ± 0.02756 | +0.0708 | within noise |
| Glass / classification / 100 | Dedup MF (mean±std) | 2.10 ± 0.30 | 2.00 ± 0.89 | -0.1000 | within noise |
| Glass / classification / 100 | Delta (mean±std) | -0.22769 ± 0.07010 | -0.24923 ± 0.07436 | -0.0215 | within noise |
| Glass / classification / 15 | Delta (mean±std) | -0.10769 ± 0.08285 | -0.05077 ± 0.05242 | +0.0569 | within noise |
| Glass / classification / 20 | Delta (mean±std) | -0.13385 ± 0.07348 | -0.09231 ± 0.06192 | +0.0415 | within noise |
| Glass / classification / 30 | Delta (mean±std) | -0.19692 ± 0.10247 | -0.10615 ± 0.10192 | +0.0908 | within noise |
| Glass / classification / 50 | Delta (mean±std) | -0.34462 ± 0.10484 | -0.24769 ± 0.10888 | +0.0969 | within noise |
| Glass / classification / 70 | Delta (mean±std) | -0.22000 ± 0.06156 | -0.23231 ± 0.10215 | -0.0123 | within noise |
| Wine / classification / 10 | Dedup MF (mean±std) | 14.20 ± 0.98 | 14.50 ± 1.20 | +0.3000 | within noise |
| Wine / classification / 10 | Delta (mean±std) | -0.02778 ± 0.03015 | -0.01111 ± 0.02062 | +0.0167 | within noise |
| Wine / classification / 15 | Dedup MF (mean±std) | 12.60 ± 1.36 | 12.80 ± 1.60 | +0.2000 | within noise |
| Wine / classification / 15 | Delta (mean±std) | -0.03704 ± 0.03884 | -0.01852 ± 0.02485 | +0.0185 | within noise |
| Wine / classification / 2 | Delta (mean±std) | +0.00000 ± 0.00000 | +0.00185 ± 0.00556 | +0.0019 | within noise |
| Wine / classification / 20 | Dedup MF (mean±std) | 10.70 ± 0.64 | 10.80 ± 1.17 | +0.1000 | within noise |
| Wine / classification / 20 | Delta (mean±std) | -0.05370 ± 0.02922 | -0.03148 ± 0.04061 | +0.0222 | within noise |
| Wine / classification / 3 | Delta (mean±std) | +0.00000 ± 0.00000 | +0.00185 ± 0.00556 | +0.0019 | within noise |
| Wine / classification / 30 | Delta (mean±std) | -0.07037 ± 0.06667 | -0.08704 ± 0.06364 | -0.0167 | within noise |
| Wine / classification / 5 | Delta (mean±std) | +0.00000 ± 0.00000 | +0.00185 ± 0.00556 | +0.0019 | within noise |
| Wine / classification / 50 | Delta (mean±std) | -0.34444 ± 0.08811 | -0.27037 ± 0.03722 | +0.0741 | within noise |
| Wine / classification / 7 | Delta (mean±std) | +0.00000 ± 0.00000 | +0.00185 ± 0.00556 | +0.0019 | within noise |
| Wine / classification / 70 | Delta (mean±std) | -0.43519 ± 0.06892 | -0.35741 ± 0.17821 | +0.0778 | within noise |

### `table_4_9_correction_pass`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Cascade → one flat FIS (union, dedup @ exact tol., argmax) | MF count | 68.7 ± 2.9 deduped | 78.0 ± 6.2 deduped | +9.3000 | **changed** |
| Base (no correction pass) | MF count | 76.0 ± 2.9 | 74.1 ± 2.9 | -1.9000 | within noise |
| Base (no correction pass) | Accuracy | 0.5385 ± 0.0596 | 0.5385 ± 0.0537 | +0.0000 | within noise |
| Cascade → one flat FIS (union, dedup @ exact tol., argmax) | Accuracy | 0.5215 ± 0.0662 | 0.5492 ± 0.0323 | +0.0277 | within noise |
| Cascade → one flat FIS (union, dedup @ exact tol., argmax) | Paired Δ vs. base | -0.0169 ± 0.0303 | +0.0108 ± 0.0482 | +0.0277 | within noise |
| Gated cascade (base + experts, routed) | MF count | 107.0 ± 12.2 raw | 105.0 ± 12.8 raw | -2.0000 | within noise |
| Gated cascade (base + experts, routed) | Accuracy | 0.5462 ± 0.0577 | 0.5815 ± 0.0382 | +0.0353 | within noise |
| Gated cascade (base + experts, routed) | Paired Δ vs. base | +0.0077 ± 0.0103 | +0.0431 ± 0.0495 | +0.0354 | within noise |

### `table_6_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete / R2 | flat | 0.605 ± 0.042 | 0.675 ± 0.039 | +0.0700 | **changed** |
| Concrete / RMSE (MPa) | flat | 10.265 ± 0.521 | 9.323 ± 0.752 | -0.9420 | **changed** |
| PhiUSIIL / accuracy | flat | 0.729 ± 0.023 | 0.997 ± 0.001 | +0.2680 | **changed** |
| PhiUSIIL / accuracy | fuzzy tree | 0.735 ± 0.029 | 0.970 ± 0.003 | +0.2350 | **changed** |
| PhiUSIIL / accuracy | mixture (HME) | 0.600 ± 0.069 | 0.999 ± 0.001 | +0.3990 | **changed** |
| Concrete / R2 | fuzzy tree | 0.616 ± 0.032 | 0.583 ± 0.067 | -0.0330 | within noise |
| Concrete / R2 | mixture (HME) | 0.689 ± 0.062 | 0.703 ± 0.057 | +0.0140 | within noise |
| Concrete / RMSE (MPa) | fuzzy tree | 10.139 ± 0.492 | 10.531 ± 0.889 | +0.3920 | within noise |
| Concrete / RMSE (MPa) | mixture (HME) | 9.065 ± 0.695 | 8.874 ± 0.828 | -0.1910 | within noise |

### `table_a1_feature_ranking`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 2 | wasserstein | SpacialCharRatioInURL (0.247) | HasSocialNet (0.867) | +0.6200 | **changed** |
| 2 | composite | HasSocialNet (0.990) | IsHTTPS (0.930) | -0.0600 | **changed** |
| 3 | wasserstein | DegitRatioInURL (0.075) | HasCopyrightInfo (0.743) | +0.6680 | **changed** |
| 3 | composite | DegitRatioInURL (0.864) | HasSocialNet (0.907) | +0.0430 | **changed** |
| 4 | wasserstein | LetterRatioInURL (0.051) | HasDescription (0.629) | +0.5780 | **changed** |
| 4 | composite | HasTitle (0.816) | HasTitle (0.769) | -0.0470 | **changed** |
| 5 | wasserstein | HasSocialNet (0.049) | DomainTitleMatchScore (0.471) | +0.4220 | **changed** |
| 5 | composite | HasCopyrightInfo (0.712) | NoOfCSS (0.744) | +0.0320 | **changed** |
| 1 | wasserstein | URLCharProb (1.000) | URLSimilarityIndex (1.000) | +0.0000 | within noise |
| 1 | composite | IsHTTPS (1.000) | URLSimilarityIndex (1.000) | +0.0000 | within noise |

### `table_a2_feature_count`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1 | wasserstein (acc / fit s) | 0.6709 / 0.13 | 0.9967 / 0.06 | +0.3258 | **changed** |
| 1 | composite (acc / fit s) | 0.4267 / 0.11 | 0.9967 / 0.06 | +0.5700 | **changed** |
| 10 | wasserstein (acc / fit s) | 0.8096 / 0.20 | 0.9995 / 0.09 | +0.1899 | **changed** |
| 10 | composite (acc / fit s) | 0.9989 / 0.17 | 0.9998 / 0.13 | +0.0009 | **changed** |
| 15 | wasserstein (acc / fit s) | 0.8096 / 0.23 | 0.9929 / 0.17 | +0.1833 | **changed** |
| 15 | composite (acc / fit s) | 0.9913 / 0.26 | 0.9999 / 0.21 | +0.0086 | **changed** |
| 2 | wasserstein (acc / fit s) | 0.7146 / 0.14 | 0.9967 / 0.06 | +0.2821 | **changed** |
| 2 | composite (acc / fit s) | 0.4267 / 0.11 | 0.9999 / 0.07 | +0.5732 | **changed** |
| 20 | wasserstein (acc / fit s) | 0.8083 / 0.27 | 0.9980 / 0.21 | +0.1897 | **changed** |
| 20 | composite (acc / fit s) | 0.9918 / 0.28 | 0.9997 / 0.22 | +0.0079 | **changed** |
| 3 | wasserstein (acc / fit s) | 0.7203 / 0.15 | 0.9967 / 0.06 | +0.2764 | **changed** |
| 3 | composite (acc / fit s) | 0.8822 / 0.12 | 0.9999 / 0.07 | +0.1177 | **changed** |
| 4 | wasserstein (acc / fit s) | 0.7286 / 0.17 | 0.9967 / 0.07 | +0.2681 | **changed** |
| 4 | composite (acc / fit s) | 0.8822 / 0.13 | 0.9999 / 0.07 | +0.1177 | **changed** |
| 5 | wasserstein (acc / fit s) | 0.7286 / 0.17 | 0.9966 / 0.07 | +0.2680 | **changed** |
| 5 | composite (acc / fit s) | 0.9292 / 0.14 | 0.9999 / 0.08 | +0.0707 | **changed** |
| 7 | wasserstein (acc / fit s) | 0.7295 / 0.18 | 0.9998 / 0.07 | +0.2703 | **changed** |
| 7 | composite (acc / fit s) | 0.9999 / 0.15 | 0.9998 / 0.10 | -0.0001 | **changed** |
| 1 | bhattacharyya (acc / fit s) | 0.9967 / 0.11 | 0.9967 / 0.06 | +0.0000 | within noise |
| 10 | bhattacharyya (acc / fit s) | 0.9999 / 0.17 | 0.9999 / 0.10 | +0.0000 | within noise |
| 15 | bhattacharyya (acc / fit s) | 0.9999 / 0.23 | 0.9999 / 0.15 | +0.0000 | within noise |
| 2 | bhattacharyya (acc / fit s) | 0.9999 / 0.11 | 0.9999 / 0.06 | +0.0000 | within noise |
| 20 | bhattacharyya (acc / fit s) | 0.9995 / 0.30 | 0.9995 / 0.22 | +0.0000 | within noise |
| 3 | bhattacharyya (acc / fit s) | 0.9999 / 0.11 | 0.9999 / 0.06 | +0.0000 | within noise |
| 4 | bhattacharyya (acc / fit s) | 0.9999 / 0.12 | 0.9999 / 0.06 | +0.0000 | within noise |
| 5 | bhattacharyya (acc / fit s) | 0.9999 / 0.12 | 0.9999 / 0.06 | +0.0000 | within noise |
| 7 | bhattacharyya (acc / fit s) | 0.9999 / 0.13 | 0.9999 / 0.07 | +0.0000 | within noise |

### `table_concrete_reconciliation`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| flat MoG-TSK 0th / log+standardized / refined | R² | 0.749 ± 0.037 | 0.747 ± 0.031 | -0.0020 | within noise |
| flat MoG-TSK 0th / log+standardized / refined | RMSE | 8.17 ± 0.38 | 8.21 ± 0.45 | +0.0400 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | R² | 0.845 ± 0.030 | 0.835 ± 0.039 | -0.0100 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | RMSE | 6.40 ± 0.43 | 6.57 ± 0.55 | +0.1700 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | R² | 0.869 ± 0.023 | 0.852 ± 0.021 | -0.0170 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | RMSE | 5.89 ± 0.37 | 6.27 ± 0.40 | +0.3800 | within noise |
| fuzzy tree / raw / n/a | R² | 0.616 ± 0.032 | 0.583 ± 0.067 | -0.0330 | within noise |
| fuzzy tree / raw / n/a | RMSE | 10.14 ± 0.49 | 10.53 ± 0.89 | +0.3900 | within noise |
| mixture of experts (HME) / log+standardized / n/a | R² | 0.789 ± 0.049 | 0.788 ± 0.052 | -0.0010 | within noise |
| mixture of experts (HME) / log+standardized / n/a | RMSE | 7.46 ± 0.61 | 7.47 ± 0.66 | +0.0100 | within noise |
| mixture of experts (HME) / raw / n/a | R² | 0.689 ± 0.062 | 0.703 ± 0.057 | +0.0140 | within noise |
| mixture of experts (HME) / raw / n/a | RMSE | 9.06 ± 0.69 | 8.87 ± 0.83 | -0.1900 | within noise |

### `table_hyperparam_normalization`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| fuzzy tree / library default | Δ min-max − raw | +0.074 | +0.106 | +0.0320 | **changed** |
| fuzzy tree / library default | Δ z-score − raw | +0.076 | +0.108 | +0.0320 | **changed** |
| mixture of experts / demo-tuned | raw features | 0.755 ± 0.028 | 0.787 ± 0.024 | +0.0320 | **changed** |
| mixture of experts / demo-tuned | Δ min-max − raw | +0.092 | +0.061 | -0.0310 | **changed** |
| mixture of experts / demo-tuned | Δ z-score − raw | +0.092 | +0.061 | -0.0310 | **changed** |
| mixture of experts / demo-tuned | RMSE raw (MPa) | 8.081 ± 0.414 | 7.547 ± 0.419 | -0.5340 | **changed** |
| mixture of experts / library default | Δ min-max − raw | +0.096 | +10.282 | +10.1860 | **changed** |
| mixture of experts / library default | Δ z-score − raw | +0.093 | +10.281 | +10.1880 | **changed** |
| mixture of experts / library default | Δ z-score − min-max | -0.003 | -0.000 | +0.0030 | **changed** |
| fuzzy tree / library default | raw features | 0.616 ± 0.032 | 0.583 ± 0.067 | -0.0330 | within noise |
| fuzzy tree / library default | RMSE raw (MPa) | 10.139 ± 0.492 | 10.531 ± 0.889 | +0.3920 | within noise |
| mixture of experts / library default | raw features | 0.694 ± 0.065 | -9.493 ± 30.632 | -10.1870 | within noise |
| mixture of experts / library default | log + min-max | 0.790 ± 0.055 | 0.789 ± 0.059 | -0.0010 | within noise |
| mixture of experts / library default | log + z-score | 0.787 ± 0.058 | 0.789 ± 0.059 | +0.0020 | within noise |
| mixture of experts / library default | RMSE raw (MPa) | 8.984 ± 0.699 | 23.672 ± 45.019 | +14.6880 | within noise |
| mixture of experts / library default | RMSE log+min-max (MPa) | 7.433 ± 0.788 | 7.443 ± 0.831 | +0.0100 | within noise |
| mixture of experts / library default | RMSE log+z-score (MPa) | 7.485 ± 0.801 | 7.448 ± 0.833 | -0.0370 | within noise |

### `table_norm_conorm_matrix`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete / HME (experts only) / R2 | min/max | 0.735 ± 0.040 | 0.782 ± 0.034 | +0.0470 | **changed** |
| Concrete / HME (experts only) / R2 | hamacher | 0.741 ± 0.038 | 0.784 ± 0.038 | +0.0430 | **changed** |
| Concrete / HME (experts only) / R2 | Best (mean spread) | **probability** (spread 1.829) | **hamacher** (spread 47.260) | +45.4310 | **changed** |
| Concrete / HME (experts only) / RMSE (MPa) | min/max | 8.396 ± 0.457 | 7.620 ± 0.612 | -0.7760 | **changed** |
| Concrete / HME (experts only) / RMSE (MPa) | hamacher | 8.299 ± 0.469 | 7.583 ± 0.674 | -0.7160 | **changed** |
| Concrete / HME (experts only) / RMSE (MPa) | Best (mean spread) | **probability** (spread 15.303) | **hamacher** (spread 58.704) | +43.4010 | **changed** |
| Concrete / flat MoG-TSK / R2 | min/max | 0.576 ± 0.037 | 0.677 ± 0.037 | +0.1010 | **changed** |
| Concrete / flat MoG-TSK / R2 | probability | 0.605 ± 0.042 | 0.675 ± 0.039 | +0.0700 | **changed** |
| Concrete / flat MoG-TSK / R2 | luk | -0.507 ± 0.254 | -3.404 ± 0.419 | -2.8970 | **changed** |
| Concrete / flat MoG-TSK / R2 | hamacher | 0.588 ± 0.041 | 0.682 ± 0.038 | +0.0940 | **changed** |
| Concrete / flat MoG-TSK / R2 | einstein | 0.607 ± 0.041 | 0.672 ± 0.039 | +0.0650 | **changed** |
| Concrete / flat MoG-TSK / R2 | Best (mean spread) | **einstein** (spread 1.114) | **hamacher** (spread 4.086) | +2.9720 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | min/max | 10.641 ± 0.545 | 9.291 ± 0.727 | -1.3500 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | probability | 10.265 ± 0.521 | 9.323 ± 0.752 | -0.9420 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | luk | 19.999 ± 1.550 | 34.264 ± 0.716 | +14.2650 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | hamacher | 10.492 ± 0.526 | 9.225 ± 0.695 | -1.2670 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | einstein | 10.236 ± 0.517 | 9.372 ± 0.759 | -0.8640 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | Best (mean spread) | **einstein** (spread 9.764) | **hamacher** (spread 25.039) | +15.2750 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | min/max | 0.727 ± 0.051 | 0.998 ± 0.001 | +0.2710 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | probability | 0.743 ± 0.044 | 0.998 ± 0.001 | +0.2550 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | luk | 0.739 ± 0.034 | 0.964 ± 0.004 | +0.2250 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | hamacher | 0.741 ± 0.046 | 0.998 ± 0.001 | +0.2570 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | einstein | 0.749 ± 0.042 | 0.998 ± 0.001 | +0.2490 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | Best (mean spread) | **einstein** (spread 0.022) | **einstein** (spread 0.034) | +0.0120 | **changed** |
| PhiUSIIL / flat MoG / accuracy | min/max | 0.740 ± 0.033 | 0.997 ± 0.001 | +0.2570 | **changed** |
| PhiUSIIL / flat MoG / accuracy | probability | 0.729 ± 0.023 | 0.997 ± 0.001 | +0.2680 | **changed** |
| PhiUSIIL / flat MoG / accuracy | luk | 0.662 ± 0.020 | 0.997 ± 0.001 | +0.3350 | **changed** |
| PhiUSIIL / flat MoG / accuracy | hamacher | 0.752 ± 0.034 | 0.997 ± 0.001 | +0.2450 | **changed** |
| PhiUSIIL / flat MoG / accuracy | einstein | 0.724 ± 0.019 | 0.997 ± 0.001 | +0.2730 | **changed** |
| PhiUSIIL / flat MoG / accuracy | Best (mean spread) | **hamacher** (spread 0.090) | **luk** (spread 0.000) | -0.0900 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | min/max | 0.573 ± 0.003 | 0.967 ± 0.003 | +0.3940 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | probability | 0.573 ± 0.003 | 0.967 ± 0.003 | +0.3940 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | luk | 0.589 ± 0.047 | 0.967 ± 0.003 | +0.3780 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | hamacher | 0.573 ± 0.003 | 0.967 ± 0.003 | +0.3940 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | einstein | 0.573 ± 0.003 | 0.967 ± 0.003 | +0.3940 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | Best (mean spread) | **luk** (spread 0.016) | **min/max** (spread 0.000) | -0.0160 | **changed** |
| Concrete / HME (experts only) / R2 | probability | 0.745 ± 0.035 | 0.773 ± 0.045 | +0.0280 | within noise |
| Concrete / HME (experts only) / R2 | luk | -1.084 ± 0.397 | -46.477 ± 129.343 | -45.3930 | within noise |
| Concrete / HME (experts only) / R2 | einstein | 0.744 ± 0.035 | 0.761 ± 0.049 | +0.0170 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | probability | 8.233 ± 0.476 | 7.759 ± 0.704 | -0.4740 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | luk | 23.536 ± 2.384 | 66.286 ± 96.928 | +42.7500 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | einstein | 8.261 ± 0.506 | 7.951 ± 0.779 | -0.3100 | within noise |

## Bit-identical

These tables produced exactly the same numbers on both sides:

- `table_3_2_memory_precision` (26 cells)
- `table_3_7_g2_downstream` (7 cells)
- `table_5_1_battery` (34 cells)
- `table_5_2_multiscale` (15 cells)
- `table_5_3_selection` (15 cells)
- `table_5_4_ch5_g1_scaling` (126 cells)
- `table_5_4_ch5_g1_scaling_raw` (1800 cells)
- `table_a7_regression_scale` (30 cells)
- `table_g5_output_partitioning` (189 cells)
- `table_g5b_skew_sweep` (48 cells)

---

> A cell counts as **changed** only if it moved by more than the larger of the two runs' reported standard deviations; smaller moves are labelled *within noise*. Wall-clock columns are always reported separately and never called a regression — this harness does not control clocks or thermals (see G4 in `NEXT_STEPS.md`).
