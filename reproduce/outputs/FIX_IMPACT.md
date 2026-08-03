# Fix impact — `full-14900hx-r2` → `full-2026-08-03`

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

<details><summary>Provenance — <code>full-2026-08-03</code></summary>

```
label:       full-2026-08-03
generated:   2026-08-03T20:34:58Z

########################################################################
## ARCHIVE ONLY -- NO GENERATOR RAN IN THIS PASS                      ##
##                                                                    ##
## The tables here were copied from reproduce/outputs/ as they stood.  ##
## They were produced by an EARLIER invocation. That invocation may or ##
## may not have run at the SHAs recorded below -- this pass reads the  ##
## SHAs as they are NOW, and cannot know what produced the files.      ##
##                                                                    ##
## Use this only to recover a run whose numeric phase completed and    ##
## whose archive step failed. Confirm the SHAs and the logs/ contents  ##
## match the run you think this is before quoting any number from it.  ##
########################################################################

tribble-fis: 4b33a0deadbe0254ed42b7f6841e3e9c4bbfdde2
tribble-cluster: e3c27e67ae2a41d636dfb472110ae2ded2e4ef82
grad-school: e7a464fecf031eac4f6d0369b40cd6c08c82f349
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
  table_concrete_reconciliation          archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_hyperparam_normalization         archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_g5_output_partitioning           archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_g5b_skew_sweep                   archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_4_1_mog_baselines                archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_6_1_model_family                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_norm_conorm_matrix               archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_4_4_openset                      archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_a1_feature_scoring               archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_pvat_scaling                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_reorder_three_arm            archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_2_memory_precision             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_4_gpu_speedups                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_5_x_ch5_selection                archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)

--- addendum: what actually produced these tables ---

The --archive-only header above is correct to warn that it reads the SHAs as they
are now. Here is what the recovery can establish, from the failed run's own output
and from git rather than from memory:

  the 13 generators ran at   grad-school dccea7e, tribble-fis 4b33a0d,
                             tribble-cluster e3c27e6
  this archive records       grad-school e7a464f (the commit at recovery time)

Both submodule SHAs are unchanged between the two. Only grad-school moved, and its
diff over reproduce/ across that range is: check_prose.py (new, reads nothing these
generators write), figures/{registry,make_figures}.py (figures only),
run_all_tables.sh (the --archive-only path and PYTHONUNBUFFERED),
tables/table_5_x_ch5_selection.py, and common.py.

The common.py change is the only one touching a shared emitter, and it is additive:
`write_markdown`/`emit` gained a `seeds=None` parameter and the footer now reads
`SEEDS if seeds is None else seeds`, which is `SEEDS` for every caller that does not
pass it. Only table_5_x_ch5_selection.py passes it. So no numeric cell in the 13
tables archived here is affected by anything that changed during the run.

table_5_x_ch5_selection is the one table NOT from that pass: it was run afterwards,
against a gated-minimax results.json whose driver had not yet recorded its own seed
sets, so its footer reads "unrecorded" rather than a seed list. Re-running
gated-minimax-selection/run_all.py and then this renderer fills it in.
```

</details>

## Summary

| Table | Cells | Verdict |
|---|---:|---|
| `table_3_1` | 16 | 11 timing |
| `table_3_1_complexity_fit` | 89 | **2 changed**, 25 timing |
| `table_3_1_three_arm` | 56 | 32 timing |
| `table_3_2_memory_precision` | 32 | identical |
| `table_3_4_gpu_speedups` | — | **new-only** |
| `table_4_1` | 7 | **3 changed**, 1 within noise, 2 timing, 1 rows added |
| `table_4_4_openset` | 9 | **3 changed**, 6 within noise |
| `table_4_4b_theta_sweep` | 28 | **18 changed** |
| `table_5_1_battery` | 34 | identical |
| `table_5_2_multiscale` | 15 | identical |
| `table_5_3_selection` | 15 | identical |
| `table_6_1` | 16 | **1 changed**, 4 within noise |
| `table_a1_feature_ranking` | 20 | identical |
| `table_a2_feature_count` | 36 | **12 changed**, 15 within noise |
| `table_concrete_reconciliation` | 34 | 16 within noise |
| `table_g5_output_partitioning` | 126 | 72 within noise |
| `table_g5b_skew_sweep` | 48 | **3 changed**, 7 within noise |
| `table_hyperparam_normalization` | — | **header-changed** |
| `table_norm_conorm_matrix` | 57 | **7 changed**, 21 within noise |

## Tables that could not be compared

- `table_3_4_gpu_speedups` — **new-only**
- `table_hyperparam_normalization` — **header-changed**
  - baseline header: `['Model', 'Hyperparameters', 'raw features', 'log + standardized', 'Δ from normalizing', 'RMSE raw (MPa)', 'RMSE log+std (MPa)']`
  - new header: `['Model', 'Hyperparameters', 'raw features', 'log + min-max', 'log + z-score', 'Δ min-max − raw', 'Δ z-score − raw', 'Δ z-score − min-max', 'RMSE raw (MPa)', 'RMSE log+min-max (MPa)', 'RMSE log+z-score (MPa)']`

## What moved

### `table_3_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,024 | classical VAT (s) | 14.438 ± 0.410 s | 14.261 ± 0.276 s | -0.1770 | timing |
| 1,024 | pVAT (s) | 0.021 ± 0.002 s | 0.020 ± 0.001 s | -0.0010 | timing |
| 1,024 | speedup | 673x | 704x | +31.0000 | timing |
| 2,048 / infeasible (>cap) | pVAT (s) | 0.075 ± 0.002 s | 0.072 ± 0.001 s | -0.0030 | timing |
| 256 | classical VAT (s) | 0.278 ± 0.012 s | 0.278 ± 0.015 s | +0.0000 | timing |
| 256 | pVAT (s) | 0.011 ± 0.026 s | 0.018 ± 0.047 s | +0.0070 | timing |
| 256 | speedup | 25x | 16x | -9.0000 | timing |
| 4,096 / infeasible (>cap) | pVAT (s) | 0.229 ± 0.006 s | 0.228 ± 0.006 s | -0.0010 | timing |
| 512 | classical VAT (s) | 1.902 ± 0.053 s | 1.916 ± 0.036 s | +0.0140 | timing |
| 512 | pVAT (s) | 0.006 ± 0.001 s | 0.006 ± 0.000 s | +0.0000 | timing |
| 512 | speedup | 311x | 312x | +1.0000 | timing |

### `table_3_1_complexity_fit`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| fitted exponent /  / 3.20 (6 pts) | stage 1 | **1.86** (11 pts) | **1.88** (11 pts) | +0.0200 | **changed** |
| fitted exponent /  / 3.20 (6 pts) | stage 2 | **1.97** (11 pts) | **1.95** (11 pts) | -0.0200 | **changed** |
| 1,000 / 10.0× | classical | 1434.42× | 1468.74× | +34.3200 | timing |
| 1,000 / 10.0× | stage 1 | 81.07× | 86.31× | +5.2400 | timing |
| 1,000 / 10.0× | stage 2 | 69.61× | 72.74× | +3.1300 | timing |
| 1,250 / 12.5× / N/A | stage 1 | 129.34× | 129.07× | -0.2700 | timing |
| 1,250 / 12.5× / N/A | stage 2 | 113.37× | 112.69× | -0.6800 | timing |
| 1,500 / 15.0× / N/A | stage 1 | 174.29× | 179.95× | +5.6600 | timing |
| 1,500 / 15.0× / N/A | stage 2 | 152.49× | 153.46× | +0.9700 | timing |
| 2,000 / 20.0× / N/A | stage 1 | 245.16× | 251.40× | +6.2400 | timing |
| 2,000 / 20.0× / N/A | stage 2 | 304.25× | 294.66× | -9.5900 | timing |
| 2,500 / 25.0× / N/A | stage 1 | 343.16× | 352.31× | +9.1500 | timing |
| 2,500 / 25.0× / N/A | stage 2 | 565.84× | 556.61× | -9.2300 | timing |
| 200 / 2.0× | classical | 7.58× | 8.25× | +0.6700 | timing |
| 200 / 2.0× | stage 1 | 3.23× | 3.21× | -0.0200 | timing |
| 200 / 2.0× | stage 2 | 3.36× | 3.45× | +0.0900 | timing |
| 3,000 / 30.0× / N/A | stage 1 | 474.52× | 471.91× | -2.6100 | timing |
| 3,000 / 30.0× / N/A | stage 2 | 952.10× | 797.97× | -154.1300 | timing |
| 300 / 3.0× | classical | 27.95× | 28.18× | +0.2300 | timing |
| 300 / 3.0× | stage 1 | 6.76× | 6.30× | -0.4600 | timing |
| 300 / 3.0× | stage 2 | 7.96× | 7.29× | -0.6700 | timing |
| 500 / 5.0× | classical | 140.20× | 133.87× | -6.3300 | timing |
| 500 / 5.0× | stage 1 | 18.15× | 16.43× | -1.7200 | timing |
| 500 / 5.0× | stage 2 | 21.14× | 19.59× | -1.5500 | timing |
| 750 / 7.5× | classical | 650.33× | 686.34× | +36.0100 | timing |
| 750 / 7.5× | stage 1 | 52.25× | 49.80× | -2.4500 | timing |
| 750 / 7.5× | stage 2 | 44.34× | 45.00× | +0.6600 | timing |

### `table_3_1_three_arm`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,000 | classical O(N³) (s) | 0.1504 ± 0.0019 s | 0.1506 ± 0.0039 s | +0.0002 | timing |
| 1,000 | stage 1 O(N²logN) (s) | 0.0122 ± 0.0003 s | 0.0132 ± 0.0014 s | +0.0010 | timing |
| 1,000 | cls/s2 | 213.7× | 206.6× | -7.1000 | timing |
| 1,000 | s1/s2 | 17.3× | 18.1× | +0.8000 | timing |
| 1,250 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0195 ± 0.0006 s | 0.0197 ± 0.0006 s | +0.0002 | timing |
| 1,250 / not run (> cap) | s1/s2 | 17.0× | 17.4× | +0.4000 | timing |
| 1,500 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0262 ± 0.0008 s | 0.0275 ± 0.0011 s | +0.0013 | timing |
| 1,500 / not run (> cap) | s1/s2 | 17.0× | 17.9× | +0.9000 | timing |
| 100 / 0.0001 ± 0.0000 s / 0.0002 ± 0.0000 s | cls/s2 | 10.4× | 10.2× | -0.2000 | timing |
| 100 / 0.0001 ± 0.0000 s / 0.0002 ± 0.0000 s | s1/s2 | 14.9× | 15.2× | +0.3000 | timing |
| 2,000 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0369 ± 0.0022 s | 0.0384 ± 0.0032 s | +0.0015 | timing |
| 2,000 / not run (> cap) | stage 2 O(N²) (s) | 0.0031 ± 0.0004 s | 0.0030 ± 0.0002 s | -0.0001 | timing |
| 2,000 / not run (> cap) | s1/s2 | 12.0× | 13.0× | +1.0000 | timing |
| 2,500 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0516 ± 0.0016 s | 0.0538 ± 0.0053 s | +0.0022 | timing |
| 2,500 / not run (> cap) | stage 2 O(N²) (s) | 0.0057 ± 0.0008 s | 0.0056 ± 0.0007 s | -0.0001 | timing |
| 2,500 / not run (> cap) | s1/s2 | 9.0× | 9.6× | +0.6000 | timing |
| 200 | classical O(N³) (s) | 0.0008 ± 0.0000 s | 0.0008 ± 0.0001 s | +0.0000 | timing |
| 200 | cls/s2 | 23.4× | 24.5× | +1.1000 | timing |
| 200 | s1/s2 | 14.3× | 14.2× | -0.1000 | timing |
| 3,000 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0714 ± 0.0009 s | 0.0720 ± 0.0017 s | +0.0006 | timing |
| 3,000 / not run (> cap) | stage 2 O(N²) (s) | 0.0096 ± 0.0015 s | 0.0080 ± 0.0002 s | -0.0016 | timing |
| 3,000 / not run (> cap) | s1/s2 | 7.4× | 9.0× | +1.6000 | timing |
| 300 / 0.0029 ± 0.0002 s / 0.0010 ± 0.0001 s | cls/s2 | 36.4× | 39.6× | +3.2000 | timing |
| 300 / 0.0029 ± 0.0002 s / 0.0010 ± 0.0001 s | s1/s2 | 12.6× | 13.2× | +0.6000 | timing |
| 500 | classical O(N³) (s) | 0.0147 ± 0.0002 s | 0.0137 ± 0.0004 s | -0.0010 | timing |
| 500 | stage 1 O(N²logN) (s) | 0.0027 ± 0.0001 s | 0.0025 ± 0.0001 s | -0.0002 | timing |
| 500 | cls/s2 | 68.8× | 69.9× | +1.1000 | timing |
| 750 | classical O(N³) (s) | 0.0682 ± 0.0027 s | 0.0704 ± 0.0013 s | +0.0022 | timing |
| 750 | stage 1 O(N²logN) (s) | 0.0079 ± 0.0007 s | 0.0076 ± 0.0002 s | -0.0003 | timing |
| 750 | stage 2 O(N²) (s) | 0.0004 ± 0.0000 s | 0.0005 ± 0.0000 s | +0.0001 | timing |
| 750 | cls/s2 | 152.1× | 156.0× | +3.9000 | timing |
| 750 | s1/s2 | 17.5× | 16.9× | -0.6000 | timing |

### `table_4_1`

Rows only in `full-2026-08-03`: `PhiUSIIL (classification) / 0.28 ± 0.03 s / acc=0.997 ± 0.001`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| PhiUSIIL (classification) | Dataset (task) | PhiUSIIL (classification) | Concrete (regression, full 2nd order) |  | **changed** |
| PhiUSIIL (classification) | MoG accuracy / R2 | acc=0.997 ± 0.001 | R2=0.842 ± 0.040 | +1.0030 | **changed** |
| PhiUSIIL (classification) | tree / RF ref | 1.000 ± 0.000 | 0.909 ± 0.019 | -0.0910 | **changed** |
| Concrete (regression) | MoG train time | 1.04 ± 0.62 s | 0.43 ± 0.01 s | -0.6100 | timing |
| Concrete (regression) | MoG accuracy / R2 | R2=0.780 ± 0.029 | R2=0.783 ± 0.030 | +0.0000 | within noise |
| PhiUSIIL (classification) | MoG train time | 0.64 ± 0.02 s | 0.46 ± 0.02 s | -0.1800 | timing |

### `table_4_4_openset`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Complement rule (this work) | Detection − false alarm | +0.169 | +0.129 | -0.0400 | **changed** |
| Isolation Forest | Detection − false alarm | +0.171 | +0.176 | +0.0050 | **changed** |
| One-class SVM | Detection − false alarm | +0.076 | +0.111 | +0.0350 | **changed** |
| Complement rule (this work) | Detection rate | 0.380 ± 0.334 | 0.491 ± 0.359 | +0.1110 | within noise |
| Complement rule (this work) | False-alarm rate | 0.211 ± 0.139 | 0.362 ± 0.262 | +0.1510 | within noise |
| Isolation Forest | Detection rate | 0.401 ± 0.324 | 0.493 ± 0.341 | +0.0920 | within noise |
| Isolation Forest | False-alarm rate | 0.230 ± 0.149 | 0.317 ± 0.160 | +0.0870 | within noise |
| One-class SVM | Detection rate | 0.281 ± 0.259 | 0.399 ± 0.300 | +0.1180 | within noise |
| One-class SVM | False-alarm rate | 0.206 ± 0.125 | 0.288 ± 0.166 | +0.0820 | within noise |

### `table_4_4b_theta_sweep`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 0.500 | detection rate | 0.754 | 0.838 | +0.0840 | **changed** |
| 0.500 | false-alarm rate | 0.532 | 0.719 | +0.1870 | **changed** |
| 0.500 | detection − false alarm | +0.222 | +0.119 | -0.1030 | **changed** |
| 0.600 | detection rate | 0.708 | 0.815 | +0.1070 | **changed** |
| 0.600 | false-alarm rate | 0.469 | 0.664 | +0.1950 | **changed** |
| 0.600 | detection − false alarm | +0.239 | +0.151 | -0.0880 | **changed** |
| 0.700 | detection rate | 0.631 | 0.762 | +0.1310 | **changed** |
| 0.700 | false-alarm rate | 0.395 | 0.616 | +0.2210 | **changed** |
| 0.700 | detection − false alarm | +0.236 | +0.146 | -0.0900 | **changed** |
| 0.800 | detection rate | 0.559 | 0.700 | +0.1410 | **changed** |
| 0.800 | false-alarm rate | 0.332 | 0.546 | +0.2140 | **changed** |
| 0.800 | detection − false alarm | +0.227 | +0.154 | -0.0730 | **changed** |
| 0.900 | detection rate | 0.473 | 0.625 | +0.1520 | **changed** |
| 0.900 | false-alarm rate | 0.277 | 0.475 | +0.1980 | **changed** |
| 0.900 | detection − false alarm | +0.196 | +0.150 | -0.0460 | **changed** |
| 0.990 | detection rate | 0.380 | 0.491 | +0.1110 | **changed** |
| 0.990 | false-alarm rate | 0.211 | 0.362 | +0.1510 | **changed** |
| 0.990 | detection − false alarm | +0.169 | +0.129 | -0.0400 | **changed** |

### `table_6_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| PhiUSIIL / accuracy / 0.997 ± 0.001 | mixture (HME) | 0.999 ± 0.001 | 1.000 ± 0.001 | +0.0010 | **changed** |
| Concrete / R2 | flat | 0.650 ± 0.056 | 0.658 ± 0.040 | +0.0080 | within noise |
| Concrete / R2 | mixture (HME) | 0.686 ± 0.060 | 0.679 ± 0.062 | -0.0070 | within noise |
| Concrete / RMSE (MPa) | flat | 9.633 ± 0.536 | 9.553 ± 0.498 | -0.0800 | within noise |
| Concrete / RMSE (MPa) | mixture (HME) | 9.114 ± 0.738 | 9.220 ± 0.802 | +0.1060 | within noise |

### `table_a2_feature_count`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 10 | wasserstein (acc / fit s) | 0.9995 / 1.23 | 0.9997 / 0.39 | +0.0002 | **changed** |
| 10 | bhattacharyya (acc / fit s) | 0.9676 / 1.83 | 0.9701 / 0.53 | +0.0025 | **changed** |
| 10 | composite (acc / fit s) | 0.9980 / 1.69 | 0.9983 / 0.57 | +0.0003 | **changed** |
| 15 | wasserstein (acc / fit s) | 0.9974 / 1.96 | 0.9957 / 0.56 | -0.0017 | **changed** |
| 15 | bhattacharyya (acc / fit s) | 0.9765 / 2.91 | 0.9788 / 0.72 | +0.0023 | **changed** |
| 20 | wasserstein (acc / fit s) | 0.9990 / 2.70 | 0.9984 / 0.69 | -0.0006 | **changed** |
| 20 | bhattacharyya (acc / fit s) | 0.9777 / 3.54 | 0.9796 / 0.79 | +0.0019 | **changed** |
| 20 | composite (acc / fit s) | 0.9995 / 3.52 | 0.9991 / 0.84 | -0.0004 | **changed** |
| 3 | bhattacharyya (acc / fit s) | 0.8455 / 0.53 | 0.8447 / 0.25 | -0.0008 | **changed** |
| 5 | wasserstein (acc / fit s) | 0.9966 / 0.70 | 0.9965 / 0.31 | -0.0001 | **changed** |
| 5 | bhattacharyya (acc / fit s) | 0.9456 / 0.95 | 0.9467 / 0.36 | +0.0011 | **changed** |
| 7 | bhattacharyya (acc / fit s) | 0.9610 / 1.32 | 0.9632 / 0.41 | +0.0022 | **changed** |
| 1 | wasserstein (acc / fit s) | 0.9967 / 0.55 | 0.9967 / 0.43 | +0.0000 | within noise |
| 1 | bhattacharyya (acc / fit s) | 0.4267 / 0.25 | 0.4267 / 0.15 | +0.0000 | within noise |
| 1 | composite (acc / fit s) | 0.4267 / 0.30 | 0.4267 / 0.19 | +0.0000 | within noise |
| 15 | composite (acc / fit s) | 0.9999 / 2.54 | 0.9999 / 0.68 | +0.0000 | within noise |
| 2 | wasserstein (acc / fit s) | 0.9967 / 0.43 | 0.9967 / 0.23 | +0.0000 | within noise |
| 2 | bhattacharyya (acc / fit s) | 0.4527 / 0.34 | 0.4527 / 0.17 | +0.0000 | within noise |
| 2 | composite (acc / fit s) | 0.9967 / 0.45 | 0.9967 / 0.25 | +0.0000 | within noise |
| 3 | wasserstein (acc / fit s) | 0.9967 / 0.51 | 0.9967 / 0.25 | +0.0000 | within noise |
| 3 | composite (acc / fit s) | 0.9967 / 0.51 | 0.9967 / 0.27 | +0.0000 | within noise |
| 4 | wasserstein (acc / fit s) | 0.9967 / 0.59 | 0.9967 / 0.27 | +0.0000 | within noise |
| 4 | bhattacharyya (acc / fit s) | 0.9160 / 0.72 | 0.9160 / 0.33 | +0.0000 | within noise |
| 4 | composite (acc / fit s) | 0.9966 / 0.71 | 0.9966 / 0.34 | +0.0000 | within noise |
| 5 | composite (acc / fit s) | 0.9966 / 0.89 | 0.9966 / 0.40 | +0.0000 | within noise |
| 7 | wasserstein (acc / fit s) | 0.9998 / 0.89 | 0.9998 / 0.33 | +0.0000 | within noise |
| 7 | composite (acc / fit s) | 0.9967 / 1.22 | 0.9967 / 0.47 | +0.0000 | within noise |

### `table_concrete_reconciliation`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| flat MoG-TSK 0th / log+standardized / closed-form only | R² | -0.334 ± 0.201 | -0.434 ± 0.241 | -0.1000 | within noise |
| flat MoG-TSK 0th / log+standardized / closed-form only | RMSE | 18.83 ± 1.21 | 19.48 ± 0.94 | +0.6500 | within noise |
| flat MoG-TSK 0th / log+standardized / refined | R² | 0.582 ± 0.072 | 0.517 ± 0.210 | -0.0650 | within noise |
| flat MoG-TSK 0th / log+standardized / refined | RMSE | 10.54 ± 0.95 | 11.11 ± 2.27 | +0.5700 | within noise |
| flat MoG-TSK 1st / log+standardized / closed-form only | R² | 0.772 ± 0.034 | 0.787 ± 0.026 | +0.0150 | within noise |
| flat MoG-TSK 1st / log+standardized / closed-form only | RMSE | 7.80 ± 0.73 | 7.54 ± 0.39 | -0.2600 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | R² | 0.836 ± 0.054 | 0.866 ± 0.029 | +0.0300 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | RMSE | 6.52 ± 0.93 | 5.94 ± 0.48 | -0.5800 | within noise |
| flat MoG-TSK 2nd / log+standardized / closed-form only | R² | 0.824 ± 0.043 | 0.832 ± 0.027 | +0.0080 | within noise |
| flat MoG-TSK 2nd / log+standardized / closed-form only | RMSE | 6.84 ± 0.94 | 6.68 ± 0.55 | -0.1600 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | R² | 0.864 ± 0.046 | 0.877 ± 0.037 | +0.0130 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | RMSE | 5.94 ± 0.82 | 5.65 ± 0.64 | -0.2900 | within noise |
| mixture of experts (HME) / log+standardized / n/a | R² | 0.763 ± 0.057 | 0.762 ± 0.061 | -0.0010 | within noise |
| mixture of experts (HME) / log+standardized / n/a | RMSE | 7.90 ± 0.93 | 7.92 ± 0.95 | +0.0200 | within noise |
| mixture of experts (HME) / raw / n/a | R² | 0.686 ± 0.060 | 0.679 ± 0.062 | -0.0070 | within noise |
| mixture of experts (HME) / raw / n/a | RMSE | 9.11 ± 0.74 | 9.22 ± 0.80 | +0.1100 | within noise |

### `table_g5_output_partitioning`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 3 / 1st / hybrid *(shipped)* | R² | 0.772 ± 0.034 | 0.787 ± 0.026 | +0.0150 | within noise |
| 3 / 1st / hybrid *(shipped)* | RMSE (MPa) | 7.80 ± 0.73 | 7.54 ± 0.39 | -0.2600 | within noise |
| 3 / 1st / hybrid *(shipped)* | tail RMSE (MPa) | 8.21 ± 1.07 | 7.93 ± 1.04 | -0.2800 | within noise |
| 3 / 1st / hybrid *(shipped)* | max err (MPa) | 30.5 ± 8.5 | 28.6 ± 6.2 | -1.9000 | within noise |
| 3 / 1st / quantile | R² | 0.773 ± 0.033 | 0.789 ± 0.026 | +0.0160 | within noise |
| 3 / 1st / quantile | RMSE (MPa) | 7.79 ± 0.72 | 7.51 ± 0.39 | -0.2800 | within noise |
| 3 / 1st / quantile | tail RMSE (MPa) | 8.36 ± 1.07 | 8.08 ± 1.05 | -0.2800 | within noise |
| 3 / 1st / quantile | max err (MPa) | 30.3 ± 8.5 | 28.5 ± 6.2 | -1.8000 | within noise |
| 3 / 1st / uniform | R² | 0.795 ± 0.035 | 0.796 ± 0.018 | +0.0010 | within noise |
| 3 / 1st / uniform | RMSE (MPa) | 7.38 ± 0.50 | 7.38 ± 0.34 | +0.0000 | within noise |
| 3 / 1st / uniform | tail RMSE (MPa) | 7.90 ± 0.99 | 8.10 ± 1.14 | +0.2000 | within noise |
| 3 / 1st / uniform | max err (MPa) | 29.1 ± 8.5 | 29.2 ± 5.6 | +0.1000 | within noise |
| 3 / 2nd / hybrid *(shipped)* | R² | 0.824 ± 0.043 | 0.832 ± 0.027 | +0.0080 | within noise |
| 3 / 2nd / hybrid *(shipped)* | RMSE (MPa) | 6.84 ± 0.94 | 6.68 ± 0.55 | -0.1600 | within noise |
| 3 / 2nd / hybrid *(shipped)* | tail RMSE (MPa) | 6.67 ± 0.94 | 6.58 ± 0.88 | -0.0900 | within noise |
| 3 / 2nd / hybrid *(shipped)* | max err (MPa) | 29.9 ± 9.1 | 29.1 ± 8.1 | -0.8000 | within noise |
| 3 / 2nd / quantile | R² | 0.826 ± 0.041 | 0.836 ± 0.025 | +0.0100 | within noise |
| 3 / 2nd / quantile | RMSE (MPa) | 6.81 ± 0.92 | 6.61 ± 0.51 | -0.2000 | within noise |
| 3 / 2nd / quantile | tail RMSE (MPa) | 6.70 ± 0.94 | 6.59 ± 0.84 | -0.1100 | within noise |
| 3 / 2nd / quantile | max err (MPa) | 29.5 ± 9.3 | 28.4 ± 7.9 | -1.1000 | within noise |
| 3 / 2nd / uniform | R² | 0.834 ± 0.032 | 0.841 ± 0.021 | +0.0070 | within noise |
| 3 / 2nd / uniform | RMSE (MPa) | 6.63 ± 0.50 | 6.50 ± 0.43 | -0.1300 | within noise |
| 3 / 2nd / uniform | tail RMSE (MPa) | 6.38 ± 0.71 | 6.50 ± 0.82 | +0.1200 | within noise |
| 3 / 2nd / uniform | max err (MPa) | 28.1 ± 9.1 | 28.5 ± 6.4 | +0.4000 | within noise |
| 4 / 1st / hybrid *(shipped)* | R² | 0.792 ± 0.023 | 0.797 ± 0.024 | +0.0050 | within noise |
| 4 / 1st / hybrid *(shipped)* | RMSE (MPa) | 7.45 ± 0.45 | 7.36 ± 0.34 | -0.0900 | within noise |
| 4 / 1st / hybrid *(shipped)* | tail RMSE (MPa) | 7.98 ± 1.07 | 8.18 ± 1.11 | +0.2000 | within noise |
| 4 / 1st / hybrid *(shipped)* | max err (MPa) | 29.4 ± 7.2 | 28.4 ± 6.5 | -1.0000 | within noise |
| 4 / 1st / quantile | R² | 0.790 ± 0.023 | 0.795 ± 0.024 | +0.0050 | within noise |
| 4 / 1st / quantile | RMSE (MPa) | 7.49 ± 0.44 | 7.40 ± 0.35 | -0.0900 | within noise |
| 4 / 1st / quantile | tail RMSE (MPa) | 8.15 ± 1.10 | 8.30 ± 1.14 | +0.1500 | within noise |
| 4 / 1st / quantile | max err (MPa) | 29.4 ± 7.0 | 28.3 ± 6.6 | -1.1000 | within noise |
| 4 / 1st / uniform | R² | 0.785 ± 0.034 | 0.799 ± 0.025 | +0.0140 | within noise |
| 4 / 1st / uniform | RMSE (MPa) | 7.56 ± 0.61 | 7.32 ± 0.40 | -0.2400 | within noise |
| 4 / 1st / uniform | tail RMSE (MPa) | 8.10 ± 1.21 | 7.90 ± 1.10 | -0.2000 | within noise |
| 4 / 1st / uniform | max err (MPa) | 30.8 ± 8.5 | 29.6 ± 5.5 | -1.2000 | within noise |
| 4 / 2nd / hybrid *(shipped)* | R² | 0.846 ± 0.027 | 0.848 ± 0.025 | +0.0020 | within noise |
| 4 / 2nd / hybrid *(shipped)* | RMSE (MPa) | 6.40 ± 0.54 | 6.35 ± 0.44 | -0.0500 | within noise |
| 4 / 2nd / hybrid *(shipped)* | tail RMSE (MPa) | 6.38 ± 0.89 | 6.73 ± 0.74 | +0.3500 | within noise |
| 4 / 2nd / hybrid *(shipped)* | max err (MPa) | 28.5 ± 7.9 | 29.3 ± 5.5 | +0.8000 | within noise |
| 4 / 2nd / quantile | R² | 0.846 ± 0.027 | 0.850 ± 0.025 | +0.0040 | within noise |
| 4 / 2nd / quantile | RMSE (MPa) | 6.39 ± 0.53 | 6.32 ± 0.44 | -0.0700 | within noise |
| 4 / 2nd / quantile | tail RMSE (MPa) | 6.37 ± 0.89 | 6.70 ± 0.71 | +0.3300 | within noise |
| 4 / 2nd / quantile | max err (MPa) | 28.5 ± 7.9 | 28.8 ± 5.7 | +0.3000 | within noise |
| 4 / 2nd / uniform | R² | 0.834 ± 0.033 | 0.845 ± 0.020 | +0.0110 | within noise |
| 4 / 2nd / uniform | RMSE (MPa) | 6.64 ± 0.73 | 6.42 ± 0.48 | -0.2200 | within noise |
| 4 / 2nd / uniform | tail RMSE (MPa) | 6.51 ± 0.88 | 6.29 ± 0.73 | -0.2200 | within noise |
| 4 / 2nd / uniform | max err (MPa) | 29.2 ± 9.7 | 28.4 ± 8.5 | -0.8000 | within noise |
| 6 / 1st / hybrid *(shipped)* | R² | 0.802 ± 0.030 | 0.808 ± 0.022 | +0.0060 | within noise |
| 6 / 1st / hybrid *(shipped)* | RMSE (MPa) | 7.25 ± 0.37 | 7.15 ± 0.27 | -0.1000 | within noise |
| 6 / 1st / hybrid *(shipped)* | tail RMSE (MPa) | 7.37 ± 0.89 | 7.37 ± 0.93 | +0.0000 | within noise |
| 6 / 1st / hybrid *(shipped)* | max err (MPa) | 30.2 ± 6.5 | 31.3 ± 5.4 | +1.1000 | within noise |
| 6 / 1st / quantile | R² | 0.801 ± 0.031 | 0.806 ± 0.023 | +0.0050 | within noise |
| 6 / 1st / quantile | RMSE (MPa) | 7.27 ± 0.39 | 7.18 ± 0.29 | -0.0900 | within noise |
| 6 / 1st / quantile | tail RMSE (MPa) | 7.43 ± 0.89 | 7.44 ± 0.93 | +0.0100 | within noise |
| 6 / 1st / quantile | max err (MPa) | 30.3 ± 6.4 | 30.8 ± 5.5 | +0.5000 | within noise |
| 6 / 1st / uniform | R² | 0.802 ± 0.027 | 0.812 ± 0.027 | +0.0100 | within noise |
| 6 / 1st / uniform | RMSE (MPa) | 7.27 ± 0.40 | 7.07 ± 0.37 | -0.2000 | within noise |
| 6 / 1st / uniform | tail RMSE (MPa) | 7.79 ± 0.91 | 7.46 ± 1.11 | -0.3300 | within noise |
| 6 / 1st / uniform | max err (MPa) | 30.2 ± 9.5 | 29.3 ± 6.1 | -0.9000 | within noise |
| 6 / 2nd / hybrid *(shipped)* | R² | 0.848 ± 0.026 | 0.852 ± 0.019 | +0.0040 | within noise |
| 6 / 2nd / hybrid *(shipped)* | RMSE (MPa) | 6.34 ± 0.45 | 6.28 ± 0.37 | -0.0600 | within noise |
| 6 / 2nd / hybrid *(shipped)* | tail RMSE (MPa) | 6.01 ± 0.85 | 6.34 ± 1.05 | +0.3300 | within noise |
| 6 / 2nd / hybrid *(shipped)* | max err (MPa) | 29.7 ± 7.1 | 32.9 ± 5.8 | +3.2000 | within noise |
| 6 / 2nd / quantile | R² | 0.849 ± 0.026 | 0.853 ± 0.020 | +0.0040 | within noise |
| 6 / 2nd / quantile | RMSE (MPa) | 6.33 ± 0.45 | 6.26 ± 0.39 | -0.0700 | within noise |
| 6 / 2nd / quantile | tail RMSE (MPa) | 6.03 ± 0.83 | 6.34 ± 1.02 | +0.3100 | within noise |
| 6 / 2nd / quantile | max err (MPa) | 29.6 ± 7.1 | 32.5 ± 5.8 | +2.9000 | within noise |
| 6 / 2nd / uniform | R² | 0.840 ± 0.029 | 0.853 ± 0.018 | +0.0130 | within noise |
| 6 / 2nd / uniform | RMSE (MPa) | 6.52 ± 0.54 | 6.26 ± 0.35 | -0.2600 | within noise |
| 6 / 2nd / uniform | tail RMSE (MPa) | 6.49 ± 0.75 | 6.40 ± 0.75 | -0.0900 | within noise |
| 6 / 2nd / uniform | max err (MPa) | 30.0 ± 10.1 | 28.2 ± 6.6 | -1.8000 | within noise |

### `table_g5b_skew_sweep`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 0.50 / +1.84 | Q − U | -0.009 | -0.008 | +0.0010 | **changed** |
| 1.00 / +5.32 | Q − U | -0.033 | -0.003 | +0.0300 | **changed** |
| 1.50 / +10.44 | Q − U | -0.152 | -0.114 | +0.0380 | **changed** |
| 0.01 / +0.05 / 0.912 ± 0.009 | uniform tail RMSE | 0.052 ± 0.006 | 0.052 ± 0.005 | +0.0000 | within noise |
| 0.50 / +1.84 | uniform R² | 0.885 ± 0.018 | 0.884 ± 0.016 | -0.0010 | within noise |
| 0.50 / +1.84 | uniform tail RMSE | 0.064 ± 0.016 | 0.066 ± 0.016 | +0.0020 | within noise |
| 1.00 / +5.32 | uniform R² | 0.761 ± 0.070 | 0.731 ± 0.083 | -0.0300 | within noise |
| 1.00 / +5.32 | uniform tail RMSE | 0.085 ± 0.029 | 0.092 ± 0.036 | +0.0070 | within noise |
| 1.50 / +10.44 | uniform R² | 0.335 ± 0.124 | 0.297 ± 0.126 | -0.0380 | within noise |
| 1.50 / +10.44 | uniform tail RMSE | 0.113 ± 0.053 | 0.118 ± 0.058 | +0.0050 | within noise |

### `table_norm_conorm_matrix`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete / HME (experts only) / R2 | Best (mean spread) | **hamacher** (spread 4.410) | **hamacher** (spread 4.400) | -0.0100 | **changed** |
| Concrete / HME (experts only) / RMSE (MPa) | Best (mean spread) | **hamacher** (spread 27.575) | **hamacher** (spread 27.615) | +0.0400 | **changed** |
| Concrete / flat MoG-TSK / R2 | Best (mean spread) | **min/max** (spread 4.412) | **probability** (spread 4.478) | +0.0660 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | Best (mean spread) | **min/max** (spread 26.021) | **probability** (spread 26.294) | +0.2730 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | luk | 0.979 ± 0.001 | 0.968 ± 0.001 | -0.0110 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | Best (mean spread) | **einstein** (spread 0.019) | **probability** (spread 0.031) | +0.0120 | **changed** |
| PhiUSIIL / flat MoG / accuracy | min/max | 0.996 ± 0.001 | 0.997 ± 0.001 | +0.0010 | **changed** |
| Concrete / HME (experts only) / R2 | min/max | 0.778 ± 0.029 | 0.785 ± 0.041 | +0.0070 | within noise |
| Concrete / HME (experts only) / R2 | probability | 0.768 ± 0.029 | 0.781 ± 0.033 | +0.0130 | within noise |
| Concrete / HME (experts only) / R2 | luk | -3.626 ± 0.397 | -3.608 ± 0.479 | +0.0180 | within noise |
| Concrete / HME (experts only) / R2 | hamacher | 0.784 ± 0.030 | 0.792 ± 0.042 | +0.0080 | within noise |
| Concrete / HME (experts only) / R2 | einstein | 0.746 ± 0.049 | 0.774 ± 0.035 | +0.0280 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | min/max | 7.696 ± 0.615 | 7.560 ± 0.828 | -0.1360 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | probability | 7.863 ± 0.459 | 7.641 ± 0.691 | -0.2220 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | luk | 35.160 ± 1.653 | 35.050 ± 1.244 | -0.1100 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | hamacher | 7.585 ± 0.607 | 7.435 ± 0.828 | -0.1500 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | einstein | 8.206 ± 0.596 | 7.771 ± 0.712 | -0.4350 | within noise |
| Concrete / flat MoG-TSK / R2 | min/max | 0.651 ± 0.052 | 0.642 ± 0.055 | -0.0090 | within noise |
| Concrete / flat MoG-TSK / R2 | probability | 0.650 ± 0.056 | 0.658 ± 0.040 | +0.0080 | within noise |
| Concrete / flat MoG-TSK / R2 | luk | -3.761 ± 0.463 | -3.821 ± 0.501 | -0.0600 | within noise |
| Concrete / flat MoG-TSK / R2 | hamacher | 0.648 ± 0.047 | 0.652 ± 0.054 | +0.0040 | within noise |
| Concrete / flat MoG-TSK / R2 | einstein | 0.624 ± 0.070 | 0.647 ± 0.041 | +0.0230 | within noise |
| Concrete / flat MoG-TSK / RMSE (MPa) | min/max | 9.621 ± 0.594 | 9.764 ± 0.844 | +0.1430 | within noise |
| Concrete / flat MoG-TSK / RMSE (MPa) | probability | 9.633 ± 0.536 | 9.553 ± 0.498 | -0.0800 | within noise |
| Concrete / flat MoG-TSK / RMSE (MPa) | luk | 35.642 ± 1.285 | 35.846 ± 1.325 | +0.2040 | within noise |
| Concrete / flat MoG-TSK / RMSE (MPa) | hamacher | 9.677 ± 0.488 | 9.630 ± 0.799 | -0.0470 | within noise |
| Concrete / flat MoG-TSK / RMSE (MPa) | einstein | 9.970 ± 0.616 | 9.697 ± 0.399 | -0.2730 | within noise |
| PhiUSIIL / HME (experts only) / accuracy | min/max | 0.998 ± 0.002 | 0.998 ± 0.001 | +0.0000 | within noise |

## Bit-identical

These tables produced exactly the same numbers on both sides:

- `table_3_2_memory_precision` (32 cells)
- `table_5_1_battery` (34 cells)
- `table_5_2_multiscale` (15 cells)
- `table_5_3_selection` (15 cells)
- `table_a1_feature_ranking` (20 cells)

---

> A cell counts as **changed** only if it moved by more than the larger of the two runs' reported standard deviations; smaller moves are labelled *within noise*. Wall-clock columns are always reported separately and never called a regression — this harness does not control clocks or thermals (see G4 in `NEXT_STEPS.md`).
