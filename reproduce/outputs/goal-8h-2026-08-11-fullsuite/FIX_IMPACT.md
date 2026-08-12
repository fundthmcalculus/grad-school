# Fix impact — `full-2026-08-03` → `uniform-2026-08-03`

Cell-by-cell diff of the archived table runs, produced by `reproduce/compare_runs.py`. Every table is listed, including the unchanged ones: confining a fix's blast radius is a claim, and it is only supported by showing the tables that did *not* move.

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
##                                                                    ##
## THIS BANNER IS NOT A VERDICT. The addendum at the end of this file  ##
## establishes what produced these tables, and this archive IS         ##
## citable -- see PROVENANCE_MAP.md note 18. Two independent readers   ##
## stopped at this banner and concluded the opposite.                  ##
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

<details><summary>Provenance — <code>uniform-2026-08-03</code></summary>

```
label:       uniform-2026-08-03
generated:   2026-08-04T02:11:34Z
tribble-fis: 1a83df8e8a90938836c62dc14628c9d3492c1559
tribble-cluster: e3c27e67ae2a41d636dfb472110ae2ded2e4ef82
grad-school: 43592d7280f9252ccfef94a9f76bcf3eabbdcf44
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
  table_4_4_openset                      ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_a1_feature_scoring               ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_1_pvat_scaling                 ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_1_reorder_three_arm            ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_2_memory_precision             ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_4_gpu_speedups                 ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_5_x_ch5_selection                ok           seeds=0,1,2,3,4,5,6,7,8,9
```

</details>

## Summary

| Table | Cells | Verdict |
|---|---:|---|
| `table_3_1` | 16 | 10 timing |
| `table_3_1_complexity_fit` | 89 | **2 changed**, 25 timing |
| `table_3_1_three_arm` | 56 | 34 timing |
| `table_3_2_memory_precision` | 32 | identical |
| `table_3_4_gpu_speedups` | 195 | **7 changed**, 55 within noise, 29 timing |
| `table_4_1` | 10 | 2 within noise, 3 timing |
| `table_4_4_openset` | 9 | identical |
| `table_4_4b_theta_sweep` | 28 | identical |
| `table_5_1_battery` | 34 | identical |
| `table_5_2_multiscale` | 15 | identical |
| `table_5_3_selection` | 15 | identical |
| `table_6_1` | 16 | 4 within noise |
| `table_a1_feature_ranking` | 20 | identical |
| `table_a2_feature_count` | 36 | 22 within noise |
| `table_concrete_reconciliation` | 34 | **3 changed**, 13 within noise |
| `table_g5_output_partitioning` | 126 | identical |
| `table_g5b_skew_sweep` | 48 | identical |
| `table_hyperparam_normalization` | 84 | **21 changed**, 24 within noise |
| `table_norm_conorm_matrix` | 57 | **8 changed**, 16 within noise |

## What moved

### `table_3_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,024 | classical VAT (s) | 14.261 ± 0.276 s | 13.704 ± 0.251 s | -0.5570 | timing |
| 1,024 | pVAT (s) | 0.020 ± 0.001 s | 0.021 ± 0.002 s | +0.0010 | timing |
| 1,024 | speedup | 704x | 664x | -40.0000 | timing |
| 2,048 / infeasible (>cap) | pVAT (s) | 0.072 ± 0.001 s | 0.086 ± 0.006 s | +0.0140 | timing |
| 256 | classical VAT (s) | 0.278 ± 0.015 s | 0.283 ± 0.015 s | +0.0050 | timing |
| 256 | pVAT (s) | 0.018 ± 0.047 s | 0.011 ± 0.027 s | -0.0070 | timing |
| 256 | speedup | 16x | 25x | +9.0000 | timing |
| 4,096 / infeasible (>cap) | pVAT (s) | 0.228 ± 0.006 s | 0.231 ± 0.011 s | +0.0030 | timing |
| 512 | classical VAT (s) | 1.916 ± 0.036 s | 1.894 ± 0.042 s | -0.0220 | timing |
| 512 | speedup | 312x | 308x | -4.0000 | timing |

### `table_3_1_complexity_fit`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| fitted exponent /  | classical | **3.20** (6 pts) | **3.15** (6 pts) | -0.0500 | **changed** |
| fitted exponent /  | stage 1 | **1.88** (11 pts) | **1.86** (11 pts) | -0.0200 | **changed** |
| 1,000 / 10.0× | classical | 1468.74× | 1242.84× | -225.9000 | timing |
| 1,000 / 10.0× | stage 1 | 86.31× | 79.17× | -7.1400 | timing |
| 1,000 / 10.0× | stage 2 | 72.74× | 63.50× | -9.2400 | timing |
| 1,250 / 12.5× / N/A | stage 1 | 129.07× | 122.86× | -6.2100 | timing |
| 1,250 / 12.5× / N/A | stage 2 | 112.69× | 101.78× | -10.9100 | timing |
| 1,500 / 15.0× / N/A | stage 1 | 179.95× | 169.19× | -10.7600 | timing |
| 1,500 / 15.0× / N/A | stage 2 | 153.46× | 137.86× | -15.6000 | timing |
| 2,000 / 20.0× / N/A | stage 1 | 251.40× | 237.38× | -14.0200 | timing |
| 2,000 / 20.0× / N/A | stage 2 | 294.66× | 275.73× | -18.9300 | timing |
| 2,500 / 25.0× / N/A | stage 1 | 352.31× | 323.13× | -29.1800 | timing |
| 2,500 / 25.0× / N/A | stage 2 | 556.61× | 480.17× | -76.4400 | timing |
| 200 / 2.0× | classical | 8.25× | 6.46× | -1.7900 | timing |
| 200 / 2.0× | stage 1 | 3.21× | 3.19× | -0.0200 | timing |
| 200 / 2.0× | stage 2 | 3.45× | 2.92× | -0.5300 | timing |
| 3,000 / 30.0× / N/A | stage 1 | 471.91× | 454.50× | -17.4100 | timing |
| 3,000 / 30.0× / N/A | stage 2 | 797.97× | 793.21× | -4.7600 | timing |
| 300 / 3.0× | classical | 28.18× | 23.38× | -4.8000 | timing |
| 300 / 3.0× | stage 1 | 6.30× | 5.91× | -0.3900 | timing |
| 300 / 3.0× | stage 2 | 7.29× | 6.55× | -0.7400 | timing |
| 500 / 5.0× | classical | 133.87× | 113.16× | -20.7100 | timing |
| 500 / 5.0× | stage 1 | 16.43× | 15.72× | -0.7100 | timing |
| 500 / 5.0× | stage 2 | 19.59× | 17.56× | -2.0300 | timing |
| 750 / 7.5× | classical | 686.34× | 571.28× | -115.0600 | timing |
| 750 / 7.5× | stage 1 | 49.80× | 50.13× | +0.3300 | timing |
| 750 / 7.5× | stage 2 | 45.00× | 40.61× | -4.3900 | timing |

### `table_3_1_three_arm`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,000 | classical O(N³) (s) | 0.1506 ± 0.0039 s | 0.1491 ± 0.0013 s | -0.0015 | timing |
| 1,000 | stage 1 O(N²logN) (s) | 0.0132 ± 0.0014 s | 0.0128 ± 0.0003 s | -0.0004 | timing |
| 1,000 | cls/s2 | 206.6× | 206.3× | -0.3000 | timing |
| 1,000 | s1/s2 | 18.1× | 17.6× | -0.5000 | timing |
| 1,250 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0197 ± 0.0006 s | 0.0198 ± 0.0008 s | +0.0001 | timing |
| 1,250 / not run (> cap) | stage 2 O(N²) (s) | 0.0011 ± 0.0000 s | 0.0012 ± 0.0001 s | +0.0001 | timing |
| 1,250 / not run (> cap) | s1/s2 | 17.4× | 17.1× | -0.3000 | timing |
| 1,500 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0275 ± 0.0011 s | 0.0273 ± 0.0023 s | -0.0002 | timing |
| 1,500 / not run (> cap) | stage 2 O(N²) (s) | 0.0015 ± 0.0001 s | 0.0016 ± 0.0001 s | +0.0001 | timing |
| 1,500 / not run (> cap) | s1/s2 | 17.9× | 17.4× | -0.5000 | timing |
| 100 / 0.0001 ± 0.0000 s / 0.0002 ± 0.0000 s | cls/s2 | 10.2× | 10.5× | +0.3000 | timing |
| 100 / 0.0001 ± 0.0000 s / 0.0002 ± 0.0000 s | s1/s2 | 15.2× | 14.2× | -1.0000 | timing |
| 2,000 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0384 ± 0.0032 s | 0.0382 ± 0.0020 s | -0.0002 | timing |
| 2,000 / not run (> cap) | stage 2 O(N²) (s) | 0.0030 ± 0.0002 s | 0.0031 ± 0.0005 s | +0.0001 | timing |
| 2,000 / not run (> cap) | s1/s2 | 13.0× | 12.2× | -0.8000 | timing |
| 2,500 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0538 ± 0.0053 s | 0.0520 ± 0.0016 s | -0.0018 | timing |
| 2,500 / not run (> cap) | stage 2 O(N²) (s) | 0.0056 ± 0.0007 s | 0.0055 ± 0.0008 s | -0.0001 | timing |
| 2,500 / not run (> cap) | s1/s2 | 9.6× | 9.5× | -0.1000 | timing |
| 200 | classical O(N³) (s) | 0.0008 ± 0.0001 s | 0.0008 ± 0.0000 s | +0.0000 | timing |
| 200 | cls/s2 | 24.5× | 23.4× | -1.1000 | timing |
| 200 | s1/s2 | 14.2× | 15.5× | +1.3000 | timing |
| 3,000 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0720 ± 0.0017 s | 0.0732 ± 0.0030 s | +0.0012 | timing |
| 3,000 / not run (> cap) | stage 2 O(N²) (s) | 0.0080 ± 0.0002 s | 0.0090 ± 0.0013 s | +0.0010 | timing |
| 3,000 / not run (> cap) | s1/s2 | 9.0× | 8.1× | -0.9000 | timing |
| 300 | classical O(N³) (s) | 0.0029 ± 0.0002 s | 0.0028 ± 0.0001 s | -0.0001 | timing |
| 300 | cls/s2 | 39.6× | 37.6× | -2.0000 | timing |
| 300 | s1/s2 | 13.2× | 12.8× | -0.4000 | timing |
| 500 | classical O(N³) (s) | 0.0137 ± 0.0004 s | 0.0136 ± 0.0002 s | -0.0001 | timing |
| 500 | cls/s2 | 69.9× | 67.9× | -2.0000 | timing |
| 500 | s1/s2 | 12.8× | 12.7× | -0.1000 | timing |
| 750 | classical O(N³) (s) | 0.0704 ± 0.0013 s | 0.0685 ± 0.0033 s | -0.0019 | timing |
| 750 | stage 1 O(N²logN) (s) | 0.0076 ± 0.0002 s | 0.0081 ± 0.0006 s | +0.0005 | timing |
| 750 | cls/s2 | 156.0× | 148.3× | -7.7000 | timing |
| 750 | s1/s2 | 16.9× | 17.5× | +0.6000 | timing |

### `table_3_4_gpu_speedups`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Pairwise distances / N=16,000, d=200, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 1.66422 ± 0.03574 | 1.83823 ± 0.13313 | +0.1740 | **changed** |
| Pairwise distances / N=16,000, d=200, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.83713 ± 0.04452 | 0.91278 ± 0.03501 | +0.0756 | **changed** |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 6.51664 ± 0.12105 | 6.68240 ± 0.13569 | +0.1658 | **changed** |
| Pairwise distances / N=16,000, d=784, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 11.39017 ± 0.06090 | 11.19383 ± 0.06488 | -0.1963 | **changed** |
| VAT front end / N=16,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 0.55451 ± 0.02104 | 0.52247 ± 0.02036 | -0.0320 | **changed** |
| VAT front end / N=48,000, float32, 9.22 GB resident / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 4.28900 | 4.53324 | +0.2442 | **changed** |
| VAT front end / N=48,000, float32, 9.22 GB resident / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 1.15596 | 1.19409 | +0.0381 | **changed** |
| Boruvka MST (device) / N=16,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.24660 ± 0.01142 | 0.23737 ± 0.00972 | -0.0092 | within noise |
| Boruvka MST (device) / N=16,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.03319 ± 0.00267 | 0.03337 ± 0.00264 | +0.0002 | within noise |
| Boruvka MST (device) / N=16,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 7.43x | 7.11x | -0.3200 | timing |
| Boruvka MST (device) / N=32,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.89526 ± 0.04364 | 0.88163 ± 0.02440 | -0.0136 | within noise |
| Boruvka MST (device) / N=32,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.13670 ± 0.00795 | 0.13568 ± 0.00776 | -0.0010 | within noise |
| Boruvka MST (device) / N=32,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 6.55x | 6.50x | -0.0500 | timing |
| Boruvka MST (device) / N=4,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.01740 ± 0.00203 | 0.01711 ± 0.00211 | -0.0003 | within noise |
| Boruvka MST (device) / N=4,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.00323 ± 0.00021 | 0.00315 ± 0.00025 | -0.0001 | within noise |
| Boruvka MST (device) / N=4,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 5.38x | 5.42x | +0.0400 | timing |
| Boruvka MST (device) / N=8,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.07500 ± 0.01386 | 0.07193 ± 0.01202 | -0.0031 | within noise |
| Boruvka MST (device) / N=8,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.00893 ± 0.00055 | 0.00882 ± 0.00048 | -0.0001 | within noise |
| Boruvka MST (device) / N=8,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 8.40x | 8.16x | -0.2400 | timing |
| Fuzzy C-Means / N=200,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | CPU (s) | 10.58187 ± 10.61534 | 10.56364 ± 10.55491 | -0.0182 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU (s) | 0.42786 ± 0.16230 | 0.41724 ± 0.17474 | -0.0106 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU speedup (CPU/GPU) | 24.73x | 25.32x | +0.5900 | timing |
| Fuzzy C-Means / N=200,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | CPU (s) | 0.99304 ± 0.99814 | 0.98191 ± 0.98562 | -0.0111 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU (s) | 0.42786 ± 0.16230 | 0.41724 ± 0.17474 | -0.0106 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU speedup (CPU/GPU) | 2.32x | 2.35x | +0.0300 | timing |
| Fuzzy C-Means / N=50,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | CPU (s) | 2.31236 ± 2.00875 | 2.31482 ± 2.00948 | +0.0025 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU (s) | 0.17701 ± 0.10801 | 0.17571 ± 0.11563 | -0.0013 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU speedup (CPU/GPU) | 13.06x | 13.17x | +0.1100 | timing |
| Fuzzy C-Means / N=50,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | CPU (s) | 0.21358 ± 0.18171 | 0.21707 ± 0.18042 | +0.0035 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU (s) | 0.17701 ± 0.10801 | 0.17571 ± 0.11563 | -0.0013 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU speedup (CPU/GPU) | 1.21x | 1.24x | +0.0300 | timing |
| Fuzzy C-Means / N=500,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | CPU (s) | 29.82402 ± 26.81687 | 29.18683 ± 25.99489 | -0.6372 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU (s) | 0.76753 ± 0.40966 | 0.74543 ± 0.41058 | -0.0221 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU speedup (CPU/GPU) | 38.86x | 39.15x | +0.2900 | timing |
| Fuzzy C-Means / N=500,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | CPU (s) | 2.88215 ± 2.55183 | 2.76255 ± 2.44033 | -0.1196 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU (s) | 0.76753 ± 0.40966 | 0.74543 ± 0.41058 | -0.0221 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU speedup (CPU/GPU) | 3.76x | 3.71x | -0.0500 | timing |
| Pairwise distances / N=16,000, d=10, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.18663 ± 0.00856 | 0.19188 ± 0.00779 | +0.0053 | within noise |
| Pairwise distances / N=16,000, d=10, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.50297 ± 0.05761 | 0.48903 ± 0.04559 | -0.0139 | within noise |
| Pairwise distances / N=16,000, d=10, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.37x | 0.39x | +0.0200 | timing |
| Pairwise distances / N=16,000, d=10, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.19343 ± 0.00822 | 0.19444 ± 0.01958 | +0.0010 | within noise |
| Pairwise distances / N=16,000, d=10, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.60612 ± 0.04998 | 0.61391 ± 0.07225 | +0.0078 | within noise |
| Pairwise distances / N=16,000, d=10, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.28779 ± 0.00720 | 0.28157 ± 0.01232 | -0.0062 | within noise |
| Pairwise distances / N=16,000, d=10, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.95888 ± 0.08445 | 0.95992 ± 0.05994 | +0.0010 | within noise |
| Pairwise distances / N=16,000, d=10, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.30x | 0.29x | -0.0100 | timing |
| Pairwise distances / N=16,000, d=200, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 1.99x | 2.01x | +0.0200 | timing |
| Pairwise distances / N=16,000, d=200, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 1.66400 ± 0.02522 | 1.68457 ± 0.03278 | +0.0206 | within noise |
| Pairwise distances / N=16,000, d=200, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.55527 ± 0.06427 | 1.53792 ± 0.04797 | -0.0173 | within noise |
| Pairwise distances / N=16,000, d=200, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 1.07x | 1.10x | +0.0300 | timing |
| Pairwise distances / N=16,000, d=200, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 1.61573 ± 0.13818 | 1.59012 ± 0.10352 | -0.0256 | within noise |
| Pairwise distances / N=16,000, d=200, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.46607 ± 0.06212 | 1.46268 ± 0.06277 | -0.0034 | within noise |
| Pairwise distances / N=16,000, d=200, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 1.10x | 1.09x | -0.0100 | timing |
| Pairwise distances / N=16,000, d=50, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.47551 ± 0.01193 | 0.47421 ± 0.01417 | -0.0013 | within noise |
| Pairwise distances / N=16,000, d=50, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.66936 ± 0.02815 | 0.64209 ± 0.03488 | -0.0273 | within noise |
| Pairwise distances / N=16,000, d=50, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.71x | 0.74x | +0.0300 | timing |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.46948 ± 0.00841 | 0.48014 ± 0.01100 | +0.0107 | within noise |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.89234 ± 0.04165 | 0.87730 ± 0.04425 | -0.0150 | within noise |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.53x | 0.55x | +0.0200 | timing |
| Pairwise distances / N=16,000, d=50, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.43572 ± 0.01726 | 0.43034 ± 0.01688 | -0.0054 | within noise |
| Pairwise distances / N=16,000, d=50, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.10304 ± 0.04926 | 1.13528 ± 0.06044 | +0.0322 | within noise |
| Pairwise distances / N=16,000, d=50, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.40x | 0.38x | -0.0200 | timing |
| Pairwise distances / N=16,000, d=784, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 7.12461 ± 0.17480 | 7.24669 ± 0.21367 | +0.1221 | within noise |
| Pairwise distances / N=16,000, d=784, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.69144 ± 0.03277 | 1.69725 ± 0.04594 | +0.0058 | within noise |
| Pairwise distances / N=16,000, d=784, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 4.21x | 4.27x | +0.0600 | timing |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 4.04915 ± 0.05781 | 4.02232 ± 0.04630 | -0.0268 | within noise |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 1.61x | 1.66x | +0.0500 | timing |
| Pairwise distances / N=16,000, d=784, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 7.31356 ± 0.30768 | 7.13415 ± 0.31643 | -0.1794 | within noise |
| VAT front end / N=16,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 1.37126 ± 0.02825 | 1.33380 ± 0.04338 | -0.0375 | within noise |
| VAT front end / N=16,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.14286 ± 0.00977 | 0.13943 ± 0.01037 | -0.0034 | within noise |
| VAT front end / N=16,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 9.60x | 9.57x | -0.0300 | timing |
| VAT front end / N=16,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.14286 ± 0.00977 | 0.13943 ± 0.01037 | -0.0034 | within noise |
| VAT front end / N=16,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 3.88x | 3.75x | -0.1300 | timing |
| VAT front end / N=32,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 6.15734 ± 0.11233 | 6.13825 ± 0.11260 | -0.0191 | within noise |
| VAT front end / N=32,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.50647 ± 0.01332 | 0.50594 ± 0.01405 | -0.0005 | within noise |
| VAT front end / N=32,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 12.16x | 12.13x | -0.0300 | timing |
| VAT front end / N=32,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 2.55655 ± 0.05406 | 2.54027 ± 0.05560 | -0.0163 | within noise |
| VAT front end / N=32,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.50647 ± 0.01332 | 0.50594 ± 0.01405 | -0.0005 | within noise |
| VAT front end / N=32,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 5.05x | 5.02x | -0.0300 | timing |
| VAT front end / N=4,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 0.09294 ± 0.00599 | 0.09080 ± 0.00296 | -0.0021 | within noise |
| VAT front end / N=4,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.01636 ± 0.00023 | 0.01646 ± 0.00054 | +0.0001 | within noise |
| VAT front end / N=4,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 5.68x | 5.52x | -0.1600 | timing |
| VAT front end / N=4,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 0.03772 ± 0.00432 | 0.03749 ± 0.00318 | -0.0002 | within noise |
| VAT front end / N=4,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.01636 ± 0.00023 | 0.01646 ± 0.00054 | +0.0001 | within noise |
| VAT front end / N=4,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 2.31x | 2.28x | -0.0300 | timing |
| VAT front end / N=48,000, float32, 9.22 GB resident / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 3.71x | 3.80x | +0.0900 | timing |
| VAT front end / N=8,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 0.34775 ± 0.01384 | 0.34548 ± 0.01291 | -0.0023 | within noise |
| VAT front end / N=8,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.04069 ± 0.00414 | 0.04006 ± 0.00321 | -0.0006 | within noise |
| VAT front end / N=8,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 8.55x | 8.62x | +0.0700 | timing |
| VAT front end / N=8,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 0.14174 ± 0.01422 | 0.13907 ± 0.01171 | -0.0027 | within noise |
| VAT front end / N=8,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.04069 ± 0.00414 | 0.04006 ± 0.00321 | -0.0006 | within noise |
| VAT front end / N=8,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 3.48x | 3.47x | -0.0100 | timing |

### `table_4_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete (regression) | MoG train time | 0.43 ± 0.01 s | 0.43 ± 0.02 s | +0.0000 | timing |
| Concrete (regression) | MoG accuracy / R2 | R2=0.783 ± 0.030 | R2=0.795 ± 0.025 | +0.0000 | within noise |
| Concrete (regression, full 2nd order) | MoG train time | 0.46 ± 0.02 s | 0.44 ± 0.04 s | -0.0200 | timing |
| Concrete (regression, full 2nd order) | MoG accuracy / R2 | R2=0.842 ± 0.040 | R2=0.852 ± 0.030 | +0.0000 | within noise |
| PhiUSIIL (classification) | MoG train time | 0.28 ± 0.03 s | 0.28 ± 0.02 s | +0.0000 | timing |

### `table_6_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete / R2 | flat | 0.658 ± 0.040 | 0.687 ± 0.049 | +0.0290 | within noise |
| Concrete / R2 | mixture (HME) | 0.679 ± 0.062 | 0.636 ± 0.087 | -0.0430 | within noise |
| Concrete / RMSE (MPa) | flat | 9.553 ± 0.498 | 9.122 ± 0.623 | -0.4310 | within noise |
| Concrete / RMSE (MPa) | mixture (HME) | 9.220 ± 0.802 | 9.785 ± 1.035 | +0.5650 | within noise |

### `table_a2_feature_count`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1 | wasserstein (acc / fit s) | 0.9967 / 0.43 | 0.9967 / 0.44 | +0.0000 | within noise |
| 10 | wasserstein (acc / fit s) | 0.9997 / 0.39 | 0.9997 / 0.42 | +0.0000 | within noise |
| 10 | bhattacharyya (acc / fit s) | 0.9701 / 0.53 | 0.9701 / 0.58 | +0.0000 | within noise |
| 10 | composite (acc / fit s) | 0.9983 / 0.57 | 0.9983 / 0.62 | +0.0000 | within noise |
| 15 | wasserstein (acc / fit s) | 0.9957 / 0.56 | 0.9957 / 0.62 | +0.0000 | within noise |
| 15 | bhattacharyya (acc / fit s) | 0.9788 / 0.72 | 0.9788 / 0.81 | +0.0000 | within noise |
| 15 | composite (acc / fit s) | 0.9999 / 0.68 | 0.9999 / 0.74 | +0.0000 | within noise |
| 2 | wasserstein (acc / fit s) | 0.9967 / 0.23 | 0.9967 / 0.25 | +0.0000 | within noise |
| 20 | wasserstein (acc / fit s) | 0.9984 / 0.69 | 0.9984 / 0.75 | +0.0000 | within noise |
| 20 | bhattacharyya (acc / fit s) | 0.9796 / 0.79 | 0.9796 / 0.92 | +0.0000 | within noise |
| 20 | composite (acc / fit s) | 0.9991 / 0.84 | 0.9991 / 0.93 | +0.0000 | within noise |
| 3 | wasserstein (acc / fit s) | 0.9967 / 0.25 | 0.9967 / 0.26 | +0.0000 | within noise |
| 3 | bhattacharyya (acc / fit s) | 0.8447 / 0.25 | 0.8447 / 0.27 | +0.0000 | within noise |
| 3 | composite (acc / fit s) | 0.9967 / 0.27 | 0.9967 / 0.28 | +0.0000 | within noise |
| 4 | wasserstein (acc / fit s) | 0.9967 / 0.27 | 0.9967 / 0.29 | +0.0000 | within noise |
| 4 | composite (acc / fit s) | 0.9966 / 0.34 | 0.9966 / 0.35 | +0.0000 | within noise |
| 5 | wasserstein (acc / fit s) | 0.9965 / 0.31 | 0.9965 / 0.32 | +0.0000 | within noise |
| 5 | bhattacharyya (acc / fit s) | 0.9467 / 0.36 | 0.9467 / 0.39 | +0.0000 | within noise |
| 5 | composite (acc / fit s) | 0.9966 / 0.40 | 0.9966 / 0.41 | +0.0000 | within noise |
| 7 | wasserstein (acc / fit s) | 0.9998 / 0.33 | 0.9998 / 0.34 | +0.0000 | within noise |
| 7 | bhattacharyya (acc / fit s) | 0.9632 / 0.41 | 0.9632 / 0.46 | +0.0000 | within noise |
| 7 | composite (acc / fit s) | 0.9967 / 0.47 | 0.9967 / 0.50 | +0.0000 | within noise |

### `table_concrete_reconciliation`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| flat MoG-TSK 0th / log+standardized / closed-form only | R² | -0.434 ± 0.241 | 0.394 ± 0.065 | +0.8280 | **changed** |
| flat MoG-TSK 0th / log+standardized / closed-form only | RMSE | 19.48 ± 0.94 | 12.73 ± 0.90 | -6.7500 | **changed** |
| flat MoG-TSK 0th / log+standardized / refined | RMSE | 11.11 ± 2.27 | 8.63 ± 0.40 | -2.4800 | **changed** |
| flat MoG-TSK 0th / log+standardized / refined | R² | 0.517 ± 0.210 | 0.720 ± 0.037 | +0.2030 | within noise |
| flat MoG-TSK 1st / log+standardized / closed-form only | R² | 0.787 ± 0.026 | 0.796 ± 0.018 | +0.0090 | within noise |
| flat MoG-TSK 1st / log+standardized / closed-form only | RMSE | 7.54 ± 0.39 | 7.38 ± 0.34 | -0.1600 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | R² | 0.866 ± 0.029 | 0.834 ± 0.045 | -0.0320 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | RMSE | 5.94 ± 0.48 | 6.59 ± 0.65 | +0.6500 | within noise |
| flat MoG-TSK 2nd / log+standardized / closed-form only | R² | 0.832 ± 0.027 | 0.841 ± 0.021 | +0.0090 | within noise |
| flat MoG-TSK 2nd / log+standardized / closed-form only | RMSE | 6.68 ± 0.55 | 6.50 ± 0.43 | -0.1800 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | R² | 0.877 ± 0.037 | 0.862 ± 0.033 | -0.0150 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | RMSE | 5.65 ± 0.64 | 6.00 ± 0.52 | +0.3500 | within noise |
| mixture of experts (HME) / log+standardized / n/a | R² | 0.762 ± 0.061 | 0.747 ± 0.053 | -0.0150 | within noise |
| mixture of experts (HME) / log+standardized / n/a | RMSE | 7.92 ± 0.95 | 8.18 ± 0.74 | +0.2600 | within noise |
| mixture of experts (HME) / raw / n/a | R² | 0.679 ± 0.062 | 0.636 ± 0.087 | -0.0430 | within noise |
| mixture of experts (HME) / raw / n/a | RMSE | 9.22 ± 0.80 | 9.78 ± 1.04 | +0.5600 | within noise |

### `table_hyperparam_normalization`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| flat MoG-TSK 1st / pipeline default | log + z-score | 0.014 ± 0.195 | 0.713 ± 0.035 | +0.6990 | **changed** |
| flat MoG-TSK 1st / pipeline default | Δ min-max − raw | +0.121 | +0.101 | -0.0200 | **changed** |
| flat MoG-TSK 1st / pipeline default | Δ z-score − raw | -0.651 | +0.018 | +0.6690 | **changed** |
| flat MoG-TSK 1st / pipeline default | Δ z-score − min-max | -0.772 | -0.083 | +0.6890 | **changed** |
| flat MoG-TSK 1st / pipeline default | RMSE log+z-score (MPa) | 16.138 ± 1.196 | 8.743 ± 0.421 | -7.3950 | **changed** |
| flat MoG-TSK 2nd / pipeline default | log + z-score | 0.787 ± 0.039 | 0.827 ± 0.028 | +0.0400 | **changed** |
| flat MoG-TSK 2nd / pipeline default | Δ min-max − raw | +0.029 | +0.037 | +0.0080 | **changed** |
| flat MoG-TSK 2nd / pipeline default | Δ z-score − raw | -0.016 | +0.023 | +0.0390 | **changed** |
| flat MoG-TSK 2nd / pipeline default | Δ z-score − min-max | -0.045 | -0.015 | +0.0300 | **changed** |
| flat MoG-TSK 2nd / pipeline default | RMSE log+z-score (MPa) | 7.510 ± 0.617 | 6.777 ± 0.473 | -0.7330 | **changed** |
| flat MoG-TSK full-2nd / pipeline default | Δ min-max − raw | +0.057 | +0.030 | -0.0270 | **changed** |
| flat MoG-TSK full-2nd / pipeline default | Δ z-score − raw | +0.020 | -0.021 | -0.0410 | **changed** |
| flat MoG-TSK full-2nd / pipeline default | Δ z-score − min-max | -0.037 | -0.051 | -0.0140 | **changed** |
| mixture of experts / demo-tuned | log + z-score | 0.698 ± 0.033 | 0.806 ± 0.031 | +0.1080 | **changed** |
| mixture of experts / demo-tuned | Δ min-max − raw | +0.059 | +0.060 | +0.0010 | **changed** |
| mixture of experts / demo-tuned | Δ z-score − raw | -0.072 | +0.032 | +0.1040 | **changed** |
| mixture of experts / demo-tuned | Δ z-score − min-max | -0.131 | -0.028 | +0.1030 | **changed** |
| mixture of experts / demo-tuned | RMSE log+z-score (MPa) | 8.969 ± 0.356 | 7.188 ± 0.472 | -1.7810 | **changed** |
| mixture of experts / library default | Δ min-max − raw | +0.071 | +0.108 | +0.0370 | **changed** |
| mixture of experts / library default | Δ z-score − raw | +0.045 | +0.096 | +0.0510 | **changed** |
| mixture of experts / library default | Δ z-score − min-max | -0.026 | -0.012 | +0.0140 | **changed** |
| flat MoG-TSK 1st / pipeline default | raw features | 0.666 ± 0.041 | 0.695 ± 0.030 | +0.0290 | within noise |
| flat MoG-TSK 1st / pipeline default | log + min-max | 0.787 ± 0.026 | 0.796 ± 0.018 | +0.0090 | within noise |
| flat MoG-TSK 1st / pipeline default | RMSE raw (MPa) | 9.439 ± 0.487 | 9.022 ± 0.357 | -0.4170 | within noise |
| flat MoG-TSK 1st / pipeline default | RMSE log+min-max (MPa) | 7.537 ± 0.388 | 7.381 ± 0.339 | -0.1560 | within noise |
| flat MoG-TSK 2nd / pipeline default | raw features | 0.804 ± 0.016 | 0.804 ± 0.030 | +0.0000 | within noise |
| flat MoG-TSK 2nd / pipeline default | log + min-max | 0.832 ± 0.027 | 0.841 ± 0.021 | +0.0090 | within noise |
| flat MoG-TSK 2nd / pipeline default | RMSE raw (MPa) | 7.251 ± 0.457 | 7.217 ± 0.531 | -0.0340 | within noise |
| flat MoG-TSK 2nd / pipeline default | RMSE log+min-max (MPa) | 6.683 ± 0.545 | 6.499 ± 0.427 | -0.1840 | within noise |
| flat MoG-TSK full-2nd / pipeline default | raw features | 0.816 ± 0.052 | 0.830 ± 0.025 | +0.0140 | within noise |
| flat MoG-TSK full-2nd / pipeline default | log + min-max | 0.873 ± 0.020 | 0.861 ± 0.026 | -0.0120 | within noise |
| flat MoG-TSK full-2nd / pipeline default | log + z-score | 0.835 ± 0.036 | 0.809 ± 0.115 | -0.0260 | within noise |
| flat MoG-TSK full-2nd / pipeline default | RMSE raw (MPa) | 6.955 ± 0.810 | 6.719 ± 0.444 | -0.2360 | within noise |
| flat MoG-TSK full-2nd / pipeline default | RMSE log+min-max (MPa) | 5.818 ± 0.419 | 6.072 ± 0.512 | +0.2540 | within noise |
| flat MoG-TSK full-2nd / pipeline default | RMSE log+z-score (MPa) | 6.594 ± 0.718 | 6.905 ± 1.718 | +0.3110 | within noise |
| mixture of experts / demo-tuned | raw features | 0.770 ± 0.035 | 0.774 ± 0.025 | +0.0040 | within noise |
| mixture of experts / demo-tuned | log + min-max | 0.829 ± 0.022 | 0.834 ± 0.027 | +0.0050 | within noise |
| mixture of experts / demo-tuned | RMSE raw (MPa) | 7.835 ± 0.704 | 7.771 ± 0.441 | -0.0640 | within noise |
| mixture of experts / demo-tuned | RMSE log+min-max (MPa) | 6.747 ± 0.432 | 6.645 ± 0.469 | -0.1020 | within noise |
| mixture of experts / library default | raw features | 0.689 ± 0.066 | 0.648 ± 0.093 | -0.0410 | within noise |
| mixture of experts / library default | log + min-max | 0.760 ± 0.060 | 0.756 ± 0.059 | -0.0040 | within noise |
| mixture of experts / library default | log + z-score | 0.734 ± 0.062 | 0.744 ± 0.066 | +0.0100 | within noise |
| mixture of experts / library default | RMSE raw (MPa) | 9.048 ± 0.735 | 9.603 ± 1.207 | +0.5550 | within noise |
| mixture of experts / library default | RMSE log+min-max (MPa) | 7.949 ± 0.975 | 8.024 ± 0.753 | +0.0750 | within noise |
| mixture of experts / library default | RMSE log+z-score (MPa) | 8.380 ± 0.878 | 8.214 ± 0.804 | -0.1660 | within noise |

### `table_norm_conorm_matrix`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete / HME (experts only) / R2 | Best (mean spread) | **hamacher** (spread 4.400) | **hamacher** (spread 4.378) | -0.0220 | **changed** |
| Concrete / HME (experts only) / RMSE (MPa) | Best (mean spread) | **hamacher** (spread 27.615) | **hamacher** (spread 27.566) | -0.0490 | **changed** |
| Concrete / flat MoG-TSK / R2 | min/max | 0.642 ± 0.055 | 0.701 ± 0.048 | +0.0590 | **changed** |
| Concrete / flat MoG-TSK / R2 | hamacher | 0.652 ± 0.054 | 0.708 ± 0.048 | +0.0560 | **changed** |
| Concrete / flat MoG-TSK / R2 | Best (mean spread) | **probability** (spread 4.478) | **hamacher** (spread 4.480) | +0.0020 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | min/max | 9.764 ± 0.844 | 8.902 ± 0.595 | -0.8620 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | hamacher | 9.630 ± 0.799 | 8.809 ± 0.613 | -0.8210 | **changed** |
| Concrete / flat MoG-TSK / RMSE (MPa) | Best (mean spread) | **probability** (spread 26.294) | **hamacher** (spread 26.829) | +0.5350 | **changed** |
| Concrete / HME (experts only) / R2 | min/max | 0.785 ± 0.041 | 0.786 ± 0.025 | +0.0010 | within noise |
| Concrete / HME (experts only) / R2 | probability | 0.781 ± 0.033 | 0.773 ± 0.019 | -0.0080 | within noise |
| Concrete / HME (experts only) / R2 | luk | -3.608 ± 0.479 | -3.583 ± 0.465 | +0.0250 | within noise |
| Concrete / HME (experts only) / R2 | hamacher | 0.792 ± 0.042 | 0.795 ± 0.024 | +0.0030 | within noise |
| Concrete / HME (experts only) / R2 | einstein | 0.774 ± 0.035 | 0.760 ± 0.024 | -0.0140 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | min/max | 7.560 ± 0.828 | 7.547 ± 0.375 | -0.0130 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | probability | 7.641 ± 0.691 | 7.788 ± 0.385 | +0.1470 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | luk | 35.050 ± 1.244 | 34.957 ± 1.253 | -0.0930 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | hamacher | 7.435 ± 0.828 | 7.391 ± 0.366 | -0.0440 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | einstein | 7.771 ± 0.712 | 8.009 ± 0.469 | +0.2380 | within noise |
| Concrete / flat MoG-TSK / R2 | probability | 0.658 ± 0.040 | 0.687 ± 0.049 | +0.0290 | within noise |
| Concrete / flat MoG-TSK / R2 | luk | -3.821 ± 0.501 | -3.772 ± 0.542 | +0.0490 | within noise |
| Concrete / flat MoG-TSK / R2 | einstein | 0.647 ± 0.041 | 0.679 ± 0.044 | +0.0320 | within noise |
| Concrete / flat MoG-TSK / RMSE (MPa) | probability | 9.553 ± 0.498 | 9.122 ± 0.623 | -0.4310 | within noise |
| Concrete / flat MoG-TSK / RMSE (MPa) | luk | 35.846 ± 1.325 | 35.638 ± 1.013 | -0.2080 | within noise |
| Concrete / flat MoG-TSK / RMSE (MPa) | einstein | 9.697 ± 0.399 | 9.246 ± 0.541 | -0.4510 | within noise |

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
