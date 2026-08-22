# Fix impact — `goal-8h-2026-08-11-fullsuite` → `full-2026-08-22`

Cell-by-cell diff of the archived table runs, produced by `reproduce/compare_runs.py`. Every table is listed, including the unchanged ones: confining a fix's blast radius is a claim, and it is only supported by showing the tables that did *not* move.

<details><summary>Provenance — <code>goal-8h-2026-08-11-fullsuite</code></summary>

```

--- backfill 2026-08-12T03:19:08Z: table_a1_feature_scoring table_4_8_mf_dedup ---
tribble-fis: 80e98d755d9649b0bad5c448bab6b88fba468e45
tribble-cluster: 85b68a8a58c004756e8112cca3a3b9b110cf4ffc
grad-school: 20dd460accad3f031e561a8096236670b4cbd4cf
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
  python           Python 3.12.3
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
  table_a1_feature_scoring               ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_4_8_mf_dedup                     ok           seeds=0,1,2,3,4,5,6,7,8,9
  table_3_1_pvat_scaling                 not-run-this-pass seeds=—
  table_3_1_reorder_three_arm            not-run-this-pass seeds=—
  table_3_2_memory_precision             not-run-this-pass seeds=—
  table_3_4_gpu_speedups                 not-run-this-pass seeds=—
  table_3_7_g2_dtw_nonmetric             not-run-this-pass seeds=—
  table_5_x_ch5_selection                not-run-this-pass seeds=—
label:       goal-8h-2026-08-11-fullsuite
generated:   2026-08-12T03:28:55Z

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
## THIS BANNER IS NOT A VERDICT. If an addendum below establishes what ##
## produced these tables, read it: the archive may well be citable.    ##
## Two independent readers stopped here and concluded it was not.      ##
########################################################################

tribble-fis: 80e98d755d9649b0bad5c448bab6b88fba468e45
tribble-cluster: 85b68a8a58c004756e8112cca3a3b9b110cf4ffc
grad-school: 20dd460accad3f031e561a8096236670b4cbd4cf
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
  python           Python 3.12.3
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
  table_4_8_mf_dedup                     archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_pvat_scaling                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_reorder_three_arm            archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_2_memory_precision             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_4_gpu_speedups                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_7_g2_dtw_nonmetric             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_5_x_ch5_selection                archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
label:       goal-8h-2026-08-11-fullsuite
generated:   2026-08-12T05:01:48Z

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
## THIS BANNER IS NOT A VERDICT. If an addendum below establishes what ##
## produced these tables, read it: the archive may well be citable.    ##
## Two independent readers stopped here and concluded it was not.      ##
########################################################################

tribble-fis: 80e98d755d9649b0bad5c448bab6b88fba468e45
tribble-cluster: 85b68a8a58c004756e8112cca3a3b9b110cf4ffc
grad-school: 20dd460accad3f031e561a8096236670b4cbd4cf
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
  python           Python 3.12.3
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
  table_4_8_mf_dedup                     archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_pvat_scaling                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_reorder_three_arm            archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_2_memory_precision             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_4_gpu_speedups                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_7_g2_dtw_nonmetric             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_5_x_ch5_selection                archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
label:       goal-8h-2026-08-11-fullsuite
generated:   2026-08-12T07:21:32Z

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
## THIS BANNER IS NOT A VERDICT. If an addendum below establishes what ##
## produced these tables, read it: the archive may well be citable.    ##
## Two independent readers stopped here and concluded it was not.      ##
########################################################################

tribble-fis: 80e98d755d9649b0bad5c448bab6b88fba468e45
tribble-cluster: 85b68a8a58c004756e8112cca3a3b9b110cf4ffc
grad-school: 20dd460accad3f031e561a8096236670b4cbd4cf
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
  python           Python 3.12.3
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
  table_4_8_mf_dedup                     archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_pvat_scaling                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_reorder_three_arm            archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_2_memory_precision             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_4_gpu_speedups                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_7_g2_dtw_nonmetric             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_5_x_ch5_selection                archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
label:       goal-8h-2026-08-11-fullsuite
generated:   2026-08-12T08:42:22Z

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
## THIS BANNER IS NOT A VERDICT. If an addendum below establishes what ##
## produced these tables, read it: the archive may well be citable.    ##
## Two independent readers stopped here and concluded it was not.      ##
########################################################################

tribble-fis: 80e98d755d9649b0bad5c448bab6b88fba468e45
tribble-cluster: 85b68a8a58c004756e8112cca3a3b9b110cf4ffc
grad-school: 20dd460accad3f031e561a8096236670b4cbd4cf
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
  python           Python 3.12.3
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
  table_4_8_mf_dedup                     archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_pvat_scaling                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_reorder_three_arm            archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_2_memory_precision             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_4_gpu_speedups                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_7_g2_dtw_nonmetric             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_5_x_ch5_selection                archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
label:       goal-8h-2026-08-11-fullsuite
generated:   2026-08-12T10:49:48Z

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
## THIS BANNER IS NOT A VERDICT. If an addendum below establishes what ##
## produced these tables, read it: the archive may well be citable.    ##
## Two independent readers stopped here and concluded it was not.      ##
########################################################################

tribble-fis: 80e98d755d9649b0bad5c448bab6b88fba468e45
tribble-cluster: 85b68a8a58c004756e8112cca3a3b9b110cf4ffc
grad-school: 20dd460accad3f031e561a8096236670b4cbd4cf
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
  python           Python 3.12.3
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
  table_4_8_mf_dedup                     archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_pvat_scaling                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_reorder_three_arm            archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_2_memory_precision             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_4_gpu_speedups                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_7_g2_dtw_nonmetric             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_5_x_ch5_selection                archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
label:       goal-8h-2026-08-11-fullsuite
generated:   2026-08-12T11:29:35Z

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
## THIS BANNER IS NOT A VERDICT. If an addendum below establishes what ##
## produced these tables, read it: the archive may well be citable.    ##
## Two independent readers stopped here and concluded it was not.      ##
########################################################################

tribble-fis: 80e98d755d9649b0bad5c448bab6b88fba468e45
tribble-cluster: 85b68a8a58c004756e8112cca3a3b9b110cf4ffc
grad-school: 42438f7bdb1a5f29e8ba06f26c1ecd9580d08b37
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
  python           Python 3.12.3
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
  table_4_8_mf_dedup                     archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_pvat_scaling                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_1_reorder_three_arm            archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_2_memory_precision             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_4_gpu_speedups                 archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_3_7_g2_dtw_nonmetric             archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
  table_5_x_ch5_selection                archived-not-run seeds=0,1,2,3,4,5,6,7,8,9 (not verified -- see the notice in PROVENANCE.txt)
```

</details>

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
```

</details>

## Summary

| Table | Cells | Verdict |
|---|---:|---|
| `table_3_1` | 16 | 10 timing |
| `table_3_1_complexity_fit` | 89 | **3 changed**, 25 timing |
| `table_3_1_three_arm` | 56 | 41 timing |
| `table_3_2_memory_precision` | 32 | **2 changed** |
| `table_3_4_gpu_speedups` | 195 | **37 changed**, 31 within noise, 31 timing |
| `table_3_7_g2_downstream` | 21 | identical |
| `table_3_7_g2_dtw_nonmetric` | 5 | **1 changed**, 2 rows removed |
| `table_4_1` | 17 | **2 changed**, 3 within noise, 5 timing |
| `table_4_4_openset` | 9 | identical |
| `table_4_4b_theta_sweep` | 28 | identical |
| `table_4_8_mf_dedup` | 30 | **24 changed**, 4 timing |
| `table_4_8_mf_dedup_sweep` | 280 | **126 changed**, 70 within noise |
| `table_5_1_battery` | 34 | identical |
| `table_5_2_multiscale` | 15 | identical |
| `table_5_3_selection` | 15 | identical |
| `table_5_4_ch5_g1_scaling` | 126 | identical |
| `table_5_4_ch5_g1_scaling_raw` | 1800 | identical |
| `table_6_1` | 16 | **5 changed**, 4 within noise |
| `table_a1_feature_ranking` | 20 | **12 changed**, 3 within noise |
| `table_a2_feature_count` | 36 | **26 changed**, 1 within noise |
| `table_a7_regression_scale` | 32 | identical |
| `table_concrete_reconciliation` | 34 | **1 changed**, 17 within noise |
| `table_g5_output_partitioning` | 189 | **11 changed**, 97 within noise |
| `table_g5b_skew_sweep` | 48 | **3 changed**, 8 within noise |
| `table_hyperparam_normalization` | 84 | **18 changed**, 30 within noise |
| `table_norm_conorm_matrix` | 39 | **20 changed**, 4 within noise |

## What moved

### `table_3_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,024 | classical VAT (s) | 19.284 ± 2.685 s | 13.387 ± 0.234 s | -5.8970 | timing |
| 1,024 | pVAT (s) | 0.023 ± 0.002 s | 0.022 ± 0.003 s | -0.0010 | timing |
| 1,024 | speedup | 827x | 604x | -223.0000 | timing |
| 2,048 / infeasible (>cap) | pVAT (s) | 0.079 ± 0.004 s | 0.076 ± 0.007 s | -0.0030 | timing |
| 256 | classical VAT (s) | 0.322 ± 0.053 s | 0.264 ± 0.012 s | -0.0580 | timing |
| 256 | pVAT (s) | 0.439 ± 1.310 s | 0.012 ± 0.031 s | -0.4270 | timing |
| 256 | speedup | 1x | 21x | +20.0000 | timing |
| 4,096 / infeasible (>cap) | pVAT (s) | 0.255 ± 0.006 s | 0.229 ± 0.007 s | -0.0260 | timing |
| 512 | classical VAT (s) | 2.189 ± 0.166 s | 1.824 ± 0.048 s | -0.3650 | timing |
| 512 | speedup | 329x | 275x | -54.0000 | timing |

### `table_3_1_complexity_fit`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| fitted exponent /  | classical | **3.15** (6 pts) | **3.11** (6 pts) | -0.0400 | **changed** |
| fitted exponent /  | stage 1 | **1.81** (11 pts) | **1.82** (11 pts) | +0.0100 | **changed** |
| fitted exponent /  | stage 2 | **1.89** (11 pts) | **1.77** (11 pts) | -0.1200 | **changed** |
| 1,000 / 10.0× | classical | 1294.66× | 1269.20× | -25.4600 | timing |
| 1,000 / 10.0× | stage 1 | 69.16× | 57.76× | -11.4000 | timing |
| 1,000 / 10.0× | stage 2 | 60.63× | 61.41× | +0.7800 | timing |
| 1,250 / 12.5× / N/A | stage 1 | 105.90× | 98.24× | -7.6600 | timing |
| 1,250 / 12.5× / N/A | stage 2 | 87.90× | 96.94× | +9.0400 | timing |
| 1,500 / 15.0× / N/A | stage 1 | 144.62× | 128.30× | -16.3200 | timing |
| 1,500 / 15.0× / N/A | stage 2 | 134.00× | 108.12× | -25.8800 | timing |
| 2,000 / 20.0× / N/A | stage 1 | 202.42× | 215.90× | +13.4800 | timing |
| 2,000 / 20.0× / N/A | stage 2 | 245.60× | 191.03× | -54.5700 | timing |
| 2,500 / 25.0× / N/A | stage 1 | 314.03× | 328.23× | +14.2000 | timing |
| 2,500 / 25.0× / N/A | stage 2 | 476.30× | 316.37× | -159.9300 | timing |
| 200 / 2.0× | classical | 7.46× | 7.51× | +0.0500 | timing |
| 200 / 2.0× | stage 1 | 3.25× | 3.23× | -0.0200 | timing |
| 200 / 2.0× | stage 2 | 3.18× | 3.61× | +0.4300 | timing |
| 3,000 / 30.0× / N/A | stage 1 | 415.18× | 459.00× | +43.8200 | timing |
| 3,000 / 30.0× / N/A | stage 2 | 640.40× | 462.68× | -177.7200 | timing |
| 300 / 3.0× | classical | 26.10× | 26.45× | +0.3500 | timing |
| 300 / 3.0× | stage 1 | 6.24× | 6.17× | -0.0700 | timing |
| 300 / 3.0× | stage 2 | 6.74× | 7.42× | +0.6800 | timing |
| 500 / 5.0× | classical | 130.50× | 132.82× | +2.3200 | timing |
| 500 / 5.0× | stage 1 | 17.13× | 16.81× | -0.3200 | timing |
| 500 / 5.0× | stage 2 | 20.49× | 20.33× | -0.1600 | timing |
| 750 / 7.5× | classical | 576.37× | 484.06× | -92.3100 | timing |
| 750 / 7.5× | stage 1 | 43.54× | 34.53× | -9.0100 | timing |
| 750 / 7.5× | stage 2 | 40.64× | 42.67× | +2.0300 | timing |

### `table_3_1_three_arm`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1,000 | classical O(N³) (s) | 0.1742 ± 0.0029 s | 0.1317 ± 0.0035 s | -0.0425 | timing |
| 1,000 | stage 1 O(N²logN) (s) | 0.0136 ± 0.0009 s | 0.0089 ± 0.0003 s | -0.0047 | timing |
| 1,000 | stage 2 O(N²) (s) | 0.0008 ± 0.0001 s | 0.0010 ± 0.0001 s | +0.0002 | timing |
| 1,000 | cls/s2 | 213.0× | 131.0× | -82.0000 | timing |
| 1,000 | s1/s2 | 16.6× | 8.9× | -7.7000 | timing |
| 1,250 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0208 ± 0.0022 s | 0.0152 ± 0.0026 s | -0.0056 | timing |
| 1,250 / not run (> cap) | stage 2 O(N²) (s) | 0.0012 ± 0.0001 s | 0.0016 ± 0.0002 s | +0.0004 | timing |
| 1,250 / not run (> cap) | s1/s2 | 17.6× | 9.5× | -8.1000 | timing |
| 1,500 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0284 ± 0.0016 s | 0.0198 ± 0.0006 s | -0.0086 | timing |
| 1,500 / not run (> cap) | stage 2 O(N²) (s) | 0.0018 ± 0.0001 s | 0.0018 ± 0.0002 s | +0.0000 | timing |
| 1,500 / not run (> cap) | s1/s2 | 15.7× | 11.2× | -4.5000 | timing |
| 100 / 0.0001 ± 0.0000 s / 0.0002 ± 0.0000 s | cls/s2 | 10.0× | 6.3× | -3.7000 | timing |
| 100 / 0.0001 ± 0.0000 s / 0.0002 ± 0.0000 s | s1/s2 | 14.6× | 9.4× | -5.2000 | timing |
| 2,000 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0398 ± 0.0016 s | 0.0333 ± 0.0007 s | -0.0065 | timing |
| 2,000 / not run (> cap) | stage 2 O(N²) (s) | 0.0033 ± 0.0003 s | 0.0031 ± 0.0002 s | -0.0002 | timing |
| 2,000 / not run (> cap) | s1/s2 | 12.0× | 10.6× | -1.4000 | timing |
| 2,500 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0618 ± 0.0016 s | 0.0506 ± 0.0010 s | -0.0112 | timing |
| 2,500 / not run (> cap) | stage 2 O(N²) (s) | 0.0064 ± 0.0008 s | 0.0052 ± 0.0003 s | -0.0012 | timing |
| 2,500 / not run (> cap) | s1/s2 | 9.6× | 9.8× | +0.2000 | timing |
| 200 | classical O(N³) (s) | 0.0010 ± 0.0000 s | 0.0008 ± 0.0000 s | -0.0002 | timing |
| 200 | stage 1 O(N²logN) (s) | 0.0006 ± 0.0000 s | 0.0005 ± 0.0000 s | -0.0001 | timing |
| 200 | stage 2 O(N²) (s) | 0.0000 ± 0.0000 s | 0.0001 ± 0.0000 s | +0.0001 | timing |
| 200 | cls/s2 | 23.4× | 13.2× | -10.2000 | timing |
| 200 | s1/s2 | 14.9× | 8.4× | -6.5000 | timing |
| 3,000 / not run (> cap) | stage 1 O(N²logN) (s) | 0.0817 ± 0.0007 s | 0.0708 ± 0.0015 s | -0.0109 | timing |
| 3,000 / not run (> cap) | stage 2 O(N²) (s) | 0.0086 ± 0.0004 s | 0.0076 ± 0.0004 s | -0.0010 | timing |
| 3,000 / not run (> cap) | s1/s2 | 9.5× | 9.3× | -0.2000 | timing |
| 300 | classical O(N³) (s) | 0.0035 ± 0.0000 s | 0.0027 ± 0.0001 s | -0.0008 | timing |
| 300 | stage 1 O(N²logN) (s) | 0.0012 ± 0.0001 s | 0.0010 ± 0.0001 s | -0.0002 | timing |
| 300 | cls/s2 | 38.7× | 22.6× | -16.1000 | timing |
| 300 | s1/s2 | 13.5× | 7.8× | -5.7000 | timing |
| 500 | classical O(N³) (s) | 0.0176 ± 0.0008 s | 0.0138 ± 0.0003 s | -0.0038 | timing |
| 500 | stage 1 O(N²logN) (s) | 0.0034 ± 0.0001 s | 0.0026 ± 0.0001 s | -0.0008 | timing |
| 500 | stage 2 O(N²) (s) | 0.0003 ± 0.0001 s | 0.0003 ± 0.0000 s | +0.0000 | timing |
| 500 | cls/s2 | 63.5× | 41.4× | -22.1000 | timing |
| 500 | s1/s2 | 12.2× | 7.8× | -4.4000 | timing |
| 750 | classical O(N³) (s) | 0.0776 ± 0.0098 s | 0.0502 ± 0.0012 s | -0.0274 | timing |
| 750 | stage 1 O(N²logN) (s) | 0.0086 ± 0.0010 s | 0.0053 ± 0.0002 s | -0.0033 | timing |
| 750 | stage 2 O(N²) (s) | 0.0005 ± 0.0001 s | 0.0007 ± 0.0001 s | +0.0002 | timing |
| 750 | cls/s2 | 141.5× | 71.9× | -69.6000 | timing |
| 750 | s1/s2 | 15.6× | 7.6× | -8.0000 | timing |

### `table_3_2_memory_precision`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| float32 / on-demand (vat_prim_mst_seq) — DEFECTIVE / 4 | ordering vs float64 (N=2,000) | 0.001 ± 0.001 | 0.999 ± 0.002 | +0.9980 | **changed** |
| float64 / on-demand (vat_prim_mst_seq) — DEFECTIVE / 8 | ordering vs float64 (N=2,000) | 0.001 ± 0.001 | 1.000 (exact) | +0.9990 | **changed** |

### `table_3_4_gpu_speedups`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Boruvka MST (device) / N=16,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.29600 ± 0.05433 | 0.21282 ± 0.01261 | -0.0832 | **changed** |
| Boruvka MST (device) / N=16,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.03502 ± 0.00455 | 0.04792 ± 0.00386 | +0.0129 | **changed** |
| Boruvka MST (device) / N=32,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 1.00111 ± 0.08641 | 0.82615 ± 0.04086 | -0.1750 | **changed** |
| Boruvka MST (device) / N=32,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.13304 ± 0.00757 | 0.19872 ± 0.01434 | +0.0657 | **changed** |
| Boruvka MST (device) / N=4,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.01932 ± 0.00224 | 0.01453 ± 0.00123 | -0.0048 | **changed** |
| Boruvka MST (device) / N=4,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.00321 ± 0.00020 | 0.00351 ± 0.00019 | +0.0003 | **changed** |
| Boruvka MST (device) / N=8,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | CPU (s) | 0.07837 ± 0.01227 | 0.06229 ± 0.01309 | -0.0161 | **changed** |
| Boruvka MST (device) / N=8,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU (s) | 0.00875 ± 0.00049 | 0.01191 ± 0.00157 | +0.0032 | **changed** |
| Pairwise distances / N=16,000, d=10, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.22048 ± 0.01976 | 0.18541 ± 0.01037 | -0.0351 | **changed** |
| Pairwise distances / N=16,000, d=10, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.21287 ± 0.01740 | 0.18467 ± 0.01001 | -0.0282 | **changed** |
| Pairwise distances / N=16,000, d=200, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 1.81962 ± 0.09180 | 0.65316 ± 0.03561 | -1.1665 | **changed** |
| Pairwise distances / N=16,000, d=200, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 1.71022 ± 0.02005 | 0.65686 ± 0.01877 | -1.0534 | **changed** |
| Pairwise distances / N=16,000, d=200, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.52900 ± 0.04712 | 1.64433 ± 0.04044 | +0.1153 | **changed** |
| Pairwise distances / N=16,000, d=200, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | exactness vs CPU | max abs Δ = 0.0e+00 | max abs Δ = 3.8e-06 | +0.0000 | **changed** |
| Pairwise distances / N=16,000, d=200, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 1.84730 ± 0.11711 | 0.84519 ± 0.04621 | -1.0021 | **changed** |
| Pairwise distances / N=16,000, d=50, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.52196 ± 0.03569 | 0.25823 ± 0.01437 | -0.2637 | **changed** |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.47682 ± 0.01414 | 0.26561 ± 0.00892 | -0.2112 | **changed** |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | exactness vs CPU | max abs Δ = 0.0e+00 | max abs Δ = 1.9e-06 | +0.0000 | **changed** |
| Pairwise distances / N=16,000, d=50, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.45056 ± 0.02828 | 0.34170 ± 0.01724 | -0.1089 | **changed** |
| Pairwise distances / N=16,000, d=784, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 7.42532 ± 0.26797 | 2.77787 ± 0.17178 | -4.6475 | **changed** |
| Pairwise distances / N=16,000, d=784, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.68712 ± 0.05318 | 2.39694 ± 0.04507 | +0.7098 | **changed** |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 6.84382 ± 0.22974 | 2.68372 ± 0.07645 | -4.1601 | **changed** |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 3.99387 ± 0.04294 | 5.09463 ± 0.04649 | +1.1008 | **changed** |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | exactness vs CPU | max abs Δ = 0.0e+00 | max abs Δ = 7.6e-06 | +0.0000 | **changed** |
| Pairwise distances / N=16,000, d=784, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 8.47066 ± 0.57599 | 4.81571 ± 0.15468 | -3.6550 | **changed** |
| Pairwise distances / N=16,000, d=784, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 11.14590 ± 0.07999 | 13.19158 ± 0.09859 | +2.0457 | **changed** |
| VAT front end / N=16,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 1.50607 ± 0.08714 | 1.24742 ± 0.03071 | -0.2587 | **changed** |
| VAT front end / N=16,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 0.63502 ± 0.07445 | 0.50816 ± 0.02025 | -0.1269 | **changed** |
| VAT front end / N=32,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.53028 ± 0.02350 | 0.62908 ± 0.03835 | +0.0988 | **changed** |
| VAT front end / N=32,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 2.67222 ± 0.13278 | 2.43062 ± 0.05290 | -0.2416 | **changed** |
| VAT front end / N=32,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.53028 ± 0.02350 | 0.62908 ± 0.03835 | +0.0988 | **changed** |
| VAT front end / N=4,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 0.09733 ± 0.00395 | 0.08745 ± 0.00560 | -0.0099 | **changed** |
| VAT front end / N=4,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 0.04157 ± 0.00387 | 0.03532 ± 0.00199 | -0.0063 | **changed** |
| VAT front end / N=48,000, float32, 9.22 GB resident / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 5.16635 | 4.39892 | -0.7674 | **changed** |
| VAT front end / N=48,000, float32, 9.22 GB resident / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 1.21737 | 1.36464 | +0.1473 | **changed** |
| VAT front end / N=8,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 0.36403 ± 0.01075 | 0.30113 ± 0.01017 | -0.0629 | **changed** |
| VAT front end / N=8,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | CPU (s) | 0.14930 ± 0.01292 | 0.13099 ± 0.01397 | -0.0183 | **changed** |
| Boruvka MST (device) / N=16,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 8.45x | 4.44x | -4.0100 | timing |
| Boruvka MST (device) / N=32,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 7.52x | 4.16x | -3.3600 | timing |
| Boruvka MST (device) / N=4,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 6.02x | 4.14x | -1.8800 | timing |
| Boruvka MST (device) / N=8,000, float64, matrix resident / pcvat.vat_prim_mst_c (Cython dense Prim) | GPU speedup (CPU/GPU) | 8.95x | 5.23x | -3.7200 | timing |
| Fuzzy C-Means / N=200,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | CPU (s) | 2.10697 ± 2.06088 | 1.93163 ± 1.94163 | -0.1753 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU (s) | 0.36445 ± 0.17667 | 0.32031 ± 0.22633 | -0.0441 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU speedup (CPU/GPU) | 5.78x | 6.03x | +0.2500 | timing |
| Fuzzy C-Means / N=200,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | CPU (s) | 1.04398 ± 1.02096 | 0.96767 ± 0.94794 | -0.0763 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU (s) | 0.36445 ± 0.17667 | 0.32031 ± 0.22633 | -0.0441 | within noise |
| Fuzzy C-Means / N=200,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU speedup (CPU/GPU) | 2.86x | 3.02x | +0.1600 | timing |
| Fuzzy C-Means / N=50,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | CPU (s) | 0.52707 ± 0.47069 | 0.40936 ± 0.35122 | -0.1177 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU (s) | 0.10732 ± 0.10040 | 0.10211 ± 0.10776 | -0.0052 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU speedup (CPU/GPU) | 4.91x | 4.01x | -0.9000 | timing |
| Fuzzy C-Means / N=50,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | CPU (s) | 0.28343 ± 0.23133 | 0.20753 ± 0.17465 | -0.0759 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU (s) | 0.10732 ± 0.10040 | 0.10211 ± 0.10776 | -0.0052 | within noise |
| Fuzzy C-Means / N=50,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU speedup (CPU/GPU) | 2.64x | 2.03x | -0.6100 | timing |
| Fuzzy C-Means / N=500,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | CPU (s) | 5.85392 ± 5.34906 | 5.38160 ± 4.82153 | -0.4723 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU (s) | 0.72417 ± 0.36976 | 0.96259 ± 0.91236 | +0.2384 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20 / fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm) | GPU speedup (CPU/GPU) | 8.08x | 5.59x | -2.4900 | timing |
| Fuzzy C-Means / N=500,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | CPU (s) | 3.05858 ± 3.15882 | 2.74843 ± 2.35473 | -0.3102 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU (s) | 0.72417 ± 0.36976 | 0.96259 ± 0.91236 | +0.2384 | within noise |
| Fuzzy C-Means / N=500,000, k=10, d=20, MATCHED formulation / gram + 2 GEMM in NumPy/BLAS (this file) | GPU speedup (CPU/GPU) | 4.22x | 2.86x | -1.3600 | timing |
| Pairwise distances / N=16,000, d=10, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.57900 ± 0.06240 | 0.53520 ± 0.04155 | -0.0438 | within noise |
| Pairwise distances / N=16,000, d=10, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.38x | 0.35x | -0.0300 | timing |
| Pairwise distances / N=16,000, d=10, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.66357 ± 0.02350 | 0.64196 ± 0.07358 | -0.0216 | within noise |
| Pairwise distances / N=16,000, d=10, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.32x | 0.29x | -0.0300 | timing |
| Pairwise distances / N=16,000, d=10, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | CPU (s) | 0.27826 ± 0.01092 | 0.27990 ± 0.01107 | +0.0016 | within noise |
| Pairwise distances / N=16,000, d=10, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.00577 ± 0.07712 | 0.95681 ± 0.08161 | -0.0490 | within noise |
| Pairwise distances / N=16,000, d=10, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.28x | 0.29x | +0.0100 | timing |
| Pairwise distances / N=16,000, d=200, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.86026 ± 0.06709 | 0.88349 ± 0.03599 | +0.0232 | within noise |
| Pairwise distances / N=16,000, d=200, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 2.12x | 0.74x | -1.3800 | timing |
| Pairwise distances / N=16,000, d=200, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 1.12x | 0.40x | -0.7200 | timing |
| Pairwise distances / N=16,000, d=200, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.60775 ± 0.12904 | 1.66698 ± 0.03414 | +0.0592 | within noise |
| Pairwise distances / N=16,000, d=200, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 1.15x | 0.51x | -0.6400 | timing |
| Pairwise distances / N=16,000, d=200, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | exactness vs CPU | max abs Δ = 2.8e-14 | max abs Δ = 7.1e-14 | +0.0000 | within noise |
| Pairwise distances / N=16,000, d=50, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.64499 ± 0.04394 | 0.62895 ± 0.02592 | -0.0160 | within noise |
| Pairwise distances / N=16,000, d=50, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.81x | 0.41x | -0.4000 | timing |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 0.90784 ± 0.03621 | 0.88194 ± 0.02942 | -0.0259 | within noise |
| Pairwise distances / N=16,000, d=50, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.53x | 0.30x | -0.2300 | timing |
| Pairwise distances / N=16,000, d=50, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU (s) | 1.14701 ± 0.06223 | 1.14066 ± 0.03591 | -0.0064 | within noise |
| Pairwise distances / N=16,000, d=50, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.39x | 0.30x | -0.0900 | timing |
| Pairwise distances / N=16,000, d=50, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | exactness vs CPU | max abs Δ = 1.4e-14 | max abs Δ = 2.8e-14 | +0.0000 | within noise |
| Pairwise distances / N=16,000, d=784, float32, fast (native acc) / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 4.40x | 1.16x | -3.2400 | timing |
| Pairwise distances / N=16,000, d=784, float32, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 1.71x | 0.53x | -1.1800 | timing |
| Pairwise distances / N=16,000, d=784, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | GPU speedup (CPU/GPU) | 0.76x | 0.37x | -0.3900 | timing |
| Pairwise distances / N=16,000, d=784, float64, high_precision / pcvat.pairwise_distances_c (C/OpenMP) | exactness vs CPU | max abs Δ = 4.3e-14 | max abs Δ = 2.8e-13 | +0.0000 | within noise |
| VAT front end / N=16,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.15583 ± 0.01252 | 0.16774 ± 0.01204 | +0.0119 | within noise |
| VAT front end / N=16,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 9.66x | 7.44x | -2.2200 | timing |
| VAT front end / N=16,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.15583 ± 0.01252 | 0.16774 ± 0.01204 | +0.0119 | within noise |
| VAT front end / N=16,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 4.08x | 3.03x | -1.0500 | timing |
| VAT front end / N=32,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | CPU (s) | 6.98682 ± 1.22449 | 5.80313 ± 0.13967 | -1.1837 | within noise |
| VAT front end / N=32,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 13.18x | 9.22x | -3.9600 | timing |
| VAT front end / N=32,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 5.04x | 3.86x | -1.1800 | timing |
| VAT front end / N=4,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.01661 ± 0.00048 | 0.01573 ± 0.00425 | -0.0009 | within noise |
| VAT front end / N=4,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 5.86x | 5.56x | -0.3000 | timing |
| VAT front end / N=4,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.01661 ± 0.00048 | 0.01573 ± 0.00425 | -0.0009 | within noise |
| VAT front end / N=4,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 2.50x | 2.25x | -0.2500 | timing |
| VAT front end / N=48,000, float32, 9.22 GB resident / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 4.24x | 3.22x | -1.0200 | timing |
| VAT front end / N=8,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU (s) | 0.04268 ± 0.00529 | 0.04601 ± 0.00911 | +0.0033 | within noise |
| VAT front end / N=8,000, float64, UNMATCHED work / pairwise_distances_c + compute_vat_c (also reorders D) | GPU speedup (CPU/GPU) | 8.53x | 6.54x | -1.9900 | timing |
| VAT front end / N=8,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU (s) | 0.04268 ± 0.00529 | 0.04601 ± 0.00911 | +0.0033 | within noise |
| VAT front end / N=8,000, float64, order only / pairwise_distances_c + vat_prim_mst_c | GPU speedup (CPU/GPU) | 3.50x | 2.85x | -0.6500 | timing |

### `table_3_7_g2_dtw_nonmetric`

Rows only in `goal-8h-2026-08-11-fullsuite`: `Crop (DTW, N=24000) / no / 23.6%`, `FordA (DTW, N=4921) / no / 0.4%`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| ECG5000 (DTW, N=5000) / no / 20.9% | Timing | 631s matrix + 0.3s reorder | 600s matrix + 0.2s reorder | -31.0000 | **changed** |

### `table_4_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| PhiUSIIL (classification) | MoG accuracy / R2 | acc=0.997 ± 0.001 | acc=0.729 ± 0.023 | -0.2680 | **changed** |
| RT-IOT2022 (12-class) | MoG accuracy / R2 | acc=0.927 ± 0.002 | acc=0.500 ± 0.244 | -0.4270 | **changed** |
| Bike Sharing (regression) | MoG train time | 0.82 ± 0.03 s | 0.11 ± 0.00 s | -0.7100 | timing |
| Bike Sharing (regression) | MoG accuracy / R2 | R2=0.939 ± 0.004 | R2=0.965 ± 0.001 | +0.0000 | within noise |
| Concrete (regression) | MoG train time | 0.41 ± 0.02 s | 0.06 ± 0.00 s | -0.3500 | timing |
| Concrete (regression) | MoG accuracy / R2 | R2=0.795 ± 0.025 | R2=0.808 ± 0.030 | +0.0000 | within noise |
| Concrete (regression, full 2nd order) | MoG train time | 0.42 ± 0.01 s | 0.07 ± 0.00 s | -0.3500 | timing |
| Concrete (regression, full 2nd order) | MoG accuracy / R2 | R2=0.852 ± 0.030 | R2=0.867 ± 0.031 | +0.0000 | within noise |
| PhiUSIIL (classification) | MoG train time | 0.28 ± 0.03 s | 0.17 ± 0.01 s | -0.1100 | timing |
| RT-IOT2022 (12-class) | MoG train time | 37.42 ± 0.64 s | 33.22 ± 0.12 s | -4.2000 | timing |

### `table_4_8_mf_dedup`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| BreastCancer / classification | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification | MF @ 1x (Δ) | 11.1 (+0.0000 ± 0.0000 acc) | 15.3 (-0.0088 ± 0.0316 acc) | +4.2000 | **changed** |
| BreastCancer / classification | Reduction @ 1x | 0.0% | 23.1% | +23.1000 | **changed** |
| BreastCancer / classification | MF @ max-lossless (Δ) | 11.1 (+0.0000 ± 0.0000 acc) | 10.5 (-0.0018 ± 0.0217 acc) | -0.6000 | **changed** |
| BreastCancer / classification | Reduction @ max-lossless | 0.0% | 47.2% | +47.2000 | **changed** |
| Concrete / regression | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression | MF @ 1x (Δ) | 62.6 (+0.0000 ± 0.0000 R²) | 33.1 (-0.0000 ± 0.0000 R²) | -29.5000 | **changed** |
| Concrete / regression | Reduction @ 1x | 0.9% | 2.4% | +1.5000 | **changed** |
| Concrete / regression | MF @ max-lossless (Δ) | 58.4 (-0.0118 ± 0.0305 R²) | 32.0 (-0.0002 ± 0.0022 R²) | -26.4000 | **changed** |
| Concrete / regression | Reduction @ max-lossless | 7.6% | 5.6% | -2.0000 | **changed** |
| Diabetes / regression | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression | MF @ 1x (Δ) | 36.2 (+0.0059 ± 0.0167 R²) | 35.2 (+0.0060 ± 0.0177 R²) | -1.0000 | **changed** |
| Diabetes / regression | Reduction @ 1x | 13.0% | 13.3% | +0.3000 | **changed** |
| Diabetes / regression | MF @ max-lossless (Δ) | 31.4 (+0.0011 ± 0.0168 R²) | 14.4 (-0.3197 ± 0.7944 R²) | -17.0000 | **changed** |
| Diabetes / regression | Reduction @ max-lossless | 24.5% | 64.5% | +40.0000 | **changed** |
| Digits / classification | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification | MF @ 1x (Δ) | 157.5 (+0.0009 ± 0.0028 acc) | 135.6 (-0.0002 ± 0.0010 acc) | -21.9000 | **changed** |
| Digits / classification | Reduction @ 1x | 8.5% | 18.9% | +10.4000 | **changed** |
| Digits / classification | MF @ max-lossless (Δ) | 96.1 (-0.0117 ± 0.0211 acc) | 129.1 (+0.0011 ± 0.0034 acc) | +33.0000 | **changed** |
| Digits / classification | Reduction @ max-lossless | 44.2% | 22.8% | -21.4000 | **changed** |
| Wine / classification | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification | MF @ 1x (Δ) | 16.6 (+0.0000 ± 0.0000 acc) | 16.2 (+0.0000 ± 0.0000 acc) | -0.4000 | **changed** |
| Wine / classification | MF @ max-lossless (Δ) | 14.6 (-0.0037 ± 0.0161 acc) | 15.8 (+0.0000 ± 0.0000 acc) | +1.2000 | **changed** |
| Wine / classification | Reduction @ max-lossless | 12.0% | 2.5% | -9.5000 | **changed** |
| BreastCancer / classification | Max-lossless × | 5× | 3× | -2.0000 | timing |
| Diabetes / regression | Max-lossless × | 2× | 7× | +5.0000 | timing |
| Digits / classification | Max-lossless × | 7× | 2× | -5.0000 | timing |
| Wine / classification | Max-lossless × | 10× | 7× | -3.0000 | timing |

### `table_4_8_mf_dedup_sweep`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| BreastCancer / classification / 0.1 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 0.1 | Dedup MF (mean±std) | 11.10 ± 0.30 | 19.90 ± 0.54 | +8.8000 | **changed** |
| BreastCancer / classification / 0.3 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 0.3 | Dedup MF (mean±std) | 11.10 ± 0.30 | 19.60 ± 0.66 | +8.5000 | **changed** |
| BreastCancer / classification / 1 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 1 | Dedup MF (mean±std) | 11.10 ± 0.30 | 15.30 ± 1.19 | +4.2000 | **changed** |
| BreastCancer / classification / 10 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 10 | Dedup MF (mean±std) | 10.20 ± 0.60 | 4.10 ± 0.30 | -6.1000 | **changed** |
| BreastCancer / classification / 10 | Delta (mean±std) | -0.00526 ± 0.00409 | -0.27485 ± 0.11693 | -0.2696 | **changed** |
| BreastCancer / classification / 100 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 100 | Delta (mean±std) | -0.56257 ± 0.01354 | -0.50351 ± 0.01600 | +0.0591 | **changed** |
| BreastCancer / classification / 15 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 15 | Dedup MF (mean±std) | 9.20 ± 0.40 | 3.90 ± 0.30 | -5.3000 | **changed** |
| BreastCancer / classification / 15 | Delta (mean±std) | +0.00117 ± 0.00819 | -0.25380 ± 0.09765 | -0.2550 | **changed** |
| BreastCancer / classification / 2 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 20 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 20 | Dedup MF (mean±std) | 8.80 ± 0.40 | 1.90 ± 0.30 | -6.9000 | **changed** |
| BreastCancer / classification / 20 | Delta (mean±std) | -0.00409 ± 0.02157 | -0.31404 ± 0.14547 | -0.3100 | **changed** |
| BreastCancer / classification / 3 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 30 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 30 | Dedup MF (mean±std) | 7.60 ± 0.49 | 1.60 ± 0.49 | -6.0000 | **changed** |
| BreastCancer / classification / 30 | Delta (mean±std) | +0.00351 ± 0.00468 | -0.33801 ± 0.16305 | -0.3415 | **changed** |
| BreastCancer / classification / 5 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 5 | Dedup MF (mean±std) | 11.10 ± 0.30 | 6.80 ± 1.17 | -4.3000 | **changed** |
| BreastCancer / classification / 5 | Delta (mean±std) | +0.00000 ± 0.00000 | -0.04912 ± 0.04211 | -0.0491 | **changed** |
| BreastCancer / classification / 50 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 50 | Dedup MF (mean±std) | 5.40 ± 0.49 | 1.00 ± 0.00 | -4.4000 | **changed** |
| BreastCancer / classification / 50 | Delta (mean±std) | -0.02456 ± 0.02277 | -0.50351 ± 0.01600 | -0.4789 | **changed** |
| BreastCancer / classification / 7 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 7 | Dedup MF (mean±std) | 10.70 ± 0.64 | 5.20 ± 0.60 | -5.5000 | **changed** |
| BreastCancer / classification / 7 | Delta (mean±std) | -0.00175 ± 0.00268 | -0.16433 ± 0.11377 | -0.1626 | **changed** |
| BreastCancer / classification / 70 | Raw MF | 11.1 | 19.9 | +8.8000 | **changed** |
| BreastCancer / classification / 70 | Dedup MF (mean±std) | 3.80 ± 0.98 | 1.00 ± 0.00 | -2.8000 | **changed** |
| Concrete / regression / 0.1 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 0.1 | Dedup MF (mean±std) | 62.60 ± 3.77 | 33.20 ± 2.71 | -29.4000 | **changed** |
| Concrete / regression / 0.3 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 0.3 | Dedup MF (mean±std) | 62.60 ± 3.77 | 33.20 ± 2.71 | -29.4000 | **changed** |
| Concrete / regression / 1 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 1 | Dedup MF (mean±std) | 62.60 ± 3.77 | 33.10 ± 2.77 | -29.5000 | **changed** |
| Concrete / regression / 10 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 10 | Dedup MF (mean±std) | 58.40 ± 3.67 | 32.00 ± 3.22 | -26.4000 | **changed** |
| Concrete / regression / 100 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 100 | Dedup MF (mean±std) | 6.20 ± 1.25 | 3.00 ± 0.63 | -3.2000 | **changed** |
| Concrete / regression / 15 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 15 | Dedup MF (mean±std) | 50.40 ± 3.50 | 29.60 ± 3.14 | -20.8000 | **changed** |
| Concrete / regression / 2 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 2 | Dedup MF (mean±std) | 62.50 ± 3.69 | 33.00 ± 2.76 | -29.5000 | **changed** |
| Concrete / regression / 20 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 20 | Dedup MF (mean±std) | 43.70 ± 3.26 | 26.80 ± 3.37 | -16.9000 | **changed** |
| Concrete / regression / 3 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 3 | Dedup MF (mean±std) | 62.10 ± 3.67 | 33.00 ± 2.76 | -29.1000 | **changed** |
| Concrete / regression / 30 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 30 | Dedup MF (mean±std) | 32.90 ± 2.70 | 21.20 ± 4.21 | -11.7000 | **changed** |
| Concrete / regression / 30 | Delta (mean±std) | -1.18618 ± 0.38488 | -0.34892 ± 0.26860 | +0.8373 | **changed** |
| Concrete / regression / 5 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 5 | Dedup MF (mean±std) | 61.20 ± 3.87 | 33.00 ± 2.76 | -28.2000 | **changed** |
| Concrete / regression / 50 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 50 | Dedup MF (mean±std) | 22.80 ± 2.36 | 12.60 ± 2.06 | -10.2000 | **changed** |
| Concrete / regression / 50 | Delta (mean±std) | -4.00261 ± 1.18617 | -0.78334 ± 0.59160 | +3.2193 | **changed** |
| Concrete / regression / 7 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 7 | Dedup MF (mean±std) | 60.40 ± 3.83 | 32.90 ± 2.77 | -27.5000 | **changed** |
| Concrete / regression / 70 | Raw MF | 63.2 | 33.9 | -29.3000 | **changed** |
| Concrete / regression / 70 | Dedup MF (mean±std) | 13.30 ± 2.33 | 8.60 ± 1.28 | -4.7000 | **changed** |
| Concrete / regression / 70 | Delta (mean±std) | -5.40018 ± 0.35341 | -2.69400 ± 1.38956 | +2.7062 | **changed** |
| Diabetes / regression / 0.1 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 0.3 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 1 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 10 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 100 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 15 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 2 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 20 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 3 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 30 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 5 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 50 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 7 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Diabetes / regression / 70 | Raw MF | 41.6 | 40.6 | -1.0000 | **changed** |
| Digits / classification / 0.1 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 0.1 | Dedup MF (mean±std) | 163.60 ± 3.20 | 138.80 ± 6.10 | -24.8000 | **changed** |
| Digits / classification / 0.3 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 0.3 | Dedup MF (mean±std) | 163.10 ± 3.11 | 137.80 ± 6.46 | -25.3000 | **changed** |
| Digits / classification / 1 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 1 | Dedup MF (mean±std) | 157.50 ± 3.29 | 135.60 ± 6.58 | -21.9000 | **changed** |
| Digits / classification / 10 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 10 | Dedup MF (mean±std) | 75.00 ± 3.87 | 69.60 ± 5.00 | -5.4000 | **changed** |
| Digits / classification / 100 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 15 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 15 | Dedup MF (mean±std) | 53.10 ± 3.91 | 47.70 ± 2.69 | -5.4000 | **changed** |
| Digits / classification / 2 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 2 | Dedup MF (mean±std) | 148.30 ± 2.83 | 129.10 ± 7.31 | -19.2000 | **changed** |
| Digits / classification / 20 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 3 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 3 | Dedup MF (mean±std) | 135.60 ± 4.43 | 121.20 ± 7.81 | -14.4000 | **changed** |
| Digits / classification / 3 | Delta (mean±std) | -0.00167 ± 0.00552 | +0.00407 ± 0.00474 | +0.0057 | **changed** |
| Digits / classification / 30 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 5 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 5 | Dedup MF (mean±std) | 115.30 ± 4.75 | 104.70 ± 6.33 | -10.6000 | **changed** |
| Digits / classification / 50 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 50 | Dedup MF (mean±std) | 12.90 ± 0.94 | 10.10 ± 1.22 | -2.8000 | **changed** |
| Digits / classification / 7 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 7 | Dedup MF (mean±std) | 96.10 ± 5.92 | 88.40 ± 4.18 | -7.7000 | **changed** |
| Digits / classification / 70 | Raw MF | 172.2 | 167.3 | -4.9000 | **changed** |
| Digits / classification / 70 | Dedup MF (mean±std) | 10.20 ± 1.08 | 6.60 ± 1.11 | -3.6000 | **changed** |
| Digits / classification / 70 | Delta (mean±std) | -0.02333 ± 0.06163 | -0.09000 ± 0.05658 | -0.0667 | **changed** |
| Wine / classification / 0.1 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 0.3 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 1 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 10 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 100 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 100 | Dedup MF (mean±std) | 2.10 ± 0.54 | 1.20 ± 0.40 | -0.9000 | **changed** |
| Wine / classification / 100 | Delta (mean±std) | -0.60926 ± 0.03037 | -0.45556 ± 0.05251 | +0.1537 | **changed** |
| Wine / classification / 15 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 2 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 20 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 20 | Delta (mean±std) | -0.02037 ± 0.02922 | -0.05370 ± 0.02922 | -0.0333 | **changed** |
| Wine / classification / 3 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 30 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 30 | Dedup MF (mean±std) | 8.70 ± 1.00 | 7.10 ± 1.22 | -1.6000 | **changed** |
| Wine / classification / 5 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 50 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 50 | Dedup MF (mean±std) | 5.60 ± 0.49 | 3.10 ± 0.94 | -2.5000 | **changed** |
| Wine / classification / 50 | Delta (mean±std) | -0.20556 ± 0.12463 | -0.34444 ± 0.08811 | -0.1389 | **changed** |
| Wine / classification / 7 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 70 | Raw MF | 16.6 | 16.2 | -0.4000 | **changed** |
| Wine / classification / 70 | Dedup MF (mean±std) | 4.70 ± 1.00 | 1.70 ± 0.46 | -3.0000 | **changed** |
| BreastCancer / classification / 1 | Delta (mean±std) | +0.00000 ± 0.00000 | -0.00877 ± 0.03163 | -0.0088 | within noise |
| BreastCancer / classification / 2 | Dedup MF (mean±std) | 11.10 ± 0.30 | 11.10 ± 1.04 | +0.0000 | within noise |
| BreastCancer / classification / 2 | Delta (mean±std) | +0.00000 ± 0.00000 | -0.00409 ± 0.03339 | -0.0041 | within noise |
| BreastCancer / classification / 3 | Dedup MF (mean±std) | 11.10 ± 0.30 | 10.50 ± 1.20 | -0.6000 | within noise |
| BreastCancer / classification / 3 | Delta (mean±std) | +0.00000 ± 0.00000 | -0.00175 ± 0.02173 | -0.0018 | within noise |
| BreastCancer / classification / 70 | Delta (mean±std) | -0.34795 ± 0.26280 | -0.50351 ± 0.01600 | -0.1556 | within noise |
| Concrete / regression / 1 | Delta (mean±std) | +0.00000 ± 0.00000 | -0.00001 ± 0.00003 | -0.0000 | within noise |
| Concrete / regression / 10 | Delta (mean±std) | -0.01184 ± 0.03048 | -0.00017 ± 0.00224 | +0.0117 | within noise |
| Concrete / regression / 100 | Delta (mean±std) | -5.41505 ± 0.33973 | -5.33977 ± 0.34757 | +0.0753 | within noise |
| Concrete / regression / 15 | Delta (mean±std) | -0.11315 ± 0.09133 | -0.08594 ± 0.08957 | +0.0272 | within noise |
| Concrete / regression / 2 | Delta (mean±std) | +0.00010 ± 0.00029 | -0.00015 ± 0.00041 | -0.0003 | within noise |
| Concrete / regression / 20 | Delta (mean±std) | -0.30388 ± 0.34377 | -0.15185 ± 0.11322 | +0.1520 | within noise |
| Concrete / regression / 3 | Delta (mean±std) | -0.00011 ± 0.00141 | -0.00015 ± 0.00041 | -0.0000 | within noise |
| Concrete / regression / 5 | Delta (mean±std) | -0.00095 ± 0.00218 | -0.00015 ± 0.00041 | +0.0008 | within noise |
| Concrete / regression / 7 | Delta (mean±std) | -0.00084 ± 0.00224 | -0.00013 ± 0.00034 | +0.0007 | within noise |
| Diabetes / regression / 0.1 | Dedup MF (mean±std) | 38.40 ± 1.43 | 37.20 ± 1.47 | -1.2000 | within noise |
| Diabetes / regression / 0.3 | Dedup MF (mean±std) | 38.40 ± 1.43 | 37.20 ± 1.47 | -1.2000 | within noise |
| Diabetes / regression / 1 | Dedup MF (mean±std) | 36.20 ± 1.78 | 35.20 ± 1.60 | -1.0000 | within noise |
| Diabetes / regression / 1 | Delta (mean±std) | +0.00588 ± 0.01667 | +0.00604 ± 0.01771 | +0.0002 | within noise |
| Diabetes / regression / 10 | Dedup MF (mean±std) | 13.20 ± 1.54 | 11.70 ± 1.42 | -1.5000 | within noise |
| Diabetes / regression / 10 | Delta (mean±std) | -0.09800 ± 0.08517 | -0.08113 ± 0.07802 | +0.0169 | within noise |
| Diabetes / regression / 100 | Delta (mean±std) | -1.13442 ± 0.47858 | -0.70208 ± 0.55624 | +0.4323 | within noise |
| Diabetes / regression / 15 | Dedup MF (mean±std) | 8.90 ± 1.14 | 8.40 ± 1.28 | -0.5000 | within noise |
| Diabetes / regression / 15 | Delta (mean±std) | -0.55585 ± 0.57566 | -1.09653 ± 1.02229 | -0.5407 | within noise |
| Diabetes / regression / 2 | Dedup MF (mean±std) | 31.40 ± 2.29 | 31.00 ± 1.90 | -0.4000 | within noise |
| Diabetes / regression / 2 | Delta (mean±std) | +0.00110 ± 0.01684 | +0.00447 ± 0.01690 | +0.0034 | within noise |
| Diabetes / regression / 20 | Dedup MF (mean±std) | 5.80 ± 1.08 | 6.40 ± 1.11 | +0.6000 | within noise |
| Diabetes / regression / 20 | Delta (mean±std) | -0.13452 ± 0.10516 | -0.33312 ± 0.69760 | -0.1986 | within noise |
| Diabetes / regression / 3 | Dedup MF (mean±std) | 26.80 ± 1.99 | 26.00 ± 1.84 | -0.8000 | within noise |
| Diabetes / regression / 3 | Delta (mean±std) | -0.01519 ± 0.02420 | +0.00144 ± 0.01861 | +0.0166 | within noise |
| Diabetes / regression / 30 | Dedup MF (mean±std) | 2.10 ± 0.70 | 2.80 ± 0.98 | +0.7000 | within noise |
| Diabetes / regression / 30 | Delta (mean±std) | -0.88968 ± 0.56571 | -0.48339 ± 0.60275 | +0.4063 | within noise |
| Diabetes / regression / 5 | Dedup MF (mean±std) | 20.00 ± 1.61 | 19.10 ± 2.17 | -0.9000 | within noise |
| Diabetes / regression / 5 | Delta (mean±std) | -0.07262 ± 0.02912 | -0.30375 ± 0.79826 | -0.2311 | within noise |
| Diabetes / regression / 50 | Delta (mean±std) | -1.08954 ± 0.52218 | -0.65855 ± 0.55875 | +0.4310 | within noise |
| Diabetes / regression / 7 | Dedup MF (mean±std) | 16.60 ± 2.50 | 14.40 ± 1.36 | -2.2000 | within noise |
| Diabetes / regression / 7 | Delta (mean±std) | -0.08956 ± 0.05221 | -0.31967 ± 0.79441 | -0.2301 | within noise |
| Diabetes / regression / 70 | Delta (mean±std) | -1.13442 ± 0.47858 | -0.70208 ± 0.55624 | +0.4323 | within noise |
| Digits / classification / 0.3 | Delta (mean±std) | +0.00019 ± 0.00056 | +0.00000 ± 0.00000 | -0.0002 | within noise |
| Digits / classification / 1 | Delta (mean±std) | +0.00093 ± 0.00278 | -0.00019 ± 0.00100 | -0.0011 | within noise |
| Digits / classification / 10 | Delta (mean±std) | -0.01889 ± 0.02704 | -0.00185 ± 0.02297 | +0.0170 | within noise |
| Digits / classification / 100 | Dedup MF (mean±std) | 3.50 ± 1.12 | 2.90 ± 0.54 | -0.6000 | within noise |
| Digits / classification / 100 | Delta (mean±std) | -0.13389 ± 0.08297 | -0.15333 ± 0.03352 | -0.0194 | within noise |
| Digits / classification / 15 | Delta (mean±std) | -0.02463 ± 0.03230 | -0.01685 ± 0.06467 | +0.0078 | within noise |
| Digits / classification / 2 | Delta (mean±std) | -0.00093 ± 0.00265 | +0.00111 ± 0.00343 | +0.0020 | within noise |
| Digits / classification / 20 | Dedup MF (mean±std) | 38.40 ± 2.33 | 36.50 ± 2.29 | -1.9000 | within noise |
| Digits / classification / 20 | Delta (mean±std) | -0.03056 ± 0.02733 | -0.03093 ± 0.06540 | -0.0004 | within noise |
| Digits / classification / 30 | Dedup MF (mean±std) | 22.60 ± 2.42 | 22.90 ± 2.55 | +0.3000 | within noise |
| Digits / classification / 30 | Delta (mean±std) | -0.03556 ± 0.04530 | -0.03389 ± 0.07083 | +0.0017 | within noise |
| Digits / classification / 5 | Delta (mean±std) | -0.00278 ± 0.01437 | +0.00704 ± 0.00872 | +0.0098 | within noise |
| Digits / classification / 50 | Delta (mean±std) | -0.02574 ± 0.06344 | -0.08944 ± 0.07553 | -0.0637 | within noise |
| Digits / classification / 7 | Delta (mean±std) | -0.01167 ± 0.02113 | -0.00167 ± 0.02693 | +0.0100 | within noise |
| Wine / classification / 0.1 | Dedup MF (mean±std) | 16.60 ± 0.66 | 16.20 ± 0.40 | -0.4000 | within noise |
| Wine / classification / 0.3 | Dedup MF (mean±std) | 16.60 ± 0.66 | 16.20 ± 0.40 | -0.4000 | within noise |
| Wine / classification / 1 | Dedup MF (mean±std) | 16.60 ± 0.66 | 16.20 ± 0.40 | -0.4000 | within noise |
| Wine / classification / 10 | Dedup MF (mean±std) | 14.60 ± 1.11 | 14.20 ± 0.98 | -0.4000 | within noise |
| Wine / classification / 10 | Delta (mean±std) | -0.00370 ± 0.01614 | -0.02778 ± 0.03015 | -0.0241 | within noise |
| Wine / classification / 15 | Dedup MF (mean±std) | 13.00 ± 1.61 | 12.60 ± 1.36 | -0.4000 | within noise |
| Wine / classification / 15 | Delta (mean±std) | -0.01667 ± 0.02103 | -0.03704 ± 0.03884 | -0.0204 | within noise |
| Wine / classification / 2 | Dedup MF (mean±std) | 16.50 ± 0.81 | 16.10 ± 0.54 | -0.4000 | within noise |
| Wine / classification / 2 | Delta (mean±std) | +0.00185 ± 0.00556 | +0.00000 ± 0.00000 | -0.0019 | within noise |
| Wine / classification / 20 | Dedup MF (mean±std) | 11.10 ± 1.37 | 10.70 ± 0.64 | -0.4000 | within noise |
| Wine / classification / 3 | Dedup MF (mean±std) | 16.50 ± 0.81 | 16.10 ± 0.54 | -0.4000 | within noise |
| Wine / classification / 3 | Delta (mean±std) | +0.00185 ± 0.00556 | +0.00000 ± 0.00000 | -0.0019 | within noise |
| Wine / classification / 30 | Delta (mean±std) | -0.07963 ± 0.06524 | -0.07037 ± 0.06667 | +0.0093 | within noise |
| Wine / classification / 5 | Dedup MF (mean±std) | 16.40 ± 0.80 | 16.00 ± 0.45 | -0.4000 | within noise |
| Wine / classification / 5 | Delta (mean±std) | +0.00185 ± 0.00556 | +0.00000 ± 0.00000 | -0.0019 | within noise |
| Wine / classification / 7 | Dedup MF (mean±std) | 16.10 ± 0.54 | 15.80 ± 0.40 | -0.3000 | within noise |
| Wine / classification / 7 | Delta (mean±std) | +0.00185 ± 0.00556 | +0.00000 ± 0.00000 | -0.0019 | within noise |
| Wine / classification / 70 | Delta (mean±std) | -0.30000 ± 0.19330 | -0.43519 ± 0.06892 | -0.1352 | within noise |

### `table_6_1`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete / R2 | flat | 0.687 ± 0.049 | 0.605 ± 0.042 | -0.0820 | **changed** |
| Concrete / RMSE (MPa) | flat | 9.122 ± 0.623 | 10.265 ± 0.521 | +1.1430 | **changed** |
| PhiUSIIL / accuracy | flat | 0.997 ± 0.001 | 0.729 ± 0.023 | -0.2680 | **changed** |
| PhiUSIIL / accuracy | fuzzy tree | 0.970 ± 0.003 | 0.735 ± 0.029 | -0.2350 | **changed** |
| PhiUSIIL / accuracy | mixture (HME) | 1.000 ± 0.001 | 0.600 ± 0.069 | -0.4000 | **changed** |
| Concrete / R2 | fuzzy tree | 0.583 ± 0.067 | 0.616 ± 0.032 | +0.0330 | within noise |
| Concrete / R2 | mixture (HME) | 0.636 ± 0.087 | 0.689 ± 0.062 | +0.0530 | within noise |
| Concrete / RMSE (MPa) | fuzzy tree | 10.531 ± 0.889 | 10.139 ± 0.492 | -0.3920 | within noise |
| Concrete / RMSE (MPa) | mixture (HME) | 9.785 ± 1.035 | 9.065 ± 0.695 | -0.7200 | within noise |

### `table_a1_feature_ranking`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 2 | wasserstein | HasSocialNet (0.867) | SpacialCharRatioInURL (0.247) | -0.6200 | **changed** |
| 2 | bhattacharyya | HasTitle (0.855) | IsHTTPS (0.954) | +0.0990 | **changed** |
| 2 | composite | URLSimilarityIndex (0.947) | HasSocialNet (0.990) | +0.0430 | **changed** |
| 3 | wasserstein | HasCopyrightInfo (0.743) | DegitRatioInURL (0.075) | -0.6680 | **changed** |
| 3 | bhattacharyya | NoOfSelfRef (0.784) | HasSocialNet (0.809) | +0.0250 | **changed** |
| 3 | composite | HasTitle (0.848) | DegitRatioInURL (0.864) | +0.0160 | **changed** |
| 4 | wasserstein | HasDescription (0.629) | LetterRatioInURL (0.051) | -0.5780 | **changed** |
| 4 | bhattacharyya | NoOfCSS (0.777) | NoOfQMarkInURL (0.773) | -0.0040 | **changed** |
| 4 | composite | NoOfCSS (0.820) | HasTitle (0.816) | -0.0040 | **changed** |
| 5 | wasserstein | DomainTitleMatchScore (0.471) | HasSocialNet (0.049) | -0.4220 | **changed** |
| 5 | bhattacharyya | NoOfImage (0.762) | IsDomainIP (0.726) | -0.0360 | **changed** |
| 5 | composite | NoOfSelfRef (0.815) | HasCopyrightInfo (0.712) | -0.1030 | **changed** |
| 1 | wasserstein | URLSimilarityIndex (1.000) | URLCharProb (1.000) | +0.0000 | within noise |
| 1 | bhattacharyya | HasSocialNet (1.000) | URLSimilarityIndex (1.000) | +0.0000 | within noise |
| 1 | composite | HasSocialNet (1.000) | IsHTTPS (1.000) | +0.0000 | within noise |

### `table_a2_feature_count`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 1 | wasserstein (acc / fit s) | 0.9967 / 0.47 | 0.6709 / 0.13 | -0.3258 | **changed** |
| 1 | bhattacharyya (acc / fit s) | 0.4267 / 0.18 | 0.9967 / 0.11 | +0.5700 | **changed** |
| 10 | wasserstein (acc / fit s) | 0.9997 / 0.75 | 0.8096 / 0.20 | -0.1901 | **changed** |
| 10 | bhattacharyya (acc / fit s) | 0.9701 / 0.85 | 0.9999 / 0.17 | +0.0298 | **changed** |
| 10 | composite (acc / fit s) | 0.9983 / 0.91 | 0.9989 / 0.17 | +0.0006 | **changed** |
| 15 | wasserstein (acc / fit s) | 0.9957 / 0.62 | 0.8096 / 0.23 | -0.1861 | **changed** |
| 15 | bhattacharyya (acc / fit s) | 0.9788 / 0.70 | 0.9999 / 0.23 | +0.0211 | **changed** |
| 15 | composite (acc / fit s) | 0.9999 / 0.63 | 0.9913 / 0.26 | -0.0086 | **changed** |
| 2 | wasserstein (acc / fit s) | 0.9967 / 0.38 | 0.7146 / 0.14 | -0.2821 | **changed** |
| 2 | bhattacharyya (acc / fit s) | 0.4527 / 0.18 | 0.9999 / 0.11 | +0.5472 | **changed** |
| 2 | composite (acc / fit s) | 0.9967 / 0.58 | 0.4267 / 0.11 | -0.5700 | **changed** |
| 20 | wasserstein (acc / fit s) | 0.9984 / 0.71 | 0.8083 / 0.27 | -0.1901 | **changed** |
| 20 | bhattacharyya (acc / fit s) | 0.9796 / 0.74 | 0.9995 / 0.30 | +0.0199 | **changed** |
| 20 | composite (acc / fit s) | 0.9991 / 1.15 | 0.9918 / 0.28 | -0.0073 | **changed** |
| 3 | wasserstein (acc / fit s) | 0.9967 / 0.36 | 0.7203 / 0.15 | -0.2764 | **changed** |
| 3 | bhattacharyya (acc / fit s) | 0.8447 / 0.61 | 0.9999 / 0.11 | +0.1552 | **changed** |
| 3 | composite (acc / fit s) | 0.9967 / 0.28 | 0.8822 / 0.12 | -0.1145 | **changed** |
| 4 | wasserstein (acc / fit s) | 0.9967 / 0.31 | 0.7286 / 0.17 | -0.2681 | **changed** |
| 4 | bhattacharyya (acc / fit s) | 0.9160 / 0.31 | 0.9999 / 0.12 | +0.0839 | **changed** |
| 4 | composite (acc / fit s) | 0.9966 / 0.34 | 0.8822 / 0.13 | -0.1144 | **changed** |
| 5 | wasserstein (acc / fit s) | 0.9965 / 0.41 | 0.7286 / 0.17 | -0.2679 | **changed** |
| 5 | bhattacharyya (acc / fit s) | 0.9467 / 0.33 | 0.9999 / 0.12 | +0.0532 | **changed** |
| 5 | composite (acc / fit s) | 0.9966 / 0.37 | 0.9292 / 0.14 | -0.0674 | **changed** |
| 7 | wasserstein (acc / fit s) | 0.9998 / 0.44 | 0.7295 / 0.18 | -0.2703 | **changed** |
| 7 | bhattacharyya (acc / fit s) | 0.9632 / 0.40 | 0.9999 / 0.13 | +0.0367 | **changed** |
| 7 | composite (acc / fit s) | 0.9967 / 0.44 | 0.9999 / 0.15 | +0.0032 | **changed** |
| 1 | composite (acc / fit s) | 0.4267 / 0.25 | 0.4267 / 0.11 | +0.0000 | within noise |

### `table_concrete_reconciliation`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| flat MoG-TSK 0th / log+standardized / refined | RMSE | 8.63 ± 0.40 | 8.17 ± 0.38 | -0.4600 | **changed** |
| flat MoG-TSK 0th / log+standardized / closed-form only | R² | 0.394 ± 0.065 | 0.453 ± 0.071 | +0.0590 | within noise |
| flat MoG-TSK 0th / log+standardized / closed-form only | RMSE | 12.73 ± 0.90 | 12.08 ± 0.84 | -0.6500 | within noise |
| flat MoG-TSK 0th / log+standardized / refined | R² | 0.720 ± 0.037 | 0.749 ± 0.037 | +0.0290 | within noise |
| flat MoG-TSK 1st / log+standardized / closed-form only | R² | 0.796 ± 0.018 | 0.803 ± 0.025 | +0.0070 | within noise |
| flat MoG-TSK 1st / log+standardized / closed-form only | RMSE | 7.38 ± 0.34 | 7.24 ± 0.32 | -0.1400 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | R² | 0.834 ± 0.045 | 0.845 ± 0.030 | +0.0110 | within noise |
| flat MoG-TSK 1st / log+standardized / refined | RMSE | 6.59 ± 0.65 | 6.40 ± 0.43 | -0.1900 | within noise |
| flat MoG-TSK 2nd / log+standardized / closed-form only | R² | 0.841 ± 0.021 | 0.851 ± 0.021 | +0.0100 | within noise |
| flat MoG-TSK 2nd / log+standardized / closed-form only | RMSE | 6.50 ± 0.43 | 6.28 ± 0.27 | -0.2200 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | R² | 0.862 ± 0.033 | 0.869 ± 0.023 | +0.0070 | within noise |
| flat MoG-TSK 2nd / log+standardized / refined | RMSE | 6.00 ± 0.52 | 5.89 ± 0.37 | -0.1100 | within noise |
| fuzzy tree / raw / n/a | R² | 0.583 ± 0.067 | 0.616 ± 0.032 | +0.0330 | within noise |
| fuzzy tree / raw / n/a | RMSE | 10.53 ± 0.89 | 10.14 ± 0.49 | -0.3900 | within noise |
| mixture of experts (HME) / log+standardized / n/a | R² | 0.747 ± 0.053 | 0.789 ± 0.049 | +0.0420 | within noise |
| mixture of experts (HME) / log+standardized / n/a | RMSE | 8.18 ± 0.74 | 7.46 ± 0.61 | -0.7200 | within noise |
| mixture of experts (HME) / raw / n/a | R² | 0.636 ± 0.087 | 0.689 ± 0.062 | +0.0530 | within noise |
| mixture of experts (HME) / raw / n/a | RMSE | 9.78 ± 1.04 | 9.06 ± 0.69 | -0.7200 | within noise |

### `table_g5_output_partitioning`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 4 / 0th / hybrid *(shipped)* | tail RMSE (MPa) | 20.01 ± 1.39 | 17.47 ± 1.36 | -2.5400 | **changed** |
| 4 / 0th / quantile | RMSE (MPa) | 13.50 ± 0.65 | 12.83 ± 0.65 | -0.6700 | **changed** |
| 4 / 0th / quantile | tail RMSE (MPa) | 21.40 ± 1.46 | 19.52 ± 1.45 | -1.8800 | **changed** |
| 4 / 2nd / hybrid *(shipped)* | max err (MPa) | 29.3 ± 5.5 | 22.2 ± 5.5 | -7.1000 | **changed** |
| 4 / 2nd / quantile | max err (MPa) | 28.8 ± 5.7 | 21.5 ± 5.0 | -7.3000 | **changed** |
| 6 / 1st / hybrid *(shipped)* | max err (MPa) | 31.3 ± 5.4 | 23.7 ± 4.1 | -7.6000 | **changed** |
| 6 / 1st / quantile | max err (MPa) | 30.8 ± 5.5 | 23.2 ± 3.3 | -7.6000 | **changed** |
| 6 / 1st / uniform | max err (MPa) | 29.3 ± 6.1 | 22.1 ± 3.7 | -7.2000 | **changed** |
| 6 / 2nd / hybrid *(shipped)* | R² | 0.852 ± 0.019 | 0.871 ± 0.019 | +0.0190 | **changed** |
| 6 / 2nd / hybrid *(shipped)* | max err (MPa) | 32.9 ± 5.8 | 22.7 ± 5.9 | -10.2000 | **changed** |
| 6 / 2nd / quantile | max err (MPa) | 32.5 ± 5.8 | 22.4 ± 5.4 | -10.1000 | **changed** |
| 3 / 0th / hybrid *(shipped)* | R² | -0.434 ± 0.241 | -0.373 ± 0.313 | +0.0610 | within noise |
| 3 / 0th / hybrid *(shipped)* | RMSE (MPa) | 19.48 ± 0.94 | 18.99 ± 1.52 | -0.4900 | within noise |
| 3 / 0th / hybrid *(shipped)* | tail RMSE (MPa) | 20.70 ± 2.06 | 19.84 ± 3.91 | -0.8600 | within noise |
| 3 / 0th / hybrid *(shipped)* | max err (MPa) | 56.3 ± 5.3 | 58.4 ± 7.2 | +2.1000 | within noise |
| 3 / 0th / quantile | R² | 0.242 ± 0.070 | 0.289 ± 0.097 | +0.0470 | within noise |
| 3 / 0th / quantile | RMSE (MPa) | 14.24 ± 0.75 | 13.76 ± 0.76 | -0.4800 | within noise |
| 3 / 0th / quantile | tail RMSE (MPa) | 22.15 ± 1.76 | 21.38 ± 1.79 | -0.7700 | within noise |
| 3 / 0th / quantile | max err (MPa) | 40.9 ± 2.9 | 39.8 ± 3.0 | -1.1000 | within noise |
| 3 / 0th / uniform | R² | 0.394 ± 0.065 | 0.453 ± 0.071 | +0.0590 | within noise |
| 3 / 0th / uniform | RMSE (MPa) | 12.73 ± 0.90 | 12.08 ± 0.84 | -0.6500 | within noise |
| 3 / 0th / uniform | tail RMSE (MPa) | 18.55 ± 1.29 | 16.84 ± 1.87 | -1.7100 | within noise |
| 3 / 0th / uniform | max err (MPa) | 38.2 ± 3.3 | 36.8 ± 4.0 | -1.4000 | within noise |
| 3 / 1st / hybrid *(shipped)* | R² | 0.787 ± 0.026 | 0.795 ± 0.027 | +0.0080 | within noise |
| 3 / 1st / hybrid *(shipped)* | RMSE (MPa) | 7.54 ± 0.39 | 7.39 ± 0.49 | -0.1500 | within noise |
| 3 / 1st / hybrid *(shipped)* | tail RMSE (MPa) | 7.93 ± 1.04 | 8.07 ± 1.14 | +0.1400 | within noise |
| 3 / 1st / hybrid *(shipped)* | max err (MPa) | 28.6 ± 6.2 | 24.6 ± 5.4 | -4.0000 | within noise |
| 3 / 1st / quantile | R² | 0.789 ± 0.026 | 0.797 ± 0.027 | +0.0080 | within noise |
| 3 / 1st / quantile | RMSE (MPa) | 7.51 ± 0.39 | 7.36 ± 0.50 | -0.1500 | within noise |
| 3 / 1st / quantile | tail RMSE (MPa) | 8.08 ± 1.05 | 8.18 ± 1.18 | +0.1000 | within noise |
| 3 / 1st / quantile | max err (MPa) | 28.5 ± 6.2 | 23.7 ± 4.9 | -4.8000 | within noise |
| 3 / 1st / uniform | R² | 0.796 ± 0.018 | 0.803 ± 0.025 | +0.0070 | within noise |
| 3 / 1st / uniform | RMSE (MPa) | 7.38 ± 0.34 | 7.24 ± 0.32 | -0.1400 | within noise |
| 3 / 1st / uniform | tail RMSE (MPa) | 8.10 ± 1.14 | 7.97 ± 1.05 | -0.1300 | within noise |
| 3 / 1st / uniform | max err (MPa) | 29.2 ± 5.6 | 26.3 ± 5.3 | -2.9000 | within noise |
| 3 / 2nd / hybrid *(shipped)* | R² | 0.832 ± 0.027 | 0.845 ± 0.021 | +0.0130 | within noise |
| 3 / 2nd / hybrid *(shipped)* | RMSE (MPa) | 6.68 ± 0.55 | 6.43 ± 0.48 | -0.2500 | within noise |
| 3 / 2nd / hybrid *(shipped)* | tail RMSE (MPa) | 6.58 ± 0.88 | 6.65 ± 0.80 | +0.0700 | within noise |
| 3 / 2nd / hybrid *(shipped)* | max err (MPa) | 29.1 ± 8.1 | 24.0 ± 7.2 | -5.1000 | within noise |
| 3 / 2nd / quantile | R² | 0.836 ± 0.025 | 0.848 ± 0.020 | +0.0120 | within noise |
| 3 / 2nd / quantile | RMSE (MPa) | 6.61 ± 0.51 | 6.36 ± 0.45 | -0.2500 | within noise |
| 3 / 2nd / quantile | tail RMSE (MPa) | 6.59 ± 0.84 | 6.64 ± 0.82 | +0.0500 | within noise |
| 3 / 2nd / quantile | max err (MPa) | 28.4 ± 7.9 | 23.4 ± 6.4 | -5.0000 | within noise |
| 3 / 2nd / uniform | R² | 0.841 ± 0.021 | 0.851 ± 0.021 | +0.0100 | within noise |
| 3 / 2nd / uniform | RMSE (MPa) | 6.50 ± 0.43 | 6.28 ± 0.27 | -0.2200 | within noise |
| 3 / 2nd / uniform | tail RMSE (MPa) | 6.50 ± 0.82 | 6.40 ± 0.82 | -0.1000 | within noise |
| 3 / 2nd / uniform | max err (MPa) | 28.5 ± 6.4 | 25.9 ± 6.5 | -2.6000 | within noise |
| 4 / 0th / hybrid *(shipped)* | R² | 0.048 ± 0.142 | 0.096 ± 0.185 | +0.0480 | within noise |
| 4 / 0th / hybrid *(shipped)* | RMSE (MPa) | 15.89 ± 0.67 | 15.43 ± 1.03 | -0.4600 | within noise |
| 4 / 0th / hybrid *(shipped)* | max err (MPa) | 49.7 ± 6.3 | 53.5 ± 7.7 | +3.8000 | within noise |
| 4 / 0th / quantile | R² | 0.319 ± 0.048 | 0.383 ± 0.070 | +0.0640 | within noise |
| 4 / 0th / quantile | max err (MPa) | 39.6 ± 2.4 | 39.2 ± 4.1 | -0.4000 | within noise |
| 4 / 0th / uniform | R² | 0.416 ± 0.070 | 0.434 ± 0.062 | +0.0180 | within noise |
| 4 / 0th / uniform | RMSE (MPa) | 12.47 ± 0.58 | 12.30 ± 0.74 | -0.1700 | within noise |
| 4 / 0th / uniform | tail RMSE (MPa) | 19.02 ± 1.25 | 18.56 ± 1.61 | -0.4600 | within noise |
| 4 / 0th / uniform | max err (MPa) | 39.6 ± 5.1 | 37.9 ± 4.9 | -1.7000 | within noise |
| 4 / 1st / hybrid *(shipped)* | R² | 0.797 ± 0.024 | 0.812 ± 0.022 | +0.0150 | within noise |
| 4 / 1st / hybrid *(shipped)* | RMSE (MPa) | 7.36 ± 0.34 | 7.08 ± 0.37 | -0.2800 | within noise |
| 4 / 1st / hybrid *(shipped)* | tail RMSE (MPa) | 8.18 ± 1.11 | 7.85 ± 1.18 | -0.3300 | within noise |
| 4 / 1st / hybrid *(shipped)* | max err (MPa) | 28.4 ± 6.5 | 23.5 ± 3.3 | -4.9000 | within noise |
| 4 / 1st / quantile | R² | 0.795 ± 0.024 | 0.811 ± 0.022 | +0.0160 | within noise |
| 4 / 1st / quantile | RMSE (MPa) | 7.40 ± 0.35 | 7.10 ± 0.38 | -0.3000 | within noise |
| 4 / 1st / quantile | tail RMSE (MPa) | 8.30 ± 1.14 | 7.97 ± 1.25 | -0.3300 | within noise |
| 4 / 1st / quantile | max err (MPa) | 28.3 ± 6.6 | 23.2 ± 3.4 | -5.1000 | within noise |
| 4 / 1st / uniform | R² | 0.799 ± 0.025 | 0.811 ± 0.033 | +0.0120 | within noise |
| 4 / 1st / uniform | RMSE (MPa) | 7.32 ± 0.40 | 7.08 ± 0.62 | -0.2400 | within noise |
| 4 / 1st / uniform | tail RMSE (MPa) | 7.90 ± 1.10 | 7.74 ± 1.34 | -0.1600 | within noise |
| 4 / 1st / uniform | max err (MPa) | 29.6 ± 5.5 | 24.1 ± 5.3 | -5.5000 | within noise |
| 4 / 2nd / hybrid *(shipped)* | R² | 0.848 ± 0.025 | 0.866 ± 0.020 | +0.0180 | within noise |
| 4 / 2nd / hybrid *(shipped)* | RMSE (MPa) | 6.35 ± 0.44 | 5.96 ± 0.43 | -0.3900 | within noise |
| 4 / 2nd / hybrid *(shipped)* | tail RMSE (MPa) | 6.73 ± 0.74 | 6.18 ± 0.84 | -0.5500 | within noise |
| 4 / 2nd / quantile | R² | 0.850 ± 0.025 | 0.868 ± 0.020 | +0.0180 | within noise |
| 4 / 2nd / quantile | RMSE (MPa) | 6.32 ± 0.44 | 5.93 ± 0.42 | -0.3900 | within noise |
| 4 / 2nd / quantile | tail RMSE (MPa) | 6.70 ± 0.71 | 6.14 ± 0.85 | -0.5600 | within noise |
| 4 / 2nd / uniform | R² | 0.845 ± 0.020 | 0.853 ± 0.027 | +0.0080 | within noise |
| 4 / 2nd / uniform | RMSE (MPa) | 6.42 ± 0.48 | 6.25 ± 0.61 | -0.1700 | within noise |
| 4 / 2nd / uniform | tail RMSE (MPa) | 6.29 ± 0.73 | 6.20 ± 0.93 | -0.0900 | within noise |
| 4 / 2nd / uniform | max err (MPa) | 28.4 ± 8.5 | 23.3 ± 7.9 | -5.1000 | within noise |
| 6 / 0th / hybrid *(shipped)* | R² | 0.326 ± 0.107 | 0.344 ± 0.135 | +0.0180 | within noise |
| 6 / 0th / hybrid *(shipped)* | RMSE (MPa) | 13.39 ± 1.05 | 13.17 ± 1.15 | -0.2200 | within noise |
| 6 / 0th / hybrid *(shipped)* | tail RMSE (MPa) | 16.44 ± 1.66 | 15.76 ± 1.58 | -0.6800 | within noise |
| 6 / 0th / hybrid *(shipped)* | max err (MPa) | 46.3 ± 6.3 | 45.6 ± 8.1 | -0.7000 | within noise |
| 6 / 0th / quantile | R² | 0.439 ± 0.070 | 0.451 ± 0.083 | +0.0120 | within noise |
| 6 / 0th / quantile | RMSE (MPa) | 12.24 ± 0.89 | 12.09 ± 0.91 | -0.1500 | within noise |
| 6 / 0th / quantile | tail RMSE (MPa) | 17.66 ± 2.10 | 17.17 ± 1.55 | -0.4900 | within noise |
| 6 / 0th / quantile | max err (MPa) | 38.9 ± 4.2 | 37.3 ± 5.6 | -1.6000 | within noise |
| 6 / 0th / uniform | R² | 0.541 ± 0.046 | 0.530 ± 0.056 | -0.0110 | within noise |
| 6 / 0th / uniform | RMSE (MPa) | 11.06 ± 0.39 | 11.19 ± 0.48 | +0.1300 | within noise |
| 6 / 0th / uniform | tail RMSE (MPa) | 15.03 ± 1.38 | 14.93 ± 1.34 | -0.1000 | within noise |
| 6 / 0th / uniform | max err (MPa) | 36.7 ± 5.0 | 33.0 ± 5.1 | -3.7000 | within noise |
| 6 / 1st / hybrid *(shipped)* | R² | 0.808 ± 0.022 | 0.822 ± 0.026 | +0.0140 | within noise |
| 6 / 1st / hybrid *(shipped)* | RMSE (MPa) | 7.15 ± 0.27 | 6.89 ± 0.52 | -0.2600 | within noise |
| 6 / 1st / hybrid *(shipped)* | tail RMSE (MPa) | 7.37 ± 0.93 | 7.39 ± 0.95 | +0.0200 | within noise |
| 6 / 1st / quantile | R² | 0.806 ± 0.023 | 0.821 ± 0.026 | +0.0150 | within noise |
| 6 / 1st / quantile | RMSE (MPa) | 7.18 ± 0.29 | 6.91 ± 0.51 | -0.2700 | within noise |
| 6 / 1st / quantile | tail RMSE (MPa) | 7.44 ± 0.93 | 7.46 ± 0.93 | +0.0200 | within noise |
| 6 / 1st / uniform | R² | 0.812 ± 0.027 | 0.818 ± 0.028 | +0.0060 | within noise |
| 6 / 1st / uniform | RMSE (MPa) | 7.07 ± 0.37 | 6.96 ± 0.44 | -0.1100 | within noise |
| 6 / 1st / uniform | tail RMSE (MPa) | 7.46 ± 1.11 | 7.35 ± 0.95 | -0.1100 | within noise |
| 6 / 2nd / hybrid *(shipped)* | RMSE (MPa) | 6.28 ± 0.37 | 5.85 ± 0.46 | -0.4300 | within noise |
| 6 / 2nd / hybrid *(shipped)* | tail RMSE (MPa) | 6.34 ± 1.05 | 6.15 ± 0.89 | -0.1900 | within noise |
| 6 / 2nd / quantile | R² | 0.853 ± 0.020 | 0.872 ± 0.019 | +0.0190 | within noise |
| 6 / 2nd / quantile | RMSE (MPa) | 6.26 ± 0.39 | 5.84 ± 0.44 | -0.4200 | within noise |
| 6 / 2nd / quantile | tail RMSE (MPa) | 6.34 ± 1.02 | 6.17 ± 0.85 | -0.1700 | within noise |
| 6 / 2nd / uniform | R² | 0.853 ± 0.018 | 0.860 ± 0.020 | +0.0070 | within noise |
| 6 / 2nd / uniform | RMSE (MPa) | 6.26 ± 0.35 | 6.11 ± 0.37 | -0.1500 | within noise |
| 6 / 2nd / uniform | tail RMSE (MPa) | 6.40 ± 0.75 | 6.25 ± 0.68 | -0.1500 | within noise |
| 6 / 2nd / uniform | max err (MPa) | 28.2 ± 6.6 | 22.9 ± 4.2 | -5.3000 | within noise |

### `table_g5b_skew_sweep`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| 0.50 / +1.84 | Q − U | -0.008 | -0.007 | +0.0010 | **changed** |
| 1.00 / +5.32 | Q − U | +0.020 | +0.001 | -0.0190 | **changed** |
| 1.50 / +10.44 | Q − U | -0.154 | -0.188 | -0.0340 | **changed** |
| 0.01 / +0.05 | uniform R² | 0.912 ± 0.009 | 0.913 ± 0.010 | +0.0010 | within noise |
| 0.01 / +0.05 | quantile R² | 0.911 ± 0.010 | 0.911 ± 0.011 | +0.0000 | within noise |
| 0.01 / +0.05 | uniform tail RMSE | 0.052 ± 0.005 | 0.052 ± 0.006 | +0.0000 | within noise |
| 0.50 / +1.84 | uniform R² | 0.884 ± 0.016 | 0.883 ± 0.020 | -0.0010 | within noise |
| 1.00 / +5.32 | uniform R² | 0.708 ± 0.088 | 0.727 ± 0.096 | +0.0190 | within noise |
| 1.00 / +5.32 | uniform tail RMSE | 0.084 ± 0.037 | 0.079 ± 0.031 | -0.0050 | within noise |
| 1.50 / +10.44 | uniform R² | 0.337 ± 0.122 | 0.371 ± 0.137 | +0.0340 | within noise |
| 1.50 / +10.44 | uniform tail RMSE | 0.097 ± 0.061 | 0.092 ± 0.054 | -0.0050 | within noise |

### `table_hyperparam_normalization`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| flat MoG-TSK 1st / pipeline default | Δ min-max − raw | +0.101 | +0.112 | +0.0110 | **changed** |
| flat MoG-TSK 1st / pipeline default | Δ z-score − raw | +0.018 | +0.031 | +0.0130 | **changed** |
| flat MoG-TSK 1st / pipeline default | Δ z-score − min-max | -0.083 | -0.080 | +0.0030 | **changed** |
| flat MoG-TSK 2nd / pipeline default | Δ min-max − raw | +0.037 | +0.045 | +0.0080 | **changed** |
| flat MoG-TSK 2nd / pipeline default | Δ z-score − raw | +0.023 | +0.030 | +0.0070 | **changed** |
| flat MoG-TSK full-2nd / pipeline default | Δ min-max − raw | +0.030 | +0.064 | +0.0340 | **changed** |
| flat MoG-TSK full-2nd / pipeline default | Δ z-score − raw | -0.021 | +0.044 | +0.0650 | **changed** |
| flat MoG-TSK full-2nd / pipeline default | Δ z-score − min-max | -0.051 | -0.020 | +0.0310 | **changed** |
| fuzzy tree / library default | Δ min-max − raw | +0.106 | +0.074 | -0.0320 | **changed** |
| fuzzy tree / library default | Δ z-score − raw | +0.108 | +0.076 | -0.0320 | **changed** |
| mixture of experts / demo-tuned | log + z-score | 0.806 ± 0.031 | 0.847 ± 0.023 | +0.0410 | **changed** |
| mixture of experts / demo-tuned | Δ min-max − raw | +0.060 | +0.092 | +0.0320 | **changed** |
| mixture of experts / demo-tuned | Δ z-score − raw | +0.032 | +0.092 | +0.0600 | **changed** |
| mixture of experts / demo-tuned | Δ z-score − min-max | -0.028 | +0.000 | +0.0280 | **changed** |
| mixture of experts / demo-tuned | RMSE log+z-score (MPa) | 7.188 ± 0.472 | 6.379 ± 0.471 | -0.8090 | **changed** |
| mixture of experts / library default | Δ min-max − raw | +0.108 | +0.096 | -0.0120 | **changed** |
| mixture of experts / library default | Δ z-score − raw | +0.096 | +0.093 | -0.0030 | **changed** |
| mixture of experts / library default | Δ z-score − min-max | -0.012 | -0.003 | +0.0090 | **changed** |
| flat MoG-TSK 1st / pipeline default | raw features | 0.695 ± 0.030 | 0.691 ± 0.040 | -0.0040 | within noise |
| flat MoG-TSK 1st / pipeline default | log + min-max | 0.796 ± 0.018 | 0.803 ± 0.025 | +0.0070 | within noise |
| flat MoG-TSK 1st / pipeline default | log + z-score | 0.713 ± 0.035 | 0.723 ± 0.042 | +0.0100 | within noise |
| flat MoG-TSK 1st / pipeline default | RMSE raw (MPa) | 9.022 ± 0.357 | 9.064 ± 0.436 | +0.0420 | within noise |
| flat MoG-TSK 1st / pipeline default | RMSE log+min-max (MPa) | 7.381 ± 0.339 | 7.240 ± 0.323 | -0.1410 | within noise |
| flat MoG-TSK 1st / pipeline default | RMSE log+z-score (MPa) | 8.743 ± 0.421 | 8.577 ± 0.335 | -0.1660 | within noise |
| flat MoG-TSK 2nd / pipeline default | raw features | 0.804 ± 0.030 | 0.806 ± 0.032 | +0.0020 | within noise |
| flat MoG-TSK 2nd / pipeline default | log + min-max | 0.841 ± 0.021 | 0.851 ± 0.021 | +0.0100 | within noise |
| flat MoG-TSK 2nd / pipeline default | log + z-score | 0.827 ± 0.028 | 0.837 ± 0.029 | +0.0100 | within noise |
| flat MoG-TSK 2nd / pipeline default | RMSE raw (MPa) | 7.217 ± 0.531 | 7.168 ± 0.432 | -0.0490 | within noise |
| flat MoG-TSK 2nd / pipeline default | RMSE log+min-max (MPa) | 6.499 ± 0.427 | 6.282 ± 0.269 | -0.2170 | within noise |
| flat MoG-TSK 2nd / pipeline default | RMSE log+z-score (MPa) | 6.777 ± 0.473 | 6.570 ± 0.347 | -0.2070 | within noise |
| flat MoG-TSK full-2nd / pipeline default | raw features | 0.830 ± 0.025 | 0.814 ± 0.032 | -0.0160 | within noise |
| flat MoG-TSK full-2nd / pipeline default | log + min-max | 0.861 ± 0.026 | 0.878 ± 0.021 | +0.0170 | within noise |
| flat MoG-TSK full-2nd / pipeline default | log + z-score | 0.809 ± 0.115 | 0.858 ± 0.030 | +0.0490 | within noise |
| flat MoG-TSK full-2nd / pipeline default | RMSE raw (MPa) | 6.719 ± 0.444 | 7.036 ± 0.576 | +0.3170 | within noise |
| flat MoG-TSK full-2nd / pipeline default | RMSE log+min-max (MPa) | 6.072 ± 0.512 | 5.678 ± 0.329 | -0.3940 | within noise |
| flat MoG-TSK full-2nd / pipeline default | RMSE log+z-score (MPa) | 6.905 ± 1.718 | 6.116 ± 0.523 | -0.7890 | within noise |
| fuzzy tree / library default | raw features | 0.583 ± 0.067 | 0.616 ± 0.032 | +0.0330 | within noise |
| fuzzy tree / library default | RMSE raw (MPa) | 10.531 ± 0.889 | 10.139 ± 0.492 | -0.3920 | within noise |
| mixture of experts / demo-tuned | raw features | 0.774 ± 0.025 | 0.755 ± 0.028 | -0.0190 | within noise |
| mixture of experts / demo-tuned | log + min-max | 0.834 ± 0.027 | 0.847 ± 0.023 | +0.0130 | within noise |
| mixture of experts / demo-tuned | RMSE raw (MPa) | 7.771 ± 0.441 | 8.081 ± 0.414 | +0.3100 | within noise |
| mixture of experts / demo-tuned | RMSE log+min-max (MPa) | 6.645 ± 0.469 | 6.379 ± 0.471 | -0.2660 | within noise |
| mixture of experts / library default | raw features | 0.648 ± 0.093 | 0.694 ± 0.065 | +0.0460 | within noise |
| mixture of experts / library default | log + min-max | 0.756 ± 0.059 | 0.790 ± 0.055 | +0.0340 | within noise |
| mixture of experts / library default | log + z-score | 0.744 ± 0.066 | 0.787 ± 0.058 | +0.0430 | within noise |
| mixture of experts / library default | RMSE raw (MPa) | 9.603 ± 1.207 | 8.984 ± 0.699 | -0.6190 | within noise |
| mixture of experts / library default | RMSE log+min-max (MPa) | 8.024 ± 0.753 | 7.433 ± 0.788 | -0.5910 | within noise |
| mixture of experts / library default | RMSE log+z-score (MPa) | 8.214 ± 0.804 | 7.485 ± 0.801 | -0.7290 | within noise |

### `table_norm_conorm_matrix`

| Row | Column | Before | After | Δ | |
|---|---|---|---|---:|---|
| Concrete / HME (experts only) / R2 | min/max | 0.786 ± 0.025 | 0.735 ± 0.040 | -0.0510 | **changed** |
| Concrete / HME (experts only) / R2 | luk | -3.583 ± 0.465 | -1.084 ± 0.397 | +2.4990 | **changed** |
| Concrete / HME (experts only) / R2 | hamacher | 0.795 ± 0.024 | 0.741 ± 0.038 | -0.0540 | **changed** |
| Concrete / HME (experts only) / R2 | Best (mean spread) | **hamacher** (spread 4.378) | **probability** (spread 1.829) | -2.5490 | **changed** |
| Concrete / HME (experts only) / RMSE (MPa) | min/max | 7.547 ± 0.375 | 8.396 ± 0.457 | +0.8490 | **changed** |
| Concrete / HME (experts only) / RMSE (MPa) | luk | 34.957 ± 1.253 | 23.536 ± 2.384 | -11.4210 | **changed** |
| Concrete / HME (experts only) / RMSE (MPa) | hamacher | 7.391 ± 0.366 | 8.299 ± 0.469 | +0.9080 | **changed** |
| Concrete / HME (experts only) / RMSE (MPa) | Best (mean spread) | **hamacher** (spread 27.566) | **probability** (spread 15.303) | -12.2630 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | min/max | 0.998 ± 0.001 | 0.727 ± 0.051 | -0.2710 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | probability | 0.998 ± 0.001 | 0.743 ± 0.044 | -0.2550 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | luk | 0.968 ± 0.001 | 0.739 ± 0.034 | -0.2290 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | hamacher | 0.998 ± 0.001 | 0.741 ± 0.046 | -0.2570 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | einstein | 0.998 ± 0.001 | 0.749 ± 0.042 | -0.2490 | **changed** |
| PhiUSIIL / HME (experts only) / accuracy | Best (mean spread) | **probability** (spread 0.031) | **einstein** (spread 0.022) | -0.0090 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | min/max | 0.967 ± 0.003 | 0.573 ± 0.003 | -0.3940 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | probability | 0.967 ± 0.003 | 0.573 ± 0.003 | -0.3940 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | luk | 0.967 ± 0.003 | 0.589 ± 0.047 | -0.3780 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | hamacher | 0.967 ± 0.003 | 0.573 ± 0.003 | -0.3940 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | einstein | 0.967 ± 0.003 | 0.573 ± 0.003 | -0.3940 | **changed** |
| PhiUSIIL / fuzzy tree (t-norm only) / accuracy | Best (mean spread) | **min/max** (spread 0.000) | **luk** (spread 0.016) | +0.0160 | **changed** |
| Concrete / HME (experts only) / R2 | probability | 0.773 ± 0.019 | 0.745 ± 0.035 | -0.0280 | within noise |
| Concrete / HME (experts only) / R2 | einstein | 0.760 ± 0.024 | 0.744 ± 0.035 | -0.0160 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | probability | 7.788 ± 0.385 | 8.233 ± 0.476 | +0.4450 | within noise |
| Concrete / HME (experts only) / RMSE (MPa) | einstein | 8.009 ± 0.469 | 8.261 ± 0.506 | +0.2520 | within noise |

## Bit-identical

These tables produced exactly the same numbers on both sides:

- `table_3_7_g2_downstream` (21 cells)
- `table_4_4_openset` (9 cells)
- `table_4_4b_theta_sweep` (28 cells)
- `table_5_1_battery` (34 cells)
- `table_5_2_multiscale` (15 cells)
- `table_5_3_selection` (15 cells)
- `table_5_4_ch5_g1_scaling` (126 cells)
- `table_5_4_ch5_g1_scaling_raw` (1800 cells)
- `table_a7_regression_scale` (32 cells)

---

> A cell counts as **changed** only if it moved by more than the larger of the two runs' reported standard deviations; smaller moves are labelled *within noise*. Wall-clock columns are always reported separately and never called a regression — this harness does not control clocks or thermals (see G4 in `NEXT_STEPS.md`).
