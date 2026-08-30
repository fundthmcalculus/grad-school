**C3 diagnostics — why the Chapter 5 antecedent arm scores what it scores on bodyfat**

| quantity | mean ± std over folds |
|---|---|
| blocks selected k (transductive, trio) | 6.5 ± 0.8 |
| fraction of points in some block core (trio) | 0.127 ± 0.022 |
| max mu over NON-members, raw U (trio) | 0.5000 ± 0.0000 |
| min mu over members, raw U (trio) | 1.0000 |
| mean max normalized membership (trio) | 0.271 ± 0.025 |
| uniform reference 1/k (trio) | 0.156 ± 0.021 |
| bottleneck-equivalence classes of 252 columns | 81.7 ± 0.6 |
| label disagreement, MS.assign vs argmax U (trio) | 0.780 ± 0.022 |
| kernel reuse identity, max abs diff (trio) | 0 |
| select_multiscale bands discovered (trio) | 0.0 |
| design condition number, graded (trio) | 6.23e+05 ± 5.86e+05 |
| design condition number, crisp (trio) | 1.75e+20 ± 2.42e+20 |
| design rank at tol 1e-08, graded (trio) | 45.6 ± 5.3 |
| design rank at tol 1e-08, crisp (trio) | 31.1 ± 3.7 |
| design columns (trio) | 45.6 ± 5.3 |
| design condition number, graded (all 13) | 3.14e+16 ± 2.22e+17 |
| design condition number, crisp (all 13) | 2.18e+37 ± 1.37e+38 |
| design rank at tol 1e-08, graded (all 13) | 200.9 ± 4.9 |
| design rank at tol 1e-08, crisp (all 13) | 95.4 ± 8.2 |
| design columns (all 13) | 423.4 ± 38.0 |
| blocks selected k (inductive, trio) | 7.5 ± 1.8 |
| train coverage (inductive, trio) | 0.199 ± 0.051 |
| max abs D* gap, point insertion vs joint graph | 0.0810 ± 0.0699 |
| max D* for scale (trio) | 0.5196 ± 0.0154 |
| zero-firing HELD-OUT rows, graded (trio) | 0.00 |
| zero-firing TRAIN rows lost from fit, graded (trio) | 0.80 ± 0.40 |
| zero-firing HELD-OUT rows, graded inductive (trio) | 0.04 ± 0.20 |
| zero-firing TRAIN rows lost, graded inductive (trio) | 0.82 ± 0.56 |
| zero-firing HELD-OUT rows, graded (all 13) | 0.20 ± 0.40 |
| zero-firing TRAIN rows lost from fit, graded (all 13) | 0.00 |
| rules with no training mass, crisp (trio) | 0.00 |
| rules with no training mass, crisp (all 13) | 0.06 ± 0.24 |
| under-filled-bucket warnings, Chapter 4 arms | 0.00 |
| mean abs prediction gap, Ch5 graded vs global 1-rule (pp) | 0.705 ± 0.193 |
| Pearson r, Ch5 graded vs global 1-rule | 0.9833 ± 0.0128 |
| paired R2, graded minus global 1-rule (trio) | -0.0085 ± 0.0089 (2/10 seeds Ch5 ahead) |
| paired R2, graded minus crisp assign (trio) | -0.0053 ± 0.0106 (4/10 seeds Ch5 ahead) |
| paired R2, graded minus crisp argmax U (trio) | -0.0064 ± 0.0114 (4/10 seeds Ch5 ahead) |
| paired R2, graded minus crisp assign, INDUCTIVE | +0.0200 ± 0.0159 (9/10 seeds Ch5 ahead) |
| paired R2, graded minus Chapter 4 (trio) | +0.0206 ± 0.0321 (6/10 seeds Ch5 ahead) |
| paired R2, graded minus OLS (trio) | -0.0148 ± 0.0085 (1/10 seeds Ch5 ahead) |

> Aggregated over every (seed, fold) pair of the run that produced `ch5_end_to_end`; the paired rows aggregate over seeds. THE SHAPE OF U: `max mu over NON-members` is 0.5 and `min mu over members` is 1.0 by construction, not by luck — a non-member of a single-linkage block reaches it only across the edge that dissolves the block, so its minimax distance is at least the block's death height, and the kernel puts half-max exactly there. U therefore has nothing in (0.5, 1), and with only a small `fraction of points in some block core` the normalized partition sits near `uniform reference 1/k`: every rule fires almost equally for every point, each rule's design block is a near-copy of every other's, and the TSK collapses toward one global polynomial. `bottleneck-equivalence classes` counts distinct membership columns — points sharing an MST bottleneck edge share a column — and is NOT a measure of boundary resolution; an earlier pass of this file over-read it as one. CONDITIONING, WITH ITS OWN CONTROL: every conditioning row is reported for the graded AND the crisp arm, because neither rank nor condition number separates the arm that blew up from the arm that did not. On all 13 features the two arms have the same column count, and the CRISP arm is both the more rank-deficient and the worse-conditioned of the two — and it is the one that scores ~19 R² points better. What distinguishes them is where the collinearity lives: a crisp arm's rule blocks have disjoint row support and so cannot be collinear with each other at all, while near-uniform graded firing makes all k blocks nearly proportional on the SAME rows, which is the ill-posedness l2_reg=1e-2 does not control. Two caveats on reading these rows: σ_max/σ_min is not comparable across arms of different numerical rank, and a design whose condition number is below 1/tol is full rank at that tolerance by arithmetic — so a rank row says something about the data only where the condition number beside it exceeds 1e8, which here is every arm except the graded trio one. ZERO FIRING: rows whose memberships underflow ZERO_FIRING_THRESHOLD = 1e-6 are left all-zero by `_normalize_firing_strengths`; a HELD-OUT such row is predicted as exactly 0.0 (on a 0–45 target one row can dominate a fold) and a TRAIN such row contributes nothing to the ridge normal equations, so the graded arms are fitted on slightly fewer rows than the crisp arms beside them. Both sides are counted for every graded arm, with the crisp arms' analogue (a rule with no training mass, whose min-norm consequent also predicts 0.0) beside them. The two prediction-agreement rows are POST-HOC, added after a first three-seed pass showed the Chapter 5 trio arm and the antecedent-free single-rule polynomial agreeing to within one seed-sigma: they compare the two arms' PREDICTIONS rather than their scores, which is the difference between Chapter 5 tying the baseline and Chapter 5 being the baseline. They measure no new model and change no score. The paired rows are the right statistic for arms that share their splits by construction; positive means the Chapter 5 arm is ahead.

> Generated by `reproduce/`; seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]. `N/A` marks a cell whose method/dataset was unavailable.

> **Machine.** host: NEX-210200 · os: Windows-11 · cpu: Intel(R) Core(TM) i9-14900HX · cores: 32 physical, 32 logical · ram: 95.6 GiB · gpu: NVIDIA GeForce RTX 4080 Laptop GPU, 12282 MiB · python: 3.13.7
>
> Wall-clock times are machine-dependent; ratios are not. Markdown tables report normalized ratios where a timing is involved, and the companion CSV carries the absolute seconds.
