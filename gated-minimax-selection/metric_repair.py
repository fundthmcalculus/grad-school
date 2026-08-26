"""One-sided metric repair for shortcut-corrupted dissimilarities.

The defense NONMETRIC_FINDINGS.md's finding 2 calls for. The minimax pipeline
is worst-case sensitive in exactly one direction: an entry that is *too small*
(a shortcut) rewires bottleneck paths globally, while entries that are too
large are simply routed around. So the repair is one-sided by design -- it
only ever *raises* entries, and only those that are provably inconsistent
with the rest of the matrix.

The instrument is the reverse triangle inequality. In any metric space,

    D_ij >= |D_ik - D_jk|   for every witness k,

so every witness provides a lower bound on the true distance, and any
quantile of those bounds is a valid lower bound too. A corrupted shortcut
violates this massively: the direct entry is tiny while the two points'
distance profiles disagree at cluster scale for most witnesses. A genuine
short pair has agreeing profiles, so its bounds are all small and the repair
leaves it alone. Formally:

    repaired(D)_ij = max( D_ij, quantile_q over k of |D_ik - D_jk| )

Two properties follow immediately and are pinned in test_metric_repair.py:

* **Identity on metric inputs, at any q.** Every witness bound is <= D_ij, so
  no quantile of them can exceed D_ij. Shortest-path, edit, and Hamming
  matrices pass through untouched -- including *real thin bridges*, which are
  metrically consistent. The repair distinguishes corruption from structure;
  geometric bridges remain ConiVAT's problem, not this function's.
* **One-sided.** Entries never decrease, so stretch-type violations (the kind
  DTW and cosine actually produce, which the minimax transform already
  tolerates) are never "fixed" into something worse.

The quantile q trades repair strength against robustness to corrupted
witnesses: with a fraction r of pairs corrupted, roughly 2r(1-r) of a pair's
witness bounds are themselves inflated (exactly one of the witness's two legs
deflated), so q must sit below ~1 - 2r(1-r) or the lift estimate lands in the
corrupted tail and over-repairs (observed empirically as over-merging at
q = 0.9, corruption rate 0.2). q = 0.5 (the median witness) lifted zero
uncorrupted entries in every no-harm test and is the recommended default;
q = 0.75 repairs harder and is safe up to rate ~0.15.

Related work: this is a cheap, one-sided special case of the metric nearness
problem (Brickell, Dhillon, Sra & Tropp, SIAM J. Matrix Anal. 2008) /
sparse metric repair (Gilbert & Jain, Allerton 2017) -- those solve the full
projection onto the metric cone; this lifts only the below-lower-bound
entries, which is the only direction the minimax transform cares about.
(Citations from memory -- verify before quoting in a chapter.)

Cost: O(n^3) time, O(n^2) memory (one witness pass per row pair, no
(n, n, n) tensor), matching the reference minimax transform's budget.
"""

from __future__ import annotations

import numpy as np


def witness_lower_bounds(D: np.ndarray, q: float = 0.5) -> np.ndarray:
    """The q-quantile over witnesses k of |D_ik - D_jk|, for every pair (i, j).

    In a metric this is a lower bound on D_ij for any q in [0, 1]. Computed
    row by row to keep memory at O(n^2).
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    LB = np.zeros((n, n))
    for i in range(n):
        # |D_ik - D_jk| for all j, k at fixed i: (n, n)
        diffs = np.abs(D[i][None, :] - D)
        LB[i] = np.quantile(diffs, q, axis=1)
    return LB


def reverse_ti_repair(D: np.ndarray, q: float = 0.5) -> np.ndarray:
    """Lift every entry to at least its witness lower bound. See module doc."""
    D = np.asarray(D, dtype=float)
    R = np.maximum(D, witness_lower_bounds(D, q))
    R = (R + R.T) / 2.0
    np.fill_diagonal(R, 0.0)
    return R
