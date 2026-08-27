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

PRIOR ART (checked 2026-08-26; see notes/BRIDGE_REPAIR.md for the full
account). This is NOT a new method. The problem is named -- Increase Only
Metric Repair, Gilbert & Jain, Allerton 2017; the increase-only variant of
Metric Violation Distance, Fan, Raichel & Van Buskirk, SODA 2018, which is
NP-complete and vertex-cover-hard to approximate. The operator is their
Algorithm 3 at a single witness. At q=1 it collapses to
||row_i(D) - row_j(D)||_inf, the classical Frechet/Kuratowski embedding of a
finite metric into l-infinity -- which is where "identity on metrics" and
"output is a metric" actually come from. The incumbent one-sided anti-bridge
transform with the same max(D_ij, .) shape is HDBSCAN's mutual reachability
distance (Campello, Moulavi & Sander, PAKDD 2013).

CAUTION on the name: the metric guarantee holds only at q=1. At the
recommended default q=0.5 the output still contains triangle violations --
this removes shortcuts, it does not restore metricity. Fine for the minimax
pipeline, which needs no metric; do not describe it as "repair" unguarded.

Only the quantile-over-witnesses aggregation, the corruption-rate estimator
and the abstention rule appear unclaimed, and "not found" is not "novel".

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


def estimate_corruption_rate(D: np.ndarray, rtol: float = 1e-9) -> float:
    """Fraction of pairs whose MEDIAN witness bound exceeds the entry itself.

    On metric data this is exactly 0 (every witness bound is <= D_ij). Under
    planted shortcut corruption it tracks the true corruption rate
    monotonically but conservatively (~0.6x on the blob benchmark): deflated
    intra-cluster pairs are invisible -- their witness bounds are ~0 anyway --
    and only deflated CROSS-cluster pairs, the ones that actually matter to
    the minimax transform, are counted. On densely non-metric data (real
    flight-profile DTW) it reads ~0.5, which is the "decline" signal
    auto_repair uses.
    """
    LB = witness_lower_bounds(D, 0.5)
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(np.mean(LB[iu] > D[iu] * (1.0 + rtol) + 1e-12))


def auto_repair(
    D: np.ndarray,
    margin: float = 0.1,
    q_min: float = 0.5,
    q_max: float = 0.9,
    decline_above: float = 0.35,
) -> tuple:
    """Repair with the quantile set from the data's own estimated corruption.

    The corrupted-witness argument in the module doc says q must sit below
    ~1 - 2r(1-r) when a fraction r of pairs is corrupted; with r estimated by
    :func:`estimate_corruption_rate`, set

        q = clip(1 - 2*r_hat*(1 - r_hat) - margin, q_min, q_max).

    Clean or metric data (r_hat = 0) gets q_max, which is free (the repair is
    identity there at any q). Heavily flagged data (r_hat > decline_above) is
    DECLINED -- returned unchanged -- because a matrix where the median
    witness disagrees with half the entries is not "metric plus sparse
    corruption", it is intrinsically non-metric (the real-DTW regime), and
    the repair's premises do not hold.

    Returns (repaired_or_original, info) where info carries r_hat, the q
    used (None if declined), and the declined flag.
    """
    r_hat = estimate_corruption_rate(D)
    if r_hat > decline_above:
        return np.asarray(D, dtype=float).copy(), {
            "r_hat": r_hat,
            "q": None,
            "declined": True,
        }
    q = float(np.clip(1.0 - 2.0 * r_hat * (1.0 - r_hat) - margin, q_min, q_max))
    return reverse_ti_repair(D, q), {"r_hat": r_hat, "q": q, "declined": False}
