"""Ruspini partitions: k triangular membership functions per input, on [0, 1].

A Ruspini partition is a fuzzy partition whose membership functions sum to one
at every point of the universe. With triangular MFs on a normalised input that
forces the whole geometry: peaks sit on a uniform grid ``c_j = j/(k-1)``, each
MF is the hat function of half-width ``h = 1/(k-1)``, and the two end functions
are the same hats clamped outside ``[0, 1]`` (equivalently, shoulders). There is
exactly one free parameter, ``k``.

Two consequences are used throughout the experiment:

1. At most two MFs are non-zero at any x, and they sum to 1.
2. Therefore the *sum* of memberships over any subset ``S`` of a variable's MFs
   is already in ``[0, 1]``, and equals 1 exactly when ``S`` is the full set.
   That makes plain summation an exact t-conorm here (no clipping, no
   double-counting) and makes "select every MF" mean "don't care" *exactly*
   rather than approximately. See ``rules.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

# Linguistic labels, so a dumped rule reads as a sentence rather than indices.
_LABELS: dict[int, list[str]] = {
    2: ["low", "high"],
    3: ["low", "medium", "high"],
    4: ["low", "med-low", "med-high", "high"],
    5: ["very low", "low", "medium", "high", "very high"],
    7: ["very low", "low", "med-low", "medium", "med-high", "high", "very high"],
}


def labels(k: int) -> list[str]:
    """Linguistic names for the k MFs of a partition."""
    return _LABELS.get(k, [f"L{j}" for j in range(k)])


def centers(k: int) -> Array:
    """Peak locations of a k-term Ruspini partition on [0, 1]."""
    if k < 2:
        raise ValueError("a Ruspini partition needs k >= 2 membership functions")
    return np.linspace(0.0, 1.0, k)


@dataclass(frozen=True)
class UnitScaler:
    """Min-max scaler fitted on training data only; test points are clipped in.

    Clipping (rather than extrapolating) is what keeps the partition-of-unity
    property on unseen data: a test value below the training minimum lands at
    x = 0, where the first MF is 1 and the rest are 0.
    """

    lo: Array
    span: Array

    @classmethod
    def fit(cls, x: Array) -> "UnitScaler":
        lo = np.min(x, axis=0)
        span = np.ptp(x, axis=0)
        # A constant feature carries no information; map it to the partition's
        # midpoint instead of dividing by zero.
        span = np.where(span > 0.0, span, 1.0)
        return cls(lo=lo.astype(np.float64), span=span.astype(np.float64))

    def transform(self, x: Array) -> Array:
        return np.clip((np.asarray(x, dtype=np.float64) - self.lo) / self.span, 0.0, 1.0)


def fuzzify(xn: Array, k: int) -> Array:
    """Membership tensor of normalised data.

    Parameters
    ----------
    xn : (n, d) array already scaled into [0, 1].
    k  : number of MFs per variable.

    Returns
    -------
    (n, d, k) array ``M`` with ``M[t, i, j] = mu_ij(x_t)`` and, by construction,
    ``M[t, i, :].sum() == 1`` for every sample and variable.
    """
    c = centers(k)
    h = 1.0 / (k - 1)
    return np.clip(1.0 - np.abs(xn[..., None] - c) / h, 0.0, 1.0)


def partition_defect(k: int, n_probe: int = 4001) -> float:
    """Max deviation from partition of unity on a dense probe grid (a self-check)."""
    xs = np.linspace(0.0, 1.0, n_probe)[:, None]
    return float(np.max(np.abs(fuzzify(xs, k).sum(axis=2) - 1.0)))
