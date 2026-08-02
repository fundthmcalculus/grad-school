"""The C-rule model: exactly one rule per class, antecedents are MF *subsets*.

Rule c reads

    IF x_1 is (A_1c) AND ... AND x_d is (A_dc) THEN class = c

where ``A_ic`` is a subset ``S[c, i, :]`` of variable i's k Ruspini membership
functions, read disjunctively ("x_1 is low OR medium"). The whole model is one
boolean tensor ``S`` of shape (C, d, k) -- and choosing that tensor is the
combinatorial problem this experiment is about.

Nothing forces the subsets to be disjoint across rules: the same MF may appear
in several rules, and different variables may contribute different numbers of
MFs to the same rule. A full row ``S[c, i, :] == True`` is a genuine don't-care
(see below); an empty row is forbidden, since it would zero the rule everywhere.

Firing strength
---------------
Per-variable disjunction, then a t-norm across variables:

    a_ic(x) = OR_{j in S[c,i]} mu_ij(x)          tau_c(x) = AND_i a_ic(x)

``sum`` is the default disjunction. On a Ruspini partition the memberships of a
variable sum to 1, so a subset sum is automatically in [0, 1] with no clipping,
and the full set gives exactly 1. Under ``max`` the full set only gives
max_j mu_ij(x) in [0.5, 1], so "don't care" would be worth as little as 0.5 --
the reason ``sum`` is the natural conorm for this geometry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ruspini import labels

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

TNORMS = ("min", "product")
DISJUNCTIONS = ("sum", "max")


def activations(m: Array, s: BoolArray, disjunction: str = "sum") -> Array:
    """Per-variable antecedent satisfaction.

    m : (n, d, k) membership tensor. s : (d, k) subset mask.
    Returns (n, d).
    """
    if disjunction == "sum":
        return np.clip(np.einsum("ndk,dk->nd", m, s.astype(np.float64)), 0.0, 1.0)
    if disjunction == "max":
        return np.max(np.where(s[None, :, :], m, 0.0), axis=2)
    raise ValueError(f"unknown disjunction {disjunction!r}")


def combine(a: Array, tnorm: str = "min") -> Array:
    """Apply the t-norm across variables: (n, d) -> (n,)."""
    if tnorm == "min":
        return np.min(a, axis=1)
    if tnorm == "product":
        return np.prod(a, axis=1)
    raise ValueError(f"unknown t-norm {tnorm!r}")


def firing(m: Array, s: BoolArray, tnorm: str = "min", disjunction: str = "sum") -> Array:
    """Firing strength of a single rule over all samples: (n,)."""
    return combine(activations(m, s, disjunction), tnorm)


@dataclass
class RuleBase:
    """A complete C-rule classifier."""

    s: BoolArray  # (C, d, k)
    tnorm: str = "min"
    disjunction: str = "sum"
    # Rule weights, fitted on training data. A broad rule fires high everywhere,
    # so raw argmax is biased toward the least specific rule; the inverse-mass
    # weight divides that bias out. ``None`` means unweighted argmax.
    weights: Array | None = None
    class_names: list[str] = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return int(self.s.shape[0])

    def firing_matrix(self, m: Array) -> Array:
        """(n, C) matrix of rule firing strengths."""
        return np.stack(
            [firing(m, self.s[c], self.tnorm, self.disjunction) for c in range(self.n_classes)],
            axis=1,
        )

    def fit_weights(self, m: Array, kind: str = "inverse-mass") -> "RuleBase":
        if kind == "none":
            self.weights = None
            return self
        if kind != "inverse-mass":
            raise ValueError(f"unknown weight kind {kind!r}")
        mass = self.firing_matrix(m).mean(axis=0)
        self.weights = 1.0 / np.maximum(mass, 1e-9)
        return self

    def predict(self, m: Array, priors: Array | None = None) -> NDArray[np.int64]:
        tau = self.firing_matrix(m)
        if self.weights is not None:
            tau = tau * self.weights[None, :]
        if priors is not None:
            # Deterministic tie-break toward the more frequent class. The nudge
            # is far below any meaningful difference in firing strength.
            tau = tau + 1e-9 * priors[None, :]
        return np.asarray(np.argmax(tau, axis=1), dtype=np.int64)

    def silent_rate(self, m: Array) -> float:
        """Fraction of samples where no rule fires at all (the model is guessing)."""
        return float(np.mean(self.firing_matrix(m).max(axis=1) <= 0.0))

    # -- reporting ---------------------------------------------------------
    def complexity(self) -> dict[str, float]:
        k = int(self.s.shape[2])
        per_rule = self.s.sum(axis=(1, 2))
        dontcare = (self.s.sum(axis=2) == k).sum(axis=1)
        return {
            "mfs_per_rule": float(np.mean(per_rule)),
            "dontcare_vars_per_rule": float(np.mean(dontcare)),
            "distinct_mfs_used": float(np.any(self.s, axis=0).sum()),
            "convex_frac": convex_fraction(self.s),
        }

    def describe(self, feature_names: list[str], class_names: list[str] | None = None) -> str:
        k = int(self.s.shape[2])
        names = labels(k)
        cls = class_names or self.class_names or [str(c) for c in range(self.n_classes)]
        out: list[str] = []
        for c in range(self.n_classes):
            terms: list[str] = []
            for i, fname in enumerate(feature_names):
                sel = np.flatnonzero(self.s[c, i])
                if sel.size == k:
                    continue  # don't care -- leave it out of the sentence
                terms.append(f"{fname} is {' or '.join(names[j] for j in sel)}")
            body = "\n     AND ".join(terms) if terms else "(always)"
            out.append(f"R{c}: IF {body}\n     THEN class = {cls[c]}")
        return "\n".join(out)


def is_convex(row: BoolArray) -> bool:
    """True if a variable's selected MFs form one contiguous run.

    Classical fuzzy modelling would require this -- "x is low or medium" is a
    linguistic term, "x is low or high (but not medium)" is a union with a hole
    and reads as two rules pretending to be one. Nothing in the subset
    formulation enforces it, so it is measured rather than assumed.
    """
    idx = np.flatnonzero(row)
    return bool(idx.size > 0 and idx[-1] - idx[0] + 1 == idx.size)


def convex_fraction(s: BoolArray) -> float:
    rows = s.reshape(-1, s.shape[-1])
    return float(np.mean([is_convex(r) for r in rows]))


def rule_objective(
    m: Array,
    in_class: NDArray[np.bool_],
    s: BoolArray,
    lam: float = 1.0,
    tnorm: str = "min",
    disjunction: str = "sum",
) -> float:
    """One-vs-rest fuzzy margin for a single rule.

        J(S) = mean_{x in c} tau(x) - lam * mean_{x not in c} tau(x)

    Both terms are non-decreasing in S, so J is a *difference of monotone*
    set functions -- not submodular. Greedy search on it is a heuristic with no
    approximation guarantee, which is precisely why the exhaustive optimum is
    worth computing where the space is small enough to allow it.
    """
    return objective_from_tau(firing(m, s, tnorm, disjunction), in_class, lam)


def objective_from_tau(tau: Array, in_class: NDArray[np.bool_], lam: float = 1.0) -> float:
    pos = float(tau[in_class].mean()) if in_class.any() else 0.0
    neg = float(tau[~in_class].mean()) if (~in_class).any() else 0.0
    return pos - lam * neg


class Evaluator:
    """Incremental objective evaluator for one class.

    Local search flips one MF of one variable at a time, which changes exactly
    one column of the (n, d) activation matrix. Caching that matrix turns each
    candidate move from O(n*d*k) into O(n*k + n*d).
    """

    def __init__(
        self,
        m: Array,
        in_class: NDArray[np.bool_],
        lam: float = 1.0,
        tnorm: str = "min",
        disjunction: str = "sum",
    ) -> None:
        self.m = m
        self.in_class = in_class
        self.lam = lam
        self.tnorm = tnorm
        self.disjunction = disjunction
        self.n_vars = m.shape[1]
        self.n_mfs = m.shape[2]
        self.n_calls = 0

    def activation_matrix(self, s: BoolArray) -> Array:
        return activations(self.m, s, self.disjunction)

    def column(self, s_row: BoolArray, i: int) -> Array:
        if self.disjunction == "sum":
            return np.clip(self.m[:, i, :] @ s_row.astype(np.float64), 0.0, 1.0)
        return np.max(np.where(s_row[None, :], self.m[:, i, :], 0.0), axis=1)

    def score(self, a: Array) -> float:
        self.n_calls += 1
        return objective_from_tau(combine(a, self.tnorm), self.in_class, self.lam)

    def score_subset(self, s: BoolArray) -> float:
        return self.score(self.activation_matrix(s))
