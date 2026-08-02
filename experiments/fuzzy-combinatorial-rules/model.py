"""Fit a C-rule Ruspini classifier: normalise, fuzzify, select, weight."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

import selection as sel
from ruspini import UnitScaler, fuzzify
from rules import RuleBase

Array = NDArray[np.float64]


@dataclass
class FitResult:
    model: RuleBase
    scaler: UnitScaler
    train_objective: float  # summed one-vs-rest margin over the C rules
    fit_seconds: float
    exhaustive_ran: bool


def fit(
    x_train: Array,
    y_train: NDArray[np.int64],
    k: int,
    selector: str,
    n_classes: int,
    lam: float = 1.0,
    tnorm: str = "min",
    disjunction: str = "sum",
    weights: str = "inverse-mass",
    convex: bool = False,
    seed: int = 0,
    class_names: list[str] | None = None,
) -> FitResult | None:
    """Return a fitted rule base, or None if the selector could not run.

    Each class is optimised independently against its own one-vs-rest margin;
    the rules only interact at prediction time, through the argmax. That
    decoupling is what makes the exhaustive optimum computable at all -- the
    joint space is ``(2^k - 1)^(d*C)``, the decoupled one is C separate
    ``(2^k - 1)^d`` problems.
    """
    start = time.perf_counter()
    scaler = UnitScaler.fit(x_train)
    m = fuzzify(scaler.transform(x_train), k)
    d = int(x_train.shape[1])

    masks = np.empty((n_classes, d, k), dtype=bool)
    total_j = 0.0
    for c in range(n_classes):
        prob = sel.Problem(
            m=m,
            in_class=(y_train == c),
            lam=lam,
            tnorm=tnorm,
            disjunction=disjunction,
            convex=convex,
            seed=seed * 1000 + c,
        )
        s = sel.select(selector, prob)
        if s is None:
            return None  # exhaustive out of budget
        masks[c] = s
        total_j += prob.score(s)

    model = RuleBase(
        s=masks,
        tnorm=tnorm,
        disjunction=disjunction,
        class_names=class_names or [str(c) for c in range(n_classes)],
    )
    model.fit_weights(m, weights)
    return FitResult(
        model=model,
        scaler=scaler,
        train_objective=total_j,
        fit_seconds=time.perf_counter() - start,
        exhaustive_ran=(selector == "exhaustive"),
    )


def memberships(scaler: UnitScaler, x: Array, k: int) -> Array:
    return fuzzify(scaler.transform(x), k)
