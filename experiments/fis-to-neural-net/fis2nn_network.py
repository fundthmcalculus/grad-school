"""The one-hidden-layer ReLU-plus-linear-skip network the conversion targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReLUNet:
    """``y = relu(X @ W1 + b1) @ w2 + X @ v + c``.

    One hidden layer plus a linear skip. The skip is not decoration: a ReLU
    layer whose knots all sit inside the data range cannot express a nonzero
    slope to the left of the first knot, and the exact 1-D conversion needs one.
    Every arm of the experiment carries the same skip, so it cannot flatter the
    warm-started arm specifically.
    """

    W1: np.ndarray  # (n_features, n_hidden)
    b1: np.ndarray  # (n_hidden,)
    w2: np.ndarray  # (n_hidden,)
    v: np.ndarray  # (n_features,)
    c: float

    @property
    def n_hidden(self) -> int:
        return int(self.W1.shape[1])

    def hidden(self, X: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(X, dtype=float) @ self.W1 + self.b1, 0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.hidden(X) @ self.w2 + X @ self.v + self.c

    def copy(self) -> "ReLUNet":
        return ReLUNet(
            self.W1.copy(), self.b1.copy(), self.w2.copy(), self.v.copy(), float(self.c)
        )

    def n_parameters(self) -> int:
        return self.W1.size + self.b1.size + self.w2.size + self.v.size + 1


def _design(net: ReLUNet, X: np.ndarray) -> np.ndarray:
    """``[hidden | X | 1]`` -- the read-out is linear in exactly these columns."""
    X = np.asarray(X, dtype=float)
    return np.hstack([net.hidden(X), X, np.ones((X.shape[0], 1))])


def solve_readout(
    net: ReLUNet, X: np.ndarray, y: np.ndarray, l2: float = 1e-6, anchor: bool = True
) -> ReLUNet:
    """Set ``w2, v, c`` to the ridge least-squares optimum for the current layer 1.

    Closed form, no gradient steps: for fixed hidden units the output is linear
    in the read-out, which is the same argument `regression.solve_tsk_consequents`
    makes for TSK consequents at fixed firing strengths. Its cost is one linear
    solve rather than an epoch budget.

    ``anchor=True`` (the default) fits the *residual* of whatever read-out the
    net already carries and adds the correction, so the ridge penalty shrinks
    toward that read-out instead of toward zero. Applied to an analytic seed
    that matters: the plain form would solve the backed-out weights away and
    keep only the knots, which is precisely the information the seed exists to
    carry. At ``l2 -> 0`` the two forms coincide, as they should.
    """
    Phi = _design(net, X)
    y = np.asarray(y, dtype=float).ravel()
    target = y - net.predict(X) if anchor else y
    n_cols = Phi.shape[1]
    penalty = l2 * np.eye(n_cols)
    penalty[-1, -1] = 0.0  # never penalize the intercept
    beta = np.linalg.solve(Phi.T @ Phi + penalty, Phi.T @ target)
    h = net.n_hidden
    n_f = net.W1.shape[0]
    out = net.copy()
    if anchor:
        out.w2 = net.w2 + beta[:h]
        out.v = net.v + beta[h : h + n_f]
        out.c = float(net.c + beta[-1])
    else:
        out.w2 = beta[:h]
        out.v = beta[h : h + n_f]
        out.c = float(beta[-1])
    return out
