"""Convert a TRIBBLE-constructed fuzzy inference system into a ReLU network.

The construction rests on one identity (Bede, Kreinovich & Toth, NAFIPS 2023 --
see ``papers/nn-fis-equivalence/``): a triangular membership function *is* a
short sum of ReLUs of the input, exactly, with no approximation anywhere.

    T(x; a, b, c) = s_a * relu(x - a) - (s_a + s_c) * relu(x - b) + s_c * relu(x - c)
    s_a = 1 / (b - a),  s_c = 1 / (c - b)

so a fuzzy term is a hidden-layer motif and its apex/foot knots are the ReLU
bias terms. Everything in this module follows from that:

* :func:`membership_to_relu` turns any of the package's membership shapes into
  that expansion (Gaussians are first fitted to triangles by the package's own
  :mod:`tribblefis.triangle_fit`, which is the only lossy step and is reported
  as such).
* :func:`fis_to_relu_net_1d` uses it to convert a one-dimensional Ruspini-
  partitioned zeroth-order TSK system into a one-hidden-layer ReLU network
  *analytically* -- no data, no fitting, agreement at machine precision. That
  is the theorem, made executable, and ``test_fis2nn.py`` pins it.
* :func:`warm_start_from_fis` is the practical n-dimensional version: the FIS's
  knots become the first layer, and the read-out is solved in closed form. In
  more than one dimension the FIS output is *not* piecewise linear -- the
  product t-norm and the firing-strength normalization both leave the PWL class
  -- so this is a warm start rather than an identity, and the experiment
  measures the gap it leaves.

Written against numpy only, deliberately: the point is that the converted
network is an ordinary MLP that any framework can consume, and a 60-line Adam
loop keeps the comparison between initializations free of framework defaults
that would otherwise differ between arms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from tribblefis.gauss_data import (
    GaussianMembership,
    GaussianMixtureModel,
    TrapezoidMembership,
    TriangularMembership,
)
from tribblefis.triangle_fit import (
    GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
    fit_triangle_to_gaussian,
)

# Knots closer together than this (in the scaled feature's units) are merged
# into one hidden unit. Duplicate knots are common -- two classes' Gaussians
# routinely land on the same mode -- and an exactly duplicated ReLU column makes
# the read-out's normal equations singular in a way ridge would silently paper
# over rather than a modeller noticing.
KNOT_MERGE_TOL = 1e-9


@dataclass(frozen=True)
class ReLUExpansion:
    """``mu(x) = bias + sum_i coeffs[i] * relu(x - knots[i])``, exactly."""

    bias: float
    knots: np.ndarray
    coeffs: np.ndarray

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if self.knots.size == 0:
            return np.full_like(x, self.bias, dtype=float)
        act = np.maximum(x[:, None] - self.knots[None, :], 0.0)
        return self.bias + act @ self.coeffs


def triangle_to_relu(mf: TriangularMembership) -> ReLUExpansion:
    """Exact ReLU expansion of a triangular term, shoulders included.

    A left shoulder (``a = -inf``) is 1 up to the apex and then falls, so it
    expands to a constant minus two ReLUs; a right shoulder (``c = +inf``) rises
    and then flattens, which is two ReLUs and no constant. Both matter: a
    Ruspini partition is exactly one left shoulder, some triangles, and one
    right shoulder, so a converter that only handled interior triangles could
    not convert a partition at all.
    """
    a, b, c = float(mf.a), float(mf.b), float(mf.c)
    knots: list[float] = []
    coeffs: list[float] = []
    bias = 0.0

    left_shoulder = np.isneginf(a)
    right_shoulder = np.isposinf(c)

    # Rising side.
    rise = 0.0
    if left_shoulder:
        bias = 1.0  # membership is already 1 to the left of the apex
    elif b > a:
        rise = 1.0 / (b - a)
        knots.append(a)
        coeffs.append(rise)
    # b == a (a vertical rise) contributes no ReLU: the term jumps to 1 at the
    # apex, which the falling side's expansion below reproduces on its own.

    # Falling side.
    fall = 0.0
    if right_shoulder:
        pass  # membership stays 1 to the right of the apex
    elif c > b:
        fall = 1.0 / (c - b)
        knots.append(c)
        coeffs.append(fall)

    # The apex knot carries whatever slope change makes the two sides meet.
    apex_coeff = -(rise + fall)
    if left_shoulder:
        apex_coeff = -fall
    if right_shoulder:
        apex_coeff = -rise
    if apex_coeff != 0.0:
        knots.append(b)
        coeffs.append(apex_coeff)

    order = np.argsort(np.asarray(knots, dtype=float)) if knots else np.array([], int)
    return ReLUExpansion(
        bias=bias,
        knots=np.asarray(knots, dtype=float)[order],
        coeffs=np.asarray(coeffs, dtype=float)[order],
    )


def trapezoid_to_relu(mf: TrapezoidMembership) -> ReLUExpansion:
    """Exact ReLU expansion of a trapezoidal term (four knots instead of three)."""
    a, b, c, d = float(mf.a), float(mf.b), float(mf.c), float(mf.d)
    knots: list[float] = []
    coeffs: list[float] = []

    if b > a:
        rise = 1.0 / (b - a)
        knots += [a, b]
        coeffs += [rise, -rise]
    if d > c:
        fall = 1.0 / (d - c)
        knots += [c, d]
        coeffs += [-fall, fall]

    order = np.argsort(np.asarray(knots, dtype=float)) if knots else np.array([], int)
    return ReLUExpansion(
        bias=0.0,
        knots=np.asarray(knots, dtype=float)[order],
        coeffs=np.asarray(coeffs, dtype=float)[order],
    )


def membership_to_relu(
    mf, half_width_sigma: float = GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH
) -> ReLUExpansion:
    """ReLU expansion of any membership shape the package produces.

    Triangles and trapezoids convert exactly. A Gaussian has infinite support
    and is not piecewise linear, so it is first replaced by the package's own
    MAE-optimal triangle fit (`tribblefis.triangle_fit`) -- the single lossy
    step in the whole pipeline, and the reason the experiment reports the
    triangularized FIS's accuracy separately from the Gaussian FIS's.
    """
    if isinstance(mf, TriangularMembership):
        return triangle_to_relu(mf)
    if isinstance(mf, TrapezoidMembership):
        return trapezoid_to_relu(mf)
    if isinstance(mf, GaussianMembership):
        return triangle_to_relu(fit_triangle_to_gaussian(mf, half_width_sigma))
    raise TypeError(f"no ReLU expansion for {type(mf).__name__}")


def merge_knots(values: Iterable[float], tol: float = KNOT_MERGE_TOL) -> np.ndarray:
    """Sort, drop non-finite, and merge knots within ``tol`` into their mean."""
    vals = np.asarray(sorted(float(v) for v in values), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return vals
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] <= tol:
            out[-1] = 0.5 * (out[-1] + v)
        else:
            out.append(v)
    return np.asarray(out, dtype=float)


def fis_knots(
    model: GaussianMixtureModel,
    features: Sequence[str],
    half_width_sigma: float = GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
    tol: float = KNOT_MERGE_TOL,
) -> dict[str, np.ndarray]:
    """Every ReLU knot the FIS implies, per feature.

    This is the object the whole "hot start" claim is about: the FIS's answer to
    *where the interesting breakpoints of each input are*, expressed in the only
    currency a ReLU network has.
    """
    knots: dict[str, np.ndarray] = {}
    for name in features:
        feature_model = model.feature_models.get(name)
        if feature_model is None:
            knots[name] = np.asarray([], dtype=float)
            continue
        raw: list[float] = []
        for label_model in feature_model.label_models.values():
            for mf in label_model.memberships:
                raw.extend(membership_to_relu(mf, half_width_sigma).knots.tolist())
        knots[name] = merge_knots(raw, tol)
    return knots


# ---------------------------------------------------------------------------
# The network
# ---------------------------------------------------------------------------


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
    net: ReLUNet, X: np.ndarray, y: np.ndarray, l2: float = 1e-6
) -> ReLUNet:
    """Set ``w2, v, c`` to the ridge least-squares optimum for the current layer 1.

    Closed form, no gradient steps: for fixed hidden units the output is linear
    in the read-out, which is the same argument `regression.solve_tsk_consequents`
    makes for TSK consequents at fixed firing strengths. The warm start inherits
    that property from the FIS it came from -- it is a *construction*, and its
    cost is one linear solve rather than an epoch budget.
    """
    Phi = _design(net, X)
    y = np.asarray(y, dtype=float).ravel()
    n_cols = Phi.shape[1]
    penalty = l2 * np.eye(n_cols)
    penalty[-1, -1] = 0.0  # never penalize the intercept
    beta = np.linalg.solve(Phi.T @ Phi + penalty, Phi.T @ y)
    h = net.n_hidden
    n_f = net.W1.shape[0]
    out = net.copy()
    out.w2 = beta[:h]
    out.v = beta[h : h + n_f]
    out.c = float(beta[-1])
    return out


# ---------------------------------------------------------------------------
# Initializations
# ---------------------------------------------------------------------------


def _axis_aligned_net(
    n_features: int, knots_per_feature: Sequence[tuple[int, np.ndarray]]
) -> ReLUNet:
    """Build layer 1 from ``(feature_index, knots)`` pairs.

    Each hidden unit reads exactly one feature: ``relu(x_f - knot)``. That is
    what the membership expansion produces, and it makes the initial network
    additive across features. Training is free to break the axis alignment --
    and the experiment's whole second half is about watching it do so.
    """
    cols_w: list[np.ndarray] = []
    cols_b: list[float] = []
    for f_idx, knots in knots_per_feature:
        for knot in np.asarray(knots, dtype=float):
            w = np.zeros(n_features, dtype=float)
            w[f_idx] = 1.0
            cols_w.append(w)
            cols_b.append(-float(knot))
    if not cols_w:
        raise ValueError("no knots: cannot build a hidden layer")
    W1 = np.column_stack(cols_w)
    b1 = np.asarray(cols_b, dtype=float)
    h = W1.shape[1]
    return ReLUNet(
        W1=W1,
        b1=b1,
        w2=np.zeros(h),
        v=np.zeros(n_features),
        c=0.0,
    )


def warm_start_from_fis(
    model: GaussianMixtureModel,
    features: Sequence[str],
    X: np.ndarray,
    y: np.ndarray,
    l2: float = 1e-6,
    half_width_sigma: float = GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH,
) -> tuple[ReLUNet, dict[str, np.ndarray]]:
    """The conversion: FIS knots become layer 1, read-out is solved in closed form.

    ``X`` must be in the same scaled coordinates the FIS was fitted in -- the
    knots are membership-function parameters, not data-derived quantities, and
    they are only meaningful in that frame.
    """
    knots = fis_knots(model, features, half_width_sigma)
    pairs = [(i, knots[name]) for i, name in enumerate(features) if knots[name].size]
    net = _axis_aligned_net(len(features), pairs)
    return solve_readout(net, X, y, l2), knots


def quantile_start(
    X: np.ndarray, n_hidden: int, y: np.ndarray, l2: float = 1e-6, eps: float = 1e-9
) -> ReLUNet:
    """Ablation: the same architecture with knots at per-feature quantiles.

    Isolates the FIS's *placement* from the ReLU-knot parameterization itself.
    If a hot start were simply an artifact of axis-aligned knots plus a
    closed-form read-out, this arm would match the converted one.
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    per_feature = max(1, n_hidden // n_features)
    qs = (np.arange(per_feature) + 0.5) / per_feature
    pairs = []
    for f in range(n_features):
        knots = merge_knots(np.quantile(X[:, f], qs), tol=eps)
        if knots.size:
            pairs.append((f, knots))
    net = _axis_aligned_net(n_features, pairs)
    return solve_readout(net, X, y, l2)


def random_feature_start(
    rng: np.random.Generator,
    X: np.ndarray,
    y: np.ndarray,
    n_hidden: int,
    l2: float = 1e-6,
) -> ReLUNet:
    """Ablation: random (He) layer 1, read-out solved in closed form.

    The classic random-features / extreme-learning-machine control. It shares
    the converted arm's closed-form read-out and differs only in where layer 1
    came from, which is the comparison that makes "the FIS knew where to put the
    knots" a falsifiable claim rather than a description.
    """
    X = np.asarray(X, dtype=float)
    n_features = X.shape[1]
    W1 = rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, n_hidden))
    b1 = np.zeros(n_hidden)
    net = ReLUNet(W1=W1, b1=b1, w2=np.zeros(n_hidden), v=np.zeros(n_features), c=0.0)
    return solve_readout(net, X, y, l2)


def he_start(
    rng: np.random.Generator, n_features: int, n_hidden: int
) -> ReLUNet:
    """The standard baseline: He-normal layer 1, small random read-out."""
    W1 = rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=n_hidden)
    v = np.zeros(n_features)
    return ReLUNet(W1=W1, b1=b1, w2=w2, v=v, c=0.0)


# ---------------------------------------------------------------------------
# The exact one-dimensional conversion (the theorem, executable)
# ---------------------------------------------------------------------------


def fis_to_relu_net_1d(
    terms: Sequence[TriangularMembership], consequents: Sequence[float]
) -> ReLUNet:
    """Convert a 1-D zeroth-order TSK system on a Ruspini partition, exactly.

    Preconditions -- all three are what make the conversion an identity rather
    than an approximation:

    * one input,
    * triangular terms forming a partition of unity (so the firing-strength
      normalization is division by 1, and the only non-PWL step disappears),
    * singleton consequents (so the output is a fixed linear combination of the
      terms).

    Under them ``y(x) = sum_l m_l * T_l(x)`` is continuous piecewise linear, and
    stacking the terms' ReLU expansions gives a one-hidden-layer network equal
    to it at every point -- no data touched, no fitting. This is the executable
    form of Bede, Kreinovich & Toth's 1-D equivalence.
    """
    if len(terms) != len(consequents):
        raise ValueError("one consequent per term")
    expansions = [triangle_to_relu(t) for t in terms]
    knots = merge_knots(np.concatenate([e.knots for e in expansions] or [np.array([])]))
    index = {round(float(k), 12): i for i, k in enumerate(knots)}

    w2 = np.zeros(len(knots))
    c = 0.0
    for m, exp in zip(consequents, expansions):
        c += float(m) * exp.bias
        for knot, coeff in zip(exp.knots, exp.coeffs):
            j = index.get(round(float(knot), 12))
            if j is None:  # merged knot: fall back to nearest
                j = int(np.argmin(np.abs(knots - knot)))
            w2[j] += float(m) * float(coeff)

    return ReLUNet(
        W1=np.ones((1, len(knots))),
        b1=-knots.astype(float),
        w2=w2,
        v=np.zeros(1),
        c=float(c),
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainHistory:
    epochs: list[int]
    train_rmse: list[float]
    test_rmse: list[float]
    val_rmse: list[float]
    seconds: list[float]


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = np.asarray(y_true, dtype=float).ravel() - np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean(d * d)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def train_adam(
    net: ReLUNet,
    X: np.ndarray,
    y: np.ndarray,
    *,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 3e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    seed: int = 0,
    eval_every: int = 1,
    y_scale: float = 1.0,
    y_center: float = 0.0,
) -> tuple[ReLUNet, TrainHistory]:
    """Minibatch Adam on the MSE, identical for every arm.

    ``y_scale``/``y_center`` map the network's (standardized) output back to the
    target's own units for reporting, so an arm is never scored in a frame of
    its own choosing.
    """
    import time

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    net = net.copy()

    params = ["W1", "b1", "w2", "v", "c"]
    m = {p: np.zeros_like(np.atleast_1d(getattr(net, p)), dtype=float) for p in params}
    vv = {p: np.zeros_like(np.atleast_1d(getattr(net, p)), dtype=float) for p in params}
    t = 0

    hist = TrainHistory([], [], [], [], [])

    def _score(Xe, ye):
        if Xe is None or ye is None:
            return float("nan")
        return rmse(
            np.asarray(ye).ravel() * y_scale + y_center,
            net.predict(Xe) * y_scale + y_center,
        )

    def record(epoch: int, elapsed: float) -> None:
        hist.epochs.append(epoch)
        hist.seconds.append(elapsed)
        hist.train_rmse.append(_score(X, y))
        hist.test_rmse.append(_score(X_test, y_test))
        hist.val_rmse.append(_score(X_val, y_val))

    start = time.perf_counter()
    record(0, 0.0)

    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for lo in range(0, n, batch_size):
            idx = order[lo : lo + batch_size]
            Xb, yb = X[idx], y[idx]
            b = Xb.shape[0]

            z = Xb @ net.W1 + net.b1
            h = np.maximum(z, 0.0)
            pred = h @ net.w2 + Xb @ net.v + net.c
            resid = pred - yb
            g_out = (2.0 / b) * resid

            grads = {
                "w2": h.T @ g_out,
                "v": Xb.T @ g_out,
                "c": np.atleast_1d(float(g_out.sum())),
            }
            g_h = np.outer(g_out, net.w2) * (z > 0.0)
            grads["W1"] = Xb.T @ g_h
            grads["b1"] = g_h.sum(axis=0)

            t += 1
            for p in params:
                g = grads[p]
                m[p] = beta1 * m[p] + (1 - beta1) * g
                vv[p] = beta2 * vv[p] + (1 - beta2) * (g * g)
                m_hat = m[p] / (1 - beta1**t)
                v_hat = vv[p] / (1 - beta2**t)
                step = lr * m_hat / (np.sqrt(v_hat) + eps)
                if p == "c":
                    net.c = float(net.c - step[0])
                else:
                    setattr(net, p, getattr(net, p) - step)

        if epoch % eval_every == 0 or epoch == epochs:
            record(epoch, time.perf_counter() - start)

    return net, hist
