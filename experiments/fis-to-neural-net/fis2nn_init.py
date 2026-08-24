"""Layer-1 initializations for the ReLU network: the converted arm and its ablations."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tribblefis.gauss_data import GaussianMixtureModel
from tribblefis.triangle_fit import GAUSSIAN_TRIANGLE_MAE_HALF_WIDTH

from fis2nn_membership import fis_knots, merge_knots
from fis2nn_network import ReLUNet, solve_readout


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


def he_start(rng: np.random.Generator, n_features: int, n_hidden: int) -> ReLUNet:
    """The standard baseline: He-normal layer 1, small random read-out."""
    W1 = rng.normal(0.0, np.sqrt(2.0 / n_features), size=(n_features, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, np.sqrt(2.0 / n_hidden), size=n_hidden)
    v = np.zeros(n_features)
    return ReLUNet(W1=W1, b1=b1, w2=w2, v=v, c=0.0)
