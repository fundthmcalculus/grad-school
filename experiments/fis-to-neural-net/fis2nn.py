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
* :func:`analytic_seed_from_fis` is the practical n-dimensional version, and is
  what the experiment actually uses. It backs the equivalence out into *every*
  weight rather than only the biases: the FIS's own one-dimensional profiles are
  sampled at its knots and decomposed by second differences, so consequents and
  gating reach the network too. In more than one dimension the FIS output is not
  piecewise linear -- the product t-norm and the firing-strength normalization
  both leave the PWL class -- so what the seed carries is the FIS's additive
  part, exactly, and the experiment measures the residual.
* :func:`warm_start_from_fis` is the weaker variant kept for comparison: FIS
  knots for the first layer, and least squares for everything else.

Written against numpy only, deliberately: the point is that the converted
network is an ordinary MLP that any framework can consume, and a 60-line Adam
loop keeps the comparison between initializations free of framework defaults
that would otherwise differ between arms.

This module is a thin re-export shim. The implementation is split by the
sections above across sibling modules -- ``fis2nn_membership`` (membership ->
ReLU expansions), ``fis2nn_network`` (the network itself), ``fis2nn_init``
(layer-1 initializations), ``fis2nn_seed`` (backing the equivalence into every
weight), ``fis2nn_convert1d`` (the exact 1-D theorem), and ``fis2nn_train``
(the shared Adam loop and metrics) -- kept together here so every existing
``import fis2nn`` / ``fis2nn.<name>`` call site (including the private
``fis2nn._axis_aligned_net`` and ``fis2nn._design`` a few callers reach into
directly) keeps working unchanged.
"""

from __future__ import annotations

from fis2nn_membership import (
    KNOT_MERGE_TOL,
    DegenerateMembership,
    ReLUExpansion,
    fis_knots,
    membership_to_relu,
    merge_knots,
    trapezoid_to_relu,
    triangle_to_relu,
)
from fis2nn_network import ReLUNet, _design, solve_readout
from fis2nn_init import (
    _axis_aligned_net,
    he_start,
    quantile_start,
    random_feature_start,
    warm_start_from_fis,
)
from fis2nn_seed import analytic_seed_from_fis, partial_dependence, pwl_to_relu_weights
from fis2nn_convert1d import fis_to_relu_net_1d
from fis2nn_train import (
    TrainHistory,
    error_rate,
    log_loss,
    logit,
    r2,
    rmse,
    sigmoid,
    train_adam,
)

__all__ = [
    "KNOT_MERGE_TOL",
    "DegenerateMembership",
    "ReLUExpansion",
    "triangle_to_relu",
    "trapezoid_to_relu",
    "membership_to_relu",
    "merge_knots",
    "fis_knots",
    "ReLUNet",
    "solve_readout",
    "warm_start_from_fis",
    "quantile_start",
    "random_feature_start",
    "he_start",
    "pwl_to_relu_weights",
    "partial_dependence",
    "analytic_seed_from_fis",
    "fis_to_relu_net_1d",
    "TrainHistory",
    "sigmoid",
    "logit",
    "error_rate",
    "log_loss",
    "rmse",
    "r2",
    "train_adam",
]
