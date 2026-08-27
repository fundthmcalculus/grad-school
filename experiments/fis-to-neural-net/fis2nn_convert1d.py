"""The exact one-dimensional conversion: the Bede/Kreinovich/Toth theorem, executable.

:func:`fis_to_relu_net_1d` uses the membership expansions in
:mod:`fis2nn_membership` to convert a one-dimensional Ruspini-partitioned
zeroth-order TSK system into a one-hidden-layer ReLU network *analytically* --
no data, no fitting, agreement at machine precision. That is the theorem, made
executable, and ``test_fis2nn.py`` pins it.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from tribblefis.gauss_data import TriangularMembership

from fis2nn_membership import merge_knots, triangle_to_relu
from fis2nn_network import ReLUNet


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
