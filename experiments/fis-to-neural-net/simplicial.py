"""Tetrahedral (simplicial) membership functions, and their exact ReLU form.

This is the n-dimensional half of the Bede/Kreinovich/Toth equivalence (IJCCC
2025). The 1-D result converts triangular membership functions to a
one-hidden-layer ReLU network; past one dimension that breaks, because a product
t-norm over per-feature triangles gives piecewise *multi*linear firing strengths
and a sum of ridge functions cannot represent a general n-D piecewise-linear
function. The fix in the paper is to replace triangles with tetrahedra -- a
simplicial partition of the input space, whose barycentric coordinates are
piecewise linear and sum to one by construction.

Implemented here on the **Freudenthal (Kuhn / CFK) triangulation** of a regular
grid, which is the one choice of simplicial complex that avoids the combinatorial
explosion the construction otherwise invites. Three facts make it work, and all
three are pinned by `test_simplicial.py`:

1. **The hat function has a closed form that is a small ReLU circuit.** For a
   grid of spacing ``h`` and a vertex ``v``, with ``d = (x - v) / h``,

       phi_v(x) = relu( 1 - relu(max_i d_i) - relu(max_i (-d_i)) )

   This is exact in every dimension -- verified against Kuhn interpolation at
   zero error, not merely to tolerance. Since ``max(a, b) = a + relu(b - a)``,
   an ``n``-fold max is ``n - 1`` ReLU units in ``ceil(log2 n)`` layers, so a
   tetrahedral membership function is **O(n) ReLU units at depth O(log n)** --
   the n-dimensional analogue of "a triangle is three ReLUs".

2. **Only n+1 hats are nonzero at any point**, whatever the dimension, and which
   ones is found by sorting the fractional coordinates: O(n log n) per sample,
   no search. The tetrahedral rule base is *sparse by construction* rather than
   pruned after the fact.

3. **The vertices that matter are bounded by the data, not by the grid.** A grid
   of K knots per feature has K**n vertices, which is unusable past a handful of
   features -- exactly the rule explosion `tribblefis.anfis` raises
   `RuleExplosionError` over. But a vertex with no data near it contributes
   nothing, and by (2) a dataset of N rows touches at most N*(n+1) of them.
   :func:`occupied_vertices` enumerates only those. This is what makes the
   construction scale: cost follows the data, not the ambient grid.

The consequence for the conversion is that the seed no longer has to be
additive. :func:`simplicial_seed_values` reads the FIS at each occupied vertex,
and the resulting interpolant carries the FIS's *interactions*, which is the
thing `analysis`-level fidelity was losing in the first-order construction.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def barycentric(X: np.ndarray, origin: np.ndarray, h: np.ndarray):
    """Freudenthal decomposition of every row of ``X``.

    Returns ``(vertices, weights)`` with shapes ``(n_samples, n+1, n)`` (integer
    grid coordinates) and ``(n_samples, n+1)``. The weights are barycentric:
    non-negative and summing to exactly 1, which is the partition-of-unity
    property the equivalence needs.

    The construction is Kuhn's: inside a cell, sort the fractional coordinates
    descending and walk the staircase path from the cell's base vertex, adding
    one unit axis step at a time. The barycentric weights are the successive
    differences of the sorted fractional coordinates -- so this is a sort, not a
    search over simplices, and costs O(n log n) per row in any dimension.
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[1]
    u = (X - origin) / h
    base = np.floor(u).astype(np.int64)
    frac = u - base

    order = np.argsort(-frac, axis=1, kind="stable")  # descending
    sorted_frac = np.take_along_axis(frac, order, axis=1)

    # lam_0 = 1 - f_(1);  lam_k = f_(k) - f_(k+1);  lam_n = f_(n)
    weights = np.empty((X.shape[0], n + 1), dtype=float)
    weights[:, 0] = 1.0 - sorted_frac[:, 0]
    if n > 1:
        weights[:, 1:n] = sorted_frac[:, :-1] - sorted_frac[:, 1:]
    weights[:, n] = sorted_frac[:, -1]

    # Vertex k is the base plus the first k axis steps of the sorted order.
    steps = np.zeros((X.shape[0], n + 1, n), dtype=np.int64)
    rows = np.arange(X.shape[0])[:, None]
    onehot = np.zeros((X.shape[0], n, n), dtype=np.int64)
    onehot[rows, np.arange(n)[None, :], order] = 1
    steps[:, 1:, :] = np.cumsum(onehot, axis=1)
    vertices = base[:, None, :] + steps
    return vertices, weights


#: Elements per temporary in :func:`hat`'s dense inner product. 2**23 float64
#: is 64 MB, which keeps the chunked evaluation inside cache-friendly memory
#: even when the vertex count runs to thousands -- a fixed row chunk does not,
#: and on bikeshare (13.9k rows, thousands of vertices, 12 features) a naive
#: 4096-row chunk asks for several gigabytes.
HAT_CHUNK_ELEMENTS = 1 << 23


def hat(
    X: np.ndarray,
    vertices: np.ndarray,
    origin: np.ndarray,
    h: np.ndarray,
    chunk: int | None = None,
) -> np.ndarray:
    """Tetrahedral membership of every row of ``X`` in every vertex's hat.

    ``(n_samples, n_vertices)``. Evaluates the closed form directly; see the
    module docstring for why it is a ReLU circuit and
    ``test_simplicial.py::test_hat_matches_kuhn_interpolation`` for the proof
    that it is the Freudenthal hat and not an approximation of one.

    Chunked over rows because the dense form materializes an
    ``(rows, vertices, n)`` array, which is the one place this construction is
    memory-hungry. The seed never needs the dense form -- :func:`barycentric`
    gives the same numbers in the sparse ``n+1`` representation -- so this is
    for gradient training, where every vertex's parameters move.
    """
    X = np.asarray(X, dtype=float)
    V = np.asarray(vertices, dtype=float)
    out = np.empty((X.shape[0], V.shape[0]), dtype=float)
    centres = origin + V * h
    if chunk is None:
        per_row = max(1, V.shape[0] * X.shape[1])
        chunk = int(np.clip(HAT_CHUNK_ELEMENTS // per_row, 1, X.shape[0] or 1))
    for lo in range(0, X.shape[0], chunk):
        hi = min(lo + chunk, X.shape[0])
        d = (X[lo:hi, None, :] - centres[None, :, :]) / h
        z = 1.0 - np.maximum(d.max(axis=2), 0.0) - np.maximum((-d).max(axis=2), 0.0)
        out[lo:hi] = np.maximum(z, 0.0)
    return out


def occupied_vertices(
    X: np.ndarray,
    origin: np.ndarray,
    h: np.ndarray,
    max_vertices: int | None = None,
    min_weight: float = 0.0,
):
    """The grid vertices the data actually reaches, most-supported first.

    Returns ``(vertices, support)``: integer grid coordinates and the summed
    barycentric weight each vertex carries over ``X``. Support -- not raw hit
    count -- because a vertex that every nearby point touches with weight 0.01
    is not doing the work of one that a few points sit almost on top of.

    ``max_vertices`` is *the* scalability knob and is a genuine truncation: the
    kept hats no longer sum to 1, so the partition-of-unity property is lost and
    the seed's read-out has to absorb the difference. Callers should report the
    retained support fraction rather than letting the truncation pass silently
    (`run_simplicial.py` does).
    """
    vertices, weights = barycentric(X, origin, h)
    flat = vertices.reshape(-1, vertices.shape[2])
    w = weights.reshape(-1)
    if min_weight > 0:
        keep = w > min_weight
        flat, w = flat[keep], w[keep]

    uniq, inverse = np.unique(flat, axis=0, return_inverse=True)
    support = np.zeros(len(uniq), dtype=float)
    np.add.at(support, inverse, w)

    order = np.argsort(-support)
    uniq, support = uniq[order], support[order]
    if max_vertices is not None and len(uniq) > max_vertices:
        uniq, support = uniq[:max_vertices], support[:max_vertices]
    return uniq, support


def grid_from_data(X: np.ndarray, resolution: int):
    """A uniform grid over the data's bounding box, ``resolution`` cells a side.

    Uniform, not knot-derived, and deliberately: the Freudenthal triangulation
    is defined on a *regular* lattice, and the closed-form hat above assumes one
    common spacing per axis. A non-uniform grid would need the hat rebuilt
    per cell, which is exactly the per-simplex enumeration this construction
    exists to avoid.
    """
    X = np.asarray(X, dtype=float)
    lo = X.min(axis=0)
    hi = X.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    # A half-cell margin each side so boundary rows sit strictly inside a cell.
    h = span / max(resolution, 1)
    return lo - 0.5 * h, h


def simplicial_seed_values(
    predict_fn, columns, feature_names, vertices, origin, h, template
):
    """Read the FIS at each vertex: the tetrahedral system's singleton consequents.

    This is the whole conversion. A TSK system with tetrahedral antecedents and
    singleton consequents ``c_v`` computes ``sum_v c_v * phi_v(x)``, which is the
    Freudenthal interpolant of whatever function the ``c_v`` were sampled from.
    Setting ``c_v = FIS(v)`` therefore makes the tetrahedral system the
    piecewise-linear interpolant *of the FIS itself* -- interactions included,
    which is what the additive first-order seed could not carry.

    ``template`` is a one-row frame supplying any columns the regressor needs
    that are not part of the converted feature set; the vertex coordinates are
    written over ``feature_names``.
    """
    import pandas as pd

    centres = origin + np.asarray(vertices, dtype=float) * h
    frame = pd.concat([template] * len(centres), ignore_index=True)
    for j, name in enumerate(feature_names):
        frame[name] = centres[:, j]
    # Columns the caller wants preserved in their original order.
    return np.asarray(predict_fn(frame[columns]), dtype=float).ravel()


@dataclass
class SimplicialNet:
    """``y = sum_v c_v * phi_v(x) + x @ skip + bias``, the tetrahedral TSK system.

    Held in its fuzzy parameterization (vertices, spacing, consequents) rather
    than as an expanded weight matrix, because that is the form the conversion
    produces and the form that stays sparse. ``to_relu_spec`` states the
    equivalent pure-ReLU network's size; `test_simplicial.py` checks the
    expansion agrees numerically on the pairwise-max motif.
    """

    vertices: np.ndarray  # (n_vertices, n) integer grid coordinates
    origin: np.ndarray  # (n,)
    h: np.ndarray  # (n,)
    c: np.ndarray  # (n_vertices,) singleton consequents
    skip: np.ndarray  # (n,)
    bias: float

    @property
    def n_hidden(self) -> int:
        return int(self.vertices.shape[0])

    @property
    def n_features(self) -> int:
        return int(self.vertices.shape[1])

    def memberships(self, X: np.ndarray) -> np.ndarray:
        return hat(X, self.vertices, self.origin, self.h)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return self.memberships(X) @ self.c + X @ self.skip + self.bias

    def predict_sparse(self, X: np.ndarray) -> np.ndarray:
        """Same value, via the n+1 active hats instead of the dense product.

        Only valid when no vertex has been truncated away -- with a `max_vertices`
        cap the active vertex for a row may not be in the kept set, and the dense
        path (which simply scores zero for missing hats) is the honest one. Used
        by the tests to show the two agree when the basis is complete.
        """
        X = np.asarray(X, dtype=float)
        vertices, weights = barycentric(X, self.origin, self.h)
        index = {tuple(v): i for i, v in enumerate(self.vertices.tolist())}
        out = np.zeros(len(X))
        for k in range(vertices.shape[1]):
            for r in range(len(X)):
                i = index.get(tuple(vertices[r, k].tolist()))
                if i is not None:
                    out[r] += weights[r, k] * self.c[i]
        return out + X @ self.skip + self.bias

    def to_relu_spec(self) -> dict:
        """Size of the equivalent pure-ReLU network.

        Each hat is two n-fold maxes (``n - 1`` ReLU units each, via
        ``max(a, b) = a + relu(b - a)``), two ReLUs for the clamps and one for
        the outer ``relu``, at depth ``ceil(log2 n) + 2``. Reported rather than
        materialized: the point is that the count is linear in ``n`` and in the
        number of vertices, with no ``K**n`` term anywhere.
        """
        n, v = self.n_features, self.n_hidden
        return {
            "relu_units": v * (2 * (n - 1) + 3),
            "depth": int(np.ceil(np.log2(max(n, 2)))) + 2,
            "vertices": v,
            "n_features": n,
            "dense_grid_would_be": "K**n",
        }


def consequents_from_fis(
    mode,
    predict_fn,
    X_scaled,
    columns,
    feature_names,
    vertices,
    origin,
    h,
    template,
    l2: float = 1e-6,
):
    """Singleton consequents for the tetrahedral rules, three ways.

    ``"vertex"``
        ``c_v = FIS(v)``: the literal reading of the equivalence, and the only
        one that needs no data at all. It is also the one that fails in high
        dimension, because a grid vertex in 8-D sits about a cell away from the
        nearest datum -- so this asks the FIS to extrapolate off the data
        manifold, where its own output is not meaningful. Measured on Concrete:
        the FIS spans 1.4..78.0 on real rows and -3.9..90.2 at grid vertices.
    ``"support"``
        A barycentric-weighted average of the FIS's predictions at the *data*:
        ``c_v = sum_i phi_v(x_i) FIS(x_i) / sum_i phi_v(x_i)``. On-manifold, but
        it smooths, and the smoothing gets worse as the grid refines and each
        vertex retains fewer points.
    ``"project"``
        Ridge solve of ``Phi c ~ FIS(X)``: the orthogonal projection of the FIS
        onto the tetrahedral basis, evaluated only where the data is. Best of
        the three wherever a vertex has support, and the default.

    None of the three consults labels -- all three convert the FIS.
    """
    if mode == "vertex":
        return (
            simplicial_seed_values(
                predict_fn, columns, feature_names, vertices, origin, h, template
            ),
            0.0,
        )

    Phi = hat(X_scaled, vertices, origin, h)
    target = np.asarray(predict_fn(X_scaled) if callable(predict_fn) else predict_fn)

    if mode == "support":
        weight = Phi.sum(axis=0)
        c = np.zeros(len(vertices))
        ok = weight > 1e-9
        c[ok] = (Phi[:, ok] * target[:, None]).sum(axis=0) / weight[ok]
        return c, 0.0

    if mode == "project":
        design = np.hstack([Phi, np.ones((len(Phi), 1))])
        penalty = l2 * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        beta = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        return beta[:-1], float(beta[-1])

    raise ValueError(f"unknown consequent mode {mode!r}")


#: Barycentric support below which a tetrahedral rule has too little data behind
#: it to be worth building. Measured, not assumed: on Concrete the hybrid's
#: fidelity improves monotonically as the grid refines while each vertex still
#: holds roughly ten rows, and becomes erratic below about five -- swinging from
#: 0.42 to 1.76 to 2.51 across neighbouring resolutions once vertices outnumber
#: rows. See `run_simplicial.py`'s support table.
TARGET_ROWS_PER_VERTEX = 10.0


def auto_resolution(
    X_scaled: np.ndarray,
    candidates=(2, 3, 4, 6, 8, 12, 16, 24),
    target_rows_per_vertex: float = TARGET_ROWS_PER_VERTEX,
):
    """Finest grid whose vertices still carry ``target_rows_per_vertex`` rows each.

    This is the scalability rule, and it is what stops the construction chasing
    resolution it cannot support. It is also why the tetrahedral basis has to be
    built on a *few* features rather than all of them: the vertex count grows
    with the subspace dimension while the row count does not, so past three or
    four features every grid fails this test at every resolution.

    Returns ``(resolution, origin, h, vertices, support)`` for the chosen grid,
    always falling back to the coarsest candidate rather than failing.
    """
    n_rows = len(X_scaled)
    chosen = None
    for res in candidates:
        origin, h = grid_from_data(X_scaled, res)
        vertices, support = occupied_vertices(X_scaled, origin, h)
        if n_rows / max(len(vertices), 1) >= target_rows_per_vertex:
            chosen = (res, origin, h, vertices, support)
        else:
            break
    if chosen is None:
        res = candidates[0]
        origin, h = grid_from_data(X_scaled, res)
        vertices, support = occupied_vertices(X_scaled, origin, h)
        chosen = (res, origin, h, vertices, support)
    return chosen


def fit_simplicial_seed(
    predict_fn,
    X_scaled: np.ndarray,
    columns,
    feature_names,
    template,
    resolution: int = 4,
    max_vertices: int | None = 512,
):
    """Convert a FIS into a tetrahedral TSK system, using no labels.

    Returns ``(net, info)``. Every consequent is the FIS's own value at a grid
    vertex, so the result is the FIS's Freudenthal interpolant restricted to the
    vertices the data reaches.
    """
    origin, h = grid_from_data(X_scaled, resolution)
    vertices, support = occupied_vertices(
        X_scaled, origin, h, max_vertices=max_vertices
    )
    c = simplicial_seed_values(
        predict_fn, columns, feature_names, vertices, origin, h, template
    )
    net = SimplicialNet(
        vertices=vertices,
        origin=origin,
        h=h,
        c=c,
        skip=np.zeros(X_scaled.shape[1]),
        bias=0.0,
    )
    all_vertices, all_support = occupied_vertices(X_scaled, origin, h)
    info = {
        "resolution": resolution,
        "vertices_kept": int(len(vertices)),
        "vertices_occupied": int(len(all_vertices)),
        "support_retained": float(support.sum() / all_support.sum()),
        "grid_vertices_if_dense": float(resolution + 1) ** X_scaled.shape[1],
        "relu_spec": net.to_relu_spec(),
    }
    return net, info


@dataclass
class SimplicialCorrection:
    """A tetrahedral correction living on a low-dimensional subspace.

    The hybrid the measurements point at. Main effects stay in the first-order
    additive seed, where every one of the N rows contributes to every 1-D
    profile; interactions go here, on the ``k`` features the FIS ranked highest,
    where a grid can still be supported. Cost is ``O(K**k)`` vertices with ``k``
    small and fixed -- it does not grow with the full feature count, which is
    what makes it scale where a full-dimensional grid does not.
    """

    columns: np.ndarray  # indices into the converted feature set
    net: SimplicialNet
    resolution: int
    rows_per_vertex: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.net.predict(np.asarray(X, dtype=float)[:, self.columns])

    def to_relu_spec(self) -> dict:
        spec = self.net.to_relu_spec()
        spec["subspace"] = int(len(self.columns))
        return spec


def fit_simplicial_correction(
    residual: np.ndarray,
    X_scaled: np.ndarray,
    subspace,
    l2: float = 1e-6,
    target_rows_per_vertex: float = TARGET_ROWS_PER_VERTEX,
    candidates=(2, 3, 4, 6, 8, 12, 16, 24),
):
    """Project a residual onto a tetrahedral basis over ``subspace``.

    ``residual`` is what the additive seed could not carry -- the FIS's own
    predictions minus the first-order seed's, so still no labels anywhere. The
    grid resolution is chosen by :func:`auto_resolution` rather than passed in,
    because the right resolution is a function of how much data each vertex
    would end up with.
    """
    subspace = np.asarray(subspace, dtype=int)
    S = np.asarray(X_scaled, dtype=float)[:, subspace]
    res, origin, h, vertices, _support = auto_resolution(
        S, candidates=candidates, target_rows_per_vertex=target_rows_per_vertex
    )
    c, bias = consequents_from_fis(
        "project",
        np.asarray(residual, dtype=float),
        S,
        None,
        None,
        vertices,
        origin,
        h,
        None,
        l2=l2,
    )
    net = SimplicialNet(
        vertices=vertices,
        origin=origin,
        h=h,
        c=c,
        skip=np.zeros(len(subspace)),
        bias=bias,
    )
    return SimplicialCorrection(
        columns=subspace,
        net=net,
        resolution=res,
        rows_per_vertex=len(S) / max(len(vertices), 1),
    )


# ---------------------------------------------------------------------------
# Triangulating on the FIS's structure instead of on a bounding box
# ---------------------------------------------------------------------------
#
# The 2025 paper's interpolation is exact because its triangulation is induced
# by the linear regions of the very network being converted -- the vertices sit
# where the function actually bends. A lattice over the data's bounding box has
# no such property, and `run_simplicial.py` measured what that costs: vertices
# land off the data manifold, and resolution is spent uniformly on a space the
# data occupies unevenly.
#
# The fix does not require abandoning the Freudenthal machinery, which needs a
# *regular* lattice. A non-uniform rectilinear complex is the image of a regular
# one under a per-axis monotone map, so warping each axis until the FIS's own
# knots land on integers gives a complex aligned to the FIS's structure while
# every hat stays exactly the closed form above. And the warp is itself
# piecewise linear -- built by the same `pwl_to_relu_weights` the first-order
# seed uses -- so the composition is still a ReLU circuit, just two blocks deep
# instead of one.


@dataclass
class AxisWarp:
    """Per-axis monotone piecewise-linear map sending a feature's knots to 0,1,2,...

    ``knots[f]`` are that feature's FIS-derived breakpoints, sorted. After
    :meth:`forward`, a unit cell of the lattice is one inter-knot interval, so
    lattice vertices coincide with the FIS's membership geometry rather than
    with an arbitrary grid. Outside the knot range the map extends linearly, so
    test rows beyond the training extent still receive distinct coordinates
    instead of being clamped on top of each other.
    """

    knots: list

    #: Knots closer together than this (in the unit-scaled feature's own units)
    #: are merged before the warp is built. Unlike the additive seed -- where a
    #: near-duplicate knot is just one more nearly-collinear ReLU column, and
    #: sweeping the tolerance from 1e-9 to 1e-2 moved test RMSE by under 2% --
    #: here it distorts the *geometry*: two knots 4e-6 apart become a full unit
    #: cell, so a lattice cell spans a gap no data can resolve. Measured on WEC,
    #: whose knot gaps span 4.6e4-to-1 (4.17e-06 to 0.19), leaving this at the
    #: `fis_knots` default drove fidelity to 16.46 against an additive seed's
    #: 1.47.
    MIN_GAP: float = 1e-3

    @staticmethod
    def from_knots(
        knots_by_feature, feature_names, min_gap: float = MIN_GAP
    ) -> "AxisWarp":
        import fis2nn  # local: keeps the two modules independently importable

        cleaned = []
        for name in feature_names:
            k = np.asarray(knots_by_feature.get(name, []), dtype=float)
            k = k[np.isfinite(k)]
            if k.size:
                k = fis2nn.merge_knots(k, tol=min_gap)
            cleaned.append(k if k.size >= 2 else np.asarray([], dtype=float))
        return AxisWarp(knots=cleaned)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Map data into knot-index coordinates. Identity on axes with no knots."""
        import fis2nn  # local: keeps the two modules independently importable

        X = np.asarray(X, dtype=float)
        U = np.array(X, dtype=float, copy=True)
        for f, k in enumerate(self.knots):
            if k.size < 2:
                continue
            base, intercept, coeffs = fis2nn.pwl_to_relu_weights(
                k, np.arange(k.size, dtype=float)
            )
            act = np.maximum(X[:, f : f + 1] - k[None, :], 0.0)
            U[:, f] = intercept + base * X[:, f] + act @ coeffs
        return U

    def relu_units(self) -> int:
        """ReLU units the warp itself costs -- one per interior knot, per axis."""
        return int(sum(max(k.size - 2, 0) for k in self.knots if k.size >= 2))


@dataclass
class WarpedCorrection:
    """A tetrahedral correction on a FIS-aligned lattice."""

    columns: np.ndarray
    warp: AxisWarp
    net: SimplicialNet
    resolution: int
    rows_per_vertex: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        S = self.warp.forward(np.asarray(X, dtype=float)[:, self.columns])
        return self.net.predict(S)

    def to_relu_spec(self) -> dict:
        spec = self.net.to_relu_spec()
        spec["subspace"] = int(len(self.columns))
        spec["warp_units"] = self.warp.relu_units()
        spec["relu_units"] += spec["warp_units"]
        return spec


def fit_warped_correction(
    residual: np.ndarray,
    X_scaled: np.ndarray,
    subspace,
    warp: AxisWarp,
    l2: float = 1e-6,
    target_rows_per_vertex: float = TARGET_ROWS_PER_VERTEX,
    candidates=(2, 3, 4, 6, 8, 12, 16, 24),
):
    """:func:`fit_simplicial_correction`, but on the FIS-aligned lattice.

    Identical in every other respect, so the pair isolates *where the vertices
    sit* from everything else about the tetrahedral construction.
    """
    subspace = np.asarray(subspace, dtype=int)
    sub_warp = AxisWarp(knots=[warp.knots[i] for i in subspace])
    S = sub_warp.forward(np.asarray(X_scaled, dtype=float)[:, subspace])
    res, origin, h, vertices, _support = auto_resolution(
        S, candidates=candidates, target_rows_per_vertex=target_rows_per_vertex
    )
    c, bias = consequents_from_fis(
        "project",
        np.asarray(residual, dtype=float),
        S,
        None,
        None,
        vertices,
        origin,
        h,
        None,
        l2=l2,
    )
    net = SimplicialNet(
        vertices=vertices,
        origin=origin,
        h=h,
        c=c,
        skip=np.zeros(len(subspace)),
        bias=bias,
    )
    return WarpedCorrection(
        columns=subspace,
        warp=sub_warp,
        net=net,
        resolution=res,
        rows_per_vertex=len(S) / max(len(vertices), 1),
    )
