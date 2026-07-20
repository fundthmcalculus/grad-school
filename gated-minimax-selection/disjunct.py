"""
Disjunct-arity discovery for the iVAT->MF approach.

For each selected block, determine how many disjuncts its membership function
needs, by counting connected components under three different notions of
"disconnected":

  mode='feature' : components in feature space (Euclidean). Detects projection
                   multimodality -- the rings fragment here.
  mode='dstar'   : components in the minimax sublevel graph (threshold D* at the
                   block birth height and count graph components among block
                   members). Hierarchy-native -- each ring stays one set.
  mode='hybrid'  : split only if disconnected in BOTH. Conservative.

The arity is read at the block's birth height (the scale at which the block is
a coherent cluster). This is deterministic: no mixture fit, no shape prior.
The chosen (fixed) t-conorm recombines disjuncts AFTER arity is decided, so the
reported arity is independent of the conorm.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _components_dstar(Dstar, members, height):
    """Connected components among `members` using edges with D* <= height."""
    members = np.asarray(sorted(members), dtype=int)
    m = len(members)
    if m <= 1:
        return np.zeros(m, dtype=int), 1
    sub = Dstar[np.ix_(members, members)]
    A = (sub <= height + 1e-12).astype(int)
    np.fill_diagonal(A, 0)
    ncomp, labels = connected_components(csr_matrix(A), directed=False)
    return labels, ncomp


def _components_feature(X, members, radius):
    """
    DEPRECATED radius-graph version (fragments on density variation).
    Kept for reference; per-axis multimodality is the right detector -- see
    axis_multimodality below.
    """
    members = np.asarray(sorted(members), dtype=int)
    m = len(members)
    if m <= 1:
        return np.zeros(m, dtype=int), 1
    P = X[members]
    diff = P[:, None, :] - P[None, :, :]
    d = np.sqrt((diff ** 2).sum(-1))
    A = (d <= radius).astype(int)
    np.fill_diagonal(A, 0)
    ncomp, labels = connected_components(csr_matrix(A), directed=False)
    return labels, ncomp


def axis_multimodality(X, members, nbins=15, min_gap_bins=1):
    """
    Detect non-convexity of the membership function as expressed over input
    variables, by looking for multimodality in the per-axis histograms of the
    block's members. Returns the MAX number of modes found on any single axis
    (that is the number of disjuncts the linguistic term needs to stay convex
    per clause). A single mode on every axis => convex => arity 1.

    This answers the projection/labelability question, which is what the OR
    operator was introduced to solve -- distinct from D*-topology.
    """
    members = np.asarray(sorted(members), dtype=int)
    if len(members) < 4:
        return 1
    P = X[members]
    max_modes = 1
    for ax in range(P.shape[1]):
        vals = P[:, ax]
        hist, _ = np.histogram(vals, bins=nbins)
        occupied = hist > 0
        # count runs of occupied bins separated by >= min_gap_bins empty bins
        modes = 0
        gap = min_gap_bins
        in_run = False
        empty_streak = min_gap_bins
        for b in occupied:
            if b:
                if not in_run and empty_streak >= min_gap_bins:
                    modes += 1
                in_run = True
                empty_streak = 0
            else:
                in_run = False
                empty_streak += 1
        max_modes = max(max_modes, modes)
    return max_modes


def block_arity(Dstar, X, block, mode='dstar', feature_radius_factor=2.5):
    """
    Return (n_disjuncts, component_labels_over_members) for one block.

    feature_radius is derived from the median nearest-neighbor spacing inside
    the block times feature_radius_factor, so it adapts to the block's scale
    (this is what lets a tight blob and a diffuse blob be judged on their own
    terms rather than a global radius).
    """
    members = np.asarray(sorted(block['members']), dtype=int)
    h_birth = block['birth']

    if mode == 'dstar':
        labels, ncomp = _components_dstar(Dstar, members, h_birth)
        return ncomp, labels

    # derive an adaptive feature radius from within-block NN spacing
    P = X[members]
    if len(P) > 1:
        diff = P[:, None, :] - P[None, :, :]
        d = np.sqrt((diff ** 2).sum(-1))
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        radius = np.median(nn) * feature_radius_factor
    else:
        radius = np.inf

    if mode == 'feature':
        nmodes = _components_feature_axis(X, members)
        return nmodes, None

    if mode == 'hybrid':
        _, nc_d = _components_dstar(Dstar, members, h_birth)
        nmodes = _components_feature_axis(X, members)
        # hybrid: report topological components, but flag projection modes too.
        # arity = max(topology, projection) because BOTH kinds of split need OR
        return max(nc_d, nmodes), None

    if mode == 'geometric':
        nc, labels = geometric_nonconvexity(X, members)
        return nc, labels

    raise ValueError(f"unknown mode {mode}")


def _components_feature_axis(X, members):
    return axis_multimodality(X, members)


def geometric_nonconvexity(X, members, occupancy_thresh=0.75, split_k_max=4):
    """
    Detect non-convexity of the membership REGION in feature space (notion #3),
    the notion that actually motivates the OR operator for shapes like rings.

    Method: hull-occupancy. A convex cluster fills its own convex hull; a ring
    (or crescent, or any cluster with a hole/concavity) leaves much of its hull
    empty. We estimate occupancy as the fraction of the hull's bounding region
    that the cluster's density actually covers, via a grid.

      occupancy = (# grid cells containing >=1 member) / (# grid cells inside hull)

    If occupancy >= thresh  -> convex -> arity 1.
    If occupancy <  thresh  -> non-convex -> we then choose how many disjuncts
      by trying k=2..split_k_max sub-splits (via k-medoids on minimax distance)
      and taking the smallest k whose pieces are each hull-convex.

    Returns (n_disjuncts, labels_over_members).
    """
    members = np.asarray(sorted(members), dtype=int)
    m = len(members)
    if m < 6:
        return 1, np.zeros(m, dtype=int)

    P = X[members]

    def occupancy(points):
        try:
            from scipy.spatial import ConvexHull, Delaunay
            if points.shape[0] < points.shape[1] + 2:
                return 1.0
            hull = ConvexHull(points)
            dela = Delaunay(points[hull.vertices])
        except Exception:
            return 1.0
        lo, hi = points.min(0), points.max(0)
        span = np.where(hi > lo, hi - lo, 1.0)
        g = 12
        axes = [np.linspace(lo[d], hi[d], g) for d in range(points.shape[1])]
        mesh = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, points.shape[1])
        inside = dela.find_simplex(mesh) >= 0
        n_inside = max(1, inside.sum())
        cell = np.floor((points - lo) / span * (g - 1)).astype(int)
        cell = np.clip(cell, 0, g - 1)
        occupied_cells = set(map(tuple, cell))
        mesh_cell = np.floor((mesh - lo) / span * (g - 1)).astype(int)
        mesh_cell = np.clip(mesh_cell, 0, g - 1)
        inside_cells = set(map(tuple, mesh_cell[inside]))
        covered = len(occupied_cells & inside_cells)
        raw = covered / len(inside_cells)

        # sampling-matched baseline: draw the SAME number of points uniformly
        # inside the hull and measure their occupancy. This removes the
        # "few points can't fill a fine grid" artifact so the ratio isolates
        # SHAPE non-convexity from sparse sampling.
        rng = np.random.default_rng(0)
        tri = dela.simplices
        # sample uniformly in the hull via barycentric coords over simplices
        n_samp = len(points)
        verts = points[hull.vertices]
        pts_in = verts[dela.simplices]  # (nsimp, d+1, d)
        nsimp = pts_in.shape[0]
        pick = rng.integers(0, nsimp, size=n_samp)
        w = rng.random((n_samp, pts_in.shape[1]))
        w /= w.sum(1, keepdims=True)
        samp = np.einsum('ij,ijk->ik', w, pts_in[pick])
        scell = np.clip(np.floor((samp - lo) / span * (g - 1)).astype(int), 0, g - 1)
        samp_cells = set(map(tuple, scell))
        base = len(samp_cells & inside_cells) / len(inside_cells)
        return raw / (base + 1e-9)

    occ = occupancy(P)
    if occ >= occupancy_thresh:
        return 1, np.zeros(m, dtype=int)

    # non-convex: find smallest k whose pieces are each convex-ish
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    Z = linkage(pdist(P), method='ward')
    for k in range(2, split_k_max + 1):
        labels = fcluster(Z, t=k, criterion='maxclust') - 1
        ok = True
        for c in range(k):
            pc = P[labels == c]
            if len(pc) >= 6 and occupancy(pc) < occupancy_thresh:
                ok = False
                break
        if ok:
            return k, labels
    labels = fcluster(Z, t=split_k_max, criterion='maxclust') - 1
    return split_k_max, labels


def disjunctive_memberships(Dstar, X, block, mode='dstar', conorm='max',
                            yager_q=2.0):
    """
    Build the disjunctive membership function for one block.

    1. decide arity via block_arity(mode)
    2. for each disjunct (component), build a per-disjunct membership from
       minimax distance to that component's members, ramped over the block's
       birth->death interval
    3. recombine the disjunct memberships with the fixed t-conorm

    Returns (mu_combined (n,), n_disjuncts, per_disjunct (k, n)).
    """
    members = np.asarray(sorted(block['members']), dtype=int)
    ncomp, comp_labels = block_arity(Dstar, X, block, mode=mode)
    h_b, h_d = block['birth'], block['death']
    eps = 1e-12
    n = Dstar.shape[0]

    per = np.zeros((ncomp, n))
    for c in range(ncomp):
        comp_members = members[comp_labels == c]
        if len(comp_members) == 0:
            continue
        d_to_c = Dstar[:, comp_members].min(axis=1)
        mu = (h_d - d_to_c) / (h_d - h_b + eps)
        mu = np.clip(mu, 0.0, 1.0)
        core = d_to_c <= h_b + eps
        mu[core] = 1.0
        per[c] = mu

    mu_comb = _tconorm(per, conorm, yager_q)
    return mu_comb, ncomp, per


def _tconorm(per, conorm, q=2.0):
    """Combine disjunct memberships (k, n) into (n,) via a fixed t-conorm."""
    if per.shape[0] == 0:
        return np.zeros(per.shape[1])
    if conorm == 'max':
        return per.max(axis=0)
    if conorm == 'probabilistic':   # 1 - prod(1 - mu)
        return 1.0 - np.prod(1.0 - per, axis=0)
    if conorm == 'lukasiewicz':     # min(1, sum)
        return np.minimum(1.0, per.sum(axis=0))
    if conorm == 'yager':           # min(1, (sum mu^q)^(1/q))
        return np.minimum(1.0, (np.sum(per ** q, axis=0)) ** (1.0 / q))
    raise ValueError(f"unknown conorm {conorm}")
