"""
Synthetic test battery + metrics for the iVAT->MF mappings.

Datasets are chosen to probe the specific failure modes we care about:

  1. two_gaussians      - easy baseline; both mappings should pass.
  2. bridged_gaussians  - THE killer test. A thin bridge of points between two
                          blobs. Single-linkage (hence iVAT, hence our MFs)
                          is known to chain across bridges. If Mapping 2 dies
                          anywhere, it dies here.
  3. concentric_rings   - non-convex, density-connected. FCM cannot do this;
                          minimax/SL structure can. Mapping 2 should WIN here.
  4. varying_density    - three blobs of very different spread. Tests whether
                          persistence heights adapt to local scale.
  5. uniform_noise      - no cluster structure. Tests that we DON'T hallucinate
                          confident memberships (tendency awareness).

Metrics:
  - Adjusted Rand Index (ARI) of argmax-defuzzified labels vs ground truth.
  - Cluster-count sensitivity: rerun at c-1, c, c+1 and report ARI stability.
    (This is the headline claim - that we avoid FCM's c-sensitivity.)
  - Convexity check on 1-D projections of each generated MF.
  - Coverage: fraction of points with max membership > 0.
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans

import ivat_mf as im


rng = np.random.default_rng(42)


def _rng(seed):
    """Deterministic per-dataset RNG so every method sees identical data.
    Falls back to the module rng only if seed is None (legacy behavior)."""
    return np.random.default_rng(seed) if seed is not None else rng


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def two_gaussians(n=120, seed=101):
    rng = _rng(seed)
    a = rng.normal([0, 0], 0.5, (n // 2, 2))
    b = rng.normal([5, 0], 0.5, (n // 2, 2))
    X = np.vstack([a, b])
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    return X, y


def bridged_gaussians(n=120, bridge=12, seed=102):
    rng = _rng(seed)
    half = (n - bridge) // 2
    a = rng.normal([0, 0], 0.5, (half, 2))
    b = rng.normal([5, 0], 0.5, (half, 2))
    # bridge: points strung along the line between the two centers
    t = np.linspace(0.15, 0.85, bridge)
    bx = np.c_[t * 5, np.zeros(bridge)] + rng.normal(0, 0.15, (bridge, 2))
    X = np.vstack([a, b, bx])
    y = np.array([0] * half + [1] * half + [-1] * bridge)  # -1 = bridge/ambiguous
    return X, y


def concentric_rings(n=160, seed=103):
    rng = _rng(seed)
    m = n // 2
    theta = rng.uniform(0, 2 * np.pi, m)
    inner = np.c_[np.cos(theta), np.sin(theta)] * 1.0 + rng.normal(0, 0.08, (m, 2))
    theta2 = rng.uniform(0, 2 * np.pi, m)
    outer = np.c_[np.cos(theta2), np.sin(theta2)] * 4.0 + rng.normal(0, 0.12, (m, 2))
    X = np.vstack([inner, outer])
    y = np.array([0] * m + [1] * m)
    return X, y


def varying_density(n=150, seed=104):
    rng = _rng(seed)
    a = rng.normal([0, 0], 0.25, (n // 3, 2))
    b = rng.normal([4, 0], 0.8, (n // 3, 2))
    c = rng.normal([8, 5], 1.5, (n // 3, 2))
    X = np.vstack([a, b, c])
    y = np.array([0] * (n // 3) + [1] * (n // 3) + [2] * (n // 3))
    return X, y


def uniform_noise(n=120, seed=105):
    rng = _rng(seed)
    X = rng.uniform(-5, 5, (n, 2))
    y = np.array([-1] * n)  # no structure
    return X, y


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def score_ari(U, y_true, Dblock=None):
    """ARI on non-ambiguous points (ground-truth label >= 0)."""
    if Dblock is not None:
        labels = im.hard_labels_proximity(U, Dblock)
    else:
        labels = im.hard_labels_from_U(U)
    mask = y_true >= 0
    if mask.sum() == 0:
        return np.nan
    return adjusted_rand_score(y_true[mask], labels[mask])


def coverage(U, tol=1e-6):
    return float(np.mean(U.max(axis=0) > tol))


def univariate_convexity(U, X, axis=0, nbins=20):
    """
    Project points onto one feature axis, bin, and check whether each MF is
    (approximately) unimodal/convex along that axis. Returns fraction of MFs
    that are unimodal (no more than one 'up-then-down' sign change in the
    binned membership profile).
    """
    x = X[:, axis]
    bins = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, bins) - 1, 0, nbins - 1)
    unimodal_flags = []
    for k in range(U.shape[0]):
        prof = np.array([U[k, idx == b].mean() if np.any(idx == b) else np.nan
                         for b in range(nbins)])
        prof = prof[~np.isnan(prof)]
        if len(prof) < 3:
            unimodal_flags.append(True)
            continue
        # count sign changes in the discrete derivative
        d = np.diff(prof)
        signs = np.sign(d[np.abs(d) > 1e-9])
        sign_changes = int(np.sum(signs[1:] != signs[:-1])) if len(signs) > 1 else 0
        # unimodal (convex-ish bump) allows at most 1 sign change (up then down)
        unimodal_flags.append(sign_changes <= 1)
    return float(np.mean(unimodal_flags))


def c_sensitivity(Dstar, y_true, c_true):
    """
    Rerun Mapping 2 at c_true-1, c_true, c_true+1 and report ARI at each.
    Small variation => robust to the cluster-count choice (the headline claim).
    """
    out = {}
    for c in [max(2, c_true - 1), c_true, c_true + 1]:
        U, _, Dblock = im.mapping2_persistence(Dstar, c)
        out[c] = score_ari(U, y_true, Dblock)
    return out


# ---------------------------------------------------------------------------
# baseline for comparison: k-means argmax (vector-space) as a sanity anchor
# ---------------------------------------------------------------------------

def kmeans_ari(X, y_true, c):
    mask = y_true >= 0
    if mask.sum() == 0:
        return np.nan
    km = KMeans(n_clusters=c, n_init=10, random_state=0).fit(X)
    return adjusted_rand_score(y_true[mask], km.labels_[mask])
