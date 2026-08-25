"""Genuinely non-Euclidean and non-metric dissimilarity datasets + diagnostics.

The existing relational battery (`relationdata.py`) is distance-matrix-only but
still *metric*: tree shortest-path distances satisfy the triangle inequality,
so they never exercise NERFCM's beta-spread or the question this module exists
to answer -- what happens to the minimax pipeline when the input dissimilarity
is not a distance at all.

Two kinds of data live here:

1. **Real dissimilarity families that arise without coordinates** -- DTW on
   time series (non-metric), Levenshtein on strings (metric, non-Euclidean),
   Hamming on categorical records, shortest paths on a community graph
   (metric, non-Euclidean), and cosine dissimilarity on topic vectors
   (non-metric in general). Each generator returns (D, y) with planted
   clusters, so ARI against ground truth is well-defined.

2. **Controlled violation injection** (`violate_pairs`) -- start from a
   Euclidean base and corrupt a chosen fraction of pairs either by
   *stretching* (inflating D_ij, which breaks Euclidean embeddability but
   leaves small edges alone) or by *shortcutting* (deflating D_ij, which is
   exactly the single-linkage bridge failure mode). The two directions are
   predicted to hurt *different* methods, which is the experiment.

Plus the diagnostics that make "non-Euclidean" quantitative rather than
asserted: triangle-inequality violation counts, the classical-MDS Gram
spectrum test for Euclidean embeddability, and an ultrametricity check.

Everything is seeded and deterministic. numpy + scipy only.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

# ---------------------------------------------------------------------------
# Dissimilarity kernels
# ---------------------------------------------------------------------------


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic dynamic-time-warping distance with |a_i - b_j| local cost.

    No window constraint, no normalization. DTW famously violates the
    triangle inequality (warping against a middle series can undercut the
    direct alignment), which is exactly why it is in this battery.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    la, lb = len(a), len(b)
    prev = np.full(lb + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, la + 1):
        cur = np.full(lb + 1, np.inf)
        cost_row = np.abs(a[i - 1] - b)
        for j in range(1, lb + 1):
            cur[j] = cost_row[j - 1] + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return float(prev[lb])


def levenshtein(s: str, t: str) -> int:
    """Edit distance (insert/delete/substitute, unit costs). Metric, but the
    resulting distance matrix is generally not Euclidean-embeddable."""
    ls, lt = len(s), len(t)
    prev = list(range(lt + 1))
    for i in range(1, ls + 1):
        cur = [i] + [0] * lt
        for j in range(1, lt + 1):
            sub = prev[j - 1] + (s[i - 1] != t[j - 1])
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, sub)
        prev = cur
    return prev[lt]


def pairwise(items, fn) -> np.ndarray:
    """Symmetric zero-diagonal dissimilarity matrix from a pairwise callable."""
    n = len(items)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(fn(items[i], items[j]))
            D[i, j] = d
            D[j, i] = d
    return D


# ---------------------------------------------------------------------------
# Diagnostics: how non-metric / non-Euclidean is a dissimilarity matrix?
# ---------------------------------------------------------------------------


def triangle_violation_stats(D: np.ndarray, rtol: float = 1e-9) -> dict:
    """Fraction of pairs (i, j) that violate the triangle inequality through
    at least one witness k, plus the worst relative violation depth.

    A pair (i, j) is violated iff D_ij > min_k (D_ik + D_kj) * (1 + rtol).
    Depth for a violated pair = D_ij / min_k(D_ik + D_kj) - 1  (how far above
    the tightest triangle bound the entry sits).
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    # tightest two-hop bound for every pair: min over k of D_ik + D_kj.
    # One (n, n) pass per k keeps memory at O(n^2).
    best = np.full((n, n), np.inf)
    for k in range(n):
        np.minimum(best, D[:, k][:, None] + D[k, :][None, :], out=best)
    iu = np.triu_indices(n, k=1)
    direct = D[iu]
    bound = best[iu]
    violated = direct > bound * (1.0 + rtol)
    depth = np.where(bound > 0, direct / np.maximum(bound, 1e-300) - 1.0, 0.0)
    return {
        "pair_violation_fraction": float(np.mean(violated)),
        "max_violation_depth": float(depth[violated].max()) if violated.any() else 0.0,
        "n_pairs": int(len(direct)),
    }


def euclidean_embeddability(D: np.ndarray) -> dict:
    """Classical-MDS test: D is Euclidean-embeddable iff the double-centered
    Gram matrix G = -0.5 * J (D o D) J is positive semidefinite.

    Reports the most negative eigenvalue relative to the largest positive one
    (`neg_ratio`); 0 means embeddable, larger means further from any Euclidean
    realization. This is the quantity NERFCM's beta-spread exists to repair.
    """
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * J @ (D * D) @ J
    G = (G + G.T) / 2.0
    eig = np.linalg.eigvalsh(G)
    lam_min, lam_max = float(eig[0]), float(eig[-1])
    neg_ratio = max(0.0, -lam_min) / max(lam_max, 1e-300)
    return {"lambda_min": lam_min, "lambda_max": lam_max, "neg_ratio": neg_ratio}


def is_ultrametric(D: np.ndarray, rtol: float = 1e-9) -> bool:
    """True iff D_ij <= max(D_ik, D_kj) * (1 + rtol) for every triple."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    for k in range(n):
        bound = np.maximum(D[:, k][:, None], D[k, :][None, :])
        if np.any(D > bound * (1.0 + rtol) + 1e-12):
            return False
    return True


# ---------------------------------------------------------------------------
# Controlled violation injection
# ---------------------------------------------------------------------------


def violate_pairs(
    D: np.ndarray,
    rate: float,
    strength: float,
    mode: str = "stretch",
    seed: int = 0,
) -> np.ndarray:
    """Corrupt a fraction of pairs of a dissimilarity matrix, symmetrically.

    mode="stretch":  D_ij <- D_ij * (1 + strength * u),  u ~ U(0, 1).
        Inflates distances. Breaks Euclidean embeddability (and eventually
        metricity) but never creates a small edge, so single-linkage merge
        order is largely preserved.
    mode="shortcut": D_ij <- D_ij * (1 - strength * u),  u ~ U(0, 1).
        Deflates distances. A deep shortcut between clusters is exactly a
        single-linkage bridge: one corrupted pair can fuse two blocks in the
        minimax transform, while an averaging method sees one outlier entry.

    `rate` is the fraction of the n(n-1)/2 pairs corrupted; `strength` scales
    the maximum fractional change. rate=0 or strength=0 returns a copy.
    """
    if mode not in ("stretch", "shortcut"):
        raise ValueError(f"mode must be 'stretch' or 'shortcut', got {mode!r}")
    D = np.asarray(D, dtype=float).copy()
    if rate <= 0.0 or strength <= 0.0:
        return D
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    n_pairs = len(iu)
    n_hit = int(round(rate * n_pairs))
    if n_hit == 0:
        return D
    hit = rng.choice(n_pairs, size=n_hit, replace=False)
    u = rng.uniform(0.0, 1.0, size=n_hit)
    factor = 1.0 + strength * u if mode == "stretch" else 1.0 - strength * u
    D[iu[hit], ju[hit]] *= factor
    D[ju[hit], iu[hit]] = D[iu[hit], ju[hit]]
    return D


# ---------------------------------------------------------------------------
# Dataset generators (each returns D, y)
# ---------------------------------------------------------------------------


def dtw_traces(n_per: int = 20, length: int = 40, seed: int = 201):
    """Three families of 1-D traces under DTW distance (non-metric).

    Family 0: slow sinusoid, random phase.
    Family 1: fast sinusoid, random phase.
    Family 2: degradation-style ramp with a random knee (flat, then decline) --
              the shape RUL trajectories take, tying this battery to the
              CMAPSS thread of the dissertation.
    All traces carry additive Gaussian noise.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, length)
    traces, labels = [], []
    for _ in range(n_per):
        phase = rng.uniform(0, 2 * np.pi)
        traces.append(np.sin(2 * np.pi * 2.0 * t + phase) + rng.normal(0, 0.15, length))
        labels.append(0)
    for _ in range(n_per):
        phase = rng.uniform(0, 2 * np.pi)
        traces.append(np.sin(2 * np.pi * 6.0 * t + phase) + rng.normal(0, 0.15, length))
        labels.append(1)
    for _ in range(n_per):
        knee = rng.uniform(0.3, 0.7)
        ramp = np.where(t < knee, 0.0, -(t - knee) / (1.0 - knee) * 2.0)
        traces.append(ramp + rng.normal(0, 0.15, length))
        labels.append(2)
    D = pairwise(traces, dtw_distance)
    return D, np.asarray(labels, dtype=int)


def edit_strings(
    n_per: int = 20, length: int = 24, max_mutations: int = 5, seed: int = 202
):
    """Three families of DNA-like strings under Levenshtein distance.

    Each family is a random seed string over {A, C, G, T}; members carry
    1..max_mutations random substitutions. Metric but non-Euclidean.
    """
    rng = np.random.default_rng(seed)
    alphabet = np.array(list("ACGT"))
    strings, labels = [], []
    for fam in range(3):
        proto = rng.choice(alphabet, size=length)
        for _ in range(n_per):
            s = proto.copy()
            k = rng.integers(1, max_mutations + 1)
            pos = rng.choice(length, size=k, replace=False)
            s[pos] = rng.choice(alphabet, size=k)
            strings.append("".join(s))
            labels.append(fam)
    D = pairwise(strings, levenshtein).astype(float)
    return D, np.asarray(labels, dtype=int)


def hamming_categorical(
    n_per: int = 20,
    n_attrs: int = 30,
    n_levels: int = 4,
    flip: float = 0.15,
    seed: int = 203,
):
    """Three clusters of categorical records under Hamming dissimilarity
    (fraction of mismatched attributes). Metric; embeddability is the question."""
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for fam in range(3):
        proto = rng.integers(0, n_levels, size=n_attrs)
        for _ in range(n_per):
            r = proto.copy()
            n_flip = rng.binomial(n_attrs, flip)
            if n_flip:
                pos = rng.choice(n_attrs, size=n_flip, replace=False)
                r[pos] = rng.integers(0, n_levels, size=n_flip)
            rows.append(r)
            labels.append(fam)
    X = np.array(rows)
    D = pairwise(list(X), lambda a, b: np.mean(a != b))
    return D, np.asarray(labels, dtype=int)


def graph_communities(
    n_per: int = 20, p_in: float = 0.15, p_out: float = 0.01, seed: int = 204
):
    """Planted 3-community random graph; D = weighted shortest-path distances.

    Metric by construction, but generically non-Euclidean. Components are
    reconnected to the largest one with a single moderate-weight edge each, so
    every distance is finite without redrawing the graph.

    Defaults chosen by a small probe (see NONMETRIC_FINDINGS.md): p_in=0.15 /
    p_out=0.01 is the regime where the planted structure is genuinely
    recoverable (NERFCM(D) reaches ~0.8 ARI) while remaining hard -- denser
    graphs concentrate shortest paths until no method sees the communities,
    which is less diagnostic than hard-but-possible.
    """
    from scipy.sparse.csgraph import connected_components, shortest_path

    rng = np.random.default_rng(seed)
    n = 3 * n_per
    y = np.repeat(np.arange(3), n_per)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            p = p_in if y[i] == y[j] else p_out
            if rng.random() < p:
                w = rng.uniform(1.0, 2.0)
                A[i, j] = A[j, i] = w
    n_comp, comp = connected_components(A > 0, directed=False)
    if n_comp > 1:
        sizes = np.bincount(comp)
        main = int(np.argmax(sizes))
        for c in range(n_comp):
            if c == main:
                continue
            a = int(rng.choice(np.where(comp == c)[0]))
            b = int(rng.choice(np.where(comp == main)[0]))
            w = rng.uniform(4.0, 6.0)
            A[a, b] = A[b, a] = w
    D = shortest_path(A, method="D", directed=False)
    return np.asarray(D, dtype=float), y


def cosine_topics(n_per: int = 20, dim: int = 50, seed: int = 205):
    """Three sparse nonnegative topic prototypes; documents mix their topic
    with background noise; D = 1 - cosine similarity (non-metric in general)."""
    rng = np.random.default_rng(seed)
    protos = []
    for _ in range(3):
        v = np.zeros(dim)
        active = rng.choice(dim, size=12, replace=False)
        v[active] = rng.gamma(2.0, 1.0, size=12)
        protos.append(v / np.linalg.norm(v))
    docs, labels = [], []
    for fam, proto in enumerate(protos):
        for _ in range(n_per):
            noise = rng.gamma(1.0, 0.15, size=dim)
            d = 0.8 * proto + noise
            docs.append(d / np.linalg.norm(d))
            labels.append(fam)
    X = np.array(docs)
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, 0.0)
    D = np.maximum((D + D.T) / 2.0, 0.0)
    return D, np.asarray(labels, dtype=int)


def euclidean_blobs(
    n_per: int = 20, sep: float = 6.5, sigma: float = 1.0, seed: int = 206
):
    """Three Gaussian blobs in 2-D at pairwise separation `sep`. Returns (D, y, X).

    sep=6.5 is calibrated so the CLEAN baseline is minimax-separable (coverage
    cover finds k=3 at ARI 1.0 on every replicate seed; at sep=4 single-linkage
    already chains the blobs and the violation sweep would start from the floor), while deep
    multiplicative shortcuts can still bridge the clusters at strength -> 1.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [sep, 0.0], [sep / 2.0, sep * np.sqrt(3) / 2.0]])
    X = np.vstack([rng.normal(c, sigma, size=(n_per, 2)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    D = squareform(pdist(X))
    return D, y, X


def spiked_random(n: int = 20, spike: float = 40.0, seed: int = 208):
    """Positive control for NERFCM's beta-spread: a random dissimilarity with
    one pair inflated far above every two-hop path.

    Realistic non-Euclidean families (DTW, cosine, stretched Euclidean) are
    formally inadmissible (negative Gram eigenvalues) yet never drive the
    NERFCM relational update negative -- beta stays 0 across every (c, m)
    probed. What DOES fire it is a spike-type violation: one entry several
    times larger than the tightest triangle bound. This generator exists so
    the diagnostics table contains a row where beta(D) > 0, proving the
    harness can detect activation and making 'beta = 0 everywhere else' a
    finding rather than a blind spot. No cluster structure; y is all zeros.
    """
    rng = np.random.default_rng(seed)
    D = rng.uniform(1.0, 2.0, size=(n, n))
    D = (D + D.T) / 2.0
    np.fill_diagonal(D, 0.0)
    D[0, 1] = D[1, 0] = spike
    return D, np.zeros(n, dtype=int)


def relational_nested_hierarchy(seed: int = 207, leaf_noise: float = 0.05):
    """Two-level relational hierarchy: 3 super-clusters x 2 sub-clusters,
    8 leaves per sub-cluster (n = 48), as tree path distances.

    Unlike `relationdata.multi_scale_hierarchy` this returns BOTH truth levels
    explicitly, uses fixed cluster sizes (no RNG in the tree topology), and
    keeps the scales cleanly separated: intra-sub 0.3, sub-sep 2.0,
    super-sep 8.0. Multiplicative noise is applied to the final distances.

    Returns (D, y_fine, y_coarse): y_fine has 6 labels, y_coarse 3.
    """
    rng = np.random.default_rng(seed)
    n_super, n_sub, n_leaf = 3, 2, 8
    # Build leaf-to-leaf distances directly from the implicit tree:
    # same sub-cluster: through the sub-root, 0.3 + 0.3
    # same super, different sub: 0.3 + 2.0 + 2.0 + 0.3
    # different super: 0.3 + 2.0 + 8.0 + 2.0 + 0.3
    n = n_super * n_sub * n_leaf
    y_fine = np.repeat(np.arange(n_super * n_sub), n_leaf)
    y_coarse = y_fine // n_sub
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if y_fine[i] == y_fine[j]:
                d = 0.6
            elif y_coarse[i] == y_coarse[j]:
                d = 4.6
            else:
                d = 12.6
            D[i, j] = D[j, i] = d
    noise = rng.uniform(1.0 - leaf_noise, 1.0 + leaf_noise, size=(n, n))
    noise = (noise + noise.T) / 2.0
    D = D * noise
    np.fill_diagonal(D, 0.0)
    return D, y_fine, y_coarse


# The non-Euclidean battery run by run_nonmetric.py: name -> (generator, k_true).
BATTERY = {
    "dtw_traces": (dtw_traces, 3),
    "edit_strings": (edit_strings, 3),
    "hamming_categorical": (hamming_categorical, 3),
    "graph_communities": (graph_communities, 3),
    "cosine_topics": (cosine_topics, 3),
}


if __name__ == "__main__":
    for name, (fn, k) in BATTERY.items():
        D, y = fn()
        tv = triangle_violation_stats(D)
        em = euclidean_embeddability(D)
        print(
            f"{name}: n={D.shape[0]} k={k} "
            f"TI-violated pairs={tv['pair_violation_fraction']:.3f} "
            f"neg-eig ratio={em['neg_ratio']:.4f}"
        )
