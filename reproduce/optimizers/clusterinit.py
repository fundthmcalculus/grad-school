"""The old way: place the membership functions with k-means or fuzzy c-means.

Before the Gaussian construction, the standard route to a rule base from data
was to cluster and read the rules off the clusters. This module builds that
starting point, in exactly the parameter layout the rest of the study uses, so
it can be dropped in beside `hot` and `cold` and compared under the same
objective, the same box and the same budget.

## What is substituted, and what is held fixed

Each Gaussian slot in the model belongs to a (feature, output bucket) pair — the
construction fits a one-dimensional mixture per pair and takes its components.
This module replaces *that fit* with a one-dimensional k-means or FCM over the
same values, and takes the cluster centres as the means and the within-cluster
spreads as the widths.

So the structure is held fixed — which features are retained, how many
components sit on each, which bucket they serve — and only the placement method
changes. That is deliberate and it is the same discipline as the cold start:
change one thing, and the comparison means something. It also bounds the claim.
This measures **k-means against a Gaussian mixture as a way of placing membership
functions**, not the classical pipeline as a whole.

The fuller "old way" — cluster the joint input–output space, let the cluster
count *be* the rule count, read each rule off a multivariate cluster — changes
the structure rather than the placement, so it belongs with the structure search
in `run_structure_study.py`. Timing claims about the classical pipeline
end-to-end should be made there, not here: the inits in this module reuse the
structure the construction discovered, so their cost is a placement cost and is
reported as one.

FCM is the author's own `tribbleclustering.fcm.fuzzy_c_means`, not a
reimplementation, for the same reason the objective is imported rather than
rewritten.
"""

from __future__ import annotations

import time

import numpy as np

# A width can never be zero: a zero-width Gaussian is a delta, contributes no
# gradient and makes the slot dead. Bounded below by a fraction of the feature's
# own range, which is the same floor `build_param_bounds` uses.
SIGMA_FLOOR_FRAC = 0.02


def _slot_groups(model):
    """{(feature, label): [(slot_index, component_index), ...]} in slot order."""
    from tribblefis.refine import _iter_gaussian_slots

    groups, n = {}, 0
    for pos, (fname, label, i, _mf) in enumerate(_iter_gaussian_slots(model)):
        groups.setdefault((fname, label), []).append((pos, i))
        n = pos + 1
    return groups, n


def _kmeans_1d(values, k, seed):
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(values.reshape(-1, 1))
    centres = km.cluster_centers_.ravel()
    labels = km.labels_
    spreads = np.array(
        [values[labels == j].std() if np.any(labels == j) else 0.0 for j in range(k)]
    )
    return centres, spreads


def _fcm_1d(values, k, m=2.0):
    """Author's FCM. Widths are the membership-weighted standard deviations.

    A crisp within-cluster std would throw away the thing that makes FCM
    different from k-means, so the spread is weighted by u^m — the same weights
    the algorithm uses to place the centre it is being measured around.
    """
    from tribbleclustering.fcm import fuzzy_c_means

    centres, u = fuzzy_c_means(values.reshape(-1, 1), k, m)
    centres = np.asarray(centres, dtype=float).ravel()
    u = np.asarray(u, dtype=float)
    if u.shape[0] != len(values):  # tolerate (k, n) orientation
        u = u.T
    w = u**m
    spreads = np.array(
        [
            (
                float(np.sqrt(np.average((values - centres[j]) ** 2, weights=w[:, j])))
                if w[:, j].sum() > 0
                else 0.0
            )
            for j in range(k)
        ]
    )
    return centres, spreads


def cluster_params(model, X_train, y_train, method, seed=0, bounds=None):
    """(vector, seconds) — antecedents placed by `method`, in slot order.

    `seconds` times the clustering alone. It is a *placement* cost: the
    structure it fills was discovered by the construction, so this is not the
    cost of an alternative pipeline and must not be quoted as one.
    """
    groups, n_slots = _slot_groups(model)
    buckets = np.asarray(y_train["y_bucket"], dtype=int)
    vec = np.zeros(2 * n_slots, dtype=float)

    start = time.perf_counter()
    for (fname, label), members in groups.items():
        col = X_train[fname].to_numpy(dtype=float)
        values = col[buckets == label]
        k = len(members)
        rng = float(col.max() - col.min()) or 1.0
        floor = SIGMA_FLOOR_FRAC * rng

        if len(values) < k or len(np.unique(values)) < k:
            # Not enough distinct values to support k clusters. Spread the
            # centres evenly over the observed range instead of failing: the
            # study needs a valid starting point from every method on every
            # seed, and a degenerate slot is a fair thing for the method to be
            # judged on rather than a reason to drop the cell.
            centres = (
                np.linspace(values.min(), values.max(), k)
                if len(values)
                else np.linspace(col.min(), col.max(), k)
            )
            spreads = np.full(k, floor)
        elif method == "kmeans":
            centres, spreads = _kmeans_1d(values, k, seed)
        elif method == "fcm":
            centres, spreads = _fcm_1d(values, k)
        else:
            raise ValueError(f"unknown method {method!r}")

        order = np.argsort(centres)  # stable slot->component assignment
        centres, spreads = centres[order], spreads[order]
        for pos, i in members:
            j = min(i, k - 1)
            vec[2 * pos] = centres[j]
            vec[2 * pos + 1] = max(float(spreads[j]), floor)
    seconds = time.perf_counter() - start

    if bounds is not None:
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        vec = np.clip(vec, lo, hi)
    return vec, seconds
