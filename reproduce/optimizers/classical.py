"""The classical route: cluster the joint space, and let the clusters be the rules.

This is how a fuzzy rule base was usually identified from data before the
Gaussian construction — Chiu's subtractive clustering, Bezdek-style FCM
identification, and the c-means family generally. One procedure, three steps:

  1. cluster the joint input-output space into `c` clusters;
  2. **each cluster becomes one rule** — so the cluster count *is* the rule
     count, chosen by the practitioner rather than derived;
  3. project each cluster onto each input axis to get that rule's membership
     function: centre from the cluster's centre, width from its spread.

The consequents are then solved by the same closed-form ridge-TSK primitive
everything else in this project uses, so the comparison is about *rule
identification* and not about how the consequents are fitted.

## Why this is the fair venue for the timing claim

The k-means and FCM inits in `clusterinit.py` replace only the *placement* of
membership functions inside a structure the Gaussian construction had already
discovered — so they cost construction + clustering, and cannot be shown as a
cheaper alternative to it. Here nothing is inherited. There is no feature
screening, no output partition, no per-bucket mixture fitting: clustering
replaces the construction outright, and the two can be timed head to head.

## Two asymmetries to keep in view when reading a result

**The rule count is an input here and an output there.** The classical route has
to be told `c`. The construction derives its rule count from the output
partition. Sweeping `c` is therefore part of running this arm honestly — a
single `c` chosen to look good would be the easiest way to rig this comparison,
in either direction.

**Parameter counts differ.** A classical model has `2 x n_features x c` free
antecedent parameters; the construction's count depends on how many mixture
components it chose per feature and bucket. The two are not the same search
problem, so an equal *evaluation* budget is not an equal *per-parameter* budget,
and the table says so.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

SIGMA_FLOOR_FRAC = 0.02


def _joint_matrix(X, y_values):
    """Standardized [X | y] — the space the clustering runs in.

    Standardized because the clustering is Euclidean and the target must not
    dominate or vanish purely through its units. X arrives already scaled by the
    shared preprocessing; y is scaled here to match.
    """
    y = np.asarray(y_values, dtype=float).reshape(-1, 1)
    y_sd = float(np.std(y)) or 1.0
    return np.hstack([np.asarray(X, dtype=float), (y - np.mean(y)) / y_sd])


def _cluster_joint(J, c, method, seed):
    """(hard labels, centres) for the joint-space clustering."""
    if method == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=c, n_init=10, random_state=seed).fit(J)
        return km.labels_, km.cluster_centers_
    if method == "fcm":
        # Supplied by `--with-editable tribble-cluster` on this script's
        # documented invocation. Not a tribble-fis dependency any more --
        # tribble-fis#233 moved it to an optional extra, since nothing in
        # `tribblefis` imports it and it made a C toolchain a hard
        # requirement of `uv sync`. See `clusterinit._import_fcm`.
        from tribbleclustering.fcm import fuzzy_c_means

        centres, u = fuzzy_c_means(J, c)
        centres = np.asarray(centres, dtype=float)
        u = np.asarray(u, dtype=float)
        if u.shape[0] != J.shape[0]:
            u = u.T
        return np.argmax(u, axis=1), centres
    raise ValueError(f"unknown method {method!r}")


def identify(X_train, y_train_values, c, method="kmeans", seed=0):
    """Build a rule base by clustering the joint space. Returns (model, y_df,
    bucket_mean, feature_names, seconds).

    `seconds` is the honest cost of this identification: the clustering and the
    projection, with nothing inherited from the Gaussian construction.
    """
    from tribblefis.gauss_data import (
        FeatureModel,
        GaussianMembership,
        GaussianMixtureModel,
        LabelModel,
    )

    features = list(X_train.columns)
    start = time.perf_counter()

    J = _joint_matrix(X_train, y_train_values)
    labels, _centres = _cluster_joint(J, c, method, seed)

    # Project each cluster onto each input axis. Empty or single-point clusters
    # keep a floored width rather than collapsing to a delta, which would make
    # the rule dead and the comparison unfair to the method rather than
    # informative about it.
    feature_models = {}
    for f in features:
        col = X_train[f].to_numpy(dtype=float)
        rng = float(col.max() - col.min()) or 1.0
        floor = SIGMA_FLOOR_FRAC * rng
        label_models = {}
        for k in range(c):
            member = col[labels == k]
            mu = float(member.mean()) if member.size else float(col.mean())
            sigma = float(member.std()) if member.size > 1 else floor
            label_models[k] = LabelModel(
                memberships=[GaussianMembership.create(mu, max(sigma, floor))]
            )
        feature_models[f] = FeatureModel(label_models=label_models)
    model = GaussianMixtureModel(feature_models=feature_models)

    # The cluster assignment plays the role the output partition plays in the
    # MoG pipeline: it is what the consequent solver fits one consequent per.
    y_vals = pd.Series(
        np.asarray(y_train_values, dtype=float).ravel(),
        index=X_train.index,
        name="y_value",
    )
    bucket = pd.Series(labels, index=X_train.index, name="y_bucket")
    y_df = pd.concat([bucket, y_vals], axis=1)
    bucket_mean = np.array(
        [
            (
                float(y_vals.to_numpy()[labels == k].mean())
                if np.any(labels == k)
                else float(y_vals.mean())
            )
            for k in range(c)
        ]
    )

    seconds = time.perf_counter() - start
    return model, y_df, bucket_mean, features, seconds
