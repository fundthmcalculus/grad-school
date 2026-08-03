"""Classification identification on PhiUSIIL: the same contest, at scale.

Concrete answered the identification question on 824 rows and 8 features, where
k-means costs nothing and the construction's cost is dominated by fitting
candidate mixtures rather than by row count. The two routes scale differently,
so that result cannot be extrapolated — which is what this module is for.
PhiUSIIL is 235,795 x 50, binary.

## How the two routes map onto a classifier

For classification the MoG construction gives **one rule per class**: for each
(feature, class) it fits a 1-D Gaussian mixture, ORs the components within the
feature, and ANDs across features. The rule count is K and is not a free
parameter — so, unlike the regression case, there is no rule count to sweep.

What *is* free is the number of components per (feature, class) — the arity of
the disjunction. So the classical route substitutes there, in the way the
literature actually does it for classification: **cluster within each class**,
and let each cluster contribute one Gaussian per feature to that class's
membership list. c clusters per class gives the same structure the construction
produces with c mixture components, and both are read by the same
`simple_gaussian_predict`.

That keeps the comparison matched at the level that matters — same rule count,
same component count, same prediction path — and leaves exactly one thing
different: whether the Gaussians are placed by a per-feature 1-D mixture fit or
by a multivariate clustering of the class.

## What is charged to whom

The construction pays for its own feature screening (`calculate_gaussian_correlation`
is O(M^2) and on 50 features that is not free). The classical route is given the
same retained feature set but is **not** charged for the screening, because
choosing features is not part of what clustering does. That asymmetry favours
the classical route in the timing column and is stated in the table rather than
hidden: if the construction still wins on time, it wins against a handicap.
"""

from __future__ import annotations

import time

import numpy as np

SIGMA_FLOOR_FRAC = 0.02


def load(sample_size):
    """(X, y) from the repo's own PhiUSIIL loader.

    y stays a pandas Series indexed like X. `calculate_gaussian_correlation`
    and `create_gaussian_membership_dict` both index it by label and call
    `.unique()`, so handing them a bare ndarray fails deep inside the library
    rather than here.
    """
    import pandas as pd
    import _fuzzy_models as FM
    out = FM.load_phiusiil(sample_size=sample_size)
    if out is None:
        raise SystemExit("PhiUSIIL unavailable; see data/.gitignore for recovery.")
    X, y = out
    return X, pd.Series(np.asarray(y), index=X.index, name="y_bucket")


def screen(X, y, top_n):
    """The construction's feature screening. Timed separately and charged to it."""
    from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                       take_top_features)
    start = time.perf_counter()
    diffs = calculate_gaussian_correlation(X, y)
    _, features = take_top_features(diffs, top_n=top_n)
    return list(features), time.perf_counter() - start


def construction(X, y, features, n_gaussians=-1):
    """Identify by the Gaussian construction. (model, seconds)."""
    from tribblefis.gauss_math import create_gaussian_membership_dict
    start = time.perf_counter()
    model = create_gaussian_membership_dict(X, y, top_n_var_names=features,
                                            n_gaussians=n_gaussians)
    return model, time.perf_counter() - start


def classical(X, y, features, c, method="kmeans", seed=0):
    """Identify by clustering within each class. (model, seconds).

    One clustering per class over the retained features; each cluster
    contributes one Gaussian per feature, so a class ends up with c components
    on every feature — the same shape the construction produces with c mixture
    components, and read by the same predictor.
    """
    from tribblefis.gauss_data import (FeatureModel, GaussianMembership,
                                       GaussianMixtureModel, LabelModel)

    y_arr = np.asarray(y)
    classes = list(dict.fromkeys(y_arr))
    Xf = X[features]
    start = time.perf_counter()

    # cluster assignment per class, done once and reused for every feature
    assign = {}
    for cls in classes:
        rows = Xf.to_numpy(dtype=float)[y_arr == cls]
        k = min(c, len(rows)) or 1
        if k == 1 or len(np.unique(rows, axis=0)) < k:
            assign[cls] = (rows, np.zeros(len(rows), dtype=int), 1)
            continue
        if method == "kmeans":
            from sklearn.cluster import KMeans
            labels = KMeans(n_clusters=k, n_init=3, random_state=seed).fit_predict(rows)
        elif method == "fcm":
            from tribbleclustering.fcm import fuzzy_c_means
            _centres, u = fuzzy_c_means(rows, k)
            u = np.asarray(u, dtype=float)
            if u.shape[0] != len(rows):
                u = u.T
            labels = np.argmax(u, axis=1)
        else:
            raise ValueError(f"unknown method {method!r}")
        assign[cls] = (rows, labels, k)

    feature_models = {}
    for j, f in enumerate(features):
        col_all = Xf[f].to_numpy(dtype=float)
        rng = float(col_all.max() - col_all.min()) or 1.0
        floor = SIGMA_FLOOR_FRAC * rng
        label_models = {}
        for cls in classes:
            rows, labels, k = assign[cls]
            mfs = []
            for m in range(k):
                vals = rows[labels == m, j]
                mu = float(vals.mean()) if vals.size else float(col_all.mean())
                sd = float(vals.std()) if vals.size > 1 else floor
                mfs.append(GaussianMembership.create(mu, max(sd, floor)))
            label_models[cls] = LabelModel(memberships=mfs)
        feature_models[f] = FeatureModel(label_models=label_models)

    model = GaussianMixtureModel(feature_models=feature_models)
    return model, time.perf_counter() - start


def accuracy(model, X_te, y_te, features):
    """Argmax over the class rules — the shipped prediction path, not a stand-in."""
    from tribblefis.gauss_math import simple_gaussian_predict
    pred = simple_gaussian_predict(X_te[features], model.to_simple_model())
    return float(np.mean(np.asarray(pred).astype(str) == np.asarray(y_te).astype(str)))


def n_membership_fns(model):
    return sum(len(lm.memberships) for fm in model.feature_models.values()
               for lm in fm.label_models.values())
