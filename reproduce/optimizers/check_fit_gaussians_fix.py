#!/usr/bin/env python3
"""Does the k-means BIC selector pick the same components the EM one did?

    uv run --project tribble-fis --with-editable tribble-cluster --with scikit-learn \
        python reproduce/optimizers/check_fit_gaussians_fix.py

`fit_gaussians` used to choose its component count by fitting a full EM mixture
at every candidate k, keeping only the winning *count*, and then running a
k-means at that k to get the placement it had just thrown away. The replacement
scores each candidate straight off the k-means partition and keeps the winner's
components.

That is a different estimator -- the hard-assignment MLE rather than the EM
optimum -- so it can disagree at the margin. This script measures how often, on
the two datasets the identification study uses, by running both selectors over
every (feature, label) group and comparing:

* the component count chosen, group by group;
* the resulting membership functions, so a matching count with different
  placement does not pass unnoticed;
* the time each takes.

Where the counts agree the memberships are identical to the last digit, because
the placement path -- k-means, then the mean and standard deviation of each
cluster -- is untouched. The *only* thing this change can move is which k wins,
and it now wins by scoring the mixture that will actually be built rather than
an EM mixture that was fitted, consulted for its component count, and
discarded.

The old path is reproduced here verbatim rather than imported, because it no
longer exists in the library. If it is ever changed back, this script is what
says whether the change was neutral.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "reproduce"))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))


def old_fit(data, max_gaussians=4, random_state=42):
    """`find_optimal_gaussians` + `fit_gaussians`, as they stood before the fix.

    Verbatim, including the prefix cap the caller could not see, the four
    discarded EM fits, and the `stats.norm.fit` per surviving cluster.
    """
    from scipy import stats
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    data = np.asarray(data, dtype=float).reshape(-1, 1)
    data = data[:20_000]
    if len(data) == 0:
        return [], 0

    if len(data) < 2:
        n_gaussians = 1
    else:
        bics = []
        n_components_range = range(1, min(max_gaussians, len(data)) + 1)
        for n in n_components_range:
            gmm = GaussianMixture(n_components=n, random_state=42)
            gmm.fit(data)
            if not gmm.converged_:
                continue
            bics.append(gmm.bic(data))
        n_gaussians = n_components_range[int(np.argmin(bics))]

    n_clusters = min(n_gaussians, len(data))
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data.copy())

    out = []
    for i in range(n_gaussians):
        cluster = data[labels == i].flatten()
        mu, std = stats.norm.fit(cluster)
        if np.isfinite(mu) and np.isfinite(std):
            out.append((float(mu), float(std)))
    return out, len(out)


def new_fit(data, max_gaussians=4, random_state=42):
    from tribblefis.gauss_math import fit_gaussian_mixture_1d

    # Matched to the old path's sample: this script is measuring the selector,
    # not the cap. The cap is measured by the sweeps.
    data = np.asarray(data, dtype=float).ravel()[:20_000]
    mfs, _k = fit_gaussian_mixture_1d(
        data, n_gaussians=0, max_gaussians=max_gaussians, random_state=random_state
    )
    return [(float(m.mu), float(m.sigma)) for m in mfs], len(mfs)


def em_bic_curve(data, max_gaussians=4):
    """The old selector's own BIC at every k -- the referee for a disagreement.

    A different selector picking a different k only matters if the criterion had
    a strong opinion. This returns the EM BIC curve so a disagreement can be
    priced: `(BIC at the k-means choice - BIC at the EM choice)` as a fraction
    of the curve's own range.

    Read it as a diagnostic, not a verdict. It is denominated in the *old*
    criterion, and the old criterion scored a model nobody ever built: the EM
    mixture was fitted, its k was kept, and its parameters were thrown away in
    favour of a k-means partition. Scoring the model that is actually delivered
    is the point of the change, so of course the two disagree where the delivered
    model and the EM mixture differ. The arbiter is held-out performance in the
    identification sweep, not this column.
    """
    from sklearn.mixture import GaussianMixture

    data = np.asarray(data, dtype=float).reshape(-1, 1)[:20_000]
    out = {}
    for n in range(1, min(max_gaussians, len(data)) + 1):
        gmm = GaussianMixture(n_components=n, random_state=42)
        gmm.fit(data)
        if gmm.converged_:
            out[n] = float(gmm.bic(data))
    return out


def groups_concrete(buckets=3):
    import _fuzzy_models as FM
    import table_concrete_reconciliation as TCR
    from tribblefis.regression import partition_output

    loaded = FM.load_concrete()
    if loaded is None:
        raise SystemExit("Concrete unavailable (no CSV, no UCI mirror).")
    prep = TCR.prepare(*loaded)
    Xt, y = prep["Xt"], prep["y"]
    y_all, _mean = partition_output(buckets, y["y_value"])
    labels = y_all["y_bucket"]
    for col in Xt.columns:
        for lv in labels.unique():
            yield f"{col}/{lv}", Xt[col][labels == lv].dropna().to_numpy(float)


def groups_phishing(n, top_n=10):
    import phishing as P

    X, y = P.load(n)
    features, _s = P.screen(X, y, top_n)
    for col in features:
        for lv in y.unique():
            yield f"{col}/{lv}", X[col][y == lv].dropna().to_numpy(float)


def run(name, groups):
    print(f"\n=== {name} ===")
    agree = disagree = 0
    deltas, moved = [], []
    t_old = t_new = 0.0

    for key, data in groups:
        if len(data) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            old, k_old = old_fit(data)
            t_old += time.perf_counter() - t0
            t0 = time.perf_counter()
            new, k_new = new_fit(data)
            t_new += time.perf_counter() - t0

        if k_old == k_new:
            agree += 1
            # Same count is not the same fit. Compare the sorted centres,
            # normalized by the column's spread so the number means something
            # across features on wildly different scales.
            scale = float(np.std(data)) or 1.0
            a = np.sort([m for m, _ in old])
            b = np.sort([m for m, _ in new])
            shift = float(np.max(np.abs(a - b))) / scale if len(a) == len(b) else np.nan
            moved.append(shift)
        else:
            disagree += 1
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                curve = em_bic_curve(data)
            if k_old in curve and k_new in curve and len(curve) > 1:
                span = max(curve.values()) - min(curve.values())
                cost = (curve[k_new] - curve[k_old]) / span if span > 0 else 0.0
            else:
                cost = float("nan")
            deltas.append((key, k_old, k_new, cost))

    total = agree + disagree
    print(f"  groups                {total}")
    print(f"  same component count  {agree}/{total} ({100.0 * agree / max(total, 1):.0f}%)")
    if deltas:
        costs = np.array([c for *_r, c in deltas if np.isfinite(c)])
        print(f"  disagreements         {len(deltas)}  "
              f"(EM-BIC given up, as a fraction of that group's own BIC range: "
              f"median {np.median(costs):.3f}, worst {costs.max():.3f})")
        for key, ko, kn, cost in deltas[:12]:
            print(f"    {key:<40} EM {ko} -> k-means {kn}   costs {cost:+.3f} of the BIC range")
        if len(deltas) > 12:
            print(f"    ... and {len(deltas) - 12} more")
    if moved:
        arr = np.array([m for m in moved if np.isfinite(m)])
        print(f"  centre shift where counts agree: median {np.median(arr):.4f} s.d., "
              f"max {arr.max():.4f} s.d.")
    print(f"  selection time        EM {1000 * t_old:8.1f} ms   "
          f"k-means {1000 * t_new:8.1f} ms   ({t_old / max(t_new, 1e-9):.1f}x)")
    return total, agree, t_old, t_new


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phishing-rows", type=int, default=50_000)
    ap.add_argument("--buckets", type=int, default=3)
    args = ap.parse_args()

    run(f"concrete, {args.buckets} output buckets", groups_concrete(args.buckets))
    run(f"phiusiil, {args.phishing_rows:,} rows", groups_phishing(args.phishing_rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
