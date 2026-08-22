#!/usr/bin/env python3
"""Attribute Table 4.1's classification collapse to one upstream function.

Symptom. Re-running `reproduce/tables/table_4_1_mog_baselines.py` at the current
pin (tribble-fis 141596e) against the archived run of record
(`reproduce/outputs/goal-8h-2026-08-11-fullsuite/`, tribble-fis 80e98d7) moves
the two CLASSIFICATION rows by margins no seed spread covers, while the three
REGRESSION rows move slightly the other way and every training time falls 5-7x:

    PhiUSIIL               0.997 +/- 0.001  ->  0.729 +/- 0.023
    RT-IOT2022 (12-class)  0.927 +/- 0.002  ->  0.500 +/- 0.244
    Concrete R2            0.795 +/- 0.025  ->  0.808 +/- 0.030
    Concrete full 2nd      0.852 +/- 0.030  ->  0.867 +/- 0.031
    Bike Sharing R2        0.939 +/- 0.004  ->  0.965 +/- 0.001

Checklist B13 recorded this bump as verified "byte-identical across the bump"
on the strength of the three R2 values, which do match. The accuracy columns
were not part of that check, and they are where the damage is.

Attribution chain, each step holding everything else fixed:

  1. The data is frozen to one .npz before either library is imported, so the
     loader (which lives in the submodule and did NOT change across the bump)
     cannot contribute.
  2. Old library vs new library on that frozen matrix: 0.9952 vs 0.7405.
  3. Bisection over the 48 commits in the range: first bad commit is
     `5237ebe` ("Replace scipy/sklearn stats functions with numba-accelerated
     implementations", #95); its parent `ce4a0fc` is good.
  4. Within #95, each replaced function is restored ONE AT A TIME at the current
     pin. Exactly one restores the accuracy: `wasserstein_distance`.

Root cause. The 1-D Wasserstein distance is the integral of the absolute CDF
difference **with respect to x**:

    W1(u,v) = integral |F_u(x) - F_v(x)| dx

`stats_numba.wasserstein_distance` returns instead the *mean* of the absolute
CDF differences over the union of the support points, with no dx weighting:

    sum(|F_u - F_v|) / len(all_quantiles)

That is a different quantity: dimensionless, bounded in [0,1], and -- the
manipulation check this script prints -- completely INVARIANT to the scale of
the data. Multiply both samples by 1000 and scipy's answer scales by 1000 while
this one does not move at all. A distance in the data's units cannot do that.

Blast radius. The function feeds `gauss_math._pairwise_label_distance`'s
"composite" score, which is the feature-differentiation screen. `mog_classifier`
runs `top_n=5`, so a wrong score selects the wrong five features and the model
is built on them. It is also the metric behind Appendix A.4 and Tables A.1/A.2.
`_pairwise_label_distance`'s own comment says it "squash[es] the unbounded
pooled-std-normalized wasserstein distance" -- the value is already bounded, so
that squash and the composite's three-term balance are both operating on a
quantity they were not designed for.

    uv run --project tribble-fis python \
        reproduce/experiments/diagnose_wasserstein_regression.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def manipulation_check() -> bool:
    """Show the defect directly, before showing what it costs downstream."""
    from scipy.stats import wasserstein_distance as sp_w

    from tribblefis.stats_numba import wasserstein_distance as tf_w

    print("=" * 78)
    print("A. The defect itself")
    print("=" * 78)
    print(f"{'case':40s} {'scipy':>12s} {'stats_numba':>12s} {'ratio':>10s}")
    cases = [
        ("u=[0,1] v=[0,2]        (analytic 0.5)", [0, 1], [0, 2]),
        ("  the same, x10        (analytic 5.0)", [0, 10], [0, 20]),
        ("  the same, x1000    (analytic 500.0)", [0, 10000], [0, 20000]),
        ("shift by 3             (analytic 3.0)", [0, 1, 2], [3, 4, 5]),
        ("  the same, x100     (analytic 300.0)", [0, 100, 200], [300, 400, 500]),
    ]
    for name, u, v in cases:
        s, t = sp_w(u, v), tf_w(u, v)
        ratio = s / t if t else float("nan")
        print(f"{name:40s} {s:12.4f} {t:12.4f} {ratio:9.2f}x")

    print()
    print("  Manipulation check -- scale the data and watch which one moves:")
    rng = np.random.default_rng(0)
    u0, v0 = rng.normal(0, 1, 500), rng.normal(1.5, 2, 500)
    vals = []
    for k in (1, 10, 100, 1000):
        s, t = sp_w(u0 * k, v0 * k), tf_w(u0 * k, v0 * k)
        vals.append(t)
        print(f"    data x{k:<5d}  scipy={s:11.4f}   stats_numba={t:11.6f}")
    invariant = max(vals) - min(vals) < 1e-12
    print(
        f"\n  => stats_numba is {'SCALE-INVARIANT (defective)' if invariant else 'scale-dependent'}"
        f"; scipy scales linearly, as W1 must."
    )
    return invariant


def downstream_cost(npz_path: str, seeds: int = 3) -> None:
    """Restore each replaced function one at a time and re-measure."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import silhouette_score as sk_sil
    from sklearn.model_selection import train_test_split
    from scipy import stats as sp_stats
    from scipy.spatial.distance import jensenshannon as sp_js
    from scipy.stats import wasserstein_distance as sp_wass

    from tribblefis import gauss_math
    from tribblefis.gaussian_classifier import TribbleClassifier

    d = np.load(npz_path, allow_pickle=True)
    X, y = d["X"], d["y"]

    originals = {
        "norm_fit": gauss_math.norm_fit,
        "norm_pdf": gauss_math.norm_pdf,
        "jensenshannon_distance": gauss_math.jensenshannon_distance,
        "wasserstein_distance": gauss_math.wasserstein_distance,
        "silhouette_score": gauss_math.silhouette_score,
        "_kmeans_labels_1d": gauss_math._kmeans_labels_1d,
    }

    def _sk_km(data, k, random_state):
        if k <= 1:
            return np.zeros(len(data), dtype=int)
        return KMeans(n_clusters=k, random_state=random_state).fit_predict(
            np.asarray(data).reshape(-1, 1)
        )

    replacements = {
        "norm_fit": lambda a: sp_stats.norm.fit(a),
        "norm_pdf": lambda x, mu, sd: sp_stats.norm.pdf(x, mu, sd),
        "jensenshannon_distance": sp_js,
        "wasserstein_distance": sp_wass,
        "silhouette_score": sk_sil,
        "_kmeans_labels_1d": _sk_km,
    }

    def measure(label: str) -> float:
        accs = []
        for seed in range(seeds):
            xtr, xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
            model = TribbleClassifier(top_n=5, random_state=seed)
            model.fit(xtr, ytr)
            accs.append(accuracy_score(yte, model.predict(xte)))
        a = np.asarray(accs)
        print(f"  {label:34s} acc = {a.mean():.4f} +/- {a.std():.4f}")
        return float(a.mean())

    print()
    print("=" * 78)
    print(f"B. What it costs downstream (PhiUSIIL {X.shape}, {seeds} seeds, top_n=5)")
    print("=" * 78)
    base = measure("current pin, unmodified")

    for name in replacements:
        setattr(gauss_math, name, replacements[name])
        measure(f"restore scipy/sklearn {name}")
        setattr(gauss_math, name, originals[name])

    for name in replacements:
        setattr(gauss_math, name, replacements[name])
    allrestored = measure("restore ALL of them")
    for name in replacements:
        setattr(gauss_math, name, originals[name])

    print()
    print(
        f"  unmodified {base:.4f} vs all-restored {allrestored:.4f}: "
        f"the gap is {allrestored - base:+.4f}"
    )


def main() -> int:
    npz = os.path.join(os.environ.get("TEMP", "/tmp"), "phi20k_diag.npz")
    if not os.path.exists(npz):
        # Freeze the matrix BEFORE importing either library, so the loader --
        # which lives in the submodule -- cannot vary between arms.
        sys.path.insert(0, os.path.join(ROOT, "tribble-fis", "tribble-tree"))
        import demo_phishing

        demo_phishing.DATA_PATH = os.path.join(
            ROOT, "data", "PhiUSIIL_Phishing_URL_Dataset.csv"
        )
        X, y = demo_phishing.load_data(sample_size=20000, random_state=42)
        np.savez(npz, X=np.asarray(X, dtype=float), y=np.asarray(y))
        print(f"froze PhiUSIIL matrix -> {npz}")

    t0 = time.perf_counter()
    manipulation_check()
    downstream_cost(npz)
    print(f"\n({time.perf_counter() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
