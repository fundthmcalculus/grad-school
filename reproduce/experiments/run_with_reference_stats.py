#!/usr/bin/env python3
"""Run a table generator with chosen `gauss_math` helpers replaced by references.

`tribble-fis` #95 replaced six scipy/sklearn functions with numpy/numba ones, and
**two of the six changed results**, by different paths:

    wasserstein_distance   -> the feature-differentiation screen  (checklist B14)
    _kmeans_labels_1d      -> the 1-D mixture initialization      (checklist D8)

Neither correction alone explains the whole of Table 4.1's drift, which is what
made the second one easy to miss. This runner substitutes any subset of them
into `gauss_math`'s namespace and then runs the generator unmodified, so the only
difference from a stock run is the functions named.

    # default: wasserstein only (what B14 is about)
    uv run --project tribble-fis python \
        reproduce/experiments/run_with_reference_stats.py \
        reproduce/tables/table_a1_feature_scoring.py

    # both, for D8's residue
    REPRO_RESTORE=wasserstein,kmeans uv run --project tribble-fis python \
        reproduce/experiments/run_with_reference_stats.py \
        reproduce/tables/table_a1_feature_scoring.py

    REPRO_RESTORE=all      every function #95 replaced
    REPRO_RESTORE=none     a stock run, for a controlled A/B in one script

Nothing is written into a submodule and no file is patched on disk; the
substitution lives only in this process. Point `REPRO_OUTPUT_DIR` somewhere of
its own so a corrected run cannot be mistaken for a stock one.

Supersedes `run_with_wasserstein_fix.py`, which did the same for one function.
"""

from __future__ import annotations

import os
import runpy
import sys

import numpy as np

# name -> (builder returning the reference callable, gauss_math attribute)
_KNOWN = (
    "wasserstein",
    "kmeans",
    "norm_fit",
    "norm_pdf",
    "jensenshannon",
    "silhouette",
)


def _references():
    """Build the reference implementations. Imported lazily: scipy and sklearn
    are dev-only for tribble-fis, and this is a diagnostic, not a dependency."""
    from scipy import stats as sp_stats
    from scipy.spatial.distance import jensenshannon as sp_js
    from scipy.stats import wasserstein_distance as sp_wass
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score as sk_sil

    def _sk_kmeans_1d(data, k, random_state):
        # sklearn seeds with k-means++; stats_numba.kmeans_1d takes a single
        # uniform-random start with no restarts, which is the difference that
        # moves the mixture fit.
        if k <= 1:
            return np.zeros(len(data), dtype=int)
        return KMeans(n_clusters=k, random_state=random_state).fit_predict(
            np.asarray(data).reshape(-1, 1)
        )

    return {
        "wasserstein": ("wasserstein_distance", sp_wass),
        "kmeans": ("_kmeans_labels_1d", _sk_kmeans_1d),
        "norm_fit": ("norm_fit", lambda a: sp_stats.norm.fit(a)),
        "norm_pdf": ("norm_pdf", lambda x, mu, sd: sp_stats.norm.pdf(x, mu, sd)),
        "jensenshannon": ("jensenshannon_distance", sp_js),
        "silhouette": ("silhouette_score", sk_sil),
    }


def _selection() -> list[str]:
    raw = os.environ.get("REPRO_RESTORE", "wasserstein").strip().lower()
    if raw in ("", "none"):
        return []
    if raw == "all":
        return list(_KNOWN)
    picked = [p.strip() for p in raw.split(",") if p.strip()]
    unknown = [p for p in picked if p not in _KNOWN]
    if unknown:
        raise SystemExit(
            f"REPRO_RESTORE: unknown {unknown}; known names are {list(_KNOWN)}"
        )
    return picked


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = os.path.abspath(sys.argv[1])
    if not os.path.exists(target):
        print(f"no such generator: {target}", file=sys.stderr)
        return 2

    picked = _selection()
    from tribblefis import gauss_math

    if picked:
        refs = _references()
        for name in picked:
            attr, fn = refs[name]
            setattr(gauss_math, attr, fn)
        print(f"[reference-stats] restored: {', '.join(picked)}")
    else:
        print("[reference-stats] restored: nothing (stock run)")

    # Prove the substitution bit rather than assume it. A runner that silently
    # failed to patch would produce a stock run under a corrected label, which is
    # worse than not running it at all.
    if "wasserstein" in picked:
        a = gauss_math.wasserstein_distance([0.0, 1.0], [0.0, 2.0])
        b = gauss_math.wasserstein_distance([0.0, 1000.0], [0.0, 2000.0])
        if not (abs(a - 0.5) < 1e-9 and abs(b - 500.0) < 1e-6):
            print(
                f"[reference-stats] ABORT: wasserstein patch did not take "
                f"(got {a} and {b}, expected 0.5 and 500.0)",
                file=sys.stderr,
            )
            return 1
        print(
            "[reference-stats] verified wasserstein: 0.5 at unit scale, 500.0 at x1000"
        )
    if "kmeans" in picked:
        # Two well-separated groups: any sane 1-D k-means splits them, but the
        # check is that the substituted callable answers at all with the
        # (data, k, random_state) signature gauss_math calls it with.
        labels = gauss_math._kmeans_labels_1d(np.array([0.0, 0.1, 5.0, 5.1]), 2, 0)
        if len(set(labels.tolist())) != 2:
            print("[reference-stats] ABORT: kmeans patch did not take", file=sys.stderr)
            return 1
        print("[reference-stats] verified kmeans: 2 clusters recovered")

    sys.argv = [target] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(target))
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
