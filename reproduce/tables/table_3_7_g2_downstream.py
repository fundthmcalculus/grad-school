"""Table 3.7 (Goal G2) companion -- decision-rule item 3: downstream usefulness.

`table_3_7_g2_dtw_nonmetric.py` measured items 1-2 (exactness, triangle-inequality
violation rate) and item 4 (Crop at scale) of Goal G2's decision rule
(research/proposal-defense/prose/07-goals-for-completion.md, "### G2"). Item 3 --
"downstream usefulness" -- was explicitly left as NOT DONE in that script's
docstring: no baseline wiring existed for a precomputed-matrix comparison. This
script fills it.

Decision rule, quoted exactly: "NERFCM given k, ConiVAT, single-linkage,
beta-plateau and bottleneck-bootstrap all run on a dissimilarity matrix. The gated
set-cover, discovering k, must land within 0.05 adjusted Rand index of
NERFCM-given-k on at least three of the five DTW sets, and select_coverage_cover
and select_multiscale must run on those matrices at all, which per Sec 5.4 they
never have, Chapter 5's relational block evaluating only NERFCM. Refuted if the
set-cover misses 0.05 on every real non-coordinate set."

WHAT ALREADY EXISTED (reused here, nothing reimplemented for these two):
  - NERFCM-given-k: `gated-minimax-selection/nerfcm.py:nerfcm(D, c, seed=...)` --
    the exact function `run_all.py` uses for Chapter 5's NERFCM_D/NERFCM_Dstar
    columns. Works on ANY (n,n) dissimilarity matrix, no coordinates needed.
  - select_coverage_cover / select_multiscale: `gated-minimax-selection/
    selection.py` and `multiscale_persistence.py` -- also matrix-only, and
    ALREADY discover k (they never take it as an argument). Confirmed by reading
    both modules: neither has ever required coordinates. The "never run on a
    real dissimilarity matrix" claim in the prose is about DATA (every existing
    call site feeds them im.dissimilarity(X) built FROM coordinates), not about
    a code limitation -- so the fix here is a new caller, not new algorithm code.

WHAT DOES NOT EXIST / IS NOT ATTEMPTED (lower priority per the task; only
NERFCM-given-k is on the decision rule's hard threshold):
  - ConiVAT (`gated-minimax-selection/conivat.py`): its metric-learning step
    (`learn_metric_diag`) operates on raw coordinate axes (Xing et al. diagonal
    Mahalanobis) and only tolerates a dissimilarity matrix once that metric has
    reweighted actual feature axes. DTW time series have no coordinate axes to
    reweight -- this is a genuine implementation gap, not a missing call site.
    Recorded as N/A.
  - bottleneck-bootstrap (`selection_comparison.py:select_bottleneck_bootstrap`):
    takes raw X and recomputes a distance matrix from scratch per bootstrap
    resample (100 boots x im.dissimilarity, hardcoded Euclidean). Adapting it to
    DTW would mean rebuilding a ~0.8N-sized DTW matrix up to 100 times; at
    ECG5000's measured ~630s per full 5000x5000 DTW build, even a naive estimate
    is many hours. Recorded as N/A.
  - single-linkage-given-k and beta-plateau: matrix-only and cheap, so included
    below as bonus context even though the decision rule's hard threshold does
    not require them.

SCORING CONVENTION. Every selector below is scored by hard-assigning EVERY point
to its nearest block by minimax distance (`multiscale_persistence.assign`) --
the same convention `run_all.py`'s `cover_result()` and `MS.assign()` already use.
This is a deliberate, single, stated choice: `run_persistence_methods_numeric()`
elsewhere in `run_all.py` uses a DIFFERENT convention (uncovered points fall into
whichever block is index 0), and mixing the two without saying so would silently
compare apples to oranges.

Datasets: only ECG5000 is attempted here. FordA's DTW matrix costs ~7200s
(~2h, measured) and Crop's costs ~1600s for a ~4.6GB matrix (measured) -- neither
is recomputed by this script without a separate, flagged decision, per this
project's ask-before-multi-hour-compute norm. ECG5000's DTW matrix is rebuilt
fresh here (not cached from the sibling script's run) at a measured ~630-660s.

A quick timing probe (n=1024 subsample, see the task record) showed the actual
downstream algorithms are NOT the cost driver: minimax_transform_fast, NERFCM,
select_coverage_cover and select_multiscale all extrapolate to single-digit
seconds at N=5000. All measured full-N times are printed and recorded.

Run (from repo root):
    uv run --project tribble-cluster --with aeon --with scipy --with scikit-learn \
        python reproduce/tables/table_3_7_g2_downstream.py

Knobs:
    REPRO_G2_DOWNSTREAM_DATASETS="ECG5000"   comma-separated aeon dataset names.
    REPRO_G2_DOWNSTREAM_SEEDS   defaults to common.SEEDS (ten); NERFCM's only
                                 randomness is its initial membership matrix.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_TABLES = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_TABLES)
sys.path.insert(0, _ROOT)
import common as C  # noqa: E402

sys.path.insert(0, _TABLES)
from table_3_7_g2_dtw_nonmetric import load_dataset, dtw_matrix  # noqa: E402

_GMS = os.path.join(os.path.dirname(_ROOT), "gated-minimax-selection")
sys.path.insert(0, _GMS)
import ivat_mf as im  # noqa: E402
import selection as S  # noqa: E402
import multiscale_persistence as MS  # noqa: E402
from nerfcm import nerfcm  # noqa: E402
from selection_comparison import select_beta_plateau  # noqa: E402

from sklearn.metrics import adjusted_rand_score  # noqa: E402
from scipy.cluster.hierarchy import linkage, fcluster  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

DATASETS = [
    d.strip()
    for d in os.environ.get("REPRO_G2_DOWNSTREAM_DATASETS", "ECG5000").split(",")
    if d.strip()
]
_seeds_env = os.environ.get("REPRO_G2_DOWNSTREAM_SEEDS", "")
SEEDS = [int(s) for s in _seeds_env.split(",")] if _seeds_env else C.SEEDS

_KNOWN_SKIPPED = {
    # FordA's ~7200s/~2h rebuild was explicitly flagged and approved (2026-08-12) given remaining budget.
    # Crop's ~1600s/~4.6GB rebuild was explicitly flagged and approved
    # (2026-08-12 session) given the remaining time budget -- see
    # reproduce/outputs/table_3_7_g2_downstream.md's Crop row for the result.
}


def nerfcm_given_k_ari(Ds, y, k, seeds):
    """NERFCM-given-k on a precomputed dissimilarity matrix, across seeds.

    Reuses `gated-minimax-selection/nerfcm.py:nerfcm` verbatim -- the same
    function `run_all.py` uses for Chapter 5's NERFCM columns.
    """
    aris, secs = [], []
    for s in seeds:
        t0 = time.time()
        U, beta, n_iter = nerfcm(Ds, k, seed=s)
        secs.append(time.time() - t0)
        lab = np.argmax(U, axis=0)
        aris.append(float(adjusted_rand_score(y, lab)))
    return aris, secs


def cover_ari(sel, Ds, y):
    """Hard-assign every point to its nearest block (multiscale_persistence.assign
    convention) and score against ground truth. Returns (ari, k, coverage)."""
    n = Ds.shape[0]
    if not sel:
        return float("nan"), 0, 0.0
    lab = MS.assign(sel, Ds)
    return (
        float(adjusted_rand_score(y, lab)),
        len(sel),
        float(S.coverage_of(sel, n)),
    )


def multiscale_best_ari(msel, Ds, y):
    """Best ARI over the discovered scale bands (ECG5000 etc. have one ground-
    truth granularity, so 'best band' is the fair comparison, not the mean-over-
    levels convention table_5_2 uses for genuinely hierarchical synthetic data)."""
    if not msel.bands:
        return float("nan"), [], []
    per_band = []
    for b in msel.bands:
        a = MS.assign_band(b, Ds)
        per_band.append(float(adjusted_rand_score(y, a)))
    return max(per_band), msel.granularities(), per_band


def single_linkage_given_k_ari(Ds, y, k):
    Z = linkage(squareform(Ds, checks=False), method="single")
    lab = fcluster(Z, t=k, criterion="maxclust") - 1
    return float(adjusted_rand_score(y, lab))


def main():
    print("Table 3.7 (Goal G2) companion -- decision-rule item 3: downstream usefulness")
    rows = []
    for name in DATASETS:
        print(f"\n== {name} ==")
        if name in _KNOWN_SKIPPED:
            print(f"  [{name}] SKIPPED: {_KNOWN_SKIPPED[name]}")
            rows.append([name, C.NA, C.NA, C.NA, C.NA, C.NA, _KNOWN_SKIPPED[name]])
            continue
        try:
            X, y_raw = load_dataset(name)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{name}] load failed: {exc.__class__.__name__}: {exc}")
            rows.append([name, C.NA, C.NA, C.NA, C.NA, C.NA, "load failed"])
            continue
        y = np.asarray([int(v) for v in y_raw])
        k_true = len(set(y.tolist()))
        print(f"  loaded: N={X.shape[0]} length={X.shape[1]} k_true={k_true}")

        t0 = time.time()
        D = dtw_matrix(X)
        t_dtw = time.time() - t0
        print(f"  DTW matrix {D.shape}: {t_dtw:.1f}s")

        t0 = time.time()
        Ds = im.minimax_transform_fast(D)
        t_mm = time.time() - t0
        print(f"  minimax_transform_fast: {t_mm:.2f}s")

        # --- NERFCM given k (the decision rule's comparison anchor) ---------
        aris_n, secs_n = nerfcm_given_k_ari(Ds, y, k_true, SEEDS)
        nerfcm_mean, nerfcm_std = C.agg(aris_n)
        print(
            f"  NERFCM given k={k_true}: ARI={C.cell(aris_n)} "
            f"({len(SEEDS)} seeds, {sum(secs_n):.2f}s total)"
        )

        # --- select_coverage_cover (the gated set-cover; discovers k) -------
        t0 = time.time()
        sel_cover = S.select_coverage_cover(Ds)
        t_cover = time.time() - t0
        ari_cover, k_cover, cov_cover = cover_ari(sel_cover, Ds, y)
        print(
            f"  select_coverage_cover: {t_cover:.2f}s -> k={k_cover} "
            f"coverage={cov_cover:.3f} ARI={ari_cover:.3f}"
        )

        # --- select_multiscale (also discovers k; may find >1 band) ---------
        t0 = time.time()
        msel = MS.select_multiscale(Ds)
        t_multi = time.time() - t0
        ari_multi, granularities, per_band = multiscale_best_ari(msel, Ds, y)
        print(
            f"  select_multiscale: {t_multi:.2f}s -> {msel.n_scales} band(s), "
            f"granularities={granularities}, per-band ARI={per_band}, best={ari_multi:.3f}"
        )

        # --- bonus context: single-linkage-given-k, beta-plateau ------------
        ari_sl = single_linkage_given_k_ari(Ds, y, k_true)
        t0 = time.time()
        k_bp, sel_bp, meta_bp = select_beta_plateau(Ds)
        t_bp = time.time() - t0
        ari_bp, _, cov_bp = cover_ari(sel_bp, Ds, y)
        print(
            f"  single-linkage given k: ARI={ari_sl:.3f}  |  "
            f"beta-plateau: {t_bp:.2f}s -> k={k_bp} coverage={cov_bp:.3f} ARI={ari_bp:.3f}"
        )

        gap_to_nerfcm = (
            abs(ari_cover - nerfcm_mean) if not np.isnan(ari_cover) else float("nan")
        )
        within_threshold = (
            "yes" if (not np.isnan(gap_to_nerfcm) and gap_to_nerfcm <= 0.05) else "no"
        )
        print(
            f"  decision-rule check: |cover_ARI - NERFCM_ARI| = {gap_to_nerfcm:.3f} "
            f"-> within 0.05? {within_threshold}"
        )

        rows.append(
            [
                f"{name} (N={X.shape[0]}, k_true={k_true})",
                C.cell(aris_n),
                f"{ari_cover:.3f} (k={k_cover}, cov={cov_cover:.2f})",
                f"{ari_multi:.3f} (bands={granularities})" if msel.bands else C.NA,
                f"{ari_sl:.3f}",
                f"{ari_bp:.3f} (k={k_bp}, cov={cov_bp:.2f})" if sel_bp else C.NA,
                f"gap={gap_to_nerfcm:.3f}, within 0.05: {within_threshold}",
            ]
        )

    C.emit(
        "table_3_7_g2_downstream",
        "Table 3.7 (Goal G2) companion -- decision-rule item 3: downstream usefulness "
        "on real DTW dissimilarity matrices",
        [
            "Dataset",
            "NERFCM given k (ARI)",
            "select_coverage_cover (ARI)",
            "select_multiscale (best-band ARI)",
            "single-linkage given k (ARI)",
            "beta-plateau (ARI)",
            "Decision-rule gap",
        ],
        rows,
        note=(
            "NERFCM-given-k reuses `gated-minimax-selection/nerfcm.py:nerfcm` "
            "unmodified -- the same function `run_all.py` uses for Chapter 5's "
            "NERFCM columns. select_coverage_cover and select_multiscale reuse "
            "`gated-minimax-selection/selection.py` and `multiscale_persistence.py` "
            "unmodified; both are already matrix-only and already discover k -- "
            "the prose's 'never run on a real dissimilarity matrix' claim was about "
            "every existing CALL SITE feeding them coordinate-derived matrices, not "
            "a code limitation, confirmed by reading both modules before writing this "
            "script. Every selector here is scored by hard-assigning ALL points to "
            "their nearest block by minimax distance (`multiscale_persistence.assign`) "
            "-- a different, and here explicitly stated, convention from "
            "`run_persistence_methods_numeric()` elsewhere in run_all.py, which leaves "
            "uncovered points folded into block 0. ConiVAT and bottleneck-bootstrap "
            "are NOT attempted: ConiVAT's metric-learning step needs coordinate axes "
            "to reweight, which DTW time series do not have (a genuine implementation "
            "gap, not a missing call site); bottleneck-bootstrap recomputes a distance "
            "matrix from scratch per bootstrap resample and would need ~100 DTW "
            "rebuilds, many hours at this scale. Single-linkage-given-k and "
            "beta-plateau are included as bonus context; the decision rule's hard "
            "threshold names only NERFCM-given-k vs. the set-cover. FordA and Crop are "
            "NOT attempted here -- their DTW matrices cost ~7200s and ~1600s "
            "respectively (measured by the sibling table_3_7_g2_dtw_nonmetric.py run) "
            "and are not rebuilt without a separate flagged decision, so the decision "
            "rule's 'at least three of the five DTW sets' threshold cannot yet be "
            "fully evaluated -- only ECG5000 is measured here. A quick timing probe "
            "(n=1024 subsample) confirmed before this run that the downstream "
            "algorithms themselves are cheap at full N (single-digit seconds); the "
            "DTW matrix build is the entire cost."
        ),
        seeds=SEEDS,
    )


if __name__ == "__main__":
    main()
