"""Table 3.1 -- the three-arm reorder study behind Chapter 3 §3.3.1.

Times the three complexity regimes against each other on identical inputs:

  arm 0  classical    O(N^3)      textbook re-scan (what the shipped VAT
                                  implementations in the literature do)
  arm 1  stage one    O(N^2 logN) priority queue with lazy deletion
                                  (`tribbleclustering.pvat.vat_prim_mst`)
  arm 2  stage two    O(N^2)      compact active set, fused relax+select
                                  (`tribbleclustering.pcvat.vat_prim_mst_c_64`)

Fairness rules, because this comparison is the whole point:
  * All three arms are COMPILED. The classical reference is numba-jitted rather
    than left in interpreted Python, so the result is about algorithms and not
    about C-versus-Python.
  * Every JIT is warmed on a tiny input before any timing.
  * Best-of-k per (n, seed) to suppress scheduler noise; reported as
    mean +/- std across seeds.
  * Every arm's ordering is checked against arm 2 for bit-identity. A timing
    number from an arm that disagrees is meaningless and is reported as such.

Run (from repo root):
    uv run --project tribble-cluster python reproduce/tables/table_3_1_reorder_three_arm.py

Knobs:
    REPRO_N_GRID="500,1000,2000,4000"   sizes to sweep
    REPRO_CUBIC_CAP="1500"             largest n at which the O(N^3) arm runs
    REPRO_SEEDS="0,1,2"                seeds
    REPRO_REPEATS="3"                  best-of-k per measurement
"""

from __future__ import annotations

import os
import sys

import numpy as np

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
import common as C  # noqa: E402

N_GRID = [int(x) for x in os.environ.get("REPRO_N_GRID", "500,1000,2000,4000").split(",")]
CUBIC_CAP = int(os.environ.get("REPRO_CUBIC_CAP", "1500"))
REPEATS = int(os.environ.get("REPRO_REPEATS", "3"))


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def distance_matrix(n, seed, dim=3):
    """Euclidean distance matrix for n uniform points (no scipy dependency)."""
    rng = np.random.RandomState(seed)
    P = rng.rand(n, dim)
    G = P @ P.T
    d = np.diag(G)
    return np.sqrt(np.maximum(d[:, None] - 2.0 * G + d[None, :], 0.0)).astype(np.float64)


# --------------------------------------------------------------------------- #
# arm 0: the classical cubic reorder, compiled so the comparison is fair
# --------------------------------------------------------------------------- #
def _make_classical():
    try:
        from numba import njit
    except ImportError:
        return None

    @njit(cache=True, nogil=True)
    def classical_vat_order(D):  # pragma: no cover - timing kernel
        n = D.shape[0]
        # seed at one endpoint of the most-dissimilar pair
        best = -1.0
        si = 0
        for i in range(n):
            for j in range(n):
                if D[i, j] > best:
                    best = D[i, j]
                    si = i
        order = np.empty(n, dtype=np.int64)
        chosen = np.zeros(n, dtype=np.bool_)
        order[0] = si
        chosen[si] = True
        for k in range(1, n):
            bd = np.inf
            bv = -1
            for v in range(n):          # every unchosen candidate ...
                if chosen[v]:
                    continue
                dv = np.inf
                for t in range(k):      # ... rescanned against the whole tree
                    a = order[t]
                    if D[a, v] < dv:
                        dv = D[a, v]
                if dv < bd:
                    bd = dv
                    bv = v
            order[k] = bv
            chosen[bv] = True
        return order

    return classical_vat_order


# --------------------------------------------------------------------------- #
# arms 1 and 2: the shipped implementations
# --------------------------------------------------------------------------- #
def _load_arms():
    heap = dense = None
    try:
        from tribbleclustering.pvat import vat_prim_mst
        heap = lambda D: np.asarray(vat_prim_mst(D.copy())[0], dtype=np.int64)  # noqa: E731
    except Exception as exc:  # noqa: BLE001
        print(f"  [arm 1] unavailable ({exc.__class__.__name__})")
    try:
        from tribbleclustering.pcvat import vat_prim_mst_c_64
        dense = lambda D: np.asarray(                                            # noqa: E731
            vat_prim_mst_c_64(np.ascontiguousarray(D))[0], dtype=np.int64)
    except Exception as exc:  # noqa: BLE001
        print(f"  [arm 2] unavailable ({exc.__class__.__name__}) -- "
              f"build the Cython extension: pip install -e '.[dev]'")
    return heap, dense


def time_best_of(fn, D, repeats=REPEATS):
    """Smallest wall-clock over `repeats` runs, plus the ordering produced."""
    best = float("inf")
    order = None
    for _ in range(repeats):
        with C.timed() as t:
            order = fn(D)
        best = min(best, t.seconds)
    return best, order


def main():
    print("Table 3.1 -- three-arm reorder timing (classical / stage one / stage two)")
    classical = _make_classical()
    heap, dense = _load_arms()
    if dense is None:
        print("  cannot proceed without the reference arm (stage two); aborting")
        return

    # Warm every JIT before timing anything.
    warm = distance_matrix(64, 999)
    for fn in (classical, heap, dense):
        if fn is not None:
            fn(warm)
    print("  JIT warm-up complete\n")

    rows = []
    for n in N_GRID:
        t_cls, t_heap, t_dense = [], [], []
        agree_cls, agree_heap = [], []
        for seed in C.SEEDS:
            D = distance_matrix(n, seed)
            td, ref = time_best_of(dense, D)
            t_dense.append(td)

            if heap is not None:
                th, oh = time_best_of(heap, D)
                t_heap.append(th)
                agree_heap.append(bool(np.array_equal(oh, ref)))

            if classical is not None and n <= CUBIC_CAP:
                tc, oc = time_best_of(classical, D, repeats=1)  # cubic: one pass
                t_cls.append(tc)
                agree_cls.append(bool(np.array_equal(oc, ref)))

        d_mean, _ = C.agg(t_dense)
        h_mean, _ = C.agg(t_heap)
        c_mean, _ = C.agg(t_cls)

        def ratio(x):
            return f"{x / d_mean:.1f}×" if (x and d_mean) else C.NA

        ok = "yes" if (all(agree_heap) if agree_heap else True) and \
                     (all(agree_cls) if agree_cls else True) else "**NO**"
        rows.append([
            f"{n:,}",
            (C.cell(t_cls, fmt="{:.4f}") + " s") if t_cls else "not run (> cap)",
            (C.cell(t_heap, fmt="{:.4f}") + " s") if t_heap else C.NA,
            C.cell(t_dense, fmt="{:.4f}") + " s",
            ratio(c_mean), ratio(h_mean), ok,
        ])
        print(f"  n={n:<6} done")

    header = ["N", "classical O(N³)", "stage 1 O(N²logN)", "stage 2 O(N²)",
              "cls/s2", "s1/s2", "orders identical"]
    C.emit("table_3_1_three_arm",
           "Table 3.1 — Reorder time across the three complexity regimes",
           header, rows,
           note=("All three arms compiled (the classical reference is numba-jitted so the "
                 "comparison is algorithmic, not C-vs-Python); JITs warmed before timing; "
                 "best-of-%d per seed. Every arm's ordering is verified bit-identical to "
                 "stage two — a timing from a disagreeing arm would be meaningless. The "
                 "cubic arm is capped at N=%d because it is genuinely O(N³)."
                 % (REPEATS, CUBIC_CAP)))


if __name__ == "__main__":
    main()
