"""Table 3.1 -- pVAT reorder time vs. a classical O(N^3) VAT reference.

The chapter's claim is that replacing the textbook linear-scan argmin with a
priority-queue reorder drops VAT from O(N^3) to O(N^2 log N). This script makes
that concrete: it times the repo's exact pVAT against a deliberately-naive
O(N^3) reference (implemented here, self-contained), across a grid of N, on
random point sets, averaged over `common.SEEDS`. The naive reference is capped
at a modest N because it is genuinely cubic; larger N report pVAT only.

Run (from repo root):  uv run --project tribble-cluster python reproduce/tables/table_3_1_pvat_scaling.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.spatial.distance import cdist

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
import common as C  # noqa: E402

N_GRID = [int(n) for n in os.environ.get("REPRO_N_GRID", "256,512,1024,2048,4096").split(",")]
NAIVE_CAP = int(os.environ.get("REPRO_NAIVE_CAP", "1024"))   # skip cubic ref above this


def classical_vat_order(D):
    """Textbook VAT: Prim-like growth with a per-step linear scan and NO
    maintained min-distance array -> genuinely O(N^3). Reference only."""
    n = len(D)
    i, _ = np.unravel_index(int(np.argmax(D)), D.shape)
    order = [int(i)]
    chosen = np.zeros(n, dtype=bool)
    chosen[i] = True
    for _ in range(n - 1):
        best_b, best_d = -1, np.inf
        for b in range(n):
            if chosen[b]:
                continue
            d = np.inf                      # min dist from b to the current tree
            for a in order:                 # <-- the O(N) rescan that makes it cubic
                if D[a, b] < d:
                    d = D[a, b]
            if d < best_d:
                best_d, best_b = d, b
        order.append(best_b)
        chosen[best_b] = True
    return order


def _resolve_pvat():
    """Find the repo's pVAT reorder entry point; return a callable D -> order/RDI."""
    candidates = [
        ("tribbleclustering", "compute_vat"),
        ("tribbleclustering.pvat", "vat_prim_mst"),
        ("tribbleclustering.pvat", "compute_vat"),
        ("tribbleclustering", "vat"),
    ]
    for mod, name in candidates:
        try:
            m = __import__(mod, fromlist=[name])
            fn = getattr(m, name, None)
            if callable(fn):
                print(f"  using pVAT: {mod}.{name}")
                return fn
        except Exception:  # noqa: BLE001
            continue
    print("  [pVAT] could not resolve the repo entry point; pVAT column -> N/A")
    print("         (edit _resolve_pvat() to point at the actual tribbleclustering API)")
    return None


def main():
    print("Table 3.1 -- pVAT vs. classical VAT reorder time")
    pvat = _resolve_pvat()
    rows = []
    for n in N_GRID:
        classical_t, pvat_t = [], []
        for seed in C.SEEDS:
            rng = np.random.RandomState(seed)
            pts = rng.rand(n, 2)
            D = cdist(pts, pts).astype(np.float64)
            if n <= NAIVE_CAP:
                with C.timed() as t:
                    classical_vat_order(D)
                classical_t.append(t.seconds)
            if pvat is not None:
                try:
                    with C.timed() as t:
                        pvat(D)
                    pvat_t.append(t.seconds)
                except Exception as exc:  # noqa: BLE001
                    print(f"    [pVAT n={n}] failed ({exc.__class__.__name__})")
        c_cell = (C.cell(classical_t, fmt="{:.3f}") + " s") if classical_t else "infeasible (>cap)"
        p_cell = (C.cell(pvat_t, fmt="{:.3f}") + " s") if pvat_t else C.NA
        cm, _ = C.agg(classical_t)
        pm, _ = C.agg(pvat_t)
        speed = f"{cm / pm:.0f}x" if (cm and pm) else C.NA
        rows.append([f"{n:,}", c_cell, p_cell, speed])

    C.emit("table_3_1", "Table 3.1 -- Reorder time: classical VAT vs. pVAT",
           ["N (points)", "classical VAT", "pVAT", "speedup"], rows,
           note=f"Random 2-D point sets; classical reference capped at N<={NAIVE_CAP} "
                "(it is genuinely cubic). Re-run under the G4 protocol on stable "
                "hardware for the citable version.")


if __name__ == "__main__":
    main()
