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

import math
import os
import pathlib
import sys

import numpy as np

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
import common as C  # noqa: E402

N_GRID = [int(x) for x in os.environ.get("REPRO_N_GRID", "500,1000,2000,4000").split(",")]
CUBIC_CAP = int(os.environ.get("REPRO_CUBIC_CAP", "1500"))
# Separate, deliberately small grid for the complexity fit: every point must be
# cheap enough that the cubic arm runs there too.
FIT_N_GRID = [int(x) for x in
              os.environ.get("REPRO_FIT_N_GRID", "100,200,300,500,750,1000").split(",")]
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
    means = []   # (classical, stage1, stage2) means per N, for the normalized view
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
        means.append((c_mean, h_mean, d_mean))
        print(f"  n={n:<6} done")

    header = ["N", "classical O(N³) (s)", "stage 1 O(N²logN) (s)", "stage 2 O(N²) (s)",
              "cls/s2", "s1/s2", "orders identical"]

    # CSV keeps absolute seconds; Markdown normalizes each row against its worst
    # (slowest) arm. Where the cubic reference ran, it IS the worst, so the other
    # two read directly as "times faster than classical" -- which is the claim.
    # Past the cubic cap the worst becomes stage one, and the stage-two cell then
    # reads as the s1/s2 ratio, which is why those columns are not repeated here.
    md_header = ["N", "classical O(N³)", "stage 1 O(N²logN)", "stage 2 O(N²)",
                 "orders identical"]
    md_rows = []
    for r, (c_m, h_m, d_m) in zip(rows, means):
        cn, hn, dn = C.normalized_worst([c_m, h_m, d_m])
        md_rows.append([r[0],
                        cn if r[1] != "not run (> cap)" else "not run (> cap)",
                        hn, dn, r[6]])

    emit_complexity_fit(classical, heap, dense)

    C.emit("table_3_1_three_arm",
           "Table 3.1 — Reorder time across the three complexity regimes",
           header, rows, md_header=md_header, md_rows=md_rows,
           note=("**The Markdown columns are normalized against the worst arm in each row** "
                 "-- the slowest arm at that N is the 1.0x baseline and the others read as "
                 "'this many times faster than the worst'. That is the machine-independent "
                 "view: absolute seconds move with thermals, governor and host, while the "
                 "ratio between arms measured in the same pass does not. Absolute seconds, "
                 "with per-seed spreads, are in the companion CSV. "
                 "All three arms compiled (the classical reference is numba-jitted so the "
                 "comparison is algorithmic, not C-vs-Python); JITs warmed before timing; "
                 "best-of-%d per seed. Every arm's ordering is verified bit-identical to "
                 "stage two — a timing from a disagreeing arm would be meaningless. The "
                 "cubic arm is capped at N=%d because it is genuinely O(N³)."
                 % (REPEATS, CUBIC_CAP)))


# --------------------------------------------------------------------------- #
# growth against the reference complexity curves
# --------------------------------------------------------------------------- #
def _ref(ns, fn):
    """A theoretical curve, normalized to its own value at the smallest N."""
    base = fn(ns[0])
    return [fn(n) / base for n in ns]


def _fit_exponent(ns, ts):
    """Least-squares slope of log(t) against log(N) -- the measured exponent."""
    pts = [(math.log(n), math.log(t)) for n, t in zip(ns, ts) if t]
    if len(pts) < 2:
        return None, len(pts)
    mx = sum(x for x, _ in pts) / len(pts)
    my = sum(y for _, y in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    if not den:
        return None, len(pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den, len(pts)


REFS = [("N²", lambda n: n * n),
        ("N² log N", lambda n: n * n * math.log2(n)),
        ("N³", lambda n: n ** 3)]


def _plot(ns, series, basename):
    """Log-log growth plot: both axes normalized to their own smallest-N value.

    Normalizing BOTH axes is what makes this portable -- x is N/N0 and y is
    t/t0, so a pure O(N^k) arm is a straight line of slope k regardless of the
    machine, the language or the constant factor. The reference curves are
    drawn recessively in grey and labelled inline: they are annotations against
    which the measured arms are read, not series competing for identity.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Validated categorical slots 1-3 (palette.md); fixed order, never cycled.
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
    x = [n / ns[0] for n in ns]

    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=200)

    for name, fn in REFS:                       # references first, so they sit behind
        y = _ref(ns, fn)
        ax.plot(x, y, ls="--", lw=1.2, color="#9a9a92", zorder=1)
        ax.annotate(name, (x[-1], y[-1]), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=8, color="#6b6b63")

    for i, (label, ts) in enumerate(series):
        xs = [n / ns[0] for n, v in zip(ns, ts) if v]
        ys = [v / ts[0] for v in ts if v]
        ax.plot(xs, ys, marker="o", ms=4.5, lw=1.8, color=COLORS[i],
                label=label, zorder=3)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("N, normalized  ($N/N_0$)", fontsize=9)
    ax.set_ylabel("time, normalized  ($t/t_0$)", fontsize=9)
    ax.set_title("Reorder growth against reference complexity curves",
                 fontsize=10, pad=8)
    ax.grid(True, which="both", lw=0.4, color="#e2e2dc", zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c9c1")
    ax.tick_params(labelsize=8, colors="#3d3d39")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.set_xlim(right=x[-1] * 1.55)             # room for the inline ref labels
    fig.tight_layout()
    C.save_figure(fig, basename)          # PNG for the Markdown, EPS for LaTeX
    plt.close(fig)


def emit_complexity_fit(classical, heap, dense):
    """Table 3.2 + its figure -- measured growth vs N^2, N^2 log N and N^3.

    Deliberately swept on a SMALL grid, separate from the main three-arm table.
    Keeping the largest N modest is what lets the cubic reference run at every
    point, so all three arms are fitted from the same number of samples instead
    of the cubic one being fitted from the two that fit under its cap. The
    large-scale envelope is a different experiment (the OpenMP/GPU work), and
    mixing the two would mean fitting an exponent across a change of machinery.
    """
    ns = FIT_N_GRID
    print(f"\n  complexity-fit sweep on N={ns} (small by design: the cubic arm "
          f"must run at every point)")
    tc, th, td = [], [], []
    for n in ns:
        c_s, h_s, d_s = [], [], []
        for seed in C.SEEDS:
            D = distance_matrix(n, seed)
            d_s.append(time_best_of(dense, D)[0])
            if heap is not None:
                h_s.append(time_best_of(heap, D)[0])
            if classical is not None:
                c_s.append(time_best_of(classical, D, repeats=1)[0])
        tc.append(C.agg(c_s)[0]); th.append(C.agg(h_s)[0]); td.append(C.agg(d_s)[0])
        print(f"    n={n:<6} done")

    arms = [("classical", tc, "N³", 3.0),
            ("stage 1", th, "N² log N", None),
            ("stage 2", td, "N²", 2.0)]

    rows = []
    for i, n in enumerate(ns):
        row = [f"{n:,}", f"{n / ns[0]:.1f}×"]
        for _, ts, _, _ in arms:
            row.append(C.normalized(ts)[i] if any(ts) else C.NA)
        for _, fn in REFS:
            row.append(f"{_ref(ns, fn)[i]:.2f}×")
        rows.append(row)

    fit = ["**fitted exponent**", ""]
    for label, ts, _, _ in arms:
        slope, npts = _fit_exponent(ns, ts)
        fit.append(f"**{slope:.2f}** ({npts} pts)" if slope is not None else C.NA)
    fit += ["2.00", "~2.1", "3.00"]
    rows.append(fit)

    _plot(ns, [(a[0], a[1]) for a in arms if any(a[1])], "fig_03_complexity_fit")

    C.emit("table_3_1_complexity_fit",
           "Table 3.2 — Measured growth against the reference complexity curves",
           ["N", "N (normalized)", "classical", "stage 1", "stage 2",
            "N² (ref.)", "N² log N (ref.)", "N³ (ref.)"],
           rows,
           note=("**Both axes are normalized**: N as N/N0 and time as t/t0, each against "
                 "its own value at the smallest N. On those axes a pure O(N^k) arm is a "
                 "straight line of slope k, independent of machine, language and constant "
                 "factor, which is what makes the comparison portable. The last row is the "
                 "least-squares slope of log(time) against log(N) -- the measured exponent "
                 "-- beside the exponent each arm is supposed to have. The grid is "
                 "deliberately SMALL: keeping the largest N modest is what lets the cubic "
                 "reference run at every point, so all three arms are fitted from the same "
                 "samples rather than the cubic one being fitted from the two that fit "
                 "under its cap. The large-scale envelope is a separate experiment "
                 "(OpenMP/GPU); fitting an exponent across a change of machinery would "
                 "measure the machinery. Companion figure: "
                 "`outputs/figures/fig_03_complexity_fit.{png,eps}` -- PNG for the "
                 "Markdown, EPS for the LaTeX build."))


if __name__ == "__main__":
    main()
