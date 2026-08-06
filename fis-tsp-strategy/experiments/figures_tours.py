"""The actual tours, side by side, on sample test instances.

Every other figure here reduces a tour to one number. That is the right thing for comparing
solvers and the wrong thing for understanding what separates them, because a 1% length difference
is invisible in a scalar and obvious in a picture: it is a handful of long edges crossing the
instance, in specific places, for structural reasons.

Three arms are drawn per instance, chosen to show the two transitions that matter:

* **greedy-edge construction** — where the local search starts. Its failure mode is the point of
  the panel: greedy edge builds excellent short edges everywhere and then has to close the tour
  with a few enormous ones, because the cities it left stranded have to be reached somehow.
* **fixed-parameter LK** — the local optimum of the 2-opt-plus-Or-opt neighbourhood, which is
  where every result before the double bridge stopped.
* **iterated (double-bridge kicks)** — the same neighbourhood plus a move that can leave it.

Edges longer than a multiple of the instance's mean edge are drawn heavier and darker, since those
are what the remaining gap consists of and they are otherwise hard to pick out of a few thousand
line segments.

Instances are chosen for structural variety rather than size: a clustered one, a uniform one and a
grid-like one behave differently and fail differently.

Run:  python figures_tours.py [--instances fl1577 pr2392 pcb3038] [--kicks 25600]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

# ``experiments/`` sits one level below the modules it imports, so the project root goes
# on sys.path before any of them. ``paths`` also owns every output location, so an
# experiment writes into the same results/ tree as the reported pipeline.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402

paths.on_path()

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

import fis  # noqa: E402
from core import build_candidates, greedy_edge_tour  # noqa: E402
from kick import iterated_lk  # noqa: E402
from lk import lk_solve  # noqa: E402
from tsplib import load, reference_length, validate_tour  # noqa: E402


K = 32
LONG_EDGE = 3.0  # an edge this many times the mean is drawn as "long"


def _draw(ax, coords, tour, title):
    seg = np.stack([coords[tour], coords[np.roll(tour, -1)]], axis=1)
    d = np.linalg.norm(seg[:, 0] - seg[:, 1], axis=1)
    long = d > LONG_EDGE * d.mean()

    ax.add_collection(
        LineCollection(seg[~long], colors="tab:blue", linewidths=0.35, alpha=0.55)
    )
    if long.any():
        ax.add_collection(
            LineCollection(seg[long], colors="tab:red", linewidths=1.4, alpha=0.95)
        )
    ax.set_title(f"{title}\n{int(long.sum())} edges > {LONG_EDGE:g}x mean", fontsize=9)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    pad = 0.03 * (hi - lo).max()
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def arms_for(inst, kicks):
    """(label, tour) for the three arms, with gap and time folded into the label."""
    cand, cand_d = build_candidates(inst.coords, K, inst.ceil)
    none = np.empty(0, np.float64)

    t0 = time.perf_counter()
    start = greedy_edge_tour(inst.coords, cand, inst.ceil)
    t_start = time.perf_counter() - t0
    validate_tour(start, inst.n)

    t0 = time.perf_counter()
    lk_tour, lk_len, _ = lk_solve(inst.coords, cand, cand_d, inst.ceil, start, K, 6, 32, 3)
    t_lk = time.perf_counter() - t0
    validate_tour(lk_tour, inst.n)

    t0 = time.perf_counter()
    it_tour, it_len, _ = iterated_lk(
        inst.coords, cand, cand_d, inst.ceil, start, K, 6, 32, 3, kicks, 24, 12345, none,
        False, fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS,
    )
    t_it = time.perf_counter() - t0
    validate_tour(it_tour, inst.n)

    return [
        (f"greedy-edge start — {inst.gap(reference_length(start, inst)):.2f}%, "
         f"{t_start:.2f}s", start),
        (f"fixed LK k32/d6/b32 — {inst.gap(lk_len):.2f}%, {t_lk:.2f}s", lk_tour),
        (f"+ {kicks:,} double-bridge kicks — {inst.gap(it_len):.2f}%, {t_it:.2f}s", it_tour),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", nargs="*", default=["fl1577", "pr2392", "pcb3038"])
    ap.add_argument("--kicks", type=int, default=25600)
    ap.add_argument("--out", default=str(paths.FIGURES / "fis_tsp_tours.png"))
    args = ap.parse_args()

    # warm the JITs so the reported times are steady-state
    warm = load("berlin52")
    wc, wcd = build_candidates(warm.coords, K, warm.ceil)
    wg = greedy_edge_tour(warm.coords, wc, warm.ceil)
    lk_solve(warm.coords, wc, wcd, warm.ceil, wg, K, 6, 32, 3)
    iterated_lk(
        warm.coords, wc, wcd, warm.ceil, wg, K, 6, 32, 3, 4, 24, 1,
        np.empty(0, np.float64), False,
        fis.NO_CHAIN_TAB, fis.NO_CHAIN_ANT, fis.NO_CHAIN_CONS,
    )

    rows = [load(name) for name in args.instances]
    fig, axes = plt.subplots(
        len(rows), 3, figsize=(13.5, 4.6 * len(rows)), squeeze=False
    )
    for r, inst in enumerate(rows):
        print(f"  {inst.name} n={inst.n}", flush=True)
        for c, (label, tour) in enumerate(arms_for(inst, args.kicks)):
            _draw(axes[r][c], inst.coords, tour, label)
            print(f"    {label}")
        axes[r][0].set_ylabel(f"{inst.name}  n={inst.n}", fontsize=10)

    fig.suptitle(
        "Red edges are longer than 3x the instance mean — the remaining gap, made visible",
        fontsize=11,
    )
    fig.tight_layout()
    Path(args.out).parent.mkdir(exist_ok=True)
    fig.savefig(args.out, dpi=120)
    plt.close(fig)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
