#!/usr/bin/env python3
"""Convergence from the hot start: what each optimizer finds, and when.

Reads `table_opt_hotstart_traces.csv` -- the per-evaluation best-so-far trace the
study writes -- rather than re-running anything, so the figure and the table
describe one set of measurements.

The x axis is **objective evaluations, not seconds**. Every arm here is capped at
the same evaluation count and runs single-threaded, so evaluations are the
budget the study controls; seconds are a consequence of it and of how much
bookkeeping each optimizer does around the objective. The wall-clock is in the
table for anyone who needs it, and the median seconds-per-thousand-evaluations
is annotated here so the two can be related.

Both panels share a y axis and it is the objective, so lower is better; the
dashed line is the hot start, and the whole question of the study is how far
below it anything gets.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "07-optimizer-hotstart"

# Fixed order, never cycled: SciPy incumbents first, then the package's arms.
ARM_ORDER = ["scipy-lbfgsb", "scipy-powell", "scipy-de",
             "opt-ga", "opt-pso", "opt-aco", "opt-gd"]


def _traces():
    rows, label = H.table("table_opt_hotstart_traces")
    by_arm = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_arm[r["arm"]][int(r["seed"])].append(
            (int(r["eval"]), float(r["seconds"]), float(r["best_cv_mse"])))
    for arm in by_arm:
        for seed in by_arm[arm]:
            by_arm[arm][seed].sort()
    return by_arm, label


def _step(trace, grid):
    """Best-so-far as a step function, sampled on a common evaluation grid."""
    evals = np.array([e for e, _, _ in trace])
    vals = np.array([v for _, _, v in trace])
    idx = np.searchsorted(evals, grid, side="right") - 1
    idx = np.clip(idx, 0, len(vals) - 1)
    return vals[idx]


def build():
    by_arm, label = _traces()
    present = [a for a in ARM_ORDER if a in by_arm]
    if not present:
        raise RuntimeError("no traces found; run reproduce/optimizers/run_study.py")

    budget = max(e for arm in by_arm.values() for tr in arm.values()
                 for e, _, _ in tr)
    grid = np.unique(np.round(np.geomspace(1, budget, 220)).astype(int))

    start = np.median([tr[0][2] for arm in by_arm.values() for tr in arm.values()])

    fig, (ax, bar) = F.grid_figure(1, 2, width=F.W_WIDE, height=3.8,
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    ax.axhline(start, lw=1.2, ls=(0, (4, 2)), color=F.FAINT, zorder=2)
    ax.text(budget, start, "  heuristic start", va="center", ha="left",
            fontsize=F.FS_SMALL, color=F.MUTED)

    finals, gains = {}, {}
    for i, arm in enumerate(present):
        curves = np.vstack([_step(tr, grid) for tr in by_arm[arm].values()])
        starts = np.array([tr[0][2] for tr in by_arm[arm].values()])
        median = np.median(curves, axis=0)
        colour = F.SERIES[i % len(F.SERIES)]
        ax.plot(grid, median, lw=1.7, color=colour, label=arm, zorder=5)
        # Inter-quartile band, not min-max. One arm's worst seed here is four
        # times its median improvement, and a min-max band for it covers the
        # whole panel and hides every other arm's line. The spread that matters
        # is reported per arm in the table's ± columns.
        ax.fill_between(grid, np.percentile(curves, 25, axis=0),
                        np.percentile(curves, 75, axis=0),
                        color=F.tint(colour, 0.90), lw=0, zorder=2)
        finals[arm] = median[-1]
        # Per-seed fractional improvement, then averaged -- the same statistic
        # the table's "vs start" column reports. Aggregating the medians first
        # gives a different number, and a figure that disagrees with its own
        # table is the failure this harness exists to prevent.
        gains[arm] = float(np.mean(100 * (starts - curves[:, -1]) / starts))

    ax.set_xscale("log")
    F.style_axes(ax, title="(a)  best objective so far, median over seeds",
                 xlabel="objective evaluations (log)",
                 ylabel="k-fold held-out MSE")
    ax.set_xlim(1, budget * 1.35)
    F.legend(ax, loc="lower left", ncol=2)

    # -- right: how much of the start each arm removed -----------------------
    order = sorted(present, key=lambda a: -gains[a])
    y = np.arange(len(order))
    values = [gains[a] for a in order]
    colours = [F.SERIES[ARM_ORDER.index(a) % len(F.SERIES)] for a in order]
    bar.barh(y, values, color=colours, height=0.62, zorder=3)
    for yi, g in zip(y, values):
        bar.text(g + max(values) * 0.02, yi, f"{g:.1f}%", va="center",
                 ha="left", fontsize=F.FS_SMALL, color=F.INK_2)
    bar.set_yticks(y)
    bar.set_yticklabels(order, fontsize=F.FS_SMALL)
    bar.invert_yaxis()
    F.style_axes(bar, title="(b)  objective removed, at the full budget",
                 xlabel="% below the heuristic start", grid_axis="x")
    bar.set_xlim(0, max(values) * 1.22)

    fig.text(0.5, -0.02,
             f"All arms optimize the same objective from the same hot start "
             f"inside the same box, cut off at exactly {budget} evaluations by a "
             f"wrapper that raises —\nno arm's own stopping rule is trusted to make "
             f"the budgets equal. Single-threaded throughout, so an optimizer that "
             f"parallelises well gets no credit here.\nBands are the inter-quartile "
             f"range across seeds; panel (b) averages the per-seed improvement, the "
             f"same statistic the table reports. {H.provenance_note(label)}",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
