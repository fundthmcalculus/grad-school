#!/usr/bin/env python3
"""What each optimizer finds from the hot start, when it finds it, and whether
the difference between them is real.

Three panels, in the order the questions should be asked.

**(a) The objective, against evaluations.** What every convergence plot shows.
Median across seeds with an inter-quartile band; the dashed line is the
heuristic start. Read on its own it says the local methods win.

**(b) Held-out R², against the evaluation budget.** The same runs, scored on the
test split at each budget checkpoint, which is the quantity the chapters
actually quote. It does not agree with (a), and that disagreement is the
study's main finding rather than an artifact: an arm can drive the
cross-validated objective down and take nothing home.

**(c) The paired difference, with its spread.** Each arm's per-seed
`R²(arm) − R²(start)`, mean and standard deviation. Pairing matters because all
arms face the identical problem at a given seed, so the start's own seed-to-seed
spread is common to them and swamps the between-arm gaps in an unpaired
comparison. The bars overlap, and the panel is drawn to make that unmissable —
the honest reading of this study is that *something* is reliably available and
*which optimizer takes it* is not yet resolved.

Panel (a) is drawn from the per-improvement trace, (b) and (c) from the budget
and per-seed CSVs. All three come out of the one run, so they cannot disagree
about what happened.
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
COLOUR = {a: F.SERIES[i % len(F.SERIES)] for i, a in enumerate(ARM_ORDER)}


def _traces():
    rows, label = H.table("table_opt_hotstart_traces")
    by_arm = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_arm[r["arm"]][int(r["seed"])].append(
            (int(r["eval"]), float(r["best_cv_mse"])))
    for arm in by_arm:
        for seed in by_arm[arm]:
            by_arm[arm][seed].sort()
    return by_arm, label


def _budget_curve():
    rows, _ = H.table("table_opt_hotstart_budget")
    by_arm = defaultdict(lambda: defaultdict(dict))
    starts = {}
    for r in rows:
        by_arm[r["arm"]][int(r["budget"])][int(r["seed"])] = float(r["r2"])
        starts[int(r["seed"])] = float(r["r2_0"])
    return by_arm, starts


def _step(trace, grid):
    evals = np.array([e for e, _ in trace])
    vals = np.array([v for _, v in trace])
    idx = np.clip(np.searchsorted(evals, grid, side="right") - 1, 0, len(vals) - 1)
    return vals[idx]


def build():
    by_arm, label = _traces()
    present = [a for a in ARM_ORDER if a in by_arm]
    if not present:
        raise RuntimeError("no traces found; run reproduce/optimizers/run_study.py")
    budget_curve, starts = _budget_curve()

    budget = max(e for arm in by_arm.values() for tr in arm.values() for e, _ in tr)
    grid = np.unique(np.round(np.geomspace(1, budget, 220)).astype(int))
    start_obj = np.median([tr[0][1] for arm in by_arm.values()
                           for tr in arm.values()])

    fig, (ax, rx, bx) = F.grid_figure(1, 3, width=F.W_WIDE + 1.4, height=3.7,
                                      gridspec_kw={"width_ratios": [1.25, 1.25, 1]})

    # -- (a) objective against evaluations -----------------------------------
    ax.axhline(start_obj, lw=1.2, ls=(0, (4, 2)), color=F.FAINT, zorder=2)
    ax.text(1.15, start_obj, " heuristic start", va="bottom", ha="left",
            fontsize=F.FS_SMALL, color=F.MUTED)
    for arm in present:
        curves = np.vstack([_step(tr, grid) for tr in by_arm[arm].values()])
        ax.plot(grid, np.median(curves, axis=0), lw=1.6, color=COLOUR[arm],
                label=arm, zorder=5)
        ax.fill_between(grid, np.percentile(curves, 25, axis=0),
                        np.percentile(curves, 75, axis=0),
                        color=F.tint(COLOUR[arm], 0.90), lw=0, zorder=2)
    ax.set_xscale("log")
    F.style_axes(ax, title="(a)  the objective it was given",
                 xlabel="objective evaluations (log)",
                 ylabel="k-fold held-out MSE")
    ax.set_xlim(1, budget * 1.1)
    F.legend(ax, loc="lower left", ncol=2)

    # -- (b) held-out R^2 against budget -------------------------------------
    seeds = sorted(starts)
    start_r2 = np.median([starts[s] for s in seeds])
    rx.axhline(start_r2, lw=1.2, ls=(0, (4, 2)), color=F.FAINT, zorder=2)
    for arm in present:
        budgets = sorted(budget_curve[arm])
        med = [np.median([budget_curve[arm][b][s] for s in budget_curve[arm][b]])
               for b in budgets]
        lo = [np.percentile(list(budget_curve[arm][b].values()), 25) for b in budgets]
        hi = [np.percentile(list(budget_curve[arm][b].values()), 75) for b in budgets]
        rx.plot(budgets, med, lw=1.6, marker="o", ms=3.5, color=COLOUR[arm], zorder=5)
        rx.fill_between(budgets, lo, hi, color=F.tint(COLOUR[arm], 0.90), lw=0,
                        zorder=2)
    rx.set_xscale("log")
    F.style_axes(rx, title="(b)  the quantity the chapters quote",
                 xlabel="evaluation budget (log)", ylabel="held-out $R^2$")
    rx.text(min(sorted(budget_curve[present[0]])), start_r2, " heuristic start",
            va="bottom", ha="left", fontsize=F.FS_SMALL, color=F.MUTED)

    # -- (c) paired delta, with spread ---------------------------------------
    rows, _ = H.table("table_opt_hotstart_seeds")
    per_arm = defaultdict(list)
    for r in rows:
        per_arm[r["arm"]].append(float(r["r2"]) - float(r["r2_0"]))
    stats = {a: (float(np.mean(per_arm[a])), float(np.std(per_arm[a])))
             for a in present if per_arm[a]}
    order = sorted(stats, key=lambda a: -stats[a][0])
    y = np.arange(len(order))
    means = [stats[a][0] for a in order]
    sds = [stats[a][1] for a in order]
    bx.barh(y, means, color=[COLOUR[a] for a in order], height=0.6, zorder=3)
    bx.errorbar(means, y, xerr=sds, fmt="none", ecolor=F.INK_2, elinewidth=1.0,
                capsize=2.5, zorder=5)
    bx.axvline(0, lw=0.9, color=F.AXIS, zorder=4)
    bx.set_yticks(y)
    bx.set_yticklabels(order, fontsize=F.FS_SMALL)
    bx.invert_yaxis()
    F.style_axes(bx, title="(c)  paired gain, ±1 s.d.",
                 xlabel="$R^2$(arm) $-$ $R^2$(start), per seed", grid_axis="x")

    fig.text(0.5, -0.02,
             "Every arm optimizes the same objective from the same hot start "
             "inside the same box, cut off at the same evaluation count by a "
             "wrapper that raises. Single-threaded,\nso an optimizer that "
             "parallelises well gets no credit. Bands are inter-quartile across "
             "seeds. **Panel (c) is the one to read for ordering, and its error "
             "bars overlap:**\nthe gain over the heuristic start is real and "
             "consistent, but no arm is separated from another by this evidence. "
             f"{H.provenance_note(label)}",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
