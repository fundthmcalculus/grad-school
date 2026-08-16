#!/usr/bin/env python3
"""PhiUSIIL: what a search is worth on a classification problem, with spreads.

The classification counterpart to `fig_07`/`fig_10`. The regression study found
the construction's *placement* worth almost nothing — a random draw in the same
box tied it before any optimization. That does not carry over, and this figure is
mostly about how badly it does not.

Six panels: **objective on the top row, test error on the bottom**, one column
per starting point. Every line is the mean across seeds and every band is ±1 s.d.
across the same seeds, so the question "is this arm actually ahead of that one"
can be answered by looking rather than by taking a ratio of two point estimates.
**Each row shares one y scale across all three columns**, which costs some detail
in the narrow panels and buys the thing worth seeing: the entire vertical range of
the hot column fits inside the bottom decade of the cold one.

Arms that never converge are drawn, not dropped. On a cold start the two
gradient-based arms sit flat at their starting objective for the whole budget —
that is a result about the cross-entropy surface, and removing them because they
have no trend would delete it.

Two things to know before reading the y axes:

**The objective is a training loss.** It is the shipped classifier fitness,
`refine._make_classifier_fitness` — training cross-entropy plus a ridge shrink
toward each arm's own starting point. Imported rather than reimplemented, on the
same principle as the regression study. So the top row is what the optimizers are
actually minimizing and the bottom row is the only outcome to quote, and where
the two disagree that disagreement is the finding.

**The bottom row is error rate on a log axis, not accuracy on a linear one.**
Two reasons, and both are forced. PhiUSIIL is saturated: the construction makes
two or three errors in ten thousand, so on a linear accuracy axis every good model
sits on the same pixel while a cold start sits four decades away, and no single
scale shows both. And a ±1 s.d. band around an accuracy of 0.9998 crosses 1.0,
which is not a possible accuracy — plotting it would draw an impossible region.
On a log error axis the three columns share one scale and can actually be
compared. Errors are measured on a large held-out split for the same
resolution reason.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "11-phishing-optimizer"

ARM_ORDER = [
    "scipy-lbfgsb",
    "scipy-powell",
    "scipy-de",
    "opt-ga",
    "opt-pso",
    "opt-aco",
    "opt-gd",
]
COLOUR = {a: F.SERIES[i % len(F.SERIES)] for i, a in enumerate(ARM_ORDER)}
LABEL = {
    "scipy-lbfgsb": "L-BFGS-B",
    "scipy-powell": "Powell",
    "scipy-de": "DE (scipy)",
    "opt-ga": "GA",
    "opt-pso": "PSO",
    "opt-aco": "ACO",
    "opt-gd": "GD",
}
INIT_ORDER = ["hot", "cold", "classical-kmeans"]
INIT_TITLE = {
    "hot": "hot: from the construction",
    "cold": "cold: random in the box",
    "classical-kmeans": "k-means within each class",
}

#: Floor for a zero error rate on a log axis. An arm that makes no test errors
#: has no place on one, so the point is floored and the caption says so rather
#: than the line quietly stopping.
ERR_FLOOR = 1e-5

#: How far to blend a series colour toward the surface for its ±1 s.d. band.
#: EPS has no alpha channel, so bands are solid blends -- which means they
#: occlude, which means seven of them need to be very light and drawn widest-first.
BAND_TINT = 0.86


def _f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def _traces():
    rows, label = H.table("table_opt_phishing_traces")
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[(r["arm"], r["init"])][int(r["seed"])].append(
            (int(r["eval"]), float(r["best_obj"]))
        )
    return out, label


def _budget():
    rows, _ = H.table("table_opt_phishing_budget")
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        out[(r["arm"], r["init"])][int(r["budget"])][int(r["seed"])] = _f(r, "acc")
    return out


def _seed_rows():
    rows, _ = H.table("table_opt_phishing_seeds")
    return rows


def _filled(per_seed, grid):
    """Each seed's best-so-far step function, forward-filled onto `grid`.

    Forward fill rather than interpolation: between two recorded improvements the
    best-so-far *is* the earlier value, and interpolating would draw progress the
    optimizer had not made yet.
    """
    stacked = []
    for pts in per_seed.values():
        if not pts:
            continue
        pts = sorted(pts)
        e = np.array([p[0] for p in pts], dtype=float)
        v = np.array([p[1] for p in pts], dtype=float)
        idx = np.clip(np.searchsorted(e, grid, side="right") - 1, 0, len(v) - 1)
        stacked.append(v[idx])
    return np.vstack(stacked) if stacked else None


def _band(ax, x, mean, sd, colour, ls, label, floor=None):
    """Mean line plus a ±1 s.d. band, widest drawn first by the caller."""
    lo, hi = mean - sd, mean + sd
    if floor is not None:
        lo = np.maximum(lo, floor)  # a log axis has no room for <= 0
    ax.fill_between(x, lo, hi, color=F.tint(colour, BAND_TINT), lw=0, zorder=2)
    ax.plot(x, mean, lw=1.8, ls=ls, color=colour, label=label, zorder=5)


def build():
    traces, label = _traces()
    budget_acc = _budget()
    seed_rows = _seed_rows()
    if not traces:
        raise RuntimeError("no traces; run run_phishing_opt_study.py")

    inits = [i for i in INIT_ORDER if any(k[1] == i for k in traces)]
    arms = [a for a in ARM_ORDER if any((a, i) in traces for i in inits)]
    max_eval = (
        max(e for per in traces.values() for pts in per.values() for e, _v in pts)
        or 2000
    )
    budgets = sorted({b for per in budget_acc.values() for b in per})

    ref = [r for r in seed_rows if r["arm"] == "none" and r["init"] == "hot"]
    heur_obj = float(np.mean([_f(r, "heuristic_obj") for r in ref])) if ref else np.nan
    heur_acc = float(np.mean([_f(r, "heuristic_acc") for r in ref])) if ref else np.nan
    n_seeds = len({r["seed"] for r in seed_rows})

    fig, axes = F.grid_figure(2, len(inits), width=F.W_WIDE + 2.4, height=7.2)
    axes = np.atleast_2d(axes)

    grid = np.unique(np.concatenate([[1], np.logspace(0, np.log10(max_eval), 160)]))
    err_axes, obj_axes = [], []

    for col, init in enumerate(inits):
        ox, ax = axes[0][col], axes[1][col]

        # -- objective ------------------------------------------------------- #
        series = []
        for arm in arms:
            per_seed = traces.get((arm, init))
            if not per_seed:
                continue
            stack = _filled(per_seed, grid)
            if stack is None:
                continue
            series.append(
                (
                    float(np.mean(stack.std(axis=0))),
                    arm,
                    stack.mean(axis=0),
                    stack.std(axis=0),
                )
            )
        # Widest band first: these are solid blends, so a wide one drawn last
        # would bury every narrow one underneath it.
        for _w, arm, mean, sd in sorted(series, reverse=True):
            _band(ox, grid, mean, sd, COLOUR[arm], "solid", LABEL[arm], floor=1e-4)
        if np.isfinite(heur_obj):
            ox.axhline(heur_obj, lw=1.3, ls=(0, (2, 2)), color=F.INK_2, zorder=6)
        ox.set_xscale("log")
        ox.set_yscale("log")
        obj_axes.append(ox)
        F.style_axes(
            ox,
            title=f"objective — {INIT_TITLE[init]}",
            xlabel="objective evaluations (log)",
            ylabel="cross-entropy + shrink (log)" if col == 0 else None,
        )
        if col == 0:
            F.legend(ox, loc="lower left", ncol=2, handlelength=2.0)
            if np.isfinite(heur_obj):
                ox.text(
                    max_eval,
                    heur_obj,
                    "the construction ",
                    va="bottom",
                    ha="right",
                    fontsize=F.FS_SMALL,
                    color=F.INK_2,
                )

        # -- error rate ------------------------------------------------------ #
        series = []
        for arm in arms:
            per_b = budget_acc.get((arm, init))
            if not per_b:
                continue
            # Per seed, then aggregate: 1 - mean(acc) and mean(1 - acc) agree,
            # but the s.d. of the error rate must come from per-seed errors.
            errs = [
                [1.0 - v for v in per_b[b].values()] if per_b.get(b) else []
                for b in budgets
            ]
            mean = np.array([np.mean(e) if e else np.nan for e in errs])
            sd = np.array([np.std(e) if e else np.nan for e in errs])
            series.append((float(np.nanmean(sd)), arm, mean, sd))
        for _w, arm, mean, sd in sorted(series, reverse=True):
            _band(
                ax,
                np.array(budgets, dtype=float),
                np.maximum(mean, ERR_FLOOR),
                sd,
                COLOUR[arm],
                "solid",
                LABEL[arm],
                floor=ERR_FLOOR,
            )
        if np.isfinite(heur_acc):
            ax.axhline(
                max(1.0 - heur_acc, ERR_FLOOR),
                lw=1.3,
                ls=(0, (2, 2)),
                color=F.INK_2,
                zorder=6,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets], fontsize=F.FS_TICK)
        ax.minorticks_off()
        err_axes.append(ax)
        F.style_axes(
            ax,
            title=f"test error — {INIT_TITLE[init]}",
            xlabel="objective evaluations (log)",
            ylabel="1 − test accuracy (log)" if col == 0 else None,
        )
        if col == 0 and np.isfinite(heur_acc):
            ax.text(
                budgets[-1],
                max(1.0 - heur_acc, ERR_FLOOR),
                "the construction ",
                va="bottom",
                ha="right",
                fontsize=F.FS_SMALL,
                color=F.INK_2,
            )

    # One scale across the bottom row, so the three columns can be compared.
    # Without it each panel autoscales to its own decade and "cold recovers to
    # within a factor of 30 of the construction" is invisible.
    for group, floor in ((err_axes, ERR_FLOOR * 0.7), (obj_axes, None)):
        if not group:
            continue
        lo = min(a.get_ylim()[0] for a in group)
        hi = max(a.get_ylim()[1] for a in group)
        for a in group:
            a.set_ylim(lo if floor is None else max(lo, floor), hi)

    fig.text(
        0.5,
        -0.012,
        f"Lines are the mean over {n_seeds} seeds and bands are ±1 s.d. over "
        f"the same seeds. Arms that never converge are drawn rather than "
        f"dropped — on a cold start the two\ngradient-based arms sit at their "
        f"starting objective for the whole budget, which is a result about "
        f"the loss surface and not missing data. The dashed line is the "
        f"Gaussian\nconstruction's own value in every panel. Top row is what "
        f"the arms minimize — the SHIPPED classifier objective, training "
        f"cross-entropy plus a ridge shrink toward each\narm's own start — so "
        f"it is a training loss; the bottom row is the only outcome to quote. "
        f"That row is ERROR RATE on a shared log axis, not accuracy: "
        f"PhiUSIIL is saturated enough that a\nlinear accuracy axis puts "
        f"every good model on one pixel and a ±1 s.d. band around 0.9998 "
        f"crosses 1.0, which is not a possible accuracy. A floor of "
        f"{ERR_FLOOR:g} keeps a zero-error\npoint on the axis. Same "
        f"features, split, box and evaluation budget throughout; only the "
        f"starting point differs. {H.provenance_note(label)}",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.6,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
