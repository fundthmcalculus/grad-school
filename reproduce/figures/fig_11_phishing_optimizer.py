#!/usr/bin/env python3
"""PhiUSIIL: what a search is worth on a classification problem, and what it costs.

The classification counterpart to `fig_07`/`fig_10`. The regression study found
the construction's *placement* worth almost nothing — a random draw in the same
box tied it before any optimization. That does not carry over, and this figure is
mostly about how badly it does not.

**(a) Objective against evaluations**, log-log, median across seeds. The
objective is the shipped classifier fitness (training cross-entropy plus a ridge
shrink), so lower is better and the construction's own value is the dashed line.
A cold draw starts three orders of magnitude above it. Whether anything closes
that gap inside the budget is the panel's question.

**(b) Test accuracy against the budget.** The outcome column, on a large held-out
split because PhiUSIIL is saturated. Two readings: the hot line is flat, because
there is nothing left to win; the cold line is a recovery curve, and where it
plateaus is what a search buys you when you do not have the construction.

**(c) Wall-clock, which is the point of the exercise.** Bars are the median
seconds an arm actually spent; the vertical line is what the construction cost.
A filled marker on a bar is the moment that arm first reached the construction's
own objective value — a bar with no marker never got there at all, which is an
answer and is labelled as one. Absolute seconds are machine-dependent and the
machine is recorded in the archive; the ratio between the bar and the line is the
portable part.
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

ARM_ORDER = ["scipy-lbfgsb", "scipy-powell", "scipy-de",
             "opt-ga", "opt-pso", "opt-aco", "opt-gd"]
COLOUR = {a: F.SERIES[i % len(F.SERIES)] for i, a in enumerate(ARM_ORDER)}
LABEL = {"scipy-lbfgsb": "L-BFGS-B", "scipy-powell": "Powell",
         "scipy-de": "DE (scipy)", "opt-ga": "GA", "opt-pso": "PSO",
         "opt-aco": "ACO", "opt-gd": "GD"}
INIT_STYLE = {"hot": ("solid", 1.9), "cold": ((0, (4, 2)), 1.4),
              "classical-kmeans": ((0, (1, 2)), 1.4)}
INIT_LABEL = {"hot": "construction", "cold": "random in the box",
              "classical-kmeans": "k-means per class"}


def _traces():
    rows, label = H.table("table_opt_phishing_traces")
    out = defaultdict(lambda: defaultdict(list))
    for r in rows:
        out[(r["arm"], r["init"])][int(r["seed"])].append(
            (int(r["eval"]), float(r["best_obj"])))
    return out, label


def _seeds():
    rows, _ = H.table("table_opt_phishing_seeds")
    return rows


def _budget():
    rows, _ = H.table("table_opt_phishing_budget")
    out = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        out[(r["arm"], r["init"])][int(r["budget"])][int(r["seed"])] = float(r["acc"])
    return out


def _median_curve(per_seed, grid):
    """Forward-fill each seed's best-so-far step function, then median."""
    stacked = []
    for pts in per_seed.values():
        if not pts:
            continue
        pts = sorted(pts)
        e = np.array([p[0] for p in pts], dtype=float)
        v = np.array([p[1] for p in pts], dtype=float)
        idx = np.clip(np.searchsorted(e, grid, side="right") - 1, 0, len(v) - 1)
        stacked.append(v[idx])
    return np.median(np.vstack(stacked), axis=0) if stacked else None


def _f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def build():
    traces, label = _traces()
    seed_rows = _seeds()
    budget_acc = _budget()
    if not traces:
        raise RuntimeError("no traces; run run_phishing_opt_study.py")

    inits = [i for i in ("hot", "cold", "classical-kmeans")
             if any(k[1] == i for k in traces)]
    arms = [a for a in ARM_ORDER if (a, inits[0]) in traces]
    max_eval = max(e for per in traces.values() for pts in per.values()
                   for e, _v in pts) or 2000

    # The construction's own reference numbers, from the `none`/hot rows.
    ref = [r for r in seed_rows if r["arm"] == "none" and r["init"] == "hot"]
    heur_obj = float(np.median([_f(r, "heuristic_obj") for r in ref])) if ref else np.nan
    heur_acc = float(np.median([_f(r, "heuristic_acc") for r in ref])) if ref else np.nan
    constr_s = float(np.median([_f(r, "construction_seconds") for r in ref])) if ref else np.nan

    fig, (ax, bx, tx) = F.grid_figure(1, 3, width=F.W_WIDE + 2.0, height=4.2,
                                      gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]})

    # -- (a) objective convergence ------------------------------------------ #
    grid = np.unique(np.concatenate([[1], np.logspace(0, np.log10(max_eval), 200)]))
    for arm in arms:
        for init in inits:
            per_seed = traces.get((arm, init))
            if not per_seed:
                continue
            med = _median_curve(per_seed, grid)
            if med is None:
                continue
            ls, lw = INIT_STYLE[init]
            ax.plot(grid, med, lw=lw, ls=ls, color=COLOUR[arm], zorder=4,
                    label=LABEL[arm] if init == "hot" else None)
    if np.isfinite(heur_obj):
        ax.axhline(heur_obj, lw=1.2, ls=(0, (2, 2)), color=F.FAINT, zorder=2)
        ax.text(1.1, heur_obj, " the construction", va="bottom", ha="left",
                fontsize=F.FS_SMALL, color=F.MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    F.style_axes(ax, title="(a)  objective against evaluations",
                 xlabel="objective evaluations (log)",
                 ylabel="cross-entropy + shrink (log)")
    F.legend(ax, loc="upper right", ncol=2, handlelength=2.2)
    ax.text(0.03, 0.03, "solid = hot · dashed = cold\ndotted = k-means per class",
            transform=ax.transAxes, fontsize=F.FS_SMALL, color=F.MUTED,
            linespacing=1.5)

    # -- (b) test accuracy against budget ----------------------------------- #
    budgets = sorted({b for per in budget_acc.values() for b in per})
    for init in inits:
        per_arm = []
        for arm in arms:
            d = budget_acc.get((arm, init))
            if not d:
                continue
            per_arm.append([float(np.median(list(d[b].values())))
                            if d.get(b) else np.nan for b in budgets])
        if not per_arm:
            continue
        med = np.nanmedian(np.vstack(per_arm), axis=0)
        ls, lw = INIT_STYLE[init]
        colour = {"hot": F.BLUE, "cold": F.ORANGE}.get(init, F.GREEN)
        bx.plot(budgets, med, lw=2.2, ls=ls, marker="o", ms=4.5, color=colour,
                zorder=5, label=f"from the {INIT_LABEL[init]}")
        for row in per_arm:
            bx.plot(budgets, row, lw=0.9, ls=ls, color=F.tint(colour, 0.55),
                    zorder=3)
    if np.isfinite(heur_acc):
        bx.axhline(heur_acc, lw=1.2, ls=(0, (2, 2)), color=F.FAINT, zorder=2)
    bx.set_xscale("log")
    bx.set_xticks(budgets)
    bx.set_xticklabels([str(b) for b in budgets], fontsize=F.FS_TICK)
    bx.minorticks_off()
    F.style_axes(bx, title="(b)  test accuracy against budget",
                 xlabel="objective evaluations (log)", ylabel="test accuracy")
    F.legend(bx, loc="lower right", handlelength=2.4)

    # -- (c) wall-clock ------------------------------------------------------ #
    bars, ticks, colours, marks = [], [], [], []
    for init in inits:
        for arm in arms:
            sel = [r for r in seed_rows
                   if r["arm"] == arm and r["init"] == init]
            if not sel:
                continue
            spent = float(np.median([_f(r, "seconds") for r in sel]))
            reached = [_f(r, "seconds_to_heuristic") for r in sel
                       if (r.get("seconds_to_heuristic") or "") != ""]
            bars.append(spent)
            ticks.append(f"{LABEL[arm]} · {INIT_LABEL[init]}")
            colours.append(F.tint(COLOUR[arm],
                                  0.0 if init == "hot" else 0.45))
            marks.append(float(np.median(reached)) if reached else None)
    y = np.arange(len(bars))
    tx.barh(y, bars, color=colours, height=0.72, zorder=3)
    for yi, (b, m) in enumerate(zip(bars, marks)):
        if m is not None and m > 0:
            tx.plot([m], [yi], marker="o", ms=5.0, color=F.INK, zorder=6)
        elif m == 0:
            # Reached at evaluation zero, i.e. it started there. True of every
            # hot arm and not a result -- flagged rather than drawn as progress.
            tx.text(b * 1.05, yi, " starts there", va="center", ha="left",
                    fontsize=F.FS_SMALL, color=F.MUTED)
        else:
            tx.text(b * 1.05, yi, " never matched", va="center", ha="left",
                    fontsize=F.FS_SMALL, color=F.MUTED)
    if np.isfinite(constr_s) and constr_s > 0:
        tx.axvline(constr_s, lw=1.4, color=F.RED, zorder=5)
        tx.text(constr_s, len(bars) - 0.3,
                f" the construction: {1000 * constr_s:.0f} ms",
                va="top", ha="left", rotation=90, fontsize=F.FS_SMALL,
                color=F.RED)
    tx.set_yticks(y)
    tx.set_yticklabels(ticks, fontsize=F.FS_SMALL)
    tx.invert_yaxis()
    tx.set_xscale("log")
    tx.set_xlim(min(constr_s * 0.4 if np.isfinite(constr_s) else 0.05, 0.05),
                max(bars) * 6.0)
    F.style_axes(tx, title="(c)  wall-clock: what each route costs",
                 xlabel="seconds, single-threaded (log)", grid_axis="x")

    fig.text(0.5, -0.02,
             "Same features, same split, same objective, same box, same "
             "evaluation budget — only the starting point differs. The objective "
             "in (a) is the SHIPPED classifier fitness\n"
             "(`refine._make_classifier_fitness`: training cross-entropy plus a "
             "ridge shrink toward each arm's own start), so it is a training loss "
             "and (b) is the only outcome to quote.\nTest accuracy is measured on "
             "a large held-out split because PhiUSIIL is saturated — a small one "
             "cannot separate two good models. In (c) a black dot is the moment "
             "an arm\nfirst reached the construction's objective value; absolute "
             "seconds are machine-dependent, the ratio to the red line is not. "
             f"Single-threaded throughout. {H.provenance_note(label)}",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
