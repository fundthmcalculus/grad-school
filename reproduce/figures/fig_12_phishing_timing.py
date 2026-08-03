#!/usr/bin/env python3
"""PhiUSIIL: what construction costs against what searching for it costs.

The companion to `fig_11`, which shows what each route *achieves*. This one shows
what each route *costs*, in the same units, because "the construction is faster"
is a claim about time and nothing else in this study measures it directly.

**(a) Wall-clock per arm.** Bars are the mean seconds an arm spent over its whole
budget, with a ±1 s.d. whisker. The red line is what the Gaussian construction
itself cost on the same data and the same machine. A black dot on a bar is the
mean moment that arm first reached the construction's own objective value; a bar
labelled *never matched* did not get there inside the budget, which is an answer
and not a gap in the data.

**(b) The same thing as a ratio**, which is the part that survives a change of
machine: how many times the construction's own cost a search had to spend before
it arrived at the construction's objective. Arms that never arrived have no bar —
their cost is a lower bound, not a number, and drawing one would invent it.

Absolute seconds are machine-dependent and the machine is recorded in the
archive's `PROVENANCE.txt`. Everything is single-threaded, which cuts both ways:
no arm can buy time with cores, and an arm that would parallelise well gets no
credit here.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "12-phishing-timing"

ARM_ORDER = ["scipy-lbfgsb", "scipy-powell", "scipy-de",
             "opt-ga", "opt-pso", "opt-aco", "opt-gd"]
COLOUR = {a: F.SERIES[i % len(F.SERIES)] for i, a in enumerate(ARM_ORDER)}
LABEL = {"scipy-lbfgsb": "L-BFGS-B", "scipy-powell": "Powell",
         "scipy-de": "DE (scipy)", "opt-ga": "GA", "opt-pso": "PSO",
         "opt-aco": "ACO", "opt-gd": "GD"}
INIT_ORDER = ["hot", "cold", "classical-kmeans"]
INIT_SHORT = {"hot": "hot", "cold": "cold", "classical-kmeans": "k-means"}


def _f(row, key, default=float("nan")):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def build():
    rows, label = H.table("table_opt_phishing_seeds")
    by = defaultdict(list)
    for r in rows:
        by[(r["arm"], r["init"])].append(r)

    ref = by.get(("none", "hot"), [])
    constr = np.array([_f(r, "construction_seconds") for r in ref])
    constr_mean = float(np.mean(constr)) if len(constr) else float("nan")
    screen = np.array([_f(r, "screen_seconds") for r in ref])
    inits = [i for i in INIT_ORDER if any(k[1] == i for k in by)]
    arms = [a for a in ARM_ORDER if any((a, i) in by for i in inits)]
    n_seeds = len({r["seed"] for r in rows})

    fig, (ax, rx) = F.grid_figure(1, 2, width=F.W_WIDE + 1.6, height=5.6,
                                 gridspec_kw={"width_ratios": [1.35, 1]})

    # -- (a) absolute wall-clock -------------------------------------------- #
    ticks, means, sds, colours, matched = [], [], [], [], []
    for init in inits:
        for arm in arms:
            sel = by.get((arm, init))
            if not sel:
                continue
            secs = np.array([_f(r, "seconds") for r in sel])
            raw = np.array([_f(r, "seconds_to_heuristic") for r in sel])
            at_zero = int(np.sum(np.isfinite(raw) & (raw <= 0)))
            reach = raw[np.isfinite(raw) & (raw > 0)]
            ticks.append(f"{LABEL[arm]} · {INIT_SHORT[init]}")
            means.append(float(np.mean(secs)))
            sds.append(float(np.std(secs)))
            colours.append(F.tint(COLOUR[arm], 0.0 if init == "hot" else 0.45))
            if len(reach):
                matched.append(("at", float(np.mean(reach)), len(reach), len(sel)))
            elif at_zero:
                # Matched at evaluation zero, i.e. it began there. True of every
                # hot arm, and not a result -- but "never matched" would be a
                # flat falsehood, which is what the `> 0` filter used to produce.
                matched.append(("start", 0.0, at_zero, len(sel)))
            else:
                matched.append(None)

    y = np.arange(len(ticks))
    ax.barh(y, means, xerr=sds, height=0.74, color=colours, zorder=3,
            error_kw=dict(ecolor=F.INK_2, elinewidth=0.9, capsize=2))
    # Labels go past the end of the whisker, not the end of the bar, or they sit
    # on top of it.
    for yi, (m, sd, hit) in enumerate(zip(means, sds, matched)):
        end = (m + sd) * 1.12
        if hit is None:
            ax.text(end, yi, "never matched", va="center", ha="left",
                    fontsize=F.FS_SMALL, color=F.MUTED)
        elif hit[0] == "start":
            ax.text(end, yi, "starts matched", va="center", ha="left",
                    fontsize=F.FS_SMALL, color=F.MUTED)
        else:
            _kind, secs, k, n = hit
            ax.plot([secs], [yi], marker="o", ms=5.2, color=F.INK, zorder=6)
            ax.text(end, yi, f"matched at {secs:.0f}s"
                             + ("" if k == n else f", {k}/{n} seeds"),
                    va="center", ha="left", fontsize=F.FS_SMALL, color=F.MUTED)
    if np.isfinite(constr_mean) and constr_mean > 0:
        ax.axvline(constr_mean, lw=1.5, color=F.RED, zorder=5)
        # Horizontal, above the plot area: rotated inside the axes it crossed
        # every bar it was meant to be a reference for.
        # Inside the axes, below the top edge: above it, the label collided with
        # the panel title.
        ax.annotate(f"the construction: {1000 * constr_mean:.0f} ms",
                    xy=(constr_mean, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -4), textcoords="offset points",
                    ha="left", va="top", fontsize=F.FS_SMALL, color=F.RED)
    ax.set_yticks(y)
    ax.set_yticklabels(ticks, fontsize=F.FS_SMALL)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(min(constr_mean * 0.35, min(means) * 0.5),
                max(m + s for m, s in zip(means, sds)) * 16.0)
    F.style_axes(ax, title="(a)  wall-clock spent, mean ±1 s.d.",
                 xlabel="seconds, single-threaded (log)", grid_axis="x")

    # -- (b) the portable version: cost of matching, as a multiple ---------- #
    labels, ratios, errs, cols = [], [], [], []
    for init in inits:
        for arm in arms:
            sel = by.get((arm, init))
            if not sel or init == "hot":
                continue          # a hot arm starts at the construction
            reach = np.array([_f(r, "seconds_to_heuristic") for r in sel])
            reach = reach[np.isfinite(reach) & (reach > 0)]
            if not len(reach) or not np.isfinite(constr_mean) or constr_mean <= 0:
                continue
            labels.append(f"{LABEL[arm]} · {INIT_SHORT[init]}")
            ratios.append(float(np.mean(reach)) / constr_mean)
            errs.append(float(np.std(reach)) / constr_mean)
            cols.append(F.tint(COLOUR[arm], 0.45))
    if labels:
        order = np.argsort(ratios)
        labels = [labels[i] for i in order]
        ratios = [ratios[i] for i in order]
        errs = [errs[i] for i in order]
        cols = [cols[i] for i in order]
        y = np.arange(len(labels))
        rx.barh(y, ratios, xerr=errs, height=0.7, color=cols, zorder=3,
                error_kw=dict(ecolor=F.INK_2, elinewidth=0.9, capsize=2))
        for yi, v, e in zip(y, ratios, errs):
            rx.text((v + e) * 1.10, yi, f"{v:,.0f}×", va="center", ha="left",
                    fontsize=F.FS_SMALL, color=F.INK_2)
        rx.set_yticks(y)
        rx.set_yticklabels(labels, fontsize=F.FS_SMALL)
        rx.invert_yaxis()
        rx.axvline(1.0, lw=1.5, color=F.RED, zorder=5)
        rx.set_xscale("log")
        rx.set_xlim(0.5, max(v + e for v, e in zip(ratios, errs)) * 3.0)
    else:
        rx.text(0.5, 0.5, "no arm reached the construction's\nobjective inside "
                          "the budget",
                transform=rx.transAxes, ha="center", va="center",
                fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.6)
    F.style_axes(rx, title="(b)  cost of matching the construction,\n"
                           "÷ the construction's own cost",
                 xlabel="multiple of the construction's cost (log)",
                 grid_axis="x")

    note = (f"Mean over {n_seeds} seeds. In (a) the whisker is ±1 s.d. of the "
            f"time spent and the black dot is the mean moment that arm first "
            f"reached the construction's own\nobjective value; bars with no dot "
            f"never reached it, which is an answer rather than a gap. (b) drops "
            f"the hot arms, which start at the construction, and drops any arm "
            f"that\nnever matched — its cost is a lower bound, not a number. ")
    if len(screen) and np.isfinite(np.mean(screen)):
        note += (f"Feature engineering, shared by every route and charged to none "
                 f"of them, cost {1000 * float(np.mean(screen)):.0f} ms.\n")
    note += (f"Absolute seconds are machine-dependent and the machine is in the "
             f"archive; the ratio in (b) is the portable part. Single-threaded "
             f"throughout, so an arm that\nparallelises well gets no credit. "
             f"{H.provenance_note(label)}")
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=F.FS_SMALL,
             color=F.MUTED, linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
