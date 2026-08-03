#!/usr/bin/env python3
"""Rule identification: the classical route against the Gaussian construction.

Both routes are asked for the same number of rules at every point, because the
rule count is an input to one and normally an output of the other, and comparing
across different counts would be comparing capacity rather than identification.

**(a) Accuracy against rule count.** Held-out R², mean and spread over seeds.
Read the *overlap*, not the ordering — three seeds is not enough to separate
these curves, and the panel is drawn so that is obvious.

**(b) Identification cost against rule count**, log scale. This is the result
that is not close: the classical route is one to two orders of magnitude
cheaper at every rule count, with spreads too small to see. Note that cost is
wall-clock here rather than iterations, unlike the optimizer study — "the
construction is cheaper" is a claim about time, so time is what is measured, and
measured single-threaded.

**(c) What each route spends its parameters on.** At a matched rule count the
construction carries roughly three times the free parameters, because it fits an
automatically-chosen number of mixture components per (feature, bucket) while
the classical route places exactly one Gaussian per (feature, rule). This is
arithmetic, not measurement, and it is why panel (a) is not a like-for-like
capacity comparison in the construction's favour.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "08-identification"

ROUTES = ["construction", "classical-kmeans", "classical-fcm"]
COLOUR = {"construction": F.BLUE, "classical-kmeans": F.ORANGE,
          "classical-fcm": F.AQUA}


def _load():
    rows, label = H.table("table_identification_sweep_seeds")
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["route"]][int(r["rules"])].append(
            (float(r["r2"]), float(r["seconds"]), int(r["n_params"])))
    return by, label


def build():
    by, label = _load()
    present = [r for r in ROUTES if r in by]
    if not present:
        raise RuntimeError("no sweep data; run run_identification_study.py")
    rules = sorted(by[present[0]])

    fig, (ax, tx, px) = F.grid_figure(1, 3, width=F.W_WIDE + 1.4, height=3.6)

    for route in present:
        cs = sorted(by[route])
        r2 = np.array([[v[0] for v in by[route][c]] for c in cs], dtype=object)
        mean = np.array([np.mean([v[0] for v in by[route][c]]) for c in cs])
        sd = np.array([np.std([v[0] for v in by[route][c]]) for c in cs])
        ax.plot(cs, mean, lw=1.8, marker="o", ms=4.5, color=COLOUR[route],
                label=route, zorder=4)
        ax.fill_between(cs, mean - sd, mean + sd, color=F.tint(COLOUR[route], 0.88),
                        lw=0, zorder=2)

        ms = [1000 * np.median([v[1] for v in by[route][c]]) for c in cs]
        tx.plot(cs, ms, lw=1.8, marker="o", ms=4.5, color=COLOUR[route], zorder=4)

        pars = [np.mean([v[2] for v in by[route][c]]) for c in cs]
        # The two classical routes have identical parameter counts by
        # construction (one Gaussian per feature per rule), so one is dashed --
        # otherwise the second is drawn exactly under the first and looks absent.
        px.plot(cs, pars, lw=1.8, marker="o", ms=4.5, color=COLOUR[route],
                ls=(0, (4, 2)) if route == "classical-kmeans" else "solid",
                zorder=4)

    F.style_axes(ax, title="(a)  accuracy at matched rule count",
                 xlabel="rules", ylabel="held-out $R^2$")
    F.legend(ax, loc="lower right")
    ax.set_xticks(rules)

    tx.set_yscale("log")
    F.style_axes(tx, title="(b)  identification cost",
                 xlabel="rules", ylabel="milliseconds, single-threaded (log)")
    tx.set_xticks(rules)

    F.style_axes(px, title="(c)  free antecedent parameters",
                 xlabel="rules", ylabel="parameters")
    px.set_xticks(rules)

    fig.text(0.5, -0.02,
             "Same data, same split, same closed-form ridge-TSK consequent solve in "
             "every row — what differs is only how the rules are identified. Bands in "
             "(a) are ±1 s.d. over\nseeds and they overlap everywhere: three seeds "
             "cannot separate these curves, and the accuracy ordering is not a result. "
             "Panel (b) is not close. Timing is wall-clock,\nsingle-threaded, median of "
             "repeats — the one place in this study where time rather than iteration "
             f"count is the quantity of interest. {H.provenance_note(label)}",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
