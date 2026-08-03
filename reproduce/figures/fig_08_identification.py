#!/usr/bin/env python3
"""Rule identification on Concrete: the classical route against the construction.

Both routes are asked for the same number of rules at every point, because the
rule count is an input to one and normally an output of the other, and comparing
across different counts would be comparing capacity rather than identification.

The first version of this figure compared the construction *with its own model
selection left in* against a classical route that is simply told how many
clusters to make. That is two questions at once, and it flattered the classical
route by an order of magnitude. Both runs are drawn here:

* **auto** — the construction chooses its component count per (feature, bucket)
  by BIC, which costs four EM fits per pair on top of the k-means it keeps, and
  leaves it with roughly three times the classical parameter count;
* **pinned** — one Gaussian per (feature, bucket), which is *exactly* the
  classical shape, so panel (c)'s lines coincide and panels (a) and (b) are
  like-for-like.

**(a) Accuracy against rule count.** Held-out R², mean and ±1 s.d. over seeds.
Read the *overlap*, not the ordering — three seeds is not enough to separate
these curves, and the panel is drawn so that is obvious.

**(b) Identification cost against rule count**, log scale. The classical route
is still cheaper at matched parameters, by about 2-5x. It was 12-63x before
pinning, and the difference between those two numbers is model selection, not
rule placement. Cost is wall-clock here rather than iterations, unlike the
optimizer study — "the construction is cheaper" is a claim about time, so time
is what is measured, and measured single-threaded.

**(c) What each route spends its parameters on.** Why (a) is only a fair
comparison for the pinned run: at a matched rule count the automatic
construction carries about three times the free parameters, while the pinned one
lands exactly on the classical line. Arithmetic, not measurement.
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

# Both archives are named rather than inferred. The panels are a comparison
# *between* two runs, so "whatever is newest" would silently plot one run
# against itself the moment either is re-archived.
AUTO = "opt-identification-2026-08-02"
PINNED = "opt-identification-pinned-2026-08-03"

CLASSICAL = ["classical-kmeans", "classical-fcm"]
COLOUR = {"construction": F.BLUE, "classical-kmeans": F.ORANGE,
          "classical-fcm": F.AQUA}
LABEL = {"classical-kmeans": "k-means", "classical-fcm": "fuzzy c-means"}


def _load(label):
    rows, src = H.table("table_identification_sweep_seeds", label)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["route"]][int(r["rules"])].append(
            (float(r["r2"]), float(r["seconds"]), int(r["n_params"])))
    return by, src


def _stats(by, route, index):
    cs = sorted(by[route])
    vals = [[v[index] for v in by[route][c]] for c in cs]
    return (cs, np.array([np.mean(v) for v in vals]),
            np.array([np.std(v) for v in vals]),
            np.array([np.median(v) for v in vals]))


def build():
    auto, auto_src = _load(AUTO)
    pinned, pinned_src = _load(PINNED)
    if "construction" not in auto or "construction" not in pinned:
        raise RuntimeError("no sweep data; run run_identification_study.py")
    rules = sorted(auto["construction"])

    fig, (ax, tx, px) = F.grid_figure(1, 3, width=F.W_WIDE + 1.4, height=3.9)

    # -- (a) accuracy ------------------------------------------------------- #
    for route in CLASSICAL:
        cs, mean, sd, _ = _stats(pinned, route, 0)
        ax.plot(cs, mean, lw=1.8, marker="o", ms=4.5, color=COLOUR[route],
                label=LABEL[route], zorder=4)
        ax.fill_between(cs, mean - sd, mean + sd,
                        color=F.tint(COLOUR[route], 0.88), lw=0, zorder=2)
    # Distinct markers, not just distinct dashes: a legend swatch is short
    # enough that a marker sits on top of the dash pattern and the two blue
    # entries become indistinguishable.
    for src, ls, mk, name in ((pinned, "solid", "o", "construction (pinned)"),
                              (auto, (0, (4, 2)), "^", "construction (auto)")):
        cs, mean, sd, _ = _stats(src, "construction", 0)
        ax.plot(cs, mean, lw=1.8, ls=ls, marker=mk, ms=4.5, color=F.BLUE,
                label=name, zorder=4)
        if ls == "solid":
            ax.fill_between(cs, mean - sd, mean + sd,
                            color=F.tint(F.BLUE, 0.88), lw=0, zorder=2)
    F.style_axes(ax, title="(a)  accuracy at matched rule count",
                 xlabel="rules", ylabel="held-out $R^2$")
    F.legend(ax, loc="lower right", handlelength=2.8)
    ax.set_xticks(rules)

    # -- (b) cost ----------------------------------------------------------- #
    for route in CLASSICAL:
        cs, _, _, med = _stats(pinned, route, 1)
        tx.plot(cs, 1000 * med, lw=1.8, marker="o", ms=4.5,
                color=COLOUR[route], label=LABEL[route], zorder=4)
    for src, ls, mk, name in ((pinned, "solid", "o", "construction (pinned)"),
                              (auto, (0, (4, 2)), "^", "construction (auto)")):
        cs, _, _, med = _stats(src, "construction", 1)
        tx.plot(cs, 1000 * med, lw=1.8, ls=ls, marker=mk, ms=4.5,
                color=F.BLUE, label=name, zorder=4)
    tx.set_yscale("log")
    lo, hi = tx.get_ylim()
    tx.set_ylim(lo * 0.55, hi)
    F.style_axes(tx, title="(b)  identification cost",
                 xlabel="rules", ylabel="milliseconds, single-threaded (log)")
    F.legend(tx, loc="lower right", handlelength=2.8)
    tx.set_xticks(rules)

    # -- (c) parameters ----------------------------------------------------- #
    # The two classical routes and the pinned construction have identical
    # parameter counts by construction (one Gaussian per feature per rule), so
    # they are drawn with different dashes -- otherwise three lines sit exactly
    # on top of each other and two of them look absent.
    for route, ls in (("classical-kmeans", (0, (4, 2))),
                      ("classical-fcm", "solid")):
        cs, mean, _, _ = _stats(pinned, route, 2)
        px.plot(cs, mean, lw=1.8, ls=ls, marker="o", ms=4.5,
                color=COLOUR[route], label=LABEL[route], zorder=4)
    cs, mean, _, _ = _stats(pinned, "construction", 2)
    px.plot(cs, mean, lw=1.4, ls=(0, (1, 2)), marker="s", ms=3.6,
            color=F.BLUE, label="construction (pinned)", zorder=5)
    cs, mean, _, _ = _stats(auto, "construction", 2)
    px.plot(cs, mean, lw=1.8, marker="^", ms=4.5, color=F.BLUE,
            label="construction (auto)", zorder=4)
    F.style_axes(px, title="(c)  free antecedent parameters",
                 xlabel="rules", ylabel="parameters")
    F.legend(px, loc="upper left", handlelength=2.8)
    px.set_xticks(rules)

    fig.text(0.5, -0.02,
             "Same data, same split, same closed-form ridge-TSK consequent solve in "
             "every row — what differs is only how the rules are identified. Bands in "
             "(a) are ±1 s.d. over seeds and they\noverlap everywhere: three seeds "
             "cannot separate these curves, and the accuracy ordering is not a result. "
             "The gap between the two blue lines in (b) is BIC component\nselection, "
             "which the classical route is never asked to do; on the pinned run the "
             "three routes carry identical parameter counts, which is the (c) lines "
             "coinciding.\nTiming is wall-clock, single-threaded, median of repeats — "
             "the one place in this study where time rather than iteration count is the "
             f"quantity of interest.\nauto: {H.provenance_note(auto_src)} · "
             f"pinned: {pinned_src}",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
