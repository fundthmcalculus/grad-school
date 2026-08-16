#!/usr/bin/env python3
"""Rule identification on Concrete: the classical route against the construction.

Both routes are asked for the same number of rules at every point, because the
rule count is an input to one and normally an output of the other, and comparing
across different counts would be comparing capacity rather than identification.

Two things had to be fixed before this comparison meant anything, and both were
in the library rather than in the method.

The first was a matched-capacity problem: the construction was choosing its own
component count while the classical route was told how many clusters to make, so
it carried three times the parameters. **pinned** fixes that — one Gaussian per
(feature, bucket) is exactly the classical shape, so panel (c)'s lines coincide
and panels (a) and (b) are like-for-like. **auto** is the construction still
choosing for itself.

The second was that choosing cost far more than it should. `find_optimal_gaussians`
fitted a full EM mixture at every candidate count, kept the winning *number*, threw
the mixtures away, and refitted by k-means. Scoring the k-means partition directly
made identification 4.1-4.7x cheaper here with equal or better held-out R². The
faint line in panel (b) is the same automatic construction before that fix, and
the distance to the line above it is the whole of what was being wasted.

**(a) Accuracy against rule count.** Held-out R², mean and ±1 s.d. over seeds.
Read the *overlap*, not the ordering — three seeds is not enough to separate
these curves, and the panel is drawn so that is obvious.

**(b) Identification cost against rule count**, log scale. With both fixes in,
the pinned construction and the classical routes are **at parity** — the ordering
changes with the rule count and the margins are tens of percent, not orders of
magnitude. The 23-84x classical advantage this study first reported was two
artifacts stacked: an unmatched parameter count and a model-selection search
nobody asked for. Cost is wall-clock here rather than iterations, unlike the
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
AUTO = "opt-identification-kmbic-2026-08-03"
PINNED = "opt-identification-kmbic-pinned-2026-08-03"
PREFIX = "opt-identification-2026-08-02"  # same sweep, before the library fix

CLASSICAL = ["classical-kmeans", "classical-fcm"]
COLOUR = {"construction": F.BLUE, "classical-kmeans": F.ORANGE, "classical-fcm": F.AQUA}
LABEL = {"classical-kmeans": "k-means", "classical-fcm": "fuzzy c-means"}


def _load(label):
    rows, src = H.table("table_identification_sweep_seeds", label)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["route"]][int(r["rules"])].append(
            (float(r["r2"]), float(r["seconds"]), int(r["n_params"]))
        )
    return by, src


def _stats(by, route, index):
    cs = sorted(by[route])
    vals = [[v[index] for v in by[route][c]] for c in cs]
    return (
        cs,
        np.array([np.mean(v) for v in vals]),
        np.array([np.std(v) for v in vals]),
        np.array([np.median(v) for v in vals]),
    )


def build():
    auto, auto_src = _load(AUTO)
    pinned, pinned_src = _load(PINNED)
    try:
        before, before_src = _load(PREFIX)
    except FileNotFoundError:
        before, before_src = None, "(missing)"
    if "construction" not in auto or "construction" not in pinned:
        raise RuntimeError("no sweep data; run run_identification_study.py")
    rules = sorted(auto["construction"])

    fig, (ax, tx, px) = F.grid_figure(1, 3, width=F.W_WIDE + 1.4, height=3.9)

    # -- (a) accuracy ------------------------------------------------------- #
    for route in CLASSICAL:
        cs, mean, sd, _ = _stats(pinned, route, 0)
        ax.plot(
            cs,
            mean,
            lw=1.8,
            marker="o",
            ms=4.5,
            color=COLOUR[route],
            label=LABEL[route],
            zorder=4,
        )
        ax.fill_between(
            cs, mean - sd, mean + sd, color=F.tint(COLOUR[route], 0.88), lw=0, zorder=2
        )
    # Distinct markers, not just distinct dashes: a legend swatch is short
    # enough that a marker sits on top of the dash pattern and the two blue
    # entries become indistinguishable.
    for src, ls, mk, name in (
        (pinned, "solid", "o", "construction (pinned)"),
        (auto, (0, (4, 2)), "^", "construction (auto)"),
    ):
        cs, mean, sd, _ = _stats(src, "construction", 0)
        ax.plot(
            cs,
            mean,
            lw=1.8,
            ls=ls,
            marker=mk,
            ms=4.5,
            color=F.BLUE,
            label=name,
            zorder=4,
        )
        if ls == "solid":
            ax.fill_between(
                cs, mean - sd, mean + sd, color=F.tint(F.BLUE, 0.88), lw=0, zorder=2
            )
    F.style_axes(
        ax,
        title="(a)  accuracy at matched rule count",
        xlabel="rules",
        ylabel="held-out $R^2$",
    )
    F.legend(ax, loc="lower right", handlelength=2.8)
    ax.set_xticks(rules)

    # -- (b) cost ----------------------------------------------------------- #
    for route in CLASSICAL:
        cs, _, _, med = _stats(pinned, route, 1)
        tx.plot(
            cs,
            1000 * med,
            lw=1.8,
            marker="o",
            ms=4.5,
            color=COLOUR[route],
            label=LABEL[route],
            zorder=4,
        )
    for src, ls, mk, name in (
        (pinned, "solid", "o", "construction (pinned)"),
        (auto, (0, (4, 2)), "^", "construction (auto)"),
    ):
        cs, _, _, med = _stats(src, "construction", 1)
        tx.plot(
            cs,
            1000 * med,
            lw=1.8,
            ls=ls,
            marker=mk,
            ms=4.5,
            color=F.BLUE,
            label=name,
            zorder=4,
        )
    # The same automatic construction before the selector was fixed. Drawn faint
    # because it is not a competitor -- it is the same code path paying for four
    # EM fits it discarded.
    if before is not None:
        cs, _, _, med = _stats(before, "construction", 1)
        tx.plot(
            cs,
            1000 * med,
            lw=1.4,
            ls=(0, (1, 2)),
            marker="v",
            ms=4.0,
            color=F.FAINT,
            label="auto, before the\nlibrary fix",
            zorder=3,
        )
    tx.set_yscale("log")
    lo, hi = tx.get_ylim()
    tx.set_ylim(lo * 0.55, hi * 3.0)
    # No legend here: five series in a narrow log panel puts the key on top of
    # the curves whichever corner it goes in, and (a) and (c) already name the
    # same four. Only the fifth needs saying, so it is said on the line itself.
    if before is not None:
        cs, _, _, med = _stats(before, "construction", 1)
        _cs, _, _, med_auto = _stats(auto, "construction", 1)
        tx.annotate(
            f"auto, before the library fix\n"
            f"({med[-1] / med_auto[-1]:.1f}× dearer at 12 rules)",
            xy=(cs[-2], 1000 * med[-2] * 1.06),
            xytext=(cs[0] + 0.05, 1000 * med[-1] * 2.4),
            fontsize=F.FS_SMALL,
            color=F.MUTED,
            linespacing=1.5,
            arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS),
        )
    F.style_axes(
        tx,
        title="(b)  identification cost",
        xlabel="rules",
        ylabel="milliseconds, single-threaded (log)",
    )
    tx.set_xticks(rules)

    # -- (c) parameters ----------------------------------------------------- #
    # The two classical routes and the pinned construction have identical
    # parameter counts by construction (one Gaussian per feature per rule), so
    # they are drawn with different dashes -- otherwise three lines sit exactly
    # on top of each other and two of them look absent.
    for route, ls in (("classical-kmeans", (0, (4, 2))), ("classical-fcm", "solid")):
        cs, mean, _, _ = _stats(pinned, route, 2)
        px.plot(
            cs,
            mean,
            lw=1.8,
            ls=ls,
            marker="o",
            ms=4.5,
            color=COLOUR[route],
            label=LABEL[route],
            zorder=4,
        )
    cs, mean, _, _ = _stats(pinned, "construction", 2)
    px.plot(
        cs,
        mean,
        lw=1.4,
        ls=(0, (1, 2)),
        marker="s",
        ms=3.6,
        color=F.BLUE,
        label="construction (pinned)",
        zorder=5,
    )
    cs, mean, _, _ = _stats(auto, "construction", 2)
    px.plot(
        cs,
        mean,
        lw=1.8,
        marker="^",
        ms=4.5,
        color=F.BLUE,
        label="construction (auto)",
        zorder=4,
    )
    F.style_axes(
        px, title="(c)  free antecedent parameters", xlabel="rules", ylabel="parameters"
    )
    F.legend(px, loc="upper left", handlelength=2.8)
    px.set_xticks(rules)

    fig.text(
        0.5,
        -0.02,
        "Same data, same split, same closed-form ridge-TSK consequent solve in "
        "every row — what differs is only how the rules are identified. Bands in "
        "(a) are ±1 s.d. over seeds and they\noverlap everywhere: three seeds "
        "cannot separate these curves, and the accuracy ordering is not a result. "
        "The gap between the two solid blue\nlines and the classical ones in (b) "
        "is now tens of percent rather than the orders of magnitude this study "
        "first reported: that gap was an unmatched\nparameter count plus a "
        "model-selection search the classical route is never asked to do, and (c) "
        "shows the first of those — on the pinned run all\nthree routes carry "
        "identical parameter counts, which is why those lines coincide. Timing is "
        "wall-clock, single-threaded, median of repeats — the one\nplace in this "
        "study where time rather than iteration count is the quantity of interest. "
        f"auto: {H.provenance_note(auto_src)} · pinned: {pinned_src}"
        + (f" · before: {before_src}" if before is not None else ""),
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
