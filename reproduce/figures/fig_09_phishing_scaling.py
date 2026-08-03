#!/usr/bin/env python3
"""PhiUSIIL: how the identification routes scale, and where the cost actually is.

Concrete said the classical route identifies a rule base 25-84x cheaper. That
was 824 rows. The two routes scale in different variables, so the interesting
question was never "which is faster" but "does that gap survive size". It does
not — and reading the timings against the library source says why, which is not
what the first version of this figure implied.

Two facts drive every panel, and both come from `fit_gaussians`:

* the construction **truncates each (feature, class) column to its first 20,000
  rows** before fitting anything, so above that its fitting cost stops growing;
* with an automatic component count it fits **four EM mixtures per (feature,
  class)** to choose one by BIC, then throws them away and runs k-means at the
  winning count.

So the flat curve is a subsampling cap, not an algorithmic property, and the
bulk of the cost is model selection rather than the fit. **(c)** is the control
that establishes the first — classical routes given the same cap flatten the
same way. **(d)** is the decomposition that establishes the second.

**(a) Training cost against training rows**, log-log. Model training only;
feature engineering is drawn separately and belongs to neither route.

**(b) Error rate, log scale.** Accuracy is the wrong axis on a saturated
dataset — 0.9997 against 0.9951 reads as a rounding difference and is a
sixteenfold difference in errors. Chapter 4 already warns that PhiUSIIL is
saturated; plotting the error makes the warning legible instead of hiding it.
This is the one panel where the construction's advantage is not explained away
by anything in (c) or (d).
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "09-phishing-scaling"

FLOOR = 1e-5      # log-axis floor for a zero error rate; drawn hollow

# The component-pinned companion run. Panel (d) is a difference between two
# archives, so the label is named rather than inferred: "whatever ran last"
# would silently subtract two unrelated runs.
PINNED = "opt-phishing-pinned-2026-08-03"

CAP = "20k"
KM, FCM = "classical-kmeans", "classical-fcm"
KM_CAP, FCM_CAP = f"classical-kmeans-{CAP}", f"classical-fcm-{CAP}"


def _load(label=None):
    rows, src = H.table("table_phishing_identification_seeds", label)
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["route"]][int(r["n_rows"])].append(
            (float(r["identify_s"]), float(r["accuracy"]),
             int(r["n_mfs"]), float(r["screen_s"])))
    # median across seeds; each seed already reports a median of repeats
    return ({route: {n: tuple(float(np.median([v[i] for v in vals]))
                              for i in range(4))
                     for n, vals in sizes.items()}
             for route, sizes in by.items()}, src)


def _cost(ax, by):
    """(a) — the headline routes, plus the shared preprocessing they all need."""
    series = [("construction", F.BLUE, "-", 1.8, "construction (auto components)"),
              (KM, F.ORANGE, "-", 1.8, "k-means"),
              (FCM, F.AQUA, "-", 1.8, "fuzzy c-means")]
    for route, colour, ls, lw, label in series:
        if route not in by:
            continue
        xs = sorted(by[route])
        ax.plot(xs, [1000 * by[route][n][0] for n in xs], lw=lw, ls=ls,
                marker="o", ms=4.0, color=colour, label=label, zorder=4)

    # Feature engineering, drawn alongside rather than inside any route's line:
    # shared preprocessing, charged to nobody, and at the largest size it costs
    # more than any route's training.
    xs = sorted(by["construction"])
    ax.plot(xs, [1000 * by["construction"][n][3] for n in xs], lw=1.4,
            ls=(0, (3, 2)), color=F.FAINT, zorder=3,
            label="feature engineering\n(shared, not training)")

    ax.set_xscale("log")
    ax.set_yscale("log")
    F.style_axes(ax, title="(a)  model training cost",
                 xlabel="training rows (log)",
                 ylabel="milliseconds, single-threaded (log)")
    F.legend(ax, loc="upper left")


def _error(ax, by):
    """(b) — the same routes as errors, because accuracy saturates here."""
    for route, colour, label in (("construction", F.BLUE, "construction"),
                                 (KM, F.ORANGE, "k-means"),
                                 (FCM, F.AQUA, "fuzzy c-means")):
        if route not in by:
            continue
        xs = sorted(by[route])
        # A route that makes zero test errors has no place on a log axis. Those
        # points are floored so the line stays continuous, and drawn hollow so a
        # floor is never read as a measured rate.
        raw = [1.0 - by[route][n][1] for n in xs]
        err = [max(e, FLOOR) for e in raw]
        ax.plot(xs, err, lw=1.8, color=colour, label=label, zorder=4)
        solid = [(n, e) for n, e, r in zip(xs, err, raw) if r > 0]
        hollow = [(n, e) for n, e, r in zip(xs, err, raw) if r <= 0]
        if solid:
            ax.plot([n for n, _ in solid], [e for _, e in solid], ls="none",
                    marker="o", ms=4.0, color=colour, zorder=5)
        if hollow:
            ax.plot([n for n, _ in hollow], [e for _, e in hollow], ls="none",
                    marker="o", ms=5.0, mfc=F.SURFACE, mec=colour, mew=1.4,
                    zorder=5)
            ax.annotate("no test errors\n(floored to plot)",
                        xy=(hollow[0][0], hollow[0][1]),
                        xytext=(hollow[0][0] * 1.6, hollow[0][1] * 4.0),
                        fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.5,
                        arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS))
    ax.set_xscale("log")
    ax.set_yscale("log")
    F.style_axes(ax, title="(b)  error rate, not accuracy",
                 xlabel="training rows (log)", ylabel="1 − test accuracy (log)")
    F.legend(ax, loc="lower right")


def _cap_control(ax, by):
    """(c) — is the flat curve an algorithm, or is it the 20,000-row cap?"""
    pairs = [(KM, KM_CAP, F.ORANGE, "k-means"),
             (FCM, FCM_CAP, F.AQUA, "fuzzy c-means")]
    drawn = False
    for full, capped, colour, label in pairs:
        if capped not in by:
            continue
        drawn = True
        xs = sorted(by[full])
        ax.plot(xs, [1000 * by[full][n][0] for n in xs], lw=1.8, marker="o",
                ms=4.0, color=colour, label=f"{label}, all rows", zorder=4)
        xs = sorted(by[capped])
        ax.plot(xs, [1000 * by[capped][n][0] for n in xs], lw=1.5,
                ls=(0, (4, 2)), marker="s", ms=3.6, mfc=F.SURFACE,
                color=colour, label=f"{label}, capped at 20k", zorder=4)
    if not drawn:
        ax.text(0.5, 0.5, f"no capped arms in this run\n"
                          f"(re-run with --cap-classical 20000)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.6)
        F.style_axes(ax, title="(c)  the subsampling control")
        return

    xs = sorted(by["construction"])
    ax.plot(xs, [1000 * by["construction"][n][0] for n in xs], lw=1.8,
            marker="o", ms=4.0, color=F.BLUE, label="construction", zorder=4)
    ax.axvline(20_000, lw=1.0, ls=(0, (2, 2)), color=F.FAINT, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # Five series need a five-row legend; headroom above the curves is cheaper
    # than a legend sitting on top of them.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 6.0)
    # After the scale is set: an axes-fraction y keeps the note off the floor
    # whatever the data does.
    ax.text(22_000, 0.04, "cap bites\nabove here",
            transform=ax.get_xaxis_transform(), va="bottom", ha="left",
            fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.4)
    F.style_axes(ax, title="(c)  the same cap, given to the classical routes",
                 xlabel="training rows (log)",
                 ylabel="milliseconds, single-threaded (log)")
    F.legend(ax, loc="upper left", ncol=1)


def _anatomy(ax, by, pinned):
    """(d) — where the construction's milliseconds go."""
    ns = sorted(by["construction"])
    if pinned is None:
        ax.text(0.5, 0.5, f"pinned companion run not found\n({PINNED})",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.6)
        F.style_axes(ax, title="(d)  where the construction's time goes")
        return

    fit = np.array([1000 * pinned["construction"][n][0] for n in ns])
    total = np.array([1000 * by["construction"][n][0] for n in ns])
    # BIC selection is the difference between choosing a component count and
    # being told one. It is a difference of two runs on the same machine, so a
    # small negative is measurement noise, not a negative cost.
    select = np.maximum(total - fit, 0.0)
    screen = np.array([1000 * by["construction"][n][3] for n in ns])

    x = np.arange(len(ns))
    ax.bar(x, fit, width=0.62, color=F.BLUE, label="mixture fit (k-means)",
           zorder=3)
    ax.bar(x, select, width=0.62, bottom=fit, color=F.VIOLET,
           label="BIC component selection", zorder=3)
    ax.bar(x, screen, width=0.62, bottom=fit + select, color=F.FAINT,
           label="feature engineering\n(not training)", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n // 1000}k" for n in ns])
    ax.set_ylim(0, float((fit + select + screen).max()) * 1.35)
    F.style_axes(ax, title="(d)  where the construction's time goes",
                 xlabel="training rows", ylabel="milliseconds",
                 grid_axis="y")
    F.legend(ax, loc="upper left")

    share = 100.0 * select[-1] / (fit[-1] + select[-1])
    ax.annotate(f"{share:.0f}% of full-size training\nis component selection",
                xy=(x[-1] - 0.30, fit[-1] + select[-1] * 0.5),
                xytext=(0.04, 0.56), textcoords="axes fraction",
                fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS,
                                connectionstyle="arc3,rad=-0.15"))


def build():
    by, label = _load()
    if "construction" not in by:
        raise RuntimeError("no data; run run_phishing_study.py")
    try:
        pinned, pinned_label = _load(PINNED)
        # `H.table` falls back to the loose outputs when an archive lacks a
        # table. That fallback is right for a headline figure and wrong here:
        # panel (d) subtracts the pinned run from the automatic one, and the
        # loose files ARE the automatic one, so the fallback would silently
        # subtract a run from itself and draw a zero selection cost.
        if "unarchived" in pinned_label:
            pinned, pinned_label = None, pinned_label
    except FileNotFoundError:
        pinned, pinned_label = None, "(missing)"

    fig, axes = F.grid_figure(2, 2, width=F.W_WIDE + 0.6, height=7.0)
    (ax_a, ax_b), (ax_c, ax_d) = axes

    _cost(ax_a, by)
    _error(ax_b, by)
    _cap_control(ax_c, by)
    _anatomy(ax_d, by, pinned)

    fig.text(0.5, -0.015,
             "Same model shape, same prediction path, same features — only the "
             "placement method differs, and the timing is model training ONLY. "
             "Three seeds, median across seeds of a\nper-seed median of three "
             "repeats, single-threaded. The 20,000-row cap in (c) is the "
             "construction's own: `fit_gaussians` truncates each (feature, "
             "class) column before fitting, so a\nclassical route that reads "
             "every row is not its control. (d) subtracts the component-pinned "
             f"run from the automatic one. {H.provenance_note(label)}"
             + (f" · pinned: {pinned_label}" if pinned is not None else ""),
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
