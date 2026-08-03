#!/usr/bin/env python3
"""PhiUSIIL: how the identification routes scale, and where the cost actually is.

Concrete said the classical route identifies a rule base 25-84x cheaper. That
was 824 rows. The two routes scale in different variables, so the interesting
question was never "which is faster" but "does that gap survive size". It does
not — and reading the timings against the library source says why, which is not
what the first version of this figure implied.

Two library defects were doing most of the talking, and both have since been
fixed in `fit_gaussians`:

* it **truncated each (feature, class) column to its first 20,000 rows** before
  fitting anything, so above that the cost stopped growing — a subsampling cap
  read as sublinear scaling;
* with an automatic component count it fitted **four EM mixtures per (feature,
  class)** to choose one by BIC, discarded them, and ran k-means at the winning
  count — 82% of training time spent choosing rather than placing.

Every route in **(a)** now reads every row. **(c)** is the before-and-after on
those two defects, and **(d)** is what is left of the selection cost.

**(a) Training cost against training rows**, log-log. Model training only;
feature engineering is drawn separately and belongs to neither route. This is
now a matched comparison in both directions — same parameter budget is (d)'s
business, but every route here sees the whole training set.

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
PINNED = "opt-phishing-kmbic-pinned-2026-08-03"
CAPPED = "opt-phishing-kmbic-capped-2026-08-03"    # after the fix, old 20k cap
BEFORE = "opt-phishing-2026-08-03"                 # before the fix, 20k cap

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


def _before_after(ax, by, capped, before):
    """(c) — what the two library fixes were worth, at every size."""
    drawn = False
    if before is not None:
        xs = sorted(before["construction"])
        ax.plot(xs, [1000 * before["construction"][n][0] for n in xs], lw=1.5,
                ls=(0, (1, 2)), marker="v", ms=4.0, color=F.FAINT, zorder=3,
                label="before the fix\n(EM select, 20k cap)")
        drawn = True
    if capped is not None:
        xs = sorted(capped["construction"])
        ax.plot(xs, [1000 * capped["construction"][n][0] for n in xs], lw=1.5,
                ls=(0, (4, 2)), marker="s", ms=3.6, mfc=F.SURFACE,
                color=F.VIOLET, zorder=4,
                label="k-means select,\nsame 20k cap")
        drawn = True
    xs = sorted(by["construction"])
    ax.plot(xs, [1000 * by["construction"][n][0] for n in xs], lw=1.8,
            marker="o", ms=4.0, color=F.BLUE, zorder=5,
            label="shipped: k-means select,\nno cap, every row")
    # The reference the construction is trying to reach. Without it the panel
    # reads as "smaller is better" with no scale for how much smaller matters.
    if KM in by:
        xs = sorted(by[KM])
        ax.plot(xs, [1000 * by[KM][n][0] for n in xs], lw=1.4, ls="-",
                marker="o", ms=3.4, color=F.ORANGE, zorder=3,
                label="k-means clustering,\nfor scale")

    ax.set_xscale("log")
    ax.set_yscale("log")
    if drawn:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 5.0)
    F.style_axes(ax, title="(c)  the two library fixes, priced",
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
    ax.annotate(f"{share:.0f}% of full-size training\nis component selection\n(was 82% before the fix)",
                xy=(x[-1] - 0.30, fit[-1] + select[-1] * 0.5),
                xytext=(0.04, 0.56), textcoords="axes fraction",
                fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.5,
                arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS,
                                connectionstyle="arc3,rad=-0.15"))


def build():
    by, label = _load()
    if "construction" not in by:
        raise RuntimeError("no data; run run_phishing_study.py")
    capped = before = None
    for companion, target in ((CAPPED, "capped"), (BEFORE, "before")):
        try:
            data, src = _load(companion)
            if "unarchived" not in src:
                if target == "capped":
                    capped = data
                else:
                    before = data
        except FileNotFoundError:
            pass

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
    _before_after(ax_c, by, capped, before)
    _anatomy(ax_d, by, pinned)

    fig.text(0.5, -0.015,
             "Same model shape, same prediction path, same features, and every "
             "route reads every row — what differs is the placement method, and "
             "the timing is model training ONLY.\nThree seeds, median across "
             "seeds of a per-seed median of three repeats, single-threaded. The "
             "20,000-row cap in (c) was the construction's own: `fit_gaussians` "
             "truncated each\n(feature, class) column before fitting anything, "
             "which is why its curve used to go flat. Both that and the "
             "discarded EM model selection are fixed; (c) prices them and (d) "
             f"shows\nwhat is left. (d) subtracts the component-pinned run from "
             f"the automatic one. {H.provenance_note(label)}"
             + (f" · pinned: {pinned_label}" if pinned is not None else ""),
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
