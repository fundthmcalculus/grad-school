#!/usr/bin/env python3
"""PhiUSIIL: how the identification routes scale, and where the gap goes.

Concrete said the classical route identifies a rule base 25-84x cheaper. That
was 824 rows. The two routes scale in different variables, so the interesting
question was never "which is faster" but "does that gap survive size". Here it
does not — it closes steadily — and the accuracy ordering is the reverse of
Concrete's.

**(a) Identification cost against training rows**, log-log. Different slopes are
the whole point: the construction's own placement step grows sublinearly in n
while a clustering grows about linearly, so the curves converge.

**(b) Error rate, log scale.** Accuracy is the wrong axis on a saturated
dataset — 0.9998 against 0.9960 reads as a rounding difference and is a
twentyfold difference in errors. Chapter 4 already warns that PhiUSIIL is
saturated; plotting the error makes the warning legible instead of hiding it.

**(c) The cost ratio against size.** What (a) implies, read directly: how many
times more expensive the construction is at each size, and how fast that
multiple is falling.
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

ROUTES = ["construction", "classical-kmeans", "classical-fcm"]
COLOUR = {"construction": F.BLUE, "classical-kmeans": F.ORANGE,
          "classical-fcm": F.AQUA}


def _load():
    rows, label = H.table("table_phishing_identification_seeds")
    by = defaultdict(dict)
    for r in rows:
        by[r["route"]][int(r["n_rows"])] = (
            float(r["identify_s"]), float(r["accuracy"]),
            int(r["n_mfs"]), float(r["screen_s"]))
    return by, label


def build():
    by, label = _load()
    present = [r for r in ROUTES if r in by]
    if not present:
        raise RuntimeError("no data; run run_phishing_study.py")
    ns = sorted(by[present[0]])

    fig, (tx, ex, rx) = F.grid_figure(1, 3, width=F.W_WIDE + 1.4, height=3.6)

    for route in present:
        xs = sorted(by[route])
        tx.plot(xs, [1000 * by[route][n][0] for n in xs], lw=1.8, marker="o",
                ms=4.5, color=COLOUR[route], label=route, zorder=4)
        # A route that makes zero test errors has no place on a log axis. Those
        # points are floored so the line stays continuous, and drawn hollow so a
        # floor is never read as a measured rate -- at the smallest size the
        # construction simply got all 800 test rows right.
        raw = [1.0 - by[route][n][1] for n in xs]
        err = [max(e, FLOOR) for e in raw]
        ex.plot(xs, err, lw=1.8, color=COLOUR[route], zorder=4)
        solid = [(n, e) for n, e, r in zip(xs, err, raw) if r > 0]
        hollow = [(n, e) for n, e, r in zip(xs, err, raw) if r <= 0]
        if solid:
            ex.plot([n for n, _ in solid], [e for _, e in solid], ls="none",
                    marker="o", ms=4.5, color=COLOUR[route], zorder=5)
        if hollow:
            ex.plot([n for n, _ in hollow], [e for _, e in hollow], ls="none",
                    marker="o", ms=5.5, mfc=F.SURFACE, mec=COLOUR[route],
                    mew=1.4, zorder=5)
            ex.annotate("no test errors\n(floored to plot)",
                        xy=(hollow[0][0], hollow[0][1]),
                        xytext=(hollow[0][0] * 1.5, hollow[0][1] * 3.2),
                        fontsize=F.FS_SMALL, color=F.MUTED, linespacing=1.5,
                        arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS))

    # The construction's screening, drawn inside its own bar of time: at the
    # largest size it is more than half the total, which is why the slope in (a)
    # is what it is.
    xs = sorted(by["construction"])
    tx.plot(xs, [1000 * by["construction"][n][3] for n in xs], lw=1.2,
            ls=(0, (3, 2)), color=F.tint(F.BLUE, 0.45), zorder=3,
            label="…of which feature screening")

    tx.set_xscale("log")
    tx.set_yscale("log")
    F.style_axes(tx, title="(a)  identification cost",
                 xlabel="training rows (log)",
                 ylabel="milliseconds, single-threaded (log)")
    F.legend(tx, loc="upper left")

    ex.set_xscale("log")
    ex.set_yscale("log")
    F.style_axes(ex, title="(b)  error rate, not accuracy",
                 xlabel="training rows (log)", ylabel="1 − test accuracy (log)")

    ratio_km = [by["construction"][n][0] / by["classical-kmeans"][n][0] for n in ns]
    ratio_fcm = [by["construction"][n][0] / by["classical-fcm"][n][0] for n in ns]
    rx.plot(ns, ratio_km, lw=1.8, marker="o", ms=4.5, color=F.ORANGE,
            label="vs k-means", zorder=4)
    rx.plot(ns, ratio_fcm, lw=1.8, marker="o", ms=4.5, color=F.AQUA,
            label="vs FCM", zorder=4)
    rx.axhline(1.0, lw=1.1, ls=(0, (3, 2)), color=F.FAINT, zorder=2)
    rx.text(ns[0], 1.0, " parity", va="bottom", ha="left", fontsize=F.FS_SMALL,
            color=F.MUTED)
    rx.set_xscale("log")
    rx.set_yscale("log")
    F.style_axes(rx, title="(c)  how much dearer the\nconstruction is",
                 xlabel="training rows (log)", ylabel="cost ratio (log)")
    F.legend(rx, loc="upper right")

    fig.text(0.5, -0.02,
             "Same model shape, same prediction path, same retained features — only "
             "the placement method differs. The construction is additionally charged "
             "for its own feature\nscreening, which the classical routes are handed "
             "free; at the largest size that screening is more than half its total, "
             "and the comparison is a handicap in the classical\nroute's favour. One "
             "seed, so read the trends and not the individual points. Timing "
             f"single-threaded, median of repeats. {H.provenance_note(label)}",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
