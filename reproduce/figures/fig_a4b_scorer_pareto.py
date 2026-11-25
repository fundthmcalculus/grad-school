#!/usr/bin/env python3
"""Figure A.4b -- what each feature scorer costs: the runtime / accuracy front.

Reads Table A.2c (`table_a2c_scorer_pareto`, `table_a2c_pooled_pareto`):
the eight library scorers at every feature budget, ten seeds, on PhiUSIIL,
Breast Cancer and Wine, timed in one process.

  (a) every (dataset, scorer, k) configuration, runtime normalized by its
      dataset's mean so the three datasets share one axis; each dataset's
      front is drawn as a step line and its front points are coloured by
      scorer. Marker shape and dash pattern carry the dataset.
  (b) the pooled view: per (scorer, k) on the common budget grid, accuracy
      averaged over the datasets and runtime as the geometric mean of the
      normalized runtimes, with its front.

Off-front points are drawn small and grey: they are context, not series. The
front marks are coloured by scorer in the same fixed slots as Figure A.4, so a
scorer is the same colour in both figures. Seed spread is not drawn; the table's
CSV carries it, and on the small datasets most neighbouring front points lie
within each other's spread.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402
from fig_a4_scorer_consistency import SCORERS  # noqa: E402

NAME = "a4b-scorer-pareto"
ARCHIVE = "a2c-pareto-dd72d5e-2026-11-25"

COLOUR = {key: colour for key, _, colour, _ in SCORERS}
LABEL = {key: name for key, name, _, _ in SCORERS}
SHORT = {"ks": "KS", "mutual_info": "MI", "jensen_shannon": "JS"}
DATASETS = [  # (name, marker, dash)
    ("PhiUSIIL", "o", "-"),
    ("Breast Cancer", "s", "--"),
    ("Wine", "^", ":"),
]


def _step(ax, pts, dash):
    """Staircase through front points sorted by runtime (attainment boundary)."""
    pts = sorted(pts)
    xs, ys = [], []
    for i, (x, y) in enumerate(pts):
        if i:
            xs.append(x)
            ys.append(pts[i - 1][1])
        xs.append(x)
        ys.append(y)
    ax.plot(xs, ys, color=F.INK_2, lw=1.0, ls=dash, zorder=2)


def _front_marks(ax, pts, marker):
    for x, y, scorer in pts:
        ax.plot(
            x,
            y,
            marker=marker,
            ls="none",
            markersize=6,
            color=COLOUR[scorer],
            markeredgecolor=F.SURFACE,
            markeredgewidth=0.9,
            zorder=4,
        )


def build():
    per, label = H.table("table_a2c_scorer_pareto", ARCHIVE)
    pooled, _ = H.table("table_a2c_pooled_pareto", ARCHIVE)
    fig, (ax_a, ax_b) = F.grid_figure(1, 2, width=F.W_WIDE, height=3.7)
    on_front = set()

    for ds, marker, dash in DATASETS:
        rows = [r for r in per if r["Dataset"] == ds]
        if not rows:
            continue
        x = [float(r["Runtime / dataset mean"]) for r in rows]
        y = [float(r["Accuracy (mean)"]) for r in rows]
        ax_a.plot(x, y, marker=marker, ls="none", markersize=3, color=F.FAINT, zorder=1)
        front = [
            (
                float(r["Runtime / dataset mean"]),
                float(r["Accuracy (mean)"]),
                r["Scorer"],
            )
            for r in rows
            if r["On front"] == "yes"
        ]
        on_front |= {s for _, _, s in front}
        _step(ax_a, [(a, b) for a, b, _ in front], dash)
        _front_marks(ax_a, front, marker)

    gcol = "Runtime / dataset mean (geometric mean)"
    acol = "Accuracy (mean over datasets)"
    ax_b.plot(
        [float(r[gcol]) for r in pooled],
        [float(r[acol]) for r in pooled],
        marker="D",
        ls="none",
        markersize=3,
        color=F.FAINT,
        zorder=1,
    )
    pfront = [
        (float(r[gcol]), float(r[acol]), r["Scorer"], r["Features kept"])
        for r in pooled
        if r["On front"] == "yes"
    ]
    on_front |= {s for _, _, s, _ in pfront}
    _step(ax_b, [(a, b) for a, b, _, _ in pfront], "-")
    _front_marks(ax_b, [(a, b, s) for a, b, s, _ in pfront], "D")
    # Selective labels: a point too close to the last labelled one (the three
    # one-feature configurations nearly coincide) is left unlabelled.
    last = None
    for a, b, s, k in sorted(pfront):
        if last and abs(math.log(a / last[0])) < 0.15 and abs(b - last[1]) < 0.02:
            continue
        ax_b.annotate(
            f"{SHORT.get(s, LABEL[s])} k={k}",
            (a, b),
            xytext=(5, -10),
            textcoords="offset points",
            fontsize=F.FS_SMALL,
            color=F.MUTED,
        )
        last = (a, b)

    F.style_axes(
        ax_a,
        title="(a) Each dataset's front",
        xlabel="fit time / dataset mean (log)",
        ylabel="accuracy, mean of ten seeds",
    )
    F.style_axes(
        ax_b,
        title="(b) Pooled over the three datasets",
        xlabel="fit time / dataset mean, geometric mean (log)",
        ylabel="accuracy, mean of the three datasets",
    )
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

    ticks = [0.1, 0.25, 0.5, 1, 2, 5, 10]
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.axvline(1.0, color=F.AXIS, lw=0.8, ls="--", zorder=0)
    ax_b.set_xlim(0.18, 4.5)

    # Legend: scorers that reach a front (colour), then datasets (shape + dash).
    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            color=COLOUR[k],
            markeredgecolor=F.SURFACE,
            markersize=6,
            label=LABEL[k],
        )
        for k, _, _, _ in SCORERS
        if k in on_front
    ]
    ax_a.legend(
        handles=[
            Line2D(
                [], [], marker=m, ls=d, color=F.INK_2, markersize=5, lw=1.0, label=ds
            )
            for ds, m, d in DATASETS
        ],
        loc="lower right",
        fontsize=F.FS_SMALL,
        frameon=False,
        labelcolor=F.INK_2,
        handlelength=2.6,
        title="front of",
        title_fontsize=F.FS_SMALL,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=F.FS_ANNOT,
        frameon=False,
        labelcolor=F.INK_2,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=2.4,
        columnspacing=1.4,
    )
    fig.text(
        0.5,
        -0.035,
        "Table A.2c: eight library scorers x every feature budget, ten 80/20 splits, "
        "fit time includes scoring. Grey: off the front. 1.0 = the dataset's\n"
        "mean fit time over every scorer at k = 1-10. Seed spread not drawn. "
        f"{H.provenance_note(label)}",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
