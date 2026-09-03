#!/usr/bin/env python3
"""Figure 4.3 -- rule count against feature count: the grid's exponential, the
construction's constant.

§4.3.4 states two formulas: a grid-partitioned rule base has prod_j N_mu_j rules,
exponential in the feature count M, and the answer-first construction has K --
one per class or output bucket, independent of M. Chapter 1 prints the two
numbers for PhiUSIIL; this figure draws the two curves and places every dataset
the document models on them, so the gap is visible as a vertical distance on a
log axis rather than as a pair of numerals.

The feature and class counts come from `reproduce/dataset_specs.yaml`, the same
source `build_pdf.py` substitutes into the prose, so the markers cannot drift
from the chapters. Concrete is a regression problem and its K is the bucket
count §4.3.2 settles on rather than a class count; the marker says so. The grid
curves are drawn at two and three sets per input, the smallest partitions
anyone uses; the real figure for a tuned grid is larger still.

Arithmetic and specifications only; no measurement appears here.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPRO)

NAME = "04-rule-count"

# (spec key, printed name, K override for regression problems, label offset in
# points, horizontal alignment). The offsets stagger the small-M labels so they
# do not sit on top of one another.
DATASETS = (
    ("glass", "Glass", None, (10, 26), "left"),
    ("wine", "Wine", None, (10, 4), "left"),
    ("digits", "Digits", None, (0, 16), "center"),
    ("phiusiil", "PhiUSIIL", None, (8, 3), "left"),
    ("rt_iot2022", "RT-IOT2022", None, (8, 4), "left"),
    ("concrete", "Concrete (3 output buckets)", 3, (10, 48), "left"),
)


def build():
    import dataset_specs as DS

    specs = DS.load_specs()
    M = np.arange(1, 100)

    fig, ax = F.figure(width=F.W_COL + 1.0, height=4.0)
    for c, colour in ((2, F.tint(F.ORANGE, 0.35)), (3, F.ORANGE)):
        ax.plot(
            M,
            float(c) ** M,
            lw=1.6,
            color=colour,
            label=f"grid: $c^M$ rules at $c$ = {c} sets per input",
            zorder=3,
        )

    xs, ys = [], []
    for key, name, k_override, (dx, dy), ha in DATASETS:
        spec = specs[key]
        m = spec["features"]
        k = k_override if k_override is not None else spec["classes"]
        xs.append(m)
        ys.append(k)
        ax.annotate(
            f"{name}: $M$ = {m}, $K$ = {k}",
            xy=(m, k),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=F.FS_SMALL,
            color=F.shade(F.BLUE, 0.25),
            ha=ha,
            va="bottom",
            arrowprops=(
                dict(arrowstyle="-", lw=0.6, color=F.AXIS) if abs(dy) > 10 else None
            ),
        )
        grid = 3.0**m
        ax.plot([m, m], [k, grid], lw=0.7, ls=(0, (2, 2)), color=F.FAINT, zorder=2)
    ax.scatter(
        xs,
        ys,
        s=40,
        color=F.BLUE,
        edgecolor=F.SURFACE,
        linewidths=0.8,
        zorder=5,
        label="this construction: $K$ rules, one per answer",
    )

    F.style_axes(
        ax,
        title="Rules in the base against features in the data",
        xlabel="$M$, features reaching the model",
        ylabel="rules (log scale)",
    )
    ax.set_yscale("log")
    ax.set_xlim(0, 100)
    ax.set_ylim(1, 1e48)
    F.legend(ax, loc="upper left")
    ax.text(
        0.0,
        -0.2,
        "Feature and class counts from `reproduce/dataset_specs.yaml`, the file the prose is built from. The dotted "
        "drop from each\ngrid curve to its marker is the rule-base explosion the answer-first factorization "
        "sidesteps (§4.3.4, Appendix A.10.9); the\nparameter count grows as $2KMp$, linear in $M$, and the only "
        "quadratic term in the fit is the $MK(K-1)/2$ antecedent screen.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
