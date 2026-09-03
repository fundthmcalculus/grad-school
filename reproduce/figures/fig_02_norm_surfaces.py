#!/usr/bin/env python3
"""Figure 2.3 -- the five t-norm / t-conorm families the library ships, as surfaces.

§2.1 defines the t-norm and t-conorm and names the families that matter later;
`table_norm_conorm_matrix.py` sweeps exactly five of them (min/max,
probability, Lukasiewicz, Hamacher, Einstein) and §4.3.5's anomaly rule is
built on the Hamacher conorm. A reader who has not seen these drawn has no
intuition for how they differ, and the one property the anomaly rule leans on
-- S(1, b) = 1 for every conorm -- is a single edge of each surface.

Two rows of five panels: T(a, b) above, S(a, b) below, each family in its own
column, the formulas from `tribblefis.gauss_math.t_norm` / `t_conorm` in the
column titles. The top edge of every conorm panel (a = 1) reads 1 across its
whole width; that edge is what saturates the anomaly aggregate once one class
firing is clipped to 1 (Appendix A.10.10). The formulas are reproduced here
rather than imported so the figure needs no submodule; they match the
library's branches line for line, including the Hamacher conorm's value of 1
at the removable singularity a = b = 1.

Computed from the formulas; nothing here is a measurement.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "02-norm-surfaces"


def _hamacher_s(a, b):
    num = a + b - 2 * a * b
    den = 1 - a * b
    out = np.ones_like(a)
    ok = np.abs(den) > 1e-12
    np.divide(num, den, out=out, where=ok)
    return out


def _hamacher_t(a, b):
    den = a + b - a * b
    out = np.zeros_like(a)
    ok = np.abs(den) > 1e-12
    np.divide(a * b, den, out=out, where=ok)
    return out


FAMILIES = [
    ("min / max", "$\\min(a,b)$", "$\\max(a,b)$", np.minimum, np.maximum),
    (
        "probability",
        "$ab$",
        "$a+b-ab$",
        lambda a, b: a * b,
        lambda a, b: a + b - a * b,
    ),
    (
        "Łukasiewicz",
        "$\\max(0,\\,a+b-1)$",
        "$\\min(1,\\,a+b)$",
        lambda a, b: np.maximum(0.0, a + b - 1),
        lambda a, b: np.minimum(1.0, a + b),
    ),
    (
        "Hamacher",
        "$\\frac{ab}{a+b-ab}$",
        "$\\frac{a+b-2ab}{1-ab}$",
        _hamacher_t,
        _hamacher_s,
    ),
    (
        "Einstein",
        "$\\frac{ab}{2-(a+b-ab)}$",
        "$\\frac{a+b}{1+ab}$",
        lambda a, b: a * b / (2.0 - (a + b - a * b)),
        lambda a, b: (a + b) / (1.0 + a * b),
    ),
]


def build():
    g = np.linspace(0.0, 1.0, 201)
    A, B = np.meshgrid(g, g, indexing="ij")  # A varies down the rows

    fig, axes = F.grid_figure(2, 5, width=F.W_WIDE, height=3.6)
    cmap = F.blue_cmap(reverse=False)
    for col, (name, t_lab, s_lab, T, S) in enumerate(FAMILIES):
        for row, (fn, lab, kind) in enumerate(((T, t_lab, "$T$"), (S, s_lab, "$S$"))):
            ax = axes[row, col]
            Z = fn(A, B)
            ax.imshow(
                Z,
                origin="lower",
                extent=(0, 1, 0, 1),
                cmap=cmap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
                rasterized=True,
            )
            ax.contour(
                B, A, Z, levels=[0.25, 0.5, 0.75], colors=[F.SURFACE], linewidths=0.6
            )
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=F.FS_SMALL - 1, colors=F.INK_2, length=2)
            for s in ax.spines.values():
                s.set_color(F.AXIS)
                s.set_linewidth(0.8)
            ax.set_title(
                f"{kind} = {lab}" if row else f"{name}\n{kind} = {lab}",
                fontsize=F.FS_SMALL,
                color=F.INK,
                pad=4,
            )
            if col == 0:
                ax.set_ylabel(
                    "t-norm (AND)\n$a$" if row == 0 else "t-conorm (OR)\n$a$",
                    fontsize=F.FS_SMALL,
                    color=F.INK_2,
                )
            if row == 1:
                ax.set_xlabel("$b$", fontsize=F.FS_SMALL, color=F.INK_2)
            if row == 1:
                # The edge the anomaly rule lives on: a = 1 -> S = 1 everywhere.
                ax.plot([0, 1], [1, 1], lw=2.2, color=F.ORANGE, zorder=5)

    fig.text(
        0.5,
        -0.02,
        "Dark is 1, light is 0; the white contours are 0.25, 0.5, 0.75. Every "
        "conorm's top edge (orange, $a = 1$) is identically 1: one saturated "
        "input saturates the aggregate, which is\nwhy the anomaly rule of §4.3.5 "
        "is a threshold on the maximum class firing wherever any class clears "
        "$1 - \\theta$ (Appendix A.10.10). The families differ only inside the "
        "square.",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
