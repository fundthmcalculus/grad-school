#!/usr/bin/env python3
"""Figure 6.4 -- the hierarchical mixture: gates over named inputs, TSK leaves.

What the caption asks the figure to emphasise is not the tree shape, which is
unremarkable, but the constraint: **gates split only on original variables.**
That is the Magdalena condition of §6.2, and §6.3.4 makes it structural rather
than aspirational -- the declarative plan can only name original inputs, so no
synthetic intermediate can enter a gate even by accident. The figure states the
condition where the gates are drawn, because that is the claim being made about
them.

The second thing it has to show is that the routing is *soft*. A point does not
land in one leaf; it reaches every leaf with a weight that is the product of the
gate memberships along the path, and the model's output is the weighted average.
Drawing crisp branches would make this look like a decision tree with extra
steps.

Schematic. The variable names are Concrete's, matching the tree §6.3.2 actually
recovers (cement, then age at the 28-day mark), so the illustration is not at
odds with the fitted model of Figure 6.2 -- but the structure here is drawn, not
fitted, and no number in it is measured.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "06-hme-structure"


def build():
    fig, ax = F.canvas(width=F.W_WIDE, height=4.8, xlim=(0, 100), ylim=(0, 100))

    ROOT_Y, MID_Y, LEAF_Y = 88, 62, 30

    F.box(
        ax,
        50,
        ROOT_Y,
        30,
        12,
        "Gate:  Cement",
        "fuzzy partition of unity\nover one named input",
        color=F.BLUE,
        title_size=F.FS_ANNOT,
        body_size=F.FS_SMALL - 0.5,
    )

    mids = [(26, "Gate:  Age"), (74, "Gate:  Water")]
    for x, title in mids:
        F.box(
            ax,
            x,
            MID_Y,
            30,
            12,
            title,
            "fuzzy partition of unity\nover one named input",
            color=F.BLUE,
            title_size=F.FS_ANNOT,
            body_size=F.FS_SMALL - 0.5,
        )

    leaves = [13, 39, 61, 87]
    for i, x in enumerate(leaves, start=1):
        F.box(
            ax,
            x,
            LEAF_Y,
            22,
            13,
            f"TSK expert {i}",
            "ridge least squares on\nthe points that reach it",
            color=F.ORANGE,
            title_size=F.FS_ANNOT,
            body_size=F.FS_SMALL - 0.5,
        )

    # Soft edges. The membership labels are the point of the figure: a branch
    # carries a weight in [0, 1], and both branches carry weight at once.
    edges = [
        ((50, ROOT_Y - 6), (26, MID_Y + 6), "$\\mu_1$", (-3.5, 1)),
        ((50, ROOT_Y - 6), (74, MID_Y + 6), "$1-\\mu_1$", (3.5, 1)),
        ((26, MID_Y - 6), (13, LEAF_Y + 6.5), "$\\mu_2$", (-3.2, 0)),
        ((26, MID_Y - 6), (39, LEAF_Y + 6.5), "$1-\\mu_2$", (3.6, 0)),
        ((74, MID_Y - 6), (61, LEAF_Y + 6.5), "$\\mu_3$", (-3.2, 0)),
        ((74, MID_Y - 6), (87, LEAF_Y + 6.5), "$1-\\mu_3$", (3.6, 0)),
    ]
    for start, end, label, offset in edges:
        F.arrow(
            ax,
            start,
            end,
            color=F.AXIS,
            lw=1.2,
            label=label,
            label_offset=offset,
            label_size=F.FS_SMALL,
            label_color=F.INK_2,
        )

    # The aggregation, drawn as a bar under the leaves rather than a seventh box:
    # it is what happens to all four outputs at once, not a step after any one.
    ax.plot([2, 98], [17, 17], lw=0.8, color=F.AXIS)
    for x in leaves:
        F.arrow(ax, (x, LEAF_Y - 6.5), (x, 18), color=F.GRID, lw=1.0)
    ax.text(
        50,
        12,
        "Output = $\\sum_\\ell w_\\ell(x)\\, f_\\ell(x)$   with   "
        "$w_\\ell(x) = \\prod_{\\mathrm{path}} \\mu$   —   every point "
        "reaches every expert, to a degree",
        ha="center",
        va="center",
        fontsize=F.FS_ANNOT,
        color=F.INK_2,
    )

    ax.text(
        50,
        4,
        "Gates name only original inputs — no synthetic intermediate can "
        "appear in one, because the declarative plan\n(§6.3.4) cannot express "
        "it. That is how the Magdalena condition of §6.2 is enforced rather "
        "than merely respected.",
        ha="center",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
        style="italic",
    )

    ax.set_ylim(-1, 96)
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
