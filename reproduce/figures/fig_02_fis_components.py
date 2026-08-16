#!/usr/bin/env python3
"""Figure 2.1 -- the anatomy of a FIS, with this work's contribution shaded in.

The standard four-stage diagram (fuzzify, fire the rules, aggregate, defuzzify)
is in every fuzzy-systems textbook, and redrawing it would add nothing. What the
caption asks for is the annotation: which pieces this work *generates from the
data* and which are *fixed by design*.

That split is the honest content of the figure, and it is deliberately
unflattering in one place. §2.1's analyzability constraints -- triangular
memberships forming a Ruspini partition, the product t-norm, weighted-average
defuzzification -- are chosen a priori, not learned, and the figure says so in
the same visual language it uses for the generated parts. A diagram that shaded
the whole pipeline as "ours" would be claiming the constraint set as a result.

Purely schematic: no measurement appears here, so there is no provenance stamp.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "02-fis-components"


def build():
    fig, ax = F.canvas(width=F.W_WIDE, height=4.4, xlim=(0, 100), ylim=(0, 100))

    Y, HBOX = 66, 26
    W, GAP = 19, 3
    xs = [7.5 + W / 2 + i * (W + GAP) for i in range(4)]

    # generated == recovered from the data by this work; fixed == chosen a priori
    stages = [
        (
            "Fuzzification",
            "membership functions\nover each input",
            True,
            "Ch. 4, Ch. 5",
        ),
        ("Rule base", "antecedents and\ntheir disjunctions", True, "Ch. 4, Ch. 6"),
        ("Inference and\naggregation", "t-norm, t-conorm\ncomplement", False, "§2.1"),
        ("Defuzzification", "weighted average\nover consequents", False, "§2.1"),
    ]

    for x, (title, body, generated, tag) in zip(xs, stages):
        color = F.BLUE if generated else F.FAINT
        F.box(
            ax,
            x,
            Y,
            W,
            HBOX,
            title,
            body,
            color=color,
            title_size=F.FS_ANNOT,
            body_size=F.FS_SMALL - 0.5,
            fill_amount=0.86 if generated else 0.93,
        )
        ax.text(
            x,
            Y - HBOX / 2 - 4,
            tag,
            ha="center",
            va="center",
            fontsize=F.FS_SMALL,
            color=F.MUTED,
        )

    for a, b in zip(xs, xs[1:]):
        F.arrow(ax, (a + W / 2, Y), (b - W / 2, Y))

    # The crisp ends. Drawn as bare labels rather than boxes: they are the
    # signal entering and leaving, not components of the system.
    ax.text(
        3.5,
        Y,
        "crisp\ninputs",
        ha="center",
        va="center",
        fontsize=F.FS_ANNOT,
        color=F.INK_2,
        linespacing=1.4,
    )
    F.arrow(ax, (3.5, Y - 8), (xs[0] - W / 2 - 0.5, Y), connection="arc3,rad=-0.25")
    ax.text(
        97,
        Y,
        "crisp\noutput",
        ha="center",
        va="center",
        fontsize=F.FS_ANNOT,
        color=F.INK_2,
        linespacing=1.4,
    )
    F.arrow(ax, (xs[-1] + W / 2, Y), (97, Y - 8), connection="arc3,rad=0.25")

    # The consequents are solved for, but they are not a *stage* -- they hang off
    # the last two boxes, which is where the closed-form solve of §2.1 lands.
    F.box(
        ax,
        (xs[2] + xs[3]) / 2,
        27,
        2 * W + GAP,
        14,
        "Consequents: closed form, not searched",
        "TSK output is linear in the coefficients\nfor fixed firing strengths (§2.1)",
        color=F.BLUE,
        title_size=F.FS_ANNOT,
        fill_amount=0.86,
    )
    for x in (xs[2], xs[3]):
        F.arrow(ax, (x, Y - HBOX / 2 - 8), (x, 34.5), color=F.GRID, lw=1.0)

    # -- legend: two fills, named --------------------------------------------
    for i, (color, text, fill) in enumerate(
        [
            (F.BLUE, "generated from the data by this work", 0.86),
            (F.FAINT, "fixed by design — the §2.1 constraints", 0.93),
        ]
    ):
        y = 11 - i * 7
        F.box(ax, 12, y, 6, 4.2, "", None, color=color, fill_amount=fill, radius=0.9)
        ax.text(17, y, text, ha="left", va="center", fontsize=F.FS_SMALL, color=F.MUTED)

    ax.set_ylim(0, 88)
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
