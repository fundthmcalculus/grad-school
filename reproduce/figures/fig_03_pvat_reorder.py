#!/usr/bin/env python3
"""Figure 3.1 -- the mergeVAT reorder against the classical linear scan.

§3.3.1 tells a three-stage story and the figure has to carry all three, because
the chapter's honesty about the second stage is part of the contribution:

  arm 0  classical    O(N^3)      re-scan every remaining pair each step
  arm 1  stage one    O(N^2 logN) priority queue with lazy deletion (published)
  arm 2  stage two    O(N^2)      compact active set, relax and select fused
                                  (unpublished; §9.3 scopes what it may claim)

The inner loop is what differs, so the inner loop is what the figure draws: one
row per arm, with the shared scaffolding (seed at the farthest pair, append the
argmin, repeat N-1 times) drawn once. Drawing three near-identical flowcharts
would bury the one box in each that is actually different.

The measured consequence of the difference is Figure 3.4's business, not this
one's -- so this is a schematic, and it carries the complexity class of each arm
rather than a timing. The caption in §3.3.1 asks for an adaptation of
`presentations/quals/slides/img/vat_prim_mst_block_diagram_v2.svg`; that file is
an export with its text baked into masks and no adaptable source, so the diagram
is redrawn here rather than traced.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "03-pvat-reorder"


def build():
    fig, ax = F.canvas(width=F.W_WIDE, height=4.8, xlim=(0, 100), ylim=(0, 100))

    # -- shared scaffolding, drawn once --------------------------------------
    ax.text(
        2,
        95,
        "Every arm shares the same outer loop",
        ha="left",
        va="center",
        fontsize=F.FS_LABEL,
        color=F.INK,
        fontweight="bold",
    )
    outer = [
        (17, "Seed", "the farther endpoint of\nthe most-dissimilar pair"),
        (
            50,
            "Repeat $N-1$ times",
            "append the nearest unplaced point\n" "to the ordered set",
        ),
        (
            84,
            "VAT ordering",
            "identical across all three arms,\n" "checked bit for bit",
        ),
    ]
    for x, title, body in outer:
        F.box(
            ax,
            x,
            82,
            30,
            14,
            title,
            body,
            color=F.FAINT,
            fill_amount=0.93,
            title_size=F.FS_ANNOT,
            body_size=F.FS_SMALL - 0.5,
        )
    F.arrow(ax, (32, 82), (35, 82))
    F.arrow(ax, (65, 82), (69, 82))

    # -- the three inner loops ------------------------------------------------
    F.arrow(ax, (50, 75), (50, 71))
    ax.text(
        50,
        67.5,
        "What differs is the step inside it — how the next minimum is found",
        ha="center",
        va="center",
        fontsize=F.FS_LABEL,
        color=F.INK,
        fontweight="bold",
    )

    arms = [
        (
            54,
            F.ORANGE,
            "classical",
            "$O(N^3)$",
            "Re-scan every\n(placed, unplaced) pair",
            "$N-1$ passes over the whole remaining matrix. The minimum is\n"
            "recomputed from scratch at each step, having just been computed.",
        ),
        (
            36,
            F.BLUE,
            "stage one — published",
            "$O(N^2\\log N)$",
            "Priority queue with\nlazy deletion",
            "Each relaxation pushes a key; the pop returns the minimum. Stale\n"
            "entries are discarded on the way out rather than removed in place.",
        ),
        (
            18,
            F.AQUA,
            "stage two — unpublished",
            "$O(N^2)$",
            "Compact active set;\nrelax and select fused",
            "The reorder only ever needs the CURRENT minimum, so the heap is\n"
            "unnecessary: one pass relaxes and selects, and the log factor goes.",
        ),
    ]

    for y, color, name, order, headline, detail in arms:
        F.box(
            ax,
            18,
            y,
            30,
            13,
            headline,
            None,
            color=color,
            fill_amount=0.88,
            title_size=F.FS_SMALL,
        )
        ax.text(
            35,
            y,
            order,
            ha="left",
            va="center",
            fontsize=F.FS_LABEL,
            color=F.shade(color, 0.25),
            fontweight="bold",
        )
        ax.text(
            50,
            y + 2.6,
            name,
            ha="left",
            va="center",
            fontsize=F.FS_SMALL,
            color=F.shade(color, 0.25),
        )
        ax.text(
            50,
            y - 2.4,
            detail,
            ha="left",
            va="center",
            fontsize=F.FS_SMALL - 0.5,
            color=F.MUTED,
            linespacing=1.5,
        )

    ax.plot([2, 98], [8, 8], lw=0.8, color=F.GRID)
    ax.text(
        2,
        3,
        "Both fast arms ship, deliberately: stage one is the portable "
        "Python/Numba path, stage two the compiled Cython kernel. They "
        "produce bit-identical\norderings, which is how each validates the "
        "other (§3.3.1).",
        ha="left",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )

    ax.set_ylim(0, 100)
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
