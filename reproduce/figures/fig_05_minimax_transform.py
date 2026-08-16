#!/usr/bin/env python3
"""Figure 5.1 -- concentric rings, before and after the minimax transform.

§5.3.1 opens by giving credit away: the single most important step in Chapter 5
is not the selection machinery, it is the minimax transform, and relational FCM
goes from an adjusted Rand index of about 0.02 on the raw dissimilarity matrix
to 1.00 on the transformed one. This figure is that sentence.

The matrices are computed here, by the chapter's own code -- `battery.concentric_rings`
for the data and `ivat_mf.minimax_transform` for the transform. The two ARI
numbers are *not* computed here: they are read from `table_5_1_battery.csv` in
the archive of record, which is where Table 5.1 gets them. A figure that
re-scored the same experiment would be a second measurement of the same thing,
with a second chance to disagree.

Both panels keep the ground-truth ordering (inner ring first, then outer). That
is the fair arrangement for this comparison: the claim is about what relational
FCM can do with the numbers, not about what a reordering can do for the eye, and
VAT-ordering the raw panel would be answering a different question.

One thing the figure shows that the prose's "indistinguishable" undersells, so
the caption states it instead: under the raw D the INNER ring does read as a
block. It is the outer ring that does not, and the reason is measurable -- its
own within-ring distances exceed many inner-to-outer distances, which is exactly
the configuration no Euclidean prototype can resolve. Those two quantities are
computed and printed rather than described.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "05-minimax-transform"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gated-minimax-selection"))


def build():
    import battery as B  # noqa: E402  -- needs the path insert above
    import ivat_mf as im  # noqa: E402

    X, y = B.concentric_rings()
    D = im.dissimilarity(X)
    Dstar = im.minimax_transform(D)

    rows, label = H.table("table_5_1_battery")
    rings = next(r for r in rows if r["Dataset"] == "concentric_rings")
    ari_raw = H.number(rings["NERFCM on raw D"])
    ari_star = H.number(rings["NERFCM on D* (given k)"])

    fig, axes = F.grid_figure(1, 3, width=F.W_WIDE, height=2.5)
    scatter, raw, star = axes

    for k, color in enumerate((F.BLUE, F.ORANGE)):
        pts = X[y == k]
        scatter.scatter(pts[:, 0], pts[:, 1], s=6, linewidths=0, color=color, zorder=3)
    scatter.set_aspect("equal")
    scatter.set_xticks([])
    scatter.set_yticks([])
    for s in scatter.spines.values():
        s.set_color(F.AXIS)
        s.set_linewidth(0.8)
    scatter.set_title(
        f"(a)  two rings, $n$ = {len(X)}", fontsize=F.FS_LABEL, color=F.INK, pad=6
    )

    F.imshow_matrix(raw, D, title="(b)  raw dissimilarity $D$")
    F.imshow_matrix(star, Dstar, title="(c)  minimax transform $D^*$")

    for ax, ari, text in (
        (raw, ari_raw, "relational FCM"),
        (star, ari_star, "relational FCM"),
    ):
        ax.text(
            0.5,
            -0.09,
            f"{text}   ARI = {ari:.2f}",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=F.FS_ANNOT,
            color=F.shade(F.BLUE if ari > 0.5 else F.ORANGE, 0.25),
            fontweight="bold",
        )

    # The measurable reason the raw panel fails, rather than an assertion about it.
    outer = y == 1
    within_outer = D[np.ix_(outer, outer)].max()
    across = D[np.ix_(~outer, outer)].min()

    fig.text(
        0.5,
        -0.05,
        "Same points, same ordering, same algorithm — only the dissimilarity "
        "changes. Under $D$ the inner ring reads as a block and the outer one "
        f"does not: its own\nwithin-ring distances reach {within_outer:.1f} while "
        f"the nearest inner-to-outer distance is {across:.1f}, which is the "
        "configuration no Euclidean prototype can resolve. Under the\nbottleneck "
        "ultrametric $D^*$ both separate, and the score moves from chance to "
        f"exact. The transform, not the selector, is what does this. "
        f"{H.provenance_note(label)}",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
