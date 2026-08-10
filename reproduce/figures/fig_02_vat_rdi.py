#!/usr/bin/env python3
"""Figure 2.2 -- a dissimilarity matrix before and after VAT reordering.

The construction is the one §2.2 names: `circle_random_clusters` from
`tribbleclustering.util`, five rings of cities spaced around a circle. It is the
package's own generator rather than a synthetic set written here, so the picture
is of the software the chapter is about.

One detail decides whether the figure says anything at all. The generator emits
points grouped by cluster, so the *unpermuted* matrix is already block-diagonal
and the left panel would show the answer before VAT had done anything. The rows
are therefore shuffled first, under a fixed seed, and the left panel is that
shuffled matrix -- an arbitrary order, which is the case VAT exists to handle.

Three panels rather than the two the caption asks for. The point set is cheap to
draw and makes the other two legible: without it a reader has to take on faith
that the five dark blocks correspond to five rings.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "02-vat-rdi"

SEED = 3
N_CLUSTERS = 5
N_CITIES = 24


def build():
    from tribbleclustering.util import circle_random_clusters, pairwise_distances
    import tribbleclustering as tc

    np.random.seed(SEED)
    X = circle_random_clusters(
        n_clusters=N_CLUSTERS,
        n_cities=N_CITIES,
        cluster_diameter=3.5,
        cluster_spacing=10.0,
    )
    truth = np.repeat(np.arange(N_CLUSTERS), N_CITIES)

    # Shuffle before measuring: an already-grouped matrix would show the answer
    # in the "before" panel. See the module docstring.
    order = np.random.RandomState(SEED).permutation(len(X))
    X, truth = X[order], truth[order]

    D = pairwise_distances(np.ascontiguousarray(X, dtype=np.float64))
    RV, _ = tc.compute_vat(D.copy())
    result = tc.compute_ivat(D.copy())
    RIV = result[0]  # iVAT returns (reordered_matrix, ordering, rdi_curve)

    fig, axes = F.grid_figure(1, 4, width=F.W_WIDE, height=2.6)
    scatter, before, after, ivat = axes

    for k in range(N_CLUSTERS):
        pts = X[truth == k]
        scatter.scatter(
            pts[:, 0],
            pts[:, 1],
            s=5,
            linewidths=0,
            color=F.SEQ_BLUE[3 + 2 * k],
            zorder=3,
        )
    scatter.set_aspect("equal")
    scatter.set_xticks([])
    scatter.set_yticks([])
    for s in scatter.spines.values():
        s.set_color(F.AXIS)
        s.set_linewidth(0.8)
    scatter.set_title(
        f"(a)  {N_CLUSTERS} rings of {N_CITIES} cities",
        fontsize=F.FS_LABEL,
        color=F.INK,
        pad=6,
    )

    vmax = float(D.max())
    F.imshow_matrix(before, D, title="(b)  raw order — speckle", vmin=0, vmax=vmax)
    F.imshow_matrix(
        after, RV, title="(c)  after VAT — five blocks", vmin=0, vmax=vmax
    )
    im = F.imshow_matrix(
        ivat, RIV, title="(d)  after VAT reordering — traversal order", vmin=0, vmax=vmax
    )

    # One shared colourbar. All panels hold the same numbers.
    cbar = fig.colorbar(im, ax=list(axes), fraction=0.022, pad=0.012, shrink=0.82)
    cbar.outline.set_edgecolor(F.AXIS)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=F.FS_SMALL, colors=F.INK_2, length=2.5, width=0.7)
    cbar.set_label("dissimilarity", fontsize=F.FS_SMALL, color=F.INK_2)

    fig.text(
        0.5,
        0.0,
        "Panels (b), (c), and (d) hold the same matrix — only the row and column order differs. "
        "(c) uses VAT (MST-based); (d) uses iVAT (minimax path). "
        "Both reorderings cross only MST edges,\nso cutting at a threshold gives single-linkage clustering.",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )

    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
