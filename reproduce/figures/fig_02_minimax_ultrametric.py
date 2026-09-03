#!/usr/bin/env python3
"""Figure 2.5 -- the minimax (bottleneck) distance: MST path, ultrametric, dendrogram.

§2.2 introduces iVAT's minimax transform in one sentence -- "the largest edge on
the lightest path between two points" -- and Chapters 3 and 5 then lean on three
facts about it: it equals the heaviest edge on the MST path (so it is cheap), it
is an ultrametric (so relational clustering works on it), and it is the
single-linkage merge height (so the iVAT image and the dendrogram are one
object). Appendix A.10.3 proves all three; this figure shows them on one small
point set.

Panel (a): the points, their MST, and one far-apart pair (i, j) picked out. The
direct dissimilarity D_ij is the dashed grey chord; the MST path between them is
blue; its heaviest edge -- the bottleneck, which *is* D*_ij -- is orange.
Panel (b): D and D* as matrices in the same VAT order. D* is blocky where D is
graded, because inside a cluster every pair shares the same bottleneck.
Panel (c): the single-linkage dendrogram; i and j merge at exactly the
bottleneck height, so D*_ij can be read off the tree.

The strong triangle inequality is checked numerically over every triple and the
worst violation is printed in the caption -- it is zero to machine precision.

Illustrative construction (three small Gaussian clumps, fixed seed); no proposal
number depends on it. Needs scipy for the MST, linkage and cophenetic distance.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "02-minimax-ultrametric"

SEED = 5
N_PER = 6


def _points():
    rng = np.random.RandomState(SEED)
    a = rng.normal((0.0, 0.0), 0.28, size=(N_PER, 2))
    b = rng.normal((3.0, 0.4), 0.28, size=(N_PER, 2))
    c = rng.normal((1.6, 2.6), 0.28, size=(N_PER, 2))
    return np.vstack([a, b, c])


def _vat_order(D):
    """Prim from the farther endpoint of the most-dissimilar pair -- textbook VAT."""
    n = len(D)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    seed = i if D[i].sum() >= D[j].sum() else j
    order = [seed]
    key = D[seed].copy()
    placed = np.zeros(n, bool)
    placed[seed] = True
    key[placed] = np.inf
    for _ in range(n - 1):
        v = int(np.argmin(key))
        order.append(v)
        placed[v] = True
        key = np.minimum(key, D[v])
        key[placed] = np.inf
    return np.array(order)


def _tree_path(T, i, j):
    """Vertices on the unique tree path from i to j (T is a symmetric adjacency)."""
    n = len(T)
    parent = {i: None}
    stack = [i]
    while stack:
        v = stack.pop()
        if v == j:
            break
        for u in range(n):
            if T[v, u] > 0 and u not in parent:
                parent[u] = v
                stack.append(u)
    path = [j]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    return path[::-1]


def build():
    from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform

    X = _points()
    n = len(X)
    D = squareform(pdist(X))
    T = minimum_spanning_tree(D).toarray()
    T = T + T.T
    Z = linkage(pdist(X), method="single")
    Dstar = squareform(cophenet(Z))  # single-linkage merge height == bottleneck

    # Strong triangle inequality, checked over every triple.
    viol = max(
        Dstar[i, j] - max(Dstar[i, k], Dstar[k, j])
        for i in range(n)
        for j in range(n)
        for k in range(n)
    )

    # The pair to illustrate: the two clusters' farthest members.
    i, j = 0, N_PER  # one from clump a, one from clump b
    path = _tree_path(T, i, j)
    edges = list(zip(path, path[1:]))
    bottleneck = max(edges, key=lambda e: D[e[0], e[1]])

    fig, (pa, pb1, pb2, pc) = F.grid_figure(
        1,
        4,
        width=F.W_WIDE + 0.6,
        height=2.75,
        gridspec_kw={"width_ratios": [1.25, 1, 1, 1.15]},
    )

    # -- (a) points, MST, the pair -------------------------------------------
    for u in range(n):
        for v in range(u + 1, n):
            if T[u, v] > 0:
                pa.plot(*X[[u, v]].T, lw=0.9, color=F.AXIS, zorder=1)
    for u, v in edges:
        pa.plot(*X[[u, v]].T, lw=2.2, color=F.BLUE, zorder=3)
    pa.plot(*X[list(bottleneck)].T, lw=3.0, color=F.ORANGE, zorder=4)
    pa.plot(*X[[i, j]].T, lw=1.0, ls=(0, (3, 2)), color=F.FAINT, zorder=2)
    pa.scatter(X[:, 0], X[:, 1], s=16, color=F.INK_2, linewidths=0, zorder=5)
    pa.scatter(X[[i, j], 0], X[[i, j], 1], s=44, color=F.BLUE, linewidths=0, zorder=6)
    pa.text(
        X[i, 0] - 0.12,
        X[i, 1] - 0.05,
        "$i$",
        ha="right",
        va="top",
        fontsize=F.FS_ANNOT,
        color=F.INK,
    )
    pa.text(
        X[j, 0] + 0.12,
        X[j, 1] - 0.05,
        "$j$",
        ha="left",
        va="top",
        fontsize=F.FS_ANNOT,
        color=F.INK,
    )
    mid = X[list(bottleneck)].mean(axis=0)
    pa.annotate(
        f"bottleneck edge\n$D^*_{{ij}}$ = {D[bottleneck]:.2f}",
        xy=tuple(mid),
        xytext=(mid[0] - 1.1, mid[1] + 1.0),
        ha="center",
        va="bottom",
        fontsize=F.FS_SMALL,
        color=F.shade(F.ORANGE, 0.25),
        arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS),
    )
    pa.text(
        (X[i, 0] + X[j, 0]) / 2,
        min(X[i, 1], X[j, 1]) - 0.45,
        f"direct $D_{{ij}}$ = {D[i, j]:.2f} (dashed)",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
    )
    pa.set_aspect("equal", adjustable="datalim")
    pa.axis("off")
    pa.set_title(
        "(a)  MST, and the path from $i$ to $j$",
        fontsize=F.FS_LABEL,
        color=F.INK,
        pad=6,
    )

    # -- (b) D and D* in VAT order -----------------------------------------------
    order = _vat_order(D)
    vmax = D.max()
    F.imshow_matrix(
        pb1, D[np.ix_(order, order)], title="(b)  $D$, VAT order", vmin=0, vmax=vmax
    )
    F.imshow_matrix(
        pb2, Dstar[np.ix_(order, order)], title="$D^*$, same order", vmin=0, vmax=vmax
    )
    pi, pj = int(np.where(order == i)[0][0]), int(np.where(order == j)[0][0])
    for ax in (pb1, pb2):
        ax.plot([pj], [pi], marker="s", ms=5, mfc="none", mec=F.ORANGE, mew=1.2)

    # -- (c) dendrogram, merge height of (i, j) -----------------------------------
    dd = dendrogram(
        Z, ax=pc, no_labels=True, color_threshold=0, above_threshold_color=F.FAINT
    )
    for coll in pc.collections:
        coll.set_linewidth(0.9)
    h = Dstar[i, j]
    pc.axhline(h, lw=1.2, ls=(0, (3, 2)), color=F.ORANGE, zorder=4)
    pc.text(
        0.02,
        h * 1.04,
        f"$i$ and $j$ merge at {h:.2f} = $D^*_{{ij}}$",
        transform=pc.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=F.FS_SMALL,
        color=F.shade(F.ORANGE, 0.25),
    )
    F.style_axes(
        pc,
        title="(c)  single-linkage dendrogram",
        ylabel="merge height",
        grid=True,
        grid_axis="y",
    )
    pc.set_xticks([])
    pc.spines["bottom"].set_visible(False)
    del dd

    fig.text(
        0.5,
        -0.03,
        "$D^*_{ij}$ is the heaviest edge on the MST path (orange), not the direct "
        "chord (dashed), and it is the height at which $i$ and $j$ merge under "
        "single linkage.\nInside a cluster every pair shares one bottleneck, so "
        "$D^*$ is block-constant where $D$ is graded. Checked over all "
        f"{n}$^3$ triples: max of $D^*_{{ij}} - \\max(D^*_{{ik}}, D^*_{{kj}})$ = "
        f"{viol:.1e} — an ultrametric (Appendix A.10.3).",
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
