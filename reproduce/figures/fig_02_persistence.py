#!/usr/bin/env python3
"""Figure 2.3 -- a single-linkage dendrogram and the persistence diagram it implies.

§2.3 defines a feature's birth as the threshold at which it forms and its death
as the threshold at which it merges into something larger. That definition is
about *components of the hierarchy*, so the persistence diagram here is over
internal dendrogram nodes -- birth is the node's own merge height, death is the
height of the merge that absorbs it. Chapter 5's blocks are the same objects,
which is the point: this figure is the one Chapter 5's selector operates on.

The alternative reading -- taking every singleton as born at zero -- is the
standard 0-dimensional barcode, and it is useless here. Under single linkage
every point is born at height 0, so the diagram degenerates to a vertical line
and nothing "stands off the diagonal". The chapter's definition is the one that
carries information about which merges to believe, and it is the one drawn.

The data is built to have exactly the structure the caption describes: two tight
clusters that survive a long way up the hierarchy, plus scattered noise whose
merges are born just under the merge that kills them and therefore sit on the
diagonal. Illustrative, not measured -- no proposal number depends on it.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "02-persistence"

SEED = 11
N_PER_CLUSTER = 14
N_NOISE = 7


def _dataset():
    """Two tight clusters, and noise scattered on a ring well outside both.

    The ring matters. Noise sprinkled uniformly over the clusters' own bounding
    box chains them together at a low threshold, and then no merge has long
    persistence -- the figure would be illustrating the opposite of its caption.
    Putting the noise outside keeps the two clusters the only early structure.

    The two clusters have deliberately *different* spreads. Equal ones are born
    at the same height and die at the same merge, so their two points land on
    top of each other in the diagram and the caption's "two" looks like one.
    Unequal spreads separate them -- and separate them along the birth axis,
    which is the density reading §2.3 sets up for Chapter 5.
    """
    rng = np.random.RandomState(SEED)
    a = rng.normal(loc=(0.0, 0.0), scale=0.34, size=(N_PER_CLUSTER, 2))
    b = rng.normal(loc=(4.2, 0.6), scale=0.15, size=(N_PER_CLUSTER, 2))
    angle = rng.uniform(0, 2 * np.pi, N_NOISE)
    radius = rng.uniform(4.6, 6.8, N_NOISE)
    noise = np.c_[2.1 + radius * np.cos(angle), 0.3 + radius * np.sin(angle)]
    return np.vstack([a, b, noise])


def _persistence(Z, n):
    """(birth, death, size) per internal node; death is None for the root.

    An internal node i has cluster id n + i. Its death is the height of whichever
    merge consumes that id -- so one pass over the linkage rows builds the whole
    parent map.
    """
    death = {}
    for i, (left, right, height, _size) in enumerate(Z):
        for child in (int(left), int(right)):
            if child >= n:
                death[child - n] = height
    return [(Z[i, 2], death.get(i), int(Z[i, 3])) for i in range(len(Z))]


def build():
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist

    X = _dataset()
    n = len(X)
    Z = linkage(pdist(X), method="single")
    feats = _persistence(Z, n)

    ceiling = float(Z[:, 2].max()) * 1.12  # where the root's death is drawn
    finite = [(b, d, s) for b, d, s in feats if d is not None]
    # "Real" here means what §2.3 says it means: persistence far above the rest.
    persistences = sorted((d - b for b, d, _ in finite), reverse=True)
    cut = persistences[1] if len(persistences) > 1 else 0.0

    fig, (dend_ax, diag_ax) = F.grid_figure(
        1, 2, width=F.W_WIDE, height=3.4, gridspec_kw={"width_ratios": [1.35, 1]}
    )

    # -- left: the dendrogram -------------------------------------------------
    dd = dendrogram(
        Z, ax=dend_ax, no_labels=True, color_threshold=0, above_threshold_color=F.FAINT
    )
    for coll in dend_ax.collections:
        coll.set_linewidth(1.0)
    # scipy draws each merge as one polyline; recolour the two that matter.
    for xs, ys in zip(dd["icoord"], dd["dcoord"]):
        birth = max(ys)
        node = min(range(len(Z)), key=lambda i: abs(Z[i, 2] - birth))
        b, d, _ = feats[node]
        if d is not None and (d - b) >= cut:
            dend_ax.plot(xs, ys, color=F.BLUE, lw=1.8, zorder=4, solid_capstyle="round")

    F.style_axes(
        dend_ax,
        title="(a)  single-linkage dendrogram",
        ylabel="merge height",
        grid=True,
        grid_axis="y",
    )
    dend_ax.set_xticks([])
    dend_ax.spines["bottom"].set_visible(False)
    # -- right: the persistence diagram --------------------------------------
    lim = ceiling * 1.05
    diag_ax.plot([0, lim], [0, lim], lw=1.0, ls="--", color=F.FAINT, zorder=1)
    diag_ax.text(
        lim * 0.62,
        lim * 0.58,
        "birth = death",
        rotation=45,
        ha="center",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        rotation_mode="anchor",
    )

    noise_b = [b for b, d, _ in finite if (d - b) < cut]
    noise_d = [d for b, d, _ in finite if (d - b) < cut]
    real_b = [b for b, d, _ in finite if (d - b) >= cut]
    real_d = [d for b, d, _ in finite if (d - b) >= cut]

    diag_ax.scatter(
        noise_b,
        noise_d,
        s=22,
        color=F.FAINT,
        linewidths=0,
        zorder=3,
        label="short persistence — noise",
    )
    diag_ax.scatter(
        real_b,
        real_d,
        s=42,
        color=F.BLUE,
        linewidths=0,
        zorder=4,
        label="long persistence — the two clusters",
    )

    root_b = next(b for b, d, _ in feats if d is None)
    diag_ax.scatter(
        [root_b],
        [ceiling],
        s=42,
        facecolor=F.SURFACE,
        edgecolor=F.BLUE,
        linewidths=1.3,
        zorder=4,
        label="the root — dies at $\\infty$",
    )

    F.style_axes(
        diag_ax,
        title="(b)  persistence diagram",
        xlabel="birth height",
        ylabel="death height",
    )
    diag_ax.set_xlim(0, lim)
    diag_ax.set_ylim(0, lim)
    diag_ax.set_aspect("equal")
    F.legend(diag_ax, loc="lower right", handletextpad=0.4)

    fig.text(
        0.5,
        -0.01,
        "Persistence is death minus birth: the vertical distance off the "
        "diagonal. The two clusters form early and survive until the merge "
        "that joins them,\nso they stand well off it; a noise merge is "
        "absorbed almost as soon as it forms and sits on it. This is the "
        "quantity Chapter 5 gates on.",
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
