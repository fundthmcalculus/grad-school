#!/usr/bin/env python3
"""Figure 3.3 -- the divide-and-conquer stitch: blocks, representatives, cross edges.

§3.3.4 describes the stitch in four clauses -- split into blocks, farthest-point
representatives per block, the strongest cross-block edges between
representatives, reconcile across those -- and Table 3.6 measures that both
ingredients are needed. Nothing in the chapter shows what the ingredients look
like, so a reader has to imagine why random representatives or a single cross
edge would fail.

Two panels on one two-moons construction, split into two blocks by a vertical
cut that deliberately goes through both moons (the case a naive per-block
ordering gets wrong: each moon is severed and the seam shows up as a spurious
cluster). Panel (a) is the split, with the seam drawn. Panel (b) is the stitch:
the r farthest-point representatives of each block, computed by the greedy
farthest-point rule, and the m shortest representative-to-representative edges
that cross the seam. Farthest-point sampling spreads the representatives along
each block's extent, so the moons' tips reach the seam and the top-m edges
connect the right halves; random representatives cluster in the interior and
the edges then connect the wrong halves, which is the light-stitch failure of
Table 3.6.

Schematic in the sense that the moons are synthetic and the reconciliation
itself is not run here; the representatives and cross edges *are* computed by
the rules the section states. No proposal number depends on this figure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "03-stitch"

SEED = 2
N_PER_MOON = 110
NOISE = 0.06
N_REPS = 6
N_CROSS = 8


def _two_moons():
    rng = np.random.RandomState(SEED)
    t = rng.uniform(0, np.pi, N_PER_MOON)
    upper = np.c_[np.cos(t), np.sin(t)]
    lower = np.c_[1 - np.cos(t), 1 - np.sin(t) - 0.5]
    X = np.vstack([upper, lower]) + rng.normal(0, NOISE, (2 * N_PER_MOON, 2))
    return X


def _farthest_point(X, r):
    """Greedy farthest-point sampling: start at the point farthest from the centroid."""
    d0 = np.linalg.norm(X - X.mean(axis=0), axis=1)
    reps = [int(np.argmax(d0))]
    dist = np.linalg.norm(X - X[reps[0]], axis=1)
    while len(reps) < r:
        nxt = int(np.argmax(dist))
        reps.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
    return np.array(reps)


def build():
    X = _two_moons()
    cut = np.median(X[:, 0])
    left, right = X[X[:, 0] < cut], X[X[:, 0] >= cut]
    rl, rr = _farthest_point(left, N_REPS), _farthest_point(right, N_REPS)

    # Shortest representative-to-representative edges across the seam.
    C = np.linalg.norm(left[rl][:, None, :] - right[rr][None, :, :], axis=2)
    flat = np.argsort(C, axis=None)[:N_CROSS]
    cross = [np.unravel_index(f, C.shape) for f in flat]

    fig, (pa, pb) = F.grid_figure(1, 2, width=F.W_WIDE, height=3.2)
    for ax in (pa, pb):
        ax.scatter(
            left[:, 0],
            left[:, 1],
            s=9,
            color=F.tint(F.BLUE, 0.35),
            linewidths=0,
            zorder=2,
        )
        ax.scatter(
            right[:, 0],
            right[:, 1],
            s=9,
            color=F.tint(F.ORANGE, 0.35),
            linewidths=0,
            zorder=2,
        )
        ax.axvline(cut, lw=1.0, ls=(0, (3, 2)), color=F.FAINT, zorder=1)
        ax.set_aspect("equal")
        ax.axis("off")

    pa.text(
        cut + 0.05,
        X[:, 1].min() - 0.05,
        "block boundary",
        ha="left",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
    )
    pa.text(
        0.5,
        -0.06,
        "each block ordered on its own; concatenating the two\n"
        "orderings puts a seam through both moons, which reads\n"
        "as spurious cluster structure (Table 3.5's 'naive block')",
        transform=pa.transAxes,
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    pa.set_title(
        "(a)  the split: two blocks, both moons severed",
        fontsize=F.FS_LABEL,
        color=F.INK,
        pad=6,
    )

    for a, b in cross:
        pb.plot(
            [left[rl[a], 0], right[rr[b], 0]],
            [left[rl[a], 1], right[rr[b], 1]],
            lw=1.1,
            color=F.AQUA,
            zorder=3,
        )
    pb.scatter(
        left[rl, 0],
        left[rl, 1],
        s=42,
        color=F.BLUE,
        edgecolor=F.SURFACE,
        linewidths=0.8,
        zorder=5,
    )
    pb.scatter(
        right[rr, 0],
        right[rr, 1],
        s=42,
        color=F.ORANGE,
        edgecolor=F.SURFACE,
        linewidths=0.8,
        zorder=5,
    )
    pb.text(
        0.5,
        -0.06,
        f"{N_REPS} farthest-point representatives per block (large dots)\n"
        f"and the {N_CROSS} shortest representative-to-representative edges\n"
        "across the seam (aqua); the reconciliation walks only these,\n"
        f"so its cost is $O(r^2)$ in $r$ = {N_REPS}, independent of block size",
        transform=pb.transAxes,
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    pb.set_title(
        "(b)  the stitch: representatives and top cross edges",
        fontsize=F.FS_LABEL,
        color=F.INK,
        pad=6,
    )

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
