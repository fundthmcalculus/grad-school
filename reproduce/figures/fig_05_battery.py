#!/usr/bin/env python3
"""Figure 5.5 -- the five battery datasets, and what the gated set-cover makes of each.

Table 5.1 is a grid of adjusted Rand indices with a discovered-k column, and its
caption spends a paragraph explaining that k-means given k ties the field on
three of four datasets and collapses only on the rings. A reader who has not
seen the five constructions has no way to judge any of that. This figure is the
five point sets, drawn once, coloured by the block the gated set-cover assigns
each point to, uncovered points hollow -- so the discovered-k, coverage and
"near-abstention on noise" columns of the table are all visible at a glance.

Every panel runs the same three calls Table 5.1's set-cover column runs:
Euclidean dissimilarity, `ivat_mf.minimax_transform`, then
`selection.select_coverage_cover` at the driver's defaults. The k, coverage and
ARI-on-covered-points in each title are computed here, not copied; if the
battery's generators or the selector move, the panel moves with them and Table
5.1 is regenerated from the same code. Ground-truth labels are drawn as marker
shapes so the reader can see agreement without a second colour system.

Computed, on the battery's own synthetic constructions. Needs scipy and
scikit-learn (the battery imports both).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "05-battery"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gated-minimax-selection"))

MARKERS = ("o", "s", "^", "D")


def build():
    import battery as B  # noqa: E402
    import ivat_mf as im  # noqa: E402
    import selection as S  # noqa: E402
    from sklearn.metrics import adjusted_rand_score

    cases = [
        ("concentric_rings", B.concentric_rings),
        ("bridged_gaussians", B.bridged_gaussians),
        ("well_separated", B.two_gaussians),
        ("varying_density", B.varying_density),
        ("uniform_noise", B.uniform_noise),
    ]
    colours = [F.BLUE, F.ORANGE, F.AQUA, F.VIOLET, F.MAGENTA, F.GREEN]

    fig, axes = F.grid_figure(1, 5, width=F.W_WIDE + 0.8, height=2.6)
    for ax, (name, gen) in zip(axes, cases):
        X, y = gen()
        n = len(X)
        Dstar = im.minimax_transform(im.dissimilarity(X))
        sel = S.select_coverage_cover(Dstar)
        label = np.full(n, -1)
        for k, b in enumerate(sel):
            label[np.fromiter(b["members"], dtype=int)] = k
        covered = label >= 0
        truth_classes = sorted({v for v in y if v >= 0})

        for t_idx, t in enumerate(sorted(set(y))):
            m_t = y == t
            marker = MARKERS[t_idx % len(MARKERS)] if t >= 0 else "x"
            unc = m_t & ~covered
            hollow = dict(facecolor=F.SURFACE, edgecolor=F.FAINT)
            if marker == "x":  # an unfilled marker takes a single colour
                hollow = dict(color=F.FAINT)
            ax.scatter(
                X[unc, 0],
                X[unc, 1],
                s=13,
                marker=marker,
                linewidths=0.7,
                zorder=2,
                **hollow,
            )
            for k in range(len(sel)):
                m = m_t & (label == k)
                if m.any():
                    ax.scatter(
                        X[m, 0],
                        X[m, 1],
                        s=13,
                        marker=marker,
                        color=colours[k % len(colours)],
                        linewidths=0,
                        zorder=3,
                    )
        # Equal data ranges rather than aspect="equal": the latter shrinks the
        # panel to the data and leaves five titles at five heights.
        cx, cy = X.mean(axis=0)
        half = 0.55 * max(np.ptp(X[:, 0]), np.ptp(X[:, 1]))
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.axis("off")
        if truth_classes and covered.any():
            ari = adjusted_rand_score(y[covered], label[covered])
            score = f"ARI {ari:.3f} on covered"
        else:
            score = "no truth partition"
        ax.set_title(
            f"{name}\n$k$ = {len(sel)}"
            + (f" (true {len(truth_classes)})" if truth_classes else " (true: none)")
            + f", cov. {covered.mean():.3f}\n{score}",
            fontsize=F.FS_SMALL,
            color=F.INK,
            pad=4,
            linespacing=1.4,
        )

    fig.text(
        0.5,
        -0.02,
        "Colour is the block the gated set-cover assigns; hollow markers are uncovered points; marker shape is the "
        "ground-truth class (× = bridge or noise).\nThe rings need the minimax transform every centroid method lacks; "
        "the bridge is the gate's recorded failure; on noise it claims an eighth of the points rather than none — "
        "Table 5.1's columns, drawn.",
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
