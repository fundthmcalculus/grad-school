#!/usr/bin/env python3
"""Figure 6.5 -- the Ruspini export: a triangular partition of unity, and apex-only refinement.

§6.3.4 exports the flat model to a shared triangular strong partition of unity
and then refines only the apex knots, and Appendix A.10.19 identifies the terms
as linear B-splines whose sum-to-one property is intrinsic. The figure draws
that construction by the rule `ruspini.build_triangular_partition` states: term
0 a left shoulder, term k-1 a right shoulder, every interior term a triangle
rising from the previous apex and falling to the next.

Panel (a): the partition on one axis at its initial knots, with the sum of all
terms drawn as the flat line it is. Panel (b): the same partition after two
apex knots move -- the refinement step -- with the sum still identically one,
because moving a knot rebuilds the two adjacent hats and nothing else. Panel
(c): what the exported rule base computes with constant consequents, the
piecewise-linear interpolant through (knot, consequent), before and after the
knot move: a function a reader can draw from the rule table by hand.

The hat functions are computed from the knot rule reproduced here rather than
imported, so the figure needs no submodule; the rule is the one the library
implements, shoulders included. Illustrative knots and consequents; nothing
here is a measurement.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "06-ruspini-export"

KNOTS = np.array([0.10, 0.32, 0.55, 0.80])
KNOTS_REFINED = np.array([0.10, 0.40, 0.50, 0.80])  # two apexes moved
CONSEQUENTS = np.array([0.2, 0.9, 0.4, 0.7])


def _partition(x, knots):
    """Hat functions on sorted knots with shoulders at both ends -- ruspini.py's rule."""
    k = len(knots)
    terms = []
    for i in range(k):
        a = -np.inf if i == 0 else knots[i - 1]
        b = knots[i]
        c = np.inf if i == k - 1 else knots[i + 1]
        mu = np.zeros_like(x)
        left = x <= b
        mu[left] = 1.0 if np.isneginf(a) else np.clip((x[left] - a) / (b - a), 0, 1)
        right = x > b
        mu[right] = 1.0 if np.isposinf(c) else np.clip((c - x[right]) / (c - b), 0, 1)
        terms.append(mu)
    return np.stack(terms)


def build():
    x = np.linspace(0.0, 1.0, 600)
    colours = (F.BLUE, F.AQUA, F.ORANGE, F.VIOLET)

    fig, axes = F.grid_figure(1, 3, width=F.W_WIDE, height=2.9)
    for ax, knots, tag in zip(
        axes[:2],
        (KNOTS, KNOTS_REFINED),
        ("(a)  exported partition", "(b)  two apex knots moved"),
    ):
        U = _partition(x, knots)
        for k, col in enumerate(colours):
            ax.plot(x, U[k], lw=1.7, color=col)
            ax.plot([knots[k]], [1.0], marker="v", ms=5, color=col, clip_on=False)
        ax.plot(
            x,
            U.sum(axis=0),
            lw=1.2,
            ls=(0, (3, 2)),
            color=F.INK_2,
            label="$\\sum_i \\mu_i(x)$",
        )
        F.style_axes(
            ax,
            title=tag,
            xlabel="feature value",
            ylabel="membership" if ax is axes[0] else None,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.5, 1.0])
        F.legend(ax, loc="lower right", bbox_to_anchor=(1.0, 0.02))
        for kx in knots:
            ax.axvline(kx, lw=0.6, ls=(0, (1, 2)), color=F.FAINT)
    moved = [i for i in range(len(KNOTS)) if KNOTS[i] != KNOTS_REFINED[i]]
    for i in moved:
        axes[1].annotate(
            "",
            xy=(KNOTS_REFINED[i], 1.06),
            xytext=(KNOTS[i], 1.06),
            arrowprops=dict(arrowstyle="->", lw=1.0, color=F.shade(colours[i], 0.2)),
        )

    # -- (c) the exported rule base as a function -----------------------------------
    pc = axes[2]
    for knots, style, name in (
        (KNOTS, "solid", "initial knots"),
        (KNOTS_REFINED, (0, (4, 2)), "refined knots"),
    ):
        U = _partition(x, knots)
        y = (U * CONSEQUENTS[:, None]).sum(axis=0)
        pc.plot(
            x,
            y,
            lw=2.0,
            ls=style,
            color=F.INK if style == "solid" else F.ORANGE,
            label=name,
        )
        pc.scatter(
            knots,
            CONSEQUENTS,
            s=22,
            color=F.INK if style == "solid" else F.ORANGE,
            zorder=5,
            linewidths=0,
        )
    F.style_axes(
        pc,
        title="(c)  output, constant consequents",
        xlabel="feature value",
        ylabel="$\\hat y(x) = \\sum_i \\mu_i(x)\\, m_i$",
    )
    pc.set_xlim(0, 1)
    F.legend(pc, loc="lower left")
    pc.text(
        0.98,
        0.04,
        "piecewise-linear through\n(knot, consequent); flat\nbeyond the outer knots",
        transform=pc.transAxes,
        ha="right",
        va="bottom",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.4,
    )

    fig.text(
        0.5,
        -0.03,
        "On each knot interval exactly two hat functions are live and they sum to one, so the partition-of-unity "
        "property survives any knot move that keeps the order — apex-only\nrefinement is free-knot linear-spline fitting "
        "under another name (Appendix A.10.19), and the rule base stays a document a reader can evaluate by hand.",
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
