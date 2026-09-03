#!/usr/bin/env python3
"""Figure 6.3 -- soft-tree routing: leaf weights that sum to one, and what they buy.

§6.3.2 says a point "flows down multiple paths with graded membership instead of
going left or right", and Appendix A.10.17 proves the resulting leaf weights
form a partition of unity. The figure shows both on a one-input, depth-two tree
with sigmoidal gates: a root split near the middle of the axis and one child
split on each side.

Panel (a): the four leaf weights w_l(x), each the product of the two gate
memberships on its path, drawn stacked so that the fact they sum to one at every
x is the flat top of the stack rather than a claim. Panel (b): the tree's
output. Crisp splits (gate memberships in {0, 1}) give the staircase-of-lines a
CART-with-linear-leaves tree produces, discontinuous at every split; the soft
gates give the convex blend of the same four leaf models, continuous, and still
linear in the leaf coefficients for fixed gates -- so the leaves are fitted by
the shared solver of Figure 6.1 with rows weighted by w_l.

Illustrative: the split positions, gate widths and leaf coefficients are chosen
for legibility; no proposal number depends on this figure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "06-soft-routing"

ROOT, LEFT, RIGHT = 0.50, 0.22, 0.76  # split thresholds
WIDTH = 0.045  # sigmoid softness, in x units
# leaf models m + c x, for leaves LL, LR, RL, RR
LEAVES = ((0.15, 1.2), (0.65, -0.9), (0.10, 1.1), (0.95, -0.4))


def _sig(x, t, w):
    return 1.0 / (1.0 + np.exp(np.clip(-(x - t) / w, -500, 500)))


def _weights(x, w):
    """Leaf weights for a depth-two tree; w -> 0 gives the crisp tree."""
    right = _sig(x, ROOT, w)
    left = 1.0 - right
    ll = left * (1.0 - _sig(x, LEFT, w))
    lr = left * _sig(x, LEFT, w)
    rl = right * (1.0 - _sig(x, RIGHT, w))
    rr = right * _sig(x, RIGHT, w)
    return np.stack([ll, lr, rl, rr])


def build():
    x = np.linspace(0.0, 1.0, 600)
    W = _weights(x, WIDTH)
    W_crisp = _weights(x, 1e-4)
    f = np.stack([m + c * x for m, c in LEAVES])
    y_soft = (W * f).sum(axis=0)
    y_crisp = (W_crisp * f).sum(axis=0)
    colours = (F.BLUE, F.AQUA, F.ORANGE, F.VIOLET)
    names = (
        f"$x \\leq {ROOT}$ and $x \\leq {LEFT}$",
        f"$x \\leq {ROOT}$ and $x > {LEFT}$",
        f"$x > {ROOT}$ and $x \\leq {RIGHT}$",
        f"$x > {ROOT}$ and $x > {RIGHT}$",
    )

    fig, (pa, pb) = F.grid_figure(
        2, 1, width=F.W_COL + 0.8, height=5.0, gridspec_kw={"height_ratios": [1, 1.4]}
    )

    # -- (a) stacked leaf weights ----------------------------------------------
    pa.stackplot(
        x, W, colors=[F.tint(c, 0.35) for c in colours], labels=names, linewidth=0
    )
    for k, c in enumerate(colours):
        pa.plot(x, W[k], lw=1.0, color=c)
    pa.axhline(1.0, lw=0.9, ls=(0, (3, 2)), color=F.INK_2)
    for t in (ROOT, LEFT, RIGHT):
        pa.axvline(t, lw=0.7, ls=(0, (1, 2)), color=F.FAINT)
    pa.text(
        0.995,
        1.03,
        "$\\sum_\\ell w_\\ell(x) = 1$ everywhere",
        ha="right",
        va="bottom",
        fontsize=F.FS_SMALL,
        color=F.INK_2,
    )
    F.style_axes(
        pa,
        title="(a)  leaf routing weights $w_\\ell(x)$ = product of the gates on the path",
        ylabel="weight",
    )
    pa.set_xlim(0, 1)
    pa.set_ylim(0, 1.15)
    pa.set_xticklabels([])
    F.legend(
        pa,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        fontsize=F.FS_SMALL - 0.5,
    )

    # -- (b) crisp tree against soft blend ----------------------------------------
    for k, ((m, c), col) in enumerate(zip(LEAVES, colours)):
        pb.plot(x, f[k], lw=0.9, color=F.tint(col, 0.55))
    pb.plot(
        x,
        y_crisp,
        lw=1.4,
        color=F.FAINT,
        label="crisp splits: one leaf per region, discontinuous",
    )
    pb.plot(
        x,
        y_soft,
        lw=2.2,
        color=F.INK,
        label="soft splits: $\\sum_\\ell w_\\ell(x)\\,f_\\ell(x)$, continuous",
    )
    for t in (ROOT, LEFT, RIGHT):
        pb.axvline(t, lw=0.7, ls=(0, (1, 2)), color=F.FAINT)
    F.style_axes(
        pb,
        title="(b)  the tree's output, with linear leaves",
        xlabel="input $x$",
        ylabel="output",
    )
    pb.set_xlim(0, 1)
    F.legend(pb, loc="upper left")

    fig.text(
        0.5,
        -0.01,
        "Each gate splits a single named input, so every weight is a product of memberships of original variables "
        "(the Magdalena condition, §6.2).\nFor fixed gates the output is linear in the leaf coefficients, so every "
        "leaf is fitted by the shared solver with rows weighted by $w_\\ell$ (Appendix A.10.17).",
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
