#!/usr/bin/env python3
"""Figure 2.2 -- one-input TSK inference, and the linearity in the consequents.

§2.1 says the single fact Chapters 4 and 6 hinge on: for fixed firing strengths
the TSK output is linear in the consequent coefficients. A sentence can state
that; a picture can show what it *means*, which is that changing one rule's
consequent moves the output by that rule's normalized firing times the change,
and nothing else.

So the figure is a one-input toy. Three Gaussian antecedents over x in [0, 1]
(top), three affine consequents f_r(x) = m_r + c_r x drawn as the faint local
lines they are (bottom), and the weighted average that blends them into the
output (bold). Then one consequent's constant is shifted by a fixed delta and
the shifted output is drawn dashed: the gap between the two curves is exactly
w_bar_2(x) * delta, the shape of the middle membership function, which is what
"linear in the coefficients" looks like on a page. Appendix A.10.1 is the
algebra; this is its picture.

Illustrative, not measured. The membership centres, widths and consequent
coefficients are chosen for legibility and no proposal number depends on them.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "02-tsk-inference"

CENTRES = (0.15, 0.50, 0.85)
SIGMA = 0.12
# (m_r, c_r): the constant and slope of each rule's consequent, on a target
# scale of roughly [0, 1] so the local lines and the blend share an axis.
CONSEQUENTS = ((0.10, 1.60), (0.85, -0.70), (0.25, 0.90))
DELTA = 0.25  # the shift applied to rule 2's constant in the second curve


def _memberships(x):
    return np.stack([np.exp(-0.5 * ((x - c) / SIGMA) ** 2) for c in CENTRES])


def _blend(x, consequents):
    w = _memberships(x)
    wbar = w / w.sum(axis=0, keepdims=True)
    f = np.stack([m + c * x for m, c in consequents])
    return wbar, f, (wbar * f).sum(axis=0)


def build():
    x = np.linspace(0.0, 1.0, 400)
    wbar, f, y = _blend(x, CONSEQUENTS)
    shifted = list(CONSEQUENTS)
    shifted[1] = (CONSEQUENTS[1][0] + DELTA, CONSEQUENTS[1][1])
    _, _, y_shift = _blend(x, shifted)

    fig, (top, bottom) = F.grid_figure(
        2, 1, width=F.W_COL + 0.6, height=4.9, gridspec_kw={"height_ratios": [1, 1.6]}
    )
    colours = (F.BLUE, F.ORANGE, F.AQUA)

    # -- top: antecedents and their normalized firings ------------------------
    w = _memberships(x)
    for r, (col, c) in enumerate(zip(colours, CENTRES)):
        top.plot(x, w[r], lw=1.8, color=col, label=f"$\\mu_{r + 1}(x)$")
        top.plot(x, wbar[r], lw=1.0, ls=(0, (2, 2)), color=col)
    top.text(
        0.99,
        0.30,
        "solid: $\\mu_r(x)$\ndashed: $\\bar w_r(x) = \\mu_r / \\sum_s \\mu_s$",
        transform=top.transAxes,
        ha="right",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    F.style_axes(top, title="(a)  three Gaussian antecedents", ylabel="membership")
    top.set_xlim(0, 1)
    top.set_ylim(0, 1.08)
    top.set_yticks([0, 0.5, 1.0])
    top.set_xticklabels([])

    # -- bottom: local consequents, the blend, and the shifted blend -----------
    for r, col in enumerate(colours):
        bottom.plot(
            x,
            f[r],
            lw=1.0,
            color=F.tint(col, 0.45),
            label=f"$f_{r + 1}(x) = m_{r + 1} + c_{r + 1} x$",
        )
    bottom.plot(x, y, lw=2.2, color=F.INK, label="$\\hat y(x) = \\sum_r \\bar w_r f_r$")
    bottom.plot(
        x,
        y_shift,
        lw=1.6,
        ls=(0, (4, 2)),
        color=F.shade(F.ORANGE, 0.2),
        label=f"same, with $m_2 \\to m_2 + {DELTA}$",
    )
    bottom.fill_between(
        x, y, y_shift, color=F.tint(F.ORANGE, 0.85), linewidth=0, zorder=0
    )
    peak = int(np.argmax(wbar[1]))
    bottom.annotate(
        "gap $= \\bar w_2(x)\\,\\Delta m_2$ —\nthe middle membership,\nscaled: linear in $m_2$",
        xy=(x[peak], (y[peak] + y_shift[peak]) / 2),
        xytext=(0.62, 0.15),
        ha="left",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS),
    )
    F.style_axes(
        bottom,
        title="(b)  affine consequents, and the weighted average that blends them",
        xlabel="input $x$",
        ylabel="output",
    )
    bottom.set_xlim(0, 1)
    bottom.set_ylim(-0.1, 1.45)
    F.legend(bottom, loc="upper left", ncol=2, columnspacing=1.2)

    fig.text(
        0.5,
        -0.01,
        "For fixed antecedents the output is a convex combination of the local "
        "consequents, so it is linear in every $(m_r, c_r)$: shifting one\n"
        "constant moves the curve by that rule's normalized firing times the "
        "shift and by nothing else. That is why the THEN side of every rule\n"
        "in Chapters 4 and 6 is solved in closed form (Appendix A.10.1, A.10.16) "
        "while the IF side, non-linear in $\\mu$ and $\\sigma$, is not.",
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
