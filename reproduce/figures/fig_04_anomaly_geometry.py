#!/usr/bin/env python3
"""Figure 4.4 -- the geometry of the anomaly rule: where it can fire, and where it wins.

§4.3.5 works out, in prose, that the complement-of-conorm rule is degenerate
"totally at the default and partially across the sweep", and that on a one-class
fit it is a bare threshold at (1 - theta) / 2. Both statements are about a
function of the class firings, so both can be drawn. Appendix A.10.10 is the
algebra; this is its picture, computed from the same formula
`tsk_firing_strengths` evaluates:

    mu_anom = 1 - S_H(clip(mu_1 + theta), ..., clip(mu_K + theta)),
    S_H(a, b) = (a + b - 2ab) / (1 - ab)     (Hamacher, the shipped conorm)

Panels (a) and (b): two known classes, mu_anom over the unit square of (mu_1,
mu_2) at theta = 0.5 and at the inherited default theta = 0.99. The dashed lines
are mu_k = 1 - theta; beyond either, one input clips to 1 and the aggregate
saturates, so mu_anom is identically zero. The white contour is where the
anomaly rule ties the best class firing -- inside it the anomaly wins the
argmax. At 0.99 that region is a sliver in the corner. Panel (c): the one-class
case, where the conorm has nothing to aggregate and the anomaly wins exactly
when mu < (1 - theta) / 2; the line is that threshold against theta.

Computed from the formula; the Hamacher branch is reproduced here rather than
imported so the figure needs no submodule. Nothing here is a measurement.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "04-anomaly-geometry"

THETAS = (0.5, 0.99)


def _hamacher_s(a, b):
    num = a + b - 2 * a * b
    den = 1 - a * b
    out = np.ones_like(a)
    ok = np.abs(den) > 1e-12
    np.divide(num, den, out=out, where=ok)
    return out


def _anomaly(mu1, mu2, theta):
    c1 = np.clip(mu1 + theta, 0.0, 1.0)
    c2 = np.clip(mu2 + theta, 0.0, 1.0)
    return 1.0 - _hamacher_s(c1, c2)


def build():
    g = np.linspace(0.0, 1.0, 301)
    M1, M2 = np.meshgrid(g, g, indexing="ij")

    fig, axes = F.grid_figure(
        1, 3, width=F.W_WIDE, height=2.9, gridspec_kw={"width_ratios": [1, 1, 1.25]}
    )
    cmap = F.blue_cmap(reverse=False)
    for ax, theta, tag in zip(axes[:2], THETAS, ("(a)", "(b)")):
        A = _anomaly(M1, M2, theta)
        ax.imshow(
            A,
            origin="lower",
            extent=(0, 1, 0, 1),
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
            rasterized=True,
            aspect="auto",
        )
        wins = A - np.maximum(M1, M2)
        ax.contour(M2, M1, wins, levels=[0.0], colors=[F.SURFACE], linewidths=1.2)
        ax.contourf(
            M2,
            M1,
            wins,
            levels=[0.0, 10.0],
            colors=[F.tint(F.ORANGE, 0.55)],
            alpha=None,
        )
        bar = 1 - theta
        ax.axvline(bar, lw=1.0, ls=(0, (3, 2)), color=F.ORANGE)
        ax.axhline(bar, lw=1.0, ls=(0, (3, 2)), color=F.ORANGE)
        ax.set_xticks([0, bar, 1] if 0.05 < bar < 0.95 else [0, 1])
        ax.set_yticks([0, bar, 1] if 0.05 < bar < 0.95 else [0, 1])
        ax.set_xticklabels([f"{v:.2g}" for v in ax.get_xticks()], fontsize=F.FS_SMALL)
        ax.set_yticklabels([f"{v:.2g}" for v in ax.get_yticks()], fontsize=F.FS_SMALL)
        for s in ax.spines.values():
            s.set_color(F.AXIS)
        ax.set_xlabel("$\\mu_2(x)$", fontsize=F.FS_SMALL, color=F.INK_2)
        ax.set_ylabel("$\\mu_1(x)$", fontsize=F.FS_SMALL, color=F.INK_2)
        ax.set_title(
            f"{tag}  two classes, $\\theta$ = {theta}",
            fontsize=F.FS_LABEL,
            color=F.INK,
            pad=5,
        )
        ax.text(
            0.97,
            0.97,
            "$\\mu_{\\rm anom} \\equiv 0$\npast either dashed\nline: $S(1,\\cdot) = 1$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=F.FS_SMALL - 0.5,
            color=F.MUTED,
            linespacing=1.3,
        )

    # -- (c) one class: a threshold ---------------------------------------------
    pc = axes[2]
    th = np.linspace(0.0, 1.0, 200)
    pc.plot(
        th,
        (1 - th) / 2,
        lw=2.0,
        color=F.BLUE,
        label="anomaly wins iff $\\mu(x) < (1-\\theta)/2$",
    )
    pc.fill_between(
        th, 0, (1 - th) / 2, color=F.tint(F.ORANGE, 0.75), linewidth=0, zorder=0
    )
    for t, (tx, ty) in zip(THETAS, ((0.52, 0.36), (0.72, 0.13))):
        pc.plot([t], [(1 - t) / 2], marker="o", ms=5, color=F.ORANGE, zorder=5)
        pc.annotate(
            f"$\\theta$ = {t}: fires below {(1 - t) / 2:.3g}",
            xy=(t, (1 - t) / 2),
            xytext=(tx, ty),
            ha="left",
            va="center",
            fontsize=F.FS_SMALL,
            color=F.MUTED,
            arrowprops=dict(arrowstyle="-", lw=0.7, color=F.AXIS),
        )
    F.style_axes(
        pc,
        title="(c)  one class: $\\theta$ is a threshold",
        xlabel="boost $\\theta$",
        ylabel="firing $\\mu(x)$",
    )
    pc.set_xlim(0, 1)
    pc.set_ylim(0, 0.55)
    pc.text(
        0.20,
        0.10,
        "labelled\nanomalous",
        ha="center",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.shade(F.ORANGE, 0.3),
        transform=pc.transData,
        linespacing=1.3,
    )
    pc.text(
        0.98,
        0.50,
        "no conorm, no family choice:\n$t\\_conorm(x, \\mathrm{None})$ returns $x$",
        ha="right",
        va="top",
        fontsize=F.FS_SMALL - 0.5,
        color=F.MUTED,
        linespacing=1.3,
    )

    fig.text(
        0.5,
        -0.03,
        "Dark is high $\\mu_{\\rm anom}$; the orange region is where the anomaly label wins the argmax. At the inherited "
        "$\\theta$ = 0.99 the rule can fire only when every class is\nbelow 0.01 and the conorm never enters the "
        "decision; across the swept band it acts on samples whose firings are all small. With one class it is "
        "a threshold, full stop (Appendix A.10.10).",
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
