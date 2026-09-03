#!/usr/bin/env python3
"""Figure 6.1 -- the shared ridge-TSK solver: the design matrix it builds, and who calls it.

§6.3.1 calls the firing-weighted ridge solve "the primitive the next two
subsections reuse", and §6.1 says the same solver fits the flat FIS of Chapter
4, the leaves of the soft tree and the experts of the mixture. That is an
architectural claim and it deserves an architectural drawing. It also has a
concrete object at its centre -- the stacked design matrix Phi of Appendix
A.10.1 -- which is easy to describe and easier to look at.

Panel (a) is a real Phi, computed from the one-input toy of Figure 2.2: forty
points, three Gaussian rules, a first-order basis, so Phi is 40 x 6 with one
[w_r | w_r x] block per rule. The rows are sorted by x so the blocks read as
the firing strengths they are: each rule lights up the rows it covers. The
columns the ridge leaves unpenalised (each rule's constant) are marked, and the
augmented rows sqrt(lambda) D^1/2 that `lstsq` sees are drawn beneath, which is
how the code avoids forming the normal equations (Appendix A.10.16).

Panel (b) is the schematic: the three consumers, what each hands the solver
(its own firing or routing weights), and the one thing that comes back.

Panel (a) is computed from the toy; panel (b) is a diagram. No proposal number
depends on this figure.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "06-ridge-solver"

CENTRES = (0.15, 0.50, 0.85)
SIGMA = 0.12
N_ROWS = 40
LAMBDA = 0.05  # exaggerated so the penalty rows are visible; the code's default is 1e-6


def _design(x):
    w = np.stack([np.exp(-0.5 * ((x - c) / SIGMA) ** 2) for c in CENTRES], axis=1)
    wbar = w / w.sum(axis=1, keepdims=True)
    phi = np.c_[np.ones_like(x), x]  # [1 | x], first order
    return (wbar[:, :, None] * phi[:, None, :]).reshape(len(x), -1)


def build():
    x = np.linspace(0.0, 1.0, N_ROWS)
    Phi = _design(x)
    P = Phi.shape[1]
    penalty = np.ones(P)
    penalty[::2] = 0.0  # column 0 of each rule block is the constant: unpenalised
    aug = np.diag(np.sqrt(LAMBDA * penalty))

    fig, (pa, pb) = F.grid_figure(
        1, 2, width=F.W_WIDE, height=4.4, gridspec_kw={"width_ratios": [1, 1.35]}
    )

    # -- (a) the design matrix, and the ridge rows beneath it ------------------
    full = np.vstack([Phi, aug])
    pa.imshow(
        np.abs(full),
        aspect="auto",
        cmap=F.blue_cmap(reverse=False),
        vmin=0,
        vmax=np.abs(full).max(),
        interpolation="nearest",
        rasterized=True,
    )
    pa.axhline(N_ROWS - 0.5, lw=1.2, color=F.ORANGE)
    for r in range(len(CENTRES)):
        pa.axvline(2 * r - 0.5, lw=0.8, color=F.SURFACE)
        pa.text(
            2 * r + 0.5,
            -1.2,
            f"rule {r + 1}\n$\\bar w_{r + 1}\\,[1\\;\\;x]$",
            ha="center",
            va="bottom",
            fontsize=F.FS_SMALL,
            color=F.INK_2,
            linespacing=1.3,
        )
        pa.plot(
            [2 * r], [N_ROWS + P + 1.2], marker="v", ms=5, color=F.AQUA, clip_on=False
        )
    pa.set_xticks(range(P))
    pa.set_xticklabels(["$m$", "$c$"] * len(CENTRES), fontsize=F.FS_SMALL)
    pa.set_yticks([0, N_ROWS - 1, N_ROWS + P - 1])
    pa.set_yticklabels(["row 1", f"row {N_ROWS}", "ridge rows"], fontsize=F.FS_SMALL)
    for s in pa.spines.values():
        s.set_color(F.AXIS)
    pa.tick_params(length=2, colors=F.INK_2)
    pa.set_title(
        f"(a)  $\\Phi$ for {len(CENTRES)} rules × [1 | $x$], {N_ROWS} rows sorted by $x$",
        fontsize=F.FS_LABEL,
        color=F.INK,
        pad=26,
    )
    pa.text(
        P - 0.4,
        N_ROWS + P / 2 - 0.5,
        f"$\\sqrt{{\\lambda}}\\,\\mathbf{{D}}^{{1/2}}$ appended:\n"
        "ridge as extra rows, so\n`lstsq` sees $\\kappa(\\Phi)$, not $\\kappa(\\Phi)^2$",
        ha="left",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.4,
    )
    pa.text(
        -0.6,
        N_ROWS + P + 1.4,
        "▼ constant columns: $d_p = 0$, never shrunk",
        ha="left",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.shade(F.AQUA, 0.3),
    )

    # -- (b) one solver, three callers ------------------------------------------
    # Laid out by hand in a 0-100 space. No tight_layout on this figure: the
    # texts here are sized for the axes as allotted, and tight_layout would
    # shrink the axes to fit them and make every box too small for its label.
    pb.set_xlim(0, 100)
    pb.set_ylim(0, 100)
    pb.axis("off")
    callers = [
        (84, "flat MoG-TSK FIS (Ch. 4)", "weights: normalized firings $\\bar w_r(x)$"),
        (58, "soft fuzzy tree (§6.3.2)", "weights: leaf routing $w_\\ell(x)$"),
        (
            32,
            "mixture of experts (§6.3.3)",
            "weights: responsibilities $h_{i\\ell}$",
        ),
    ]
    for y, title, body in callers:
        F.box(
            pb,
            27,
            y,
            50,
            16,
            title,
            body,
            color=F.BLUE,
            title_size=F.FS_SMALL,
            body_size=F.FS_SMALL - 1,
        )
        F.arrow(pb, (52, y), (62, 58), color=F.AXIS)
    F.box(
        pb,
        80,
        58,
        34,
        30,
        "one ridge-TSK solve",
        "$\\min_\\beta \\|\\mathbf{y} - \\Phi\\beta\\|^2 + \\lambda\\beta^\\top\\mathbf{D}\\beta$\n"
        "lstsq on $[\\Phi;\\ \\sqrt{\\lambda}\\mathbf{D}^{1/2}]$\n"
        "pinned columns → RHS",
        color=F.ORANGE,
        title_size=F.FS_SMALL,
        body_size=F.FS_SMALL - 1,
    )
    F.arrow(pb, (80, 43), (80, 24), color=F.AXIS)
    F.box(
        pb,
        80,
        14,
        34,
        16,
        "consequents $(m_r, c_r)$",
        "exact for the given weights",
        color=F.AQUA,
        title_size=F.FS_SMALL,
        body_size=F.FS_SMALL - 1,
    )
    pb.text(
        24,
        9,
        "a caller changes only the weights it\n"
        "stacks into $\Phi$; the solve, its conditioning\n"
        "fix and the pinning are shared\n(A.10.16–A.10.18)",
        ha="center",
        va="center",
        fontsize=F.FS_SMALL - 0.5,
        color=F.MUTED,
        linespacing=1.4,
    )
    pb.set_title(
        "(b)  one primitive, three consumers", fontsize=F.FS_LABEL, color=F.INK, pad=8
    )

    fig.subplots_adjust(left=0.10, right=0.99, top=0.84, bottom=0.10, wspace=0.55)
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
