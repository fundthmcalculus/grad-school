#!/usr/bin/env python3
"""Figure 5.3 -- the membership function read off the hierarchy, and a caveat.

§5.3.4 gives the persistence ramp

    mu_B(x) = clip((death_B - d_B(x)) / (death_B - birth_B), 0, 1)

and says, rightly, that every term in it is a merge height the hierarchy
already supplies -- nothing is fitted. The figure draws that. It also draws
something the prose does not mention and a reader plotting this themselves would
hit immediately.

**The ramp is crisp on an ultrametric.** d_B(x) is the bottleneck height at
which x joins B. For a member that is at most birth_B; for a non-member it is at
least death_B, because death_B is by definition the height at which B first
absorbs anything outside itself. So no point can land strictly inside the ramp,
and mu_B takes only the values 1 and 0. This is not an implementation
shortcoming -- it follows from the geometry -- and `multiscale_persistence`
knows it: the docstring of `block_membership` records the ramp as "CRISP by
construction ... kept for the record", and the shipped default is a Gaussian in
minimax distance with half-max at the death height, which grades the non-member
skirt and is genuinely fuzzy.

So the figure shows both, with the data on the axis: the ramp with every sample
plotted at its own d_B, which makes the empty interval visible rather than
asserted, and the shipped kernel over the same axis. The right panel is the
resulting partition either way -- membership is still read off the hierarchy,
which is §5.3.4's actual claim and survives intact.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402

NAME = "05-persistence-ramp"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gated-minimax-selection"))


def build():
    import battery_hierarchical as BH  # noqa: E402
    import ivat_mf as im  # noqa: E402
    import multiscale_persistence as MS  # noqa: E402

    X, y_fine, y_med, y_coarse = BH.three_level_hierarchy()
    Dstar = im.minimax_transform(im.dissimilarity(X))
    sel = MS.select_multiscale(Dstar)

    band = sel.bands[0]  # the finest scale
    block = max(band.blocks, key=lambda b: b["persistence"])
    members = np.fromiter(block["members"], dtype=int)
    d = Dstar[members, :].min(axis=0)  # d_B(x) for every point
    birth, death = block["birth"], block["death"]

    fig, (curve_ax, part_ax) = F.grid_figure(
        1, 2, width=F.W_WIDE, height=3.4, gridspec_kw={"width_ratios": [1.25, 1]}
    )

    # -- left: the two membership functions over d_B --------------------------
    # symlog on the distance axis. Members sit at exactly d = 0 so a pure log
    # scale cannot show them, and a linear one cannot show anything else: the
    # non-members in this construction run out to d = 200 while birth and death
    # are under 6, which squashes the whole interesting region into one pixel.
    grid = np.concatenate(
        [np.linspace(0, death * 2, 500), np.geomspace(death * 2, d.max() * 1.15, 300)]
    )
    ramp = np.clip((death - grid) / (death - birth), 0.0, 1.0)
    gaussian = np.exp(-np.log(2.0) * (grid / death) ** 2)

    curve_ax.plot(grid, ramp, lw=2.0, color=F.BLUE, label="persistence ramp, §5.3.4")
    curve_ax.plot(
        grid, gaussian, lw=2.0, ls=(0, (4, 2)), color=F.ORANGE, label="shipped kernel"
    )

    # The interval the ramp slopes across, shaded so its emptiness is a region
    # rather than a claim in a sentence.
    curve_ax.axvspan(birth, death, color=F.tint(F.FAINT, 0.94), zorder=0)
    for x, style, align, dx in (
        (birth, "birth", "right", -0.04),
        (death, "death", "left", 0.06),
    ):
        curve_ax.axvline(x, lw=1.0, ls=(0, (2, 2)), color=F.FAINT, zorder=1)
        curve_ax.text(
            x + dx * x,
            1.19,
            f"{style}$_B$ = {x:.2f}",
            ha=align,
            va="center",
            fontsize=F.FS_SMALL,
            color=F.MUTED,
        )

    # Every sample at its own d_B. The gap between the two clumps is the point.
    inside = d <= birth + 1e-9
    curve_ax.plot(
        d[inside],
        np.full(inside.sum(), -0.11),
        marker="|",
        ls="none",
        ms=6,
        mew=0.9,
        color=F.BLUE,
        zorder=4,
    )
    curve_ax.plot(
        d[~inside],
        np.full((~inside).sum(), -0.11),
        marker="|",
        ls="none",
        ms=6,
        mew=0.9,
        color=F.FAINT,
        zorder=3,
    )
    curve_ax.annotate(
        "no sample lands in the\nshaded interval — on an\n"
        "ultrametric a non-member\njoins at height $\\geq$ death$_B$",
        xy=((birth + death) / 2, -0.11),
        xytext=(death * 3.0, 0.34),
        ha="left",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.6,
        arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS),
    )

    F.style_axes(
        curve_ax,
        title=f"(a)  one block's membership, $|B|$ = {len(members)}",
        xlabel="minimax distance to the block,  $d_B(x)$   "
        "(symlog: linear below the death height)",
        ylabel="$\\mu_B(x)$",
    )
    curve_ax.set_xscale("symlog", linthresh=death)
    curve_ax.set_xlim(-death * 0.06, d.max() * 1.2)
    curve_ax.set_ylim(-0.18, 1.28)
    curve_ax.set_yticks([0, 0.5, 1.0])
    F.legend(curve_ax, loc="upper right", bbox_to_anchor=(1.0, 0.88))

    # -- right: the partition the band's memberships produce ------------------
    U = MS.band_memberships(band, Dstar)
    order = np.argsort(y_fine, kind="stable")
    im_ = part_ax.imshow(
        U[:, order],
        aspect="auto",
        interpolation="nearest",
        cmap=F.blue_cmap(reverse=False),
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    part_ax.set_yticks(range(len(band.blocks)))
    part_ax.set_yticklabels(
        [f"block {i + 1}" for i in range(len(band.blocks))], fontsize=F.FS_SMALL
    )
    part_ax.set_xticks([])
    for s in part_ax.spines.values():
        s.set_color(F.AXIS)
        s.set_linewidth(0.8)
    part_ax.set_title(
        f"(b)  the band's fuzzy partition, {len(band.blocks)} blocks "
        f"× {Dstar.shape[0]} points",
        fontsize=F.FS_LABEL,
        color=F.INK,
        pad=6,
    )
    part_ax.set_xlabel(
        "samples, ordered by ground-truth fine label",
        fontsize=F.FS_SMALL,
        color=F.INK_2,
    )
    cbar = fig.colorbar(im_, ax=part_ax, fraction=0.035, pad=0.02)
    cbar.outline.set_edgecolor(F.AXIS)
    cbar.outline.set_linewidth(0.8)
    cbar.ax.tick_params(labelsize=F.FS_SMALL, colors=F.INK_2, length=2.5, width=0.7)
    cbar.set_label("$\\mu$", fontsize=F.FS_SMALL, color=F.INK_2)

    fig.text(
        0.5,
        -0.02,
        "Both curves are read off the hierarchy: birth and death are merge "
        "heights the dendrogram already supplies, and nothing is fitted — which "
        "is §5.3.4's claim,\nand it holds for either. What the rug shows is that "
        "the ramp cannot be the fuzzy one: on an ultrametric its sloped interval "
        "is empty by construction, so it takes\nonly 0 and 1. The shipped kernel "
        "grades the non-member skirt instead, and argmax over it still reproduces "
        "the crisp labels of panel (b).",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.6,
    )
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
