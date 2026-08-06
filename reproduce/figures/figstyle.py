"""Shared style for every proposal figure.

Sixteen figures are called out in the proposal. Before this module there was
exactly one real generator -- Chapter 3's complexity fit, written inline in
`tables/table_3_1_reorder_three_arm.py` -- and fifteen 640x480 grey placeholder
PNGs. Producing the rest one script at a time would have produced fifteen
figures that agree about nothing: different fonts, different blues, different
sizes on the page.

So the style lives here and the generators stay thin. A generator decides *what
to draw*; this file decides what it looks like. The values below are lifted from
the complexity-fit figure, which was already drawn against the validated
palette, so the fifteen new figures match the one that existed.

Two constraints shape the choices, both from `common.save_figure`:

  * **No alpha.** EPS has no alpha channel, so a transparent fill flattens
    differently in the two formats and the PNG and EPS drift apart. Every soft
    fill here is a *solid* colour precomputed by `tint()` -- blended against the
    surface at authoring time rather than at export time.
  * **Vector where possible.** Raster panels (the VAT images of Ch 2 and Ch 5)
    are unavoidable, and `imshow_matrix` keeps them small enough that the EPS
    stays a reasonable size. Everything else is pure vector.

Print figures are light-mode only. There is no dark variant, because a
dissertation is printed on paper.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # reproduce/ -> `import common`
import common as C  # noqa: E402

# Matplotlib stamps a %%CreationDate into every EPS, so re-running an unchanged
# generator produced a diff in fourteen tracked files and told you nothing. The
# figures are tracked artifacts; a diff should mean the picture changed.
# Matplotlib honours SOURCE_DATE_EPOCH for its PS/PDF metadata, so pinning it
# makes the output byte-reproducible. `setdefault`, so a caller that has set it
# for its own reasons wins.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1735689600")  # 2025-01-01T00:00:00Z

# --------------------------------------------------------------------------- #
# palette -- validated categorical slots, light mode (references/palette.md)
# --------------------------------------------------------------------------- #
# Fixed order, never cycled. A figure needing a ninth series does not get a
# ninth hue; it gets faceted or folded into "other".
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",
]  # 8 red

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES

# Sequential blue ramp, light -> dark. Used for magnitude (the VAT images).
SEQ_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]

SURFACE = "#fcfcfb"  # chart surface
INK = "#0b0b0b"  # primary text
INK_2 = "#3d3d39"  # tick labels
MUTED = "#6b6b63"  # annotation text
GRID = "#e2e2dc"  # gridlines
AXIS = "#c9c9c1"  # spines, connectors
FAINT = "#9a9a92"  # reference curves, de-emphasised marks

# Figure widths, in inches, matched to the document's text block. Two sizes
# only -- a figure that needs a third is usually two figures.
W_COL = 5.4  # single column, the complexity-fit width
W_WIDE = 7.2  # full text width, for side-by-side panels

DPI = 200

# Type scale. Small, because these are reduced onto a printed page: a 9pt label
# in a 5.4in figure lands at roughly 9pt on paper.
FS_TITLE = 10
FS_LABEL = 9
FS_TICK = 8
FS_ANNOT = 8
FS_SMALL = 7


# --------------------------------------------------------------------------- #
# colour helpers
# --------------------------------------------------------------------------- #
def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(rgb):
    return "#" + "".join(f"{max(0, min(255, int(round(c)))):02x}" for c in rgb)


def tint(color, amount=0.85, toward=SURFACE):
    """Blend `color` toward the surface and return a SOLID hex.

    `amount` is how much of the surface to mix in: 0.0 returns the colour
    unchanged, 1.0 returns the surface. This exists so that a soft fill is a
    real colour rather than an alpha, which is what keeps the PNG and the EPS
    identical -- see the module docstring.
    """
    a, b = _hex_to_rgb(color), _hex_to_rgb(toward)
    return _rgb_to_hex([x + (y - x) * amount for x, y in zip(a, b)])


def shade(color, amount=0.25):
    """Blend `color` toward black -- for an edge that has to read against its fill."""
    return tint(color, amount, toward="#000000")


def blue_cmap(reverse=True):
    """The sequential blue ramp as a matplotlib colormap.

    Default `reverse=True` gives dark-for-small, which is the VAT convention:
    a dark diagonal block is a set of mutually *close* points.
    """
    from matplotlib.colors import LinearSegmentedColormap

    steps = list(reversed(SEQ_BLUE)) if reverse else list(SEQ_BLUE)
    return LinearSegmentedColormap.from_list("tribble_blue", steps)


# --------------------------------------------------------------------------- #
# figure construction
# --------------------------------------------------------------------------- #
def _pyplot():
    """Import pyplot with a headless backend, once, at call time.

    Deferred rather than done at module import so that `make_figures.py` can
    import the registry -- and print what it would run -- on a machine with no
    matplotlib installed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def figure(width=W_COL, height=4.0, **kwargs):
    """A single-axes figure at the house size and dpi."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(width, height), dpi=DPI, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def grid_figure(nrows, ncols, width=W_WIDE, height=4.0, **kwargs):
    """A multi-panel figure at the house size and dpi."""
    plt = _pyplot()
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), dpi=DPI, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    for ax in (axes.ravel() if hasattr(axes, "ravel") else [axes]):
        ax.set_facecolor(SURFACE)
    return fig, axes


def canvas(width=W_WIDE, height=4.0, xlim=(0, 100), ylim=(0, 100)):
    """A blank drawing surface in arbitrary units -- for the schematics.

    Block diagrams are laid out by hand in a 0-100 coordinate space rather than
    by a plotting routine, so the axes, ticks and spines are all removed and
    the aspect is left free.
    """
    fig, ax = figure(width, height)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax


def style_axes(ax, title=None, xlabel=None, ylabel=None, grid=True, grid_axis="both"):
    """The recessive-furniture treatment: no top/right spine, faint grid, small type."""
    if title:
        ax.set_title(title, fontsize=FS_TITLE, color=INK, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL, color=INK_2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL, color=INK_2)
    if grid:
        ax.grid(True, axis=grid_axis, which="major", lw=0.4, color=GRID, zorder=0)
        ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=FS_TICK, colors=INK_2, length=3, width=0.8)
    return ax


def legend(ax, **kwargs):
    """A frameless legend in the house type size.

    Present whenever a panel carries two or more series -- identity is never
    left to colour alone.
    """
    opts = dict(fontsize=FS_ANNOT, frameon=False, labelcolor=INK_2)
    opts.update(kwargs)
    leg = ax.legend(**opts)
    return leg


def panel_label(ax, text, x=0.0, y=1.02):
    """The '(a)' / '(b)' marker above a panel in a multi-panel figure."""
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=FS_LABEL,
        color=INK,
        va="bottom",
        ha="left",
        fontweight="bold",
    )


# --------------------------------------------------------------------------- #
# schematic primitives
# --------------------------------------------------------------------------- #
def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    body=None,
    color=BLUE,
    fill_amount=0.88,
    edge_amount=0.0,
    title_size=FS_LABEL,
    body_size=FS_SMALL,
    radius=1.6,
    lw=1.2,
    zorder=2,
    title_weight="bold",
    dashed=False,
):
    """A rounded box with a bold title and optional body text, centred on (x, y).

    `x, y` is the CENTRE, which makes a hand-laid-out diagram far easier to keep
    aligned than corner coordinates would.
    """
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=tint(color, fill_amount),
        edgecolor=color if edge_amount == 0.0 else shade(color, edge_amount),
        linewidth=lw,
        zorder=zorder,
        linestyle=(0, (3, 2)) if dashed else "solid",
    )
    ax.add_patch(patch)

    if body:
        ax.text(
            x,
            y + h * 0.20,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=INK,
            fontweight=title_weight,
            zorder=zorder + 1,
        )
        ax.text(
            x,
            y - h * 0.20,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=MUTED,
            zorder=zorder + 1,
            linespacing=1.35,
        )
    else:
        ax.text(
            x,
            y,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=INK,
            fontweight=title_weight,
            zorder=zorder + 1,
            linespacing=1.35,
        )
    return patch


def arrow(
    ax,
    start,
    end,
    color=AXIS,
    lw=1.4,
    label=None,
    label_offset=(0, 2.2),
    label_color=MUTED,
    style="-|>",
    zorder=1,
    connection="arc3,rad=0",
    label_size=FS_SMALL,
):
    """A connector between two points, optionally labelled at its midpoint."""
    from matplotlib.patches import FancyArrowPatch

    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        color=color,
        lw=lw,
        zorder=zorder,
        connectionstyle=connection,
        shrinkA=1.5,
        shrinkB=1.5,
    )
    ax.add_patch(patch)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(
            mx + label_offset[0],
            my + label_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=label_size,
            color=label_color,
            zorder=zorder + 2,
        )
    return patch


def badge(ax, x, y, text, color=BLUE, size=FS_SMALL, pad=0.32):
    """A small filled pill -- for the '(Ch. 3)' and 'ARI 1.00' annotations."""
    return ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=shade(color, 0.35),
        zorder=6,
        bbox=dict(
            boxstyle=f"round,pad={pad}",
            facecolor=tint(color, 0.86),
            edgecolor=tint(color, 0.55),
            linewidth=0.7,
        ),
    )


def caption(ax, text, y=-0.02, size=FS_SMALL, color=MUTED):
    """A note under a panel, in figure-relative coordinates."""
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=size,
        color=color,
        linespacing=1.4,
    )


def imshow_matrix(ax, M, cmap=None, title=None, vmin=None, vmax=None):
    """A dissimilarity-matrix panel: square, no ticks, rasterised for the EPS.

    `rasterized=True` matters. An N x N `imshow` exported as vector EPS becomes
    N^2 filled rectangles; at N = 300 that is a 90,000-path file measured in
    tens of megabytes. Rasterising the image while leaving the frame and labels
    vector keeps the EPS small and the type sharp.
    """
    im = ax.imshow(
        M,
        cmap=cmap or blue_cmap(),
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(AXIS)
        s.set_linewidth(0.8)
    if title:
        ax.set_title(title, fontsize=FS_LABEL, color=INK, pad=6)
    return im


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def harness_name(prose_name):
    """`03-complexity-fit` -> `fig_03_complexity_fit`.

    The one figure that already existed established this mapping; deriving it
    rather than tabulating it means a new figure cannot be added with an
    inconsistent name.
    """
    return "fig_" + prose_name.replace("-", "_")


def save(fig, prose_name):
    """Write PNG + EPS into the harness figure directory.

    Deliberately NOT written straight into `prose/fig/`. Every experimental
    artifact in this project lands under `reproduce/outputs/` first, so that
    what the document shows can always be traced to a harness run; the copy into
    the document is a separate, explicit step (`make_figures.py --install`, or
    `build_pdf.py` at build time).
    """
    paths = C.save_figure(fig, harness_name(prose_name))
    _pyplot().close(fig)
    return paths
