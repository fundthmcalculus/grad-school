#!/usr/bin/env python3
"""Figure 5.2 -- band discovery on the log-birth spectrum, and what it recovers.

The chapter says this is its key figure, and the review agrees, so it runs the
real selector: `multiscale_persistence.select_multiscale` on the minimax
transform of `battery_hierarchical.three_level_hierarchy`, which is the same
dataset and the same call behind Table 5.2's `three_level_hierarchy` row.
Nothing here is staged -- the band edges drawn are the edges the algorithm
found, and the granularities in the panel titles are `sel.granularities()`.

Two rendering decisions.

**The partitions are strips, not scatters.** The construction separates its
levels by factors of 200, 28 and 5, so a 2-D scatter of it is one visible pair
of dots: the fine structure this figure exists to show is smaller than the
marker. Ordering the samples by their ground-truth fine label and drawing each
recovered partition as a row of coloured cells shows all three levels at once,
and shows the nesting -- eight runs collapsing to four, then to two -- which a
scatter could not.

**The spectrum is drawn as blocks on a log axis with the gaps shaded**, because
the gaps are what the method keys on. A histogram of birth heights would bury
them: the interesting feature is empty space, and a histogram draws empty space
as nothing at all.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "05-band-discovery"

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "gated-minimax-selection"))

DATASET = "three_level_hierarchy"


def build():
    import battery_hierarchical as BH   # noqa: E402
    import ivat_mf as im                # noqa: E402
    import multiscale_persistence as MS  # noqa: E402
    from sklearn.metrics import adjusted_rand_score

    X, y_fine, y_med, y_coarse = BH.three_level_hierarchy()
    Dstar = im.minimax_transform(im.dissimilarity(X))
    sel = MS.select_multiscale(Dstar)

    # Bands come back fine -> coarse; the truth levels are named the same way.
    truths = [("fine", y_fine), ("medium", y_med), ("coarse", y_coarse)]
    order = np.argsort(y_fine, kind="stable")

    fig = F._pyplot().figure(figsize=(F.W_WIDE, 4.4), dpi=F.DPI)
    fig.patch.set_facecolor(F.SURFACE)
    gs = fig.add_gridspec(len(sel.bands), 2, width_ratios=[1.15, 1],
                          hspace=0.55, wspace=0.22)

    # -- left: the spectrum, spanning the full height ------------------------
    spec = fig.add_subplot(gs[:, 0])
    spec.set_facecolor(F.SURFACE)

    blocks, n = _significant(MS, Dstar)
    births = np.array([b["birth"] for b in blocks])
    sizes = np.array([b["size"] for b in blocks])
    spec.scatter(births, sizes, s=26, color=F.tint(F.BLUE, 0.35), linewidths=0,
                 zorder=3)

    edges = [float(np.exp(e)) for e in sel.band_edges_log]
    lo, hi = births.min() / 2.2, births.max() * 2.2
    bounds = [lo] + edges + [hi]
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        if i < len(sel.bands):
            spec.axvspan(a, b, color=F.tint(F.SERIES[i % 8], 0.93), zorder=0)
            spec.text(np.sqrt(a * b), sizes.max() * 1.32,
                      f"band {i + 1}\n$k$ = {sel.bands[i].k}", ha="center",
                      va="center", fontsize=F.FS_SMALL,
                      color=F.shade(F.SERIES[i % 8], 0.3), linespacing=1.5)
    for e in edges:
        spec.axvline(e, lw=1.1, ls=(0, (3, 2)), color=F.FAINT, zorder=2)

    spec.set_xscale("log")
    F.style_axes(spec, title="(a)  the log-birth spectrum",
                 xlabel="block birth height  (log scale)",
                 ylabel="block size (points)")
    spec.set_xlim(lo, hi)
    spec.set_ylim(0, sizes.max() * 1.55)

    # -- right: one strip per recovered band ---------------------------------
    for i, band in enumerate(sel.bands):
        ax = fig.add_subplot(gs[i, 1])
        ax.set_facecolor(F.SURFACE)
        assigned = MS.assign_band(band, Dstar)
        # Relabel by first appearance along the strip, so that spatially adjacent
        # runs take slot-adjacent hues -- which is the pairlist the categorical
        # palette is validated against.
        seq = assigned[order]
        remap, nxt = {}, 0
        for v in seq:
            if v not in remap:
                remap[v] = nxt
                nxt += 1
        labels = np.array([remap[v] for v in seq])
        ax.imshow(labels[None, :], aspect="auto", interpolation="nearest",
                  cmap=_partition_cmap(band.k), vmin=0, vmax=max(band.k - 1, 1),
                  rasterized=True)
        # A surface-coloured rule at each boundary: the 2px gap between fills.
        for boundary in np.flatnonzero(np.diff(labels)) + 1:
            ax.axvline(boundary - 0.5, color=F.SURFACE, lw=1.6, zorder=4)
        ax.set_yticks([])
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_color(F.AXIS)
            s.set_linewidth(0.8)

        name, truth = truths[i] if i < len(truths) else ("", None)
        ari = adjusted_rand_score(truth, assigned) if truth is not None else float("nan")
        panel = "(b)  " if i == 0 else ""
        ax.set_title(f"{panel}band {i + 1} — {band.k} clusters recovered   "
                     f"(vs {name} truth: ARI {ari:.2f})",
                     fontsize=F.FS_SMALL, color=F.INK, pad=5)

    fig.text(0.5, -0.02,
             f"Left: one dot per persistence-significant block, placed at its birth height. "
             f"Bands are the runs between gaps in the log-birth axis, and the "
             f"granularities\n{sel.granularities()} fall out of the selection rather "
             f"than being asked for. Right: the partition each band recovers, samples "
             f"ordered by the finest ground-truth label,\nso a recovered cluster is a "
             f"contiguous run of colour and the nesting is visible. "
             f"{DATASET}, $n$ = {n}; the same dataset and call as Table 5.2.",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.6)
    return fig


def _significant(MS, Dstar):
    """The blocks the gate admits -- the population band discovery runs on."""
    import selection as S
    blocks, n = S._all_blocks(Dstar)
    return MS.significant_blocks(blocks, n), n


def _partition_cmap(k):
    """Categorical slots in fixed order, as a discrete colormap."""
    from matplotlib.colors import ListedColormap
    return ListedColormap([F.SERIES[i % len(F.SERIES)] for i in range(max(k, 2))])


if __name__ == "__main__":
    F.save(build(), NAME)
