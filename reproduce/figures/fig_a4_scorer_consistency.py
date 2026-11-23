#!/usr/bin/env python3
"""Figure A.4 -- what each feature scorer buys: accuracy, and how consistently it selects.

Two panels over the same fits (Table A.2: PhiUSIIL, 20,000 rows, ten seeds,
`TribbleClassifier(top_n=k, scoring_method=<scorer>)`):

  (a) test accuracy against the number of features kept, one line per scorer;
  (b) the share of the k kept features that every one of the ten seeds'
      fitted models agrees on (`top_features_`, after the 0.85 redundancy
      filter): 1.0 means the selection does not depend on the split at all.
      It is strict by design -- one seed that swaps one feature costs 1/k.

Read together they separate two things a single accuracy column conflates: a
scorer can be accurate with a selection that wanders from split to split, or
stable on a selection that is simply poor. Both panels read the per-fit file
`table_a2_feature_runs.csv`, so they cannot disagree with Table A.2 or A.2b.

Marker shape carries the scorer family (circle: the Gaussian / moment scorers
the library shipped first; square: the rank/bin scorers of tribble-fis #253),
so the grouping survives greyscale and colour-vision deficiency, and the legend
names every series.
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "a4-scorer-consistency"
ARCHIVE = "bump-dd72d5e-2026-11-22"

# Fixed slot order (figstyle.SERIES), assigned to the entity, never to rank.
SCORERS = [
    ("wasserstein", "Wasserstein", F.BLUE, "o"),
    ("bhattacharyya", "Bhattacharyya", F.ORANGE, "o"),
    ("composite", "Composite", F.AQUA, "o"),
    ("higher_order", "Higher-order", F.YELLOW, "o"),
    ("hellinger", "Hellinger", F.MAGENTA, "s"),
    ("jensen_shannon", "Jensen–Shannon", F.GREEN, "s"),
    ("ks", "Kolmogorov–Smirnov", F.VIOLET, "s"),
    ("mutual_info", "Mutual information", F.RED, "s"),
]


def _load():
    rows, label = H.table("table_a2_feature_runs", ARCHIVE)
    acc = defaultdict(list)
    sets = defaultdict(list)
    for r in rows:
        key = (r["scorer"], int(r["k"]))
        acc[key].append(float(r["accuracy"]))
        sets[key].append(set(r["features"].split("|")) - {""})
    ks = sorted({k for _, k in acc})
    return acc, sets, ks, label


def build():
    acc, sets, ks, label = _load()
    fig, (ax_a, ax_c) = F.grid_figure(1, 2, width=F.W_WIDE, height=3.5)

    for key, name, colour, marker in SCORERS:
        if (key, ks[0]) not in acc:
            continue
        a = [sum(acc[(key, k)]) / len(acc[(key, k)]) for k in ks]
        c = [len(set.intersection(*sets[(key, k)])) / k for k in ks]
        style = dict(
            color=colour,
            marker=marker,
            markersize=4.5,
            lw=1.6,
            markeredgecolor=F.SURFACE,
            markeredgewidth=0.8,
            label=name,
            zorder=3,
        )
        ax_a.plot(ks, a, **style)
        ax_c.plot(ks, c, **style)

    F.style_axes(
        ax_a,
        title="(a) Test accuracy",
        xlabel="features kept (top_n)",
        ylabel="accuracy, mean of ten seeds",
    )
    F.style_axes(
        ax_c,
        title="(b) Selection shared by all ten seeds",
        xlabel="features kept (top_n)",
        ylabel="share of kept features in every seed",
    )
    for ax in (ax_a, ax_c):
        ax.set_xticks(ks)
        ax.set_xlim(0, ks[-1] + 1)
    ax_a.set_ylim(0.4, 1.0)
    ax_c.set_ylim(-0.03, 1.05)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        fontsize=F.FS_ANNOT,
        frameon=False,
        labelcolor=F.INK_2,
        bbox_to_anchor=(0.5, 0.0),
        handlelength=2.2,
        columnspacing=1.4,
    )
    fig.text(
        0.5,
        -0.035,
        "PhiUSIIL (leak-free), 20,000 rows, ten 80/20 splits; the same fits as "
        "Tables A.2 and A.2b. Circles: Gaussian/moment scorers; squares: rank/bin\n"
        "scorers (tribble-fis #253), whose lines coincide wherever they overlap. "
        f"{H.provenance_note(label)}",
        ha="center",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
