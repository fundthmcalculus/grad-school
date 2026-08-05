#!/usr/bin/env python3
"""Figure 4.3 -- the correction-rule pass, quantified on Glass.

This figure was retargeted, not merely filled in. It was scoped as a
before/after confusion matrix on RT-IOT2022, and that experiment still cannot
be run: RT-IOT2022 is not among the datasets `reproduce/` can load, and
nothing about that has changed. What has changed is the claim the figure was
actually standing in for -- "the accuracy contribution of the correction pass
has not been isolated" (Ch 4 §4.3.1) -- which `table_4_9_correction_pass.py`
now measures directly, on Glass, ten paired seeds. A confusion matrix on an
unavailable dataset would still be invented; a bar chart of a real paired
measurement on an available one is not, so this figure draws the second thing
instead of continuing to wait for the first.

Three arms, same construction, same splits. **Base** is the flat classifier
with no correction pass. **Gated cascade** adds
`MixtureOfGaussiansFuzzySequenceClassifier`'s confused-pair expert layers,
routed by anomaly level and confidence margin -- this is what §4.3.1 means by
"the correction pass." **-> Flat** unions every layer's membership functions
into one `GaussianMixtureModel` (`.augment()` twice), deduplicates at exact
tolerance (`rtol=atol=0`, no numeric merging -- see Table 4.8 for what
tolerance costs), and predicts by a single plain argmax: one deployable FIS,
gating logic removed.

Reading the two panels together is the point. The cascade panel's own MF count
carries "raw" in its cell because nothing has been merged yet; the flat panel's
count is what survives exact-tolerance dedup. Accuracy moves much less than MF
count does: the gated cascade buys +0.031 over base, flattening keeps +0.014 of
that, well under half -- so the gating logic the flattened arm discards was
doing real work, not decoration.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "04-rtiot-confusion"   # prose/fig basename is unchanged; only the content moves

# Named rather than inferred: this archive carries three tables and would
# never be "newest" by luck, but a figure this small should not depend on
# that being true forever either.
ARCHIVE = "mf-dedup-2026-08-05"

ARMS = ["Base (no correction pass)",
        "Gated cascade (base + experts, routed)",
        "Cascade → one flat FIS (union, dedup @ exact tol., argmax)"]
SHORT = ["Base\nno pass", "Cascade\nraw, gated", "→ Flat\ndeduped"]
COLOUR = [F.MUTED, F.BLUE, F.ORANGE]


def _load():
    rows, label = H.table("table_4_9_correction_pass", ARCHIVE)
    by_arm = {r["Arm"]: r for r in rows}
    return [by_arm[a] for a in ARMS], label


def _bar_panel(ax, values, spreads, title, ylabel, fmt):
    x = range(len(values))
    bars = ax.bar(x, values, color=[F.tint(c, 0.15) for c in COLOUR],
                  edgecolor=[F.shade(c, 0.15) for c in COLOUR],
                  linewidth=1.0, width=0.62, zorder=3)
    ax.errorbar(x, values, yerr=spreads, fmt="none", ecolor=F.INK_2,
               elinewidth=1.0, capsize=3, capthick=1.0, zorder=4)
    for xi, v, s in zip(x, values, spreads):
        ax.text(xi, v + s + (max(values) * 0.03), fmt.format(v),
                ha="center", va="bottom", fontsize=F.FS_SMALL, color=F.INK_2)
    ax.set_xticks(list(x))
    ax.set_xticklabels(SHORT, fontsize=F.FS_TICK)
    F.style_axes(ax, title=title, ylabel=ylabel, grid_axis="y")
    ax.set_ylim(0, max(values) * 1.28)
    return bars


def build():
    rows, label = _load()
    mf = [H.number(r["MF count"]) for r in rows]
    mf_sd = [H.spread(r["MF count"]) for r in rows]
    acc = [H.number(r["Accuracy"]) for r in rows]
    acc_sd = [H.spread(r["Accuracy"]) for r in rows]
    delta = [H.number(r["Paired Δ vs. base"]) for r in rows]
    delta_sd = [H.spread(r["Paired Δ vs. base"]) for r in rows]

    fig, (ax_mf, ax_acc) = F.grid_figure(1, 2, width=F.W_WIDE, height=3.6)

    _bar_panel(ax_mf, mf, mf_sd, "Membership-function count", "MF count",
              "{:.1f}")
    _bar_panel(ax_acc, acc, acc_sd, "Accuracy, and the paired Δ vs. base",
              "accuracy", "{:.3f}")
    for xi, d, ds in zip(range(1, 3), delta[1:], delta_sd[1:]):
        ax_acc.text(xi, 0.05, f"Δ {d:+.3f}\n± {ds:.3f}", ha="center",
                    va="bottom", fontsize=F.FS_SMALL, color=F.shade(F.BLUE, 0.3),
                    transform=ax_acc.get_xaxis_transform(), linespacing=1.4)

    fig.text(0.5, -0.04,
             "Glass, ten paired seeds, same splits across arms. Base -> gated cascade "
             "is §4.3.1's correction pass; cascade -> flat is Table 4.8's exact-tolerance "
             "dedup applied to the union of every layer. "
             f"{H.provenance_note(label)}",
             ha="center", va="top", fontsize=F.FS_SMALL, color=F.MUTED,
             linespacing=1.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
