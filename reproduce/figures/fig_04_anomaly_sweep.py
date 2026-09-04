#!/usr/bin/env python3
"""Figure 4.6 -- the open-set operating curve, plotted from Table 4.6's own CSV.

Reads `table_4_4b_theta_sweep.csv` from the archive of record rather than
re-running the sweep. That is the right dependency: the table and the figure are
the same experiment, and re-running would give a second set of numbers to keep
in agreement with the first for no gain. Regenerate the table (with
`REPRO_THETA_SWEEP=1`) and this figure follows.

The two rates share an axis because they are the same kind of quantity, so
Youden's J -- their difference, which is the column the table reports and the
quantity §4.4 argues about -- is simply the vertical gap between the curves. It
is shaded rather than drawn as a third line, and the best operating point is
marked with its value.

Three things the prose says about this curve are annotated on it, because they
are the reasons to show a curve instead of a number: the knob is monotone, the
inherited default of theta = 0.99 is a poor operating point, and past theta =
1.1 the boost saturates the aggregate and the rule stops firing at all.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as F  # noqa: E402
import harness_data as H  # noqa: E402

NAME = "04-anomaly-sweep"

DEFAULT_THETA = 0.99  # the value inherited from the BETH configuration


def build():
    rows, label = H.table("table_4_4b_theta_sweep")
    theta = [H.number(r["θ"]) for r in rows]
    detection = [H.number(r["detection rate"]) for r in rows]
    false_alarm = [H.number(r["false-alarm rate"]) for r in rows]
    j = [d - f for d, f in zip(detection, false_alarm)]

    fig, ax = F.figure(width=F.W_COL + 1.1, height=3.8)

    ax.fill_between(
        theta, false_alarm, detection, color=F.tint(F.BLUE, 0.90), zorder=1, linewidth=0
    )
    ax.plot(
        theta,
        detection,
        marker="o",
        ms=4.5,
        lw=1.8,
        color=F.BLUE,
        label="detection rate (unseen class)",
        zorder=4,
    )
    ax.plot(
        theta,
        false_alarm,
        marker="o",
        ms=4.5,
        lw=1.8,
        color=F.ORANGE,
        label="false-alarm rate (known classes)",
        zorder=4,
    )

    best = max(range(len(j)), key=lambda i: j[i])
    ax.annotate(
        f"best $J$ = {j[best]:+.3f}  at  $\\theta$ = {theta[best]:.2f}",
        xy=(theta[best], (detection[best] + false_alarm[best]) / 2),
        xytext=(theta[best] + 0.02, 0.90),
        fontsize=F.FS_SMALL,
        color=F.shade(F.BLUE, 0.3),
        ha="left",
        arrowprops=dict(arrowstyle="-", lw=0.8, color=F.AXIS),
    )

    ax.axvline(DEFAULT_THETA, lw=1.0, ls=(0, (3, 2)), color=F.FAINT, zorder=2)
    default_j = j[min(range(len(theta)), key=lambda i: abs(theta[i] - DEFAULT_THETA))]
    ax.text(
        1.015,
        0.78,
        f"inherited default $\\theta$ = {DEFAULT_THETA}\n"
        f"$J$ = {default_j:+.3f} — about seven-\ntenths of what is available",
        ha="left",
        va="center",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.6,
    )

    saturated = [
        t for t, d, f in zip(theta, detection, false_alarm) if d == 0 and f == 0
    ]
    if saturated:
        ax.annotate(
            f"past $\\theta$ = {min(saturated):.1f} the boost\n"
            f"saturates the aggregate —\nthe rule stops firing",
            xy=(min(saturated), 0.015),
            xytext=(1.015, 0.30),
            ha="left",
            va="center",
            fontsize=F.FS_SMALL,
            color=F.MUTED,
            linespacing=1.6,
            arrowprops=dict(arrowstyle="->", lw=0.8, color=F.AXIS),
        )

    F.style_axes(
        ax,
        title="Open-set operating curve on Glass, leave-one-class-out",
        xlabel="anomaly boost  $\\theta$",
        ylabel="rate",
    )
    ax.set_xlim(0.455, 1.30)
    ax.set_ylim(-0.03, 1.0)
    F.legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    # The forgiving-knob claim, restricted to the range the prose makes it over.
    flat = [v for t, v in zip(theta, j) if 0.5 <= t <= 0.8]
    ax.text(
        0.0,
        -0.30,
        "The shaded band is $J$ = detection $-$ false alarm, the column Table 4.6 "
        f"reports. Across $\\theta \\in [0.5, 0.8]$ it stays\nbetween "
        f"{min(flat):+.3f} and {max(flat):+.3f}, so the choice within that range "
        "is nearly free — the knob is forgiving rather than\ndelicate, which is "
        f"the argument for reporting a curve instead of a number. "
        f"{H.provenance_note(label)}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=F.FS_SMALL,
        color=F.MUTED,
        linespacing=1.6,
    )
    return fig


if __name__ == "__main__":
    F.save(build(), NAME)
