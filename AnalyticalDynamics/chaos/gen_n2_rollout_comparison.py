"""Generate n2_rollout_comparison_all.png: real FIS predictions vs truth, training edge.

Fits the same four time-as-input configurations Table 4 reports (40 rules, 120
rules, 300 rules, 8-harmonic encoding) via `compare_families.run_with_time` and
plots each one's *actual* predicted trajectory against a DOP853 reference.

This previously synthesized the divergence with hand-tuned noise
(`error_scale * (exp(0.3 t) - 1)`) rather than running the model -- illustrative,
but not a reproduction of anything. It now shows what the models actually do,
titled with the real RMSE each run computes, not a transcription of Table 4.
"""

from __future__ import annotations

import numpy as np

import compare_families as cf
import pendulum_data as pdata
import plots

DURATION = pdata.TEST_T_END
N_STEPS = int(round(DURATION / pdata.H))

#: Display clip for the extrapolation region only -- 300 rules reaches ~338,555
#: degrees past the window (Table 4), which would flatten every other curve to a
#: line at this axis scale. The clip is cosmetic: titles report the true RMSE,
#: computed from the unclipped prediction.
YLIM_PAD_DEG = 60.0


def main():
    split = cf.fts.load(2, friction=True)
    t = np.arange(N_STEPS) * pdata.H
    inw = t < pdata.T_END

    state0 = np.array(
        [np.deg2rad(pdata.THETA1_DEG), 0.0, np.deg2rad(pdata.TEST_THETA2_DEG), 0.0]
    )
    rhs = pdata.rhs_for(2, friction=True)
    truth = pdata.integrate_dop853(rhs, state0, t)
    truth_deg = np.rad2deg(truth[:, [0, 2]])

    print(
        f"Friction double pendulum, trained on {pdata.T_END:.0f} s, "
        f"scored to {DURATION:.0f} s against a DOP853 reference.\n"
    )
    rows, preds = cf.run_with_time(split, truth_deg, t, inw)

    fig, axes = plots.plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor(plots.SURFACE)
    axes = axes.flatten()

    lo = float(truth_deg[:, 0].min()) - YLIM_PAD_DEG
    hi = float(truth_deg[:, 0].max()) + YLIM_PAD_DEG

    for ax, (label, _cfg), row in zip(axes, cf.WITH_TIME, rows):
        pred_deg = np.clip(preds[label][:, 0], lo, hi)
        ax.plot(
            t,
            truth_deg[:, 0],
            color=plots.INK,
            linewidth=2.5,
            label="Truth (DOP853)",
            zorder=3,
        )
        ax.plot(
            t,
            pred_deg,
            color=plots.RED,
            linewidth=1.6,
            label="FIS prediction (clipped)",
            zorder=2,
        )
        ax.axvline(
            pdata.T_END, color=plots.RED, linestyle=":", linewidth=1.5, alpha=0.7
        )
        ax.fill_between([pdata.T_END, t[-1]], lo, hi, alpha=0.08, color=plots.RED)
        ax.set_ylim(lo, hi)
        plots._style(
            ax,
            title=(
                f"{label}: in-window {row['in_window_rmse_deg']:.3g}°, "
                f"extrap {row['extrap_rmse_deg']:.4g}°"
            ),
            xlabel="Time (s)",
            ylabel="θ₁ (degrees)",
        )
        ax.legend(loc="best", fontsize=plots.FS_SMALL, frameon=False)

    fig.suptitle(
        "Friction double pendulum: FIS diverges at the training-window edge (t = 10 s)",
        fontsize=plots.FS_TITLE,
        color=plots.INK,
        y=0.995,
    )
    fig.tight_layout()
    path = plots._save(fig, "n2_rollout_comparison_all")
    print(f"\nwrote {path}")
    return rows


if __name__ == "__main__":
    main()
