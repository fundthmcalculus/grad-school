"""Why the frictionless held-out initial condition is unlearnable, measured.

The paper's time-step operator interpolates between initial conditions: the
held-out IC [120, 2.05] sits exactly between the trained ICs [120, 2.0] and
[120, 2.1]. Interpolation can only work while those two neighbours still
resemble each other. This script measures how long that holds.

Once the two bracketing training trajectories have separated by a large fraction
of the holdout's own angular range, the training set offers two mutually
contradictory answers for the same query and nothing to choose between them. Any
model is then pushed toward the conditional mean over the initial-condition grid,
which caps achievable R^2 well below 1 regardless of architecture. That is a
property of the dataset, not of the model, and it is why every model in the paper
scores ~0.2 on the frictionless holdout and ~0.99 with friction.

Two reference errors are reported for scale, neither of which is a lower bound:
`midpoint_rmse_scaled` (average the two bracketing trained trajectories) and
`constant_mean_rmse_scaled` (predict the trajectory's own mean, i.e. R^2 = 0).

Writes results/bracket.csv and figures/fig_bracket.png.
"""

from __future__ import annotations

import csv

import numpy as np

import plots
from fis_timestep import RESULT_DIR, load

#: The two trained ICs that bracket the 2.05 deg holdout.
LOWER_DEG, UPPER_DEG = 2.0, 2.1


def _row(split, value):
    idx = int(np.argmin(np.abs(split.ic_deg[:, split.swept_index] - value)))
    assert abs(split.ic_deg[idx, split.swept_index] - value) < 1e-6, (
        f"{split.label}: no trained IC at {value} deg"
    )
    return idx


def measure(split):
    """Separation of the bracketing pair, and the holdout's distance from each."""
    lo = split.theta_deg[_row(split, LOWER_DEG)]
    hi = split.theta_deg[_row(split, UPPER_DEG)]
    mid = split.holdout_theta_deg

    # Largest angular disagreement across links, at each time.
    sep = np.max(np.abs(hi - lo), axis=1)
    # How far the truth sits from the midpoint of its own bracket. This is the
    # error of one specific naive estimator -- average the two nearest trained
    # trajectories -- not a lower bound on what any model can achieve. A model
    # that regresses toward the conditional mean over all 31 ICs can and does
    # beat it once the bracket has diverged.
    interp_err = np.max(np.abs(mid - 0.5 * (lo + hi)), axis=1)

    span = float(np.max(np.ptp(mid, axis=0)))
    # First time the bracket is wider than a tenth of the holdout's own range.
    thresh = 0.1 * span
    over = np.nonzero(sep > thresh)[0]
    t_decorrelate = float(split.t[over[0]]) if over.size else float("inf")

    # Exponential divergence rate fitted to the bracket separation while it is
    # still growing, i.e. the finite-time Lyapunov estimate this dataset implies.
    growing = sep > 0
    if t_decorrelate < np.inf and np.isfinite(t_decorrelate):
        fit_to = max(int(t_decorrelate / (split.t[1] - split.t[0])), 10)
    else:
        fit_to = sep.size
    m = growing[:fit_to]
    lam = float("nan")
    if m.sum() > 10:
        lam = float(np.polyfit(split.t[:fit_to][m], np.log(sep[:fit_to][m]), 1)[0])

    return {
        "dataset": split.label,
        "holdout_range_deg": round(span, 2),
        "bracket_sep_final_deg": round(float(sep[-1]), 2),
        "bracket_sep_max_deg": round(float(sep.max()), 2),
        "t_decorrelate_s": round(t_decorrelate, 3),
        "lyapunov_per_s": round(lam, 3),
        "midpoint_err_final_deg": round(float(interp_err[-1]), 2),
        "midpoint_err_rms_deg": round(float(np.sqrt(np.mean(interp_err**2))), 2),
        # Error of the naive "average the two bracketing trained trajectories"
        # estimator, in the paper's scaled units. A reference point, not a bound.
        "midpoint_rmse_scaled": round(float(np.sqrt(np.mean(interp_err**2)) / span), 4),
        # The other reference point: predict this trajectory's own mean. Scaled
        # R^2 is 0 by construction here, so this is the RMSE that R^2 = 0 means.
        "constant_mean_rmse_scaled": round(
            float(np.mean(np.std(split.holdout_theta_scaled, axis=0))), 4
        ),
        "t": split.t,
        "sep": sep,
    }


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for n_links in (2, 3):
        for friction in (True, False):
            split = load(n_links, friction)
            m = measure(split)
            curves[split.label] = (m.pop("t"), m.pop("sep"))
            rows.append(m)
            print(
                f"{m['dataset']:20s} holdout range {m['holdout_range_deg']:8.1f} deg | "
                f"bracket [2.0, 2.1] separates to {m['bracket_sep_final_deg']:8.1f} deg | "
                f"decorrelates at t={m['t_decorrelate_s']:.2f} s | "
                f"lambda={m['lyapunov_per_s']:.2f}/s | "
                f"midpoint RMSE {m['midpoint_rmse_scaled']:.4f} | "
                f"R2=0 RMSE {m['constant_mean_rmse_scaled']:.4f}"
            )

    with open(RESULT_DIR / "bracket.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    fig, ax = plots.plt.subplots(figsize=(6.4, 4.0))
    fig.patch.set_facecolor(plots.SURFACE)
    colors = {"double_friction": plots.BLUE, "triple_friction": plots.AQUA,
              "double_frictionless": plots.ORANGE, "triple_frictionless": plots.YELLOW}
    for label, (t, sep) in curves.items():
        ax.semilogy(t, np.maximum(sep, 1e-6), color=colors[label], linewidth=1.3,
                    label=label.replace("_", " "))
    plots._style(
        ax,
        title="Separation of the two trained ICs bracketing the 2.05° holdout",
        xlabel="t (seconds)",
        ylabel="max angular separation (deg, log scale)",
    )
    ax.legend(fontsize=plots.FS_SMALL, frameon=False, loc="lower right")
    path = plots._save(fig, "fig_bracket")
    print(f"\nwrote {RESULT_DIR / 'bracket.csv'} and {path}")


if __name__ == "__main__":
    main()
