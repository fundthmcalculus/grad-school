"""Why the frictionless held-out initial condition is unlearnable, measured.

The paper's time-step operator interpolates between initial conditions: the
held-out IC [120, 2.05] sits exactly between the trained ICs [120, 2.0] and
[120, 2.1]. Interpolation can only work while those two neighbors still
resemble each other. This script measures how long that holds.

Once the two bracketing training trajectories have separated by a large fraction
of the holdout's own angular range, the training set offers two mutually
contradictory answers for the same query and nothing to choose between them. Any
model is then pushed toward the conditional mean over the initial-condition grid,
which caps achievable R^2 well below 1 regardless of architecture. That is a
property of the dataset, not of the model, and it is why every model in the paper
scores ~0.2 on the frictionless holdout and ~0.99 with friction.

The separation is also what makes the friction problems easy, and how easy depends
on chain length. At n=2 and n=3 the bracketing pair stays inside 4% of the target's
range for the whole 10 s window, so the midpoint baseline reaches R^2 0.9999 and
beats every model in the paper. At n=5 it separates to 24%, the baseline degrades
twentyfold, and interpolating the grid stops being close to sufficient. Run this
before concluding anything from a friction-variant score.

Two reference errors are reported for scale, neither of which is a lower bound:
`midpoint_rmse_scaled` (average the two bracketing trained trajectories) and
`constant_mean_rmse_scaled` (predict the trajectory's own mean, i.e. R^2 = 0).

`draw_bracket_separation` is a second, independent diagnostic of the same
phenomenon at n=2 only: it integrates the bracketing pair itself (rather than
reading it out of the sweep dataset) so the divergence is visible without going
through `measure()`'s bookkeeping, across both regimes side by side.

Standalone run prints the table, writes results/bracket.json (a plain dump, not
gated by run_all.py's cache -- see pipeline_cache.py), and both figures.
"""

from __future__ import annotations

import json

import numpy as np

import pendulum_data as pdata
import plots
from fis_timestep import RESULT_DIR, load

#: The two trained ICs that bracket the 2.05 deg holdout.
LOWER_DEG, UPPER_DEG = 2.0, 2.1


def _row(split, value):
    idx = int(np.argmin(np.abs(split.ic_deg[:, split.swept_index] - value)))
    assert (
        abs(split.ic_deg[idx, split.swept_index] - value) < 1e-6
    ), f"{split.label}: no trained IC at {value} deg"
    return idx


def _extend_to_holdout(split, ic_row_deg):
    """Integrate one trained IC out to the holdout's full 20 s.

    `split.theta_deg` only holds the 10 s training window; comparing the
    bracketing pair against the holdout at t = 20 s (Table 3) needs the same
    two initial conditions continued to 20 s, not the truncated array.
    """
    rhs = pdata.rhs_for(split.n_links, split.friction)
    state0 = np.zeros(2 * split.n_links)
    state0[0::2] = np.deg2rad(ic_row_deg)
    traj = pdata.rk4_integrate(rhs, state0, n_steps=split.holdout_t.size)
    return np.rad2deg(traj[:, 0::2])


def measure(split):
    """Separation of the bracketing pair, and the holdout's distance from each."""
    lo = _extend_to_holdout(split, split.ic_deg[_row(split, LOWER_DEG)])
    hi = _extend_to_holdout(split, split.ic_deg[_row(split, UPPER_DEG)])
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
    t_decorrelate = float(split.holdout_t[over[0]]) if over.size else float("inf")

    # Exponential divergence rate fitted to the bracket separation while it is
    # still growing, i.e. the finite-time Lyapunov estimate this dataset implies.
    growing = sep > 0
    if t_decorrelate < np.inf and np.isfinite(t_decorrelate):
        fit_to = max(int(t_decorrelate / (split.holdout_t[1] - split.holdout_t[0])), 10)
    else:
        fit_to = sep.size
    m = growing[:fit_to]
    lam = float("nan")
    if m.sum() > 10:
        lam = float(
            np.polyfit(split.holdout_t[:fit_to][m], np.log(sep[:fit_to][m]), 1)[0]
        )

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
        "t": split.holdout_t,
        "sep": sep,
    }


def measure_all(n_links_list=pdata.N_LINKS, log=print):
    """Every (n_links, friction) dataset's bracket measurement. Pure: no file I/O.

    Returns (rows, curves): `rows` are `measure()`'s scalar fields (its `t`/`sep`
    arrays popped out), `curves` maps label -> (t, sep) for the figure.
    """
    rows, curves = [], {}
    for n_links in n_links_list:
        for friction in (True, False):
            split = load(n_links, friction)
            m = measure(split)
            curves[split.label] = (m.pop("t"), m.pop("sep"))
            rows.append(m)
            log(
                f"{m['dataset']:20s} holdout range {m['holdout_range_deg']:8.1f} deg | "
                f"bracket [2.0, 2.1] separates to {m['bracket_sep_final_deg']:8.1f} deg | "
                f"decorrelates at t={m['t_decorrelate_s']:.2f} s | "
                f"lambda={m['lyapunov_per_s']:.2f}/s | "
                f"midpoint RMSE {m['midpoint_rmse_scaled']:.4f} | "
                f"R2=0 RMSE {m['constant_mean_rmse_scaled']:.4f}"
            )
    return rows, curves


def draw_fig_bracket(curves):
    """figures/fig_bracket.png: bracket separation against time, all datasets."""
    fig, ax = plots.plt.subplots(figsize=(6.6, 4.2))
    fig.patch.set_facecolor(plots.SURFACE)
    colors = plots.regime_colors(curves)
    for label, (t, sep) in sorted(curves.items()):
        ax.semilogy(
            t,
            np.maximum(sep, 1e-6),
            color=colors[label],
            linewidth=1.3,
            label=label.replace("_", " "),
        )
    plots._style(
        ax,
        title="Separation of the two trained ICs bracketing the 2.05° holdout",
        xlabel="t (seconds)",
        ylabel="max angular separation (deg, log scale)",
    )
    ax.legend(fontsize=plots.FS_SMALL, frameon=False, loc="lower right")
    return plots._save(fig, "fig_bracket")


def draw_bracket_separation():
    """figures/trajectory_snapshots.png: bracket separation, friction vs frictionless.

    A second view of the same n=2 phenomenon `measure_all` reports on every chain
    length: integrates the [2.0, 2.1] deg bracketing pair directly (rather than
    reading it out of the sweep dataset) so the two regimes sit side by side on
    one plot. Physics comes from `pendulum_data`, matching every other script here
    -- there is exactly one derivation of the equations of motion in this
    repository, not one per figure.
    """
    theta1_rad = np.radians(pdata.THETA1_DEG)
    ic_lower = np.array([theta1_rad, 0.0, np.radians(LOWER_DEG), 0.0])
    ic_upper = np.array([theta1_rad, 0.0, np.radians(UPPER_DEG), 0.0])

    configs = [("friction", pdata.DAMPING), ("frictionless", 0.0)]
    n_steps = int(round((pdata.TEST_T_END - pdata.T_START) / pdata.H))
    t_all = pdata.time_points(t_end=pdata.TEST_T_END)

    fig, axes = plots.plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(plots.SURFACE)
    for ax, (regime, damping) in zip(axes, configs):
        rhs = lambda r, t: pdata.rhs_double_reference(  # noqa: E731
            r, t, damping1=damping, damping2=damping
        )
        traj_lower = pdata.rk4_integrate(rhs, ic_lower, n_steps=n_steps)
        traj_upper = pdata.rk4_integrate(rhs, ic_upper, n_steps=n_steps)

        sep_1 = np.abs(np.degrees(traj_lower[:, 0] - traj_upper[:, 0]))
        sep_2 = np.abs(np.degrees(traj_lower[:, 2] - traj_upper[:, 2]))
        separation = np.maximum(sep_1, sep_2)

        ax.semilogy(t_all, separation, linewidth=2.5, color=plots.BLUE, label="Separation")
        ax.axvline(
            x=pdata.T_END,
            color=plots.RED,
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Training edge",
        )
        ax.set_ylim([1e-2, 1e3])
        plots._style(
            ax,
            title=f"Double pendulum, {regime}",
            xlabel="Time (s)",
            ylabel="Max angle separation (°)",
        )
        ax.legend(loc="best", fontsize=plots.FS_SMALL, frameon=False)

    fig.suptitle(
        "Bracket Separation: How Two Training ICs Diverge Over Time",
        fontsize=plots.FS_TITLE,
        color=plots.INK,
        y=1.00,
    )
    fig.tight_layout()
    return plots._save(fig, "trajectory_snapshots")


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows, curves = measure_all()

    with open(RESULT_DIR / "bracket.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": rows}, fh, indent=2)

    fig_path = draw_fig_bracket(curves)
    snap_path = draw_bracket_separation()
    print(f"\nwrote {RESULT_DIR / 'bracket.json'}, {fig_path} and {snap_path}")


if __name__ == "__main__":
    main()
