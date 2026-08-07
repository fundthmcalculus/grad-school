"""Time-as-input versus state-as-input, on the friction double pendulum.

Two formulations are compared, both trained on the same 10 s of the same 31 initial
conditions and both scored on the same held-out [120, 2.05] deg trajectory out to
20 s.

  WITH the time variable -- the paper's time-step operator (fis_timestep.py):
      (theta_1(0), theta_2(0), t) -> (theta_1(t), theta_2(t))
      One vectorized query answers any t. Time is an input, so the model's domain
      of validity is the training window and nothing carries across its edge.

  WITHOUT the time variable -- a learned right-hand side (stable_extrapolation.py):
      (theta_1, omega_1, theta_2, omega_2) -> (alpha_1, alpha_2), then integrate.
      Time never enters, so there is no window to leave; what matters instead is
      whether the rolled-out state stays where the training data was.

Restricted to the friction double pendulum on purpose. The frictionless references
are not converged past about 11.5 s (pendulum_data.reference_convergence), so a
10-20 s score there would compare a surrogate against a reference that is itself a
property of the step size. Friction at n=2 agrees with DOP853 to 0.0018 deg over
20 s, so the comparison is meaningful.

Three axes are recorded as well as accuracy, because they are where the two families
genuinely trade against each other:

  * training time     -- seconds to fit, measured
  * input variables   -- how many the model actually consumes after constants are
                         dropped; the time-step operator sees 2, the state-space
                         one sees 4, the harmonic encoding sees 18
  * rollout time      -- seconds to produce the 20 s trajectory. The time-step
                         operator answers in one batched call; the state-space one
                         must integrate 4000 steps x 4 stages, so it pays per
                         query what it saves in generality. Reported because a
                         comparison that omitted it would flatter the state-space
                         family.

Run: python compare_families.py
"""

from __future__ import annotations

import csv
import time

import numpy as np

import fis_timestep as fts
import pendulum_data as pdata
import plots
import stable_extrapolation as se

RESULT_DIR = pdata.DATA_DIR.parent / "results"

DURATION = pdata.TEST_T_END
N_STEPS = int(round(DURATION / pdata.H))

#: (label, FisConfig) for the time-as-input family.
WITH_TIME = [
    ("time-step, 40 rules", fts.FisConfig(n_output_buckets=40)),
    ("time-step, 120 rules", fts.FisConfig(n_output_buckets=120, tsk_order="full-2nd")),
    (
        "time-step, 300 rules",
        fts.FisConfig(n_output_buckets=300, tsk_order="full-2nd", l2_reg=1e-9),
    ),
    (
        "time-step, 8 harmonics",
        fts.FisConfig(n_output_buckets=120, encoding="harmonic", n_harmonics=8),
    ),
]


def _rmse_deg(pred_deg, true_deg, mask):
    return float(np.sqrt(np.mean((pred_deg[mask] - true_deg[mask]) ** 2)))


def run_with_time(split, truth_deg, t, inw):
    """Fit and score each time-as-input model over the full 20 s.

    Returns (rows, preds): `preds[label]` is that model's real predicted
    trajectory in degrees over the full 20 s, always built (not gated behind a
    flag) since it costs nothing beyond what scoring already computes and lets
    a caller like gen_n2_rollout_comparison.py plot genuine model output instead
    of re-deriving or, worse, synthesizing it.
    """
    rows, preds = [], {}
    for label, cfg in WITH_TIME:
        X, names, Y = fts.build_pooled(split, cfg)
        rng = np.random.default_rng(42)
        perm = rng.permutation(X.shape[0])
        tr = perm[: int(0.8 * X.shape[0])]

        model = fts.FisOperator(cfg)
        t0 = time.perf_counter()
        model.fit(X[tr], names, Y[tr])
        fit_s = time.perf_counter() - t0

        Xh, _ = fts.encode(
            split.holdout_ic_deg[None, :], t, cfg.encoding, cfg.n_harmonics
        )
        t0 = time.perf_counter()
        pred_scaled = model.predict(Xh)
        query_s = time.perf_counter() - t0

        span = (split.holdout_range[:, 1] - split.holdout_range[:, 0])[None, :]
        lo = split.holdout_range[:, 0][None, :]
        pred_deg = pred_scaled * span + lo
        preds[label] = pred_deg

        rows.append(
            {
                "family": "with time",
                "model": label,
                "n_variables": len(model.names_),
                "variables": " ".join(model.names_),
                "train_seconds": round(fit_s, 2),
                "rollout_seconds": round(query_s, 3),
                "in_window_rmse_deg": round(_rmse_deg(pred_deg, truth_deg, inw), 4),
                "extrap_rmse_deg": round(_rmse_deg(pred_deg, truth_deg, ~inw), 4),
            }
        )
        print(
            f"  {label:26s} vars={rows[-1]['n_variables']:2d} "
            f"train={fit_s:7.1f}s  in-window {rows[-1]['in_window_rmse_deg']:10.3f} deg  "
            f"past 10s {rows[-1]['extrap_rmse_deg']:12.3f} deg",
            flush=True,
        )
    return rows, preds


def run_without_time(truth_deg, t, inw, state0):
    """Fit and roll out each state-as-input model over the full 20 s."""
    X, Y, _ = se.training_pairs()
    specs = [
        ("state-space, 40 rules", lambda: se.DynamicsFIS(n_output_buckets=40), False),
        ("state-space, 120 rules", lambda: se.DynamicsFIS(n_output_buckets=120), False),
        (
            "state-space, 120 + guard",
            lambda: se.DynamicsFIS(n_output_buckets=120),
            True,
        ),
        ("physics basis (grey box)", se.PhysicsFIS, False),
    ]
    rows = []
    for label, factory, guard in specs:
        model = factory()
        t0 = time.perf_counter()
        model.fit(X, Y)
        fit_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred, diverged, n_corr = se.rollout(
            model, state0, N_STEPS, dissipation_guard=guard
        )
        roll_s = time.perf_counter() - t0
        pred_deg = np.rad2deg(pred[:, [0, 2]])

        # The physics basis consumes engineered features, not raw states: five for
        # alpha_1 and four for alpha_2. Report the larger, and name them, so the
        # variable count is not silently comparing different things.
        n_vars = 5 if isinstance(model, se.PhysicsFIS) else 4
        names = (
            "physics basis terms"
            if isinstance(model, se.PhysicsFIS)
            else " ".join(se.STATE_LABELS)
        )

        rows.append(
            {
                "family": "without time",
                "model": label,
                "n_variables": n_vars,
                "variables": names,
                "train_seconds": round(fit_s, 2),
                "rollout_seconds": round(roll_s, 2),
                "in_window_rmse_deg": round(_rmse_deg(pred_deg, truth_deg, inw), 4),
                "extrap_rmse_deg": round(_rmse_deg(pred_deg, truth_deg, ~inw), 4),
                "diverged_at_s": diverged,
                "guard_corrections": n_corr,
            }
        )
        print(
            f"  {label:26s} vars={n_vars:2d} train={fit_s:7.1f}s  "
            f"in-window {rows[-1]['in_window_rmse_deg']:10.3f} deg  "
            f"past 10s {rows[-1]['extrap_rmse_deg']:12.3f} deg",
            flush=True,
        )
    return rows


# ---------------------------------------------------------------------------
FAM_COLOUR = {"with time": plots.ORANGE, "without time": plots.BLUE}
FLOOR = 1e-6  # log-axis floor; the physics basis is below any plot resolution


def _bars(rows, key, title, ylabel, name, logy=True, sort_asc=True):
    """One sorted bar chart, coloured by family."""
    data = sorted(rows, key=lambda r: r[key], reverse=not sort_asc)
    labels = [r["model"] for r in data]
    vals = [max(r[key], FLOOR) if logy else r[key] for r in data]
    cols = [FAM_COLOUR[r["family"]] for r in data]

    fig, ax = plots.plt.subplots(figsize=(0.85 * len(labels) + 2.6, 4.3))
    fig.patch.set_facecolor(plots.SURFACE)
    x = np.arange(len(labels), dtype=float)
    ax.bar(x, vals, 0.66, color=cols, edgecolor=plots.INK, linewidth=0.6)
    for xi, v, r in zip(x, vals, data):
        ax.text(
            xi,
            v,
            f" {r[key]:.4g}",
            ha="center",
            va="bottom",
            fontsize=plots.FS_SMALL,
            color=plots.INK_2,
            rotation=90,
        )
    if logy:
        ax.set_yscale("log")
        ax.set_ylim(FLOOR, max(vals) * 60)
    else:
        ax.set_ylim(0, max(vals) * 1.35)
    plots._style(ax, title=title, ylabel=ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, fontsize=plots.FS_TICK, color=plots.INK_2, rotation=30, ha="right"
    )
    handles = [
        plots.plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor=plots.INK)
        for c in FAM_COLOUR.values()
    ]
    ax.legend(
        handles,
        [f"{k} variable" for k in FAM_COLOUR],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        fontsize=plots.FS_SMALL,
        frameon=False,
        ncol=2,
    )
    return plots._save(fig, name)


def _paired(rows):
    """In-window against past-the-window error, side by side per model."""
    data = sorted(rows, key=lambda r: r["extrap_rmse_deg"])
    labels = [r["model"] for r in data]
    a = [max(r["in_window_rmse_deg"], FLOOR) for r in data]
    b = [max(r["extrap_rmse_deg"], FLOOR) for r in data]

    fig, ax = plots.plt.subplots(figsize=(0.95 * len(labels) + 2.8, 4.4))
    fig.patch.set_facecolor(plots.SURFACE)
    x = np.arange(len(labels), dtype=float)
    w = 0.38
    ax.bar(
        x - w / 2,
        a,
        w,
        color=plots.AQUA,
        edgecolor=plots.INK,
        linewidth=0.6,
        label="0-10 s (inside training window)",
    )
    ax.bar(
        x + w / 2,
        b,
        w,
        color=plots.RED,
        edgecolor=plots.INK,
        linewidth=0.6,
        label="10-20 s (past the window)",
    )
    ax.set_yscale("log")
    ax.set_ylim(FLOOR, max(b) * 60)
    plots._style(
        ax,
        title="Friction double pendulum: error inside vs past the training window",
        ylabel="angle RMSE (deg, log)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels, fontsize=plots.FS_TICK, color=plots.INK_2, rotation=30, ha="right"
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.34),
        fontsize=plots.FS_SMALL,
        frameon=False,
        ncol=2,
    )
    return plots._save(fig, "fig_family_inwindow_vs_extrap")


def _tradeoff(rows):
    """Extrapolation error against training time, marker area = variable count."""
    fig, ax = plots.plt.subplots(figsize=(7.0, 4.6))
    fig.patch.set_facecolor(plots.SURFACE)
    for fam, colour in FAM_COLOUR.items():
        sub = [r for r in rows if r["family"] == fam]
        ax.scatter(
            [r["train_seconds"] for r in sub],
            [max(r["extrap_rmse_deg"], FLOOR) for r in sub],
            s=[28 * r["n_variables"] for r in sub],
            color=colour,
            edgecolor=plots.INK,
            linewidth=0.7,
            label=f"{fam} variable",
            zorder=5,
        )
    # Number the markers and key them off to the side. Several models sit within a
    # factor of two of each other on both axes, so per-point text labels overprint
    # however the offsets are alternated; an index plus a key cannot collide.
    by_x = sorted(rows, key=lambda r: r["train_seconds"])
    for i, r in enumerate(by_x, 1):
        ax.annotate(
            str(i),
            (r["train_seconds"], max(r["extrap_rmse_deg"], FLOOR)),
            textcoords="offset points",
            xytext=(0, -3),
            ha="center",
            fontsize=plots.FS_SMALL,
            color="white",
            weight="bold",
            zorder=7,
        )
    key = "\n".join(
        f"{i}. {r['model']}  —  {r['n_variables']} vars, {r['train_seconds']:.4g} s"
        for i, r in enumerate(by_x, 1)
    )
    # Placed right of the leftmost marker: the physics-basis point sits at the
    # bottom-left, which is otherwise the emptiest corner.
    ax.text(
        0.28,
        0.42,
        key,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=plots.FS_SMALL,
        color=plots.INK_2,
        linespacing=1.5,
    )
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlim(
        min(r["train_seconds"] for r in rows) * 0.35,
        max(r["train_seconds"] for r in rows) * 9,
    )
    plots._style(
        ax,
        title="Cost against extrapolation accuracy (marker area = input variables)",
        xlabel="training time (s, log)",
        ylabel="10-20 s angle RMSE (deg, log)",
    )
    # Legend below the axes: the interesting corner is bottom-left and an inset
    # legend covered the physics-basis point sitting in it.
    ax.legend(
        fontsize=plots.FS_SMALL,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
    )
    return plots._save(fig, "fig_family_tradeoff")


# ---------------------------------------------------------------------------
def main():
    split = fts.load(2, friction=True)
    t = np.arange(N_STEPS) * pdata.H
    inw = t < pdata.T_END
    state0 = np.array(
        [np.deg2rad(pdata.THETA1_DEG), 0.0, np.deg2rad(pdata.TEST_THETA2_DEG), 0.0]
    )
    # One reference for both families, DOP853 so neither shares an integrator with
    # it. RK4 at the paper's step differs from this by 0.0018 deg here, so the
    # choice does not move any number -- it removes an objection.
    truth_deg = np.rad2deg(se.truth(state0, N_STEPS)[:, [0, 2]])

    print(
        f"Friction double pendulum, trained on {pdata.T_END:.0f} s, "
        f"scored to {DURATION:.0f} s against a DOP853 reference.\n"
    )
    print("WITH the time variable (time-step operator):")
    rows, _ = run_with_time(split, truth_deg, t, inw)
    print("\nWITHOUT the time variable (learned dynamics + RK4):")
    rows += run_without_time(truth_deg, t, inw, state0)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    order = [
        "family",
        "model",
        "n_variables",
        "train_seconds",
        "rollout_seconds",
        "in_window_rmse_deg",
        "extrap_rmse_deg",
    ]
    fields = order + [f for f in fields if f not in order]
    with open(
        RESULT_DIR / "family_comparison.csv", "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    paths = [
        _bars(
            rows,
            "extrap_rmse_deg",
            "Friction double pendulum: error 10-20 s, past the training window",
            "angle RMSE (deg, log)",
            "fig_family_extrap_rmse",
        ),
        _bars(
            rows,
            "in_window_rmse_deg",
            "Friction double pendulum: error 0-10 s, inside the training window",
            "angle RMSE (deg, log)",
            "fig_family_inwindow_rmse",
        ),
        _bars(
            rows,
            "train_seconds",
            "Training time",
            "seconds (log)",
            "fig_family_train_time",
        ),
        _bars(
            rows,
            "n_variables",
            "Input variables consumed",
            "count",
            "fig_family_variables",
            logy=False,
            sort_asc=False,
        ),
        _bars(
            rows,
            "rollout_seconds",
            "Time to produce one 20 s trajectory",
            "seconds (log)",
            "fig_family_rollout_time",
        ),
        _paired(rows),
        _tradeoff(rows),
    ]
    print(
        f"\nwrote {RESULT_DIR / 'family_comparison.csv'} and "
        f"{len(paths)} figures to {plots.FIG_DIR}"
    )
    return rows


if __name__ == "__main__":
    main()
