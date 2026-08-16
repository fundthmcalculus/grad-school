"""A friction double pendulum surrogate that stays stable past the training window.

The time-step operator this repository reproduces from arXiv:2504.13453 maps
``(theta(0), t) -> theta(t)``. Time is an *input feature*, so the model's domain of
validity is exactly the time window it was trained on: at t = 20 s the input scaler
returns 2.0, every Gaussian membership is deep in its tail, and the prediction
diverges by four orders of magnitude within one timestep of the boundary
(REPRODUCTION_REPORT.md section 9). No amount of tuning fixes that -- it is what
the formulation is.

This module changes the formulation instead. The FIS learns the *dynamics*

    (theta_1, omega_1, theta_2, omega_2)  ->  (alpha_1, alpha_2)

and a plain RK4 integrator advances the state. Time never enters the model, so
there is no window to leave. What matters instead is whether the rolled-out state
stays inside the region of state space the training trajectories covered.

Why friction is what makes this work. The undamped double pendulum is chaotic and
volume-preserving: an error in the learned acceleration is amplified at the
Lyapunov rate (lambda ~ 1.8/s, measured in bracket_diagnostic.py) and the rollout
leaves the training region no matter how accurate the fit. With damping the flow is
*contracting*: trajectories decay toward the hanging equilibrium, small errors
shrink rather than grow, and the state moves toward smaller angles and rates --
i.e. toward the interior of the training distribution, not out of it. Extrapolating
in time becomes interpolating in state.

Run: python stable_extrapolation.py
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent / "tribble-fis" / "src"))

from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import MinMaxScaler  # noqa: E402

from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

import pendulum_data as pdata  # noqa: E402

RESULT_DIR = HERE / "results"

#: Inputs and outputs of the learned dynamics.
STATE_LABELS = ["theta_1", "omega_1", "theta_2", "omega_2"]
ACCEL_LABELS = ["alpha_1", "alpha_2"]


@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ---------------------------------------------------------------------------
def training_pairs(damping=pdata.DAMPING, n_steps=pdata.N_STEPS):
    """(state, acceleration) samples from every training initial condition.

    Same 31 initial conditions and same 10 s window as the time-step study, so the
    two formulations see exactly the same information. The accelerations are taken
    from the true right-hand side at each visited state -- this is a fit to the
    dynamics, not a finite-difference of the trajectory, so integrator error does
    not enter the labels.
    """
    rhs = _friction_rhs(damping)
    ic_deg = pdata._initial_conditions(2, pdata.TRAIN_THETA2_DEG)

    states, accels = [], []
    for row in ic_deg:
        s0 = np.array([np.deg2rad(row[0]), 0.0, np.deg2rad(row[1]), 0.0])
        traj = pdata.rk4_integrate(rhs, s0, n_steps=n_steps)
        d = np.array([rhs(s, 0.0) for s in traj])
        states.append(traj)
        accels.append(d[:, [1, 3]])
    return np.vstack(states), np.vstack(accels), ic_deg


def _friction_rhs(damping):
    return lambda r, t: pdata.rhs_double_reference(r, t, damping, damping)


# ---------------------------------------------------------------------------
def physics_basis(states):
    """Numerator terms of the equations of motion, divided by the known denominator.

    The black-box route above cannot work, and the reason is measurable: the
    damping term is 1.2% of the acceleration, while a TSK fit of the acceleration
    leaves a residual 8x *larger* than that at 40 rules and still 0.3x at 300. The
    surrogate's effective dissipation is therefore fit noise, its sign is arbitrary,
    and the rollout gains energy until it leaves the region the training data
    covered.

    So give the model the structure instead of making it discover it. The double
    pendulum's accelerations are

        alpha_i = (sum of a few trig/quadratic terms) / D,
        D = L_i (2 m1 + m2 - m2 cos(2 theta_1 - 2 theta_2)),

    and D depends only on the angles and on masses and lengths, which are known
    constants of the rig. Dividing through by D makes each acceleration *exactly
    linear* in the features below with constant coefficients, so a least-squares
    consequent recovers those constants rather than approximating a curved surface.
    The damping feature (omega_i / D) is then one column among equals instead of a
    1% perturbation buried under fit error.

    This is a grey-box assumption and a strong one: it presumes the functional form
    of the dynamics, which is exactly what a surrogate is often meant to avoid. It
    is the same trade this repository already documented at
    ../N3_N5_FUZZY_REGRESSION_REPORT.md section 4.5. Stated plainly so the result is
    not read as more than it is.

    Returns (F1, F2): design matrices for alpha_1 and alpha_2.
    """
    s = np.asarray(states, float).reshape(-1, 4)
    th1, om1, th2, om2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    m1, m2 = pdata.M1, pdata.M2
    l1, l2 = pdata.L1, pdata.L2
    delta = th1 - th2
    den = 2 * m1 + m2 - m2 * np.cos(2 * th1 - 2 * th2)
    d1, d2 = l1 * den, l2 * den

    f1 = np.column_stack(
        [
            np.sin(th1) / d1,  # -g(2m1+m2)
            np.sin(th1 - 2 * th2) / d1,  # -m2 g
            np.sin(delta) * om2**2 / d1,  # -2 m2 L2
            np.sin(delta) * om1**2 * np.cos(delta) / d1,  # -2 m2 L1
            om1 / d1,  # -damping1
        ]
    )
    f2 = np.column_stack(
        [
            np.sin(delta) * om1**2 / d2,  # 2 L1 (m1+m2)
            np.sin(delta) * np.cos(th1) / d2,  # 2 g (m1+m2)
            np.sin(delta) * om2**2 * np.cos(delta) / d2,  # 2 L2 m2
            om2 / d2,  # -damping2
        ]
    )
    return f1, f2


#: True coefficients of `physics_basis`, for checking what the fit recovers.
def true_coefficients():
    g, m1, m2, l1, l2, c = (
        pdata.G,
        pdata.M1,
        pdata.M2,
        pdata.L1,
        pdata.L2,
        pdata.DAMPING,
    )
    return (
        np.array([-g * (2 * m1 + m2), -m2 * g, -2 * m2 * l2, -2 * m2 * l1, -c]),
        np.array([2 * l1 * (m1 + m2), 2 * g * (m1 + m2), 2 * l2 * m2, -c]),
    )


class PhysicsFIS:
    """Order-1 TSK on the physics basis: one linear consequent per acceleration.

    With the basis above the target is exactly linear, so a single rule with a
    no-intercept least-squares consequent is the whole model. That is a degenerate
    fuzzy system by design -- partitioning the input space cannot help fit a
    function that is already a plane, and this repository measured that penalty
    directly (N3_N5 report section 4.7: many rules still had not reached the plain
    least-squares ceiling).
    """

    def fit(self, states, accels):
        f1, f2 = physics_basis(states)
        self.coef_ = [
            np.linalg.lstsq(f1, accels[:, 0], rcond=None)[0],
            np.linalg.lstsq(f2, accels[:, 1], rcond=None)[0],
        ]
        return self

    def accel(self, state):
        # physics_basis always returns a 2-D design matrix; a single state is one
        # row of it, so take element 0 rather than assuming a scalar.
        f1, f2 = physics_basis(state)
        return np.array([(f1 @ self.coef_[0])[0], (f2 @ self.coef_[1])[0]])

    def rhs(self, state, _t=0.0):
        a = self.accel(state)
        return np.array([state[1], a[0], state[3], a[1]])

    @property
    def label(self):
        return "physics-basis-lstsq"


# ---------------------------------------------------------------------------
class DynamicsFIS:
    """A collection of TSK systems approximating the equations of motion.

    One FIS per acceleration component. Inputs are min-max scaled over the training
    states.

    `clip_inputs` clips the state to the training box before evaluation. The
    expectation was that this would turn the out-of-box blow-up into saturation --
    return the acceleration learned at the nearest covered state rather than an
    unbounded linear extrapolation.

    Measured, it does the opposite: tail amplitude ratio 41 unclipped versus 1468
    clipped on the 20 s rollout, twenty-five times *worse*. Clipping freezes the
    acceleration at the boundary value, so once the state leaves the box, the
    restoring term stops growing with displacement while the rate keeps
    integrating -- the clip removes precisely the feedback that would have pulled
    back the state. Saturating the input is different from saturating the output.
    Kept as an ablation because the intuition is appealing and wrong; use
    `dissipation_guard` in `rollout` instead, which bounds the rollout by
    constraining energy rather than inputs.
    """

    def __init__(
        self,
        n_output_buckets=60,
        tsk_order="full-2nd",
        l2_reg=1e-9,
        output_partition="quantile",
        clip_inputs=False,
        random_state=42,
    ):
        self.kw = dict(
            n_output_buckets=n_output_buckets,
            tsk_order=tsk_order,
            l2_reg=l2_reg,
            output_partition=output_partition,
            top_p=1.0,
            random_state=random_state,
        )
        self.clip_inputs = clip_inputs

    def fit(self, X, Y):
        self.scaler_ = MinMaxScaler(clip=self.clip_inputs).fit(X)
        Xs = self.scaler_.transform(X)
        self.models_ = []
        with _quiet():
            for j in range(Y.shape[1]):
                m = TribbleRegressor(**self.kw)
                m.fit(Xs, Y[:, j])
                self.models_.append(m)
        return self

    def accel(self, state):
        """Accelerations at one state (4,) -> (2,)."""
        xs = self.scaler_.transform(np.asarray(state, float).reshape(1, -1))
        return np.array([float(m.predict(xs)[0]) for m in self.models_])

    def rhs(self, state, _t=0.0):
        a = self.accel(state)
        return np.array([state[1], a[0], state[3], a[1]])

    @property
    def label(self):
        return (
            f"nb{self.kw['n_output_buckets']}_{self.kw['tsk_order']}"
            f"_{self.kw['output_partition']}"
            f"{'_clipped' if self.clip_inputs else ''}"
        )


# ---------------------------------------------------------------------------
def mechanical_energy(state):
    """Total energy of the double pendulum, pivot as the potential zero."""
    th1, om1, th2, om2 = np.asarray(state, float)
    m1, m2, l1, l2, g = pdata.M1, pdata.M2, pdata.L1, pdata.L2, pdata.G
    ke = 0.5 * m1 * (l1 * om1) ** 2 + 0.5 * m2 * (
        (l1 * om1) ** 2 + (l2 * om2) ** 2 + 2 * l1 * l2 * om1 * om2 * np.cos(th1 - th2)
    )
    pe = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return ke + pe, ke, pe


def apply_dissipation_guard(prev_state, new_state):
    """Forbid the surrogate from adding energy, without assuming the equations.

    The middle tier between "black-box, unstable" and "assume the exact equations".
    It uses one fact that the word *friction* already gives us -- the true system
    cannot gain mechanical energy -- and enforces exactly that, nothing more.

    Kinetic energy is homogeneous of degree 2 in the angular rates, so rescaling
    both rates by lambda scales KE by lambda^2 and the correction is closed-form:
    lambda = sqrt((E_prev - PE_new) / KE_new). Angles are untouched, so the
    configuration the integrator produced is kept, and only its speed is bled off.

    This bounds the rollout but does not aim it: it constrains the *magnitude* of
    the state, not its direction on the energy surface. This repository measured
    that distinction before -- exact energy projection cut drift by sixteen orders
    of magnitude and improved tracking by 5% (N3_N5 report section 4.3) -- so
    expect stability from this, not accuracy.
    """
    e_prev, _, _ = mechanical_energy(prev_state)
    e_new, ke_new, pe_new = mechanical_energy(new_state)
    if e_new <= e_prev or ke_new <= 0.0:
        return new_state, False
    lam = np.sqrt(max(0.0, (e_prev - pe_new) / ke_new))
    out = np.asarray(new_state, float).copy()
    out[1] *= lam
    out[3] *= lam
    return out, True


def rollout(model, state0, n_steps, h=pdata.H, bound=1e4, dissipation_guard=False):
    """RK4 the learned dynamics forward, stopping if the state stops being finite.

    `bound` is a divergence tripwire, not a stabiliser: it only decides when to
    stop recording, and where it fires is reported. A surrogate that needs it has
    already failed.
    """
    q = np.asarray(state0, float).copy()
    out = np.empty((n_steps, 4))
    diverged_at = None
    n_corrected = 0
    for i in range(n_steps):
        out[i] = q
        if not np.all(np.isfinite(q)) or np.max(np.abs(q)) > bound:
            out[i:] = q
            diverged_at = i * h
            break
        k1 = h * model.rhs(q)
        k2 = h * model.rhs(q + 0.5 * k1)
        k3 = h * model.rhs(q + 0.5 * k2)
        k4 = h * model.rhs(q + k3)
        nxt = q + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if dissipation_guard:
            nxt, hit = apply_dissipation_guard(q, nxt)
            n_corrected += hit
        q = nxt
    return out, diverged_at, n_corrected


def truth(state0, n_steps, damping=pdata.DAMPING, h=pdata.H, method="DOP853"):
    """Reference trajectory to score rollouts against.

    Defaults to scipy's 8th-order adaptive DOP853 at rtol 1e-12 rather than the
    paper's fixed-step RK4. Two reasons.

    First, independence. Scoring an RK4 rollout of the learned dynamics against an
    RK4 rollout of the true dynamics lets integration error cancel between them, so
    an agreement of 1e-11 degrees measures the model and the integrator together
    and cannot separate them. A different-order adaptive reference removes that
    cancellation, and the surrogate has to be right on its own.

    Second, accuracy where it is actually needed. Over 20 s, RK4 at h = 0.005
    differs from DOP853 by 0.0018 deg on the double friction chain -- irrelevant --
    but by 7.4 deg on the quintuple friction chain, which is the same size as the
    surrogate errors being reported there. Training data stays on the paper's RK4
    grid; only the yardstick changes.
    """
    rhs = _friction_rhs(damping)
    if method == "RK4":
        return pdata.rk4_integrate(rhs, state0, n_steps=n_steps, h=h)
    t = np.arange(n_steps) * h
    return pdata.integrate_dop853(rhs, state0, t)


# ---------------------------------------------------------------------------
def evaluate(pred, true_traj, t, train_t_end=pdata.T_END):
    """Angle-error metrics in degrees, split at the training-window edge."""
    th_p = np.rad2deg(pred[:, [0, 2]])
    th_t = np.rad2deg(true_traj[:, [0, 2]])
    err = np.abs(th_p - th_t)
    inw = t < train_t_end

    def block(mask):
        return {
            "rmse_deg": float(np.sqrt(np.mean((th_p[mask] - th_t[mask]) ** 2))),
            "max_err_deg": float(np.max(err[mask])),
            "r2": float(
                r2_score(th_t[mask], th_p[mask], multioutput="uniform_average")
            ),
        }

    out = {
        "in_window": block(inw),
        "extrap": block(~inw),
        "full": block(np.ones_like(inw)),
    }
    worst = np.max(err, axis=1)
    over = np.flatnonzero(worst > 10.0)
    out["t_10deg"] = float(t[over[0]]) if over.size else float("inf")
    # Two tail diagnostics, because "bounded" and "still moving" are different
    # failures and one number cannot separate them. A surrogate that pumps energy
    # has amp_ratio >> 1; one that has been damped to a standstill has amp_ratio
    # near 1 but activity_ratio near 0, and only the second number catches it.
    tail = t > (t[-1] - 2.0)
    out["tail_amp_ratio"] = float(
        np.max(np.abs(th_p[tail])) / max(np.max(np.abs(th_t[tail])), 1e-9)
    )
    out["tail_activity_ratio"] = float(
        np.mean(np.std(th_p[tail], axis=0))
        / max(np.mean(np.std(th_t[tail], axis=0)), 1e-9)
    )
    return out


# ---------------------------------------------------------------------------
def error_decomposition(model, state0, n_steps, h=pdata.H):
    """Separate model error from integrator error. Four rollouts, two axes.

    Scoring an RK4 rollout of the learned dynamics against an RK4 rollout of the
    true dynamics makes integration error cancel between them, because both
    integrators commit the same error on nearly the same trajectory. That
    comparison flatters the model by orders of magnitude and cannot be used to
    claim accuracy -- it was reported here as 3e-11 deg over 120 s before this
    function existed, when the model's actual error is 4e-9 and the number being
    measured was neither.

    Crossing {RK4, DOP853} against {learned rhs, true rhs} separates them:

        RK4(true)     vs DOP853(true)     -> integrator alone
        DOP853(learn) vs DOP853(true)     -> model alone
        RK4(learn)    vs DOP853(true)     -> what a user of the paper's grid gets
        RK4(learn)    vs RK4(true)        -> the cancelling comparison, for contrast

    Returns those four as a dict of max angular difference in degrees.
    """
    t = np.arange(n_steps) * h
    true_rhs = _friction_rhs(pdata.DAMPING)

    rk4_true = pdata.rk4_integrate(true_rhs, state0, n_steps=n_steps, h=h)
    dop_true = pdata.integrate_dop853(true_rhs, state0, t)
    rk4_learn, _, _ = rollout(model, state0, n_steps, h=h)
    dop_learn = pdata.integrate_dop853(model.rhs, state0, t)

    def diff(a, b):
        return float(
            np.max(np.abs(np.rad2deg(a[:, [0, 2]]) - np.rad2deg(b[:, [0, 2]])))
        )

    return {
        "integrator_only_deg": diff(rk4_true, dop_true),
        "model_only_deg": diff(dop_learn, dop_true),
        "model_plus_integrator_deg": diff(rk4_learn, dop_true),
        "cancelling_comparison_deg": diff(rk4_learn, rk4_true),
    }


def main(duration=20.0, n_buckets=120):
    import csv

    n_steps = int(round(duration / pdata.H))
    t = np.arange(n_steps) * pdata.H
    s0 = np.array(
        [np.deg2rad(pdata.THETA1_DEG), 0.0, np.deg2rad(pdata.TEST_THETA2_DEG), 0.0]
    )
    true_traj = truth(s0, n_steps)

    print(
        f"Friction double pendulum, trained on {pdata.T_END:.0f} s, "
        f"rolled out to {duration:.0f} s from the held-out IC "
        f"[{pdata.THETA1_DEG:g}, {pdata.TEST_THETA2_DEG:g}] deg\n"
    )

    X, Y, _ = training_pairs()
    black = DynamicsFIS(n_output_buckets=n_buckets).fit(X, Y)
    clipped = DynamicsFIS(n_output_buckets=n_buckets, clip_inputs=True).fit(X, Y)
    grey = PhysicsFIS().fit(X, Y)

    runs = [
        ("state-space FIS", black, False),
        ("state-space FIS + clipped inputs", clipped, False),
        ("state-space FIS + dissipation guard", black, True),
        ("physics-basis FIS (grey box)", grey, False),
    ]

    rows, curves = [], {}
    for name, model, guard in runs:
        pred, diverged, n_corr = rollout(model, s0, n_steps, dissipation_guard=guard)
        m = evaluate(pred, true_traj, t)
        curves[name] = pred
        rows.append(
            {
                "model": name,
                "config": model.label,
                "dissipation_guard": guard,
                "train_t_end_s": pdata.T_END,
                "rollout_t_end_s": duration,
                "in_window_rmse_deg": round(m["in_window"]["rmse_deg"], 4),
                "extrap_rmse_deg": round(m["extrap"]["rmse_deg"], 4),
                "extrap_r2": round(m["extrap"]["r2"], 6),
                "tail_amplitude_ratio": round(m["tail_amp_ratio"], 4),
                "tail_activity_ratio": round(m["tail_activity_ratio"], 4),
                "t_error_10deg_s": m["t_10deg"],
                "guard_corrections": n_corr,
                "diverged_at_s": diverged,
            }
        )
        print(
            f"  {name:36s} in-window {m['in_window']['rmse_deg']:9.3f} deg | "
            f"past {pdata.T_END:.0f}s {m['extrap']['rmse_deg']:11.3f} deg | "
            f"amplitude x{m['tail_amp_ratio']:.3f} | "
            f"still moving {m['tail_activity_ratio']:.3f}"
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(
        RESULT_DIR / "extrapolation_models.csv", "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    dec = error_decomposition(grey, s0, n_steps)
    print("\n  physics-basis error decomposition (max deg over the rollout):")
    for k, v in dec.items():
        print(f"    {k:30s} {v:.3e}")
    with open(
        RESULT_DIR / "error_decomposition.csv", "w", newline="", encoding="utf-8"
    ) as fh:
        w = csv.DictWriter(fh, fieldnames=["quantity", "max_deg"])
        w.writeheader()
        w.writerows([{"quantity": k, "max_deg": v} for k, v in dec.items()])

    path = _plot(t, true_traj, curves, duration)
    print(
        f"\nwrote {RESULT_DIR / 'extrapolation_models.csv'}, "
        f"{RESULT_DIR / 'error_decomposition.csv'} and {path}"
    )
    return rows


def _plot(t, true_traj, curves, duration):
    import plots

    fig, axes = plots.plt.subplots(2, 1, figsize=(7.8, 5.6), sharex=True)
    fig.patch.set_facecolor(plots.SURFACE)
    colour = {
        "state-space FIS": plots.RED,
        "state-space FIS + clipped inputs": plots.YELLOW,
        "state-space FIS + dissipation guard": plots.ORANGE,
        "physics-basis FIS (grey box)": plots.AQUA,
    }
    th_true = np.rad2deg(true_traj[:, [0, 2]])
    for j, ax in enumerate(axes):
        ax.plot(
            t,
            th_true[:, j],
            color=plots.BLUE,
            linewidth=3.0,
            label="actual (RK4)",
            zorder=2,
        )
        for name, pred in curves.items():
            ax.plot(
                t,
                np.rad2deg(pred[:, [0, 2]])[:, j],
                color=colour[name],
                linewidth=1.1,
                linestyle="--",
                label=name,
                zorder=5,
            )
        lo, hi = th_true[:, j].min(), th_true[:, j].max()
        pad = (hi - lo) * 0.35
        ax.set_ylim(lo - pad, hi + pad)
        ax.axvline(pdata.T_END, color=plots.INK, linewidth=1.1, linestyle=":")
        plots._style(ax, ylabel=rf"$\theta_{j + 1}$ (deg)")
        if j == 0:
            ax.text(
                pdata.T_END,
                hi + pad,
                " training data ends",
                ha="left",
                va="top",
                fontsize=plots.FS_SMALL,
                color=plots.INK,
            )
            ax.legend(loc="lower left", fontsize=plots.FS_SMALL, frameon=False, ncol=2)
    axes[-1].set_xlabel("t (seconds)", fontsize=plots.FS_LABEL, color=plots.INK)
    axes[0].set_title(
        f"Friction double pendulum: {pdata.T_END:.0f} s of training, "
        f"{duration:.0f} s of rollout",
        fontsize=plots.FS_TITLE,
        color=plots.INK,
    )
    fig.tight_layout()
    return plots._save(fig, "fig_extrapolation_models")


if __name__ == "__main__":
    main()
