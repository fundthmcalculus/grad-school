"""
n=2 rollout-stability experiment: physics-informed energy conservation.

Every rollout so far (angle-only, angle+velocity, moving-average features)
lets the fuzzy TSK regressor predict a completely free-form state delta,
with nothing stopping the rolled-out trajectory from gaining or losing
mechanical energy every step -- exactly the kind of error a real
(conservative, undamped) double pendulum cannot make. This tries the
natural fix: after each predicted step, project the predicted state back
onto the constant-energy manifold by rescaling the predicted angular
velocities.

Why rescaling omega works exactly (not approximately): kinetic energy

    T(omega_1, omega_2; theta_1, theta_2) =
        0.5*m1*l1^2*omega_1^2
      + 0.5*m2*(l1^2*omega_1^2 + l2^2*omega_2^2 + 2*l1*l2*omega_1*omega_2*cos(theta_1-theta_2))

is EXACTLY homogeneous of degree 2 in (omega_1, omega_2) jointly -- every
term is either omega_i^2 or the omega_1*omega_2 cross term, with theta held
fixed. So T(lambda*omega_1, lambda*omega_2) = lambda^2 * T(omega_1, omega_2)
for ANY lambda, with no approximation. Given a raw predicted state
(theta_new, omega_raw) and the true reference energy E0, solving

    T(lambda*omega_raw) + V(theta_new) = E0
    =>  lambda = sqrt(max(0, (E0 - V(theta_new)) / T(omega_raw)))

gives a closed-form correction that makes the corrected state land exactly
on the E0 energy shell, every step, by construction -- not a soft penalty,
an exact projection. This does not fix chaos (the corrected state can still
be on the wrong point of the right energy shell), but it removes the
additional, unphysical failure mode of energy drift compounding on top of
chaotic sensitivity.
"""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_fuzzy_ode import initialize_model
from n_pendulum_fuzzy_regression import (
    load_mimo_data,
    train_mimo,
    run_iterative_prediction,
)

DT = 0.01
THRESHOLD = 0.5
FEATURES = ["theta_1", "omega_1", "theta_2", "omega_2"]


def kinetic_energy(theta1, theta2, omega1, omega2, m1, m2, l1, l2):
    return 0.5 * m1 * l1**2 * omega1**2 + 0.5 * m2 * (
        l1**2 * omega1**2
        + l2**2 * omega2**2
        + 2 * l1 * l2 * omega1 * omega2 * np.cos(theta1 - theta2)
    )


def potential_energy(theta1, theta2, m1, m2, l1, l2, g):
    return -(m1 + m2) * g * l1 * np.cos(theta1) - m2 * g * l2 * np.cos(theta2)


def total_energy(theta1, theta2, omega1, omega2, m1, m2, l1, l2, g):
    return kinetic_energy(
        theta1, theta2, omega1, omega2, m1, m2, l1, l2
    ) + potential_energy(theta1, theta2, m1, m2, l1, l2, g)


def energy_correct_rollout(
    regressor, test_trajectory, params, n_steps=None, correct=True
):
    """Closed-loop rollout seeded with only the initial condition.

    If correct=True, rescales (omega_1, omega_2) after every step so the
    state lands exactly on the E0 energy shell. If correct=False, this is
    the plain uncorrected rollout (for a same-seed, same-model comparison).
    """
    m1, m2, l1, l2, g = params
    state0 = test_trajectory[FEATURES].iloc[0]
    E0 = total_energy(
        state0["theta_1"],
        state0["theta_2"],
        state0["omega_1"],
        state0["omega_2"],
        m1,
        m2,
        l1,
        l2,
        g,
    )

    total_steps = n_steps if n_steps is not None else len(test_trajectory) - 1
    state = state0.copy()
    rows = [state.to_dict()]
    n_clamped = 0

    for step in range(total_steps):
        x_now = pd.DataFrame([state[FEATURES].to_dict()])
        delta = regressor.predict(x_now).iloc[0]
        theta1_new = state["theta_1"] + delta["theta_1"]
        theta2_new = state["theta_2"] + delta["theta_2"]
        omega1_raw = state["omega_1"] + delta["omega_1"]
        omega2_raw = state["omega_2"] + delta["omega_2"]

        if correct:
            V_new = potential_energy(theta1_new, theta2_new, m1, m2, l1, l2, g)
            T_raw = kinetic_energy(
                theta1_new, theta2_new, omega1_raw, omega2_raw, m1, m2, l1, l2
            )
            budget = E0 - V_new
            if T_raw <= 1e-12 or budget <= 0:
                lam = 0.0
                n_clamped += 1
            else:
                lam = np.sqrt(budget / T_raw)
            omega1_new, omega2_new = lam * omega1_raw, lam * omega2_raw
        else:
            omega1_new, omega2_new = omega1_raw, omega2_raw

        state = pd.Series(
            {
                "theta_1": theta1_new,
                "omega_1": omega1_new,
                "theta_2": theta2_new,
                "omega_2": omega2_new,
            }
        )

        if not np.isfinite(state.values).all() or np.any(np.abs(state.values) > 1e4):
            for _ in range(total_steps - step):
                rows.append({f: np.nan for f in FEATURES})
            break
        rows.append(state.to_dict())

    if correct and n_clamped:
        print(
            f"  (energy budget exhausted / clamped to omega=0 on {n_clamped}/{total_steps} steps)"
        )
    return pd.DataFrame(rows)


def time_to_threshold(t, err, threshold=THRESHOLD):
    idx = np.where(err > threshold)[0]
    return t[idx[0]] if len(idx) else None


if __name__ == "__main__":
    train_results, test_results = initialize_model()
    tst = test_results.trajectories[0]
    pendulum = train_results.model
    params = (pendulum.m1, pendulum.m2, pendulum.l1, pendulum.l2, pendulum.g)

    print("Training angle+velocity MIMO regressor (same as the earlier ablation)...")
    Xtr, ytr = load_mimo_data(train_results.trajectories, FEATURES, window_size=1)
    Xte, yte = load_mimo_data([tst], FEATURES, window_size=1)
    res = train_mimo(FEATURES, Xtr, ytr, Xte, yte, window_size=1, n_bins=3)

    print("\nRollout WITHOUT energy correction (baseline)...")
    pred_raw = energy_correct_rollout(res["regressor"], tst, params, correct=False)
    print("\nRollout WITH energy correction...")
    pred_corr = energy_correct_rollout(res["regressor"], tst, params, correct=True)

    t = np.arange(len(pred_raw)) * DT
    actual = tst[FEATURES].iloc[: len(pred_raw)].reset_index(drop=True)

    err_raw = np.abs(pred_raw["theta_1"].values - actual["theta_1"].values)
    err_corr = np.abs(pred_corr["theta_1"].values - actual["theta_1"].values)

    ttt_raw = time_to_threshold(t, err_raw)
    ttt_corr = time_to_threshold(t, err_corr)
    print(f"\nTime to {THRESHOLD} rad error:")
    print(
        f"  uncorrected: {ttt_raw:.2f}s"
        if ttt_raw is not None
        else "  uncorrected: never"
    )
    print(
        f"  energy-corrected: {ttt_corr:.2f}s"
        if ttt_corr is not None
        else "  energy-corrected: never"
    )

    E0 = total_energy(
        actual["theta_1"].iloc[0],
        actual["theta_2"].iloc[0],
        actual["omega_1"].iloc[0],
        actual["omega_2"].iloc[0],
        *params,
    )
    E_actual = total_energy(
        actual["theta_1"],
        actual["theta_2"],
        actual["omega_1"],
        actual["omega_2"],
        *params,
    )
    E_raw = total_energy(
        pred_raw["theta_1"],
        pred_raw["theta_2"],
        pred_raw["omega_1"],
        pred_raw["omega_2"],
        *params,
    )
    E_corr = total_energy(
        pred_corr["theta_1"],
        pred_corr["theta_2"],
        pred_corr["omega_1"],
        pred_corr["omega_2"],
        *params,
    )

    print(f"\nEnergy drift (max |E(t)-E0|, E0={E0:.4f}):")
    print(f"  actual (integrator truth): {np.max(np.abs(E_actual - E0)):.2e}")
    print(f"  uncorrected rollout:       {np.max(np.abs(E_raw.dropna() - E0)):.2e}")
    print(f"  energy-corrected rollout:  {np.max(np.abs(E_corr.dropna() - E0)):.2e}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].semilogy(
        t, np.maximum(err_raw, 1e-6), color="#ff6b6b", lw=1.6, label="uncorrected"
    )
    axes[0].semilogy(
        t, np.maximum(err_corr, 1e-6), color="#2ca02c", lw=1.6, label="energy-corrected"
    )
    axes[0].axhline(
        THRESHOLD,
        color="#888",
        linestyle="--",
        lw=1,
        label=f"{THRESHOLD} rad threshold",
    )
    if ttt_raw is not None:
        axes[0].axvline(ttt_raw, color="#ff6b6b", linestyle=":", lw=1, alpha=0.6)
    if ttt_corr is not None:
        axes[0].axvline(ttt_corr, color="#2ca02c", linestyle=":", lw=1, alpha=0.6)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(r"$|\theta_1^{pred}-\theta_1^{actual}|$ (rad, log)")
    axes[0].set_title("Rollout Error: With vs. Without Energy Correction")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")

    axes[1].plot(t, E_actual - E0, color="#00d4ff", lw=2, label="actual (truth)")
    axes[1].plot(
        t, E_raw - E0, color="#ff6b6b", lw=1.4, label="uncorrected rollout", alpha=0.85
    )
    axes[1].plot(
        t,
        E_corr - E0,
        color="#2ca02c",
        lw=1.4,
        label="energy-corrected rollout",
        alpha=0.85,
    )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("$E(t) - E_0$")
    axes[1].set_title("Energy Drift Over the Rollout")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        "n=2: Physics-Informed Energy-Conserving Velocity Correction", fontweight="bold"
    )
    plt.tight_layout()
    fig.savefig(
        "figures/n2_energy_conservation_rollout.png", dpi=200, bbox_inches="tight"
    )
    print("\nSaved figures/n2_energy_conservation_rollout.png")
