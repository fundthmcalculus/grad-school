"""
Closes the open question flagged in DOUBLE_PENDULUM_REPORT.md / N3_N5:
every rollout result so far used angle-only inputs (theta_1, theta_2), never
velocities. Does adding (theta, omega) fix the sub-second open-loop
collapse, or is the 16-trajectory/1.5deg training manifold too narrow
regardless of feature set?

Answer: it helps the cross-sectional (single-step) fit meaningfully, and
roughly doubles the time-to-0.5rad-error (0.32s -> 0.6s), but does not fix
the fundamental problem -- the velocity-augmented rollout error oscillates
back above threshold repeatedly rather than saturating monotonically, but
never stays below it. Chaos amplification dominates well within the 30s
horizon either way.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from test_fuzzy_ode import initialize_model
from n_pendulum_fuzzy_regression import load_mimo_data, train_mimo, run_iterative_prediction

DT = 0.01
THRESHOLD = 0.5


def rollout_error(train_trajectories, test_trajectory, feature_names):
    Xtr, ytr = load_mimo_data(train_trajectories, feature_names, window_size=1)
    Xte, yte = load_mimo_data([test_trajectory], feature_names, window_size=1)
    res = train_mimo(feature_names, Xtr, ytr, Xte, yte, window_size=1, n_bins=3)

    seed = test_trajectory[feature_names].iloc[:1].reset_index(drop=True)
    n_steps = len(test_trajectory) - 1
    predicted = run_iterative_prediction(res['regressor'], seed, feature_names, n_steps, window_size=1)

    t = np.arange(len(predicted)) * DT
    err = np.abs(predicted['theta_1'].values - test_trajectory['theta_1'].values[:len(predicted)])
    return t, err, res


def time_to_threshold(t, err, threshold=THRESHOLD):
    idx = np.where(err > threshold)[0]
    return t[idx[0]] if len(idx) else None


if __name__ == '__main__':
    train_results, test_results = initialize_model()
    tst = test_results.trajectories[0]

    print("Angles only (theta_1, theta_2)...")
    t_a, err_a, res_a = rollout_error(train_results.trajectories, tst, ['theta_1', 'theta_2'])
    print(f"  time to {THRESHOLD} rad: {time_to_threshold(t_a, err_a):.2f}s")

    print("\nAngles + velocities (theta_1, omega_1, theta_2, omega_2)...")
    t_b, err_b, res_b = rollout_error(train_results.trajectories, tst,
                                       ['theta_1', 'omega_1', 'theta_2', 'omega_2'])
    print(f"  time to {THRESHOLD} rad: {time_to_threshold(t_b, err_b):.2f}s")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.semilogy(t_a, np.maximum(err_a, 1e-6), color='#00d4ff', lw=1.6, label='angles only (theta_1, theta_2)')
    ax.semilogy(t_b, np.maximum(err_b, 1e-6), color='#ffb703', lw=1.6, label='angles + velocities (theta, omega)')
    ax.axhline(THRESHOLD, color='#888', linestyle='--', lw=1, label=f'{THRESHOLD} rad threshold')
    for t, err, c in [(t_a, err_a, '#00d4ff'), (t_b, err_b, '#ffb703')]:
        ttt = time_to_threshold(t, err)
        if ttt is not None:
            ax.axvline(ttt, color=c, linestyle=':', lw=1, alpha=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'$|\theta_1^{pred}(t) - \theta_1^{actual}(t)|$ (rad, log scale)')
    ax.set_title('n=2 Rollout Error: Does Adding Velocity Inputs Help?')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    fig.savefig('figures/rollout_error_velocity_ablation.png', dpi=200, bbox_inches='tight')
    print("\nSaved figures/rollout_error_velocity_ablation.png")
