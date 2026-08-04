"""
n=3 rollout-stability experiment: physics-inspired consequent equations,
generalizing n2_physics_informed_v2_rational.py via the automated symbolic
term-extraction in n_pendulum_physics_basis.py.

Same recipe that won 5x for n=2: divide each Cramer's-rule numerator term
by the exact, known denominator det(M(theta)) (m, l are known constants),
fit one sparse linear (no-intercept) consequent equation per output
(omega_i's equation sees only the terms belonging to omega_i's own
numerator -- 18 terms for n=3, vs 4 and 3 for the double pendulum's two
outputs), and roll theta forward from the UPDATED omega each step
(semi-implicit Euler).

Same train/test scenario as the original n=3 fuzzy study
(n_pendulum_fuzzy_regression.py): fan configuration base
theta=[120,60,0] deg, sweep theta_3 by [1.5,3.0] deg in 0.1 deg steps for
16 training trajectories, test at +2.05 deg -- so the comparison against
that study's black-box result (0.48s to 0.5rad error) is apples-to-apples.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from n_pendulum_fuzzy_regression import generate_family
from n_pendulum_physics_basis import derive_physics_basis, compute_features, state_cols

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


class PhysicsConsequentRegressorN:
    """n physics-structured consequent equations, one per angular acceleration,
    each a plain no-intercept linear regression on its own numerator terms."""

    def __init__(self, n):
        self.n = n
        self.models = [LinearRegression(fit_intercept=False) for _ in range(n)]

    def fit(self, X_per_output, y):
        for i in range(self.n):
            self.models[i].fit(X_per_output[i], y[:, i])
        return self

    def predict_row(self, basis, theta_row, omega_row):
        theta_arrs = [np.array([v]) for v in theta_row]
        omega_arrs = [np.array([v]) for v in omega_row]
        _denom, feats = compute_features(basis, theta_arrs, omega_arrs)
        return np.array([self.models[i].predict(feats[i])[0] for i in range(self.n)])


def build_dataset(trajectories, basis, theta_cols, omega_cols):
    n = basis.n
    all_X = [[] for _ in range(n)]
    all_y = []
    for df in trajectories:
        theta_arrs = [df[c].values[:-1] for c in theta_cols]
        omega_arrs = [df[c].values[:-1] for c in omega_cols]
        _denom, feats = compute_features(basis, theta_arrs, omega_arrs)
        for i in range(n):
            all_X[i].append(feats[i])
        all_y.append(np.diff(df[omega_cols].values, axis=0))
    X_per_output = [np.vstack(x) for x in all_X]
    y = np.vstack(all_y)
    return X_per_output, y


def rollout(regressor, basis, test_trajectory, theta_cols, omega_cols, dt, n_steps=None):
    n = basis.n
    theta = test_trajectory[theta_cols].iloc[0].values.astype(float)
    omega = test_trajectory[omega_cols].iloc[0].values.astype(float)
    total_steps = n_steps if n_steps is not None else len(test_trajectory) - 1
    rows = [dict(zip(theta_cols + omega_cols, np.concatenate([theta, omega])))]

    for step in range(total_steps):
        delta_omega = regressor.predict_row(basis, theta, omega)
        omega_new = omega + delta_omega
        theta_new = theta + omega_new * dt  # semi-implicit (Euler-Cromer)

        state = np.concatenate([theta_new, omega_new])
        if not np.isfinite(state).all() or np.any(np.abs(state) > 1e4):
            for _ in range(total_steps - step):
                rows.append({c: np.nan for c in theta_cols + omega_cols})
            break
        theta, omega = theta_new, omega_new
        rows.append(dict(zip(theta_cols + omega_cols, state)))

    return pd.DataFrame(rows)


def time_to_threshold(t, err, threshold=0.5):
    idx = np.where(err > threshold)[0]
    return t[idx[0]] if len(idx) else None


if __name__ == '__main__':
    n = 3
    dt = 0.01
    theta_cols, omega_cols = state_cols(n)

    print("Deriving physics basis terms for n=3 (Cramer's rule + known denominator)...")
    t0 = time.perf_counter()
    basis = derive_physics_basis(n, m_vals=(1.0, 1.0, 1.0), l_vals=(1.0, 1.0, 1.0), g_val=9.81)
    print(f"  derived in {time.perf_counter() - t0:.2f}s; "
          f"term counts per output: {[len(t) for t in basis.per_output_terms]}")

    print("\nGenerating n=3 training set (same fan-configuration scenario as the original study)...")
    family = generate_family(n, base_thetas_deg=[120.0, 60.0, 0.0], sweep_index=2,
                              sweep_deltas_deg=np.arange(1.5, 3.00001, 0.1), test_delta_deg=2.05,
                              dt=dt, duration=30.0)

    print("\nBuilding physics-basis features...")
    X_train, y_train = build_dataset(family.train_trajectories, basis, theta_cols, omega_cols)
    X_test, y_test = build_dataset([family.test_trajectory], basis, theta_cols, omega_cols)
    print(f"  training rows: {X_train[0].shape[0]}, feature counts per output: {[x.shape[1] for x in X_train]}")

    print("\nFitting n physics-structured consequent equations...")
    t1 = time.perf_counter()
    regressor = PhysicsConsequentRegressorN(n).fit(X_train, y_train)
    print(f"  fit time {time.perf_counter() - t1:.2f}s")

    print("\nCross-sectional (single-step delta-omega) fit on held-out test trajectory:")
    for i, col in enumerate(omega_cols):
        pred = regressor.models[i].predict(X_test[i])
        mse = mean_squared_error(y_test[:, i], pred)
        print(f"  {col}: R2={r2_score(y_test[:, i], pred):.6f}  RMSE={np.sqrt(mse):.6f}  "
              f"MAE={mean_absolute_error(y_test[:, i], pred):.6f}")

    print("\nRunning corrected open-loop rollout...")
    predicted = rollout(regressor, basis, family.test_trajectory, theta_cols, omega_cols, dt)
    t = np.arange(len(predicted)) * dt
    actual = family.test_trajectory[theta_cols + omega_cols].iloc[:len(predicted)].reset_index(drop=True)
    err_theta1 = np.abs(predicted['theta_1'].values - actual['theta_1'].values)

    ttt = time_to_threshold(t, err_theta1)
    print(f"  time to 0.5 rad error (theta_1): {ttt:.2f}s" if ttt is not None else "  never exceeded 0.5 rad")
    valid = ~predicted['theta_1'].isna()
    if valid.sum() > 2:
        for col in theta_cols + omega_cols:
            a = actual[col].values[valid.values]
            p = predicted[col].values[valid.values]
            print(f"  {col}: MAE={np.mean(np.abs(a - p)):.4f}  R2={r2_score(a, p):.4f}  "
                  f"(valid={valid.sum()}/{len(predicted)})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(t, actual['theta_1'].values, color='#00d4ff', lw=2, label='Actual')
    axes[0].plot(t[valid.values], predicted['theta_1'].values[valid.values], color='#ff1744', lw=1.6,
                 label='Predicted (physics-informed, n=3)', alpha=0.85)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel(r'$\theta_1$ (rad)')
    axes[0].set_title('Rollout: Actual vs. Predicted')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(t, np.maximum(err_theta1, 1e-8), color='#2ca02c', lw=1.6,
                      label='physics-informed rational, n=3')
    axes[1].axhline(0.5, color='#888', linestyle='--', lw=1, label='0.5 rad threshold')
    if ttt is not None:
        axes[1].axvline(ttt, color='#2ca02c', linestyle=':', lw=1, alpha=0.6)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(r'$|\theta_1^{pred}-\theta_1^{actual}|$ (rad, log)')
    axes[1].set_title('Rollout Error Growth')
    axes[1].legend()
    axes[1].grid(alpha=0.3, which='both')

    fig.suptitle('n=3: Physics-Inspired Consequent Equations (Automated via Symbolic Cramer\'s Rule)',
                 fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'n3_physics_informed_rollout.png', dpi=200, bbox_inches='tight')
    print("\nSaved figures/n3_physics_informed_rollout.png")
