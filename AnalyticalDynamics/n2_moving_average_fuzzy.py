"""
n=2 rollout-stability experiment: fuzzy TSK regressor with moving-average
position features, on a larger training set.

Inputs per timestep, per angle (theta_1, theta_2): the current value, a
3-sample trailing moving average, and a 9-sample trailing moving average --
6 input features total. This is a compact alternative to the raw-lagged
MIMO windowing used earlier (window={1,3,5,7,10}), which degraded past
window=3 as raw lagged columns multiplied faster than they added
information. Multi-scale moving averages compress the same "recent history"
signal into a fixed 6 columns regardless of how far back the long average
looks, and (like a lagged window) implicitly encode a finite-difference-ish
velocity signal without predicting omega separately.

Output: same as every other regressor in this project -- the one-step
state DELTA (delta theta_1, delta theta_2), not omega and not an
integration step. Rollout is the same closed-loop procedure as
n_pendulum_fuzzy_regression.py: predict delta, add to current state, and
feed the result back in (correctly seeded -- see that module's docstring
for the leakage bug this avoids repeating).

Training set: widened relative to the original 16-trajectory study (theta_2
in a narrow 1.5 deg band) to theta_2 in a full 30 deg band, more than an
order of magnitude wider, to see whether that alone helps rollout
generalization independent of the feature change.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from test_fuzzy_ode import DoublePendulum
from n_pendulum_fuzzy_regression import mimo_feature_steps  # unused here, kept for column-name parity if needed

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "tribble-fis" / "src"))
from tribblefis.gaussian_regressor import MimoGaussianPredictor

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

MA_SHORT = 3
MA_LONG = 9
FEATURE_NAMES = ['theta_1', 'theta_2']
INPUT_COLS = ['theta_1', 'theta_2', 'ma3_theta_1', 'ma3_theta_2', 'ma9_theta_1', 'ma9_theta_2']


def add_moving_average_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing moving averages (inclusive of the current sample) per angle."""
    out = df.copy()
    for feat in FEATURE_NAMES:
        out[f'ma3_{feat}'] = out[feat].rolling(MA_SHORT, min_periods=1).mean()
        out[f'ma9_{feat}'] = out[feat].rolling(MA_LONG, min_periods=1).mean()
    return out


def build_dataset(trajectories):
    all_X, all_y = [], []
    for df in trajectories:
        feat_df = add_moving_average_features(df)
        X = feat_df[INPUT_COLS].iloc[:-1].values
        y = np.diff(feat_df[FEATURE_NAMES].values, axis=0)
        all_X.append(X)
        all_y.append(y)
    return np.vstack(all_X), np.vstack(all_y)


def generate_training_set(theta1_deg=120.0, theta2_range_deg=(-15.0, 15.0), step_deg=0.3,
                           test_theta2_deg=7.35, dt=0.01, duration=30.0):
    pendulum = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    theta2_grid = np.arange(theta2_range_deg[0], theta2_range_deg[1] + 1e-9, step_deg)

    trajectories = []
    for theta2 in theta2_grid:
        state0 = (theta1_deg * np.pi / 180, 0.0, theta2 * np.pi / 180, 0.0)
        df = pendulum.simulate(state0, duration=duration, dt=dt)
        trajectories.append(df)

    test_state0 = (theta1_deg * np.pi / 180, 0.0, test_theta2_deg * np.pi / 180, 0.0)
    test_trajectory = pendulum.simulate(test_state0, duration=duration, dt=dt)

    return pendulum, trajectories, test_trajectory, theta2_grid


def rollout(regressor, test_trajectory, dt, n_steps=None):
    """Closed-loop rollout seeded with only the first MA_LONG raw rows (no leakage)."""
    seed_len = MA_LONG
    buffer = test_trajectory[FEATURE_NAMES].iloc[:seed_len].reset_index(drop=True)
    total_steps = n_steps if n_steps is not None else len(test_trajectory) - seed_len

    predicted_rows = [buffer.iloc[i].to_dict() for i in range(seed_len)]

    for step in range(total_steps):
        window_df = pd.DataFrame(predicted_rows[-max(MA_LONG, seed_len):])
        feat_df = add_moving_average_features(window_df)
        x_now = feat_df[INPUT_COLS].iloc[-1:].reset_index(drop=True)

        delta = regressor.predict(x_now)
        current = predicted_rows[-1]
        new_row = {feat: current[feat] + delta.iloc[0][feat] for feat in FEATURE_NAMES}

        if not np.isfinite(list(new_row.values())).all() or any(abs(v) > 1e4 for v in new_row.values()):
            # Pad remainder with NaN and stop -- genuine numerical blow-up.
            for _ in range(total_steps - step):
                predicted_rows.append({feat: np.nan for feat in FEATURE_NAMES})
            break
        predicted_rows.append(new_row)

    return pd.DataFrame(predicted_rows)


def time_to_threshold(t, err, threshold=0.5):
    idx = np.where(err > threshold)[0]
    return t[idx[0]] if len(idx) else None


if __name__ == '__main__':
    print("Generating widened n=2 training set (theta_2 in [-15, 15] deg, step 0.3 deg)...")
    t0 = time.perf_counter()
    pendulum, trajectories, test_trajectory, theta2_grid = generate_training_set()
    print(f"  {len(trajectories)} training trajectories generated in {time.perf_counter() - t0:.2f}s "
          f"(vs. 16 in the original study)")

    print("\nBuilding moving-average feature dataset...")
    X_train, y_train = build_dataset(trajectories)
    X_test, y_test = build_dataset([test_trajectory])
    print(f"  X_train={X_train.shape}, y_train={y_train.shape}")

    print("\nTraining fuzzy TSK MIMO regressor (current + MA3 + MA9 -> delta state)...")
    t1 = time.perf_counter()
    regressor = MimoGaussianPredictor(n_output_buckets=3, tsk_order="1st",
                                       optimize_coefficients=True, random_state=42)
    X_train_df = pd.DataFrame(X_train, columns=INPUT_COLS)
    y_train_df = pd.DataFrame(y_train, columns=FEATURE_NAMES)
    X_test_df = pd.DataFrame(X_test, columns=INPUT_COLS)
    y_test_df = pd.DataFrame(y_test, columns=FEATURE_NAMES)
    regressor.fit(X_train_df, y_train_df)
    print(f"  fit time {time.perf_counter() - t1:.2f}s")

    y_pred_df = regressor.predict(X_test_df)
    print("\nCross-sectional (single-step delta) fit on held-out test trajectory:")
    for col in FEATURE_NAMES:
        mse = mean_squared_error(y_test_df[col], y_pred_df[col])
        print(f"  {col}: R2={r2_score(y_test_df[col], y_pred_df[col]):.4f}  "
              f"RMSE={np.sqrt(mse):.4f}  MAE={mean_absolute_error(y_test_df[col], y_pred_df[col]):.4f}")

    print("\nRunning corrected open-loop rollout...")
    dt = 0.01
    predicted = rollout(regressor, test_trajectory, dt)
    t = np.arange(len(predicted)) * dt
    actual = test_trajectory[FEATURE_NAMES].iloc[:len(predicted)].reset_index(drop=True)
    err_theta1 = np.abs(predicted['theta_1'].values - actual['theta_1'].values)

    ttt = time_to_threshold(t, err_theta1)
    print(f"  time to 0.5 rad error (theta_1): {ttt:.2f}s" if ttt is not None else "  never exceeded 0.5 rad")
    valid = ~predicted['theta_1'].isna()
    if valid.sum() > 2:
        for col in FEATURE_NAMES:
            a = actual[col].values[valid.values]
            p = predicted[col].values[valid.values]
            print(f"  {col}: MAE={np.mean(np.abs(a - p)):.4f}  R2={r2_score(a, p):.4f}  "
                  f"(valid={valid.sum()}/{len(predicted)})")

    # Comparison figure against the earlier angle-only and angle+velocity results.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(t, actual['theta_1'].values, color='#00d4ff', lw=2, label='Actual')
    axes[0].plot(t[valid.values], predicted['theta_1'].values[valid.values], color='#ff1744', lw=1.6,
                 label='Predicted (MA features)', alpha=0.85)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel(r'$\theta_1$ (rad)')
    axes[0].set_title('Rollout: Actual vs. Predicted')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(t, np.maximum(err_theta1, 1e-6), color='#2ca02c', lw=1.6,
                      label=f'current+MA3+MA9 ({len(trajectories)} traj.)')
    axes[1].axhline(0.5, color='#888', linestyle='--', lw=1, label='0.5 rad threshold')
    if ttt is not None:
        axes[1].axvline(ttt, color='#2ca02c', linestyle=':', lw=1, alpha=0.6)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(r'$|\theta_1^{pred}-\theta_1^{actual}|$ (rad, log)')
    axes[1].set_title('Rollout Error Growth')
    axes[1].legend()
    axes[1].grid(alpha=0.3, which='both')

    fig.suptitle('n=2: Moving-Average-Feature Fuzzy Regressor, Wider Training Set', fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'n2_moving_average_rollout.png', dpi=200, bbox_inches='tight')
    print("\nSaved figures/n2_moving_average_rollout.png")
