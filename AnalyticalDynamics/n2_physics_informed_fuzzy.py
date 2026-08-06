"""
n=2 rollout-stability experiment: physics-inspired (Lagrangian-structured)
consequent features.

Every prior ablation (angle-only, angle+velocity, moving-average) fed the
fuzzy TSK regressor generic state features and let it fit a fully free-form
local-linear consequent per rule. This experiment instead builds the input
feature columns from the exact nonlinear basis terms that appear in the
TRUE equations of motion (see DOUBLE_PENDULUM_REPORT.md SS2-3):

    alpha1 = [-g(2m1+m2)sin(th1) - m2 g sin(th1-2th2)
              - 2 sin(D) m2 (om2^2 l2 + om1^2 l1 cos(D))] / denom1(th)
    alpha2 = [2 sin(D) (om1^2 l1(m1+m2) + g(m1+m2)cos(th1)
              + om2^2 l2 m2 cos(D))] / denom2(th)          [D = th1-th2]

The basis terms {sin(th1), sin(th1-2*th2), sin(D)*om1^2, sin(D)*om2^2,
sin(D)*om1^2*cos(D), sin(D)*om2^2*cos(D), sin(D)*cos(th1), cos(D), cos(2D)}
are computed EXACTLY from the current state (they require no knowledge
beyond th1, th2, om1, om2 -- no mass/length/gravity values), and become
the fuzzy regressor's inputs in place of raw (th1,om1,th2,om2). The
regressor still fits its consequent coefficients from data, same as every
other experiment in this project -- the difference is that the functional
FORM is constrained to what the physics can actually produce, rather than
left as an arbitrary affine function of raw state. The hypothesis: bounded,
physically-sane basis functions should extrapolate more gracefully outside
the training manifold than raw polynomial/affine consequents can, since
trig functions can't blow up the way an arbitrary local-linear plane can.

Target and rollout procedure are unchanged from n_pendulum_fuzzy_regression.py
(predict delta-state, closed-loop rollout seeded with only the initial
condition -- no leakage).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from test_fuzzy_ode import initialize_model

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tribble-fis" / "src"))
from tribblefis.gaussian_regressor import MimoGaussianPredictor

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

STATE_COLS = ["theta_1", "omega_1", "theta_2", "omega_2"]
PHYSICS_BASIS_COLS = [
    "sin_th1",
    "sin_th1_minus_2th2",
    "sinD_om1sq",
    "sinD_om2sq",
    "sinD_om1sq_cosD",
    "sinD_om2sq_cosD",
    "sinD_costh1",
    "cosD",
    "cos2D",
]


def physics_basis_features(df: pd.DataFrame) -> pd.DataFrame:
    th1, om1, th2, om2 = df["theta_1"], df["omega_1"], df["theta_2"], df["omega_2"]
    D = th1 - th2
    sinD, cosD = np.sin(D), np.cos(D)
    out = pd.DataFrame(
        {
            "sin_th1": np.sin(th1),
            "sin_th1_minus_2th2": np.sin(th1 - 2 * th2),
            "sinD_om1sq": sinD * om1**2,
            "sinD_om2sq": sinD * om2**2,
            "sinD_om1sq_cosD": sinD * om1**2 * cosD,
            "sinD_om2sq_cosD": sinD * om2**2 * cosD,
            "sinD_costh1": sinD * np.cos(th1),
            "cosD": cosD,
            "cos2D": np.cos(2 * D),
        },
        index=df.index if hasattr(df, "index") else None,
    )
    return out


def build_dataset(trajectories):
    all_X, all_y = [], []
    for df in trajectories:
        feat_df = physics_basis_features(df)
        X = feat_df.iloc[:-1].values
        y = np.diff(df[STATE_COLS].values, axis=0)
        all_X.append(X)
        all_y.append(y)
    return np.vstack(all_X), np.vstack(all_y)


def rollout(regressor, test_trajectory, dt, n_steps=None):
    """Closed-loop rollout seeded with only the initial condition (no leakage)."""
    state = test_trajectory[STATE_COLS].iloc[0].copy()
    total_steps = n_steps if n_steps is not None else len(test_trajectory) - 1
    rows = [state.to_dict()]

    for step in range(total_steps):
        x_now = physics_basis_features(pd.DataFrame([state.to_dict()]))
        delta = regressor.predict(x_now).iloc[0]
        new_state = {col: state[col] + delta[col] for col in STATE_COLS}

        if not np.isfinite(list(new_state.values())).all() or any(
            abs(v) > 1e4 for v in new_state.values()
        ):
            for _ in range(total_steps - step):
                rows.append({c: np.nan for c in STATE_COLS})
            break
        state = pd.Series(new_state)
        rows.append(new_state)

    return pd.DataFrame(rows)


def time_to_threshold(t, err, threshold=0.5):
    idx = np.where(err > threshold)[0]
    return t[idx[0]] if len(idx) else None


if __name__ == "__main__":
    dt = 0.01
    print(
        "Generating n=2 training set (same 16-trajectory scenario as the original study)..."
    )
    train_results, test_results = initialize_model()
    tst = test_results.trajectories[0]

    print("\nBuilding physics-basis feature dataset...")
    X_train, y_train = build_dataset(train_results.trajectories)
    X_test, y_test = build_dataset([tst])
    print(f"  X_train={X_train.shape}, y_train={y_train.shape}")

    print(
        "\nTraining fuzzy TSK MIMO regressor (physics-basis features -> delta state)..."
    )
    t0 = time.perf_counter()
    regressor = MimoGaussianPredictor(
        n_output_buckets=3, tsk_order="1st", optimize_coefficients=True, random_state=42
    )
    X_train_df = pd.DataFrame(X_train, columns=PHYSICS_BASIS_COLS)
    y_train_df = pd.DataFrame(y_train, columns=STATE_COLS)
    X_test_df = pd.DataFrame(X_test, columns=PHYSICS_BASIS_COLS)
    y_test_df = pd.DataFrame(y_test, columns=STATE_COLS)
    regressor.fit(X_train_df, y_train_df)
    print(f"  fit time {time.perf_counter() - t0:.2f}s")

    y_pred_df = regressor.predict(X_test_df)
    print("\nCross-sectional (single-step delta) fit on held-out test trajectory:")
    for col in STATE_COLS:
        mse = mean_squared_error(y_test_df[col], y_pred_df[col])
        print(
            f"  {col}: R2={r2_score(y_test_df[col], y_pred_df[col]):.4f}  "
            f"RMSE={np.sqrt(mse):.4f}  MAE={mean_absolute_error(y_test_df[col], y_pred_df[col]):.4f}"
        )

    print("\nRunning corrected open-loop rollout...")
    predicted = rollout(regressor, tst, dt)
    t = np.arange(len(predicted)) * dt
    actual = tst[STATE_COLS].iloc[: len(predicted)].reset_index(drop=True)
    err_theta1 = np.abs(predicted["theta_1"].values - actual["theta_1"].values)

    ttt = time_to_threshold(t, err_theta1)
    print(
        f"  time to 0.5 rad error (theta_1): {ttt:.2f}s"
        if ttt is not None
        else "  never exceeded 0.5 rad"
    )
    valid = ~predicted["theta_1"].isna()
    if valid.sum() > 2:
        for col in STATE_COLS:
            a = actual[col].values[valid.values]
            p = predicted[col].values[valid.values]
            print(
                f"  {col}: MAE={np.mean(np.abs(a - p)):.4f}  R2={r2_score(a, p):.4f}  "
                f"(valid={valid.sum()}/{len(predicted)})"
            )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(t, actual["theta_1"].values, color="#00d4ff", lw=2, label="Actual")
    axes[0].plot(
        t[valid.values],
        predicted["theta_1"].values[valid.values],
        color="#ff1744",
        lw=1.6,
        label="Predicted (physics-basis features)",
        alpha=0.85,
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(r"$\theta_1$ (rad)")
    axes[0].set_title("Rollout: Actual vs. Predicted")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(
        t,
        np.maximum(err_theta1, 1e-6),
        color="#9467bd",
        lw=1.6,
        label="physics-basis features, 16 traj",
    )
    axes[1].axhline(0.5, color="#888", linestyle="--", lw=1, label="0.5 rad threshold")
    if ttt is not None:
        axes[1].axvline(ttt, color="#9467bd", linestyle=":", lw=1, alpha=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$|\theta_1^{pred}-\theta_1^{actual}|$ (rad, log)")
    axes[1].set_title("Rollout Error Growth")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle(
        "n=2: Physics-Inspired (Lagrangian-Structured) Consequent Features",
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "n2_physics_informed_rollout.png", dpi=200, bbox_inches="tight"
    )
    print("\nSaved figures/n2_physics_informed_rollout.png")
