"""
n=2 rollout-stability experiment, v2: physics-inspired consequent with the
KNOWN rational (division) structure included, not just the numerator terms.

n2_physics_informed_fuzzy.py (v1) fed the raw numerator basis terms
{sin(th1), sin(D)*om1^2, cos(D), ...} into the standard multi-rule fuzzy
TSK machinery and did worse than a plain angle+velocity baseline (0.21s vs
0.60s). The reason: a locally-LINEAR consequent cannot represent a
division by a state-dependent denominator, and clustering the fuzzy
antecedents on the transformed basis features (rather than raw state)
scrambled the partitioning.

The true equations of motion are

    alpha1 = [-g(2m1+m2)sin(th1) - m2 g sin(th1-2th2)
              - 2 sin(D) m2 (om2^2 l2 + om1^2 l1 cos(D))] / denom1(th)
    alpha2 = [2 sin(D) (om1^2 l1(m1+m2) + g(m1+m2)cos(th1)
              + om2^2 l2 m2 cos(D))] / denom2(th)
    denom1(th) = l1(2m1+m2-m2*cos(2D)),  denom2(th) = l2(2m1+m2-m2*cos(2D))

denom1/denom2 depend only on theta (no mass/length values need to be
*learned* to compute them -- l1,l2,m1,m2 are known system constants, same
assumption already used for the energy-conservation experiment). Dividing
the numerator basis terms by the EXACT, known denominator turns the
target into something that genuinely IS linear in the resulting features,
with fixed coefficients (-(2m1+m2)g, -m2*g, -2*m2*l2, -2*m2*l1, ...) that a
plain linear regression can recover exactly.

Sanity check confirmed this: fitting each omega's 3-4 relevant features
with plain sklearn LinearRegression gets R^2=0.99 and recovers the true
physical coefficients (scaled by dt) almost exactly. Routing the SAME
features through MimoGaussianPredictor's fuzzy-clustering machinery (even
at its floor of 2 output buckets) instead gave R^2 < 0 -- the clustering
step is unstable on this feature parameterization (the 1/denom division
produces a very different, harder-to-partition value range than raw state).
Since the whole point here is "one physics-structured consequent equation"
per output, not a fuzzy partition of many, this uses plain linear
regression directly: mathematically, a TSK system with exactly one rule
that always fires is linear regression, so this is not a departure from
the "consequent equation" framing, just skipping machinery that isn't
needed (and is actively unstable) for a single global rule. Each output's
equation also only sees the basis terms that physically belong to it
(omega_1's equation never sees any of omega_2's terms and vice versa) --
a second, sparser form of physics-informed structure beyond the basis
functions themselves.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from test_fuzzy_ode import initialize_model

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

STATE_COLS = ["theta_1", "omega_1", "theta_2", "omega_2"]
RATIONAL_BASIS_COLS = [
    "sin_th1_over_d1",
    "sin_th1m2th2_over_d1",
    "sinD_om2sq_over_d1",
    "sinD_om1sq_cosD_over_d1",
    "sinD_om1sq_over_d2",
    "sinD_costh1_over_d2",
    "sinD_om2sq_cosD_over_d2",
]


def rational_basis_features(
    df: pd.DataFrame, l1: float, l2: float, m1: float, m2: float
) -> pd.DataFrame:
    th1, om1, th2, om2 = df["theta_1"], df["omega_1"], df["theta_2"], df["omega_2"]
    D = th1 - th2
    sinD, cosD = np.sin(D), np.cos(D)
    bracket = 2 * m1 + m2 - m2 * np.cos(2 * D)  # shared bracket in both denominators
    denom1 = l1 * bracket
    denom2 = l2 * bracket
    return pd.DataFrame(
        {
            "sin_th1_over_d1": np.sin(th1) / denom1,
            "sin_th1m2th2_over_d1": np.sin(th1 - 2 * th2) / denom1,
            "sinD_om2sq_over_d1": sinD * om2**2 / denom1,
            "sinD_om1sq_cosD_over_d1": sinD * om1**2 * cosD / denom1,
            "sinD_om1sq_over_d2": sinD * om1**2 / denom2,
            "sinD_costh1_over_d2": sinD * np.cos(th1) / denom2,
            "sinD_om2sq_cosD_over_d2": sinD * om2**2 * cosD / denom2,
        }
    )


ALPHA1_COLS = [
    "sin_th1_over_d1",
    "sin_th1m2th2_over_d1",
    "sinD_om2sq_over_d1",
    "sinD_om1sq_cosD_over_d1",
]
ALPHA2_COLS = ["sinD_om1sq_over_d2", "sinD_costh1_over_d2", "sinD_om2sq_cosD_over_d2"]


class PhysicsConsequentRegressor:
    """Two physics-structured consequent equations, one per angular acceleration.

    Each is a plain (no-intercept) linear regression restricted to exactly
    the basis terms that physically belong to it -- omega_1's equation
    never sees any of omega_2's terms and vice versa. Mathematically
    equivalent to a degenerate single-rule TSK consequent per output.
    """

    def __init__(self):
        self.lr1 = LinearRegression(fit_intercept=False)
        self.lr2 = LinearRegression(fit_intercept=False)

    def fit(self, X_df: pd.DataFrame, y_df: pd.DataFrame):
        self.lr1.fit(X_df[ALPHA1_COLS].values, y_df["omega_1"].values)
        self.lr2.fit(X_df[ALPHA2_COLS].values, y_df["omega_2"].values)
        return self

    def predict(self, X_df: pd.DataFrame) -> pd.DataFrame:
        p1 = self.lr1.predict(X_df[ALPHA1_COLS].values)
        p2 = self.lr2.predict(X_df[ALPHA2_COLS].values)
        return pd.DataFrame({"omega_1": p1, "omega_2": p2})


OMEGA_COLS = ["omega_1", "omega_2"]


def build_dataset(trajectories, l1, l2, m1, m2):
    """Target is delta-OMEGA only (angular acceleration integrated over dt).

    Delta-theta is not fit at all: it is exactly omega*dt to leading order,
    a kinematic identity, not something acceleration-shaped features (which
    only see omega^2, never raw omega -- so can't even see its sign) have
    any business predicting. Rolling theta forward from the *updated* omega
    (semi-implicit/Euler-Cromer) below is the "integrate, stepwise" step.
    """
    all_X, all_y = [], []
    for df in trajectories:
        feat_df = rational_basis_features(df, l1, l2, m1, m2)
        X = feat_df.iloc[:-1].values
        y = np.diff(df[OMEGA_COLS].values, axis=0)
        all_X.append(X)
        all_y.append(y)
    return np.vstack(all_X), np.vstack(all_y)


def rollout(regressor, test_trajectory, l1, l2, m1, m2, dt, n_steps=None):
    state = test_trajectory[STATE_COLS].iloc[0].copy()
    total_steps = n_steps if n_steps is not None else len(test_trajectory) - 1
    rows = [state.to_dict()]

    for step in range(total_steps):
        x_now = rational_basis_features(pd.DataFrame([state.to_dict()]), l1, l2, m1, m2)
        delta_omega = regressor.predict(x_now).iloc[0]
        omega1_new = state["omega_1"] + delta_omega["omega_1"]
        omega2_new = state["omega_2"] + delta_omega["omega_2"]
        # Semi-implicit (Euler-Cromer) integration: advance theta using the
        # UPDATED omega, not the old one -- the "integrate, stepwise" step.
        theta1_new = state["theta_1"] + omega1_new * dt
        theta2_new = state["theta_2"] + omega2_new * dt
        new_state = {
            "theta_1": theta1_new,
            "omega_1": omega1_new,
            "theta_2": theta2_new,
            "omega_2": omega2_new,
        }

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
    print("Generating n=2 training set (same 16-trajectory scenario as v1)...")
    train_results, test_results = initialize_model()
    tst = test_results.trajectories[0]
    pendulum = train_results.model
    l1, l2, m1, m2 = pendulum.l1, pendulum.l2, pendulum.m1, pendulum.m2

    print("\nBuilding rational (numerator/known-denominator) basis features...")
    X_train, y_train = build_dataset(train_results.trajectories, l1, l2, m1, m2)
    X_test, y_test = build_dataset([tst], l1, l2, m1, m2)
    print(f"  X_train={X_train.shape}, y_train={y_train.shape}")

    print(
        "\nFitting two physics-structured consequent equations "
        "(one per angular acceleration, each restricted to its own basis terms)..."
    )
    t0 = time.perf_counter()
    regressor = PhysicsConsequentRegressor()
    X_train_df = pd.DataFrame(X_train, columns=RATIONAL_BASIS_COLS)
    y_train_df = pd.DataFrame(y_train, columns=OMEGA_COLS)
    X_test_df = pd.DataFrame(X_test, columns=RATIONAL_BASIS_COLS)
    y_test_df = pd.DataFrame(y_test, columns=OMEGA_COLS)
    regressor.fit(X_train_df, y_train_df)
    print(f"  fit time {time.perf_counter() - t0:.2f}s")
    print(f"  alpha1 coefficients {dict(zip(ALPHA1_COLS, regressor.lr1.coef_))}")
    print(f"  alpha2 coefficients {dict(zip(ALPHA2_COLS, regressor.lr2.coef_))}")

    y_pred_df = regressor.predict(X_test_df)
    print(
        "\nCross-sectional (single-step delta-omega) fit on held-out test trajectory:"
    )
    for col in OMEGA_COLS:
        mse = mean_squared_error(y_test_df[col], y_pred_df[col])
        print(
            f"  {col}: R2={r2_score(y_test_df[col], y_pred_df[col]):.6f}  "
            f"RMSE={np.sqrt(mse):.6f}  MAE={mean_absolute_error(y_test_df[col], y_pred_df[col]):.6f}"
        )

    print("\nRunning corrected open-loop rollout...")
    predicted = rollout(regressor, tst, l1, l2, m1, m2, dt)
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
                f"  {col}: MAE={np.mean(np.abs(a - p)):.6f}  R2={r2_score(a, p):.6f}  "
                f"(valid={valid.sum()}/{len(predicted)})"
            )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(t, actual["theta_1"].values, color="#00d4ff", lw=2, label="Actual")
    axes[0].plot(
        t[valid.values],
        predicted["theta_1"].values[valid.values],
        color="#ff1744",
        lw=1.6,
        label="Predicted (rational physics basis)",
        alpha=0.85,
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(r"$\theta_1$ (rad)")
    axes[0].set_title("Rollout: Actual vs. Predicted")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(
        t,
        np.maximum(err_theta1, 1e-8),
        color="#d62728",
        lw=1.6,
        label="rational physics basis, 16 traj, single global consequent",
    )
    axes[1].axhline(0.5, color="#888", linestyle="--", lw=1, label="0.5 rad threshold")
    if ttt is not None:
        axes[1].axvline(ttt, color="#d62728", linestyle=":", lw=1, alpha=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$|\theta_1^{pred}-\theta_1^{actual}|$ (rad, log)")
    axes[1].set_title("Rollout Error Growth")
    axes[1].legend()
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle(
        "n=2: Physics-Inspired Consequent, v2 -- Known Rational (Division) Structure",
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "n2_physics_informed_v2_rollout.png", dpi=200, bbox_inches="tight"
    )
    print("\nSaved figures/n2_physics_informed_v2_rollout.png")
