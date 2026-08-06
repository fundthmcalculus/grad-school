"""
Fuzzy TSK surrogate-modeling study for the n-link pendulum, generalizing
test_double_pendulum.py (n=2) to arbitrary chain length n.

Same train/test scenario as the n=2 study: fix a base configuration,
sweep one joint's angle over a narrow 1.5 deg band in 0.1 deg steps to get
16 "nearby" training trajectories, hold out a test trajectory whose swept
angle sits between two training grid points, then see whether a fuzzy
TSK regressor can learn the one-step state-delta map from the family.

Trimmed relative to the n=2 study to keep runtime reasonable at higher n:
MIMO window sizes {1, 3} only (not {1,3,5,7,10}), and no
"actual + 2-nearest-training-neighbors" overlay animation (kept as a
static plot instead). Both trims are called out in the written report,
not silently applied.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from n_pendulum_symbolic import NPendulum
from n_pendulum_animation import chain_xy, chain_energy
from ode_helpers import (
    load_and_prepare_data,
    train_and_evaluate_single_step,
    find_nearest_trajectories,
)

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "tribble-fis" / "src"))
from tribblefis.gaussian_regressor import MimoGaussianPredictor

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


@dataclass
class NPendulumFamily:
    pendulum: NPendulum
    train_trajectories: list
    test_trajectory: pd.DataFrame
    base_thetas_deg: list
    sweep_index: int
    dt: float
    duration: float


def generate_family(
    n,
    base_thetas_deg,
    sweep_index,
    sweep_deltas_deg,
    test_delta_deg,
    m_vals=None,
    l_vals=None,
    g=9.81,
    dt=0.01,
    duration=30.0,
) -> NPendulumFamily:
    pendulum = NPendulum(n, m_vals=m_vals, l_vals=l_vals, g=g)

    train_trajectories = []
    for delta in sweep_deltas_deg:
        thetas = list(base_thetas_deg)
        thetas[sweep_index] = thetas[sweep_index] + delta
        state0 = np.zeros(2 * n)
        state0[0::2] = np.array(thetas) * np.pi / 180.0
        df = pendulum.simulate(tuple(state0), duration=duration, dt=dt)
        train_trajectories.append(df)

    thetas = list(base_thetas_deg)
    thetas[sweep_index] = thetas[sweep_index] + test_delta_deg
    state0 = np.zeros(2 * n)
    state0[0::2] = np.array(thetas) * np.pi / 180.0
    test_trajectory = pendulum.simulate(tuple(state0), duration=duration, dt=dt)

    return NPendulumFamily(
        pendulum=pendulum,
        train_trajectories=train_trajectories,
        test_trajectory=test_trajectory,
        base_thetas_deg=base_thetas_deg,
        sweep_index=sweep_index,
        dt=dt,
        duration=duration,
    )


def check_family_energy(family: NPendulumFamily):
    """Sanity check: every generated trajectory should conserve energy."""
    p = family.pendulum
    max_pct = 0.0
    for df in family.train_trajectories + [family.test_trajectory]:
        theta = df[[f"theta_{i}" for i in range(1, p.n + 1)]].values
        omega = df[[f"omega_{i}" for i in range(1, p.n + 1)]].values
        E = chain_energy(theta, omega, p.m_vals, p.l_vals, p.g)
        pct = 100 * np.max(np.abs(E - E[0])) / abs(E[0]) if E[0] != 0 else 0.0
        max_pct = max(max_pct, pct)
    print(
        f"  Family energy check: max drift across all trajectories = {max_pct:.5f}% of |E0|"
    )
    return max_pct


def mimo_feature_steps(window_size, feature_names):
    if window_size == 1:
        return [(feat, 0, feat) for feat in feature_names]
    return [
        (f"{feat}_step{i}", i, feat)
        for i in range(window_size)
        for feat in feature_names
    ]


def get_mimo_df(df, window_size, feature_names):
    steps = mimo_feature_steps(window_size, feature_names)
    X = pd.DataFrame()
    for step_name, offset, col in steps:
        X[step_name] = df[col].iloc[offset : -(window_size - offset)].values
    return X


def load_mimo_data(trajectories, feature_names, window_size=1):
    all_X, all_y = [], []
    for df in trajectories:
        y = np.diff(df[feature_names].iloc[window_size - 1 :].values, axis=0)
        if window_size == 1:
            X = df[feature_names].iloc[:-1].values
        else:
            X = get_mimo_df(df, window_size, feature_names).values
        all_X.append(X)
        all_y.append(y)
    return np.vstack(all_X), np.vstack(all_y)


def train_mimo(feature_names, X_train, y_train, X_test, y_test, window_size, n_bins=3):
    steps = mimo_feature_steps(window_size, feature_names)
    input_cols = [s[0] for s in steps]
    regressor = MimoGaussianPredictor(
        n_output_buckets=n_bins,
        tsk_order="1st",
        optimize_coefficients=True,
        random_state=42,
    )
    X_train_df = pd.DataFrame(X_train, columns=input_cols)
    y_train_df = pd.DataFrame(y_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=input_cols)
    y_test_df = pd.DataFrame(y_test, columns=feature_names)
    regressor.fit(X_train_df, y_train_df)
    y_pred_df = regressor.predict(X_test_df)

    metrics = {}
    for col in feature_names:
        mse = mean_squared_error(y_test_df[col], y_pred_df[col])
        metrics[col] = dict(
            mse=mse,
            rmse=np.sqrt(mse),
            mae=mean_absolute_error(y_test_df[col], y_pred_df[col]),
            r2=r2_score(y_test_df[col], y_pred_df[col]),
        )
    return dict(
        regressor=regressor,
        metrics=metrics,
        y_test=y_test_df,
        y_pred=y_pred_df,
        window_size=window_size,
        input_cols=input_cols,
    )


def run_iterative_prediction(
    regressor, initial_window_df, feature_names, n_steps, window_size=1
):
    running_state = initial_window_df.copy()
    if window_size > 1:
        running_state = get_mimo_df(initial_window_df, window_size, feature_names)
    diverged_at = None

    for step in range(n_steps):
        if diverged_at:
            running_state = pd.concat(
                [
                    running_state,
                    pd.DataFrame(
                        [np.full(running_state.shape[1], np.nan)],
                        columns=running_state.columns,
                    ),
                ],
                ignore_index=True,
            )
            continue
        next_delta_df = regressor.predict(running_state[-window_size:])
        if window_size == 1:
            new_state = running_state.iloc[-1, :] + next_delta_df
        else:
            steps = mimo_feature_steps(window_size, feature_names)
            new_row = {}
            for step_name, offset, col in steps:
                if offset < window_size - 1:
                    next_step_name = f"{col}_step{offset + 1}"
                    new_row[step_name] = running_state.iloc[-1][next_step_name]
                else:
                    most_recent = f"{col}_step{window_size - 1}"
                    new_row[step_name] = (
                        running_state.iloc[-1][most_recent] + next_delta_df.iloc[0][col]
                    )
            new_state = pd.DataFrame([new_row])

        if np.any(np.isnan(new_state)) or np.any(np.abs(new_state) > 1e4):
            diverged_at = step + 1
            print(f"    Warning: diverged at step {diverged_at}")
        else:
            running_state = pd.concat([running_state, new_state], ignore_index=True)

    return running_state


def render_comparison_gif(
    family: NPendulumFamily,
    predicted_theta_df,
    out_path,
    max_frames=320,
    fps=28,
    trail_len=60,
    figsize=(11, 6),
    dpi=110,
):
    p = family.pendulum
    n = p.n
    feature_names = [f"theta_{i}" for i in range(1, n + 1)]
    actual_theta = (
        family.test_trajectory[feature_names].iloc[: len(predicted_theta_df)].values
    )
    predicted_theta = predicted_theta_df[feature_names].values

    x_act, y_act = chain_xy(actual_theta, p.l_vals)
    pred_filled = pd.DataFrame(predicted_theta).ffill().fillna(0.0).values
    x_pred, y_pred = chain_xy(pred_filled, p.l_vals)

    n_samples = len(predicted_theta)
    step = max(1, n_samples // max_frames)
    frame_idx = np.arange(0, n_samples, step)
    n_frames = len(frame_idx)
    total_l = sum(p.l_vals)

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    for ax, title in zip(axes, ["Actual", "Predicted (fuzzy TSK, iterative rollout)"]):
        ax.set_facecolor("#16213e")
        ax.set_xlim(-total_l * 1.15, total_l * 1.15)
        ax.set_ylim(-total_l * 1.15, total_l * 0.35)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, color="#888")
        ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")

    fig.suptitle(
        f"{n}-Link Pendulum: Actual vs. Fuzzy-TSK Prediction",
        color="white",
        fontsize=14,
        fontweight="bold",
    )
    time_text = fig.text(0.5, 0.02, "", ha="center", color="#aaaaaa", fontsize=10)

    (trail_a,) = axes[0].plot([], [], "-", color="#00d4ff", linewidth=1.1, alpha=0.55)
    (chain_a,) = axes[0].plot(
        [], [], "o-", color="#e0e0e0", lw=2.2, ms=5, markerfacecolor="white"
    )
    (bob_a,) = axes[0].plot([], [], "o", color="#00d4ff", ms=8)

    (trail_p,) = axes[1].plot([], [], "-", color="#ff6b6b", linewidth=1.1, alpha=0.55)
    (chain_p,) = axes[1].plot(
        [], [], "o-", color="#e0e0e0", lw=2.2, ms=5, markerfacecolor="white"
    )
    (bob_p,) = axes[1].plot([], [], "o", color="#ff6b6b", ms=8)

    def init():
        for artist in [trail_a, chain_a, bob_a, trail_p, chain_p, bob_p]:
            artist.set_data([], [])
        time_text.set_text("")
        return trail_a, chain_a, bob_a, trail_p, chain_p, bob_p, time_text

    def update(frame):
        i = frame_idx[frame]
        t_start = max(0, i - trail_len * step)
        trail_a.set_data(
            x_act[t_start : i + 1 : max(1, step // 2 + 1), -1],
            y_act[t_start : i + 1 : max(1, step // 2 + 1), -1],
        )
        xs_a = np.concatenate([[0.0], x_act[i]])
        ys_a = np.concatenate([[0.0], y_act[i]])
        chain_a.set_data(xs_a, ys_a)
        bob_a.set_data([x_act[i, -1]], [y_act[i, -1]])

        trail_p.set_data(
            x_pred[t_start : i + 1 : max(1, step // 2 + 1), -1],
            y_pred[t_start : i + 1 : max(1, step // 2 + 1), -1],
        )
        xs_p = np.concatenate([[0.0], x_pred[i]])
        ys_p = np.concatenate([[0.0], y_pred[i]])
        chain_p.set_data(xs_p, ys_p)
        bob_p.set_data([x_pred[i, -1]], [y_pred[i, -1]])

        time_text.set_text(f"t = {i * family.dt:.2f} s")
        return trail_a, chain_a, bob_a, trail_p, chain_p, bob_p, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=True
    )
    writer = animation.PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    print(
        f"  Saved {out_path.name}: {n_frames} frames, {out_path.stat().st_size / 1e6:.2f} MB"
    )


def run_study(
    n,
    base_thetas_deg,
    sweep_index,
    label,
    out_prefix,
    m_vals=None,
    l_vals=None,
    g=9.81,
    dt=0.01,
    duration=30.0,
    window_sizes=(1, 3),
    n_bins=3,
):
    print("=" * 70)
    print(f"FUZZY TSK REGRESSION STUDY: n={n} ({label})")
    print("=" * 70)

    sweep_deltas = np.arange(1.5, 3.00001, 0.1)
    t0 = time.perf_counter()
    family = generate_family(
        n,
        base_thetas_deg,
        sweep_index,
        sweep_deltas,
        test_delta_deg=2.05,
        m_vals=m_vals,
        l_vals=l_vals,
        g=g,
        dt=dt,
        duration=duration,
    )
    print(
        f"Generated {len(family.train_trajectories)} training trajectories "
        f"+ 1 test trajectory in {time.perf_counter() - t0:.2f}s"
    )
    check_family_energy(family)

    feature_names = [f"theta_{i}" for i in range(1, n + 1)]

    print("\nSingle-step model (predicts absolute next theta_1)...")
    X_tr, y_tr = load_and_prepare_data(
        family.train_trajectories, feature_names, feature_names, window_size=1
    )
    X_te, y_te = load_and_prepare_data(
        [family.test_trajectory], feature_names, feature_names, window_size=1
    )
    single_step = train_and_evaluate_single_step(
        n_bins, feature_names, X_tr, y_tr, X_te, y_te
    )

    mimo_results = {}
    for ws in window_sizes:
        print(f"\nMIMO model, window={ws}...")
        Xw_tr, yw_tr = load_mimo_data(
            family.train_trajectories, feature_names, window_size=ws
        )
        Xw_te, yw_te = load_mimo_data(
            [family.test_trajectory], feature_names, window_size=ws
        )
        t1 = time.perf_counter()
        res = train_mimo(
            feature_names, Xw_tr, yw_tr, Xw_te, yw_te, window_size=ws, n_bins=n_bins
        )
        print(f"  fit+eval time {time.perf_counter() - t1:.2f}s")
        for col in feature_names:
            m = res["metrics"][col]
            print(f"  {col}: R2={m['r2']:.4f} RMSE={m['rmse']:.4f}")
        mimo_results[ws] = res

    best_ws = min(
        window_sizes
    )  # window=1 for iterative-rollout comparability with the n=2 study
    print(f"\nIterative rollout (window={best_ws})...")
    # Seed with ONLY the first `best_ws` rows of the test trajectory -- not the whole
    # trajectory. The original n=2 pipeline seeded with the full actual test dataframe,
    # so its "prediction" for the first len(actual) rows was a verbatim copy of the
    # ground truth, not a real rollout (see DOUBLE_PENDULUM_REPORT.md addendum). This is
    # a genuine open-loop rollout: every row past the seed is generated purely from the
    # model's own prior predictions.
    # Only the theta_i columns -- not omega_i. The regressor's predicted delta
    # DataFrame has exactly `feature_names` columns; adding a Series with extra
    # (omega) index entries to it produces NaN in those entries via pandas'
    # column alignment, which then trips the divergence check on step 1 even
    # though nothing actually diverged.
    seed = family.test_trajectory[feature_names].iloc[:best_ws].reset_index(drop=True)
    n_steps = len(family.test_trajectory) - best_ws
    predicted = run_iterative_prediction(
        mimo_results[best_ws]["regressor"],
        seed,
        feature_names,
        n_steps,
        window_size=best_ws,
    )

    valid_mask = ~predicted[feature_names[0]].isna()
    n_valid = valid_mask.sum()
    print(
        f"  Valid rollout steps: {n_valid}/{len(predicted)} "
        f"({n_valid * dt:.2f}s / {duration:.2f}s)"
    )
    if n_valid > 2:
        for col in feature_names:
            act = family.test_trajectory[col].values[: len(predicted)][
                valid_mask.values
            ]
            pred = predicted[col].values[valid_mask.values]
            print(
                f"  {col}: MAE={np.mean(np.abs(act - pred)):.4f}  R2={r2_score(act, pred):.4f}"
            )

    print("\nRendering actual-vs-predicted GIF...")
    render_comparison_gif(
        family, predicted, FIG_DIR.parent / f"{out_prefix}_fuzzy_comparison.gif"
    )

    print("\nPlotting nearest-training-trajectory comparison...")
    nearest = find_nearest_trajectories(
        family.test_trajectory, family.train_trajectories, k=2, features=feature_names
    )
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes.flatten()
    t_test = np.arange(len(family.test_trajectory)) * dt
    for idx, col in enumerate(feature_names):
        ax = axes[idx]
        ax.plot(
            t_test,
            family.test_trajectory[col].values,
            color="#00d4ff",
            lw=2,
            label="Test (reference)",
            zorder=5,
        )
        t_pred = np.arange(len(predicted)) * dt
        ax.plot(
            t_pred[valid_mask],
            predicted[col].values[valid_mask],
            color="#ff1744",
            lw=1.6,
            label="Test (predicted)",
            alpha=0.85,
            zorder=4,
        )
        for rank, (tr_idx, dist, tr_df) in enumerate(nearest):
            t_tr = np.arange(len(tr_df)) * dt
            ax.plot(
                t_tr,
                tr_df[col].values,
                "--",
                color=["#00DD00", "#00AA00"][rank],
                lw=1.2,
                alpha=0.6,
                label=f"Train {tr_idx} (d={dist:.3f})",
            )
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(
        f"n={n} ({label}): Test vs. Predicted vs. Nearest Training",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / f"{out_prefix}_nearest_training.png", dpi=180, bbox_inches="tight"
    )
    plt.close(fig)

    return dict(
        family=family,
        single_step=single_step,
        mimo=mimo_results,
        predicted=predicted,
        n_valid=n_valid,
    )


if __name__ == "__main__":
    print("\n\n### n=3 STUDY ###\n")
    result3 = run_study(
        n=3,
        base_thetas_deg=[120.0, 60.0, 0.0],
        sweep_index=2,
        label="fan configuration, sweeping theta_3",
        out_prefix="n3",
    )

    print("\n\n### n=5 STUDY (inverted zigzag) ###\n")
    result5 = run_study(
        n=5,
        base_thetas_deg=[170.0, -170.0, 170.0, -170.0, 170.0],
        sweep_index=4,
        label="inverted zigzag, sweeping theta_5",
        out_prefix="n5",
    )

    print("\nAll studies complete.")
