"""
Cross-n comparison of the fuzzy TSK surrogate's open-loop rollout accuracy,
using the corrected (non-leaking) rollout from n_pendulum_fuzzy_regression.py.

Produces one figure: |predicted theta_1 - actual theta_1| vs time, log scale,
for n=2 (double pendulum), n=3 (fan), and n=5 (inverted zigzag). Also reports
the time at which each crosses a fixed 0.5 rad error threshold, as a single
comparable "how long before the surrogate is useless" number per chain length.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from test_fuzzy_ode import initialize_model
from n_pendulum_fuzzy_regression import (
    generate_family,
    load_mimo_data,
    train_mimo,
    run_iterative_prediction,
)

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
THRESHOLD = 0.5  # rad


def rollout_error(feature_names, train_trajectories, test_trajectory, dt):
    Xtr, ytr = load_mimo_data(train_trajectories, feature_names, window_size=1)
    Xte, yte = load_mimo_data([test_trajectory], feature_names, window_size=1)
    res = train_mimo(feature_names, Xtr, ytr, Xte, yte, window_size=1, n_bins=3)

    seed = test_trajectory[feature_names].iloc[:1].reset_index(drop=True)
    n_steps = len(test_trajectory) - 1
    predicted = run_iterative_prediction(
        res["regressor"], seed, feature_names, n_steps, window_size=1
    )

    t = np.arange(len(predicted)) * dt
    err = np.abs(
        predicted["theta_1"].values
        - test_trajectory["theta_1"].values[: len(predicted)]
    )
    return t, err


def time_to_threshold(t, err, threshold=THRESHOLD):
    idx = np.where(err > threshold)[0]
    return t[idx[0]] if len(idx) else None


if __name__ == "__main__":
    dt = 0.01
    results = {}

    print("n=2 (double pendulum)...")
    train2, test2 = initialize_model()
    t2, err2 = rollout_error(
        ["theta_1", "theta_2"], train2.trajectories, test2.trajectories[0], dt
    )
    results[2] = (t2, err2)

    print("n=3 (fan configuration)...")
    fam3 = generate_family(
        3,
        [120.0, 60.0, 0.0],
        2,
        np.arange(1.5, 3.00001, 0.1),
        test_delta_deg=2.05,
        dt=dt,
        duration=30.0,
    )
    t3, err3 = rollout_error(
        ["theta_1", "theta_2", "theta_3"],
        fam3.train_trajectories,
        fam3.test_trajectory,
        dt,
    )
    results[3] = (t3, err3)

    print("n=5 (inverted zigzag)...")
    fam5 = generate_family(
        5,
        [170.0, -170.0, 170.0, -170.0, 170.0],
        4,
        np.arange(1.5, 3.00001, 0.1),
        test_delta_deg=2.05,
        dt=dt,
        duration=30.0,
    )
    t5, err5 = rollout_error(
        [f"theta_{i}" for i in range(1, 6)],
        fam5.train_trajectories,
        fam5.test_trajectory,
        dt,
    )
    results[5] = (t5, err5)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {2: "#00d4ff", 3: "#ffb703", 5: "#ff1744"}
    for n, (t, err) in results.items():
        err_plot = np.maximum(err, 1e-6)
        ax.semilogy(t, err_plot, color=colors[n], lw=1.6, label=f"n={n}")
        ttt = time_to_threshold(t, err)
        if ttt is not None:
            ax.axvline(ttt, color=colors[n], linestyle=":", lw=1, alpha=0.6)
            print(f"n={n}: time to |err(theta_1)| > {THRESHOLD} rad = {ttt:.2f} s")
        else:
            print(f"n={n}: never exceeded {THRESHOLD} rad")

    ax.axhline(
        THRESHOLD,
        color="#888",
        linestyle="--",
        lw=1,
        label=f"{THRESHOLD} rad threshold",
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$|\theta_1^{pred}(t) - \theta_1^{actual}(t)|$ (rad, log scale)")
    ax.set_title(
        "Open-Loop Rollout Error Growth vs. Chain Length\n(fuzzy TSK, window=1, corrected rollout)"
    )
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "rollout_error_vs_n.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved figures/rollout_error_vs_n.png")
