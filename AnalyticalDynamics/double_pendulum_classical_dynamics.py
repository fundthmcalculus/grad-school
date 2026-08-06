"""
Classical-mechanics diagnostics for the Lagrangian double pendulum.

Complements test_double_pendulum.py (which focuses on fuzzy-regression
surrogate modeling) with the physics checks a dynamics write-up needs:
energy conservation of the integrated trajectory, sensitivity to initial
conditions (the signature of chaos), and the Cartesian bob trajectories.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from test_fuzzy_ode import DoublePendulum
from ode_helpers import angles_to_xy

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def total_energy(df, pendulum: DoublePendulum):
    """Kinetic + potential energy at every sample of a simulated trajectory."""
    th1, om1, th2, om2 = (
        df["theta_1"].values,
        df["omega_1"].values,
        df["theta_2"].values,
        df["omega_2"].values,
    )
    m1, m2, l1, l2, g = pendulum.m1, pendulum.m2, pendulum.l1, pendulum.l2, pendulum.g
    T = 0.5 * m1 * l1**2 * om1**2 + 0.5 * m2 * (
        l1**2 * om1**2 + l2**2 * om2**2 + 2 * l1 * l2 * om1 * om2 * np.cos(th1 - th2)
    )
    V = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
    return T + V, T, V


def plot_energy_conservation(pendulum, dt=0.001, duration=20.0):
    state0 = (120 * np.pi / 180, 0.0, -10 * np.pi / 180, 0.0)
    df = pendulum.simulate(state0, duration=duration, dt=dt)
    E, T, V = total_energy(df, pendulum)
    t = np.arange(len(df)) * dt

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(t, T, label="Kinetic $T$", color="#d62728")
    axes[0].plot(t, V, label="Potential $V$", color="#1f77b4")
    axes[0].plot(t, E, label="Total $E=T+V$", color="#2ca02c", lw=2)
    axes[0].set_ylabel("Energy (J, unit mass/length)")
    axes[0].set_title(f"Energy Budget of the Undamped Double Pendulum ($dt$={dt}s)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    drift = E - E[0]
    axes[1].plot(t, drift, color="#9467bd")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("$E(t) - E(0)$")
    axes[1].set_title(
        f"Energy Drift (max |drift| = {np.max(np.abs(drift)):.2e} J) — integrator error, not physics"
    )
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "energy_conservation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(
        f"Energy drift: max |dE| = {np.max(np.abs(drift)):.3e} J "
        f"({100 * np.max(np.abs(drift)) / np.abs(E[0]):.4f}% of |E0|)"
    )
    return df, E


def plot_chaos_sensitivity(pendulum, dt=0.01, duration=20.0, delta0=1e-4):
    base = (120 * np.pi / 180, 0.0, 60 * np.pi / 180, 0.0)
    perturbed = (base[0], base[1], base[2] + delta0, base[3])

    df_a = pendulum.simulate(base, duration=duration, dt=dt)
    df_b = pendulum.simulate(perturbed, duration=duration, dt=dt)
    t = np.arange(len(df_a)) * dt

    sep = np.sqrt(
        (df_a["theta_1"] - df_b["theta_1"]) ** 2
        + (df_a["theta_2"] - df_b["theta_2"]) ** 2
    )
    sep = sep.values
    sep[sep < 1e-14] = 1e-14

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(
        t, df_a["theta_2"], label=f"$\\theta_2(0)$={base[2]:.4f} rad", color="#00d4ff"
    )
    axes[0].plot(
        t,
        df_b["theta_2"],
        label=f"$\\theta_2(0)$={perturbed[2]:.4f} rad",
        color="#ff1744",
        alpha=0.8,
    )
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel(r"$\theta_2$ (rad)")
    axes[0].set_title(f"Two Trajectories, $\\Delta\\theta_2(0)$ = {delta0:.0e} rad")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(t, sep, color="#e07b00")
    # Fit an exponential to the early-time growth to estimate the largest Lyapunov exponent.
    fit_mask = (t > 0.5) & (t < duration * 0.5) & (sep < 1.0)
    if fit_mask.sum() > 10:
        p = np.polyfit(t[fit_mask], np.log(sep[fit_mask]), 1)
        lyap = p[0]
        axes[1].semilogy(
            t[fit_mask],
            np.exp(np.polyval(p, t[fit_mask])),
            "k--",
            label=f"fit: $\\lambda\\approx${lyap:.2f} s$^{{-1}}$",
        )
        axes[1].legend()
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$\|\Delta\theta\|$ (rad)")
    axes[1].set_title("Exponential Divergence of Nearby Trajectories")
    axes[1].grid(alpha=0.3, which="both")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "chaos_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    if fit_mask.sum() > 10:
        print(
            f"Estimated largest Lyapunov exponent: {lyap:.3f} 1/s "
            f"(e-folding time {1/lyap:.2f} s)"
            if lyap > 0
            else "No positive exponent found in fit window."
        )
    return df_a, df_b


def plot_trajectory_snapshots(pendulum, dt=0.01, duration=8.0):
    ics = [
        (
            170 * np.pi / 180,
            0.0,
            0.0 * np.pi / 180,
            0.0,
            "#00d4ff",
            "near-inverted (chaotic)",
        ),
        (
            30 * np.pi / 180,
            0.0,
            30 * np.pi / 180,
            0.0,
            "#ff9800",
            "small-angle (regular)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    for ax, (th1, om1, th2, om2, color, label) in zip(axes, ics):
        df = pendulum.simulate((th1, om1, th2, om2), duration=duration, dt=dt)
        x1, y1, x2, y2 = angles_to_xy(
            df["theta_1"].values, df["theta_2"].values, pendulum.l1, pendulum.l2
        )
        n = len(x2)
        for i in range(0, n - 1, 3):
            ax.plot(
                x2[i : i + 2],
                y2[i : i + 2],
                color=color,
                alpha=0.15 + 0.7 * i / n,
                lw=1,
            )
        ax.plot([0, x1[-1]], [0, y1[-1]], "o-", color="#333", lw=2, ms=5)
        ax.plot(
            [x1[-1], x2[-1]],
            [y1[-1], y2[-1]],
            "o-",
            color="#333",
            lw=2,
            ms=5,
            markerfacecolor=color,
        )
        ax.plot(0, 0, "ks", ms=8)
        ax.set_title(
            f"{label}\n$\\theta_1(0)$={th1*180/np.pi:.0f}°, $\\theta_2(0)$={th2*180/np.pi:.0f}°",
            fontsize=11,
        )
        ax.set_aspect("equal")
        ax.set_xlim(-2.2, 2.2)
        ax.set_ylim(-2.2, 0.5)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Bob-2 Trajectory Over {duration}s (color fades light→dark with time)",
        fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(FIG_DIR / "trajectory_snapshots.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pendulum = DoublePendulum(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
    print("Generating energy-conservation figure...")
    plot_energy_conservation(pendulum)
    print("Generating chaos-sensitivity figure...")
    plot_chaos_sensitivity(pendulum)
    print("Generating trajectory-snapshot figure...")
    plot_trajectory_snapshots(pendulum)
    print("Done.")
