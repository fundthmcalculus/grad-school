"""Generate trajectory_snapshots.png: bracket separation over time.

Shows how the two bracketing training ICs (2.0 and 2.1 deg) diverge over time for
both friction and frictionless regimes. Damping keeps them coherent; frictionless
causes exponential divergence.
"""

from math import cos, sin
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Constants from pendulum_data
G = 9.81
L1 = L2 = 1.0
M1 = M2 = 1.0
DAMPING = 0.15
T_START = 0.0
T_END = 10.0
TEST_T_END = 20.0
N_STEPS = 2000
H = (T_END - T_START) / N_STEPS  # 0.005 s

THETA1_DEG = 120.0


def rhs_double_reference(r, _t, damping1=0.0, damping2=0.0):
    """Double pendulum equations of motion (from arXiv:2504.13453 reference code)."""
    theta1, omega1, theta2, omega2 = r
    denom = 2 * M1 + M2 - M2 * cos(2 * theta1 - 2 * theta2)

    fomega1 = (
        -G * (2 * M1 + M2) * sin(theta1)
        - M2 * G * sin(theta1 - 2 * theta2)
        - 2
        * sin(theta1 - theta2)
        * M2
        * (omega2**2 * L2 + omega1**2 * L1 * cos(theta1 - theta2))
        - damping1 * omega1
    ) / (L1 * denom)

    fomega2 = (
        2
        * sin(theta1 - theta2)
        * (
            omega1**2 * L1 * (M1 + M2)
            + G * (M1 + M2) * cos(theta1)
            + omega2**2 * L2 * M2 * cos(theta1 - theta2)
        )
        - damping2 * omega2
    ) / (L2 * denom)

    return np.array([omega1, fomega1, omega2, fomega2], float)


def rk4_integrate(rhs, state0, n_steps, h=H):
    """Classical RK4 integration on a fixed grid."""
    trajectory = np.zeros((n_steps, len(state0)))
    trajectory[0] = state0

    for i in range(n_steps - 1):
        k1 = rhs(trajectory[i], 0)
        k2 = rhs(trajectory[i] + h * k1 / 2, 0)
        k3 = rhs(trajectory[i] + h * k2 / 2, 0)
        k4 = rhs(trajectory[i] + h * k3, 0)
        trajectory[i + 1] = trajectory[i] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return trajectory


def time_points(t_end=T_END):
    """Generate time points on the integration grid."""
    return np.arange(T_START, t_end, H)


def main():
    """Generate the bracket separation figure."""
    theta1_rad = np.radians(THETA1_DEG)
    ic_lower = np.array([theta1_rad, 0.0, np.radians(2.0), 0.0])  # 2.0 deg
    ic_upper = np.array([theta1_rad, 0.0, np.radians(2.1), 0.0])  # 2.1 deg

    configs = [
        ("friction", DAMPING),
        ("frictionless", 0.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (regime, damping) in enumerate(configs):
        ax = axes[idx]

        # Integrate to t=20 for both trajectories
        n_steps = int(round((TEST_T_END - T_START) / H))
        traj_lower = rk4_integrate(
            lambda r, t: rhs_double_reference(r, t, damping1=damping, damping2=damping),
            ic_lower,
            n_steps=n_steps,
        )
        traj_upper = rk4_integrate(
            lambda r, t: rhs_double_reference(r, t, damping1=damping, damping2=damping),
            ic_upper,
            n_steps=n_steps,
        )

        # Get time vector
        t_all = time_points(t_end=TEST_T_END)

        # Calculate separation: max angle difference across both joints (in degrees)
        sep_1 = np.abs(np.degrees(traj_lower[:, 0] - traj_upper[:, 0]))
        sep_2 = np.abs(np.degrees(traj_lower[:, 2] - traj_upper[:, 2]))
        separation = np.maximum(sep_1, sep_2)

        # Plot on log scale
        ax.semilogy(
            t_all, separation, linewidth=2.5, color="#1f77b4", label="Separation"
        )
        ax.axvline(
            x=10,
            color="red",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label="Training edge",
        )
        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Max angle separation (°)", fontsize=12)
        ax.set_title(f"Double pendulum, {regime}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim([1e-2, 1e3])
        ax.legend(loc="best", fontsize=11)

    plt.suptitle(
        "Bracket Separation: How Two Training ICs Diverge Over Time",
        fontsize=14,
        fontweight="bold",
        y=1.00,
    )
    plt.tight_layout()
    output_path = Path(__file__).parent / "figures" / "trajectory_snapshots.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"✓ Created {output_path}")
    plt.close()


if __name__ == "__main__":
    main()
