from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

from odemodel import OdeSystem
from tribblefis.gaussian_regressor import MimoGaussianPredictor
from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor

# The FIS sees the two angles and their history -- current value, short-term
# moving average, and the average over the steps preceding that window -- and
# predicts the angular velocities. Accelerations are never modelled: the Euler
# step integrates theta directly from the predicted omega.
THETA_FEATURES = ['theta_1', 'theta_2']
OMEGA_TARGETS = ['omega_1', 'omega_2']
WINDOW_SIZE = 50  # short-term moving average spans this many steps
MEMORY_SIZE = 25  # long-term average spans the MEMORY_SIZE steps before that window
# Rows of history needed before a feature row has a fully populated long-term average.
HISTORY_LEN = WINDOW_SIZE + MEMORY_SIZE

# A 0th-order consequent, against the library default of '1st'. The three features
# per angle are near-collinear -- at dt=0.01 a 50-step average trails the current
# value by ~0.2 rad against a ~4.5 rad range, correlation 0.98 -- so a consequent
# that is linear in them is ill-conditioned and extrapolates without bound: it
# spikes to |omega| in the hundreds a few steps into a rollout and then locks the
# angles at a fixed point. The piecewise-constant consequent cannot extrapolate,
# and tracks ~3x closer over the full run. Widening the window is what supplies the
# velocity information; 50/25 tracks theta_1 to 0.1 rad for ~0.8 s, against ~0.2 s
# at the 5/3 window.
TSK_ORDER = '0th'

# Categorical slots 1 and 2 of the validated light-mode palette.
COLOR_ACTUAL = '#2a78d6'
COLOR_PREDICTED = '#eb6834'
COLOR_MUTED = '#52514e'


@dataclass
class PendulumParameters:
    theta1: float
    omega1: float
    theta2: float
    omega2: float
    dt: float
    duration: float

    @property
    def np(self) -> np.ndarray:
        return np.array([self.theta1, self.omega1, self.theta2, self.omega2])


@dataclass
class DataSimulation:
    # TODO - Make this more general, or generic-typed?
    params: PendulumParameters
    model: OdeSystem
    trajectories: list[pd.DataFrame]

    def get_combined_data(self: 'DataSimulation') -> tuple[DataFrame, DataFrame]:
        """Angles and their moving averages as inputs, angular velocities as targets.

        Both sides are taken at the same time step, so the fitted model is a
        state-to-velocity map that an Euler step can integrate directly. Rows
        whose long-term average has too little history behind it are dropped, so
        every training row carries a fully defined feature vector.

        `include_time` is deliberately off: `time_step` is an absolute row index,
        which would sit far outside its training range during a rollout.
        """
        extractor = MemoryWindowFeatureExtractor(window_size=WINDOW_SIZE, memory_size=MEMORY_SIZE)
        all_x = []
        all_y = []
        for trajectory in self.trajectories:
            features = extractor.prepare_sequences(trajectory, THETA_FEATURES, include_time=False)
            valid = ~features.isna().any(axis=1).values
            all_x.append(features[valid])
            all_y.append(trajectory[OMEGA_TARGETS][valid])
        return pd.concat(all_x, ignore_index=True), pd.concat(all_y, ignore_index=True)




class DoublePendulum(OdeSystem):
    """Double pendulum simulator using Lagrangian mechanics.

    TODO - alpha_2 was wrong until 797b69a: a sum where the cited reference has a
    product, so the system gained energy instead of conserving it. Figure 6.3 and
    anything else measured through this generator -- test_double_pendulum.py
    imports initialize_model from here -- was fit against the uncorrected
    equations and has to be re-run before it is quoted.
    """

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g

    @property
    def state_labels(self) -> list[str]:
        return ['theta_1', 'omega_1', 'theta_2', 'omega_2']

    @property
    def derivative_labels(self) -> list[str]:
        return ["omega_1", "alpha_1", "omega_2", "alpha_2"]

    def potential_energy(self, theta1, theta2):
        """Gravitational potential, measured from the pivot: zero with both arms horizontal."""
        return -self.m1 * self.g * self.l1 * np.cos(theta1) \
            - self.m2 * self.g * (self.l1 * np.cos(theta1) + self.l2 * np.cos(theta2))

    def kinetic_energy(self, theta1, omega1, theta2, omega2):
        """Kinetic energy, including the l1*l2 cross term between the two arms."""
        return 0.5 * self.m1 * self.l1**2 * omega1**2 \
            + 0.5 * self.m2 * (self.l1**2 * omega1**2 + self.l2**2 * omega2**2
                               + 2 * self.l1 * self.l2 * omega1 * omega2 * np.cos(theta1 - theta2))

    def energy(self, theta1, omega1, theta2, omega2):
        """Total energy. Conserved exactly by this system, so it scores a rollout."""
        return self.kinetic_energy(theta1, omega1, theta2, omega2) \
            + self.potential_energy(theta1, theta2)

    def equations_of_motion(self, state, t):
        """
        Compute double pendulum equations of motion using Lagrangian approach.

        State: [theta_1, omega_1, theta_2, omega_2]
        Returns: [omega_1, alpha_1, omega_2, alpha_2]
        """
        theta1, omega1, theta2, omega2 = state
        # Found here: https://web.mit.edu/jorloff/www/chaosTalk/double-pendulum/double-pendulum-en.html
        delta_theta = theta1 - theta2

        # Common terms
        denom1 = self.l1 *(2*self.m1 + self.m2 - self.m2 *np.cos(2*delta_theta))
        num11 = -self.g*(2*self.m1 + self.m2)*np.sin(theta1)
        num12 = -self.m2*self.g*np.sin(delta_theta - theta2) # theta1-2theta2
        num13 = -2*np.sin(delta_theta)*self.m2*(omega2**2 * self.l2 + omega1**2 *self.l1 * np.cos(delta_theta))
        alpha1 = (num11 + num12 + num13) / denom1

        num21 = omega1**2 *self.l1 *(self.m1+self.m2)
        num22 = self.g*(self.m1+self.m2)*np.cos(theta1)
        num23 = omega2**2 * self.l2*self.m2 * np.cos(delta_theta)
        denom2 = self.l2 *(2*self.m1 + self.m2 - self.m2 * np.cos(2*delta_theta))
        alpha2 = 2*np.sin(delta_theta)*(num21 + num22 + num23) / denom2

        return [omega1, alpha1, omega2, alpha2]


class DoublePendulumDamped(OdeSystem):
    """Double pendulum simulator with damping using Lagrangian mechanics with dissipation."""

    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81, c1=0.1, c2=0.1):
        self.m1 = m1
        self.m2 = m2
        self.l1 = l1
        self.l2 = l2
        self.g = g
        self.c1 = c1
        self.c2 = c2

    @property
    def state_labels(self) -> list[str]:
        return ['theta_1', 'omega_1', 'theta_2', 'omega_2']

    @property
    def derivative_labels(self) -> list[str]:
        return ["omega_1", "alpha_1", "omega_2", "alpha_2"]

    def equations_of_motion(self, state, t):
        """
        Compute damped double pendulum equations of motion using Lagrangian approach with Rayleigh dissipation.

        Damping dissipation function: F = (1/2)*c1*omega_1^2 + (1/2)*c2*omega_2^2

        State: [theta_1, omega_1, theta_2, omega_2]
        Returns: [omega_1, alpha_1, omega_2, alpha_2]
        """
        theta1, omega1, theta2, omega2 = state
        delta_theta = theta1 - theta2

        # Common denominator term: μ = m₁ + m₂*sin²(Δ)
        mu = self.m1 + self.m2 * np.sin(delta_theta)**2

        # Isolated equation for α₁ (upper pendulum angular acceleration)
        num1_1 = -self.g * (self.m1 + self.m2) * np.sin(theta1)
        num1_2 = self.m2 * self.g * np.sin(theta2) * np.cos(delta_theta)
        num1_3 = -self.m2 * self.l2 * omega2**2 * np.sin(delta_theta)
        num1_4 = -self.m2 * self.l1 * omega1**2 * np.sin(delta_theta) * np.cos(delta_theta)
        num1_5 = -(self.c1 * omega1) / self.l1
        num1_6 = (self.c2 * omega2 * np.cos(delta_theta)) / self.l2
        alpha1 = (num1_1 + num1_2 + num1_3 + num1_4 + num1_5 + num1_6) / (self.l1 * mu)

        # Isolated equation for α₂ (lower pendulum angular acceleration)
        num2_1 = self.g * (self.m1 + self.m2) * np.sin(theta1) * np.cos(delta_theta)
        num2_2 = -self.g * (self.m1 + self.m2) * np.sin(theta2)
        num2_3 = self.l1 * omega1**2 * (self.m1 + self.m2) * np.sin(delta_theta)
        num2_4 = self.m2 * self.l2 * omega2**2 * np.sin(delta_theta) * np.cos(delta_theta)
        num2_5 = (self.c1 * omega1 * np.cos(delta_theta)) / self.l1
        num2_6 = -(self.c2 * omega2 * (self.m1 + self.m2)) / (self.m2 * self.l2)
        alpha2 = (num2_1 + num2_2 + num2_3 + num2_4 + num2_5 + num2_6) / (self.l2 * mu)

        return [omega1, alpha1, omega2, alpha2]


def test_tribble_ode():
    """Fit a memory-augmented FIS on the angles alone and roll it out with Euler."""
    # 1) Create the simulation data for various initial conditions.
    train_results, test_results = initialize_model()
    X_combined, y_combined = train_results.get_combined_data()
    print(f"Training rows: {len(X_combined)}")
    print(f"Inputs:  {list(X_combined.columns)}")
    print(f"Outputs: {list(y_combined.columns)}")
    ode_m = MimoGaussianPredictor(tsk_order=TSK_ORDER)
    ode_m.fit(X_combined, y_combined)

    # 2) Euler rollout: theta_{n+1} = theta_n + omega_hat_n * dt. Only theta is
    # integrated, and only theta feeds back in, so the accelerations never enter.
    extractor = MemoryWindowFeatureExtractor(window_size=WINDOW_SIZE, memory_size=MEMORY_SIZE)
    actual_trajectory = test_results.trajectories[0]
    dt = test_results.params.dt

    # The moving averages need HISTORY_LEN samples behind them, so the rollout is
    # seeded with that many rows of the true trajectory. Error is zero by
    # construction over the seed, which is why the metrics below skip it.
    theta_rows = list(actual_trajectory[THETA_FEATURES].iloc[:HISTORY_LEN].values)
    omega_rows = [np.full(len(OMEGA_TARGETS), np.nan) for _ in range(HISTORY_LEN)]

    for _ in range(len(actual_trajectory) - HISTORY_LEN):
        history = pd.DataFrame(theta_rows[-HISTORY_LEN:], columns=THETA_FEATURES)
        features = extractor.prepare_sequences(history, THETA_FEATURES, include_time=False)
        omega = ode_m.predict(features.iloc[-1:]).values.flatten()
        theta = theta_rows[-1] + omega * dt
        if not np.isfinite(theta).all():
            print(f"Warning: Euler diverged at step {len(theta_rows)}, stopping early.")
            break
        theta_rows.append(theta)
        omega_rows.append(omega)

    predicted = pd.DataFrame(theta_rows, columns=THETA_FEATURES)
    predicted[OMEGA_TARGETS] = np.array(omega_rows)
    print(f"Euler rollout produced {len(predicted)} of {len(actual_trajectory)} steps "
          f"({HISTORY_LEN} seeded from truth).")

    # 3) Report how long the rollout tracks, rather than only whether it stayed finite.
    n = min(len(predicted), len(actual_trajectory))
    print("\nRollout accuracy past the seed window:")
    for col in THETA_FEATURES + OMEGA_TARGETS:
        error = np.abs(actual_trajectory[col].values[HISTORY_LEN:n] - predicted[col].values[HISTORY_LEN:n])
        print(f"  {col:8s} MAE={np.nanmean(error):8.4f}")
    theta1_error = np.abs(actual_trajectory['theta_1'].values[:n] - predicted['theta_1'].values[:n])
    for tol in (0.01, 0.1, 0.5, 1.0):
        exceeded = np.flatnonzero(theta1_error > tol)
        when = f"{exceeded[0] * dt:.2f} s" if exceeded.size else "never"
        print(f"  |theta_1 error| first exceeds {tol:>4}: {when}")

    # 4) Plot the traces: one panel per state, actual against the FIS rollout.
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Memory-augmented FIS rollout: angles and angular velocities',
                 fontsize=14, fontweight='bold')

    t_actual = np.arange(len(actual_trajectory)) * dt
    t_predicted = np.arange(len(predicted)) * dt
    for ax, col in zip(axes.flat, THETA_FEATURES + OMEGA_TARGETS):
        ax.plot(t_actual, actual_trajectory[col].values, '-',
                color=COLOR_ACTUAL, linewidth=2, label='Actual')
        ax.plot(t_predicted, predicted[col].values, '-',
                color=COLOR_PREDICTED, linewidth=2, label='FIS rollout')
        ax.axvline(HISTORY_LEN * dt, color=COLOR_MUTED, linestyle=':', linewidth=1,
                   label='end of seed window')
        units = 'rad' if col.startswith('theta') else 'rad/s'
        ax.set_title(col, fontsize=11)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel(f'{col} ({units})', fontsize=10)
        ax.grid(True, alpha=0.25)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)
        ax.legend(loc='best', fontsize=9, framealpha=0.9)

    fig.tight_layout()
    output_file = Path(__file__).parent / "tribble_ode_traces.png"
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nTraces saved to: {output_file}")

    print("Test completed successfully!")


def initialize_model() -> tuple[DataSimulation, DataSimulation]:
    pendulum = DoublePendulum()
    trajectories = []
    theta2s = np.arange(1.5, 3.00001, 0.1)  # TODO - 0.1
    train_params = PendulumParameters(theta1=120 * np.pi / 180,
                                omega1=0.0,
                                omega2=0.0,
                                dt=0.01,
                                duration=30.0,
                                theta2=0.0)
    for ij in range(len(theta2s)):
        theta2 = theta2s[ij]
        theta2 *= np.pi / 180
        ic = tuple([train_params.theta1, train_params.omega1, theta2, train_params.omega2])
        df = pendulum.simulate(ic, duration=train_params.duration, dt=train_params.dt)
        trajectories.append(df)

    print(f"Generated {len(trajectories)} trajectories")
    print(f"First trajectory shape: {trajectories[0].shape}")

    train_results = DataSimulation(
        trajectories=trajectories,
        params=train_params,
        model=pendulum,
    )

    # Test trajectory
    test_params = PendulumParameters(theta1=train_params.theta1, theta2=2.05 * np.pi / 180.0,
                                     omega1=train_params.omega1, omega2=train_params.omega2,
                                     dt=train_params.dt, duration=train_params.duration)
    test_ic = np.array([test_params.theta1, test_params.omega1, test_params.theta2, test_params.omega2])
    actual_trajectory = pendulum.simulate(test_ic, duration=train_params.duration, dt=train_params.dt)
    test_results = DataSimulation(
        trajectories=[actual_trajectory],
        params=test_params,
        model=pendulum,
    )
    return train_results, test_results


if __name__ == "__main__":
    test_tribble_ode()