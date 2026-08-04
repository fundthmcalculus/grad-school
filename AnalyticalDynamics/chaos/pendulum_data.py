"""Paper-faithful RK4 data generation for arXiv:2504.13453.

Ramachandruni et al., "Using Machine Learning and Neural Networks to Analyze and
Predict Chaos in Multi-Pendulum and Chaotic Systems" (arXiv:2504.13453).

The generator here reproduces the *time-step approach* dataset of section 2.2 of
that paper, as it is actually implemented in the authors' codebase
(CTNN-ASDRP/ICBBT-IEEE-Xplore-...-Codebase-, notebook `NEW/NEWLSTM (5).ipynb`
cell 4 and `NEW FRICTION/NEW Friction LSTM (1) (1).ipynb` cell 4):

  * fixed-step classical RK4, h = 10/2000 = 0.005 s, t in [0, 10), 2000 samples
  * g = 9.81, l1 = l2 = 1 m, m1 = m2 = 1 kg, released from rest
  * theta_1(0) = 120 deg held fixed; theta_2(0) swept 0.0 .. 3.0 deg by 0.1
  * friction variant: -damping_i * omega_i added to the *numerator* of the
    omega_i equation, damping1 = damping2 = 0.15
  * angles stored in DEGREES (the authors convert inside the integration loop)

Two deliberate departures from the reference notebooks, both fixed here and both
recorded in REPRODUCTION_REPORT.md:

  1. The reference `angles` list contains 31 entries, two of which are typos
     ([122, 0.7] and [122, 1.8] where the pattern demands 120). The paper text
     says 30 initial angles. We sweep the intended 31-point grid 0.0..3.0 with
     theta_1(0) = 120 throughout.
  2. The reference double-pendulum trajectories omit theta_2(0) = 0.7 and 1.8
     from the saved .npy files (they were saved under the 122 names). We keep
     the full grid.

Triple pendulum: the paper uses Yesilyurt's n-point-mass formulation. We use
this repository's already-validated symbolic Lagrangian model
(`n_pendulum_symbolic.NPendulum`, validated in `n_pendulum_validation.py` against
the hand-derived n=2 equations to machine precision over 200 random states), and
`validate_against_reference()` below re-checks the n=2 reduction against the
paper's own closed-form right-hand side every run. Do not trust one derivation.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from math import cos, sin
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from n_pendulum_symbolic import make_state_space  # noqa: E402

# ---------------------------------------------------------------------------
# Paper constants (section 2, "Initial conditions")
# ---------------------------------------------------------------------------
G = 9.81
L1 = L2 = L3 = 1.0
M1 = M2 = M3 = 1.0
DAMPING = 0.15  # damping1 == damping2 (== damping3) in the reference notebooks

T_START = 0.0
T_END = 10.0
N_STEPS = 2000
H = (T_END - T_START) / N_STEPS  # 0.005 s

#: theta_1(0) = 120 deg, theta_2(0) swept over this grid (degrees).
TRAIN_THETA2_DEG = np.round(np.arange(0.0, 3.0 + 1e-9, 0.1), 2)
#: The paper's held-out "in-between" initial condition.
TEST_THETA2_DEG = 2.05
THETA1_DEG = 120.0

DATA_DIR = Path(__file__).resolve().parent / "data"


def time_points() -> np.ndarray:
    """The 2000 sample times, matching ``np.arange(0, 10, 0.005)``."""
    return np.arange(T_START, T_END, H)


# ---------------------------------------------------------------------------
# Double pendulum: the paper's own right-hand side, transcribed verbatim
# ---------------------------------------------------------------------------
def rhs_double_reference(r, _t, damping1=0.0, damping2=0.0):
    """The reference codebase's `f`, transcribed from Preprocessing.py:19-33.

    State ordering is the authors': [theta1, omega1, theta2, omega2], radians.
    Damping terms sit inside the numerator exactly as the friction notebook has
    them, which is dimensionally a torque-per-unit-something rather than a
    clean viscous joint torque. We reproduce it as written rather than
    "correcting" it, because the paper's friction numbers depend on it.
    """
    theta1, omega1, theta2, omega2 = r
    denom = 2 * M1 + M2 - M2 * cos(2 * theta1 - 2 * theta2)

    fomega1 = (
        -G * (2 * M1 + M2) * sin(theta1)
        - M2 * G * sin(theta1 - 2 * theta2)
        - 2 * sin(theta1 - theta2) * M2 * (omega2**2 * L2 + omega1**2 * L1 * cos(theta1 - theta2))
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


def rk4_integrate(rhs, state0, n_steps=N_STEPS, h=H):
    """Classical RK4 on a fixed grid, in the authors' loop order.

    Returns an (n_steps, len(state0)) array of states. The state at index i is
    the state *before* the i-th step is taken, so row 0 is exactly state0 --
    this is what the reference notebooks store.
    """
    q = np.asarray(state0, dtype=float).copy()
    out = np.empty((n_steps, q.size), dtype=float)
    t = T_START
    for i in range(n_steps):
        out[i] = q
        k1 = h * rhs(q, t)
        k2 = h * rhs(q + 0.5 * k1, t + 0.5 * h)
        k3 = h * rhs(q + 0.5 * k2, t + 0.5 * h)
        k4 = h * rhs(q + k3, t + h)
        q = q + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += h
    return out


# ---------------------------------------------------------------------------
# N-pendulum via this repo's validated symbolic model
# ---------------------------------------------------------------------------
def make_rhs_n(n, damping=0.0):
    """Right-hand side for the n-link chain, interleaved [th_1, om_1, ...].

    Built from `n_pendulum_symbolic.make_state_space`, which forms the
    Euler-Lagrange equations with SymPy and solves M(q) qdd = f(q, qd)
    numerically each call. Damping is applied the same way the reference
    notebooks apply it: subtracted from the acceleration of each link,
    proportional to that link's angular rate.
    """
    base, _model = make_state_space(n, tuple([1.0] * n), tuple([1.0] * n), G)

    if damping == 0.0:
        return base

    def rhs(state, t):
        d = np.asarray(base(state, t), dtype=float).copy()
        d[1::2] -= damping * np.asarray(state, dtype=float)[1::2]
        return d

    return rhs


# ---------------------------------------------------------------------------
# Provenance check: two independent derivations must agree
# ---------------------------------------------------------------------------
def validate_against_reference(n_trials=200, seed=42, tol=1e-9):
    """Assert the symbolic n=2 model reproduces the paper's closed form.

    Both are point-mass double pendulums with absolute angles from the downward
    vertical, so they must agree exactly. Raises AssertionError with the worst
    offender if they do not.
    """
    rng = np.random.default_rng(seed)
    rhs_sym = make_rhs_n(2, damping=0.0)
    worst = 0.0
    worst_state = None
    for _ in range(n_trials):
        state = rng.uniform(-np.pi, np.pi, 4)
        state[1::2] = rng.uniform(-5.0, 5.0, 2)
        a = np.asarray(rhs_sym(state, 0.0), float)
        b = rhs_double_reference(state, 0.0)
        err = float(np.max(np.abs(a - b)))
        if err > worst:
            worst, worst_state = err, state.copy()
    assert worst < tol, (
        f"symbolic n=2 disagrees with the paper's closed form by {worst:.3e} "
        f"at state {worst_state}"
    )
    return worst


def energy_drift_double(damping=0.0):
    """Relative energy drift of an undamped 120 deg / 0 deg run over 10 s.

    A sanity gate on the integrator: for damping=0 this must be tiny. Returns
    (E0, max relative |dE/E_scale|). Energy uses the pivot as the potential
    zero, so E can pass through 0; we normalise by the total kinetic+potential
    swing instead of by E0.
    """
    traj = rk4_integrate(
        lambda r, t: rhs_double_reference(r, t, damping, damping),
        [np.deg2rad(THETA1_DEG), 0.0, 0.0, 0.0],
    )
    th1, om1, th2, om2 = traj[:, 0], traj[:, 1], traj[:, 2], traj[:, 3]
    ke = 0.5 * M1 * (L1 * om1) ** 2 + 0.5 * M2 * (
        (L1 * om1) ** 2 + (L2 * om2) ** 2 + 2 * L1 * L2 * om1 * om2 * np.cos(th1 - th2)
    )
    pe = -(M1 + M2) * G * L1 * np.cos(th1) - M2 * G * L2 * np.cos(th2)
    e = ke + pe
    scale = float(np.ptp(pe))
    return float(e[0]), float(np.max(np.abs(e - e[0])) / scale)


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    """One (n_links, friction) dataset in the paper's time-step layout.

    theta_deg has shape (n_ics, 2000, n_links) and holds angles in DEGREES,
    matching what the reference notebooks save to .npy.
    """

    n_links: int
    friction: bool
    ic_deg: np.ndarray  # (n_ics, n_links) initial angles in degrees
    theta_deg: np.ndarray  # (n_ics, n_steps, n_links)
    t: np.ndarray  # (n_steps,)

    @property
    def label(self):
        name = {2: "double", 3: "triple"}[self.n_links]
        return f"{name}_{'friction' if self.friction else 'frictionless'}"


def _initial_conditions(n_links, theta2_grid_deg):
    """Initial-angle rows for the sweep.

    Double:  [120, x] for x in the grid.
    Triple:  [120, 0, x] -- the paper varies the *third* angle
             ("the initial angles for the new approach started at
             [120, 0, 0.1] and we incremented the third angle by 0.1
             until we reached 3.0", Fig. 18B caption).
    """
    grid = np.asarray(theta2_grid_deg, dtype=float).reshape(-1, 1)
    lead = np.full((grid.size, 1), THETA1_DEG)
    if n_links == 2:
        return np.hstack([lead, grid])
    if n_links == 3:
        return np.hstack([lead, np.zeros_like(grid), grid])
    raise ValueError(f"unsupported n_links={n_links}")


def generate(n_links, friction, theta_grid_deg=None):
    """Integrate every initial condition in the sweep."""
    if theta_grid_deg is None:
        theta_grid_deg = TRAIN_THETA2_DEG
    ic_deg = _initial_conditions(n_links, theta_grid_deg)
    damping = DAMPING if friction else 0.0

    if n_links == 2:
        rhs = lambda r, t: rhs_double_reference(r, t, damping, damping)  # noqa: E731
    else:
        rhs = make_rhs_n(n_links, damping)

    theta = np.empty((ic_deg.shape[0], N_STEPS, n_links), dtype=float)
    for k, row in enumerate(ic_deg):
        state0 = np.zeros(2 * n_links)
        state0[0::2] = np.deg2rad(row)
        traj = rk4_integrate(rhs, state0)
        theta[k] = np.rad2deg(traj[:, 0::2])

    return Dataset(n_links, friction, ic_deg, theta, time_points())


def save(ds, out_dir=DATA_DIR):
    """Write one dataset as .npz plus a flat tidy .csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{ds.label}.npz"
    np.savez_compressed(
        npz, ic_deg=ds.ic_deg, theta_deg=ds.theta_deg, t=ds.t, n_links=ds.n_links,
        friction=ds.friction,
    )

    n_ics, n_steps, n_links = ds.theta_deg.shape
    cols = {"t": np.tile(ds.t, n_ics)}
    for j in range(n_links):
        cols[f"theta{j + 1}_init_deg"] = np.repeat(ds.ic_deg[:, j], n_steps)
    for j in range(n_links):
        cols[f"theta{j + 1}_deg"] = ds.theta_deg[:, :, j].reshape(-1)
    header = ",".join(cols)
    table = np.column_stack(list(cols.values()))
    csv = out_dir / f"{ds.label}.csv"
    np.savetxt(csv, table, delimiter=",", header=header, comments="", fmt="%.10g")
    return npz, csv


def main():
    print("Validating the symbolic n=2 model against the paper's closed form ...")
    worst = validate_against_reference()
    print(f"  max |d(state)/dt| disagreement over 200 random states: {worst:.3e}")

    e0, drift = energy_drift_double(damping=0.0)
    print(f"  undamped 10 s run: E0 = {e0:.6f} J, max relative energy drift = {drift:.3e}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for n_links in (2, 3):
        for friction in (False, True):
            train = generate(n_links, friction)
            npz, csv = save(train)
            test = generate(n_links, friction, [TEST_THETA2_DEG])
            test_ds = Dataset(n_links, friction, test.ic_deg, test.theta_deg, test.t)
            tnpz = DATA_DIR / f"{test_ds.label}_holdout.npz"
            np.savez_compressed(
                tnpz, ic_deg=test_ds.ic_deg, theta_deg=test_ds.theta_deg, t=test_ds.t,
                n_links=n_links, friction=friction,
            )
            print(
                f"  {train.label}: train {train.theta_deg.shape} -> {npz.name}, {csv.name}; "
                f"holdout {test_ds.theta_deg.shape} -> {tnpz.name}"
            )


if __name__ == "__main__":
    main()
