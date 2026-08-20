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

from n_pendulum_animation import chain_energy  # noqa: E402
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

#: Single source of truth for dataset naming, imported by every other module here.
#: The paper covers n=2 and n=3 only; n=5 is this repository's extension, which the
#: symbolic Lagrangian model in n_pendulum_symbolic.py supports without changes.
SYSTEM_NAMES = {1: "single", 2: "double", 3: "triple", 4: "quadruple", 5: "quintuple"}


def system_name(n_links):
    """'double', 'triple', 'quintuple', ... for use in labels and filenames."""
    try:
        return SYSTEM_NAMES[n_links]
    except KeyError:
        raise ValueError(
            f"no name for n_links={n_links}; add it to SYSTEM_NAMES"
        ) from None


def dataset_label(n_links, friction):
    return f"{system_name(n_links)}_{'friction' if friction else 'frictionless'}"


#: theta_1(0) = 120 deg; the last link's angle is swept over this grid (degrees).
TRAIN_THETA2_DEG = np.round(np.arange(0.0, 3.0 + 1e-9, 0.1), 2)
#: The paper's held-out "in-between" initial condition.
TEST_THETA2_DEG = 2.05
THETA1_DEG = 120.0

DATA_DIR = Path(__file__).resolve().parent / "data"


#: The held-out trajectory is integrated twice as far as the training window so the
#: second half tests extrapolation in *time*, not just to an unseen initial angle.
#: Training is unchanged at 10 s; only the holdout is longer.
TEST_T_END = 20.0
TEST_N_STEPS = int(round((TEST_T_END - T_START) / H))  # 4000


def time_points(t_end=T_END) -> np.ndarray:
    """Sample times on the paper's grid: ``np.arange(0, t_end, 0.005)``."""
    return np.arange(T_START, t_end, H)


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


def energy_drift(n_links, use_reference_rhs=None):
    """Relative energy drift of the undamped [120, 0, ..., 0] run over 10 s.

    A sanity gate on the integrator, run for every n. Returns
    (E0, max |dE| / potential swing). Energy uses the pivot as the potential zero,
    so E passes through zero for some initial conditions; normalising by the
    potential swing rather than by E0 keeps the ratio meaningful when it does.

    Energy comes from `n_pendulum_animation.chain_energy`, the same function the
    existing n=3 and n=5 validation in n_pendulum_validation.py uses -- so a drift
    figure here is comparable to the ones already recorded for this repository.

    use_reference_rhs selects which derivation to integrate; it defaults to the
    paper's own closed form at n=2 and the symbolic model elsewhere. Passing it
    explicitly lets the two be compared at n=2, where both exist.
    """
    if use_reference_rhs is None:
        use_reference_rhs = n_links == 2
    if use_reference_rhs:
        if n_links != 2:
            raise ValueError("the paper's closed-form RHS exists only for n=2")
        rhs = lambda r, t: rhs_double_reference(r, t, 0.0, 0.0)  # noqa: E731
    else:
        rhs = make_rhs_n(n_links, damping=0.0)

    state0 = np.zeros(2 * n_links)
    state0[0] = np.deg2rad(THETA1_DEG)
    traj = rk4_integrate(rhs, state0)

    theta, omega = traj[:, 0::2], traj[:, 1::2]
    ones = tuple([1.0] * n_links)
    e = chain_energy(theta, omega, ones, ones, G)
    # Potential-only swing, as the normaliser: recompute with omega zeroed.
    pe = chain_energy(theta, np.zeros_like(omega), ones, ones, G)
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
        return dataset_label(self.n_links, self.friction)


def _initial_conditions(n_links, theta_grid_deg):
    """Initial-angle rows for the sweep: [120, 0, ..., 0, x] for x in the grid.

    This is the paper's own pattern, read off its two cases and continued:

      Double: [120, x]        -- theta_1 fixed at 120, sweep theta_2.
      Triple: [120, 0, x]     -- "the initial angles for the new approach started
                                 at [120, 0, 0.1], and we incremented the third
                                 angle by 0.1 until we reached 3.0" (Fig. 18B
                                 caption).

    So the swept angle is always the *last* link and every intermediate link starts
    at rest hanging straight down. For n=5 that gives [120, 0, 0, 0, x], which is
    an extrapolation of the paper's convention -- the paper has no n=5 experiment.
    """
    grid = np.asarray(theta_grid_deg, dtype=float).reshape(-1, 1)
    if n_links < 2:
        raise ValueError(f"n_links must be >= 2, got {n_links}")
    lead = np.full((grid.size, 1), THETA1_DEG)
    middle = np.zeros((grid.size, n_links - 2))
    return np.hstack([lead, middle, grid])


def generate(n_links, friction, theta_grid_deg=None, n_steps=N_STEPS):
    """Integrate every initial condition in the sweep.

    n_steps defaults to the paper's 2000 (10 s). The held-out trajectory is
    generated with TEST_N_STEPS (20 s) so its second half lies beyond anything the
    model was trained on.
    """
    if theta_grid_deg is None:
        theta_grid_deg = TRAIN_THETA2_DEG
    ic_deg = _initial_conditions(n_links, theta_grid_deg)
    damping = DAMPING if friction else 0.0

    if n_links == 2:
        rhs = lambda r, t: rhs_double_reference(r, t, damping, damping)  # noqa: E731
    else:
        rhs = make_rhs_n(n_links, damping)

    theta = np.empty((ic_deg.shape[0], n_steps, n_links), dtype=float)
    for k, row in enumerate(ic_deg):
        state0 = np.zeros(2 * n_links)
        state0[0::2] = np.deg2rad(row)
        traj = rk4_integrate(rhs, state0, n_steps=n_steps)
        theta[k] = np.rad2deg(traj[:, 0::2])

    return Dataset(n_links, friction, ic_deg, theta, time_points(T_START + n_steps * H))


def save(ds, out_dir=DATA_DIR):
    """Write one dataset as .npz plus a flat tidy .csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{ds.label}.npz"
    np.savez_compressed(
        npz,
        ic_deg=ds.ic_deg,
        theta_deg=ds.theta_deg,
        t=ds.t,
        n_links=ds.n_links,
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


def rk4_order_check(n_links, refinements=2):
    """Confirm energy drift falls as h^4, so drift is discretization not derivation.

    At the paper's h = 0.005 the n=5 chain drifts ~5e-5 of its potential swing --
    two orders worse than n=2, which invites the question of whether the symbolic
    n=5 equations are wrong. They are not: halving h cuts the drift by ~16x, which
    is RK4's order. A derivation error would not obey the integrator's convergence
    rate. Returns [(h, drift), ...] finest-last.
    """
    out = []
    for k in range(refinements + 1):
        div = 2**k
        rhs = make_rhs_n(n_links, damping=0.0)
        state0 = np.zeros(2 * n_links)
        state0[0] = np.deg2rad(THETA1_DEG)
        traj = rk4_integrate(rhs, state0, n_steps=N_STEPS * div, h=H / div)
        theta, omega = traj[:, 0::2], traj[:, 1::2]
        ones = tuple([1.0] * n_links)
        e = chain_energy(theta, omega, ones, ones, G)
        pe = chain_energy(theta, np.zeros_like(omega), ones, ones, G)
        out.append((H / div, float(np.max(np.abs(e - e[0])) / np.ptp(pe))))
    return out


def integrate_dop853(rhs, state0, t_eval, rtol=1e-12, atol=1e-14):
    """High-accuracy reference integration with scipy's 8th-order Dormand-Prince.

    The paper specifies fixed-step RK4 at h = 0.005 and the reproduction datasets
    keep it -- changing the training integrator would be changing the experiment.
    This exists for the other job: an *independent* check on whether that choice is
    accurate enough, and a reference to score long rollouts against.

    Independence is the point. `reference_convergence` refines the RK4 step and
    checks self-agreement, which catches step-size error but shares any structural
    bias with itself. DOP853 is a different order, different coefficients, and
    adaptive, so agreement between the two is evidence rather than consistency.

    `rhs` takes (state, t) in this module's odeint-style convention; solve_ivp wants
    (t, state), so it is flipped here rather than at every call site.
    """
    from scipy.integrate import solve_ivp

    sol = solve_ivp(
        lambda t, y: rhs(y, t),
        (float(t_eval[0]), float(t_eval[-1])),
        np.asarray(state0, dtype=float),
        method="DOP853",
        t_eval=np.asarray(t_eval, dtype=float),
        rtol=rtol,
        atol=atol,
        dense_output=False,
    )
    if not sol.success:
        raise RuntimeError(f"DOP853 failed: {sol.message}")
    return sol.y.T


def rhs_for(n_links, friction):
    """The right-hand side used for a given dataset, paper's form at n=2."""
    damping = DAMPING if friction else 0.0
    if n_links == 2:
        return lambda r, t: rhs_double_reference(r, t, damping, damping)
    return make_rhs_n(n_links, damping)


def cross_check_integrators(
    n_links, friction, duration=TEST_T_END, rtol=1e-12, atol=1e-14
):
    """Max angular disagreement (deg) between RK4 at h=0.005 and DOP853.

    Returns (max_delta_deg, t_first_exceeds_10deg). For a converged reference this
    is small and the threshold is never crossed; for a chaotic one it is large and
    tells you where the paper's step size stops resolving the trajectory.
    """
    rhs = rhs_for(n_links, friction)
    state0 = np.zeros(2 * n_links)
    state0[0] = np.deg2rad(THETA1_DEG)
    state0[-2] = np.deg2rad(TEST_THETA2_DEG)

    n_steps = int(round(duration / H))
    t = time_points(duration)
    a = np.rad2deg(rk4_integrate(rhs, state0, n_steps=n_steps)[:, 0::2])
    b = np.rad2deg(integrate_dop853(rhs, state0, t, rtol=rtol, atol=atol)[:, 0::2])
    delta = np.abs(a - b)
    over = np.flatnonzero(np.max(delta, axis=1) > 10.0)
    return float(np.max(delta)), (float(t[over[0]]) if over.size else float("inf"))


def reference_convergence(
    n_links, friction, duration=TEST_T_END, refinements=3, threshold_deg=10.0
):
    """How long the reference trajectory itself is trustworthy.

    Energy drift says the integrator is self-consistent; it says nothing about
    whether the *trajectory* is converged. On a chaotic system those differ
    enormously, because any step-size error is amplified at the Lyapunov rate. So
    integrate the held-out initial condition at h, h/2, h/4, ... and report where
    successive refinements stop agreeing. Beyond that time the "ground truth" is a
    property of the step size, not of the pendulum, and no surrogate should be
    scored against it.

    Measured on the paper's h = 0.005 over 20 s: the friction chains agree to 0.00
    degrees under 8x refinement -- fully converged, nothing to fix. The frictionless
    ones disagree by hundreds of degrees and part company from about t = 11.5 s,
    which is well inside the 10-20 s extrapolation window. Returns
    [(h, max_abs_delta_deg, t_exceeds_threshold), ...]; the first row is the
    coarsest and has no predecessor to compare against.
    """
    damping = DAMPING if friction else 0.0
    if n_links == 2:
        rhs = lambda r, t: rhs_double_reference(r, t, damping, damping)  # noqa: E731
    else:
        rhs = make_rhs_n(n_links, damping)

    state0 = np.zeros(2 * n_links)
    state0[0] = np.deg2rad(THETA1_DEG)
    state0[-2] = np.deg2rad(TEST_THETA2_DEG)

    base_steps = int(round(duration / H))
    out, prev = [], None
    for k in range(refinements + 1):
        div = 2**k
        traj = rk4_integrate(rhs, state0, n_steps=base_steps * div, h=H / div)
        theta = np.rad2deg(traj[::div, 0::2])
        if prev is None:
            out.append((H, None, None))
        else:
            delta = np.abs(theta - prev)
            over = np.flatnonzero(np.max(delta, axis=1) > threshold_deg)
            out.append(
                (
                    H / div,
                    float(np.max(delta)),
                    float(over[0] * H) if over.size else float("inf"),
                )
            )
        prev = theta
    return out


#: Which chains to build. n=2 and n=3 are the paper's; n=5 extends it.
N_LINKS = (2, 3, 5)


def collect_provenance(n_trials=200, seed=42, tol=1e-9):
    """The correctness checks `main()` prints, as a dict for the pipeline's log.

    Raises AssertionError under the same conditions main() would: symbolic/
    reference disagreement above `tol`, or RK4 convergence slower than order 3
    (ratio <= 8x per halving, versus the ~16x order-4 expects).
    """
    worst = validate_against_reference(n_trials=n_trials, seed=seed, tol=tol)

    energy = {}
    e0, drift = energy_drift(2, use_reference_rhs=True)
    energy["double_closed_form"] = {"E0": e0, "drift_ratio": drift}
    for n_links in N_LINKS:
        e0, drift = energy_drift(n_links, use_reference_rhs=False)
        energy[f"n{n_links}_symbolic"] = {"E0": e0, "drift_ratio": drift}

    worst_n = max(N_LINKS)
    steps = rk4_order_check(worst_n)
    ratios = [a[1] / b[1] for a, b in zip(steps, steps[1:])]
    assert min(ratios) > 8.0, (
        f"n={worst_n} energy drift is not converging at RK4's order "
        f"(ratios {ratios}); suspect the derivation, not the step size"
    )
    return {
        "symbolic_vs_reference_max_error": worst,
        "energy_drift": energy,
        "rk4_order_check_n": worst_n,
        "rk4_step_sizes": [h for h, _ in steps],
        "rk4_drifts": [d for _, d in steps],
        "rk4_order_ratios": ratios,
    }


def generate_all(n_links_list=N_LINKS, out_dir=DATA_DIR):
    """Build and save every (n_links, friction) train + holdout dataset.

    The holdout runs to TEST_T_END, double the training window, so its second
    half is extrapolation in time as well as to an unseen angle. Returns one
    summary dict per dataset (paths, shapes, the swept IC grid), for the
    pipeline's stage_data JSON log.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for n_links in n_links_list:
        for friction in (False, True):
            train = generate(n_links, friction)
            npz, csv = save(train, out_dir)
            test = generate(n_links, friction, [TEST_THETA2_DEG], n_steps=TEST_N_STEPS)
            tnpz = out_dir / f"{test.label}_holdout.npz"
            np.savez_compressed(
                tnpz,
                ic_deg=test.ic_deg,
                theta_deg=test.theta_deg,
                t=test.t,
                n_links=n_links,
                friction=friction,
                train_t_end=T_END,
            )
            datasets.append(
                {
                    "label": train.label,
                    "n_links": n_links,
                    "friction": friction,
                    "npz": str(npz),
                    "csv": str(csv),
                    "holdout_npz": str(tnpz),
                    "train_shape": list(train.theta_deg.shape),
                    "holdout_shape": list(test.theta_deg.shape),
                    "ic_grid_deg": train.ic_deg[:, -1].tolist(),
                }
            )
    return datasets


def main():
    print("Validating the symbolic n=2 model against the paper's closed form ...")
    prov = collect_provenance()
    print(
        f"  max |d(state)/dt| disagreement over 200 random states: "
        f"{prov['symbolic_vs_reference_max_error']:.3e}"
    )

    print("Energy conservation of the undamped [120, 0, ...] run over 10 s:")
    d = prov["energy_drift"]["double_closed_form"]
    print(
        f"  n=2 (paper's closed form): E0 = {d['E0']:+.6f} J, "
        f"max drift / PE swing = {d['drift_ratio']:.3e}"
    )
    for n_links in N_LINKS:
        d = prov["energy_drift"][f"n{n_links}_symbolic"]
        print(
            f"  n={n_links} (symbolic)          : E0 = {d['E0']:+.6f} J, "
            f"max drift / PE swing = {d['drift_ratio']:.3e}"
        )

    print(
        f"  n={prov['rk4_order_check_n']} h-refinement drift: "
        + " -> ".join(f"{d:.2e}" for d in prov["rk4_drifts"])
        + "  (ratios "
        + ", ".join(f"{r:.1f}x" for r in prov["rk4_order_ratios"])
        + ", RK4 expects ~16x)"
    )

    datasets = generate_all()
    for ds in datasets:
        print(
            f"  {ds['label']}: train {tuple(ds['train_shape'])} -> "
            f"{Path(ds['npz']).name}, {Path(ds['csv']).name}; "
            f"holdout {tuple(ds['holdout_shape'])} -> {Path(ds['holdout_npz']).name}"
        )


if __name__ == "__main__":
    main()
