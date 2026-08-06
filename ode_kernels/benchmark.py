"""Speed + accuracy comparison: ode_kernels vs. scipy.integrate.solve_ivp.

    python ode_kernels/benchmark.py

Three problems, chosen to stress different things:

  * ``decay20``   -- a 20-dimensional linear decay (cheap RHS, many steps):
                     isolates per-step Cython overhead from RHS cost.
  * ``van_der_pol`` -- classic mildly-nonlinear 2D oscillator at increasing
                     stiffness (mu), the standard non-stiff-solver stress test.
  * ``pleiades``   -- the 7-body planar gravitational problem (28 ODEs, a
                     standard non-stiff high-accuracy benchmark from Hairer &
                     Wanner), run at tight tolerance where high-order methods
                     should earn their keep.

For each, every ode_kernels method is timed against the closest scipy
equivalent at matched tolerances (same rtol/atol), reporting wall time,
function-eval count, and accuracy against a DOP853-at-1e-13 reference
trajectory. Writes a table to stdout and a bar chart to
``figures/benchmark.png``.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402

from ode_kernels import ode12, ode23, ode45, ode56, ode67, ode78  # noqa: E402

FIG_DIR = Path(__file__).parent / "figures"
METHODS = {
    "ode12": ode12, "ode23": ode23, "ode45": ode45,
    "ode56": ode56, "ode67": ode67, "ode78": ode78,
}
SCIPY_EQUIV = {"ode23": "RK23", "ode45": "RK45"}


def _decay20(t, y):
    return -np.arange(1, 21) * y


def _van_der_pol(t, y, mu=5.0):
    return [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0]]


def _pleiades(t, y):
    # 7 bodies, planar, masses = body index (Hairer, Norsett & Wanner test).
    n = 7
    x, yy, vx, vy = y[:n], y[n:2 * n], y[2 * n:3 * n], y[3 * n:4 * n]
    ax = np.zeros(n)
    ay = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dx = x[j] - x[i]
            dy = yy[j] - yy[i]
            r3 = (dx * dx + dy * dy) ** 1.5
            mj = j + 1
            ax[i] += mj * dx / r3
            ay[i] += mj * dy / r3
    return np.concatenate([vx, vy, ax, ay])


def _pleiades_ic():
    x0 = np.array([3.0, 3.0, -1.0, -3.0, 2.0, -2.0, 2.0])
    y0 = np.array([3.0, -3.0, 2.0, 0.0, 0.0, -4.0, 4.0])
    vx0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.75, -1.5])
    vy0 = np.array([0.0, 0.0, 0.0, -1.25, 1.0, 0.0, 0.0])
    return np.concatenate([x0, y0, vx0, vy0])


PROBLEMS = {
    "decay20": dict(f=_decay20, t_span=(0.0, 5.0), y0=np.ones(20), rtol=1e-8, atol=1e-11),
    "van_der_pol": dict(f=_van_der_pol, t_span=(0.0, 20.0), y0=[2.0, 0.0], rtol=1e-8, atol=1e-11),
    # Looser than the other two: the Pleiades problem has close gravitational
    # encounters, and ode12 (order 2) is excluded below because it needs an
    # impractically large step count to hold even this tolerance through
    # them -- exactly the regime a 2nd-order method isn't for.
    "pleiades": dict(f=_pleiades, t_span=(0.0, 2.0), y0=_pleiades_ic(), rtol=1e-8, atol=1e-10),
}
PROBLEM_SKIP = {"pleiades": {"ode12"}}


def _time_call(fn, repeats=3):
    best = np.inf
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        r = fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best, result = dt, r
    return best, result


def run():
    rows = []
    for prob_name, spec in PROBLEMS.items():
        f, t_span, y0, rtol, atol = (spec["f"], spec["t_span"], spec["y0"],
                                      spec["rtol"], spec["atol"])
        ref = solve_ivp(f, t_span, y0, method="DOP853", rtol=1e-13, atol=1e-13,
                         dense_output=True)
        y_ref_end = ref.sol(t_span[1])

        print(f"\n=== {prob_name} (n={np.size(y0)}, rtol={rtol:.0e}) ===")
        hdr = f"{'method':10s} {'time_ms':>10s} {'nfev':>8s} {'end_err':>12s}"
        print(hdr)

        skip = PROBLEM_SKIP.get(prob_name, set())
        for name, solver in METHODS.items():
            if name in skip:
                print(f"{name:10s} {'skipped':>10s}")
                continue
            ms, res = _time_call(lambda: solver(f, t_span, y0, rtol=rtol, atol=atol))
            err = np.max(np.abs(res.y[:, -1] - y_ref_end))
            print(f"{name:10s} {ms * 1e3:10.3f} {res.nfev:8d} {err:12.3e}")
            rows.append((prob_name, name, ms * 1e3, res.nfev, err))

        for ode_name, scipy_name in SCIPY_EQUIV.items():
            ms, res = _time_call(
                lambda: solve_ivp(f, t_span, y0, method=scipy_name, rtol=rtol, atol=atol)
            )
            err = np.max(np.abs(res.y[:, -1] - y_ref_end))
            label = f"scipy.{scipy_name}"
            print(f"{label:10s} {ms * 1e3:10.3f} {res.nfev:8d} {err:12.3e}")
            rows.append((prob_name, label, ms * 1e3, res.nfev, err))

    _plot(rows)
    return rows


def _plot(rows):
    problems = list(PROBLEMS)
    fig, axes = plt.subplots(1, len(problems), figsize=(6 * len(problems), 5))
    if len(problems) == 1:
        axes = [axes]

    for ax, prob_name in zip(axes, problems):
        names = [r[1] for r in rows if r[0] == prob_name]
        times = [r[2] for r in rows if r[0] == prob_name]
        colors = ["tab:blue" if not n.startswith("scipy") else "tab:orange" for n in names]
        ax.bar(names, times, color=colors)
        ax.set_ylabel("wall time (ms)")
        ax.set_title(prob_name)
        ax.tick_params(axis="x", rotation=45)
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    p = FIG_DIR / "benchmark.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    run()
