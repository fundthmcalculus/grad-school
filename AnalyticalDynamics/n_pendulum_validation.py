"""
Validation suite for the general n-link pendulum built in n_pendulum_symbolic.py.

Three checks, in increasing order of how much they trust each other:
  1. n=1 reduces algebraically to the textbook simple pendulum.
  2. n=2 agrees with the independently hand-derived, bug-fixed
     DoublePendulum.equations_of_motion to machine precision, over random
     states -- not just at one nice initial condition.
  3. n=3 and n=5 conserve energy under free (undamped) evolution, the same
     check that caught the alpha2 bug in the n=2 case.

This is a validation pass, not a chaos/results study of the triple or
quintuple pendulum -- that's queued as future work once explicitly wanted.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

from test_fuzzy_ode import DoublePendulum
from n_pendulum_symbolic import build_n_pendulum, make_state_space

FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def check_n1_reduces_to_simple_pendulum():
    model = build_n_pendulum(1)
    # M = m1 l1^2, f = -g m1 l1 sin(theta1)  =>  thetaddot = -(g/l1) sin(theta1)
    import sympy as sp

    m1, l1, g = model.m[0], model.l[0], model.g
    thetaddot = sp.simplify((model.f[0] / model.M[0, 0]))
    expected = -g / l1 * sp.sin(model.theta[0])
    ok = sp.simplify(thetaddot - expected) == 0
    print(f"n=1 reduces to simple pendulum: {ok}  ({thetaddot} == {expected})")
    return ok


def check_n2_matches_hand_derivation(n_trials=200, seed=42):
    pend = DoublePendulum(m1=1.3, m2=0.7, l1=1.1, l2=0.9, g=9.81)
    rhs2, _ = make_state_space(2, (pend.m1, pend.m2), (pend.l1, pend.l2), pend.g)

    rng = np.random.default_rng(seed)
    max_err = 0.0
    for _ in range(n_trials):
        state = rng.uniform(-3, 3, size=4)
        a_hand = np.array(pend.equations_of_motion(state, 0.0))
        a_sym = rhs2(state, 0.0)
        max_err = max(max_err, np.max(np.abs(a_hand - a_sym)))
    print(
        f"n=2 vs hand-derived DoublePendulum: max abs error over {n_trials} random "
        f"states = {max_err:.3e}"
    )
    return max_err


def chain_energy(theta, omega, m_vals, l_vals, g):
    n = theta.shape[1]
    xc = np.zeros(theta.shape[0])
    yc = np.zeros(theta.shape[0])
    xdc = np.zeros(theta.shape[0])
    ydc = np.zeros(theta.shape[0])
    E = np.zeros(theta.shape[0])
    for i in range(n):
        xc = xc + l_vals[i] * np.sin(theta[:, i])
        yc = yc - l_vals[i] * np.cos(theta[:, i])
        xdc = xdc + l_vals[i] * omega[:, i] * np.cos(theta[:, i])
        ydc = ydc + l_vals[i] * omega[:, i] * np.sin(theta[:, i])
        E += 0.5 * m_vals[i] * (xdc**2 + ydc**2) + m_vals[i] * g * yc
    return E


def chain_xy(theta, l_vals):
    n = theta.shape[1]
    x = np.zeros_like(theta)
    y = np.zeros_like(theta)
    xc = np.zeros(theta.shape[0])
    yc = np.zeros(theta.shape[0])
    for i in range(n):
        xc = xc + l_vals[i] * np.sin(theta[:, i])
        yc = yc - l_vals[i] * np.cos(theta[:, i])
        x[:, i] = xc
        y[:, i] = yc
    return x, y


def validate_n(n, theta0_deg, dt=0.001, duration=8.0, g=9.81):
    m_vals = tuple([1.0] * n)
    l_vals = tuple([1.0] * n)

    t0 = time.perf_counter()
    rhs, _model = make_state_space(n, m_vals, l_vals, g)
    build_s = time.perf_counter() - t0

    state0 = np.zeros(2 * n)
    state0[0::2] = np.array(theta0_deg) * np.pi / 180.0

    t1 = time.perf_counter()
    tspan = np.arange(0, duration, dt)
    sol = odeint(rhs, state0, tspan)
    integrate_s = time.perf_counter() - t1

    theta = sol[:, 0::2]
    omega = sol[:, 1::2]
    E = chain_energy(theta, omega, m_vals, l_vals, g)
    drift = np.max(np.abs(E - E[0]))
    pct = 100 * drift / abs(E[0]) if E[0] != 0 else float("nan")

    print(
        f"n={n}: build {build_s:.2f}s, integrate {integrate_s:.2f}s ({len(tspan)} steps), "
        f"energy drift {drift:.3e} J ({pct:.5f}% of |E0|)"
    )

    return dict(
        n=n,
        build_s=build_s,
        integrate_s=integrate_s,
        drift=drift,
        pct=pct,
        theta=theta,
        omega=omega,
        t=tspan,
        m_vals=m_vals,
        l_vals=l_vals,
    )


def plot_build_time_scaling(results):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ns = [r["n"] for r in results]
    build_times = [r["build_s"] for r in results]
    ax.semilogy(ns, build_times, "o-", color="#1f77b4")
    ax.set_xlabel("Chain length $n$")
    ax.set_ylabel("Symbolic build + lambdify time (s, log scale)")
    ax.set_title("Cost of Deriving the $n$-Link Pendulum Symbolically")
    ax.set_xticks(ns)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "n_pendulum_build_scaling.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_chain_snapshots(results_by_n):
    fig, axes = plt.subplots(1, len(results_by_n), figsize=(6.5 * len(results_by_n), 6))
    if len(results_by_n) == 1:
        axes = [axes]
    for ax, r in zip(axes, results_by_n):
        n = r["n"]
        x, y = chain_xy(r["theta"], r["l_vals"])
        n_frames = len(r["t"])
        step = max(1, n_frames // 400)
        for i in range(0, n_frames - step, step):
            alpha = 0.15 + 0.7 * i / n_frames
            ax.plot(x[i, -1:], y[i, -1:], ".", color="#00d4ff", alpha=alpha, ms=3)
        # Final configuration of the whole chain
        xf = np.concatenate([[0.0], x[-1]])
        yf = np.concatenate([[0.0], y[-1]])
        ax.plot(xf, yf, "o-", color="#333", lw=2, ms=6)
        ax.plot(0, 0, "ks", ms=8)
        total_l = sum(r["l_vals"])
        ax.set_xlim(-total_l * 1.1, total_l * 1.1)
        ax.set_ylim(-total_l * 1.1, total_l * 0.3)
        ax.set_aspect("equal")
        ax.set_title(
            f'n={n}: last-bob trace over {r["t"][-1]:.0f}s\n'
            f"(final configuration in black)",
            fontsize=11,
        )
        ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(
        FIG_DIR / "n_pendulum_chain_snapshots.png", dpi=200, bbox_inches="tight"
    )
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 70)
    print("VALIDATION 1: n=1 algebraic reduction to the simple pendulum")
    print("=" * 70)
    check_n1_reduces_to_simple_pendulum()

    print("\n" + "=" * 70)
    print("VALIDATION 2: n=2 vs. independently hand-derived, bug-fixed code")
    print("=" * 70)
    check_n2_matches_hand_derivation()

    print("\n" + "=" * 70)
    print("VALIDATION 3: energy conservation for n=1..5")
    print("=" * 70)
    results = []
    results.append(validate_n(1, [120]))
    results.append(validate_n(2, [120, 60]))
    results.append(validate_n(3, [120, 60, 10]))
    results.append(validate_n(4, [120, 60, 10, -20]))
    results.append(validate_n(5, [120, 60, 10, -20, 5]))

    print("\nGenerating figures...")
    plot_build_time_scaling(results)
    plot_chain_snapshots([results[2], results[4]])  # n=3, n=5
    print("Done.")
