"""
Chaotic-regime animations for the n=3 (triple) and n=5 (quintuple) pendulum,
built on the symbolic n-link model in n_pendulum_symbolic.py.

Each run: integrate at a fine dt, confirm energy conservation (same
diagnostic used throughout this project to catch modeling bugs), then
render a dark-themed GIF of the whole chain swinging, with a fading trail
on the last bob.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import odeint

from n_pendulum_symbolic import make_state_space

OUT_DIR = Path(__file__).parent
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


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


def simulate(n, theta0_deg, m_vals=None, l_vals=None, g=9.81, dt=0.005, duration=15.0):
    m_vals = m_vals or tuple([1.0] * n)
    l_vals = l_vals or tuple([1.0] * n)

    t0 = time.perf_counter()
    rhs, _model = make_state_space(n, m_vals, l_vals, g)
    print(f"  n={n}: symbolic model built in {time.perf_counter() - t0:.2f}s")

    state0 = np.zeros(2 * n)
    state0[0::2] = np.array(theta0_deg) * np.pi / 180.0

    tspan = np.arange(0, duration, dt)
    t1 = time.perf_counter()
    sol = odeint(rhs, state0, tspan)
    print(f"  n={n}: integrated {len(tspan)} steps in {time.perf_counter() - t1:.2f}s")

    theta, omega = sol[:, 0::2], sol[:, 1::2]
    E = chain_energy(theta, omega, m_vals, l_vals, g)
    drift = np.max(np.abs(E - E[0]))
    pct = 100 * drift / abs(E[0]) if E[0] != 0 else float("nan")
    print(
        f"  n={n}: energy drift {drift:.3e} J ({pct:.5f}% of |E0|) -- "
        f"{'OK' if pct < 1.0 else 'SUSPICIOUSLY LARGE'}"
    )

    return dict(
        n=n,
        t=tspan,
        theta=theta,
        omega=omega,
        m_vals=m_vals,
        l_vals=l_vals,
        E=E,
        theta0_deg=theta0_deg,
    )


def render_gif(
    sim, out_path, max_frames=320, fps=28, trail_len=60, figsize=(6, 6), dpi=110
):
    n = sim["n"]
    theta, l_vals = sim["theta"], sim["l_vals"]
    x, y = chain_xy(theta, l_vals)
    total_l = sum(l_vals)

    n_samples = len(sim["t"])
    step = max(1, n_samples // max_frames)
    frame_idx = np.arange(0, n_samples, step)
    n_frames = len(frame_idx)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")
    ax.set_xlim(-total_l * 1.15, total_l * 1.15)
    ax.set_ylim(-total_l * 1.15, total_l * 0.35)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, color="#888")
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.set_title(
        f"{n}-Link Pendulum  "
        + r"($\theta_0$="
        + f'{sim["theta0_deg"]}'
        + r"$\degree$)",
        color="white",
        fontsize=12,
        fontweight="bold",
    )
    time_text = ax.text(
        0.02, 0.02, "", transform=ax.transAxes, color="#aaaaaa", fontsize=9
    )

    (trail,) = ax.plot([], [], "-", color="#00d4ff", linewidth=1.1, alpha=0.55)
    (chain_line,) = ax.plot(
        [], [], "o-", color="#e0e0e0", lw=2.2, ms=5, markerfacecolor="white"
    )
    (last_bob,) = ax.plot([], [], "o", color="#ff6b6b", ms=8)
    (pivot,) = ax.plot([0], [0], "s", color="#ffffff", ms=6)

    def init():
        trail.set_data([], [])
        chain_line.set_data([], [])
        last_bob.set_data([], [])
        time_text.set_text("")
        return trail, chain_line, last_bob, pivot, time_text

    def update(frame):
        i = frame_idx[frame]
        t_start = max(0, i - trail_len * step)
        trail.set_data(
            x[t_start : i + 1 : max(1, step // 2 + 1), -1],
            y[t_start : i + 1 : max(1, step // 2 + 1), -1],
        )
        xs = np.concatenate([[0.0], x[i]])
        ys = np.concatenate([[0.0], y[i]])
        chain_line.set_data(xs, ys)
        last_bob.set_data([x[i, -1]], [y[i, -1]])
        time_text.set_text(f't = {sim["t"][i]:.2f} s')
        return trail, chain_line, last_bob, pivot, time_text

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, init_func=init, blit=True
    )
    writer = animation.PillowWriter(fps=fps)
    ani.save(str(out_path), writer=writer)
    plt.close(fig)
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved {out_path.name}: {n_frames} frames, {size_mb:.2f} MB")


if __name__ == "__main__":
    print("Simulating n=3 (triple pendulum), chaotic regime...")
    sim3 = simulate(3, theta0_deg=[150, 90, 30], duration=15.0, dt=0.005)
    print("Rendering n=3 GIF...")
    render_gif(sim3, OUT_DIR / "triple_pendulum.gif")

    print("\nSimulating n=5 (quintuple pendulum), chaotic regime...")
    sim5 = simulate(5, theta0_deg=[150, 120, 90, 60, 30], duration=15.0, dt=0.005)
    print("Rendering n=5 GIF...")
    render_gif(sim5, OUT_DIR / "quintuple_pendulum.gif")

    print("\nDone.")
