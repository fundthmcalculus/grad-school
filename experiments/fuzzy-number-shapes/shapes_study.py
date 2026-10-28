"""Membership-shape study: does model quality stop improving past the trapezoid?

Question
    A first-order TSK model on a 2-input grid of K=5 fuzzy numbers per input.
    Each input's fuzzy numbers share one shape family. Shapes compared:

      piecewise-linear n-gons (n = 3..7 knots):
          triangle (3), trapezoid (4), pentagon (5), hexagon (6), heptagon (7)
      smooth:
          gaussian (2 params), elliptic (2 params: centre, half-width)

    n-gon convention: n knots x_0 < ... < x_{n-1}; the membership at the knots
    is a symmetric ramp (0 -> 1 in equal steps; a single peak for odd n, a
    two-knot plateau for even n). Knots are free, so an n-gon has n parameters.

Fitting (the baseline, not the method under test)
    Consequents are solved by linear least squares inside the residual
    (variable projection). Antecedent parameters go to scipy least_squares
    with its default finite-difference Jacobian. Seeds 0..9 (common.SEEDS
    protocol): the seed sets the training sample and a small init jitter.

Pre-registered expectations (written before the run)
    E1  triangle -> trapezoid gives the largest RMSE drop of the n-gon family.
    E2  each further knot (pentagon, hexagon, heptagon) gives a smaller gain
        than the one before it, with gains within about one std of zero.
    E3  the smooth shapes (gaussian, elliptic) reach at least the heptagon's
        RMSE with fewer antecedent parameters.
    E4  manipulation check: the target is not representable by any shape at
        K=5 without error, so RMSE must move when the shape changes.

Run:  python shapes_study.py            (writes results.csv and results.md)
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
SEEDS = range(10)
K = 5  # fuzzy numbers per input
N_TRAIN = 500
NOISE_SD = 0.05
GRID = 40  # noise-free test grid per axis

SHAPES = {
    "triangle": ("pl", 3),
    "trapezoid": ("pl", 4),
    "pentagon": ("pl", 5),
    "hexagon": ("pl", 6),
    "heptagon": ("pl", 7),
    "gaussian": ("gauss", 2),
    "elliptic": ("ell", 2),
}


def target(x1, x2):
    return np.sin(2 * np.pi * x1) * (x2 + 0.5) + 0.8 * np.exp(
        -((x1 - 0.6) ** 2 + (x2 - 0.4) ** 2) / 0.02
    )


def pl_heights(n):
    """Membership at the n knots: symmetric ramp, odd n = peak, even n = plateau."""
    if n % 2:
        s = (n - 1) // 2
        left = [j / s for j in range(s)] + [1.0]
        return np.array(left + left[-2::-1])
    s = (n - 2) // 2
    left = [0.0] + [j / s for j in range(1, s + 1)]
    return np.array(left + left[::-1])


def mf_matrix(shape, theta, x):
    """Membership of the K fuzzy numbers of one input, shape (K, N)."""
    kind, n = SHAPES[shape]
    out = np.empty((K, x.size))
    for k in range(K):
        t = theta[k * n : (k + 1) * n]
        if kind == "pl":
            knots = t[0] + np.concatenate(([0.0], np.cumsum(np.exp(t[1:]))))
            out[k] = np.interp(x, knots, pl_heights(n), left=0.0, right=0.0)
        elif kind == "gauss":
            out[k] = np.exp(-0.5 * ((x - t[0]) / np.exp(t[1])) ** 2)
        else:  # elliptic
            z = (x - t[0]) / np.exp(t[1])
            out[k] = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return out


def init_theta(shape, rng):
    """Both inputs: centres on a unit grid, each fuzzy number spanning one grid spacing either side."""
    kind, n = SHAPES[shape]
    centres = np.linspace(0.0, 1.0, K)
    h = 1.0 / (K - 1)
    one = []
    for c in centres:
        if kind == "pl":
            spacing = np.full(n - 1, 2 * h / (n - 1))
            one += [c - h, *np.log(spacing)]
        else:
            one += [c, np.log(h)]
    theta = np.concatenate([one, one]).astype(float)
    return theta + rng.normal(0.0, 0.05, theta.size)


def design(shape, theta1, theta2, X):
    """Normalised TSK regressor matrix, shape (N, 3*K*K)."""
    m1 = mf_matrix(shape, theta1, X[:, 0])
    m2 = mf_matrix(shape, theta2, X[:, 1])
    fire = (m1[:, None, :] * m2[None, :, :]).reshape(K * K, -1)
    fire = fire / (fire.sum(0, keepdims=True) + 1e-12)
    ones = np.ones(X.shape[0])
    cols = [fire.T * ones[:, None], fire.T * X[:, :1], fire.T * X[:, 1:2]]
    return np.concatenate(cols, axis=1)


def fit(shape, X, y, theta0):
    p = SHAPES[shape][1] * K

    def solve(theta):
        Phi = design(shape, theta[:p], theta[p : 2 * p], X)
        beta = np.linalg.lstsq(Phi, y, rcond=None)[0]
        return Phi, beta

    def residual(theta):
        Phi, beta = solve(theta)
        return Phi @ beta - y

    t0 = time.perf_counter()
    res = least_squares(residual, theta0, method="trf", max_nfev=200)
    elapsed = time.perf_counter() - t0
    Phi, beta = solve(res.x)
    return res.x, beta, elapsed, res.nfev


def predict(shape, theta, beta, X):
    p = SHAPES[shape][1] * K
    return design(shape, theta[:p], theta[p : 2 * p], X) @ beta


def run():
    grid = np.linspace(0.0, 1.0, GRID)
    g1, g2 = np.meshgrid(grid, grid)
    Xte = np.column_stack([g1.ravel(), g2.ravel()])
    yte = target(Xte[:, 0], Xte[:, 1])

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.0, 1.0, (N_TRAIN, 2))
        y = target(X[:, 0], X[:, 1]) + rng.normal(0.0, NOISE_SD, N_TRAIN)
        for shape, (kind, n) in SHAPES.items():
            theta0 = init_theta(shape, np.random.default_rng(1000 + seed))
            theta, beta, secs, nfev = fit(shape, X, y, theta0)
            rmse_te = np.sqrt(np.mean((predict(shape, theta, beta, Xte) - yte) ** 2))
            rmse_tr = np.sqrt(np.mean((predict(shape, theta, beta, X) - y) ** 2))
            rows.append(
                dict(
                    shape=shape,
                    seed=seed,
                    antecedent_params=2 * K * n,
                    rmse_train=rmse_tr,
                    rmse_test=rmse_te,
                    nfev=nfev,
                    seconds=secs,
                )
            )
            print(f"seed {seed} {shape:10s} test RMSE {rmse_te:.4f}  nfev {nfev}  {secs:.2f}s")
    return rows


def write(rows):
    with open(HERE / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    order = list(SHAPES)
    lines = [
        "| shape | antecedent params | test RMSE (mean ± sd) | train RMSE | gain vs previous | nfev | seconds |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    prev = None
    for shape in order:
        r = [x for x in rows if x["shape"] == shape]
        te = np.array([x["rmse_test"] for x in r])
        tr = np.array([x["rmse_train"] for x in r])
        gain = "" if prev is None else f"{prev.mean() - te.mean():+.4f}"
        lines.append(
            f"| {shape} | {r[0]['antecedent_params']} | {te.mean():.4f} ± {te.std(ddof=1):.4f} "
            f"| {tr.mean():.4f} | {gain} "
            f"| {np.mean([x['nfev'] for x in r]):.0f} | {np.mean([x['seconds'] for x in r]):.2f} |"
        )
        prev = te
    text = "\n".join(lines) + f"\n\nSeeds {list(SEEDS)[0]}..{list(SEEDS)[-1]} (n={len(list(SEEDS))}), K={K}, N_train={N_TRAIN}, noise sd={NOISE_SD}.\n"
    (HERE / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    write(run())
