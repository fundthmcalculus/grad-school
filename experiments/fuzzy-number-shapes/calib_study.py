"""Calibrated footprint of uncertainty, under two noise models.

Stage 1: fit centres, spreads and consequents by least squares (local_pent_study.fit).
Stage 2: freeze everything except the footprint parameters and minimise the mean
         Winkler interval score (alpha = 0.1, i.e. nominal 90%) on the noisy training
         targets. The intervals become predictive intervals, not mean-function bands.

Noise models
    x_hetero  sd = 0.02 where x1 < 0.5, else 0.15. Observable from x.
    regime    a hidden 0/1 regime z ~ Bernoulli(1/2) per point; sd = 0.02 if z = 0,
              0.15 if z = 1. z is not an input, so no footprint of x can see it.
              Negative control for width tracking.

Arms (per variant and noise model, per seed)
    uncal  stage-1 footprint, KM bounds (the previous study)
    cal    footprint calibrated by Winkler score

Evaluation on held-out noisy targets; group = (sd > 0.1), i.e. the high-noise side.

Pre-registered expectations (written before the run)
    C1  x_hetero, gauss_local, cal: noisy-test coverage in [0.85, 0.95].
    C2  x_hetero, gauss_local, cal: Spearman(width, sd) > 0.5.
    C3  x_hetero, gauss_local, cal: mean width ratio high-noise / low-noise >= 1.5.
    C4  regime: |Spearman(width, group)| < 0.2 (cannot track hidden regime).
    C5  regime: conditional coverage gap |cov(group 0) - cov(group 1)| > 0.10.
    C6  x_hetero: cal Winkler score < uncal Winkler score for every variant (median over seeds).

Run:  python calib_study.py   (writes results_calib.csv and results_calib.md)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import local_pent_study as lp  # noqa: E402
from shapes_study import K, N_TRAIN, GRID, target  # noqa: E402

ALPHA = 0.1
SEEDS = range(10)
VARIANT_NAMES = ["gauss_global", "gauss_local", "pent_fou"]
NOISE_MODELS = ["x_hetero", "regime"]
N_TEST_RANDOM = 1600


def sd_x(X):
    return np.where(X[:, 0] < 0.5, 0.02, 0.15)


def make_train(noise, seed):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 1.0, (N_TRAIN, 2))
    if noise == "x_hetero":
        sd = sd_x(X)
    else:
        sd = np.where(rng.integers(0, 2, N_TRAIN) == 0, 0.02, 0.15)
    y = target(X[:, 0], X[:, 1]) + rng.normal(0.0, sd)
    return X, y


def make_test(noise, seed):
    rng = np.random.default_rng(10_000 + seed)
    if noise == "x_hetero":
        grid = np.linspace(0.0, 1.0, GRID)
        g1, g2 = np.meshgrid(grid, grid)
        X = np.column_stack([g1.ravel(), g2.ravel()])
        sd = sd_x(X)
    else:
        X = rng.uniform(0.0, 1.0, (N_TEST_RANDOM, 2))
        sd = np.where(rng.integers(0, 2, N_TEST_RANDOM) == 0, 0.02, 0.15)
    f = target(X[:, 0], X[:, 1])
    y = f + rng.normal(0.0, sd)
    return X, f, y, sd


def winkler(y, lo, hi, alpha=ALPHA):
    return (hi - lo) + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def footprint_index(variant):
    """Indices of the footprint parameters and their bounds."""
    kind, P, _, _ = lp.VARIANTS[variant]
    n = 2 * K * P
    if variant == "gauss_global":
        idx = [i for i in range(n) if i % lp.PG == 2]
        bounds = [(-6.0, 1.0)] * len(idx)
    elif variant == "gauss_local":
        idx = [i for i in range(n) if i % lp.PG in (2, 3)]
        bounds = [(-6.0, 1.0) if i % lp.PG == 2 else (-3.0, 3.0) for i in idx]
    else:
        idx = [i for i in range(n) if i % lp.PP == lp.PP - 1]
        bounds = [(-6.0, 1.0)] * len(idx)
    return np.array(idx), bounds


def calibrate(variant, theta, X, y):
    kind, P, _, _ = lp.VARIANTS[variant]
    idx, bounds = footprint_index(variant)

    def objective(v):
        th = theta.copy()
        th[idx] = v
        beta = lp.model(kind, P, th, X, y)[2]
        _, lo, hi, _ = lp.predict_all(kind, P, th, beta, X)
        return float(winkler(y, lo, hi).mean())

    res = minimize(objective, theta[idx], method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 40})
    th = theta.copy()
    th[idx] = res.x
    return th, res.nit, res.fun


def run():
    rows = []
    for noise in NOISE_MODELS:
        for seed in SEEDS:
            X, y = make_train(noise, seed)
            Xte, f, yte, sd = make_test(noise, seed)
            group = sd > 0.1
            for variant in VARIANT_NAMES:
                kind, P, _, _ = lp.VARIANTS[variant]
                t0 = time.perf_counter()
                theta1, _, _ = lp.fit(variant, X, y, seed)
                theta2, nit, wfit = calibrate(variant, theta1, X, y)
                secs = time.perf_counter() - t0
                for arm, th in [("uncal", theta1), ("cal", theta2)]:
                    beta = lp.model(kind, P, th, X, y)[2]
                    point, lo, hi, und = lp.predict_all(kind, P, th, beta, Xte)
                    width = hi - lo
                    wk = winkler(yte, lo, hi)
                    covered = (yte >= lo) & (yte <= hi)
                    rows.append(dict(
                        noise=noise, variant=variant, seed=seed, arm=arm,
                        coverage=float(covered.mean()),
                        cov_group0=float(covered[~group].mean()),
                        cov_group1=float(covered[group].mean()),
                        mean_width=float(width.mean()),
                        width_ratio=float(width[group].mean() / max(width[~group].mean(), 1e-12)),
                        rho_width_sd=float(spearmanr(width, sd).correlation) if width.std() > 0 else 0.0,
                        rho_width_group=float(spearmanr(width, group).correlation) if width.std() > 0 else 0.0,
                        winkler=float(wk.mean()),
                        rmse_noisefree=float(np.sqrt(np.mean((point - f) ** 2))),
                        frac_undefined=und,
                    ))
                    print(f"{noise:9s} seed {seed} {variant:13s} {arm:5s} cov {rows[-1]['coverage']:.3f} "
                          f"width {rows[-1]['mean_width']:.3f} ratio {rows[-1]['width_ratio']:.2f} "
                          f"rho_sd {rows[-1]['rho_width_sd']:+.2f} wink {rows[-1]['winkler']:.3f} "
                          f"(cal {secs:.0f}s, nit {nit})", flush=True)
    return rows


def write(rows):
    with open(HERE / "results_calib.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| noise | variant | arm | coverage | cov (low sd) | cov (high sd) | mean width | width ratio hi/lo | Spearman(width, sd) | Spearman(width, group) | Winkler | RMSE (noise-free) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for noise in NOISE_MODELS:
        for variant in VARIANT_NAMES:
            for arm in ["uncal", "cal"]:
                r = [x for x in rows if x["noise"] == noise and x["variant"] == variant and x["arm"] == arm]
                m = lambda k: np.mean([x[k] for x in r])
                lines.append(f"| {noise} | {variant} | {arm} | {m('coverage'):.3f} | {m('cov_group0'):.3f} "
                             f"| {m('cov_group1'):.3f} | {m('mean_width'):.3f} | {m('width_ratio'):.2f} "
                             f"| {m('rho_width_sd'):+.2f} | {m('rho_width_group'):+.2f} | {m('winkler'):.3f} "
                             f"| {m('rmse_noisefree'):.4f} |")
    text = "\n".join(lines) + ("\n\nSeeds 0..9, K=5, N_train=500. Test: 40x40 grid (x_hetero) or 1600 random points (regime), "
                               "with fresh noisy targets. group = high noise (sd > 0.1). Winkler at alpha = 0.1. "
                               "Calibration: L-BFGS-B on footprint parameters only, maxiter 40.\n")
    (HERE / "results_calib.md").write_text(text)
    print(text)


if __name__ == "__main__":
    write(run())
