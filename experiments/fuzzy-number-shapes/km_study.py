"""Karnik-Mendel type reduction for the interval type-2 TSK, and a calibration test.

Changes from type2_study.py
    1. Output interval by Karnik-Mendel (KM) type reduction of the interval
       type-2 output, instead of the min/max of two firing-extreme outputs.
       Rules have firing intervals [F_l, F_u] and scalar consequent values y_r(x).
       KM finds the extremes of sum(w_r y_r) / sum(w_r) over w_r in [F_l, F_u].
    2. Heteroscedastic training noise: sd = 0.02 where x1 < 0.5, 0.15 otherwise.
       A useful interval width should follow the local noise. (Last study used
       homoscedastic noise, so that check could not be made.)

Pre-registered expectations (written before the run)
    K1  KM matches brute-force optimisation over all switch points (verification).
    K2  collapse control (w = 0) gives zero width under KM.
    K3  Spearman correlation between KM width and local noise sd, on the test
        grid, is > 0.3 for the FOU model (the interval tracks noise).
    K4  KM interval covers the noise-free target at >= 90% of test points.
    K5  KM interval covers noisy test targets at < 50% (model-uncertainty
        intervals are not predictive intervals; this shows the gap).

Run:  python km_study.py   (writes results_km.csv and results_km.md)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from shapes_study import K, N_TRAIN, GRID, target  # noqa: E402
import type2_study as t2  # noqa: E402
from type2_study import it2_model, fit  # noqa: E402

# Consequent ridge. 1e-6 (the type2_study value) gives consequent coefficients ~120 and
# intervals ~35 wide (see results_km_ridge1e-6.md). 1e-2 was chosen after a sweep on seeds 0-1,
# which are also in the evaluation below, so treat that choice as tuned, not held out.
CONSEQ_RIDGE = 1e-2
t2.RIDGE = CONSEQ_RIDGE

SEEDS = range(10)


def hetero_sd(X):
    return np.where(X[:, 0] < 0.5, 0.02, 0.15)


def karnik_mendel(yr, fl, fu, left):
    """Karnik-Mendel endpoint. yr, fl, fu have shape (R, N): rules by samples."""
    order = np.argsort(yr, axis=0)
    y = np.take_along_axis(yr, order, 0)
    L = np.take_along_axis(fl, order, 0)
    U = np.take_along_axis(fu, order, 0)
    R, N = y.shape
    idx = np.arange(R)[:, None]
    w = 0.5 * (L + U)
    yc = (w * y).sum(0) / w.sum(0)
    k_prev = np.full(N, -1)
    for _ in range(200):
        k = np.clip((y <= yc).sum(0), 1, R - 1)
        mask = idx < k  # rules 0..k-1 (smallest consequents)
        if left:
            w = np.where(mask, U, L)
        else:
            w = np.where(mask, L, U)
        yc = (w * y).sum(0) / w.sum(0)
        if np.array_equal(k, k_prev):
            break
        k_prev = k
    return yc


def brute_endpoint(yr, fl, fu, left):
    """Minimum (left) or maximum (right) over all switch points, for one sample."""
    order = np.argsort(yr)
    y, L, U = yr[order], fl[order], fu[order]
    R = y.size
    best = np.inf if left else -np.inf
    for k in range(1, R):
        if left:
            w = np.concatenate([U[:k], L[k:]])
        else:
            w = np.concatenate([L[:k], U[k:]])
        v = (w * y).sum() / w.sum()
        best = min(best, v) if left else max(best, v)
    return best


def verify_km(rng):
    worst = 0.0
    for _ in range(300):
        R = 25
        yr = rng.normal(size=(R, 1))
        fu = rng.uniform(0.01, 1.0, size=(R, 1))
        fl = fu * rng.uniform(0.0, 1.0, size=(R, 1))
        for left in (True, False):
            km = karnik_mendel(yr, fl, fu, left)[0]
            bf = brute_endpoint(yr[:, 0], fl[:, 0], fu[:, 0], left)
            worst = max(worst, abs(km - bf))
    print(f"verify KM vs brute force over switch points: worst |diff| {worst:.2e}")
    assert worst < 1e-9, f"KM disagrees with brute force: {worst:.2e}"
    return worst


def predict_all(theta, beta, X):
    """Point (Nie-Tan), min/max bounds (previous study), and KM bounds."""
    _, _, _, Fu, Fl = it2_model(theta, X, np.zeros(X.shape[0]))
    N = X.shape[0]
    B = np.stack([np.ones(N), X[:, 0], X[:, 1]])
    R = K * K
    yhat = beta.reshape(3, R).T @ B  # (R, N)
    point = ((0.5 * (Fu + Fl)) * yhat).sum(0) / ((0.5 * (Fu + Fl)).sum(0) + 1e-12)
    a = (Fl * yhat).sum(0) / (Fl.sum(0) + 1e-12)
    b = (Fu * yhat).sum(0) / (Fu.sum(0) + 1e-12)
    nt_lo, nt_hi = np.minimum(a, b), np.maximum(a, b)
    km_lo = np.minimum(karnik_mendel(yhat, Fl, Fu, True), karnik_mendel(yhat, Fl, Fu, False))
    km_hi = np.maximum(karnik_mendel(yhat, Fl, Fu, True), karnik_mendel(yhat, Fl, Fu, False))
    return point, (nt_lo, nt_hi), (km_lo, km_hi)


def run():
    rng_v = np.random.default_rng(0)
    verify_km(rng_v)

    grid = np.linspace(0.0, 1.0, GRID)
    g1, g2 = np.meshgrid(grid, grid)
    Xte = np.column_stack([g1.ravel(), g2.ravel()])
    yte = target(Xte[:, 0], Xte[:, 1])
    sd_te = hetero_sd(Xte)

    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X = rng.uniform(0.0, 1.0, (N_TRAIN, 2))
        y = target(X[:, 0], X[:, 1]) + rng.normal(0.0, hetero_sd(X))
        y_noisy_te = yte + np.random.default_rng(500 + seed).normal(0.0, sd_te)
        for arm, fou in [("km_fou", True), ("km_collapse", False)]:
            theta, secs, njev = fit(X, y, seed, fou)
            beta = it2_model(theta, X, y)[2]
            point, nt, km = predict_all(theta, beta, Xte)
            rmse = float(np.sqrt(np.mean((point - yte) ** 2)))
            for name, (lo, hi) in [("nie_tan", nt), ("km", km)]:
                width = hi - lo
                rows.append(dict(
                    arm=arm, bounds=name, seed=seed, rmse_test=rmse,
                    cov_noisefree=float(np.mean((yte >= lo) & (yte <= hi))),
                    cov_noisy=float(np.mean((y_noisy_te >= lo) & (y_noisy_te <= hi))),
                    mean_width=float(np.mean(width)),
                    rho_width_sd=float(spearmanr(width, sd_te).correlation) if width.std() > 0 else 0.0,
                    width_lowsd=float(np.mean(width[sd_te < 0.1])),
                    width_highsd=float(np.mean(width[sd_te > 0.1])),
                ))
                print(f"seed {seed} {arm:12s} {name:8s} RMSE {rmse:.4f} cov(nf) {rows[-1]['cov_noisefree']:.3f} "
                      f"cov(noisy) {rows[-1]['cov_noisy']:.3f} width {rows[-1]['mean_width']:.3f} "
                      f"rho {rows[-1]['rho_width_sd']:+.2f}", flush=True)
    return rows


def write(rows):
    with open(HERE / "results_km.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| arm | bounds | cov. noise-free | cov. noisy | mean width | width (low sd) | width (high sd) | Spearman(width, sd) |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ["km_collapse", "km_fou"]:
        for b in ["nie_tan", "km"]:
            r = [x for x in rows if x["arm"] == arm and x["bounds"] == b]
            m = lambda k: np.mean([x[k] for x in r])
            lines.append(f"| {arm} | {b} | {m('cov_noisefree'):.3f} | {m('cov_noisy'):.3f} | {m('mean_width'):.3f} "
                         f"| {m('width_lowsd'):.3f} | {m('width_highsd'):.3f} | {m('rho_width_sd'):+.2f} |")
    text = "\n".join(lines) + ("\n\nSeeds 0..9, K=5, N_train=500, heteroscedastic noise sd 0.02 (x1<0.5) / 0.15. "
                               "Coverage on a 40x40 grid. Spearman computed per seed, averaged.\n")
    (HERE / "results_km.md").write_text(text)
    print(text)


if __name__ == "__main__":
    write(run())
