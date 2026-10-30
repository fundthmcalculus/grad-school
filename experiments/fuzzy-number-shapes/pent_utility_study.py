"""Does the pentagon footprint add anything? Compared with simple post-hoc intervals.

Every method is scored on the same held-out noisy targets (calib_study.make_test) and the
same seeds. The question is whether the footprint beats an interval built from the residuals
of a plain point predictor, which is the obvious alternative.

Point predictors
    pent_type1   pentagon, type-1 (pent_collapse): the point prediction for B1 and B2
    pent_cal     pentagon footprint, calibrated (calib_study): point for B3 and for PENT
    gauss_cal    Gaussian local footprint, calibrated: point and intervals for GAUSS

Methods
    B1  pent_type1 point +/- global 90% quantile of |training residual|
    B2  pent_type1 point +/- local 90% quantile of |residual| over k=100 nearest training points
        (uses x; the strongest non-footprint baseline)
    B3  pent_cal point +/- global 90% quantile of |training residual| (footprint point, no footprint interval)
    PENT  pentagon footprint, calibrated (KM bounds, as in calib_study)
    GAUSS Gaussian local footprint, calibrated (KM bounds)

Pre-registered decision rule (written before the run)
    D1  PENT has the lowest mean Winkler on x_hetero AND beats B2 on paired per-seed Winkler
        in at least 8 of 10 seeds. Otherwise the pentagon footprint is not shown to add utility.
    D2  On regime (negative control), no method tracks the hidden regime (|Spearman| < 0.2).
    D3  PENT beats GAUSS on paired per-seed Winkler in at least 7 of 10 seeds on x_hetero.
        Otherwise the pentagon-versus-Gaussian gap is not reliable.

Run:  python pent_utility_study.py   (writes results_utility.csv and results_utility.md)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calib_study as cs  # noqa: E402
import local_pent_study as lp  # noqa: E402

K_NN = 100
NOISE_MODELS = ["x_hetero", "regime"]
SEEDS = range(10)
METHODS = ["B1", "B2", "B3", "PENT", "GAUSS"]


def global_halfwidth(r, n):
    return np.full(n, np.quantile(np.abs(r), 0.9))


def local_halfwidth(Xtr, rtr, Xq):
    _, idx = cKDTree(Xtr).query(Xq, k=K_NN)
    return np.quantile(np.abs(rtr[idx]), 0.9, axis=1)


def point_of(kind, P, theta, X, y, Xq):
    beta = lp.model(kind, P, theta, X, y)[2]
    return lp.predict_all(kind, P, theta, beta, Xq)


def run():
    rows = []
    for noise in NOISE_MODELS:
        for seed in SEEDS:
            X, y = cs.make_train(noise, seed)
            Xte, f, yte, sd = cs.make_test(noise, seed)
            group = sd > 0.1

            th_t1, _, _ = lp.fit("pent_collapse", X, y, seed)
            p_tr_t1, _, _, _ = point_of("pent", lp.PP, th_t1, X, y, X)
            p_te_t1, _, _, _ = point_of("pent", lp.PP, th_t1, X, y, Xte)
            r_t1 = y - p_tr_t1

            th_pf, _, _ = lp.fit("pent_fou", X, y, seed)
            th_pc, _, _ = cs.calibrate("pent_fou", th_pf, X, y)
            p_tr_pc, _, _, _ = point_of("pent", lp.PP, th_pc, X, y, X)
            p_te_pc, lo_pc, hi_pc, und_pc = point_of("pent", lp.PP, th_pc, X, y, Xte)
            r_pc = y - p_tr_pc

            th_gf, _, _ = lp.fit("gauss_local", X, y, seed)
            th_gc, _, _ = cs.calibrate("gauss_local", th_gf, X, y)
            p_te_gc, lo_gc, hi_gc, und_gc = point_of("gauss", lp.PG, th_gc, X, y, Xte)

            cands = {
                "B1": (p_te_t1, p_te_t1 - global_halfwidth(r_t1, len(Xte)), p_te_t1 + global_halfwidth(r_t1, len(Xte))),
                "B2": (p_te_t1, p_te_t1 - local_halfwidth(X, r_t1, Xte), p_te_t1 + local_halfwidth(X, r_t1, Xte)),
                "B3": (p_te_pc, p_te_pc - global_halfwidth(r_pc, len(Xte)), p_te_pc + global_halfwidth(r_pc, len(Xte))),
                "PENT": (p_te_pc, lo_pc, hi_pc),
                "GAUSS": (p_te_gc, lo_gc, hi_gc),
            }
            for name, (pt, lo, hi) in cands.items():
                width = hi - lo
                cov = (yte >= lo) & (yte <= hi)
                rows.append(dict(
                    noise=noise, method=name, seed=seed,
                    coverage=float(cov.mean()),
                    cov_group0=float(cov[~group].mean()),
                    cov_group1=float(cov[group].mean()),
                    mean_width=float(width.mean()),
                    width_ratio=float(width[group].mean() / max(width[~group].mean(), 1e-12)),
                    rho_width_sd=float(spearmanr(width, sd).correlation) if width.std() > 0 else 0.0,
                    rho_width_group=float(spearmanr(width, group).correlation) if width.std() > 0 else 0.0,
                    winkler=float(cs.winkler(yte, lo, hi).mean()),
                    rmse_noisefree=float(np.sqrt(np.mean((pt - f) ** 2))),
                ))
                print(f"{noise:9s} seed {seed} {name:5s} cov {rows[-1]['coverage']:.3f} "
                      f"width {rows[-1]['mean_width']:.3f} rho {rows[-1]['rho_width_sd']:+.2f} "
                      f"wink {rows[-1]['winkler']:.3f} rmse {rows[-1]['rmse_noisefree']:.4f}", flush=True)
    return rows


def paired(rows, noise, a, b):
    """Per-seed Winkler difference a - b (negative = a better)."""
    d = []
    for seed in SEEDS:
        wa = [x["winkler"] for x in rows if x["noise"] == noise and x["method"] == a and x["seed"] == seed][0]
        wb = [x["winkler"] for x in rows if x["noise"] == noise and x["method"] == b and x["seed"] == seed][0]
        d.append(wa - wb)
    d = np.array(d)
    return d.mean(), d.std(ddof=1), int((d < 0).sum())


def write(rows):
    with open(HERE / "results_utility.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| noise | method | coverage | cov (low sd) | cov (high sd) | mean width | width ratio hi/lo | Spearman(width, sd) | Winkler | RMSE (noise-free) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for noise in NOISE_MODELS:
        for m in METHODS:
            r = [x for x in rows if x["noise"] == noise and x["method"] == m]
            mean = lambda k: np.mean([x[k] for x in r])
            lines.append(f"| {noise} | {m} | {mean('coverage'):.3f} | {mean('cov_group0'):.3f} | {mean('cov_group1'):.3f} "
                         f"| {mean('mean_width'):.3f} | {mean('width_ratio'):.2f} | {mean('rho_width_sd'):+.2f} "
                         f"| {mean('winkler'):.3f} | {mean('rmse_noisefree'):.4f} |")
    lines += ["", "Paired per-seed Winkler difference (row minus column; negative favours the row). Mean, sd over seeds, seeds where row < column.", "",
              "| noise | comparison | mean diff | sd | seeds row better |", "|---|---|---:|---:|---:|"]
    for noise in NOISE_MODELS:
        for a, b in [("PENT", "B2"), ("PENT", "B1"), ("PENT", "B3"), ("PENT", "GAUSS"), ("B3", "B2")]:
            md, sdv, nb = paired(rows, noise, a, b)
            lines.append(f"| {noise} | {a} vs {b} | {md:+.3f} | {sdv:.3f} | {nb}/10 |")
    text = "\n".join(lines) + ("\n\nSeeds 0..9, N_train=500. Test: 40x40 grid (x_hetero) or 1600 random points (regime). "
                               "Winkler at alpha = 0.1. group = high noise (sd > 0.1). B2 uses k = 100 nearest training residuals.\n")
    (HERE / "results_utility.md").write_text(text)
    print(text)


if __name__ == "__main__":
    write(run())
