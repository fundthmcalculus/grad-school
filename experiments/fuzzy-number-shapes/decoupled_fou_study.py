"""Decoupled type-2: the footprint sets the interval only, never the point.

The problem with calib_study.py: calibrating the footprint also moves the Nie-Tan point
prediction (noise-free RMSE rose from 0.062 to 0.097 for the local Gaussian footprint).

Here the point comes from the type-1 model on the fuzzy-number centres. Its consequents
are solved on the centre firing, so the footprint does not enter the point at all. The
footprint is then calibrated only for the interval: KM bounds on the upper and lower firing,
with the type-1 consequents, scored by Winkler on training targets.

Arms (per seed and noise model)
    B2_gauss   kNN residual interval (k = 100) on the Gaussian type-1 point
    B2_pent    kNN residual interval (k = 100) on the pentagon type-1 point
    DEC_gauss  Gaussian local footprint, calibrated for the interval only
    DEC_pent   pentagon footprint, calibrated for the interval only

Pre-registered expectations (written before the run)
    E1  DEC_* noise-free RMSE equals the corresponding type-1 point RMSE exactly
        (the footprint cannot touch the point). Checked in code, abort on mismatch.
    E2  x_hetero: DEC_gauss tracks noise better than the type-1 gaussian B2-style
        baselines, Spearman(width, sd) > 0.5.
    E3  x_hetero: DEC_gauss Winkler < B2_gauss on mean. (Expected to fail: kNN residuals are
        a strong baseline. Reported either way.)
    E4  x_hetero: DEC_gauss Winkler within 0.05 of B2_gauss (the footprint is not much worse
        than the strongest simple baseline).

Run:  python decoupled_fou_study.py  (writes results_decoupled.csv and results_decoupled.md)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calib_study as cs  # noqa: E402
import local_pent_study as lp  # noqa: E402

K_NN = 100
NOISE_MODELS = ["x_hetero", "regime"]
SEEDS = range(10)
ARMS = ["B2_gauss", "B2_pent", "DEC_gauss", "DEC_pent"]


def halfwidth_knn(Xtr, rtr, Xq):
    _, idx = cKDTree(Xtr).query(Xq, k=K_NN)
    return np.quantile(np.abs(rtr[idx]), 0.9, axis=1)


def set_footprint(variant, theta_c):
    """Centre parameters from a collapsed fit, with a footprint switched on at a default size."""
    th = theta_c.copy()
    if variant == "gauss_local":
        th[2 :: lp.PG] = np.log(0.1)
        th[3 :: lp.PG] = 0.0
    else:
        th[5 :: lp.PP] = np.log(0.1)
    return th


def calibrate_interval(variant, theta_fp, beta, X, y):
    """Optimise footprint parameters only, to minimise training Winkler. beta is fixed."""
    kind, P, _, _ = lp.VARIANTS[variant]
    idx, bounds = cs.footprint_index(variant)

    def objective(v):
        th = theta_fp.copy()
        th[idx] = v
        _, lo, hi, _ = lp.predict_all(kind, P, th, beta, X)
        return float(cs.winkler(y, lo, hi).mean())

    res = minimize(objective, theta_fp[idx], method="L-BFGS-B", bounds=bounds, options={"maxiter": 40})
    th = theta_fp.copy()
    th[idx] = res.x
    return th


def run():
    rows = []
    for noise in NOISE_MODELS:
        for seed in SEEDS:
            X, y = cs.make_train(noise, seed)
            Xte, f, yte, sd = cs.make_test(noise, seed)
            group = sd > 0.1

            out = {}
            for kind_name, collapse, variant in [("gauss", "gauss_collapse", "gauss_local"),
                                                 ("pent", "pent_collapse", "pent_fou")]:
                kind, P, _, _ = lp.VARIANTS[collapse]
                th_c, _, _ = lp.fit(collapse, X, y, seed)
                beta = lp.model(kind, P, th_c, X, y)[2]
                tr_point = lp.predict_all(kind, P, th_c, beta, X)[0]
                te_point = lp.predict_all(kind, P, th_c, beta, Xte)[0]
                r_tr = y - tr_point
                out[f"B2_{kind_name}"] = (te_point,
                                          te_point - halfwidth_knn(X, r_tr, Xte),
                                          te_point + halfwidth_knn(X, r_tr, Xte))
                t0 = time.perf_counter()
                th_fp = calibrate_interval(variant, set_footprint(variant, th_c), beta, X, y)
                secs = time.perf_counter() - t0
                _, lo, hi, _ = lp.predict_all(lp.VARIANTS[variant][0], lp.VARIANTS[variant][1],
                                              th_fp, beta, Xte)
                # point stays the type-1 point of the centres (footprint-free)
                out[f"DEC_{kind_name}"] = (te_point, lo, hi)
                print(f"  calibrated {variant} in {secs:.0f}s", flush=True)

            # E1: the decoupled point equals the type-1 point (same centres, same beta)
            for kind_name in ["gauss", "pent"]:
                assert np.allclose(out[f"DEC_{kind_name}"][0], out[f"B2_{kind_name}"][0], atol=1e-12), \
                    "decoupled point differs from type-1 point"

            for arm in ARMS:
                pt, lo, hi = out[arm]
                width = hi - lo
                cov = (yte >= lo) & (yte <= hi)
                rows.append(dict(
                    noise=noise, arm=arm, seed=seed,
                    coverage=float(cov.mean()),
                    cov_group0=float(cov[~group].mean()),
                    cov_group1=float(cov[group].mean()),
                    mean_width=float(width.mean()),
                    width_ratio=float(width[group].mean() / max(width[~group].mean(), 1e-12)),
                    rho_width_sd=float(spearmanr(width, sd).correlation) if width.std() > 0 else 0.0,
                    winkler=float(cs.winkler(yte, lo, hi).mean()),
                    rmse_noisefree=float(np.sqrt(np.mean((pt - f) ** 2))),
                ))
                print(f"{noise:9s} seed {seed} {arm:9s} cov {rows[-1]['coverage']:.3f} "
                      f"width {rows[-1]['mean_width']:.3f} rho {rows[-1]['rho_width_sd']:+.2f} "
                      f"wink {rows[-1]['winkler']:.3f} rmse {rows[-1]['rmse_noisefree']:.4f}", flush=True)
    return rows


def paired(rows, noise, a, b):
    d = []
    for seed in SEEDS:
        wa = [x["winkler"] for x in rows if x["noise"] == noise and x["arm"] == a and x["seed"] == seed][0]
        wb = [x["winkler"] for x in rows if x["noise"] == noise and x["arm"] == b and x["seed"] == seed][0]
        d.append(wa - wb)
    d = np.array(d)
    return d.mean(), d.std(ddof=1), int((d < 0).sum())


def write(rows):
    with open(HERE / "results_decoupled.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| noise | arm | coverage | cov (low sd) | cov (high sd) | mean width | width ratio hi/lo | Spearman(width, sd) | Winkler | RMSE (noise-free) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for noise in NOISE_MODELS:
        for arm in ARMS:
            r = [x for x in rows if x["noise"] == noise and x["arm"] == arm]
            m = lambda k: np.mean([x[k] for x in r])
            lines.append(f"| {noise} | {arm} | {m('coverage'):.3f} | {m('cov_group0'):.3f} | {m('cov_group1'):.3f} "
                         f"| {m('mean_width'):.3f} | {m('width_ratio'):.2f} | {m('rho_width_sd'):+.2f} "
                         f"| {m('winkler'):.3f} | {m('rmse_noisefree'):.4f} |")
    lines += ["", "Paired per-seed Winkler difference (row minus column; negative favours the row).", "",
              "| noise | comparison | mean diff | sd | seeds row better |", "|---|---|---:|---:|---:|"]
    for noise in NOISE_MODELS:
        for a, b in [("DEC_gauss", "B2_gauss"), ("DEC_pent", "B2_pent"), ("DEC_pent", "DEC_gauss"), ("B2_pent", "B2_gauss")]:
            md, sdv, nb = paired(rows, noise, a, b)
            lines.append(f"| {noise} | {a} vs {b} | {md:+.3f} | {sdv:.3f} | {nb}/10 |")
    text = "\n".join(lines) + ("\n\nSeeds 0..9, N_train=500. Point is the type-1 centre model for every arm; only the interval differs. "
                               "Winkler at alpha = 0.1. group = high noise (sd > 0.1).\n")
    (HERE / "results_decoupled.md").write_text(text)
    print(text)


if __name__ == "__main__":
    write(run())
