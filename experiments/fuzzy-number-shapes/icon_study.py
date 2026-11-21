"""Interval consequents: a type-2 fix for the under-covering footprint.

Diagnosis (fou_solutions_study.py and the checks behind it)
    The calibrated footprint covers the noise-free target 95-98% of the time, so its mean is
    right. The misses are noise tails: on missed training points |y - f| is about 0.15, close to
    the high-noise sd. The footprint can only reweight the rule firings, and that does not
    supply enough noise-level width. Per-rule footprints (RULE arm) did not fix it either.

Fix tested here: interval consequents (type-2 TSK with interval-valued consequents).
    Each rule r has a consequent interval [yhat_r - d_r, yhat_r + d_r], where yhat_r is the
    type-1 linear consequent and d_r >= 0 is a fitted half-width. The output interval is
        lo = KM_left ( yhat - d ; F ),    hi = KM_right( yhat + d ; F )
    with KM the Karnik-Mendel endpoint (karnik_mendel). Firing is the centre firing F (no footprint),
    so the width comes from the consequent widths alone. Rules form a grid over x, so per-rule
    half-widths can follow local noise. 25 parameters, fitted by Winkler on training data.

Arms
    B2   kNN residual interval (k = 100) on the same centre-model point
    DEC  gauss_local footprint, interval only (decoupled_fou_study baseline)
    ICON interval consequents, as above

Pre-registered expectations (written before the run)
    I1  x_hetero: ICON training and held-out coverage both in [0.85, 0.95].
    I2  x_hetero: ICON Winkler < DEC Winkler (mean over seeds).
    I3  x_hetero: ICON Winkler within 0.03 of B2.
    I4  x_hetero: ICON Spearman(width, sd) > 0.5.
    I5  regime: |Spearman(width, regime)| < 0.2 (cannot see a hidden regime).
    I6  point is the centre model for every arm, so RMSE is identical across arms (checked in code).

Run:  python icon_study.py   (writes results_icon.csv and results_icon.md)
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import calib_study as cs  # noqa: E402
import decoupled_fou_study as dec  # noqa: E402
import local_pent_study as lp  # noqa: E402
from km_study import karnik_mendel  # noqa: E402
from shapes_study import K  # noqa: E402

R = K * K
K_NN = 100
SEEDS = range(10)
NOISE_MODELS = ["x_hetero", "regime"]
ARMS = ["B2", "DEC", "ICON"]


def consequent_yhat(beta, X):
    return beta.reshape(3, R).T @ np.stack([np.ones(X.shape[0]), X[:, 0], X[:, 1]])


def icon_bounds(F, yhat, d):
    lo = karnik_mendel(yhat - d[:, None], F, F, True)
    hi = karnik_mendel(yhat + d[:, None], F, F, False)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def fit_icon(F, yhat, y):
    def objective(d):
        lo, hi = icon_bounds(F, yhat, d)
        return float(cs.winkler(y, lo, hi).mean())

    d0 = np.full(R, 0.1)
    res = minimize(
        objective,
        d0,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * R,
        options={"maxiter": 100},
    )
    return res.x


def centre_firing(theta_c, X):
    _, _, _, Fu, _ = lp.model("gauss", lp.PG, theta_c, X, np.zeros(X.shape[0]))
    return Fu  # footprint off, so Fu == Fl == centre firing


def run():
    rows = []
    for noise in NOISE_MODELS:
        for seed in SEEDS:
            X, y = cs.make_train(noise, seed)
            Xte, f, yte, sd = cs.make_test(noise, seed)
            group = sd > 0.1

            th_c, _, _ = lp.fit("gauss_collapse", X, y, seed)
            beta = lp.model("gauss", lp.PG, th_c, X, y)[2]
            te_point = lp.predict_all("gauss", lp.PG, th_c, beta, Xte)[0]
            tr_point = lp.predict_all("gauss", lp.PG, th_c, beta, X)[0]
            pts = {}

            arms = {}
            r_tr = y - tr_point
            _, idx = cKDTree(X).query(Xte, k=K_NN)
            qk = np.quantile(np.abs(r_tr[idx]), 0.9, axis=1)
            arms["B2"] = (te_point - qk, te_point + qk, (tr_point - 0, None))

            th_fp = dec.calibrate_interval(
                "gauss_local", dec.set_footprint("gauss_local", th_c), beta, X, y
            )
            _, lo, hi, _ = lp.predict_all("gauss", lp.PG, th_fp, beta, Xte)
            _, tlo, thi, _ = lp.predict_all("gauss", lp.PG, th_fp, beta, X)
            arms["DEC"] = (lo, hi, (tlo, thi))

            F_tr = centre_firing(th_c, X)
            F_te = centre_firing(th_c, Xte)
            yhat_tr = consequent_yhat(beta, X)
            yhat_te = consequent_yhat(beta, Xte)
            d = fit_icon(F_tr, yhat_tr, y)
            tlo, thi = icon_bounds(F_tr, yhat_tr, d)
            lo, hi = icon_bounds(F_te, yhat_te, d)
            arms["ICON"] = (lo, hi, (tlo, thi))

            for arm in ARMS:
                lo, hi, train = arms[arm]
                pt = te_point
                width = hi - lo
                cov = (yte >= lo) & (yte <= hi)
                if train[1] is None:
                    trc = float("nan")
                    trw = float("nan")
                else:
                    trc = float(np.mean((y >= train[0]) & (y <= train[1])))
                    trw = float(cs.winkler(y, train[0], train[1]).mean())
                rows.append(
                    dict(
                        noise=noise,
                        arm=arm,
                        seed=seed,
                        coverage=float(cov.mean()),
                        cov_group0=float(cov[~group].mean()),
                        cov_group1=float(cov[group].mean()),
                        train_coverage=trc,
                        mean_width=float(width.mean()),
                        rho_width_sd=(
                            float(spearmanr(width, sd).correlation)
                            if width.std() > 0
                            else 0.0
                        ),
                        rho_width_group=(
                            float(spearmanr(width, group).correlation)
                            if width.std() > 0
                            else 0.0
                        ),
                        winkler=float(cs.winkler(yte, lo, hi).mean()),
                        train_winkler=trw,
                        rmse_noisefree=float(np.sqrt(np.mean((pt - f) ** 2))),
                    )
                )
                print(
                    f"{noise:9s} seed {seed} {arm:5s} cov {rows[-1]['coverage']:.3f} "
                    f"train_cov {trc:.3f} rho {rows[-1]['rho_width_sd']:+.2f} "
                    f"wink {rows[-1]['winkler']:.3f} rmse {rows[-1]['rmse_noisefree']:.4f}",
                    flush=True,
                )
    return rows


def write(rows):
    with open(HERE / "results_icon.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| noise | arm | train coverage | held-out coverage | cov (low sd) | cov (high sd) | mean width | Spearman(width, sd) | Winkler | RMSE (noise-free) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for noise in NOISE_MODELS:
        for arm in ARMS:
            r = [x for x in rows if x["noise"] == noise and x["arm"] == arm]
            m = lambda k: np.nanmean([x[k] for x in r])
            lines.append(
                f"| {noise} | {arm} | {m('train_coverage'):.3f} | {m('coverage'):.3f} | {m('cov_group0'):.3f} "
                f"| {m('cov_group1'):.3f} | {m('mean_width'):.3f} | {m('rho_width_sd'):+.2f} "
                f"| {m('winkler'):.3f} | {m('rmse_noisefree'):.4f} |"
            )
    lines += [
        "",
        "Paired per-seed Winkler difference (row minus column; negative favours the row).",
        "",
        "| noise | comparison | mean diff | sd | seeds row better |",
        "|---|---|---:|---:|---:|",
    ]
    for noise in NOISE_MODELS:
        for a, b in [("ICON", "DEC"), ("ICON", "B2")]:
            d = []
            for seed in SEEDS:
                wa = [
                    x["winkler"]
                    for x in rows
                    if x["noise"] == noise and x["arm"] == a and x["seed"] == seed
                ][0]
                wb = [
                    x["winkler"]
                    for x in rows
                    if x["noise"] == noise and x["arm"] == b and x["seed"] == seed
                ][0]
                d.append(wa - wb)
            d = np.array(d)
            lines.append(
                f"| {noise} | {a} vs {b} | {d.mean():+.3f} | {d.std(ddof=1):.3f} | {int((d < 0).sum())}/10 |"
            )
    text = "\n".join(lines) + (
        "\n\nSeeds 0..9, N_train=500. Point is the type-1 Gaussian centre model for every arm. "
        "Winkler at alpha = 0.1. B2 has no training interval, so its train columns are blank.\n"
    )
    (HERE / "results_icon.md").write_text(text)
    print(text)


if __name__ == "__main__":
    write(run())
