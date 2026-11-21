"""Can an interval type-2 footprint reach nominal coverage and beat a residual interval?

Problem found in decoupled_fou_study.py: the calibrated footprint under-covers (about 0.79 on
held-out noisy targets, against 0.90 nominal), its width tracks noise only partly, and it
loses to the kNN residual interval (B2) on every seed. Training coverage was no higher than test
coverage, so the shortfall is the footprint family or the optimiser, not overfitting.

Candidate fixes tested here, all with the point fixed to the type-1 Gaussian centre model
(as in decoupled_fou_study.py), so only the interval changes:

    DEC       gauss_local footprint, 40-iteration cap (baseline from the decoupled study)
    LONG      the same footprint, 300-iteration cap. Tests whether the optimiser is the cause.
    RULE      per-rule footprint: each of the K*K = 25 rules gets its own upper gain
              exp(a_r) and lower shrink exp(-b_r), a_r, b_r in [0, 2]. The firing intervals
              are [F_r exp(-b_r), F_r exp(a_r)], with F_r the centre firing. Calibrated by
              Winkler on training, consequents fixed. 50 parameters.
    B2        kNN residual interval (k = 100) on the same point. Reference.

Pre-registered expectations (written before the run)
    S1  RULE reaches nominal coverage on x_hetero: training coverage >= 0.88 and held-out
        coverage >= 0.85.
    S2  LONG improves over DEC on training coverage by >= 0.03 (the optimiser is part of the cause).
    S3  RULE Winkler < DEC Winkler on x_hetero (mean over seeds).
    S4  RULE Winkler within 0.03 of B2 on x_hetero. (Pre-registered as a stretch goal.)
    S5  RULE does not track the hidden regime on the regime noise model (|Spearman| < 0.2).
    S6  RULE train-to-held-out Winkler gap < 0.05 (50 parameters on 500 points should not overfit much).

Run:  python fou_solutions_study.py   (writes results_fou_solutions.csv and results_fou_solutions.md)
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
ARMS = ["B2", "DEC", "LONG", "RULE"]


def centre_firing(theta_c, X):
    """Centre firing per rule, shape (R, N), from the Gaussian centre model."""
    _, _, _, Fu, _ = lp.model("gauss", lp.PG, theta_c, X, np.zeros(X.shape[0]))
    return Fu  # with the footprint switched off, Fu == Fl == centre firing


def rule_bounds(F, yhat, a, b):
    """KM output bounds with per-rule firing intervals [F exp(-b), F exp(a)]."""
    Fu = F * np.exp(a)[:, None]
    Fl = F * np.exp(-b)[:, None]
    lo = np.minimum(
        karnik_mendel(yhat, Fl, Fu, True), karnik_mendel(yhat, Fl, Fu, False)
    )
    hi = np.maximum(
        karnik_mendel(yhat, Fl, Fu, True), karnik_mendel(yhat, Fl, Fu, False)
    )
    return lo, hi


def fit_rule(theta_c, beta, X, y):
    F = centre_firing(theta_c, X)
    yhat = beta.reshape(3, R).T @ np.stack([np.ones(X.shape[0]), X[:, 0], X[:, 1]])

    def objective(p):
        a, b = p[:R], p[R:]
        lo, hi = rule_bounds(F, yhat, a, b)
        return float(cs.winkler(y, lo, hi).mean())

    p0 = np.full(2 * R, 0.1)
    res = minimize(
        objective,
        p0,
        method="L-BFGS-B",
        bounds=[(0.0, 2.0)] * (2 * R),
        options={"maxiter": 60},
    )
    return res.x, res.fun


def rule_predict(theta_c, beta, p, X):
    F = centre_firing(theta_c, X)
    yhat = beta.reshape(3, R).T @ np.stack([np.ones(X.shape[0]), X[:, 0], X[:, 1]])
    return rule_bounds(F, yhat, p[:R], p[R:])


def cover(y, lo, hi):
    return float(np.mean((y >= lo) & (y <= hi)))


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
            r_tr = y - tr_point

            arms = {}
            # B2: kNN residual on the same point
            _, idx = cKDTree(X).query(Xte, k=K_NN)
            qk = np.quantile(np.abs(r_tr[idx]), 0.9, axis=1)
            _, idx_tr = cKDTree(X).query(X, k=K_NN)
            arms["B2"] = (te_point - qk, te_point + qk, None)

            # DEC: 40-iteration cap, and LONG: 300-iteration cap, same footprint model
            for name, cap in [("DEC", 40), ("LONG", 300)]:
                th_fp = dec.set_footprint("gauss_local", th_c)
                idx_fp, bounds = cs.footprint_index("gauss_local")

                def objective(v, th_fp=th_fp, idx_fp=idx_fp):
                    th = th_fp.copy()
                    th[idx_fp] = v
                    _, lo, hi, _ = lp.predict_all("gauss", lp.PG, th, beta, X)
                    return float(cs.winkler(y, lo, hi).mean())

                res = minimize(
                    objective,
                    th_fp[idx_fp],
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": cap},
                )
                th_best = th_fp.copy()
                th_best[idx_fp] = res.x
                _, tlo, thi, _ = lp.predict_all("gauss", lp.PG, th_best, beta, X)
                _, lo, hi, _ = lp.predict_all("gauss", lp.PG, th_best, beta, Xte)
                arms[name] = (lo, hi, (tlo, thi))

            # RULE: per-rule footprint
            p, _ = fit_rule(th_c, beta, X, y)
            tlo, thi = rule_predict(th_c, beta, p, X)
            lo, hi = rule_predict(th_c, beta, p, Xte)
            arms["RULE"] = (lo, hi, (tlo, thi))

            for arm in ARMS:
                lo, hi, train = arms[arm]
                width = hi - lo
                cov = (yte >= lo) & (yte <= hi)
                if train is None:
                    trc = trw = float("nan")
                else:
                    trc = cover(y, *train)
                    trw = float(cs.winkler(y, *train).mean())
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
                        winkler=float(cs.winkler(yte, lo, hi).mean()),
                        train_winkler=trw,
                    )
                )
                print(
                    f"{noise:9s} seed {seed} {arm:5s} cov {rows[-1]['coverage']:.3f} "
                    f"train_cov {trc:.3f} rho {rows[-1]['rho_width_sd']:+.2f} "
                    f"wink {rows[-1]['winkler']:.3f} train_wink {trw:.3f}",
                    flush=True,
                )
    return rows


def write(rows):
    with open(HERE / "results_fou_solutions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [
        "| noise | arm | train coverage | held-out coverage | cov (low sd) | cov (high sd) | mean width | Spearman(width, sd) | Winkler held-out | Winkler train |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for noise in NOISE_MODELS:
        for arm in ARMS:
            r = [x for x in rows if x["noise"] == noise and x["arm"] == arm]
            m = lambda k: np.nanmean([x[k] for x in r])
            lines.append(
                f"| {noise} | {arm} | {m('train_coverage'):.3f} | {m('coverage'):.3f} | {m('cov_group0'):.3f} "
                f"| {m('cov_group1'):.3f} | {m('mean_width'):.3f} | {m('rho_width_sd'):+.2f} "
                f"| {m('winkler'):.3f} | {m('train_winkler'):.3f} |"
            )
    text = "\n".join(lines) + (
        "\n\nSeeds 0..9, N_train=500. Point is the type-1 Gaussian centre model for every arm. "
        "Train columns are n/a (nan, shown as blank in the mean) for B2 because it has no fitted parameters on the training set. "
        "Winkler at alpha = 0.1.\n"
    )
    (HERE / "results_fou_solutions.md").write_text(text)
    print(text)


if __name__ == "__main__":
    write(run())
