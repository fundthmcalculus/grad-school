"""Interval type-2 extension of the TSK study: analytic gradients through a footprint of uncertainty.

Model
    Each fuzzy number is a Gaussian with an uncertain standard deviation
    (Mendel's FOU): sigma_u = exp(s + w), sigma_l = exp(s - w), w = exp(v) > 0.
    Upper membership mu_u (larger sigma) >= lower membership mu_l everywhere.
    Parameters per fuzzy number: centre c, log-sigma s, log-width-spread v (3).
    K=5 per input, 2 inputs: 30 antecedent parameters.

    Firing interval [F_l, F_u] per rule is the product of the lower and upper
    memberships. Point prediction is the Nie-Tan type reduction: the consequent
    is solved by least squares on the averaged firing (F_l + F_u)/2, normalised.
    Output interval [y_l, y_u] uses the lower and upper firing with the same
    consequents (the Wu-Mendel uncertainty bounds, simplified).

Arms
    it2_fou      interval type-2, free footprint (v fitted)
    it2_collapse control: the same model with v dropped (w = 0), i.e. type-1
                 Gaussian through the type-2 code path. It should match the
                 type-1 gaussian of gradient_study.py (trf_analytic arm).

Pre-registered expectations (written before the run)
    T1  collapse control reproduces the type-1 gaussian medians within 0.005
        (checks that the type-2 code path changes nothing when the FOU is zero).
    T2  the free FOU lowers the median test RMSE relative to the collapse control.
    T3  the output interval covers the noise-free test target at >= 90% of test
        points, and its width is larger where the training data are noisier
        (reported as a correlation; not a pass/fail).
    T4  manipulation check: the fitted w values differ between seeds and are
        not all at the init value (otherwise the FOU never moved).

Run:  python type2_study.py   (writes results_type2.csv and results_type2.md)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from shapes_study import K, N_TRAIN, NOISE_SD, GRID, target  # noqa: E402

SEEDS = range(10)
P = 3  # centre, log sigma, log w (w = exp(v))
RIDGE = 1e-6  # same consequent ridge as gradient_study.py


def mf_it2(theta, x):
    """Upper/lower Gaussian memberships for one input and their derivatives wrt theta (K*P, N)."""
    N = x.size
    Mu = np.zeros((K, N))
    Ml = np.zeros((K, N))
    dMu = np.zeros((K * P, N))
    dMl = np.zeros((K * P, N))
    for k in range(K):
        c, s, v = theta[k * P : (k + 1) * P]
        w = np.exp(v)
        su = np.exp(s + w)
        sl = np.exp(s - w)
        zu = (x - c) / su
        zl = (x - c) / sl
        mu = np.exp(-0.5 * zu * zu)
        ml = np.exp(-0.5 * zl * zl)
        Mu[k] = mu
        Ml[k] = ml
        dMu[k * P] = mu * zu / su
        dMl[k * P] = ml * zl / sl
        dMu[k * P + 1] = mu * zu * zu
        dMl[k * P + 1] = ml * zl * zl
        dMu[k * P + 2] = mu * zu * zu * w
        dMl[k * P + 2] = -ml * zl * zl * w
    return Mu, Ml, dMu, dMl


def it2_model(theta, X, y):
    """Residual r, Jacobian J (N, A) wrt the full theta (2*K*P), consequents beta, firing parts."""
    p = K * P
    N = X.shape[0]
    R = K * K
    Mu1, Ml1, dMu1, dMl1 = mf_it2(theta[:p], X[:, 0])
    Mu2, Ml2, dMu2, dMl2 = mf_it2(theta[p:], X[:, 1])
    Fu = (Mu1[:, None, :] * Mu2[None, :, :]).reshape(R, N)
    Fl = (Ml1[:, None, :] * Ml2[None, :, :]).reshape(R, N)
    F = 0.5 * (Fu + Fl)
    S = F.sum(0) + 1e-12
    W = F / S
    B = np.stack([np.ones(N), X[:, 0], X[:, 1]])
    Phi = (W[None] * B[:, None, :]).reshape(3 * R, N).T
    G = Phi.T @ Phi + RIDGE * np.eye(3 * R)
    beta = np.linalg.solve(G, Phi.T @ y)
    r = Phi @ beta - y

    A = 2 * p
    dF = np.zeros((A, K, K, N))
    for a in range(p):
        i = a // P
        dF[a, i] = 0.5 * (dMu1[a][None, :] * Mu2 + dMl1[a][None, :] * Ml2)
    for a in range(p):
        j = a // P
        dF[p + a, :, j] = 0.5 * (Mu1 * dMu2[a][None, :] + Ml1 * dMl2[a][None, :])
    dF = dF.reshape(A, R, N)
    dS = dF.sum(1)
    dW = (dF - W[None] * dS[:, None, :]) / S
    dPhi = (dW[:, None] * B[None, :, None, :]).reshape(A, 3 * R, N).transpose(0, 2, 1)

    V = np.einsum("anc,c->an", dPhi, beta)
    T = np.einsum("anc,n->ac", dPhi, -r) - V @ Phi
    dbeta = np.linalg.solve(G, T.T)
    J = V.T + Phi @ dbeta
    return r, J, beta, Fu, Fl


def predict_interval(theta, beta, X):
    """Point (Nie-Tan) prediction and output bounds from lower and upper firing."""
    _, _, _, Fu, Fl = it2_model(theta, X, np.zeros(X.shape[0]))
    N = X.shape[0]
    B = np.stack([np.ones(N), X[:, 0], X[:, 1]])
    R = K * K
    coef = beta.reshape(3, R)  # consequent b, rule r (matches Phi column b*R + r)
    yhat = coef.T @ B  # (R, N) rule outputs

    def tsk(Fx):
        return (Fx * yhat).sum(0) / (Fx.sum(0) + 1e-12)

    point = tsk(0.5 * (Fu + Fl))
    # the two bounds can cross when consequents differ across rules; order them
    a, b = tsk(Fl), tsk(Fu)
    return point, np.minimum(a, b), np.maximum(a, b)


def expand(x_red, fou, idx_full):
    theta = np.zeros(2 * K * P)
    theta[idx_full] = x_red
    return theta


def make_init(seed, fou):
    rng = np.random.default_rng(1000 + seed)
    centres = np.linspace(0.0, 1.0, K)
    h = 1.0 / (K - 1)
    one = []
    for c in centres:
        one += [c, np.log(h), np.log(0.1)]
    full = np.concatenate([one, one]).astype(float)
    full = full + rng.normal(0.0, 0.05, full.size)
    idx = np.arange(2 * K * P)
    if not fou:
        idx = np.array([i for i in idx if i % P != 2])
    return full, idx


def fit(X, y, seed, fou):
    full0, idx = make_init(seed, fou)
    x0 = full0[idx]

    def expand_theta(x):
        th = np.zeros(2 * K * P)
        th[idx] = x
        if not fou:
            th[[i for i in range(2 * K * P) if i % P == 2]] = -np.inf  # w = 0
        return th

    def res(x):
        return it2_model(expand_theta(x), X, y)[0]

    def jac(x):
        J = it2_model(expand_theta(x), X, y)[1]
        return J[:, idx]

    t0 = time.perf_counter()
    sol = least_squares(res, x0, jac=jac, method="trf", max_nfev=200)
    secs = time.perf_counter() - t0
    return expand_theta(sol.x), secs, sol.njev


def verify():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (400, 2))
    y = target(X[:, 0], X[:, 1])
    full0, _ = make_init(0, True)
    J = it2_model(full0, X, y)[1]
    h = 1e-5
    cols = []
    for a in range(full0.size):
        e = np.zeros_like(full0)
        e[a] = h
        cols.append((it2_model(full0 + e, X, y)[0] - it2_model(full0 - e, X, y)[0]) / (2 * h))
    Jfd = np.column_stack(cols)
    rel = np.max(np.abs(J - Jfd)) / np.max(np.abs(Jfd))
    print(f"verify it2 analytic vs central FD: max rel. err {rel:.2e}")
    assert rel < 1e-5, f"type-2 Jacobian disagrees with finite differences: {rel:.2e}"


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
        for arm, fou in [("it2_fou", True), ("it2_collapse", False)]:
            theta, secs, njev = fit(X, y, seed, fou)
            beta = it2_model(theta, X, y)[2]
            point, lo, hi = predict_interval(theta, beta, Xte)
            rmse = float(np.sqrt(np.mean((point - yte) ** 2)))
            covered = float(np.mean((yte >= lo) & (yte <= hi)))
            width = float(np.mean(hi - lo))
            w_fit = np.exp(theta[2::P]).tolist() if fou else []
            rows.append(dict(arm=arm, seed=seed, rmse_test=rmse, coverage=covered,
                             mean_width=width, njev=njev, seconds=secs,
                             w_mean=float(np.mean(w_fit)) if fou else 0.0,
                             w_init=float(np.exp(np.log(0.1))) if fou else 0.0))
            print(f"seed {seed} {arm:13s} RMSE {rmse:.4f} coverage {covered:.3f} width {width:.3f} "
                  f"w {rows[-1]['w_mean']:.3f} jac {njev} {secs:.2f}s", flush=True)
    return rows


def write(rows):
    with open(HERE / "results_type2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| arm | test RMSE median | test RMSE mean ± sd | interval coverage (noise-free test) | mean interval width | Jacobians | fitted mean w |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ["it2_collapse", "it2_fou"]:
        r = [x for x in rows if x["arm"] == arm]
        te = np.array([x["rmse_test"] for x in r])
        lines.append(f"| {arm} | {np.median(te):.4f} | {te.mean():.4f} ± {te.std(ddof=1):.4f} "
                     f"| {np.mean([x['coverage'] for x in r]):.3f} | {np.mean([x['mean_width'] for x in r]):.4f} "
                     f"| {np.mean([x['njev'] for x in r]):.0f} | {np.mean([x['w_mean'] for x in r]):.3f} |")
    text = "\n".join(lines) + "\n\nSeeds 0..9, K=5, N_train=500, noise sd=0.05. Gaussian FOU, Nie-Tan point prediction.\n"
    (HERE / "results_type2.md").write_text(text)
    print(text)


if __name__ == "__main__":
    verify()
    write(run())
