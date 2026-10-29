"""Local footprint of uncertainty, and pentagonal interval type-2 fuzzy numbers.

Two type-2 shape families, both with analytic Jacobians and KM type reduction.

gauss (P=4 per fuzzy number: c, s, v0, v1)
    Upper / lower Gaussian with uncertain spread:
        L_upper(x) = s + w(x),  L_lower(x) = s - w(x)
        w(x) = exp(v0 + v1 (x - c))          <- local: the spread depends on x
    gauss_global fixes v1 = 0 (the Part 3 model); gauss_local fits it.

pent (P=6 per fuzzy number: 5 knot params, v)
    Pentagon with knots t_0..t_4 (peak at t_2 = centre), heights (0, .5, 1, .5, 0).
    Upper membership: knots scaled about the peak by lam_u = exp(+w), w = exp(v).
    Lower membership: knots scaled by lam_l = exp(-w).
    Scaling about the peak keeps upper >= lower pointwise.
    pent_collapse drops v (w = 0): the type-1 pentagon through the type-2 path.

Model and fitting as in type2_study.py / km_study.py: consequents by least squares on the
averaged firing (ridge 1e-2), Nie-Tan point prediction, Karnik-Mendel output bounds.

Pre-registered expectations (written before the run)
    LP1  gauss_local has higher Spearman(width, local noise sd) than gauss_global,
         and > 0.3.
    LP2  gauss_local fits a non-trivial spread: mean |v1| > 0.1.
    LP3  pent_collapse gives zero KM width (manipulation check).
    LP4  pent_fou median test RMSE is no worse than gauss_global's (+0.003 allowed).
    LP5  pent_fou tracks local noise: Spearman > 0.3.
    LP6  pent_fou covers the noise-free target at >= 90%.

Run:  python local_pent_study.py   (writes results_local_pent.csv and results_local_pent.md)
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from shapes_study import K, N_TRAIN, NOISE_SD, GRID, target  # noqa: E402
from km_study import hetero_sd, karnik_mendel  # noqa: E402

RIDGE = 1e-2
SEEDS = range(10)
NK = 5  # pentagon knots
PG = 4  # gauss params per fuzzy number
PP = 6  # pentagon params per fuzzy number
H_PENT = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
VARIANTS = {
    # name: (kind, P, fou, local)
    "gauss_global": ("gauss", PG, True, False),
    "gauss_local": ("gauss", PG, True, True),
    "pent_fou": ("pent", PP, True, False),
    "pent_collapse": ("pent", PP, False, False),
    "gauss_collapse": ("gauss", PG, False, False),
}


# ------------------------------------------------------------------ membership
def pl_and_G(knots, x):
    """Pentagon membership at x and d mu / d knot_j, shapes (N,) and (N, NK)."""
    N = x.size
    inside = (x >= knots[0]) & (x <= knots[-1])
    j = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, NK - 2)
    # A knot collapse (fitted segment width -> 0) degenerates the pentagon; clamp so the
    # derivative stays finite and the degeneracy shows up in the fit, not as NaN.
    w = np.maximum(knots[j + 1] - knots[j], 1e-9)
    u = x - knots[j]
    dh = H_PENT[j + 1] - H_PENT[j]
    mu = np.where(inside, H_PENT[j] + dh * u / w, 0.0)
    G = np.zeros((N, NK))
    rows = np.arange(N)
    G[rows, j] = np.where(inside, dh * (u - w) / w**2, 0.0)
    G[rows, j + 1] = np.where(inside, -dh * u / w**2, 0.0)
    return mu, G


def mf_gauss(theta, x):
    """Upper/lower Gaussian memberships with (local) uncertain spread, and derivatives."""
    N = x.size
    Mu = np.zeros((K, N))
    Ml = np.zeros((K, N))
    dMu = np.zeros((K * PG, N))
    dMl = np.zeros((K * PG, N))
    for k in range(K):
        c, s, v0, v1 = theta[k * PG : (k + 1) * PG]
        dx = x - c
        w = np.exp(v0 + v1 * dx)
        for sgn, M, dM in ((1.0, Mu, dMu), (-1.0, Ml, dMl)):
            L = s + sgn * w
            sig = np.exp(L)
            z = dx / sig
            mu = np.exp(-0.5 * z * z)
            dLdc = sgn * (-v1 * w)  # w depends on c through dx
            M[k] = mu
            dM[k * PG + 0] = z * mu / sig + z * z * mu * dLdc
            dM[k * PG + 1] = z * z * mu
            dM[k * PG + 2] = z * z * mu * sgn * w
            dM[k * PG + 3] = z * z * mu * sgn * w * dx
    return Mu, Ml, dMu, dMl


def mf_pent(theta, x):
    """Upper/lower scaled pentagons and derivatives wrt the knot params and v."""
    N = x.size
    Mu = np.zeros((K, N))
    Ml = np.zeros((K, N))
    dMu = np.zeros((K * PP, N))
    dMl = np.zeros((K * PP, N))
    Dt = np.zeros((NK, NK))  # d knot_m / d theta_p (knot params: t0, 4 log-increments)
    Dt[:, 0] = 1.0
    eye = np.eye(NK)
    for k in range(K):
        th = theta[k * PP : (k + 1) * PP]
        incr = th[1:5]
        for p in range(1, NK):
            Dt[p:, p] = np.exp(incr[p - 1])
        t = th[0] + np.concatenate(([0.0], np.cumsum(np.exp(incr))))
        tc = t[2]
        w = np.exp(th[5])
        for sgn, M, dM in ((1.0, Mu, dMu), (-1.0, Ml, dMl)):
            lam = np.exp(sgn * w)
            Kn = tc + lam * (t - tc)
            mu, G = pl_and_G(Kn, x)
            M[k] = mu
            # d K_j / d t_m = lam*delta_jm + (1-lam)*delta_m2
            DK = lam * eye + (1 - lam) * np.outer(np.ones(NK), eye[2])
            dK_dv = sgn * w * lam * (t - tc)
            dM[k * PP : k * PP + 5] = (G @ DK @ Dt).T
            dM[k * PP + 5] = G @ dK_dv
    return Mu, Ml, dMu, dMl


def mf(kind, theta, x):
    return mf_gauss(theta, x) if kind == "gauss" else mf_pent(theta, x)


# ------------------------------------------------------------------------ TSK
def model(kind, P, theta, X, y):
    """Residual, Jacobian wrt full theta, beta, firing bounds. Same structure as type2_study."""
    p = K * P
    N = X.shape[0]
    R = K * K
    Mu1, Ml1, dMu1, dMl1 = mf(kind, theta[:p], X[:, 0])
    Mu2, Ml2, dMu2, dMl2 = mf(kind, theta[p:], X[:, 1])
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


def predict_all(kind, P, theta, beta, X):
    _, _, _, Fu, Fl = model(kind, P, theta, X, np.zeros(X.shape[0]))
    N = X.shape[0]
    B = np.stack([np.ones(N), X[:, 0], X[:, 1]])
    R = K * K
    yhat = beta.reshape(3, R).T @ B
    Fm = 0.5 * (Fu + Fl)
    point = (Fm * yhat).sum(0) / (Fm.sum(0) + 1e-12)
    lo = np.minimum(karnik_mendel(yhat, Fl, Fu, True), karnik_mendel(yhat, Fl, Fu, False))
    hi = np.maximum(karnik_mendel(yhat, Fl, Fu, True), karnik_mendel(yhat, Fl, Fu, False))
    # Where the lower firing vanishes for every rule, the KM weights are 0/0 and the output
    # interval is undefined. Such points collapse to the point prediction; the fraction is returned.
    und = np.isnan(lo) | np.isnan(hi)
    lo = np.where(und, point, lo)
    hi = np.where(und, point, hi)
    return point, lo, hi, float(und.mean())


# ----------------------------------------------------------- init and fitting
def init_full(kind, seed):
    rng = np.random.default_rng(1000 + seed)
    centres = np.linspace(0.0, 1.0, K)
    h = 1.0 / (K - 1)
    one = []
    if kind == "gauss":
        for c in centres:
            one += [c, np.log(h), np.log(0.1), 0.0]
    else:
        for c in centres:
            one += [c - h, *([np.log(2 * h / 4)] * 4), np.log(0.1)]
    full = np.concatenate([one, one]).astype(float)
    return full + rng.normal(0.0, 0.05, full.size)


def fit(variant, X, y, seed):
    kind, P, fou, local = VARIANTS[variant]
    full0 = init_full(kind, seed)
    idx = np.arange(full0.size)
    if kind == "gauss" and not local:
        idx = np.array([i for i in idx if i % PG != 3])  # v1 fixed at 0
    default = np.zeros(full0.size)
    if not fou:
        if kind == "gauss":
            idx = np.array([i for i in idx if i % PG not in (2, 3)])  # drop v0, v1
            default[2::PG] = -50.0  # w = exp(-50) ~ 0
        else:
            idx = np.array([i for i in idx if i % PP != PP - 1])  # drop v
            default[P - 1 :: P] = -50.0
    x0 = full0[idx]

    def expand(x):
        th = default.copy()
        th[idx] = x
        return th

    def res(x):
        return model(kind, P, expand(x), X, y)[0]

    def jac(x):
        return model(kind, P, expand(x), X, y)[1][:, idx]

    t0 = time.perf_counter()
    sol = least_squares(res, x0, jac=jac, method="trf", max_nfev=200)
    return expand(sol.x), time.perf_counter() - t0, sol.njev


# --------------------------------------------------------------- verification
def verify():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (400, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1]
    for variant in VARIANTS:
        kind, P, fou, local = VARIANTS[variant]
        full0 = init_full(kind, 0)
        J = model(kind, P, full0, X, y)[1]
        h = 1e-6
        cols = []
        for a in range(full0.size):
            e = np.zeros_like(full0)
            e[a] = h
            cols.append((model(kind, P, full0 + e, X, y)[0] - model(kind, P, full0 - e, X, y)[0]) / (2 * h))
        Jfd = np.column_stack(cols)
        rel = np.max(np.abs(J - Jfd)) / np.max(np.abs(Jfd))
        print(f"verify {variant:14s} analytic vs central FD: max rel. err {rel:.2e}", flush=True)
        assert rel < 1e-5, f"{variant}: Jacobian disagrees with finite differences ({rel:.2e})"
    # upper >= lower pointwise for the pentagon FOU
    xg = np.linspace(0, 1, 201)
    th = init_full("pent", 0)
    Mu, Ml, _, _ = mf_pent(th, xg)
    assert np.all(Mu >= Ml - 1e-12), "pentagon upper below lower"
    print("verify pentagon upper >= lower pointwise: OK", flush=True)


# ------------------------------------------------------------------------ run
def run():
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
        y_noisy = yte + np.random.default_rng(500 + seed).normal(0.0, sd_te)
        for variant in VARIANTS:
            kind, P, fou, local = VARIANTS[variant]
            theta, secs, njev = fit(variant, X, y, seed)
            beta = model(kind, P, theta, X, y)[2]
            point, lo, hi, und = predict_all(kind, P, theta, beta, Xte)
            width = hi - lo
            rmse = float(np.sqrt(np.mean((point - yte) ** 2)))
            if kind == "gauss":
                vs = theta.reshape(2 * K, PG)[:, 3]
                fitted = float(np.mean(np.abs(vs)))
                wbar = float(np.mean(np.exp(theta.reshape(2 * K, PG)[:, 2])))
            else:
                vs = theta.reshape(2 * K, PP)[:, 5]
                fitted = float(np.mean(np.abs(vs)))
                wbar = float(np.mean(np.exp(vs)))
            rho = float(spearmanr(width, sd_te).correlation) if width.std() > 0 else 0.0
            rows.append(dict(
                variant=variant, seed=seed, rmse_test=rmse,
                cov_noisefree=float(np.mean((yte >= lo) & (yte <= hi))),
                cov_noisy=float(np.mean((y_noisy >= lo) & (y_noisy <= hi))),
                mean_width=float(np.mean(width)),
                width_lowsd=float(np.mean(width[sd_te < 0.1])),
                width_highsd=float(np.mean(width[sd_te > 0.1])),
                rho_width_sd=rho, abs_v1_or_v=fitted, w_mean=wbar,
                jacobians=njev, seconds=secs, frac_undefined=und,
            ))
            print(f"seed {seed} {variant:14s} RMSE {rmse:.4f} cov(nf) {rows[-1]['cov_noisefree']:.3f} "
                  f"width {rows[-1]['mean_width']:.3f} rho {rho:+.2f} w {wbar:.3f} {secs:.1f}s", flush=True)
    return rows


def write(rows):
    with open(HERE / "results_local_pent.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = ["| variant | RMSE median | RMSE mean ± sd | cov. noise-free | cov. noisy | mean width | width low sd | width high sd | Spearman(width, sd) | mean abs(v1) | mean w | undefined grid points |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in VARIANTS:
        r = [x for x in rows if x["variant"] == v]
        m = lambda k: np.mean([x[k] for x in r])
        te = np.array([x["rmse_test"] for x in r])
        lines.append(f"| {v} | {np.median(te):.4f} | {te.mean():.4f} ± {te.std(ddof=1):.4f} "
                     f"| {m('cov_noisefree'):.3f} | {m('cov_noisy'):.3f} | {m('mean_width'):.3f} "
                     f"| {m('width_lowsd'):.3f} | {m('width_highsd'):.3f} | {m('rho_width_sd'):+.2f} "
                     f"| {m('abs_v1_or_v'):.3f} | {m('w_mean'):.3f} | {m('frac_undefined') * 100:.2f}% |")
    text = "\n".join(lines) + ("\n\nSeeds 0..9, K=5, N_train=500, heteroscedastic noise sd 0.02 (x1<0.5) / 0.15. "
                               "Consequent ridge 1e-2. KM output bounds. mean abs(v1) is the fitted local slope for gauss_local "
                               "and the fitted v for pent variants, averaged over fuzzy numbers.\n")
    (HERE / "results_local_pent.md").write_text(text)
    print(text)


if __name__ == "__main__":
    verify()
    write(run())
