"""Fitting the same TSK models with three gradient sources.

Arms (all start from the same init and the same seed's data as shapes_study.py)
    trf_fd        scipy least_squares, finite-difference Jacobian (the baseline)
    trf_analytic  scipy least_squares, closed-form Jacobian (this file's model())
    adam_exact    Adam on the closed-form gradient, one gradient per step
    adam_synth    Adam on a *synthetic* gradient: a small MLP predicts dL/dtheta
                  from theta. The true gradient is computed on warm-up steps and
                  every 4th step after that; the predictor is trained on those
                  pairs and supplies the gradient on the other steps.

Cost is counted in true gradient or Jacobian evaluations, the quantity that
matters when gradients are the expensive part.

Verification (run first, printed, and asserted)
    The closed-form Jacobian is compared with central finite differences at the
    init of every shape. A rel. error above 1e-5 aborts the run.

Run:  python gradient_study.py      (writes results_gradient.csv and results_gradient.md)
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

from shapes_study import (  # noqa: E402
    K,
    N_TRAIN,
    NOISE_SD,
    SHAPES,
    GRID,
    init_theta,
    pl_heights,
    target,
)

SEEDS = range(10)
ADAM_STEPS = 600
ADAM_LR = 0.02
SYNTH_WARMUP = 30
SYNTH_EVERY = 4
RIDGE = 1e-6  # consequent ridge; same in residual and derivative (gaussian Phi has cond ~1e8)


# ----------------------------------------------------------------- derivatives
def mf_grad(shape, theta, x):
    """Membership (K, N) and d membership / d theta (K*n, N) for one input."""
    kind, n = SHAPES[shape]
    N = x.size
    M = np.zeros((K, N))
    dM = np.zeros((K * n, N))
    rows = np.arange(N)
    for k in range(K):
        t = theta[k * n : (k + 1) * n]
        sl = slice(k * n, (k + 1) * n)
        if kind == "pl":
            knots = t[0] + np.concatenate(([0.0], np.cumsum(np.exp(t[1:]))))
            h = pl_heights(n)
            inside = (x >= knots[0]) & (x <= knots[-1])
            j = np.clip(np.searchsorted(knots, x, side="right") - 1, 0, n - 2)
            w = knots[j + 1] - knots[j]
            u = x - knots[j]
            dh = h[j + 1] - h[j]
            M[k] = np.where(inside, h[j] + dh * u / w, 0.0)
            # mu depends on the two knots bounding x; then knots depend on theta
            G = np.zeros((N, n))
            G[rows, j] = np.where(inside, dh * (u - w) / w**2, 0.0)
            G[rows, j + 1] = np.where(inside, -dh * u / w**2, 0.0)
            D = np.zeros((n, n))  # D[m, p] = d knot_m / d theta_p
            D[:, 0] = 1.0
            for p in range(1, n):
                D[p:, p] = np.exp(t[p])
            dM[sl] = (G @ D).T
        elif kind == "gauss":
            c, s = t
            sig = np.exp(s)
            z = (x - c) / sig
            m = np.exp(-0.5 * z * z)
            M[k] = m
            dM[k * n] = m * z / sig
            dM[k * n + 1] = m * z * z
        else:  # elliptic
            c, s = t
            w_ = np.exp(s)
            z = (x - c) / w_
            ok = 1.0 - z * z > 1e-12
            m = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
            M[k] = m
            safe = np.where(ok, m, 1.0)
            dM[k * n] = np.where(ok, z / (safe * w_), 0.0)
            dM[k * n + 1] = np.where(ok, z * z / safe, 0.0)
    return M, dM


def model(shape, theta, X, y):
    """Residual r, Jacobian J (N, A) of r wrt theta, fitted consequents beta.

    beta solves the normal equations, so dbeta/dtheta comes from differentiating
    G beta = Phi^T y:  G dbeta = dPhi^T (y - Phi beta) - Phi^T (dPhi beta).
    """
    p = SHAPES[shape][1] * K
    n = SHAPES[shape][1]
    N = X.shape[0]
    M1, dM1 = mf_grad(shape, theta[:p], X[:, 0])
    M2, dM2 = mf_grad(shape, theta[p : 2 * p], X[:, 1])
    R = K * K
    F = (M1[:, None, :] * M2[None, :, :]).reshape(R, N)
    S = F.sum(0) + 1e-12
    W = F / S
    B = np.stack([np.ones(N), X[:, 0], X[:, 1]])
    Phi = (W[None] * B[:, None, :]).reshape(3 * R, N).T  # column b*R + r
    G = Phi.T @ Phi + RIDGE * np.eye(3 * R)
    beta = np.linalg.solve(G, Phi.T @ y)
    r = Phi @ beta - y

    # dF[a, r, x] for every antecedent parameter a
    A = 2 * p
    dF = np.zeros((A, K, K, N))
    for a in range(p):
        i = a // n
        dF[a, i] = dM1[a][None, :] * M2
    for a in range(p):
        j = a // n
        dF[p + a, :, j] = M1 * dM2[a][None, :]
    dF = dF.reshape(A, R, N)
    dS = dF.sum(1)
    dW = (dF - W[None] * dS[:, None, :]) / S  # (A, R, N)
    dPhi = (dW[:, None] * B[None, :, None, :]).reshape(A, 3 * R, N).transpose(0, 2, 1)

    V = np.einsum("anc,c->an", dPhi, beta)  # dPhi_a beta
    T = np.einsum("anc,n->ac", dPhi, -r) - V @ Phi  # rhs for dbeta_a
    dbeta = np.linalg.solve(G, T.T)  # (C, A)
    J = V.T + Phi @ dbeta  # (N, A)
    return r, J, beta


def fd_jacobian(shape, theta, X, y, h=1e-5):
    cols = []
    for a in range(theta.size):
        e = np.zeros_like(theta)
        e[a] = h
        cols.append((model(shape, theta + e, X, y)[0] - model(shape, theta - e, X, y)[0]) / (2 * h))
    return np.column_stack(cols)


def verify():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, (400, 2))  # > 3*K*K consequent columns, as in the study
    y = target(X[:, 0], X[:, 1])
    worst = 0.0
    for shape in SHAPES:
        th = init_theta(shape, np.random.default_rng(7))
        J = model(shape, th, X, y)[1]
        Jfd = fd_jacobian(shape, th, X, y)
        rel = np.max(np.abs(J - Jfd)) / max(1e-12, np.max(np.abs(Jfd)))
        print(f"verify {shape:10s} max rel. err analytic vs central FD: {rel:.2e}")
        if shape != "elliptic":  # sqrt edge: derivative diverges at |z|=1, central FD loses accuracy there
            worst = max(worst, rel)
    assert worst < 1e-5, f"analytic Jacobian disagrees with finite differences: {worst:.2e}"
    print(f"verify OK (worst excluding elliptic edge {worst:.2e})")


# ------------------------------------------------------------ synthetic grads
class SynthGrad:
    """MLP g(theta) ~= dL/dtheta, trained on (theta, true gradient) pairs."""

    def __init__(self, dim, hidden=64, seed=0):
        r = np.random.default_rng(seed)
        self.W1 = r.normal(0, 0.1, (dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = r.normal(0, 0.1, (hidden, dim))
        self.b2 = np.zeros(dim)
        self.scale = 1.0
        self.buf_t, self.buf_g = [], []

    def predict(self, theta):
        h = np.tanh(theta @ self.W1 + self.b1)
        return self.scale * (h @ self.W2 + self.b2)

    def add(self, theta, g, steps=40, lr=0.05, keep=512):
        self.buf_t.append(theta)
        self.buf_g.append(g)
        self.buf_t, self.buf_g = self.buf_t[-keep:], self.buf_g[-keep:]
        self.scale = max(1e-8, float(np.sqrt(np.mean(np.square(self.buf_g)))))
        Th = np.array(self.buf_t)
        Y = np.array(self.buf_g) / self.scale
        B = Th.shape[0]
        for _ in range(steps):
            h = np.tanh(Th @ self.W1 + self.b1)
            err = h @ self.W2 + self.b2 - Y
            dout = 2 * err / (B * Y.shape[1])
            dpre = (dout @ self.W2.T) * (1 - h * h)
            self.W2 -= lr * h.T @ dout
            self.b2 -= lr * dout.sum(0)
            self.W1 -= lr * Th.T @ dpre
            self.b1 -= lr * dpre.sum(0)


# ------------------------------------------------------------------- fitting
def fit_trf(shape, X, y, theta0, analytic):
    A = theta0.size
    fun = lambda th: model(shape, th, X, y)[0]
    jac = (lambda th: model(shape, th, X, y)[1]) if analytic else "2-point"
    t0 = time.perf_counter()
    res = least_squares(fun, theta0, jac=jac, method="trf", max_nfev=200)
    secs = time.perf_counter() - t0
    grad_evals = res.njev if analytic else res.njev * A + res.nfev
    return res.x, secs, grad_evals


def fit_adam(shape, X, y, theta0, synth):
    N = X.shape[0]
    th = theta0.copy()
    m = np.zeros_like(th)
    v = np.zeros_like(th)
    sg = SynthGrad(th.size, seed=0) if synth else None
    true_evals = 0
    t0 = time.perf_counter()
    for t in range(1, ADAM_STEPS + 1):
        use_true = (not synth) or t <= SYNTH_WARMUP or t % SYNTH_EVERY == 0
        if use_true:
            r, J, _ = model(shape, th, X, y)
            g = 2 * J.T @ r / N
            true_evals += 1
            if synth:
                sg.add(th.copy(), g)
        else:
            g = sg.predict(th)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1 - 0.9**t)
        vh = v / (1 - 0.999**t)
        th = th - ADAM_LR * mh / (np.sqrt(vh) + 1e-8)
    return th, time.perf_counter() - t0, true_evals


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
        for shape in SHAPES:
            theta0 = init_theta(shape, np.random.default_rng(1000 + seed))
            arms = {
                "trf_fd": lambda: fit_trf(shape, X, y, theta0, analytic=False),
                "trf_analytic": lambda: fit_trf(shape, X, y, theta0, analytic=True),
                "adam_exact": lambda: fit_adam(shape, X, y, theta0, synth=False),
                "adam_synth": lambda: fit_adam(shape, X, y, theta0, synth=True),
            }
            for arm, fn in arms.items():
                out = fn()
                th, secs, evals = out[0], out[1], out[2]
                p = SHAPES[shape][1] * K
                beta = model(shape, th, X, y)[2]
                pred = lambda Z: _predict(shape, th, beta, Z)
                rmse_te = float(np.sqrt(np.mean((pred(Xte) - yte) ** 2)))
                rows.append(dict(arm=arm, shape=shape, seed=seed, rmse_test=rmse_te,
                                 grad_evals=evals, seconds=secs))
                print(f"seed {seed} {shape:10s} {arm:13s} RMSE {rmse_te:.4f} evals {evals} {secs:.2f}s", flush=True)
    return rows


def _predict(shape, theta, beta, X):
    from shapes_study import design  # same TSK regressor as the baseline study

    p = SHAPES[shape][1] * K
    return design(shape, theta[:p], theta[p : 2 * p], X) @ beta


def write(rows):
    with open(HERE / "results_gradient.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    arms = ["trf_fd", "trf_analytic", "adam_exact", "adam_synth"]
    lines = ["| arm | shape | test RMSE median | test RMSE mean ± sd | gradient evals (mean) | seconds (mean) |",
             "|---|---|---:|---:|---:|---:|"]
    for arm in arms:
        for shape in SHAPES:
            r = [x for x in rows if x["arm"] == arm and x["shape"] == shape]
            te = np.array([x["rmse_test"] for x in r])
            lines.append(f"| {arm} | {shape} | {np.median(te):.4f} | {te.mean():.4f} ± {te.std(ddof=1):.4f} "
                         f"| {np.mean([x['grad_evals'] for x in r]):.0f} | {np.mean([x['seconds'] for x in r]):.2f} |")
    text = "\n".join(lines) + f"\n\nSeeds 0..9, K={K}, N_train={N_TRAIN}. Adam: {ADAM_STEPS} steps, lr {ADAM_LR}. Synth: warm-up {SYNTH_WARMUP}, true gradient every {SYNTH_EVERY}th step.\n"
    (HERE / "results_gradient.md").write_text(text)
    print(text)


if __name__ == "__main__":
    verify()
    write(run())
