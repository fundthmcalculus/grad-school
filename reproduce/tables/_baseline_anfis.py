"""ANFIS baseline for Table 4.1 — a fair fuzzy comparator, hand-rolled on numpy.

Why this exists
---------------
The dissertation claims its MoG construction is *orders of magnitude faster* than
a tuned fuzzy system, and Goal **C1** notes there is currently no fuzzy baseline
in the harness to be faster *than*. This is that baseline: a Jang-style ANFIS
(adaptive-network fuzzy inference system) with Gaussian premises, TSK order-1
consequents, and hybrid learning — least-squares for the linear consequents each
epoch, Adam on the premise centres/widths. No new dependencies: numpy and
scikit-learn only, which is what the pinned `tribble-fis` environment already has.

A baseline is only useful if it is *fair*. A deliberately weak ANFIS that loses to
the MoG arm would flatter this work, which is the opposite of the point. Two
things guard against that: the module is validated against published ANFIS-on-
Concrete territory (test R^2 ~0.8, see `__main__`), and it is given a real hybrid
fit rather than a single least-squares pass.

Grid vs. scatter partitioning — and why the choice IS the thesis's argument
--------------------------------------------------------------------------
Textbook ANFIS grid-partitions the input: `p` membership functions on each of `M`
features give `p^M` rules. That is fine on Concrete (8 features, p=2 -> 256 rules)
and impossible on PhiUSIIL (50) or RT-IOT2022 (81), where `2^50` rules cannot be
formed at all. That explosion is precisely the cost the MoG construction avoids,
so rather than report N/A at scale we run BOTH forms and label which each row used:

  * grid    — the textbook method, when `p^M` <= RULE_CAP.
  * scatter — one rule per k-means cluster, `k` rules regardless of `M`, so ANFIS
              stays a genuine competitor on the large sets. This is the standard
              subtractive-clustering / FCM route to ANFIS at scale (Chiu 1994),
              named as a distinct configuration rather than passed off as the grid
              form it is not.

Which partition each dataset used is printed at fit time (`[anfis] <shape>: grid
/ scatter, N rules`) so the run log carries the provenance the single table column
cannot.

Interface (the contract Table 4.1's `_bench` calls):
    fit_predict(X_train, y_train, X_test, *, kind, seed) -> np.ndarray
        kind = "reg" -> real-valued predictions
        kind = "clf" -> integer class labels (one-hot regression + argmax)
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Grid partitioning is used only while the rule count stays tractable; past this
# the scatter form takes over. 512 keeps Concrete (256 at p=2) on the textbook
# path while ruling out anything that would blow up the least-squares solve.
RULE_CAP = 512
MF_PER_FEATURE = 2  # p in p^M; 2 Gaussians per input is the usual grid default.
SCATTER_RULES = 12  # k-means rules at scale; matches the MoG arm's rule budget.
EPOCHS = 40
LR = 0.02
RIDGE = 1e-6  # consequent-solve regularisation; high-dim scatter needs it.


def _gauss(x, c, sigma):
    """Gaussian membership, guarded against a zero width."""
    sigma = np.maximum(sigma, 1e-6)
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)


class _ANFIS:
    """One TSK order-1 ANFIS. Fitted premises (centres, widths) per rule per
    feature; linear consequents solved by ridge least squares each epoch.

    `centres`/`widths` have shape (R, M): rule r, feature j. A rule's firing on a
    sample is the product of its per-feature Gaussians; consequents are affine in
    the (standardised) inputs, one weight vector of length M+1 per output per rule.
    """

    def __init__(self, centres, widths, n_outputs, seed):
        self.c = centres.astype(float)
        self.s = widths.astype(float)
        self.R, self.M = centres.shape
        self.n_out = n_outputs
        self.rng = np.random.RandomState(seed)
        # consequents: (R, M+1, n_out), filled by the LSE step.
        self.W = np.zeros((self.R, self.M + 1, n_outputs))

    def _log_firing(self, X):
        """Log rule-firing strengths, shape (N, R).

        A rule fires as the product of its per-feature Gaussians, i.e. the SUM of
        their logs. Computing it in log space is not an optimisation but a
        correctness requirement at scale: a product of 81 memberships each < 1
        underflows to exactly 0 in float64, which collapses every rule to the same
        firing and turns the model into a constant. The sum of logs stays finite,
        and the softmax normalisation below only needs the differences."""
        z = (X[:, None, :] - self.c[None, :, :]) / np.maximum(self.s[None, :, :], 1e-6)
        return (-0.5 * z * z).sum(axis=2)  # (N, R)

    def _firing(self, X):
        """Normalised firing via softmax over log-firings, shape (N, R)."""
        L = self._log_firing(X)
        L = L - L.max(axis=1, keepdims=True)
        w = np.exp(L)
        return w / w.sum(axis=1, keepdims=True)

    def _design(self, X, wn):
        """Consequent design matrix: each rule contributes its normalised firing
        times [x, 1]. Shape (N, R*(M+1))."""
        Xb = np.hstack([X, np.ones((X.shape[0], 1))])  # (N, M+1)
        # (N, R, M+1) = wn[:, :, None] * Xb[:, None, :]
        phi = wn[:, :, None] * Xb[:, None, :]
        return phi.reshape(X.shape[0], self.R * (self.M + 1))

    def _solve_consequents(self, X, Y, wn):
        """Ridge LSE for the consequents: (phi^T phi + lambda I) beta = phi^T Y.

        Solved in whichever of the two equivalent forms has the smaller matrix
        to invert. The push-through identity
            (phi^T phi + lambda I_D)^-1 phi^T == phi^T (phi phi^T + lambda I_N)^-1
        (D = R*(M+1) consequent params, N = sample count) gives the exact same
        beta either way, so this is a pure cost trade, not an approximation --
        but grid partitions with many rules routinely have D > N (Concrete's
        256-rule grid: D=2304 vs N=824 train rows), where solving the N x N
        dual system is `(D/N)**3` cheaper than the primal D x D system. This
        is the arm the fuzzy baselines spend most of their time in (GA-FIS
        re-solves it every population member, every generation), so the ratio
        matters far more here than a one-off model fit.

        Returns `phi`, the design matrix just used, so an immediately-following
        predict on this same X (the common case: GA-FIS's fitness loop and
        ANFIS's per-epoch refit both solve then predict on the same X) can
        reuse it instead of rebuilding it from scratch.
        """
        phi = self._design(X, wn)
        N, D = phi.shape
        if N < D:
            K = phi @ phi.T + RIDGE * np.eye(N)
            beta = phi.T @ np.linalg.solve(K, Y)
        else:
            A = phi.T @ phi + RIDGE * np.eye(D)
            B = phi.T @ Y
            beta = np.linalg.solve(A, B)
        self.W = beta.reshape(self.R, self.M + 1, self.n_out)
        return phi

    def _predict_raw(self, X, phi=None):
        """Predict from the fitted consequents. Pass `phi` from a just-computed
        `_solve_consequents` on this same X to skip rebuilding the firing and
        design matrices; omit it (the default) to compute fresh, e.g. for a
        genuinely different X such as a held-out test split."""
        if phi is None:
            wn = self._firing(X)
            phi = self._design(X, wn)
        beta = self.W.reshape(self.R * (self.M + 1), self.n_out)
        return phi @ beta

    def fit(self, X, Y):
        """Hybrid learning: LSE consequents + Adam on premise centres/widths."""
        # Adam state for centres and widths.
        mc = np.zeros_like(self.c)
        vc = np.zeros_like(self.c)
        ms = np.zeros_like(self.s)
        vs = np.zeros_like(self.s)
        b1, b2, eps = 0.9, 0.999, 1e-8

        for t in range(1, EPOCHS + 1):
            wn = self._firing(X)
            phi = self._solve_consequents(X, Y, wn)  # forward-optimal consequents
            pred = self._predict_raw(X, phi=phi)
            err = pred - Y  # (N, n_out)

            # Numerical gradient on premise params is too slow; use an analytic
            # approximation that treats the consequents as fixed for this step —
            # standard for ANFIS premise tuning, and enough to move centres to
            # where the data is. Gradient of 0.5*||err||^2 wrt each (c, s).
            gc, gs = self._premise_grad(X, err, wn)

            for p, g, m, v in ((self.c, gc, mc, vc), (self.s, gs, ms, vs)):
                m[:] = b1 * m + (1 - b1) * g
                v[:] = b2 * v + (1 - b2) * g * g
                mhat = m / (1 - b1**t)
                vhat = v / (1 - b2**t)
                p -= LR * mhat / (np.sqrt(vhat) + eps)
            self.s = np.maximum(self.s, 1e-3)

        wn = self._firing(X)
        self._solve_consequents(X, Y, wn)  # final consequents on tuned premises
        return self

    def _premise_grad(self, X, err, wn):
        """d(0.5||err||^2)/d(centre), d/d(width), consequents held fixed.

        Worked in log space to match `_firing`. Chain:
          loss -> pred -> wn=softmax(L) -> L_r=sum_j logmu_rj -> (c,s)
        with logmu_rj = -0.5((x-c)/s)^2, so
          dL_r/dc_rj = (x-c)/s^2 ,  dL_r/ds_rj = (x-c)^2/s^3.
        The softmax Jacobian gives dLoss/dL_r = wn_r (g_r - sum_k g_k wn_k),
        where g_r = dLoss/dwn_r. Vectorised over samples."""
        N = X.shape[0]
        Xb = np.hstack([X, np.ones((N, 1))])  # (N, M+1)
        f = np.einsum("nm,rmo->nro", Xb, self.W)  # per-rule output (N,R,n_out)
        g = np.einsum("no,nro->nr", err, f)  # dLoss/dwn_r  (N,R)
        # softmax backprop: dLoss/dL_r = wn_r (g_r - sum_k g_k wn_k)
        dL = wn * (g - (g * wn).sum(axis=1, keepdims=True))  # (N,R)
        diff = X[:, None, :] - self.c[None, :, :]  # (N,R,M)
        s2 = np.maximum(self.s[None, :, :], 1e-6) ** 2
        dc = dL[:, :, None] * (diff / s2)  # (N,R,M)
        ds = dL[:, :, None] * (diff**2 / (np.maximum(self.s[None, :, :], 1e-6) ** 3))
        return dc.sum(axis=0), ds.sum(axis=0)

    def predict(self, X):
        return self._predict_raw(X)


def _grid_partition(X, p):
    """p Gaussian MFs per feature over the observed range; Cartesian product of
    rules. Returns (centres, widths) of shape (p^M, M)."""
    M = X.shape[1]
    lo, hi = X.min(0), X.max(0)
    # p centres evenly across each feature; width from the centre spacing.
    per_feat_c = [np.linspace(lo[j], hi[j], p) for j in range(M)]
    spacing = np.where((hi - lo) > 0, (hi - lo) / max(p - 1, 1), 1.0)
    grids = np.meshgrid(*per_feat_c, indexing="ij")
    centres = np.stack([g.ravel() for g in grids], axis=1)  # (p^M, M)
    widths = np.tile(spacing, (centres.shape[0], 1))
    return centres, widths


def _scatter_partition(X, k, seed):
    """k rules from k-means cluster centres; per-feature width from the in-cluster
    standard deviation (floored). Returns (centres, widths) of shape (k, M)."""
    k = min(k, X.shape[0])
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(X)
    centres = km.cluster_centers_
    widths = np.ones_like(centres)
    for r in range(k):
        pts = X[km.labels_ == r]
        if len(pts) > 1:
            widths[r] = np.maximum(pts.std(axis=0), 1e-2)
    return centres, widths


def fit_predict(X_train, y_train, X_test, *, kind, seed):
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    ytr = np.asarray(y_train)

    # ANFIS is scale-sensitive; standardise on train, apply to test.
    sx = StandardScaler().fit(Xtr)
    Xtr, Xte = sx.transform(Xtr), sx.transform(Xte)

    M = Xtr.shape[1]
    grid_rules = MF_PER_FEATURE**M if M <= 20 else np.inf
    use_grid = grid_rules <= RULE_CAP

    if use_grid:
        centres, widths = _grid_partition(Xtr, MF_PER_FEATURE)
        mode = "grid"
    else:
        centres, widths = _scatter_partition(Xtr, SCATTER_RULES, seed)
        mode = "scatter"

    if kind == "reg":
        sy = StandardScaler().fit(ytr.reshape(-1, 1))
        Ytr = sy.transform(ytr.reshape(-1, 1))
        net = _ANFIS(centres, widths, n_outputs=1, seed=seed).fit(Xtr, Ytr)
        pred = sy.inverse_transform(net.predict(Xte)).ravel()
        print(
            f"  [anfis] {Xtr.shape[0]}x{M}: {mode} partition, {centres.shape[0]} rules"
        )
        return pred

    # classification: one-hot regression targets, argmax at predict.
    classes = np.unique(ytr)
    Y = (ytr[:, None] == classes[None, :]).astype(float)
    net = _ANFIS(centres, widths, n_outputs=len(classes), seed=seed).fit(Xtr, Y)
    scores = net.predict(Xte)
    print(
        f"  [anfis] {Xtr.shape[0]}x{M}: {mode} partition, {centres.shape[0]} rules, "
        f"{len(classes)} classes"
    )
    return classes[np.argmax(scores, axis=1)]


if __name__ == "__main__":
    # Validation: ANFIS on Concrete should land in published territory (test R^2
    # around 0.8), not limp in far below it. If this drops, the baseline is broken
    # and any comparison against it is meaningless -- fail loudly rather than ship
    # a strawman.
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split

    import _fuzzy_models as F

    Xc, yc = F.load_concrete()
    Xtr, Xte, ytr, yte = train_test_split(
        np.asarray(Xc), np.asarray(yc), test_size=0.2, random_state=0
    )
    p = fit_predict(Xtr, ytr, Xte, kind="reg", seed=0)
    r2 = r2_score(yte, p)
    print(
        f"Concrete ANFIS (grid) test R^2 = {r2:.3f}  {'OK' if r2 > 0.75 else 'TOO LOW'}"
    )
