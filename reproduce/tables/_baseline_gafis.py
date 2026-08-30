"""GA-tuned FIS baseline for Table 4.1 — the second fuzzy comparator.

Where the ANFIS baseline (`_baseline_anfis.py`) tunes its premise parameters by
gradient descent, this one tunes them with a genetic algorithm: the classic
"GA-optimised fuzzy system" the dissertation names alongside ANFIS as the kind of
stochastic-search fuzzy modelling its answer-first construction is meant to
replace. Same rule structure and the same ridge-least-squares consequent solve;
only the premise search differs.

It shares scaffolding with the ANFIS module (firing in log space, LSE consequents)
so the two arms differ in exactly one thing — how the antecedent Gaussians are
placed — which is the comparison Table 4.1 is making. No new dependencies: numpy
and scikit-learn only.

This is expected to be the *slowest* arm, and that is on-message rather than a
defect: a population of fuzzy systems each re-solved every generation is precisely
the cost the single-pass MoG construction avoids. Population and generations are
kept modest so the table still finishes, and the budget is stated in the run log.

Interface (Table 4.1's `_bench` contract):
    fit_predict(X_train, y_train, X_test, *, kind, seed) -> np.ndarray
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from _baseline_anfis import (
    RULE_CAP,
    MF_PER_FEATURE,
    SCATTER_RULES,
    _ANFIS,
    _grid_partition,
    _scatter_partition,
)

POP = 20
GENERATIONS = 15
TOURNAMENT = 3
MUT_RATE = 0.3
MUT_SCALE = 0.15
ELITES = 2
FITNESS_SUBSAMPLE = 4000  # cap fitness evals at scale; full data used at predict.


def _score_individual(centres, widths, X, Y, seed):
    """Train fitness of one (centres, widths): LSE consequents, then negative MSE
    on the same data. No gradient — the GA is the only search here."""
    net = _ANFIS(centres, widths, n_outputs=Y.shape[1], seed=seed)
    wn = net._firing(X)
    phi = net._solve_consequents(X, Y, wn)
    pred = net._predict_raw(X, phi=phi)
    return -np.mean((pred - Y) ** 2), net


def _evolve(centres0, widths0, X, Y, seed):
    rng = np.random.RandomState(seed)
    R, M = centres0.shape
    # genome = concatenated (centres, widths), perturbed around the seed partition.
    dim = 2 * R * M

    def unpack(g):
        c = g[: R * M].reshape(R, M)
        s = np.maximum(g[R * M :].reshape(R, M), 1e-3)
        return c, s

    base = np.concatenate([centres0.ravel(), widths0.ravel()])
    # spread the initial population around the grid/scatter seed.
    scale = np.concatenate(
        [
            np.full(R * M, 0.3),  # centre jitter (inputs are standardised)
            np.full(R * M, 0.3),  # width jitter
        ]
    )
    pop = base[None, :] + rng.normal(0, 1, (POP, dim)) * scale[None, :]
    pop[0] = base  # keep the unperturbed seed as one individual

    best_fit, best_net = -np.inf, None
    for _ in range(GENERATIONS):
        scored = []
        for g in pop:
            c, s = unpack(g)
            fit, net = _score_individual(c, s, X, Y, seed)
            scored.append((fit, g, net))
        scored.sort(key=lambda t: t[0], reverse=True)
        if scored[0][0] > best_fit:
            best_fit, best_net = scored[0][0], scored[0][2]

        # elitism + tournament selection + BLX-alpha crossover + Gaussian mutation
        new = [scored[i][1] for i in range(ELITES)]
        elite_g = [s[1] for s in scored]
        while len(new) < POP:
            a = min(rng.choice(POP, TOURNAMENT), key=lambda i: -scored[i][0])
            b = min(rng.choice(POP, TOURNAMENT), key=lambda i: -scored[i][0])
            ga, gb = elite_g[a], elite_g[b]
            lo, hi = np.minimum(ga, gb), np.maximum(ga, gb)
            span = hi - lo
            child = rng.uniform(lo - 0.2 * span, hi + 0.2 * span)  # BLX-0.2
            mask = rng.rand(dim) < MUT_RATE
            child[mask] += rng.normal(0, MUT_SCALE, mask.sum())
            new.append(child)
        pop = np.array(new)

    return best_net


def fit_predict(X_train, y_train, X_test, *, kind, seed):
    Xtr = np.asarray(X_train, dtype=float)
    Xte = np.asarray(X_test, dtype=float)
    ytr = np.asarray(y_train)

    sx = StandardScaler().fit(Xtr)
    Xtr, Xte = sx.transform(Xtr), sx.transform(Xte)

    M = Xtr.shape[1]
    grid_rules = MF_PER_FEATURE**M if M <= 20 else np.inf
    if grid_rules <= RULE_CAP:
        centres, widths = _grid_partition(Xtr, MF_PER_FEATURE)
        mode = "grid"
    else:
        centres, widths = _scatter_partition(Xtr, SCATTER_RULES, seed)
        mode = "scatter"

    # fitness on a subsample keeps POP*GENERATIONS solves affordable at scale;
    # the winning system's consequents are refit on all of Xtr before predicting.
    if len(Xtr) > FITNESS_SUBSAMPLE:
        idx = np.random.RandomState(seed).choice(
            len(Xtr), FITNESS_SUBSAMPLE, replace=False
        )
        Xfit = Xtr[idx]
    else:
        idx = slice(None)
        Xfit = Xtr

    if kind == "reg":
        sy = StandardScaler().fit(ytr.reshape(-1, 1))
        Ytr = sy.transform(ytr.reshape(-1, 1))
        net = _evolve(centres, widths, Xfit, Ytr[idx], seed)
        wn = net._firing(Xtr)  # refit consequents on the full train set
        net._solve_consequents(Xtr, Ytr, wn)
        pred = sy.inverse_transform(net.predict(Xte)).ravel()
        print(
            f"  [gafis] {Xtr.shape[0]}x{M}: {mode}, {centres.shape[0]} rules, "
            f"pop {POP} x {GENERATIONS} gen"
        )
        return pred

    classes = np.unique(ytr)
    Y = (ytr[:, None] == classes[None, :]).astype(float)
    net = _evolve(centres, widths, Xfit, Y[idx], seed)
    wn = net._firing(Xtr)
    net._solve_consequents(Xtr, Y, wn)
    scores = net.predict(Xte)
    print(
        f"  [gafis] {Xtr.shape[0]}x{M}: {mode}, {centres.shape[0]} rules, "
        f"{len(classes)} classes, pop {POP} x {GENERATIONS} gen"
    )
    return classes[np.argmax(scores, axis=1)]


if __name__ == "__main__":
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
        f"Concrete GA-FIS (grid) test R^2 = {r2:.3f}  {'OK' if r2 > 0.6 else 'TOO LOW'}"
    )
