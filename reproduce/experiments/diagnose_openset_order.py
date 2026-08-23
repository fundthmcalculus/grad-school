#!/usr/bin/env python3
"""C16: is the complement rule's order dependence floating-point accumulation?

`profile_openset_cost.py` established the *fact* -- feeding the same 82 features
in ranked vs column order changes 29% of predictions, deterministically. This
pins the *mechanism*. The anomaly membership is

    boosted = clip(class_firings + theta, 0, 1)
    mu_anom = 1 - S(boosted_1, ..., boosted_K)          # S a t-conorm

and `t_conorm(x, None)` reduces the columns with a left fold:
`z = S(S(S(0, c_0), c_1), ...)`, in column order. A t-conorm is associative in
exact arithmetic, so *reordering the same columns* must not change the answer --
unless the cause is float accumulation order. So this holds the fold, the model
and the boosted matrix fixed and only permutes the reduction order:

    column  : the order the model ships (control -- reproduces the table)
    reversed: same values, reduction reassociated
    sorted  : ascending per fold, a canonical order
    pairwise: balanced tree reduction (numpy's stable pattern)

If reversed alone moves predictions, the columns are identical and only the
association changed -- that is float accumulation order, conclusively. The size
of the move at theta=0.99 vs across the swept band is the second question.

    uv run --project tribble-fis python \
        reproduce/experiments/diagnose_openset_order.py [--held-out N]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))

CONORM = os.environ.get("REPRO_ANOM_CONORM", "hamacher")


def _fold(held_index: int):
    import _fuzzy_models as F  # noqa: F401  (ensures dataset loaders are importable)
    from sklearn.model_selection import train_test_split

    import _fuzzy_models as FM

    got = FM.load_rt_iot2022()
    if got is None:
        raise SystemExit("RT-IOT2022 unavailable")
    X, y = got
    classes = sorted(np.unique(y))
    held = classes[held_index]
    mask = np.asarray(y) != held
    Xk = X[mask].reset_index(drop=True)
    yk = pd.Series(np.asarray(y)[mask]).reset_index(drop=True)
    Xtr, Xte, ytr, _ = train_test_split(Xk, yk, test_size=0.2, random_state=0)
    return held, Xtr, ytr, Xte


def _class_firings(X, model, norms):
    """Reproduce simple_gaussian_predict's per-class rule_firing, without the
    anomaly column -- the theta-independent part."""
    from tribblefis.gauss_math import t_conorm, t_norm

    n = len(X)
    n_rules = len(model.rules)
    rf = np.ones((n, n_rules))
    for i, rule in enumerate(model.rules):
        for feature_name, mf_ids in rule.antecedents.items():
            if feature_name not in X.columns:
                continue
            matched = model.get_mfs(mf_ids)
            local = np.zeros(n)
            for mf in matched:
                local = t_conorm(
                    local, mf.evaluate(X[feature_name].values), norms.t_conorm
                )
            rf[:, i] = t_norm(local, rf[:, i], norms.t_norm)
    return rf


def _reduce(boosted, conorm, order):
    """Reduce boosted (n x K) to (n,) under a t-conorm, in a chosen order."""
    from tribblefis.gauss_math import t_conorm

    K = boosted.shape[1]
    if order == "column":
        idx = list(range(K))
    elif order == "reversed":
        idx = list(range(K - 1, -1, -1))
    elif order == "sorted":
        idx = None  # per-row sort below
    elif order == "pairwise":
        cols = [boosted[:, j].astype(float) for j in range(K)]
        while len(cols) > 1:
            nxt = []
            for a in range(0, len(cols) - 1, 2):
                nxt.append(t_conorm(cols[a], cols[a + 1], conorm))
            if len(cols) % 2:
                nxt.append(cols[-1])
            cols = nxt
        return cols[0]
    else:
        raise ValueError(order)

    if order == "sorted":
        srt = np.sort(boosted, axis=1)  # ascending, per row
        z = np.zeros(boosted.shape[0])
        for j in range(K):
            z = t_conorm(z, srt[:, j], conorm)
        return z

    z = np.zeros(boosted.shape[0])
    for j in idx:
        z = t_conorm(z, boosted[:, j].astype(float), conorm)
    return z


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", type=int, default=0)
    args = ap.parse_args()

    from tribblefis.gauss_data import AnomalyParameters, resolve_norm_pair
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        t_complement,
        take_top_features,
    )

    held, Xtr, ytr, Xte = _fold(args.held_out)
    print(f"held-out class : {held}   conorm : {CONORM}")
    print(f"fold           : train {Xtr.shape}, test {Xte.shape}")

    diffs = calculate_gaussian_correlation(Xtr, ytr)
    _, top_vars = take_top_features(diffs, top_n=len(Xtr.columns))
    memb = create_gaussian_membership_dict(Xtr, ytr, top_n_var_names=top_vars)
    params = AnomalyParameters(
        include_anomaly=True,
        threshold=0.99,
        label="anomaly",
        norm_conorm=CONORM,
        member_function="gaussian",
    )
    model = memb.to_simple_model(params)
    norms = params.norms()

    class_firing = _class_firings(Xte, model, norms)[:, :]  # (n, K) all are class rules
    labels = [r.consequent for r in model.rules] + ["anomaly"]
    K = class_firing.shape[1]
    print(f"model          : {len(model.rules)} class rules, K={K} columns to reduce")

    thetas = [0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    orders = ["column", "reversed", "sorted", "pairwise"]

    print("\n  agreement of final prediction vs the shipped 'column' order,")
    print("  and anomaly-flag rate, at each theta:\n")
    header = (
        f"    {'theta':>6} "
        + " ".join(f"{o:>10}" for o in orders)
        + f"   {'flag(col)':>10}"
    )
    print(header)
    for th in thetas:
        boosted = np.clip(class_firing + th, 0.0, 1.0)
        preds = {}
        for o in orders:
            mu = t_complement(_reduce(boosted, norms.t_conorm, o))
            rf = np.column_stack([class_firing, mu])
            preds[o] = np.array([labels[k] for k in np.argmax(rf, axis=1)])
        base = preds["column"]
        agree = {o: float(np.mean(preds[o] == base)) for o in orders}
        flag = float(np.mean(base == "anomaly"))
        row = f"    {th:>6.2f} " + " ".join(f"{agree[o]:>10.4f}" for o in orders)
        print(row + f"   {flag:>10.4f}")

    print("\n  reading: 'column' is 1.0000 by definition. Any other column below")
    print("  1.0000 is the SAME values reduced in a different association --")
    print("  i.e. float accumulation order changing the prediction.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
