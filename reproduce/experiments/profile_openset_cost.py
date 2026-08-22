#!/usr/bin/env python3
"""Where `table_4_4_openset` spends 3h38m, and what can safely be cut.

Three questions, in order, because the answer to each decides whether the next
one is worth asking:

  A. COST      Which stage of one leave-one-class-out fold costs what?
  B. NECESSITY The antecedent screen ranks all 82 features and then keeps all 82,
               so the ranking looks discardable. Is it?
  C. CONTROL   If dropping it changes the answer, is that because the ORDER
               matters, or because the fold is not deterministic? Without this
               the B result means nothing.

Run on a quiet machine -- it is a wall-clock measurement, and the suite running
alongside it will show up in the numbers.

    uv run --project tribble-fis python \
        reproduce/experiments/profile_openset_cost.py [--held-out N]

Findings as measured on 2026-08-22 are written up in
`reproduce/outputs/OPENSET_COST_2026-08-22.md`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))

THRESHOLD = float(os.environ.get("REPRO_ANOM_THRESHOLD", "0.99"))
CONORM = os.environ.get("REPRO_ANOM_CONORM", "hamacher")


def _fold(held_index: int):
    import _fuzzy_models as F
    from sklearn.model_selection import train_test_split

    got = F.load_rt_iot2022()
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


def _params():
    from tribblefis.gauss_data import AnomalyParameters

    return AnomalyParameters(
        include_anomaly=True,
        threshold=THRESHOLD,
        label="anomaly",
        norm_conorm=CONORM,
        member_function="gaussian",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", type=int, default=0)
    args = ap.parse_args()

    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        simple_gaussian_predict,
        take_top_features,
    )

    held, Xtr, ytr, Xte = _fold(args.held_out)
    params = _params()
    print(f"held-out class : {held}")
    print(
        f"fold           : train {Xtr.shape}, test {Xte.shape}, "
        f"{len(np.unique(ytr))} known classes"
    )

    # --- A. cost ------------------------------------------------------------
    timings = {}
    t = time.perf_counter()
    diffs = calculate_gaussian_correlation(Xtr, ytr)
    timings["calculate_gaussian_correlation (screen)"] = time.perf_counter() - t

    t = time.perf_counter()
    _, ranked = take_top_features(diffs, top_n=len(Xtr.columns))
    timings["take_top_features"] = time.perf_counter() - t

    t = time.perf_counter()
    memb = create_gaussian_membership_dict(Xtr, ytr, top_n_var_names=ranked)
    timings["create_gaussian_membership_dict"] = time.perf_counter() - t

    t = time.perf_counter()
    model = memb.to_simple_model(params)
    timings["to_simple_model"] = time.perf_counter() - t

    t = time.perf_counter()
    pred_ranked = simple_gaussian_predict(Xte, model)
    timings["simple_gaussian_predict"] = time.perf_counter() - t

    total = sum(timings.values())
    print("\nA. Cost of one complement-rule fold")
    print(f"  {'stage':40s} {'seconds':>9s} {'share':>7s}")
    for name, v in timings.items():
        print(f"  {name:40s} {v:9.2f} {100 * v / total:6.1f}%")
    print(f"  {'TOTAL':40s} {total:9.2f}")

    m, k = Xtr.shape[1], len(np.unique(ytr))
    print(
        f"\n  screen is O(M*K^2): {m} features x {k * (k - 1) // 2} class pairs "
        f"= {m * k * (k - 1) // 2} pairwise distances per fold"
    )
    print(f"  and its ranking is then DISCARDED: top_n={m} keeps all {len(ranked)}")

    # --- B/C. necessity, with the determinism control FIRST -----------------
    def predict(order):
        mb = create_gaussian_membership_dict(Xtr, ytr, top_n_var_names=order)
        p = simple_gaussian_predict(Xte, mb.to_simple_model(params))
        return np.asarray([str(v) == "anomaly" for v in p], dtype=bool)

    plain = list(Xtr.columns)
    a1 = np.asarray([str(v) == "anomaly" for v in pred_ranked], dtype=bool)
    a2 = predict(ranked)
    b1 = predict(plain)
    b2 = predict(plain)

    print("\nC. Control before conclusion -- is the fold deterministic?")
    print(f"  ranked vs ranked (same input twice) : {(a1 == a2).mean():.4f}")
    print(f"  column vs column                    : {(b1 == b2).mean():.4f}")
    print("\nB. Does the discarded ranking change the answer?")
    print(f"  ranked vs column order              : {(a1 == b1).mean():.4f}")
    print(f"  anomaly-flag rate, ranked           : {a1.mean():.4f}")
    print(f"  anomaly-flag rate, column order     : {b1.mean():.4f}")

    deterministic = (a1 == a2).all() and (b1 == b2).all()
    order_matters = not (a1 == b1).all()
    print()
    if deterministic and order_matters:
        print("  => The fold is deterministic AND the feature ORDER changes the")
        print("     prediction. The screen cannot be skipped, and the complement")
        print("     rule is order-dependent on a list §4.3.5 describes as combined")
        print("     by a commutative, associative t-conorm.")
    elif not deterministic:
        print("  => The fold is NOT deterministic; the order comparison says nothing.")
    else:
        print(
            "  => Order does not matter; the screen is safe to skip when top_n is all."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
