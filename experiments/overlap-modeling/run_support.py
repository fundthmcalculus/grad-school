"""Stage 3: does COMPACT support rescue per-bucket local consequents?

Stage 2 found that per-bucket consequent solving makes every rule a much better
approximator of its own region (local R2 0.70 -> 0.95 on concrete) while making
the blended model worse. The counterpoint this stage tests:

> We have full support everywhere because the membership functions are Gaussian,
> which have effectively infinite support. What if we switched to trapezoids, or
> applied a non-linear clamp to zero at 2.75-3 sd?

That is the right objection. A Gaussian is strictly positive everywhere, so every
rule fires -- however faintly -- at every point, and a local model gets blended in
a long way from any data it was fitted on. Compact support removes that by
construction.

It also introduces the opposite failure, which is why `coverage` is recorded for
every arm: a compactly supported rule set can leave points that **no** rule
covers. `_normalize_firing_strengths` returns an all-zero row there and the model
predicts exactly 0 -- a finite number, so it passes every NaN filter and lands in
the R2 as a large error with nothing to attribute it to.

Arms
----
    gaussian                  infinite support                    [reference]
    clamped/k (smooth)        Gaussian zeroed past k sd, meeting the axis
                              continuously -- the "non-linear clamp"
    clamped/k (hard)          the same with a plain truncation, so the cost of the
                              discontinuity is priced rather than assumed
    trapezoid                 the library's fast histogram fitter
    ruspini/tol               each feature re-expressed as a shared triangular
                              partition of unity

crossed with ``consequent_fit in {global, local}`` and ``overlap in {0, 0.5}``.
The local arm is the one the counterpoint is about: if compact support is what
per-bucket fitting was missing, `local` should close on `global` as k falls --
right up until coverage collapses.

Usage
-----
    python experiments/overlap-modeling/run_support.py
    python experiments/overlap-modeling/run_support.py --quick

Writes ``outputs/support_results.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_experiment import (  # noqa: E402
    BUCKETS, DATASETS, DEFAULT_DATASETS, L2_REG, ORDERS, PARTITION, SEEDS,
    _r2, dump_payload, prepare, provenance, split3,
)
from run_local import train_fold_buckets  # noqa: E402
from overlap import OverlapTribbleRegressor  # noqa: E402

# 2.75 and 3.0 are the values the counterpoint named; the rest bracket them so the
# coverage cliff can be located rather than guessed at.
CLAMP_KS = (2.0, 2.5, 2.75, 3.0, 3.5, 4.0)
HARD_KS = (2.75, 3.0)          # priced against their smooth twins only where it matters
RUSPINI_TOLS = (0.02, 0.05)
TAUS = (0.0, 0.5)


def membership_configs():
    """(name, kwargs) for every antecedent shape under test."""
    out = [("gaussian", dict(membership="gaussian"))]
    for k in CLAMP_KS:
        out.append((f"clamped-smooth/{k:g}",
                    dict(membership="clamped", clamp_k=k, clamp_smooth=True)))
    for k in HARD_KS:
        out.append((f"clamped-hard/{k:g}",
                    dict(membership="clamped", clamp_k=k, clamp_smooth=False)))
    out.append(("trapezoid", dict(membership="trapezoid")))
    for tol in RUSPINI_TOLS:
        out.append((f"ruspini/{tol:g}", dict(membership="ruspini", ruspini_tol=tol)))
    return out


def arm_configs():
    out = []
    for shape, kw in membership_configs():
        for fit in ("global", "local"):
            for tau in TAUS:
                out.append((shape, fit, f"{shape}|{fit}|t{tau:g}",
                            dict(consequent_fit=fit, overlap=tau,
                                 overlap_shape="flat", **kw)))
    return out


def run_cell(dataset, n_buckets, order, seed, configs):
    warnings.filterwarnings("ignore")
    X, y = DATASETS[dataset]()
    inner, val, test = split3(X, y, seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare(inner, [inner, val, test])
    hard_tr = train_fold_buckets(ytr, ytr, n_buckets)
    hard_te = train_fold_buckets(ytr, yte, n_buckets)

    records = []
    for shape, fit, label, kwargs in configs:
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = OverlapTribbleRegressor(
                    n_output_buckets=n_buckets, output_partition=PARTITION,
                    tsk_order=order, l2_reg=L2_REG, pin_extremes=False,
                    random_state=seed, **kwargs).fit(Xtr, ytr)
                fit_s = time.perf_counter() - t0
                r2_val, drop_val = _r2(yva, model.predict(Xva))
                r2_test, drop_test = _r2(yte, model.predict(Xte))
                cov = model.coverage(Xte)
                local_test = model.local_approximation_r2(Xte, yte, hard_te)
                local_train = model.local_approximation_r2(Xtr, ytr, hard_tr)
        except Exception as exc:                       # noqa: BLE001
            records.append(dict(
                dataset=dataset, n_buckets=n_buckets, order=order, seed=seed,
                shape=shape, fit=fit, label=label,
                error=f"{type(exc).__name__}: {exc}"))
            continue
        records.append(dict(
            dataset=dataset, n_buckets=n_buckets, order=order, seed=seed,
            shape=shape, fit=fit, label=label, overlap=kwargs["overlap"],
            r2_val=r2_val, r2_test=r2_test,
            local_r2_train=local_train, local_r2_test=local_test,
            uncovered=cov["uncovered"], mean_active=cov["mean_active"],
            active_frac=cov["active_frac"],
            dropped_val=drop_val, dropped_test=drop_test, fit_seconds=fit_s,
            n_rules=int(model.n_rules_)))
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--buckets", default=",".join(str(b) for b in BUCKETS))
    ap.add_argument("--orders", default=",".join(ORDERS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "outputs",
                                                  "support_results.json"))
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    buckets = [int(b) for b in args.buckets.split(",")]
    orders = args.orders.split(",")
    if args.quick:
        datasets, seeds, buckets, orders = ["concrete"], [0, 1, 2], [5], ["2nd"]

    configs = arm_configs()
    cells = [(d, b, o, s) for d in datasets for b in buckets for o in orders for s in seeds]
    print(f"{len(cells)} cells x {len(configs)} arms = {len(cells) * len(configs)} fits "
          f"on {args.jobs} workers")

    from joblib import Parallel, delayed
    t0 = time.time()
    batches = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_cell)(d, b, o, s, configs) for d, b, o, s in cells)
    records = [r for batch in batches for r in batch]
    elapsed = time.time() - t0

    payload = dict(provenance=provenance(), wall_clock_seconds=elapsed,
                   clamp_ks=list(CLAMP_KS), hard_ks=list(HARD_KS),
                   ruspini_tols=list(RUSPINI_TOLS), taus=list(TAUS),
                   records=records)
    written = dump_payload(payload, args.out)

    n_err = sum("error" in r for r in records)
    print(f"\n{len(records)} records in {elapsed:.1f}s ({n_err} errors) -> {written}")
    for r in records:
        if "error" in r:
            print(f"  {r['dataset']}/{r['label']}: {r['error']}")
            break


if __name__ == "__main__":
    main()
