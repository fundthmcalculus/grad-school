"""Stage 4: fix the trapezoid fitter's endpoint defect, then vary its support.

Stage 3 found compactly supported antecedents unusable: the histogram trapezoid
fitter left 77% of concrete's test rows and 100% of bikeshare's covered by no
rule at all. This stage traces that to a defect and asks the question stage 3
could not:

> For the histogram fit operation, what if we reduced the number of buckets,
> thereby increasing the effective support?

The diagnosis first, because it changes what `n_bins` can possibly do.
`TrapezoidMembership.evaluate` rises with a strict inequality (``x > a``), so
membership is exactly **0 at x == a** -- correct for an open trapezoid. But
`fit_trapezoids_fast` sets ``a = bin_edges[0]``, the minimum of the data it was
fitted to. So the smallest observed value, and everything tied with it, gets zero
membership from the term fitted to describe it. On concrete's scaled features 55%
of rows sit exactly at FlyAsh's minimum, 44% at Slag's, 36% at
Superplasticizer's; under the ``min`` t-norm one dead feature zeroes the whole
rule. That is the 77%, and no bin count can fix it -- the left edge tracks the
data minimum at every ``n_bins``.

`overlap.pad_trapezoids` re-seats each term so its fitted range is the *plateau*
and the support extends ``trapz_pad`` times the region width beyond it. The same
hazard is already handled one module over: `regression.partition_output` nudges
``edges[0] -= 1e-9`` so the smallest value lands in bucket 0 instead of becoming
NaN.

Arms: ``trapz_bins`` x ``trapz_pad`` x ``consequent_fit``, with the Gaussian model
as the reference. Two questions, separable only because both knobs are swept:

* does padding make compact support usable at all (``pad=0`` vs ``pad>0``)?
* with the defect removed, do *fewer bins* -- wider support -- help, as predicted?

Usage
-----
    python experiments/overlap-modeling/run_trapz.py
    python experiments/overlap-modeling/run_trapz.py --quick

Writes ``outputs/trapz_results.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_experiment import (  # noqa: E402
    BUCKETS, DATASETS, DEFAULT_DATASETS, L2_REG, ORDERS, PARTITION, SEEDS,
    _r2, prepare, provenance, split3,
)
from run_local import train_fold_buckets  # noqa: E402
from overlap import OverlapTribbleRegressor  # noqa: E402

BINS = (1, 3, 5, 12, 50)
PADS = (0.0, 0.05, 0.15, 0.25, 0.5)


def arm_configs():
    out = []
    for fit in ("global", "local"):
        out.append(("gaussian", fit, f"gaussian|{fit}",
                    dict(membership="gaussian", consequent_fit=fit)))
        for bins in BINS:
            for pad in PADS:
                out.append(("trapezoid", fit, f"trapz|b{bins}|p{pad:g}|{fit}",
                            dict(membership="trapezoid", trapz_bins=bins,
                                 trapz_pad=pad, consequent_fit=fit)))
    return out


def run_cell(dataset, n_buckets, order, seed, configs):
    warnings.filterwarnings("ignore")
    X, y = DATASETS[dataset]()
    inner, val, test = split3(X, y, seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare(inner, [inner, val, test])
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
        except Exception as exc:                       # noqa: BLE001
            records.append(dict(
                dataset=dataset, n_buckets=n_buckets, order=order, seed=seed,
                shape=shape, fit=fit, label=label,
                error=f"{type(exc).__name__}: {exc}"))
            continue
        records.append(dict(
            dataset=dataset, n_buckets=n_buckets, order=order, seed=seed,
            shape=shape, fit=fit, label=label,
            trapz_bins=kwargs.get("trapz_bins"), trapz_pad=kwargs.get("trapz_pad"),
            r2_val=r2_val, r2_test=r2_test, local_r2_test=local_test,
            uncovered=cov["uncovered"], active_frac=cov["active_frac"],
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
    ap.add_argument("--out",
                    default=os.path.join(HERE, "outputs", "trapz_results.json"))
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
                   bins=list(BINS), pads=list(PADS), records=records)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=1)

    n_err = sum("error" in r for r in records)
    print(f"\n{len(records)} records in {elapsed:.1f}s ({n_err} errors) -> {args.out}")
    for r in records:
        if "error" in r:
            print(f"  {r['dataset']}/{r['label']}: {r['error']}")
            break


if __name__ == "__main__":
    main()
