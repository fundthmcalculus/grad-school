"""Stage 2: with a real per-bucket consequent solve, is the deficit the FIT or the BLEND?

Stage 1 (`run_experiment.py`) found that per-bucket ("local") consequent solving
loses badly to TRIBBLE's global firing-weighted ridge, and that overlap recovers
much of that loss without closing it. It never asked *why* the local arm loses,
because it only ever scored the blended prediction. There are two candidate
explanations and they call for opposite fixes:

**the fit** -- each rule is a poor approximator of its own region, so no way of
combining the rules can help; or

**the aggregation** -- each rule is a *good* approximator of its own region, and
the damage is done by the firing-weighted blend, which mixes rules in where they
do not apply. TRIBBLE's firing strengths come from x-space membership functions
fitted per y-bucket; they are not indicators of "this row belongs to bucket r",
so nothing guarantees that a rule fires only where it is competent.

The diagnostic that separates them is `local_r2`: R2 of each row's *own-bucket*
rule's crisp output, ignoring the blend entirely. If local solving raises
`local_r2` while lowering test R2, the fit is fine and the blend is the problem.

Arms
----
    baseline        global solve, blended prediction               [stage-1 reference]
    local-free      per-bucket solve, free intercept, blended
    local-residual  per-bucket solve with the constant pinned at the bucket
                    centroid and corrections fitted to the residual -- the
                    library's own `compute_*_order_corrections` formulation
    local-wta       per-bucket solve, winner-take-all prediction
    local-recal     per-bucket solve, per-rule affine recalibration of the blend
    local-sharp     per-bucket solve, firing strengths raised to gamma before
                    normalization -- a blend-concentration knob, gamma -> inf
                    being winner-take-all
    shrink-local    global solve with the local fit as the ridge's prior
    global-wta      global solve, winner-take-all           [is WTA good per se?]
    global-recal    global solve, recalibrated blend        [does recal help anything?]
    global-sharp    global solve, same gamma sweep          [is sharpening good per se?]

The last three are controls, one per aggregation fix. If `global-wta`,
`global-recal` or `global-sharp` improves the global solve by as much, that fix is
not telling us anything about *local* fitting -- it is just a better aggregation,
or more free parameters, and it would have to be reported as such.

`local-sharp`'s exponent is applied in the solve as well as at predict time, so
each arm is a self-consistent model rather than a mis-weighted one.

`shrink-local` runs with the antecedent overlap **off**, so it isolates the
consequent prior from the forward-pass change stage 1 already showed to be a
null (`soft-random`). Band shape is `flat` throughout -- it won 56 of 56
local-family cells in stage 1.

Usage
-----
    python experiments/overlap-modeling/run_local.py
    python experiments/overlap-modeling/run_local.py --quick

Writes ``outputs/local_results.json``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_experiment import (  # noqa: E402
    BUCKETS,
    DATASETS,
    DEFAULT_DATASETS,
    L2_REG,
    ORDERS,
    PARTITION,
    SEEDS,
    _r2,
    dump_payload,
    prepare,
    provenance,
    split3,
)
from overlap import OverlapTribbleRegressor  # noqa: E402

FRACTIONS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
# Blend-concentration exponents. gamma>1 pushes toward winner-take-all, <1 toward
# a uniform average. Swept at two overlap widths so the two knobs can be told
# apart, and mirrored on the global solve as a control.
SHARPEN = (0.5, 2.0, 4.0, 8.0)
SHARPEN_TAUS = (0.0, 0.5)


def arm_configs():
    """(arm, label, kwargs) for every stage-2 configuration, in report order."""
    out = [("baseline", "baseline", dict(consequent_fit="global", overlap=0.0))]
    families = [
        ("local-free", dict(consequent_fit="local")),
        ("local-residual", dict(consequent_fit="local-residual")),
        ("local-wta", dict(consequent_fit="local", predict_mode="wta")),
        ("local-recal", dict(consequent_fit="local", blend_recalibrate=True)),
        # Antecedents held hard on purpose -- see the module docstring.
        (
            "shrink-local",
            dict(
                consequent_fit="shrink-local",
                overlap_antecedents=False,
                overlap_means=False,
            ),
        ),
    ]
    for arm, base in families:
        for f in FRACTIONS:
            out.append(
                (arm, f"{arm}/{f:g}", dict(overlap=f, overlap_shape="flat", **base))
            )
    # Does concentrating the blend rescue a set of good local approximators?
    for tau in SHARPEN_TAUS:
        for gamma in SHARPEN:
            out.append(
                (
                    "local-sharp",
                    f"local-sharp/{tau:g}/{gamma:g}",
                    dict(
                        consequent_fit="local",
                        overlap=tau,
                        overlap_shape="flat",
                        blend_sharpen=gamma,
                    ),
                )
            )
    out.append(
        (
            "global-wta",
            "global-wta",
            dict(consequent_fit="global", overlap=0.0, predict_mode="wta"),
        )
    )
    out.append(
        (
            "global-recal",
            "global-recal",
            dict(consequent_fit="global", overlap=0.0, blend_recalibrate=True),
        )
    )
    for gamma in SHARPEN:
        out.append(
            (
                "global-sharp",
                f"global-sharp/{gamma:g}",
                dict(consequent_fit="global", overlap=0.0, blend_sharpen=gamma),
            )
        )
    return out


def train_fold_buckets(y_fit, y_other, n_buckets):
    """Assign ``y_other`` to buckets using ``y_fit``'s equal-frequency edges.

    The local-approximation diagnostic needs to know which rule is *responsible*
    for a row, which is a statement about that row's target -- so it is a
    diagnostic, not something a predictor could compute. Even so the edges come
    from the fitting fold, not from the fold being scored: re-deriving quantiles
    on the test fold would move the boundaries between arms' scores and make the
    numbers incomparable across folds.
    """
    edges = np.quantile(
        np.asarray(y_fit, dtype=float), np.linspace(0.0, 1.0, n_buckets + 1)
    )[1:-1]
    return np.digitize(np.asarray(y_other, dtype=float), edges)


def run_cell(dataset, n_buckets, order, seed, configs):
    warnings.filterwarnings("ignore")
    X, y = DATASETS[dataset]()
    inner, val, test = split3(X, y, seed)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare(inner, [inner, val, test])
    hard_tr = train_fold_buckets(ytr, ytr, n_buckets)
    hard_te = train_fold_buckets(ytr, yte, n_buckets)

    records = []
    for arm, label, kwargs in configs:
        t0 = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                model = OverlapTribbleRegressor(
                    n_output_buckets=n_buckets,
                    output_partition=PARTITION,
                    tsk_order=order,
                    l2_reg=L2_REG,
                    pin_extremes=False,
                    random_state=seed,
                    **kwargs,
                ).fit(Xtr, ytr)
                fit_s = time.perf_counter() - t0
                r2_val, drop_val = _r2(yva, model.predict(Xva))
                r2_test, drop_test = _r2(yte, model.predict(Xte))
                local_train = model.local_approximation_r2(Xtr, ytr, hard_tr)
                local_test = model.local_approximation_r2(Xte, yte, hard_te)
        except Exception as exc:  # noqa: BLE001
            records.append(
                dict(
                    dataset=dataset,
                    n_buckets=n_buckets,
                    order=order,
                    seed=seed,
                    arm=arm,
                    label=label,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        records.append(
            dict(
                dataset=dataset,
                n_buckets=n_buckets,
                order=order,
                seed=seed,
                arm=arm,
                label=label,
                r2_val=r2_val,
                r2_test=r2_test,
                local_r2_train=local_train,
                local_r2_test=local_test,
                dropped_val=drop_val,
                dropped_test=drop_test,
                fit_seconds=fit_s,
                n_rules=int(model.n_rules_),
                overlap=kwargs.get("overlap", 0.0),
            )
        )
    return records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--buckets", default=",".join(str(b) for b in BUCKETS))
    ap.add_argument("--orders", default=",".join(ORDERS))
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--out", default=os.path.join(HERE, "outputs", "local_results.json")
    )
    args = ap.parse_args()

    datasets = args.datasets.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]
    buckets = [int(b) for b in args.buckets.split(",")]
    orders = args.orders.split(",")
    if args.quick:
        datasets, seeds, buckets, orders = ["concrete"], [0, 1, 2], [5], ["2nd"]

    configs = arm_configs()
    cells = [
        (d, b, o, s) for d in datasets for b in buckets for o in orders for s in seeds
    ]
    print(
        f"{len(cells)} cells x {len(configs)} arms = {len(cells) * len(configs)} fits "
        f"on {args.jobs} workers"
    )

    from joblib import Parallel, delayed

    t0 = time.time()
    batches = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(run_cell)(d, b, o, s, configs) for d, b, o, s in cells
    )
    records = [r for batch in batches for r in batch]
    elapsed = time.time() - t0

    payload = dict(
        provenance=provenance(),
        wall_clock_seconds=elapsed,
        fractions=list(FRACTIONS),
        records=records,
    )
    written = dump_payload(payload, args.out)

    n_err = sum("error" in r for r in records)
    print(f"\n{len(records)} records in {elapsed:.1f}s ({n_err} errors) -> {written}")
    for r in records:
        if "error" in r:
            print(f"  {r['dataset']}/{r['label']}: {r['error']}")
            break


if __name__ == "__main__":
    main()
