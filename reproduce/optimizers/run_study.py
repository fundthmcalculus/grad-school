#!/usr/bin/env python3
"""Hot-start optimizer study: how much is left to find, and who finds it.

    uv run --project tribble-fis --with-editable tribble-opt \
        python reproduce/optimizers/run_study.py --smoke
    uv run --project tribble-fis --with-editable tribble-opt \
        python reproduce/optimizers/run_study.py

Every arm starts from the same tribble-fis fit, optimizes the same k-fold
held-out MSE inside the same box, and is stopped at the same evaluation count.
What is reported per arm:

  * **CV MSE** before and after, and the fraction of the start it removed;
  * **test R^2**, because that is what the chapters quote and because an arm can
    lower the cross-validated objective and lose on held-out data -- §6.3.5's
    finding, and the one this study is built to re-test properly;
  * **beat start**, how many seeds improved on the hot start at all;
  * **evaluations** spent and **wall-clock** seconds.

Knobs:
    --arms a,b,c        subset of arms (default: all)
    --seeds 0,1,2       seed list (default: common.SEEDS)
    --budget 2000       objective evaluations per arm per seed
    --radius 1.0        trust-region fraction; 1.0 = full parameter box
    --order 2nd         TSK consequent order
    --smoke             3 seeds, 300 evaluations -- for wiring, NOT citable
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "reproduce"))

import arms as A          # noqa: E402
import common as C        # noqa: E402
import problem as P       # noqa: E402
from budget import BudgetedObjective  # noqa: E402


def run_one(arm, dataset, seed, budget, radius, order, hp):
    """One (arm, seed) measurement. Returns a record dict, never raises."""
    prob = P.build(dataset=dataset, seed=seed, order=order, radius=radius)
    obj = BudgetedObjective(prob.fitness, max_evals=budget, x0=prob.x0).start()

    error = None
    try:
        A.run(arm, obj, prob, seed, **hp)
    except Exception as exc:  # noqa: BLE001 -- one bad arm must not kill the run
        error = f"{exc.__class__.__name__}: {exc}"

    r2, rmse = prob.score(obj.best_x)
    r2_0, rmse_0 = prob.score(prob.x0)
    return {
        "arm": arm, "dataset": dataset, "seed": seed,
        "n_params": prob.n_params,
        "cv_mse_0": obj.f0, "cv_mse": obj.best_f,
        "improvement": obj.improvement(),
        "beat_start": obj.beat_start(),
        "r2_0": r2_0, "r2": r2, "rmse_0": rmse_0, "rmse": rmse,
        "evals": obj.n_evals, "seconds": obj.seconds,
        "trace": obj.trace, "error": error,
    }


def _agg(records, key):
    vals = [r[key] for r in records if r[key] is not None and np.isfinite(r[key])]
    return C.agg(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="concrete", choices=sorted(P.DATASETS))
    ap.add_argument("--arms", default=",".join(A.ARMS))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--order", default="2nd")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.budget = args.seeds or "0,1,2", min(args.budget, 300)
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds else C.SEEDS)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in A.ARMS]
    if unknown:
        print(f"unknown arm(s): {', '.join(unknown)}; have {', '.join(A.ARMS)}",
              file=sys.stderr)
        return 2

    print(f"hot-start study: {args.dataset}, order={args.order}, "
          f"radius={args.radius}, budget={args.budget} evals/arm/seed, "
          f"seeds={seeds}")

    rows, records = [], []
    for arm in arm_names:
        per_arm = []
        for seed in seeds:
            rec = run_one(arm, args.dataset, seed, args.budget, args.radius,
                          args.order, hp={})
            per_arm.append(rec)
            records.append(rec)
            flag = "" if rec["error"] is None else f"  [{rec['error']}]"
            print(f"  {arm:<14} seed {seed}: cv {rec['cv_mse_0']:.5f} -> "
                  f"{rec['cv_mse']:.5f}   R2 {rec['r2_0']:.3f} -> {rec['r2']:.3f}   "
                  f"{rec['evals']:>5} evals  {rec['seconds']:6.1f}s{flag}")

        imp_mean, _ = _agg(per_arm, "improvement")
        beat = sum(1 for r in per_arm if r["beat_start"])
        rows.append([
            arm,
            A.HOT_START[arm],
            C.cell([r["cv_mse"] for r in per_arm], fmt="{:.5f}"),
            "—" if imp_mean is None else f"{100 * imp_mean:+.1f}%",
            C.cell([r["r2"] for r in per_arm]),
            f"{beat}/{len(per_arm)}",
            C.cell([r["evals"] for r in per_arm], fmt="{:.0f}"),
            C.cell([r["seconds"] for r in per_arm], fmt="{:.1f}"),
        ])

    ref = [r for r in records if r["arm"] == arm_names[0]]
    n_params = ref[0]["n_params"] if ref else 0
    r2_start, r2_start_sd = _agg(records, "r2_0")

    C.emit(
        "table_opt_hotstart",
        f"Optimizer study — improvement on the tribble-fis hot start "
        f"({args.dataset}, order {args.order})",
        ["arm", "hot start via", "CV MSE", "vs start", "test R²",
         "beat start", "evals", "seconds"],
        rows,
        note=(f"Every arm optimizes the same k-fold held-out MSE from the same "
              f"{n_params}-parameter hot start inside the same box, and is cut off "
              f"at exactly {args.budget} objective evaluations by a wrapper that "
              f"raises — no arm's own stopping rule is trusted to make the budgets "
              f"equal. Trust-region radius {args.radius} (1.0 = the full parameter "
              f"box from `build_param_bounds`). Heuristic start scores R² "
              f"{r2_start:.3f} ± {r2_start_sd:.3f} before any search. All arms run "
              f"single-threaded, so an optimizer that parallelises well gets no "
              f"credit here. `beat start` counts seeds where the arm improved the "
              f"objective at all."))

    _write_traces(records)
    return 0


def _write_traces(records):
    """Per-evaluation convergence traces, for the figure and for re-analysis."""
    path = os.path.join(C.OUTPUT_DIR, "table_opt_hotstart_traces.csv")
    header = ["arm", "seed", "eval", "seconds", "best_cv_mse"]
    rows = [[r["arm"], r["seed"], e, f"{s:.4f}", f"{v:.6f}"]
            for r in records for (e, s, v) in r["trace"]]
    C.write_csv(path, header, rows)
    print(f"  wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
