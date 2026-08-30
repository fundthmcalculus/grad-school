#!/usr/bin/env python3
"""Hot-start optimizer study: how much is left to find, and who finds it.

    uv run --project tribble-fis --with-editable tribble-opt \
        python reproduce/optimizers/run_study.py --smoke
    uv run --project tribble-fis --with-editable tribble-opt \
        python reproduce/optimizers/run_study.py

The `--init fcm` and `--init classical-fcm` arms additionally need
`--with-editable tribble-cluster` on the invocation: tribble-fis#233 moved
`tribble-clustering` out of tribble-fis's dependencies into an optional extra
(nothing in `tribblefis` imports it, and as a git source with Cython extensions
it made a C toolchain a hard requirement of `uv sync`), so those arms no longer
get it for free. `clusterinit._import_fcm` says so if you forget. The other
inits -- `hot`, `cold`, `kmeans` -- are unaffected.

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

import arms as A  # noqa: E402
import common as C  # noqa: E402
import problem as P  # noqa: E402
from budget import BudgetedObjective  # noqa: E402


def checkpoints_for(budget):
    """Budgets at which to snapshot, roughly a quarter-decade apart."""
    grid = [125, 250, 500, 1000, 2000, 4000, 8000]
    return [c for c in grid if c < budget] + [budget]


def run_one(arm, dataset, seed, budget, radius, order, hp, init="hot"):
    """One (arm, seed, init) measurement. Returns a record dict, never raises."""
    prob = P.build(dataset=dataset, seed=seed, order=order, radius=radius, init=init)
    obj = BudgetedObjective(
        prob.fitness, max_evals=budget, x0=prob.x0, checkpoints=checkpoints_for(budget)
    ).start()

    error = None
    try:
        A.run(arm, obj, prob, seed, **hp)
    except Exception as exc:  # noqa: BLE001 -- one bad arm must not kill the run
        error = f"{exc.__class__.__name__}: {exc}"
    obj.finalize()

    r2, rmse = prob.score(obj.best_x)
    r2_0, rmse_0 = prob.score(prob.x0)
    # The budget curve: what this arm would have delivered had it been stopped
    # earlier. Scoring is cheap next to the search, so the whole curve comes out
    # of the one run rather than out of one run per budget.
    curve = []
    for cp, (x_cp, f_cp, secs) in sorted(obj.snapshots.items()):
        r2_cp, _ = prob.score(x_cp) if x_cp is not None else (float("nan"),) * 2
        curve.append((cp, f_cp, r2_cp, secs))
    # The headline for a cold run: how many evaluations it needed to reach what
    # the Gaussian construction hands over for free. `None` means "never, within
    # this budget", which is itself the answer and must not read as missing data.
    evals_to_heuristic = None
    for n, _secs, value in obj.trace:
        if value <= prob.heuristic_cv:
            evals_to_heuristic = n
            break

    return {
        "curve": curve,
        "init": init,
        "heuristic_cv": prob.heuristic_cv,
        "heuristic_r2": prob.heuristic_r2,
        "evals_to_heuristic": evals_to_heuristic,
        "arm": arm,
        "dataset": dataset,
        "seed": seed,
        "n_params": prob.n_params,
        "cv_mse_0": obj.f0,
        "cv_mse": obj.best_f,
        "improvement": obj.improvement(),
        "beat_start": obj.beat_start(),
        "r2_0": r2_0,
        "r2": r2,
        "rmse_0": rmse_0,
        "rmse": rmse,
        "evals": obj.n_evals,
        "seconds": obj.seconds,
        "init_seconds": prob.meta.get("init_seconds", 0.0),
        "construction_seconds": prob.meta.get("construction_seconds", 0.0),
        "trace": obj.trace,
        "error": error,
    }


STARTED_FROM = {
    "hot": "Gaussian construction",
    "cold": "random point in the box",
    "kmeans": "1-D k-means per (feature, bucket)",
    "fcm": "1-D fuzzy c-means",
}


def _agg(records, key):
    vals = [r[key] for r in records if r[key] is not None and np.isfinite(r[key])]
    return C.agg(vals)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="concrete", choices=sorted(P.DATASETS))
    ap.add_argument("--arms", default=",".join(A.ARMS))
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--order", default="2nd")
    ap.add_argument(
        "--init", default="hot", help="comma-separated: hot, cold, kmeans, fcm"
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--archive",
        metavar="LABEL",
        help="copy the outputs into reproduce/outputs/<LABEL>/ with a "
        "PROVENANCE.txt — labelled archives are tracked, loose "
        "files in outputs/ are scratch",
    )
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.budget = args.seeds or "0,1,2", min(args.budget, 300)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else C.SEEDS
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arm_names if a not in A.ARMS]
    if unknown:
        print(
            f"unknown arm(s): {', '.join(unknown)}; have {', '.join(A.ARMS)}",
            file=sys.stderr,
        )
        return 2

    print(
        f"hot-start study: {args.dataset}, order={args.order}, "
        f"radius={args.radius}, budget={args.budget} evals/arm/seed, "
        f"seeds={seeds}"
    )

    inits = [i.strip() for i in args.init.split(",") if i.strip()]
    rows, records = [], []
    for init in inits:
        for arm in arm_names:
            per_arm = []
            for seed in seeds:
                rec = run_one(
                    arm,
                    args.dataset,
                    seed,
                    args.budget,
                    args.radius,
                    args.order,
                    hp={},
                    init=init,
                )
                per_arm.append(rec)
                records.append(rec)
                flag = "" if rec["error"] is None else f"  [{rec['error']}]"
                reach = (
                    ""
                    if rec["evals_to_heuristic"] is None
                    else f"  reached heuristic @{rec['evals_to_heuristic']}"
                )
                print(
                    f"  [{init}] {arm:<14} seed {seed}: cv {rec['cv_mse_0']:.5f} -> "
                    f"{rec['cv_mse']:.5f}   R2 {rec['r2_0']:.3f} -> {rec['r2']:.3f}   "
                    f"{rec['evals']:>5} evals{reach}{flag}"
                )

            imp_mean, _ = _agg(per_arm, "improvement")
            beat = sum(1 for r in per_arm if r["beat_start"])
            # Paired, per seed. Every arm faced the identical problem at each seed,
            # so the seed-to-seed spread of the *start* (0.755-0.872 in R^2 here) is
            # common to all of them and swamps the between-arm differences when the
            # columns are compared as independent means. The paired delta removes it,
            # and it is the statistic the ordering should be read from.
            # Paired against the HEURISTIC in both modes, not against each run's own
            # start. A cold run's own start is a random point, so "improvement on the
            # start" flatters it enormously and means nothing; the question is
            # whether it caught up with the construction.
            d_r2 = [
                r["r2"] - r["heuristic_r2"]
                for r in per_arm
                if np.isfinite(r["r2"]) and np.isfinite(r["heuristic_r2"])
            ]
            won = sum(1 for d in d_r2 if d > 0)
            reached = [
                r["evals_to_heuristic"]
                for r in per_arm
                if r["evals_to_heuristic"] is not None
            ]
            rows.append(
                [
                    init,
                    arm,
                    STARTED_FROM.get(init, init),
                    C.cell([r["cv_mse"] for r in per_arm], fmt="{:.5f}"),
                    C.cell([r["r2"] for r in per_arm]),
                    C.cell(d_r2, fmt="{:+.3f}") if d_r2 else C.NA,
                    f"{won}/{len(d_r2)}",
                    (
                        C.cell(reached, fmt="{:.0f}")
                        if len(reached) == len(per_arm)
                        else (
                            f"{len(reached)}/{len(per_arm)} seeds"
                            if reached
                            else "never"
                        )
                    ),
                    f"{beat}/{len(per_arm)}",
                    C.cell([r["evals"] for r in per_arm], fmt="{:.0f}"),
                    C.cell(
                        [
                            1000 * (r["construction_seconds"] + r["init_seconds"])
                            for r in per_arm
                        ],
                        fmt="{:.0f}",
                    ),
                ]
            )

    ref = [r for r in records if r["arm"] == arm_names[0]]
    n_params = ref[0]["n_params"] if ref else 0
    r2_start, r2_start_sd = _agg(records, "heuristic_r2")

    C.emit(
        "table_opt_hotstart",
        f"Optimizer study — the Gaussian construction as a starting point "
        f"({args.dataset}, order {args.order})",
        [
            "init",
            "arm",
            "started from",
            "CV MSE",
            "test R²",
            "Δ R² vs heuristic (paired)",
            "R² wins",
            "evals to reach heuristic",
            "beat own start",
            "evals",
            "start-up cost (ms)",
        ],
        rows,
        note=(
            f"Every arm optimizes the same k-fold held-out MSE inside the same "
            f"box over the same {n_params} antecedent parameters, and is cut off "
            f"at exactly {args.budget} objective evaluations by a wrapper that "
            f"raises — no arm's own stopping rule is trusted to make the budgets "
            f"equal. **The budget is evaluations, not time.** Wall-clock is kept "
            f"per seed in the companion CSV but is not a variable this study "
            f"controls, and every arm runs single-threaded, so parallelism is "
            f"deliberately out of scope. `init=hot` starts from the Gaussian "
            f"construction's own antecedents; `init=cold` from a uniform random "
            f"point in the same box, everything else identical. Both are scored "
            f"against the SAME reference — the heuristic model, R² "
            f"{r2_start:.3f} ± {r2_start_sd:.3f} — because a cold run's own start "
            f"is a random point and improvement on it means nothing. `evals to "
            f"reach heuristic` is how many evaluations a run needed before its "
            f"objective matched what the construction supplies for free: the "
            f"price of not having it, and the number this study exists to "
            f"produce. Trust-region radius {args.radius} (1.0 = the full box from "
            f"`build_param_bounds`), centred on whichever point that run starts "
            f"from.\n>\n"
            f"> **`start-up cost` is not a pipeline comparison, and cannot be "
            f"read as one.** Every init in this table needs the Gaussian "
            f"construction first: it supplies the structure (which features, how "
            f"many components per bucket) and the box that `build_param_bounds` "
            f"derives. The k-means and FCM inits then *replace the placement* "
            f"inside that structure, so their cost is construction + clustering "
            f"and they are strictly more expensive than the construction alone. "
            f"Showing k-means as a cheaper alternative would require the "
            f"classical joint-space identification, where clustering chooses the "
            f"rules instead of inheriting them — that changes the structure and "
            f"belongs with `run_structure_study.py`."
        ),
    )

    _write_traces(records)
    _write_seeds(records)
    _write_curve(records)
    if args.archive:
        _archive(args.archive, args, seeds, arm_names, n_params)
    return 0


ARTIFACTS = [
    "table_opt_hotstart.md",
    "table_opt_hotstart.csv",
    "table_opt_hotstart_seeds.csv",
    "table_opt_hotstart_traces.csv",
    "table_opt_hotstart_budget.csv",
]


def _archive(label, args, seeds, arm_names, n_params=0):
    """Copy this run's artifacts under a label, with the provenance to read them.

    Loose files in `reproduce/outputs/` are scratch by policy — whatever ran
    last, untracked. A labelled directory is tracked, which is what makes a
    result quotable: the evidence survives the machine that produced it, and a
    later run can be diffed against it. Same convention as
    `run_all_tables.sh <label>`, and this writes the same kind of
    `PROVENANCE.txt` so `harness_data` can find it as an archive.
    """
    import shutil
    import subprocess
    import time

    dest = os.path.join(C.OUTPUT_DIR, label)
    os.makedirs(dest, exist_ok=True)

    def sha(path):
        try:
            return subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    lines = [
        f"label:       {label}",
        f"generated:   {stamp}",
        f"tribble-fis: {sha(os.path.join(ROOT, 'tribble-fis'))}",
        f"tribble-opt: {sha(os.path.join(ROOT, 'tribble-opt'))}",
        f"grad-school: {sha(ROOT)}",
        f"seeds:       {','.join(map(str, seeds))}",
        "",
        "study:       reproduce/optimizers/run_study.py",
        f"dataset:     {args.dataset}",
        f"order:       {args.order}",
        f"radius:      {args.radius}",
        f"budget:      {args.budget} objective evaluations per arm per seed",
        # Recorded because it is not a constant: the parameter count follows the
        # model the construction produces, so a library change moves it (144
        # before the identification fix, 136 after). Anything that converts
        # evaluations into generations needs it -- scipy's differential evolution
        # takes `popsize` as a MULTIPLIER of the dimension, so its population is
        # popsize x this number.
        f"params:      {n_params} antecedent parameters",
        f"arms:        {','.join(arm_names)}",
        f"init:        {args.init}",
        "",
        C.machine_block().strip(),
        "",
    ]
    with open(os.path.join(dest, "PROVENANCE.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    copied = 0
    for name in ARTIFACTS:
        src = os.path.join(C.OUTPUT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
            copied += 1
    print(f"  archived {copied} artifact(s) -> " f"{os.path.relpath(dest, ROOT)}")


def _write_curve(records):
    """Held-out R² as a function of the evaluation budget, per arm and seed."""
    path = os.path.join(C.OUTPUT_DIR, "table_opt_hotstart_budget.csv")
    header = [
        "arm",
        "init",
        "seed",
        "budget",
        "cv_mse",
        "r2",
        "r2_0",
        "heuristic_r2",
        "seconds",
    ]
    rows = [
        [
            r["arm"],
            r["init"],
            r["seed"],
            cp,
            f"{f:.6f}",
            f"{r2:.6f}",
            f"{r['r2_0']:.6f}",
            f"{r['heuristic_r2']:.6f}",
            f"{secs:.2f}",
        ]
        for r in records
        for (cp, f, r2, secs) in r["curve"]
    ]
    C.write_csv(path, header, rows)
    print(f"  wrote {path}")


def _write_seeds(records):
    """Per-(arm, seed) records — the paired analysis reads these.

    Without this file the only way to recover a paired comparison is to parse
    the run log, which is not an artifact anyone should be quoting from.
    """
    path = os.path.join(C.OUTPUT_DIR, "table_opt_hotstart_seeds.csv")
    header = [
        "arm",
        "init",
        "seed",
        "cv_mse_0",
        "cv_mse",
        "improvement",
        "beat_start",
        "r2_0",
        "r2",
        "heuristic_r2",
        "heuristic_cv",
        "evals_to_heuristic",
        "rmse_0",
        "rmse",
        "evals",
        "seconds",
        "init_seconds",
        "construction_seconds",
        "error",
    ]
    rows = [
        [
            r["arm"],
            r["init"],
            r["seed"],
            f"{r['cv_mse_0']:.6f}",
            f"{r['cv_mse']:.6f}",
            "" if r["improvement"] is None else f"{r['improvement']:.6f}",
            int(r["beat_start"]),
            f"{r['r2_0']:.6f}",
            f"{r['r2']:.6f}",
            f"{r['heuristic_r2']:.6f}",
            f"{r['heuristic_cv']:.6f}",
            "" if r["evals_to_heuristic"] is None else r["evals_to_heuristic"],
            f"{r['rmse_0']:.4f}",
            f"{r['rmse']:.4f}",
            r["evals"],
            f"{r['seconds']:.2f}",
            f"{r['init_seconds']:.4f}",
            f"{r['construction_seconds']:.4f}",
            r["error"] or "",
        ]
        for r in records
    ]
    C.write_csv(path, header, rows)
    print(f"  wrote {path}")


def _write_traces(records):
    """Per-evaluation convergence traces, for the figure and for re-analysis."""
    path = os.path.join(C.OUTPUT_DIR, "table_opt_hotstart_traces.csv")
    header = ["arm", "init", "seed", "eval", "seconds", "best_cv_mse"]
    rows = [
        [r["arm"], r["init"], r["seed"], e, f"{s:.4f}", f"{v:.6f}"]
        for r in records
        for (e, s, v) in r["trace"]
    ]
    C.write_csv(path, header, rows)
    print(f"  wrote {path}")


if __name__ == "__main__":
    sys.exit(main())
