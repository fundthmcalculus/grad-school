#!/usr/bin/env python3
"""Optimizer study on PhiUSIIL: what is left to find on a classification problem.

    uv run --project tribble-fis --with-editable tribble-opt \
        --with-editable tribble-cluster \
        python reproduce/optimizers/run_phishing_opt_study.py --smoke
    uv run --project tribble-fis --with-editable tribble-opt \
        --with-editable tribble-cluster \
        python reproduce/optimizers/run_phishing_opt_study.py --archive <label>

The sibling of `run_study.py`, which does the same thing on Concrete regression.
Every arm starts from the same construction, optimizes the same shipped
classifier objective inside the same box, and is stopped at the same evaluation
count. `problem_cls.py` says what that objective is and why it is imported
rather than written here.

## Time is a reported result here, not a by-product

The Concrete study deliberately demotes wall-clock: its question is how many
*iterations* an optimizer needs, and seconds are machine-dependent. This study
keeps that axis and adds an explicit timing one, because on a classification
problem the interesting claim is a cost claim — the construction hands over a
working rule base in milliseconds, and the question is what a search costs to
match it.

Three timing numbers, kept separate because they answer different questions:

  construction (ms)     what the Gaussian construction itself costs. The number
                        the "how much faster" claim rests on.
  feature engineering   the O(M^2) screen. Shared by every route, charged to
                        none of them, reported because it is real.
  seconds to match      wall-clock before a COLD search first reaches the
                        construction's own objective value. This is the direct
                        comparison: the same quantity, in the same units, for
                        the two ways of getting a rule base.

Ratios are the portable part; absolute seconds are not, and the machine is
recorded. Everything is single-threaded so an arm cannot buy time with cores.

## Two sample sizes, set independently

`--train-rows` sets the cost of an objective evaluation; `--test-rows` sets the
resolution of the accuracy column. PhiUSIIL is saturated — the construction makes
two or three errors in ten thousand — so those pull in opposite directions and are
sized separately. `problem_cls.py` explains at more length, and the accuracy
column is reported with **error counts** beside it, because a rate standing on two
events invites over-reading.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

# The optimizers package draws a tqdm bar per generation. Across 100+ arm-runs
# that is thousands of carriage-return updates in the log, which buries the one
# line per run that carries the result.
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "reproduce"))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))

import arms as A  # noqa: E402
import common as C  # noqa: E402
import problem_cls as PC  # noqa: E402
from budget import BudgetedObjective  # noqa: E402

CHECKPOINTS = (125, 250, 500, 1000, 2000)

STARTED_FROM = {
    "hot": "Gaussian construction",
    "cold": "random point in the box",
    "classical-kmeans": "k-means within each class",
    "classical-fcm": "fuzzy c-means within each class",
}


def _one(arm, init, seed, args):
    prob = PC.build(seed=seed, radius=args.radius, n_train=args.train_rows,
                    n_test=args.test_rows, top_n=args.top_n, init=init,
                    components=args.components)
    checkpoints = tuple(c for c in CHECKPOINTS if c <= args.budget) or (args.budget,)
    # `.start()` is not optional: it begins the clock and scores x0 off-budget,
    # which is what fills `f0` and `best_x`. Without it every arm reports a
    # 0-dimensional best point and no starting objective.
    obj = BudgetedObjective(prob.fitness, args.budget, x0=prob.x0,
                            checkpoints=checkpoints).start()
    error = None
    try:
        A.run(arm, obj, prob, seed)
    except Exception as exc:  # noqa: BLE001 -- one bad arm must not kill the run
        error = f"{exc.__class__.__name__}: {exc}"
    obj.finalize()

    acc, err = prob.score(obj.best_x)
    acc_0, err_0 = prob.score(prob.x0)

    curve = []
    for cp, (x_cp, f_cp, secs) in sorted(obj.snapshots.items()):
        a_cp, _ = prob.score(x_cp) if x_cp is not None else (float("nan"),) * 2
        curve.append((cp, f_cp, a_cp, secs))

    # Evaluations AND seconds before the search first reached what the
    # construction supplies for free. `None` means "never, inside this budget",
    # which is an answer and must not read as missing data.
    evals_to_heuristic = seconds_to_heuristic = None
    for n, secs, value in obj.trace:
        if value <= prob.heuristic_obj:
            evals_to_heuristic, seconds_to_heuristic = n, secs
            break

    return {
        "arm": arm, "init": init, "seed": seed, "curve": curve,
        "n_params": prob.n_params, "n_mfs": prob.meta["n_mfs"],
        "n_train": prob.meta["n_train"],
        "heuristic_obj": prob.heuristic_obj,
        "heuristic_acc": prob.heuristic_score,
        "evals_to_heuristic": evals_to_heuristic,
        "seconds_to_heuristic": seconds_to_heuristic,
        "obj_0": obj.f0, "obj": obj.best_f,
        "improvement": obj.improvement(), "beat_start": obj.beat_start(),
        "acc_0": acc_0, "acc": acc, "err_0": err_0, "err": err,
        "n_test": prob.meta["n_test"],
        # The count, not just the rate. Two errors in 48,000 is a rate of
        # 0.00004 and reads as a precise measurement; as "2 errors" it reads as
        # what it is.
        "errors": err * prob.meta["n_test"],
        "errors_0": err_0 * prob.meta["n_test"],
        "evals": obj.n_evals, "seconds": obj.seconds,
        "construction_seconds": prob.meta["construction_seconds"],
        "screen_seconds": prob.meta["screen_seconds"],
        "init_seconds": prob.meta["init_seconds"],
        "trace": obj.trace, "error": error,
    }


def _write_seeds(records):
    path = os.path.join(C.OUTPUT_DIR, "table_opt_phishing_seeds.csv")
    C.write_csv(path,
                ["arm", "init", "seed", "n_train", "n_params", "n_mfs",
                 "obj_0", "obj", "improvement", "beat_start",
                 "acc_0", "acc", "heuristic_acc", "heuristic_obj",
                 "evals_to_heuristic", "seconds_to_heuristic",
                 "evals", "seconds", "construction_seconds", "screen_seconds",
                 "init_seconds", "error"],
                [[r["arm"], r["init"], r["seed"], r["n_train"], r["n_params"],
                  r["n_mfs"], f"{r['obj_0']:.6f}", f"{r['obj']:.6f}",
                  f"{r['improvement']:.6f}", int(r["beat_start"]),
                  f"{r['acc_0']:.6f}", f"{r['acc']:.6f}",
                  f"{r['heuristic_acc']:.6f}", f"{r['heuristic_obj']:.6f}",
                  "" if r["evals_to_heuristic"] is None else r["evals_to_heuristic"],
                  "" if r["seconds_to_heuristic"] is None
                  else f"{r['seconds_to_heuristic']:.4f}",
                  r["evals"], f"{r['seconds']:.3f}",
                  f"{r['construction_seconds']:.6f}",
                  f"{r['screen_seconds']:.6f}", f"{r['init_seconds']:.6f}",
                  r["error"] or ""]
                 for r in records])
    return path


def _write_traces(records):
    path = os.path.join(C.OUTPUT_DIR, "table_opt_phishing_traces.csv")
    rows = []
    for r in records:
        for n, secs, value in r["trace"]:
            rows.append([r["arm"], r["init"], r["seed"], n, f"{secs:.4f}",
                         f"{value:.6f}"])
    C.write_csv(path, ["arm", "init", "seed", "eval", "seconds", "best_obj"], rows)
    return path


def _write_curve(records):
    path = os.path.join(C.OUTPUT_DIR, "table_opt_phishing_budget.csv")
    rows = []
    for r in records:
        for cp, f_cp, a_cp, secs in r["curve"]:
            rows.append([r["arm"], r["init"], r["seed"], cp, f"{f_cp:.6f}",
                         f"{a_cp:.6f}", f"{r['acc_0']:.6f}",
                         f"{r['heuristic_acc']:.6f}", f"{secs:.3f}"])
    C.write_csv(path, ["arm", "init", "seed", "budget", "obj", "acc", "acc_0",
                       "heuristic_acc", "seconds"], rows)
    return path


def _agg(records, key):
    vals = [r[key] for r in records
            if r.get(key) is not None and np.isfinite(r[key])]
    return C.agg(vals)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", default=",".join(A.ARMS))
    ap.add_argument("--init", default="hot,cold",
                    help="comma-separated: hot, cold, classical-kmeans, classical-fcm")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--train-rows", type=int, default=16_000,
                    help="training rows; sets the cost of an objective "
                         "evaluation. See problem_cls.py on why this is capped.")
    ap.add_argument("--test-rows", type=int, default=48_000,
                    help="test rows, sized independently: PhiUSIIL is saturated, "
                         "so a small test set cannot resolve the accuracy column")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--components", type=int, default=None,
                    help="pin components per (feature, class); default lets the "
                         "construction choose by BIC")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--archive", metavar="LABEL")
    args = ap.parse_args()

    if args.smoke:
        args.seeds, args.budget = "0", min(args.budget, 200)
        args.train_rows, args.test_rows = 4_000, 12_000

    seeds = [int(s) for s in args.seeds.split(",")]
    C.SEEDS = seeds
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    inits = [i.strip() for i in args.init.split(",") if i.strip()]
    unknown = [a for a in arm_names if a not in A.ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}; have {sorted(A.ARMS)}")

    print(f"phishing optimizer study: train={args.train_rows}, "
          f"test={args.test_rows}, top_n={args.top_n}, "
          f"budget={args.budget}, radius={args.radius}, seeds={seeds}, "
          f"arms={arm_names}, init={inits}, threads=1")

    records = []
    wall0 = time.perf_counter()
    for init in inits:
        for seed in seeds:
            for arm in arm_names:
                rec = _one(arm, init, seed, args)
                records.append(rec)
                _write_seeds(records)
                match = ("—" if rec["seconds_to_heuristic"] is None
                         else f"{rec['seconds_to_heuristic']:.1f}s"
                              f" @{rec['evals_to_heuristic']}")
                print(f"  [{init}] {arm:<14} seed {seed}: "
                      f"obj {rec['obj_0']:.5f} -> {rec['obj']:.5f}   "
                      f"acc {rec['acc_0']:.4f} -> {rec['acc']:.4f}   "
                      f"{rec['seconds']:6.1f}s / {rec['evals']} evals   "
                      f"match {match}"
                      + (f"   ERROR {rec['error']}" if rec["error"] else ""))
    print(f"  total wall-clock: {(time.perf_counter() - wall0) / 60:.1f} min")

    _write_traces(records)
    _write_seeds(records)
    _write_curve(records)
    _emit(records, args, seeds, inits, arm_names)
    if args.archive:
        _archive(args.archive, args, seeds, inits, arm_names)
    return 0


ARTIFACTS = ["table_opt_phishing.md", "table_opt_phishing.csv",
             "table_opt_phishing_timing.md", "table_opt_phishing_timing.csv",
             "table_opt_phishing_seeds.csv", "table_opt_phishing_traces.csv",
             "table_opt_phishing_budget.csv"]


def _emit(records, args, seeds, inits, arm_names):
    ref = [r for r in records if r["arm"] == "none" and r["init"] == "hot"]
    n_params = ref[0]["n_params"] if ref else 0
    n_mfs = ref[0]["n_mfs"] if ref else 0
    n_train = ref[0]["n_train"] if ref else 0

    rows = []
    for init in inits:
        for arm in arm_names:
            sel = [r for r in records if r["arm"] == arm and r["init"] == init]
            if not sel:
                continue
            paired = [r["acc"] - r["heuristic_acc"] for r in sel]
            wins = sum(1 for d in paired if d > 0)
            reached = [r for r in sel if r["seconds_to_heuristic"] is not None]
            rows.append([
                init, arm, STARTED_FROM.get(init, init),
                C.cell([r["obj"] for r in sel], fmt="{:.5f}"),
                C.cell([r["acc"] for r in sel], fmt="{:.4f}"),
                C.cell([r["errors"] for r in sel], fmt="{:.0f}"),
                C.cell(paired, fmt="{:+.4f}"),
                f"{wins}/{len(sel)}",
                (C.cell([r["seconds_to_heuristic"] for r in reached], fmt="{:.1f}")
                 + (f" ({len(reached)}/{len(sel)})" if len(reached) < len(sel) else "")
                 if reached else "never"),
                C.cell([r["seconds"] for r in sel], fmt="{:.1f}"),
            ])

    C.emit(
        "table_opt_phishing",
        f"Optimizer study on PhiUSIIL — the construction as a starting point "
        f"({n_train:,} training rows, {args.top_n} features)",
        ["init", "arm", "started from", "objective", "test accuracy",
         "test errors", "Δ accuracy vs construction (paired)", "acc wins",
         "seconds to match the construction", "seconds spent"],
        rows,
        note=(f"Binary classification, so the construction gives **one rule per "
              f"class** and the rule count is not a free parameter; what is free "
              f"is the {n_mfs} membership functions' placement, i.e. "
              f"{n_params} antecedent parameters. The objective is the shipped "
              f"classifier fitness — `refine._make_classifier_fitness`, training "
              f"cross-entropy plus a ridge shrink toward each arm's own `x0` at "
              f"`l2_shrink={PC.L2_SHRINK}` — imported rather than reimplemented, "
              f"because an optimizer measured against a target the shipped code "
              f"does not use is measuring nothing anybody runs. **It is a "
              f"training loss**, so `test accuracy` is the only outcome column to "
              f"quote and the gap between the two is the point. Budget: "
              f"**{args.budget} objective evaluations** per arm per seed, "
              f"enforced by a wrapper that raises. Trust-region radius "
              f"{args.radius} ({'the full box' if args.radius >= 1.0 else 'shrunk around x0'}). "
              f"{args.train_rows:,} training rows and {args.test_rows:,} test "
              f"rows, sized independently over one stratified split: training "
              f"size sets the cost of an evaluation, test size sets the "
              f"resolution of the accuracy column, and PhiUSIIL is saturated "
              f"enough that a small test set cannot separate two good models. "
              f"Applied identically to every arm and reported — not the "
              f"invisible cap `fit_gaussians` used to apply. Single-threaded. "
              f"Seeds: {','.join(map(str, seeds))}."))

    # Timing, on its own, because it answers a different question from accuracy
    # and because mixing the two invites reading a machine-dependent number as a
    # property of the method.
    trows = []
    if ref:
        trows.append(["the construction itself", "—",
                      C.cell([1000 * r["construction_seconds"] for r in ref],
                             fmt="{:.0f}"), "—"])
        trows.append(["feature engineering (shared, not training)", "—",
                      C.cell([1000 * r["screen_seconds"] for r in ref],
                             fmt="{:.0f}"), "—"])
    for init in inits:
        for arm in arm_names:
            if arm == "none":
                continue
            sel = [r for r in records if r["arm"] == arm and r["init"] == init]
            reached = [r for r in sel if r["seconds_to_heuristic"] is not None]
            if not sel:
                continue
            ratio = "—"
            if reached and ref:
                base = float(np.median([r["construction_seconds"] for r in ref]))
                med = float(np.median([r["seconds_to_heuristic"] for r in reached]))
                if base > 0:
                    ratio = f"{med / base:,.0f}×"
            trows.append([
                f"{init} · {arm}",
                C.cell([1000 * r["seconds_to_heuristic"] for r in reached],
                       fmt="{:.0f}") if reached else "never",
                C.cell([1000 * r["seconds"] for r in sel], fmt="{:.0f}"),
                ratio,
            ])

    C.emit(
        "table_opt_phishing_timing",
        "PhiUSIIL optimizer study — wall-clock, single-threaded",
        ["what", "to match the construction (ms)", "full budget (ms)",
         "cost of matching, ÷ the construction"],
        trows,
        note=("The construction's own cost against what a search costs to reach "
              "the same objective value — the same quantity in the same units, "
              "which is the comparison the 'how much faster' claim needs. "
              "`never` means the arm did not reach the construction's objective "
              "inside the budget, which for a **hot** start is expected: it "
              "begins there. Read the hot rows as the cost of *improving on* the "
              "construction and the cold rows as the cost of *reaching* it. "
              "Absolute milliseconds are machine-dependent and the machine is "
              "recorded below; the ratio column is the portable part. "
              "Single-threaded throughout, so no arm can buy time with cores — "
              "which also means an optimizer that parallelises well gets no "
              "credit here."))


def _archive(label, args, seeds, inits, arm_names):
    import shutil
    import subprocess
    dest = os.path.join(C.OUTPUT_DIR, label)
    os.makedirs(dest, exist_ok=True)

    def sha(path):
        try:
            rev = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                                 capture_output=True, text=True,
                                 check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"
        try:
            dirty = subprocess.run(["git", "-C", path, "status", "--porcelain",
                                    "--untracked-files=no"],
                                   capture_output=True, text=True,
                                   check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return rev
        return f"{rev}-dirty" if dirty else rev

    lines = [
        f"label:       {label}",
        f"generated:   {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"tribble-fis: {sha(os.path.join(ROOT, 'tribble-fis'))}",
        f"tribble-opt: {sha(os.path.join(ROOT, 'tribble-opt'))}",
        f"tribble-cluster: {sha(os.path.join(ROOT, 'tribble-cluster'))}",
        f"grad-school: {sha(ROOT)}",
        f"seeds:       {','.join(map(str, seeds))}",
        "",
        "study:       reproduce/optimizers/run_phishing_opt_study.py",
        "dataset:     PhiUSIIL (binary classification)",
        f"train rows:  {args.train_rows}",
        f"test rows:   {args.test_rows}",
        f"top_n:       {args.top_n} features",
        f"components:  {args.components or 'chosen by BIC'}",
        f"radius:      {args.radius}",
        f"budget:      {args.budget} objective evaluations per arm per seed",
        f"objective:   refine._make_classifier_fitness, l2_shrink={PC.L2_SHRINK}",
        f"arms:        {','.join(arm_names)}",
        f"init:        {','.join(inits)}",
        "threads:     1 (OMP/BLAS pinned before numpy import)",
        "",
        C.machine_block().strip(),
        "",
    ]
    with open(os.path.join(dest, "PROVENANCE.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for name in ARTIFACTS:
        src = os.path.join(C.OUTPUT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
    print(f"  archived -> {os.path.relpath(dest, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
