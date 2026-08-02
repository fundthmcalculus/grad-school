#!/usr/bin/env python3
"""How it was usually done, against how it can be done now — at matched rule count.

    uv run --project tribble-fis --with-editable tribble-cluster --with scikit-learn \
        python reproduce/optimizers/run_identification_study.py --archive <label>

Identification only: no optimizer runs here. The question is what each route
produces *before* anyone starts searching, and what it costs to produce it.

Three routes, all swept over the same rule counts so the comparison is
like-for-like at every point:

  construction       partition the output into c buckets, screen the features,
                     fit a 1-D Gaussian mixture per (feature, bucket);
  classical-kmeans   cluster the joint input-output space into c clusters, one
                     rule per cluster, membership functions projected off each;
  classical-fcm      the same, with the author's fuzzy c-means.

Sweeping the construction's bucket count as well as the classical cluster count
is the point. The rule count is an input to the classical route and normally an
output of the construction, and comparing a 3-rule construction against a 9-rule
classical model would be comparing capacity, not identification. Here both are
asked for c rules and both answer.

## Timing

Wall-clock matters in this study — unlike the optimizer study next door, where
the budget is iterations — because "the construction is cheaper than clustering"
is a claim about time. So the measurement is made carefully:

  * **single-threaded**, forced through the OpenMP/BLAS environment before numpy
    or sklearn is imported. scikit-learn's k-means is threaded and would
    otherwise be timed with more cores than the construction uses.
  * **median of `--repeats` runs**, not the first, which pays import and JIT
    costs that belong to neither method.
  * **`--kmeans-n-init` is reported**, because scikit-learn defaults to ten
    restarts and that is a quality-versus-time choice rather than an intrinsic
    property of k-means. Quoting k-means as slow without saying how many
    restarts it ran would be a rigged comparison.

Run it on an otherwise idle machine. The table records the host.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Before numpy, before sklearn: the timing comparison is single-threaded and
# these are read at import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(ROOT, "reproduce"))
sys.path.insert(0, os.path.join(ROOT, "reproduce", "tables"))

import common as C  # noqa: E402

DEFAULT_GRID = "2,3,4,6,8,12"


def _data(dataset, seed, test_size=0.2):
    from sklearn.model_selection import train_test_split
    import _fuzzy_models as FM
    import table_concrete_reconciliation as TCR

    if dataset != "concrete":
        raise SystemExit(f"dataset {dataset!r} not wired up here yet")
    loaded = FM.load_concrete()
    if loaded is None:
        raise SystemExit("Concrete unavailable (no CSV, no UCI mirror).")
    prep = TCR.prepare(*loaded)
    Xtr, Xte, ytr, yte = train_test_split(prep["Xt"], prep["y"],
                                          test_size=test_size, random_state=seed)
    return Xtr, Xte, ytr, yte, prep["span"]


def _construction(Xtr, ytr, c, seed):
    """Identify by the Gaussian construction at c output buckets. (model, y, ...)"""
    from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                       create_gaussian_membership_dict,
                                       take_top_features)
    from tribblefis.regression import partition_output

    start = time.perf_counter()
    y_all, bucket_mean = partition_output(c, ytr["y_value"])
    diffs = calculate_gaussian_correlation(Xtr, y_all["y_bucket"])
    _, features = take_top_features(diffs, top_n=len(Xtr.columns))
    model = create_gaussian_membership_dict(Xtr, y_all["y_bucket"],
                                            top_n_var_names=features,
                                            n_gaussians=-1)
    seconds = time.perf_counter() - start
    return model, y_all, bucket_mean, list(features), seconds


def _classical(Xtr, ytr, c, seed, method, n_init=10):
    import classical as CL
    CL_KM_N_INIT[0] = n_init
    return CL.identify(Xtr, ytr["y_value"], c, method, seed=seed)


CL_KM_N_INIT = [10]      # patched into classical._cluster_joint below


def _install_n_init_hook():
    """Let the sweep control k-means restarts without editing `classical.py`.

    The restart count is a quality/time dial and the table has to report it, so
    it has to be settable from here; but `classical.py` is the artifact the
    study's claims rest on and should not grow a knob that exists only for a
    sweep.
    """
    import classical as CL
    original = CL._cluster_joint

    def hooked(J, c, method, seed):
        if method == "kmeans":
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=c, n_init=CL_KM_N_INIT[0],
                        random_state=seed).fit(J)
            return km.labels_, km.cluster_centers_
        return original(J, c, method, seed)

    CL._cluster_joint = hooked


def _evaluate(model, Xtr, Xte, ytr_df, yte, features, bucket_mean, c, order,
              l2_reg, span, n_folds=3, seed=0):
    """(cv_mse, test_r2, n_mfs, n_params) for an identified model."""
    from tribblefis.refine import (_make_folds, _make_kfold_fitness,
                                   extract_gaussian_params)
    from sklearn.metrics import r2_score
    from tribblefis.regression import predict_tsk, solve_tsk_consequents

    folds = _make_folds(len(Xtr), n_folds, 0.2, seed)
    fitness = _make_kfold_fitness(model, Xtr, ytr_df, folds, features, c, order,
                                  l2_reg, "raw", None)
    params = extract_gaussian_params(model)
    cv = float(fitness(params))
    try:
        corr, ybm = solve_tsk_consequents(
            Xtr, model, features, bucket_mean, ytr_df, n_output_buckets=c,
            order=order, l2_reg=l2_reg, basis="raw", cross_pairs=None,
            verbose=False)
        pred = predict_tsk(Xte, model, features, ybm, corr, order=order,
                           basis="raw", cross_pairs=None)
        truth = np.asarray(yte["y_value"], dtype=float).ravel()
        pred = np.asarray(pred, dtype=float).ravel()
        keep = ~np.isnan(pred)
        r2 = float(r2_score(truth[keep], pred[keep])) if np.any(keep) else float("nan")
    except Exception:  # noqa: BLE001
        r2 = float("nan")
    n_mfs = sum(len(lm.memberships) for fm in model.feature_models.values()
                for lm in fm.label_models.values())
    return cv, r2, n_mfs, len(params)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="concrete")
    ap.add_argument("--rules", default=DEFAULT_GRID)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--order", default="2nd")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--kmeans-n-init", type=int, default=10)
    ap.add_argument("--archive", metavar="LABEL")
    args = ap.parse_args()

    _install_n_init_hook()
    grid = [int(x) for x in args.rules.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    routes = [("construction", _construction),
              ("classical-kmeans",
               lambda X, y, c, s: _classical(X, y, c, s, "kmeans", args.kmeans_n_init)),
              ("classical-fcm",
               lambda X, y, c, s: _classical(X, y, c, s, "fcm"))]

    print(f"identification sweep: {args.dataset}, rules={grid}, seeds={seeds}, "
          f"repeats={args.repeats}, kmeans n_init={args.kmeans_n_init}, "
          f"threads=1")

    records = []
    for seed in seeds:
        Xtr, Xte, ytr, yte, span = _data(args.dataset, seed)
        for name, build in routes:
            for c in grid:
                times = []
                out = None
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    out = build(Xtr, ytr, c, seed)
                    times.append(time.perf_counter() - t0)
                model, y_df, bucket_mean, features, _inner = out
                cv, r2, n_mfs, n_params = _evaluate(
                    model, Xtr, Xte, y_df, yte, features, bucket_mean, c,
                    args.order, args.l2, span, seed=seed)
                rec = {"route": name, "seed": seed, "rules": c,
                       "seconds": float(np.median(times)), "cv_mse": cv,
                       "r2": r2, "n_mfs": n_mfs, "n_params": n_params}
                records.append(rec)
                print(f"  {name:<18} c={c:<3} R2={r2:6.3f}  cv={cv:.5f}  "
                      f"mfs={n_mfs:<4} params={n_params:<4} "
                      f"{1000 * rec['seconds']:8.1f} ms")

    rows = []
    for name, _ in routes:
        for c in grid:
            sel = [r for r in records if r["route"] == name and r["rules"] == c]
            if not sel:
                continue
            rows.append([
                name, str(c),
                C.cell([r["n_mfs"] for r in sel], fmt="{:.0f}"),
                C.cell([r["n_params"] for r in sel], fmt="{:.0f}"),
                C.cell([r["r2"] for r in sel]),
                C.cell([r["cv_mse"] for r in sel], fmt="{:.5f}"),
                C.cell([1000 * r["seconds"] for r in sel], fmt="{:.1f}"),
            ])

    C.emit(
        "table_identification_sweep",
        f"Identification at matched rule count — construction against the "
        f"classical route ({args.dataset}, order {args.order})",
        ["route", "rules", "membership fns", "free params", "test R²",
         "CV MSE", "identify (ms)"],
        rows,
        note=(f"**Identification only — no optimizer runs in this table.** Each "
              f"route is asked for the same number of rules and reports what it "
              f"produced and what it cost. The construction's bucket count is "
              f"swept alongside the classical cluster count, because the rule "
              f"count is an input to one and normally an output of the other, and "
              f"comparing across different rule counts would be comparing capacity "
              f"rather than identification. Timing is the **median of "
              f"{args.repeats}** repeats, forced **single-threaded** through the "
              f"OpenMP/BLAS environment before numpy is imported — scikit-learn's "
              f"k-means is threaded and would otherwise be timed on more cores "
              f"than the construction uses. k-means ran **n_init="
              f"{args.kmeans_n_init}** restarts; that is a quality-versus-time "
              f"dial, not a property of the algorithm, and quoting k-means as slow "
              f"without saying so would rig the comparison. Consequents come from "
              f"the same closed-form ridge-TSK solve in every row, so what is "
              f"compared is rule identification alone. Seeds: "
              f"{','.join(map(str, seeds))}."))

    path = os.path.join(C.OUTPUT_DIR, "table_identification_sweep_seeds.csv")
    C.write_csv(path, ["route", "seed", "rules", "n_mfs", "n_params", "r2",
                       "cv_mse", "seconds"],
                [[r["route"], r["seed"], r["rules"], r["n_mfs"], r["n_params"],
                  f"{r['r2']:.6f}", f"{r['cv_mse']:.6f}", f"{r['seconds']:.6f}"]
                 for r in records])
    print(f"  wrote {path}")

    if args.archive:
        _archive_here(args.archive, args, seeds, grid)
    return 0


def _archive_here(label, args, seeds, grid):
    import shutil
    import subprocess
    dest = os.path.join(C.OUTPUT_DIR, label)
    os.makedirs(dest, exist_ok=True)

    def sha(path):
        try:
            return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                                  capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    lines = [
        f"label:       {label}",
        f"generated:   {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"tribble-fis: {sha(os.path.join(ROOT, 'tribble-fis'))}",
        f"tribble-cluster: {sha(os.path.join(ROOT, 'tribble-cluster'))}",
        f"grad-school: {sha(ROOT)}",
        f"seeds:       {','.join(map(str, seeds))}",
        "",
        "study:       reproduce/optimizers/run_identification_study.py",
        f"dataset:     {args.dataset}",
        f"rule grid:   {','.join(map(str, grid))}",
        f"order:       {args.order}",
        f"repeats:     {args.repeats} (median reported)",
        f"kmeans n_init: {args.kmeans_n_init}",
        "threads:     1 (OMP/BLAS pinned before numpy import)",
        "",
        C.machine_block().strip(),
        "",
    ]
    with open(os.path.join(dest, "PROVENANCE.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for name in ("table_identification_sweep.md", "table_identification_sweep.csv",
                 "table_identification_sweep_seeds.csv"):
        src = os.path.join(C.OUTPUT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
    print(f"  archived -> {os.path.relpath(dest, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
