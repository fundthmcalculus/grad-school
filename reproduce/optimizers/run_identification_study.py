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
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
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
    Xtr, Xte, ytr, yte = train_test_split(
        prep["Xt"], prep["y"], test_size=test_size, random_state=seed
    )
    return Xtr, Xte, ytr, yte, prep["span"]


PIN_COMPONENTS = [-1]  # set from --pin-components; -1 means "choose by BIC"
MAX_SAMPLES = [None]  # set from --max-samples; None means "use every row"


def _construction(Xtr, ytr, c, seed):
    """Identify by the Gaussian construction at c output buckets.

    Returns (model, y, bucket_mean, features, train_seconds, screen_seconds).

    The two timers are separate on purpose. Ranking the features is *feature
    engineering* -- a preprocessing decision, made once, whose output any
    identification route consumes. Training is the output partition and the
    mixture fit. Folding the screen into the training number would compare a
    pipeline against a training step, which is not the comparison being made.

    With `PIN_COMPONENTS` left at -1 the construction chooses its own component
    count per (feature, bucket) by BIC, which costs four EM fits per pair on top
    of the k-means it actually keeps. Pinning it to 1 produces exactly the
    classical route's shape -- one Gaussian per (feature, bucket) -- so the two
    can be compared at equal parameter counts, and separates "the construction
    is dearer" from "model selection is dearer".
    """
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        take_top_features,
    )
    from tribblefis.regression import partition_output

    t0 = time.perf_counter()
    y_all, bucket_mean = partition_output(c, ytr["y_value"])
    t_part = time.perf_counter() - t0

    t0 = time.perf_counter()
    diffs = calculate_gaussian_correlation(Xtr, y_all["y_bucket"])
    _, features = take_top_features(diffs, top_n=len(Xtr.columns))
    screen_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    model = create_gaussian_membership_dict(
        Xtr,
        y_all["y_bucket"],
        top_n_var_names=features,
        n_gaussians=PIN_COMPONENTS[0],
        max_samples=MAX_SAMPLES[0],
    )
    train_seconds = t_part + (time.perf_counter() - t0)
    return model, y_all, bucket_mean, list(features), train_seconds, screen_seconds


def _classical(Xtr, ytr, c, seed, method, n_init=10):
    """Same signature as `_construction`; the classical route does no screening,
    so its feature-engineering time is zero and it uses every column."""
    import classical as CL

    CL_KM_N_INIT[0] = n_init
    model, y_df, bm, feats, secs = CL.identify(
        Xtr, ytr["y_value"], c, method, seed=seed
    )
    return model, y_df, bm, feats, secs, 0.0


CL_KM_N_INIT = [10]  # patched into classical._cluster_joint below


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

            km = KMeans(n_clusters=c, n_init=CL_KM_N_INIT[0], random_state=seed).fit(J)
            return km.labels_, km.cluster_centers_
        return original(J, c, method, seed)

    CL._cluster_joint = hooked


def _evaluate(
    model,
    Xtr,
    Xte,
    ytr_df,
    yte,
    features,
    bucket_mean,
    c,
    order,
    l2_reg,
    span,
    n_folds=3,
    seed=0,
):
    """(cv_mse, test_r2, n_mfs, n_params) for an identified model."""
    from tribblefis.refine import (
        _make_folds,
        _make_kfold_fitness,
        extract_gaussian_params,
    )
    from sklearn.metrics import r2_score
    from tribblefis.regression import predict_tsk, solve_tsk_consequents

    folds = _make_folds(len(Xtr), n_folds, 0.2, seed)
    fitness = _make_kfold_fitness(
        model, Xtr, ytr_df, folds, features, c, order, l2_reg, "raw", None
    )
    params = extract_gaussian_params(model)
    cv = float(fitness(params))
    try:
        corr, ybm = solve_tsk_consequents(
            Xtr,
            model,
            features,
            bucket_mean,
            ytr_df,
            n_output_buckets=c,
            order=order,
            l2_reg=l2_reg,
            basis="raw",
            cross_pairs=None,
            verbose=False,
        )
        pred = predict_tsk(
            Xte, model, features, ybm, corr, order=order, basis="raw", cross_pairs=None
        )
        truth = np.asarray(yte["y_value"], dtype=float).ravel()
        pred = np.asarray(pred, dtype=float).ravel()
        keep = ~np.isnan(pred)
        r2 = float(r2_score(truth[keep], pred[keep])) if np.any(keep) else float("nan")
    except Exception:  # noqa: BLE001
        r2 = float("nan")
    n_mfs = sum(
        len(lm.memberships)
        for fm in model.feature_models.values()
        for lm in fm.label_models.values()
    )
    return cv, r2, n_mfs, len(params)


def _write_seed_csv(records):
    path = os.path.join(C.OUTPUT_DIR, "table_identification_sweep_seeds.csv")
    C.write_csv(
        path,
        [
            "route",
            "seed",
            "rules",
            "n_mfs",
            "n_params",
            "r2",
            "cv_mse",
            "seconds",
            "screen_seconds",
        ],
        [
            [
                r["route"],
                r["seed"],
                r["rules"],
                r["n_mfs"],
                r["n_params"],
                f"{r['r2']:.6f}",
                f"{r['cv_mse']:.6f}",
                f"{r['seconds']:.6f}",
                f"{r.get('screen_seconds', 0.0):.6f}",
            ]
            for r in records
        ],
    )
    return path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default="concrete")
    ap.add_argument("--rules", default=DEFAULT_GRID)
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--order", default="2nd")
    ap.add_argument("--l2", type=float, default=1e-2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--kmeans-n-init", type=int, default=10)
    ap.add_argument(
        "--pin-components",
        type=int,
        default=-1,
        metavar="N",
        help="force N Gaussians per (feature, bucket) instead of "
        "choosing by BIC. N=1 gives exactly the classical "
        "route's parameter count, which is the matched "
        "comparison; the default -1 leaves model selection in.",
    )
    ap.add_argument(
        "--max-samples",
        type=int,
        default=0,
        metavar="N",
        help="cap the rows each (feature, bucket) fit sees; 0 (the "
        "default) uses every row. The library used to apply "
        "N=20000 unconditionally, as a prefix rather than a "
        "sample, with no way for a caller to see it.",
    )
    ap.add_argument("--archive", metavar="LABEL")
    args = ap.parse_args()

    PIN_COMPONENTS[0] = args.pin_components
    MAX_SAMPLES[0] = args.max_samples or None
    _install_n_init_hook()
    grid = [int(x) for x in args.rules.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    # The Markdown footer prints `common.SEEDS` -- the ten-seed protocol the
    # table harness runs. This study takes `--seeds`, so leaving it alone makes
    # a three-seed run claim ten.
    C.SEEDS = seeds
    routes = [
        ("construction", _construction),
        (
            "classical-kmeans",
            lambda X, y, c, s: _classical(X, y, c, s, "kmeans", args.kmeans_n_init),
        ),
        ("classical-fcm", lambda X, y, c, s: _classical(X, y, c, s, "fcm")),
    ]

    print(
        f"identification sweep: {args.dataset}, rules={grid}, seeds={seeds}, "
        f"repeats={args.repeats}, kmeans n_init={args.kmeans_n_init}, "
        f"threads=1"
    )

    records = []
    for seed in seeds:
        Xtr, Xte, ytr, yte, span = _data(args.dataset, seed)
        for name, build in routes:
            for c in grid:
                times, screens = [], []
                out = None
                for _ in range(args.repeats):
                    out = build(Xtr, ytr, c, seed)
                    times.append(out[4])
                    screens.append(out[5])
                model, y_df, bucket_mean, features = out[0], out[1], out[2], out[3]
                cv, r2, n_mfs, n_params = _evaluate(
                    model,
                    Xtr,
                    Xte,
                    y_df,
                    yte,
                    features,
                    bucket_mean,
                    c,
                    args.order,
                    args.l2,
                    span,
                    seed=seed,
                )
                rec = {
                    "route": name,
                    "seed": seed,
                    "rules": c,
                    "seconds": float(np.median(times)),
                    "screen_seconds": float(np.median(screens)),
                    "cv_mse": cv,
                    "r2": r2,
                    "n_mfs": n_mfs,
                    "n_params": n_params,
                }
                records.append(rec)
                print(
                    f"  {name:<18} c={c:<3} R2={r2:6.3f}  cv={cv:.5f}  "
                    f"mfs={n_mfs:<4} params={n_params:<4} "
                    f"train {1000 * rec['seconds']:8.1f} ms"
                    f"  (+{1000 * rec['screen_seconds']:.0f} ms feat.eng.)"
                )
                # Flush after every measurement. The first attempt at this sweep
                # was killed by a timeout after ~40 minutes and left nothing at
                # all, because the emit happened only at the end. A partial run
                # should cost the rows it did not reach, not the ones it did.
                _write_seed_csv(records)

    rows = []
    for name, _ in routes:
        for c in grid:
            sel = [r for r in records if r["route"] == name and r["rules"] == c]
            if not sel:
                continue
            rows.append(
                [
                    name,
                    str(c),
                    C.cell([r["n_mfs"] for r in sel], fmt="{:.0f}"),
                    C.cell([r["n_params"] for r in sel], fmt="{:.0f}"),
                    C.cell([r["r2"] for r in sel]),
                    C.cell([r["cv_mse"] for r in sel], fmt="{:.5f}"),
                    C.cell([1000 * r["seconds"] for r in sel], fmt="{:.1f}"),
                    C.cell([1000 * r["screen_seconds"] for r in sel], fmt="{:.1f}"),
                ]
            )

    C.emit(
        "table_identification_sweep",
        f"Identification at matched rule count — construction against the "
        f"classical route ({args.dataset}, order {args.order})",
        [
            "route",
            "rules",
            "membership fns",
            "free params",
            "test R²",
            "CV MSE",
            "train (ms)",
            "feat. eng. (ms)",
        ],
        rows,
        note=(
            f"**Identification only — no optimizer runs in this table.** Each "
            f"route is asked for the same number of rules and reports what it "
            f"produced and what it cost. The construction's bucket count is "
            f"swept alongside the classical cluster count, because the rule "
            f"count is an input to one and normally an output of the other, and "
            f"comparing across different rule counts would be comparing capacity "
            f"rather than identification. **The timing column is model "
            f"training only.** Feature engineering — the construction's O(M^2) "
            f"screen — is its own column and is charged to neither route: it is "
            f"a preprocessing decision whose output an identification route "
            f"consumes, and folding it in would compare a pipeline against a "
            f"training step. On Concrete it also selects nothing, since "
            f"`top_n` is the full feature count; it only ranks. Timing is the "
            f"**median of "
            f"{args.repeats}** repeats, forced **single-threaded** through the "
            f"OpenMP/BLAS environment before numpy is imported — scikit-learn's "
            f"k-means is threaded and would otherwise be timed on more cores "
            f"than the construction uses. k-means ran **n_init="
            f"{args.kmeans_n_init}** restarts; that is a quality-versus-time "
            f"dial, not a property of the algorithm, and quoting k-means as slow "
            f"without saying so would rig the comparison. Consequents come from "
            f"the same closed-form ridge-TSK solve in every row, so what is "
            f"compared is rule identification alone. "
            + (
                f"The construction is **pinned to {args.pin_components} "
                f"Gaussian(s) per (feature, bucket)**, so its parameter count "
                f"matches the classical route's and its BIC model selection — "
                f"four EM fits per pair, kept out of what the classical route "
                f"is asked to do — is excluded. "
                if args.pin_components > 0
                else f"The construction **chooses its own component count by BIC**, "
                f"which costs four EM fits per (feature, bucket) beyond the "
                f"k-means it keeps, and gives it more parameters than the "
                f"classical route; `--pin-components 1` is the matched run. "
            )
            + f"Seeds: {','.join(map(str, seeds))}."
        ),
    )

    print(f"  wrote {_write_seed_csv(records)}")

    if args.archive:
        _archive_here(args.archive, args, seeds, grid)
    return 0


def _archive_here(label, args, seeds, grid):
    import shutil
    import subprocess

    dest = os.path.join(C.OUTPUT_DIR, label)
    os.makedirs(dest, exist_ok=True)

    def sha(path):
        """HEAD, plus `-dirty` when the tree has uncommitted changes.

        Without the suffix a stamp reads as "this commit produced these numbers"
        when the numbers may have come from an edited working tree -- which is
        exactly what happens while a library fix is being measured before it is
        committed.
        """
        try:
            rev = subprocess.run(
                ["git", "-C", path, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unknown"
        try:
            dirty = subprocess.run(
                ["git", "-C", path, "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            return rev
        return f"{rev}-dirty" if dirty else rev

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
        f"max samples: {args.max_samples or 'all rows'}",
        f"pin components: {args.pin_components}"
        + (
            " (BIC model selection)"
            if args.pin_components < 0
            else " (fixed; BIC selection excluded)"
        ),
        "threads:     1 (OMP/BLAS pinned before numpy import)",
        "",
        C.machine_block().strip(),
        "",
    ]
    with open(os.path.join(dest, "PROVENANCE.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for name in (
        "table_identification_sweep.md",
        "table_identification_sweep.csv",
        "table_identification_sweep_seeds.csv",
    ):
        src = os.path.join(C.OUTPUT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
    print(f"  archived -> {os.path.relpath(dest, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
