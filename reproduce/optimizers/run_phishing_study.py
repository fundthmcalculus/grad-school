#!/usr/bin/env python3
"""Does the identification cost gap survive a real dataset? Sweep n and see.

    uv run --project tribble-fis --with-editable tribble-cluster --with scikit-learn \
        python reproduce/optimizers/run_phishing_study.py --smoke
    uv run --project tribble-fis --with-editable tribble-cluster --with scikit-learn \
        python reproduce/optimizers/run_phishing_study.py --archive <label>

On Concrete (824 x 8) the classical route identified a rule base 25-84x faster
than the Gaussian construction. That cannot be extrapolated: the construction's
cost there was dominated by fitting candidate mixtures, k-means on 824 rows is
free, and the two scale in different variables. So the question here is not
"which is faster" but **"where do the two curves cross, if they cross"** — which
is why the sweep is over sample size, not just over component count.

Both routes produce the same model shape (K class rules, c components per
feature per class) and are read by the same `simple_gaussian_predict`, so what
differs is only how the Gaussians are placed. See `phishing.py`.

Timing is single-threaded, median of `--repeats`, with OMP/BLAS pinned before
numpy imports — same protocol as the Concrete sweep, and for the same reason:
here cost is the claim rather than a by-product.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

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
import phishing as P  # noqa: E402


def _write(records):
    path = os.path.join(C.OUTPUT_DIR, "table_phishing_identification_seeds.csv")
    C.write_csv(path, ["route", "seed", "n_rows", "components", "n_features",
                       "n_mfs", "accuracy", "identify_s", "screen_s"],
                [[r["route"], r["seed"], r["n_rows"], r["components"],
                  r["n_features"], r["n_mfs"], f"{r['accuracy']:.6f}",
                  f"{r['identify_s']:.6f}", f"{r['screen_s']:.6f}"]
                 for r in records])
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="5000,20000,50000,120000")
    ap.add_argument("--components", type=int, default=3,
                    help="clusters per class for the classical route; the "
                         "construction chooses its own unless --pin-components")
    ap.add_argument("--pin-components", action="store_true",
                    help="force the construction to the same component count, "
                         "so the two have identical parameter counts")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--cap-classical", type=int, default=0, metavar="N",
                    help="also run classical routes clustering on at most N "
                         "rows per class, the same cap the construction "
                         "applies internally (fit_gaussians max_samples=20000). "
                         "Without this the construction is compared against "
                         "routes that read every row, and part of its scaling "
                         "advantage is just that it does not.")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-samples", type=int, default=0, metavar="N",
                    help="cap the rows each (feature, class) fit sees; 0 (the "
                         "default) uses every row. The library used to apply "
                         "N=20000 unconditionally, as a prefix rather than a "
                         "sample, with no way for a caller to see it.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--archive", metavar="LABEL")
    args = ap.parse_args()

    if args.smoke:
        args.sizes, args.repeats, args.seeds = "5000,20000", 1, "0"
    sizes = [int(s) for s in args.sizes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    # The Markdown footer prints `common.SEEDS`, which is the ten-seed protocol
    # the table harness runs. This study takes `--seeds`, so the footer would
    # otherwise claim ten seeds for a three-seed run.
    C.SEEDS = seeds

    print(f"phishing identification sweep: sizes={sizes}, components="
          f"{args.components}, top_n={args.top_n}, seeds={seeds}, "
          f"repeats={args.repeats}, threads=1")

    from sklearn.model_selection import train_test_split

    records = []
    for n in sizes:
        X, y = P.load(n)
        for seed in seeds:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                                  random_state=seed, stratify=y)
            features, screen_s = P.screen(Xtr, ytr, args.top_n)

            routes = [
                ("construction", lambda: P.construction(
                    Xtr, ytr, features,
                    n_gaussians=args.components if args.pin_components else -1,
                    max_samples=args.max_samples or None)),
                ("classical-kmeans", lambda: P.classical(
                    Xtr, ytr, features, args.components, "kmeans", seed)),
                ("classical-fcm", lambda: P.classical(
                    Xtr, ytr, features, args.components, "fcm", seed)),
            ]
            if args.cap_classical:
                cap = args.cap_classical
                routes += [
                    (f"classical-kmeans-{cap // 1000}k", lambda: P.classical(
                        Xtr, ytr, features, args.components, "kmeans", seed,
                        max_samples=cap)),
                    (f"classical-fcm-{cap // 1000}k", lambda: P.classical(
                        Xtr, ytr, features, args.components, "fcm", seed,
                        max_samples=cap)),
                ]
            for name, build in routes:
                times, model = [], None
                for _ in range(args.repeats):
                    t0 = time.perf_counter()
                    model, _inner = build()
                    times.append(time.perf_counter() - t0)
                acc = P.accuracy(model, Xte, yte, features)
                # Model training only. Feature engineering is reported as its
                # own line and charged to nobody: selecting features is a
                # preprocessing decision that any route needs made for it, and
                # folding it into one route's training time compares a pipeline
                # against a training step. Both routes here consume the SAME
                # selected feature set.
                charged = float(np.median(times))
                rec = {"route": name, "seed": seed, "n_rows": len(Xtr),
                       "components": args.components,
                       "n_features": len(features),
                       "n_mfs": P.n_membership_fns(model), "accuracy": acc,
                       "identify_s": charged, "screen_s": screen_s}
                records.append(rec)
                _write(records)
                print(f"  n={len(Xtr):<7} {name:<18} acc={acc:.4f}  "
                      f"mfs={rec['n_mfs']:<4} train {1000 * charged:8.1f} ms")

    # Feature engineering, reported once per size: it is shared, it is not
    # training, and it belongs to neither route.
    fe_rows = []
    for n in sizes:
        sel = [r for r in records if r["n_rows"] == int(n * 0.8)]
        if sel:
            fe_rows.append([f"{sel[0]['n_rows']:,}", f"{sel[0]['n_features']}",
                            C.cell([1000 * r["screen_s"] for r in sel],
                                   fmt="{:.0f}")])
    C.emit("table_phishing_feature_engineering",
           "PhiUSIIL — feature engineering, shared by every route",
           ["train rows", "features kept", "screening (ms)"], fe_rows,
           note=("`calculate_gaussian_correlation` + `take_top_features`, the "
                 "O(M^2) screen that ranks 50 features and keeps the top "
                 f"{args.top_n}. **This is preprocessing, not model training**, "
                 "and it is deliberately kept out of the training comparison: "
                 "every route in that table consumes the same selected feature "
                 "set, so charging the screen to whichever route happens to "
                 "implement it would compare a pipeline against a training step. "
                 "It is reported here because it is a real cost that someone "
                 "pays, and because on this dataset it is not small."))

    rows = []
    for name in dict.fromkeys(r["route"] for r in records):
        for n in sizes:
            sel = [r for r in records if r["route"] == name
                   and r["n_rows"] == int(n * 0.8)]
            if not sel:
                continue
            rows.append([
                name, f"{sel[0]['n_rows']:,}",
                C.cell([r["n_mfs"] for r in sel], fmt="{:.0f}"),
                C.cell([r["accuracy"] for r in sel], fmt="{:.4f}"),
                C.cell([1000 * r["identify_s"] for r in sel], fmt="{:.0f}"),
            ])

    C.emit(
        "table_phishing_identification",
        "PhiUSIIL — identification cost and accuracy against sample size",
        ["route", "train rows", "membership fns", "test accuracy",
         "train (ms)"],
        rows,
        note=(f"Binary classification, so the rule count is fixed at K = 2 for "
              f"every route and there is no rule count to sweep — what is swept "
              f"is the **sample size**, because the Concrete result (classical "
              f"25-84x cheaper on 824 rows) cannot be extrapolated: the two "
              f"routes scale in different variables. Both produce the same model "
              f"shape and are read by the same `simple_gaussian_predict`, so what "
              f"differs is only how the Gaussians are placed: a per-feature 1-D "
              f"mixture fit against a multivariate clustering within each class. "
              f"Classical routes use {args.components} clusters per class; the "
              f"construction "
              + (f"is pinned to the same component count"
                 if args.pin_components else
                 f"chooses its own component count per (feature, class), so its "
                 f"membership-function count differs and the comparison is "
                 f"shape-matched only when `--pin-components` is passed") +
              f". Top {args.top_n} features by the construction's own screening, "
              f"used by all three routes. **Timing is model training only** — the shared "
              f"feature-engineering cost is in its own table and is charged to "
              f"neither route, because it is preprocessing that both consume the "
              f"output of. "
              + (f"The `-{args.cap_classical // 1000}k` rows are the honest "
                 f"control on scaling: the construction truncates every "
                 f"(feature, class) column to its first "
                 f"{args.cap_classical:,} rows inside `fit_gaussians`, so its "
                 f"fitting cost stops growing above that and only the row "
                 f"filtering still scales. Those rows give the classical route "
                 f"the same cap, which is what separates *the construction "
                 f"scales better* from *the construction reads less data*. "
                 if args.cap_classical else "")
              + f"Single-threaded, median of {args.repeats}."))

    print(f"  wrote {_write(records)}")
    if args.archive:
        _archive(args.archive, args, sizes, seeds)
    return 0


def _archive(label, args, sizes, seeds):
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

    lines = [f"label:       {label}",
             f"generated:   {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
             f"tribble-fis: {sha(os.path.join(ROOT, 'tribble-fis'))}",
             f"tribble-cluster: {sha(os.path.join(ROOT, 'tribble-cluster'))}",
             f"grad-school: {sha(ROOT)}",
             f"seeds:       {','.join(map(str, seeds))}", "",
             "study:       reproduce/optimizers/run_phishing_study.py",
             "dataset:     PhiUSIIL (binary classification)",
             f"sizes:       {','.join(map(str, sizes))}",
             f"components:  {args.components}"
             + (" (construction pinned to match)" if args.pin_components else ""),
             f"top_n:       {args.top_n}",
             f"max samples: {args.max_samples or 'all rows'}",
             f"repeats:     {args.repeats} (median reported)",
             "threads:     1 (OMP/BLAS pinned before numpy import)", "",
             C.machine_block().strip(), ""]
    with open(os.path.join(dest, "PROVENANCE.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    for name in ("table_phishing_identification.md",
                 "table_phishing_identification.csv",
                 "table_phishing_identification_seeds.csv"):
        src = os.path.join(C.OUTPUT_DIR, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dest, name))
    print(f"  archived -> {os.path.relpath(dest, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
