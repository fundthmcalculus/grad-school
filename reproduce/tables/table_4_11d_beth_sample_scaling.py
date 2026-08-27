"""Table 4.11(d) -- BETH: every arm trained on the SAME number of samples.

Table 4.11(c) compared training times that were not comparable, and this table
exists to correct that. In (c) each arm saw a different training set:

    Tribble one-class     763,144 rows (the whole benign split)
    One-class SVM          20,000 rows (capped -- libsvm is O(n^2)-O(n^3))
    Isolation Forest      763,144 rows nominally, but `max_samples=256` by
                          default, so each tree is built from 256 rows and the
                          fit cost barely moves with n at all

So (c)'s training-time panel was reading three different experiments, and its
flat Isolation Forest line was an artifact of that default rather than a property
of the algorithm. Wall-clock is only a comparison when the work is the same.

WHAT THIS SWEEP FIXES
---------------------
Every arm is fitted on the SAME subsample -- the identical rows, drawn once per
(n, seed) and handed to all five arms -- for n from 1,000 to 20,000. 20,000 is
the ceiling because that is where the one-class SVM's cap sat in (c); pushing it
higher would drop the SVM out of the comparison and re-create the problem this
table exists to fix.

Isolation Forest appears TWICE, because there is no single honest way to put it
in this sweep:

  * `max_samples=256` -- the library default, and what (c) measured. Its fit cost
    is near-flat in n by construction. Kept so (c)'s number stays traceable.
  * `max_samples=n`   -- every tree built from all n rows, which is the arm that
    is actually doing the same amount of work as the others.

Reporting only the first would repeat (c)'s mistake; reporting only the second
would silently change what "Isolation Forest" means between two tables.

Quality is measured at every n as well, because "how much data does this need?"
is the question a sample-count sweep is really being asked, and reporting time
alone would leave it hanging. Inference time is re-measured at every n too: it is
independent of n for the fuzzy arms and for Isolation Forest, but NOT for the
one-class SVM, whose support-vector count grows with the training set -- so its
scoring cost rises with n while everyone else's stays put.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_4_11d_beth_sample_scaling.py
"""

from __future__ import annotations

import os
import sys
import warnings

# Thread caps before numpy -- see table_4_11_beth_anomaly.py. n_jobs=-1 on a
# 32-core host hung this machine and segfaulted the process.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("REPRO_BLAS_THREADS", "8"))

import numpy as np  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import table_4_11_beth_anomaly as T411  # noqa: E402

FA_BUDGET = float(os.environ.get("REPRO_BETH_FA_BUDGET", "0.01"))
IF_TREES = int(os.environ.get("REPRO_BETH_IF_TREES", "200"))
N_JOBS = int(os.environ.get("REPRO_BETH_N_JOBS", "8"))

# Training-set sizes. Ceiling is 20,000 -- the one-class SVM's cap in 4.11(c);
# beyond it that arm leaves the comparison.
_ns = os.environ.get("REPRO_BETH_NSWEEP", "")
N_SWEEP = [int(n) for n in _ns.split(",")] if _ns else [1000, 2000, 5000, 10000, 20000]

# Feature count the sweep is run at. Fixed, so this table varies exactly one
# thing; 4.11(c) is the sweep over features at fixed sample count.
TOP_N = int(os.environ.get("REPRO_BETH_SCALING_TOPN", "0"))  # 0 => all features


def _threshold(scores_val, budget):
    thr = float(np.quantile(scores_val, 1.0 - budget))
    return thr, float((scores_val > thr).mean())


def main():
    print("Table 4.11(d) -- BETH: matched training-set size across every arm")
    got = T411.load_splits()
    if got is None:
        print("  BETH unavailable -- nothing emitted")
        return
    Xtr, Xva, Xte, _is_anom_va, is_anom_te, meta = got
    n_feat = meta["n_feat"]
    k = TOP_N or n_feat

    from tribblefis.one_class import TribbleOneClassDetector

    # Resolve the feature set ONCE so every arm at every n sees identical
    # columns, and so the table can state them rather than imply them.
    probe = TribbleOneClassDetector(
        score="surprisal",
        norm_conorm=T411.CONORM,
        feature_selection="variance" if k < n_feat else "all",
        top_n=k if k < n_feat else -1,
        random_state=42,
    ).fit(Xtr.iloc[:20000])
    features = list(probe.top_features_)
    print(f"  features ({len(features)}): {features}")
    Xtr_f, Xva_f, Xte_f = Xtr[features], Xva[features], Xte[features]

    ARMS = [
        "Tribble one-class (surprisal)",
        "Tribble one-class (complement)",
        "Isolation Forest (max_samples=256, library default)",
        "Isolation Forest (max_samples=n, matched work)",
        "One-class SVM",
    ]

    acc = {
        (arm, n): {"fit": [], "inf": [], "auc": [], "det": [], "fa": []}
        for arm in ARMS
        for n in N_SWEEP
    }

    for n in N_SWEEP:
        for seed in C.SEEDS:
            # ONE subsample per (n, seed), shared by every arm. This is the whole
            # point of the table: same rows, same count, same work.
            idx = np.random.RandomState(seed).choice(len(Xtr_f), n, replace=False)
            Xfit = Xtr_f.iloc[idx]

            def _run(arm, make, score):
                try:
                    with C.timed() as t_fit:
                        est = make()
                    s_va = score(est, Xva_f)
                    with C.timed() as t_inf:
                        s_te = score(est, Xte_f)
                    thr, _ = _threshold(s_va, FA_BUDGET)
                    det, fa = T411._rates(s_te > thr, is_anom_te)
                    a = acc[(arm, n)]
                    a["fit"].append(t_fit.seconds)
                    a["inf"].append(t_inf.seconds)
                    a["auc"].append(float(roc_auc_score(is_anom_te, s_te)))
                    a["det"].append(det)
                    a["fa"].append(fa)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"    n={n} {arm} seed {seed}: {exc.__class__.__name__}: {exc}"
                    )

            oc_score = lambda e, X: -np.asarray(e.score_samples(X))  # noqa: E731

            for score_mode, arm in (
                ("surprisal", ARMS[0]),
                ("complement", ARMS[1]),
            ):
                _run(
                    arm,
                    lambda sm=score_mode: TribbleOneClassDetector(
                        score=sm,
                        norm_conorm=T411.CONORM,
                        feature_selection="all",
                        contamination=FA_BUDGET,
                        random_state=seed,
                    ).fit(Xfit),
                    oc_score,
                )

            _run(
                ARMS[2],
                lambda: IsolationForest(
                    n_estimators=IF_TREES, random_state=seed, n_jobs=N_JOBS
                ).fit(Xfit),
                oc_score,
            )
            _run(
                ARMS[3],
                lambda nn=n: IsolationForest(
                    n_estimators=IF_TREES,
                    max_samples=nn,
                    random_state=seed,
                    n_jobs=N_JOBS,
                ).fit(Xfit),
                oc_score,
            )
            _run(
                ARMS[4],
                lambda: OneClassSVM(nu=FA_BUDGET, gamma="scale").fit(Xfit),
                lambda e, X: -np.asarray(e.decision_function(X)),
            )

        for arm in ARMS:
            a = acc[(arm, n)]
            if not a["fit"]:
                continue
            fm, _ = C.agg(a["fit"])
            im, _ = C.agg(a["inf"])
            am, _ = C.agg(a["auc"])
            print(f"    n={n:6,} {arm:52s} fit={fm:6.3f}s inf={im:6.3f}s auc={am:.4f}")

    header = [
        "Method",
        "Training samples",
        "Train time (s)",
        "Inference time (s)",
        "ROC-AUC",
        "Detection rate",
        "False-alarm rate",
        "Detection − false alarm",
    ]
    rows = []
    for arm in ARMS:
        for n in N_SWEEP:
            a = acc[(arm, n)]
            if not a["fit"]:
                rows.append([arm, f"{n:,}", C.NA, C.NA, C.NA, C.NA, C.NA, C.NA])
                continue
            dm, _ = C.agg(a["det"])
            fm_, _ = C.agg(a["fa"])
            rows.append(
                [
                    arm,
                    f"{n:,}",
                    C.cell(a["fit"], fmt="{:.3f}"),
                    C.cell(a["inf"], fmt="{:.3f}"),
                    C.cell(a["auc"], fmt="{:.4f}"),
                    C.cell(a["det"]),
                    C.cell(a["fa"]),
                    f"{dm - fm_:+.3f}",
                ]
            )

    C.emit(
        "table_4_11d_beth_sample_scaling",
        "Table 4.11(d) — BETH: matched training-set size across every arm",
        header,
        rows,
        note=(
            f"**Every arm is fitted on the identical rows**: one subsample is drawn per "
            f"(n, seed) from the {meta['n_train']:,}-row benign training split and handed "
            f"to all five arms, so the training-time column is a comparison rather than "
            f"five different experiments. This corrects Table 4.11(c), where Tribble saw "
            f"all {meta['n_train']:,} rows, the one-class SVM saw a 20,000-row cap, and "
            f"Isolation Forest saw {meta['n_train']:,} rows nominally but built each tree "
            f"from 256 (`max_samples` default) — so (c)'s flat Isolation Forest training "
            f"line was that default, not a property of the algorithm. n tops out at "
            f"20,000 because that is where the SVM's cap sat in (c); going higher would "
            f"drop it out of the comparison. **Isolation Forest appears twice on purpose**: "
            f"at `max_samples=256` (the library default, so (c)'s number stays traceable) "
            f"and at `max_samples=n` (every tree built from all n rows — the arm doing the "
            f"same work as the others). Quality is reported at every n because "
            f"'how much data does this need?' is what a sample sweep is really asking. "
            f"Inference time is re-measured at every n: it is independent of n for the "
            f"fuzzy arms and both forests, but **not** for the one-class SVM, whose "
            f"support-vector count grows with the training set, so its scoring cost rises "
            f"with n while everyone else's stays flat. Features held fixed at all "
            f"{len(features)}: {', '.join(features)} — see Table 4.11(c) for the sweep over "
            f"feature count at fixed sample size. Threshold is the {1 - FA_BUDGET:.3f} "
            f"quantile of each arm's own benign-validation scores ({FA_BUDGET:.1%} budget); "
            f"scored on the full {meta['n_test']:,}-row test split. All cells are mean ± "
            f"sample std across common.SEEDS ({C.SEEDS}) — every arm is stochastic here, "
            f"including the fuzzy ones, because which n rows are drawn is itself random."
        ),
    )


if __name__ == "__main__":
    main()
