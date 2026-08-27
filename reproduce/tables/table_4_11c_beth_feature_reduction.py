"""Table 4.11(c) -- BETH: quality, train time and inference time vs feature reduction.

Table 4.11 answers "which detector", at one fixed feature set (all eight
surviving columns). This answers the question a deployment actually asks: what
does feature reduction buy, and what does it cost? Both halves are needed --
"with reduction" is meaningless without the "without reduction" row beside it.

WHAT IS MEASURED, ACROSS EVERY ARM AND EVERY FEATURE COUNT
----------------------------------------------------------
  quality         ROC-AUC on the test split (threshold-free), plus detection and
                  false-alarm rate at a calibrated operating point
  train time      wall-clock fit on the benign training split
  inference time  wall-clock scoring of the full 188,967-row test split, and the
                  throughput that implies (rows/second)

Inference time is reported because it is the number that decides whether a host
detector can run online, and Table 4.11 does not measure it: a model that fits in
3 s and scores 189k events in 0.13 s is a different proposition from one that
fits in 0.1 s and scores them in 30 s. On this dataset the two costs rank the
arms differently, which is precisely why quoting only training time would
mislead.

ONE CALIBRATION FOR EVERY ARM
-----------------------------
Table 4.11 matched the baselines to the fuzzy arm by setting their
`contamination`/`nu`. That is the convention `table_4_4_openset.py` established,
but it leans on each library's own threshold placement -- and Isolation Forest's
`contamination` places the cut on the *training* score distribution, which is why
its detection rate reads as ~0 there while its AUC is 0.90.

This table instead applies ONE procedure to all four arms: the decision threshold
is the (1 - budget) quantile of that arm's own anomaly scores on the benign
VALIDATION split. Every arm is then calibrated identically, on data containing no
positives, and the detection/false-alarm columns become directly comparable
rather than comparable-after-allowing-for-each-library's-defaults. AUC is
threshold-free and unaffected either way.

FEATURE REDUCTION IS APPLIED IDENTICALLY TOO
--------------------------------------------
`TribbleOneClassDetector(feature_selection="variance", top_n=k)` picks the k
highest-variance columns -- an unsupervised filter, which is the only kind
available here, since with a single class there is no separation to rank on. The
selected names are read back off the fitted detector's `top_features_` and the
baselines are fitted on *exactly those columns*, so at every k all four arms see
identical input. Selecting independently per arm would have made the curves
incomparable in a way no note could repair.

k = n_features is the "without reduction" control and is not a separate code
path: it is the same sweep at its endpoint.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_4_11c_beth_feature_reduction.py
"""

from __future__ import annotations

import os
import sys
import warnings

# Thread caps before numpy -- see table_4_11_beth_anomaly.py's note. n_jobs=-1 on
# a 32-core host hung this machine and segfaulted the process; that was thread
# oversubscription, not memory.
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
OCSVM_TRAIN_CAP = int(os.environ.get("REPRO_OCSVM_TRAIN_CAP", "20000"))
IF_TREES = int(os.environ.get("REPRO_BETH_IF_TREES", "200"))
N_JOBS = int(os.environ.get("REPRO_BETH_N_JOBS", "8"))

# Feature counts to sweep. Default 2..8 -- 8 is "no reduction" on BETH's eight
# surviving columns, so the control is the sweep's own endpoint.
_ks = os.environ.get("REPRO_BETH_TOPN_SWEEP", "")
TOPN_SWEEP = [int(k) for k in _ks.split(",")] if _ks else None


def _threshold(scores_val, budget):
    """(threshold, realized validation false-alarm rate) at a benign-only budget."""
    thr = float(np.quantile(scores_val, 1.0 - budget))
    return thr, float((scores_val > thr).mean())


def _measure(fit, score, Xva, Xte, is_anom_te):
    """Fit / calibrate / score one arm, timing train and inference separately.

    `fit` is a zero-arg callable returning a fitted estimator; `score` maps
    (estimator, X) -> "higher means more anomalous". Inference time is the
    scoring of the FULL test split, timed on its own rather than folded into a
    combined fit-and-predict figure, because those two costs are paid at
    different times by different parts of a deployment.
    """
    with C.timed() as t_fit:
        est = fit()
    s_va = score(est, Xva)
    with C.timed() as t_inf:
        s_te = score(est, Xte)
    thr, realized = _threshold(s_va, FA_BUDGET)
    flagged = s_te > thr
    det, fa = T411._rates(flagged, is_anom_te)
    return {
        "auc": float(roc_auc_score(is_anom_te, s_te)),
        "det": det,
        "fa": fa,
        "val_fa": realized,
        "fit_s": t_fit.seconds,
        "inf_s": t_inf.seconds,
        "rows_per_s": len(Xte) / t_inf.seconds if t_inf.seconds > 0 else float("nan"),
    }


def _agg_runs(runs):
    """Mean of each metric across seeds, plus the std of the ones we plot."""
    out = {}
    for key in ("auc", "det", "fa", "val_fa", "fit_s", "inf_s", "rows_per_s"):
        vals = [r[key] for r in runs if r is not None and np.isfinite(r[key])]
        m, s = C.agg(vals)
        out[key] = m
        out[key + "_std"] = s
    return out


def main():
    print(
        "Table 4.11(c) -- BETH quality / train time / inference time vs feature count"
    )
    got = T411.load_splits()
    if got is None:
        print("  BETH unavailable -- nothing emitted")
        return
    Xtr, Xva, Xte, _is_anom_va, is_anom_te, meta = got
    n_feat = meta["n_feat"]
    ks = TOPN_SWEEP or list(range(2, n_feat + 1))
    print(f"  sweeping top_n = {ks} (k={n_feat} is the no-reduction control)")

    from tribblefis.one_class import TribbleOneClassDetector

    records = []
    for k in ks:
        reduced = k < n_feat

        # The fuzzy arms first: their fitted `top_features_` defines the column
        # subset every other arm at this k is then given.
        selected = None
        for score_mode in ("surprisal", "complement"):

            def _fit(sm=score_mode, kk=k):
                return TribbleOneClassDetector(
                    score=sm,
                    norm_conorm=T411.CONORM,
                    feature_selection="variance" if kk < n_feat else "all",
                    top_n=kk if kk < n_feat else -1,
                    contamination=FA_BUDGET,
                    random_state=42,
                ).fit(Xtr)

            # Deterministic given the fixed splits -- fitted once, no seed loop.
            est_holder = {}

            def _fit_capture(f=_fit):
                est_holder["est"] = f()
                return est_holder["est"]

            r = _measure(
                _fit_capture,
                lambda e, X: -np.asarray(e.score_samples(X)),
                Xva,
                Xte,
                is_anom_te,
            )
            if selected is None:
                selected = list(est_holder["est"].top_features_)
            r.update(
                arm=f"Tribble one-class ({score_mode})",
                k=k,
                reduced=reduced,
                seeds=1,
                features=",".join(selected),
            )
            for key in ("auc", "det", "fa", "val_fa", "fit_s", "inf_s", "rows_per_s"):
                r[key + "_std"] = 0.0
            records.append(r)
            print(
                f"    k={k} {r['arm']:34s} auc={r['auc']:.4f} det={r['det']:.3f} "
                f"fa={r['fa']:.3f} fit={r['fit_s']:.2f}s inf={r['inf_s']:.3f}s"
            )

        Xtr_k, Xva_k, Xte_k = Xtr[selected], Xva[selected], Xte[selected]

        for arm in ("Isolation Forest", "One-class SVM"):
            runs = []
            for seed in C.SEEDS:
                try:
                    if arm == "Isolation Forest":
                        runs.append(
                            _measure(
                                lambda s=seed: IsolationForest(
                                    n_estimators=IF_TREES,
                                    random_state=s,
                                    n_jobs=N_JOBS,
                                ).fit(Xtr_k),
                                lambda e, X: -np.asarray(e.score_samples(X)),
                                Xva_k,
                                Xte_k,
                                is_anom_te,
                            )
                        )
                    else:
                        idx = np.random.RandomState(seed).choice(
                            len(Xtr_k),
                            min(OCSVM_TRAIN_CAP, len(Xtr_k)),
                            replace=False,
                        )
                        runs.append(
                            _measure(
                                lambda i=idx: OneClassSVM(
                                    nu=FA_BUDGET, gamma="scale"
                                ).fit(Xtr_k.iloc[i]),
                                lambda e, X: -np.asarray(e.decision_function(X)),
                                Xva_k,
                                Xte_k,
                                is_anom_te,
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"    k={k} {arm} seed {seed}: {exc.__class__.__name__}: {exc}"
                    )
            if not runs:
                continue
            a = _agg_runs(runs)
            a.update(
                arm=arm,
                k=k,
                reduced=reduced,
                seeds=len(runs),
                features=",".join(selected),
            )
            records.append(a)
            print(
                f"    k={k} {arm:34s} auc={a['auc']:.4f} det={a['det']:.3f} "
                f"fa={a['fa']:.3f} fit={a['fit_s']:.2f}s inf={a['inf_s']:.3f}s"
            )

    header = [
        "Method",
        "features",
        "reduced?",
        "ROC-AUC",
        "Detection rate",
        "False-alarm rate",
        "Detection − false alarm",
        "Train time (s)",
        "Inference time (s)",
        "Inference throughput (rows/s)",
        "seeds",
        "selected features",
    ]
    rows = []
    for r in records:
        rows.append(
            [
                r["arm"],
                r["k"],
                "yes" if r["reduced"] else "no (control)",
                f"{r['auc']:.4f}"
                + (f" ± {r['auc_std']:.4f}" if r.get("auc_std") else ""),
                f"{r['det']:.3f}"
                + (f" ± {r['det_std']:.3f}" if r.get("det_std") else ""),
                f"{r['fa']:.3f}" + (f" ± {r['fa_std']:.3f}" if r.get("fa_std") else ""),
                f"{r['det'] - r['fa']:+.3f}",
                f"{r['fit_s']:.2f}"
                + (f" ± {r['fit_s_std']:.2f}" if r.get("fit_s_std") else ""),
                f"{r['inf_s']:.3f}"
                + (f" ± {r['inf_s_std']:.3f}" if r.get("inf_s_std") else ""),
                f"{r['rows_per_s']:,.0f}",
                r["seeds"],
                r["features"],
            ]
        )

    C.emit(
        "table_4_11c_beth_feature_reduction",
        "Table 4.11(c) — BETH: quality, train time and inference time vs feature reduction",
        header,
        rows,
        note=(
            f"Every arm at every feature count, on BETH's shipped splits: fit on the "
            f"full {meta['n_train']:,}-row benign training split, threshold calibrated on "
            f"the {meta['n_val']:,}-row benign validation split at a {FA_BUDGET:.1%} "
            f"false-alarm budget, scored on the {meta['n_test']:,}-row test split. "
            f"**k={n_feat} is the no-reduction control** — the same sweep at its endpoint, "
            f"not a separate code path — so every 'with reduction' row has its 'without' "
            f"row in the same table. Feature reduction is "
            f"`TribbleOneClassDetector(feature_selection='variance', top_n=k)`, an "
            f"unsupervised filter (with one class there is no separation to rank on); the "
            f"selected names are read off the fitted detector and **the baselines are "
            f"fitted on exactly those columns**, so at each k all four arms see identical "
            f"input. Selecting per arm would have made the curves incomparable. "
            f"**One calibration for all four arms**: the threshold is the "
            f"{1 - FA_BUDGET:.3f} quantile of each arm's own scores on the benign "
            f"validation split. This deliberately differs from Table 4.11, which matched "
            f"baselines by `contamination`/`nu` — that leans on each library's own "
            f"threshold placement, and Isolation Forest's `contamination` sets the cut on "
            f"the *training* score distribution, which is why its detection rate reads "
            f"near zero there. ROC-AUC is threshold-free and identical under either "
            f"convention. Inference time is the scoring of the full test split, timed "
            f"separately from the fit because the two costs are paid at different times; "
            f"throughput is rows/second at that measurement. "
            f"**DO NOT COMPARE THE TRAIN-TIME COLUMN ACROSS ARMS IN THIS TABLE.** Each arm "
            f"here trains on its own protocol's sample count — the fuzzy arms on all "
            f"{meta['n_train']:,} rows, the one-class SVM on a {OCSVM_TRAIN_CAP:,}-row "
            f"subsample (libsvm is O(n²)–O(n³)), and Isolation Forest on "
            f"{meta['n_train']:,} rows nominally but with `max_samples=256`, so each tree "
            f"is built from 256. Those are three different amounts of work, so the seconds "
            f"are three different experiments and a cross-arm reading of them is invalid; "
            f"the flat Isolation Forest train-time line in particular is that default, not "
            f"a property of the algorithm. **Table 4.11(d) "
            f"(`table_4_11d_beth_sample_scaling.py`) is the matched-sample-count "
            f"comparison** — same rows, same count, every arm — and it is the one to quote "
            f"for timing. Within a single arm the trend across feature count is still "
            f"valid, which is what this column is for. The fuzzy arms are deterministic "
            f"given the fixed splits and are fitted once (no ±); the baseline rows are "
            f"mean ± std across common.SEEDS ({C.SEEDS})."
        ),
    )


if __name__ == "__main__":
    main()
