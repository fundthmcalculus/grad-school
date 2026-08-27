"""Table 4.11 -- BETH host-telemetry anomaly detection, one-class protocol.

BETH is the large-scale anomaly-detection set Chapter 4 §4.3.5 names as the
motivating case for the complement rule: train on benign host telemetry, then be
shown novel attack behaviour. Table 4.4/4.7 could not run it -- the BETH files
were absent, and the note in `table_4_4_openset.py` says so -- and its
leave-one-class-out protocol needs >=3 classes, which a binary set does not have.
This generator runs the protocol BETH actually supports, on the full set.

WHAT THE SPLITS ARE, AND WHY THIS IS NOT A SUPERVISED TABLE
-----------------------------------------------------------
grad-school #95 specifies "train on the training split, report AUC vs. RF /
ANFIS baselines." That is not implementable, and the reason is a property of the
dataset rather than of the harness. Counting `evil` per shipped split:

    train  763,144 rows   evil=1:       0      (100.0% benign)
    val    188,967 rows   evil=1:       0      (100.0% benign)
    test   188,967 rows   evil=1: 158,432      ( 83.8% anomalous)

**Every positive in BETH is in the test split.** A supervised classifier fitted
on the training split sees one class and cannot learn a decision boundary at all;
scikit-learn's Random Forest fits it happily and predicts the constant 0, which
scores 16.2% accuracy and AUC 0.5 on test while reporting no error. So the
supervised arms are marked N/A **with the reason in the cell**, which is the
honest answer; they are not quietly omitted, because an absent row reads as "not
tried."

No arm in this table is fitted on test-split labels. There is a supervised
feature-separability diagnostic -- it necessarily trains on the only split that
has positives -- and it is printed to stdout and kept out of the emitted table on
purpose; `_separability_probe` documents why at length.

This is not a defect in the dataset. BETH ships this way on purpose: it is a
one-class benchmark, and the benign-only validation split is exactly what you
need to calibrate a false-alarm budget without ever touching a positive label.
That is the protocol below.

TWO FEATURES ARE DROPPED, AND ONE OF THEM IS A LABEL
----------------------------------------------------
`_fuzzy_models.load_beth()` takes X as "every numeric column but the last",
which keeps `sus` -- BETH's *heuristic suspicion label*, shipped alongside
`evil` as the second of two annotations. On the test split it is not merely
correlated with the target, it contains it:

    sus=1 for 158,432 / 158,432 evil rows (100.0%), and for 13,027 benign rows

so `sus` alone is a detector at 1.000 detection / 0.427 false alarm, and any arm
given it is scoring the annotator, not the telemetry. `timestamp` is a
within-capture session clock; it separates the three files, not the behaviour.
Both are dropped by name here, which is what
`FuzzySystemsExperiments/beth-anomaly.py` -- the script Chapter 4's BETH
discussion is written from -- has always done. Eight features remain.

The drop is done HERE rather than in `load_beth()` on purpose: `load_beth` is
also called by `table_4_4_openset.py`, and narrowing a shared loader's return
would silently change what an already-archived generator measures. AGENTS.md
non-negotiable 4.

PROTOCOL
--------
  fit        benign training split (763,144 rows), UnitScalar fitted on it alone
  calibrate  theta chosen on the benign VALIDATION split as the grid value with
             the highest false-alarm rate still inside the budget
             (REPRO_BETH_FA_BUDGET, default 0.01). No positive label is touched.
  test       detection rate on the 158,432 anomalous rows, false-alarm rate on
             the 30,535 benign rows, Youden's J, and ROC-AUC.

  Baselines get the SAME calibration treatment: contamination is set to the
  complement rule's realized validation false-alarm rate, so the arms sit at a
  matched operating point rather than at whatever default each ships with. This
  mirrors `table_4_4_openset.py` except that the matching happens on validation
  data, which is what having a validation split buys.

SEEDS. The complement rule, its theta calibration and the scaler are
deterministic given the three fixed files -- there is no train_test_split to
reseed, so a ten-seed loop over them would report ten identical numbers and a
+/-0.000 that means "no randomness", not "no spread". They are therefore fitted
ONCE, and `common.SEEDS` governs only the genuinely stochastic arms (Isolation
Forest's trees, the one-class SVM's subsample, and the stdout separability
probe's split). Which cells carry a seed spread and which are exact is stated in
the note.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_4_11_beth_anomaly.py
"""

from __future__ import annotations

import os
import sys
import warnings

# --- thread caps, set BEFORE numpy/sklearn import ---------------------------
#
# These are load-bearing on this workstation, not tidiness. The host has 32
# logical cores and its OpenBLAS is compiled for fewer ("precompiled NUM_THREADS
# exceeded" on every run). With `n_jobs=-1` inside a ten-seed loop, joblib's
# loky backend SPAWNS 32 interpreters on Windows and each one builds its own BLAS
# pool: ~32x32 threads contending for 32 cores. The observed result was the whole
# machine hanging and the process dying with SIGSEGV (rc=139) at a
# nondeterministic seed -- never a Python traceback, because the fault is in
# native thread setup, not in this script.
#
# It was never a memory problem: BETH's three frames are 91 MB combined on a
# 95.6 GB host. Diagnosing it as one would have led to downsampling the training
# split, which is exactly what grad-school #95 says not to do.
#
# The vars must be set before the first numpy import -- BLAS reads them at load
# time -- which is why they sit above the imports rather than in main().
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("REPRO_BLAS_THREADS", "8"))

import numpy as np  # noqa: E402 -- must follow the thread caps above
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest, RandomForestClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit, train_test_split  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import _fuzzy_models as F  # noqa: E402

# Columns that are annotations or split identifiers, not telemetry. See the
# module docstring -- `sus` is a label and dropping it is not optional.
LEAKY_COLUMNS = ["sus", "timestamp"]

CONORM = os.environ.get("REPRO_ANOM_CONORM", "hamacher")

# The theta grid the calibration picks from, and the grid the operating curve is
# reported on. It brackets both the value `beth-anomaly.py` inherited (0.99) and
# the band Table 4.6's sweep found usable (0.5-0.8), so the curve can be read
# against both rather than only confirming one.
THETA_GRID = [
    float(t)
    for t in os.environ.get(
        "REPRO_BETH_THETAS", "0.5,0.6,0.7,0.8,0.9,0.95,0.99,0.999"
    ).split(",")
]

# False-alarm budget the operating point is calibrated to, on validation.
FA_BUDGET = float(os.environ.get("REPRO_BETH_FA_BUDGET", "0.01"))

# libsvm's fit is O(n^2)-O(n^3); 763k rows is not a fit, it is a hang. Same cap
# and same reasoning as table_4_4_openset.py, recorded in the note rather than
# left silent. The complement rule and Isolation Forest see all 763,144 rows.
OCSVM_TRAIN_CAP = int(os.environ.get("REPRO_OCSVM_TRAIN_CAP", "20000"))

# Isolation Forest's default max_samples=256 already makes its fit cheap on
# 763k rows; this exists so a smoke run can shrink the forest, not the data.
IF_TREES = int(os.environ.get("REPRO_BETH_IF_TREES", "200"))

# Worker count for the sklearn ensembles. Deliberately NOT -1 -- see the thread
# note at the top of this file. Eight workers saturate the useful parallelism of
# a 200-tree forest on 8 features while leaving the machine usable; -1 on this
# 32-core host hung it. Raise with REPRO_BETH_N_JOBS if a host can take it.
N_JOBS = int(os.environ.get("REPRO_BETH_N_JOBS", "8"))


def _rates(flagged, is_anom):
    """(detection rate on anomalies, false-alarm rate on benign)."""
    det = float(flagged[is_anom].mean()) if is_anom.any() else float("nan")
    fa = float(flagged[~is_anom].mean()) if (~is_anom).any() else float("nan")
    return det, fa


def _trapz(ys, xs):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(ys, xs))


def _sweep_auc(points):
    """Trapezoidal ROC-AUC from a set of measured (false-alarm, detection) points.

    The complement rule emits a hard label, not a score, so there is no
    `roc_auc_score` to call on it. Sweeping theta traces its ROC directly, and
    the area under the resulting operating points is the same quantity -- but
    computed from ~8 points rather than from every threshold, so it is reported
    under its own column heading and never presented as interchangeable with the
    baselines' full-resolution AUC. The (0,0) and (1,1) endpoints are appended
    because a detector that flags nothing and one that flags everything are both
    reachable by construction, not by measurement.
    """
    pts = sorted(set([(0.0, 0.0), *points, (1.0, 1.0)]))
    return _trapz([p[1] for p in pts], [p[0] for p in pts])


def load_splits():
    """BETH's three shipped splits, leak-free and scaled. None if unavailable.

    Returns (X_train, X_val, X_test, is_anom_val, is_anom_test, meta).
    """
    splits = F.load_beth()
    if splits is None:
        return None

    # The training y is intentionally unused past the positive-count report
    # below: it is constant 0, which is the whole premise of this table.
    (Xtr, _ytr), (Xva, yva), (Xte, yte) = (  # noqa: F841 -- _ytr is constant 0
        splits["train"],
        splits["val"],
        splits["test"],
    )

    dropped = [c for c in LEAKY_COLUMNS if c in Xtr.columns]
    Xtr, Xva, Xte = (
        Xtr.drop(columns=dropped),
        Xva.drop(columns=dropped),
        Xte.drop(columns=dropped),
    )
    print(
        f"  [beth] dropped {dropped} (annotation / split clock); "
        f"{Xtr.shape[1]} features left"
    )

    n_pos = {k: int(np.asarray(v[1]).sum()) for k, v in splits.items()}
    print(
        "  [beth] positives per split: "
        + ", ".join(
            f"{k}={n_pos[k]}/{len(splits[k][0])}" for k in ("train", "val", "test")
        )
    )
    if n_pos["train"] > 0:
        # A future BETH release with positives in train would make the supervised
        # arms legitimate, and this table would then be understating them. Say so
        # rather than silently keeping the N/A.
        print(
            "  [beth] NOTE: the training split now contains positives -- the "
            "supervised N/A rows in this table are no longer justified and the "
            "protocol should be revisited."
        )

    # UnitScalar fitted on TRAIN ONLY and applied to all three. Fitting per file
    # would scale each split by its own min/max, which is the exact silent bug
    # `FuzzySystemsExperiments/beth-anomaly.py` documents having had.
    sc = F._scaler("unit", 2).set_output(transform="pandas").fit(Xtr)
    print(f"  [beth] auto-logged features: {list(sc.log_features_)}")
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    # How many DISTINCT feature vectors the test split actually contains. This
    # is not a curiosity: system-call telemetry repeats itself, and if the
    # repetition is heavy then any within-test-split supervised reference is
    # scoring rows whose exact feature vector it was trained on. Measured, not
    # assumed, because the answer decides whether that reference means anything.
    # `pd.factorize` over a MultiIndex of the rows, NOT a per-row string join:
    # the join built ~1.5M interned Python strings and peaked at 341 MB
    # (measured) to answer a question that factorize answers in 72 MB.
    groups, uniques = pd.factorize(pd.MultiIndex.from_frame(Xte))
    print(
        f"  [beth] test split: {len(Xte):,} rows but only {len(uniques):,} distinct "
        f"feature vectors ({len(uniques) / len(Xte):.2%})"
    )

    meta = {
        "n_train": len(Xtr),
        "n_val": len(Xva),
        "n_test": len(Xte),
        "n_feat": Xtr.shape[1],
        "dropped": dropped,
        "logged": list(sc.log_features_),
        "n_pos": n_pos,
        "n_distinct_test": len(uniques),
        "test_groups": groups,
    }
    return Xtr, Xva, Xte, np.asarray(yva) == 1, np.asarray(yte) == 1, meta


def _build_complement_model(Xtr):
    """Fit the MoG membership base on benign-only data, anomaly rule enabled.

    Single-class fit is the point, not a degenerate case: the complement rule's
    whole claim is that the fuzzy complement of the known-class aggregate is a
    detector for whatever is not in the base. With one known class the
    between-class differentiator scores are all identically zero, so the top-n
    screen cannot rank features and every feature is kept (top_p=1.0) -- the
    same choice `beth-anomaly.py` makes, and the honest one here, since a screen
    with nothing to discriminate on would otherwise pick an arbitrary subset.
    """
    from tribblefis.gauss_data import AnomalyParameters
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        take_top_features,
    )

    y = pd.Series("regular", index=Xtr.index, name="y_value")
    diffs = calculate_gaussian_correlation(Xtr, y)
    _, top_vars = take_top_features(diffs, top_p=1.0)
    memb = create_gaussian_membership_dict(Xtr, y, top_n_var_names=top_vars)
    params = AnomalyParameters(
        include_anomaly=True,
        threshold=THETA_GRID[0],  # overridden per-theta by the sweep
        label="anomaly",
        norm_conorm=CONORM,
        member_function="gaussian",
    )
    return memb.to_simple_model(params), top_vars


def _flag_sweep(model, X, thetas):
    """{theta: bool 'flagged anomalous'} in one class-firing pass.

    theta enters only at the anomaly step, so the class firing is computed once
    and reused across the grid -- the same reuse `table_4_4_openset.py` verified
    bit-identical to a theta-at-a-time loop.
    """
    from tribblefis.gauss_math import simple_gaussian_predict_sweep

    preds = simple_gaussian_predict_sweep(X, model, list(thetas))
    return {
        th: np.asarray([str(p) == "anomaly" for p in preds[th]], dtype=bool)
        for th in thetas
    }


def _calibrate(val_flags, is_anom_val, budget):
    """Pick theta on validation under a false-alarm budget.

    The validation split is 100% benign, so there is no detection rate to
    maximize on it and nothing resembling J to tune -- the only thing it can
    measure is the false-alarm rate, which is precisely what a benign-only
    holdout is for. Among the thetas that stay inside the budget, take the one
    with the HIGHEST false alarm: within a fixed budget, more flagging is more
    sensitivity, and picking the quietest theta would throw away detection the
    budget had already paid for. If none fits, fall back to the single quietest
    theta and say so -- the alternative is emitting a calibrated-looking cell
    that never met its budget.
    """
    fa = {th: _rates(f, is_anom_val)[1] for th, f in val_flags.items()}
    ok = {th: v for th, v in fa.items() if np.isfinite(v) and v <= budget}
    if ok:
        theta = max(ok, key=lambda t: ok[t])
        return theta, fa[theta], True, fa
    theta = min(fa, key=lambda t: fa[t] if np.isfinite(fa[t]) else np.inf)
    print(
        f"  [calibrate] no theta meets the {budget:.1%} validation false-alarm "
        f"budget; quietest is theta={theta} at {fa[theta]:.3f} -- reported as "
        f"UNCALIBRATED"
    )
    return theta, fa[theta], False, fa


def _separability_probe(Xte, is_anom_te, meta):
    """Feature-separability diagnostic. STDOUT ONLY -- never a table row.

    THIS IS NOT A BASELINE AND MUST NOT BECOME ONE. It fits a Random Forest on
    half the test split and scores it on the other half, so it trains on labels
    from the only split that has any. That is not a held-out measurement of
    anything, and it is not comparable to the one-class arms in the table, which
    never see a positive.

    Two reasons it stays out of the emitted table rather than going in with a
    caveat:

    1. BETH's test split holds `n_test` rows drawn from only ~`n_distinct_test`
       distinct feature vectors (~2%), so an i.i.d. row split trains on a copy of
       nearly every row it then scores. That arm reports ~1.000 AUC. It is a
       lookup table succeeding, not a model generalizing. The grouped variant
       (GroupShuffleSplit on feature-vector identity, so no vector straddles the
       split) removes that specific inflation -- but both halves still come from
       one capture, the same hosts and the same session, so even the grouped
       number is an in-capture resubstitution estimate, not a generalization one.
    2. A cell that is only safe to read alongside a long note is a cell that
       gets quoted without the note. `PROVENANCE_MAP.md` records this repository
       doing exactly that more than once, including a generator that emitted one
       dataset's table under another's filename. The N/A rows with their reason
       are the honest answer to grad-school #95's request for RF/ANFIS arms;
       this probe only answers the softer question of whether the eight
       surviving features separate at all once labels exist, and printing it is
       sufficient for that.

    Returns the grouped result dict for the caller's logging, and nothing enters
    `rows`.
    """
    print(
        "\n  [probe] feature separability -- DIAGNOSTIC ONLY, not a table row and "
        "not a baseline."
    )
    print(
        f"  [probe] fits on labels from the test split itself; the "
        f"{meta['n_test']:,} test rows hold only {meta['n_distinct_test']:,} "
        f"distinct feature vectors "
        f"({meta['n_distinct_test'] / meta['n_test']:.2%}), so the i.i.d. variant "
        f"is a lookup table and the grouped variant is still within one capture."
    )

    results = {}
    for tag, groups in (("grouped", meta["test_groups"]), ("i.i.d.", None)):
        out = {"det": [], "fa": [], "auc": []}
        for seed in C.SEEDS:
            if groups is None:
                Xa, Xb, ya, yb = train_test_split(
                    Xte,
                    is_anom_te,
                    test_size=0.5,
                    random_state=seed,
                    stratify=is_anom_te,
                )
            else:
                gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
                ia, ib = next(gss.split(Xte, is_anom_te, groups=groups))
                Xa, Xb = Xte.iloc[ia], Xte.iloc[ib]
                ya, yb = is_anom_te[ia], is_anom_te[ib]
            if len(np.unique(ya)) < 2 or len(np.unique(yb)) < 2:
                # A grouped split can land every positive on one side; that is
                # not a number, so drop the seed rather than report it.
                print(f"    [probe:{tag}] seed {seed}: single-class split, skipped")
                continue
            rf = RandomForestClassifier(
                n_estimators=200, random_state=seed, n_jobs=N_JOBS
            ).fit(Xa, ya)
            d, f = _rates(rf.predict(Xb).astype(bool), np.asarray(yb, dtype=bool))
            out["det"].append(d)
            out["fa"].append(f)
            try:
                out["auc"].append(roc_auc_score(yb, rf.predict_proba(Xb)[:, 1]))
            except Exception:  # noqa: BLE001
                pass
        if out["det"]:
            dm, _ = C.agg(out["det"])
            fm, _ = C.agg(out["fa"])
            am, _ = C.agg(out["auc"]) if out["auc"] else (None, None)
            print(
                f"    [probe:{tag}] detection={dm:.3f} false-alarm={fm:.3f} "
                f"J={dm - fm:+.3f}"
                + (f" auc={am:.3f}" if am is not None else "")
                + f"  (n={len(out['det'])} seeds)"
            )
        results[tag] = out
    return results.get("grouped", {})


def main():
    print("Table 4.11 -- BETH anomaly detection (one-class, full 763k training split)")
    got = load_splits()
    if got is None:
        print("  BETH unavailable -- nothing emitted")
        return
    Xtr, Xva, Xte, is_anom_va, is_anom_te, meta = got
    print(
        f"  train={meta['n_train']:,} val={meta['n_val']:,} test={meta['n_test']:,} "
        f"features={meta['n_feat']} test positives={int(is_anom_te.sum()):,}"
    )

    arms: dict = {}

    def add(arm, det, fa, auc=None, secs=None):
        a = arms.setdefault(arm, {"det": [], "fa": [], "auc": [], "t": []})
        if np.isfinite(det) and np.isfinite(fa):
            a["det"].append(det)
            a["fa"].append(fa)
        if auc is not None and np.isfinite(auc):
            a["auc"].append(auc)
        if secs is not None:
            a["t"].append(secs)

    # ---- complement rule: deterministic, fitted once, calibrated on validation
    theta = None
    val_fa = float("nan")
    calibrated = False
    curve = []
    try:
        with C.timed() as t_fit:
            model, _top_vars = _build_complement_model(Xtr)
        print(
            f"  [complement] fitted on {meta['n_train']:,} benign rows "
            f"in {t_fit.seconds:.1f}s"
        )

        val_flags = _flag_sweep(model, Xva, THETA_GRID)
        theta, val_fa, calibrated, val_fa_all = _calibrate(
            val_flags, is_anom_va, FA_BUDGET
        )
        print(
            f"  [complement] theta={theta} chosen on validation "
            f"(false alarm {val_fa:.4f}, budget {FA_BUDGET:.1%})"
        )

        test_flags = _flag_sweep(model, Xte, THETA_GRID)
        for th in THETA_GRID:
            d, f = _rates(test_flags[th], is_anom_te)
            curve.append((th, val_fa_all[th], d, f))
        sweep_auc = _sweep_auc([(f, d) for _, _, d, f in curve])

        d, f = _rates(test_flags[theta], is_anom_te)
        add("**Complement rule (this work)**", d, f, auc=sweep_auc, secs=t_fit.seconds)
        target_fa = val_fa
    except Exception as exc:  # noqa: BLE001
        print(f"  [complement] failed ({exc.__class__.__name__}: {exc}) -> N/A")
        arms.setdefault(
            "**Complement rule (this work)**", {"det": [], "fa": [], "auc": [], "t": []}
        )
        target_fa = FA_BUDGET

    # Baselines are matched to the complement rule's VALIDATION false-alarm rate,
    # clipped to the range sklearn accepts for `contamination` / `nu`.
    cont = float(
        min(max(target_fa if np.isfinite(target_fa) else FA_BUDGET, 1e-4), 0.5)
    )
    print(f"  [baselines] contamination/nu matched to {cont:.4f}")

    for seed in C.SEEDS:
        ocsvm_idx = np.random.RandomState(seed).choice(
            len(Xtr), min(OCSVM_TRAIN_CAP, len(Xtr)), replace=False
        )
        for arm, est, Xfit in (
            (
                "Isolation Forest",
                IsolationForest(
                    contamination=cont,
                    n_estimators=IF_TREES,
                    random_state=seed,
                    n_jobs=N_JOBS,
                ),
                Xtr,
            ),
            ("One-class SVM", OneClassSVM(nu=cont, gamma="scale"), Xtr.iloc[ocsvm_idx]),
        ):
            try:
                with C.timed() as t:
                    est.fit(Xfit)
                d, f = _rates(est.predict(Xte) == -1, is_anom_te)
                auc = None
                try:
                    # -score_samples is "more anomalous = larger", which is the
                    # orientation roc_auc_score needs against is_anom.
                    auc = roc_auc_score(is_anom_te, -est.score_samples(Xte))
                except Exception:  # noqa: BLE001
                    pass
                add(arm, d, f, auc=auc, secs=t.seconds)
                print(
                    f"    [{arm}] seed {seed}: det={d:.3f} fa={f:.3f} "
                    f"auc={auc if auc is None else round(auc, 3)} ({t.seconds:.1f}s)"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"    [{arm}] seed {seed}: {exc.__class__.__name__}: {exc}")

    # ---- headline table -----------------------------------------------------
    order = ["**Complement rule (this work)**", "Isolation Forest", "One-class SVM"]
    rows = []
    for arm in order:
        v = arms.get(arm)
        if not v or not v["det"]:
            rows.append([arm, C.NA, C.NA, C.NA, C.NA, C.NA])
            continue
        dm, _ = C.agg(v["det"])
        fm, _ = C.agg(v["fa"])
        rows.append(
            [
                arm,
                C.cell(v["det"]),
                C.cell(v["fa"]),
                f"{dm - fm:+.3f}",
                C.cell(v["auc"]) if v["auc"] else C.NA,
                C.cell(v["t"], fmt="{:.1f}") + " s" if v["t"] else C.NA,
            ]
        )

    # The supervised arms the issue asked for, marked with WHY they are N/A
    # rather than silently omitted -- an absent row reads as "not tried".
    untrainable = (
        f"N/A -- untrainable (0 positives in the "
        f"{meta['n_train']:,}-row training split)"
    )
    rows.append(
        [
            "Random Forest (supervised, BETH protocol)",
            untrainable,
            C.NA,
            C.NA,
            C.NA,
            C.NA,
        ]
    )
    rows.append(
        ["ANFIS (supervised, BETH protocol)", untrainable, C.NA, C.NA, C.NA, C.NA]
    )

    # Separability probe -- STDOUT ONLY, deliberately not a row. See
    # `_separability_probe` for why it is not allowed into the table.
    _separability_probe(Xte, is_anom_te, meta)

    cal_txt = (
        f"theta={theta} calibrated on the benign validation split at a "
        f"{FA_BUDGET:.1%} false-alarm budget (realized {val_fa:.4f})"
        if calibrated
        else f"theta={theta} is the quietest grid value and MISSED the "
        f"{FA_BUDGET:.1%} validation budget at {val_fa:.4f} -- this operating "
        f"point is uncalibrated"
    )

    # The gap between the calibrated validation false-alarm rate and the one
    # actually realized on test is the most load-bearing caveat in this table, so
    # it is computed and stated rather than left for a reader to divide out.
    cr = arms.get("**Complement rule (this work)**") or {"fa": []}
    transfer = ""
    if cr["fa"]:
        test_fa = cr["fa"][0]
        if np.isfinite(val_fa) and val_fa > 0:
            transfer = (
                f" **The calibration does not transfer**: the θ that costs "
                f"{val_fa:.4f} false alarms on validation costs {test_fa:.4f} on "
                f"test, {test_fa / val_fa:.1f}× more. Both splits are benign-only "
                f"draws from the same capture, so this is a property of BETH's "
                f"benign test rows, not of the rule -- and it means a false-alarm "
                f"budget set on BETH's validation split cannot be believed on its "
                f"test split. Every arm here is matched at the validation rate, so "
                f"they are affected alike and remain comparable to each other."
            )
    C.emit(
        "table_4_11_beth_anomaly",
        "Table 4.11 — BETH host-telemetry anomaly detection (one-class)",
        [
            "Method",
            "Detection rate",
            "False-alarm rate",
            "Detection − false alarm",
            "ROC-AUC",
            "Train time",
        ],
        rows,
        note=(
            f"BETH's shipped splits, used as shipped: fit on the full "
            f"{meta['n_train']:,}-row training split, operating point calibrated on the "
            f"{meta['n_val']:,}-row validation split, scored on the "
            f"{meta['n_test']:,}-row test split ({int(is_anom_te.sum()):,} anomalous / "
            f"{int((~is_anom_te).sum()):,} benign). No downsampling of the training "
            f"set. **This is a one-class table because BETH is a one-class benchmark: "
            f"all {meta['n_pos']['test']:,} positives are in the test split and the "
            f"training and validation splits are 100% benign.** The supervised rows are "
            f"therefore N/A with the reason given, not omitted. **No arm in this table "
            f"is fitted on test-split labels.** A supervised feature-separability "
            f"diagnostic is printed to stdout by the generator and deliberately kept out "
            f"of this table: it would have to train on the only split that has "
            f"positives, and the test split's {meta['n_test']:,} rows hold just "
            f"{meta['n_distinct_test']:,} distinct feature vectors "
            f"({meta['n_distinct_test'] / meta['n_test']:.2%}), so any within-test split "
            f"is a lookup table or, grouped, an in-capture resubstitution estimate — "
            f"neither is a number that belongs beside the one-class arms.{transfer} "
            f"Features: {meta['n_feat']} numeric columns after dropping "
            f"{meta['dropped']} — `sus` is BETH's second *label* (1 for 100% of evil "
            f"rows) and `timestamp` is a per-capture session clock that separates the "
            f"files rather than the behaviour; both are dropped in "
            f"`FuzzySystemsExperiments/beth-anomaly.py` too. UnitScalar fitted on train "
            f"alone, auto-logging {meta['logged']}. {cal_txt}; {CONORM} conorm. "
            f"Baselines' contamination/nu is matched to the complement rule's validation "
            f"false-alarm rate ({cont:.4f}) so the arms sit at a matched operating point. "
            f"'Detection − false alarm' is Youden's J. The complement rule emits a hard "
            f"label, so its ROC-AUC is the trapezoidal area under its measured theta "
            f"sweep ({len(THETA_GRID)} points, endpoints appended) and is coarser than "
            f"the baselines' score-based AUC — read it as an operating-curve summary, "
            f"not as an interchangeable number. The complement rule and its calibration "
            f"are deterministic given the fixed splits and are fitted once, so their "
            f"cells carry no ±; the ± on the baseline and reference rows is across "
            f"common.SEEDS ({C.SEEDS}). One-class SVM is fitted on a random "
            f"{OCSVM_TRAIN_CAP:,}-row subsample per seed (libsvm is O(n²)–O(n³); "
            f"{meta['n_train']:,} rows is not a fit), while the complement rule and "
            f"Isolation Forest see the full training split."
        ),
    )

    # ---- operating curve ---------------------------------------------------
    if curve:
        print("\n  theta operating curve:")
        print(
            f"    {'theta':>8} {'val FA':>10} {'detection':>12} "
            f"{'test FA':>10} {'J':>8}"
        )
        crows = []
        for th, vfa, d, f in curve:
            print(f"    {th:8.3f} {vfa:10.4f} {d:12.3f} {f:10.4f} {d - f:+8.3f}")
            crows.append(
                [
                    f"{th:.3f}",
                    f"{vfa:.4f}",
                    f"{d:.3f}",
                    f"{f:.4f}",
                    f"{d - f:+.3f}",
                    "**chosen**" if th == theta else "",
                ]
            )
        C.emit(
            "table_4_11_beth_theta_sweep",
            "Table 4.11(b) — BETH complement-rule operating curve vs. the boost θ",
            [
                "θ",
                "validation false-alarm rate",
                "test detection rate",
                "test false-alarm rate",
                "detection − false alarm",
                "chosen",
            ],
            crows,
            note=(
                f"One model, fitted once on the benign training split; θ enters only at "
                f"the anomaly step, so the whole curve is one class-firing pass. The "
                f"validation column is what the operating point is chosen on — it is "
                f"measurable without any positive label, which is the entire value of "
                f"BETH's benign-only validation split. The test columns are the "
                f"consequence of that choice and are NOT used to make it. The marked row "
                f"is the θ the {FA_BUDGET:.1%} budget selected. Deterministic: no seed "
                f"averaging, and none is implied."
            ),
        )


if __name__ == "__main__":
    main()
