"""Table 4.11 -- BETH host-telemetry anomaly detection, one-class protocol.

BETH is the large-scale anomaly-detection set Chapter 4 §4.3.5 names as the
motivating case for the complement rule: train on benign host telemetry, then be
shown novel attack behaviour. Table 4.4/4.7 could not run it -- its
leave-one-class-out protocol needs >=3 classes and BETH is binary. This generator
runs the protocol BETH actually supports, on the full set.

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
scores 16.2% accuracy and AUC 0.5 on test while reporting no error.

So there are no supervised arms here, and no placeholder rows for them either.
**The Random-Forest family's answer to this problem is Isolation Forest**, which
is in the table: a forest of random trees, fitted with no labels, scoring by
expected isolation depth. It is the tree-ensemble counterpart of the one-class
SVM, and it is the right RF-shaped comparison for a one-class task. (A literal
`RandomForestClassifier` can be pressed into one-class service by the
Shi-Horvath synthetic-contrast trick -- train it to separate the real data from
a sample of the product of its marginals, then read normality off the predicted
probability. That is a genuinely different estimator, not a baseline this table
needs, and it is deliberately not implemented here.)

No arm in this table is fitted on test-split labels. There is a supervised
feature-separability diagnostic -- it necessarily trains on the only split that
has positives -- and it is printed to stdout and kept out of the emitted table on
purpose; `_separability_probe` documents why at length.

WHICH DETECTOR, AND WHY NOT THE HAND-ASSEMBLED ONE
--------------------------------------------------
The fuzzy arm is `tribblefis.one_class.TribbleOneClassDetector` -- the library's
own one-class estimator, an sklearn `OutlierMixin` with `score_samples` /
`decision_function` / `predict`.

An earlier version of this generator assembled the same thing by hand out of
`create_gaussian_membership_dict` + `AnomalyParameters` + `simple_gaussian_predict`,
i.e. by bending the *multi-class* path into single-class use. The two agree
exactly on the operating point (det 0.9930 / false alarm 0.1502 either way), so
this is not a correction of a wrong number -- but a second copy of a library
capability is how two tables silently drift apart, and the hand-rolled version
could only emit a hard label, which cost this table a real ROC-AUC. See the
saturation note below for what that concealed.

THE COMPLEMENT SCORE SATURATES ON BETH AT EIGHT FEATURES
--------------------------------------------------------
`TribbleOneClassDetector` offers `score="complement"` (`1 - max firing`, the
formulation Chapter 4 argues for and the library's default) and
`score="surprisal"` (`sum_j -log membership_j`). Under the product t-norm these
are monotone transforms of one another, so *in exact arithmetic they have
identical ROC-AUC*. Measured on BETH they do not:

    complement   AUC 0.928     resolves 1,508 distinct scores
    surprisal    AUC 0.990     resolves 3,997 distinct scores
    Spearman(complement, surprisal) = 0.812   (exact arithmetic: 1.000)

The test split contains 4,002 distinct feature vectors, so ~4,002 is the ceiling
on distinct scores any detector can produce. The surprisal reaches 3,997 of them;
the complement collapses them onto 1,508, which is where the AUC goes. (Both
scores leave most test *rows* tied at the maximum -- 75% and 85% -- but that is
BETH repeating itself, not a numerical fault, so the distinct-score count is the
diagnostic and the tied-row fraction is not.)

The library's module docstring puts the onset of this "past roughly 60 features";
BETH reaches it at **eight**, because the log-scaled process/thread identifiers
are heavy-tailed enough that a typical point's summed z^2 already exceeds
float64's resolution against 1.0. So the threshold is a property of the data's
tails, not of the feature count alone.

Consequence for this table: **surprisal is the arm to read**, and the AUC
difference between the two rows is a floating-point resolution artifact, not a
modelling result. Both are reported, because dropping the complement row would
hide a caveat that Chapter 4's own default walks into.

PROTOCOL
--------
  fit        benign training split (763,144 rows), UnitScalar fitted on it alone
  calibrate  the decision threshold is the (1 - budget) quantile of the
             detector's anomaly scores on the benign VALIDATION split
             (REPRO_BETH_FA_BUDGET, default 0.01). Exact, not grid-limited, and
             it touches no positive label -- which is the whole value of a
             benign-only validation split.
  test       detection rate on the 158,432 anomalous rows, false-alarm rate on
             the 30,535 benign rows, Youden's J, and ROC-AUC from the continuous
             score.

  Baselines get the SAME treatment: contamination/nu is set to the fuzzy arm's
  realized validation false-alarm rate, so every arm sits at a matched operating
  point rather than at whatever default it ships with. This mirrors
  `table_4_4_openset.py`, except that the matching happens on validation data.

SEEDS. The fuzzy detector, its calibration and the scaler are deterministic given
the three fixed files -- there is no train_test_split to reseed, so a ten-seed
loop over them would report ten identical numbers and a +/-0.000 that means "no
randomness", not "no spread". They are fitted ONCE, and `common.SEEDS` governs
only the genuinely stochastic arms (Isolation Forest's trees, the one-class SVM's
subsample, the stdout probe's split). Which cells carry a seed spread and which
are exact is stated in the note.

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
# 95.6 GB host, and the measured peak resident set is 521 MB. Diagnosing it as
# memory pressure would have led to downsampling the training split, which is
# exactly what grad-school #95 says not to do.
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

# Columns that are annotations or split identifiers, not telemetry. `sus` is
# BETH's *second label* -- it is 1 for 158,432 of 158,432 evil rows, so alone it
# detects at 1.000/0.427 and any arm given it is scoring the annotator rather
# than the telemetry. `timestamp` is a per-capture session clock whose ranges
# separate the three files rather than the behaviour. Both are dropped by
# `FuzzySystemsExperiments/beth-anomaly.py` too -- the script Chapter 4's BETH
# discussion is written from.
#
# The drop is done HERE rather than in `load_beth()` on purpose: `load_beth` is
# also called by `table_4_4_openset.py`, and narrowing a shared loader's return
# would silently change what an already-archived generator measures. AGENTS.md
# non-negotiable 4.
LEAKY_COLUMNS = ["sus", "timestamp"]

CONORM = os.environ.get("REPRO_ANOM_CONORM", "probability")

# False-alarm budget the headline operating point is calibrated to, on validation.
FA_BUDGET = float(os.environ.get("REPRO_BETH_FA_BUDGET", "0.01"))

# Budgets the operating curve is reported at. This is the curve a user actually
# picks a threshold on, and every point is chosen using benign data only.
FA_SWEEP = [
    float(b)
    for b in os.environ.get(
        "REPRO_BETH_FA_SWEEP", "0.001,0.005,0.01,0.02,0.05,0.10"
    ).split(",")
]

# The two aggregation modes, reported side by side. `surprisal` is the one to
# read on BETH; `complement` is Chapter 4's formulation and the library default,
# kept so the saturation artifact is visible rather than argued about.
SCORES = ["surprisal", "complement"]

# libsvm's fit is O(n^2)-O(n^3); 763k rows is not a fit, it is a hang. Same cap
# and same reasoning as table_4_4_openset.py, recorded in the note rather than
# left silent. The fuzzy arm and Isolation Forest see all 763,144 rows.
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
        # A future BETH release with positives in train would make supervised
        # arms legitimate, and this table would then be understating them.
        print(
            "  [beth] NOTE: the training split now contains positives -- this "
            "table's one-class-only protocol should be revisited."
        )

    # UnitScalar fitted on TRAIN ONLY and applied to all three. Fitting per file
    # would scale each split by its own min/max, which is the exact silent bug
    # `FuzzySystemsExperiments/beth-anomaly.py` documents having had.
    sc = F._scaler("unit", 2).set_output(transform="pandas").fit(Xtr)
    print(f"  [beth] auto-logged features: {list(sc.log_features_)}")
    Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

    # How many DISTINCT feature vectors the test split holds. Not a curiosity:
    # system-call telemetry repeats itself, and heavy repetition means any
    # within-test-split supervised reference is scoring rows whose exact feature
    # vector it trained on. `pd.factorize` over a row MultiIndex rather than a
    # per-row string join -- the join built ~1.5M interned strings and peaked at
    # 341 MB (measured) to answer what factorize answers in 72 MB.
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


def _fit_detector(Xtr, score):
    """The library's one-class detector on benign-only data.

    `TribbleOneClassDetector` is the sanctioned entry point for exactly this
    setting -- abundant normal data, no anomalies at fit time -- and it handles
    the single-class case the multi-class classifier cannot: its feature
    selection is unsupervised (`"all"` here, since with one class there is no
    separation to rank on) and it exposes a continuous `score_samples` instead
    of only a hard label.

    `contamination` is passed for the sake of anyone calling `predict()` on the
    returned object, but this table does NOT use it: the operating point comes
    from the validation quantile in `_calibrate`, because contamination places
    the threshold on the *training* score distribution and BETH's validation
    split is the holdout that exists to be calibrated on.
    """
    from tribblefis.one_class import TribbleOneClassDetector

    return TribbleOneClassDetector(
        score=score,
        norm_conorm=CONORM,
        feature_selection="all",
        contamination=FA_BUDGET,
        random_state=42,
    ).fit(Xtr)


def _anomaly_scores(det, X):
    """Continuous "higher means more anomalous" score.

    `score_samples` follows sklearn's convention -- higher is more NORMAL -- so
    it is negated once, here, rather than at each of the four call sites.
    """
    return -np.asarray(det.score_samples(X))


def _calibrate(scores_val, budget):
    """Threshold at the (1 - budget) quantile of the benign validation scores.

    The validation split is 100% benign, so there is no detection rate to
    maximize on it and nothing resembling J to tune -- the only thing it can
    measure is the false-alarm rate, which is precisely what a benign-only
    holdout is for. A continuous score makes this exact rather than grid-limited:
    the previous revision picked from an eight-value theta grid and could only
    land near the budget.

    Ties matter here, which is the saturation story in miniature: with 85% of the
    complement arm's scores equal, `> threshold` can realize a false-alarm rate
    well under the budget because the tied mass sits on the boundary. The
    realized rate is returned and reported rather than the requested one.
    """
    thr = float(np.quantile(scores_val, 1.0 - budget))
    realized = float((scores_val > thr).mean())
    return thr, realized


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
       dataset's table under another's filename.

    It answers only the softer question of whether the surviving features
    separate at all once labels exist, and printing it is sufficient for that.
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

    rows = []
    curve_rows = []
    target_fa = float("nan")
    sat = {}

    # ---- fuzzy one-class arms: deterministic, fitted once per score ---------
    for score in SCORES:
        try:
            with C.timed() as t_fit:
                det = _fit_detector(Xtr, score)
            s_va = _anomaly_scores(det, Xva)
            s_te = _anomaly_scores(det, Xte)
        except Exception as exc:  # noqa: BLE001
            print(f"  [tribble:{score}] failed ({exc.__class__.__name__}: {exc})")
            rows.append([f"Tribble one-class ({score})", C.NA, C.NA, C.NA, C.NA, C.NA])
            continue

        thr, realized = _calibrate(s_va, FA_BUDGET)
        flagged = s_te > thr
        d, f = _rates(flagged, is_anom_te)
        auc = roc_auc_score(is_anom_te, s_te)

        # Resolution diagnostic: what fraction of test scores are tied at the
        # maximum. This is the number that explains the two arms' AUC gap, so it
        # is measured rather than asserted.
        sat[score] = {
            "tied_at_max": float(np.mean(s_te >= s_te.max() - 1e-15)),
            "distinct": int(len(np.unique(s_te))),
            "auc": auc,
        }
        print(
            f"  [tribble:{score}] fit={t_fit.seconds:.1f}s thr@val={realized:.4f} "
            f"det={d:.3f} fa={f:.3f} auc={auc:.4f} "
            f"tied_at_max={sat[score]['tied_at_max']:.2%} "
            f"distinct_scores={sat[score]['distinct']:,}"
        )

        label = (
            f"**Tribble one-class ({score})**"
            if score == "surprisal"
            else (f"Tribble one-class ({score}, saturates — see note)")
        )
        rows.append(
            [
                label,
                f"{d:.3f}",
                f"{f:.3f}",
                f"{d - f:+.3f}",
                f"{auc:.3f}",
                f"{t_fit.seconds:.1f} s",
            ]
        )

        # The operating curve, from the same fitted model: thresholds chosen on
        # validation only, consequences read on test.
        for budget in FA_SWEEP:
            b_thr, b_real = _calibrate(s_va, budget)
            bd, bf = _rates(s_te > b_thr, is_anom_te)
            curve_rows.append(
                [
                    score,
                    f"{budget:.3f}",
                    f"{b_real:.4f}",
                    f"{bd:.3f}",
                    f"{bf:.4f}",
                    f"{bd - bf:+.3f}",
                    "**chosen**" if budget == FA_BUDGET else "",
                ]
            )

        if score == "surprisal":
            target_fa = realized

    # Baselines matched to the surprisal arm's realized VALIDATION false-alarm
    # rate, clipped to the range sklearn accepts for `contamination` / `nu`.
    cont = float(
        min(max(target_fa if np.isfinite(target_fa) else FA_BUDGET, 1e-4), 0.5)
    )
    print(f"  [baselines] contamination/nu matched to {cont:.4f}")

    arms: dict = {}
    for seed in C.SEEDS:
        ocsvm_idx = np.random.RandomState(seed).choice(
            len(Xtr), min(OCSVM_TRAIN_CAP, len(Xtr)), replace=False
        )
        for arm, est, Xfit in (
            (
                "Isolation Forest (the RF-family one-class detector)",
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
            a = arms.setdefault(arm, {"det": [], "fa": [], "auc": [], "t": []})
            try:
                with C.timed() as t:
                    est.fit(Xfit)
                d, f = _rates(est.predict(Xte) == -1, is_anom_te)
                a["det"].append(d)
                a["fa"].append(f)
                a["t"].append(t.seconds)
                try:
                    a["auc"].append(roc_auc_score(is_anom_te, -est.score_samples(Xte)))
                except Exception:  # noqa: BLE001
                    pass
                print(f"    [{arm.split(' (')[0]}] seed {seed}: det={d:.3f} fa={f:.3f}")
            except Exception as exc:  # noqa: BLE001
                print(f"    [{arm}] seed {seed}: {exc.__class__.__name__}: {exc}")

    for arm, v in arms.items():
        if not v["det"]:
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
                C.cell(v["t"], fmt="{:.1f}") + " s",
            ]
        )

    _separability_probe(Xte, is_anom_te, meta)

    sat_txt = ""
    if "surprisal" in sat and "complement" in sat:
        sc_, cp = sat["surprisal"], sat["complement"]
        n_vec = meta["n_distinct_test"]
        sat_txt = (
            f" **The two fuzzy rows differ only by floating-point resolution, not by "
            f"model.** Under the product t-norm the complement and the surprisal are "
            f"monotone transforms of one another, so in exact arithmetic their ROC-AUC "
            f"is identical; measured here they are {cp['auc']:.3f} and {sc_['auc']:.3f}. "
            f"The evidence is score resolution against the ceiling the data itself sets: "
            f"the test split contains {n_vec:,} distinct feature vectors, so {n_vec:,} is "
            f"the most distinct scores any detector can produce. The surprisal resolves "
            f"{sc_['distinct']:,} of them ({sc_['distinct'] / n_vec:.1%}); the complement "
            f"resolves {cp['distinct']:,} ({cp['distinct'] / n_vec:.1%}), collapsing "
            f"distinct behaviours onto equal scores so the ordering AUC needs is not "
            f"there. (Both rows show most test *rows* tied at the maximum — "
            f"{sc_['tied_at_max']:.0%} and {cp['tied_at_max']:.0%} — but that is BETH "
            f"repeating itself, not a numerical fault, which is why the distinct-score "
            f"count is the diagnostic and the tied-row fraction is not.) "
            f"`tribblefis.one_class`'s docstring puts the onset of complement saturation "
            f"past ~60 features; BETH reaches it at {meta['n_feat']}, because the "
            f"log-scaled process and thread identifiers are heavy-tailed enough that a "
            f"typical point's summed z² already exceeds float64's resolution against "
            f"1.0 — so the threshold is a property of the tails, not of the feature "
            f"count alone. Read the surprisal row; the complement row is Chapter 4's "
            f"default and is kept visible for that reason."
        )

    transfer = ""
    if np.isfinite(target_fa) and target_fa > 0:
        test_fa = None
        for r in rows:
            if r[0].startswith("**Tribble one-class (surprisal)"):
                test_fa = float(r[2])
        if test_fa is not None:
            transfer = (
                f" **The calibration does not transfer**: the threshold costing "
                f"{target_fa:.4f} false alarms on validation costs {test_fa:.4f} on "
                f"test, {test_fa / target_fa:.1f}× more. Both splits are benign-only "
                f"draws from the same capture, so this is a property of BETH's benign "
                f"test rows, not of any detector -- and it means a false-alarm budget "
                f"set on BETH's validation split cannot be believed on its test split. "
                f"Every arm is matched at the validation rate, so they are affected "
                f"alike and remain comparable to each other."
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
            f"{meta['n_train']:,}-row training split, threshold calibrated on the "
            f"{meta['n_val']:,}-row validation split, scored on the "
            f"{meta['n_test']:,}-row test split ({int(is_anom_te.sum()):,} anomalous / "
            f"{int((~is_anom_te).sum()):,} benign). No downsampling of the training "
            f"set. **This is a one-class table because BETH is a one-class benchmark: "
            f"all {meta['n_pos']['test']:,} positives are in the test split and the "
            f"training and validation splits are 100% benign.** A supervised arm is not "
            f"reported because none can be trained — a Random Forest fitted on the "
            f"training split predicts the constant 0 without raising anything (16.2% "
            f"accuracy, AUC 0.5). **Isolation Forest is the Random-Forest family's "
            f"one-class detector** and is the RF-shaped comparison this task admits: an "
            f"unlabelled forest of random trees scoring by isolation depth, the "
            f"tree-ensemble counterpart of the one-class SVM. No arm here is fitted on "
            f"test-split labels; a supervised separability diagnostic is printed to "
            f"stdout and deliberately kept out of this table, since the test split's "
            f"{meta['n_test']:,} rows hold only {meta['n_distinct_test']:,} distinct "
            f"feature vectors ({meta['n_distinct_test'] / meta['n_test']:.2%}) and any "
            f"within-test split is therefore a lookup table or an in-capture "
            f"resubstitution estimate. The fuzzy arms are "
            f"`tribblefis.one_class.TribbleOneClassDetector`, the library's own "
            f"one-class estimator, rather than a hand-assembly of the multi-class path; "
            f"both give the same operating point, and using the library's API is what "
            f"keeps this table from drifting against it.{sat_txt}{transfer} "
            f"Features: {meta['n_feat']} numeric columns after dropping "
            f"{meta['dropped']} — `sus` is BETH's second *label* (1 for 100% of evil "
            f"rows) and `timestamp` is a per-capture session clock; both are dropped in "
            f"`FuzzySystemsExperiments/beth-anomaly.py` too. UnitScalar fitted on train "
            f"alone, auto-logging {meta['logged']}. Threshold = the "
            f"{1 - FA_BUDGET:.3f} quantile of the detector's benign-validation scores "
            f"(a {FA_BUDGET:.1%} budget; exact, not grid-limited), {CONORM} conorm. "
            f"Baselines' contamination/nu is matched to the surprisal arm's realized "
            f"validation false-alarm rate ({cont:.4f}). 'Detection − false alarm' is "
            f"Youden's J. The fuzzy arms and their calibration are deterministic given "
            f"the fixed splits and are fitted once, so their cells carry no ±; the ± on "
            f"the baseline rows is across common.SEEDS ({C.SEEDS}). One-class SVM is "
            f"fitted on a random {OCSVM_TRAIN_CAP:,}-row subsample per seed (libsvm is "
            f"O(n²)–O(n³); {meta['n_train']:,} rows is not a fit), while the fuzzy arms "
            f"and Isolation Forest see the full training split."
        ),
    )

    if curve_rows:
        print("\n  operating curve (threshold chosen on validation only):")
        _resolution_txt = "; ".join(
            f"{k} resolves {v['distinct']:,} distinct scores" for k, v in sat.items()
        )
        C.emit(
            "table_4_11_beth_fa_sweep",
            "Table 4.11(b) — BETH operating curve vs. the validation false-alarm budget",
            [
                "score",
                "budget",
                "realized validation false-alarm rate",
                "test detection rate",
                "test false-alarm rate",
                "detection − false alarm",
                "chosen",
            ],
            curve_rows,
            note=(
                f"One fitted model per score; only the threshold moves. Each threshold "
                f"is the (1 − budget) quantile of the detector's scores on the "
                f"benign-only validation split, so every row is selected without any "
                f"positive label — which is the entire value of BETH shipping a "
                f"benign validation split. The test columns are the consequence of that "
                f"choice and are NOT used to make it. Realized validation rates fall "
                f"below their budget where scores are tied at the threshold, which is "
                f"why the complement rows plateau and, at the tightest budget, collapse "
                f"to zero detection while the surprisal holds: {_resolution_txt}. This "
                f"is the strict-operating-point failure that ROC-AUC alone hides. The "
                f"marked rows are the {FA_BUDGET:.1%} budget the headline table quotes. "
                f"Deterministic: no seed averaging, and none is implied."
            ),
        )


if __name__ == "__main__":
    main()
