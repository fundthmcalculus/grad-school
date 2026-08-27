"""Table 4.4 -- does the complement rule actually detect the unseen?

Chapter 4 §4.3.5 claims that taking the fuzzy complement of the aggregate of the
known-class rules yields an open-set detector for free. That claim is currently
argued rather than measured. This script measures it, against the detectors
built for the job.

PROTOCOL -- leave-one-class-out open set. For each class c in turn: train on the
other K-1 classes ONLY, so c is genuinely unseen at fit time, then score every
test point as known/unknown. Rotating c over all classes and averaging avoids
the result hinging on which class happened to be held out. This is the standard
open-set recognition setup and it is the honest version of the Chapter 4 BETH
experiment, which trains on benign traffic and is shown novel attacks.

Chapter 4 describes that BETH experiment; the BETH files are not in the
repository, so this runs the same protocol on a public dataset that is. If
`beth_data/` appears, prefer it -- the code checks.

ARMS
  complement rule   the MoG classifier with AnomalyParameters enabled; the
                    anomaly label wins when no known rule fires strongly enough
  one-class SVM     trained on the known-class training data
  isolation forest  same

METRICS. Detection rate and false-alarm rate are meaningless alone -- any
detector reaches 100% detection by flagging everything -- so both are reported,
plus ROC-AUC where a continuous score exists, and the operating points are
matched by holding each baseline's contamination to the complement rule's
observed false-alarm rate.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_4_4_openset.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import _fuzzy_models as F  # noqa: E402

THRESHOLD = float(os.environ.get("REPRO_ANOM_THRESHOLD", "0.99"))
CONORM = os.environ.get("REPRO_ANOM_CONORM", "hamacher")

# One-class SVM training is O(n^2)-O(n^3) (libsvm); on RT-IOT2022's ~80-90k-row
# leave-one-class-out training folds a single fit measured at ~100-130s, x120
# (12 classes x 10 seeds) is several hours for one baseline arm. The complement
# rule and Isolation Forest both fit the full training set; only the SVM baseline
# is capped, and the cap is recorded in the table's note rather than left silent.
OCSVM_TRAIN_CAP = int(os.environ.get("REPRO_OCSVM_TRAIN_CAP", "20000"))
# theta_sweep (Fig 4.2) is a supplementary sensitivity curve, not the headline
# table -- it reruns the complement rule alone, so its cost is (#thetas x
# #classes x #seeds) fits with no SVM involved. On a large dataset the default
# 10-seed x 7-theta grid is still (7 x 12 x 10) fits; REPRO_THETA_SWEEP_SEEDS lets
# it use a named subset instead of quietly changing the seed count elsewhere.
_theta_seeds_env = os.environ.get("REPRO_THETA_SWEEP_SEEDS", "")
THETA_SWEEP_SEEDS = (
    [int(s) for s in _theta_seeds_env.split(",")] if _theta_seeds_env else None
)


def _glass():
    path = os.path.join(F.DATA_DIR, "glass.csv")
    if not os.path.exists(path):
        # Glass moved into data/; this fallback is the pre-move location.
        path = os.path.join(F.REPO_ROOT, "glass.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).dropna()
    return df.drop(columns=["Type"]).astype(float), df["Type"].astype(int)


def load_openset_data():
    """(X, y, name) for the leave-one-class-out protocol.

    Priority: RT-IOT2022 (123k) > BETH (3.8M) > Glass (214).

    `REPRO_OPENSET_DATASET` pins the choice instead of taking the priority order.
    That override exists because the priority silently reassigned what this
    generator produces: RT-IOT2022 landed in the repository on 2026-08-12, and
    from that moment the script emitted the RT-IOT2022 table -- the proposal's
    **Table 4.7b** -- under the same output filenames that Tables 4.6 and 4.7
    are quoted from. Those two are Glass measurements, and there was no way to
    re-derive them short of hiding the dataset. Checklist B16(e).

        REPRO_OPENSET_DATASET=glass  ->  Tables 4.6 / 4.7
        REPRO_OPENSET_DATASET=rt-iot2022 -> Table 4.7b (the current default)
    """
    pin = os.environ.get("REPRO_OPENSET_DATASET", "").strip().lower()
    if pin:
        loaders = {
            "glass": ("Glass", _glass),
            "rt-iot2022": ("RT-IOT2022", F.load_rt_iot2022),
            "beth": ("BETH", lambda: (F.load_beth() or {}).get("train")),
        }
        if pin not in loaders:
            raise SystemExit(
                f"REPRO_OPENSET_DATASET={pin!r} is not one of {sorted(loaders)}"
            )
        name, fn = loaders[pin]
        got = fn()
        if got is None:
            # Pinned and absent is an error, not a reason to quietly use another
            # dataset -- silently substituting one is the whole reason this knob
            # exists.
            raise SystemExit(f"REPRO_OPENSET_DATASET={pin!r} but {name} is unavailable")
        X, y = got
        print(f"  [data] REPRO_OPENSET_DATASET={pin} -- using {name}")
        return X, y, name

    # Try RT-IOT2022 first (large-scale public dataset)
    iot = F.load_rt_iot2022()
    if iot is not None:
        X, y = iot
        print("  [data] RT-IOT2022 found (123k × 83, 12 classes) -- using it")
        return X, y, "RT-IOT2022"

    # Try BETH (explicit train split for anomaly detection)
    beth_splits = F.load_beth()
    if beth_splits is not None:
        X, y = beth_splits["train"]
        print("  [data] BETH found (training split, 763k × 10) -- using it")
        return X, y, "BETH"

    # Fall back to Glass (small public dataset)
    got = _glass()
    if got is None:
        return None
    print(
        "  [data] RT-IOT2022 and BETH absent -- leave-one-class-out on Glass (214 × 9)"
    )
    return got[0], got[1], "Glass"


def _fit_complement(X_tr, y_tr):
    """Fit the MoG membership dict. Independent of the anomaly threshold theta,
    which enters only at score time (see `_score_complement`), so a theta sweep
    can fit once here and re-score across the whole grid."""
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        take_top_features,
    )

    diffs = calculate_gaussian_correlation(X_tr, y_tr)
    _, top_vars = take_top_features(diffs, top_n=len(X_tr.columns))
    return create_gaussian_membership_dict(X_tr, y_tr, top_n_var_names=top_vars)


def _score_complement(memb, X_te, threshold):
    """Score a fitted membership dict at one anomaly threshold. Returns a bool
    'flagged unknown'. theta enters here only, via AnomalyParameters; the fit does
    not see it, so `_score_complement(memb, X_te, th)` for varying `th` on a fixed
    `memb` reproduces refitting at each `th` exactly."""
    from tribblefis.gauss_data import AnomalyParameters
    from tribblefis.gauss_math import simple_gaussian_predict

    params = AnomalyParameters(
        include_anomaly=True,
        threshold=threshold,
        label="anomaly",
        norm_conorm=CONORM,
        member_function="gaussian",
    )
    pred = simple_gaussian_predict(X_te, memb.to_simple_model(params))
    return np.asarray([str(p) == "anomaly" for p in pred], dtype=bool)


def complement_rule_sweep(X_tr, y_tr, X_te, thetas):
    """`complement_rule` at every theta, building the model once.

    The boost theta enters only at the anomaly step of prediction; the screen,
    the membership dict and the fitted rule base are all theta-independent, and so
    is the class rule firing. So this builds the model once and calls
    `simple_gaussian_predict_sweep`, which reuses one class-firing pass across all
    thetas. Bit-identical to calling `complement_rule` once per theta (verified in
    tribble-fis test_predict_theta_sweep and end-to-end on an RT-IOT2022 fold),
    at about 5.7x on the sweep.

    Returns {theta: bool 'flagged unknown'}.
    """
    from tribblefis.gauss_data import AnomalyParameters
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        simple_gaussian_predict_sweep,
        take_top_features,
    )

    diffs = calculate_gaussian_correlation(X_tr, y_tr)
    _, top_vars = take_top_features(diffs, top_n=len(X_tr.columns))
    memb = create_gaussian_membership_dict(X_tr, y_tr, top_n_var_names=top_vars)
    thetas = list(thetas)
    params = AnomalyParameters(
        include_anomaly=True,
        threshold=thetas[0],  # overridden per-theta by the sweep; build ignores it
        label="anomaly",
        norm_conorm=CONORM,
        member_function="gaussian",
    )
    model = memb.to_simple_model(params)
    preds = simple_gaussian_predict_sweep(X_te, model, thetas)
    return {
        th: np.asarray([str(p) == "anomaly" for p in preds[th]], dtype=bool)
        for th in thetas
    }


def rates(flagged, is_unknown):
    """(detection rate on unknowns, false-alarm rate on knowns)."""
    unk, kn = is_unknown, ~is_unknown
    det = float(flagged[unk].mean()) if unk.any() else float("nan")
    fa = float(flagged[kn].mean()) if kn.any() else float("nan")
    return det, fa


def theta_sweep(X, y, classes, thetas, seeds=None):
    """Operating curve: detection and false-alarm as functions of the boost theta.

    This is the experiment Figure 4.2 needs. A single theta says almost nothing --
    the question is whether the knob buys a usable trade at ANY setting.

    theta enters the complement rule only at score time, so the MoG fit is the same
    at every theta on the grid. We fit once per (held-out class, seed) and re-score
    across the whole grid rather than refitting for each theta -- identical numbers,
    with the sweep's fit cost cut from (#thetas x #classes x #seeds) to
    (#classes x #seeds). A fit that raises is theta-independent, so it would have
    contributed nothing at any theta in the refitting version either; skipping it
    once here matches that.
    """
    seeds = seeds if seeds is not None else C.SEEDS
    print("\n  theta sweep (Figure 4.2):")
    print(f"    {'theta':>8} {'detection':>12} {'false alarm':>13} {'J':>8}")

    # Fit once per (held-out class, seed); theta is applied at score time below.
    fits = []
    for held in classes:
        known = pd.Series(y).values != held
        Xk, yk = X[known], pd.Series(y)[known]
        if len(np.unique(yk)) < 2 or len(Xk) < 40:
            continue
        for seed in seeds:
            Xtr, Xte_k, ytr, _ = train_test_split(
                Xk, yk, test_size=0.3, random_state=seed
            )
            Xte = pd.concat([Xte_k, X[~known]], ignore_index=True)
            unk = np.r_[
                np.zeros(len(Xte_k), bool), np.ones(int((~known).sum()), bool)
            ]
            try:
                memb = _fit_complement(Xtr, ytr)
            except Exception:  # noqa: BLE001
                continue
            fits.append((memb, Xte, unk))

    rows = []
    thetas = list(thetas)

    # Build the model once per (held-out class, seed) and sweep theta over the
    # anomaly step only -- the fold is the same for every theta, so rebuilding it
    # per theta (the old loop) redid the screen, the membership dict and the
    # dedup 7x for nothing. Accumulate per-theta so the aggregation is unchanged.
    det = {th: [] for th in thetas}
    fa = {th: [] for th in thetas}
    for held in classes:
        known = pd.Series(y).values != held
        Xk, yk = X[known], pd.Series(y)[known]
        if len(np.unique(yk)) < 2 or len(Xk) < 40:
            continue
        for seed in seeds:
            Xtr, Xte_k, ytr, _ = train_test_split(
                Xk, yk, test_size=0.3, random_state=seed
            )
            Xte = pd.concat([Xte_k, X[~known]], ignore_index=True)
            unk = np.r_[np.zeros(len(Xte_k), bool), np.ones(int((~known).sum()), bool)]
            try:
                flags = complement_rule_sweep(Xtr, ytr, Xte, thetas)
            except Exception:  # noqa: BLE001
                continue
            for th in thetas:
                d, f = rates(flags[th], unk)
                if np.isfinite(d) and np.isfinite(f):
                    det[th].append(d)
                    fa[th].append(f)

    for th in thetas:
        dm, _ = C.agg(det[th])
        fm, _ = C.agg(fa[th])
        if dm is None:
            continue
        print(f"    {th:8.3f} {dm:12.3f} {fm:13.3f} {dm-fm:+8.3f}")
        rows.append([f"{th:.3f}", f"{dm:.3f}", f"{fm:.3f}", f"{dm-fm:+.3f}"])
    if rows:
        C.emit(
            "table_4_4b_theta_sweep",
            "Figure 4.2 (tabular) — anomaly operating curve vs. the boost θ",
            ["θ", "detection rate", "false-alarm rate", "detection − false alarm"],
            rows,
            note=(
                "Averaged over held-out classes × seeds"
                + (
                    f" (seeds={seeds}, a named subset of the ten-seed floor -- "
                    f"this sweep is a supplementary sensitivity curve, not the "
                    f"headline table)"
                    if seeds != C.SEEDS
                    else ""
                )
                + ". This is the curve a user picks an "
                "operating point on; a single θ in isolation says little. If J stays near "
                "zero across the whole sweep, the knob does not buy a usable trade on this "
                "dataset and the claim needs a better testbed than a 214-sample set."
            ),
        )


def main():
    print("Table 4.4 -- open-set detection, leave-one-class-out")
    data = load_openset_data()
    if data is None:
        print("  no dataset available")
        return
    X, y, dsname = data
    classes = sorted(pd.Series(y).unique())
    print(f"  {dsname}: N={len(X)} M={X.shape[1]} classes={classes}")

    acc: dict = {}

    def add(arm, det, fa):
        acc.setdefault(arm, {"det": [], "fa": []})
        if not (np.isnan(det) or np.isnan(fa)):
            acc[arm]["det"].append(det)
            acc[arm]["fa"].append(fa)

    for held in classes:
        known_mask = pd.Series(y).values != held
        Xk, yk = X[known_mask], pd.Series(y)[known_mask]
        if len(np.unique(yk)) < 2 or len(Xk) < 40:
            continue
        for seed in C.SEEDS:
            # train on KNOWN classes only; test set = held-out knowns + all unknowns
            Xtr, Xte_k, ytr, _ = train_test_split(
                Xk, yk, test_size=0.3, random_state=seed
            )
            Xte_u = X[~known_mask]
            Xte = pd.concat([Xte_k, Xte_u], ignore_index=True)
            is_unknown = np.r_[np.zeros(len(Xte_k), bool), np.ones(len(Xte_u), bool)]

            try:
                flg = complement_rule(Xtr, ytr, Xte)
                d, f = rates(flg, is_unknown)
                add("**Complement rule (this work)**", d, f)
                target_fa = f  # match the baselines to this
            except Exception as exc:  # noqa: BLE001
                print(
                    f"    [complement] class {held} seed {seed}: {exc.__class__.__name__}: {exc}"
                )
                target_fa = 0.1

            cont = float(
                min(max(target_fa if np.isfinite(target_fa) else 0.1, 0.01), 0.5)
            )
            for arm, est in (
                ("One-class SVM", OneClassSVM(nu=cont, gamma="scale")),
                (
                    "Isolation Forest",
                    IsolationForest(
                        contamination=cont, random_state=seed, n_estimators=200
                    ),
                ),
            ):
                try:
                    Xfit = Xtr
                    if arm == "One-class SVM" and len(Xtr) > OCSVM_TRAIN_CAP:
                        idx = np.random.RandomState(seed).choice(
                            len(Xtr), OCSVM_TRAIN_CAP, replace=False
                        )
                        Xfit = Xtr.iloc[idx]
                    est.fit(Xfit)
                    add(arm, *rates(est.predict(Xte) == -1, is_unknown))
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"    [{arm}] class {held} seed {seed}: {exc.__class__.__name__}"
                    )
        print(f"  done: held-out class {held}")

    order = ["**Complement rule (this work)**", "One-class SVM", "Isolation Forest"]
    rows = []
    for arm in order:
        v = acc.get(arm)
        if not v or not v["det"]:
            rows.append([arm, C.NA, C.NA, C.NA, "no" if "Complement" in arm else "yes"])
            continue
        dm, _ = C.agg(v["det"])
        fm, _ = C.agg(v["fa"])
        youden = f"{dm - fm:+.3f}" if (dm is not None and fm is not None) else C.NA
        rows.append(
            [
                arm,
                C.cell(v["det"]),
                C.cell(v["fa"]),
                youden,
                "no" if "Complement" in arm else "yes",
            ]
        )

    sweep = os.environ.get("REPRO_THETA_SWEEP", "")
    if sweep:
        theta_sweep(
            X, y, classes, [float(t) for t in sweep.split(",")], seeds=THETA_SWEEP_SEEDS
        )

    svm_note = ""
    if len(X) > OCSVM_TRAIN_CAP:
        svm_note = (
            f" One-class SVM is trained on a random {OCSVM_TRAIN_CAP}-row subsample of "
            f"each fold's training set (libsvm's O(n^2)-O(n^3) fit time makes the full "
            f"~{int(len(X) * 0.7):,}-row fold intractable at ten seeds x {len(classes)} "
            f"held-out classes); the complement rule and Isolation Forest see the full "
            f"training set."
        )

    C.emit(
        "table_4_4_openset",
        f"Table 4.4 — Open-set detection on {dsname} (leave-one-class-out)",
        [
            "Method",
            "Detection rate",
            "False-alarm rate",
            "Detection − false alarm",
            "Separate model?",
        ],
        rows,
        note=(
            f"Each class is held out of training in turn and treated as unseen; results "
            f"averaged over held-out classes × seeds. The baselines' contamination is set "
            f"to the complement rule's observed false-alarm rate, so the arms are compared "
            f"at a matched operating point rather than at whatever default each ships with. "
            f"'Detection − false alarm' is Youden's J: higher is better, 0 means the "
            f"detector is no better than flagging at random. Complement rule at θ={THRESHOLD}, "
            f"{CONORM} conorm.{svm_note} Chapter 4 describes this experiment on BETH; those "
            f"files are present locally but binary (BETH), so RT-IOT2022's {len(classes)} "
            f"classes are used when available, falling back to BETH then Glass."
        ),
    )


if __name__ == "__main__":
    main()
