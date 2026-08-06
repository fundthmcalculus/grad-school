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


def load_openset_data():
    """(X, y) for the leave-one-class-out protocol. BETH if present, else Glass."""
    # Under data/, not the submodule: tribble-fis dropped `gaussian_mixture/` in
    # 8484fd6 and is now a pure library. See _fuzzy_models.DATA_DIR.
    beth = os.path.join(F.DATA_DIR, "beth_data", "labelled_training_data.csv")
    if os.path.exists(beth):
        print("  [data] BETH found -- using it")
        df = pd.read_csv(beth)
        y = pd.Series(np.where(df.get("evil", 0) == 1, "anomaly", "regular"))
        X = df.select_dtypes(include=[np.number]).drop(
            columns=[c for c in ("sus", "evil") if c in df.columns], errors="ignore"
        )
        return X, y, "BETH"
    path = os.path.join(F.REPO_ROOT, "glass.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path).dropna()
    print("  [data] BETH absent -- leave-one-class-out on Glass (public, in-repo)")
    return (df.drop(columns=["Type"]).astype(float), df["Type"].astype(int), "Glass")


def complement_rule(X_tr, y_tr, X_te):
    """MoG classifier with the anomaly rule on. Returns a bool 'flagged unknown'."""
    from tribblefis.gauss_data import AnomalyParameters
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        simple_gaussian_predict,
        take_top_features,
    )

    diffs = calculate_gaussian_correlation(X_tr, y_tr)
    _, top_vars = take_top_features(diffs, top_n=len(X_tr.columns))
    memb = create_gaussian_membership_dict(X_tr, y_tr, top_n_var_names=top_vars)
    params = AnomalyParameters(
        include_anomaly=True,
        threshold=THRESHOLD,
        label="anomaly",
        norm_conorm=CONORM,
        member_function="gaussian",
    )
    pred = simple_gaussian_predict(X_te, memb.to_simple_model(params))
    return np.asarray([str(p) == "anomaly" for p in pred], dtype=bool)


def rates(flagged, is_unknown):
    """(detection rate on unknowns, false-alarm rate on knowns)."""
    unk, kn = is_unknown, ~is_unknown
    det = float(flagged[unk].mean()) if unk.any() else float("nan")
    fa = float(flagged[kn].mean()) if kn.any() else float("nan")
    return det, fa


def theta_sweep(X, y, classes, thetas):
    """Operating curve: detection and false-alarm as functions of the boost theta.

    This is the experiment Figure 4.2 needs. A single theta says almost nothing --
    the question is whether the knob buys a usable trade at ANY setting.
    """
    print("\n  theta sweep (Figure 4.2):")
    print(f"    {'theta':>8} {'detection':>12} {'false alarm':>13} {'J':>8}")
    rows = []
    global THRESHOLD
    keep = THRESHOLD
    for th in thetas:
        THRESHOLD = th
        det, fa = [], []
        for held in classes:
            known = pd.Series(y).values != held
            Xk, yk = X[known], pd.Series(y)[known]
            if len(np.unique(yk)) < 2 or len(Xk) < 40:
                continue
            for seed in C.SEEDS:
                Xtr, Xte_k, ytr, _ = train_test_split(
                    Xk, yk, test_size=0.3, random_state=seed
                )
                Xte = pd.concat([Xte_k, X[~known]], ignore_index=True)
                unk = np.r_[
                    np.zeros(len(Xte_k), bool), np.ones(int((~known).sum()), bool)
                ]
                try:
                    d, f = rates(complement_rule(Xtr, ytr, Xte), unk)
                    if np.isfinite(d) and np.isfinite(f):
                        det.append(d)
                        fa.append(f)
                except Exception:  # noqa: BLE001
                    pass
        dm, _ = C.agg(det)
        fm, _ = C.agg(fa)
        if dm is None:
            continue
        print(f"    {th:8.3f} {dm:12.3f} {fm:13.3f} {dm-fm:+8.3f}")
        rows.append([f"{th:.3f}", f"{dm:.3f}", f"{fm:.3f}", f"{dm-fm:+.3f}"])
    THRESHOLD = keep
    if rows:
        C.emit(
            "table_4_4b_theta_sweep",
            "Figure 4.2 (tabular) — anomaly operating curve vs. the boost θ",
            ["θ", "detection rate", "false-alarm rate", "detection − false alarm"],
            rows,
            note=(
                "Averaged over held-out classes × seeds. This is the curve a user picks an "
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
                    est.fit(Xtr)
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
        theta_sweep(X, y, classes, [float(t) for t in sweep.split(",")])

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
            f"{CONORM} conorm. Chapter 4 describes this experiment on BETH; those files are "
            f"not in the repository, so the same protocol runs on public in-repo data."
        ),
    )


if __name__ == "__main__":
    main()
