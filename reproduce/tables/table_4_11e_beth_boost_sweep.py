"""Table 4.11(e) -- does each detector's own knob buy anything over a plain threshold?

Every arm in this family ships a knob that moves its operating point, and Chapter
4 leans on one of them: the anomaly rule's **boost theta**. This table asks the
only question that decides whether such a knob is a contribution or a
reparameterisation:

    At a MATCHED false-alarm rate, does turning the knob detect more than simply
    moving the decision threshold on the arm's own continuous score?

If the answer is "no" for a knob, that knob is a threshold in disguise. That is
not a criticism of the code -- a threshold is a perfectly good thing to have --
but it is a claim the chapter should not make about it.

THE ANALYTIC PREDICTION FOR THETA, WHICH THIS TABLE EXISTS TO CHECK
------------------------------------------------------------------
`gauss_math._anomaly_argmax` computes the anomaly column as

    boosted   = clip(class_firing + theta, 0, 1)
    anomaly   = complement(conorm(boosted))          # conorm over CLASS columns

and then argmaxes. With **one** known class there is exactly one class column, and
`t_conorm(x, None, ...)` aggregates column-wise -- so the conorm has nothing to
aggregate and is the identity. The anomaly label therefore wins exactly when

    1 - (firing + theta) > firing   <=>   firing < (1 - theta) / 2

so in the one-class setting theta IS a hard threshold on firing strength, the
norm/conorm family is irrelevant (there is one column to aggregate), and the
boost cannot express any decision a firing threshold cannot. The proposal's §4.3
already argues a weaker version of this for the multi-class case -- that at
theta = 0.99 the rule degenerates to a max-membership rejector -- and the
one-class reduction makes it total at every theta, not just the shipped default.

That is a derivation, not a measurement, so this table measures it. `delta
detection` is the number to read: if theta is a threshold in disguise it is
0.000 across the whole sweep.

THE OTHER TWO KNOBS ARE NOT THE SAME KIND OF THING
--------------------------------------------------
  Isolation Forest, `contamination` -- provably a pure threshold knob: it only
      sets `offset_`, computed as a quantile of the TRAINING scores, and never
      touches the trees. One fit per seed is therefore sufficient and correct,
      and its delta must come back 0.000. That is a CONTROL: it tells us the
      delta metric returns zero when it should, which is what licenses reading a
      non-zero delta elsewhere as real.

  One-class SVM, `nu` -- NOT a threshold knob. It enters the QP objective as an
      upper bound on the fraction of margin errors and a lower bound on the
      fraction of support vectors, so every value of `nu` is a different fitted
      model. Here a non-zero delta is expected and meaningful: it is the only
      knob in this table that can trade one decision surface for another rather
      than slide along one.

PROTOCOL
--------
Every arm is fitted on the SAME 20,000-row benign subsample per seed (matched, as
in Table 4.11(d) -- the sample-count confound that table corrects is not
re-introduced here). No wall-clock is reported: this table is about decision
curves, and timing questions belong to 4.11(d).

For each knob setting we record the realized test false-alarm rate and detection
rate. The control detection is then read off the SAME arm's continuous score,
thresholded to land on that identical false-alarm rate -- so the two are compared
at one operating point rather than across two curves.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_4_11e_beth_boost_sweep.py
"""

from __future__ import annotations

import os
import sys
import warnings

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, os.environ.get("REPRO_BLAS_THREADS", "8"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import IsolationForest  # noqa: E402
from sklearn.svm import OneClassSVM  # noqa: E402

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import table_4_11_beth_anomaly as T411  # noqa: E402

FIT_N = int(os.environ.get("REPRO_BETH_BOOST_FIT_N", "20000"))

_th = os.environ.get("REPRO_BETH_THETA_GRID", "")
THETAS = (
    [float(t) for t in _th.split(",")]
    if _th
    else [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
)
_cg = os.environ.get("REPRO_BETH_CONTAM_GRID", "")
CONTAMS = (
    [float(c) for c in _cg.split(",")]
    if _cg
    else [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30]
)


# Emitted .md/.csv are written UTF-8 by `common.write_markdown`, but this repo's
# generators print ASCII to stdout on purpose: a Windows console is cp1252, and a
# Greek theta in a progress line aborts the whole run with UnicodeEncodeError
# AFTER the measurement has been paid for. Table labels keep the real glyphs.
_ASCII = {"θ": "theta", "ν": "nu", "Δ": "delta", "±": "+/-"}


def _p(msg):
    """print(), with the non-cp1252 glyphs transliterated for the console."""
    for k, v in _ASCII.items():
        msg = msg.replace(k, v)
    print(msg)


def _rates(flagged, is_anom):
    return T411._rates(flagged, is_anom)


def _det_at_fa(scores, is_anom, target_fa):
    """Detection of a plain score threshold placed to realize `target_fa`.

    The threshold is the (1 - target_fa) quantile of the BENIGN test scores, so
    the control lands on the same false-alarm rate as the knob setting it is
    being compared against. Comparing at a matched operating point is the only
    way to ask "did the knob detect more?" -- two curves read at different false
    alarm rates cannot answer it.

    Returns (detection, realized_fa). `nan` if the target is unreachable.
    """
    if not np.isfinite(target_fa):
        return float("nan"), float("nan")
    benign = scores[~is_anom]
    if target_fa <= 0:
        thr = float(np.max(scores)) + 1.0  # flag nothing
    else:
        thr = float(np.quantile(benign, 1.0 - target_fa))
    flagged = scores > thr
    det, fa = _rates(flagged, is_anom)
    return det, fa


def _build_theta_model(Xfit):
    """Membership model with the anomaly rule enabled, theta swept separately.

    theta enters only at the anomaly step, so the model is theta-independent and
    is built once -- the same reuse `table_4_4_openset.py` verified bit-identical
    to a theta-at-a-time loop.
    """
    from tribblefis.gauss_data import AnomalyParameters
    from tribblefis.gauss_math import (
        calculate_gaussian_correlation,
        create_gaussian_membership_dict,
        take_top_features,
    )

    y = pd.Series("normal", index=Xfit.index, name="y_value")
    diffs = calculate_gaussian_correlation(Xfit, y)
    _, top_vars = take_top_features(diffs, top_p=1.0)
    memb = create_gaussian_membership_dict(Xfit, y, top_n_var_names=top_vars)
    params = AnomalyParameters(
        include_anomaly=True,
        threshold=THETAS[0],
        label="anomaly",
        norm_conorm=T411.CONORM,
        member_function="gaussian",
    )
    return memb.to_simple_model(params)


def _theta_flags(model, X, thetas):
    from tribblefis.gauss_math import simple_gaussian_predict_sweep

    preds = simple_gaussian_predict_sweep(X, model, list(thetas))
    return {
        t: np.asarray([str(p) == "anomaly" for p in preds[t]], dtype=bool)
        for t in thetas
    }


def main():
    print("Table 4.11(e) -- knob vs plain threshold, at matched false-alarm rate")
    got = T411.load_splits()
    if got is None:
        print("  BETH unavailable -- nothing emitted")
        return
    Xtr, _Xva, Xte, _is_anom_va, is_anom_te, meta = got
    print(f"  fitting every arm on the same {FIT_N:,}-row benign subsample per seed")

    from tribblefis.one_class import TribbleOneClassDetector

    # (arm, knob value) -> lists over seeds
    acc: dict = {}

    def add(arm, knob, det, fa, ctrl_det):
        a = acc.setdefault((arm, knob), {"det": [], "fa": [], "ctrl": [], "d": []})
        if np.isfinite(det) and np.isfinite(fa):
            a["det"].append(det)
            a["fa"].append(fa)
        if np.isfinite(ctrl_det):
            a["ctrl"].append(ctrl_det)
            if np.isfinite(det):
                a["d"].append(det - ctrl_det)

    for seed in C.SEEDS:
        idx = np.random.RandomState(seed).choice(len(Xtr), FIT_N, replace=False)
        Xfit = Xtr.iloc[idx]

        # ---- arm 1: the boost theta, against a firing-strength threshold ----
        try:
            model = _build_theta_model(Xfit)
            flags = _theta_flags(model, Xte, THETAS)
            # The control score is the complement, 1 - max firing: a monotone
            # function of the very quantity the theta rule thresholds, from the
            # same fitted memberships.
            ctrl = -np.asarray(
                TribbleOneClassDetector(
                    score="complement",
                    norm_conorm=T411.CONORM,
                    feature_selection="all",
                    random_state=seed,
                )
                .fit(Xfit)
                .score_samples(Xte)
            )
            for t in THETAS:
                det, fa = _rates(flags[t], is_anom_te)
                cdet, _ = _det_at_fa(ctrl, is_anom_te, fa)
                add("Tribble boost θ", t, det, fa, cdet)
        except Exception as exc:  # noqa: BLE001
            print(f"    [theta] seed {seed}: {exc.__class__.__name__}: {exc}")

        # ---- arm 2: Isolation Forest contamination (a pure threshold: control)
        try:
            # ONE fit. `contamination` only sets `offset_` from a quantile of the
            # training scores and never touches the trees, so refitting per value
            # would burn time to produce the same forest.
            forest = IsolationForest(
                n_estimators=200, max_samples=FIT_N, random_state=seed, n_jobs=8
            ).fit(Xfit)
            s_tr = -np.asarray(forest.score_samples(Xfit))
            s_te = -np.asarray(forest.score_samples(Xte))
            for c in CONTAMS:
                offset = float(np.quantile(s_tr, 1.0 - c))
                det, fa = _rates(s_te > offset, is_anom_te)
                cdet, _ = _det_at_fa(s_te, is_anom_te, fa)
                add("iForest contamination", c, det, fa, cdet)
        except Exception as exc:  # noqa: BLE001
            print(f"    [iforest] seed {seed}: {exc.__class__.__name__}: {exc}")

        # ---- arm 3: one-class SVM nu (refits -- the only non-threshold knob) --
        try:
            # The control is ONE fixed model's score, thresholded. nu refits, so
            # the comparison is "a different surface" vs "sliding along one".
            base = OneClassSVM(nu=0.01, gamma="scale").fit(Xfit)
            s_base = -np.asarray(base.decision_function(Xte))
            for nu in CONTAMS:
                est = OneClassSVM(nu=nu, gamma="scale").fit(Xfit)
                det, fa = _rates(est.predict(Xte) == -1, is_anom_te)
                cdet, _ = _det_at_fa(s_base, is_anom_te, fa)
                add("OC-SVM ν", nu, det, fa, cdet)
        except Exception as exc:  # noqa: BLE001
            print(f"    [ocsvm] seed {seed}: {exc.__class__.__name__}: {exc}")

        print(f"    seed {seed} done")

    ORDER = [
        ("Tribble boost θ", THETAS),
        ("iForest contamination", CONTAMS),
        ("OC-SVM ν", CONTAMS),
    ]
    rows = []
    verdicts = {}
    noise = {}
    for arm, grid in ORDER:
        worst = 0.0
        det_sds = []
        for v in grid:
            a = acc.get((arm, v))
            if not a or not a["det"]:
                rows.append([arm, f"{v:g}", C.NA, C.NA, C.NA, C.NA, C.NA])
                continue
            dm, dsd = C.agg(a["det"])
            fm, _ = C.agg(a["fa"])
            cm, _ = C.agg(a["ctrl"]) if a["ctrl"] else (None, None)
            if dsd:
                det_sds.append(dsd)
            # Mean carries a sign (the direction of the effect is the point); the
            # STD does not. `C.cell(fmt="{:+.4f}")` applied the sign to both and
            # rendered "-0.0001 ± +0.0004", which reads as a signed dispersion.
            delta = C.NA
            if a["d"]:
                mean_d, sd_d = C.agg(a["d"])
                delta = f"{mean_d:+.4f}" + (f" ± {sd_d:.4f}" if sd_d else "")
                worst = max(worst, abs(mean_d))
            rows.append(
                [
                    arm,
                    f"{v:g}",
                    C.cell(a["fa"], fmt="{:.4f}"),
                    C.cell(a["det"]),
                    f"{cm:.3f}" if cm is not None else C.NA,
                    delta,
                    f"{dm - fm:+.3f}",
                ]
            )
        verdicts[arm] = worst
        # The noise floor the verdict is judged against: the arm's own
        # seed-to-seed dispersion in the detection column. A |delta| below the
        # spread of the quantity it is a difference of cannot be a real effect,
        # and that is a measured bar rather than a constant someone chose.
        noise[arm] = float(np.median(det_sds)) if det_sds else float("nan")
        _p(
            f"  [{arm}] largest abs delta-detection {worst:.4f} "
            f"vs detection seed-spread {noise[arm]:.4f}"
        )

    # The bar is the arm's own detection seed-spread, not a chosen constant: a
    # difference smaller than the dispersion of the quantity being differenced
    # is not an effect. `nan` spread (a fully deterministic column) falls back
    # to a strict absolute bar.
    verdict_txt = "; ".join(
        f"**{arm}**: largest |Δ| {w:.4f} against a detection seed-spread of "
        + (f"{noise[arm]:.4f}" if np.isfinite(noise[arm]) else "0")
        + " — "
        + (
            "below the noise floor, so a threshold in disguise on this data"
            if w <= (noise[arm] if np.isfinite(noise[arm]) else 0.001)
            else "ABOVE the noise floor, so it does something a threshold cannot"
        )
        for arm, w in verdicts.items()
    )

    C.emit(
        "table_4_11e_beth_boost_sweep",
        "Table 4.11(e) — BETH: does each arm's knob beat a plain score threshold?",
        [
            "Knob",
            "Setting",
            "Realized test false-alarm",
            "Detection (knob)",
            "Detection (plain threshold, matched FA)",
            "Δ detection",
            "Youden's J",
        ],
        rows,
        note=(
            f"The question: at a **matched false-alarm rate**, does turning each arm's own "
            f"knob detect more than simply moving the decision threshold on that same arm's "
            f"continuous score? 'Δ detection' is knob minus plain threshold; **0.000 across "
            f"a grid means the knob is a reparameterisation of the threshold**, which is a "
            f"fine thing to be but not a claim to make about it. The control threshold is "
            f"the (1 − realized FA) quantile of the arm's own benign *test* scores, so both "
            f"numbers in a row sit at one operating point rather than on two curves. "
            f"**Analytic prediction for θ, which this table checks rather than assumes:** "
            f"`_anomaly_argmax` forms the anomaly column as "
            f"`complement(conorm(clip(class_firing + θ, 0, 1)))`, and with a single known "
            f"class there is exactly one class column — `t_conorm(x, None, …)` aggregates "
            f"column-wise, so it is the identity. The anomaly label therefore wins exactly "
            f"when `firing < (1 − θ)/2`, i.e. **in the one-class setting θ is a hard "
            f"threshold on firing strength and the norm/conorm family is irrelevant because "
            f"there is one column to aggregate.** Chapter 4 §4.3 argues a weaker version of "
            f"this for the multi-class case (that θ=0.99 degenerates to a max-membership "
            f"rejector); the one-class reduction makes it total at every θ. "
            f"**Isolation Forest's `contamination` is the method's control**: it provably "
            f"only sets `offset_` from a quantile of the training scores and never touches "
            f"the trees, so its Δ must be 0.000 — that it comes back 0.000 is what licenses "
            f"reading a non-zero Δ elsewhere as real, and one fit per seed is therefore "
            f"correct rather than a shortcut. **`ν` is the one genuinely different knob**: it "
            f"enters libsvm's QP objective, so every value is a different fitted model, and "
            f"its control is one fixed (ν=0.01) model's score thresholded. Measured verdicts "
            f"— {verdict_txt}. Every arm is fitted on the SAME {FIT_N:,}-row benign "
            f"subsample per seed, so the sample-count confound Table 4.11(d) corrects is not "
            f"re-introduced; no wall-clock is reported because this table is about decision "
            f"curves and timing belongs to (d). Scored on the full {meta['n_test']:,}-row "
            f"test split ({int(is_anom_te.sum()):,} anomalous); mean ± sample std across "
            f"common.SEEDS ({C.SEEDS})."
        ),
    )


if __name__ == "__main__":
    main()
