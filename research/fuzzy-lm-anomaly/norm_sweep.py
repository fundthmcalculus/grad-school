"""Stage 3c -- the anomaly rule over the full (t-norm, t-conorm) grid.

`tsk_firing_strengths` drives both operators from one `norm_conorm` string, so
only four *matched* choices are reachable and the Hamacher conorm it uses is
out of range. Here we extract the fitted membership functions from the tribble
model and re-aggregate them ourselves, which makes three things possible:

* every family in `norms.py`, all axiom-checked;
* **mismatched** pairs -- a conjunctive T with a disjunctive S and vice versa;
* clipping mu+theta back into [0,1] before aggregation, which the library
  omits (theta can push a membership above 1, leaving the operator's domain).

The aggregation reproduces the library's structure exactly:

    feature_mu[c][f] = S over the MFs of (class c, feature f)
    firing[c]        = T over features
    mu_anom          = 1 - S over classes of clip(firing[c] + theta)

Only the operators change, so any difference in the results is attributable to
the operator choice alone.
"""

import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from tribblefis.gauss_data import AnomalyParameters
from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
    tsk_firing_strengths,
)

from analyze import DATA, SEED, fpr_at_tpr, make_splits
from detect_fis import build_features, discover_modes
from norms import FAMILIES, TCONORMS, TNORMS, reduce_op

warnings.filterwarnings("ignore")


def membership_tensors(F, idx, model, feats):
    """Per-(class, feature) membership evaluations for the given rows.

    Returns (classes, {class: [ (n, m_cf) array per feature ]}).
    """
    X = F.iloc[idx][feats]
    classes = list(next(iter(model.feature_models.values())).ordered_keys)
    out = {}
    for c in classes:
        per_feat = []
        for f in feats:
            fm = model.feature_models[f]
            if c not in fm.label_models:
                continue
            col = X[f].to_numpy()
            per_feat.append(np.stack([mf.evaluate(col)
                                      for mf in fm.label_models[c].memberships], axis=1))
        out[c] = per_feat
    return classes, out


def anomaly_score(classes, tens, tname, sname, theta):
    """mu_anom under an arbitrary (T, S) pair, with mu+theta clipped to [0,1]."""
    T, S = TNORMS[tname], TCONORMS[sname]
    firings = []
    for c in classes:
        feat_mu = [reduce_op(S, [m[:, k] for k in range(m.shape[1])]) for m in tens[c]]
        firings.append(np.clip(reduce_op(T, feat_mu) + theta, 0.0, 1.0))
    return 1.0 - reduce_op(S, firings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--top-n", type=int, default=8)
    ap.add_argument("--k-modes", type=int, default=2)
    ap.add_argument("--matched-only", action="store_true")
    args = ap.parse_args()

    meta = pd.read_parquet(DATA / "capture_meta.parquet")
    cfg = json.loads((DATA / "best_repr.json").read_text())
    cfg = {"pooling": cfg["pooling"], "layer": int(cfg["layer"]),
           "k": int(cfg["k"]), "l2": bool(cfg["l2"])}
    hidden = np.load(DATA / f"capture_hidden_{cfg['pooling']}.npy")
    sp = make_splits(meta)

    variants = {
        "fused": dict(use_hidden=True, use_stats=True),
        "stats": dict(use_hidden=False, use_stats=True),
        "hidden": dict(use_hidden=True, use_stats=False),
    }

    rows, parity = [], []
    for vname, kw in variants.items():
        F = build_features(meta, hidden, sp, cfg, **kw)
        _, labels, sil = discover_modes(F, sp["fit"], k_fixed=args.k_modes, verbose=False)
        y_modes = pd.Series([f"mode{c}" for c in labels])
        with contextlib.redirect_stdout(io.StringIO()):
            diff = calculate_gaussian_correlation(
                F.iloc[sp["fit"]].reset_index(drop=True), y_modes)
            _, feats = take_top_features(diff, top_n=args.top_n)
            model = create_gaussian_membership_dict(
                F.iloc[sp["fit"]][feats].reset_index(drop=True), y_modes,
                top_n_var_names=feats)
        print(f"[{vname}] K={args.k_modes} sil={sil:.3f} antecedents={len(feats)} "
              f"mfs={model.n_membership_functions}")

        for tag, pk in [("TriviaQA", "test_tq"), ("FalsePremise", "test_fp")]:
            ix = np.concatenate([sp["test_neg"], sp[pk]])
            y = np.concatenate([np.zeros(len(sp["test_neg"])), np.ones(len(sp[pk]))])
            classes, tens = membership_tensors(F, ix, model, feats)

            # parity check: our re-aggregation vs the library, on its own norms
            for lib_norm in ("min/max", "probability"):
                ours = anomaly_score(classes, tens,
                                     {"min/max": "minimum", "probability": "product"}[lib_norm],
                                     {"min/max": "minimum", "probability": "product"}[lib_norm],
                                     args.theta)
                fs, lab = tsk_firing_strengths(
                    F.iloc[ix][feats].reset_index(drop=True), model,
                    AnomalyParameters(True, args.theta, "anomaly", lib_norm, "gaussian"))
                theirs = fs[:, lab.index("anomaly")]
                parity.append({"variant": vname, "family": tag, "norm": lib_norm,
                               "max_abs_diff": float(np.nanmax(np.abs(ours - theirs)))})

            names = list(FAMILIES)
            pairs = ([(n, n) for n in names] if args.matched_only
                     else [(t, s) for t in names for s in names])
            for tn, sn in pairs:
                with np.errstate(all="ignore"):
                    s = anomaly_score(classes, tens, tn, sn, args.theta)
                s = np.asarray(s, dtype=float)
                bad = float(np.mean(~np.isfinite(s)))
                rec = {"variant": vname, "family": tag, "tnorm": tn, "sconorm": sn,
                       "matched": tn == sn, "nonfinite": bad}
                if bad > 0 or np.ptp(s) == 0:
                    rows.append({**rec, "auroc": np.nan, "auprc": np.nan,
                                 "fpr@95tpr": np.nan})
                else:
                    rows.append({**rec, "auroc": roc_auc_score(y, s),
                                 "auprc": average_precision_score(y, s),
                                 "fpr@95tpr": fpr_at_tpr(y, s)})

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "norm_sweep.csv", index=False)

    par = pd.DataFrame(parity)
    print(f"\n{'='*76}\nPARITY vs tribblefis (our re-aggregation on its own norms)\n{'='*76}")
    print(par.to_string(index=False, float_format=lambda v: f"{v:.2e}"))

    print(f"\n{'='*76}\nMATCHED DUAL PAIRS -- AUROC\n{'='*76}")
    for tag in ("TriviaQA", "FalsePremise"):
        m = df[(df.family == tag) & df.matched].pivot_table(
            index="tnorm", columns="variant", values="auroc", dropna=False)
        print(f"\n--- {tag} ---")
        print(m.reindex(list(FAMILIES)).to_string(float_format=lambda v: f"{v:.3f}",
                                                  na_rep="  --  "))

    if not args.matched_only:
        print(f"\n{'='*76}\nBEST PAIRS OVERALL (mismatched allowed)\n{'='*76}")
        for tag in ("TriviaQA", "FalsePremise"):
            sub = df[(df.family == tag)].dropna(subset=["auroc"]).nlargest(10, "auroc")
            print(f"\n--- {tag} (top 10 of {df[df.family==tag].auroc.notna().sum()} valid"
                  f" / {len(df[df.family==tag])} tried) ---")
            print(sub[["variant", "tnorm", "sconorm", "matched", "auroc", "auprc",
                       "fpr@95tpr"]].to_string(index=False,
                                               float_format=lambda v: f"{v:.3f}"))

        print(f"\n{'='*76}\nDEGENERATION RATE BY OPERATOR\n{'='*76}")
        deg = pd.concat([
            df.groupby("tnorm")["auroc"].apply(lambda s: s.isna().mean()).rename("as T"),
            df.groupby("sconorm")["auroc"].apply(lambda s: s.isna().mean()).rename("as S"),
        ], axis=1)
        print(deg.reindex(list(FAMILIES)).to_string(float_format=lambda v: f"{v:.2f}"))

    for tag in ("TriviaQA", "FalsePremise"):
        base = pd.read_csv(DATA / f"baselines_{tag.lower()}.csv")
        best = df[df.family == tag].dropna(subset=["auroc"]).nlargest(1, "auroc")
        if len(best):
            b = best.iloc[0]
            print(f"\n{tag}: best baseline {base.iloc[0]['detector']} "
                  f"{base.iloc[0]['auroc']:.3f} | best FIS pair "
                  f"T={b.tnorm}/S={b.sconorm} ({b.variant}) {b.auroc:.3f}")


if __name__ == "__main__":
    main()
