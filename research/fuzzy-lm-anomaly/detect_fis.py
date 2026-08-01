"""Stage 3b -- the tribble anomaly rule as a hallucination detector.

Transplants the Ch 4.3.5 "none of the above" construction from host telemetry
onto language-model internals:

    mu_anom(x) = t_complement( t_conorm( mu_1 + theta, ..., mu_K + theta ) )

Nothing here is trained to recognise hallucination. We fit a Mixture-of-Gaussian
FIS to the *known-good* behaviour of the frozen SLM and let the anomaly rule
fire wherever no legitimate rule matches -- exactly the BETH benign-only,
open-set protocol.

The K classes are not given. Ch 4.3.5 had real class labels; here the only
label is "truthful", and `calculate_gaussian_correlation` scores features by
*pairwise* label comparison, so a single class yields zero discriminative
signal. We therefore recover K **behavioural modes** by clustering the truthful
fit split in PCA space (which is also the answer to the PCA/SVD question) and
use those cluster ids as the FIS antecedent classes.

Reports FIS anomaly firing against the same test split as `analyze.py`.
"""

import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from tribblefis.gauss_data import AnomalyParameters
from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
    tsk_firing_strengths,
)

from analyze import DATA, SCALAR_COLS, SEED, fpr_at_tpr, make_splits

warnings.filterwarnings("ignore")


def build_features(meta, hidden, sp, cfg, use_hidden=True, use_stats=True):
    """Feature table: PCA components of the residual stream + distribution stats.

    Scaler and PCA are fit on the truthful fit split only, so the hallucination
    class never influences the representation.
    """
    parts = []
    if use_hidden:
        X = hidden[:, cfg["layer"], :].astype(np.float32)
        if cfg["l2"]:
            X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
        sc = StandardScaler().fit(X[sp["fit"]])
        p = PCA(n_components=cfg["k"], random_state=SEED).fit(sc.transform(X[sp["fit"]]))
        Z = p.transform(sc.transform(X))
        parts.append(pd.DataFrame(Z, columns=[f"PC{i+1:02d}" for i in range(Z.shape[1])]))
    if use_stats:
        parts.append(meta[SCALAR_COLS].reset_index(drop=True).astype(float))
    return pd.concat(parts, axis=1)


def discover_modes(F, fit_idx, k_fixed=0, k_range=(2, 9), verbose=True):
    """Cluster the known-good split into behavioural modes.

    With k_fixed=0 the mode count is chosen by silhouette; otherwise K is
    forced, so the sweep can ask whether more behavioural modes buy accuracy.
    """
    Xf = StandardScaler().fit_transform(F.iloc[fit_idx].to_numpy())
    if k_fixed:
        km = KMeans(n_clusters=k_fixed, n_init=10, random_state=SEED).fit(Xf)
        return k_fixed, km.labels_, silhouette_score(Xf, km.labels_)
    best = (None, -2, None)
    for k in range(*k_range):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xf)
        s = silhouette_score(Xf, km.labels_)
        if verbose:
            print(f"   K={k:<2} silhouette={s:.3f}")
        if s > best[1]:
            best = (k, s, km)
    return best[0], best[2].labels_, best[1]


def fit_fis(F, sp, labels, feature_names):
    """Fit the MoG FIS on the discovered known-good modes.

    theta only enters the anomaly aggregation, never the fit, so the model is
    built once and reused across the whole theta sweep.
    """
    X_fit = F.iloc[sp["fit"]][feature_names].reset_index(drop=True)
    y_fit = pd.Series([f"mode{c}" for c in labels])

    model = create_gaussian_membership_dict(X_fit, y_fit, top_n_var_names=feature_names)
    print(f"   rules={model.n_rules}  mfs={model.n_membership_functions}  "
          f"possible_rules={model.possible_rules}")

    def score(idx, anomaly):
        fs, lab = tsk_firing_strengths(F.iloc[idx][feature_names].reset_index(drop=True),
                                       model, anomaly)
        return fs[:, lab.index(anomaly.label)]

    return score, model


def evaluate(score_fn, anomaly, sp, tag, pos_key, rows, name, extra):
    """Score one test family, tolerating degenerate anomaly columns.

    The Hamacher t-norm is x*y/(x+y-x*y), which is 0/0 when both memberships
    underflow -- unavoidable once the antecedent count is large enough that
    some Gaussian evaluates to exactly 0. We record such runs as NaN with the
    non-finite fraction rather than dropping them, since that fragility is
    itself a finding about the Ch 4.3.5 construction at high dimension.
    """
    ix = np.concatenate([sp["test_neg"], sp[pos_key]])
    y = np.concatenate([np.zeros(len(sp["test_neg"])), np.ones(len(sp[pos_key]))])
    s = np.asarray(score_fn(ix, anomaly), dtype=float)

    bad = float(np.mean(~np.isfinite(s)))
    row = {"family": tag, "detector": name, **extra, "nonfinite": bad}
    if bad > 0 or np.ptp(s) == 0:
        rows.append({**row, "auroc": np.nan, "auprc": np.nan, "fpr@95tpr": np.nan})
        return
    rows.append({**row, "auroc": roc_auc_score(y, s),
                 "auprc": average_precision_score(y, s), "fpr@95tpr": fpr_at_tpr(y, s)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta", type=float, default=0.5,
                    help="anomaly boost (rank-invariant for min/max; sets operating point)")
    args = ap.parse_args()
    NORMS = ["min/max", "probability", "hamacher"]

    meta = pd.read_parquet(DATA / "capture_meta.parquet")
    cfg = json.loads((DATA / "best_repr.json").read_text())
    cfg = {"pooling": cfg["pooling"], "layer": int(cfg["layer"]),
           "k": int(cfg["k"]), "l2": bool(cfg["l2"])}
    hidden = np.load(DATA / f"capture_hidden_{cfg['pooling']}.npy")
    sp = make_splits(meta)
    print(f"representation (selected on val): {cfg}")

    variants = {
        "FIS fused (hidden+stats)": dict(use_hidden=True, use_stats=True),
        "FIS stats only": dict(use_hidden=False, use_stats=True),
        "FIS hidden only": dict(use_hidden=True, use_stats=False),
    }

    # theta is rank-invariant for the min/max conorm (it adds a constant to every
    # class firing before aggregation), so it moves the operating point but not
    # AUROC. Verified empirically; we hold it fixed and sweep what does matter:
    # antecedent budget, mode count, and the norm pair.
    TOP_N = [8, 12, 24, 48]
    K_GRID = [0, 2, 4, 8, 16]          # 0 = silhouette-selected
    rows = []

    for name, kw in variants.items():
        print(f"\n{'='*72}\n{name}\n{'='*72}")
        F = build_features(meta, hidden, sp, cfg, **kw)
        n_feat = F.shape[1]
        _, _, sil = discover_modes(F, sp["fit"])
        for K in K_GRID:
            Kx, labels, sil = discover_modes(F, sp["fit"], k_fixed=K, verbose=False)
            y_modes = pd.Series([f"mode{c}" for c in labels])
            diff = calculate_gaussian_correlation(
                F.iloc[sp["fit"]].reset_index(drop=True), y_modes)
            for tn in TOP_N:
                if tn > n_feat:
                    continue
                _, feats = take_top_features(diff, top_n=tn)
                with contextlib.redirect_stdout(io.StringIO()):
                    score, model = fit_fis(F, sp, labels, feats)
                for norm in NORMS:
                    anomaly = AnomalyParameters(include_anomaly=True, threshold=args.theta,
                                                label="anomaly", norm_conorm=norm,
                                                member_function="gaussian")
                    for tag, pk in [("TriviaQA", "test_tq"), ("FalsePremise", "test_fp")]:
                        evaluate(score, anomaly, sp, tag, pk, rows, name,
                                 {"norm": norm, "K": Kx, "top_n": tn,
                                  "silhouette": sil, "mfs": model.n_membership_functions})
            print(f"   K={Kx:<2} (sil {sil:.3f}) swept top_n={TOP_N}")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "fis_results.csv", index=False)

    nan_rate = df.groupby("norm")["auroc"].apply(lambda s: s.isna().mean())
    print(f"\n{'='*72}\nNUMERICAL STABILITY (fraction of runs that degenerated)\n{'='*72}")
    print(nan_rate.to_string(float_format=lambda v: f"{v:.2f}"))

    for tag in ("TriviaQA", "FalsePremise"):
        sub = (df[df["family"] == tag].dropna(subset=["auroc"])
               .sort_values("auroc", ascending=False))
        print(f"\n{'='*72}\nFIS ANOMALY RULE -- {tag}  (top 10 of {len(sub)} valid)\n{'='*72}")
        print(sub.drop(columns=["family"]).head(10).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))

        base = pd.read_csv(DATA / f"baselines_{tag.lower()}.csv")
        print(f"\nbest baseline: {base.iloc[0]['detector']} "
              f"(AUROC {base.iloc[0]['auroc']:.3f})")
        if len(sub):
            b = sub.iloc[0]
            print(f"best FIS     : {b['detector']} norm={b['norm']} K={b['K']} "
                  f"top_n={b['top_n']} (AUROC {b['auroc']:.3f})")


if __name__ == "__main__":
    main()
