"""Stage 6 -- seed sweep with error bars, plus training/scoring cost.

§8's headline rested on one split of 170 matched pairs per class. This re-runs
the whole comparison over N independent splits and reports mean ± std, so the
claim either survives with error bars or does not.

Each seed re-draws the fit/val/test split, which also re-fits everything that
depends on it: the truthful centroid, the PCA basis, the KMeans modes, the
antecedent ranking, and the rule base. The length-matched subsample is redrawn
too. Nothing is carried across seeds.

The decisive statistic is **paired**: per seed, (FIS − best baseline) on the
length-matched split. A mean advantage that is large relative to its std across
seeds is real; one that flips sign between seeds is not.

Also records cost, since an interpretable detector that is slow to fit is a
different proposition from one that is not:

  feat_ms   one-time feature construction over all 7,500 rows
  fit_ms    split-dependent fitting -- basis, clustering, ranking, rule base
  score_ms  scoring cost, normalised per 1,000 samples
"""

import argparse
import contextlib
import io
import json
import sys
import time
import warnings
from pathlib import Path

# Windows consoles default to cp1252, which cannot encode the report's Delta /
# plus-minus / middot glyphs; without this the run dies AFTER doing all the work.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
)

from analyze import DATA, SCALAR_COLS, fpr_at_tpr, make_splits
from length_control import exact_match_on_length
from nopca import POOLING, f_centroid, f_stats
from norm_sweep import anomaly_score, membership_tensors

warnings.filterwarnings("ignore")

TOP_N, K_MODES = 8, 2
PAIR = ("dombi2", "einstein")
THETA = 0.5


class Timer:
    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t) * 1e3


def fit_fis(F, sp, seed, top_n=TOP_N, K=K_MODES):
    """Cluster known-good modes, rank antecedents, fit the rule base."""
    feats0 = list(F.iloc[sp["fit"]].var().sort_values(ascending=False).index[:top_n])
    Xf = StandardScaler().fit_transform(F.iloc[sp["fit"]][feats0].to_numpy())
    labels = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Xf).labels_
    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        diff = calculate_gaussian_correlation(
            F.iloc[sp["fit"]].reset_index(drop=True), y_modes)
        _, feats = take_top_features(diff, top_n=top_n)
        model = create_gaussian_membership_dict(
            F.iloc[sp["fit"]][feats].reset_index(drop=True), y_modes,
            top_n_var_names=feats)

    def fn(ix):
        classes, tens = membership_tensors(F, ix, model, feats)
        with np.errstate(all="ignore"):
            return anomaly_score(classes, tens, *PAIR, THETA)
    return fn, feats, model


def build_detectors(meta, Fc, Fs, Fp, sp, seed, feat_ms):
    """Return {name: (scorer, fit_ms, feat_ms)} for one split."""
    out = {}

    with Timer() as t:
        fis_c, feats_c, model_c = fit_fis(Fc, sp, seed)
    out["FIS · centroid (PCA-free)"] = (fis_c, t.ms, feat_ms["centroid"],
                                        model_c.n_membership_functions)

    with Timer() as t:
        fis_p, _, model_p = fit_fis(Fp, sp, seed)
    out["FIS · PCA (64 comp)"] = (fis_p, t.ms, feat_ms["pca"],
                                  model_p.n_membership_functions)

    with Timer() as t:
        ssc = StandardScaler().fit(Fs.iloc[sp["fit"]].to_numpy())
        lw = LedoitWolf().fit(ssc.transform(Fs.iloc[sp["fit"]].to_numpy()))
    out["Mahalanobis · stats"] = (
        lambda ix: lw.mahalanobis(ssc.transform(Fs.iloc[ix].to_numpy())),
        t.ms, feat_ms["stats"], np.nan)

    with Timer() as t:
        isc = StandardScaler().fit(Fs.iloc[sp["fit"]].to_numpy())
        iso = IsolationForest(random_state=seed).fit(
            isc.transform(Fs.iloc[sp["fit"]].to_numpy()))
    out["IsolationForest · stats"] = (
        lambda ix: -iso.score_samples(isc.transform(Fs.iloc[ix].to_numpy())),
        t.ms, feat_ms["stats"], np.nan)

    with Timer() as t:
        csc = StandardScaler().fit(Fc.iloc[sp["fit"]][feats_c].to_numpy())
        oc = OneClassSVM(nu=.1, gamma="scale").fit(
            csc.transform(Fc.iloc[sp["fit"]][feats_c].to_numpy()))
    out["OneClassSVM · centroid"] = (
        lambda ix: -oc.score_samples(csc.transform(Fc.iloc[ix][feats_c].to_numpy())),
        t.ms, feat_ms["centroid"], np.nan)

    # training-free reference detectors
    for nm, col in (("perplexity", "perplexity"), ("mean entropy", "ent_mean"),
                    ("n_tokens (control)", "n_tokens")):
        out[nm] = (lambda ix, c=col: meta.iloc[ix][c].to_numpy(float),
                   0.0, 0.0, np.nan)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--report-only", action="store_true",
                    help="re-report from seed_sweep.csv without refitting")
    args = ap.parse_args()

    meta = pd.read_parquet(DATA / "capture_meta.parquet")
    hidden = np.load(DATA / f"capture_hidden_{POOLING}.npy")
    cfg = json.loads((DATA / "best_repr.json").read_text())
    layer, ncomp = int(cfg["layer"]), int(cfg["k"])

    if args.report_only:
        df = pd.read_csv(DATA / "seed_sweep.csv")
        report(df, df.seed.nunique())
        return

    rows = []
    for seed in range(args.seeds):
        sp = make_splits(meta, seed=seed)
        rng = np.random.default_rng(1000 + seed)
        feat_ms = {}

        with Timer() as t:
            Fs = f_stats(meta, hidden, sp)
        feat_ms["stats"] = t.ms
        with Timer() as t:
            Fc = f_centroid(meta, hidden, sp).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0)
        feat_ms["centroid"] = t.ms
        with Timer() as t:
            X = hidden[:, layer, :].astype(np.float32)
            psc = StandardScaler().fit(X[sp["fit"]])
            pca = PCA(n_components=ncomp, random_state=seed).fit(
                psc.transform(X[sp["fit"]]))
            Z = pca.transform(psc.transform(X))
            Fp = pd.DataFrame(Z, columns=[f"PC{i+1:02d}" for i in range(ncomp)])
        feat_ms["pca"] = t.ms

        dets = build_detectors(meta, Fc, Fs, Fp, sp, seed, feat_ms)

        for tag, pk in [("FalsePremise", "test_fp"), ("TriviaQA", "test_tq")]:
            neg, pos = sp["test_neg"], sp[pk]
            mneg, mpos = exact_match_on_length(meta, neg, pos, rng)
            for name, (fn, fit_ms, fms, mfs) in dets.items():
                for cond, (a, b) in (("raw", (neg, pos)),
                                     ("matched", (mneg, mpos))):
                    if len(a) < 30:
                        continue
                    ix = np.concatenate([a, b])
                    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
                    with Timer() as t:
                        s = np.asarray(fn(ix), dtype=float)
                    ok = np.isfinite(s).all() and np.ptp(s) > 0
                    rows.append({
                        "seed": seed, "family": tag, "condition": cond,
                        "detector": name,
                        "auroc": roc_auc_score(y, s) if ok else np.nan,
                        "fpr@95tpr": fpr_at_tpr(y, s) if ok else np.nan,
                        "n_pos": len(b), "n_neg": len(a),
                        "feat_ms": fms, "fit_ms": fit_ms,
                        "score_ms_per_1k": t.ms / len(ix) * 1000,
                        "n_mfs": mfs,
                    })
        print(f"  seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "seed_sweep.csv", index=False)
    report(df, args.seeds)


def report(df, n_seeds):
    # ---------------- accuracy with error bars ----------------------------
    for tag in ("FalsePremise", "TriviaQA"):
        print(f"\n{'='*88}\n{tag} — AUROC over {n_seeds} seeds (mean ± std)\n{'='*88}")
        sub = df[df.family == tag]
        piv = sub.pivot_table(index="detector", columns="condition",
                              values="auroc", aggfunc=["mean", "std"])
        tbl = pd.DataFrame({
            "raw": piv[("mean", "raw")].map("{:.3f}".format) + " ± "
                   + piv[("std", "raw")].map("{:.3f}".format),
            "matched": piv[("mean", "matched")].map("{:.3f}".format) + " ± "
                       + piv[("std", "matched")].map("{:.3f}".format),
        }).reindex(piv[("mean", "matched")].sort_values(ascending=False).index)
        print(tbl.to_string())

        # paired test: FIS vs the best rival, per seed, on matched data
        m = sub[sub.condition == "matched"].pivot_table(
            index="seed", columns="detector", values="auroc")
        fis = "FIS · centroid (PCA-free)"
        if fis in m:
            rivals = [c for c in m.columns if c not in (fis, "n_tokens (control)")]
            best_rival = m[rivals].mean().idxmax()
            d = (m[fis] - m[best_rival]).dropna()
            if len(d) > 1:
                tstat, p = sstats.wilcoxon(d) if len(d) >= 6 else (np.nan, np.nan)
                print(f"\n  paired advantage of FIS over best rival "
                      f"({best_rival}):")
                print(f"    mean Δ = {d.mean():+.3f} ± {d.std():.3f}  "
                      f"(min {d.min():+.3f}, max {d.max():+.3f})")
                print(f"    wins {int((d > 0).sum())}/{len(d)} seeds"
                      + (f" · Wilcoxon p = {p:.4f}" if np.isfinite(p) else ""))

    # ---------------- cost ------------------------------------------------
    print(f"\n{'='*88}\nTRAINING AND SCORING COST (mean over seeds)\n{'='*88}")
    cost = (df.groupby("detector")
            .agg(feat_ms=("feat_ms", "mean"), fit_ms=("fit_ms", "mean"),
                 score_ms_per_1k=("score_ms_per_1k", "mean"),
                 n_mfs=("n_mfs", "mean"))
            .sort_values("fit_ms"))
    cost["total_train_ms"] = cost.feat_ms + cost.fit_ms
    print(cost.to_string(float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main()
