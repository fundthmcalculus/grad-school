"""Stage 16 -- the false-premise niche, properly controlled.

§19 found the one class where the fuzzy rule beat entropy (0.786 vs 0.561), but
on a template-confounded comparison. This re-tests it on `prompts_v3.jsonl`,
where real and invented subjects share the surface form and both sides produce
fluent, discursive answers.

Controls are stacked so the cost of each is visible:

  raw               no matching
  length            n_tokens matched
  length+template   + template matched  <- the honest number
  + entropy         + entropy quartile matched (the hardest test)

Feature families from §18 are compared head to head, since `deltaref` (late-layer
update geometry) was the best there and is a different view from `centroid`.

Reports accuracy alongside tunable parameters, training time and inference speed,
per the standing-report convention (§14).
"""

import argparse
import contextlib
import io
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tribblefis.gauss_math import create_gaussian_membership_dict
from analyze import DATA, SCALAR_COLS, fpr_at_tpr
from features_ext import build
from nopca import POOLING
from norm_sweep import anomaly_score, membership_tensors
from seed_sweep import K_MODES, PAIR, THETA, Timer
from template_control import match

warnings.filterwarnings("ignore")

SEEDS, TOP_N = 8, 8
FAMILIES = ["centroid", "delta", "deltaref", "geom", "curve", "agg"]


def splits(meta, seed):
    rng = np.random.default_rng(seed)
    good = np.flatnonzero((meta.family == "longform_real")
                          & (meta.label == "grounded"))
    bad = np.flatnonzero((meta.family == "longform_fake")
                         & (meta.label == "hallucination"))
    rng.shuffle(good)
    rng.shuffle(bad)
    return {"fit": good[:int(.6 * len(good))],
            "test_neg": good[int(.6 * len(good)):], "test_pos": bad}


def fit_fis_on(F, fit, seed, top_n=TOP_N):
    feats0 = list(F.iloc[fit].var().sort_values(ascending=False).index[:top_n])
    Xf = StandardScaler().fit_transform(F.iloc[fit][feats0].to_numpy())
    labels = KMeans(n_clusters=K_MODES, n_init=10, random_state=seed).fit(Xf).labels_
    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                           take_top_features)
        diff = calculate_gaussian_correlation(
            F.iloc[fit].reset_index(drop=True), y_modes)
        _, feats = take_top_features(diff, top_n=top_n)
        model = create_gaussian_membership_dict(
            F.iloc[fit][feats].reset_index(drop=True), y_modes,
            top_n_var_names=feats)

    def score(ix):
        classes, tens = membership_tensors(F, ix, model, feats)
        with np.errstate(all="ignore"):
            return np.asarray(anomaly_score(classes, tens, *PAIR, THETA), float)
    return score, feats, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()

    meta = pd.read_parquet(DATA / "capture_v3_meta.parquet")
    H = np.load(DATA / f"capture_v3_hidden_{POOLING}.npy")
    print("label counts:")
    print(pd.crosstab(meta.family, meta.label).to_string())

    rows = []
    for seed in range(args.seeds):
        sp = splits(meta, seed)
        rng = np.random.default_rng(17000 + seed)
        fam = build(meta, H, sp["fit"])
        Fs = meta[SCALAR_COLS].reset_index(drop=True).astype(float)

        # entropy-quartile key for the hardest condition
        ent = meta["ent_mean"].astype(float)
        m2 = meta.copy()
        m2["_eb"] = np.digitize(ent, np.quantile(ent.iloc[np.concatenate(
            [sp["test_neg"], sp["test_pos"]])], [.25, .5, .75]))

        conds = {
            "raw": None,
            "length": ["n_tokens"],
            "length+template": ["template", "n_tokens"],
            "length+template+entropy": ["template", "n_tokens", "_eb"],
        }

        detectors = {}
        for fname in FAMILIES:
            F = fam[fname]
            with Timer() as t:
                sc, feats, model = fit_fis_on(F, sp["fit"], seed)
            detectors[f"FIS · {fname}"] = (
                sc, t.ms, 2 * model.n_membership_functions + 1,
                f"{model.n_rules}R/{model.n_membership_functions}MF")

        # one-class SVM on the best family's antecedents, for a same-features rival
        Fb = fam["deltaref"]
        fb = list(Fb.iloc[sp["fit"]].var().sort_values(ascending=False).index[:TOP_N])
        with Timer() as t:
            csc = StandardScaler().fit(Fb.iloc[sp["fit"]][fb].to_numpy())
            oc = OneClassSVM(nu=.1, gamma="scale").fit(
                csc.transform(Fb.iloc[sp["fit"]][fb].to_numpy()))
        detectors["OneClassSVM · deltaref"] = (
            lambda ix: -oc.score_samples(csc.transform(Fb.iloc[ix][fb].to_numpy())),
            t.ms, oc.support_vectors_.shape[0] * TOP_N + oc.support_vectors_.shape[0] + 1,
            f"{oc.support_vectors_.shape[0]}SV")

        d = Fs.shape[1]
        with Timer() as t:
            ssc = StandardScaler().fit(Fs.iloc[sp["fit"]].to_numpy())
            lw = LedoitWolf().fit(ssc.transform(Fs.iloc[sp["fit"]].to_numpy()))
        detectors["Mahalanobis · stats"] = (
            lambda ix: lw.mahalanobis(ssc.transform(Fs.iloc[ix].to_numpy())),
            t.ms, d + d * (d + 1) // 2, f"{d}f")
        for nm, col in (("mean entropy", "ent_mean"), ("perplexity", "perplexity"),
                        ("n_tokens (control)", "n_tokens")):
            detectors[nm] = (lambda ix, c=col: meta.iloc[ix][c].to_numpy(float),
                             0.0, 0, "threshold")

        for cond, keys in conds.items():
            if keys is None:
                a, b = sp["test_neg"], sp["test_pos"]
            else:
                a, b = match(m2, sp["test_neg"], sp["test_pos"], keys, rng)
            if len(a) < 20:
                continue
            ix = np.concatenate([a, b])
            y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
            for nm, (fn, tms, npar, struct) in detectors.items():
                with Timer() as t:
                    s = np.asarray(fn(ix), float)
                ok = np.isfinite(s).all() and np.ptp(s) > 0
                rows.append({"seed": seed, "condition": cond, "detector": nm,
                             "auroc": roc_auc_score(y, s) if ok else np.nan,
                             "fpr95": fpr_at_tpr(y, s) if ok else np.nan,
                             "n_neg": len(a), "n_pos": len(b),
                             "train_ms": tms, "n_params": npar,
                             "samples_per_sec": len(ix) / max(t.ms / 1000, 1e-9),
                             "structure": struct})
        print(f"  seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "falsepremise_deep.csv", index=False)

    order = list(conds)
    piv = df.pivot_table(index="detector", columns="condition", values="auroc",
                         aggfunc=["mean", "std"])
    cols = [c for c in order if ("mean", c) in piv.columns]
    tbl = pd.DataFrame({c: piv[("mean", c)].map("{:.3f}".format) + " ± "
                        + piv[("std", c)].map("{:.3f}".format) for c in cols})
    key = ("mean", cols[-1])
    tbl = tbl.reindex(piv[key].sort_values(ascending=False).index)
    print(f"\n{'='*104}\nAUROC by control, long-form template-matched probes "
          f"({args.seeds} seeds)\n{'='*104}")
    print(tbl.to_string())
    print(f"\nmean test sizes:\n"
          f"{df.groupby('condition')[['n_neg','n_pos']].mean().round(0).to_string()}")

    # standing report on the strictest condition
    strict = cols[-1]
    s = df[df.condition == strict]
    rep = (s.groupby("detector")
           .agg(auroc=("auroc", "mean"), auroc_std=("auroc", "std"),
                fpr95=("fpr95", "mean"), n_params=("n_params", "mean"),
                train_ms=("train_ms", "mean"),
                samples_per_sec=("samples_per_sec", "mean"))
           .sort_values("auroc", ascending=False))
    rep["structure"] = s.groupby("detector")["structure"].first()
    print(f"\n{'='*104}\nSTANDING REPORT — condition '{strict}'\n{'='*104}")
    out = rep.copy()
    out["AUROC"] = (out.auroc.map("{:.3f}".format) + " ± "
                    + out.auroc_std.map("{:.3f}".format))
    out["params"] = out.n_params.map(lambda v: f"{int(v):,}")
    out["train"] = out.train_ms.map(lambda v: f"{v:,.0f} ms")
    out["infer"] = out.samples_per_sec.map(lambda v: f"{v:,.0f}/s")
    print(out[["AUROC", "fpr95", "params", "train", "infer", "structure"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))

    # paired test: best FIS family vs mean entropy, on the strictest condition
    m = s.pivot_table(index="seed", columns="detector", values="auroc")
    fis_cols = [c for c in m.columns if c.startswith("FIS")]
    if fis_cols and "mean entropy" in m.columns:
        best = m[fis_cols].mean().idxmax()
        dd = (m[best] - m["mean entropy"]).dropna()
        p = sstats.wilcoxon(dd)[1] if len(dd) >= 6 else np.nan
        print(f"\npaired, condition '{strict}': {best} vs mean entropy")
        print(f"  mean Δ = {dd.mean():+.3f} ± {dd.std():.3f} "
              f"(min {dd.min():+.3f}, max {dd.max():+.3f}) · "
              f"wins {int((dd > 0).sum())}/{len(dd)}"
              + (f" · p = {p:.4f}" if np.isfinite(p) else ""))

    # per-subtype breakdown for the winner
    print(f"\n{'='*104}\nPER-SUBTYPE (condition 'length+template', best FIS family)"
          f"\n{'='*104}")
    sp0 = splits(meta, 0)
    fam0 = build(meta, H, sp0["fit"])
    bestfam = (best.split("· ")[1] if fis_cols and "·" in best else "deltaref")
    sc, _, _ = fit_fis_on(fam0.get(bestfam, fam0["deltaref"]), sp0["fit"], 0)
    rng = np.random.default_rng(0)
    for t in sorted(meta.template.unique()):
        neg = np.array([i for i in sp0["test_neg"] if meta.iloc[i].template == t])
        pos = np.array([i for i in sp0["test_pos"] if meta.iloc[i].template == t])
        if len(neg) < 8 or len(pos) < 8:
            continue
        a, b = match(meta, neg, pos, ["n_tokens"], rng)
        if len(a) < 8:
            continue
        ix = np.concatenate([a, b])
        y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
        f_au = roc_auc_score(y, sc(ix))
        e_au = roc_auc_score(y, meta.iloc[ix].ent_mean.to_numpy(float))
        print(f"  {t:<12} n={len(ix):>4}   FIS·{bestfam} {f_au:.3f}   "
              f"entropy {e_au:.3f}   Δ {f_au - e_au:+.3f}")


if __name__ == "__main__":
    main()
