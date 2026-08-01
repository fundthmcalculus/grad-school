"""Stage 8 -- the template control. Is it fabrication, or prompt style?

§8 removed the length confound. This removes the remaining one. In the v1 probe
set the fabricated questions were templated and the truthful comparison set was
not, so a detector could separate them on surface form. `build_prompts_v2.py`
fixes that at the source: every fabricated question has a real-entity twin in
the identical surface form.

    real : "What is the capital city of Portugal?"    -> gradeable, truthful
    fake : "What is the capital city of Brazendia?"   -> necessarily fabricated

Both sides are fit and tested within the same five templates and the same three
phrasings, so template identity carries no information by construction. Three
conditions are reported side by side so the cost of each confound is visible:

  raw              no matching
  length           exact match on n_tokens
  length+template  exact match on (template, n_tokens)   <- the honest number

Everything is fit on **accurate (question, answer) pairs only** and swept over
N seeds.
"""

import argparse
import contextlib
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze import DATA, SCALAR_COLS, fpr_at_tpr
from nopca import POOLING, drop_constant, f_centroid, f_stats
from seed_sweep import fit_fis

warnings.filterwarnings("ignore")


def splits_v2(meta, seed):
    """Fit on template_real correct; test truthful = held-out template_real correct."""
    rng = np.random.default_rng(seed)
    real_ok = np.flatnonzero((meta.family == "template_real")
                             & (meta.label == "correct"))
    fake_bad = np.flatnonzero((meta.family == "template_fake")
                              & (meta.label == "hallucination"))
    rng.shuffle(real_ok)
    rng.shuffle(fake_bad)
    cut = int(0.6 * len(real_ok))
    return {"fit": real_ok[:cut], "test_neg": real_ok[cut:], "test_pos": fake_bad}


def match(meta, neg, pos, keys, rng):
    """Exact matching: keep min(#pos, #neg) within each cell of `keys`."""
    dn = meta.iloc[neg][keys].astype(str).agg("|".join, axis=1).to_numpy()
    dp = meta.iloc[pos][keys].astype(str).agg("|".join, axis=1).to_numpy()
    kn, kp = [], []
    for cell in np.union1d(np.unique(dn), np.unique(dp)):
        a, b = neg[dn == cell], pos[dp == cell]
        k = min(len(a), len(b))
        if k:
            kn.append(rng.choice(a, k, replace=False))
            kp.append(rng.choice(b, k, replace=False))
    return (np.concatenate(kn) if kn else np.array([], int),
            np.concatenate(kp) if kp else np.array([], int))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--prefix", default="capture_v2")
    args = ap.parse_args()

    meta = pd.read_parquet(DATA / f"{args.prefix}_meta.parquet")
    hidden = np.load(DATA / f"{args.prefix}_hidden_{POOLING}.npy")

    print("label counts by family:")
    print(pd.crosstab(meta.family, meta.label).to_string())
    n_real_ok = int(((meta.family == "template_real") & (meta.label == "correct")).sum())
    n_fake_bad = int(((meta.family == "template_fake")
                      & (meta.label == "hallucination")).sum())
    print(f"\ntemplate_real correct : {n_real_ok}")
    print(f"template_fake halluc  : {n_fake_bad}")
    if n_real_ok < 100:
        print("\nnot enough gradeable real answers to run the control")
        return

    CONDITIONS = [("raw", None), ("length", ["n_tokens"]),
                  ("length+template", ["template", "n_tokens"])]
    rows = []

    for seed in range(args.seeds):
        sp = splits_v2(meta, seed)
        rng = np.random.default_rng(5000 + seed)
        Fc = drop_constant(f_centroid(meta, hidden, sp).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0), sp["fit"])
        Fs = f_stats(meta, hidden, sp)

        with contextlib.redirect_stdout(io.StringIO()):
            fis, feats, model = fit_fis(Fc, sp, seed)
        ssc = StandardScaler().fit(Fs.iloc[sp["fit"]].to_numpy())
        lw = LedoitWolf().fit(ssc.transform(Fs.iloc[sp["fit"]].to_numpy()))
        csc = StandardScaler().fit(Fc.iloc[sp["fit"]][feats].to_numpy())
        oc = OneClassSVM(nu=.1, gamma="scale").fit(
            csc.transform(Fc.iloc[sp["fit"]][feats].to_numpy()))

        dets = {
            "FIS · centroid (PCA-free)": fis,
            "OneClassSVM · centroid": lambda ix: -oc.score_samples(
                csc.transform(Fc.iloc[ix][feats].to_numpy())),
            "Mahalanobis · stats": lambda ix: lw.mahalanobis(
                ssc.transform(Fs.iloc[ix].to_numpy())),
            "perplexity": lambda ix: meta.iloc[ix].perplexity.to_numpy(float),
            "mean entropy": lambda ix: meta.iloc[ix].ent_mean.to_numpy(float),
            "n_tokens (control)": lambda ix: meta.iloc[ix].n_tokens.to_numpy(float),
        }

        for cond, keys in CONDITIONS:
            if keys is None:
                a, b = sp["test_neg"], sp["test_pos"]
            else:
                a, b = match(meta, sp["test_neg"], sp["test_pos"], keys, rng)
            if len(a) < 30:
                continue
            ix = np.concatenate([a, b])
            y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
            for name, fn in dets.items():
                s = np.asarray(fn(ix), dtype=float)
                ok = np.isfinite(s).all() and np.ptp(s) > 0
                rows.append({"seed": seed, "condition": cond, "detector": name,
                             "auroc": roc_auc_score(y, s) if ok else np.nan,
                             "fpr@95tpr": fpr_at_tpr(y, s) if ok else np.nan,
                             "n_neg": len(a), "n_pos": len(b)})
        print(f"  seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "template_control.csv", index=False)

    print(f"\n{'='*92}\nAUROC by confound condition — mean ± std over "
          f"{args.seeds} seeds\n{'='*92}")
    piv = df.pivot_table(index="detector", columns="condition",
                         values="auroc", aggfunc=["mean", "std"])
    order = [c for c, _ in CONDITIONS if ("mean", c) in piv.columns]
    tbl = pd.DataFrame({c: piv[("mean", c)].map("{:.3f}".format) + " ± "
                        + piv[("std", c)].map("{:.3f}".format) for c in order})
    key = ("mean", order[-1])
    print(tbl.reindex(piv[key].sort_values(ascending=False).index).to_string())

    sizes = df.groupby("condition")[["n_neg", "n_pos"]].mean().round(0)
    print(f"\nmean test sizes:\n{sizes.to_string()}")

    strict = order[-1]
    m = df[df.condition == strict].pivot_table(index="seed", columns="detector",
                                               values="auroc")
    fis = next((c for c in m.columns if "centroid" in c), None)
    rivals = [c for c in m.columns if c != fis and "n_tokens" not in c]
    if fis and rivals:
        best = m[rivals].mean().idxmax()
        d = (m[fis] - m[best]).dropna()
        p = sstats.wilcoxon(d)[1] if len(d) >= 6 else np.nan
        print(f"\npaired advantage under '{strict}' vs {best}:")
        print(f"  mean Δ = {d.mean():+.3f} ± {d.std():.3f} "
              f"(min {d.min():+.3f}) · wins {int((d>0).sum())}/{len(d)}"
              + (f" · p = {p:.4f}" if np.isfinite(p) else ""))


if __name__ == "__main__":
    main()
