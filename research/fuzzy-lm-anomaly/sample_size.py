"""Stage 26 -- was the fuzzy advantage a small-sample effect?

Section 24 measured FIS − Mahalanobis at +0.036 (SmolLM2) and +0.032 (Qwen), both
8/8 seeds, on the v3 probe set (471 real subjects, ~280 fit rows). On the 2.7x
larger v4 set the same comparison falls to +0.014 (p = 0.078) and +0.007
(p = 0.383) for those same two models.

There is an obvious candidate explanation. Mahalanobis estimates a full 19x19
covariance -- **209 parameters** -- so at ~280 fit rows it is badly under-determined
(n/p ≈ 1.3), while the fuzzy rule fits ~95. If the fuzzy advantage came from
Mahalanobis being *under-fit* rather than from the rule class being better, then
holding the data fixed and varying only the fit-set size should reproduce the
whole effect.

This sweeps the fit-set size on the v4 data, everything else identical, and reports
the paired advantage at each size. A curve that decays toward zero says the v3
result was a small-sample artefact; a flat curve says it was not.
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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                   take_top_features, tsk_firing_strengths)
import fis_config as CFG
from analyze import DATA, SCALAR_COLS
from template_control import match

warnings.filterwarnings("ignore")

SEEDS, TOP_N, K = 8, 6, 4
SIZES = [120, 200, 280, 400, 550, 760]     # 280 ≈ the v3 fit size


def run(meta, F, seed, n_fit, rng):
    good = np.flatnonzero((meta.family == "longform_real")
                          & (meta.label == "grounded"))
    bad = np.flatnonzero((meta.family == "longform_fake")
                         & (meta.label == "hallucination"))
    g = good.copy()
    rng.shuffle(g)
    # test negatives are held FIXED across sizes so only the fit set varies
    test_neg, pool = g[:400], g[400:]
    fit = pool[:n_fit]
    if len(fit) < n_fit:
        return None

    sub = F.iloc[fit]
    feats0 = list(sub.var().sort_values(ascending=False).index[:TOP_N])
    Xf = StandardScaler().fit_transform(sub[feats0].to_numpy())
    labels = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Xf).labels_
    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        diff = calculate_gaussian_correlation(sub.reset_index(drop=True), y_modes,
                                              method=CFG.METRIC)
        _, feats = take_top_features(diff, top_n=TOP_N)
        model = CFG.build_memberships(sub[feats].reset_index(drop=True), y_modes,
                                      feats, membership=CFG.MEMBERSHIP)

    ssc = StandardScaler().fit(sub.to_numpy())
    lw = LedoitWolf().fit(ssc.transform(sub.to_numpy()))

    a, b = match(meta, test_neg, bad, ["template", "n_tokens"], rng)
    if len(a) < 30:
        return None
    ix = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])

    ap = CFG.anomaly_params()
    fs, lab = tsk_firing_strengths(F.iloc[ix][feats].reset_index(drop=True),
                                   model, ap)
    s_fis = np.asarray(fs[:, lab.index("anomaly")], float)
    s_mah = lw.mahalanobis(ssc.transform(F.iloc[ix].to_numpy()))
    return (roc_auc_score(y, s_fis), roc_auc_score(y, s_mah),
            roc_auc_score(y, meta.iloc[ix].ent_mean.to_numpy(float)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["smollm2", "qwen", "gemma", "lfm"])
    args = ap.parse_args()

    rows = []
    for m in args.models:
        meta = pd.read_parquet(DATA / f"capture_v4_{m}_meta.parquet")
        F = meta[SCALAR_COLS].reset_index(drop=True).astype(float)
        for n_fit in SIZES:
            for seed in range(SEEDS):
                r = run(meta, F, seed, n_fit, np.random.default_rng(31000 + seed))
                if r is None:
                    continue
                rows.append({"model": m, "n_fit": n_fit, "seed": seed,
                             "fis": r[0], "mahalanobis": r[1], "entropy": r[2],
                             "d_mahal": r[0] - r[1]})
        print(f"  {m} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "sample_size.csv", index=False)

    print(f"\n{'='*88}\nPAIRED FIS − MAHALANOBIS vs FIT-SET SIZE "
          f"(v4 data, fixed test set, {SEEDS} seeds)\n{'='*88}")
    piv = df.pivot_table(index="n_fit", columns="model", values="d_mahal",
                         aggfunc="mean")
    print(piv.to_string(float_format=lambda v: f"{v:+.3f}"))
    print("\n  n/p for Mahalanobis (209 parameters):")
    print("   " + "  ".join(f"{n}:{n/209:.1f}x" for n in SIZES))

    print(f"\n{'='*88}\nSIGNIFICANCE AT EACH SIZE (Wilcoxon over seeds)\n{'='*88}")
    for m in args.models:
        out = []
        for n_fit in SIZES:
            d = df[(df.model == m) & (df.n_fit == n_fit)].d_mahal.dropna()
            if len(d) < 6:
                continue
            p = sstats.wilcoxon(d)[1]
            out.append(f"n={n_fit}: {d.mean():+.3f}{'*' if p < 0.05 else ' '}")
        print(f"  {m:9s} " + "  ".join(out))
    print("\n  * p < 0.05")

    # absolute AUROCs, to show whether both detectors improve or only one
    print(f"\n{'='*88}\nABSOLUTE AUROC vs FIT SIZE (mean over models and seeds)\n{'='*88}")
    g = df.groupby("n_fit")[["fis", "mahalanobis", "entropy"]].mean()
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
