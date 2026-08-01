"""Stage 28 -- can the detector be switched without labels?

§27: the fuzzy rule beats mean entropy exactly where entropy is weak (crossover at
entropy AUROC ≈ 0.61), but a zero-parameter blend of the two gains nothing — they
are complementary *across* regimes, not *within* a cell. So the value, if any, is
in **choosing** the detector per deployment.

Three rungs of increasing realism:

  1. ORACLE      switch on the true per-cell entropy AUROC. Cheating; it exists
                 only to bound what switching could ever be worth.
  2. CALIBRATED  estimate entropy's AUROC from k labelled examples and switch on
                 the estimate. The realistic deployment story; the honest failure
                 mode is that the estimate is too noisy at small k.
  3. LABEL-FREE  predict entropy's AUROC from statistics of the KNOWN-GOOD split
                 alone (variance, range, skew, bimodality of entropy over grounded
                 generations). Keeps the open-set protocol intact. The predictor is
                 fitted on some models and tested on HELD-OUT models, so it is
                 validated out of sample rather than in it.

Every arm uses fixed detector configurations (§26): no search budget for anyone.
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
from sklearn.linear_model import LinearRegression
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

SEEDS, TOP_N, K, MIN_CELL = 6, 6, 4, 15
CROSSOVER = 0.608          # from §27's linear fit
KS = [20, 50, 100, 200]


def fit_fis(F, fit, seed):
    sub = F.iloc[fit]
    f0 = list(sub.var().sort_values(ascending=False).index[:TOP_N])
    Xf = StandardScaler().fit_transform(sub[f0].to_numpy())
    lab = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Xf).labels_
    ym = pd.Series([f"mode{c}" for c in lab])
    with contextlib.redirect_stdout(io.StringIO()):
        diff = calculate_gaussian_correlation(sub.reset_index(drop=True), ym,
                                              method=CFG.METRIC)
        _, feats = take_top_features(diff, top_n=TOP_N)
        model = CFG.build_memberships(sub[feats].reset_index(drop=True), ym,
                                      feats, membership=CFG.MEMBERSHIP)
    return feats, model


def known_good_entropy_stats(ent_fit):
    """Label-free descriptors of entropy on the known-good split only."""
    q = np.percentile(ent_fit, [10, 25, 50, 75, 90])
    return {
        "kg_std": float(np.std(ent_fit)),
        "kg_iqr": float(q[3] - q[1]),
        "kg_range": float(q[4] - q[0]),
        "kg_skew": float(sstats.skew(ent_fit)),
        "kg_kurt": float(sstats.kurtosis(ent_fit)),
        "kg_mean": float(np.mean(ent_fit)),
        "kg_cv": float(np.std(ent_fit) / (np.mean(ent_fit) + 1e-9)),
        # bimodality coefficient: high => the known-good set has structure
        "kg_bimod": float((sstats.skew(ent_fit) ** 2 + 1)
                          / (sstats.kurtosis(ent_fit) + 3 + 1e-9)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["smollm2", "qwen", "gemma", "lfm"])
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    rows = []

    for m in args.models:
        f = DATA / f"capture_v4_{m}_meta.parquet"
        if not f.exists():
            print(f"  skip {m} (no capture)")
            continue
        meta = pd.read_parquet(f)
        F = meta[SCALAR_COLS].reset_index(drop=True).astype(float)
        good = np.flatnonzero((meta.family == "longform_real")
                              & (meta.label == "grounded"))
        bad = np.flatnonzero((meta.family == "longform_fake")
                             & (meta.label == "hallucination"))

        for seed in range(args.seeds):
            rng = np.random.default_rng(41000 + seed)
            g = good.copy()
            rng.shuffle(g)
            cut = int(.6 * len(g))
            fit, test_neg = g[:cut], g[cut:]
            feats, model = fit_fis(F, fit, seed)

            for tmpl in sorted(meta.template.unique()):
                neg = test_neg[meta.iloc[test_neg].template.to_numpy() == tmpl]
                pos = bad[meta.iloc[bad].template.to_numpy() == tmpl]
                if len(neg) < MIN_CELL or len(pos) < MIN_CELL:
                    continue
                a, b = match(meta, neg, pos, ["n_tokens"], rng)
                if len(a) < MIN_CELL:
                    continue
                ix = np.concatenate([a, b])
                y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
                s_ent = meta.iloc[ix].ent_mean.to_numpy(float)
                fs, lb = tsk_firing_strengths(
                    F.iloc[ix][feats].reset_index(drop=True), model,
                    CFG.anomaly_params())
                s_fis = np.asarray(fs[:, lb.index("anomaly")], float)
                if not np.isfinite(s_fis).all() or np.ptp(s_fis) == 0:
                    continue

                au_e, au_f = roc_auc_score(y, s_ent), roc_auc_score(y, s_fis)

                # rung 2: AUROC estimated from k labelled examples
                est = {}
                for k in KS:
                    kk = min(k // 2, len(a), len(b))
                    if kk < 5:
                        est[k] = np.nan
                        continue
                    sa = rng.choice(len(a), kk, replace=False)
                    sb = rng.choice(len(b), kk, replace=False) + len(a)
                    sub = np.concatenate([sa, sb])
                    est[k] = roc_auc_score(y[sub], s_ent[sub])

                # entropy on the known-good FIT split -- label-free descriptors
                kg = known_good_entropy_stats(
                    meta.iloc[fit].ent_mean.to_numpy(float))

                rows.append({"model": m, "template": tmpl, "seed": seed,
                             "n": len(ix), "auroc_entropy": au_e,
                             "auroc_fis": au_f,
                             **{f"est_k{k}": v for k, v in est.items()}, **kg})
        print(f"  {m} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "switching.csv", index=False)
    df["best"] = df[["auroc_entropy", "auroc_fis"]].max(axis=1)

    print(f"\n{'='*90}\nSWITCHING — {len(df)} cell-seeds, "
          f"{df.groupby(['model','template']).ngroups} cells\n{'='*90}")
    base = df.auroc_entropy.mean()
    print(f"  always entropy            {base:.4f}   (the incumbent)")
    print(f"  always FIS                {df.auroc_fis.mean():.4f}")
    print(f"  ORACLE per-cell best      {df.best.mean():.4f}   "
          f"(+{df.best.mean()-base:.4f})  <- upper bound, cheating")

    # rung 1: switch on TRUE entropy AUROC
    pick = np.where(df.auroc_entropy < CROSSOVER, df.auroc_fis, df.auroc_entropy)
    print(f"  oracle threshold rule     {pick.mean():.4f}   "
          f"(+{pick.mean()-base:.4f})  <- still uses the true AUROC")

    # rung 2: switch on a k-labelled estimate
    print(f"\n  rung 2 — switch on an AUROC estimated from k labelled examples:")
    for k in KS:
        c = f"est_k{k}"
        ok = df[c].notna()
        p = np.where(df.loc[ok, c] < CROSSOVER,
                     df.loc[ok, "auroc_fis"], df.loc[ok, "auroc_entropy"])
        agree = ((df.loc[ok, c] < CROSSOVER)
                 == (df.loc[ok, "auroc_entropy"] < CROSSOVER)).mean()
        print(f"    k={k:>3}  net {p.mean():.4f} ({p.mean()-base:+.4f})   "
              f"switch agrees with oracle {agree:.1%}")

    # rung 3: predict entropy AUROC from known-good statistics, held-out models
    print(f"\n  rung 3 — LABEL-FREE: predict entropy AUROC from the known-good "
          f"split, tested on held-out models:")
    kgc = [c for c in df.columns if c.startswith("kg_")]
    cell = df.groupby(["model", "template"]).agg(
        {**{c: "mean" for c in kgc},
         "auroc_entropy": "mean", "auroc_fis": "mean"}).reset_index()
    preds, truths, gains = [], [], []
    for held in cell.model.unique():
        tr, te = cell[cell.model != held], cell[cell.model == held]
        if len(tr) < 8 or len(te) < 2:
            continue
        lr = LinearRegression().fit(tr[kgc].to_numpy(), tr.auroc_entropy)
        ph = lr.predict(te[kgc].to_numpy())
        preds += list(ph)
        truths += list(te.auroc_entropy)
        g = np.where(ph < CROSSOVER, te.auroc_fis, te.auroc_entropy)
        gains += list(g - te.auroc_entropy)
    if len(preds) > 3:
        r, pv = sstats.pearsonr(truths, preds)
        print(f"    predicted vs true entropy AUROC: r = {r:+.3f} (p = {pv:.3f}), "
              f"n = {len(preds)} held-out cells")
        print(f"    net gain from switching on the prediction: "
              f"{np.mean(gains):+.4f}")
        print(f"    (oracle threshold rule gains "
              f"{pick.mean()-base:+.4f} for comparison)")

    print(f"\n{'='*90}\nPER MODEL\n{'='*90}")
    pm = df.groupby("model").agg(entropy=("auroc_entropy", "mean"),
                                 fis=("auroc_fis", "mean"),
                                 oracle=("best", "mean")).sort_values("entropy")
    pm["oracle_gain"] = pm.oracle - pm.entropy
    print(pm.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
