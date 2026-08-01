"""Stage 22 -- factorial comparison of coefficient metric, membership function,
and norm pair, against tribble-fis main.

The library now exposes natively what earlier stages had to reimplement:

  * `calculate_gaussian_correlation(X, y, method=...)`  -- issue #30, PR #34.
    The hard-coded blend is gone; "bhattacharyya" (default) and "wasserstein".
  * `resolve_norm_pair` / `AnomalyParameters(t_norm=, t_conorm=,
    allow_mixed_norms=)` and an "einstein" family -- PR #32. Mixed-family pairs
    are now *gated*: they are not De Morgan duals, and the anomaly rule's
    complement construction 1 - S(...) depends on that duality for its meaning,
    so asking for one has to be explicit.
  * `create_trapz_membership_dict` -- a trapezoid analogue of the Gaussian
    builder, returning the same container, so the anomaly rule accepts it.

This runs the full factorial on the honest task (v3 long-form, template matched)
over the 19 output statistics -- the representation §21 established as the one
that survives the controls.

  metric              bhattacharyya | wasserstein
  membership function gaussian | trapezoid
  norm pair           5 De Morgan families + 3 explicitly-mixed pairs

Everything else is held at the modal §21 selection (6 antecedents, K=4 modes, no
whitening) so the factorial isolates these three factors. All numbers carry the
standing cost columns.
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

from tribblefis.gauss_data import AnomalyParameters
from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                   take_top_features, tsk_firing_strengths)
import fis_config as CFG

from analyze import DATA, SCALAR_COLS, fpr_at_tpr
from seed_sweep import Timer
from template_control import match

warnings.filterwarnings("ignore")

SEEDS, TOP_N, K = 8, 6, 4
METRICS = ["bhattacharyya", "wasserstein"]
# Built via fis_config.build_memberships, which verifies the family actually
# produced -- section 22 measured this factor at +/-0.262 AUROC.
MFS = {"gaussian": "gaussian", "trapezoid": "trap"}
DE_MORGAN = ["min/max", "probability", "luk", "hamacher", "einstein"]
MIXED = [("hamacher", "einstein"), ("probability", "einstein"),
         ("min/max", "hamacher")]
THETA = 0.5


def splits(meta, seed):
    rng = np.random.default_rng(seed)
    good = np.flatnonzero((meta.family == "longform_real")
                          & (meta.label == "grounded"))
    bad = np.flatnonzero((meta.family == "longform_fake")
                         & (meta.label == "hallucination"))
    rng.shuffle(good)
    rng.shuffle(bad)
    g = int(.55 * len(good))
    return {"fit": good[:g], "test_neg": good[g:], "test_pos": bad}


def build_model(F, fit, metric, mf_name, seed):
    """Antecedents ranked by `metric`; membership functions built by `mf_name`."""
    feats0 = list(F.iloc[fit].var().sort_values(ascending=False).index[:TOP_N])
    Xf = StandardScaler().fit_transform(F.iloc[fit][feats0].to_numpy())
    labels = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Xf).labels_
    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        diff = calculate_gaussian_correlation(
            F.iloc[fit].reset_index(drop=True), y_modes, method=metric)
        _, feats = take_top_features(diff, top_n=TOP_N)
        model = CFG.build_memberships(
            F.iloc[fit][feats].reset_index(drop=True), y_modes, feats,
            membership=MFS[mf_name])
    return feats, model


def score(F, feats, model, ix, tn, sn):
    """Anomaly column via the library, with the norm pair resolved explicitly."""
    ap = (CFG.anomaly_params(theta=THETA, norm=tn) if tn == sn
          else CFG.anomaly_params(theta=THETA, t_norm=tn, t_conorm=sn))
    fs, lab = tsk_firing_strengths(F.iloc[ix][feats].reset_index(drop=True),
                                   model, ap)
    return np.asarray(fs[:, lab.index("anomaly")], float)


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--seeds", type=int, default=SEEDS)
    args = ap_.parse_args()

    meta = pd.read_parquet(DATA / "capture_v3_meta.parquet")
    ent = meta["ent_mean"].astype(float)
    F = meta[SCALAR_COLS].reset_index(drop=True).astype(float)
    pairs = [(n, n) for n in DE_MORGAN] + MIXED
    rows = []

    for seed in range(args.seeds):
        sp = splits(meta, seed)
        rng = np.random.default_rng(23000 + seed)
        m2 = meta.copy()
        m2["_eb"] = np.digitize(ent, np.quantile(
            ent.iloc[np.concatenate([sp["test_neg"], sp["test_pos"]])],
            [.25, .5, .75]))
        conds = {"length+template": ["template", "n_tokens"],
                 "length+template+entropy": ["template", "n_tokens", "_eb"]}

        built = {}
        for metric in METRICS:
            for mf in MFS:
                with Timer() as t:
                    feats, model = build_model(F, sp["fit"], metric, mf, seed)
                built[(metric, mf)] = (feats, model, t.ms)

        # Mahalanobis reference on the same 19 statistics
        d = F.shape[1]
        with Timer() as t_m:
            ssc = StandardScaler().fit(F.iloc[sp["fit"]].to_numpy())
            lw = LedoitWolf().fit(ssc.transform(F.iloc[sp["fit"]].to_numpy()))

        for cond, keys in conds.items():
            a, b = match(m2, sp["test_neg"], sp["test_pos"], keys, rng)
            if len(a) < 20:
                continue
            ix = np.concatenate([a, b])
            y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])

            for (metric, mf), (feats, model, tms) in built.items():
                for tn, sn in pairs:
                    with Timer() as t:
                        s = score(F, feats, model, ix, tn, sn)
                    ok = np.isfinite(s).all() and np.ptp(s) > 0
                    rows.append({
                        "seed": seed, "condition": cond, "metric": metric,
                        "mf": mf, "tnorm": tn, "sconorm": sn,
                        "de_morgan": tn == sn,
                        "auroc": roc_auc_score(y, s) if ok else np.nan,
                        "fpr95": fpr_at_tpr(y, s) if ok else np.nan,
                        "n_params": 2 * model.n_membership_functions + 1,
                        "n_mfs": model.n_membership_functions,
                        "train_ms": tms,
                        "samples_per_sec": len(ix) / max(t.ms / 1000, 1e-9)})
            v = lw.mahalanobis(ssc.transform(F.iloc[ix].to_numpy()))
            rows.append({"seed": seed, "condition": cond, "metric": "-",
                         "mf": "Mahalanobis", "tnorm": "-", "sconorm": "-",
                         "de_morgan": True, "auroc": roc_auc_score(y, v),
                         "fpr95": fpr_at_tpr(y, v),
                         "n_params": d + d * (d + 1) // 2, "n_mfs": 0,
                         "train_ms": t_m.ms, "samples_per_sec": np.nan})
        print(f"  seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "compare_variants.csv", index=False)

    fis = df[df.mf != "Mahalanobis"]
    mah = df[df.mf == "Mahalanobis"]

    for cond in ("length+template", "length+template+entropy"):
        s = fis[fis.condition == cond]
        mref = mah[mah.condition == cond].auroc.mean()
        print(f"\n{'='*100}\nCONDITION '{cond}'  "
              f"(Mahalanobis reference {mref:.3f}, {args.seeds} seeds)\n{'='*100}")

        print("\n-- coefficient metric x membership function (mean over norm pairs) --")
        p1 = s.pivot_table(index="mf", columns="metric", values="auroc",
                           aggfunc=["mean", "std"])
        for m in METRICS:
            if ("mean", m) not in p1.columns:
                continue
        print(pd.DataFrame({m: p1[("mean", m)].map("{:.3f}".format) + " ± "
                            + p1[("std", m)].map("{:.3f}".format)
                            for m in METRICS if ("mean", m) in p1.columns}
                           ).to_string())

        print("\n-- norm pair (mean over metric x MF; De Morgan pairs first) --")
        p2 = (s.assign(pair=s.tnorm + " / " + s.sconorm)
              .groupby(["de_morgan", "pair"])["auroc"]
              .agg(["mean", "std", "count"])
              .sort_values(["de_morgan", "mean"], ascending=[False, False]))
        print(p2.to_string(float_format=lambda v: f"{v:.3f}"))

        print("\n-- best 8 full configurations --")
        g = (s.groupby(["metric", "mf", "tnorm", "sconorm", "de_morgan"])
             .agg(auroc=("auroc", "mean"), std=("auroc", "std"),
                  fpr95=("fpr95", "mean"), params=("n_params", "mean"),
                  mfs=("n_mfs", "mean"), train_ms=("train_ms", "mean"))
             .reset_index().sort_values("auroc", ascending=False))
        print(g.head(8).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"\n  configurations beating Mahalanobis ({mref:.3f}): "
              f"{int((g.auroc > mref).sum())} of {len(g)}")

    # main-effect summary across both conditions
    print(f"\n{'='*100}\nMAIN EFFECTS (condition 'length+template')\n{'='*100}")
    s = fis[fis.condition == "length+template"]
    for factor in ("metric", "mf", "de_morgan"):
        g = s.groupby(factor)["auroc"].agg(["mean", "std", "count"])
        print(f"\n{factor}:")
        print(g.to_string(float_format=lambda v: f"{v:.3f}"))
        lv = list(g.index)
        if len(lv) == 2:
            x = s[s[factor] == lv[0]].groupby("seed")["auroc"].mean()
            z = s[s[factor] == lv[1]].groupby("seed")["auroc"].mean()
            dd = (x - z).dropna()
            if len(dd) >= 6:
                p = sstats.wilcoxon(dd)[1]
                print(f"  paired {lv[0]} - {lv[1]}: {dd.mean():+.3f} ± "
                      f"{dd.std():.3f}, wins {int((dd>0).sum())}/{len(dd)}, "
                      f"p = {p:.4f}")


if __name__ == "__main__":
    main()
