"""Stage 25 -- a second-pass specialist with abstention, for the FPR@95 problem.

The defect. Under entropy matching the single rule base wins on ranking
(0.789 vs 0.760 for Mahalanobis) but its **FPR@95TPR is 0.961** -- to catch 95%
of fabrications it flags essentially everything. This is structural, not a
tuning miss: section 3.4 proved theta is rank-invariant, so no threshold search
can move a point on the ROC curve that the ranking does not already offer. The
tail of the ranking is simply bad, and only a different *ranking* fixes it.

The fuzzy answer, from Ch 4.3.1: a **cascade of specialists with abstention**.
The first rule base handles what it is confident about and abstains in the band
where its own known-good modes overlap the anomaly rule; a second rule base,
fitted only on the known-good examples that fall in that band, re-ranks it.

    pass 1   mu_anom from the full known-good fit
             confident-clean  -> score as is
             confident-anomalous -> score as is
             UNCERTAIN BAND   -> defer
    pass 2   a rule base fitted on the known-good rows inside the band only,
             so its modes describe the hard region rather than the bulk;
             its mu_anom re-ranks the deferred rows *within* the band

Ranks are composed, not averaged: deferred rows keep their band position and are
reordered inside it. That cannot hurt the head of the ranking (where pass 1 is
already confident) and can only improve the tail -- which is exactly where
FPR@95 lives.

Everything stays open-set: pass 2 is fitted on known-good rows only, selected by
pass 1's score, never using a label.
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
from analyze import DATA, SCALAR_COLS, fpr_at_tpr
from seed_sweep import Timer
from template_control import match

warnings.filterwarnings("ignore")

SEEDS, TOP_N, K = 8, 6, 4
TASKS = {
    "capture_v3": ("longform_real", "grounded", "longform_fake"),
    "capture_v3_qwen": ("longform_real", "grounded", "longform_fake"),
    "capture_v4_smollm2": ("longform_real", "grounded", "longform_fake"),
    "capture_v4_qwen": ("longform_real", "grounded", "longform_fake"),
    "capture_v4_gemma": ("longform_real", "grounded", "longform_fake"),
    "capture_v4_lfm": ("longform_real", "grounded", "longform_fake"),
    "capture_v2": ("template_real", "correct", "template_fake"),
}


def splits(meta, seed, task):
    neg_fam, neg_lab, pos_fam = TASKS[task]
    rng = np.random.default_rng(seed)
    good = np.flatnonzero((meta.family == neg_fam) & (meta.label == neg_lab))
    bad = np.flatnonzero((meta.family == pos_fam) & (meta.label == "hallucination"))
    rng.shuffle(good)
    rng.shuffle(bad)
    g = int(.6 * len(good))
    return {"fit": good[:g], "test_neg": good[g:], "test_pos": bad}


def fit_pass(F, fit_idx, seed, top_n=TOP_N, k=K):
    """One rule base, fitted on the given known-good rows only."""
    if len(fit_idx) < 40:
        return None, None
    sub = F.iloc[fit_idx]
    feats0 = list(sub.var().sort_values(ascending=False).index[:top_n])
    Xf = StandardScaler().fit_transform(sub[feats0].to_numpy())
    kk = min(k, max(2, len(fit_idx) // 20))
    labels = KMeans(n_clusters=kk, n_init=10, random_state=seed).fit(Xf).labels_
    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        diff = calculate_gaussian_correlation(sub.reset_index(drop=True), y_modes,
                                              method=CFG.METRIC)
        _, feats = take_top_features(diff, top_n=top_n)
        model = CFG.build_memberships(sub[feats].reset_index(drop=True), y_modes,
                                      feats, membership=CFG.MEMBERSHIP)
    return feats, model


def anomaly_of(F, feats, model, idx):
    ap = CFG.anomaly_params()
    fs, lab = tsk_firing_strengths(F.iloc[idx][feats].reset_index(drop=True),
                                   model, ap)
    return np.asarray(fs[:, lab.index("anomaly")], float)


def cascade_scores(F, sp, seed, band=(0.35, 0.85)):
    """Pass-1 score, and the cascade score that re-ranks the uncertain band.

    `band` is expressed in quantiles of pass 1's score over the KNOWN-GOOD fit
    rows -- so the band is defined without ever consulting a label.
    """
    feats1, m1 = fit_pass(F, sp["fit"], seed)
    if m1 is None:
        return None
    s_fit = anomaly_of(F, feats1, m1, sp["fit"])
    lo, hi = np.quantile(s_fit, band)

    # pass 2 fits ONLY on known-good rows sitting inside the uncertain band
    inner = sp["fit"][(s_fit >= lo) & (s_fit <= hi)]
    feats2, m2 = fit_pass(F, inner, seed, top_n=TOP_N, k=2)

    def score(idx):
        s1 = anomaly_of(F, feats1, m1, idx)
        if m2 is None:
            return s1, s1
        s2 = anomaly_of(F, feats2, m2, idx)
        # compose ranks: keep the band's position, reorder within it
        out = s1.copy()
        inb = (s1 >= lo) & (s1 <= hi)
        if inb.sum() > 1:
            r = sstats.rankdata(s2[inb]) / (inb.sum() + 1.0)
            out[inb] = lo + (hi - lo) * r
        return s1, out

    return score, (lo, hi), (m1, m2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--prefix", default="capture_v3", choices=list(TASKS))
    args = ap.parse_args()

    meta = pd.read_parquet(DATA / f"{args.prefix}_meta.parquet")
    ent = meta["ent_mean"].astype(float)
    F = meta[SCALAR_COLS].reset_index(drop=True).astype(float)
    rows = []

    for seed in range(args.seeds):
        sp = splits(meta, seed, args.prefix)
        rng = np.random.default_rng(29000 + seed)
        m2m = meta.copy()
        m2m["_eb"] = np.digitize(ent, np.quantile(
            ent.iloc[np.concatenate([sp["test_neg"], sp["test_pos"]])],
            [.25, .5, .75]))
        conds = {"length+template": ["template", "n_tokens"],
                 "length+template+entropy": ["template", "n_tokens", "_eb"]}

        with Timer() as t:
            built = cascade_scores(F, sp, seed)
        if built is None:
            continue
        score, (lo, hi), (m1, mm2) = built
        n_par1 = 2 * m1.n_membership_functions + 1
        n_par2 = (2 * mm2.n_membership_functions + 1) if mm2 is not None else 0

        ssc = StandardScaler().fit(F.iloc[sp["fit"]].to_numpy())
        lw = LedoitWolf().fit(ssc.transform(F.iloc[sp["fit"]].to_numpy()))

        for cond, keys in conds.items():
            a, b = match(m2m, sp["test_neg"], sp["test_pos"], keys, rng)
            if len(a) < 20:
                continue
            ix = np.concatenate([a, b])
            y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
            s1, sc = score(ix)
            for nm, s, npar in (("FIS single pass", s1, n_par1),
                                ("FIS cascade (2 pass)", sc, n_par1 + n_par2),
                                ("Mahalanobis · stats",
                                 lw.mahalanobis(ssc.transform(F.iloc[ix].to_numpy())),
                                 209),
                                ("mean entropy",
                                 meta.iloc[ix].ent_mean.to_numpy(float), 0)):
                rows.append({"seed": seed, "condition": cond, "detector": nm,
                             "auroc": roc_auc_score(y, s),
                             "fpr95": fpr_at_tpr(y, s),
                             "fpr90": fpr_at_tpr(y, s, 0.90),
                             "fpr80": fpr_at_tpr(y, s, 0.80),
                             "n_params": npar, "train_ms": t.ms,
                             "band_lo": lo, "band_hi": hi})
        print(f"  seed {seed}: band [{lo:.3f}, {hi:.3f}], "
              f"pass2 {'fitted' if mm2 is not None else 'skipped'}")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / f"cascade_{args.prefix}.csv", index=False)

    for cond in ("length+template", "length+template+entropy"):
        s = df[df.condition == cond]
        if not len(s):
            continue
        g = (s.groupby("detector")
             .agg(auroc=("auroc", "mean"), auroc_std=("auroc", "std"),
                  fpr95=("fpr95", "mean"), fpr90=("fpr90", "mean"),
                  fpr80=("fpr80", "mean"), n_params=("n_params", "mean"))
             .sort_values("auroc", ascending=False))
        print(f"\n{'='*94}\n{args.prefix} — condition '{cond}' "
              f"({args.seeds} seeds)\n{'='*94}")
        out = g.copy()
        out["AUROC"] = (out.auroc.map("{:.3f}".format) + " ± "
                        + out.auroc_std.map("{:.3f}".format))
        print(out[["AUROC", "fpr95", "fpr90", "fpr80", "n_params"]]
              .to_string(float_format=lambda v: f"{v:.3f}"))

        m = s.pivot_table(index="seed", columns="detector",
                          values=["auroc", "fpr95"])
        if ("auroc", "FIS cascade (2 pass)") in m.columns:
            for metric, better in (("auroc", "higher"), ("fpr95", "lower")):
                d = (m[(metric, "FIS cascade (2 pass)")]
                     - m[(metric, "FIS single pass")]).dropna()
                p = sstats.wilcoxon(d)[1] if len(d) >= 6 and d.std() > 0 else np.nan
                win = int((d < 0).sum()) if better == "lower" else int((d > 0).sum())
                print(f"  cascade − single pass, {metric} ({better} is better): "
                      f"{d.mean():+.3f} ± {d.std():.3f}, improves {win}/{len(d)}"
                      + (f", p = {p:.4f}" if np.isfinite(p) else ""))


if __name__ == "__main__":
    main()
