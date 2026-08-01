"""Stage 13 -- heuristic screen over the expanded variable families.

Ranks every one of the ~330 derived variables on the honest template-matched
task, and asks the only question that matters now: does a variable separate
fabrication **among generations that entropy already scores the same**? Mean
entropy reaches 0.964 here (§14), so raw separation is not interesting -- a
variable correlated 0.95 with entropy adds nothing.

  auroc           univariate separation under template+length matching
  auroc_entmatch  the same, with entropy ALSO matched (decile cells)
  rho_entropy     Spearman correlation with mean entropy -- redundancy

**A note on a method that was tried and rejected.** The first version of this
screen residualised each variable on (entropy, n_tokens) with a linear model fit
on the truthful split, then scored the residual. That is invalid here: the
nuisance model is fit only on truthful data, so on high-entropy fabrications it
extrapolates badly and the residual re-encodes entropy through its own prediction
error. The tell was unmissable -- residualising *raised* `N03_ratio` from 0.632
to 0.958. Matching (as used for length and template throughout this study)
conditions on the nuisance without extrapolating, so it is used instead.

The supervised check uses **nested** selection: variables are ranked on the
training fold only and evaluated on the held-out fold, because ranking on the
full test set and then cross-validating on it leaks the selection. With ~260
matched samples and 331 candidate variables, n < p, so even the nested number is
an optimistic ceiling rather than a deployable detector.
"""

import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze import DATA
from features_ext import build
from nopca import POOLING
from template_control import match, splits_v2

warnings.filterwarnings("ignore")

SEEDS = 4
NUISANCE = ["ent_mean", "n_tokens"]
# Coarse: truthful and fabricated entropy distributions barely overlap
# (entropy AUROC 0.968), so fine bins leave no matchable pairs at all.
N_ENT_BINS = 4


def main():
    meta = pd.read_parquet(DATA / "capture_v2_meta.parquet")
    H = np.load(DATA / f"capture_v2_hidden_{POOLING}.npy")
    recs = []

    for seed in range(SEEDS):
        sp = splits_v2(meta, seed)
        rng = np.random.default_rng(11000 + seed)
        fam = build(meta, H, sp["fit"])

        a, b = match(meta, sp["test_neg"], sp["test_pos"],
                     ["template", "n_tokens"], rng)
        if len(a) < 30:
            continue
        ix = np.concatenate([a, b])
        y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])

        ent_te = meta.iloc[ix]["ent_mean"].to_numpy(float)

        # Entropy-matched subset: bin entropy into deciles over the matched test
        # set, then keep min(#pos, #neg) per decile. Conditions on entropy
        # without extrapolating any model onto the positives.
        ent_all = meta["ent_mean"].astype(float)
        edges = np.quantile(ent_te, np.linspace(0, 1, N_ENT_BINS + 1))
        meta_bin = pd.Series(np.digitize(ent_all, edges[1:-1]), index=meta.index)
        meta2 = meta.copy()
        meta2["_entbin"] = meta_bin.values
        a2, b2 = match(meta2, a, b, ["template", "n_tokens", "_entbin"], rng)
        ix2 = np.concatenate([a2, b2])
        y2 = np.concatenate([np.zeros(len(a2)), np.ones(len(b2))])
        ok2 = len(a2) >= 12

        for fname, F in fam.items():
            for col in F.columns:
                v_te = F.iloc[ix][col].to_numpy(float)
                if np.ptp(v_te) == 0:
                    continue
                au = roc_auc_score(y, v_te)
                am = np.nan
                if ok2:
                    v2 = F.iloc[ix2][col].to_numpy(float)
                    if np.ptp(v2) > 0:
                        a_ = roc_auc_score(y2, v2)
                        am = max(a_, 1 - a_)
                recs.append({
                    "seed": seed, "family": fname, "feature": col,
                    # symmetric: a variable separating in either direction counts
                    "auroc": max(au, 1 - au),
                    "auroc_entmatch": am,
                    "rho_entropy": abs(sstats.spearmanr(v_te, ent_te).statistic),
                })
        print(f"  seed {seed}: screened {len(fam)} families; "
              f"entropy-matched n={len(a2)}+{len(b2)}")

    df = pd.DataFrame(recs)
    g = (df.groupby(["family", "feature"])
         .agg(auroc=("auroc", "mean"),
              auroc_entmatch=("auroc_entmatch", "mean"),
              rho_entropy=("rho_entropy", "mean"),
              entmatch_std=("auroc_entmatch", "std"))
         .reset_index())
    g.to_csv(DATA / "correlate_screen.csv", index=False)

    ent = g[g.feature == "ent_mean"]
    ent_au = float(ent.auroc.iloc[0]) if len(ent) else np.nan
    print(f"\n{'='*92}\nSCREEN OVER {len(g)} VARIABLES "
          f"({SEEDS} seeds, template+length matched)\n{'='*92}")
    print(f"reference: mean entropy univariate AUROC = {ent_au:.3f}\n")

    print("per-family best:")
    fam_best = (g.groupby("family")
                .agg(n=("feature", "count"),
                     best_auroc=("auroc", "max"),
                     best_entmatch=("auroc_entmatch", "max"),
                     median_rho=("rho_entropy", "median"))
                .sort_values("best_entmatch", ascending=False))
    print(fam_best.to_string(float_format=lambda v: f"{v:.3f}"))

    print("\ntop 20 by ENTROPY-MATCHED AUROC "
          "(separates among equal-entropy generations):")
    top = g.nlargest(20, "auroc_entmatch")
    print(top[["family", "feature", "auroc", "auroc_entmatch", "rho_entropy"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\nlow-redundancy candidates (rho_entropy < 0.3):")
    dec = g[g.rho_entropy < 0.3].nlargest(12, "auroc_entmatch")
    print(dec[["family", "feature", "auroc", "auroc_entmatch", "rho_entropy"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nseed-to-seed std of the entropy-matched column is "
          f"{g.entmatch_std.median():.3f} (median), so treat anything within "
          f"~2x that of 0.5 as noise.")

    # ---- supervised incremental check (upper bound, not a detector) -------
    print(f"\n{'='*92}\nINCREMENTAL VALUE OVER ENTROPY "
          f"(2-fold CV logistic regression — SUPERVISED upper bound)\n{'='*92}")
    sp = splits_v2(meta, 0)
    rng = np.random.default_rng(11000)
    fam = build(meta, H, sp["fit"])
    a, b = match(meta, sp["test_neg"], sp["test_pos"],
                 ["template", "n_tokens"], rng)
    ix = np.concatenate([a, b])
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    allF = pd.concat(fam.values(), axis=1)
    allF = allF.loc[:, ~allF.columns.duplicated()]

    def nested_auroc(k):
        """Rank on the training fold only, then evaluate on the held-out fold."""
        s = np.zeros(len(y))
        for tr, te in StratifiedKFold(2, shuffle=True, random_state=0).split(
                np.zeros(len(y)), y):
            if k == 0:
                cols = ["ent_mean"]
            else:
                sc = {}
                for c in allF.columns:
                    v = allF.iloc[ix[tr]][c].to_numpy(float)
                    if np.ptp(v) == 0:
                        continue
                    au = roc_auc_score(y[tr], v)
                    sc[c] = max(au, 1 - au)
                ranked = [c for c, _ in sorted(sc.items(), key=lambda t: -t[1])
                          if c != "ent_mean"][:k]
                cols = ["ent_mean"] + ranked
            sca = StandardScaler().fit(allF.iloc[ix[tr]][cols].to_numpy())
            m = LogisticRegression(max_iter=2000).fit(
                sca.transform(allF.iloc[ix[tr]][cols].to_numpy()), y[tr])
            s[te] = m.predict_proba(
                sca.transform(allF.iloc[ix[te]][cols].to_numpy()))[:, 1]
        return roc_auc_score(y, s)

    print(f"  entropy alone                             {nested_auroc(0):.3f}")
    for k in (3, 8, 20):
        print(f"  entropy + top-{k:<2} (selected in-fold)        {nested_auroc(k):.3f}")
    print(f"\n  n = {len(y)} matched samples vs {allF.shape[1]} candidate "
          f"variables (n < p), so treat these as a ceiling, not a detector.")


if __name__ == "__main__":
    main()
