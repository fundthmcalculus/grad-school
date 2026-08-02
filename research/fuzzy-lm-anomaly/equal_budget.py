"""Stage 30 -- the equal-budget comparison.

§26 showed the fuzzy rule's apparent win came from a 120-candidate configuration
search scored against labelled validation positives, which its rivals never
received. §27-§29 then worked at *zero* budget for everyone. Neither is the
comparison a paper needs. This is: **every detector family gets a search of
comparable size, on the same validation positives, with the same criterion, and
is reported on the same disjoint test half.**

The budgets are deliberately matched at ~120 candidates:

  FIS               whiten(2) x antecedents(5) x modes(3) x norm pair(4)  = 120
  Mahalanobis       estimator(4) x shrinkage(6) x feature subset(5)       = 120
  IsolationForest   trees(4) x max_features(3) x contamination(4)
                    x feature subset(2)                                   =  96
  OneClassSVM       nu(5) x gamma(5) x kernel(2) x feature subset(2)      = 100
  single statistic  19 statistics x 2 directions                          =  38

The last one matters and is easy to miss. **"Mean entropy" is one arbitrary choice
out of nineteen output statistics.** A threshold detector allowed to *select* its
statistic on validation is the like-for-like rival to a rule base allowed to select
its configuration — comparing a searched FIS against a fixed entropy threshold is
the same error as §26, pointed the other way. Its budget is smaller because the
family genuinely is; that is reported rather than padded.

Also reported: the same table at zero budget, so the effect of the search itself is
visible per family rather than assumed.
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
from sklearn.covariance import (EmpiricalCovariance, LedoitWolf, MinCovDet,
                                OAS, ShrunkCovariance)
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                   take_top_features, tsk_firing_strengths)
import fis_config as CFG
from analyze import DATA, SCALAR_COLS, fpr_at_tpr
from template_control import match

warnings.filterwarnings("ignore")

SEEDS = 6
MODELS = ["smollm2", "qwen", "gemma", "lfm", "sl135", "sl1700"]
NORMS = [("min/max", "min/max"), ("probability", "probability"),
         ("hamacher", "hamacher"), ("einstein", "einstein")]


def splits(meta, seed):
    rng = np.random.default_rng(seed)
    good = np.flatnonzero((meta.family == "longform_real")
                          & (meta.label == "grounded"))
    bad = np.flatnonzero((meta.family == "longform_fake")
                         & (meta.label == "hallucination"))
    rng.shuffle(good)
    rng.shuffle(bad)
    g = int(.55 * len(good))
    return {"fit": good[:g], "test_neg": good[g:],
            "val_pos": bad[:len(bad) // 2], "test_pos": bad[len(bad) // 2:]}


def topk(F, fit, k):
    return list(F.iloc[fit].var().sort_values(ascending=False).index[:k])


# --------------------------------------------------------------------------
# candidate generators: each yields (name, scorer) built on the FIT split only
# --------------------------------------------------------------------------

def cands_fis(F, fit, seed):
    out = []
    for whiten in (False, True):
        if whiten:
            sc = StandardScaler().fit(F.iloc[fit].to_numpy())
            from sklearn.decomposition import PCA
            p = PCA(whiten=True, random_state=0).fit(sc.transform(F.iloc[fit].to_numpy()))
            Z = p.transform(sc.transform(F.to_numpy()))
            Fx = pd.DataFrame(Z, columns=[f"W{i:02d}" for i in range(Z.shape[1])])
        else:
            Fx = F
        for tn in (4, 6, 8, 12, 19):
            for k in (2, 3, 4):
                feats0 = topk(Fx, fit, tn)
                Xf = StandardScaler().fit_transform(Fx.iloc[fit][feats0].to_numpy())
                lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xf).labels_
                ym = pd.Series([f"mode{c}" for c in lab])
                with contextlib.redirect_stdout(io.StringIO()):
                    diff = calculate_gaussian_correlation(
                        Fx.iloc[fit].reset_index(drop=True), ym, method=CFG.METRIC)
                    _, feats = take_top_features(diff, top_n=tn)
                    model = CFG.build_memberships(
                        Fx.iloc[fit][feats].reset_index(drop=True), ym, feats,
                        membership=CFG.MEMBERSHIP)
                for tnm, snm in NORMS:
                    def mk(Fx=Fx, feats=feats, model=model, tnm=tnm, snm=snm):
                        def f(ix):
                            fs, lb = tsk_firing_strengths(
                                Fx.iloc[ix][feats].reset_index(drop=True), model,
                                CFG.anomaly_params(norm=tnm))
                            return np.asarray(fs[:, lb.index("anomaly")], float)
                        return f
                    out.append((f"w{int(whiten)}_t{tn}_k{k}_{tnm}", mk()))
    return out


def cands_mahalanobis(F, fit, seed):
    out = []
    ests = {"empirical": lambda: EmpiricalCovariance(),
            "ledoit": lambda: LedoitWolf(), "oas": lambda: OAS(),
            "mcd": lambda: MinCovDet(random_state=seed, support_fraction=0.9)}
    for ename, mk_e in ests.items():
        for shr in (None, 0.0, 0.05, 0.1, 0.3, 0.6):
            for k in (4, 6, 8, 12, 19):
                feats = topk(F, fit, k)
                sc = StandardScaler().fit(F.iloc[fit][feats].to_numpy())
                X = sc.transform(F.iloc[fit][feats].to_numpy())
                try:
                    est = ShrunkCovariance(shrinkage=shr) if shr is not None else mk_e()
                    est.fit(X)
                except Exception:
                    continue
                def mk(est=est, sc=sc, feats=feats):
                    return lambda ix: est.mahalanobis(
                        sc.transform(F.iloc[ix][feats].to_numpy()))
                out.append((f"{ename}_s{shr}_k{k}", mk()))
    return out


def cands_iforest(F, fit, seed):
    out = []
    for n in (50, 100, 200, 400):
        for mf in (0.5, 0.7, 1.0):
            for cont in ("auto", 0.05, 0.1, 0.2):
                for k in (8, 19):
                    feats = topk(F, fit, k)
                    sc = StandardScaler().fit(F.iloc[fit][feats].to_numpy())
                    m = IsolationForest(n_estimators=n, max_features=mf,
                                        contamination=cont, random_state=seed
                                        ).fit(sc.transform(F.iloc[fit][feats].to_numpy()))
                    def mk(m=m, sc=sc, feats=feats):
                        return lambda ix: -m.score_samples(
                            sc.transform(F.iloc[ix][feats].to_numpy()))
                    out.append((f"n{n}_mf{mf}_c{cont}_k{k}", mk()))
    return out


def cands_ocsvm(F, fit, seed):
    out = []
    for nu in (0.01, 0.05, 0.1, 0.2, 0.5):
        for gam in ("scale", "auto", 0.01, 0.1, 1.0):
            for kern in ("rbf", "sigmoid"):
                for k in (8, 19):
                    feats = topk(F, fit, k)
                    sc = StandardScaler().fit(F.iloc[fit][feats].to_numpy())
                    try:
                        m = OneClassSVM(nu=nu, gamma=gam, kernel=kern).fit(
                            sc.transform(F.iloc[fit][feats].to_numpy()))
                    except Exception:
                        continue
                    def mk(m=m, sc=sc, feats=feats):
                        return lambda ix: -m.score_samples(
                            sc.transform(F.iloc[ix][feats].to_numpy()))
                    out.append((f"nu{nu}_g{gam}_{kern}_k{k}", mk()))
    return out


def cands_threshold(F, fit, seed):
    """A single output statistic, thresholded. 'Mean entropy' is one of these."""
    out = []
    for c in SCALAR_COLS:
        for sign in (1, -1):
            out.append((f"{c}_{'+' if sign > 0 else '-'}",
                        (lambda c=c, sign=sign: lambda ix: sign * F.iloc[ix][c]
                         .to_numpy(float))()))
    return out


FAMILIES = {"FIS": cands_fis, "Mahalanobis": cands_mahalanobis,
            "IsolationForest": cands_iforest, "OneClassSVM": cands_ocsvm,
            "single statistic": cands_threshold}
FIXED = {"FIS": "w0_t6_k4_hamacher", "Mahalanobis": "ledoit_sNone_k19",
         "IsolationForest": "n100_mf1.0_cauto_k19",
         "OneClassSVM": "nu0.1_gscale_rbf_k8", "single statistic": "ent_mean_+"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    rows = []

    for m in args.models:
        f = DATA / f"capture_v4_{m}_meta.parquet"
        if not f.exists():
            continue
        meta = pd.read_parquet(f)
        ent = meta["ent_mean"].astype(float)
        F = meta[SCALAR_COLS].reset_index(drop=True).astype(float)

        for seed in range(args.seeds):
            sp = splits(meta, seed)
            rng = np.random.default_rng(53000 + seed)
            m2 = meta.copy()
            m2["_eb"] = np.digitize(ent, np.quantile(
                ent.iloc[np.concatenate([sp["test_neg"], sp["test_pos"]])],
                [.25, .5, .75]))

            va, vb = match(m2, sp["test_neg"], sp["val_pos"],
                           ["template", "n_tokens"], rng)
            vix, vy = (np.concatenate([va, vb]),
                       np.concatenate([np.zeros(len(va)), np.ones(len(vb))]))

            # candidates depend only on the fit split, so build them once per
            # seed rather than once per condition (halves the runtime)
            built = {fam: gen(F, sp["fit"], seed) for fam, gen in FAMILIES.items()}

            for cond, keys in (("length+template", ["template", "n_tokens"]),
                               ("length+template+entropy",
                                ["template", "n_tokens", "_eb"])):
                ta, tb = match(m2, sp["test_neg"], sp["test_pos"], keys, rng)
                if len(ta) < 20:
                    continue
                tix, ty = (np.concatenate([ta, tb]),
                           np.concatenate([np.zeros(len(ta)), np.ones(len(tb))]))

                for fam, cands in built.items():
                    best, best_v = None, -np.inf
                    fixed_auc = np.nan
                    for name, fn in cands:
                        s = np.asarray(fn(vix), float)
                        if not np.isfinite(s).all() or np.ptp(s) == 0:
                            continue
                        v = roc_auc_score(vy, s)
                        if v > best_v:
                            best_v, best = v, (name, fn)
                        if name == FIXED.get(fam):
                            st = np.asarray(fn(tix), float)
                            fixed_auc = (roc_auc_score(ty, st)
                                         if np.isfinite(st).all()
                                         and np.ptp(st) > 0 else np.nan)
                    if best is None:
                        continue
                    st = np.asarray(best[1](tix), float)
                    rows.append({
                        "model": m, "seed": seed, "condition": cond,
                        "family": fam, "budget": len(cands),
                        "selected": best[0], "val_auroc": best_v,
                        "test_auroc": roc_auc_score(ty, st),
                        "test_fpr95": fpr_at_tpr(ty, st),
                        "fixed_auroc": fixed_auc})
            print(f"  {m} seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "equal_budget.csv", index=False)

    for cond in ("length+template", "length+template+entropy"):
        s = df[df.condition == cond]
        if not len(s):
            continue
        g = (s.groupby("family")
             .agg(budget=("budget", "max"),
                  searched=("test_auroc", "mean"), sd=("test_auroc", "std"),
                  fixed=("fixed_auroc", "mean"),
                  fpr95=("test_fpr95", "mean"))
             .sort_values("searched", ascending=False))
        g["search_gain"] = g.searched - g.fixed
        print(f"\n{'='*94}\nEQUAL-BUDGET COMPARISON — '{cond}' "
              f"({args.seeds} seeds x {s.model.nunique()} models)\n{'='*94}")
        print(g.to_string(float_format=lambda v: f"{v:.4f}"))

        piv = s.pivot_table(index=["model", "seed"], columns="family",
                            values="test_auroc")
        if "FIS" in piv.columns:
            print(f"\n  paired vs FIS (searched, same budget):")
            for other in [c for c in piv.columns if c != "FIS"]:
                d = (piv["FIS"] - piv[other]).dropna()
                if len(d) < 6:
                    continue
                p = sstats.wilcoxon(d)[1]
                print(f"    FIS − {other:<17} {d.mean():+.4f} ± {d.std():.4f}  "
                      f"wins {int((d > 0).sum())}/{len(d)}  p = {p:.4f}")

    print(f"\n{'='*94}\nMOST-SELECTED CONFIGURATION PER FAMILY\n{'='*94}")
    for fam in FAMILIES:
        sub = df[df.family == fam]
        if len(sub):
            print(f"  {fam:<17} {sub.selected.value_counts().index[0]}  "
                  f"({sub.selected.value_counts().iloc[0]}/{len(sub)} runs)")


if __name__ == "__main__":
    main()
