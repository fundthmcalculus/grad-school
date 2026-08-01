"""Stage 17 -- a fuzzy rule over the output statistics (the representation that works).

Section 20 established that the only representation surviving every control is the
19 output-distribution statistics: Mahalanobis over them holds 0.799 with entropy
matched, where mean entropy itself falls to 0.642. Every hidden-state family
failed.

So the question for the dissertation is no longer "can fuzzy beat entropy on
activations" (no) but the sharper one:

    Can a SMALL, READABLE fuzzy rule over the output statistics match the
    full-covariance Mahalanobis detector on that same representation?

If yes, the fuzzy contribution is real and defensible: same accuracy, far fewer
parameters, and a rule base you can print -- which Mahalanobis cannot offer.

Protocol. Hyperparameters (antecedent count, mode count, norm pair, whitening)
are selected on a VALIDATION half of the positives and reported on the disjoint
test half, so the headline is not the maximum of a grid. Rule bases are still fit
on grounded data only. Both matched conditions from section 20 are reported, with
the standing cost columns.

On whitening: a 19x19 linear decorrelation is included as a variant because the
factorized-Gaussian antecedents of a MoG FIS assume independence, and these 19
statistics are strongly correlated. This is NOT the dimensionality reduction that
was dropped in section 7 -- no components are discarded, and it is reported as its
own labelled row so the comparison stays honest.
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
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                   take_top_features)
import fis_config as CFG

CFG_NORM = CFG.NORM
from analyze import DATA, SCALAR_COLS, fpr_at_tpr
from norm_sweep import anomaly_score, membership_tensors
from seed_sweep import Timer
from template_control import match

warnings.filterwarnings("ignore")

SEEDS = 8
TOP_NS = [4, 6, 8, 12, 19]
K_MODES = [2, 3, 4]
NORMS = [("product", "product"), ("hamacher", "hamacher"),
         ("dombi2", "einstein"), ("minimum", "minimum")]
THETA = 0.5


# (negative family, negative label, positive family) per capture set.
TASKS = {
    "capture_v3":      ("longform_real", "grounded", "longform_fake"),
    "capture_v3_qwen": ("longform_real", "grounded", "longform_fake"),
    "capture_v2":      ("template_real", "correct", "template_fake"),
    # v4: the expanded probe set (1,272 real / 4,052 fake), four models,
    # all captured in bfloat16 so precision is identical across architectures.
    "capture_v4_smollm2": ("longform_real", "grounded", "longform_fake"),
    "capture_v4_qwen":    ("longform_real", "grounded", "longform_fake"),
    "capture_v4_gemma":   ("longform_real", "grounded", "longform_fake"),
    "capture_v4_lfm":     ("longform_real", "grounded", "longform_fake"),
}


def splits(meta, seed, task="capture_v3"):
    """Known-good -> fit/test. Fabrications -> val (selection) / test (reporting)."""
    neg_fam, neg_lab, pos_fam = TASKS[task]
    rng = np.random.default_rng(seed)
    good = np.flatnonzero((meta.family == neg_fam) & (meta.label == neg_lab))
    bad = np.flatnonzero((meta.family == pos_fam)
                         & (meta.label == "hallucination"))
    rng.shuffle(good)
    rng.shuffle(bad)
    g = int(.55 * len(good))
    return {"fit": good[:g], "test_neg": good[g:],
            "val_pos": bad[:len(bad) // 2], "test_pos": bad[len(bad) // 2:]}


def make_features(meta, fit, whiten):
    F = meta[SCALAR_COLS].reset_index(drop=True).astype(float)
    if not whiten:
        return F
    sc = StandardScaler().fit(F.iloc[fit].to_numpy())
    p = PCA(whiten=True, random_state=0).fit(sc.transform(F.iloc[fit].to_numpy()))
    Z = p.transform(sc.transform(F.to_numpy()))
    return pd.DataFrame(Z, columns=[f"W{i:02d}" for i in range(Z.shape[1])])


def fit_fis(F, fit, top_n, k, seed):
    feats0 = list(F.iloc[fit].var().sort_values(ascending=False).index[:top_n])
    Xf = StandardScaler().fit_transform(F.iloc[fit][feats0].to_numpy())
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Xf).labels_
    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        diff = calculate_gaussian_correlation(
            F.iloc[fit].reset_index(drop=True), y_modes, method=CFG.METRIC)
        _, feats = take_top_features(diff, top_n=top_n)
        # explicit membership family, verified -- see fis_config for why
        model = CFG.build_memberships(
            F.iloc[fit][feats].reset_index(drop=True), y_modes, feats,
            membership=CFG.MEMBERSHIP)
    return feats, model


def score_with(F, feats, model, ix, tn, sn):
    classes, tens = membership_tensors(F, ix, model, feats)
    with np.errstate(all="ignore"):
        return np.asarray(anomaly_score(classes, tens, tn, sn, THETA), float)


def auroc(y, s):
    return (roc_auc_score(y, s)
            if np.isfinite(s).all() and np.ptp(s) > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--no-select", action="store_true",
                    help="skip the validation configuration search and use "
                         "fis_config's declared defaults. The search scores "
                         "candidate configurations against LABELLED validation "
                         "positives, which the rival detectors never receive -- "
                         "so this flag is the like-for-like comparison.")
    ap.add_argument("--select-on", default="auroc", choices=["auroc", "fpr95"],
                    help="validation criterion. Selecting on AUROC optimises the "
                         "whole ranking and can pick a configuration with a poor "
                         "high-recall tail; fpr95 targets the tail directly.")
    ap.add_argument("--prefix", default="capture_v3", choices=list(TASKS),
                    help="which capture set / task to analyse")
    args = ap.parse_args()

    meta = pd.read_parquet(DATA / f"{args.prefix}_meta.parquet")
    print(f"task={args.prefix}  n={len(meta):,}  "
          f"known-good={TASKS[args.prefix][0]}/{TASKS[args.prefix][1]}  "
          f"fabrications={TASKS[args.prefix][2]}")
    ent = meta["ent_mean"].astype(float)
    rows, sel_log = [], []

    for seed in range(args.seeds):
        sp = splits(meta, seed, args.prefix)
        rng = np.random.default_rng(19000 + seed)
        m2 = meta.copy()
        m2["_eb"] = np.digitize(ent, np.quantile(
            ent.iloc[np.concatenate([sp["test_neg"], sp["test_pos"]])],
            [.25, .5, .75]))
        conds = {"length+template": ["template", "n_tokens"],
                 "length+template+entropy": ["template", "n_tokens", "_eb"]}

        # ---- select the FIS configuration on the VALIDATION positives ----
        Fraw = make_features(meta, sp["fit"], False)
        Fwht = make_features(meta, sp["fit"], True)
        va, vb = match(m2, sp["test_neg"], sp["val_pos"],
                       ["template", "n_tokens"], rng)
        vix = np.concatenate([va, vb])
        vy = np.concatenate([np.zeros(len(va)), np.ones(len(vb))])

        best, cache = (None, -1), {}
        if args.no_select:
            feats, model = fit_fis(Fraw, sp["fit"], 6, 4, seed)
            cache[(False, 6, 4)] = (feats, model)
            best = ((False, 6, 4, CFG_NORM, CFG_NORM), float("nan"))
        for whiten, F in (() if args.no_select else ((False, Fraw), (True, Fwht))):
            for tn_ in TOP_NS:
                for k in K_MODES:
                    feats, model = fit_fis(F, sp["fit"], tn_, k, seed)
                    cache[(whiten, tn_, k)] = (feats, model)
                    for nt, ns in NORMS:
                        sv = score_with(F, feats, model, vix, nt, ns)
                        a = auroc(vy, sv)
                        if not np.isfinite(a):
                            continue
                        # maximise: AUROC, or the negated FPR@95 so lower is better
                        crit = a if args.select_on == "auroc" else -fpr_at_tpr(vy, sv)
                        if crit > best[1]:
                            best = ((whiten, tn_, k, nt, ns), crit)
        (whiten, tn_, k, nt, ns), val_au = best
        sel_log.append({"seed": seed, "whiten": whiten, "top_n": tn_, "K": k,
                        "tnorm": nt, "sconorm": ns, "val_criterion": val_au,
                        "select_on": args.select_on})

        F = Fwht if whiten else Fraw
        feats, model = cache[(whiten, tn_, k)]
        with Timer() as t_fit:
            fit_fis(F, sp["fit"], tn_, k, seed)     # timed representative fit

        # ---- rivals on the same 19 statistics ----------------------------
        d = Fraw.shape[1]
        with Timer() as t_mah:
            ssc = StandardScaler().fit(Fraw.iloc[sp["fit"]].to_numpy())
            lw = LedoitWolf().fit(ssc.transform(Fraw.iloc[sp["fit"]].to_numpy()))
        with Timer() as t_oc:
            oc = OneClassSVM(nu=.1, gamma="scale").fit(
                ssc.transform(Fraw.iloc[sp["fit"]].to_numpy()))
        with Timer() as t_if:
            iso = IsolationForest(random_state=seed).fit(
                ssc.transform(Fraw.iloc[sp["fit"]].to_numpy()))

        nsv = oc.support_vectors_.shape[0]
        dets = {
            f"FIS · stats{' (whitened)' if whiten else ''}":
                (lambda ix: score_with(F, feats, model, ix, nt, ns),
                 t_fit.ms, 2 * model.n_membership_functions + 1,
                 f"{model.n_rules}R/{model.n_membership_functions}MF/{tn_}ant"),
            "Mahalanobis · stats":
                (lambda ix: lw.mahalanobis(ssc.transform(Fraw.iloc[ix].to_numpy())),
                 t_mah.ms, d + d * (d + 1) // 2, f"{d}f full cov"),
            "OneClassSVM · stats":
                (lambda ix: -oc.score_samples(ssc.transform(Fraw.iloc[ix].to_numpy())),
                 t_oc.ms, nsv * d + nsv + 1, f"{nsv}SV"),
            "IsolationForest · stats":
                (lambda ix: -iso.score_samples(ssc.transform(Fraw.iloc[ix].to_numpy())),
                 t_if.ms, int(sum((e.tree_.children_left != -1).sum() * 2
                                  for e in iso.estimators_)), "100 trees"),
            "mean entropy": (lambda ix: meta.iloc[ix].ent_mean.to_numpy(float),
                             0.0, 0, "threshold"),
            "perplexity": (lambda ix: meta.iloc[ix].perplexity.to_numpy(float),
                           0.0, 0, "threshold"),
            "n_tokens (control)": (lambda ix: meta.iloc[ix].n_tokens.to_numpy(float),
                                   0.0, 0, "threshold"),
        }

        for cond, keys in conds.items():
            a, b = match(m2, sp["test_neg"], sp["test_pos"], keys, rng)
            if len(a) < 20:
                continue
            ix = np.concatenate([a, b])
            y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
            for nm, (fn, tms, npar, struct) in dets.items():
                with Timer() as t:
                    s = np.asarray(fn(ix), float)
                rows.append({"seed": seed, "condition": cond,
                             "detector": nm.replace(" (whitened)", ""),
                             "whitened": "(whitened)" in nm,
                             "auroc": auroc(y, s),
                             "fpr95": fpr_at_tpr(y, s) if np.isfinite(s).all() else np.nan,
                             "n_neg": len(a), "n_pos": len(b), "train_ms": tms,
                             "n_params": npar, "structure": struct,
                             "samples_per_sec": len(ix) / max(t.ms / 1000, 1e-9)})
        print(f"  seed {seed}: whiten={whiten} top_n={tn_} K={k} "
              f"T={nt}/S={ns} (val {args.select_on} {val_au:+.3f})")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / (f"fuzzy_stats_{args.prefix}_"
                f"{'fixed' if args.no_select else args.select_on}.csv"), index=False)
    pd.DataFrame(sel_log).to_csv(
        DATA / f"fuzzy_stats_selection_{args.prefix}_{args.select_on}.csv", index=False)

    for cond in ("length+template", "length+template+entropy"):
        s = df[df.condition == cond]
        if not len(s):
            continue
        rep = (s.groupby("detector")
               .agg(auroc=("auroc", "mean"), auroc_std=("auroc", "std"),
                    fpr95=("fpr95", "mean"), n_params=("n_params", "mean"),
                    train_ms=("train_ms", "mean"),
                    samples_per_sec=("samples_per_sec", "mean"))
               .sort_values("auroc", ascending=False))
        rep["structure"] = s.groupby("detector")["structure"].first()
        out = rep.copy()
        out["AUROC"] = (out.auroc.map("{:.3f}".format) + " ± "
                        + out.auroc_std.map("{:.3f}".format))
        out["params"] = out.n_params.map(lambda v: f"{int(v):,}")
        out["train"] = out.train_ms.map(lambda v: f"{v:,.0f} ms")
        out["infer"] = out.samples_per_sec.map(lambda v: f"{v:,.0f}/s")
        print(f"\n{'='*104}\nFUZZY RULE OVER THE OUTPUT STATISTICS — "
              f"condition '{cond}' ({args.seeds} seeds)\n{'='*104}")
        print(out[["AUROC", "fpr95", "params", "train", "infer", "structure"]]
              .to_string(float_format=lambda v: f"{v:.3f}"))

        m = s.pivot_table(index="seed", columns="detector", values="auroc")
        if "FIS · stats" in m and "Mahalanobis · stats" in m:
            dd = (m["FIS · stats"] - m["Mahalanobis · stats"]).dropna()
            p = sstats.wilcoxon(dd)[1] if len(dd) >= 6 else np.nan
            print(f"\n  paired FIS vs Mahalanobis (same 19 statistics):")
            print(f"    mean Δ = {dd.mean():+.3f} ± {dd.std():.3f} "
                  f"(min {dd.min():+.3f}, max {dd.max():+.3f}) · "
                  f"wins {int((dd > 0).sum())}/{len(dd)}"
                  + (f" · p = {p:.4f}" if np.isfinite(p) else ""))
            r = rep.loc["FIS · stats"], rep.loc["Mahalanobis · stats"]
            print(f"    parameter ratio: {r[1].n_params / max(r[0].n_params,1):.1f}x "
                  f"fewer for the FIS ({int(r[0].n_params)} vs "
                  f"{int(r[1].n_params)})")

    print(f"\nselected configurations across seeds:")
    print(pd.DataFrame(sel_log).to_string(index=False,
                                          float_format=lambda v: f"{v:.3f}"))





def print_best_rule(seed=0, top_n=6, k=4, tn="minimum", sn="minimum"):
    """Print the winning rule base -- the interpretability payoff."""
    from print_rule import describe, plot_mfs, OUT
    meta = pd.read_parquet(DATA / "capture_v3_meta.parquet")
    sp = splits(meta, seed)
    F = make_features(meta, sp["fit"], False)
    feats, model = fit_fis(F, sp["fit"], top_n, k, seed)
    print(f"\nselected config: top_n={top_n} K={k} T={tn} S={sn} (no whitening)")
    describe(model, F, sp["fit"], feats)
    OUT.mkdir(exist_ok=True)
    plot_mfs(model, F, sp["fit"], feats, OUT / "membership_functions_stats")


if __name__ == "__main__":
    import sys as _s
    if "--print-rule" in _s.argv:
        print_best_rule()
    else:
        main()
