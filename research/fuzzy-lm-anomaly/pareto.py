"""Stage 11 -- the standing report: accuracy, parameters, train time, inference.

Every detector comparison from here on reports four things, so cost is never
implicit:

  tunable parameters  continuous values fitted from data
  train time          feature construction + fitting, on the fit split
  inference speed     samples scored per second
  accuracy            AUROC on the honest (template-matched) task

Then it computes the actual Pareto front -- a detector is dominated if another
is at least as good on accuracy AND cheaper on the cost axis. This is reported
as measured, not as hoped: if a zero-parameter baseline dominates the fuzzy
rule, that is what the table will say.

Parameter counts (continuous, fitted from data):
  FIS            2 per Gaussian MF (mu, sigma) + 1 for theta
                 (antecedent choice and mode count are discrete structure,
                  reported separately, not counted here)
  Mahalanobis    d means + d(d+1)/2 covariance entries
  OneClassSVM    n_SV x d support vectors + n_SV duals + 1 offset
  IsolationForest 2 per internal node (feature index + threshold), all trees
  entropy / perplexity / n_tokens   0 fitted parameters (a threshold only)
"""

import contextlib
import io
import sys
import time
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
mpl.use("Agg")

from analyze import DATA, fpr_at_tpr
from nopca import POOLING, drop_constant, f_centroid, f_stats
from seed_sweep import Timer, fit_fis
from template_control import match, splits_v2

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "figures"
SEEDS = 6
SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#4a3aa7"


def n_params_iforest(iso):
    return int(sum((e.tree_.children_left != -1).sum() * 2
                   for e in iso.estimators_))


def main():
    meta = pd.read_parquet(DATA / "capture_v2_meta.parquet")
    hidden = np.load(DATA / f"capture_v2_hidden_{POOLING}.npy")
    rows = []

    for seed in range(SEEDS):
        sp = splits_v2(meta, seed)
        rng = np.random.default_rng(9000 + seed)

        with Timer() as t_fc:
            Fc = drop_constant(f_centroid(meta, hidden, sp).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0), sp["fit"])
        with Timer() as t_fs:
            Fs = f_stats(meta, hidden, sp)

        entries = {}

        with Timer() as t:
            with contextlib.redirect_stdout(io.StringIO()):
                fis, feats, model = fit_fis(Fc, sp, seed)
        entries["FIS · centroid"] = (
            fis, t_fc.ms + t.ms, 2 * model.n_membership_functions + 1,
            f"{model.n_rules} rules, {model.n_membership_functions} MFs, "
            f"{len(feats)} antecedents")

        d = Fs.shape[1]
        with Timer() as t:
            ssc = StandardScaler().fit(Fs.iloc[sp["fit"]].to_numpy())
            lw = LedoitWolf().fit(ssc.transform(Fs.iloc[sp["fit"]].to_numpy()))
        entries["Mahalanobis · stats"] = (
            lambda ix: lw.mahalanobis(ssc.transform(Fs.iloc[ix].to_numpy())),
            t_fs.ms + t.ms, d + d * (d + 1) // 2, f"{d} features, full covariance")

        with Timer() as t:
            csc = StandardScaler().fit(Fc.iloc[sp["fit"]][feats].to_numpy())
            oc = OneClassSVM(nu=.1, gamma="scale").fit(
                csc.transform(Fc.iloc[sp["fit"]][feats].to_numpy()))
        nsv = oc.support_vectors_.shape[0]
        entries["OneClassSVM · centroid"] = (
            lambda ix: -oc.score_samples(csc.transform(Fc.iloc[ix][feats].to_numpy())),
            t_fc.ms + t.ms, nsv * len(feats) + nsv + 1,
            f"{nsv} support vectors x {len(feats)} dims")

        with Timer() as t:
            isc = StandardScaler().fit(Fs.iloc[sp["fit"]].to_numpy())
            iso = IsolationForest(random_state=seed).fit(
                isc.transform(Fs.iloc[sp["fit"]].to_numpy()))
        entries["IsolationForest · stats"] = (
            lambda ix: -iso.score_samples(isc.transform(Fs.iloc[ix].to_numpy())),
            t_fs.ms + t.ms, n_params_iforest(iso),
            f"{len(iso.estimators_)} trees")

        for nm, col in (("mean entropy", "ent_mean"), ("perplexity", "perplexity"),
                        ("n_tokens (control)", "n_tokens")):
            entries[nm] = (lambda ix, c=col: meta.iloc[ix][c].to_numpy(float),
                           0.0, 0, "threshold only")

        a, b = match(meta, sp["test_neg"], sp["test_pos"],
                     ["template", "n_tokens"], rng)
        if len(a) < 30:
            continue
        ix = np.concatenate([a, b])
        y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])

        for nm, (fn, train_ms, n_par, note) in entries.items():
            with Timer() as t:
                s = np.asarray(fn(ix), dtype=float)
            ok = np.isfinite(s).all() and np.ptp(s) > 0
            rows.append({
                "detector": nm, "seed": seed,
                "auroc": roc_auc_score(y, s) if ok else np.nan,
                "fpr@95tpr": fpr_at_tpr(y, s) if ok else np.nan,
                "train_ms": train_ms, "n_params": n_par,
                "samples_per_sec": len(ix) / max(t.ms / 1000, 1e-9),
                "structure": note,
            })
        print(f"  seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "pareto.csv", index=False)

    g = (df.groupby("detector")
         .agg(auroc=("auroc", "mean"), auroc_std=("auroc", "std"),
              fpr95=("fpr@95tpr", "mean"), n_params=("n_params", "mean"),
              train_ms=("train_ms", "mean"),
              samples_per_sec=("samples_per_sec", "mean"))
         .sort_values("auroc", ascending=False))
    g["structure"] = df.groupby("detector")["structure"].first()

    print(f"\n{'='*104}\nSTANDING REPORT — template-matched task, {SEEDS} seeds"
          f"\n{'='*104}")
    show = g.copy()
    show["AUROC"] = (show.auroc.map("{:.3f}".format) + " ± "
                     + show.auroc_std.map("{:.3f}".format))
    show["params"] = show.n_params.map(lambda v: f"{int(v):,}")
    show["train"] = show.train_ms.map(lambda v: f"{v:,.0f} ms")
    show["infer"] = show.samples_per_sec.map(lambda v: f"{v:,.0f}/s")
    print(show[["AUROC", "fpr95", "params", "train", "infer", "structure"]]
          .to_string(float_format=lambda v: f"{v:.3f}"))

    # ---- Pareto front: maximise AUROC, minimise each cost axis -----------
    print(f"\n{'='*104}\nPARETO ANALYSIS (maximise AUROC, minimise cost)\n{'='*104}")
    for cost, label in (("n_params", "tunable parameters"),
                        ("train_ms", "training time"),
                        ("samples_per_sec", "inference speed (higher better)")):
        asc = cost == "samples_per_sec"
        front = []
        for nm, r in g.iterrows():
            dominated = any(
                (o.auroc >= r.auroc)
                and ((o[cost] >= r[cost]) if asc else (o[cost] <= r[cost]))
                and (o.auroc > r.auroc
                     or ((o[cost] > r[cost]) if asc else (o[cost] < r[cost])))
                for on, o in g.iterrows() if on != nm)
            if not dominated:
                front.append(nm)
        print(f"\nAUROC vs {label}:")
        for nm in g.index:
            mark = "  ON FRONT" if nm in front else "  dominated"
            print(f"  {nm:<24} AUROC {g.loc[nm,'auroc']:.3f}  "
                  f"{cost} {g.loc[nm,cost]:>12,.0f}{mark}")

    # ---- figure ----------------------------------------------------------
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.5), facecolor=SURFACE)
    for ax, (cost, xlabel, logx) in zip(axes, [
            ("n_params", "Tunable parameters  (← fewer is better)", True),
            ("train_ms", "Training time, ms  (← faster is better)", True)]):
        for nm, r in g.iterrows():
            is_fis = "FIS" in nm
            c = S1 if is_fis else (S2 if "n_tokens" in nm else INK_MUTED)
            x = max(r[cost], 0.5)
            ax.scatter(x, r.auroc, s=95 if is_fis else 55, color=c,
                       zorder=4, edgecolor=SURFACE, linewidth=1.4)
            ax.annotate(f"{nm.split(' · ')[0]}", (x, r.auroc),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=7.4, color=c,
                        fontweight="bold" if is_fis else "normal")
        if logx:
            ax.set_xscale("symlog", linthresh=1)
        ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
        ax.set_xlabel(xlabel, fontsize=8.5, color=INK_2)
        ax.set_ylabel("AUROC  (↑ higher is better)", fontsize=8.5, color=INK_2)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=INK_2, length=0)
    fig.suptitle("Cost vs accuracy — template-matched task "
                 f"({SEEDS} seeds)", x=0.02, ha="left", fontsize=11.5,
                 color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pareto.{ext}", dpi=200, facecolor=SURFACE)
    print(f"\nwrote {OUT / 'pareto.png'}")


if __name__ == "__main__":
    main()
