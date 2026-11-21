"""Embedding -> PCA -> TRIBBLE fuzzy classifier: semantic separation and which dimensions matter.

Supervised, not clustering: the labels are used. For each corpus (MiniLM 384-d embeddings):

  pipeline   PCA(d) fitted on the base set, d in {16, 32}
  FIS        tribblefis.gaussian_classifier.TribbleClassifier, feature selection by the
             Wasserstein differentiation score (calculate_gaussian_correlation), top_n in
             {4, 8, 16, 32} (only top_n <= d), correlation pruning disabled
  baseline   logistic regression on all d PCA components (same features, no FIS)

Ten seeds: each seed draws a bootstrap resample of the base set (the fit data) and sets the
classifier's random state. Evaluation: accuracy and macro-F1 on the 500 held-out test queries.

Which dimensions matter: each PCA component's Wasserstein score is mapped back to the 384
original MiniLM coordinates through its loadings, importance_j = sum_i score_i * |V_ij|
(normalised). Reported: the top-10 original coordinates by mean importance across seeds, and
the fraction of seeds in which each appears in its top 10 (stability).

Manipulation checks, recorded: (C1) top_n actually changes the selected feature set across
the sweep; (C2) the bootstrap changes the selected set across seeds, so the seed sweep is not
vacuous.

Run (repo root), in the tribble-fis environment:
    uv run --project tribble-fis --with scikit-learn --with pandas --with matplotlib \\
      python reproduce/experiments/ivat_embeddings/run_fis_pca.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.join(_ROOT, "reproduce"))
import common as C  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from tribblefis import gauss_math as G  # noqa: E402
from tribblefis.gaussian_classifier import TribbleClassifier  # noqa: E402

CACHE = os.environ.get(
    "REPRO_EMBED_CACHE", os.path.expanduser("~/.cache/grad-school/embeddings")
)
CORPORA = ["ag_news", "emotion", "newsgroups20"]
PCA_DIMS = [16, 32]
TOP_NS = [4, 8, 16, 32]
SEEDS = list(C.SEEDS)
RAW_JSON = os.path.join(C.OUTPUT_DIR, "fis_pca_raw.json")


def scores_dict(df, y, method="wasserstein"):
    rank = G.calculate_gaussian_correlation(df, pd.Series(y), method=method)
    return {str(f): float(s) for f, s in rank}


def main():
    raw, rows, dim_rows = {}, [], []
    for corp in CORPORA:
        d = np.load(os.path.join(CACHE, f"{corp}.npz"))
        Xb = d["st_base"].astype(np.float64)
        Xq = d["st_query"].astype(np.float64)
        yb, yq = d["y_base"], d["y_query"]
        raw[corp] = {}
        print(f"== {corp}: N={len(Xb)} classes={len(np.unique(yb))}", flush=True)
        for pd_ in PCA_DIMS:
            pca = PCA(pd_, random_state=0).fit(Xb)
            Zb, Zq = pca.transform(Xb), pca.transform(Xq)
            cols = [f"pc{i}" for i in range(pd_)]
            key = f"d={pd_}"
            raw[corp][key] = {"lr": [], "fis": {}, "importance": []}
            # logistic regression on all PCA components (no FIS)
            lr_acc, lr_f1 = [], []
            for s in SEEDS:
                idx = np.random.default_rng(s).choice(len(Zb), len(Zb), replace=True)
                lr = LogisticRegression(max_iter=2000, random_state=s).fit(
                    Zb[idx], yb[idx]
                )
                p = lr.predict(Zq)
                lr_acc.append(accuracy_score(yq, p))
                lr_f1.append(f1_score(yq, p, average="macro"))
            raw[corp][key]["lr"] = {"acc": lr_acc, "macro_f1": lr_f1}
            print(f"  {key} LR acc={np.mean(lr_acc):.3f}", flush=True)

            for tn in TOP_NS:
                if tn > pd_:
                    continue
                accs, f1s, fits, sels = [], [], [], []
                for s in SEEDS:
                    idx = np.random.default_rng(s).choice(
                        len(Zb), len(Zb), replace=True
                    )
                    df = pd.DataFrame(Zb[idx], columns=cols)
                    dq = pd.DataFrame(Zq, columns=cols)
                    t0 = time.perf_counter()
                    m = TribbleClassifier(
                        top_n=tn, correlation_threshold=0.0, random_state=s
                    ).fit(df, pd.Series(yb[idx]))
                    fits.append(time.perf_counter() - t0)
                    p = np.asarray(m.predict(dq))
                    accs.append(accuracy_score(yq, p))
                    f1s.append(f1_score(yq, p, average="macro"))
                    sels.append(sorted(str(f) for f in m.top_features_))
                raw[corp][key]["fis"][f"top_n={tn}"] = {
                    "acc": accs,
                    "macro_f1": f1s,
                    "fit_sec": fits,
                    "selected": sels,
                }
                distinct = len({tuple(x) for x in sels})
                rows.append(
                    [
                        corp,
                        key,
                        str(tn),
                        f"{np.mean(accs):.3f} ± {np.std(accs, ddof=1):.3f}",
                        f"{np.mean(f1s):.3f}",
                        f"{np.mean(fits):.2f}",
                        str(distinct),
                        f"{np.mean(lr_acc):.3f}",
                    ]
                )
                print(
                    f"  {key} top_n={tn:2d} FIS acc={np.mean(accs):.3f} ± {np.std(accs, ddof=1):.3f} "
                    f"macroF1={np.mean(f1s):.3f} fit={np.mean(fits):.2f}s distinct-selections={distinct}",
                    flush=True,
                )

            # which original MiniLM coordinates matter: map Wasserstein scores through PCA loadings
            V = pca.components_  # (d, 384)
            imp_runs = []
            for s in SEEDS:
                idx = np.random.default_rng(s).choice(len(Zb), len(Zb), replace=True)
                sc = scores_dict(pd.DataFrame(Zb[idx], columns=cols), yb[idx])
                si = np.array([sc.get(c, 0.0) for c in cols])
                imp = (si[:, None] * np.abs(V)).sum(axis=0)
                imp = imp / imp.sum()
                imp_runs.append(imp)
            imp_runs = np.array(imp_runs)
            mean_imp = imp_runs.mean(axis=0)
            top10 = np.argsort(-mean_imp)[:10]
            in_top10 = np.mean(
                [np.isin(np.arange(384), np.argsort(-r)[:10]) for r in imp_runs], axis=0
            )
            raw[corp][key]["importance"] = {
                "mean": mean_imp.tolist(),
                "top10": top10.tolist(),
                "top10_stability": in_top10[top10].tolist(),
            }
            for rank_, j in enumerate(top10, 1):
                dim_rows.append(
                    [
                        corp,
                        key,
                        str(rank_),
                        f"MiniLM coord {int(j)}",
                        f"{mean_imp[j]:.4f}",
                        f"{in_top10[j]:.2f}",
                    ]
                )

    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    with open(RAW_JSON, "w") as fh:
        json.dump(raw, fh)
    C.emit(
        "table_fis_pca_accuracy",
        "Embedding -> PCA -> TRIBBLE fuzzy classifier: held-out accuracy and fit time",
        [
            "corpus",
            "PCA",
            "top_n",
            "FIS accuracy (mean ± std)",
            "FIS macro-F1",
            "fit seconds",
            "distinct selections",
            "LR accuracy (all PCA dims)",
        ],
        rows,
        note="Ten seeds (bootstrap resample of the base set per seed). Evaluated on the 500 held-out test queries. distinct selections > 1 means the bootstrap changes the selected features (manipulation check C2).",
        seeds=SEEDS,
    )
    C.emit(
        "table_fis_pca_dimensions",
        "Which original MiniLM coordinates matter: importance through PCA loadings",
        [
            "corpus",
            "PCA",
            "rank",
            "original coordinate",
            "mean importance",
            "top-10 stability (fraction of seeds)",
        ],
        dim_rows,
        note="importance_j = sum_i Wasserstein score_i * |loading_ij|, normalised. Stability = fraction of the ten bootstrap seeds in which the coordinate is in the top 10.",
        seeds=SEEDS,
    )

    # figure: accuracy vs top_n against the LR baseline
    fig, axes = plt.subplots(len(CORPORA), len(PCA_DIMS), figsize=(9, 9), sharey="row")
    for r, corp in enumerate(CORPORA):
        for c, pd_ in enumerate(PCA_DIMS):
            ax = axes[r, c]
            key = f"d={pd_}"
            tns = [t for t in TOP_NS if f"top_n={t}" in raw[corp][key]["fis"]]
            mu = [np.mean(raw[corp][key]["fis"][f"top_n={t}"]["acc"]) for t in tns]
            sd = [
                np.std(raw[corp][key]["fis"][f"top_n={t}"]["acc"], ddof=1) for t in tns
            ]
            ax.errorbar(tns, mu, yerr=sd, marker="o", capsize=3, label="TRIBBLE FIS")
            ax.axhline(
                np.mean(raw[corp][key]["lr"]["acc"]),
                color="k",
                ls="--",
                lw=1,
                label="logistic regression (all dims)",
            )
            ax.set_title(f"{corp}, PCA-{pd_}", fontsize=9)
            ax.set_xlabel("features selected (top_n)", fontsize=8)
            ax.set_xticks(tns)
            ax.grid(alpha=0.3)
            if c == 0:
                ax.set_ylabel("held-out accuracy", fontsize=8)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.suptitle(
        "Embedding -> PCA -> TRIBBLE fuzzy classifier (MiniLM): accuracy vs features kept",
        fontsize=10,
    )
    fig.tight_layout()
    C.save_figure(fig, "fig_fis_pca_accuracy")
    plt.close(fig)

    # figure: which original coordinates matter (top-10 importance, PCA-16)
    fig, axes = plt.subplots(1, len(CORPORA), figsize=(13, 4))
    for ax, corp in zip(axes, CORPORA):
        imp = raw[corp]["d=16"]["importance"]
        top = imp["top10"]
        mean = np.array(imp["mean"])
        stab = imp["top10_stability"]
        ax.barh([f"coord {j}" for j in top][::-1], mean[top][::-1], color="tab:blue")
        for i, st in enumerate(stab[::-1]):
            ax.text(
                mean[top][::-1][i], i, f" stability {st:.1f}", va="center", fontsize=7
            )
        ax.set_title(f"{corp}, PCA-16: top-10 MiniLM coordinates", fontsize=9)
        ax.set_xlabel("normalised importance", fontsize=8)
    fig.tight_layout()
    C.save_figure(fig, "fig_fis_pca_dimensions")
    plt.close(fig)
    print("raw:", RAW_JSON, flush=True)


if __name__ == "__main__":
    main()
