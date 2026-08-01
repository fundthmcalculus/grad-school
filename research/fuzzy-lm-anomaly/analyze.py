"""Stage 3a -- does the signal exist, and where does it live?

Establishes the ground the fuzzy detector has to stand on:

1. **SVD spectrum** of the truthful residual stream -- is the known-good
   manifold low-rank enough for a compact fuzzy rule base?
2. **Representation sweep** over (pooling site, layer, #components,
   L2-normalisation), scored by Mahalanobis AUROC.
3. **Baselines** -- max-softmax, entropy, perplexity, one-class SVM,
   isolation forest, so the FIS has something to beat.

Protocol. The detector is fit on **truthful training answers only**; no
hallucination is ever seen during fitting (open-set, mirroring the BETH
benign-only protocol of Ch 4.3.5). Every representation choice is selected on a
**validation** split and reported on a disjoint **test** split, so the headline
numbers are not tuned on the data they are quoted against.

Evaluation is within-family: truthful and hallucinated TriviaQA answers come
from the same prompt distribution, so nothing separates on prompt topic. The
false-premise family is reported separately as a harder, novel open-set.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings("ignore")

DATA = Path(__file__).parent / "data"
SEED = 0
POOLINGS = ["prompt", "first", "mean"]

SCALAR_COLS = [
    "ent_mean", "ent_min", "ent_max", "ent_std", "ent_first",
    "maxp_mean", "maxp_min", "maxp_max", "maxp_std", "maxp_first",
    "margin_mean", "margin_min", "margin_std", "margin_first",
    "logp_mean", "logp_min", "logp_std", "perplexity", "n_tokens",
]


def fpr_at_tpr(y, s, tpr_target=0.95):
    """False-positive rate at the threshold catching `tpr_target` of hallucinations."""
    thresh = np.quantile(np.sort(s[y == 1]), 1 - tpr_target)
    return float((s[y == 0] >= thresh).mean())


def make_splits(meta):
    """Truthful -> 60/20/20 fit/val/test. Hallucinations -> 50/50 val/test."""
    rng = np.random.default_rng(SEED)
    tq = meta["family"] == "triviaqa"

    def shuffled(mask):
        ix = np.flatnonzero(mask)
        rng.shuffle(ix)
        return ix

    good = shuffled(tq & (meta["label"] == "correct"))
    a, b = int(.6 * len(good)), int(.8 * len(good))
    tq_h = shuffled(tq & (meta["label"] == "hallucination"))
    fp_h = shuffled((meta["family"] == "falsepremise") & (meta["label"] == "hallucination"))

    return {
        "fit": good[:a], "val_neg": good[a:b], "test_neg": good[b:],
        "val_tq": tq_h[: len(tq_h) // 2], "test_tq": tq_h[len(tq_h) // 2:],
        "val_fp": fp_h[: len(fp_h) // 2], "test_fp": fp_h[len(fp_h) // 2:],
    }


def embed(hidden, sp, layer, k, l2):
    """Fit scaler+PCA on the truthful fit split only; return a transform closure."""
    X = hidden[:, layer, :].astype(np.float32)
    if l2:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sc = StandardScaler().fit(X[sp["fit"]])
    p = PCA(n_components=k, random_state=SEED).fit(sc.transform(X[sp["fit"]]))
    return lambda ix: p.transform(sc.transform(X[ix]))


def score_pair(fn, sp, pos_key, split):
    """Mahalanobis AUROC for one (neg, pos) split pair."""
    neg, pos = sp[f"{split}_neg"], sp[pos_key]
    lw = LedoitWolf().fit(fn(sp["fit"]))
    s = lw.mahalanobis(fn(np.concatenate([neg, pos])))
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    return roc_auc_score(y, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer-stride", type=int, default=2)
    args = ap.parse_args()

    meta = pd.read_parquet(DATA / "capture_meta.parquet")
    hid = {p: np.load(DATA / f"capture_hidden_{p}.npy") for p in POOLINGS}
    n_layers = hid["mean"].shape[1]
    sp = make_splits(meta)

    print("split sizes:", {k: len(v) for k, v in sp.items()})

    # ---------------- 1. SVD spectrum --------------------------------------
    print("\n" + "=" * 72)
    print("1. SVD SPECTRUM OF THE TRUTHFUL RESIDUAL STREAM (pooling=mean)")
    print("=" * 72)
    print(f"{'layer':>6} {'k@90%':>7} {'k@95%':>7} {'k@99%':>7} {'eff.rank':>9} {'top1':>7}")
    for li in list(range(0, n_layers, 8)) + [n_layers - 1]:
        H = StandardScaler().fit_transform(hid["mean"][sp["fit"], li, :].astype(np.float32))
        ev = np.linalg.svd(H, compute_uv=False) ** 2
        ev /= ev.sum()
        cum = np.cumsum(ev)
        eff = float(np.exp(-(ev * np.log(ev + 1e-12)).sum()))
        print(f"{li:>6} {np.searchsorted(cum, .9)+1:>7} {np.searchsorted(cum, .95)+1:>7} "
              f"{np.searchsorted(cum, .99)+1:>7} {eff:>9.1f} {ev[0]:>7.3f}")

    # ---------------- 2. representation sweep (selected on VAL) ------------
    print("\n" + "=" * 72)
    print("2. REPRESENTATION SWEEP -- Mahalanobis AUROC on the VALIDATION split")
    print("=" * 72)
    recs = []
    for pool in POOLINGS:
        for l2 in (False, True):
            for k in (8, 16, 32, 64):
                for li in range(0, n_layers, args.layer_stride):
                    fn = embed(hid[pool], sp, li, k, l2)
                    recs.append({"pooling": pool, "l2": l2, "k": k, "layer": li,
                                 "val_tq": score_pair(fn, sp, "val_tq", "val"),
                                 "val_fp": score_pair(fn, sp, "val_fp", "val")})
    sweep = pd.DataFrame(recs)
    sweep.to_csv(DATA / "representation_sweep.csv", index=False)

    print("\nbest 8 configurations by validation TriviaQA AUROC:")
    print(sweep.nlargest(8, "val_tq").to_string(index=False,
                                                float_format=lambda v: f"{v:.3f}"))
    print("\nbest per pooling site (val TriviaQA):")
    print(sweep.loc[sweep.groupby("pooling")["val_tq"].idxmax()]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    best = sweep.loc[sweep["val_tq"].idxmax()]
    print(f"\nSELECTED (on val): pooling={best.pooling} layer={int(best.layer)} "
          f"k={int(best.k)} l2={bool(best.l2)}")

    # ---------------- 3. baselines on the untouched TEST split -------------
    print("\n" + "=" * 72)
    print("3. TEST-SPLIT BASELINES (representation chosen on val, never on test)")
    print("=" * 72)

    fn = embed(hid[best.pooling], sp, int(best.layer), int(best.k), bool(best.l2))
    ssc = StandardScaler().fit(meta.iloc[sp["fit"]][SCALAR_COLS].to_numpy())
    S = lambda ix: ssc.transform(meta.iloc[ix][SCALAR_COLS].to_numpy())

    Zfit, Sfit = fn(sp["fit"]), S(sp["fit"])
    oc = OneClassSVM(nu=.1, gamma="scale").fit(Zfit)
    iso_h = IsolationForest(random_state=SEED).fit(Zfit)
    iso_s = IsolationForest(random_state=SEED).fit(Sfit)
    lw_h, lw_s = LedoitWolf().fit(Zfit), LedoitWolf().fit(Sfit)
    lw_f = LedoitWolf().fit(np.hstack([Zfit, Sfit]))

    for tag, pos_key in [("TriviaQA", "test_tq"), ("FalsePremise", "test_fp")]:
        ix = np.concatenate([sp["test_neg"], sp[pos_key]])
        y = np.concatenate([np.zeros(len(sp["test_neg"])), np.ones(len(sp[pos_key]))])
        m, Z, Ss = meta.iloc[ix], fn(ix), S(ix)

        rows = []

        def rep(name, s):
            rows.append({"detector": name, "auroc": roc_auc_score(y, s),
                         "auprc": average_precision_score(y, s),
                         "fpr@95tpr": fpr_at_tpr(y, s)})

        rep("max-softmax (1-maxp)", -m["maxp_mean"].to_numpy())
        rep("mean entropy", m["ent_mean"].to_numpy())
        rep("perplexity", m["perplexity"].to_numpy())
        rep("margin (neg)", -m["margin_mean"].to_numpy())
        rep("Mahalanobis (hidden)", lw_h.mahalanobis(Z))
        rep("OneClassSVM (hidden)", -oc.score_samples(Z))
        rep("IsolationForest (hidden)", -iso_h.score_samples(Z))
        rep("Mahalanobis (stats)", lw_s.mahalanobis(Ss))
        rep("IsolationForest (stats)", -iso_s.score_samples(Ss))
        rep("Mahalanobis (fused)", lw_f.mahalanobis(np.hstack([Z, Ss])))

        df = pd.DataFrame(rows).sort_values("auroc", ascending=False)
        print(f"\n--- {tag}: hallucination vs truthful (n_pos={len(sp[pos_key])}, "
              f"n_neg={len(sp['test_neg'])}) ---")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        df.to_csv(DATA / f"baselines_{tag.lower()}.csv", index=False)

    pd.Series({"pooling": best.pooling, "layer": int(best.layer),
               "k": int(best.k), "l2": bool(best.l2)}).to_json(DATA / "best_repr.json")


if __name__ == "__main__":
    main()
