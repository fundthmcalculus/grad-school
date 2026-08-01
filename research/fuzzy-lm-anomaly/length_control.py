"""Stage 5 -- is the false-premise result just answer length?

`n_tokens` alone scores AUROC 0.853 on the false-premise family (median 48
tokens vs 21 for truthful answers -- confabulations run to the generation cap).
That is higher than every baseline in §3.1, so any detector evaluated on the
raw split may be reading length rather than fabrication.

Control: **exact matching on the confounder.** For each distinct `n_tokens`
value, keep k = min(#positives, #negatives) from each side. The two classes then
have identical length distributions by construction, so `n_tokens` becomes
exactly uninformative (AUROC 0.5) and any surviving separation is not length.

Every detector is scored on the raw split and the matched split so the drop
attributable to length is visible per detector.
"""

import contextlib
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from tribblefis.gauss_math import (
    calculate_gaussian_correlation,
    create_gaussian_membership_dict,
    take_top_features,
)

from analyze import DATA, SCALAR_COLS, SEED, fpr_at_tpr, make_splits
from nopca import POOLING, BEST_LAYER, f_centroid, f_stats
from norm_sweep import anomaly_score, membership_tensors

warnings.filterwarnings("ignore")


def exact_match_on_length(meta, neg, pos, rng):
    """Keep min(#pos, #neg) per n_tokens value so the marginals coincide."""
    ln = meta.iloc[neg].n_tokens.to_numpy()
    lp = meta.iloc[pos].n_tokens.to_numpy()
    keep_n, keep_p = [], []
    for v in np.union1d(np.unique(ln), np.unique(lp)):
        i_n, i_p = neg[ln == v], pos[lp == v]
        k = min(len(i_n), len(i_p))
        if k == 0:
            continue
        keep_n.append(rng.choice(i_n, k, replace=False))
        keep_p.append(rng.choice(i_p, k, replace=False))
    return (np.concatenate(keep_n) if keep_n else np.array([], int),
            np.concatenate(keep_p) if keep_p else np.array([], int))


def build_centroid_fis(F, sp, top_n=8, K=2):
    """The best PCA-free configuration from stage 4."""
    feats0 = list(F.iloc[sp["fit"]].var().sort_values(ascending=False).index[:top_n])
    Xf = StandardScaler().fit_transform(F.iloc[sp["fit"]][feats0].to_numpy())
    labels = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Xf).labels_
    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        diff = calculate_gaussian_correlation(
            F.iloc[sp["fit"]].reset_index(drop=True), y_modes)
        _, feats = take_top_features(diff, top_n=top_n)
        model = create_gaussian_membership_dict(
            F.iloc[sp["fit"]][feats].reset_index(drop=True), y_modes,
            top_n_var_names=feats)

    def fn(ix):
        classes, tens = membership_tensors(F, ix, model, feats)
        with np.errstate(all="ignore"):
            return anomaly_score(classes, tens, "dombi2", "einstein", 0.5)
    return fn, feats


def main():
    rng = np.random.default_rng(SEED)
    meta = pd.read_parquet(DATA / "capture_meta.parquet")
    hidden = np.load(DATA / f"capture_hidden_{POOLING}.npy")
    sp = make_splits(meta)

    Fc = f_centroid(meta, hidden, sp).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    Fs = f_stats(meta, hidden, sp)
    cent_fn, cent_feats = build_centroid_fis(Fc, sp)
    print(f"centroid FIS antecedents: {cent_feats}")

    # Mahalanobis on raw stats (the strongest PCA-free baseline from stage 4)
    ssc = StandardScaler().fit(Fs.iloc[sp["fit"]].to_numpy())
    lw = LedoitWolf().fit(ssc.transform(Fs.iloc[sp["fit"]].to_numpy()))

    DETECTORS = {
        "n_tokens (confound probe)": lambda ix: meta.iloc[ix].n_tokens.to_numpy(float),
        "perplexity": lambda ix: meta.iloc[ix].perplexity.to_numpy(float),
        "mean entropy": lambda ix: meta.iloc[ix].ent_mean.to_numpy(float),
        "Mahalanobis · stats": lambda ix: lw.mahalanobis(
            ssc.transform(Fs.iloc[ix].to_numpy())),
        "FIS · centroid (PCA-free)": cent_fn,
    }

    for tag, pk in [("FalsePremise", "test_fp"), ("TriviaQA", "test_tq")]:
        neg, pos = sp["test_neg"], sp[pk]
        mneg, mpos = exact_match_on_length(meta, neg, pos, rng)

        print(f"\n{'='*78}\n{tag}\n{'='*78}")
        print(f"raw     : {len(neg)} truthful vs {len(pos)} hallucinated")
        print(f"matched : {len(mneg)} vs {len(mpos)} "
              f"(exact on n_tokens; {len(mpos)/max(len(pos),1):.0%} of positives kept)")
        if len(mneg) < 30:
            print("  matched sample too small to interpret -- skipping")
            continue

        rows = []
        for name, fn in DETECTORS.items():
            out = {"detector": name}
            for lab, (a, b) in (("raw", (neg, pos)), ("matched", (mneg, mpos))):
                ix = np.concatenate([a, b])
                y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
                s = np.asarray(fn(ix), dtype=float)
                out[f"auroc_{lab}"] = (roc_auc_score(y, s)
                                       if np.isfinite(s).all() and np.ptp(s) > 0
                                       else np.nan)
            out["delta"] = out["auroc_matched"] - out["auroc_raw"]
            rows.append(out)

        df = pd.DataFrame(rows).sort_values("auroc_matched", ascending=False)
        print(df.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
        df.to_csv(DATA / f"length_control_{tag.lower()}.csv", index=False)


if __name__ == "__main__":
    main()
