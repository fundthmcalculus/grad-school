"""Stage 10 -- which feature-ranking statistic should pick the antecedents?

`tribblefis.calculate_gaussian_correlation` computes four separation statistics
per feature -- Bhattacharyya coefficient, Jensen-Shannon distance, overlap
coefficient, and histogram correlation -- then hard-codes a blend:
`mean(arithmetic_mean, geometric_mean)` over all four. None is individually
selectable, and three are computed from *fitted Gaussians* while the fourth is
computed from *empirical histograms*, so the blend mixes parametric and
non-parametric evidence and is not interpretable as any single divergence.

This reimplements each statistic separately, adds four non-parametric
alternatives, and scores them the only way that matters here: use each to pick
the top-k antecedents, fit the FIS, and measure detection AUROC.

  parametric (Gaussian-fit, as in the library)
    bhattacharyya   1 - sum sqrt(p q)
    jensen_shannon  JS distance between the fitted pdfs
    overlap         1 - sum min(p, q)
    blend           the library's current combination

  non-parametric (empirical, no Gaussian assumption)
    hist_corr       1 - corr(hist_a, hist_b)      (the library's fourth term)
    ks              Kolmogorov-Smirnov statistic
    wasserstein      Earth-mover distance, scaled by the pooled std
    auc             |2*AUC - 1|, i.e. rank separation
    mutual_info     mutual information with the mode label

Reported on the template-matched v2 task, which §8b establishes as the honest
one.
"""

import contextlib
import io
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as sstats
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tribblefis.gauss_math import create_gaussian_membership_dict
from analyze import DATA
from nopca import POOLING, drop_constant, f_centroid
from norm_sweep import anomaly_score, membership_tensors
from seed_sweep import K_MODES, PAIR, THETA
from template_control import match, splits_v2

warnings.filterwarnings("ignore")

TOP_N, SEEDS = 8, 6


# --------------------------------------------------------------------------
# ranking statistics: each takes (values, binary mode mask) -> separation score
# --------------------------------------------------------------------------

def _pdfs(a, b, lo, hi, n=100):
    xs = np.linspace(lo, hi, n)
    pa = sstats.norm.pdf(xs, *sstats.norm.fit(a))
    pb = sstats.norm.pdf(xs, *sstats.norm.fit(b))
    sa, sb = pa.sum(), pb.sum()
    if sa <= 0 or sb <= 0:
        return None, None
    return pa / sa, pb / sb


def s_bhattacharyya(a, b, lo, hi):
    p, q = _pdfs(a, b, lo, hi)
    return np.nan if p is None else 1 - np.sum(np.sqrt(p * q))


def s_jensen_shannon(a, b, lo, hi):
    p, q = _pdfs(a, b, lo, hi)
    return np.nan if p is None else jensenshannon(p, q)


def s_overlap(a, b, lo, hi):
    p, q = _pdfs(a, b, lo, hi)
    return np.nan if p is None else 1 - np.sum(np.minimum(p, q))


def s_hist_corr(a, b, lo, hi):
    if not np.isfinite([lo, hi]).all() or hi - lo <= 0:
        return np.nan
    ha, _ = np.histogram(a, bins=100, range=(lo, hi), density=True)
    hb, _ = np.histogram(b, bins=100, range=(lo, hi), density=True)
    if ha.std() == 0 or hb.std() == 0:
        return np.nan
    return 1 - np.corrcoef(ha, hb)[0, 1]


def s_ks(a, b, lo, hi):
    return sstats.ks_2samp(a, b).statistic


def s_wasserstein(a, b, lo, hi):
    pooled = np.concatenate([a, b]).std()
    return np.nan if pooled == 0 else sstats.wasserstein_distance(a, b) / pooled


def s_auc(a, b, lo, hi):
    y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    v = np.concatenate([a, b])
    return abs(2 * roc_auc_score(y, v) - 1)


def s_blend(a, b, lo, hi):
    """The library's current behaviour, reimplemented for comparison."""
    vals = [s_bhattacharyya(a, b, lo, hi), s_jensen_shannon(a, b, lo, hi),
            s_overlap(a, b, lo, hi), s_hist_corr(a, b, lo, hi)]
    v = np.array([x for x in vals if np.isfinite(x)])
    if not len(v):
        return np.nan
    am = v.mean()
    gm = np.prod(np.clip(v, 0, None)) ** (1 / len(v))
    return (am + gm) / 2


PAIRWISE = {
    "bhattacharyya": s_bhattacharyya, "jensen_shannon": s_jensen_shannon,
    "overlap": s_overlap, "blend (library)": s_blend,
    "hist_corr": s_hist_corr, "ks": s_ks, "wasserstein": s_wasserstein,
    "auc": s_auc,
}


def rank(F, fit_idx, labels, method, top_n=TOP_N):
    sub = F.iloc[fit_idx]
    if method == "mutual_info":
        mi = mutual_info_classif(sub.to_numpy(), labels, random_state=0)
        return list(pd.Series(mi, index=sub.columns)
                    .sort_values(ascending=False).index[:top_n])
    if method == "variance":
        return list(sub.var().sort_values(ascending=False).index[:top_n])

    fn = PAIRWISE[method]
    classes = np.unique(labels)
    scores = {}
    for col in sub.columns:
        v = sub[col].to_numpy()
        lo, hi = v.min(), v.max()
        tot, n = 0.0, 0
        for i in range(len(classes)):
            for j in range(i + 1, len(classes)):
                a, b = v[labels == classes[i]], v[labels == classes[j]]
                if len(a) < 2 or len(b) < 2:
                    continue
                s = fn(a, b, lo, hi)
                if np.isfinite(s):
                    tot += s
                    n += 1
        scores[col] = tot / n if n else -np.inf
    return list(pd.Series(scores).sort_values(ascending=False).index[:top_n])


def main():
    meta = pd.read_parquet(DATA / "capture_v2_meta.parquet")
    hidden = np.load(DATA / f"capture_v2_hidden_{POOLING}.npy")
    methods = list(PAIRWISE) + ["mutual_info", "variance"]
    rows = []

    for seed in range(SEEDS):
        sp = splits_v2(meta, seed)
        rng = np.random.default_rng(7000 + seed)
        F = drop_constant(f_centroid(meta, hidden, sp).replace(
            [np.inf, -np.inf], np.nan).fillna(0.0), sp["fit"])

        Xf = StandardScaler().fit_transform(
            F.iloc[sp["fit"]][list(F.var().sort_values(ascending=False)
                                   .index[:TOP_N])].to_numpy())
        labels = KMeans(n_clusters=K_MODES, n_init=10,
                        random_state=seed).fit(Xf).labels_
        y_modes = pd.Series([f"mode{c}" for c in labels])

        a, b = match(meta, sp["test_neg"], sp["test_pos"],
                     ["template", "n_tokens"], rng)
        if len(a) < 30:
            continue
        ix = np.concatenate([a, b])
        y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])

        for m in methods:
            feats = rank(F, sp["fit"], labels, m)
            with contextlib.redirect_stdout(io.StringIO()):
                model = create_gaussian_membership_dict(
                    F.iloc[sp["fit"]][feats].reset_index(drop=True), y_modes,
                    top_n_var_names=feats)
            classes, tens = membership_tensors(F, ix, model, feats)
            with np.errstate(all="ignore"):
                s = np.asarray(anomaly_score(classes, tens, *PAIR, THETA),
                               dtype=float)
            ok = np.isfinite(s).all() and np.ptp(s) > 0
            rows.append({"seed": seed, "ranker": m,
                         "auroc": roc_auc_score(y, s) if ok else np.nan,
                         "n_layers_selected": len({f[:3] for f in feats}),
                         "features": ",".join(feats)})
        print(f"  seed {seed} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "ranker_compare.csv", index=False)

    g = df.groupby("ranker")["auroc"].agg(["mean", "std", "count"])
    g = g.sort_values("mean", ascending=False)
    print(f"\n{'='*80}\nFEATURE RANKER vs detection AUROC "
          f"(template-matched, {SEEDS} seeds, top-{TOP_N})\n{'='*80}")
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))

    lib = g.loc["blend (library)", "mean"] if "blend (library)" in g.index else np.nan
    print(f"\nlibrary blend: {lib:.3f}")
    best = g.index[0]
    print(f"best ranker  : {best} ({g.iloc[0]['mean']:.3f}, "
          f"{g.iloc[0]['mean'] - lib:+.3f} vs the blend)")
    print("\nmost-selected features per ranker:")
    for m in g.index:
        sel = df[df.ranker == m].features.str.split(",").explode().value_counts()
        print(f"  {m:<16} {', '.join(sel.index[:5])}")


if __name__ == "__main__":
    main()
