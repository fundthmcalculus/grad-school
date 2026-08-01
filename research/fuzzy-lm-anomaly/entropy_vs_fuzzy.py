"""Stage 27 -- when does the fuzzy rule beat entropy?

§24/§26 left one live pattern. Across four models the fuzzy rule's advantage over
a mean-entropy threshold tracked how badly entropy was doing:

    gemma   entropy 0.553  ->  FIS +0.096
    qwen    entropy 0.735  ->  FIS -0.050
    lfm     entropy 0.816  ->  FIS -0.058
    smollm2 entropy 0.840  ->  FIS -0.004

Four points is an anecdote. This turns it into a measurement: **4 models x 13
templates = 52 cells**, each a self-contained detection problem, and asks whether
(FIS - entropy) is predicted by entropy's own AUROC in that cell.

Two design rules, both learned the hard way:

* **Equal budget (§26).** The FIS uses `fis_config`'s declared defaults with **no
  configuration search**. Entropy has no hyperparameters, so neither detector gets
  a supervised search. This is the like-for-like comparison.
* **Template is held constant within a cell**, so the §11 template confound cannot
  operate; length is matched inside each cell as well.

Also tests a zero-parameter fusion. If the two detectors are complementary, the
rank-average of their scores should beat both — and because it fits nothing, it
costs no search budget and stays budget-fair.
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tribblefis.gauss_math import (calculate_gaussian_correlation,
                                   take_top_features, tsk_firing_strengths)
import fis_config as CFG
from analyze import DATA, SCALAR_COLS
from template_control import match

warnings.filterwarnings("ignore")

SEEDS, TOP_N, K = 6, 6, 4
MODELS = ["smollm2", "qwen", "gemma", "lfm"]
MIN_CELL = 15


def fit_global_fis(F, fit, seed):
    """One rule base per model, fitted on ALL its grounded data. Fixed config."""
    sub = F.iloc[fit]
    feats0 = list(sub.var().sort_values(ascending=False).index[:TOP_N])
    Xf = StandardScaler().fit_transform(sub[feats0].to_numpy())
    labels = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(Xf).labels_
    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        diff = calculate_gaussian_correlation(sub.reset_index(drop=True), y_modes,
                                              method=CFG.METRIC)
        _, feats = take_top_features(diff, top_n=TOP_N)
        model = CFG.build_memberships(sub[feats].reset_index(drop=True), y_modes,
                                      feats, membership=CFG.MEMBERSHIP)
    return feats, model


def fis_score(F, feats, model, ix):
    fs, lab = tsk_firing_strengths(F.iloc[ix][feats].reset_index(drop=True),
                                   model, CFG.anomaly_params())
    return np.asarray(fs[:, lab.index("anomaly")], float)


def rank01(x):
    return sstats.rankdata(x) / (len(x) + 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    rows = []

    for m in MODELS:
        meta = pd.read_parquet(DATA / f"capture_v4_{m}_meta.parquet")
        F = meta[SCALAR_COLS].reset_index(drop=True).astype(float)
        good = np.flatnonzero((meta.family == "longform_real")
                             & (meta.label == "grounded"))
        bad = np.flatnonzero((meta.family == "longform_fake")
                            & (meta.label == "hallucination"))

        for seed in range(args.seeds):
            rng = np.random.default_rng(37000 + seed)
            g = good.copy()
            rng.shuffle(g)
            cut = int(.6 * len(g))
            fit, test_neg = g[:cut], g[cut:]
            feats, model = fit_global_fis(F, fit, seed)

            for tmpl in sorted(meta.template.unique()):
                neg = test_neg[meta.iloc[test_neg].template.to_numpy() == tmpl]
                pos = bad[meta.iloc[bad].template.to_numpy() == tmpl]
                if len(neg) < MIN_CELL or len(pos) < MIN_CELL:
                    continue
                a, b = match(meta, neg, pos, ["n_tokens"], rng)
                if len(a) < MIN_CELL:
                    continue
                ix = np.concatenate([a, b])
                y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])

                s_ent = meta.iloc[ix].ent_mean.to_numpy(float)
                s_fis = fis_score(F, feats, model, ix)
                if not np.isfinite(s_fis).all() or np.ptp(s_fis) == 0:
                    continue
                # zero-parameter fusion: average of within-cell ranks
                s_fuse = rank01(s_ent) + rank01(s_fis)

                rows.append({
                    "model": m, "template": tmpl, "seed": seed, "n": len(ix),
                    "entropy": roc_auc_score(y, s_ent),
                    "fis": roc_auc_score(y, s_fis),
                    "fusion": roc_auc_score(y, s_fuse)})
        print(f"  {m} done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "entropy_vs_fuzzy.csv", index=False)
    df["d_fis"] = df.fis - df.entropy
    df["d_fuse_best"] = df.fusion - df[["entropy", "fis"]].max(axis=1)

    cell = df.groupby(["model", "template"]).agg(
        entropy=("entropy", "mean"), fis=("fis", "mean"),
        fusion=("fusion", "mean"), n=("n", "mean")).reset_index()
    cell["d_fis"] = cell.fis - cell.entropy

    print(f"\n{'='*88}\n{len(cell)} CELLS (model x template), fixed configuration, "
          f"{args.seeds} seeds\n{'='*88}")

    r, p = sstats.pearsonr(cell.entropy, cell.d_fis)
    rs, ps = sstats.spearmanr(cell.entropy, cell.d_fis)
    print(f"\nHYPOTHESIS: the fuzzy rule helps more where entropy is weaker.")
    print(f"  corr(entropy AUROC, FIS − entropy):")
    print(f"    Pearson  r = {r:+.3f}  (p = {p:.2e})")
    print(f"    Spearman r = {rs:+.3f}  (p = {ps:.2e})")
    lo, hi = cell.entropy.median(), cell.entropy.median()
    weak, strong = cell[cell.entropy <= lo], cell[cell.entropy > hi]
    print(f"\n  entropy WEAK  cells (AUROC <= {lo:.3f}, n={len(weak)}): "
          f"FIS − entropy = {weak.d_fis.mean():+.3f}")
    print(f"  entropy STRONG cells (AUROC >  {hi:.3f}, n={len(strong)}): "
          f"FIS − entropy = {strong.d_fis.mean():+.3f}")
    if len(weak) > 2 and len(strong) > 2:
        u, pu = sstats.mannwhitneyu(weak.d_fis, strong.d_fis)
        print(f"  Mann-Whitney p = {pu:.4f}")

    # crossover: below what entropy AUROC does the FIS win?
    fitline = np.polyfit(cell.entropy, cell.d_fis, 1)
    cross = -fitline[1] / fitline[0] if fitline[0] != 0 else np.nan
    print(f"\n  linear fit: (FIS − entropy) = {fitline[0]:+.3f}·entropy "
          f"{fitline[1]:+.3f}")
    print(f"  crossover: the fuzzy rule is ahead when entropy AUROC < {cross:.3f}")
    print(f"  cells where FIS beats entropy: "
          f"{int((cell.d_fis > 0).sum())}/{len(cell)}")

    print(f"\n{'='*88}\nPER MODEL\n{'='*88}")
    pm = df.groupby("model").agg(entropy=("entropy", "mean"), fis=("fis", "mean"),
                                 fusion=("fusion", "mean"),
                                 d_fis=("d_fis", "mean")).sort_values("entropy")
    print(pm.to_string(float_format=lambda v: f"{v:.3f}"))

    print(f"\n{'='*88}\nZERO-PARAMETER RANK FUSION (entropy + FIS)\n{'='*88}")
    print(f"  mean over cells: entropy {cell.entropy.mean():.3f}  "
          f"FIS {cell.fis.mean():.3f}  fusion {cell.fusion.mean():.3f}")
    d = df.groupby(["model", "template"])["d_fuse_best"].mean()
    print(f"  fusion − best single detector: {d.mean():+.3f} ± {d.std():.3f}, "
          f"beats both in {int((d > 0).sum())}/{len(d)} cells")
    dd = (cell.fusion - cell.entropy)
    pw = sstats.wilcoxon(dd)[1] if len(dd) >= 6 else np.nan
    print(f"  fusion − entropy: {dd.mean():+.3f}, wins {int((dd>0).sum())}/{len(dd)}"
          + (f", p = {pw:.2e}" if np.isfinite(pw) else ""))

    print(f"\n{'='*88}\nWEAKEST-ENTROPY CELLS\n{'='*88}")
    print(cell.nsmallest(10, "entropy")[
        ["model", "template", "n", "entropy", "fis", "fusion", "d_fis"]]
        .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
