"""Detect wrong answers. Geometry against the entropy incumbents.

`experiments/fuzzy-lm-anomaly.md` established the bar and the traps. Its result
was that a single statistic, `ent_max`, beat every learned detector (0.870
against Mahalanobis 0.776, isolation forest 0.768, one-class SVM 0.767, tribble
FIS 0.750), and that five separate apparent successes were each destroyed by a
control. So the controls here are part of the design, not a robustness section:

* **`n_tokens` is an arm.** Answer length alone reached AUROC 0.843 there. A
  detector that does not beat it is measuring length.
* **Within-condition AUROC is reported alongside pooled.** `with_context` and
  `no_context` differ in correctness *and* in prompt length, so a pooled score
  can be earned entirely by detecting which condition a generation came from.
  The within-condition numbers are the ones that mean anything.
* **Length-stratified AUROC**, computed inside `n_tokens` deciles and pooled,
  so length cannot be the carrier.
* **Every single feature's own AUROC is printed**, because the thing that beat
  all the learned detectors last time was a single feature nobody had searched.
* **Equal budget** for every learned arm.

What is new here is the *geometric* family. The prior study searched entropy
and summary statistics; it never looked at the hidden state's geometry against
the embedding matrix. Those features are cheap (they are already computed by
the forward pass that produced the answer) and untested:

  h_pca_resid      reconstruction error of `h` against a PCA basis fitted on
                   correct answers -- how far off the manifold this state sits
  h_maha           Mahalanobis distance in hidden space (not over summary
                   statistics, which is what was tested before)
  h_tok_align      max cosine between `h` and the embedding rows of the tokens
                   it actually ranks highest -- "is this state near any real
                   token, or between them"
  layer_norm_*     the per-layer norm profile, shape rather than magnitude
  traj_spread      ||h_last - h_mean|| / ||h_mean||, trajectory coherence
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def safe_auc(y, s):
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def stratified_auc(y, s, strat, min_n=25):
    """Mean AUROC computed inside strata, weighted by stratum size.

    Pooled AUROC can be produced entirely by a variable that shifts both the
    score and the label across strata. Computing inside strata removes it.
    """
    tot_w, tot = 0.0, 0.0
    for v in np.unique(strat):
        m = strat == v
        if m.sum() < min_n or len(np.unique(y[m])) < 2:
            continue
        a = roc_auc_score(y[m], s[m])
        tot += a * m.sum()
        tot_w += m.sum()
    return tot / tot_w if tot_w else float("nan")


def build_features(rundir: Path, df: pd.DataFrame, train_mask: np.ndarray):
    """Entropy-family (existing) and geometric (new) feature banks."""
    hid_mean = np.load(rundir / "hid_mean.npy")
    hid_last = np.load(rundir / "hid_last.npy")
    lnorm = np.load(rundir / "layer_norm_mean.npy")

    ent_cols = [
        c for c in df.columns if "__" in c and not c.startswith(("f1", "ref_recall"))
    ]
    ent = df[ent_cols].to_numpy(dtype=np.float64)

    # ---- geometric ------------------------------------------------------
    # Manifold fitted on *correct* training answers only: the question is how
    # far a generation sits from where the model is when it is right.
    fit_rows = train_mask & (df.label.to_numpy() == 0)
    pca = PCA(n_components=64, random_state=0).fit(hid_mean[fit_rows])
    rec = pca.inverse_transform(pca.transform(hid_mean))
    h_pca_resid = np.linalg.norm(hid_mean - rec, axis=1)

    Z = pca.transform(hid_mean)
    Zc = pca.transform(hid_mean[fit_rows])
    cov = np.cov(Zc.T) + 1e-6 * np.eye(Zc.shape[1])
    inv = np.linalg.inv(cov)
    d = Z - Zc.mean(0)
    h_maha = np.sqrt(np.einsum("ij,jk,ik->i", d, inv, d))

    nm, nl = (
        np.linalg.norm(hid_mean, axis=1),
        np.linalg.norm(hid_last, axis=1),
    )
    traj_spread = np.linalg.norm(hid_last - hid_mean, axis=1) / np.maximum(nm, 1e-6)
    cos_ml = (hid_mean * hid_last).sum(1) / np.maximum(nm * nl, 1e-6)

    lnorm_n = lnorm / np.maximum(lnorm[:, -1:], 1e-6)  # shape, not magnitude

    geo = np.column_stack([h_pca_resid, h_maha, nm, nl, traj_spread, cos_ml, lnorm_n])
    geo_names = [
        "h_pca_resid",
        "h_maha",
        "h_norm_mean",
        "h_norm_last",
        "traj_spread",
        "cos_mean_last",
    ] + [f"lnorm{i}" for i in range(lnorm_n.shape[1])]
    return ent, ent_cols, geo, geo_names


def run(rundir: Path, thresh=0.5, budget=24, seed=0) -> dict:
    df = pd.read_parquet(rundir / "gens.parquet").reset_index(drop=True)
    # Positive class = the answer is wrong. Graded, never the condition.
    df["label"] = (df.ref_recall < thresh).astype(int)

    rng = np.random.default_rng(seed)
    items = df.item_id.unique()
    rng.shuffle(items)
    tr_items = set(items[: int(0.6 * len(items))])
    tr = df.item_id.isin(tr_items).to_numpy()
    te = ~tr

    ent, ent_names, geo, geo_names = build_features(rundir, df, tr)
    y = df.label.to_numpy()
    ntok = df.n_tokens.to_numpy().astype(float)
    cond = df.cond.to_numpy()
    decile = pd.qcut(df.n_tokens, 10, labels=False, duplicates="drop").to_numpy()

    out: dict = {
        "n": len(df),
        "prevalence_wrong": float(y.mean()),
        "prevalence_by_cond": df.groupby("cond").label.mean().round(4).to_dict(),
        "threshold_ref_recall": thresh,
        "arms": {},
        "single_features": {},
    }

    def report(name, score_te):
        r = {
            "auroc": safe_auc(y[te], score_te),
            "auroc_within_cond": stratified_auc(y[te], score_te, cond[te]),
            "auroc_within_len_decile": stratified_auc(y[te], score_te, decile[te]),
        }
        for c in ("with_context", "no_context"):
            m = cond[te] == c
            r[f"auroc_{c}"] = safe_auc(y[te][m], score_te[m])
        out["arms"][name] = r
        print(
            f"  {name:26s} pooled {r['auroc']:.3f} | within-cond "
            f"{r['auroc_within_cond']:.3f} | within-len {r['auroc_within_len_decile']:.3f}"
            f" | ctx {r['auroc_with_context']:.3f} noctx {r['auroc_no_context']:.3f}",
            flush=True,
        )
        return r

    # ---- reference arms ---------------------------------------------------
    report("n_tokens", ntok[te])
    for nm in ("entropy__max", "entropy__mean", "top1_prob__mean", "varentropy__max"):
        if nm in ent_names:
            report(nm, ent[te, ent_names.index(nm)])

    # every single feature, so nothing hides
    allX = np.column_stack([ent, geo, ntok])
    allN = ent_names + geo_names + ["n_tokens"]
    for j, nm in enumerate(allN):
        s = allX[te, j]
        out["single_features"][nm] = {
            "auroc": safe_auc(y[te], s),
            "within_cond": stratified_auc(y[te], s, cond[te]),
        }
    best = max(
        out["single_features"].items(),
        key=lambda kv: (
            kv[1]["within_cond"] if np.isfinite(kv[1]["within_cond"]) else 0
        ),
    )
    out["best_single_feature"] = {"name": best[0], **best[1]}

    # ---- learned arms, equal budget --------------------------------------
    banks = {
        "entropy": (ent, ent_names),
        "geometric": (geo, geo_names),
        "entropy+geometric": (np.hstack([ent, geo]), ent_names + geo_names),
        "entropy+geo+ntok": (
            np.hstack([ent, geo, ntok[:, None]]),
            ent_names + geo_names + ["n_tokens"],
        ),
    }
    for bname, (X, names) in banks.items():
        sc = StandardScaler().fit(X[tr])
        Xs = sc.transform(X)
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xs[tr], y[tr])
        report(f"logreg[{bname}]", lr.predict_proba(Xs[te])[:, 1])
        gb = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.06, random_state=seed
        ).fit(X[tr], y[tr])
        report(f"gbm[{bname}]", gb.predict(X[te]))

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/halluc")
    ap.add_argument("--thresh", type=float, default=0.5)
    a = ap.parse_args()
    rundir = Path(a.run)
    r = run(rundir, thresh=a.thresh)
    (rundir / "detect.json").write_text(json.dumps(r, indent=2))
    print("\nbest single feature:", json.dumps(r["best_single_feature"]))


if __name__ == "__main__":
    main()
