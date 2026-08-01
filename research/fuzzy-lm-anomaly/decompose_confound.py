"""Stage 8b -- why did the FIS advantage vanish under template matching?

`template_control.py` shows the fuzzy rule losing badly (0.668 vs 0.965 for mean
entropy) once truthful and fabricated share a template. But that experiment
changed TWO things at once relative to §9:

  1. template matching (the intended manipulation), and
  2. the truthful distribution -- from TriviaQA-correct (broad, ~5,300 diverse
     questions) to template_real-correct (narrow, 632 answers over 5 curated
     fact types).

Either could explain the collapse, and they have opposite implications:

  * If the FIS is strong with a TriviaQA-truthful set and weak with a
    template-matched one, the §9 advantage was **template novelty** -- a
    confound, and the finding is dead.
  * If the FIS is weak in both, it is the **fit set** -- narrow and small
    (379 examples, 66 features) -- and the finding survives conditional on a
    broad known-good manifold.

This runs the 2x2 so the two causes separate. Fabrications are held fixed
(template_fake) in every cell; only the truthful side changes.
"""

import contextlib
import io
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from analyze import DATA, fpr_at_tpr
from nopca import POOLING, drop_constant, f_centroid, f_stats
from seed_sweep import fit_fis
from template_control import match

warnings.filterwarnings("ignore")

SEEDS = 6


def main():
    meta = pd.read_parquet(DATA / "capture_v2_meta.parquet")
    hidden = np.load(DATA / f"capture_v2_hidden_{POOLING}.npy")

    tq_ok = np.flatnonzero((meta.family == "triviaqa") & (meta.label == "correct"))
    tr_ok = np.flatnonzero((meta.family == "template_real")
                           & (meta.label == "correct"))
    fake = np.flatnonzero((meta.family == "template_fake")
                          & (meta.label == "hallucination"))

    # Cell definitions: (name, truthful pool, subsample truthful to this size?)
    CELLS = [
        ("triviaqa-truthful (broad, n=5286)", tq_ok, None),
        ("triviaqa-truthful subsampled to 632", tq_ok, len(tr_ok)),
        ("template-truthful (narrow, n=632)", tr_ok, None),
    ]
    rows = []

    for cname, pool, cap in CELLS:
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            p = pool.copy()
            rng.shuffle(p)
            if cap:
                p = p[:cap]
            cut = int(0.6 * len(p))
            sp = {"fit": p[:cut], "test_neg": p[cut:]}

            Fc = drop_constant(f_centroid(meta, hidden, sp).replace(
                [np.inf, -np.inf], np.nan).fillna(0.0), sp["fit"])
            Fs = f_stats(meta, hidden, sp)
            with contextlib.redirect_stdout(io.StringIO()):
                fis, feats, _ = fit_fis(Fc, sp, seed)
            ssc = StandardScaler().fit(Fs.iloc[sp["fit"]].to_numpy())
            lw = LedoitWolf().fit(ssc.transform(Fs.iloc[sp["fit"]].to_numpy()))

            dets = {
                "FIS · centroid": fis,
                "Mahalanobis · stats": lambda ix: lw.mahalanobis(
                    ssc.transform(Fs.iloc[ix].to_numpy())),
                "mean entropy": lambda ix: meta.iloc[ix].ent_mean.to_numpy(float),
                "n_tokens": lambda ix: meta.iloc[ix].n_tokens.to_numpy(float),
            }
            # length-matched throughout, so length is never the explanation
            a, b = match(meta, sp["test_neg"], fake, ["n_tokens"], rng)
            if len(a) < 30:
                continue
            ix = np.concatenate([a, b])
            y = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
            for nm, fn in dets.items():
                s = np.asarray(fn(ix), dtype=float)
                ok = np.isfinite(s).all() and np.ptp(s) > 0
                rows.append({"truthful_set": cname, "seed": seed, "detector": nm,
                             "auroc": roc_auc_score(y, s) if ok else np.nan,
                             "n_neg": len(a), "n_pos": len(b),
                             "n_fit": len(sp["fit"])})
        print(f"  {cname}: done")

    df = pd.DataFrame(rows)
    df.to_csv(DATA / "decompose_confound.csv", index=False)

    print(f"\n{'='*94}\nAUROC vs the SAME fabrications, varying only the truthful "
          f"set (length-matched, {SEEDS} seeds)\n{'='*94}")
    piv = df.pivot_table(index="detector", columns="truthful_set",
                         values="auroc", aggfunc=["mean", "std"])
    cols = [c for _, c, _ in
            [(0, n, 0) for n, _, _ in CELLS] if ("mean", c) in piv.columns]
    tbl = pd.DataFrame({c: piv[("mean", c)].map("{:.3f}".format) + " ± "
                        + piv[("std", c)].map("{:.3f}".format) for c in cols})
    print(tbl.to_string())
    print(f"\nmean fit-set size per cell:")
    print(df.groupby('truthful_set')['n_fit'].mean().round(0).to_string())

    print(f"\n{'-'*94}")
    fis = df[df.detector == "FIS · centroid"].groupby("truthful_set")["auroc"].mean()
    broad = fis.get("triviaqa-truthful (broad, n=5286)", np.nan)
    small = fis.get("triviaqa-truthful subsampled to 632", np.nan)
    narrow = fis.get("template-truthful (narrow, n=632)", np.nan)
    print("Interpretation:")
    print(f"  broad TriviaQA truthful        FIS {broad:.3f}")
    print(f"  same, subsampled to n=632      FIS {small:.3f}   "
          f"<- isolates fit-set SIZE")
    print(f"  template-matched truthful      FIS {narrow:.3f}   "
          f"<- isolates the TEMPLATE/distribution")
    if np.isfinite(broad) and np.isfinite(small) and np.isfinite(narrow):
        print(f"\n  cost of shrinking the fit set : {small - broad:+.3f}")
        print(f"  cost of matching the template : {narrow - small:+.3f}")


if __name__ == "__main__":
    main()
