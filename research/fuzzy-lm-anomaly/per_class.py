"""Stage 14 -- one detector per class of error, and the transfer matrix.

Under the open-set protocol the rule base is fit on truthful data only, so
nothing about it is class-specific. A "specialist" is therefore made by letting
each error class choose *which variables the rule watches*: antecedents are
selected on a validation slice of that class (supervised selection), while the
rule base itself is still fit on truthful data alone (open-set fitting).

The interesting output is not the diagonal, it is the **transfer matrix**:
specialist selected for class i, evaluated on class j. That distinguishes two
possibilities that the aggregate numbers cannot:

  * hallucination is ONE phenomenon  -> off-diagonal ~ diagonal
  * detectors learn prompt-family STYLE -> diagonal high, off-diagonal at chance

Section 12 already hinted at the second (a falsepremise-selected detector was at
chance on template_fake). This measures it across all seven classes at once.

Note on the confound: a single common truthful set (TriviaQA-correct) is used for
every class so the matrix has a comparable basis, which means template matching
is not possible here and the absolute numbers are inflated by prompt-family
style. That is deliberate -- the matrix is the instrument for *exposing* that
inflation, not a bid to avoid it. Length is matched throughout.
"""

import contextlib
import io
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tribblefis.gauss_math import create_gaussian_membership_dict
from analyze import DATA
from features_ext import build
from nopca import POOLING
from norm_sweep import anomaly_score, membership_tensors
from seed_sweep import K_MODES, PAIR, THETA
from template_control import match

warnings.filterwarnings("ignore")

SEEDS, TOP_N = 4, 8


def error_classes(meta):
    """Seven classes of error, kept only where there are enough samples."""
    out = {}
    for t in ("capital", "symbol", "novel", "currency", "film"):
        ix = np.flatnonzero((meta.family == "template_fake")
                            & (meta.template == t)
                            & (meta.label == "hallucination"))
        if len(ix) >= 120:
            out[f"fake:{t}"] = ix
    fp = np.flatnonzero((meta.family == "falsepremise")
                        & (meta.label == "hallucination"))
    if len(fp) >= 120:
        out["falsepremise"] = fp
    tq = np.flatnonzero((meta.family == "triviaqa")
                        & (meta.label == "hallucination"))
    if len(tq) >= 120:
        out["triviaqa_error"] = tq
    return out


def select(F, fit, val_neg, val_pos, top_n=TOP_N):
    """Rank variables by separation on a VALIDATION slice of one error class."""
    y = np.concatenate([np.zeros(len(val_neg)), np.ones(len(val_pos))])
    ix = np.concatenate([val_neg, val_pos])
    sc = {}
    for c in F.columns:
        v = F.iloc[ix][c].to_numpy(float)
        if np.ptp(v) == 0:
            continue
        au = roc_auc_score(y, v)
        sc[c] = max(au, 1 - au)
    return [c for c, _ in sorted(sc.items(), key=lambda t: -t[1])[:top_n]]


def fit_open_set(F, fit, feats, seed):
    """Rule base fit on truthful data only -- no error examples involved."""
    Xf = StandardScaler().fit_transform(F.iloc[fit][feats].to_numpy())
    labels = KMeans(n_clusters=K_MODES, n_init=10, random_state=seed).fit(Xf).labels_
    y_modes = pd.Series([f"mode{c}" for c in labels])
    with contextlib.redirect_stdout(io.StringIO()):
        model = create_gaussian_membership_dict(
            F.iloc[fit][feats].reset_index(drop=True), y_modes,
            top_n_var_names=feats)

    def score(ix):
        classes, tens = membership_tensors(F, ix, model, feats)
        with np.errstate(all="ignore"):
            return np.asarray(anomaly_score(classes, tens, *PAIR, THETA), float)
    return score, model


def main():
    meta = pd.read_parquet(DATA / "capture_v2_meta.parquet")
    H = np.load(DATA / f"capture_v2_hidden_{POOLING}.npy")
    classes = error_classes(meta)
    print("error classes:")
    for k, v in classes.items():
        print(f"  {k:<16} n={len(v):>6}")

    recs = []
    for seed in range(SEEDS):
        rng = np.random.default_rng(13000 + seed)
        truth = np.flatnonzero((meta.family == "triviaqa") & (meta.label == "correct"))
        rng.shuffle(truth)
        a, b = int(.5 * len(truth)), int(.7 * len(truth))
        fit, val_neg, test_neg = truth[:a], truth[a:b], truth[b:]

        F = pd.concat(build(meta, H, fit).values(), axis=1)
        F = F.loc[:, ~F.columns.duplicated()]

        # split each error class into a selection half and a test half
        halves = {}
        for cname, ix in classes.items():
            i = ix.copy()
            rng.shuffle(i)
            halves[cname] = (i[:len(i) // 2], i[len(i) // 2:])

        # one specialist per class, plus a generalist selected on all pooled
        selections = {c: select(F, fit, val_neg, halves[c][0]) for c in classes}
        pooled = np.concatenate([halves[c][0] for c in classes])
        rng.shuffle(pooled)
        selections["GENERALIST"] = select(F, fit, val_neg, pooled[:2000])

        for sname, feats in selections.items():
            score, model = fit_open_set(F, fit, feats, seed)
            for cname, (_, te_pos) in halves.items():
                na, nb = match(meta, test_neg, te_pos, ["n_tokens"], rng)
                if len(na) < 30:
                    continue
                ix = np.concatenate([na, nb])
                y = np.concatenate([np.zeros(len(na)), np.ones(len(nb))])
                s = score(ix)
                ok = np.isfinite(s).all() and np.ptp(s) > 0
                recs.append({"seed": seed, "specialist": sname,
                             "eval_class": cname,
                             "auroc": roc_auc_score(y, s) if ok else np.nan,
                             "n": len(ix), "mfs": model.n_membership_functions,
                             "feats": ",".join(feats)})
            # entropy reference on the same matched sets
            for cname, (_, te_pos) in halves.items():
                if sname != "GENERALIST":
                    continue
                na, nb = match(meta, test_neg, te_pos, ["n_tokens"], rng)
                if len(na) < 30:
                    continue
                ix = np.concatenate([na, nb])
                y = np.concatenate([np.zeros(len(na)), np.ones(len(nb))])
                v = meta.iloc[ix].ent_mean.to_numpy(float)
                recs.append({"seed": seed, "specialist": "mean entropy",
                             "eval_class": cname,
                             "auroc": roc_auc_score(y, v), "n": len(ix),
                             "mfs": 0, "feats": "ent_mean"})
        print(f"  seed {seed} done")

    df = pd.DataFrame(recs)
    df.to_csv(DATA / "per_class.csv", index=False)

    M = df.pivot_table(index="specialist", columns="eval_class", values="auroc")
    order = list(classes)
    M = M.reindex(index=[c for c in list(classes) + ["GENERALIST", "mean entropy"]
                         if c in M.index], columns=order)
    print(f"\n{'='*100}\nTRANSFER MATRIX — specialist (row) evaluated on class "
          f"(column), {SEEDS} seeds\n{'='*100}")
    print(M.to_string(float_format=lambda v: f"{v:.3f}"))

    spec = M.loc[[c for c in classes if c in M.index], order]
    diag = np.array([spec.loc[c, c] for c in spec.index if c in spec.columns])
    off = np.array([spec.loc[i, j] for i in spec.index for j in spec.columns
                    if i != j and np.isfinite(spec.loc[i, j])])
    print(f"\nFIS specialists   diagonal mean {np.nanmean(diag):.3f}   "
          f"off-diagonal mean {np.nanmean(off):.3f}   "
          f"transfer gap {np.nanmean(diag) - np.nanmean(off):+.3f}")
    if "GENERALIST" in M.index:
        print(f"FIS generalist    mean over classes {M.loc['GENERALIST'].mean():.3f}")
    if "mean entropy" in M.index:
        e = M.loc["mean entropy"]
        print(f"mean entropy      mean over classes {e.mean():.3f}   "
              f"(per class: {', '.join(f'{c}={e[c]:.2f}' for c in order)})")

    print(f"\nvariables each specialist chose:")
    for c in list(classes) + ["GENERALIST"]:
        f = df[df.specialist == c].feats.dropna()
        if len(f):
            top = f.str.split(",").explode().value_counts().index[:5]
            print(f"  {c:<16} {', '.join(top)}")


if __name__ == "__main__":
    main()
