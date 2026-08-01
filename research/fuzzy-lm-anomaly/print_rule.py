"""Stage 7 -- print the rule base, and use theta as the operating point.

The whole point of routing this through a fuzzy inference system rather than a
one-class SVM is that the result is legible. The SVM that scores 0.789 on the
identical 8 features cannot be written down; this can:

    RULE 1  IF L26_dist is LOW  AND L26_cos is HIGH ... THEN behaviour is normal
    ANOMALY IF no rule above fires THEN flag the output

Two deliverables:

1. The rule base in linguistic form, with each Gaussian membership function
   named LOW / MEDIUM / HIGH by where its mean sits in the fit-split
   distribution of that feature, plus a membership-function figure.
2. The theta operating-point table. §3.4 proved theta cannot change ranking, so
   the honest way to report it is as a *deployment* knob: warn on the top X% of
   outputs, and report the precision and recall that buys.
"""

import argparse
import contextlib
import io
import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

mpl.use("Agg")

from analyze import DATA, make_splits
from nopca import POOLING, f_centroid
from norm_sweep import anomaly_score, membership_tensors
from seed_sweep import PAIR, THETA, fit_fis

warnings.filterwarnings("ignore")

OUT = Path(__file__).parent / "figures"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e5e4e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#4a3aa7"


def term_for(mu, q):
    """Name a membership function by where its centre sits in the feature's
    fit-split quantiles: q = (q10, q30, q70, q90)."""
    if mu <= q[0]:
        return "VERY LOW"
    if mu <= q[1]:
        return "LOW"
    if mu <= q[2]:
        return "MEDIUM"
    if mu <= q[3]:
        return "HIGH"
    return "VERY HIGH"


def describe(model, F, fit_idx, feats):
    """Emit the rule base in linguistic form."""
    classes = list(next(iter(model.feature_models.values())).ordered_keys)
    qs = {f: np.quantile(F.iloc[fit_idx][f].to_numpy(), [.10, .30, .70, .90])
          for f in feats}

    print(f"\n{'='*78}\nRULE BASE — {len(classes)} normal-behaviour rules + 1 anomaly rule"
          f"\n{'='*78}")
    print(f"antecedents ({len(feats)}): {', '.join(feats)}")
    print(f"membership functions: {model.n_membership_functions}")
    print("\nFeature meaning: L{n}_dist = Euclidean distance of the layer-n "
          "residual state to the\ncentroid of truthful behaviour; L{n}_cos = "
          "cosine similarity to that same centroid.\nBoth are computed against "
          "the fit split only, so no hallucination informs them.\n")

    for ci, c in enumerate(classes, 1):
        print(f"RULE {ci}  ({c})")
        print("  IF", end="")
        parts = []
        for f in feats:
            fm = model.feature_models[f]
            if c not in fm.label_models:
                continue
            terms = [term_for(mf.mu, qs[f]) for mf in fm.label_models[c].memberships]
            uniq = sorted(set(terms), key=terms.index)
            parts.append(f"{f} is {' or '.join(uniq)}")
        print(("\n  AND ".join(f" {p}" for p in parts)))
        print("  THEN the model is behaving normally\n")

    T, S = PAIR
    print(f"ANOMALY RULE  (Ch 4.3.5 'none of the above')")
    print(f"  mu_anom = 1 - S(mu_1 + theta, ..., mu_K + theta)")
    print(f"  with T = {T}, S = {S}, theta = {THETA}")
    print("  IF none of the rules above fires strongly")
    print("  THEN flag the output as suspect — no known behaviour matched\n")


def plot_mfs(model, F, fit_idx, feats, path):
    """One panel per antecedent: the fitted MFs over the fit-split histogram."""
    n = len(feats)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.35 * nrow),
                             facecolor=SURFACE)
    axes = np.atleast_1d(axes).ravel()
    classes = list(next(iter(model.feature_models.values())).ordered_keys)
    colors = [S1, S2, S3, "#1baf7a"]

    for ax, f in zip(axes, feats):
        col = F.iloc[fit_idx][f].to_numpy()
        lo, hi = np.percentile(col, [0.5, 99.5])
        xs = np.linspace(lo, hi, 400)
        ax.hist(col, bins=40, range=(lo, hi), density=True, color=GRID,
                zorder=1)
        fm = model.feature_models[f]
        for ci, c in enumerate(classes):
            if c not in fm.label_models:
                continue
            for mf in fm.label_models[c].memberships:
                ys = mf.evaluate(xs)
                scale = ax.get_ylim()[1] * 0.92
                ax.plot(xs, ys * scale, color=colors[ci % len(colors)], lw=1.8,
                        zorder=3, label=c if mf is fm.label_models[c].memberships[0]
                        else None)
        ax.set_title(f, fontsize=8.5, color=INK, loc="left")
        ax.set_yticks([])
        ax.tick_params(labelsize=7, colors=INK_2, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_facecolor(SURFACE)
    for ax in axes[n:]:
        ax.axis("off")

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8, ncol=len(l), loc="lower center",
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Fitted membership functions over the known-good distribution",
                 x=0.02, ha="left", fontsize=11, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, facecolor=SURFACE)
    print(f"wrote {path.with_suffix('.png')}")


def operating_points(score_fn, sp, pos_key, rates=(0.01, 0.02, 0.05, 0.10, 0.20)):
    """What a deployment gets for warning on the top X% of outputs."""
    neg, pos = sp["test_neg"], sp[pos_key]
    ix = np.concatenate([neg, pos])
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.asarray(score_fn(ix), dtype=float)

    rows = []
    for r in rates:
        thr = np.quantile(s, 1 - r)
        flag = s >= thr
        tp = int((flag & (y == 1)).sum())
        fp = int((flag & (y == 0)).sum())
        rows.append({
            "warn_rate": f"{r:.0%}",
            "n_flagged": int(flag.sum()),
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(int((y == 1).sum()), 1),
            "false_alarms": fp,
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    meta = pd.read_parquet(DATA / "capture_meta.parquet")
    hidden = np.load(DATA / f"capture_hidden_{POOLING}.npy")
    sp = make_splits(meta, seed=args.seed)
    F = f_centroid(meta, hidden, sp).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    with contextlib.redirect_stdout(io.StringIO()):
        score, feats, model = fit_fis(F, sp, args.seed)

    describe(model, F, sp["fit"], feats)
    OUT.mkdir(exist_ok=True)
    plot_mfs(model, F, sp["fit"], feats, OUT / "membership_functions")

    print(f"\n{'='*78}\nOPERATING POINTS — theta as a deployment knob\n{'='*78}")
    print("theta cannot change AUROC (§3.4); it selects where on the curve you "
          "sit.\nEquivalently: warn on the top X% of outputs by anomaly "
          "membership.\n")
    for tag, pk in [("false-premise", "test_fp"), ("TriviaQA", "test_tq")]:
        df = operating_points(score, sp, pk)
        base = len(sp[pk]) / (len(sp[pk]) + len(sp["test_neg"]))
        print(f"--- {tag} (base rate {base:.1%}) ---")
        print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        df.to_csv(DATA / f"operating_points_{tag.replace('-', '')}.csv", index=False)
        print()


if __name__ == "__main__":
    main()
