"""Figures for the negative results and the per-class transfer matrix.

negative_results.png
  A -- the template control: raw -> length-matched -> length+template-matched.
       The FIS falls; the confidence baselines rise to near-ceiling.
  B -- the decomposition: the same fabrications against three truthful sets,
       isolating the template from the fit-set size.

transfer_matrix.png
  Specialist (row) x evaluation class (column). Diverging palette centred on
  0.5 = chance, because the polarity (above/below chance) is the point --
  several off-diagonal cells fall BELOW chance.

Same design tokens as `plot_results.py`.
"""

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
mpl.use("Agg")

from analyze import DATA

OUT = Path(__file__).parent / "figures"
SURFACE, INK, INK_2, INK_MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e5e4e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#4a3aa7"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
})


def panel_template(ax):
    d = pd.read_csv(DATA / "template_control.csv")
    agg = d.groupby(["detector", "condition"])["auroc"].agg(["mean", "std"])
    order = ["raw", "length", "length+template"]
    style = [("centroid", S1, 2.4, "tribble FIS · centroid", 0.0),
             ("entropy", INK_MUTED, 1.6, "Mean entropy", 0.022),
             ("Mahalanobis", INK_MUTED, 1.6, "Mahalanobis · stats", -0.020),
             ("perplexity", INK_MUTED, 1.6, "Perplexity", -0.055),
             ("n_tokens", S2, 1.8, "n_tokens (control)", 0.0)]
    x = np.arange(len(order))
    for key, c, lw, lab, dy in style:
        rows = [i for i in agg.index.get_level_values(0).unique() if key in i]
        if not rows:
            continue
        r = rows[0]
        ys = [agg.loc[(r, o), "mean"] for o in order]
        es = [agg.loc[(r, o), "std"] for o in order]
        ax.errorbar(x, ys, yerr=es, color=c, lw=lw, marker="o", ms=5.5,
                    mec=SURFACE, mew=1.3, capsize=3, elinewidth=1.0, zorder=3)
        ax.text(x[-1] + 0.09, ys[-1] + dy, f"{lab}\n{ys[-1]:.3f}", color=c,
                fontsize=7.4, va="center", ha="left", linespacing=1.3,
                fontweight="bold" if c == S1 else "normal")
    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.08, 0.507, "chance", fontsize=7.4, color=INK_MUTED, va="bottom")
    ax.set_xticks(x, ["raw", "length\nmatched", "length +\ntemplate matched"])
    ax.set_xlim(-0.35, 2.95)
    ax.set_ylim(0.44, 1.02)
    ax.set_ylabel("AUROC  (↑ higher is better)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("A · Adding controls reverses the ranking", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold", pad=8)


def panel_decompose(ax):
    d = pd.read_csv(DATA / "decompose_confound.csv")
    agg = d.groupby(["detector", "truthful_set"])["auroc"].agg(["mean", "std"])
    sets = ["triviaqa-truthful (broad, n=5286)",
            "triviaqa-truthful subsampled to 632",
            "template-truthful (narrow, n=632)"]
    labels = ["TriviaQA truthful\n(broad, n=5286)", "same, subsampled\nto n=632",
              "template-matched\ntruthful (n=632)"]
    # Four series need four distinguishable swatches: two greys would make the
    # legend ambiguous. Blue/violet/orange are the validated categorical slots;
    # grey stays for the one reference series.
    dets = [("FIS · centroid", S1, "tribble FIS · centroid"),
            ("mean entropy", S3, "Mean entropy"),
            ("Mahalanobis · stats", INK_MUTED, "Mahalanobis · stats"),
            ("n_tokens", S2, "n_tokens (control)")]
    x = np.arange(len(sets))
    w = 0.19
    for i, (key, c, lab) in enumerate(dets):
        rows = [r for r in agg.index.get_level_values(0).unique() if
                key.split(" · ")[0].split()[0] in r]
        if not rows:
            continue
        r = rows[0]
        ys = [agg.loc[(r, s), "mean"] if (r, s) in agg.index else np.nan
              for s in sets]
        es = [agg.loc[(r, s), "std"] if (r, s) in agg.index else np.nan
              for s in sets]
        off = (i - (len(dets) - 1) / 2) * w
        ax.bar(x + off, ys, width=w * 0.9, yerr=es, color=c, zorder=3,
               error_kw=dict(elinewidth=1, capsize=2.5, ecolor=INK_MUTED),
               label=lab)
        for xi, v in zip(x + off, ys):
            if np.isfinite(v):
                ax.text(xi, v + 0.018, f"{v:.2f}", ha="center", fontsize=6.6,
                        color=INK_2, zorder=4)
    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.set_xticks(x, labels, fontsize=7.6)
    ax.set_ylim(0.35, 1.03)
    ax.set_ylabel("AUROC  (↑ higher is better)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=7.4, ncol=2, loc="upper left",
              handlelength=1.0, columnspacing=1.0)
    ax.set_title("B · Same fabrications, different truthful set", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold", pad=8)


def fig_negative():
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.18], wspace=0.30,
                          left=0.075, right=0.80, top=0.775, bottom=0.145)
    panel_template(fig.add_subplot(gs[0, 0]))
    panel_decompose(fig.add_subplot(gs[0, 1]))
    fig.suptitle("Negative results — the fuzzy rule's advantage was a confound",
                 x=0.075, y=0.955, ha="left", fontsize=12.5, color=INK,
                 fontweight="bold")
    fig.text(0.075, 0.895,
             "Left: each control added moves the fuzzy rule down and the "
             "confidence baselines up. Right: against the SAME fabrications, the "
             "fuzzy rule is at chance\nwith a broad truthful set — so the earlier "
             "0.906 was prompt-family style, not fabrication. Shrinking the fit "
             "set costs only −0.028.",
             ha="left", va="top", fontsize=8.5, color=INK_2, linespacing=1.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"negative_results.{ext}", dpi=200)
    print(f"wrote {OUT / 'negative_results.png'}")


def fig_transfer():
    d = pd.read_csv(DATA / "per_class.csv")
    M = d.pivot_table(index="specialist", columns="eval_class", values="auroc")
    cols = [c for c in ["fake:capital", "fake:symbol", "fake:novel",
                        "fake:currency", "fake:film", "falsepremise",
                        "triviaqa_error"] if c in M.columns]
    rows = [r for r in cols + ["GENERALIST", "mean entropy"] if r in M.index]
    M = M.reindex(index=rows, columns=cols)

    # diverging: polarity about chance is the point (cells fall below 0.5)
    cmap = LinearSegmentedColormap.from_list(
        "div", [S2, "#f2f1ec", S1])
    norm = TwoSlopeNorm(vcenter=0.5, vmin=0.30, vmax=1.0)

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    im = ax.imshow(M.to_numpy(), cmap=cmap, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iloc[i, j]
            if not np.isfinite(v):
                continue
            own = M.index[i] == M.columns[j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=INK, fontweight="bold" if own else "normal")
            if own:
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                           edgecolor=INK, lw=1.8, zorder=5))
    ax.set_xticks(range(len(cols)), cols, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    for k, lab in enumerate(ax.get_yticklabels()):
        if rows[k] in ("GENERALIST", "mean entropy"):
            lab.set_style("italic")
    ax.set_xlabel("evaluated on this class of error", fontsize=8.5)
    ax.set_ylabel("detector specialised for", fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("AUROC  (0.5 = chance)", fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=0)

    spec = M.loc[[r for r in rows if r in cols], cols]
    diag = np.array([spec.loc[c, c] for c in spec.index])
    off = np.array([spec.loc[i, j] for i in spec.index for j in spec.columns
                    if i != j])
    fig.suptitle("Per-class specialists do not transfer", x=0.008, y=0.975,
                 ha="left", fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.008, 0.915,
             f"Boxed diagonal = specialist on its own class (mean "
             f"{np.nanmean(diag):.3f}).  Off-diagonal mean {np.nanmean(off):.3f} "
             f"— a transfer gap of {np.nanmean(diag)-np.nanmean(off):+.3f}.\n"
             f"Several off-diagonal cells fall BELOW chance, so a detector tuned "
             f"for one kind of fabrication can be actively misleading on another.",
             ha="left", va="top", fontsize=8.4, color=INK_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"transfer_matrix.{ext}", dpi=200)
    print(f"wrote {OUT / 'transfer_matrix.png'}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_negative()
    fig_transfer()
