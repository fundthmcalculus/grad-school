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


def _main():
    OUT.mkdir(exist_ok=True)
    fig_negative()
    fig_transfer()
    fig_falsepremise()
    fig_fuzzy_stats()
    fig_models()
    fig_entropy_regime()


def fig_falsepremise():
    """The false-premise niche under stacked controls."""
    d = pd.read_csv(DATA / "falsepremise_deep.csv")
    agg = d.groupby(["detector", "condition"])["auroc"].agg(["mean", "std"])
    order = ["raw", "length", "length+template", "length+template+entropy"]
    style = [("Mahalanobis · stats", INK_MUTED, 1.7, "Mahalanobis · stats", 0.010),
             ("mean entropy", S3, 1.7, "Mean entropy", -0.012),
             ("FIS · agg", S1, 2.4, "tribble FIS · agg (best family)", 0.012),
             ("FIS · deltaref", S1, 1.3, "FIS · deltaref", -0.014),
             ("n_tokens (control)", S2, 1.7, "n_tokens (control)", 0.0)]
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(order))
    for key, c, lw, lab, dy in style:
        if key not in agg.index.get_level_values(0):
            continue
        ys = [agg.loc[(key, o), "mean"] for o in order]
        es = [agg.loc[(key, o), "std"] for o in order]
        ax.errorbar(x, ys, yerr=es, color=c, lw=lw, marker="o", ms=5.5,
                    mec=SURFACE, mew=1.3, capsize=3, elinewidth=1.0, zorder=3,
                    alpha=1.0 if lw > 1.5 else 0.75)
        ax.text(x[-1] + 0.08, ys[-1] + dy, f"{lab}\n{ys[-1]:.3f}", color=c,
                fontsize=7.4, va="center", ha="left", linespacing=1.3,
                fontweight="bold" if lw > 2 else "normal")
    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.1, 0.507, "chance", fontsize=7.4, color=INK_MUTED, va="bottom")
    ax.set_xticks(x, ["raw", "+ length\nmatched", "+ template\nmatched",
                      "+ entropy\nmatched"])
    ax.set_xlim(-0.4, 4.55)
    ax.set_ylim(0.44, 0.94)
    ax.set_ylabel("AUROC  (↑ higher is better)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    fig.suptitle("The false-premise niche does not survive its own control",
                 x=0.012, y=0.975, ha="left", fontsize=12, color=INK,
                 fontweight="bold")
    fig.text(0.012, 0.905,
             "Long-form probes with real-subject twins in identical surface "
             "forms, so both sides are fluent and discursive.\nEvery fuzzy family "
             "stays well below the output-distribution baselines once template is "
             "matched.",
             ha="left", va="top", fontsize=8.4, color=INK_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"falsepremise_control.{ext}", dpi=200)
    print(f"wrote {OUT / 'falsepremise_control.png'}")





def fig_fuzzy_stats():
    """The positive result: fuzzy rule over the output statistics."""
    d = pd.read_csv(DATA / "fuzzy_stats.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), facecolor=SURFACE)
    conds = ["length+template", "length+template+entropy"]
    titles = ["A · Template + length matched", "B · + entropy matched (hardest)"]

    for ax, cond, title in zip(axes, conds, titles):
        s = d[d.condition == cond]
        g = (s.groupby("detector")
             .agg(auroc=("auroc", "mean"), std=("auroc", "std"),
                  params=("n_params", "mean")))
        # Pareto front: maximise AUROC, minimise parameters
        front = [n for n, r in g.iterrows()
                 if not any((o.auroc >= r.auroc) and (o.params <= r.params)
                            and (o.auroc > r.auroc or o.params < r.params)
                            for on, o in g.iterrows() if on != n)]
        for n, r in g.iterrows():
            is_fis = n.startswith("FIS")
            on_front = n in front
            c = S1 if is_fis else (S2 if "n_tokens" in n else INK_MUTED)
            x = max(r.params, 0.6)
            ax.errorbar(x, r.auroc, yerr=r["std"], color=c, marker="o",
                        ms=11 if is_fis else 7, mec=INK if on_front else SURFACE,
                        mew=1.6 if on_front else 1.2, capsize=2.5,
                        elinewidth=1.0, zorder=4, linestyle="none")
            ax.annotate(f"{n.split(' · ')[0]}\n{r.auroc:.3f}", (x, r.auroc),
                        textcoords="offset points", xytext=(10, -2),
                        fontsize=7.2, color=c, linespacing=1.25,
                        fontweight="bold" if is_fis else "normal")
        fr = g.loc[front].sort_values("params")
        ax.plot(np.maximum(fr.params, 0.6), fr.auroc, color=INK, lw=1,
                ls=(0, (5, 3)), zorder=2, alpha=0.55)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(-0.4, 5e5)
        ax.set_ylim(0.45, 0.96)
        ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xlabel("Tunable parameters  (← fewer is better)", fontsize=8.5)
        ax.set_ylabel("AUROC  (↑ higher is better)", fontsize=8.5)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.set_facecolor(SURFACE)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(labelsize=7.5, colors=INK_2, length=0)
        ax.set_title(title, loc="left", fontsize=10.2, color=INK,
                     fontweight="bold", pad=8)

    fig.suptitle("A small fuzzy rule over the OUTPUT statistics is on the "
                 "Pareto front", x=0.012, y=0.975, ha="left", fontsize=12.5,
                 color=INK, fontweight="bold")
    fig.text(0.012, 0.905,
             "Same 19 output-distribution statistics for every learned detector. "
             "Dashed line = Pareto front (ringed markers). The fuzzy rule uses "
             "4 rules over 6 antecedents\n(80 parameters) and beats "
             "full-covariance Mahalanobis by +0.040 on 8/8 seeds (p = 0.0078) "
             "with 2.6x fewer parameters.",
             ha="left", va="top", fontsize=8.4, color=INK_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fuzzy_stats_pareto.{ext}", dpi=200)
    print(f"wrote {OUT / 'fuzzy_stats_pareto.png'}")





def fig_models():
    """Cross-model / cross-task replication (section 24)."""
    runs = [("v3 long-form\nSmolLM2-360M", "fuzzy_stats.csv"),
            ("v3 long-form\nQwen2.5-0.5B", "fuzzy_stats_capture_v3_qwen.csv"),
            ("v2 short-factual\nSmolLM2-360M", "fuzzy_stats_capture_v2.csv")]
    dets = [("FIS · stats", S1, "tribble FIS · stats"),
            ("Mahalanobis · stats", INK_MUTED, "Mahalanobis · stats"),
            ("mean entropy", S3, "Mean entropy")]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # --- A: grouped bars, condition length+template ----------------------
    ax = axes[0]
    x = np.arange(len(runs))
    w = 0.26
    for i, (key, c, lab) in enumerate(dets):
        ys, es = [], []
        for _, f in runs:
            d = pd.read_csv(DATA / f)
            s = d[(d.condition == "length+template")
                  & (d.detector.str.contains(key.split(" · ")[0], regex=False))]
            ys.append(s.auroc.mean())
            es.append(s.auroc.std())
        off = (i - 1) * w
        ax.bar(x + off, ys, width=w * 0.9, yerr=es, color=c, zorder=3, label=lab,
               error_kw=dict(elinewidth=1, capsize=2.5, ecolor=INK_MUTED))
        for xi, v in zip(x + off, ys):
            ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", fontsize=6.8,
                    color=INK_2, zorder=4)
    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.set_xticks(x, [r[0] for r in runs], fontsize=7.8)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("AUROC  (↑ higher is better)")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=7.8, loc="lower left", handlelength=1.0)
    ax.set_title("A · Detector accuracy, template + length matched", loc="left",
                 fontsize=10.2, color=INK, fontweight="bold", pad=8)

    # --- B: paired advantage, the actual claim ---------------------------
    ax = axes[1]
    from scipy import stats as ss
    labels, deltas, errs, colors, notes = [], [], [], [], []
    for name, f in runs:
        d = pd.read_csv(DATA / f)
        m = d[d.condition == "length+template"].pivot_table(
            index="seed", columns="detector", values="auroc")
        fis = [c for c in m.columns if c.startswith("FIS")][0]
        for rival, col in (("Mahalanobis", INK_MUTED), ("mean entropy", S3)):
            rc = [c for c in m.columns if rival.split()[0] in c][0]
            dd = (m[fis] - m[rc]).dropna()
            p = ss.wilcoxon(dd)[1] if len(dd) >= 6 else np.nan
            labels.append(f"{name.splitlines()[1]}\nvs {rival}")
            deltas.append(dd.mean())
            errs.append(dd.std())
            colors.append(S1 if p < 0.05 and dd.mean() > 0 else
                          (S2 if dd.mean() < 0 else INK_MUTED))
            notes.append(f"{int((dd>0).sum())}/{len(dd)}  p={p:.3f}")
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, deltas, xerr=errs, height=0.6, color=colors, zorder=3,
            error_kw=dict(elinewidth=1, capsize=2.5, ecolor=INK_MUTED))
    # clear the error-bar whisker, not just the bar end
    for y_, v, e, n in zip(yy, deltas, errs, notes):
        off = (e + 0.004) if v >= 0 else -(e + 0.004)
        ax.text(v + off, y_, n, va="center",
                ha="left" if v >= 0 else "right", fontsize=6.8, color=INK_2)
    ax.axvline(0, color=INK, lw=1, zorder=4)
    ax.set_yticks(yy, labels, fontsize=7.2)
    ax.set_xlim(-0.075, 0.088)
    ax.set_xlabel("paired Δ AUROC vs rival  (→ fuzzy rule better)", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("B · Paired advantage (blue = p < 0.05)", loc="left",
                 fontsize=10.2, color=INK, fontweight="bold", pad=8)

    fig.suptitle("Cross-model replication: the Mahalanobis result holds, the "
                 "entropy result does not", x=0.008, y=0.975, ha="left",
                 fontsize=12.2, color=INK, fontweight="bold")
    fig.text(0.008, 0.906,
             "Same 19 output statistics throughout, so the comparison is "
             "architecture-independent. The fuzzy rule beats full-covariance "
             "Mahalanobis on both models\n(+0.036 and +0.032, both 8/8 seeds, "
             "p = 0.008) but beats mean entropy only on SmolLM2 long-form — it "
             "ties on Qwen and loses on the short-factual task.",
             ha="left", va="top", fontsize=8.3, color=INK_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"model_comparison.{ext}", dpi=200)
    print(f"wrote {OUT / 'model_comparison.png'}")





def fig_entropy_regime():
    """Where the fuzzy rule earns its keep: the entropy-weakness regime."""
    from scipy import stats as ss
    d = pd.read_csv(DATA / "entropy_vs_fuzzy.csv")
    d["d_fis"] = d.fis - d.entropy
    c = (d.groupby(["model", "template"])
         .agg(entropy=("entropy", "mean"), fis=("fis", "mean"),
              d_fis=("d_fis", "mean"), n=("n", "mean")).reset_index())

    cols = {"gemma": S1, "smollm2": S2, "qwen": S3, "lfm": INK_MUTED}
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    for m, g in c.groupby("model"):
        ax.scatter(g.entropy, g.d_fis, s=34, color=cols[m], label=m,
                   edgecolor=SURFACE, linewidth=0.9, zorder=4)
    k, b = np.polyfit(c.entropy, c.d_fis, 1)
    xs = np.linspace(c.entropy.min() - .02, c.entropy.max() + .02, 50)
    ax.plot(xs, k * xs + b, color=INK, lw=1.4, ls=(0, (5, 3)), zorder=3)
    cross = -b / k
    ax.axhline(0, color=INK_MUTED, lw=1, zorder=2)
    ax.axvline(cross, color=INK_MUTED, lw=1, ls=(0, (2, 3)), zorder=2)
    ax.annotate(f"crossover\nentropy = {cross:.2f}", (cross, c.d_fis.max()),
                textcoords="offset points", xytext=(6, -6), fontsize=7.4,
                color=INK_2, linespacing=1.3)
    r, p = ss.pearsonr(c.entropy, c.d_fis)
    ax.text(0.98, 0.96, f"r = {r:+.3f}\np = {p:.1e}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.4, color=INK, linespacing=1.4)
    ax.set_xlabel("entropy AUROC in that cell  (→ entropy doing better)",
                  fontsize=8.5)
    ax.set_ylabel("FIS − entropy  (↑ fuzzy rule better)", fontsize=8.5)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=INK_2, length=0)
    ax.legend(frameon=False, fontsize=7.6, loc="lower left", handlelength=0.9)
    ax.set_title("A · The fuzzy rule wins exactly where entropy fails",
                 loc="left", fontsize=10.2, color=INK, fontweight="bold", pad=8)

    ax = axes[1]
    lo = c[c.entropy < cross].sort_values("d_fis", ascending=True)
    yy = np.arange(len(lo))
    ax.barh(yy, lo.d_fis, height=0.62, color=[cols[m] for m in lo.model], zorder=3)
    for y_, v, e in zip(yy, lo.d_fis, lo.entropy):
        ax.text(v + 0.006, y_, f"entropy {e:.2f}", va="center", fontsize=6.8,
                color=INK_2)
    ax.axvline(0, color=INK, lw=1, zorder=4)
    ax.set_yticks(yy, [f"{m} · {t}" for m, t in zip(lo.model, lo.template)],
                  fontsize=7.2)
    ax.set_xlim(-0.09, 0.36)
    ax.set_xlabel("FIS − entropy", fontsize=8.5)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    ax.tick_params(labelsize=7.5, colors=INK_2, length=0)
    ax.set_title(f"B · Cells below the crossover (entropy < {cross:.2f})",
                 loc="left", fontsize=10.2, color=INK, fontweight="bold", pad=8)

    fig.suptitle("The fuzzy anomaly rule covers entropy's blind spot",
                 x=0.008, y=0.975, ha="left", fontsize=12.2, color=INK,
                 fontweight="bold")
    fig.text(0.008, 0.905,
             "44 cells (4 models x 13 templates), fixed configuration for both "
             "detectors so neither gets a search budget. Template is constant "
             "within a cell and length is\nmatched. The relationship survives a "
             "split-half check for the shared-term artefact (r = −0.78 / −0.72 on "
             "disjoint seeds), and corr(entropy, FIS) is only +0.22.",
             ha="left", va="top", fontsize=8.3, color=INK_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"entropy_regime.{ext}", dpi=200)
    print(f"wrote {OUT / 'entropy_regime.png'}")


if __name__ == "__main__":
    _main()
