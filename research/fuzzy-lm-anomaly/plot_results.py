"""Comparison figures for the fuzzy-LM anomaly study.

comparison.png
  Panel A -- detector comparison on the raw test split, both probe families,
             with the FIS shown both with PCA and PCA-free.
  Panel B -- the length control. `n_tokens` alone scores 0.853 on false-premise,
             so panel A's raw numbers are partly confounded. Exact matching on
             n_tokens removes it; what survives is not length.

layers.png
  Validation AUROC by layer and pooling site (where the signal lives).

Panels A and B report different things on purpose and are labelled as such;
panel B is the one to trust for the false-premise family.

Palette: categorical slots 1/2/7 of the reference palette, six checks passing
under `validate_palette.js --mode light` (blue-orange ΔE 24.7 protan). Panel B
uses semantic roles rather than five categorical hues: blue = the subject,
orange = the confound control, muted grey = reference baselines.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e5e4e0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#4a3aa7"

mpl.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": GRID,
    "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False,
})

RENAME = {
    "max-softmax (1-maxp)": "Max-softmax", "mean entropy": "Mean entropy",
    "perplexity": "Perplexity", "margin (neg)": "Margin",
    "Mahalanobis (hidden)": "Mahalanobis · hidden",
    "OneClassSVM (hidden)": "One-class SVM · hidden",
    "IsolationForest (hidden)": "Isolation forest · hidden",
    "Mahalanobis (stats)": "Mahalanobis · stats",
    "IsolationForest (stats)": "Isolation forest · stats",
    "Mahalanobis (fused)": "Mahalanobis · fused",
}
FIS_PCA = "tribble FIS · with PCA"
FIS_FREE = "tribble FIS · PCA-free"
KEYS = [("triviaqa", "TriviaQA"), ("falsepremise", "FalsePremise")]


def load_panel_a():
    frames = {}
    nop = pd.read_csv(DATA / "nopca_results.csv")
    fis = pd.read_csv(DATA / "fis_results.csv")
    ns = pd.read_csv(DATA / "norm_sweep.csv")
    for key, tag in KEYS:
        b = pd.read_csv(DATA / f"baselines_{key}.csv")[["detector", "auroc"]]
        b["detector"] = b["detector"].map(lambda d: RENAME.get(d, d))
        pca = float(np.nanmax([fis.loc[fis.family == tag, "auroc"].max(),
                               ns.loc[ns.family == tag, "auroc"].max()]))
        free = float(nop.loc[(nop.family == tag)
                             & nop.detector.str.startswith("FIS"), "auroc"].max())
        frames[key] = pd.concat([b, pd.DataFrame(
            [{"detector": FIS_PCA, "auroc": pca},
             {"detector": FIS_FREE, "auroc": free}])], ignore_index=True)
    return frames


def panel_a(ax, frames):
    m = frames["triviaqa"].merge(frames["falsepremise"], on="detector",
                                 how="outer", suffixes=("_tq", "_fp"))
    m = m.sort_values("auroc_fp", ascending=True).reset_index(drop=True)
    y = np.arange(len(m))
    h = 0.36
    ax.barh(y + h / 2 + 0.012, m.auroc_fp, height=h, color=S2,
            label="False-premise (novel open-set)", zorder=3)
    ax.barh(y - h / 2 - 0.012, m.auroc_tq, height=h, color=S1,
            label="TriviaQA (in-distribution)", zorder=3)

    for yi, (vt, vf) in enumerate(zip(m.auroc_tq, m.auroc_fp)):
        for v, off in ((vt, -h / 2 - 0.012), (vf, h / 2 + 0.012)):
            if np.isfinite(v):
                ax.text(v + 0.006, yi + off, f"{v:.3f}", va="center", ha="left",
                        fontsize=7.2, color=INK_2, zorder=4)

    ax.axvline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(0.5, len(m) - 0.3, " chance", fontsize=7.5, color=INK_MUTED,
            ha="left", va="center")

    ax.set_yticks(y, list(m.detector))
    for t in ax.get_yticklabels():
        if t.get_text() in (FIS_PCA, FIS_FREE):
            t.set_color(INK)
            t.set_fontweight("bold")

    ax.set_xlim(0.45, 0.94)
    ax.set_ylim(-0.75, len(m) - 0.1)
    ax.set_xlabel("AUROC — hallucination vs truthful (raw test split)\n"
                  "→ higher is better  ·  0.5 = chance")
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("A · Detector comparison — raw split", loc="left",
                 fontsize=10.5, color=INK, fontweight="bold", pad=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right", handlelength=1.1,
              borderaxespad=0.8)


def panel_b(ax):
    d = pd.read_csv(DATA / "length_control_falsepremise.csv")
    style = {
        "FIS · centroid (PCA-free)": (S1, 2.4, 1.0, "tribble FIS · centroid\n(PCA-free)"),
        "n_tokens (confound probe)": (S2, 2.0, 1.0, "n_tokens\n(confound probe)"),
        "Mahalanobis · stats": (INK_MUTED, 1.6, 0.9, "Mahalanobis · stats"),
        "perplexity": (INK_MUTED, 1.6, 0.9, "Perplexity"),
        "mean entropy": (INK_MUTED, 1.6, 0.9, "Mean entropy"),
    }
    # the CSV round-trips '·' through cp1252 on Windows; match on a prefix
    def find(row_key):
        for v in d.detector:
            if v.split()[0] == row_key.split()[0]:
                return v
        return None

    x = [0, 1]
    for key, (color, lw, alpha, lab) in style.items():
        r = d[d.detector.str.startswith(key.split()[0])]
        if r.empty:
            continue
        r = r.iloc[0]
        ys = [r.auroc_raw, r.auroc_matched]
        ax.plot(x, ys, color=color, lw=lw, alpha=alpha, marker="o", ms=6.5,
                mec=SURFACE, mew=1.4, zorder=3,
                solid_capstyle="round")
        ax.text(1.055, ys[1], f"{lab}  {ys[1]:.3f}", color=color, fontsize=7.8,
                va="center", ha="left", linespacing=1.3,
                fontweight="bold" if color is S1 else "normal")
        ax.text(-0.055, ys[0], f"{ys[0]:.3f}", color=color, fontsize=7.6,
                va="center", ha="right")

    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(-0.055, 0.5, "chance", fontsize=7.5, color=INK_MUTED, va="center",
            ha="right")

    ax.set_xticks(x, ["raw split\n(353 vs 716)", "length-matched\n(170 vs 170)"])
    ax.set_xlim(-0.42, 1.72)
    ax.set_ylim(0.47, 0.93)
    ax.set_ylabel("AUROC — false-premise family\n↑ higher is better")
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.set_title("B · Controlling for answer length", loc="left", fontsize=10.5,
                 color=INK, fontweight="bold", pad=8)


def fig_layers():
    sw = pd.read_csv(DATA / "representation_sweep.csv")
    g = sw.groupby(["pooling", "layer"])["val_tq"].max().reset_index()
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for pool, (c, lab) in {"prompt": (S1, "Prompt state"),
                           "first": (S2, "First token"),
                           "mean": (S3, "Mean over answer")}.items():
        dd = g[g.pooling == pool].sort_values("layer")
        ax.plot(dd.layer, dd.val_tq, color=c, lw=2, marker="o", ms=3.4,
                mec=SURFACE, mew=0.8, zorder=3, label=lab)
    ax.axhline(0.5, color=INK_MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.text(32.6, 0.5, "chance", fontsize=7.5, color=INK_MUTED, va="center")
    ax.axvline(20, color=INK_MUTED, lw=0.9, ls=(0, (2, 3)), zorder=2)
    ax.annotate("layer 20 selected", xy=(20, 0.595), xytext=(11.5, 0.606),
                fontsize=7.6, color=INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9))
    ax.set_xlabel("Residual-stream layer (0 = embeddings, 32 = final)")
    ax.set_ylabel("AUROC — validation split\n↑ higher is better")
    ax.set_xlim(-1, 36.5)
    ax.set_ylim(0.330, 0.620)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 32])
    ax.yaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.legend(frameon=False, fontsize=7.8, ncol=3, handlelength=1.1,
              columnspacing=1.4, loc="lower center",
              bbox_to_anchor=(0.5, -0.015), borderaxespad=0.0)
    ax.set_title("Where the signal lives — TriviaQA, validation split",
                 loc="left", fontsize=10.5, color=INK, fontweight="bold", pad=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"layers.{ext}", dpi=200)


def main():
    OUT.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(13.4, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.42, 1], wspace=0.16,
                          left=0.145, right=0.795, top=0.80, bottom=0.165)
    panel_a(fig.add_subplot(gs[0, 0]), load_panel_a())
    panel_b(fig.add_subplot(gs[0, 1]))

    fig.suptitle("Fuzzy anomaly detection on a frozen SmolLM2-360M-Instruct",
                 x=0.145, y=0.965, ha="left", fontsize=12.5, color=INK,
                 fontweight="bold")
    fig.text(0.145, 0.915,
             "Fit on accurate (question, answer) pairs only — no hallucination "
             "seen during fitting. Answer length alone scores 0.853 on the raw\n"
             "false-premise split, so it must be controlled: once it is, the "
             "fuzzy rule leads and the confidence baselines collapse to chance.",
             ha="left", va="top", fontsize=8.6, color=INK_2, linespacing=1.45)

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"comparison.{ext}", dpi=200)
    fig_layers()
    print(f"wrote {OUT}/comparison.[png|pdf] and layers.[png|pdf]")


if __name__ == "__main__":
    main()
