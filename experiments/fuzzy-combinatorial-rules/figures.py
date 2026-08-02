"""Figures for the experiment.

    python figures.py [--dataset iris] [--k 5] [--seed 0]

Colour follows the repository's data-viz conventions: a single-hue sequential
ramp wherever the encoded quantity is ordered (the MFs of a partition are
ordered low->high, so they are *not* a categorical set), categorical hues only
where identity is the job, and identity never carried by colour alone -- every
marked line is directly labelled.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

import datasets as ds  # noqa: E402
import model as mdl  # noqa: E402
import selection as sel  # noqa: E402
from ruspini import UnitScaler, centers, fuzzify, labels  # noqa: E402

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b8b7b0"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e8f0fb", "#2a78d6", "#12365f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "grid.color": "#e6e5df",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def fig_partition(ks: tuple[int, ...] = (3, 5, 7)) -> None:
    """The Ruspini geometry itself: hats on a uniform grid, summing to one."""
    xs = np.linspace(0, 1, 1001)
    fig, axes = plt.subplots(1, len(ks), figsize=(3.1 * len(ks), 2.4), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), ks):
        m = fuzzify(xs[:, None], k)[:, 0, :]
        for j in range(k):
            ax.plot(xs, m[:, j], lw=2, color=SEQ(j / max(k - 1, 1)))
            ax.annotate(labels(k)[j], (centers(k)[j], 1.04), ha="center",
                        fontsize=7.5, color=INK_2, rotation=30)
        ax.plot(xs, m.sum(axis=1), lw=1.4, ls="--", color=INK_2)
        ax.annotate("sum = 1", (0.5, 1.0), xytext=(0.5, 0.86), ha="center",
                    fontsize=8, color=INK_2)
        ax.set_title(f"k = {k}", fontsize=10)
        ax.set_xlabel("normalised input")
        ax.set_ylim(0, 1.22)
        ax.grid(axis="y", lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("membership")
    fig.suptitle("Ruspini partitions: one free parameter, and the memberships sum to one",
                 fontsize=11, y=1.10)
    _save(fig, "01-ruspini-partition.png")


def fig_rule_masks(dataset: str, k: int, selector: str, seed: int) -> None:
    """Which MFs each rule claims -- and how often the rules claim the same one."""
    data = ds.load(dataset)
    xtr, _, ytr, _ = train_test_split(data.x, data.y, test_size=0.3,
                                      random_state=seed, stratify=data.y)
    fit = mdl.fit(xtr, ytr, k, selector, data.n_classes, seed=seed,
                  class_names=data.class_names)
    assert fit is not None
    s = fit.model.s
    n_c, d, _ = s.shape

    fig, axes = plt.subplots(1, n_c + 1, figsize=(2.3 * (n_c + 1), 0.42 * d + 1.9),
                             sharey=True)
    for c in range(n_c):
        ax = axes[c]
        ax.imshow(s[c].astype(float), cmap=SEQ, vmin=0, vmax=1.4, aspect="auto")
        for i in range(d):
            for j in range(k):
                if s[c, i, j]:
                    ax.text(j, i, "•", ha="center", va="center", color=SURFACE, fontsize=9)
        ax.set_title(f"R{c}: {data.class_names[c]}", fontsize=9.5)
        ax.set_xticks(range(k), labels(k), rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(d), data.feature_names, fontsize=7.5)
        ax.grid(False)

    overlap = s.sum(axis=0)  # how many rules use each (variable, MF)
    ax = axes[-1]
    im = ax.imshow(overlap, cmap=SEQ, vmin=0, vmax=n_c, aspect="auto")
    for i in range(d):
        for j in range(k):
            ax.text(j, i, str(int(overlap[i, j])), ha="center", va="center",
                    fontsize=8, color=SURFACE if overlap[i, j] > n_c / 2 else INK_2)
    ax.set_title("rules sharing this MF", fontsize=9.5)
    ax.set_xticks(range(k), labels(k), rotation=45, ha="right", fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.05, ticks=range(n_c + 1))
    fig.suptitle(
        f"{dataset}, k={k}, selector `{selector}` — antecedent subsets are not exclusive",
        fontsize=11, y=1.02)
    _save(fig, "02-rule-masks.png")


def fig_landscape(dataset: str, k: int, seed: int) -> None:
    """Every subset combination scored, with the heuristics placed in it.

    Plotted as a survival curve -- how many combinations score at least J --
    rather than a histogram, because the selectors all land in the extreme
    right tail where a histogram cannot separate them but rank can.
    """
    data = ds.load(dataset)
    xtr, _, ytr, _ = train_test_split(data.x, data.y, test_size=0.3,
                                      random_state=seed, stratify=data.y)
    d = int(xtr.shape[1])
    if not sel.exhaustive_feasible(d, k):
        print(f"landscape: {dataset} k={k} out of enumeration budget, skipped")
        return
    scaler = UnitScaler.fit(xtr)
    m = fuzzify(scaler.transform(xtr), k)

    fig, axes = plt.subplots(1, data.n_classes, figsize=(3.5 * data.n_classes, 3.2))
    for c in range(data.n_classes):
        ax = np.atleast_1d(axes)[c]
        prob = sel.Problem(m=m, in_class=(ytr == c), seed=seed * 1000 + c)
        scores = sel.all_subset_scores(prob)
        assert scores is not None
        ordered = np.sort(scores)[::-1]
        ax.plot(ordered, np.arange(1, ordered.size + 1), lw=2, color=MUTED)
        ax.set_yscale("log")

        marks = [("optimum", float(ordered[0]))]
        for name in ("greedy", "mst_core", "mst_mf", "mass"):
            probe = sel.Problem(m=m, in_class=(ytr == c), seed=seed * 1000 + c)
            got = sel.select(name, probe)
            assert got is not None
            marks.append((name, probe.score(got)))
        span = float(ordered[0] - ordered[-1]) or 1.0
        for idx, (name, val) in enumerate(marks):
            rank = int(np.sum(scores >= val - 1e-12))
            colour = BLUE if name == "optimum" else ORANGE
            note = f"optimum ({rank} tied)" if name == "optimum" else f"{name} — rank {rank}"
            ax.plot([val], [rank], "o", ms=8, color=colour, zorder=4,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            ax.annotate(
                note, (val, rank), xytext=(0.03, 0.90 - 0.075 * idx),
                textcoords="axes fraction", fontsize=7.5, color=colour,
                ha="left", va="center",
                arrowprops={"arrowstyle": "-", "color": colour, "lw": 0.7,
                            "alpha": 0.55, "shrinkA": 2, "shrinkB": 5},
            )
        ax.set_title(f"{data.class_names[c]} — {scores.size:,} combinations", fontsize=9.5)
        ax.set_xlabel("one-vs-rest margin J(S)")
        ax.set_xlim(float(ordered[-1]) - 0.04 * span, float(ordered[0]) + 0.10 * span)
        ax.set_ylim(0.6, ordered.size * 12)
        ax.grid(axis="y", lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("combinations scoring at least J (log)")
    fig.suptitle(f"{dataset}, k={k}: where the selectors land in the full enumeration",
                 fontsize=11, y=1.02)
    _save(fig, "03-landscape.png")


def fig_mst(dataset: str, k: int, seed: int) -> None:
    """The co-firing MST over membership functions, and the component it keeps."""
    from selection import _cofiring_distance, _mst_edges  # noqa: PLC0415

    data = ds.load(dataset)
    xtr, _, ytr, _ = train_test_split(data.x, data.y, test_size=0.3,
                                      random_state=seed, stratify=data.y)
    d = int(xtr.shape[1])
    scaler = UnitScaler.fit(xtr)
    m = fuzzify(scaler.transform(xtr), k)
    flat = m.reshape(m.shape[0], d * k)

    fig, axes = plt.subplots(1, data.n_classes, figsize=(3.3 * data.n_classes, 0.44 * d + 2.1))
    for c in range(data.n_classes):
        ax = np.atleast_1d(axes)[c]
        in_c = ytr == c
        mc = flat[in_c]
        prior = float(in_c.mean())
        total = flat.sum(axis=0)
        precision = np.where(total > 0, mc.sum(axis=0) / np.maximum(total, 1e-9), 0.0)
        benefit = (precision - prior) * mc.mean(axis=0)

        rows, cols, w = _mst_edges(_cofiring_distance(mc))
        prob = sel.Problem(m=m, in_class=in_c, seed=seed * 1000 + c)
        chosen = sel.select("mst_mf", prob)
        assert chosen is not None
        # A don't-care row means the variable was absent from the kept component.
        kept = chosen & (chosen.sum(axis=1) < k)[:, None]

        pos = np.array([[j, i] for i in range(d) for j in range(k)], dtype=float)
        for u, v, weight in zip(rows, cols, w):
            ax.plot(*zip(pos[u], pos[v]), lw=0.8,
                    color=MUTED, alpha=float(np.clip(1.15 - weight, 0.12, 1.0)))
        span = float(np.ptp(benefit)) or 1.0
        ax.scatter(pos[:, 0], pos[:, 1], s=52,
                   c=SEQ((benefit - benefit.min()) / span), zorder=3,
                   edgecolors=SURFACE, linewidths=1.2)
        sel_idx = np.flatnonzero(kept.reshape(-1))
        ax.scatter(pos[sel_idx, 0], pos[sel_idx, 1], s=160, facecolors="none",
                   edgecolors=AQUA, linewidths=2.0, zorder=4)
        ax.set_title(f"{data.class_names[c]}", fontsize=9.5)
        ax.set_xticks(range(k), labels(k), rotation=45, ha="right", fontsize=7.5)
        ax.set_yticks(range(d), data.feature_names if c == 0 else [""] * d, fontsize=7.5)
        ax.invert_yaxis()
        ax.grid(False)
    fig.suptitle(
        f"{dataset}, k={k}: MST over membership functions (edge = co-firing on the class); "
        f"ringed nodes are the kept component",
        fontsize=11, y=1.03)
    _save(fig, "04-mst-membership-graph.png")


def fig_convexity(tag: str = "main") -> None:
    """How often a freely-chosen antecedent is still one contiguous run."""
    import json  # noqa: PLC0415

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "outputs", f"results-{tag}.json")
    if not os.path.exists(path):
        print(f"convexity: {path} missing, skipped")
        return
    with open(path) as handle:
        run = json.load(handle)

    selectors = ["mass", "mst_mf", "mst_core", "greedy", "anneal"]
    names = list(run["datasets"])
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 2.7), sharey=True)
    for ax, dname in zip(np.atleast_1d(axes), names):
        ks = sorted(int(k) for k in run["datasets"][dname])
        ends: list[tuple[float, str, int]] = []
        for idx, name in enumerate(selectors):
            ys, xs = [], []
            for k in ks:
                got = run["datasets"][dname].get(str(k), {}).get(name)
                if got and got.get("convex_frac"):
                    xs.append(k)
                    ys.append(float(np.mean(got["convex_frac"])))
            if not xs:
                continue
            ax.plot(xs, ys, "-o", lw=2, ms=6, color=SEQ(0.25 + 0.75 * idx / (len(selectors) - 1)),
                    markeredgecolor=SURFACE, markeredgewidth=1.0, label=name)
            ends.append((ys[-1], name, idx))
        # Lines that finish on top of each other would otherwise stack their
        # labels; push each one up to a minimum separation.
        placed = 0.0
        for y_end, name, idx in sorted(ends):
            y_label = max(y_end, placed + 0.035) if placed else y_end
            placed = y_label
            ax.annotate(name, (ks[-1], y_end), xytext=(ks[-1] + 0.28, y_label),
                        textcoords="data", fontsize=7.5, color=INK_2, va="center",
                        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7}
                        if abs(y_label - y_end) > 1e-9 else None)
        ax.set_title(dname, fontsize=10)
        ax.set_xlabel("k (MFs per variable)")
        ax.set_xticks(ks)
        ax.set_xlim(min(ks) - 0.3, max(ks) + 1.6)
        ax.set_ylim(0.55, 1.03)
        ax.grid(axis="y", lw=0.6)
    np.atleast_1d(axes)[0].set_ylabel("fraction of antecedents contiguous")
    fig.suptitle("Free subset search drifts away from convex linguistic terms as k grows",
                 fontsize=11, y=1.03)
    _save(fig, "05-convexity.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="iris")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--selector", default="greedy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    fig_partition()
    fig_rule_masks(args.dataset, args.k, args.selector, args.seed)
    fig_landscape(args.dataset, 3, args.seed)
    fig_mst(args.dataset, args.k, args.seed)
    fig_convexity()


if __name__ == "__main__":
    main()
