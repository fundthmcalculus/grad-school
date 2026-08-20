"""What the fitted rule bases actually say, drawn so they can be read.

The reason to prefer a fuzzy rule base to a black box is that you can read it afterwards. That
only holds if someone actually looks, so this draws the trained `EFFORT` and `CHAIN` bases in the
two halves they consist of:

* **the membership functions** — where each input's LOW / MED / HIGH sit after fitting. This is
  what decides *when* a rule fires, and it is the half usually left at its defaults. Centres and
  widths were both fitted, so a term can migrate or narrow, and a narrow term is a rule that
  applies to a sliver of the input range.
* **the consequents** — a grid of rule against output, which is the rule base as a table. The
  hand-written values are shown beside the fitted ones and the difference between them, because
  the interesting question is not what the fitted base says but *where the optimiser disagreed
  with the reasoning it started from*.

Reading the difference panel: red means the optimiser wants more effort than was written, blue
means less. A row that is uniformly one colour is a rule whose whole premise was mis-set; a row
that is mixed is one where the optimiser kept the trigger but redistributed effort across the
four LK parameters, which is a subtler and more interesting kind of disagreement.

Run:  python figures_fis.py [--tuned tuned_small.npz] [--scale small]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import fis  # noqa: E402
import paths  # noqa: E402

TERM = ("LOW", "MED", "HIGH")
TERM_COLOUR = ("tab:blue", "tab:orange", "tab:green")
E_OUT = ("breadth", "deep breadth", "depth", "Or-opt seg")


def _rule_labels(ant, names):
    """One readable sentence per rule, from its antecedent row."""
    out = []
    for r in range(ant.shape[0]):
        parts = [
            f"{names[i]} {TERM[ant[r, i]]}"
            for i in range(ant.shape[1])
            if ant[r, i] >= 0
        ]
        out.append(" & ".join(parts) if parts else "(always)")
    return out


def _draw_mf(ax, tab, names, title):
    """Membership functions, one input per row, offset vertically."""
    xs = np.linspace(0.0, 1.0, tab.shape[2])
    offs = np.arange(len(names))[::-1]
    for i, off in enumerate(offs):
        ax.axhline(off, color="0.9", lw=0.8, zorder=0)
        for t in range(tab.shape[1]):
            ax.plot(
                xs,
                off + 0.85 * tab[i, t],
                color=TERM_COLOUR[t],
                lw=1.8,
                label=TERM[t] if i == 0 else None,
            )
        ax.text(0.012, off + 0.60, names[i], fontsize=8.5, color="black")
    ax.set_yticks(offs)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.15, len(names) - 0.05)
    ax.set_xlabel("normalised input value")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=7, loc="upper right", ncol=3)


def _draw_cons(
    ax, cons, labels, out_names, title, cmap="viridis", vlim=None, diff=False
):
    if diff:
        m = float(np.abs(cons).max()) if vlim is None else vlim
        im = ax.imshow(cons, cmap="coolwarm", vmin=-m, vmax=m, aspect="auto")
    else:
        im = ax.imshow(cons, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(out_names)))
    ax.set_xticklabels(out_names, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(title, fontsize=10)
    for r in range(cons.shape[0]):
        for c in range(cons.shape[1]):
            v = cons[r, c]
            ax.text(
                c,
                r,
                f"{v:+.2f}" if diff else f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color="white" if (not diff and v < 0.55) else "black",
            )
    return im


def figure(tuned, scale, out):
    e_ant, e_hand, _, _, _ = fis.effort_base(scale)
    h_ant, h_hand, _, _, _ = fis.chain_base(scale)
    e_names = fis.effort_inputs(scale)
    h_names = fis.chain_inputs(scale)

    e_fit = np.asarray(tuned["effort_cons"])
    h_fit = np.asarray(tuned["chain_cons"])
    e_tab = np.asarray(tuned["effort_tab"])
    h_tab = np.asarray(tuned["chain_tab"])

    fig = plt.figure(figsize=(19.5, 11.0))
    gs = fig.add_gridspec(
        2, 4, width_ratios=[1.25, 1.0, 1.0, 1.0], hspace=0.32, wspace=0.42
    )

    _draw_mf(
        fig.add_subplot(gs[0, 0]),
        e_tab,
        e_names,
        "EFFORT — fitted membership functions",
    )
    e_labels = _rule_labels(e_ant, e_names)
    _draw_cons(
        fig.add_subplot(gs[0, 1]),
        e_hand,
        e_labels,
        E_OUT,
        "EFFORT consequents: hand-written",
    )
    im = _draw_cons(
        fig.add_subplot(gs[0, 2]),
        e_fit,
        e_labels,
        E_OUT,
        "EFFORT consequents: GA-fitted + polished",
    )
    fig.colorbar(im, ax=fig.axes[-1], fraction=0.046, label="effort (0 = least)")
    imd = _draw_cons(
        fig.add_subplot(gs[0, 3]),
        e_fit - e_hand,
        e_labels,
        E_OUT,
        "difference (fitted − hand-written)",
        diff=True,
    )
    fig.colorbar(imd, ax=fig.axes[-1], fraction=0.046, label="red = more effort")

    _draw_mf(
        fig.add_subplot(gs[1, 0]), h_tab, h_names, "CHAIN — fitted membership functions"
    )
    h_labels = _rule_labels(h_ant, h_names)
    _draw_cons(
        fig.add_subplot(gs[1, 1]),
        h_hand,
        h_labels,
        ["keep going"],
        "CHAIN consequents: hand-written",
    )
    _draw_cons(
        fig.add_subplot(gs[1, 2]),
        h_fit,
        h_labels,
        ["keep going"],
        "CHAIN consequents: GA-fitted + polished",
    )
    _draw_cons(
        fig.add_subplot(gs[1, 3]),
        h_fit - h_hand,
        h_labels,
        ["keep going"],
        "difference (fitted − hand-written)",
        diff=True,
    )

    fig.suptitle(
        f"The trained rule bases, '{scale}' scale — {e_ant.shape[0]} EFFORT rules over "
        f"{e_ant.shape[1]} inputs, {h_ant.shape[0]} CHAIN rules over {h_ant.shape[1]}",
        fontsize=12,
    )
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def describe(tuned, scale):
    """The rules the optimiser moved most, in words. A figure is for looking; this is for
    quoting, and for noticing when 'interpretable' has quietly stopped being true."""
    e_ant, e_hand, _, _, _ = fis.effort_base(scale)
    e_fit = np.asarray(tuned["effort_cons"])
    labels = _rule_labels(e_ant, fis.effort_inputs(scale))
    delta = e_fit - e_hand
    order = np.argsort(-np.abs(delta).mean(axis=1))
    print(f"\nEFFORT rules the optimiser disagreed with most ({scale} scale):")
    for r in order[:6]:
        moves = ", ".join(
            f"{E_OUT[c]} {e_hand[r, c]:.2f}->{e_fit[r, c]:.2f}"
            for c in range(delta.shape[1])
        )
        print(f"  IF {labels[r]:<28s} {moves}")

    tab = np.asarray(tuned["effort_tab"])
    xs = np.linspace(0.0, 1.0, tab.shape[2])
    print("\nfitted term supports (where each term is non-zero):")
    for i, name in enumerate(fis.effort_inputs(scale)):
        spans = []
        for t in range(tab.shape[1]):
            nz = xs[tab[i, t] > 1e-6]
            if nz.size:
                spans.append(f"{TERM[t]} [{nz.min():.2f},{nz.max():.2f}]")
            else:
                spans.append(f"{TERM[t]} empty")
        print(f"  {name:>12s}  " + "  ".join(spans))


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small", choices=("small", "large"))
    ap.add_argument("--tuned", default=None, help="default: results/tuned_<scale>.npz")
    ap.add_argument("--out", default=str(paths.FIGURES / "fis_tsp_rulebase.png"))
    args = ap.parse_args()
    paths.ensure()
    if args.tuned is None:
        args.tuned = str(paths.tuned(args.scale))
    tuned = np.load(args.tuned)
    # The file's own record of the scale it was fitted at wins over the flag: its consequents
    # are only meaningful against that scale's antecedents, so drawing them under the other
    # scale's input names would label every axis wrongly and still render.
    scale = str(tuned["scale"]) if "scale" in tuned else args.scale
    print(f"wrote {figure(tuned, scale, args.out)}")
    describe(tuned, scale)


if __name__ == "__main__":
    main()
