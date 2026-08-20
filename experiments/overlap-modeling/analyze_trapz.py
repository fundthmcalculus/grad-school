"""Turn ``outputs/trapz_results.json`` into the stage-4 tables.

Two questions, and the sweep separates them because both knobs vary:

* does padding make compact support usable at all (`pad=0` against `pad>0`)?
* with the endpoint defect removed, does coarsening the histogram -- widening the
  support -- help, as predicted?

Usage: python experiments/overlap-modeling/analyze_trapz.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_experiment import load_payload  # noqa: E402

CELL = ["dataset", "n_buckets", "order", "seed"]


def _pm(v, nd=4):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return "--"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1) if v.size > 1 else 0.0:.{nd}f}"


def _mean(v, nd=3):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    return "--" if v.size == 0 else f"{v.mean():.{nd}f}"


def _wilcoxon(d):
    d = np.asarray([x for x in d if np.isfinite(x)], dtype=float)
    if d.size < 6 or np.allclose(d, 0):
        return None
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(d).pvalue)
    except Exception:  # noqa: BLE001
        return None


def grid(df, bins, pads, field, fit, nd=3) -> list[str]:
    lines = [
        "| n_bins | " + " | ".join(f"pad={p:g}" for p in pads) + " |",
        "|---|" + "---|" * len(pads),
    ]
    for b in bins:
        cells = []
        for p in pads:
            sel = df[
                (df["shape"] == "trapezoid")
                & (df.fit == fit)
                & (df.trapz_bins == b)
                & (np.isclose(df.trapz_pad, p))
            ]
            cells.append(_mean(sel[field], nd) if len(sel) else "--")
        lines.append(f"| {b} | " + " | ".join(cells) + " |")
    return lines


def grids(df, bins, pads) -> str:
    lines = [
        "## The (n_bins × pad) grid",
        "",
        "`pad=0` is the library's fitted geometry, whose left edge sits on the data",
        "minimum. Reading down a `pad>0` column answers the coarsen-the-histogram",
        "question; reading across a row answers whether padding was the blocker.",
        "",
    ]
    for dataset, part in df.groupby("dataset"):
        gauss = part[part["shape"] == "gaussian"]
        lines += [
            f"### {dataset}",
            "",
            f"Gaussian reference: global `{_mean(gauss[gauss.fit == 'global'].r2_test)}`"
            f" · per-bucket `{_mean(gauss[gauss.fit == 'local'].r2_test)}`",
            "",
            "**test R² (global solve)**",
            "",
        ]
        lines += grid(part, bins, pads, "r2_test", "global")
        lines += ["", "**uncovered fraction**", ""]
        lines += grid(part, bins, pads, "uncovered", "global")
        lines += ["", "**active_frac (rules firing per row)**", ""]
        lines += grid(part, bins, pads, "active_frac", "global")
        lines += ["", "**test R² (per-bucket solve)**", ""]
        lines += grid(part, bins, pads, "r2_test", "local")
        lines.append("")
    return "\n".join(lines) + "\n"


def paired(df) -> str:
    """Every trapezoid config paired against the Gaussian in the same cell."""
    lines = [
        "## Paired against Gaussian, same cell",
        "",
        "Config chosen per cell on **validation** R² within each family, then scored",
        "on test and paired against the Gaussian arm of the same cell and solver.",
        "Positive means trapezoids beat infinite support.",
        "",
        "| family | fit | dataset | test R² | Δ vs gaussian | wins | Wilcoxon p |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for fit in ("global", "local"):
        base = (
            df[(df["shape"] == "gaussian") & (df.fit == fit)]
            .set_index(CELL)[["r2_test"]]
            .rename(columns={"r2_test": "base"})
        )
        for name, sub in (
            (
                "trapezoid, padded (pad>0)",
                df[(df["shape"] == "trapezoid") & (df.fit == fit) & (df.trapz_pad > 0)],
            ),
            (
                "trapezoid, unpadded (pad=0)",
                df[
                    (df["shape"] == "trapezoid")
                    & (df.fit == fit)
                    & (np.isclose(df.trapz_pad, 0.0))
                ],
            ),
        ):
            if sub.empty:
                continue
            pick = sub.loc[sub.groupby(CELL)["r2_val"].idxmax()].set_index(CELL)
            joined = pick.join(base, how="inner")
            for dataset, part in joined.groupby(level="dataset"):
                d = (part.r2_test - part.base).to_numpy()
                p = _wilcoxon(d)
                lines.append(
                    f"| {name} | {fit} | {dataset} | {_pm(part.r2_test)} | {_pm(d)} | "
                    f"{int((d > 0).sum())}/{len(d)} | "
                    f"{'--' if p is None else f'{p:.2g}'} |"
                )
    return "\n".join(lines) + "\n"


def bins_effect(df, bins) -> str:
    """With the defect removed, does coarsening the histogram help?"""
    lines = [
        "## Does coarsening the histogram help, once padded?",
        "",
        "Padded configs only (`pad>0`), pooled over pad. Paired against the same",
        "cell's `n_bins=50` (the library default) at the same pad and solver, so the",
        "bin count is the only thing varying.",
        "",
        "| fit | dataset | " + " | ".join(f"{b} bins" for b in bins) + " |",
        "|---|---|" + "---|" * len(bins),
    ]
    for fit in ("global", "local"):
        sub = df[(df["shape"] == "trapezoid") & (df.fit == fit) & (df.trapz_pad > 0)]
        if sub.empty:
            continue
        key = CELL + ["trapz_pad"]
        ref = (
            sub[sub.trapz_bins == 50]
            .set_index(key)[["r2_test"]]
            .rename(columns={"r2_test": "ref"})
        )
        for dataset in sorted(sub.dataset.unique()):
            cells = []
            for b in bins:
                cur = sub[(sub.trapz_bins == b) & (sub.dataset == dataset)].set_index(
                    key
                )
                j = cur.join(ref, how="inner")
                cells.append(_pm(j.r2_test - j.ref, 4) if len(j) else "--")
            lines.append(f"| {fit} | {dataset} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def figure(df, bins, pads, path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None

    datasets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(
        2, len(datasets), figsize=(3.6 * len(datasets), 6.2), sharex=True, squeeze=False
    )
    # `squeeze=False` plus an explicit reshape: with one dataset, subplots returns
    # a 1-D array and `np.atleast_2d` orients it (1, 2) rather than (2, 1), so
    # `axes[row, col]` silently addresses the wrong panel or raises. This is the
    # failure `WORKINGDOC.md` records from `gated-minimax-selection/run_all.py`.
    axes = np.asarray(axes).reshape(2, len(datasets))
    cmap = plt.get_cmap("viridis")
    for col, dataset in enumerate(datasets):
        part = df[df.dataset == dataset]
        gauss = part[part["shape"] == "gaussian"]
        for row, field, label in (
            (0, "r2_test", "test R²"),
            (1, "uncovered", "uncovered fraction"),
        ):
            ax = axes[row, col]
            for i, p in enumerate(pads):
                ys = [
                    part[
                        (part["shape"] == "trapezoid")
                        & (part.fit == "global")
                        & (part.trapz_bins == b)
                        & (np.isclose(part.trapz_pad, p))
                    ][field].mean()
                    for b in bins
                ]
                ax.plot(
                    bins,
                    ys,
                    "-o",
                    ms=3.5,
                    color=cmap(i / max(1, len(pads) - 1)),
                    label=f"pad={p:g}",
                )
            if row == 0:
                ax.axhline(
                    gauss[gauss.fit == "global"].r2_test.mean(),
                    ls="-",
                    color="k",
                    lw=1.3,
                    label="gaussian",
                )
            ax.set_xscale("log")
            ax.set_xticks(bins)
            ax.set_xticklabels([str(b) for b in bins])
            ax.grid(alpha=0.25)
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(dataset, fontsize=10)
            else:
                ax.set_xlabel("n_bins")
                ax.set_ylim(-0.03, 1.03)

    h, la = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        h,
        la,
        loc="lower center",
        ncol=7,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    # The title states what the sweep measured, not what was predicted: padding is
    # the whole effect and the bin count barely moves anything once it is applied.
    fig.suptitle(
        "Trapezoid antecedents: padding is the whole effect; "
        "the bin count barely matters",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", default=os.path.join(HERE, "outputs", "trapz_results.json")
    )
    args = ap.parse_args()
    payload = load_payload(args.results)
    df = pd.DataFrame(payload["records"])
    errors = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    if "error" in df:
        df = df[df.error.isna()]
    bins, pads = payload["bins"], payload["pads"]

    out = os.path.join(HERE, "outputs")
    prov = payload["provenance"]
    header = (
        f"<!-- generated by analyze_trapz.py from "
        f"{os.path.basename(args.results)}; repo {prov['repo_commit'][:7]}, "
        f"tribble-fis {prov['tribble_fis_commit'][:7]}, "
        f"{len(prov['seeds'])} seeds -->\n\n"
    )

    written = {
        "trapz_grid.md": grids(df, bins, pads),
        "trapz_paired.md": paired(df),
        "trapz_bins.md": bins_effect(df, bins),
    }
    for name, body in written.items():
        with open(os.path.join(out, name), "w") as fh:
            fh.write(header + body)
        print(f"wrote outputs/{name}")
    drawn = figure(df, bins, pads, os.path.join(out, "trapz_grid.png"))
    print(f"wrote outputs/{os.path.basename(drawn)}" if drawn else "figure skipped")

    print(f"\n{len(df)} scored records, {len(errors)} errors")
    print("\n" + written["trapz_paired.md"])
    print(written["trapz_bins.md"])


if __name__ == "__main__":
    main()
