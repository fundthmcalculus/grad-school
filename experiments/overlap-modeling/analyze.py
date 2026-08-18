"""Turn ``outputs/results.json`` into the tables RESULTS.md quotes.

Every table here is generated from the run of record, never transcribed. Two
kinds of table, kept apart on purpose:

* **Headline** -- the overlap width and band shape are chosen per cell on the
  *validation* fold, and the table reports what that choice then scored on
  *test*, paired against the baseline in the same cell. This is the only table
  that answers "would adopting overlap have helped", because it is the only one
  where the width was not chosen with the answer in hand.
* **Diagnostic curves** -- test R2 as a function of the width, pooled. Useful for
  reading the *shape* of the effect and useless for deciding whether to adopt it:
  taking the best column of a curve is selection on test.

Usage: python experiments/overlap-modeling/analyze.py [--results PATH]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CELL = ["dataset", "n_buckets", "order", "seed"]
FAMILIES = ["soft-ante", "soft-random", "local-overlap", "full-overlap"]


def load(path):
    with open(path) as fh:
        payload = json.load(fh)
    df = pd.DataFrame(payload["records"])
    errors = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    return payload, df[df.get("error").isna()] if "error" in df else df, errors


def _fmt(x, nd=4):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _pm(values, nd=4):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return "--"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1) if v.size > 1 else 0.0:.{nd}f}"


def _wilcoxon(diffs):
    """Two-sided Wilcoxon signed-rank p on paired differences, or None."""
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    if d.size < 6 or np.allclose(d, 0):
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:                                  # noqa: BLE001
        return None


# --------------------------------------------------------------------------
def headline(df) -> str:
    """Validation-selected width per cell, scored on test, paired vs baseline."""
    base = (df[df.arm == "baseline"]
            .set_index(CELL)[["r2_test", "r2_val", "fit_seconds"]]
            .rename(columns=lambda c: c + "_base"))

    lines = ["## Headline — overlap width chosen on validation, scored on test", "",
             "One row per (family, dataset). The width and band shape are picked",
             "inside each (dataset, buckets, order, seed) cell by validation R², then",
             "that cell's test R² is paired against the baseline's. `Δ test R²` is the",
             "mean paired difference: **positive means overlap helped**.", "",
             "| family | dataset | cells | baseline test R² | selected test R² | Δ test R² | wins | Wilcoxon p |",
             "|---|---|---:|---|---|---|---:|---:|"]

    rows = []
    for family in FAMILIES:
        fam = df[df.arm == family]
        if fam.empty:
            continue
        # argmax on validation inside each cell -- never on test.
        pick = fam.loc[fam.groupby(CELL)["r2_val"].idxmax()].set_index(CELL)
        joined = pick.join(base, how="inner")
        for dataset, part in joined.groupby(level="dataset"):
            diffs = (part.r2_test - part.r2_test_base).to_numpy()
            p = _wilcoxon(diffs)
            rows.append((family, dataset, len(part)))
            lines.append(
                f"| {family} | {dataset} | {len(part)} | "
                f"{_pm(part.r2_test_base)} | {_pm(part.r2_test)} | "
                f"{_pm(diffs)} | {int((diffs > 0).sum())}/{len(diffs)} | "
                f"{'--' if p is None else f'{p:.2g}'} |")

    # The same question for the fusion penalty, whose knob is lambda not a width.
    fus = df[df.arm == "fusion"]
    if not fus.empty:
        pick = fus.loc[fus.groupby(CELL)["r2_val"].idxmax()].set_index(CELL)
        joined = pick.join(base, how="inner")
        for dataset, part in joined.groupby(level="dataset"):
            diffs = (part.r2_test - part.r2_test_base).to_numpy()
            p = _wilcoxon(diffs)
            lines.append(
                f"| fusion | {dataset} | {len(part)} | {_pm(part.r2_test_base)} | "
                f"{_pm(part.r2_test)} | {_pm(diffs)} | "
                f"{int((diffs > 0).sum())}/{len(diffs)} | "
                f"{'--' if p is None else f'{p:.2g}'} |")

    # soft-ante against its own control, cell by cell and with the same number of
    # validation candidates on each side, which is the comparison that isolates
    # "the boundary got softer" from "every membership fit got more rows".
    sr = df[df.arm == "soft-random"]
    sa = df[df.arm == "soft-ante"]
    if not sr.empty and not sa.empty:
        pick_sa = sa.loc[sa.groupby(CELL)["r2_val"].idxmax()].set_index(CELL)
        pick_sr = (sr.loc[sr.groupby(CELL)["r2_val"].idxmax()].set_index(CELL)
                   [["r2_test"]].rename(columns={"r2_test": "r2_test_rand"}))
        joined = pick_sa.join(pick_sr, how="inner").join(base, how="inner")
        lines += ["", "### soft-ante against its own control", "",
                  "`soft-random` borrows the same rows-per-fit with the same weights from",
                  "the same neighbours, drawn uniformly instead of at the shared edge, and",
                  "is selected over the same 14 candidates. `Δ vs random` is therefore the",
                  "part of the gain attributable to the *boundary*, with the extra data and",
                  "the extra selection freedom held fixed.", "",
                  "| dataset | cells | baseline | soft-random | soft-ante | Δ random−base "
                  "| Δ ante−random | wins | Wilcoxon p |",
                  "|---|---:|---|---|---|---|---|---:|---:|"]
        for dataset, part in joined.groupby(level="dataset"):
            d_rand = (part.r2_test_rand - part.r2_test_base).to_numpy()
            d_ante = (part.r2_test - part.r2_test_rand).to_numpy()
            p = _wilcoxon(d_ante)
            lines.append(
                f"| {dataset} | {len(part)} | {_pm(part.r2_test_base)} | "
                f"{_pm(part.r2_test_rand)} | {_pm(part.r2_test)} | {_pm(d_rand)} | "
                f"{_pm(d_ante)} | {int((d_ante > 0).sum())}/{len(d_ante)} | "
                f"{'--' if p is None else f'{p:.2g}'} |")

    # The local family's own control: local-overlap against local-hard, which is
    # the comparison that isolates the overlap from the switch to a local fit.
    lh = (df[df.arm == "local-hard"].set_index(CELL)[["r2_test"]]
          .rename(columns={"r2_test": "r2_test_lh"}))
    if not lh.empty:
        lines += ["", "### The local family against its own control", "",
                  "`local-hard` is the same per-rule fit with hard bucket edges, so this",
                  "is the overlap's effect with the local/global change held fixed —",
                  "the comparison the request is really about.", "",
                  "| family | dataset | cells | local-hard test R² | selected test R² "
                  "| Δ test R² | wins | Wilcoxon p |",
                  "|---|---|---:|---|---|---|---:|---:|"]
        for family in ["local-overlap", "full-overlap"]:
            fam = df[df.arm == family]
            if fam.empty:
                continue
            pick = fam.loc[fam.groupby(CELL)["r2_val"].idxmax()].set_index(CELL)
            joined = pick.join(lh, how="inner")
            for dataset, part in joined.groupby(level="dataset"):
                diffs = (part.r2_test - part.r2_test_lh).to_numpy()
                p = _wilcoxon(diffs)
                lines.append(
                    f"| {family} | {dataset} | {len(part)} | {_pm(part.r2_test_lh)} | "
                    f"{_pm(part.r2_test)} | {_pm(diffs)} | "
                    f"{int((diffs > 0).sum())}/{len(diffs)} | "
                    f"{'--' if p is None else f'{p:.2g}'} |")
    return "\n".join(lines) + "\n"


def curves(df, fractions, shapes) -> str:
    """Test R² against overlap width. Diagnostic — do not select a width from this."""
    lines = ["## Diagnostic — test R² against overlap width", "",
             "**Not a selection table.** Reading off the best column is selection on",
             "test; the headline table above is the honest version. What this is for is",
             "the *shape* of each family's response to the width.", ""]
    for dataset, part in df.groupby("dataset"):
        base = part[part.arm == "baseline"].r2_test
        lh = part[part.arm == "local-hard"].r2_test
        lines += [f"### {dataset}", "",
                  f"baseline `{_pm(base)}`  ·  local-hard `{_pm(lh)}`", "",
                  "| family | shape | " + " | ".join(f"τ={f:g}" for f in fractions) + " |",
                  "|---|---|" + "---|" * len(fractions)]
        for family in FAMILIES:
            for shape in shapes:
                cells = []
                for f in fractions:
                    sel = part[(part.arm == family) & (part.overlap_shape == shape)
                               & (np.isclose(part.overlap, f))]
                    cells.append(_pm(sel.r2_test, 3) if len(sel) else "--")
                lines.append(f"| {family} | {shape} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def by_buckets(df, fractions) -> str:
    """Does overlap matter more when there are more edges to soften?"""
    lines = ["## Does the overlap matter more with more buckets?", "",
             "More buckets means more internal edges, so if hard edges are the problem",
             "the overlap's benefit should grow with the bucket count. Δ is the",
             "validation-selected family minus the baseline, in the same cells.", "",
             "| family | dataset | " + " | ".join(f"{b} buckets" for b in sorted(df.n_buckets.unique())) + " |",
             "|---|---|" + "---|" * df.n_buckets.nunique()]
    base = df[df.arm == "baseline"].set_index(CELL)[["r2_test"]].rename(
        columns={"r2_test": "base"})
    for family in FAMILIES:
        fam = df[df.arm == family]
        if fam.empty:
            continue
        pick = fam.loc[fam.groupby(CELL)["r2_val"].idxmax()].set_index(CELL).join(base)
        for dataset, part in pick.groupby(level="dataset"):
            cells = []
            for b in sorted(df.n_buckets.unique()):
                sel = part[part.index.get_level_values("n_buckets") == b]
                cells.append(_pm(sel.r2_test - sel.base, 3) if len(sel) else "--")
            lines.append(f"| {family} | {dataset} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def fusion(df, fusion_regs) -> str:
    lines = ["## The fusion penalty — overlap as neighbour agreement instead of shared data", "",
             "`fusion_reg` weights `Σ_r ||c_{r+1} − c_r||²` inside the exact global solve.",
             "λ→∞ is the limit of total data sharing between neighbours: one polynomial",
             "for every rule.", "",
             "| dataset | baseline | " + " | ".join(f"λ={lam:g}" for lam in fusion_regs) + " |",
             "|---|---|" + "---|" * len(fusion_regs)]
    for dataset, part in df.groupby("dataset"):
        cells = [_pm(part[part.arm == "baseline"].r2_test, 3)]
        for lam in fusion_regs:
            sel = part[(part.arm == "fusion") & (np.isclose(part.fusion_reg, lam))]
            cells.append(_pm(sel.r2_test, 3) if len(sel) else "--")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def mechanism(df, fractions) -> str:
    """Confirm the antecedent overlap actually softened the membership functions."""
    lines = ["## Mechanism check — did the antecedents actually get softer?", "",
             "Mean overlap coefficient `Σ min / Σ max` between adjacent buckets'",
             "membership envelopes, averaged over features and adjacent pairs. If this",
             "does not rise with τ then `overlap_antecedents` is not doing what it says",
             "and no accuracy reading from it means anything.", "",
             "| dataset | τ=0 (baseline) | " + " | ".join(f"τ={f:g}" for f in fractions) + " |",
             "|---|---|" + "---|" * len(fractions)]
    for dataset, part in df.groupby("dataset"):
        cells = [_fmt(part[part.arm == "baseline"].overlap_area.mean(), 3)]
        for f in fractions:
            sel = part[(part.arm == "soft-ante") & (np.isclose(part.overlap, f))]
            cells.append(_fmt(sel.overlap_area.mean(), 3) if len(sel) else "--")
        lines.append(f"| {dataset} | " + " | ".join(cells) + " |")
    lines += ["", "Rule counts and fit cost, to rule out the overlap buying accuracy with",
              "either:", "",
              "| arm | mean rules | mean features | mean fit s |", "|---|---:|---:|---:|"]
    for arm, part in df.groupby("arm"):
        lines.append(f"| {arm} | {part.n_rules.mean():.2f} | "
                     f"{part.n_features.mean():.2f} | {part.fit_seconds.mean():.3f} |")
    return "\n".join(lines) + "\n"


def figure(df, fractions, path) -> str | None:
    """Test R2 against overlap width, one panel per dataset, with the two controls.

    The point of drawing this is that the three families respond to the width in
    visibly different ways, which a table of means makes you reconstruct in your
    head: the local family climbs steeply from its own control, and the global
    family sits flat on the baseline. Diagnostic, like `curves.md`.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                   # noqa: BLE001
        return None

    datasets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.1 * len(datasets), 3.6),
                             sharey=False)
    axes = np.atleast_1d(axes)
    colors = {"soft-ante": "#1f77b4", "soft-random": "#2ca02c",
              "local-overlap": "#d62728", "full-overlap": "#9467bd"}
    for ax, dataset in zip(axes, datasets):
        part = df[df.dataset == dataset]
        base = part[part.arm == "baseline"].r2_test.mean()
        lh = part[part.arm == "local-hard"].r2_test.mean()
        ax.axhline(base, color="k", ls="-", lw=1.4, label="baseline (global, hard)")
        ax.axhline(lh, color="0.55", ls=":", lw=1.4, label="local-hard (control)")
        for family, color in colors.items():
            for shape, ls in (("flat", "-"), ("ramp", "--")):
                ys = [part[(part.arm == family) & (part.overlap_shape == shape)
                           & (np.isclose(part.overlap, f))].r2_test.mean()
                      for f in fractions]
                ax.plot(fractions, ys, ls, color=color, marker="o", ms=3.5, lw=1.3,
                        label=f"{family} ({shape})")
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("overlap fraction τ")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("test R² (mean over cells)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Test R² against overlap width — diagnostic, not a selection table",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "outputs", "results.json"))
    args = ap.parse_args()

    payload, df, errors = load(args.results)
    fractions, shapes = payload["fractions"], payload["shapes"]
    out = os.path.join(HERE, "outputs")

    header = (f"<!-- generated by analyze.py from {os.path.basename(args.results)}; "
              f"repo {payload['provenance']['repo_commit'][:7]}, "
              f"tribble-fis {payload['provenance']['tribble_fis_commit'][:7]}, "
              f"{len(payload['provenance']['seeds'])} seeds -->\n\n")

    written = {
        "headline.md": headline(df),
        "curves.md": curves(df, fractions, shapes),
        "by_buckets.md": by_buckets(df, fractions),
        "fusion.md": fusion(df, payload["fusion_regs"]),
        "mechanism.md": mechanism(df, fractions),
    }
    for name, body in written.items():
        with open(os.path.join(out, name), "w") as fh:
            fh.write(header + body)
        print(f"wrote outputs/{name}")

    drawn = figure(df, fractions, os.path.join(out, "overlap_curves.png"))
    print(f"wrote outputs/{os.path.basename(drawn)}" if drawn
          else "matplotlib unavailable; figure skipped")

    dropped = df[["dropped_val", "dropped_test"]].to_numpy().sum() if "dropped_test" in df else 0
    print(f"\n{len(df)} scored records, {len(errors)} errors, "
          f"{int(dropped)} non-finite predictions dropped")
    if len(errors):
        print(errors[["dataset", "n_buckets", "order", "label", "error"]].head(10).to_string())
    print("\n" + written["headline.md"])


if __name__ == "__main__":
    main()
