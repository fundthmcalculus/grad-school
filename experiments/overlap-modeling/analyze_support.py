"""Turn ``outputs/support_results.json`` into the stage-3 tables.

Stage 3 tests whether COMPACT antecedent support rescues per-bucket local
consequents. Two columns decide it and they pull against each other:

    active_frac  what fraction of the rules fire on a typical row -- how local
                 the model actually is
    uncovered    what fraction of rows NO rule covers -- rows the model answers
                 with exactly 0, which is finite and so passes every NaN filter

A membership shape only helps if it drives `active_frac` down while holding
`uncovered` at zero. The point of tabulating both is that any shape can win one
of them alone.

Usage: python experiments/overlap-modeling/analyze_support.py
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


def _shape_order(df):
    """Membership shapes in a readable order: gaussian, clamps by k, then the rest."""
    shapes = list(df.shape_.unique()) if "shape_" in df else list(df["shape"].unique())

    def key(s):
        if s == "gaussian":
            return (0, 0.0, "")
        if s.startswith("clamped-smooth/"):
            return (1, float(s.split("/")[1]), s)
        if s.startswith("clamped-hard/"):
            return (2, float(s.split("/")[1]), s)
        if s == "trapezoid":
            return (3, 0.0, s)
        return (4, 0.0, s)

    return sorted(shapes, key=key)


def support_table(df) -> str:
    """The central table: locality, coverage and accuracy for every shape."""
    lines = [
        "## Compact support: locality, coverage, accuracy",
        "",
        "`active_frac` is the fraction of rules firing on a typical row (1.0 = every",
        "rule everywhere, which is what infinite support gives). `uncovered` is the",
        "fraction of rows no rule covers -- those are answered with exactly 0.",
        "`local R²` scores each row's own-bucket rule alone; `test R²` scores the",
        "blended model. All at overlap τ=0 so the shape is the only thing varying.",
        "",
    ]
    for dataset, part in df.groupby("dataset"):
        part = part[np.isclose(part.overlap, 0.0)]
        lines += [
            f"### {dataset}",
            "",
            "| membership | active_frac | uncovered | local R² (global) "
            "| test R² (global) | local R² (local fit) | test R² (local fit) |",
            "|---|---:|---:|---|---|---|---|",
        ]
        for shape in _shape_order(part):
            sub = part[part["shape"] == shape]
            g = sub[sub.fit == "global"]
            lo = sub[sub.fit == "local"]
            if g.empty and lo.empty:
                continue
            lines.append(
                f"| {shape} | {_mean(g.active_frac)} | {_mean(g.uncovered)} | "
                f"{_mean(g.local_r2_test)} | **{_mean(g.r2_test)}** | "
                f"{_mean(lo.local_r2_test)} | {_mean(lo.r2_test)} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def paired_vs_gaussian(df) -> str:
    """Every shape paired against the Gaussian baseline in the same cell."""
    lines = [
        "## Paired against Gaussian, same cell",
        "",
        "Cell-by-cell paired differences against `gaussian` at the same",
        "(dataset, buckets, order, seed, fit, τ). Positive means the compact shape",
        "beat infinite support. τ is chosen per cell on **validation** R² within",
        "each shape, so no width is selected on test.",
        "",
        "| membership | fit | dataset | test R² | Δ vs gaussian | wins | Wilcoxon p |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for fit in ("global", "local"):
        base = (
            df[(df["shape"] == "gaussian") & (df.fit == fit)]
            .loc[lambda d: d.groupby(CELL)["r2_val"].idxmax()]
            .set_index(CELL)[["r2_test"]]
            .rename(columns={"r2_test": "base"})
        )
        for shape in _shape_order(df):
            if shape == "gaussian":
                continue
            sub = df[(df["shape"] == shape) & (df.fit == fit)]
            if sub.empty:
                continue
            pick = sub.loc[sub.groupby(CELL)["r2_val"].idxmax()].set_index(CELL)
            joined = pick.join(base, how="inner")
            for dataset, part in joined.groupby(level="dataset"):
                d = (part.r2_test - part.base).to_numpy()
                p = _wilcoxon(d)
                lines.append(
                    f"| {shape} | {fit} | {dataset} | {_pm(part.r2_test)} | {_pm(d)} | "
                    f"{int((d > 0).sum())}/{len(d)} | "
                    f"{'--' if p is None else f'{p:.2g}'} |"
                )
    return "\n".join(lines) + "\n"


def locality_gap(df) -> str:
    """Does the local fit close on the global one as support tightens?"""
    lines = [
        "## Does the local fit close on the global one as support tightens?",
        "",
        "The counterpoint's prediction: with rules silent outside their own region,",
        "per-bucket consequents should stop being blended in where they do not apply,",
        "so `gap` (local minus global test R², same cell) should shrink toward 0 as",
        "`active_frac` falls. Both at τ=0.",
        "",
        "| membership | active_frac | uncovered | gap = local − global test R² |",
        "|---|---:|---:|---|",
    ]
    for shape in _shape_order(df):
        sub = df[(df["shape"] == shape) & np.isclose(df.overlap, 0.0)]
        g = sub[sub.fit == "global"].set_index(CELL)[["r2_test"]]
        lo = sub[sub.fit == "local"].set_index(CELL)[["r2_test"]]
        j = lo.join(g, how="inner", lsuffix="_local", rsuffix="_global")
        if j.empty:
            continue
        gap = (j.r2_test_local - j.r2_test_global).to_numpy()
        lines.append(
            f"| {shape} | {_mean(sub.active_frac)} | {_mean(sub.uncovered)} | "
            f"{_pm(gap)} |"
        )
    return "\n".join(lines) + "\n"


def figure(df, path) -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return None

    ks = sorted(
        {
            float(s.split("/")[1])
            for s in df["shape"].unique()
            if s.startswith("clamped-smooth/")
        }
    )
    datasets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(
        2, len(datasets), figsize=(3.6 * len(datasets), 6.0), sharex=True
    )
    axes = np.atleast_2d(axes)
    for col, dataset in enumerate(datasets):
        part = df[(df.dataset == dataset) & np.isclose(df.overlap, 0.0)]
        gauss = part[part["shape"] == "gaussian"]

        def series(field, fit):
            return [
                part[(part["shape"] == f"clamped-smooth/{k:g}") & (part.fit == fit)][
                    field
                ].mean()
                for k in ks
            ]

        ax = axes[0, col]
        ax.plot(
            ks,
            series("r2_test", "global"),
            "-o",
            ms=3.5,
            color="#1f77b4",
            label="global solve",
        )
        ax.plot(
            ks,
            series("r2_test", "local"),
            "-s",
            ms=3.5,
            color="#d62728",
            label="per-bucket solve",
        )
        ax.axhline(
            gauss[gauss.fit == "global"].r2_test.mean(),
            ls="-",
            color="k",
            lw=1.3,
            label="gaussian, global",
        )
        ax.axhline(
            gauss[gauss.fit == "local"].r2_test.mean(),
            ls=":",
            color="#d62728",
            lw=1.3,
            label="gaussian, per-bucket",
        )
        ax.set_title(dataset, fontsize=10)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("test R²")

        ax2 = axes[1, col]
        ax2.plot(
            ks,
            series("active_frac", "global"),
            "-o",
            ms=3.5,
            color="#2ca02c",
            label="active_frac (locality)",
        )
        ax2.plot(
            ks,
            series("uncovered", "global"),
            "-^",
            ms=3.5,
            color="#ff7f0e",
            label="uncovered (dead zones)",
        )
        ax2.set_xlabel("clamp cutoff k (σ)")
        ax2.grid(alpha=0.25)
        ax2.set_ylim(-0.03, 1.03)
        if col == 0:
            ax2.set_ylabel("fraction")

    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        h1 + h2,
        l1 + l2,
        loc="lower center",
        ncol=6,
        fontsize=7.5,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Clamping a Gaussian: accuracy against the locality/coverage trade-off",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results", default=os.path.join(HERE, "outputs", "support_results.json")
    )
    args = ap.parse_args()
    payload = load_payload(args.results)
    df = pd.DataFrame(payload["records"])
    errors = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    if "error" in df:
        df = df[df.error.isna()]

    out = os.path.join(HERE, "outputs")
    prov = payload["provenance"]
    header = (
        f"<!-- generated by analyze_support.py from "
        f"{os.path.basename(args.results)}; repo {prov['repo_commit'][:7]}, "
        f"tribble-fis {prov['tribble_fis_commit'][:7]}, "
        f"{len(prov['seeds'])} seeds -->\n\n"
    )

    written = {
        "support_shapes.md": support_table(df),
        "support_paired.md": paired_vs_gaussian(df),
        "support_locality.md": locality_gap(df),
    }
    for name, body in written.items():
        with open(os.path.join(out, name), "w") as fh:
            fh.write(header + body)
        print(f"wrote outputs/{name}")
    drawn = figure(df, os.path.join(out, "support_clamp.png"))
    print(f"wrote outputs/{os.path.basename(drawn)}" if drawn else "figure skipped")

    print(f"\n{len(df)} scored records, {len(errors)} errors")
    print("\n" + written["support_locality.md"])


if __name__ == "__main__":
    main()
