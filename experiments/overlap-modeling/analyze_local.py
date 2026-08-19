"""Turn ``outputs/local_results.json`` into the stage-2 tables.

Stage 2 asks one question: with a genuine per-bucket consequent solve, is the
local family's deficit the FIT or the AGGREGATION? So the central table puts two
columns side by side that stage 1 never separated --

    local R2   how well each row's own-bucket rule approximates y, blend ignored
    test R2    how well the blended model predicts y

-- because the whole finding is that per-bucket solving moves those two in
*opposite* directions.

Usage: python experiments/overlap-modeling/analyze_local.py
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


def _pm(v, nd=3):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    if v.size == 0:
        return "--"
    return f"{v.mean():.{nd}f} ± {v.std(ddof=1) if v.size > 1 else 0.0:.{nd}f}"


def _mean(v, nd=3):
    v = np.asarray([x for x in v if np.isfinite(x)], dtype=float)
    return "--" if v.size == 0 else f"{v.mean():.{nd}f}"


def _wilcoxon(diffs):
    d = np.asarray([x for x in diffs if np.isfinite(x)], dtype=float)
    if d.size < 6 or np.allclose(d, 0):
        return None
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d).pvalue)
    except Exception:                                   # noqa: BLE001
        return None


def fit_vs_blend(df) -> str:
    """The central table: local approximation quality against blended accuracy."""
    lines = ["## Fit or aggregation? Local approximation against blended accuracy", "",
             "`local R²` scores each row's **own-bucket rule alone**, ignoring the blend;",
             "`test R²` scores the blended model. If per-bucket solving raises the first",
             "while lowering the second, each rule *is* the better local approximator the",
             "idea predicts, and the blend is what loses the accuracy.", "",
             "Means over all cells per dataset. τ is the overlap width.", ""]
    for dataset, part in df.groupby("dataset"):
        lines += [f"### {dataset}", "",
                  "| arm | τ | local R² (train) | local R² (test) | test R² |",
                  "|---|---:|---|---|---|"]
        base = part[part.arm == "baseline"]
        lines.append(f"| baseline (global) | — | {_mean(base.local_r2_train)} | "
                     f"{_mean(base.local_r2_test)} | **{_mean(base.r2_test)}** |")
        for arm in ["local-free", "local-residual", "local-wta", "local-recal",
                    "local-sharp", "shrink-local"]:
            sub = part[part.arm == arm]
            if sub.empty:
                continue
            for tau, grp in sub.groupby("overlap"):
                lines.append(f"| {arm} | {tau:g} | {_mean(grp.local_r2_train)} | "
                             f"{_mean(grp.local_r2_test)} | {_mean(grp.r2_test)} |")
        for arm in ["global-wta", "global-recal", "global-sharp"]:
            sub = part[part.arm == arm]
            if not sub.empty:
                lines.append(f"| {arm} *(control)* | — | {_mean(sub.local_r2_train)} | "
                             f"{_mean(sub.local_r2_test)} | {_mean(sub.r2_test)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def aggregation_fixes(df) -> str:
    """Do the three aggregation fixes recover the local family's deficit?"""
    base = df[df.arm == "baseline"].set_index(CELL)[["r2_test"]].rename(
        columns={"r2_test": "base"})
    lines = ["## Do the aggregation fixes recover the deficit?", "",
             "Each local arm's width (and γ where it has one) is chosen per cell on",
             "**validation** R², then scored on test and paired against the baseline in",
             "the same cell. `recovered` is the fraction of `local-free`'s own deficit",
             "that the fix closes, so 100% would mean the fix reaches the baseline. It",
             "is left blank for the `global-*` controls, which never had that deficit.", "",
             "| arm | dataset | selected test R² | Δ vs baseline | recovered | wins | Wilcoxon p |",
             "|---|---|---|---|---:|---:|---:|"]
    ref = {}
    order = ["local-free", "local-residual", "local-wta", "local-recal",
             "local-sharp", "shrink-local", "global-wta", "global-recal",
             "global-sharp"]
    for arm in order:
        fam = df[df.arm == arm]
        if fam.empty:
            continue
        pick = fam.loc[fam.groupby(CELL)["r2_val"].idxmax()].set_index(CELL)
        joined = pick.join(base, how="inner")
        for dataset, part in joined.groupby(level="dataset"):
            diffs = (part.r2_test - part.base).to_numpy()
            if arm == "local-free":
                ref[dataset] = float(np.nanmean(diffs))
            deficit = ref.get(dataset)
            # Only meaningful for the local family: the global controls never had
            # `local-free`'s deficit, so "recovering" it is not a thing they do.
            rec = "--"
            if arm.startswith("local") or arm == "shrink-local":
                if deficit and np.isfinite(deficit) and deficit < 0:
                    rec = f"{100.0 * (1.0 - np.nanmean(diffs) / deficit):.0f}%"
            p = _wilcoxon(diffs)
            lines.append(
                f"| {arm} | {dataset} | {_pm(part.r2_test)} | {_pm(diffs)} | {rec} | "
                f"{int((diffs > 0).sum())}/{len(diffs)} | "
                f"{'--' if p is None else f'{p:.2g}'} |")
    return "\n".join(lines) + "\n"


def sharpen_table(df, gammas) -> str:
    lines = ["## The blend-concentration exponent γ", "",
             "Firing strengths raised to γ before normalization, in the solve and at",
             "predict time alike. γ→∞ is winner-take-all, γ=1 is TSK's own weighting.",
             "`global-sharp` is the control: if the global solve likes the same γ, the",
             "exponent is not a statement about local fitting.", "",
             "| dataset | arm | γ=1 | " + " | ".join(f"γ={g:g}" for g in gammas) + " |",
             "|---|---|---|" + "---|" * len(gammas)]
    for dataset, part in df.groupby("dataset"):
        for arm, unity_arm, taus in (("local-sharp", "local-free", sorted(
                part[part.arm == "local-sharp"].overlap.unique())),
                ("global-sharp", "baseline", [None])):
            for tau in taus:
                sub = part[part.arm == arm]
                unity = part[part.arm == unity_arm]
                if tau is not None:
                    sub = sub[np.isclose(sub.overlap, tau)]
                    unity = unity[np.isclose(unity.overlap, tau)]
                if sub.empty:
                    continue
                name = arm if tau is None else f"{arm} (τ={tau:g})"
                cells = [_mean(unity.r2_test)]
                for g in gammas:
                    cells.append(_mean(sub[sub.label.str.endswith(f"/{g:g}")].r2_test))
                lines.append(f"| {dataset} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def figure(df, path) -> str | None:
    """The two curves that tell the story, per dataset."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:                                   # noqa: BLE001
        return None

    datasets = sorted(df.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.1 * len(datasets), 3.7))
    axes = np.atleast_1d(axes)
    for ax, dataset in zip(axes, datasets):
        part = df[df.dataset == dataset]
        loc = part[part.arm == "local-free"].groupby("overlap")
        taus = sorted(part[part.arm == "local-free"].overlap.unique())
        ax.plot(taus, [loc.get_group(t).local_r2_test.mean() for t in taus],
                "-o", ms=3.5, color="#d62728",
                label="local R² of own-bucket rule\n(per-bucket solve)")
        ax.plot(taus, [loc.get_group(t).r2_test.mean() for t in taus],
                "-s", ms=3.5, color="#1f77b4", label="test R² of the blend")
        base = part[part.arm == "baseline"]
        ax.axhline(base.local_r2_test.mean(), ls=":", color="#d62728", lw=1.3,
                   label="local R², global solve")
        ax.axhline(base.r2_test.mean(), ls="-", color="k", lw=1.4,
                   label="test R², global solve (baseline)")
        ax.set_title(dataset, fontsize=10)
        ax.set_xlabel("overlap fraction τ")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("R² (mean over cells)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.10))
    fig.suptitle("Per-bucket solving makes every rule a better local approximator "
                 "and the blended model worse", fontsize=10)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(HERE, "outputs",
                                                      "local_results.json"))
    args = ap.parse_args()
    payload = load_payload(args.results)
    df = pd.DataFrame(payload["records"])
    errors = df[df.get("error").notna()] if "error" in df else df.iloc[:0]
    if "error" in df:
        df = df[df.error.isna()]

    out = os.path.join(HERE, "outputs")
    prov = payload["provenance"]
    header = (f"<!-- generated by analyze_local.py from "
              f"{os.path.basename(args.results)}; repo {prov['repo_commit'][:7]}, "
              f"tribble-fis {prov['tribble_fis_commit'][:7]}, "
              f"{len(prov['seeds'])} seeds -->\n\n")

    from run_local import SHARPEN  # noqa: E402  -- one source of truth for the grid
    written = {
        "local_fit_vs_blend.md": fit_vs_blend(df),
        "local_aggregation.md": aggregation_fixes(df),
        "local_sharpen.md": sharpen_table(df, SHARPEN),
    }
    for name, body in written.items():
        with open(os.path.join(out, name), "w") as fh:
            fh.write(header + body)
        print(f"wrote outputs/{name}")
    drawn = figure(df, os.path.join(out, "local_fit_vs_blend.png"))
    print(f"wrote outputs/{os.path.basename(drawn)}" if drawn else "figure skipped")

    print(f"\n{len(df)} scored records, {len(errors)} errors")
    print("\n" + written["local_aggregation.md"])


if __name__ == "__main__":
    main()
