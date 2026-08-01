"""Export the study's result tables in three consumable forms.

`FINDINGS.md` is the narrative record and already carries every table as
Markdown. This adds the two machine-readable forms:

  report/results.xlsx   one sheet per table -- for pasting into the dissertation
  report/summary.json   headline numbers only -- for programmatic checks
  report/TABLES.md      the same tables regenerated from the CSVs, so a stale
                        hand-written number in FINDINGS.md can be caught

Everything is derived from the CSVs in `data/`, so nothing here is transcribed
by hand and it cannot drift from what the scripts actually produced.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "report"

FIS = "centroid (PCA-free)"           # ASCII key; '·' does not survive the CSVs


def mean_std(df, value="auroc"):
    g = df.groupby(["detector", "condition"])[value].agg(["mean", "std", "count"])
    return g.reset_index()


def build():
    sw = pd.read_csv(DATA / "seed_sweep.csv")
    n_seeds = int(sw.seed.nunique())
    tables, summary = {}, {"n_seeds": n_seeds}

    # --- 1. accuracy with error bars, per family -------------------------
    for fam in ("FalsePremise", "TriviaQA"):
        s = sw[sw.family == fam]
        g = mean_std(s)
        piv = g.pivot(index="detector", columns="condition", values=["mean", "std"])
        out = pd.DataFrame({
            "auroc_raw_mean": piv[("mean", "raw")],
            "auroc_raw_std": piv[("std", "raw")],
            "auroc_matched_mean": piv[("mean", "matched")],
            "auroc_matched_std": piv[("std", "matched")],
        }).sort_values("auroc_matched_mean", ascending=False).round(4)
        tables[f"auroc_{fam}"] = out.reset_index()

        # paired advantage of the FIS over its best rival
        m = s[s.condition == "matched"].pivot_table(
            index="seed", columns="detector", values="auroc")
        fis_col = next((c for c in m.columns if FIS in c), None)
        rivals = [c for c in m.columns if c != fis_col and "n_tokens" not in c]
        if fis_col and rivals:
            best = m[rivals].mean().idxmax()
            d = (m[fis_col] - m[best]).dropna()
            summary[fam] = {
                "fis_matched_mean": round(float(m[fis_col].mean()), 4),
                "fis_matched_std": round(float(m[fis_col].std()), 4),
                "best_rival": best,
                "best_rival_matched_mean": round(float(m[best].mean()), 4),
                "paired_delta_mean": round(float(d.mean()), 4),
                "paired_delta_std": round(float(d.std()), 4),
                "paired_delta_min": round(float(d.min()), 4),
                "seeds_won": int((d > 0).sum()),
                "seeds_total": int(len(d)),
            }

    # --- 2. cost ---------------------------------------------------------
    cost = (sw.groupby("detector")
            .agg(feat_ms=("feat_ms", "mean"), fit_ms=("fit_ms", "mean"),
                 score_ms_per_1k=("score_ms_per_1k", "mean"),
                 n_mfs=("n_mfs", "mean"))
            .assign(total_train_ms=lambda d: d.feat_ms + d.fit_ms)
            .sort_values("total_train_ms").round(2))
    tables["cost"] = cost.reset_index()

    # --- 3. supporting sweeps -------------------------------------------
    for name, path, cols in [
        ("representation_sweep", "representation_sweep.csv", None),
        ("norm_pair_sweep", "norm_sweep.csv", None),
        ("pca_free_representations", "nopca_results.csv", None),
    ]:
        f = DATA / path
        if f.exists():
            tables[name] = pd.read_csv(f) if cols is None else pd.read_csv(f)[cols]

    # --- 4. label counts from the capture -------------------------------
    meta = pd.read_parquet(DATA / "capture_meta.parquet")
    tables["label_counts"] = (pd.crosstab(meta.family, meta.label)
                              .reset_index())
    summary["n_generations"] = int(len(meta))
    return tables, summary, n_seeds


def main():
    OUT.mkdir(exist_ok=True)
    tables, summary, n_seeds = build()

    with pd.ExcelWriter(OUT / "results.xlsx", engine="openpyxl") as xl:
        for name, df in tables.items():
            df.to_excel(xl, sheet_name=name[:31], index=False)

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2),
                                      encoding="utf-8")

    lines = ["# Result tables (generated — do not edit)",
             "",
             f"Regenerated from `data/*.csv` by `export_report.py`. "
             f"{n_seeds} seeds, {summary['n_generations']:,} generations.",
             ""]
    for name in ("auroc_FalsePremise", "auroc_TriviaQA", "cost", "label_counts"):
        if name not in tables:
            continue
        lines += [f"## {name}", "", tables[name].to_markdown(index=False), ""]
    lines += ["## Headline", "", "```json", json.dumps(summary, indent=2), "```"]
    (OUT / "TABLES.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {OUT / 'results.xlsx'}  ({len(tables)} sheets)")
    print(f"wrote {OUT / 'summary.json'}")
    print(f"wrote {OUT / 'TABLES.md'}")
    for fam in ("FalsePremise", "TriviaQA"):
        if fam in summary:
            s = summary[fam]
            print(f"  {fam}: FIS {s['fis_matched_mean']:.3f} ± "
                  f"{s['fis_matched_std']:.3f} vs {s['best_rival']} "
                  f"{s['best_rival_matched_mean']:.3f} | Δ {s['paired_delta_mean']:+.3f} "
                  f"| {s['seeds_won']}/{s['seeds_total']} seeds")


if __name__ == "__main__":
    main()
