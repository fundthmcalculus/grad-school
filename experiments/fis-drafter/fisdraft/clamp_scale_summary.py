"""Synthesize the clamp scaling study across models: where does the
exogenous-response subspace live, and is its dimensionality universal?

Reads clamp_scan.json from several stream runs and reports, per model:
  peak causal layer (as a fraction of depth), the centrality plateau (band of
  depth where centrality is >= 70% of peak), and an effective-rank estimate from
  the k-sweep behav_KL knee (smallest k reaching >= half the k=32 KL). Also emits
  tidy plot rows: centrality vs fractional depth, one series per model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def eff_rank(ksweep: dict) -> int:
    ks = sorted(int(k) for k in ksweep)
    kl = (
        {int(k): ksweep[str(k)]["clamp"]["behav_KL"] for k in ks}
        if all(isinstance(k, str) for k in ksweep)
        else {int(k): v["clamp"]["behav_KL"] for k, v in ksweep.items()}
    )
    kmax = max(ks)
    half = 0.5 * kl[kmax]
    for k in ks:
        if kl[k] >= half:
            return k
    return kmax


def summarize(runs):
    rows, plot = [], []
    for rd in runs:
        p = Path(rd) / "clamp_scan.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        c = {int(L): v for L, v in r["causal_centrality"].items()}
        nL = r["n_layers"] + 1
        peak = r["peak_layer"]
        peakv = c[peak]
        # a causal subspace exists only if centrality is clearly positive; when the
        # fitted subspace is no more causal than random (peak <= ~0), there is none.
        has_subspace = peakv > 0.05
        band = [L for L, v in c.items() if v >= 0.7 * peakv] if has_subspace else []
        er = eff_rank(r["k_sweep_at_peak"]) if has_subspace else None
        rows.append(
            {
                "model": r["model"].split("/")[-1],
                "n_layers": r["n_layers"],
                "hidden": r["hidden"],
                "subspace": "yes" if has_subspace else "NONE (<=random)",
                "peak_centrality": round(peakv, 3),
                "peak_layer": peak if has_subspace else None,
                "peak_frac": round(peak / nL, 2) if has_subspace else None,
                "plateau_frac": (
                    f"{min(band)/nL:.2f}-{max(band)/nL:.2f}" if band else "-"
                ),
                "eff_rank": er,
                "eff_rank_frac_of_d": round(er / r["hidden"], 4) if er else None,
            }
        )
        for L, v in c.items():
            plot.append(
                {
                    "model": r["model"].split("/")[-1],
                    "frac_depth": round(L / nL, 3),
                    "centrality": round(v, 3),
                }
            )
    return rows, plot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--runs",
        nargs="+",
        default=[
            "runs/stream_smol360",
            "runs/stream_qwen1p5b",
            "runs/stream_qwen3b",
            "runs/stream_qwen7b",
        ],
    )
    a = ap.parse_args()
    rows, plot = summarize(a.runs)
    Path("runs/clamp_scale_summary.json").write_text(
        json.dumps({"rows": rows, "plot": plot}, indent=2)
    )
    print(
        f"{'model':<26}{'nL':>4}{'d':>6}{'subspace':>18}{'peakC':>7}"
        f"{'peak%':>7}{'plateau%depth':>15}{'effRank':>8}"
    )
    for r in rows:
        pf = f"{r['peak_frac']:.2f}" if r["peak_frac"] is not None else "-"
        print(
            f"{r['model']:<26}{r['n_layers']:>4}{r['hidden']:>6}{r['subspace']:>18}"
            f"{r['peak_centrality']:>7.2f}{pf:>7}{r['plateau_frac']:>15}"
            f"{str(r['eff_rank']):>8}"
        )


if __name__ == "__main__":
    main()
