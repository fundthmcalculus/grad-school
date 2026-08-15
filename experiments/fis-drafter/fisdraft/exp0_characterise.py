"""Experiment 0 -- is there any signal in tier-A features at all?

Settles one thing: whether the shape of the next-token distribution at step t
is predictable from its own recent history. If it is not, variant V1 is dead
and no fuzzy system needs to be built to discover that.

The control that makes this non-trivial: a raw autocorrelation of entropy
along generation is *not* evidence of step-to-step dynamics. Different prompts
sit at different average entropies, so pooling all steps together produces a
healthy-looking correlation that is entirely between-prompt variance -- the
same shape of artefact as the prompt-family confound that reached ~0.9 AUROC
in `experiments/fuzzy-lm-anomaly.md`. So every correlation here is reported
twice: pooled, and within-prompt (after removing each prompt's own mean).
The within-prompt number is the real one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SHAPE_COLS = [
    "entropy",
    "varentropy",
    "renyi2",
    "top1_prob",
    "log_margin_12",
    "nucleus_90",
    "tail_slope",
    "mass_top10",
]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return float("nan")
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def within_prompt_corr(df: pd.DataFrame, x: str, y: str) -> float:
    """Correlation after removing each prompt's own mean from both columns."""
    g = df.groupby("prompt_id")
    xa = (df[x] - g[x].transform("mean")).to_numpy()
    ya = (df[y] - g[y].transform("mean")).to_numpy()
    return _pearson(xa, ya)


def variance_decomposition(df: pd.DataFrame, col: str) -> dict:
    """How much of this statistic's variance is between prompts vs within?

    If nearly all of it is between-prompt, a 'predictor' can score well by
    learning the prompt and nothing about the dynamics.
    """
    g = df.groupby("prompt_id")[col]
    grand = df[col].mean()
    n = g.count()
    between = float((n * (g.mean() - grand) ** 2).sum() / len(df))
    total = float(df[col].var(ddof=0))
    return {
        "total_var": total,
        "between_prompt_frac": between / total if total > 0 else float("nan"),
    }


def run(rundir: Path) -> dict:
    df = pd.read_parquet(rundir / "steps.parquet")
    out: dict = {"n_rows": int(len(df)), "n_prompts": int(df.prompt_id.nunique())}

    # ---- marginal distribution of entropy -------------------------------
    e = df.entropy.to_numpy()
    out["entropy"] = {
        "mean": float(e.mean()),
        "std": float(e.std()),
        "quantiles": {
            str(q): float(np.quantile(e, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)
        },
        "frac_below_0.1_nats": float((e < 0.1).mean()),
        "frac_above_3_nats": float((e > 3.0).mean()),
    }
    # Fraction of steps that are near-deterministic matters directly: those are
    # the steps any drafter gets for free, and they inflate every average.
    out["top1_prob"] = {
        "mean": float(df.top1_prob.mean()),
        "frac_above_0.9": float((df.top1_prob > 0.9).mean()),
        "frac_above_0.5": float((df.top1_prob > 0.5).mean()),
    }

    # ---- persistence: pooled vs within-prompt ---------------------------
    persistence = {}
    for col in SHAPE_COLS:
        row = {"var_decomp": variance_decomposition(df, col)}
        for lag in (1, 2, 3):
            pc = f"prev{lag}_{col}" if f"prev{lag}_{col}" in df.columns else None
            if pc is None:
                continue
            row[f"pooled_lag{lag}"] = _pearson(df[col].to_numpy(), df[pc].to_numpy())
            row[f"within_lag{lag}"] = within_prompt_corr(df, col, pc)
        persistence[col] = row
    out["persistence"] = persistence

    # ---- does the position in the sequence explain it? -------------------
    out["step_effects"] = {
        "corr_entropy_step": _pearson(
            df.entropy.to_numpy(), df.step.to_numpy().astype(float)
        ),
        "entropy_by_step_decile": (
            df.groupby(pd.qcut(df.step, 10, duplicates="drop"), observed=True)
            .entropy.mean()
            .round(4)
            .to_dict()
        ),
    }
    out["step_effects"]["entropy_by_step_decile"] = {
        str(k): v for k, v in out["step_effects"]["entropy_by_step_decile"].items()
    }

    # ---- regime separation on the synthetic probes -----------------------
    syn = df[df.source == "synthetic"]
    if len(syn):
        out["synthetic_regimes"] = (
            syn.groupby("category")[
                ["entropy", "top1_prob", "nucleus_90", "tail_slope"]
            ]
            .agg(["mean", "std", "count"])
            .round(4)
            .to_dict()
        )
        out["synthetic_regimes"] = {
            f"{a}|{b}": v for (a, b), v in out["synthetic_regimes"].items()
        }

    # ---- is the tail exponent actually stable? ---------------------------
    # The smoke run hinted tail_slope sits near -1.6 regardless of regime. If
    # that holds it is a finding in itself (Zipfian tail with a near-invariant
    # exponent) AND it means tail_slope carries little predictive signal --
    # a feature worth dropping rather than modelling.
    out["tail"] = {
        "slope_mean": float(df.tail_slope.mean()),
        "slope_std": float(df.tail_slope.std()),
        "slope_iqr": [
            float(np.quantile(df.tail_slope, 0.25)),
            float(np.quantile(df.tail_slope, 0.75)),
        ],
        "fit_resid_mean": float(df.tail_fit_resid.mean()),
        "between_prompt_frac": variance_decomposition(df, "tail_slope")[
            "between_prompt_frac"
        ],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rundir = Path(a.run)
    res = run(rundir)
    dest = Path(a.out) if a.out else rundir / "exp0_characterise.json"
    dest.write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
