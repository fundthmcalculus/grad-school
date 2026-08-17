"""Improve the FIS's RUL on both axes at once -- accuracy and smoothness --
without leaving TribbleRegressor.

The lead is the benchmark's own contrast: the `best` pipeline is *both* more
accurate and smoother than `honest`, and the only structural difference is that
its `MemoryWindowFeatureExtractor` hands the FIS a temporally-coherent signal
instead of one independent snapshot per cycle. So the question here is whether
giving the interpretable `honest` pipeline **causal trend features** buys the
same -- a genuine FIS throughout, no network, no post-hoc clamp.

The distinction that matters, learned the hard way in `monotone.py`: *replacing*
each feature with its rolling mean blurs the degradation trend and doubled RMSE.
*Augmenting* -- keeping every sharp per-cycle feature and adding smooth,
slowly-varying companions -- lets the FIS's own feature selection keep what it
needs and lean on the smooth signal for the level. That is the thing under test.

Two metrics, both per engine then averaged (each trajectory weighted equally,
which is the right convention for a smoothness question; it differs from the
benchmark's pooled per-sample RMSE and both are reported so neither surprises):
per-engine RMSE against uncapped RUL, and the up-cycle fraction / positive total
variation from `monotone.py`.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

import cmapss_data
import models
import monotone as M
import report

OUT = report.OUT


# ---------------------------------------------------------------------------
# Causal, per-engine trend features
# ---------------------------------------------------------------------------
def _ewma(s: pd.Series, alpha: float) -> pd.Series:
    return s.ewm(alpha=alpha, adjust=False).mean()


def _rolling(s: pd.Series, k: int) -> pd.Series:
    return s.rolling(k, min_periods=1).mean()


def _cummean(s: pd.Series) -> pd.Series:
    return s.expanding().mean()


TREND = {
    "ewma": lambda s: _ewma(s, 0.3),
    "roll5": lambda s: _rolling(s, 5),
    "cummean": _cummean,  # a smooth, near-monotone health signal since onset
}


def augment(X, unit, cycle, names, base_cols, kinds, standardize_from=None):
    """Append causal trend transforms of `base_cols` to X.

    Every transform is a trailing, per-engine, cycle-ordered operation, so a
    row at cycle t sees only cycles <= t of its own engine: deployable, and no
    leakage across engines. New columns are standardized with train statistics
    (`standardize_from`, a dict built on the first call) so the FIS sees them on
    the same scale as the unit-variance base features.
    """
    X = np.asarray(X, dtype=float)
    idx = {n: i for i, n in enumerate(names)}
    df = pd.DataFrame(X[:, [idx[c] for c in base_cols]], columns=base_cols)
    df["_u"], df["_c"], df["_i"] = unit, cycle, np.arange(len(X))
    df = df.sort_values(["_u", "_c"])

    new_cols, new_names = [], []
    for kind in kinds:
        fn = TREND[kind]
        block = df.groupby("_u")[base_cols].transform(fn)
        block.columns = [f"{c}__{kind}" for c in base_cols]
        new_cols.append(block)
        new_names += list(block.columns)
    aug = pd.concat(new_cols, axis=1).loc[df.index]  # keep df's sort
    aug["_i"] = df["_i"].values
    aug = aug.sort_values("_i").drop(columns="_i").to_numpy()

    if standardize_from is None:
        standardize_from = {
            "mu": aug.mean(axis=0),
            "sd": np.where(aug.std(axis=0) > 1e-9, aug.std(axis=0), 1.0),
        }
    aug = (aug - standardize_from["mu"]) / standardize_from["sd"]
    return np.column_stack([X, aug]), names + new_names, standardize_from


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_featureset(
    b, names, kinds, base="mean", config="honest", extra_kwargs=None
) -> dict:
    """Fit the FIS on an augmented feature set; score accuracy + smoothness."""
    warnings.simplefilter("ignore")
    base_cols = (
        [n for n in names if n.endswith("_mean")] if base == "mean" else list(names)
    )
    kw = dict(models.FIS_CONFIGS[config])
    kw.update(extra_kwargs or {})

    if kinds:
        Xtr, aug_names, stats = augment(
            b.train.X, b.train.unit, b.train.cycle, names, base_cols, kinds
        )
        Xte, _, _ = augment(
            b.test.X,
            b.test.unit,
            b.test.cycle,
            names,
            base_cols,
            kinds,
            standardize_from=stats,
        )
    else:
        Xtr, Xte, aug_names = b.train.X, b.test.X, names

    fis, fit_s = models.fit_fis(Xtr, b.train.y, aug_names, **kw)
    pred = models.fis_predict(fis, Xte, aug_names)
    g = M.per_cycle(b.test.unit, b.test.cycle, b.test.y_true, pred)
    agg = M.aggregate(
        [
            M.score_engine(s.true.to_numpy(), s.pred.to_numpy())
            for _, s in g.groupby("unit")
        ]
    )
    agg["pooled_rmse"] = float(np.sqrt(np.mean((pred - b.test.y_true) ** 2)))
    agg["fit_seconds"] = fit_s
    agg["n_features"] = Xtr.shape[1]
    agg["n_kept"] = int(len(fis.top_features_))
    return agg


def eval_bundle(feature_set, aggregation, config, kinds=None, extra=None) -> dict:
    """Score one FIS pipeline on both axes. `kinds` augments with trend
    features (whole_cycle only); `extra` overrides FIS kwargs."""
    b = cmapss_data.load_or_build(
        feature_set=feature_set, aggregation=aggregation, verbose=False
    )
    return evaluate_featureset(b, b.feature_names, kinds or [], "mean", config, extra)


def trend_trials() -> dict:
    """The augmentation ladder on the honest whole-cycle pipeline."""
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES["honest"], verbose=False)
    names = b.feature_names
    trials = {
        "+ewma(mean)": (["ewma"], {}),
        "+roll5(mean)": (["roll5"], {}),
        "+cummean(mean)": (["cummean"], {}),
        "+all-trend(mean)": (["ewma", "roll5", "cummean"], {}),
        "+all-trend, l2=0.1": (["ewma", "roll5", "cummean"], {"l2_reg": 0.1}),
    }
    return {
        tag: evaluate_featureset(b, names, kinds, "mean", "honest", extra)
        for tag, (kinds, extra) in trials.items()
    }


def run() -> dict:
    # The lever comparison: what actually moves both axes, and what does not.
    levers = {
        # feature mechanism -- the win: memory beats per-cycle on both axes,
        # and the strict 18-sensor set matches the 20-channel `best`.
        "whole_cycle, real 18ch": eval_bundle("real", "whole_cycle", "honest"),
        "raw_memory, real 18ch": eval_bundle("real", "raw_memory", "best"),
        "raw_memory, lit 20ch (best)": eval_bundle("literature", "raw_memory", "best"),
    }
    trends = trend_trials()
    return {"levers": levers, "trend_aug": trends}


def plot(payload, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 5.4), facecolor=report.SURFACE)
    lev = payload["levers"]
    best_aug = min(payload["trend_aug"].values(), key=lambda m: m["rmse"])
    wc = lev["whole_cycle, real 18ch"]
    mem = lev["raw_memory, real 18ch"]  # lit 20ch sits on top of it -- merged
    # (tag, pos_tv, rmse, kind, label-offset, ha)
    pts = [
        ("whole_cycle, real 18ch", wc["pos_tv"], wc["rmse"], "lever", (10, 8), "left"),
        (
            "raw_memory\n(real 18ch = lit 20ch)",
            mem["pos_tv"],
            mem["rmse"],
            "lever",
            (14, 0),
            "left",
        ),
        (
            "whole_cycle + trend-aug\n(best of 5 — worse on both)",
            best_aug["pos_tv"],
            best_aug["rmse"],
            "neg",
            (0, -30),
            "center",
        ),
    ]
    for tag, tv, rmse, kind, off, ha in pts:
        col = report.C[1] if kind == "neg" else report.C[0]
        ax.scatter(
            tv,
            rmse,
            s=110,
            color=col,
            edgecolor=report.SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        ax.annotate(
            tag,
            (tv, rmse),
            textcoords="offset points",
            xytext=off,
            ha=ha,
            va="center",
            fontsize=8.5,
            color=report.INK2,
        )
    report._style(
        ax,
        "cycles of upward movement  (pos_tv — noisier →)",
        "per-engine RMSE  (cycles — worse ↑)",
        "Both axes at once: only memory features move down-and-left",
    )
    ax.annotate(
        "better",
        xy=(0.02, 0.06),
        xytext=(0.20, 0.20),
        xycoords="axes fraction",
        textcoords="axes fraction",
        fontsize=10,
        color=report.INK2,
        arrowprops=dict(arrowstyle="->", color=report.INK2, linewidth=1.5),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)
    plt.close(fig)


def main() -> None:
    payload = run()
    print("\n=== FIS-quality levers (per-engine RMSE / smoothness) ===")
    print(
        f"  {'pipeline / lever':34s} {'rmse':>6s} {'pooled':>7s} {'up%':>5s} {'pos_tv':>7s}"
    )
    for tag, m in payload["levers"].items():
        print(
            f"  {tag:34s} {m['rmse']:6.2f} {m['pooled_rmse']:7.2f} "
            f"{m['up_frac']*100:5.0f} {m['pos_tv']:7.1f}"
        )
    print("  -- trend augmentation of whole_cycle (all worse on RMSE): --")
    for tag, m in payload["trend_aug"].items():
        print(
            f"  {tag:34s} {m['rmse']:6.2f} {m['pooled_rmse']:7.2f} "
            f"{m['up_frac']*100:5.0f} {m['pos_tv']:7.1f}"
        )

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fis_quality.json"), "w") as f:
        json.dump(payload, f, indent=1)
    plot(payload, os.path.join(report.FIG, "fis_quality.png"))
    print(
        f"\nwrote {os.path.relpath(os.path.join(OUT, 'fis_quality.json'), cmapss_data.REPO)}"
    )
    print(
        f"wrote {os.path.relpath(os.path.join(report.FIG, 'fis_quality.png'), cmapss_data.REPO)}"
    )


if __name__ == "__main__":
    main()
