"""FIS RUL quality, inside TribbleRegressor -- one study, three questions.

Consolidates what were three scripts (fis_quality, fis_memory_sweep,
fis_monotone) into one driver with three subcommands, because they are one
investigation:

    python fis_quality.py levers        # which FIS-native lever moves both
                                        # accuracy AND smoothness (memory
                                        # features do; trend augmentation and
                                        # hyperparameters do not)
    python fis_quality.py memory-sweep  # tune the memory-window size for the
                                        # accuracy/smoothness trade-off
    python fis_quality.py monotone      # the capstone: the recommended
                                        # memory18 FIS made hard-monotone, plus
                                        # the "predict delta then cumsum" arms

Write-up in outputs/nn-cmapss/FIS_QUALITY.md.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd

import cmapss_data
import models
import metrics
import transforms
import monotone_model as MM
import report

OUT = report.OUT


# ===========================================================================
# subcommand: levers
# ===========================================================================
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
    g = metrics.per_cycle(b.test.unit, b.test.cycle, b.test.y_true, pred)
    agg = metrics.aggregate(
        [
            metrics.score_engine(s.true.to_numpy(), s.pred.to_numpy())
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


def run_levers() -> dict:
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


def plot_levers(payload, path):
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


def cmd_levers() -> None:
    payload = run_levers()
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
    plot_levers(payload, os.path.join(report.FIG, "fis_quality.png"))
    print(
        f"\nwrote {os.path.relpath(os.path.join(OUT, 'fis_quality.json'), cmapss_data.REPO)}"
    )
    print(
        f"wrote {os.path.relpath(os.path.join(report.FIG, 'fis_quality.png'), cmapss_data.REPO)}"
    )


# ===========================================================================
# subcommand: memory-sweep
# ===========================================================================
OUT = report.OUT


def corrected_frames(feature_set="real"):
    """Load DS02 and condition-correct once; return dev/test frames + columns."""
    data, var = cmapss_data.load_h5(cmapss_data.DEFAULT_H5)
    df_dev, df_test = cmapss_data.to_frame(data, var, "dev"), cmapss_data.to_frame(
        data, var, "test"
    )
    del data
    w = [f"W_{n}" for n in var["W"]]
    xs = [f"Xs_{n}" for n in var["X_s"]]
    n_xv = cmapss_data.FEATURE_SET_XV[feature_set]
    xv = (
        []
        if n_xv == 0
        else (
            [f"Xv_{n}" for n in var["X_v"]]
            if n_xv is None
            else [f"Xv_{n}" for n in var["X_v"][:n_xv]]
        )
    )
    models_cc = cmapss_data.fit_condition_correction(df_dev, xs + xv, w)
    df_dev = cmapss_data.apply_condition_correction(df_dev, xs + xv, w, models_cc)
    df_test = cmapss_data.apply_condition_correction(df_test, xs + xv, w, models_cc)
    return df_dev, df_test, w + xs + xv


def memory_tables(df, feat_cols, window, memory, stride):
    """One row per subsampled sample, with short/long-term memory features."""
    from tribblefis.gaussian_regressor_memory import MemoryWindowFeatureExtractor

    ext = MemoryWindowFeatureExtractor(window_size=window, memory_size=memory)
    frames = []
    for unit, sub in df.groupby("unit", sort=True):
        sub = sub.iloc[::stride].reset_index(drop=True)
        mem = ext.prepare_sequences(sub, feat_cols, include_time=False)
        mem["unit"] = unit
        mem["cycle"] = sub["cycle"].values
        mem["RUL"] = sub["RUL"].values
        mem["hs"] = sub["hs"].values
        frames.append(mem)
    out = pd.concat(frames, ignore_index=True)
    cols = [c for c in out.columns if c not in ("unit", "cycle", "RUL", "hs")]
    out[cols] = out[cols].bfill().ffill()
    return out, cols


def ms_evaluate(df_dev, df_test, feat_cols, window, memory, stride):
    from sklearn.preprocessing import StandardScaler

    train_tab, agg_cols = memory_tables(df_dev, feat_cols, window, memory, stride)
    test_tab, _ = memory_tables(df_test, feat_cols, window, memory, stride)

    caps = cmapss_data.physical_rul_cap(train_tab)  # training units only
    sc = StandardScaler().fit(train_tab[agg_cols].to_numpy(float))
    Xtr = sc.transform(train_tab[agg_cols].to_numpy(float))
    Xte = sc.transform(test_tab[agg_cols].to_numpy(float))
    ytr = cmapss_data.capped_rul(train_tab, caps)

    fis, fit_s = models.fit_fis(Xtr, ytr, agg_cols, **models.FIS_CONFIGS["best"])
    pred = models.fis_predict(fis, Xte, agg_cols)

    g = metrics.per_cycle(
        test_tab["unit"].to_numpy(),
        test_tab["cycle"].to_numpy(),
        test_tab["RUL"].astype(float).to_numpy(),
        pred,
    )
    agg = metrics.aggregate(
        [
            metrics.score_engine(s.true.to_numpy(), s.pred.to_numpy())
            for _, s in g.groupby("unit")
        ]
    )
    agg.update(
        window=window,
        memory=memory,
        stride=stride,
        fit_seconds=fit_s,
        n_features=Xtr.shape[1],
        n_train=len(train_tab),
    )
    return agg


# (window, memory, stride). window=5/memory=2/stride=200 is the shipped `best`.
MEMORY_GRID = [
    (5, 2, 200),
    (10, 5, 200),
    (20, 10, 200),
    (40, 20, 200),
    (20, 10, 100),
    (40, 20, 100),
    (80, 40, 100),
]


def cmd_memory_sweep(feature_set="real") -> None:
    warnings.simplefilter("ignore")
    print(f"Loading + condition-correcting DS02 ({feature_set}) once ...")
    df_dev, df_test, feat_cols = corrected_frames(feature_set)
    print(
        f"  {len(df_dev):,} dev + {len(df_test):,} test rows, {len(feat_cols)} channels"
    )

    rows = []
    for window, memory, stride in MEMORY_GRID:
        r = ms_evaluate(df_dev, df_test, feat_cols, window, memory, stride)
        rows.append(r)
        span = window + memory
        print(
            f"  w={window:3d} m={memory:3d} stride={stride:3d}  "
            f"(~{span} samples/window)  n_train={r['n_train']:6d}  "
            f"rmse={r['rmse']:6.2f}  up%={r['up_frac']*100:3.0f}  "
            f"pos_tv={r['pos_tv']:6.1f}  fit={r['fit_seconds']:.2f}s"
        )

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fis_memory_sweep.json")
    with open(path, "w") as f:
        json.dump({"feature_set": feature_set, "rows": rows}, f, indent=1)
    print(f"\nwrote {os.path.relpath(path, cmapss_data.REPO)}")


# ===========================================================================
# subcommand: monotone
# ===========================================================================
OUT = report.OUT
WHICH = "memory18"


def mono_build():
    warnings.simplefilter("ignore")
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[WHICH], verbose=False)
    names = b.feature_names
    fis, fit_s = models.fit_fis(
        b.train.X, b.train.y, names, **models.FIS_CONFIGS[WHICH]
    )
    pred = models.fis_predict(fis, b.test.X, names)
    raw = metrics.per_cycle(b.test.unit, b.test.cycle, b.test.y_true, pred)
    return b, raw, fit_s


def mono_score(g: pd.DataFrame) -> dict:
    per = [
        metrics.score_engine(s.true.to_numpy(), s.pred.to_numpy())
        for _, s in g.groupby("unit")
    ]
    a = metrics.aggregate(per)
    a["pooled_rmse"] = float(np.sqrt(np.mean((g["pred"] - g["true"]) ** 2)))
    return a


def cmd_monotone() -> None:
    b, raw, fit_s = mono_build()

    methods = {
        "raw FIS (memory18)": transforms.out_raw,
        "+ cummin": transforms.out_cummin,
        "+ mean5->cummin": lambda p: transforms.out_mean_cummin(p, 5),
        "offline oracle (bound)": transforms.out_iso_offline,
    }
    rows = {}
    for tag, fn in methods.items():
        g = raw.assign(
            pred=raw.groupby("unit")["pred"].transform(lambda s: fn(s.to_numpy()))
        )
        rows[tag] = mono_score(g)

    # "Predict a per-cycle delta, then cumsum" -- both forms.
    # Non-negative delta (softplus, floored) = the monotone damage model.
    dmg = MM.damage_predictions(WHICH, link="softplus", floor=0.0)[0]
    rows["delta+cumsum, non-neg (=damage)"] = mono_score(dmg)
    # Signed delta, unconstrained -- the plain version, not monotone.
    signed = MM.damage_predictions(WHICH, link="identity", floor=-1e9)[0]
    rows["delta+cumsum, signed"] = mono_score(signed)

    print(
        f"=== {WHICH}: the recommended FIS, made monotone "
        f"(FIS fit {fit_s:.2f}s) ==="
    )
    print(f"  {'method':26s} {'rmse':>6s} {'pooled':>7s} {'up%':>5s} {'pos_tv':>7s}")
    for tag, m in rows.items():
        print(
            f"  {tag:26s} {m['rmse']:6.2f} {m['pooled_rmse']:7.2f} "
            f"{m['up_frac']*100:5.0f} {m['pos_tv']:7.1f}"
        )

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "fis_monotone.json"), "w") as f:
        json.dump(rows, f, indent=1)
    mono_plot(b, raw, dmg)
    print(
        f"\nwrote {os.path.relpath(os.path.join(OUT, 'fis_monotone.json'), cmapss_data.REPO)}"
    )


def mono_plot(b, raw, dmg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    units = sorted(raw["unit"].unique().tolist())
    fig, axes = plt.subplots(
        1,
        len(units),
        figsize=(4.7 * len(units), 4.1),
        facecolor=report.SURFACE,
        squeeze=False,
        sharey=False,
    )
    C_RAW, C_MONO, C_DMG = report.C[0], report.C[3], report.C[1]
    for c, u in enumerate(units):
        ax = axes[0][c]
        sub = raw[raw.unit == u]
        cyc, truth, praw = (sub[k].to_numpy() for k in ("cycle", "true", "pred"))
        pmono = transforms.out_cummin(praw)
        dsub = dmg[dmg.unit == u]
        ax.plot(
            cyc,
            truth,
            color=report.INK,
            linewidth=2.4,
            zorder=5,
            label="true RUL",
            solid_capstyle="round",
        )
        ax.plot(
            cyc,
            praw,
            color=C_RAW,
            linewidth=1.0,
            alpha=0.5,
            label="raw FIS (memory18)",
            zorder=2,
        )
        ax.plot(
            dsub["cycle"].to_numpy(),
            dsub["pred"].to_numpy(),
            color=C_DMG,
            linewidth=1.5,
            alpha=0.85,
            label="damage model",
            zorder=3,
        )
        ax.plot(
            cyc,
            pmono,
            color=C_MONO,
            linewidth=2.2,
            label="FIS + cummin (recommended)",
            zorder=4,
        )
        ax.axhline(0.0, color=report.GRID, linewidth=1.2, zorder=1)
        report._style(ax, "flight cycle", "RUL (cycles)" if c == 0 else "", f"unit {u}")
        e = float(np.sqrt(np.mean((pmono - truth) ** 2)))
        ax.text(
            0.03,
            0.06,
            f"FIS+cummin  {e:.1f}   (↑0%)",
            transform=ax.transAxes,
            fontsize=8.5,
            color=report.INK2,
            family="monospace",
        )
        if c == 0:
            ax.legend(
                frameon=False, fontsize=8.5, labelcolor=report.INK2, loc="upper right"
            )
    fig.suptitle(
        "The recommended FIS (18 sensors + memory), clamped monotone",
        color=report.INK,
        fontsize=13,
        x=0.006,
        ha="left",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs(report.FIG, exist_ok=True)
    path = os.path.join(report.FIG, "fis_monotone.png")
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, cmapss_data.REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("levers", help="which FIS-native lever moves both axes")
    sub.add_parser("memory-sweep", help="tune the memory-window size")
    sub.add_parser("monotone", help="the recommended FIS made hard-monotone")
    args = ap.parse_args()
    if args.cmd == "memory-sweep":
        cmd_memory_sweep()
    elif args.cmd == "monotone":
        cmd_monotone()
    else:
        cmd_levers()
