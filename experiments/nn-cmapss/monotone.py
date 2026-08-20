"""How noisy is the FIS's RUL, and what makes it near-monotonically decreasing?

True RUL falls by exactly one every cycle, so *any* cycle where the prediction
rises is pure noise. The raw FIS rises on ~40% of cycles on the `honest`
pipeline (see the baseline row below). This measures both axes -- accuracy and
monotonicity -- for a ladder of candidate fixes, and keeps two distinctions the
choice actually turns on:

* **Source vs. output.** Smoothing the *inputs* the FIS sees reduces the noise
  it generates; smoothing its *output* masks noise after the fact. The former
  is the real answer to "reduce the noisiness of the FIS"; the latter is the
  cheap one. Both are measured, and labelled.

* **Causal vs. oracle.** Onboard, RUL at cycle t may use only cycles <= t, so a
  transform that looks at the whole trajectory is not deployable -- it is an
  upper bound on what any monotone post-processing could achieve, not a
  candidate. Every method is tagged `causal` or `oracle`.

The scoring convention matches the rest of this experiment: per-sample RMSE
against uncapped ground-truth RUL, on the three official DS02 test engines,
after collapsing to one row per (unit, cycle) because RUL is per-cycle.

This is a *driver*: the metrics it uses live in `metrics.py` and the monotone
operators in `transforms.py`; it only orchestrates and reports.
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
import report

OUT = report.OUT


# ---------------------------------------------------------------------------
# Input smoothing -- the source-side fix: smooth the features, refit the FIS
# ---------------------------------------------------------------------------
def smooth_features(
    X: np.ndarray, unit: np.ndarray, cycle: np.ndarray, k: int
) -> np.ndarray:
    """Causal trailing-mean each feature within each engine, in cycle order.

    Returns a matrix aligned row-for-row with `X`. The FIS then sees a signal
    whose flight-to-flight jitter is already averaged down, so its output is
    smoother by construction rather than by post-processing -- and because the
    window is trailing, it is deployable.
    """
    df = pd.DataFrame(X)
    df["_u"], df["_c"], df["_i"] = unit, cycle, np.arange(len(X))
    df = df.sort_values(["_u", "_c"])
    feat = [c for c in df.columns if c not in ("_u", "_c", "_i")]
    df[feat] = df.groupby("_u")[feat].transform(
        lambda s: s.rolling(k, min_periods=1).mean()
    )
    return df.sort_values("_i")[feat].to_numpy()


def run(which: str) -> dict:
    warnings.simplefilter("ignore")
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which], verbose=False)
    names = b.feature_names
    kw = models.FIS_CONFIGS[which]

    # Raw predictions, once.
    fis, fit_s = models.fit_fis(b.train.X, b.train.y, names, **kw)
    pred = models.fis_predict(fis, b.test.X, names)
    g = metrics.per_cycle(b.test.unit, b.test.cycle, b.test.y_true, pred)

    results = {}
    for tag, (fn, causal, side) in transforms.OUTPUT_METHODS.items():
        agg = metrics.aggregate(transforms.apply_output(g, fn))
        results[tag] = dict(**agg, causal=causal, side=side, fit_seconds=fit_s)

    # Source-side: smooth inputs, refit, predict. Train is smoothed too, so the
    # model is trained on the same signal it will see -- otherwise the refit
    # would be predicting a smoothed input with a raw-input model.
    for k in (5, 10):
        Xtr = smooth_features(b.train.X, b.train.unit, b.train.cycle, k)
        Xte = smooth_features(b.test.X, b.test.unit, b.test.cycle, k)
        f2, s2 = models.fit_fis(Xtr, b.train.y, names, **kw)
        p2 = models.fis_predict(f2, Xte, names)
        g2 = metrics.per_cycle(b.test.unit, b.test.cycle, b.test.y_true, p2)
        agg = metrics.aggregate(transforms.apply_output(g2, transforms.out_raw))
        results[f"input_smooth_k{k}"] = dict(
            **agg, causal="causal", side="source (refit)", fit_seconds=s2
        )
        # ...and input smoothing composed with the recommended output monotone.
        agg2 = metrics.aggregate(
            transforms.apply_output(g2, lambda p: transforms.out_mean_cummin(p, 5))
        )
        results[f"input_smooth_k{k}+mean5_cummin"] = dict(
            **agg2, causal="causal", side="source+output", fit_seconds=s2
        )

    return dict(bundle=which, n_engines=int(g["unit"].nunique()), methods=results)


def to_table(res: dict) -> pd.DataFrame:
    rows = []
    for tag, m in res["methods"].items():
        rows.append(
            dict(
                method=tag,
                causal=m["causal"],
                side=m["side"],
                up_frac=m["up_frac"],
                pos_tv=m["pos_tv"],
                max_up=m["max_up"],
                rmse=m["rmse"],
                mae=m["mae"],
            )
        )
    return pd.DataFrame(rows)


def main(bundles=("honest", "best")) -> None:
    payload = {}
    for which in bundles:
        res = run(which)
        payload[which] = res
        df = to_table(res)
        raw = df[df.method == "raw"].iloc[0]
        print(
            f"\n=== {which}  (raw: up_frac {raw.up_frac:.2f}, "
            f"pos_tv {raw.pos_tv:.0f} cyc, rmse {raw.rmse:.2f}) ==="
        )
        show = df.copy()
        show["up_frac"] = show["up_frac"].map(lambda v: f"{v:.2f}")
        for c in ("pos_tv", "max_up", "rmse", "mae"):
            show[c] = show[c].map(lambda v: f"{v:.2f}")
        print(show.to_string(index=False))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "monotone.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {os.path.relpath(path, cmapss_data.REPO)}")


# ---------------------------------------------------------------------------
# Figure: raw vs recommended-causal vs oracle, on the test-engine trajectories
# ---------------------------------------------------------------------------
def plot(bundles=("honest", "best")) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    warnings.simplefilter("ignore")
    runs = []
    for which in bundles:
        b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which], verbose=False)
        names = b.feature_names
        fis, _ = models.fit_fis(
            b.train.X, b.train.y, names, **models.FIS_CONFIGS[which]
        )
        pred = models.fis_predict(fis, b.test.X, names)
        runs.append(
            (which, metrics.per_cycle(b.test.unit, b.test.cycle, b.test.y_true, pred))
        )
    units = sorted(runs[0][1]["unit"].unique().tolist())

    fig, axes = plt.subplots(
        len(runs),
        len(units),
        figsize=(4.6 * len(units), 3.9 * len(runs)),
        facecolor=report.SURFACE,
        squeeze=False,
        sharex="col",
    )
    C_RAW, C_MONO = report.C[0], report.C[1]
    for r, (tag, g) in enumerate(runs):
        for c, u in enumerate(units):
            ax = axes[r][c]
            sub = g[g.unit == u]
            cyc = sub["cycle"].to_numpy()
            truth = sub["true"].to_numpy()
            praw = sub["pred"].to_numpy()
            pmono = transforms.out_mean_cummin(praw, 5)
            porc = transforms.out_iso_offline(praw)
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
                linewidth=1.1,
                alpha=0.55,
                label="raw FIS",
                zorder=2,
            )
            ax.plot(
                cyc,
                porc,
                color=report.INK2,
                linewidth=1.3,
                alpha=0.9,
                linestyle=":",
                label="oracle (offline monotone)",
                zorder=3,
            )
            ax.plot(
                cyc,
                pmono,
                color=C_MONO,
                linewidth=2.0,
                label="mean5 -> cummin (causal)",
                zorder=4,
            )
            ax.axhline(0.0, color=report.GRID, linewidth=1.2, zorder=1)
            report._style(
                ax,
                "flight cycle" if r == len(runs) - 1 else "",
                "RUL (cycles)" if c == 0 else "",
                f"unit {u} — `{tag}`",
            )
            raw_m = metrics.monotonicity(praw)
            ax.text(
                0.03,
                0.06,
                f"raw ↑{raw_m['up_frac']:.0%}  →  causal ↑0%",
                transform=ax.transAxes,
                fontsize=8.5,
                color=report.INK2,
                family="monospace",
            )
            if r == 0 and c == 0:
                ax.legend(
                    frameon=False,
                    fontsize=8.5,
                    labelcolor=report.INK2,
                    loc="upper right",
                )
    fig.suptitle(
        "RUL, raw against near-monotone: every cycle should lose RUL, not gain it",
        color=report.INK,
        fontsize=13,
        x=0.006,
        ha="left",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    path = os.path.join(report.FIG, "monotone.png")
    os.makedirs(report.FIG, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, cmapss_data.REPO)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*", default=["honest", "best"])
    a = ap.parse_args()
    main(a.bundles)
    plot(tuple(a.bundles))
