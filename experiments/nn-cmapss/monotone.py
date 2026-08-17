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
import report

OUT = report.OUT


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def monotonicity(pred: np.ndarray) -> dict:
    """How far a per-engine prediction sequence is from monotone-decreasing.

    `pred` is ordered by ascending cycle. A perfect predictor of a quantity
    that only ever falls has every one of these at zero.
    """
    d = np.diff(np.asarray(pred, dtype=float))
    up = d[d > 0.0]
    return dict(
        up_frac=float((d > 0.0).mean()) if d.size else 0.0,
        pos_tv=float(up.sum()),  # total cycles of "RUL went up"
        max_up=float(up.max()) if up.size else 0.0,
    )


def score_engine(true: np.ndarray, pred: np.ndarray) -> dict:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    m = monotonicity(pred)
    m["rmse"] = float(np.sqrt(np.mean((pred - true) ** 2)))
    m["mae"] = float(np.mean(np.abs(pred - true)))
    return m


# ---------------------------------------------------------------------------
# Output transforms -- applied to the prediction sequence, per engine
# ---------------------------------------------------------------------------
def _antitonic(y: np.ndarray) -> np.ndarray:
    """L2-optimal monotone *non-increasing* fit (pool-adjacent-violators)."""
    from sklearn.isotonic import IsotonicRegression

    x = np.arange(len(y))
    return IsotonicRegression(increasing=False).fit(x, y).predict(x)


def out_raw(p):
    return np.asarray(p, dtype=float)


def out_mean(p, k):
    return pd.Series(p).rolling(k, min_periods=1).mean().to_numpy()


def out_ewma(p, alpha):
    return pd.Series(p).ewm(alpha=alpha, adjust=False).mean().to_numpy()


def out_cummin(p):
    """RUL revised only downward: the running minimum. Hard-monotone, causal --
    but a single early low outlier pins the whole trajectory under it."""
    return np.minimum.accumulate(np.asarray(p, dtype=float))


def out_ewma_cummin(p, alpha=0.3):
    """Smooth first, then clamp to non-increasing. The causal method that gets
    hard monotonicity without letting one raw outlier set the floor."""
    return np.minimum.accumulate(out_ewma(p, alpha))


def out_mean_cummin(p, k=5):
    """Trailing mean, then clamp to non-increasing. The recommended causal
    hard-monotone estimator: on the noisy `honest` pipeline it costs only
    +0.2 RMSE over raw (against +0.8 for the ewma variant and +7 for a bare
    running min), because a short symmetric-within-window average knocks down
    the spikes a running min would otherwise adopt as its floor."""
    return np.minimum.accumulate(out_mean(p, k))


def out_iso_causal(p):
    """Antitonic regression re-fit on cycles 0..t at each t, reported at t.

    Deployable: every fit sees only the past. More robust than a running min
    because pooling averages an outlier against its neighbours instead of
    adopting it as the floor.
    """
    p = np.asarray(p, dtype=float)
    return np.array([_antitonic(p[: t + 1])[-1] for t in range(len(p))])


def out_iso_offline(p):
    """Antitonic regression over the whole trajectory. NOT causal -- the L2-best
    monotone fit, and thus the bound every causal method above is chasing."""
    return _antitonic(np.asarray(p, dtype=float))


OUTPUT_METHODS = {
    "raw": (out_raw, "causal", "output"),
    "mean_k5": (lambda p: out_mean(p, 5), "causal", "output"),
    "ewma_0.3": (lambda p: out_ewma(p, 0.3), "causal", "output"),
    "cummin": (out_cummin, "causal", "output"),
    "ewma_cummin": (out_ewma_cummin, "causal", "output"),
    "mean5_cummin": (lambda p: out_mean_cummin(p, 5), "causal", "output"),
    "iso_causal": (out_iso_causal, "causal", "output"),
    "iso_offline": (out_iso_offline, "oracle", "output"),
}

# The one recommended for deployment, referenced by the plot and report.
RECOMMENDED = "mean5_cummin"


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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def per_cycle(unit, cycle, true, pred) -> pd.DataFrame:
    """One row per (unit, cycle); RUL is constant within a cycle."""
    df = pd.DataFrame({"unit": unit, "cycle": cycle, "true": true, "pred": pred})
    return (
        df.groupby(["unit", "cycle"], as_index=False)
        .mean()
        .sort_values(["unit", "cycle"])
    )


def aggregate(per_engine: list[dict]) -> dict:
    """Mean over engines -- each engine is one trajectory, weighted equally."""
    keys = ("up_frac", "pos_tv", "max_up", "rmse", "mae")
    return {k: float(np.mean([e[k] for e in per_engine])) for k in keys}


def apply_output(g: pd.DataFrame, fn) -> list[dict]:
    out = []
    for _, sub in g.groupby("unit"):
        out.append(score_engine(sub["true"].to_numpy(), fn(sub["pred"].to_numpy())))
    return out


def run(which: str) -> dict:
    warnings.simplefilter("ignore")
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which], verbose=False)
    names = b.feature_names
    kw = models.FIS_CONFIGS[which]

    # Raw predictions, once.
    fis, fit_s = models.fit_fis(b.train.X, b.train.y, names, **kw)
    pred = models.fis_predict(fis, b.test.X, names)
    g = per_cycle(b.test.unit, b.test.cycle, b.test.y_true, pred)

    results = {}
    for tag, (fn, causal, side) in OUTPUT_METHODS.items():
        agg = aggregate(apply_output(g, fn))
        results[tag] = dict(**agg, causal=causal, side=side, fit_seconds=fit_s)

    # Source-side: smooth inputs, refit, predict. Train is smoothed too, so the
    # model is trained on the same signal it will see -- otherwise the refit
    # would be predicting a smoothed input with a raw-input model.
    for k in (5, 10):
        Xtr = smooth_features(b.train.X, b.train.unit, b.train.cycle, k)
        Xte = smooth_features(b.test.X, b.test.unit, b.test.cycle, k)
        f2, s2 = models.fit_fis(Xtr, b.train.y, names, **kw)
        p2 = models.fis_predict(f2, Xte, names)
        g2 = per_cycle(b.test.unit, b.test.cycle, b.test.y_true, p2)
        agg = aggregate(apply_output(g2, out_raw))
        results[f"input_smooth_k{k}"] = dict(
            **agg, causal="causal", side="source (refit)", fit_seconds=s2
        )
        # ...and input smoothing composed with the recommended output monotone.
        agg2 = aggregate(apply_output(g2, lambda p: out_mean_cummin(p, 5)))
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
        runs.append((which, per_cycle(b.test.unit, b.test.cycle, b.test.y_true, pred)))
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
            pmono = out_mean_cummin(praw, 5)
            porc = out_iso_offline(praw)
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
            raw_m = monotonicity(praw)
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
