"""The full stack: the recommended FIS, made hard-monotone.

`fis_quality.py` settled on the `memory18` pipeline -- the strict 18 real
sensors through the memory-window extractor -- as the best FIS on both accuracy
and smoothness (6.19 per-engine RMSE, but still ~36% up-cycles). This applies
the monotone work from `monotone.py` / `monotone_model.py` to *that* pipeline,
so the end product is one model that is accurate, smooth, and monotone-
decreasing by the guarantee rather than by luck.

Everything here is on the same `memory18` FIS predictions, per test engine:
the causal clamps, the offline oracle bound, and the damage-accumulation model
fit on `memory18`'s own features.
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
import monotone_model as MM
import report

OUT = report.OUT
WHICH = "memory18"


def build():
    warnings.simplefilter("ignore")
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[WHICH], verbose=False)
    names = b.feature_names
    fis, fit_s = models.fit_fis(
        b.train.X, b.train.y, names, **models.FIS_CONFIGS[WHICH]
    )
    pred = models.fis_predict(fis, b.test.X, names)
    raw = M.per_cycle(b.test.unit, b.test.cycle, b.test.y_true, pred)
    return b, raw, fit_s


def score(g: pd.DataFrame) -> dict:
    per = [
        M.score_engine(s.true.to_numpy(), s.pred.to_numpy())
        for _, s in g.groupby("unit")
    ]
    a = M.aggregate(per)
    a["pooled_rmse"] = float(np.sqrt(np.mean((g["pred"] - g["true"]) ** 2)))
    return a


def main() -> None:
    b, raw, fit_s = build()

    methods = {
        "raw FIS (memory18)": M.out_raw,
        "+ cummin": M.out_cummin,
        "+ mean5->cummin": lambda p: M.out_mean_cummin(p, 5),
        "offline oracle (bound)": M.out_iso_offline,
    }
    rows = {}
    for tag, fn in methods.items():
        g = raw.assign(
            pred=raw.groupby("unit")["pred"].transform(lambda s: fn(s.to_numpy()))
        )
        rows[tag] = score(g)

    # "Predict a per-cycle delta, then cumsum" -- both forms.
    # Non-negative delta (softplus, floored) = the monotone damage model.
    dmg = MM.damage_predictions(WHICH, link="softplus", floor=0.0)[0]
    rows["delta+cumsum, non-neg (=damage)"] = score(dmg)
    # Signed delta, unconstrained -- the plain version, not monotone.
    signed = MM.damage_predictions(WHICH, link="identity", floor=-1e9)[0]
    rows["delta+cumsum, signed"] = score(signed)

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
    plot(b, raw, dmg)
    print(
        f"\nwrote {os.path.relpath(os.path.join(OUT, 'fis_monotone.json'), cmapss_data.REPO)}"
    )


def plot(b, raw, dmg) -> None:
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
        pmono = M.out_cummin(praw)
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
    main()
