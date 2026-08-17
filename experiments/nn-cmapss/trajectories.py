"""RUL-vs-cycle overlay: the FIS and the network on the same held-out engines.

The aggregate tables say the network wins on the cheap pipeline and ties on the
rich one. They do not say *where* the two disagree, and on a prognostics problem
that is most of the question: an RMSE is one number over a whole trajectory, but
what an operator cares about is whether the curve tracks near end of life, and
whether the model errs early (safe) or late (not).

So this plots the thing being predicted. One panel per official test engine,
ground-truth RUL against the two models' predictions, for both pipelines.
Nothing is refit beyond what the benchmark already selected: the configuration,
learning rate, stopping epoch and seed are read out of `arms_*.json`, and each
curve's RMSE is checked against the value recorded there before it is drawn --
so a plot can never quietly disagree with the table it illustrates.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import cmapss_data
import models  # noqa: F401  -- puts experiments/fis-to-neural-net on sys.path
import report

import fis2nn  # noqa: E402

OUT = report.OUT
FIG = report.FIG

# Truth is the reference, not a third category, so it is drawn in ink rather
# than taking a categorical slot: two models, two hues, in the palette's fixed
# order. Three series on one panel also sits inside the all-pairs cap.
C_FIS, C_NN = report.C[0], report.C[1]


def median_seed_row(rows: list, arm: str) -> dict:
    """The seed whose test RMSE is the median -- the run the tables quote."""
    cand = [r for r in rows if r["arm"] == arm]
    cand.sort(key=lambda r: r["final_test"]["rmse"])
    return cand[len(cand) // 2]


def rebuild(which: str, arms_file: str, arm: str = "he") -> dict:
    """Refit the FIS and one network arm exactly as `arms.py` selected them."""
    with open(os.path.join(OUT, arms_file)) as f:
        res = json.load(f)
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which], verbose=False)
    names = b.feature_names
    fis_kwargs = models.FIS_CONFIGS[res["fis_config"]]

    fis, _ = models.fit_fis(b.train.X, b.train.y, names, **fis_kwargs)
    pred_fis = models.fis_predict(fis, b.test.X, names)
    fis_rmse = models.evaluate(b.test, pred_fis)["rmse"]
    assert abs(fis_rmse - res["references"]["fis"]["test"]["rmse"]) < 1e-6, (
        f"FIS refit disagrees with {arms_file}: "
        f"{fis_rmse:.4f} vs {res['references']['fis']['test']['rmse']:.4f}"
    )

    row = median_seed_row(res["arms"], arm)
    y_center = float(np.mean(b.train.y))
    y_scale = float(np.std(b.train.y)) or 1.0
    cols = np.array([names.index(f) for f in fis.top_features_], dtype=int)
    Xtr, Xte = b.train.X[:, cols], b.test.X[:, cols]

    rng = np.random.default_rng(1000 + row["seed"])
    net = fis2nn.he_start(rng, Xtr.shape[1], res["n_hidden"])
    trained, _ = fis2nn.train_adam(
        net,
        Xtr,
        (b.train.y - y_center) / y_scale,
        epochs=int(row["selected_epoch"]),
        batch_size=res["batch_size"],
        lr=row["lr"],
        seed=row["seed"],
        track_train=False,
    )
    pred_nn = trained.predict(Xte) * y_scale + y_center
    nn_rmse = models.evaluate(b.test, pred_nn)["rmse"]
    assert abs(nn_rmse - row["final_test"]["rmse"]) < 1e-6, (
        f"{arm} refit disagrees with {arms_file}: "
        f"{nn_rmse:.4f} vs {row['final_test']['rmse']:.4f}"
    )

    return dict(
        bundle=which,
        split=b.test,
        pred_fis=pred_fis,
        pred_nn=pred_nn,
        fis_rmse=fis_rmse,
        nn_rmse=nn_rmse,
        n_hidden=res["n_hidden"],
        seed=row["seed"],
    )


def per_cycle(split, values: np.ndarray) -> pd.DataFrame:
    """Collapse to one point per (unit, cycle).

    The `raw_memory` pipeline carries ~4 rows per cycle, so a raw scatter draws
    a band rather than a trajectory. RUL is constant within a cycle by
    construction, so averaging the predictions loses nothing about the target
    and makes the two pipelines directly comparable on one axis.
    """
    df = pd.DataFrame(
        {"unit": split.unit, "cycle": split.cycle, "true": split.y_true, "pred": values}
    )
    return df.groupby(["unit", "cycle"], as_index=False).mean()


def main(arm: str = "he") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [
        ("honest", rebuild("honest", "arms_honest.json", arm)),
        ("best", rebuild("best", "arms_best.json", arm)),
    ]
    units = sorted(set(runs[0][1]["split"].unit.tolist()))

    fig, axes = plt.subplots(
        len(runs),
        len(units),
        figsize=(4.6 * len(units), 3.9 * len(runs)),
        facecolor=report.SURFACE,
        squeeze=False,
        sharex="col",
    )

    for r, (tag, run) in enumerate(runs):
        agg = per_cycle(run["split"], run["pred_fis"])
        agg_nn = per_cycle(run["split"], run["pred_nn"])
        for c, unit in enumerate(units):
            ax = axes[r][c]
            m = agg["unit"] == unit
            cyc = agg.loc[m, "cycle"].to_numpy()
            truth = agg.loc[m, "true"].to_numpy()
            p_fis = agg.loc[m, "pred"].to_numpy()
            p_nn = agg_nn.loc[agg_nn["unit"] == unit, "pred"].to_numpy()

            ax.plot(
                cyc,
                truth,
                color=report.INK,
                linewidth=2.4,
                label="true RUL",
                zorder=4,
                solid_capstyle="round",
            )
            ax.plot(cyc, p_fis, color=C_FIS, linewidth=1.6, alpha=0.95, label="FIS")
            ax.plot(
                cyc,
                p_nn,
                color=C_NN,
                linewidth=1.6,
                alpha=0.95,
                label=f"network ({run['n_hidden']} hidden)",
            )
            e_fis = float(np.sqrt(np.mean((truth - p_fis) ** 2)))
            e_nn = float(np.sqrt(np.mean((truth - p_nn) ** 2)))
            report._style(
                ax,
                "flight cycle" if r == len(runs) - 1 else "",
                "RUL (cycles)" if c == 0 else "",
                f"unit {unit} — `{tag}`",
            )
            # Per-panel error, because the aggregate RMSE hides that one engine
            # carries most of it: on `honest`, unit 14 is roughly twice the
            # other two for both models.
            ax.text(
                0.03,
                0.06,
                f"FIS {e_fis:.1f}   net {e_nn:.1f}",
                transform=ax.transAxes,
                fontsize=8.5,
                color=report.INK2,
                family="monospace",
            )
            # RUL cannot be negative, and both models cross zero in the last
            # few cycles of every engine -- a real defect that an RMSE over the
            # whole trajectory averages away, so the reference line stays.
            ax.axhline(0.0, color=report.GRID, linewidth=1.2, zorder=1)

    fig.suptitle(
        "Predicted RUL against ground truth, three held-out engines",
        color=report.INK,
        fontsize=13,
        x=0.006,
        ha="left",
        y=0.995,
    )
    # One legend for six panels, at figure level: repeating it per panel would
    # cost data area, and placing it inside panel one covered unit 11's curve.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9.5,
        labelcolor=report.INK2,
        ncol=3,
        loc="upper left",
        bbox_to_anchor=(0.004, 0.972),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    path = os.path.join(FIG, "trajectories.png")
    fig.savefig(path, dpi=150, facecolor=report.SURFACE)
    plt.close(fig)

    for tag, run in runs:
        print(
            f"  {tag:7s} FIS rmse {run['fis_rmse']:6.2f}   "
            f"network rmse {run['nn_rmse']:6.2f}  "
            f"(seed {run['seed']}, {run['n_hidden']} hidden)"
        )
    print(f"wrote {os.path.relpath(path, cmapss_data.REPO)}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "he")
