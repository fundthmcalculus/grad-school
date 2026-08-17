"""Per-engine RUL vs. TIME for the 'best' pipeline (raw-memory, 1 Hz stream).

Companion to cmapss_rul_engine_trajectories.py (which uses the whole-cycle
'honest' pipeline, one prediction per flight cycle). This one uses the
raw-memory 'best_full_de_minmax' champion: the 1 Hz sensor stream subsampled
at stride 200 (so ~one sample every 200 s) through a memory window, giving
MANY predictions per flight cycle -- i.e. RUL predicted as a function of
elapsed flight time, not just per cycle.

Each file is fit independently on its own training units (best_full_de_minmax
hyperparameters, reused unchanged -- physical + 2 virtual sensors, full-2nd
order, MinMaxScaler). Every engine (train and test) is plotted: predicted
RUL vs. approximate flight time, with the raw per-sample prediction (thin)
and a moving average (bold) over the true RUL (grey).

Outputs (under FuzzySystemsExperiments/outputs/, gitignored):
  cmapss_rul_time_best_<DSxx>.png     per-engine RUL-vs-time grid per file

Time axis: the raw stream is 1 Hz and subsampled at stride STRIDE, so each
plotted sample is ~STRIDE seconds apart; x is cumulative flight time in
hours (sample_ordinal * STRIDE / 3600), an approximation of true elapsed
time. Fits on each file's TRAINING data only.
"""

import contextlib
import glob
import io
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import cmapss_rul_full_analysis as m
from tribblefis.gaussian_regressor import TribbleRegressor

H5_DIR = "NASA-CMAPSS"
OUT_DIR = "FuzzySystemsExperiments/outputs"
STRIDE = 200  # matches aggregate_raw_memory's default subsample stride
SAMPLE_DT_S = STRIDE  # 1 Hz stream -> STRIDE seconds between subsampled rows
MA_WINDOW = 30  # samples (~half a flight cycle) for the moving average

# best_full_de_minmax champion (see cmapss_rul_full_analysis.py PIPELINES)
CHAMPION_KW = dict(
    tsk_order="full-2nd",
    n_gaussians=4,
    top_p=0.9622893249863613,
    detect_interactions=False,
    norm_conorm="hamacher",
    l2_reg=0.01502536299852122,
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
TRUE_C = "#52514e"
PRED_TRAIN_C = "#2a78d6"
PRED_TEST_C = "#eb6834"


def fit_one_file(h5_path, dataset):
    data, var = m.load_h5(h5_path)
    df_dev = m.to_frame(data, var, "dev", dataset)
    df_test = m.to_frame(data, var, "test", dataset)
    del data
    w = [f"W_{n}" for n in var["W"]]
    xs = [f"Xs_{n}" for n in var["X_s"]]
    xv = [f"Xv_{n}" for n in var["X_v"]]
    correct = xs + xv[: m.CORRECT_N_XV]
    models = m.fit_condition_correction(df_dev, correct, w)
    df_dev = m.apply_condition_correction(df_dev, correct, w, models)
    df_test = m.apply_condition_correction(df_test, correct, w, models)
    feat_cols = w + xs + xv[:2]  # best: physical + 2 virtual sensors
    train_tab = m.aggregate_raw_memory(df_dev, feat_cols, stride=STRIDE)
    test_tab = m.aggregate_raw_memory(df_test, feat_cols, stride=STRIDE)
    agg = [
        c
        for c in train_tab.columns
        if c not in ("dataset", "unit", "cycle", "RUL", "hs")
    ]
    caps = m.physical_rul_cap(train_tab)
    y_train = m.capped_rul(train_tab, caps)
    sc = MinMaxScaler().fit(train_tab[agg].to_numpy(np.float64))
    Xtr = sc.transform(train_tab[agg].to_numpy(np.float64))
    Xte = sc.transform(test_tab[agg].to_numpy(np.float64))
    mdl = TribbleRegressor(random_state=42, max_samples=2000, **CHAMPION_KW)
    with contextlib.redirect_stdout(io.StringIO()):
        mdl.fit(Xtr, y_train)
    return (
        train_tab.assign(RUL_pred=mdl.predict(Xtr)),
        test_tab.assign(RUL_pred=mdl.predict(Xte)),
    )


def plot_file(dataset, train_tab, test_tab, out_path):
    engines = [("train", u) for u in sorted(train_tab["unit"].unique())] + [
        ("test", u) for u in sorted(test_tab["unit"].unique())
    ]
    ncol = min(5, len(engines))
    nrow = int(np.ceil(len(engines) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(3.6 * ncol, 2.9 * nrow), facecolor=SURFACE, squeeze=False
    )
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (split, unit) in enumerate(engines):
        ax = axes.flat[i]
        ax.set_visible(True)
        tab = train_tab if split == "train" else test_tab
        pred_c = PRED_TRAIN_C if split == "train" else PRED_TEST_C
        sub = tab[tab["unit"] == unit].reset_index(drop=True)  # already time-ordered
        t_h = np.arange(len(sub)) * SAMPLE_DT_S / 3600.0  # approx flight time (h)
        ma = sub["RUL_pred"].rolling(MA_WINDOW, center=True, min_periods=1).mean()
        rmse = float(np.sqrt(mean_squared_error(sub["RUL"], sub["RUL_pred"])))
        ma_rmse = float(np.sqrt(mean_squared_error(sub["RUL"], ma)))
        ax.plot(t_h, sub["RUL"], color=TRUE_C, lw=1.8, zorder=3)
        ax.plot(t_h, sub["RUL_pred"], color=pred_c, lw=0.5, alpha=0.3, zorder=4)
        ax.plot(t_h, ma, color=pred_c, lw=1.8, zorder=5)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_SECONDARY, labelsize=7)
        ax.grid(color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        tag = "TEST " if split == "test" else ""
        ax.set_title(
            f"{tag}unit {unit}  (RMSE {rmse:.1f} raw / {ma_rmse:.1f} MA)",
            fontsize=8.5,
            color=(PRED_TEST_C if split == "test" else INK),
            loc="left",
        )
        if i % ncol == 0:
            ax.set_ylabel("RUL (cycles)", fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel("≈ flight time (h)", fontsize=8, color=INK_SECONDARY)
    handles = [
        plt.Line2D([0], [0], color=TRUE_C, lw=2, label="true RUL"),
        plt.Line2D(
            [0], [0], color=INK_SECONDARY, lw=1, alpha=0.5, label="per-sample (1 Hz)"
        ),
        plt.Line2D(
            [0], [0], color=INK_SECONDARY, lw=2.2, label=f"moving avg ({MA_WINDOW} smp)"
        ),
        plt.Line2D([0], [0], color=PRED_TRAIN_C, lw=2.2, label="train engine"),
        plt.Line2D([0], [0], color=PRED_TEST_C, lw=2.2, label="test engine"),
    ]
    fig.suptitle(
        f"{dataset}: per-engine RUL vs. time (best raw-memory fit, 1 Hz stream)",
        fontsize=13,
        color=INK,
        x=0.01,
        ha="left",
    )
    top = 1 - 0.9 / (2.9 * nrow)
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, top + 0.5 * (1 - top)),
        ncol=5,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    fig.tight_layout(rect=[0, 0, 1, top])
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for h5_path in sorted(glob.glob(os.path.join(H5_DIR, "*.h5"))):
        dataset = (
            os.path.basename(h5_path).replace("N-CMAPSS_", "").replace(".h5", "")
        ).split("-")[0]
        try:
            train_tab, test_tab = fit_one_file(h5_path, dataset)
        except Exception as e:
            print(f"{dataset}: SKIPPED ({e!r})")
            continue
        te_rmse = float(
            np.sqrt(mean_squared_error(test_tab["RUL"], test_tab["RUL_pred"]))
        )
        plot_file(
            dataset,
            train_tab,
            test_tab,
            f"{OUT_DIR}/cmapss_rul_time_best_{dataset}.png",
        )
        print(
            f"{dataset}: test_rmse={te_rmse:.2f}  "
            f"({len(test_tab)} test samples over {test_tab['unit'].nunique()} engines) -> plotted"
        )


if __name__ == "__main__":
    main()
