"""The iterative training-residual sweep on the *pooled* all-datasets model.

On DS02 alone the fuzzy system is bias-limited and boosting mostly bought
overfitting. The pooled raw_memory train set is ~220k rows across all nine
N-CMAPSS files -- far more data than a 190-term full-2nd system can memorise --
so the hypothesis is that here reducing the training residual buys *honest* test
skill rather than overfitting.

Reuses `cmapss_all_datasets.gather` (restricted to raw_memory) to load, condition-
correct and featurise every file, then reproduces `TribblePredictiveHealth`'s
cap -> 30k subsample -> scale before dropping in boosted / staged regressors.
Train residual is measured on the 30k rows the model actually fits; test is the
full pooled test set, per-sample RMSE (the report's headline metric). Run from
the repo root (needs NASA-CMAPSS/):

    python experiments/cmapss-ds02-fis/iterative_pooled.py
"""

import contextlib
import csv
import io
import os
import time

import numpy as np  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from _ds02_harness import bootstrap, rmse  # noqa: E402

bootstrap("FuzzySystemsExperiments", os.path.dirname(__file__))
import cmapss_all_datasets as cad  # noqa: E402
from tribble_predictive_health.preprocessing import cap_rul, onset_caps  # noqa: E402
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

OUT = "outputs/ds02-iterative"
os.makedirs(OUT, exist_ok=True)
MAX_TRAIN = 30_000
SEED = 42

BASE = dict(
    tsk_order="full-2nd",
    top_p=0.95,
    n_output_buckets=2,
    norm_conorm="hamacher",
    l2_reg=0.01,
    max_samples=2000,
)


def _fit(cfg, X, y, seed):
    reg = TribbleRegressor(random_state=seed, **cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        reg.fit(X, y)
    return reg


def prep(h5_dir):
    """Pooled raw_memory -> (X_tr, y_tr, X_te, y_te), matching the report's
    cap/subsample/scale for the raw_memory config."""
    cad.CONFIGS = {"raw_memory": cad.CONFIGS["raw_memory"]}  # skip whole_cycle
    pooled, processed, skipped = cad.gather(h5_dir)
    train, test, cols = pooled["raw_memory"]
    print(
        f"  pooled {len(processed)} datasets, train {len(train):,} test {len(test):,}"
    )

    caps = onset_caps(train, unit_col="engine")  # caps on the full train
    sub = (
        train.sample(MAX_TRAIN, random_state=SEED) if len(train) > MAX_TRAIN else train
    )
    y_tr = np.asarray(cap_rul(sub, caps, unit_col="engine"), float)
    scaler = StandardScaler().fit(sub[cols].to_numpy(float))
    X_tr = scaler.transform(sub[cols].to_numpy(float))
    X_te = scaler.transform(test[cols].to_numpy(float))
    y_te = test["rul"].to_numpy(float)
    return X_tr, y_tr, X_te, y_te


def boost(X_tr, y_tr, X_te, y_te, cfg, n_stages, eta):
    F_tr = np.zeros_like(y_tr)
    F_te = np.zeros(len(y_te))
    curve = []
    for m in range(n_stages):
        step = 1.0 if m == 0 else eta
        reg = _fit(cfg, X_tr, y_tr - F_tr, seed=SEED + m)
        F_tr = F_tr + step * reg.predict(X_tr)
        F_te = F_te + step * reg.predict(X_te)
        curve.append((rmse(y_tr, F_tr), rmse(y_te, F_te)))
    return curve


def main(h5_dir):
    t0 = time.perf_counter()
    print(f"Loading + pooling raw_memory from {h5_dir} ...")
    X_tr, y_tr, X_te, y_te = prep(h5_dir)
    print(f"  fit matrix train {X_tr.shape}  test {X_te.shape}")
    rows = []

    b = _fit(BASE, X_tr, y_tr, seed=SEED)
    tr, te = rmse(y_tr, b.predict(X_tr)), rmse(y_te, b.predict(X_te))
    print(f"\nbaseline: train {tr:.3f}  test {te:.3f}")
    rows.append(("baseline", 1, tr, te))

    print("\n== residual boosting (strong base) ==")
    for eta in (1.0, 0.5, 0.3):
        curve = boost(X_tr, y_tr, X_te, y_te, BASE, 12, eta)
        i = min(range(len(curve)), key=lambda k: curve[k][0])
        j = min(range(len(curve)), key=lambda k: curve[k][1])
        print(
            f"  eta={eta}: train {curve[0][0]:.2f}->{curve[-1][0]:.2f} (min {curve[i][0]:.2f}@{i+1})  "
            f"test min {curve[j][1]:.2f}@{j+1} end {curve[-1][1]:.2f}"
        )
        for m, (a, c) in enumerate(curve):
            rows.append((f"boost eta={eta}", m + 1, a, c))

    print("\n== staged rule growth (full-2nd) ==")
    for nb in (
        2,
        3,
        4,
        6,
        8,
    ):  # 12 buckets OOMs the 128k-row full-2nd predict on a 15GB box
        r = _fit({**BASE, "n_output_buckets": nb}, X_tr, y_tr, seed=SEED)
        tr, te = rmse(y_tr, r.predict(X_tr)), rmse(y_te, r.predict(X_te))
        print(
            f"  buckets={nb:2d} rules={int(r.model_.n_rules):2d}: train {tr:.3f}  test {te:.3f}"
        )
        rows.append((f"rules buckets={nb}", nb, tr, te))

    csv_path = os.path.join(OUT, "iterative_pooled.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "param", "train_rmse", "test_rmse"])
        w.writerows(rows)
    print(f"\nwrote {csv_path}")
    _plot(rows)
    print(f"Total wall time: {time.perf_counter() - t0:.0f}s")


def _plot(rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  (skip plot: {exc})")
        return
    base = next(r for r in rows if r[0] == "baseline")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax in (axL, axR):
        ax.axhline(base[2], ls=":", c="k", lw=1, label="baseline train")
        ax.axhline(base[3], ls="--", c="0.5", lw=1, label="baseline test")
    for r in rows:
        pass
    boosts = {}
    for r in rows:
        if r[0].startswith("boost"):
            boosts.setdefault(r[0], []).append((r[1], r[2], r[3]))
    for name, pts in boosts.items():
        pts.sort()
        axL.plot(
            [p[0] for p in pts], [p[1] for p in pts], "-o", ms=3, label=f"{name} train"
        )
        axL.plot(
            [p[0] for p in pts],
            [p[2] for p in pts],
            "--",
            alpha=0.6,
            label=f"{name} test",
        )
    axL.set_title("residual boosting")
    axL.set_xlabel("boosting stage")
    rc = sorted([(r[1], r[2], r[3]) for r in rows if r[0].startswith("rules")])
    axR.plot(
        [p[0] for p in rc], [p[1] for p in rc], "-o", ms=3, c="tab:green", label="train"
    )
    axR.plot(
        [p[0] for p in rc],
        [p[2] for p in rc],
        "--",
        c="tab:green",
        alpha=0.6,
        label="test",
    )
    axR.set_title("staged rule growth")
    axR.set_xlabel("output buckets (rules)")
    axL.set_ylabel("RMSE (cycles)")
    for ax in (axL, axR):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Pooled all-datasets: iterative reduction of the training residual", y=1.02
    )
    fig.tight_layout()
    p = os.path.join(OUT, "iterative_pooled.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--h5-dir", default="NASA-CMAPSS")
    main(ap.parse_args().h5_dir)
