"""Hyperparameter sweep: how small and how fast can the network be?

Trains the He-initialized arm -- the "just train a neural net on this dataset"
baseline, with no FIS anywhere in it -- across a grid of width, learning rate
and batch size, and reads epochs-to-target off the validation curve rather than
re-training per epoch budget. Selection is on the two held-out *dev* engines
(18, 20). The official test engines (11, 14, 15) are not touched by anything in
this file.

Two feature spaces are swept, because the fis-to-neural-net study's one
unambiguous positive result was that TRIBBLE's feature selection helps a
randomly-initialized network:

  space="all"  -- every aggregated column
  space="fis"  -- only the columns a TribbleRegressor's `top_features_` kept

The FIS is fit once per (space="fis") sweep purely to obtain that column list;
its cost is reported separately and charged to the arms that use it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np

import cmapss_data
import models  # noqa: F401  -- puts experiments/fis-to-neural-net on sys.path

import fis2nn  # noqa: E402  -- only importable after `models` extends sys.path

OUT = os.path.join(cmapss_data.REPO, "outputs", "nn-cmapss")

GRID = dict(
    n_hidden=[8, 16, 32, 64, 128, 256],
    lr=[1e-3, 3e-3, 1e-2, 3e-2],
    batch_size=[32, 128],
)
SEEDS = (0, 1, 2)


def train_curve(
    net,
    Xf,
    yf,
    Xv,
    yv_true,
    y_center,
    y_scale,
    val_split,
    epochs,
    batch_size,
    lr,
    seed,
    eval_every=1,
):
    """Train once, return the whole validation curve in cycle units.

    `train_adam`'s own `val_rmse` is computed against the standardized target,
    which is fine for selection but not comparable to a published RMSE, so the
    curve is re-scored here in cycles. `y_scale`/`y_center` do that mapping.
    """
    trained, hist = fis2nn.train_adam(
        net,
        Xf,
        yf,
        X_val=Xv,
        y_val=(yv_true - y_center) / y_scale,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        eval_every=eval_every,
        track_train=False,
    )
    # val_rmse is in standardized units; y is standardized by a *constant*
    # affine map, so multiplying by y_scale recovers cycles exactly.
    curve = np.asarray(hist.val_rmse, dtype=float) * y_scale
    return (
        trained,
        np.asarray(hist.epochs, dtype=float),
        curve,
        np.asarray(hist.seconds),
    )


def epochs_to_target(epochs, curve, seconds, target):
    """First recorded point at or below `target`, as (epoch, seconds)."""
    hit = np.nonzero(curve <= target)[0]
    if hit.size == 0:
        return None, None
    i = int(hit[0])
    return float(epochs[i]), float(seconds[i])


def run(which: str, epochs: int, spaces, out_name: str) -> None:
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which])
    names = b.feature_names
    y_center = float(np.mean(b.fit.y))
    y_scale = float(np.std(b.fit.y)) or 1.0
    y_fit_s = (b.fit.y - y_center) / y_scale

    # Grounding numbers every config below has to beat to mean anything.
    const = float(np.mean(b.fit.y))
    base = {
        "constant_mean": models.evaluate(b.val, np.full(len(b.val), const))["rmse"],
    }
    from sklearn.linear_model import RidgeCV

    t0 = time.perf_counter()
    ridge = RidgeCV(alphas=np.logspace(-3, 4, 20)).fit(b.fit.X, b.fit.y)
    base["ridge_seconds"] = time.perf_counter() - t0
    base["ridge"] = models.evaluate(b.val, ridge.predict(b.val.X))["rmse"]
    print(
        f"baselines (val): constant-mean {base['constant_mean']:.2f}  "
        f"ridge {base['ridge']:.2f} ({base['ridge_seconds']:.2f}s)"
    )

    # The FIS, once: its feature list defines the "fis" space, and its own
    # validation RMSE is the number the network is trying to reach.
    fis, fis_seconds = models.fit_fis(
        b.fit.X, b.fit.y, names, **models.FIS_CONFIGS[which]
    )
    fis_val = models.evaluate(b.val, models.fis_predict(fis, b.val.X, names))
    fis_index = np.array([names.index(f) for f in fis.top_features_], dtype=int)
    print(
        f"FIS ({which} config): {fis_seconds:.2f}s  val rmse {fis_val['rmse']:.2f}  "
        f"kept {len(fis_index)}/{len(names)} features"
    )

    targets = {
        "fis_parity": fis_val["rmse"],
        "ridge_parity": base["ridge"],
        "rmse_15": 15.0,
        "rmse_12": 12.0,
        "rmse_10": 10.0,
    }

    rows = []
    keys = list(GRID)
    combos = list(itertools.product(*GRID.values()))
    total = len(combos) * len(spaces) * len(SEEDS)
    done = 0
    t_sweep = time.perf_counter()

    for space in spaces:
        cols = fis_index if space == "fis" else np.arange(len(names))
        Xf, Xv = b.fit.X[:, cols], b.val.X[:, cols]
        for combo in combos:
            cfg = dict(zip(keys, combo))
            per_seed = []
            for seed in SEEDS:
                rng = np.random.default_rng(1000 + seed)
                net = fis2nn.he_start(rng, Xf.shape[1], cfg["n_hidden"])
                t0 = time.perf_counter()
                _, ep, curve, secs = train_curve(
                    net,
                    Xf,
                    y_fit_s,
                    Xv,
                    b.val.y_true,
                    y_center,
                    y_scale,
                    b.val,
                    epochs,
                    cfg["batch_size"],
                    cfg["lr"],
                    seed,
                )
                wall = time.perf_counter() - t0
                best_i = int(np.argmin(curve))
                rec = dict(
                    space=space,
                    seed=seed,
                    **cfg,
                    best_val_rmse=float(curve[best_i]),
                    best_epoch=float(ep[best_i]),
                    seconds_to_best=float(secs[best_i]),
                    seconds_total=wall,
                    final_val_rmse=float(curve[-1]),
                )
                for tname, tval in targets.items():
                    e, s = epochs_to_target(ep, curve, secs, tval)
                    rec[f"epochs_to_{tname}"] = e
                    rec[f"seconds_to_{tname}"] = s
                per_seed.append(rec)
                done += 1
            rows.extend(per_seed)
            med = float(np.median([r["best_val_rmse"] for r in per_seed]))
            print(
                f"[{done:4d}/{total}] {space:4s} h={cfg['n_hidden']:4d} "
                f"lr={cfg['lr']:<6g} bs={cfg['batch_size']:4d}  "
                f"val_rmse(med)={med:7.2f}  "
                f"epoch={np.median([r['best_epoch'] for r in per_seed]):6.0f}  "
                f"s={np.median([r['seconds_to_best'] for r in per_seed]):6.2f}"
            )

    payload = dict(
        bundle=which,
        epochs=epochs,
        seeds=list(SEEDS),
        grid=GRID,
        n_features=len(names),
        n_fit=len(b.fit),
        n_val=len(b.val),
        baselines=base,
        fis=dict(
            seconds=fis_seconds,
            val=fis_val,
            n_features_kept=len(fis_index),
            features=list(fis.top_features_),
            config=models.FIS_CONFIGS[which],
        ),
        targets=targets,
        sweep_seconds=time.perf_counter() - t_sweep,
        rows=rows,
    )
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, out_name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(
        f"\nwrote {os.path.relpath(path, cmapss_data.REPO)} "
        f"({len(rows)} rows, {payload['sweep_seconds']:.1f}s)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="honest", choices=list(cmapss_data.BUNDLES))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--spaces", nargs="+", default=["fis", "all"])
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--grid",
        default=None,
        help='JSON overriding GRID, e.g. \'{"n_hidden":[2,4,8],"lr":[0.1]}\'. '
        "The first sweep put its winner at the *edge* of the default grid in "
        "both width (8, the smallest offered) and learning rate (0.03, the "
        "largest); a result on a grid boundary is a statement about the grid, "
        "so the boundary gets pushed rather than quoted.",
    )
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    a = ap.parse_args()
    if a.grid:
        GRID.update(json.loads(a.grid))
    SEEDS = tuple(range(a.seeds))
    run(a.bundle, a.epochs, a.spaces, a.out or f"sweep_{a.bundle}.json")
