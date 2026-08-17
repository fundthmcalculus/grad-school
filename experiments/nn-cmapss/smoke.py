"""Smallest thing that can fail: fit the FIS on DS02, convert it, train 3 epochs.

Run before anything expensive. It checks the parts that are easy to get
silently wrong -- the FIS's feature subspace lining up with the network's
columns, the y standardization round-tripping, the seed reproducing the FIS it
came from -- and prints the conversion's fidelity, which is the number that
predicts whether a hot start can help at all on this dataset.
"""

import time

import numpy as np

import cmapss_data
import models
import metrics


def main(which: str = "honest") -> None:
    cfg = {
        "honest": dict(feature_set="real", aggregation="whole_cycle"),
        "best": dict(feature_set="literature", aggregation="raw_memory"),
    }[which]
    fis_kwargs = models.FIS_CONFIGS[which]

    print(f"=== smoke: {which} ===")
    b = cmapss_data.load_or_build(**cfg)
    names = b.feature_names
    print(f"  {len(b.fit)} fit rows x {len(names)} features")

    y_center = float(np.mean(b.fit.y))
    y_scale = float(np.std(b.fit.y)) or 1.0

    fis, fis_s = models.fit_fis(b.fit.X, b.fit.y, names, **fis_kwargs)
    pred_val = models.fis_predict(fis, b.val.X, names)
    m = metrics.evaluate(b.val, pred_val)
    print(
        f"  FIS: {fis_s:.2f}s  rules={fis.model_.n_rules}  "
        f"mfs={fis.model_.n_membership_functions}  "
        f"features_kept={len(fis.top_features_)}/{len(names)}"
    )
    print(f"  FIS val: rmse={m['rmse']:.2f}  endpoint_rmse={m['rmse_endpoint']:.2f}")

    t0 = time.perf_counter()
    conv = models.Conversion(fis, names, b.fit.X, y_center, y_scale, background=256)
    print(
        f"  conversion: {time.perf_counter() - t0:.2f}s  n_hidden={conv.n_hidden}  "
        f"(knots over {len(conv.features)} features)"
    )

    # Fidelity: how closely the label-free seed reproduces the FIS itself. In
    # 1-D this would be the equivalence and ~0; here it is the size of the FIS's
    # non-additive part, which no axis-aligned seed can carry.
    Xv_s = conv.subspace(b.val.X)
    seed_val = conv.net.predict(Xv_s) * y_scale + y_center
    fid = float(np.sqrt(np.mean((seed_val - pred_val) ** 2)))
    print(
        f"  seed-vs-FIS fidelity on val: rmse={fid:.3f} cycles "
        f"(relative {fid / (np.std(pred_val) or 1):.3f})"
    )
    print(f"  seed val: {metrics.evaluate(b.val, seed_val)['rmse']:.2f} rmse")

    y_fit_s = (b.fit.y - y_center) / y_scale
    y_val_s = (b.val.y_true - y_center) / y_scale
    nets, secs, space = models.make_starts(conv, b.fit.X, y_fit_s, seed=0)
    print(
        f"\n  {'arm':14s} {'setup_s':>8s} {'start_rmse':>11s} {'3ep_rmse':>9s} {'train_s':>8s}"
    )
    for arm, net in nets.items():
        Xf, Xv = models.matrices(conv, space[arm], b.fit.X, b.val.X)
        start = metrics.evaluate(b.val, net.predict(Xv) * y_scale + y_center)["rmse"]
        t0 = time.perf_counter()
        trained, _ = fis2nn_train(net, Xf, y_fit_s, Xv, y_val_s, epochs=3)
        dt = time.perf_counter() - t0
        after = metrics.evaluate(b.val, trained.predict(Xv) * y_scale + y_center)[
            "rmse"
        ]
        print(f"  {arm:14s} {secs[arm]:8.3f} {start:11.2f} {after:9.2f} {dt:8.2f}")


def fis2nn_train(net, X, y, Xv, yv, epochs):
    import fis2nn

    return fis2nn.train_adam(
        net,
        X,
        y,
        X_val=Xv,
        y_val=yv,
        epochs=epochs,
        batch_size=64,
        lr=3e-3,
        seed=0,
        track_train=False,
    )


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "honest")
