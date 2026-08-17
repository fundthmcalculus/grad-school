"""The benchmark proper: every arm, selected on validation, scored on test.

Protocol, applied identically to all of them:

  select  train on the 4 fit engines (2, 5, 10, 16), read the curve on the 2
          validation engines (18, 20). Everything tunable -- stopping epoch,
          read-out ridge -- is chosen here.
  confirm refit on all 6 dev engines with the selected settings, score once on
          the 3 official held-out engines (11, 14, 15).

Nothing selects on the test engines. That is the one place this differs from
the DOE it is benchmarked against, whose Factor-D grid minimizes
`rmse_test_true` directly; the DOE's own configuration is carried here as a
reference row and labelled test-selected, so both numbers are visible.

Arms
----
`fis`               TribbleRegressor -- the incumbent.
`ridge-fis`         ridge on the FIS's selected columns: what you get from
                    TRIBBLE's feature selection alone, with no fuzzy inference
                    and no network. The control that decides whether the FIS's
                    *rules* contribute anything.
`he` / `he-all`     the network from a standard random init, on the FIS's
                    columns and on all of them. The gap between these two is
                    TRIBBLE's feature selection, measured.
`quantile`, `elm`   closed-form initializations: knots at per-feature
                    quantiles, and random features. Both solve their read-out
                    against the labels in one ridge solve.
`hot-analytic`      the FIS converted to a network with no labels at all.
`hot`               the same seed plus one anchored ridge solve -- the warm
                    start under test.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import cmapss_data
import models  # noqa: F401  -- puts experiments/fis-to-neural-net on sys.path

import fis2nn  # noqa: E402

OUT = os.path.join(cmapss_data.REPO, "outputs", "nn-cmapss")

# Read-out ridge for the closed-form arms. On 309 rows with 324 hidden units
# the normal equations are wildly underdetermined and l2=1e-6 sends the
# validation RMSE past 300 cycles; the penalty is a real hyperparameter here,
# not a numerical guard, so it is selected on validation like everything else.
L2_CHOICES = (1e-6, 1e-3, 1e-1, 1.0, 10.0, 100.0)

CLOSED_FORM_ARMS = ("quantile", "elm", "hot", "quantile-all")
ALL_ARMS = ("he", "he-all", "quantile", "quantile-all", "elm", "hot-analytic", "hot")

# Learning rate is selected per arm, on validation, from this ladder. A single
# global rate is not neutral between these arms: a random init needs a rate big
# enough to travel, while a warm start is already near a good solution and the
# same rate walks it straight back out (at lr=0.03 the `hot` arm's best epoch is
# 0 -- selection declining to train at all). Tuning the baseline's rate and
# handing the same one to the arm under test would be a thumb on the scale in
# the direction of the conclusion, so each arm gets its own.
LR_LADDER = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2)


def rmse_cycles(split, y_pred) -> float:
    return float(np.sqrt(np.mean((split.y_true - np.asarray(y_pred).ravel()) ** 2)))


def curve_on(net, hist_epochs, *_):  # pragma: no cover - kept for symmetry
    raise NotImplementedError


def train_and_curve(
    net,
    Xtr,
    ytr_s,
    Xev,
    yev_true,
    y_center,
    y_scale,
    epochs,
    batch_size,
    lr,
    seed,
    eval_every=1,
):
    """Train once; return (trained net, epochs, RMSE-in-cycles curve, seconds)."""
    trained, hist = fis2nn.train_adam(
        net,
        Xtr,
        ytr_s,
        X_val=Xev,
        y_val=(np.asarray(yev_true, float) - y_center) / y_scale,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        eval_every=eval_every,
        track_train=False,
    )
    return (
        trained,
        np.asarray(hist.epochs, float),
        np.asarray(hist.val_rmse, float) * y_scale,
        np.asarray(hist.seconds, float),
    )


def to_target(epochs, curve, seconds, setup_seconds, target):
    """(epoch, train seconds, total seconds incl. setup) at the first point
    reaching `target` -- None if the arm never gets there."""
    hit = np.nonzero(curve <= target)[0]
    if hit.size == 0:
        return None
    i = int(hit[0])
    return dict(
        epoch=float(epochs[i]),
        train_seconds=float(seconds[i]),
        total_seconds=float(seconds[i] + setup_seconds),
        rmse=float(curve[i]),
    )


def build_conversion(bundle_split, names, fis_kwargs, seed, background):
    """Fit a FIS on one split and convert it, returning both plus timings."""
    y_center = float(np.mean(bundle_split.y))
    y_scale = float(np.std(bundle_split.y)) or 1.0
    fis, fis_seconds = models.fit_fis(
        bundle_split.X, bundle_split.y, names, seed=seed, **fis_kwargs
    )
    conv = models.Conversion(
        fis, names, bundle_split.X, y_center, y_scale, seed=seed, background=background
    )
    return fis, fis_seconds, conv, y_center, y_scale


def select_l2(conv, arm, X_fit, y_fit_s, X_val, y_val_true, y_center, y_scale, seed):
    """Pick the read-out ridge for one closed-form arm, on validation."""
    best = (None, np.inf)
    for l2 in L2_CHOICES:
        nets, _, space = models.make_starts(
            conv, X_fit, y_fit_s, l2=l2, seed=seed, arms=(arm,)
        )
        (Xv,) = models.matrices(conv, space[arm], X_val)
        pred = nets[arm].predict(Xv) * y_scale + y_center
        r = float(np.sqrt(np.mean((np.asarray(y_val_true, float) - pred) ** 2)))
        if np.isfinite(r) and r < best[1]:
            best = (l2, r)
    return best[0] if best[0] is not None else 1.0, best[1]


def run(
    which,
    epochs,
    lr,
    batch_size,
    n_hidden,
    seeds,
    background,
    out_name,
    fis_config=None,
):
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which])
    names = b.feature_names
    fis_kwargs = models.FIS_CONFIGS[fis_config or which]

    result = dict(
        bundle=which,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        n_hidden=n_hidden,
        seeds=list(seeds),
        fis_config=fis_config or which,
        fis_kwargs=fis_kwargs,
        background=background,
        n_features=len(names),
        sizes={k: len(getattr(b, k)) for k in ("fit", "val", "train", "test")},
        build_seconds=b.build_seconds,
    )

    # ---------------- reference models, on the confirm split ----------------
    print(f"=== {which} :: reference models (train -> test) ===")
    refs = {}
    const = float(np.mean(b.train.y))
    refs["constant-mean"] = dict(
        setup_seconds=0.0, test=models.evaluate(b.test, np.full(len(b.test), const))
    )

    fis_tr, fis_tr_s = models.fit_fis(b.train.X, b.train.y, names, **fis_kwargs)
    pred_test = models.fis_predict(fis_tr, b.test.X, names)
    refs["fis"] = dict(
        setup_seconds=fis_tr_s,
        test=models.evaluate(b.test, pred_test),
        n_rules=int(fis_tr.model_.n_rules),
        n_mfs=int(fis_tr.model_.n_membership_functions),
        n_features_kept=len(fis_tr.top_features_),
    )
    fis_index = np.array([names.index(f) for f in fis_tr.top_features_], dtype=int)
    print(
        f"  fis            {fis_tr_s:6.2f}s  test rmse {refs['fis']['test']['rmse']:6.2f}"
        f"  endpoint {refs['fis']['test']['rmse_endpoint']:6.2f}"
        f"  ({len(fis_index)}/{len(names)} features, {refs['fis']['n_mfs']} MFs)"
    )

    # Ridge on the FIS's columns, with alpha chosen on the validation engines
    # -- the same selection budget every network arm gets, so the control is
    # not handicapped by sklearn's row-wise internal CV, which cross-validates
    # across rows of the *same* engine and badly misjudges engine-level
    # extrapolation (12.77 val on all columns, against 6.0 for the network).
    from sklearn.linear_model import Ridge

    fis_fit, _ = models.fit_fis(b.fit.X, b.fit.y, names, **fis_kwargs)
    fit_index = np.array([names.index(f) for f in fis_fit.top_features_], dtype=int)
    for tag, sel_cols, fin_cols in (
        ("ridge-fis", fit_index, fis_index),
        ("ridge-all", np.arange(len(names)), np.arange(len(names))),
    ):
        best_a, best_r = None, np.inf
        for alpha in np.logspace(-3, 5, 25):
            m = Ridge(alpha=alpha).fit(b.fit.X[:, sel_cols], b.fit.y)
            r = rmse_cycles(b.val, m.predict(b.val.X[:, sel_cols]))
            if r < best_r:
                best_a, best_r = alpha, r
        t0 = time.perf_counter()
        m = Ridge(alpha=best_a).fit(b.train.X[:, fin_cols], b.train.y)
        secs = time.perf_counter() - t0
        refs[tag] = dict(
            setup_seconds=secs + (fis_tr_s if tag == "ridge-fis" else 0.0),
            alpha=float(best_a),
            val_rmse=float(best_r),
            test=models.evaluate(b.test, m.predict(b.test.X[:, fin_cols])),
        )
        print(
            f"  {tag:14s} {refs[tag]['setup_seconds']:6.2f}s  "
            f"test rmse {refs[tag]['test']['rmse']:6.2f}  "
            f"endpoint {refs[tag]['test']['rmse_endpoint']:6.2f}  (alpha={best_a:.3g})"
        )
    result["references"] = refs

    # ---------------- the conversion, on both splits ------------------------
    # Two conversions: one from the FIS fit on the 4 selection engines (used to
    # choose stopping epoch and ridge), one from the FIS fit on all 6 (used for
    # the reported test numbers). Reusing the second for selection would let
    # every hot arm's starting point see the validation engines.
    print("  converting FIS -> ReLU seed ...")
    t0 = time.perf_counter()
    _, sel_fis_s, conv_sel, yc_sel, ys_sel = build_conversion(
        b.fit, names, fis_kwargs, seeds[0], background
    )
    _, fin_fis_s, conv_fin, yc_fin, ys_fin = build_conversion(
        b.train, names, fis_kwargs, seeds[0], background
    )
    print(
        f"    n_hidden(sel)={conv_sel.n_hidden} n_hidden(fin)={conv_fin.n_hidden}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    # Fidelity: how well the label-free seed reproduces the FIS it came from.
    seed_test = conv_fin.net.predict(conv_fin.subspace(b.test.X)) * ys_fin + yc_fin
    result["conversion"] = dict(
        n_hidden_sel=conv_sel.n_hidden,
        n_hidden_fin=conv_fin.n_hidden,
        analytic_seconds=conv_fin.analytic_seconds,
        fis_seconds=fin_fis_s,
        n_fis_features=len(conv_fin.features),
        fidelity_rmse_vs_fis=float(np.sqrt(np.mean((seed_test - pred_test) ** 2))),
        fidelity_relative=float(
            np.sqrt(np.mean((seed_test - pred_test) ** 2)) / (np.std(pred_test) or 1.0)
        ),
        seed_test=models.evaluate(b.test, seed_test),
    )
    print(
        f"    seed-vs-FIS fidelity on test: {result['conversion']['fidelity_rmse_vs_fis']:.2f} "
        f"cycles (relative {result['conversion']['fidelity_relative']:.3f})"
    )

    # ---------------- the arms ---------------------------------------------
    y_fit_s = (b.fit.y - yc_sel) / ys_sel
    y_train_s = (b.train.y - yc_fin) / ys_fin
    width = n_hidden or conv_fin.n_hidden

    targets = dict(
        fis_parity=refs["fis"]["test"]["rmse"],
        ridge_parity=refs["ridge-fis"]["test"]["rmse"],
    )
    for t in (20.0, 15.0, 12.0, 10.0, 9.0, 8.0):
        targets[f"rmse_{t:g}"] = t
    result["targets"] = targets

    print(f"\n=== {which} :: arms (width {width}, {len(seeds)} seeds) ===")
    arm_rows = []
    for arm in ALL_ARMS:
        for seed in seeds:
            # --- selection: 4 engines -> 2 engines ---
            l2, l2_val = (
                select_l2(
                    conv_sel,
                    arm,
                    b.fit.X,
                    y_fit_s,
                    b.val.X,
                    b.val.y_true,
                    yc_sel,
                    ys_sel,
                    seed,
                )
                if arm in CLOSED_FORM_ARMS
                else (1e-6, float("nan"))
            )
            nets, setup_sel, space = models.make_starts(
                conv_sel,
                b.fit.X,
                y_fit_s,
                n_hidden=width,
                l2=l2,
                seed=seed,
                arms=(arm,),
            )
            Xf, Xv = models.matrices(conv_sel, space[arm], b.fit.X, b.val.X)
            ladder = LR_LADDER if lr is None else (lr,)
            best_epoch, best_lr, best_val = 0, ladder[0], np.inf
            for cand_lr in ladder:
                _, ep, val_curve, _ = train_and_curve(
                    nets[arm],
                    Xf,
                    y_fit_s,
                    Xv,
                    b.val.y_true,
                    yc_sel,
                    ys_sel,
                    epochs,
                    batch_size,
                    cand_lr,
                    seed,
                )
                finite = np.where(np.isfinite(val_curve), val_curve, np.inf)
                i = int(np.argmin(finite))
                if finite[i] < best_val:
                    best_epoch, best_lr, best_val = (
                        int(ep[i]),
                        cand_lr,
                        float(finite[i]),
                    )

            # --- confirm: 6 engines -> 3 engines, at the selected settings ---
            nets_f, setup_fin, space_f = models.make_starts(
                conv_fin,
                b.train.X,
                y_train_s,
                n_hidden=width,
                l2=l2,
                seed=seed,
                arms=(arm,),
            )
            setup = setup_fin[arm] + (fin_fis_s if arm.startswith("hot") else 0.0)
            Xt, Xte = models.matrices(conv_fin, space_f[arm], b.train.X, b.test.X)
            start_pred = nets_f[arm].predict(Xte) * ys_fin + yc_fin
            # Selection is allowed to answer "zero epochs". For a closed-form
            # arm whose ridge solve already sits at its best, gradient descent
            # only makes it worse, and training one epoch anyway (because a
            # `max(best_epoch, 1)` looked harmless) reported that arm at the
            # diverged point rather than at the point selection actually chose.
            if best_epoch == 0:
                trained, train_seconds = nets_f[arm], 0.0
                ep_f = np.array([0.0])
                test_curve = np.array([rmse_cycles(b.test, start_pred)])
                secs_f = np.array([0.0])
            else:
                t0 = time.perf_counter()
                trained, ep_f, test_curve, secs_f = train_and_curve(
                    nets_f[arm],
                    Xt,
                    y_train_s,
                    Xte,
                    b.test.y_true,
                    yc_fin,
                    ys_fin,
                    best_epoch,
                    batch_size,
                    best_lr,
                    seed,
                )
                train_seconds = time.perf_counter() - t0
            final_pred = trained.predict(Xte) * ys_fin + yc_fin

            row = dict(
                arm=arm,
                seed=seed,
                space=space[arm],
                n_hidden=width,
                # What the arm actually built, which is not always what was
                # asked for: `quantile_start` floors at one knot per feature,
                # so a request for 8 units over 90 columns silently yields 90,
                # and the hot arms take their width from the FIS's knot count.
                n_hidden_actual=int(nets_f[arm].n_hidden),
                l2=l2,
                l2_val_rmse=l2_val,
                selected_epoch=best_epoch,
                lr=best_lr,
                val_rmse_at_selected=float(best_val),
                setup_seconds=float(setup),
                train_seconds=float(train_seconds),
                total_seconds=float(setup + train_seconds),
                n_parameters=int(nets_f[arm].n_parameters()),
                start_test=models.evaluate(b.test, start_pred),
                final_test=models.evaluate(b.test, final_pred),
                to_target={
                    k: to_target(ep_f, test_curve, secs_f, setup, v)
                    for k, v in targets.items()
                },
                curve=dict(
                    epochs=ep_f.tolist(),
                    test_rmse=test_curve.tolist(),
                    seconds=secs_f.tolist(),
                ),
            )
            arm_rows.append(row)
        med = np.median([r["final_test"]["rmse"] for r in arm_rows if r["arm"] == arm])
        med_start = np.median(
            [r["start_test"]["rmse"] for r in arm_rows if r["arm"] == arm]
        )
        med_tot = np.median([r["total_seconds"] for r in arm_rows if r["arm"] == arm])
        med_ep = np.median([r["selected_epoch"] for r in arm_rows if r["arm"] == arm])
        lrs = sorted({r["lr"] for r in arm_rows if r["arm"] == arm})
        print(
            f"  {arm:14s} start {med_start:8.2f} -> final {med:6.2f}  "
            f"epochs {med_ep:5.0f}  lr {'/'.join(f'{v:g}' for v in lrs):22s} "
            f"total {med_tot:6.2f}s"
        )

    result["arms"] = arm_rows
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, out_name)
    with open(path, "w") as f:
        json.dump(result, f, indent=1)
    print(f"\nwrote {os.path.relpath(path, cmapss_data.REPO)}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="honest", choices=list(cmapss_data.BUNDLES))
    ap.add_argument("--fis-config", default=None, choices=list(models.FIS_CONFIGS))
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument(
        "--lr",
        type=float,
        default=None,
        help="fix the learning rate for every arm; omit to select "
        "it per arm on validation from LR_LADDER",
    )
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument(
        "--n-hidden", type=int, default=8, help="0 = use the FIS's own knot count"
    )
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--background", type=int, default=256)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run(
        a.bundle,
        a.epochs,
        a.lr,
        a.batch_size,
        a.n_hidden,
        tuple(range(a.seeds)),
        a.background,
        a.out or f"arms_{a.bundle}.json",
        a.fis_config,
    )
