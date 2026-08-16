"""Driver: does a TRIBBLE-constructed FIS make a better ReLU network than random init?

Five arms per (dataset, seed), all sharing one architecture, one optimizer, and
one epoch budget, so the only thing that differs is where the initial weights
came from:

    fis        the TRIBBLE TSK regressor itself (the reference to beat/match)
    hot        FIS knots -> layer 1, read-out solved in closed form   [the conversion]
    quantile   per-feature quantile knots, same closed-form read-out  [placement ablation]
    elm        He-random layer 1, same closed-form read-out           [random-features control]
    he         He-random layer 1, random read-out                     [standard NN baseline]

Every arm's hidden width is the number of knots the FIS produced, so no arm gets
a capacity advantage. Wall clock for `hot` is charged the FIS fit that produced
it; a warm start you cannot afford is not a warm start.

Usage
-----
    python experiments/fis-to-neural-net/run_experiment.py               # everything
    python experiments/fis-to-neural-net/run_experiment.py --datasets concrete
    FIS2NN_SEEDS=0,1,2 python experiments/fis-to-neural-net/run_experiment.py --epochs 50

Writes ``results.json`` (every curve, every seed) and ``results_summary.md``
next to this file.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "reproduce", "tables"))

import fis2nn  # noqa: E402
import _fuzzy_models as fm  # noqa: E402
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402
from tribblefis.triangle_fit import fit_triangles_to_mixture  # noqa: E402
from tribblefis.regression import predict_tsk  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("FIS2NN_SEEDS", "0,1,2,3,4,5,6,7,8,9").split(",")]

# Learning rates swept per arm, on an inner validation split of the training
# fold of the *first* seed only, then pinned for the rest of that dataset's
# seeds. Per arm rather than shared: a warm-started net and a randomly
# initialized one are not at comparable points on the loss surface, and forcing
# one step size on both hands the result to whichever arm the shared value
# happened to suit. Pinned after the first seed rather than re-swept every
# time: the sweep costs 4x the run, and re-selecting per seed would let each
# arm see 4 draws of its own noise.
LR_GRID = (3e-4, 1e-3, 3e-3, 1e-2)

# Two families of control, because they answer different questions.
#
# `quantile` and `elm` run on the features TRIBBLE kept, so they differ from
# `hot` in *where the knots came from* and nothing else -- the controlled
# ablation. But they are then not honest stand-ins for "what you would have
# done without a FIS", because feature selection is itself something the FIS
# did: on WEC that is 8 columns out of 301, handed to the control for free.
# `quantile-all` and `he-all` therefore run on every raw column, which is the
# from-scratch baseline a practitioner actually faces.
ARMS = ("hot", "quantile", "elm", "he", "quantile-all", "he-all")
FIS_FEATURE_ARMS = {"hot", "quantile", "elm", "he"}

# `fis_kwargs` reaches TribbleRegressor untouched.
#
# WEC needs `top_n=12`, and the reason is a finding rather than a convenience:
# the converted network's width is the FIS's membership-function count, and
# nothing bounds that. Left at the default `top_p=0.95`, TRIBBLE keeps 300 of
# WEC's 301 columns and builds 3,686 membership functions -- an 8,751-unit
# hidden layer with 2.6M parameters in W1, ~300x the training set. Capping the
# FIS's own feature selection is the natural knob, and it is the FIS's knob,
# not one this experiment invented.
DATASETS = {
    "concrete": dict(loader=fm.load_concrete, buckets=4, order="1st", sample=None,
                     fis_kwargs={}),
    "bikeshare": dict(loader=None, buckets=5, order="1st", sample=None,
                      fis_kwargs={}),
    "wec": dict(loader=None, buckets=4, order="1st", sample=None,
                fis_kwargs={"top_n": 12}),
}


# `cnt`, bikeshare's target, is *exactly* `casual + registered`, and
# `_fuzzy_models.load_bikeshare` leaves both columns in X -- it drops only `cnt`
# and `instant`. Any model that finds that sum scores a near-zero RMSE without
# learning anything about bike demand, which is what the first run of this
# experiment did: the converted network reached 0.897 RMSE against the FIS's
# 33.9, on a target whose standard deviation is ~181.
#
# The shared loader is deliberately not patched here. Proposal Tables 4.1 and
# 6.1 quote numbers from it, and silently changing what it returns would move
# archived results without any table announcing it -- the exact failure mode
# `WORKINGDOC.md` documents. This experiment uses its own leak-free loader and
# reports the problem instead.
LEAKY_BIKESHARE_COLUMNS = ["casual", "registered"]


def load_bikeshare_noleak():
    """Bike sharing with the two components of the target removed."""
    loaded = fm.load_bikeshare()
    if loaded is None:
        return None
    X, y = loaded
    return X.drop(columns=LEAKY_BIKESHARE_COLUMNS, errors="raise"), y


def load_wec():
    """Wave Energy Converters (Sydney, 100 buoys): 2,319 x 301 -> total power.

    The high-dimensional partner. TRIBBLE's feature selection does real work
    here -- it keeps a handful of the 301 columns -- so the converted network is
    much narrower than the input, which is the regime where "the FIS told us
    which knots matter" is worth the most.
    """
    path = os.path.join(REPO, "data", "WEC_Sydney_100.csv")
    df = pd.read_csv(path).dropna()
    y = df["Total_Power"].astype(float)
    y.name = "y_value"
    X = df.drop(columns=["Total_Power"]).select_dtypes(include=[np.number]).astype(float)
    return X, y


DATASETS["bikeshare"]["loader"] = load_bikeshare_noleak
DATASETS["wec"]["loader"] = load_wec


def split(X, y, seed, test_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    cut = int(len(X) * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]
    return (
        X.iloc[tr].reset_index(drop=True),
        y.iloc[tr].reset_index(drop=True),
        X.iloc[te].reset_index(drop=True),
        y.iloc[te].reset_index(drop=True),
    )


def prepare(X_tr, y_tr, X_te, y_te):
    """Leak-free unit scaling of X on the train fold; y standardized for training.

    `_fuzzy_models.fit_scaler` exists precisely so the archive's fit-on-everything
    default can be avoided at a call site like this one.
    """
    scaler, _logged = fm.fit_scaler(X_tr, scaler="unit")
    Xtr = fm.apply_scaler(scaler, X_tr)
    Xte = fm.apply_scaler(scaler, X_te)
    y_center = float(np.mean(y_tr))
    y_scale = float(np.std(y_tr)) or 1.0
    return Xtr, Xte, y_center, y_scale


def fis_predict(model, reg, X):
    """Predict with an arbitrary (possibly triangularized) mixture model.

    Goes through the same `predict_tsk` the regressor's own `.predict` uses, so
    swapping the membership shapes is the only difference between the Gaussian
    and triangle numbers.
    """
    return predict_tsk(
        X,
        model,
        reg.top_features_,
        reg.y_bucket_mean_,
        reg.corr_terms_,
        order=reg.tsk_order,
        basis=reg.consequent_basis,
        cross_pairs=getattr(reg, "cross_pairs_", None),
        norms=reg.norms_ if hasattr(reg, "norms_") else None,
    )


def inner_split(n, seed, val_frac=0.15):
    """Carve a validation fold out of the *training* fold.

    Learning-rate choice and early stopping both happen here. Reading either off
    the test fold -- the obvious shortcut when every arm is being compared on
    one curve -- would give each arm a free peek proportional to how jagged its
    curve is, which is exactly the axis the arms differ on.
    """
    rng = np.random.default_rng(90_000 + seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - val_frac))
    return idx[:cut], idx[cut:]


def run_one(name, cfg, seed, epochs, batch_size, l2, lr_cache, quiet=True):
    """One (dataset, seed) cell: fit the FIS, convert it, train every arm."""
    X, y = cfg["loader"]()
    if cfg["sample"] and len(X) > cfg["sample"]:
        idx = np.random.RandomState(0).choice(len(X), cfg["sample"], replace=False)
        X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

    X_tr, y_tr, X_te, y_te = split(X, y, seed)
    Xtr, Xte, y_center, y_scale = prepare(X_tr, y_tr, X_te, y_te)

    out = {"dataset": name, "seed": seed, "n_train": len(Xtr), "n_test": len(Xte),
           "n_features_raw": Xtr.shape[1]}

    # --- 1. the FIS ---------------------------------------------------------
    t0 = time.perf_counter()
    reg = TribbleRegressor(
        n_output_buckets=cfg["buckets"],
        tsk_order=cfg["order"],
        random_state=seed,
        **cfg.get("fis_kwargs", {}),
    )
    import contextlib, io

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink if quiet else sys.stdout):
        reg.fit(Xtr, y_tr)
    fis_seconds = time.perf_counter() - t0

    feats = list(reg.top_features_)
    y_fis_te = np.asarray(reg.predict(Xte)).ravel()
    y_fis_tr = np.asarray(reg.predict(Xtr)).ravel()
    out["fis"] = {
        "seconds": fis_seconds,
        "test_rmse": fis2nn.rmse(y_te, y_fis_te),
        "test_r2": fis2nn.r2(y_te, y_fis_te),
        "train_rmse": fis2nn.rmse(y_tr, y_fis_tr),
        "n_rules": int(reg.model_.n_rules),
        "n_mfs": int(reg.model_.n_membership_functions),
        "features_kept": feats,
    }

    # --- 2. the triangularized FIS (the one that converts exactly) -----------
    tri_model = fit_triangles_to_mixture(reg.model_)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            y_tri_te = np.asarray(fis_predict(tri_model, reg, Xte)).ravel()
        out["fis_triangular"] = {
            "test_rmse": fis2nn.rmse(y_te, y_tri_te),
            "test_r2": fis2nn.r2(y_te, y_tri_te),
        }
    except Exception as exc:  # noqa: BLE001
        out["fis_triangular"] = {"error": f"{type(exc).__name__}: {exc}"}

    # --- 3. the conversion --------------------------------------------------
    Xtr_a = Xtr[feats].to_numpy(dtype=float)
    Xte_a = Xte[feats].to_numpy(dtype=float)
    Xtr_all = Xtr.to_numpy(dtype=float)
    Xte_all = Xte.to_numpy(dtype=float)
    ytr_s = (np.asarray(y_tr, dtype=float) - y_center) / y_scale
    yte_s = (np.asarray(y_te, dtype=float) - y_center) / y_scale

    t0 = time.perf_counter()
    net_hot, knots = fis2nn.warm_start_from_fis(reg.model_, feats, Xtr_a, ytr_s, l2=l2)
    convert_seconds = time.perf_counter() - t0

    n_hidden = net_hot.n_hidden
    out["conversion"] = {
        "seconds": convert_seconds,
        "n_hidden": n_hidden,
        "n_parameters": net_hot.n_parameters(),
        "knots_per_feature": {f: int(knots[f].size) for f in feats},
    }

    # Fidelity of the conversion *as a conversion*: the same layer 1, with the
    # read-out solved against the FIS's own predictions instead of the labels.
    net_distill = fis2nn.solve_readout(
        fis2nn._axis_aligned_net(len(feats), [(i, knots[f]) for i, f in enumerate(feats) if knots[f].size]),
        Xtr_a,
        (y_fis_tr - y_center) / y_scale,
        l2=l2,
    )
    d_te = net_distill.predict(Xte_a) * y_scale + y_center
    out["conversion"]["fidelity_rmse_vs_fis"] = fis2nn.rmse(y_fis_te, d_te)
    out["conversion"]["fidelity_relative"] = fis2nn.rmse(y_fis_te, d_te) / (
        float(np.std(y_fis_te)) or 1.0
    )

    # --- 4. the arms --------------------------------------------------------
    # Everything from here trains on the inner-train fold only; `val` picks the
    # learning rate and the stopping epoch, `test` is scored and never consulted.
    i_tr, i_val = inner_split(len(Xtr_a), seed)
    Xin, yin = Xtr_a[i_tr], ytr_s[i_tr]
    Xval, yval = Xtr_a[i_val], ytr_s[i_val]

    # Re-derive every initialization on the inner fold, so no arm's starting
    # point has seen the validation rows it is about to be selected on. For
    # `hot` that means re-solving the read-out only -- the knots come from the
    # FIS and are not refitted here.
    Xall_in, Xall_val, Xall_te = Xtr_all[i_tr], Xtr_all[i_val], Xte_all

    net_hot_in, _ = fis2nn.warm_start_from_fis(reg.model_, feats, Xin, yin, l2=l2)
    r_init = np.random.default_rng(1000 + seed)
    starts = {
        "hot": net_hot_in,
        "quantile": fis2nn.quantile_start(Xin, n_hidden, yin, l2=l2),
        "elm": fis2nn.random_feature_start(r_init, Xin, yin, n_hidden, l2=l2),
        "he": fis2nn.he_start(r_init, Xin.shape[1], n_hidden),
        "quantile-all": fis2nn.quantile_start(Xall_in, n_hidden, yin, l2=l2),
        "he-all": fis2nn.he_start(r_init, Xall_in.shape[1], n_hidden),
    }
    setup_seconds = {a: (fis_seconds + convert_seconds if a == "hot" else 0.0) for a in ARMS}

    def train(net0, lr, arm):
        on_fis_feats = arm in FIS_FEATURE_ARMS
        Xa, Xv, Xt = (
            (Xin, Xval, Xte_a) if on_fis_feats else (Xall_in, Xall_val, Xall_te)
        )
        return fis2nn.train_adam(
            net0, Xa, yin,
            X_test=Xt, y_test=yte_s,
            X_val=Xv, y_val=yval,
            epochs=epochs, batch_size=batch_size, lr=lr, seed=seed,
            y_scale=y_scale, y_center=y_center,
        )

    out["arms"] = {}
    for arm in ARMS:
        key = (name, arm)
        if key not in lr_cache:
            swept = {}
            for lr in LR_GRID:
                _, h = train(starts[arm], lr, arm)
                swept[lr] = float(np.min(h.val_rmse))
            lr_cache[key] = min(swept, key=swept.get)
            out.setdefault("lr_sweep", {})[arm] = {str(k): v for k, v in swept.items()}
        lr = lr_cache[key]

        t0 = time.perf_counter()
        trained, hist = train(starts[arm], lr, arm)
        train_seconds = time.perf_counter() - t0
        X_score = Xte_a if arm in FIS_FEATURE_ARMS else Xte_all

        test_curve = np.asarray(hist.test_rmse, dtype=float)
        val_curve = np.asarray(hist.val_rmse, dtype=float)
        stop = int(np.argmin(val_curve))  # early stopping, decided on val alone
        out["arms"][arm] = {
            "lr": lr,
            "epoch0_test_rmse": float(test_curve[0]),
            "selected_epoch": int(hist.epochs[stop]),
            "selected_test_rmse": float(test_curve[stop]),
            "final_test_rmse": float(test_curve[-1]),
            "final_test_r2": fis2nn.r2(y_te, trained.predict(X_score) * y_scale + y_center),
            "setup_seconds": setup_seconds[arm],
            "train_seconds": train_seconds,
            "seconds_per_epoch": train_seconds / max(epochs, 1),
            "curve_epochs": list(hist.epochs),
            "curve_test_rmse": [float(v) for v in test_curve],
            "curve_val_rmse": [float(v) for v in val_curve],
            "curve_train_rmse": [float(v) for v in hist.train_rmse],
        }

    # How long each arm takes to *first* reach two reference qualities, in
    # epochs and in wall-clock seconds -- with the FIS fit charged to `hot`,
    # since a warm start you cannot afford is not a warm start.
    def first_hit(rec, target):
        curve = np.asarray(rec["curve_test_rmse"])
        hit = np.flatnonzero(curve <= target)
        if not hit.size:
            return None, None
        ep = int(rec["curve_epochs"][hit[0]])
        return ep, rec["setup_seconds"] + ep * rec["seconds_per_epoch"]

    fis_target = out["fis"]["test_rmse"]
    # What the better of the two randomly-initialized arms eventually reaches:
    # the honest "when do you get what training from scratch would have given
    # you" line, rather than a threshold chosen to flatter the warm start.
    baseline_target = min(out["arms"][a]["selected_test_rmse"] for a in ("he", "he-all"))
    for rec in out["arms"].values():
        rec["epochs_to_fis_parity"], rec["seconds_to_fis_parity"] = first_hit(rec, fis_target)
        rec["epochs_to_baseline"], rec["seconds_to_baseline"] = first_hit(rec, baseline_target)
    out["baseline_target_rmse"] = baseline_target

    return out


def provenance():
    def git(*args, cwd=REPO):
        try:
            return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
        except Exception:  # noqa: BLE001
            return "unknown"

    return {
        "grad_school_commit": git("rev-parse", "--short", "HEAD"),
        "tribble_fis_commit": git("rev-parse", "--short", "HEAD", cwd=os.path.join(REPO, "tribble-fis")),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "seeds": SEEDS,
    }


def agg(values):
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return float("nan"), float("nan")
    return float(np.mean(v)), float(np.std(v))


def summarize(results, cfg_meta):
    """Render the tables that the write-up quotes."""
    lines = ["# FIS -> ReLU network: measured results", ""]
    lines.append(f"Seeds: {cfg_meta['seeds']} · epochs: {cfg_meta['epochs']} · "
                 f"batch: {cfg_meta['batch_size']} · commit `{cfg_meta['provenance']['grad_school_commit']}` · "
                 f"tribble-fis `{cfg_meta['provenance']['tribble_fis_commit']}`")
    lines.append("")

    by_ds: dict[str, list] = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)

    for ds, rows in by_ds.items():
        n_hidden, _ = agg([r["conversion"]["n_hidden"] for r in rows])
        lines += [
            f"## {ds}",
            "",
            f"{rows[0]['n_train']} train / {rows[0]['n_test']} test · "
            f"{rows[0]['n_features_raw']} raw features · "
            f"FIS keeps {len(rows[0]['fis']['features_kept'])} · "
            f"hidden width {n_hidden:.0f}",
            "",
            "| model | test RMSE | test R2 | epoch-0 RMSE | epochs to FIS parity | total s |",
            "|---|---|---|---|---|---|",
        ]
        m, s = agg([r["fis"]["test_rmse"] for r in rows])
        m2, s2 = agg([r["fis"]["test_r2"] for r in rows])
        t, ts = agg([r["fis"]["seconds"] for r in rows])
        lines.append(f"| tribble FIS | {m:.3f} ± {s:.3f} | {m2:.3f} ± {s2:.3f} | — | — | {t:.2f} ± {ts:.2f} |")

        if "test_rmse" in rows[0].get("fis_triangular", {}):
            m, s = agg([r["fis_triangular"]["test_rmse"] for r in rows])
            m2, s2 = agg([r["fis_triangular"]["test_r2"] for r in rows])
            lines.append(f"| tribble FIS (triangularized) | {m:.3f} ± {s:.3f} | {m2:.3f} ± {s2:.3f} | — | — | — |")

        for arm in ARMS:
            recs = [r["arms"][arm] for r in rows]
            m, s = agg([x["selected_test_rmse"] for x in recs])
            m2, s2 = agg([x["final_test_r2"] for x in recs])
            e0, e0s = agg([x["epoch0_test_rmse"] for x in recs])
            par = [x["epochs_to_fis_parity"] for x in recs]
            hit = [p for p in par if p is not None]
            par_txt = (f"{np.mean(hit):.0f} ({len(hit)}/{len(par)})" if hit else f"never (0/{len(par)})")
            t, ts = agg([x["setup_seconds"] + x["train_seconds"] for x in recs])
            lines.append(
                f"| nn-{arm} (lr {recs[0]['lr']:g}) | {m:.3f} ± {s:.3f} | {m2:.3f} ± {s2:.3f} | "
                f"{e0:.3f} ± {e0s:.3f} | {par_txt} | {t:.2f} ± {ts:.2f} |"
            )

        lines += [
            "",
            "Cost to reach the quality the best randomly-initialized arm ends at "
            f"({np.mean([r['baseline_target_rmse'] for r in rows]):.3f} RMSE):",
            "",
            "| model | epochs | wall-clock s (FIS fit charged to nn-hot) |",
            "|---|---|---|",
        ]
        for arm in ARMS:
            recs = [r["arms"][arm] for r in rows]
            eps = [x["epochs_to_baseline"] for x in recs]
            secs = [x["seconds_to_baseline"] for x in recs]
            hit = [e for e in eps if e is not None]
            if not hit:
                lines.append(f"| nn-{arm} | never (0/{len(eps)}) | — |")
                continue
            em, es = agg(eps)
            sm, ss = agg(secs)
            lines.append(
                f"| nn-{arm} | {em:.1f} ± {es:.1f} ({len(hit)}/{len(eps)} seeds) | {sm:.2f} ± {ss:.2f} |"
            )

        f, fs = agg([r["conversion"]["fidelity_relative"] for r in rows])
        lines += [
            "",
            f"Conversion fidelity (read-out solved against the FIS's own predictions rather "
            f"than the labels): relative RMSE {f:.3f} ± {fs:.3f} of the FIS output's own "
            f"standard deviation.",
            "",
        ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--out", default=os.path.join(HERE, "results.json"))
    args = ap.parse_args()

    results = []
    lr_cache: dict[tuple[str, str], float] = {}
    for name in args.datasets:
        cfg = DATASETS[name]
        for seed in SEEDS:
            t0 = time.perf_counter()
            try:
                res = run_one(name, cfg, seed, args.epochs, args.batch_size, args.l2, lr_cache)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name} seed {seed}] FAILED {type(exc).__name__}: {exc}")
                continue
            results.append(res)
            arms = res["arms"]
            print(
                f"  [{name} seed {seed}] fis {res['fis']['test_rmse']:.3f} | "
                f"hot {arms['hot']['epoch0_test_rmse']:.3f}->{arms['hot']['selected_test_rmse']:.3f} | "
                f"he {arms['he']['epoch0_test_rmse']:.3f}->{arms['he']['selected_test_rmse']:.3f} | "
                f"{time.perf_counter() - t0:.1f}s"
            )

    meta = {
        "seeds": SEEDS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "l2": args.l2,
        "learning_rates": {f"{d}/{a}": lr for (d, a), lr in lr_cache.items()},
        "lr_grid": list(LR_GRID),
        "provenance": provenance(),
    }
    with open(args.out, "w") as fh:
        json.dump({"meta": meta, "results": results}, fh, indent=1)
    summary_path = os.path.join(HERE, "results_summary.md")
    with open(summary_path, "w") as fh:
        fh.write(summarize(results, meta))
    print(f"\nwrote {os.path.relpath(args.out, REPO)} and {os.path.relpath(summary_path, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
