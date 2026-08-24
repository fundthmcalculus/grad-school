"""PhiUSIIL at full scale: the regime where a warm start can actually pay.

Every rung of the earlier ladder trains in under 13 seconds, so a 0.2-1.6 s
TRIBBLE fit is 4-34% overhead and there is no room for a warm start to
amortize -- that, not the conversion, is why Part 1's hot arms lost on
wall-clock. PhiUSIIL is 235,795 rows x 50 numeric features, 17x bikeshare and
229x Concrete, and a network takes minutes on it. If the hot start does not pay
here it does not pay anywhere.

It is also the first *classification* rung, so two things change:

* The conversion seeds a **logit**. The network's scalar output is what a
  sigmoid is applied to, so the FIS's probability is mapped through
  `fis2nn.logit` before being backed out; converting in probability space and
  then squashing again would compose two sigmoids and misplace every weight.
* Training minimizes binary cross-entropy rather than MSE. Only the output-layer
  gradient differs, so it is the same `fis2nn.train_adam` with `loss="bce"`.

Two measurement notes that the write-up has to carry:

* **Scaling is load-bearing, not incidental.** On raw features TRIBBLE scores
  0.730; with the repository's own log + min-max treatment (`_fuzzy_models`'s
  `fit_scaler`, fitted on the training fold only) it scores 0.994. Every arm
  here gets the same scaled inputs.
* **Accuracy barely discriminates on this dataset.** A single threshold on
  `URLSimilarityIndex` alone scores 0.9914 on the test fold. Everything lands
  near 99%, so the tables report *error rate* and log-loss, and the headline is
  time-to-target rather than a league table of accuracies.

    python experiments/fis-to-neural-net/run_phiusiil.py
    FIS2NN_SEEDS=0 python experiments/fis-to-neural-net/run_phiusiil.py --epochs 20

Writes `outputs/phiusiil_results.json` and `outputs/phiusiil.md`.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd

from _bootstrap import add_repo_paths

HERE, REPO = add_repo_paths(__file__, ("reproduce", "tables"))

#: Every generated artifact goes here. Kept out of the source directory so the
#: scripts and the things they produce never have to be told apart by eye, and
#: so `outputs/.gitignore` can drop derived CSVs without a rule that could ever
#: match a hand-written file.
OUTPUTS = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

import _fuzzy_models as fm  # noqa: E402
import fis2nn  # noqa: E402
from tribblefis.gaussian_classifier import TribbleClassifier  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("FIS2NN_SEEDS", "0,1,2").split(",")]
LR_GRID = (1e-3, 3e-3, 1e-2)
DATA = os.path.join(REPO, "data", "PhiUSIIL_Phishing_URL_Dataset.csv")

# TRIBBLE's own feature cap. Five is what `tribble-tree/demo_phishing.py` uses,
# and it is also the best of the settings swept here (0.9944 against 0.9905 for
# the unrestricted default and 0.9883 at twelve) -- so this is the FIS's
# preferred configuration, not one chosen to make the conversion look good.
FIS_TOP_N = 5

# All the non-FIS arms start at the base rate by construction: without either
# the FIS or a least-squares solve there is nothing better to initialize a
# classifier's output layer to. The comparison is therefore time-to-target, not
# epoch-0 parity, which is the metric that matters anyway.
# `hot-all` exists to remove a confound the first run exposed: `hot` trains on
# the five columns TRIBBLE kept, while `he-all` trains on all of them, so the
# two differ in *inputs* as well as in initialization. Without the dominant
# feature that difference dominates everything -- five columns cap the model at
# ~1.2% error while forty-nine reach 0.01%. `hot-all` takes the whole feature
# set, places ReLU knots only on the five the FIS chose, and lets the linear
# skip cover the rest at zero initial weight: it starts exactly where `hot`
# starts and is free to learn what the FIS discarded.
ARMS = ("hot", "hot-all", "hot-anova", "quantile", "he", "he-all")
FIS_FEATURE_ARMS = {"hot", "hot-anova", "quantile", "he"}

# How far the FIS's probability is clipped before being read as a logit.
# `TribbleClassifier` saturates hard here -- 53.7% of its test-fold outputs sit
# at a probability of exactly 0 or 1 -- so the clip decides the whole dynamic
# range of the conversion target.
LOGIT_EPS = 1e-2


#: Dropping this makes PhiUSIIL a real problem rather than a lookup.
#: `URLSimilarityIndex` alone, thresholded, scores 0.9914 on the test fold --
#: within a fraction of a point of every model in the table below, TRIBBLE and
#: neural network alike. With it present the dataset cannot distinguish
#: initializations, because there is nothing left to learn after one feature.
DOMINANT_FEATURE = "URLSimilarityIndex"


def load(drop=()):
    """Full PhiUSIIL, numeric features only, integer label.

    Recovered into `data/` from `tribble-fis` history -- see `data/.gitignore`,
    which records the exact `git show` that reproduces it (57 MB, not vendored).
    """
    df = pd.read_csv(DATA, encoding="utf-8-sig").dropna()
    df.columns = [c.strip() for c in df.columns]
    y = df["label"].astype(int).to_numpy()
    X = df.drop(columns=["label"]).select_dtypes(include=[np.number]).astype(float)
    if drop:
        X = X.drop(columns=[c for c in drop if c in X.columns])
    return X, y


def split(n, seed, test_frac=0.2, val_frac=0.15):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_frac))
    tr_all, te = idx[:cut], idx[cut:]
    inner = int(len(tr_all) * (1 - val_frac))
    return tr_all[:inner], tr_all[inner:], te


def run_cell(X, y, seed, epochs, batch_size, lr_cache, eval_batches):
    i_tr, i_val, i_te = split(len(X), seed)
    X_tr, X_val, X_te = X.iloc[i_tr], X.iloc[i_val], X.iloc[i_te]
    y_tr, y_val, y_te = (
        y[i_tr].astype(float),
        y[i_val].astype(float),
        y[i_te].astype(float),
    )

    # Scaling fitted on the inner-training fold only. It is charged to the hot
    # arm's setup below because the FIS needs it; the random arms are scored on
    # the same inputs, which is generous to them and keeps the comparison about
    # initialization rather than preprocessing.
    t0 = time.perf_counter()
    scaler, _logged = fm.fit_scaler(X_tr, scaler="unit")
    Str, Sval, Ste = (fm.apply_scaler(scaler, f) for f in (X_tr, X_val, X_te))
    scale_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        clf = TribbleClassifier(top_n=FIS_TOP_N, random_state=seed)
        clf.fit(Str, y_tr.astype(int))
    fis_seconds = time.perf_counter() - t0

    feats = list(clf.top_features_)

    def fis_logit(frame):
        with contextlib.redirect_stdout(io.StringIO()):
            proba = np.asarray(clf.predict_proba(frame), dtype=float)
        return fis2nn.logit(proba[:, 1], eps=LOGIT_EPS)

    fis_te_logit = fis_logit(Ste)
    out = {
        "seed": seed,
        "n_train": len(i_tr),
        "n_val": len(i_val),
        "n_test": len(i_te),
        "n_features_raw": Str.shape[1],
        "scale_seconds": scale_seconds,
        "fis": {
            "seconds": fis_seconds,
            "error_rate": fis2nn.error_rate(y_te, fis_te_logit),
            "log_loss": fis2nn.log_loss(y_te, fis_te_logit),
            "features_kept": feats,
            "n_mfs": int(clf.model_.n_membership_functions),
        },
    }

    Atr = Str[feats].to_numpy(dtype=float)
    Aval = Sval[feats].to_numpy(dtype=float)
    Ate = Ste[feats].to_numpy(dtype=float)
    Ftr, Fval, Fte = (f.to_numpy(dtype=float) for f in (Str, Sval, Ste))

    # --- the conversion ----------------------------------------------------
    # Two routes to the same first layer, differing only in how the read-out is
    # obtained, and on a saturating classifier they are not close.
    #
    # `hot-anova` is the route that worked for regression: average the FIS's
    # response over the data one feature at a time and decompose each profile by
    # second differences. It fails here for a specific and instructive reason --
    # 53.7% of the FIS's logits are clipped extremes, so the profile averages are
    # dominated by them and the additive decomposition's (F-1) * baseline
    # centering term compounds it. The seed's *ranking* survives (AUC 0.996) but
    # its level does not: raw error 0.574, and 0.032 even after a two-parameter
    # Platt rescale.
    #
    # `hot` instead projects the FIS's logit onto the same ReLU basis by one
    # ridge solve. Still no labels -- the targets are the FIS's own outputs --
    # and it lands below the FIS's own error rate.
    t0 = time.perf_counter()
    knots = fis2nn.fis_knots(clf.model_, feats)
    pairs = [(i, knots[f]) for i, f in enumerate(feats) if knots[f].size]
    basis = fis2nn._axis_aligned_net(len(feats), pairs)
    net_hot = fis2nn.solve_readout(basis, Atr, fis_logit(Str), l2=1e-6, anchor=False)
    convert_seconds = time.perf_counter() - t0

    # Same knots, same read-out target, but embedded in the full feature space.
    t0 = time.perf_counter()
    col_of = {name: j for j, name in enumerate(Str.columns)}
    wide_pairs = [(col_of[feats[i]], k) for i, k in pairs]
    wide_basis = fis2nn._axis_aligned_net(Ftr.shape[1], wide_pairs)
    net_hot_all = fis2nn.solve_readout(
        wide_basis, Ftr, fis_logit(Str), l2=1e-6, anchor=False
    )
    convert_all_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    net_anova = fis2nn.analytic_seed_from_fis(
        fis_logit, Str, feats, knots, background_size=256, seed=seed
    )
    anova_seconds = time.perf_counter() - t0

    n_hidden = net_hot.n_hidden
    out["conversion"] = {
        "seconds": convert_seconds,
        "anova_seconds": anova_seconds,
        "n_hidden": n_hidden,
        "n_parameters": net_hot.n_parameters(),
        "seed_error_rate": fis2nn.error_rate(y_te, net_hot.predict(Ate)),
        "seed_all_error_rate": fis2nn.error_rate(y_te, net_hot_all.predict(Fte)),
        "all_seconds": convert_all_seconds,
        "anova_seed_error_rate": fis2nn.error_rate(y_te, net_anova.predict(Ate)),
        "fidelity_logit_rmse": fis2nn.rmse(fis_te_logit, net_hot.predict(Ate)),
    }

    rng = np.random.default_rng(1000 + seed)
    starts = {
        "hot": net_hot,
        "hot-all": net_hot_all,
        "hot-anova": net_anova,
        "quantile": fis2nn.quantile_start(Atr, n_hidden, np.zeros(len(Atr))),
        "he": fis2nn.he_start(rng, Atr.shape[1], n_hidden),
        "he-all": fis2nn.he_start(rng, Ftr.shape[1], n_hidden),
    }
    # Every non-FIS arm starts predicting the base rate rather than 0.5, which
    # is the strongest label-free init available to it and costs one scalar.
    base_logit = float(fis2nn.logit(np.mean(y_tr)))
    for arm, net in starts.items():
        if not arm.startswith("hot"):
            net.w2[:] = 0.0
            net.v[:] = 0.0
            net.c = base_logit

    setup = {
        "hot": scale_seconds + fis_seconds + convert_seconds,
        "hot-all": scale_seconds + fis_seconds + convert_all_seconds,
        "hot-anova": scale_seconds + fis_seconds + anova_seconds,
        "quantile": scale_seconds,
        "he": scale_seconds,
        "he-all": scale_seconds,
    }

    def train(net0, lr, arm):
        on_fis = arm in FIS_FEATURE_ARMS
        Xa, Xv, Xt = (Atr, Aval, Ate) if on_fis else (Ftr, Fval, Fte)
        return fis2nn.train_adam(
            net0,
            Xa,
            y_tr,
            X_test=Xt,
            y_test=y_te,
            X_val=Xv,
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            loss="bce",
            metric_fn=fis2nn.error_rate,
            eval_batches=eval_batches,
            track_train=False,
        )

    out["arms"] = {}
    for arm in ARMS:
        if arm not in lr_cache:
            swept = {}
            for lr in LR_GRID:
                _, h = train(starts[arm], lr, arm)
                swept[lr] = float(np.min(h.val_rmse))
            lr_cache[arm] = min(swept, key=swept.get)
            out.setdefault("lr_sweep", {})[arm] = {str(k): v for k, v in swept.items()}
        lr = lr_cache[arm]

        t0 = time.perf_counter()
        trained, hist = train(starts[arm], lr, arm)
        train_seconds = time.perf_counter() - t0
        test_curve = np.asarray(hist.test_rmse, dtype=float)
        stop = int(np.argmin(np.asarray(hist.val_rmse, dtype=float)))
        Xt = Ate if arm in FIS_FEATURE_ARMS else Fte
        out["arms"][arm] = {
            "lr": lr,
            "epoch0_error": float(test_curve[0]),
            "selected_epoch": float(hist.epochs[stop]),
            "selected_error": float(test_curve[stop]),
            "final_error": float(test_curve[-1]),
            "final_log_loss": fis2nn.log_loss(y_te, trained.predict(Xt)),
            "setup_seconds": setup[arm],
            "train_seconds": train_seconds,
            "seconds_per_epoch": train_seconds / max(epochs, 1),
            "curve_epochs": [float(e) for e in hist.epochs],
            "curve_test_error": [float(v) for v in test_curve],
            "curve_val_error": [float(v) for v in hist.val_rmse],
        }
    return out


def seconds_to(rec, target):
    curve = np.asarray(rec["curve_test_error"], dtype=float)
    hit = np.flatnonzero(curve <= target)
    if not hit.size:
        return None, None
    epoch = float(rec["curve_epochs"][hit[0]])
    return epoch, rec["setup_seconds"] + epoch * rec["seconds_per_epoch"]


def summarize(rows, meta):
    def m(fn):
        vals = [fn(r) for r in rows]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    dropped = meta.get("dropped") or []
    lines = [
        "# PhiUSIIL at full scale"
        + (" — without the dominant feature" if dropped else ""),
        "",
        f"{rows[0]['n_train']:,} train / {rows[0]['n_val']:,} val / "
        f"{rows[0]['n_test']:,} test · {rows[0]['n_features_raw']} numeric features · "
        f"seeds {meta['seeds']} · {meta['epochs']} epochs · batch {meta['batch_size']}",
        "",
        (
            f"**`{DOMINANT_FEATURE}` excluded.** Thresholded on its own it scores "
            "0.9914 on the test fold, within a fraction of a point of every model "
            "below, so with it present the dataset cannot distinguish "
            "initializations at all."
            if dropped
            else "Error rate, not accuracy: everything on this dataset lands near 99%, "
            f"and a single threshold on `{DOMINANT_FEATURE}` alone scores 0.9914, so "
            "accuracy cannot separate the arms."
        )
        + " All arms see the same log + min-max scaled inputs (fitted on the training "
        "fold); on raw features the FIS scores 0.730 rather than 0.994, so the scaling "
        "is load-bearing.",
        "",
        "| model | test error | log loss | epoch-0 error | setup s | s/epoch | total s |",
        "|---|---|---|---|---|---|---|",
        f"| tribble FIS ({FIS_TOP_N} features, {m(lambda r: r['fis']['n_mfs']):.0f} MFs) | "
        f"{m(lambda r: r['fis']['error_rate']):.4f} | "
        f"{m(lambda r: r['fis']['log_loss']):.4f} | — | "
        f"{m(lambda r: r['scale_seconds'] + r['fis']['seconds']):.1f} | — | "
        f"{m(lambda r: r['scale_seconds'] + r['fis']['seconds']):.1f} |",
    ]
    for arm in ARMS:
        lines.append(
            f"| nn-{arm} (lr {rows[0]['arms'][arm]['lr']:g}) | "
            f"{m(lambda r: r['arms'][arm]['selected_error']):.4f} | "
            f"{m(lambda r: r['arms'][arm]['final_log_loss']):.4f} | "
            f"{m(lambda r: r['arms'][arm]['epoch0_error']):.4f} | "
            f"{m(lambda r: r['arms'][arm]['setup_seconds']):.1f} | "
            f"{m(lambda r: r['arms'][arm]['seconds_per_epoch']):.1f} | "
            f"{m(lambda r: r['arms'][arm]['setup_seconds'] + r['arms'][arm]['train_seconds']):.0f} |"
        )

    lines += [
        "",
        f"Conversion: {m(lambda r: r['conversion']['n_hidden']):.0f} hidden units, "
        f"{m(lambda r: r['conversion']['seconds']):.1f} s, seeded error "
        f"{m(lambda r: r['conversion']['seed_error_rate']):.4f} before a single "
        "gradient step -- against the FIS's own "
        f"{m(lambda r: r['fis']['error_rate']):.4f}. The partial-dependence route "
        f"that worked for regression seeds {m(lambda r: r['conversion']['anova_seed_error_rate']):.4f} "
        "instead: the FIS saturates, so its profile averages are dominated by "
        "clipped extremes.",
        "",
        "## Time to target",
        "",
        "Wall clock to *first* reach each test error rate, with scaling, the FIS fit "
        "and the conversion all charged to `nn-hot`. Targets are absolute error rates "
        "every arm faces identically.",
        "",
    ]
    targets = [0.05, 0.02, 0.01, 0.007, 0.005]
    lines += [
        "| arm | " + " | ".join(f"err <= {t:.3f}" for t in targets) + " |",
        "|" + "---|" * (len(targets) + 1),
    ]
    secs = {}
    for arm in ARMS:
        cells = []
        for t in targets:
            vals = [
                s
                for s in (seconds_to(r["arms"][arm], t)[1] for r in rows)
                if s is not None
            ]
            if not vals:
                cells.append("never")
                secs.setdefault(arm, []).append(None)
            else:
                txt = f"{np.mean(vals):.1f}s"
                if len(vals) < len(rows):
                    txt += f" ({len(vals)}/{len(rows)})"
                cells.append(txt)
                secs.setdefault(arm, []).append(float(np.mean(vals)))
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Speedup of `nn-hot` over each other arm at the same target:",
        "",
        "| arm | " + " | ".join(f"err <= {t:.3f}" for t in targets) + " |",
        "|" + "---|" * (len(targets) + 1),
    ]
    for arm in ARMS:
        if arm == "hot":
            continue
        cells = []
        for i in range(len(targets)):
            mine, theirs = secs["hot"][i], secs[arm][i]
            if mine is None:
                cells.append("hot never")
            elif theirs is None:
                cells.append("only hot arrives")
            else:
                cells.append(f"{theirs / mine:.1f}x")
        lines.append(f"| vs `{arm}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def guard_output(path, force):
    """Refuse to overwrite an existing run of record unless asked twice.

    Writing this file cost tens of minutes; a smoke run with the default `--out`
    costs seconds and silently replaces it. That happened once while this
    directory was being reorganized -- a one-seed, five-epoch synth1d run landed
    on top of the ten-seed, 150-epoch `results.json` -- and it is the same
    failure `WORKINGDOC.md` catalogues under REPRO_OUTPUT_DIR. Recovering it
    needed `git checkout`, which only worked because the file happened to be
    staged.
    """
    if os.path.exists(path) and not force:
        raise SystemExit(
            f"{os.path.relpath(path, REPO)} already exists.\n"
            "Pass --force to replace it, or --out <path> to write elsewhere "
            "(which is what a smoke run should do)."
        )
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument(
        "--eval-batches",
        type=int,
        default=25,
        help="record the curve every N minibatches; an epoch is 313 updates here, "
        "and every arm crosses the loose targets inside the first one",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--drop-dominant",
        action="store_true",
        help=f"exclude {DOMINANT_FEATURE}, which alone scores 0.9914 and makes "
        "every arm look identical",
    )
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing run of record"
    )
    args = ap.parse_args()

    drop = (DOMINANT_FEATURE,) if args.drop_dominant else ()
    tag = "_hard" if args.drop_dominant else ""
    out_path = args.out or os.path.join(OUTPUTS, f"phiusiil{tag}_results.json")
    md_path = os.path.join(OUTPUTS, f"phiusiil{tag}.md")

    if not os.path.exists(DATA):
        print(f"missing {DATA}\nRecover it with the command in data/.gitignore.")
        return 1

    t0 = time.perf_counter()
    X, y = load(drop)
    print(
        f"loaded {X.shape[0]:,} x {X.shape[1]} in {time.perf_counter() - t0:.1f}s"
        + (f"  (dropped {DOMINANT_FEATURE})" if drop else ""),
        flush=True,
    )

    rows, lr_cache = [], {}
    for seed in SEEDS:
        t0 = time.perf_counter()
        rec = run_cell(
            X, y, seed, args.epochs, args.batch_size, lr_cache, args.eval_batches
        )
        rows.append(rec)
        a = rec["arms"]
        print(
            f"  [seed {seed}] fis {rec['fis']['error_rate']:.4f} | "
            f"hot {a['hot']['epoch0_error']:.4f}->{a['hot']['selected_error']:.4f} | "
            f"he-all {a['he-all']['epoch0_error']:.4f}->{a['he-all']['selected_error']:.4f} | "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    meta = {
        "seeds": SEEDS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rates": lr_cache,
        "top_n": FIS_TOP_N,
        "dropped": list(drop),
    }
    with open(guard_output(out_path, args.force), "w") as fh:
        json.dump({"meta": meta, "results": rows}, fh, indent=1)
    with open(md_path, "w") as fh:
        fh.write(summarize(rows, meta))
    print(f"\nwrote {os.path.relpath(md_path, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
