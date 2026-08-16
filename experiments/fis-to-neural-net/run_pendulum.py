"""The warm start on a genuinely slow-converging problem.

`find_slow_problem.py` ranked candidates by minibatch updates to reach R2 >= 0.9.
PhiUSIIL needed **25**; Concrete needed 386; the damped double-pendulum
time-step operator needs **3,444** -- 138x PhiUSIIL, on 62,000 real rows. That is
the first problem in this whole experiment where the cost of a TRIBBLE fit and
the cost of the training it would replace are the same order of magnitude, which
is the only regime in which a warm start can repay itself.

The target is the operator from `AnalyticalDynamics/chaos`:

    (theta_1(0), theta_2(0), t)  ->  theta_1(t)

for the damped (friction) n=2 chain. Friction rather than the frictionless case
on purpose: it is the better-conditioned of the two, and the frictionless one is
capacity-limited at this width rather than merely slow (it stalls at R2 0.76
however long it trains, so it measures the network's size, not its convergence).

Two conversions are compared, because Part 4 showed the choice matters:

    hot        knots from the FIS, read-out projected onto the FIS's own output
               by one ridge solve -- no labels
    hot-anova  the partial-dependence route of Parts 1-3 -- no labels

Protocol note: rows are split at random, matching how the 3,444-update figure
was measured, so the numbers here are directly comparable to it. That is *not*
the chaos study's own protocol, which holds out a whole in-between initial
condition and is a much harder extrapolation test. This measures convergence
speed, not the generalization claim that directory makes.

    python experiments/fis-to-neural-net/run_pendulum.py
    FIS2NN_SEEDS=0 python experiments/fis-to-neural-net/run_pendulum.py --epochs 20

Writes `pendulum_results.json` and `pendulum.md`.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "AnalyticalDynamics", "chaos"))

import fis2nn  # noqa: E402
from find_slow_problem import p_pendulum  # noqa: E402
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("FIS2NN_SEEDS", "0,1,2").split(",")]
LR_GRID = (1e-3, 3e-3, 1e-2)
COLUMNS = ["theta1_0", "theta2_0", "t"]
ARMS = ("hot", "hot-anova", "quantile", "he")

#: Output buckets for the FIS. Swept in the probe: R2 0.50 at 4, 0.74 at 8,
#: 0.84 at 16, for 0.48/0.68/1.28 s of fit. Sixteen is the setting used here --
#: the warm start's ceiling is the FIS's own quality, so a deliberately weak FIS
#: would make the comparison meaningless in the arm's own favour by giving it
#: nothing to lose.
BUCKETS = 16

#: R2 levels each arm has to reach. 0.90 is the level `find_slow_problem.py`
#: measured 3,444 updates against, so it is the directly comparable row.
TARGETS = (0.80, 0.85, 0.90, 0.93, 0.95)

#: The frictionless chain is a different problem and needs its own settings.
#:
#: It is NOT capacity-limited, which is what a first look suggested: widening
#: the network from 128 to 1024 units moves its best R2 only from 0.725 to
#: 0.771, and the FIS plateaus in the same place (0.558 at 16 buckets, 0.757 at
#: 64). Both methods hit the same ceiling from opposite directions, which is the
#: signature of an irreducible component rather than an under-powered model:
#: without damping the double pendulum's trajectories separate exponentially, so
#: past some horizon in `t` the map (theta_2(0), t) -> theta_1(t) is not a
#: function anything can learn from a 0.1-degree grid of initial conditions.
#: R2 0.9 is therefore unreachable here, not slow, and asking for it would
#: measure nothing. The targets below are the ones this problem admits.
FRICTIONLESS_BUCKETS = 32
FRICTIONLESS_TARGETS = (0.40, 0.50, 0.60, 0.65, 0.70)


def prepare(seed, friction=True):
    X, y, note = p_pendulum(2, friction)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_te = int(0.15 * len(X))
    n_val = int(0.15 * len(X))
    te, val, tr = idx[:n_te], idx[n_te : n_te + n_val], idx[n_te + n_val :]

    frames = {
        k: pd.DataFrame(X[i], columns=COLUMNS)
        for k, i in (("tr", tr), ("val", val), ("te", te))
    }
    mu, sd = frames["tr"].mean(), frames["tr"].std().replace(0, 1.0)
    scaled = {k: (v - mu) / sd for k, v in frames.items()}
    ym, ys = float(y[tr].mean()), float(y[tr].std()) or 1.0
    targets = {k: (y[i] - ym) / ys for k, i in (("tr", tr), ("val", val), ("te", te))}
    return scaled, targets, note


def r2_curve(hist, y_true):
    """Turn the trainer's standardized-RMSE curve into R2 on the same rows."""
    var = float(np.var(np.asarray(y_true))) or 1.0
    return 1.0 - np.asarray(hist, dtype=float) ** 2 / var


def run_cell(seed, epochs, batch_size, eval_batches, l2, friction, buckets, targets_r2):
    scaled, targets, note = prepare(seed, friction)
    Xtr, Xval, Xte = (scaled[k] for k in ("tr", "val", "te"))
    ytr, yval, yte = (targets[k] for k in ("tr", "val", "te"))

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        reg = TribbleRegressor(
            n_output_buckets=buckets, tsk_order="1st", random_state=seed
        )
        reg.fit(Xtr, pd.Series(ytr, name="y_value"))
    fis_seconds = time.perf_counter() - t0

    feats = list(reg.top_features_)

    def fis_fn(frame):
        with contextlib.redirect_stdout(io.StringIO()):
            return np.asarray(reg.predict(frame), dtype=float).ravel()

    Atr, Aval, Ate = (f[feats].to_numpy(dtype=float) for f in (Xtr, Xval, Xte))
    fis_te = fis_fn(Xte)

    out = {
        "seed": seed,
        "note": note,
        "n_train": len(Atr),
        "n_test": len(Ate),
        "features_kept": feats,
        "fis": {
            "seconds": fis_seconds,
            "r2": fis2nn.r2(yte, fis_te),
            "n_mfs": int(reg.model_.n_membership_functions),
        },
    }

    knots = fis2nn.fis_knots(reg.model_, feats)
    pairs = [(i, knots[f]) for i, f in enumerate(feats) if knots[f].size]
    basis = fis2nn._axis_aligned_net(len(feats), pairs)

    t0 = time.perf_counter()
    net_hot = fis2nn.solve_readout(basis, Atr, fis_fn(Xtr), l2=l2, anchor=False)
    convert_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    net_anova = fis2nn.analytic_seed_from_fis(
        fis_fn, Xtr, feats, knots, background_size=256, seed=seed
    )
    anova_seconds = time.perf_counter() - t0

    n_hidden = net_hot.n_hidden
    out["conversion"] = {
        "n_hidden": n_hidden,
        "seconds": convert_seconds,
        "anova_seconds": anova_seconds,
        "seed_r2": fis2nn.r2(yte, net_hot.predict(Ate)),
        "anova_seed_r2": fis2nn.r2(yte, net_anova.predict(Ate)),
    }

    rng = np.random.default_rng(1000 + seed)
    t0 = time.perf_counter()
    net_quantile = fis2nn.quantile_start(Atr, n_hidden, ytr, l2=l2)
    quantile_seconds = time.perf_counter() - t0
    starts = {
        "hot": net_hot,
        "hot-anova": net_anova,
        "quantile": net_quantile,
        "he": fis2nn.he_start(rng, Atr.shape[1], n_hidden),
    }
    # `quantile` gets its read-out from one ridge solve too, so it is charged
    # for it -- otherwise the arm that does the most work before epoch 0 is the
    # only one shown as free. It also fits the *labels* directly, where the hot
    # arms fit the FIS's output and never see y during setup; that asymmetry is
    # deliberate (it is what "no labels" means) and favours quantile.
    setup = {
        "hot": fis_seconds + convert_seconds,
        "hot-anova": fis_seconds + anova_seconds,
        "quantile": quantile_seconds,
        "he": 0.0,
    }
    out["conversion"]["quantile_seconds"] = quantile_seconds

    n_batches = max(1, int(np.ceil(len(Atr) / batch_size)))
    out["arms"] = {}
    for arm in ARMS:
        swept = {}
        for lr in LR_GRID:
            _, h = fis2nn.train_adam(
                starts[arm],
                Atr,
                ytr,
                X_test=Ate,
                y_test=yte,
                X_val=Aval,
                y_val=yval,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                seed=seed,
                eval_batches=eval_batches,
                track_train=False,
            )
            swept[lr] = float(np.min(h.val_rmse))
        lr = min(swept, key=swept.get)

        t0 = time.perf_counter()
        trained, hist = fis2nn.train_adam(
            starts[arm],
            Atr,
            ytr,
            X_test=Ate,
            y_test=yte,
            X_val=Aval,
            y_val=yval,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            seed=seed,
            eval_batches=eval_batches,
            track_train=False,
        )
        seconds = time.perf_counter() - t0
        curve = r2_curve(hist.test_rmse, yte)
        per_update = seconds / max(epochs * n_batches, 1)

        rec = {
            "lr": lr,
            "epoch0_r2": float(curve[0]),
            "final_r2": float(curve[-1]),
            "best_r2": float(np.nanmax(curve)),
            "setup_seconds": setup[arm],
            "train_seconds": seconds,
            "seconds_per_update": per_update,
            "curve_updates": [float(e) * n_batches for e in hist.epochs],
            "curve_r2": [float(v) for v in curve],
        }
        for tgt in targets_r2:
            hit = np.flatnonzero(curve >= tgt)
            if hit.size:
                u = float(rec["curve_updates"][hit[0]])
                rec[f"updates_to_{tgt}"] = u
                rec[f"seconds_to_{tgt}"] = setup[arm] + u * per_update
            else:
                rec[f"updates_to_{tgt}"] = None
                rec[f"seconds_to_{tgt}"] = None
        out["arms"][arm] = rec
    return out


def summarize(rows, meta):
    TARGETS = tuple(meta["targets"])

    def m(fn):
        vals = [fn(r) for r in rows]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else None

    kind = "Damped" if meta["friction"] else "Frictionless"
    lines = [
        f"# The warm start on a slow-converging problem — {kind.lower()} n=2",
        "",
        f"{kind} n=2 double-pendulum time-step operator, {rows[0]['n_train']:,} train / "
        f"{rows[0]['n_test']:,} test rows, 3 inputs · seeds {meta['seeds']} · "
        f"{meta['epochs']} epochs · batch {meta['batch_size']}.",
        "",
        (
            "This is the problem `find_slow_problem.py` measured at **3,444 updates** "
            "for a from-scratch network to reach R2 0.9 — against **25** for PhiUSIIL."
            if meta["friction"]
            else "R2 0.9 is unreachable here, not merely slow: widening the network "
            "from 128 to 1024 units moves its ceiling only from 0.725 to 0.771, and "
            "the FIS plateaus in the same place. Without damping the trajectories "
            "separate exponentially, so past some horizon in `t` the operator is not "
            "a learnable function of a 0.1-degree initial-condition grid. The targets "
            "below are the ones this problem admits."
        )
        + " The FIS fit and the conversion are charged to the hot arms.",
        "",
        "| model | R2 at start | best R2 | setup s | s/update |",
        "|---|---|---|---|---|",
        f"| tribble FIS ({meta['buckets']} buckets, {m(lambda r: r['fis']['n_mfs']):.0f} MFs) | — | "
        f"{m(lambda r: r['fis']['r2']):.4f} | {m(lambda r: r['fis']['seconds']):.2f} | — |",
    ]
    for arm in ARMS:
        lines.append(
            f"| nn-{arm} (lr {rows[0]['arms'][arm]['lr']:g}) | "
            f"{m(lambda r: r['arms'][arm]['epoch0_r2']):.4f} | "
            f"{m(lambda r: r['arms'][arm]['best_r2']):.4f} | "
            f"{m(lambda r: r['arms'][arm]['setup_seconds']):.2f} | "
            f"{1000 * m(lambda r: r['arms'][arm]['seconds_per_update']):.2f} ms |"
        )

    lines += [
        "",
        f"Conversion: {m(lambda r: r['conversion']['n_hidden']):.0f} hidden units. "
        f"The projection seed starts at R2 {m(lambda r: r['conversion']['seed_r2']):.4f} "
        f"against the FIS's own {m(lambda r: r['fis']['r2']):.4f}; the "
        "partial-dependence seed starts at "
        f"{m(lambda r: r['conversion']['anova_seed_r2']):.4f}.",
        "",
        "## Time to target — the whole question",
        "",
        "Wall clock to *first* reach each R2, FIS fit and conversion included in the "
        "hot arms' totals.",
        "",
        "| arm | " + " | ".join(f"R2 {t}" for t in TARGETS) + " |",
        "|" + "---|" * (len(TARGETS) + 1),
    ]
    secs = {}
    for arm in ARMS:
        cells = []
        for t in TARGETS:
            v = m(lambda r: r["arms"][arm][f"seconds_to_{t}"])
            n_hit = sum(
                1 for r in rows if r["arms"][arm][f"seconds_to_{t}"] is not None
            )
            secs.setdefault(arm, []).append(v)
            if v is None:
                cells.append("never")
            else:
                cells.append(
                    f"{v:.2f}s"
                    + (f" ({n_hit}/{len(rows)})" if n_hit < len(rows) else "")
                )
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Updates to the same targets (setup excluded):",
        "",
        "| arm | " + " | ".join(f"R2 {t}" for t in TARGETS) + " |",
        "|" + "---|" * (len(TARGETS) + 1),
    ]
    for arm in ARMS:
        cells = []
        for t in TARGETS:
            v = m(lambda r: r["arms"][arm][f"updates_to_{t}"])
            cells.append("never" if v is None else f"{v:,.0f}")
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Speedup of `nn-hot` over each arm, wall clock at the same target:",
        "",
        "| arm | " + " | ".join(f"R2 {t}" for t in TARGETS) + " |",
        "|" + "---|" * (len(TARGETS) + 1),
    ]
    for arm in ARMS:
        if arm == "hot":
            continue
        cells = []
        for i in range(len(TARGETS)):
            mine, theirs = secs["hot"][i], secs[arm][i]
            if mine is None:
                cells.append("hot never")
            elif theirs is None:
                cells.append("only hot arrives")
            else:
                cells.append(f"{theirs / mine:.2f}x")
        lines.append(f"| vs `{arm}` | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument(
        "--frictionless",
        action="store_true",
        help="the undamped chain: a different problem, see FRICTIONLESS_*",
    )
    ap.add_argument("--buckets", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    friction = not args.frictionless
    buckets = args.buckets or (BUCKETS if friction else FRICTIONLESS_BUCKETS)
    targets_r2 = TARGETS if friction else FRICTIONLESS_TARGETS
    tag = "" if friction else "_frictionless"
    out_path = args.out or os.path.join(HERE, f"pendulum{tag}_results.json")
    md_path = os.path.join(HERE, f"pendulum{tag}.md")

    rows = []
    for seed in SEEDS:
        t0 = time.perf_counter()
        rec = run_cell(
            seed,
            args.epochs,
            args.batch_size,
            args.eval_batches,
            args.l2,
            friction,
            buckets,
            targets_r2,
        )
        rows.append(rec)
        a = rec["arms"]
        print(
            f"  [seed {seed}] fis R2 {rec['fis']['r2']:.4f} | "
            f"hot {a['hot']['epoch0_r2']:.4f}->{a['hot']['best_r2']:.4f} | "
            f"he {a['he']['epoch0_r2']:.4f}->{a['he']['best_r2']:.4f} | "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    meta = {
        "seeds": SEEDS,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "buckets": buckets,
        "targets": list(targets_r2),
        "friction": friction,
    }
    with open(out_path, "w") as fh:
        json.dump({"meta": meta, "results": rows}, fh, indent=1)
    with open(md_path, "w") as fh:
        fh.write(summarize(rows, meta))
    print(f"\nwrote {os.path.relpath(md_path, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
