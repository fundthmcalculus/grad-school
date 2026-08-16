"""Two follow-ups the tetrahedral results pointed at, measured against each other.

`run_simplicial.py` established that the tetrahedral construction is exactly and
cheaply representable but support-bound, and that the usable form is a hybrid:
additive main effects plus a tetrahedral correction on a few features. It left
two arbitrary choices inside that hybrid, and this script removes both.

**Where the vertices sit.** The 2025 paper's interpolation is exact because its
triangulation comes from the linear regions of the object being converted. A
lattice over the data's bounding box has no such property. `simplicial.AxisWarp`
warps each axis until the FIS's own membership knots land on lattice integers,
so a unit cell is one inter-knot interval and the complex is aligned to the FIS's
structure -- while every hat stays exactly the closed form, because the warped
lattice is still regular. The warp is piecewise linear, so the composition is
still a ReLU circuit; it just costs one extra unit per interior knot per axis.

**Which features the correction spans.** The hybrid took the FIS's top-`k`
features by differentiation score, which ranks *main* effects -- an odd basis on
which to choose where to model *interactions*. `gauss_math.calculate_interaction_scores`
already scores feature pairs for joint lift beyond either alone, which is the
question actually being asked.

Crossing both gives four arms per subspace size, differing in exactly one
choice at a time:

    lattice  x importance     the incumbent from run_simplicial.py
    lattice  x interaction    subspace chosen by pair lift
    warped   x importance     vertices on the FIS's knots
    warped   x interaction    both

    python experiments/fis-to-neural-net/run_warped.py
    python experiments/fis-to-neural-net/run_warped.py --datasets synth1d concrete

Writes `warped_results.json` and `warped.md`. Leaves `run_simplicial.py`'s
outputs untouched.
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
sys.path.insert(0, os.path.join(REPO, "reproduce", "tables"))

import fis2nn  # noqa: E402
import simplicial  # noqa: E402
from run_experiment import DATASETS, prepare, split  # noqa: E402
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("FIS2NN_SEEDS", "0,1,2,3,4").split(",")]
SUBSPACE_DIMS = (2, 3, 4)
GEOMETRIES = ("lattice", "warped")
SELECTORS = ("importance", "interaction")


def interaction_ranking(reg, Xtr, y_tr, feats, buckets):
    """Feature indices ordered by the lift of the pairs they appear in.

    `calculate_interaction_scores` scores a *derived product column* with the
    same metric as the univariate ranking, and that metric sums a pairwise
    distance over every pair of **labels**. So it must be handed the bucketed
    target, not the raw one: passing 824 distinct y values turns an O(buckets^2)
    scoring into an O(n^2) one and the call never finishes. The regressor
    buckets internally via `partition_output`; this reproduces that step with
    the same arguments so the ranking sees exactly the labels the FIS saw.

    Computed once per cell and sliced per subspace size. Returns the ordering
    and the top positive lift, falling back to the importance order if the
    scorer is unavailable -- a missing ranking should degrade this arm to the
    incumbent, not crash the run.
    """
    try:
        from tribblefis.gauss_math import calculate_interaction_scores
        from tribblefis.regression import partition_output

        with contextlib.redirect_stdout(io.StringIO()):
            y_part, _means = partition_output(
                buckets, pd.Series(np.asarray(y_tr, dtype=float), name="y_value")
            )
            scores = calculate_interaction_scores(
                Xtr[feats].reset_index(drop=True),
                y_part["y_bucket"].reset_index(drop=True),
                reg.feature_differentiators_,
                candidate_pool=feats,
            )
    except Exception:  # noqa: BLE001
        return list(range(len(feats))), None

    index = {name: i for i, name in enumerate(feats)}
    order: list[int] = []
    top_lift = None
    for entry in scores or []:
        fi, fj, lift = entry[0], entry[1], float(entry[2])
        if lift <= 0:
            break
        if top_lift is None:
            top_lift = lift
        for nm in (fi, fj):
            i = index.get(nm)
            if i is not None and i not in order:
                order.append(i)
    for i in range(len(feats)):  # pad by importance rank
        if i not in order:
            order.append(i)
    return order, top_lift


def run_cell(name, cfg, seed, l2):
    X, y = cfg["loader"]()
    X_tr, y_tr, X_te, y_te = split(X, y, seed)
    Xtr, Xte, y_center, y_scale = prepare(X_tr, y_tr, X_te, y_te)

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        reg = TribbleRegressor(
            n_output_buckets=cfg["buckets"],
            tsk_order=cfg["order"],
            random_state=seed,
            **cfg.get("fis_kwargs", {}),
        )
        reg.fit(Xtr, y_tr)
    fis_seconds = time.perf_counter() - t0
    feats = list(reg.top_features_)

    def fis_fn(frame):
        with contextlib.redirect_stdout(io.StringIO()):
            return np.asarray(reg.predict(frame), dtype=float).ravel()

    def scaled_fis(frame):
        return (fis_fn(frame) - y_center) / y_scale

    Xtr_a = Xtr[feats].to_numpy(dtype=float)
    Xte_a = Xte[feats].to_numpy(dtype=float)
    fis_tr_s, fis_te_s = scaled_fis(Xtr), scaled_fis(Xte)
    fis_sd = float(np.std(fis_te_s)) or 1.0

    def fidelity(pred):
        return fis2nn.rmse(fis_te_s, pred) / fis_sd

    knots = fis2nn.fis_knots(reg.model_, feats)
    t0 = time.perf_counter()
    add_net = fis2nn.analytic_seed_from_fis(
        scaled_fis, Xtr, feats, knots, background_size=256, seed=seed
    )
    add_seconds = time.perf_counter() - t0
    add_tr, add_te = add_net.predict(Xtr_a), add_net.predict(Xte_a)
    residual = fis_tr_s - add_tr

    warp = simplicial.AxisWarp.from_knots(knots, feats)
    out = {
        "dataset": name,
        "seed": seed,
        "n_train": len(Xtr),
        "n_features": len(feats),
        "fis_seconds": fis_seconds,
        "fis_test_rmse": fis2nn.rmse(y_te, fis_fn(Xte)),
        "additive": {
            "fidelity": fidelity(add_te),
            "seconds": add_seconds,
            "n_hidden": add_net.n_hidden,
        },
        "warp_units": warp.relu_units(),
        "arms": [],
    }

    inter_order, top_lift = interaction_ranking(reg, Xtr, y_tr, feats, cfg["buckets"])
    out["top_lift"] = top_lift
    out["interaction_order"] = [feats[i] for i in inter_order[: max(SUBSPACE_DIMS)]]

    for k in SUBSPACE_DIMS:
        if k > len(feats):
            continue
        subspaces = {
            "importance": list(range(k)),
            "interaction": inter_order[:k],
        }
        for selector in SELECTORS:
            cols = subspaces[selector]
            for geometry in GEOMETRIES:
                t0 = time.perf_counter()
                if geometry == "warped":
                    corr = simplicial.fit_warped_correction(
                        residual, Xtr_a, cols, warp, l2=l2
                    )
                else:
                    corr = simplicial.fit_simplicial_correction(
                        residual, Xtr_a, cols, l2=l2
                    )
                seconds = time.perf_counter() - t0
                pred = add_te + corr.predict(Xte_a)
                out["arms"].append(
                    {
                        "k": k,
                        "selector": selector,
                        "geometry": geometry,
                        "columns": [feats[i] for i in cols],
                        "resolution": corr.resolution,
                        "vertices": corr.net.n_hidden,
                        "rows_per_vertex": corr.rows_per_vertex,
                        "relu_units": corr.to_relu_spec()["relu_units"],
                        "seconds": seconds,
                        "fidelity": fidelity(pred),
                        "test_rmse": fis2nn.rmse(
                            y_te, np.asarray(pred) * y_scale + y_center
                        ),
                        "total_seconds": fis_seconds + add_seconds + seconds,
                    }
                )
    return out


def mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(results):
    lines = [
        "# FIS-aligned vertices, and interaction-chosen subspaces",
        "",
        f"Seeds: {SEEDS}. **fidelity** is the conversion's RMSE against the FIS it "
        "came from, relative to that FIS output's own standard deviation -- 0 means "
        "it reproduces the FIS exactly. Two choices vary independently: where the "
        "lattice vertices sit (`lattice` = data bounding box, `warped` = on the "
        "FIS's own knots) and how the correction's subspace is chosen "
        "(`importance` = top-k by differentiation score, `interaction` = features "
        "spanned by the highest-lift pairs).",
        "",
    ]
    by_ds: dict[str, list] = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)

    for ds, rows in by_ds.items():
        add = mean([r["additive"]["fidelity"] for r in rows])
        lines += [
            f"## {ds} — {rows[0]['n_train']} rows, {rows[0]['n_features']} features",
            "",
            f"Additive seed fidelity **{add:.3f}**. "
            f"Warp costs {mean([r['warp_units'] for r in rows]):.0f} extra ReLU units "
            "across all axes.",
            "",
            "| k | subspace | geometry | K | vertices | rows/vertex | ReLU units | fidelity | vs additive | s |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        keys = []
        for r in rows:
            for a in r["arms"]:
                key = (a["k"], a["selector"], a["geometry"])
                if key not in keys:
                    keys.append(key)
        for key in sorted(keys):
            recs = [
                a
                for r in rows
                for a in r["arms"]
                if (a["k"], a["selector"], a["geometry"]) == key
            ]
            fid = mean([a["fidelity"] for a in recs])
            delta = (
                (add - fid) / add * 100 if np.isfinite(add) and add else float("nan")
            )
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | "
                f"{mean([a['resolution'] for a in recs]):.0f} | "
                f"{mean([a['vertices'] for a in recs]):.0f} | "
                f"{mean([a['rows_per_vertex'] for a in recs]):.1f} | "
                f"{mean([a['relu_units'] for a in recs]):.0f} | "
                f"**{fid:.3f}** | {delta:+.0f}% | "
                f"{mean([a['seconds'] for a in recs]):.2f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--out", default=os.path.join(HERE, "warped_results.json"))
    args = ap.parse_args()

    results = []
    for name in args.datasets:
        for seed in SEEDS:
            t0 = time.perf_counter()
            try:
                rec = run_cell(name, DATASETS[name], seed, args.l2)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name} seed {seed}] FAILED {type(exc).__name__}: {exc}")
                continue
            results.append(rec)
            best = (
                min(rec["arms"], key=lambda a: a["fidelity"]) if rec["arms"] else None
            )
            print(
                f"  [{name} seed {seed}] additive {rec['additive']['fidelity']:.3f} -> "
                + (
                    f"best {best['fidelity']:.3f} "
                    f"({best['geometry']}/{best['selector']}, k={best['k']})"
                    if best
                    else "no arms"
                )
                + f" | {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    with open(args.out, "w") as fh:
        json.dump({"seeds": SEEDS, "results": results}, fh, indent=1)
    path = os.path.join(HERE, "warped.md")
    with open(path, "w") as fh:
        fh.write(summarize(results))
    print(f"\nwrote {os.path.relpath(path, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
