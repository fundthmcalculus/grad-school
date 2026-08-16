"""The tetrahedral construction: does it fix the fidelity gap, and does it scale?

The first-order (axis-aligned) seed carries only the FIS's *additive* part, and
that is where every failure above one dimension came from -- fidelity 0.29 on
Concrete, 1.03 on bikeshare, 1.17 on WEC, where 0 would mean the seed reproduces
the FIS exactly. The tetrahedral construction has no such ceiling in principle:
a Freudenthal interpolant of the FIS carries interactions too.

Four things are measured, in the order the argument needs them.

**1. Cost.** A dense tetrahedral partition of ``K`` cells a side in ``n``
features has ``(K+1)**n`` rules -- 43 million for Concrete at ``K=8``, which is
the explosion `tribblefis.anfis` raises `RuleExplosionError` over. Nothing here
enumerates it: only vertices the data reaches are built, and the table reports
that count against the dense one it replaces.

**2. Where the consequents come from.** ``c_v = FIS(v)``, the literal reading of
the equivalence, is the one choice that needs no data -- and the one that fails
in high dimension, because a grid vertex sits off the data manifold where the
FIS's own output is extrapolation. Three estimators are compared.

**3. The statistical ceiling.** Computational scaling is not the binding
constraint; support is. Each vertex needs data behind it, the vertex count grows
with the subspace dimension while the row count does not, and the fidelity turns
erratic once vertices outnumber rows.

**4. The hybrid that follows.** Main effects in the additive seed, where all N
rows feed every 1-D profile; interactions in a tetrahedral basis over the few
features the FIS ranked highest, at a resolution chosen to keep support.

    python experiments/fis-to-neural-net/run_simplicial.py
    python experiments/fis-to-neural-net/run_simplicial.py --datasets synth1d concrete

Writes `simplicial_results.json` and `simplicial.md`.
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "reproduce", "tables"))

import fis2nn  # noqa: E402
import simplicial  # noqa: E402
from run_experiment import DATASETS, prepare, split  # noqa: E402
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("FIS2NN_SEEDS", "0,1,2,3,4").split(",")]
CONSEQUENT_MODES = ("vertex", "support", "project")
SUBSPACE_DIMS = (1, 2, 3, 4, 5)
FULL_RESOLUTIONS = (2, 4, 8)
MAX_VERTICES = 8192


def ridge_readout(Phi, X, y, l2=1e-6):
    """Closed-form ridge over ``[hats | X | 1]`` -- one solve, no epochs."""
    design = np.hstack([Phi, X, np.ones((len(X), 1))])
    penalty = l2 * np.eye(design.shape[1])
    penalty[-1, -1] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return beta[: Phi.shape[1]], beta[Phi.shape[1] : -1], float(beta[-1])


def run_cell(name, cfg, seed, l2, max_vertices):
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

    # `top_features_` comes back ranked by differentiation score, so a prefix of
    # it is "the k features the FIS thinks matter most" with no extra machinery.
    feats = list(reg.top_features_)
    columns = list(Xtr.columns)

    def fis_fn(frame):
        with contextlib.redirect_stdout(io.StringIO()):
            return np.asarray(reg.predict(frame), dtype=float).ravel()

    def scaled_fis(frame):
        return (fis_fn(frame) - y_center) / y_scale

    Xtr_a = Xtr[feats].to_numpy(dtype=float)
    Xte_a = Xte[feats].to_numpy(dtype=float)
    ytr_s = (np.asarray(y_tr, dtype=float) - y_center) / y_scale
    fis_tr_s, fis_te_s = scaled_fis(Xtr), scaled_fis(Xte)
    fis_sd = float(np.std(fis_te_s)) or 1.0

    def fidelity(pred_scaled):
        return fis2nn.rmse(fis_te_s, pred_scaled) / fis_sd

    def rmse_orig(pred_scaled):
        return fis2nn.rmse(y_te, np.asarray(pred_scaled) * y_scale + y_center)

    out = {
        "dataset": name,
        "seed": seed,
        "n_train": len(Xtr),
        "n_features": len(feats),
        "fis_seconds": fis_seconds,
        "fis_test_rmse": fis2nn.rmse(y_te, fis_fn(Xte)),
    }

    # --- incumbent: the first-order additive seed --------------------------
    t0 = time.perf_counter()
    knots = fis2nn.fis_knots(reg.model_, feats)
    add_net = fis2nn.analytic_seed_from_fis(
        scaled_fis, Xtr, feats, knots, background_size=256, seed=seed
    )
    add_seconds = time.perf_counter() - t0
    add_tr, add_te = add_net.predict(Xtr_a), add_net.predict(Xte_a)
    out["additive"] = {
        "seconds": add_seconds,
        "n_hidden": add_net.n_hidden,
        "fidelity": fidelity(add_te),
        "test_rmse": rmse_orig(add_te),
    }

    # --- 2. consequent estimators, on a full-dimensional grid --------------
    template = Xtr.iloc[[0]].copy()
    for col in columns:
        template[col] = float(Xtr[col].median())

    out["consequent_modes"] = []
    for res in FULL_RESOLUTIONS:
        origin, h = simplicial.grid_from_data(Xtr_a, res)
        vertices, support = simplicial.occupied_vertices(
            Xtr_a, origin, h, max_vertices=max_vertices
        )
        row = {
            "resolution": res,
            "vertices": int(len(vertices)),
            "rows_per_vertex": len(Xtr_a) / max(len(vertices), 1),
            "dense_grid": float(res + 1) ** len(feats),
        }
        for mode in CONSEQUENT_MODES:
            src = scaled_fis if mode == "vertex" else fis_tr_s
            try:
                c, bias = simplicial.consequents_from_fis(
                    mode,
                    src,
                    Xtr_a,
                    columns,
                    feats,
                    vertices,
                    origin,
                    h,
                    template,
                    l2=l2,
                )
                net = simplicial.SimplicialNet(
                    vertices=vertices,
                    origin=origin,
                    h=h,
                    c=c,
                    skip=np.zeros(len(feats)),
                    bias=bias,
                )
                row[mode] = fidelity(net.predict(Xte_a))
            except Exception as exc:  # noqa: BLE001
                row[mode] = None
                row[f"{mode}_error"] = type(exc).__name__
        row["relu_spec"] = simplicial.SimplicialNet(
            vertices, origin, h, np.zeros(len(vertices)), np.zeros(len(feats)), 0.0
        ).to_relu_spec()
        out["consequent_modes"].append(row)

    # --- 4. the hybrid: additive main effects + tetrahedral interactions ----
    residual_tr = fis_tr_s - add_tr
    out["hybrid"] = []
    for k in SUBSPACE_DIMS:
        if k > len(feats):
            continue
        t0 = time.perf_counter()
        corr = simplicial.fit_simplicial_correction(
            residual_tr, Xtr_a, list(range(k)), l2=l2
        )
        corr_seconds = time.perf_counter() - t0
        hybrid_te = add_te + corr.predict(Xte_a)
        hybrid_tr = add_tr + corr.predict(Xtr_a)

        # And what one closed-form solve on the labels buys in the same basis.
        t0 = time.perf_counter()
        Phi = corr.net.memberships(Xtr_a[:, corr.columns])
        c, skip, bias = ridge_readout(Phi, Xtr_a, ytr_s - add_tr, l2=l2)
        solve_seconds = time.perf_counter() - t0
        fitted = simplicial.SimplicialNet(
            corr.net.vertices, corr.net.origin, corr.net.h, c, skip, bias
        )
        fit_te = (
            add_te
            + fitted.memberships(Xte_a[:, corr.columns]) @ c
            + Xte_a @ skip
            + bias
        )

        out["hybrid"].append(
            {
                "k": k,
                "resolution": corr.resolution,
                "vertices": corr.net.n_hidden,
                "rows_per_vertex": corr.rows_per_vertex,
                "dense_grid": float(corr.resolution + 1) ** k,
                "relu_spec": corr.to_relu_spec(),
                "seconds": corr_seconds,
                "fidelity": fidelity(hybrid_te),
                "train_fidelity": fis2nn.rmse(fis_tr_s, hybrid_tr)
                / (np.std(fis_tr_s) or 1),
                "test_rmse": rmse_orig(hybrid_te),
                "solve_seconds": solve_seconds,
                "fit_test_rmse": rmse_orig(fit_te),
                "total_seconds": fis_seconds
                + add_seconds
                + corr_seconds
                + solve_seconds,
            }
        )

    return out


def mean(rows, fn):
    vals = [fn(r) for r in rows]
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def summarize(results):
    lines = [
        "# The tetrahedral construction",
        "",
        f"Seeds: {SEEDS}. **fidelity** is the converted model's RMSE against the "
        "FIS it came from, relative to that FIS output's own standard deviation: "
        "0 means the conversion reproduces the FIS exactly. The first-order "
        "additive construction is the number to beat.",
        "",
    ]
    by_ds: dict[str, list] = {}
    for r in results:
        by_ds.setdefault(r["dataset"], []).append(r)

    for ds, rows in by_ds.items():
        n_f = rows[0]["n_features"]
        n_tr = rows[0]["n_train"]
        lines += [
            f"## {ds} — {n_tr} rows, {n_f} features",
            "",
            f"FIS test RMSE {mean(rows, lambda r: r['fis_test_rmse']):.3f}, fit in "
            f"{mean(rows, lambda r: r['fis_seconds']):.2f} s. "
            f"Additive seed: fidelity **{mean(rows, lambda r: r['additive']['fidelity']):.3f}**, "
            f"{rows[0]['additive']['n_hidden']} units, "
            f"{mean(rows, lambda r: r['additive']['seconds']):.2f} s.",
            "",
            "### Where the consequents come from (full-dimensional grid)",
            "",
            "| K | vertices | dense grid | rows/vertex | `c_v = FIS(v)` | support-weighted | projected |",
            "|---|---|---|---|---|---|---|",
        ]
        for i in range(len(rows[0]["consequent_modes"])):
            recs = [r["consequent_modes"][i] for r in rows]

            def m(key):
                return mean(recs, lambda x: x.get(key))

            lines.append(
                f"| {recs[0]['resolution']} | {m('vertices'):.0f} | "
                f"{recs[0]['dense_grid']:.3g} | {m('rows_per_vertex'):.2f} | "
                f"{m('vertex'):.3f} | {m('support'):.3f} | {m('project'):.3f} |"
            )

        lines += [
            "",
            "### Hybrid: additive main effects + tetrahedral interactions on the top *k* features",
            "",
            "Grid resolution is chosen automatically to keep roughly "
            f"{simplicial.TARGET_ROWS_PER_VERTEX:.0f} rows behind every vertex.",
            "",
            "| k | K | vertices | dense grid | rows/vertex | ReLU units | depth |"
            " fidelity | test RMSE | +1 solve | total s |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for i in range(len(rows[0]["hybrid"])):
            recs = [r["hybrid"][i] for r in rows if i < len(r["hybrid"])]
            if not recs:
                continue

            def m(key):
                return mean(recs, lambda x: x.get(key))

            lines.append(
                f"| {recs[0]['k']} | {m('resolution'):.0f} | {m('vertices'):.0f} | "
                f"{m('dense_grid'):.0f} | {m('rows_per_vertex'):.1f} | "
                f"{mean(recs, lambda x: x['relu_spec']['relu_units']):.0f} | "
                f"{recs[0]['relu_spec']['depth']} | **{m('fidelity'):.3f}** | "
                f"{m('test_rmse'):.3f} | {m('fit_test_rmse'):.3f} | "
                f"{m('total_seconds'):.2f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--max-vertices", type=int, default=MAX_VERTICES)
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--out", default=os.path.join(HERE, "simplicial_results.json"))
    args = ap.parse_args()

    results = []
    for name in args.datasets:
        for seed in SEEDS:
            t0 = time.perf_counter()
            try:
                rec = run_cell(name, DATASETS[name], seed, args.l2, args.max_vertices)
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name} seed {seed}] FAILED {type(exc).__name__}: {exc}")
                continue
            results.append(rec)
            best = (
                min(rec["hybrid"], key=lambda h: h["fidelity"])
                if rec["hybrid"]
                else None
            )
            print(
                f"  [{name} seed {seed}] additive {rec['additive']['fidelity']:.3f} -> "
                + (
                    f"hybrid {best['fidelity']:.3f} (k={best['k']}, K={best['resolution']}, "
                    f"{best['vertices']} vertices)"
                    if best
                    else "no hybrid"
                )
                + f" | {time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    with open(args.out, "w") as fh:
        json.dump({"seeds": SEEDS, "results": results}, fh, indent=1)
    path = os.path.join(HERE, "simplicial.md")
    with open(path, "w") as fh:
        fh.write(summarize(results))
    print(f"\nwrote {os.path.relpath(path, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
