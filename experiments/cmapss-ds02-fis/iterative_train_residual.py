"""Iterative methods to reduce the *training* residual on N-CMAPSS DS02.

The single DS02 fuzzy system is bias-limited: train RMSE ~= test RMSE (both ~6.5
cycles), so the model is not overfitting -- it has room to fit the training data
harder. This script asks how far several iterative capacity-adding schemes push
the training residual down, and what each one costs on the held-out engines
(the honest counterweight: a method that only shrinks train error while test
error climbs has bought overfitting, not skill).

Methods compared (all wrap the same TRIBBLE `TribbleRegressor`):

  * baseline            -- one full-2nd, 2-bucket fuzzy system (the DS02 default).
  * residual boosting   -- additive stages, each new system fit on the running
                           residual with shrinkage `eta`; F <- F + eta * g_m.
                           Two base learners: a *weak* 1st-order/2-bucket stump
                           (classic slow boosting) and the *strong* full-2nd
                           default (aggressive). Swept over stages x eta.
  * staged rule growth  -- one system, but more rules (output buckets): capacity
                           added by partitioning the output finer, not additively.

Writes a CSV + a train-vs-test descent plot to outputs/ds02-iterative/. Run from
the repo root:

    python experiments/cmapss-ds02-fis/iterative_train_residual.py
"""

import contextlib
import csv
import io
import os
import time

import numpy as np  # noqa: E402
from _ds02_harness import bootstrap, load, rmse  # noqa: E402

bootstrap(os.path.dirname(__file__))

from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

OUT = "outputs/ds02-iterative"
os.makedirs(OUT, exist_ok=True)

# DS02 default base-learner knobs, shared by baseline and the "strong" booster.
STRONG = dict(
    tsk_order="full-2nd",
    top_p=0.95,
    n_output_buckets=2,
    norm_conorm="hamacher",
    l2_reg=0.01,
    max_samples=2000,
)
# A deliberately weak stump: 1st-order consequents, 2 buckets, same norm.
WEAK = dict(
    tsk_order="1st",
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


def boost(d, base_cfg, n_stages, eta):
    """Additive residual boosting. Returns per-stage (train_rmse, test_rmse)."""
    X_tr, y_tr, X_te, y_te = d["X_tr"], d["y_tr"], d["X_te"], d["y_te"]
    F_tr = np.zeros_like(y_tr)
    F_te = np.zeros(len(y_te))
    curve = []
    for m in range(n_stages):
        resid = y_tr - F_tr
        step = 1.0 if m == 0 else eta  # first stage full-step (fit y itself)
        reg = _fit(base_cfg, X_tr, resid, seed=42 + m)
        F_tr = F_tr + step * reg.predict(X_tr)
        F_te = F_te + step * reg.predict(X_te)
        curve.append((rmse(y_tr, F_tr), rmse(y_te, F_te)))
    return curve


def staged_rules(d, base_cfg, bucket_grid):
    X_tr, y_tr, X_te, y_te = d["X_tr"], d["y_tr"], d["X_te"], d["y_te"]
    out = []
    for nb in bucket_grid:
        cfg = {**base_cfg, "n_output_buckets": nb}
        reg = _fit(cfg, X_tr, y_tr, seed=42)
        out.append(
            (
                nb,
                int(reg.model_.n_rules),
                rmse(y_tr, reg.predict(X_tr)),
                rmse(y_te, reg.predict(X_te)),
            )
        )
    return out


def main():
    t0 = time.perf_counter()
    print("Loading + featurising DS02 ...")
    d = load()
    print(f"  train {d['X_tr'].shape}  test {d['X_te'].shape}")

    rows = []  # method, variant, stage/param, n_rules, train, test

    base = _fit(STRONG, d["X_tr"], d["y_tr"], seed=42)
    b_tr = rmse(d["y_tr"], base.predict(d["X_tr"]))
    b_te = rmse(d["y_te"], base.predict(d["X_te"]))
    print(f"\nbaseline (full-2nd, 2 buckets): train {b_tr:.3f}  test {b_te:.3f}")
    rows.append(("baseline", "full-2nd/2", 1, int(base.model_.n_rules), b_tr, b_te))

    N_STAGES = 12
    print("\n== residual boosting ==")
    for tag, cfg in (("weak(1st/2)", WEAK), ("strong(full-2nd/2)", STRONG)):
        for eta in (1.0, 0.5, 0.3):
            curve = boost(d, cfg, N_STAGES, eta)
            best = min(range(len(curve)), key=lambda i: curve[i][0])
            print(
                f"  {tag:20s} eta={eta:>3}: "
                f"train {curve[0][0]:.2f}->{curve[-1][0]:.2f} "
                f"(min {curve[best][0]:.2f}@{best+1})  "
                f"test end {curve[-1][1]:.2f}"
            )
            for m, (tr, te) in enumerate(curve):
                rows.append((f"boost:{tag}", f"eta={eta}", m + 1, None, tr, te))

    print("\n== staged rule growth ==")
    for tag, cfg in (("full-2nd", STRONG), ("1st", WEAK)):
        for nb, nr, tr, te in staged_rules(d, cfg, [2, 3, 4, 6, 8, 12]):
            print(
                f"  {tag:9s} buckets={nb:2d} rules={nr:2d}: train {tr:.3f}  test {te:.3f}"
            )
            rows.append((f"rules:{tag}", f"buckets={nb}", nb, nr, tr, te))

    csv_path = os.path.join(OUT, "iterative_train_residual.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "variant", "param", "n_rules", "train_rmse", "test_rmse"])
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
        print(f"  (skipping plot: {exc})")
        return
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    base = next(r for r in rows if r[0] == "baseline")
    for ax in (axL, axR):
        ax.axhline(base[4], ls=":", c="k", lw=1, label="baseline train")
        ax.axhline(base[5], ls="--", c="0.5", lw=1, label="baseline test")
    # boosting curves
    boosts = {}
    for r in rows:
        if r[0].startswith("boost:"):
            boosts.setdefault((r[0], r[1]), []).append((r[2], r[4], r[5]))
    for (method, variant), pts in boosts.items():
        pts.sort()
        st = [p[0] for p in pts]
        ax = axL if "weak" in method else axR
        ax.plot(st, [p[1] for p in pts], "-o", ms=3, label=f"{variant} train")
        ax.plot(st, [p[2] for p in pts], "--", alpha=0.6, label=f"{variant} test")
    axL.set_title("residual boosting: weak base (1st/2)")
    axR.set_title("residual boosting: strong base (full-2nd/2)")
    for ax in (axL, axR):
        ax.set_xlabel("boosting stage")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axL.set_ylabel("RMSE (cycles)")
    fig.suptitle("DS02: iterative reduction of the training residual", y=1.02)
    fig.tight_layout()
    p = os.path.join(OUT, "iterative_train_residual.png")
    fig.savefig(p, dpi=130, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
