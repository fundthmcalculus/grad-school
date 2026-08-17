"""Select the FIS's own hyperparameters on validation, so both sides are honest.

The DOE's Factor-D grid minimizes `rmse_test_true` (`cmapss_rul.py:508`), and
`cmapss_rul_best.py`'s `PIPELINES` hardcodes the winners. So the FIS
configurations this benchmark compares against are **test-selected**, while
every network number is validation-selected. On the `best` bundle that matters:
the FIS lands at 6.48 against the network's 6.25, a margin smaller than the
advantage a test-set selection can buy.

This runs the same grid against the validation engines instead, then confirms
the winner once on test -- the FIS's number under the network's rules. The gap
between the two is the size of the selection advantage, measured rather than
argued about.

What it found, so the next reader does not have to re-run it: on `best` the two
protocols select the **identical** configuration, so the published 6.48 owes
nothing to test selection. On `honest` the validation protocol picks a *worse*
model (16.06 against 11.23) while scoring better on validation -- two engines is
not enough to select on. The objection is real in principle and empty in fact
here, which is a better answer than either assuming it away or conceding it.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import cmapss_data
import models

OUT = os.path.join(cmapss_data.REPO, "outputs", "nn-cmapss")

# The DOE's Factor-D grid, minus the corners it documents as unusable:
# `top_p=1.0` (blows the seconds budget on wide pipelines) and
# `norm_conorm='luk'` (numerically catastrophic at 37.99 RMSE).
D_GRID = dict(
    tsk_order=["0th", "1st", "full-2nd"],
    n_gaussians=[0, 3, 5],
    top_p=[0.90, 0.95],
    detect_interactions=[False],
    norm_conorm=["probability", "hamacher"],
    l2_reg=[1e-6, 0.01],
)


def run(which: str, out_name: str) -> None:
    b = cmapss_data.load_or_build(**cmapss_data.BUNDLES[which])
    names = b.feature_names
    keys = list(D_GRID)
    rows = []
    t_all = time.perf_counter()

    for combo in itertools.product(*D_GRID.values()):
        kwargs = dict(zip(keys, combo))
        try:
            fis, secs = models.fit_fis(b.fit.X, b.fit.y, names, **kwargs)
            m = models.evaluate(b.val, models.fis_predict(fis, b.val.X, names))
        except Exception as exc:  # noqa: BLE001
            rows.append(dict(**kwargs, error=f"{type(exc).__name__}: {exc}"))
            continue
        rows.append(
            dict(
                **kwargs,
                fit_seconds=secs,
                val_rmse=m["rmse"],
                val_rmse_endpoint=m["rmse_endpoint"],
                n_features_kept=len(fis.top_features_),
            )
        )

    ok = [r for r in rows if "val_rmse" in r]
    best = min(ok, key=lambda r: r["val_rmse"])
    cfg = {k: best[k] for k in keys}
    print(f"  best on validation: {cfg}")
    print(f"    val rmse {best['val_rmse']:.2f}  ({best['fit_seconds']:.2f}s fit)")

    fis, secs = models.fit_fis(b.train.X, b.train.y, names, **cfg)
    test = models.evaluate(b.test, models.fis_predict(fis, b.test.X, names))

    # The DOE's published configuration, refit and scored the same way, for the
    # side-by-side.
    doe_cfg = models.FIS_CONFIGS[which]
    fis_doe, secs_doe = models.fit_fis(b.train.X, b.train.y, names, **doe_cfg)
    test_doe = models.evaluate(b.test, models.fis_predict(fis_doe, b.test.X, names))
    val_doe = models.evaluate(
        b.val,
        models.fis_predict(
            models.fit_fis(b.fit.X, b.fit.y, names, **doe_cfg)[0], b.val.X, names
        ),
    )

    print(
        f"  val-selected config -> test rmse {test['rmse']:.2f} "
        f"(endpoint {test['rmse_endpoint']:.2f}), fit {secs:.2f}s"
    )
    print(
        f"  DOE (test-selected) -> test rmse {test_doe['rmse']:.2f} "
        f"(endpoint {test_doe['rmse_endpoint']:.2f}), fit {secs_doe:.2f}s, "
        f"val {val_doe['rmse']:.2f}"
    )
    print(
        f"  selection advantage: {test_doe['rmse'] - test['rmse']:+.2f} cycles "
        f"of test RMSE"
    )

    payload = dict(
        bundle=which,
        n_configs=len(rows),
        n_ok=len(ok),
        grid_seconds=time.perf_counter() - t_all,
        val_selected=dict(
            config=cfg,
            val=best["val_rmse"],
            test=test,
            fit_seconds=secs,
            n_features_kept=best["n_features_kept"],
        ),
        doe_selected=dict(
            config=doe_cfg, val=val_doe["rmse"], test=test_doe, fit_seconds=secs_doe
        ),
        rows=rows,
    )
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, out_name)
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    print(
        f"  wrote {os.path.relpath(path, cmapss_data.REPO)} "
        f"({len(ok)}/{len(rows)} configs ok, {payload['grid_seconds']:.0f}s)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("bundles", nargs="*", default=["honest", "best"])
    a = ap.parse_args()
    for which in a.bundles:
        print(f"=== FIS grid on validation: {which} ===")
        run(which, f"sweep_fis_{which}.json")
