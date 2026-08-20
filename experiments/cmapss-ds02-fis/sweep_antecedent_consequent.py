"""#6 antecedent granularity (n_gaussians) and #7 RBF consequents on DS02.

Both are regressor-internal knobs, so they reuse the default DS02 featurisation
(_ds02_harness.load) -- no re-featurising. Baseline is the shipped DS02 default
(full-2nd, 2 buckets, top_p 0.95, hamacher; per-sample test RMSE ~6.48).

  #6  n_gaussians: Gaussians per feature per output bucket. 0 = automatic
      (~unimodal). >1 gives multi-modal antecedents -- genuine antecedent
      capacity the fuzzy semantics support (unlike boosting).
  #7  consequent_basis="gaussian-rbf": nonlinear consequents instead of the
      raw polynomial. Swept over rbf_n_centers x rbf_gamma; a shot at cutting
      DS02's *bias* (where more rules / boosting could not).

Writes CSV to outputs/ds02-iterative/. Run from the repo root:

    python experiments/cmapss-ds02-fis/sweep_antecedent_consequent.py
"""

import contextlib
import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _ds02_harness import load, rmse  # noqa: E402

from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

OUT = "outputs/ds02-iterative"
os.makedirs(OUT, exist_ok=True)
BASE = dict(
    top_p=0.95,
    n_output_buckets=2,
    norm_conorm="hamacher",
    l2_reg=0.01,
    max_samples=2000,
)


def fit_eval(cfg, d):
    reg = TribbleRegressor(random_state=42, **cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        reg.fit(d["X_tr"], d["y_tr"])
    return (
        rmse(d["y_tr"], reg.predict(d["X_tr"])),
        rmse(d["y_te"], reg.predict(d["X_te"])),
    )


def main():
    print("Loading + featurising DS02 ...")
    d = load()
    rows = []

    def run(sweep, param, cfg):
        try:
            a, b = fit_eval(cfg, d)
            print(f"  {sweep:12s} {str(param):26s} train {a:5.2f}  test {b:5.2f}")
            rows.append((sweep, str(param), a, b))
        except Exception as exc:  # noqa: BLE001
            print(f"  {sweep:12s} {str(param):26s} FAILED: {type(exc).__name__}: {exc}")
            rows.append((sweep, str(param), None, f"ERR:{type(exc).__name__}"))

    print("\n== baseline (full-2nd) ==")
    run("baseline", "full-2nd/auto-gauss", {**BASE, "tsk_order": "full-2nd"})

    # #6 antecedent granularity, at both full-2nd and the cheaper 1st order
    print("\n== #6 n_gaussians ==")
    for order in ("full-2nd", "1st"):
        for ng in (1, 2, 3):
            run(
                "n_gaussians",
                f"{order}/ng={ng}",
                {**BASE, "tsk_order": order, "n_gaussians": ng},
            )

    # #7 RBF consequents
    print("\n== #7 gaussian-rbf consequents ==")
    for order in ("1st", "2nd"):
        for nc in (2, 3, 5):
            for gamma in (0.5, 1.0, 2.0):
                run(
                    "rbf",
                    f"{order}/nc={nc}/g={gamma}",
                    {
                        **BASE,
                        "tsk_order": order,
                        "consequent_basis": "gaussian-rbf",
                        "rbf_n_centers": nc,
                        "rbf_gamma": gamma,
                    },
                )

    with open(
        os.path.join(OUT, "sweep_antecedent_consequent.csv"), "w", newline=""
    ) as f:
        w = csv.writer(f)
        w.writerow(["sweep", "param", "train_rmse", "test_rmse"])
        w.writerows(rows)
    print(f"\nwrote {os.path.join(OUT, 'sweep_antecedent_consequent.csv')}")


if __name__ == "__main__":
    main()
