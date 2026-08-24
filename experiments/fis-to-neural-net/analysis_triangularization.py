"""Why the triangularized FIS collapses, measured as a function of feature count.

H2 said replacing the fitted FIS's Gaussians with `tribblefis.triangle_fit`'s
MAE-optimal triangles would cost little accuracy. On Concrete it costs
everything. This script isolates the mechanism rather than leaving it as an
anomaly in a results table, because the mechanism is the interesting part: it is
not a bad triangle fit, it is compact support meeting a product t-norm.

A Gaussian is positive everywhere, so every rule fires a little for every input
and the firing-strength normalization always has something to divide by. A
triangle is zero outside its feet. Under the product t-norm a rule's strength is
the product over features, so **one** feature landing outside its triangles
zeroes that rule -- and if that happens for every rule, the row's total firing
is zero and `_normalize_firing_strengths`'s documented convention returns 0 for
it. The probability of at least one such feature compounds with the number of
features, so the failure is a dimension effect, and this script shows the curve.

    python experiments/fis-to-neural-net/analysis_triangularization.py

Writes `outputs/triangularization.md`.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

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
from run_experiment import DATASETS, prepare, split  # noqa: E402
from tribblefis.gauss_math import tsk_firing_strengths  # noqa: E402
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402
from tribblefis.regression import predict_tsk  # noqa: E402
from tribblefis.triangle_fit import fit_triangles_to_mixture  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("FIS2NN_SEEDS", "0,1,2,3,4").split(",")]
# Feature counts to sweep on Concrete. `top_n` is TRIBBLE's own knob, so the
# sweep changes only how many features the FIS is allowed to keep.
FEATURE_COUNTS = [1, 2, 3, 4, 5, 6, 7, 8]


def dead_row_fraction(X, model, feats):
    with contextlib.redirect_stdout(io.StringIO()):
        fs, _ = tsk_firing_strengths(X[feats], model, norms=None)
    return float(np.mean(fs.sum(axis=1) <= 1e-6))


def one_cell(X, y, seed, top_n, buckets=4, order="1st"):
    X_tr, y_tr, X_te, y_te = split(X, y, seed)
    Xtr, Xte, _yc, _ys = prepare(X_tr, y_tr, X_te, y_te)
    with contextlib.redirect_stdout(io.StringIO()):
        reg = TribbleRegressor(
            n_output_buckets=buckets, tsk_order=order, random_state=seed, top_n=top_n
        )
        reg.fit(Xtr, y_tr)
    feats = list(reg.top_features_)
    tri = fit_triangles_to_mixture(reg.model_)

    def predict(model):
        with contextlib.redirect_stdout(io.StringIO()):
            return np.asarray(
                predict_tsk(
                    Xte,
                    model,
                    feats,
                    reg.y_bucket_mean_,
                    reg.corr_terms_,
                    order=reg.tsk_order,
                    basis=reg.consequent_basis,
                )
            ).ravel()

    return {
        "n_features": len(feats),
        "gauss_rmse": fis2nn.rmse(y_te, predict(reg.model_)),
        "tri_rmse": fis2nn.rmse(y_te, predict(tri)),
        "gauss_dead": dead_row_fraction(Xte, reg.model_, feats),
        "tri_dead": dead_row_fraction(Xte, tri, feats),
    }


def main() -> int:
    lines = [
        "# Why triangularization collapses (H2)",
        "",
        "Concrete, `top_n` swept over TRIBBLE's own feature selector, "
        f"{len(SEEDS)} seeds. **dead rows** = test rows whose total firing "
        "strength is <= 1e-6 across every rule, which "
        "`regression._normalize_firing_strengths` maps to a prediction of 0.",
        "",
        "| features kept | Gaussian RMSE | triangular RMSE | Gaussian dead rows | triangular dead rows |",
        "|---|---|---|---|---|",
    ]
    X, y = fm.load_concrete()
    for top_n in FEATURE_COUNTS:
        cells = [one_cell(X, y, s, top_n) for s in SEEDS]
        n_f = int(np.mean([c["n_features"] for c in cells]))
        g = np.mean([c["gauss_rmse"] for c in cells])
        t = np.mean([c["tri_rmse"] for c in cells])
        gd = np.mean([c["gauss_dead"] for c in cells])
        td = np.mean([c["tri_dead"] for c in cells])
        lines.append(
            f"| {n_f} | {g:.2f} | {t:.2f} | {100 * gd:.1f}% | {100 * td:.1f}% |"
        )
        print(lines[-1], flush=True)

    lines += [
        "",
        "Other datasets at their experiment settings:",
        "",
        "| dataset | features kept | Gaussian RMSE | triangular RMSE | triangular dead rows |",
        "|---|---|---|---|---|",
    ]
    for name in ("bikeshare", "wec"):
        cfg = DATASETS[name]
        Xd, yd = cfg["loader"]()
        cells = [
            one_cell(
                Xd,
                yd,
                s,
                cfg.get("fis_kwargs", {}).get("top_n", -1),
                buckets=cfg["buckets"],
                order=cfg["order"],
            )
            for s in SEEDS
        ]
        n_f = int(np.mean([c["n_features"] for c in cells]))
        lines.append(
            f"| {name} | {n_f} | {np.mean([c['gauss_rmse'] for c in cells]):.2f} | "
            f"{np.mean([c['tri_rmse'] for c in cells]):.2f} | "
            f"{100 * np.mean([c['tri_dead'] for c in cells]):.1f}% |"
        )
        print(lines[-1], flush=True)

    out = os.path.join(OUTPUTS, "triangularization.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {os.path.relpath(out, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
