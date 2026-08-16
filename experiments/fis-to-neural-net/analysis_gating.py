"""Does the FIS's gating choice still matter, once the seed is backed out?

Backing the seed out of the FIS's *response* rather than its internal gates was
supposed to make the t-norm/t-conorm choice structurally irrelevant: a product
t-norm makes firing strengths piecewise multilinear and kills any gate-level
ReLU conversion, but it does not stop us evaluating the FIS at a knot. That is
an argument, not a measurement. This script measures it.

For each De Morgan norm family `tribblefis` offers, it fits the same TRIBBLE
regressor, backs out the same analytic seed, and reports both the FIS's own
accuracy and the seed's fidelity to it. Two things could show up:

* the families differ in **FIS accuracy** -- a modelling fact about TRIBBLE,
  already known from `experiments/fis-acceleration` (which found `min/max` the
  worst of four and moved the default to `probability`);
* the families differ in **seed fidelity** -- which would mean the gating choice
  reaches the conversion after all, and the structural argument above is
  incomplete.

    python experiments/fis-to-neural-net/analysis_gating.py

Writes `outputs/gating.md`.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "reproduce", "tables"))

#: Every generated artifact goes here. Kept out of the source directory so the
#: scripts and the things they produce never have to be told apart by eye, and
#: so `outputs/.gitignore` can drop derived CSVs without a rule that could ever
#: match a hand-written file.
OUTPUTS = os.path.join(HERE, "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

import _fuzzy_models as fm  # noqa: E402
import fis2nn  # noqa: E402
from run_experiment import load_synth1d, prepare, split  # noqa: E402
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402

SEEDS = [int(s) for s in os.environ.get("FIS2NN_SEEDS", "0,1,2,3,4").split(",")]


def norm_families():
    """Whatever De Morgan pairs this build of `tribblefis` exposes."""
    from tribblefis import gauss_data

    for attr in ("NORM_FAMILIES", "NORM_PAIRS", "DE_MORGAN_FAMILIES"):
        found = getattr(gauss_data, attr, None)
        if isinstance(found, dict):
            return list(found)
        if isinstance(found, (list, tuple)):
            return list(found)
    return ["probability", "min-max", "lukasiewicz", "einstein"]


def one(X, y, seed, family, buckets, order, fis_kwargs):
    X_tr, y_tr, X_te, y_te = split(X, y, seed)
    Xtr, Xte, y_center, y_scale = prepare(X_tr, y_tr, X_te, y_te)
    with contextlib.redirect_stdout(io.StringIO()):
        reg = TribbleRegressor(
            n_output_buckets=buckets,
            tsk_order=order,
            random_state=seed,
            norm_conorm=family,
            **fis_kwargs,
        )
        reg.fit(Xtr, y_tr)
    feats = list(reg.top_features_)

    def fis_fn(frame):
        with contextlib.redirect_stdout(io.StringIO()):
            return np.asarray(reg.predict(frame), dtype=float).ravel()

    y_fis = fis_fn(Xte)
    knots = fis2nn.fis_knots(reg.model_, feats)
    net = fis2nn.analytic_seed_from_fis(
        lambda fr: (fis_fn(fr) - y_center) / y_scale,
        Xtr,
        feats,
        knots,
        background_size=256,
        seed=seed,
    )
    seed_pred = net.predict(Xte[feats].to_numpy(float)) * y_scale + y_center
    return {
        "fis_rmse": fis2nn.rmse(y_te, y_fis),
        "seed_rmse": fis2nn.rmse(y_te, seed_pred),
        "fidelity": fis2nn.rmse(y_fis, seed_pred) / (float(np.std(y_fis)) or 1.0),
        "n_hidden": net.n_hidden,
    }


def main() -> int:
    families = norm_families()
    cases = [
        ("synth1d", load_synth1d(), 6, "1st", {}),
        ("concrete", fm.load_concrete(), 4, "1st", {}),
    ]
    lines = [
        "# Does the gating choice reach the conversion? (measured)",
        "",
        f"Seeds: {SEEDS}. **seed fidelity** is the analytic seed's RMSE against the "
        "FIS it was backed out of, relative to that FIS output's own standard "
        "deviation -- 0 means the seed reproduces the FIS exactly.",
        "",
    ]
    for name, data, buckets, order, kw in cases:
        if data is None:
            continue
        X, y = data
        lines += [
            f"## {name}",
            "",
            "| norm family | FIS test RMSE | seed test RMSE | seed fidelity | hidden |",
            "|---|---|---|---|---|",
        ]
        for family in families:
            try:
                cells = [one(X, y, s, family, buckets, order, kw) for s in SEEDS]
            except Exception as exc:  # noqa: BLE001
                lines.append(
                    f"| {family} | unsupported ({type(exc).__name__}) | — | — | — |"
                )
                print(lines[-1], flush=True)
                continue
            lines.append(
                f"| {family} | {np.mean([c['fis_rmse'] for c in cells]):.3f} "
                f"± {np.std([c['fis_rmse'] for c in cells]):.3f} | "
                f"{np.mean([c['seed_rmse'] for c in cells]):.3f} | "
                f"{np.mean([c['fidelity'] for c in cells]):.3f} | "
                f"{np.mean([c['n_hidden'] for c in cells]):.0f} |"
            )
            print(f"[{name}] {lines[-1]}", flush=True)
        lines.append("")

    out = os.path.join(OUTPUTS, "gating.md")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\nwrote {os.path.relpath(out, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
