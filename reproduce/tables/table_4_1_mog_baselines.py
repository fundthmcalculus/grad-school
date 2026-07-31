"""Table 4.1 -- Mixture-of-Gaussians FIS training time and accuracy vs. baselines.

The point of this table is the *speed* claim: MoG trains a competitive model
without any GA/GD. Columns: MoG train time and accuracy/R2, then the baselines a
reviewer will ask for -- a scikit-learn tree/forest reference (always available),
and ANFIS / GA-tuned FIS (optional adapters; N/A unless their deps are present).
Every number is mean +/- std across `common.SEEDS`.

Run (from repo root):  uv run --project tribble-fis python reproduce/tables/table_4_1_mog_baselines.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C            # noqa: E402
import _fuzzy_models as _fm   # noqa: E402

# Optional baselines -- resolve once; None => the column renders as N/A.
(anfis_fit_predict,) = C.optional_import("_baseline_anfis", ["fit_predict"])
(gafis_fit_predict,) = C.optional_import("_baseline_gafis", ["fit_predict"])


def _bench(kind, X, y, mog_factory, score):
    """Return dict col -> list-of-per-seed (train_seconds, score)."""
    cols = {c: {"t": [], "s": []} for c in ["mog", "rf", "anfis", "gafis"]}
    for seed in C.SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)

        model = mog_factory(seed)
        if model is not None:
            with C.timed() as t:
                try:
                    p = np.asarray(model.fit(Xtr, ytr).predict(Xte))
                except Exception:  # noqa: BLE001
                    p = None
            if p is not None:
                cols["mog"]["t"].append(t.seconds)
                cols["mog"]["s"].append(score(yte, p))

        RF = RandomForestRegressor if kind == "reg" else RandomForestClassifier
        with C.timed() as t:
            p = RF(n_estimators=200, random_state=seed).fit(Xtr, ytr).predict(Xte)
        cols["rf"]["t"].append(t.seconds)
        cols["rf"]["s"].append(score(yte, p))

        for name, fn in (("anfis", anfis_fit_predict), ("gafis", gafis_fit_predict)):
            if fn is None:
                continue
            try:
                with C.timed() as t:
                    p = np.asarray(fn(Xtr, ytr, Xte, kind=kind, seed=seed))
                cols[name]["t"].append(t.seconds)
                cols[name]["s"].append(score(yte, p))
            except Exception as exc:  # noqa: BLE001
                print(f"  [{name}] failed ({exc.__class__.__name__}); -> N/A")
    return cols


def _row(label, metric_name, cols):
    order = ["mog", "anfis", "gafis", "rf"]
    return [label,
            C.cell(cols["mog"]["t"], fmt="{:.2f}") + " s" if cols["mog"]["t"] else C.NA,
            f"{metric_name}=" + C.cell(cols["mog"]["s"]) if cols["mog"]["s"] else C.NA,
            *[C.cell(cols[k]["s"]) if k != "mog" else "" for k in order if k != "mog"]]


def main():
    print("Table 4.1 -- MoG training time & accuracy vs. baselines")
    rows = []

    concrete = _fm.load_concrete()
    if concrete is not None:
        X, y = concrete
        cols = _bench("reg", X, y, _fm.mog_regressor, r2_score)
        rows.append(_row("Concrete (regression)", "R2", cols))
    else:
        rows.append(["Concrete (regression)", C.NA, C.NA, C.NA, C.NA, C.NA])

    phi = _fm.load_phiusiil()
    if phi is not None:
        X, y = phi
        cols = _bench("clf", X, y, _fm.mog_classifier, accuracy_score)
        rows.append(_row("PhiUSIIL (classification)", "acc", cols))
    else:
        rows.append(["PhiUSIIL (classification)", C.NA, C.NA, C.NA, C.NA, C.NA])

    header = ["Dataset (task)", "MoG train time", "MoG accuracy / R2",
              "ANFIS", "GA-tuned FIS", "tree / RF ref"]
    C.emit("table_4_1", "Table 4.1 -- Training time and accuracy", header, rows,
           note="MoG columns measured; ANFIS / GA-FIS fill in only if their adapters "
                "(reproduce/tables/_baseline_anfis.py, _baseline_gafis.py) are present. "
                "The RF reference is scikit-learn. Times are wall-clock training seconds.")


if __name__ == "__main__":
    main()
