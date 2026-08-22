"""Norm/conorm comparison -- does the choice of fuzzy operator matter?

Every model here combines memberships with two connectives: a t-norm for the rule
AND, and a t-conorm for the per-feature OR. Until `tribble-fis#32` the choice was
a default nobody had swept, and for regression it could not be swept at all --
`tsk_firing_strengths` read the operator off the anomaly parameters, which
regression never supplies, so every regressor silently ran at "min/max".

This sweeps the five De Morgan pairs across both datasets and every model that
accepts the selection, so "we used min/max" stops being an unexamined default.

WHY ONLY THE DIAGONAL. Each family's t-norm and t-conorm are De Morgan duals
under N(x) = 1 - x. Mixing families is possible (`allow_mixed_norms=True`) but is
deliberately not the default, and is not swept here: the interesting case for a
mismatched pair is the anomaly rule, whose complement construction assumes
duality, and that experiment needs the open-set harness rather than this one.

WHAT EACH COLUMN MEANS PER MODEL -- they are not all the same thing:
  flat MoG      both operators, from the named family
  fuzzy tree    t-norm ONLY. Path weights are a pure AND; a tree has no OR to
                apply a conorm to, so `t_conorm` would have nothing to act on.
  HME           the EXPERTS' operators. The gate is deliberately excluded: its
                responsibilities are a product of partition-of-unity weights, and
                that product is what keeps leaf responsibilities summing to 1 --
                the property that makes the model a mixture of experts at all.
                A general t-norm would break the normalisation, so the gate is
                fixed by the model's semantics rather than being a free axis.

Run (from repo root):
    uv run --project tribble-fis python reproduce/tables/table_norm_conorm_matrix.py

Knobs:
    REPRO_SEEDS="0,1,2,3,4"
    REPRO_NORM_FAMILIES="min/max,probability,luk,hamacher,einstein"
    REPRO_PHIUSIIL_N="20000"   sample cap for the classification rows
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import _fuzzy_models as _fm  # noqa: E402

FAMILIES = [
    f.strip()
    for f in os.environ.get(
        "REPRO_NORM_FAMILIES", "min/max,probability,luk,hamacher,einstein"
    ).split(",")
]
PHIUSIIL_N = int(os.environ.get("REPRO_PHIUSIIL_N", "20000"))


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


# --- model builders, one per (task, model), parameterised by family -----------
# Each returns an estimator or None. `None` becomes N/A rather than aborting the
# table, and the reason is printed once so a blank cell is never mysterious.


def mog_regressor(family, seed):
    # Renamed upstream: MixtureOfGaussiansFuzzyRegressor -> TribbleRegressor.
    # This file was missed by the B12(a) sweep that fixed the same rename in
    # table_a1_feature_scoring and table_4_8_mf_dedup, so BOTH flat-MoG rows of
    # this table have been silently N/A ever since -- the skip path works, which
    # is why nothing failed and nobody noticed. Try the new name first and keep
    # the old one as a fallback so the generator still runs against an older pin.
    import tribblefis.gaussian_regressor as gr

    cls = getattr(gr, "TribbleRegressor", None) or gr.MixtureOfGaussiansFuzzyRegressor
    return cls(
        n_output_buckets=3,
        tsk_order="1st",
        top_n=-1,
        norm_conorm=family,
        random_state=seed,
    )


def mog_classifier(family, seed):
    # Same rename as above: MixtureOfGaussiansFuzzyClassifier -> TribbleClassifier.
    import tribblefis.gaussian_classifier as gc

    cls = getattr(gc, "TribbleClassifier", None) or gc.MixtureOfGaussiansFuzzyClassifier
    return cls(top_n=5, norm_conorm=family, random_state=seed)


def tree_regressor(family, seed):
    import fuzzytree

    cls = getattr(fuzzytree, "FuzzyRegressionTree", None)
    if cls is None:
        return None
    return cls(
        tsk_order="1st",
        criterion="variance",
        max_depth=3,
        n_terms=2,
        top_n=4,
        min_soft_count=20,
        t_norm=family,
        random_state=seed,
    )


def tree_classifier(family, seed):
    import fuzzytree

    cls = getattr(fuzzytree, "FuzzyClassificationTree", None) or getattr(
        fuzzytree, "FuzzyTreeClassifier", None
    )
    if cls is None:
        return None
    return cls(max_depth=3, n_terms=2, top_n=4, t_norm=family, random_state=seed)


def hme_regressor(family, seed):
    """HME with the family applied to its EXPERTS. The gate stays a product."""
    import fuzzytree

    cls = getattr(fuzzytree, "HierarchicalFuzzyExpertsRegressor", None)
    if cls is None:
        return None
    return cls(
        criterion="variance",
        max_depth=2,
        n_gate_terms=2,
        top_n=4,
        min_soft_count=40,
        min_expert_samples=60,
        random_state=seed,
        expert_kwargs={
            "n_output_buckets": 3,
            "tsk_order": "1st",
            "norm_conorm": family,
        },
    )


def hme_classifier(family, seed):
    import fuzzytree

    cls = getattr(fuzzytree, "HierarchicalFuzzyExpertsClassifier", None)
    if cls is None:
        return None
    return cls(
        max_depth=2,
        n_gate_terms=2,
        top_n=4,
        random_state=seed,
        expert_kwargs={"norm_conorm": family},
    )


REGRESSION_MODELS = [
    ("flat MoG-TSK", mog_regressor),
    ("fuzzy tree (t-norm only)", tree_regressor),
    ("HME (experts only)", hme_regressor),
]
CLASSIFICATION_MODELS = [
    ("flat MoG", mog_classifier),
    ("fuzzy tree (t-norm only)", tree_classifier),
    ("HME (experts only)", hme_classifier),
]


def sweep(X, y, models, metrics, task):
    """-> {(model_name, metric_name): {family: [per-seed values]}}"""
    out = {(m, k): {f: [] for f in FAMILIES} for m, _ in models for k, _ in metrics}
    complained = set()
    for family in FAMILIES:
        for seed in C.SEEDS:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )
            for model_name, build in models:
                try:
                    est = build(family, seed)
                    if est is None:
                        raise RuntimeError("estimator class not found")
                    pred = np.asarray(est.fit(Xtr, ytr).predict(Xte))
                except Exception as exc:  # noqa: BLE001 - report once, cell -> N/A
                    key = (model_name, family)
                    if key not in complained:
                        complained.add(key)
                        print(
                            f"  [skip] {task} / {model_name} / {family}: "
                            f"{exc.__class__.__name__}: {exc}"
                        )
                    continue
                for metric_name, fn in metrics:
                    try:
                        out[(model_name, metric_name)][family].append(
                            float(fn(yte, pred))
                        )
                    except Exception:  # noqa: BLE001
                        pass
        print(f"  {task}: {family} done")
    return out


def rows_from(sweep_out, dataset, models, metrics):
    """One row per (model, metric); one column per family, plus the best family."""
    rows = []
    for model_name, _ in models:
        for metric_name, _ in metrics:
            per_family = sweep_out[(model_name, metric_name)]
            cells = [C.cell(per_family[f]) for f in FAMILIES]
            # "Best" is only meaningful where at least two families produced a
            # number; a single populated column is not a comparison.
            means = {f: C.agg(per_family[f])[0] for f in FAMILIES}
            means = {f: m for f, m in means.items() if m is not None}
            if len(means) < 2:
                best = C.NA
            else:
                lower_is_better = "RMSE" in metric_name
                pick = (
                    min(means, key=means.get)
                    if lower_is_better
                    else max(means, key=means.get)
                )
                spread = max(means.values()) - min(means.values())
                best = f"**{pick}** (spread {spread:.3f})"
            rows.append([dataset, model_name, metric_name, *cells, best])
    return rows


def main():
    print("Norm/conorm matrix -- De Morgan diagonal")
    print(f"  families: {FAMILIES}")
    print(f"  seeds:    {C.SEEDS}")
    rows = []

    concrete = _fm.load_concrete()
    if concrete is None:
        print("  [concrete] unavailable; regression rows -> N/A")
        for model_name, _ in REGRESSION_MODELS:
            for metric_name in ("R2", "RMSE (MPa)"):
                rows.append(
                    [
                        "Concrete",
                        model_name,
                        metric_name,
                        *([C.NA] * len(FAMILIES)),
                        C.NA,
                    ]
                )
    else:
        X, y = concrete
        metrics = [("R2", r2_score), ("RMSE (MPa)", _rmse)]
        rows += rows_from(
            sweep(X, y, REGRESSION_MODELS, metrics, "Concrete"),
            "Concrete",
            REGRESSION_MODELS,
            metrics,
        )

    phiusiil = _fm.load_phiusiil(sample_size=PHIUSIIL_N)
    if phiusiil is None:
        print("  [phiusiil] unavailable; classification rows -> N/A")
        for model_name, _ in CLASSIFICATION_MODELS:
            rows.append(
                ["PhiUSIIL", model_name, "accuracy", *([C.NA] * len(FAMILIES)), C.NA]
            )
    else:
        X, y = phiusiil
        metrics = [("accuracy", accuracy_score)]
        rows += rows_from(
            sweep(X, y, CLASSIFICATION_MODELS, metrics, "PhiUSIIL"),
            "PhiUSIIL",
            CLASSIFICATION_MODELS,
            metrics,
        )

    header = ["Dataset", "Model", "Metric", *FAMILIES, "Best (mean spread)"]
    C.emit(
        "table_norm_conorm_matrix",
        "Norm/conorm comparison — the five De Morgan pairs",
        header,
        rows,
        note=(
            "Each column names a FAMILY whose t-norm and t-conorm are De Morgan "
            "duals under N(x)=1-x; mixed pairs are an opt-in advanced setting and "
            "are not swept here. The columns do not mean the same thing for every "
            "model: the flat MoG uses both operators; the fuzzy tree uses the "
            "t-norm only, since path weights are a pure AND with no OR to apply a "
            "conorm to; and the HME row varies its EXPERTS only — its gate is a "
            "product of partition-of-unity weights by construction, which is what "
            "keeps leaf responsibilities summing to 1, so it is not a free axis. "
            "'Best' reports the winning family and the spread between the best and "
            "worst mean, which is the number that says whether this axis is worth "
            "tuning at all. Regression could not be swept before tribble-fis#32."
        ),
    )


if __name__ == "__main__":
    main()
