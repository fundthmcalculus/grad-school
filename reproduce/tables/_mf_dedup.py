"""Shared dedup machinery and dataset loaders for Table 4.8.

`GaussianMixtureModel` in `gauss_data.py` already has dedup machinery
(`get_deduplicated_membership_fcns`, `to_simple_model`), but the merge tolerance
is a hardcoded module constant (`rtol=1e-2, atol=1e-3` in `_is_close`) with no
way to sweep it from the outside. Filed upstream as tribble-fis#85. Until that
lands, the functions below reimplement the same construction with the
tolerance exposed, so the sweep does not require hand-patching the pinned
submodule. `to_simple_model_tol` and `build_deduped_model` are intentionally
byte-for-byte equivalent to `GaussianMixtureModel.to_simple_model()` /
`get_deduplicated_membership_fcns()` at ``multiplier=1.0`` (the library
default) -- that equivalence is what lets this table claim it measures the
*library's own* dedup, not a reimplementation of it.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, load_digits, load_diabetes

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TABLES)
import _fuzzy_models as F  # noqa: E402

from tribblefis.gauss_data import (  # noqa: E402
    GaussianMixtureModel,
    FeatureModel,
    LabelModel,
    GaussianMembership,
    Rule,
    SimpleGaussianClassifierModel,
)
from tribblefis.gauss_math import simple_gaussian_predict  # noqa: E402
from tribblefis.gaussian_classifier import (  # noqa: E402
    TribbleClassifier,
    TribbleSequenceClassifier,
)
from tribblefis.gaussian_regressor import TribbleRegressor  # noqa: E402
from tribblefis.regression import predict_tsk  # noqa: E402

# Library default: rtol=1e-2, atol=1e-3 (gauss_data.py, GaussianMixtureModel._is_close).
# Every multiplier below scales both relative and absolute tolerance together,
# so "1x" reproduces the shipped behaviour exactly.
LIB_RTOL, LIB_ATOL = 1e-2, 1e-3

# Log-spaced sweep, dense enough to localise an inflection point without
# O(n^2)-pairwise dedup at each of an excessive number of points (a few
# hundred MFs per fit; even 14 points x 6 datasets x 10 seeds finishes in
# well under a minute -- see the module docstring in the caller for the
# measured wall-clock).
MULTIPLIERS = [
    0.1,
    0.3,
    1.0,
    2.0,
    3.0,
    5.0,
    7.0,
    10.0,
    15.0,
    20.0,
    30.0,
    50.0,
    70.0,
    100.0,
]


def _close(mf, other, rtol, atol):
    if type(mf) != type(other):
        return False
    if isinstance(mf, GaussianMembership):
        return bool(
            np.isclose(mf.mu, other.mu, rtol=rtol, atol=atol)
            and np.isclose(mf.sigma, other.sigma, rtol=rtol, atol=atol)
        )
    return False


def dedup_map(all_mfs, rtol, atol):
    """Replacement map {mf -> canonical_mf}, transitively resolved.

    O(n^2) pairwise -- matches the library's own `get_deduplicated_membership_
    fcns`. At the scales here (a few hundred MFs per fit) this is a few
    hundred thousand comparisons, well under the wall-clock this table's
    seed/dataset/tolerance product already spends on model fitting.
    """
    to_replace = {}
    for i, mf in enumerate(all_mfs):
        for other in all_mfs[i + 1 :]:
            if other in to_replace:
                continue
            if _close(mf, other, rtol, atol):
                to_replace[other] = mf
    for key in list(to_replace.keys()):
        cur = to_replace[key]
        seen = {key}
        while cur in to_replace and cur not in seen:
            seen.add(cur)
            cur = to_replace[cur]
        to_replace[key] = cur
    return to_replace


def to_simple_model_tol(
    model: GaussianMixtureModel, rtol, atol, anomaly_params=None
) -> SimpleGaussianClassifierModel:
    """`GaussianMixtureModel.to_simple_model()`, with the tolerance exposed."""
    dedup_mfs = dedup_map(model.all_membership_fcns, rtol, atol)
    rules = []
    for label in model.all_output_labels:
        antecedent_ids = {}
        for feature_name, feature_model in model.feature_models.items():
            label_model = feature_model.label_models.get(label, None)
            if label_model is None:
                continue
            antecedent_ids[feature_name] = [
                dedup_mfs.get(mf, mf).id for mf in label_model.memberships
            ]
        rules.append(Rule(antecedents=antecedent_ids, consequent=label))
    required_ids = {u for r in rules for lst in r.antecedents.values() for u in lst}
    input_mfs = [mf for mf in model.all_membership_fcns if mf.id in required_ids]
    return SimpleGaussianClassifierModel(
        input_mfs=input_mfs, rules=rules, anomaly_params=anomaly_params
    )


def build_deduped_model(
    model: GaussianMixtureModel, rtol, atol
) -> GaussianMixtureModel:
    """Same feature/label structure, memberships swapped for their dedup
    representative -- for regression, which has no SimpleGaussianClassifierModel
    equivalent (tribble-fis#85, follow-up 3)."""
    rep = dedup_map(model.all_membership_fcns, rtol, atol)
    new_feature_models = {}
    for fname, fmodel in model.feature_models.items():
        new_label_models = {}
        for label, lmodel in fmodel.label_models.items():
            new_label_models[label] = LabelModel(
                memberships=[rep.get(mf, mf) for mf in lmodel.memberships]
            )
        new_feature_models[fname] = FeatureModel(label_models=new_label_models)
    return GaussianMixtureModel(new_feature_models, anomaly_params=model.anomaly_params)


# --- dataset loaders ---------------------------------------------------------
# Classification: Glass (in-repo CSV), Wine / Breast Cancer / Digits (bundled
# with scikit-learn -- no network, no missing-file risk, which is why they were
# chosen over the dissertation's other named datasets: PhiUSIIL and RT-IOT2022
# are not in this repository (see Ch4 Table 4.4's own `not run` cells), and
# nothing here should silently substitute a different dataset for a named one.
# Regression: Concrete (in-repo CSV, via `_fuzzy_models.load_concrete`),
# Diabetes (bundled).
def load_glass():
    # Glass moved into data/ when the loaders were refactored; three call sites
    # kept looking for it at the repo root, where it has not been since. The
    # failure is silent in two of them (load_glass returns None and the rows are
    # dropped), which is why Table 4.8's Glass row and Table 4.9 -- checklist C4's
    # headline correction-pass measurement -- have been absent from every archive
    # since the move. Prefer data/, fall back to the old root path.
    path = os.path.join(F.DATA_DIR, "glass.csv")
    if not os.path.exists(path):
        path = os.path.join(F.REPO_ROOT, "glass.csv")
    if not os.path.exists(path):
        print("  [glass] not found in data/ or repo root; rows -> N/A")
        return None
    df = pd.read_csv(path).dropna()
    return df.drop(columns=["Type"]).astype(float), df["Type"].astype(int)


def load_wine_ds():
    d = load_wine()
    return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target)


def load_breast_cancer_ds():
    d = load_breast_cancer()
    return pd.DataFrame(d.data, columns=d.feature_names), pd.Series(d.target)


def load_digits_ds():
    d = load_digits()
    cols = [f"px{i}" for i in range(d.data.shape[1])]
    return pd.DataFrame(d.data, columns=cols), pd.Series(d.target)


def load_diabetes_ds():
    d = load_diabetes()
    y = pd.Series(d.target, name="y_value")
    return pd.DataFrame(d.data, columns=d.feature_names), y


CLASSIFICATION_DATASETS = [
    ("Glass", load_glass),
    ("Wine", load_wine_ds),
    ("BreastCancer", load_breast_cancer_ds),
    ("Digits", load_digits_ds),
]

REGRESSION_DATASETS = [
    ("Concrete", F.load_concrete),
    ("Diabetes", load_diabetes_ds),
]
