"""Dataset loaders and model adapters shared by Tables 4.1 and 6.1.

Runs under the tribble-fis environment (``uv run`` inside ``tribble-fis``), which
is where ``tribblefis`` and the ``fuzzytree`` package are importable. Everything
is wrapped so an unavailable model or dataset yields ``None`` (rendered as N/A)
instead of aborting the whole table.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- locate repo root and put the fuzzytree package on the path --------------
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))  # .../grad-school
FIS = os.path.join(REPO_ROOT, "tribble-fis")
sys.path.insert(0, os.path.join(FIS, "tribble-tree"))  # for `import fuzzytree`
sys.path.insert(0, os.path.dirname(_HERE))  # reproduce/ -> `import common`
import common as C  # noqa: E402

# The dataset loaders and DATA_DIR now live in the top-level `repro_data`
# package, so every experiment shares one definition instead of each carrying
# its own copy. They are re-exported here UNCHANGED, so existing
# `import _fuzzy_models` callers keep working and no reported number moves. The
# preprocessing (dropped columns, errata, leaky features) lives in
# repro_data/loaders.py, documented against reproduce/dataset_specs.yaml.
sys.path.insert(0, REPO_ROOT)  # repo root -> `import repro_data`
from repro_data import (  # noqa: E402,F401 -- re-exported for existing callers
    CONCRETE_COLS,
    DATA_DIR,
    load_beth,
    load_bikeshare,
    load_bodyfat,
    load_concrete,
    load_glass,
    load_phiusiil,
    load_rt_iot2022,
    load_shuttle,
    load_wec,
)


def _first_attr(mod, *names):
    for n in names:
        obj = getattr(mod, n, None)
        if obj is not None:
            return obj
    return None


# --- model factories (return a fitted-predict callable, or None) -------------
def _try(build):
    """Return the constructed estimator, or None if construction fails."""
    try:
        return build()
    except Exception as exc:  # noqa: BLE001
        print(f"  [model] construction failed ({exc.__class__.__name__}); -> N/A")
        return None


# --- normalization ------------------------------------------------------------
# `tribble-fis` PR #67 (a385a1a) DELETED `gauss_math.standard_transform` and
# `gauss_math.detect_and_apply_log_transform` and replaced them with two sklearn
# transformers in `tribblefis.scaling`:
#
#   UnitScalar     min-max to [0, 1], log1p-ing wide-dynamic-range features first
#   StandardScalar z-score (mu=0, sigma=1), same log-transform step
#
# `standard_transform` applied MIN-MAX despite its name, so `UnitScalar` -- not
# `StandardScalar` -- is its behaviour-preserving successor. The two wrappers
# below exist because the deleted helpers had three call shapes the sklearn
# surface does not: Series-in/Series-out, a column *subset*, and a DataFrame out
# (the tribblefis transforms index features by name, so a bare ndarray raises).
# They are wrappers, not reimplementations: the arithmetic is `tribblefis.scaling`'s.
#
# Two details are load-bearing for reproducing the archived numbers:
#   * `log_dynamic_range=2` is passed explicitly. The old calls passed
#     `min_dynamic_range=2`; `UnitScalar`'s default is 3.0, which on Concrete
#     would drop `Slag` from the logged set and change every cell.
#   * the scaler is FIT ON THE FULL FRAME, before the train/test split, exactly
#     as the deleted helpers were called. This is transductive and would be wrong
#     in a deployment pipeline, but reproducing the archive means reproducing it.
SCALERS = ("unit", "standard")


def _scaler(kind, log_dynamic_range):
    from tribblefis.scaling import StandardScalar, UnitScalar

    if kind == "unit":
        return UnitScalar(log_dynamic_range=log_dynamic_range)
    if kind == "standard":
        return StandardScalar(log_dynamic_range=log_dynamic_range)
    raise ValueError(f"scaler must be one of {SCALERS}, got {kind!r}")


def unit_scale(X, column=None):
    """Min-max to [0, 1] with no log step: the exact behavior of the deleted
    `gauss_math.standard_transform`, verified bit-for-bit against it.

    Accepts a Series (returns a Series), or a DataFrame with an optional
    `column` subset (returns a DataFrame, untouched columns passed through).
    """
    if isinstance(X, pd.Series):
        scaled = _scaler("unit", None).fit_transform(X.to_frame()).ravel()
        return pd.Series(scaled, index=X.index, name=X.name)

    cols = (
        list(X.columns)
        if column is None
        else ([column] if isinstance(column, str) else list(column))
    )
    out = X.copy()
    out[cols] = _scaler("unit", None).fit_transform(X[cols].copy())
    return out


def fit_scaler(X, scaler="unit", log_dynamic_range=2):
    """Fit a feature scaler and return it, for callers that must not leak the test fold.

    `normalize()` below calls `fit_transform` on whatever frame it is handed, and every
    generator hands it the full dataset before splitting. That is deliberate and
    documented above -- reproducing the archive means reproducing it -- but it means
    there is no way to fit on a training fold with these helpers, so measuring what the
    transduction is worth was not possible without a second copy of the treatment.
    Handing back the fitted scaler makes the leak-free variant a two-line change at the
    call site instead of a duplicate implementation that can drift.

    Returns (fitted_scaler, names_of_logged_columns).
    """
    sc = _scaler(scaler, log_dynamic_range)
    sc.fit(X.copy())
    return sc, list(sc.log_features_)


def apply_scaler(sc, X):
    """Transform with an already-fitted scaler, preserving index and column names."""
    return pd.DataFrame(sc.transform(X.copy()), index=X.index, columns=X.columns)


def unit_scale_with(lo, hi, y):
    """Min-max a target using bounds supplied from elsewhere -- normally a train fold.

    `unit_scale()` derives its bounds from its argument, so scaling a target before
    splitting puts the test fold's min and max into the transform. R^2 is invariant
    under an affine map of the target, so on its own that biases nothing; it matters
    because the *bucket* boundaries and bucket means computed downstream from the scaled
    target are not affine, and those do reach the prediction path.
    """
    span = float(hi) - float(lo)
    if span == 0:
        return pd.Series(np.zeros(len(y)), index=y.index, name=y.name)
    return (y - float(lo)) / span


def normalize(X, scaler="unit"):
    """`concrete.py`'s feature treatment: auto log-transform, then scale.

    Shared rather than duplicated, because a second copy of this that drifted
    would silently make two tables incomparable -- which is the exact failure
    this harness exists to catch.

    `scaler="unit"` (the default) is what every archived "log+std" number in
    this repository actually measured: log + min-max to [0, 1]. `"standard"` is
    genuine z-score, which had never been measured before Table 4.1's third arm.

    Returns (transformed_X, names_of_logged_columns).
    """
    sc = _scaler(scaler, 2)
    Xt = pd.DataFrame(sc.fit_transform(X.copy()), index=X.index, columns=X.columns)
    return Xt, list(sc.log_features_)


def mog_regressor(seed, tsk_order="1st"):
    """The flat MoG-TSK regressor at one consequent order.

    `tsk_order` is exposed because Table 4.5 quotes a full-second-order R² whose
    training time was never measured -- the accuracy came from
    `table_concrete_reconciliation.py`, which sweeps orders but does not time
    them, so the cell sat as `*pending*` rather than borrow the 1st-order row's
    seconds. Same object, same preprocessing, one keyword: the alternative was a
    second copy of this constructor, which is how two tables drift apart.
    """
    from tribblefis.gaussian_regressor import TribbleRegressor

    return _try(
        lambda: TribbleRegressor(
            n_output_buckets=3, tsk_order=tsk_order, top_n=-1, random_state=seed
        )
    )


def mog_classifier(seed):
    from tribblefis.gaussian_classifier import TribbleClassifier

    return _try(lambda: TribbleClassifier(top_n=5, random_state=seed))


# --- Ruspini derivation --------------------------------------------------------
# Both helpers convert an *already-fitted* MoG model into its explicit Ruspini
# form (tribblefis.ruspini.ruspinize_model) rather than fitting a second,
# independent model -- `RuspiniFuzzyClassifier` would otherwise redo the
# TribbleClassifier fit `mog_classifier` already did. `cluster_joint_terms`
# defaults on: restricting each rule to actually-observed joint term
# combinations, instead of the marginal Cartesian product, is the whole point
# of exercising this path in the quick scripts.


def ruspinize_classifier(model, X, y, cluster_joint_terms=True, min_cluster_frac=0.05):
    """Derive a RuspiniPartitionModel from a fitted TribbleClassifier."""
    from tribblefis.ruspini import ruspinize_model

    return ruspinize_model(
        model.model_,
        X,
        y,
        cluster_joint_terms=cluster_joint_terms,
        min_cluster_frac=min_cluster_frac,
    )


def ruspinize_regressor(model, X, y, cluster_joint_terms=True, min_cluster_frac=0.05):
    """Derive a RuspiniPartitionModel from a fitted TribbleRegressor's output-
    bucket mixture.

    Regression has no rule-per-class notion, but `TribbleRegressor.model_` is
    built the same way as the classifier's -- one Gaussian mixture per output
    *bucket*, standing in for the class label -- so the same ruspinize_model
    entry point applies directly. Returns (RuspiniPartitionModel, bucket_mean)
    where `bucket_mean[b]` is the value bucket `b`'s rule stands for; see
    `ruspini_predict_regression` for how that's read back into a prediction.

    Deliberately recomputes `bucket_mean` from `partition_output` here rather
    than reading `model.y_bucket_mean_`: the model's own attribute has by then
    been overwritten by `solve_tsk_consequents` with intercepts solved jointly
    against *its* firing strengths (from the original Gaussian mixture) -- not
    the Ruspini partition's triangular firing, which can differ a lot,
    especially once `cluster_joint_terms` changes the rule count. The raw
    per-bucket target mean is the value that actually pairs with this
    (unrelated) firing computation.
    """
    from tribblefis.ruspini import ruspinize_model
    from tribblefis.regression import partition_output

    y_series = pd.Series(np.asarray(y).flatten(), name="y_value")
    y_partitioned, bucket_mean = partition_output(
        model.n_output_buckets, y_series, method=model.output_partition
    )
    rm = ruspinize_model(
        model.model_,
        X,
        y_partitioned["y_bucket"],
        cluster_joint_terms=cluster_joint_terms,
        min_cluster_frac=min_cluster_frac,
    )
    return rm, bucket_mean


def ruspini_predict_regression(rm, bucket_mean, X):
    """Defuzzify a bucket-classification RuspiniPartitionModel into a
    continuous prediction: each bucket rule's normalised firing weights that
    bucket's mean value (order-0 TSK; no linear consequent correction)."""
    proba, labels = rm.class_proba(X)
    values = np.array([bucket_mean[int(lab)] for lab in labels])
    return proba @ values


# --- Ruspini refinement ---------------------------------------------------------
# `refine_ruspini_partition` moves the partition's apex knots against a
# cross-entropy objective (see tribblefis.refine); `method="coordinate"` -- its
# default -- is the cheap one-knot-at-a-time L-BFGS search, as opposed to
# `method="optimizers"`'s population search. That's the "basic" refinement
# these quick scripts want: a fast pass to report alongside the unrefined
# Ruspini numbers, not a tuned search.


def refine_classifier(rm, X, y, **kwargs):
    """Refine a classifier's Ruspini partition. Returns (refined_rm, info)."""
    from tribblefis.refine import refine_ruspini_partition

    kwargs.setdefault("method", "coordinate")
    kwargs.setdefault("seed", 42)
    kwargs.setdefault("verbose", False)
    return refine_ruspini_partition(rm, X, y, **kwargs)


def refine_regressor(model, rm, X, y, **kwargs):
    """Refine a regressor's Ruspini partition.

    `model` supplies the output-bucket partition scheme
    (n_output_buckets/output_partition) so `y` gets bucketed exactly the way
    `ruspinize_regressor` bucketed it -- refine_ruspini_partition scores
    against discrete labels, and those labels have to be the same ones the
    partition's rules were built against. Returns (refined_rm, info).
    """
    from tribblefis.refine import refine_ruspini_partition
    from tribblefis.regression import partition_output

    y_series = pd.Series(np.asarray(y).flatten(), name="y_value")
    y_partitioned, _ = partition_output(
        model.n_output_buckets, y_series, method=model.output_partition
    )
    kwargs.setdefault("method", "coordinate")
    kwargs.setdefault("seed", 42)
    kwargs.setdefault("verbose", False)
    return refine_ruspini_partition(rm, X, y_partitioned["y_bucket"], **kwargs)


def plot_membership_functions(rm, X, basename, max_features=6):
    """Save a simple per-feature plot of a RuspiniPartitionModel's triangular
    membership functions, for (up to `max_features` of) its selected inputs.

    Returns the path(s) written (see `common.save_figure`).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = rm.feature_order[:max_features]
    terms = rm.feature_terms()
    fig, axes = plt.subplots(
        len(features), 1, figsize=(6, 2.0 * len(features)), squeeze=False
    )
    for ax, f in zip(axes[:, 0], features):
        col = X[f].to_numpy(dtype=float) if f in X.columns else np.asarray(rm.apexes[f])
        lo, hi = float(np.min(col)), float(np.max(col))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        xs = np.linspace(lo - pad, hi + pad, 400)
        for i, t in enumerate(terms[f]):
            ax.plot(xs, t.evaluate(xs), label=f"term {i}")
        ax.set_title(f, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6, ncol=min(len(terms[f]), 4), loc="lower right")
    fig.tight_layout()
    written = C.save_figure(fig, basename, formats=("png",))
    plt.close(fig)
    return written


def tree_regressor(seed):
    import fuzzytree

    cls = _first_attr(fuzzytree, "FuzzyRegressionTree")
    return _try(lambda: cls(random_state=seed)) if cls else None


def tree_classifier(seed):
    import fuzzytree

    cls = _first_attr(fuzzytree, "FuzzyClassificationTree", "FuzzyTreeClassifier")
    return _try(lambda: cls(random_state=seed)) if cls else None


def hme_regressor(seed):
    import fuzzytree

    cls = _first_attr(fuzzytree, "HierarchicalFuzzyExpertsRegressor")
    return _try(lambda: cls(random_state=seed)) if cls else None


def hme_classifier(seed):
    import fuzzytree

    cls = _first_attr(fuzzytree, "HierarchicalFuzzyExpertsClassifier")
    return _try(lambda: cls(random_state=seed)) if cls else None


def fit_predict(model, X_tr, y_tr, X_te):
    """Fit and predict; return predictions or None on any failure."""
    if model is None:
        return None
    try:
        return np.asarray(model.fit(X_tr, y_tr).predict(X_te))
    except Exception as exc:  # noqa: BLE001
        print(f"  [model] fit/predict failed ({exc.__class__.__name__}); -> N/A")
        return None
