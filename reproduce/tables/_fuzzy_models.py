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

# Datasets live HERE, never in the submodule. tribble-fis used to carry
# `gaussian_mixture/` with the benchmark data in it; upstream removed that
# directory in 8484fd6 to make the library pure, and the experiments moved to
# this repository. Caching a fetched dataset back into the submodule would
# recreate a directory upstream deliberately deleted and leave a pinned
# submodule permanently dirty -- the same failure this harness exists to prevent.
# Override with GRAD_SCHOOL_DATA.
DATA_DIR = os.environ.get("GRAD_SCHOOL_DATA", os.path.join(REPO_ROOT, "data"))


def _first_attr(mod, *names):
    for n in names:
        obj = getattr(mod, n, None)
        if obj is not None:
            return obj
    return None


# --- datasets ----------------------------------------------------------------
CONCRETE_COLS = [
    "Cement",
    "Slag",
    "FlyAsh",
    "Water",
    "Superplasticizer",
    "CoarseAgg",
    "FineAgg",
    "Age",
    "Strength",
]


def load_concrete():
    """UCI Concrete: 8 mixture/age features -> compressive strength (MPa).

    Loads from ``data/Concrete_Data.csv`` or local spreadsheet fallback.
    Returns (X, y) or None.
    """
    csv_path = os.path.join(DATA_DIR, "Concrete_Data.csv")

    if not os.path.exists(csv_path):
        df = None
        try:  # try local spreadsheet

            xls = os.path.join(
                REPO_ROOT, "AEEM6097", "project-data", "Concrete_Data.xls"
            )
            if os.path.exists(xls):
                try:
                    df = pd.read_excel(xls)
                    df.columns = CONCRETE_COLS[: len(df.columns)]
                    print("  [concrete] read from the local .xls")
                except ImportError:
                    print(
                        "  [concrete] the local file is a legacy .xls and needs `xlrd`; "
                        "either `pip install xlrd` or let the UCI fetch handle it"
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [concrete] .xls unreadable ({exc.__class__.__name__})")

        if df is None:
            return None
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  [concrete] cached -> {os.path.relpath(csv_path, REPO_ROOT)}")

    df = pd.read_csv(csv_path).dropna()
    df.columns = [c.strip() for c in df.columns]
    # y is returned as a *named Series*, matching gaussian_mixture/concrete.py's
    # loader: the tribblefis transforms index it by column name, so a bare ndarray
    # raises deep inside the scaling/partition path.
    y = df["Strength"].astype(float)
    y.name = "y_value"
    X = df.drop(columns=["Strength"]).select_dtypes(include=[np.number]).astype(float)
    return X, y


def load_phiusiil(sample_size=20000):
    """PhiUSIIL phishing. Loads from repo loader or data/PhiUSIIL_Phishing_URL_Dataset.csv.

    Returns (X, y) or None if unavailable.
    """
    local = os.path.join(DATA_DIR, "PhiUSIIL_Phishing_URL_Dataset.csv")
    try:
        sys.path.insert(0, os.path.join(FIS, "tribble-tree"))
        import demo_phishing  # noqa: E402  -- repo loader, exact same features

        if os.path.exists(local):
            demo_phishing.DATA_PATH = local
        X, y = demo_phishing.load_data(sample_size=sample_size, random_state=42)
        print(
            f"  [phiusiil] repo loader, data from {os.path.relpath(local, REPO_ROOT)}"
            if os.path.exists(local)
            else "  [phiusiil] repo loader, bundled path"
        )
        return X, np.asarray(y)
    except Exception as exc:  # noqa: BLE001
        print(f"  [phiusiil] unavailable ({exc.__class__.__name__}); column -> N/A")
        return None


def load_rt_iot2022(sample_size=None):
    """RT-IOT2022: 123k rows × 83 features, 12 classes (open-set detection).

    Returns (X, y) or None if file not found.
    """
    local = os.path.join(DATA_DIR, "RT_IOT2022.csv")
    try:
        df = pd.read_csv(local)
        y = df.iloc[:, -1]  # last column is the target
        X = df.iloc[:, :-1]
        X = X.select_dtypes(include=[np.number]).astype(float)
        y = np.asarray(y)
        if sample_size and len(X) > sample_size:
            idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
            X, y = X.iloc[idx], y[idx]
        print(
            f"  [rt-iot2022] loaded {os.path.relpath(local, REPO_ROOT)}: "
            f"{len(X)} rows × {X.shape[1]} features"
        )
        return X, y
    except FileNotFoundError:
        print(f"  [rt-iot2022] file not found at {os.path.relpath(local, REPO_ROOT)}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  [rt-iot2022] failed to load ({exc.__class__.__name__}); column -> N/A")
        return None


def load_beth(combine=True):
    """BETH host telemetry: binary classification. Reads from data/beth/*.csv.

    If combine=True (default), concatenates all CSV files in the directory.
    Returns (X, y) or None if directory not found.
    """
    beth_dir = os.path.join(DATA_DIR, "beth")
    try:
        if not os.path.isdir(beth_dir):
            print(f"  [beth] directory not found at {os.path.relpath(beth_dir, REPO_ROOT)}")
            return None

        csv_files = sorted([f for f in os.listdir(beth_dir) if f.endswith(".csv")])
        if not csv_files:
            print(f"  [beth] no CSV files found in {os.path.relpath(beth_dir, REPO_ROOT)}")
            return None

        dfs = []
        for csv_file in csv_files:
            path = os.path.join(beth_dir, csv_file)
            dfs.append(pd.read_csv(path))

        df = pd.concat(dfs, ignore_index=True) if combine else dfs[0]
        y = df.iloc[:, -1]  # last column is the target
        X = df.iloc[:, :-1]
        X = X.select_dtypes(include=[np.number]).astype(float)
        y = np.asarray(y)
        print(
            f"  [beth] loaded {len(csv_files)} file(s) from "
            f"{os.path.relpath(beth_dir, REPO_ROOT)}: {len(X)} rows × {X.shape[1]} features"
        )
        return X, y
    except Exception as exc:  # noqa: BLE001
        print(f"  [beth] failed to load ({exc.__class__.__name__}); column -> N/A")
        return None


def load_shuttle(sample_size=None):
    """Shuttle: 58k rows × 7 features, 7 classes (structure discovery flagship).

    Loads from data/shuttle.csv. Returns (X, y) or None if file not found.
    """
    local = os.path.join(DATA_DIR, "shuttle.csv")

    if os.path.exists(local):
        try:
            df = pd.read_csv(local)
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
            X = X.select_dtypes(include=[np.number]).astype(float)
            y = np.asarray(y)
            if sample_size and len(X) > sample_size:
                idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
                X, y = X.iloc[idx], y[idx]
            print(
                f"  [shuttle] loaded {os.path.relpath(local, REPO_ROOT)}: "
                f"{len(X)} rows × {X.shape[1]} features"
            )
            return X, y
        except Exception as exc:  # noqa: BLE001
            print(f"  [shuttle] failed to load ({exc.__class__.__name__}); column -> N/A")
            return None
    else:
        print(f"  [shuttle] file not found at {os.path.relpath(local, REPO_ROOT)}")
        return None


def load_bikeshare(target_col="cnt", sample_size=None):
    """Bike Sharing Demand: 17.4k rows × 16 features, regression (demand prediction).

    Kaggle dataset: https://www.kaggle.com/datasets/c1730b3c7d4311e6a6202040f0db4ec7b826f619
    File: bikeshare-hour.csv (extracted from the Kaggle zip)

    Args:
        target_col: column name for the target (default 'cnt' = count of bikes rented).
        sample_size: if set, randomly sample to this size (for quick tests).

    Returns (X, y) or None if file not found.
    """
    local = os.path.join(DATA_DIR, "bikeshare-hour.csv")
    try:
        df = pd.read_csv(local)
        if target_col not in df.columns:
            print(
                f"  [bikeshare] target column '{target_col}' not found; "
                f"available: {list(df.columns)}"
            )
            return None

        y = df[target_col].astype(float)
        y.name = "y_value"
        # Drop non-numeric and index columns (dteday, instant, etc.)
        X = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors="ignore")
        # Drop obvious ID/index columns if present
        X = X.drop(columns=["instant"], errors="ignore").astype(float)

        if sample_size and len(X) > sample_size:
            idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)

        print(
            f"  [bikeshare] loaded {os.path.relpath(local, REPO_ROOT)}: "
            f"{len(X)} rows × {X.shape[1]} features"
        )
        return X, y
    except FileNotFoundError:
        print(f"  [bikeshare] file not found at {os.path.relpath(local, REPO_ROOT)}")
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"  [bikeshare] failed to load ({exc.__class__.__name__}); column -> N/A")
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
    """Min-max to [0, 1] with no log step: the exact behaviour of the deleted
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
