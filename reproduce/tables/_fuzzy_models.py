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
REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))          # .../grad-school
FIS = os.path.join(REPO_ROOT, "tribble-fis")
sys.path.insert(0, os.path.join(FIS, "tribble-tree"))       # for `import fuzzytree`


def _first_attr(mod, *names):
    for n in names:
        obj = getattr(mod, n, None)
        if obj is not None:
            return obj
    return None


# --- datasets ----------------------------------------------------------------
CONCRETE_COLS = ["Cement", "Slag", "FlyAsh", "Water", "Superplasticizer",
                 "CoarseAgg", "FineAgg", "Age", "Strength"]


def load_concrete():
    """UCI Concrete: 8 mixture/age features -> compressive strength (MPa).

    Resolution order, so this works on a fresh clone without manual setup:
      1. the repo CSV, if a previous run already cached it;
      2. UCI via ``ucimlrepo`` (id 165) -- then cached as that CSV;
      3. the legacy ``.xls`` in AEEM6097 (needs ``xlrd``, often absent).
    Returns (X, y) or None, printing which route it took.
    """
    csv_path = os.path.join(FIS, "gaussian_mixture", "Concrete_Data.csv")

    if not os.path.exists(csv_path):
        df = None
        try:                                    # 2. authoritative source
            from ucimlrepo import fetch_ucirepo
            ds = fetch_ucirepo(id=165)
            df = ds.data.features.copy()
            df["Strength"] = np.asarray(ds.data.targets).ravel()
            df.columns = CONCRETE_COLS[: len(df.columns)]
            print("  [concrete] fetched from UCI (id 165)")
        except Exception as exc:                # noqa: BLE001
            print(f"  [concrete] UCI fetch unavailable ({exc.__class__.__name__})")

        if df is None:                          # 3. local spreadsheet
            xls = os.path.join(REPO_ROOT, "AEEM6097", "project-data", "Concrete_Data.xls")
            if os.path.exists(xls):
                try:
                    df = pd.read_excel(xls)
                    df.columns = CONCRETE_COLS[: len(df.columns)]
                    print("  [concrete] read from the local .xls")
                except ImportError:
                    print("  [concrete] the local file is a legacy .xls and needs `xlrd`; "
                          "either `pip install xlrd` or let the UCI fetch handle it")
                except Exception as exc:        # noqa: BLE001
                    print(f"  [concrete] .xls unreadable ({exc.__class__.__name__})")

        if df is None:
            return None
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  [concrete] cached -> {os.path.relpath(csv_path, REPO_ROOT)}")

    df = pd.read_csv(csv_path).dropna()
    df.columns = [c.strip() for c in df.columns]
    y = df["Strength"].astype(float).to_numpy()
    X = df.drop(columns=["Strength"]).select_dtypes(include=[np.number]).astype(float)
    return X, y


def load_phiusiil(sample_size=20000):
    """PhiUSIIL phishing. Reuse the repo's own loader if importable; else fetch
    via ucimlrepo (id 967); else return None so the column shows N/A."""
    try:
        sys.path.insert(0, os.path.join(FIS, "tribble-tree"))
        import demo_phishing  # noqa: E402  -- repo loader, exact same features
        X, y = demo_phishing.load_data(sample_size=sample_size, random_state=42)
        return X, np.asarray(y)
    except Exception as exc:  # noqa: BLE001
        print(f"  [phiusiil] repo loader unavailable ({exc.__class__.__name__}); trying ucimlrepo")
    try:
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=967)
        X = ds.data.features.select_dtypes(include=[np.number]).dropna(axis=1)
        y = np.asarray(ds.data.targets).ravel()
        if sample_size and len(X) > sample_size:
            idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
            X, y = X.iloc[idx], y[idx]
        return X, y
    except Exception as exc:  # noqa: BLE001
        print(f"  [phiusiil] unavailable ({exc.__class__.__name__}); column -> N/A")
        return None


# --- model factories (return a fitted-predict callable, or None) -------------
def _try(build):
    """Return the constructed estimator, or None if construction fails."""
    try:
        return build()
    except Exception as exc:  # noqa: BLE001
        print(f"  [model] construction failed ({exc.__class__.__name__}); -> N/A")
        return None


def mog_regressor(seed):
    from tribblefis.gaussian_regressor import MixtureOfGaussiansFuzzyRegressor
    return _try(lambda: MixtureOfGaussiansFuzzyRegressor(
        n_output_buckets=3, tsk_order="1st", top_n=-1, random_state=seed))


def mog_classifier(seed):
    from tribblefis.gaussian_classifier import MixtureOfGaussiansFuzzyClassifier
    return _try(lambda: MixtureOfGaussiansFuzzyClassifier(top_n=5, random_state=seed))


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
