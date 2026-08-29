"""Canonical dataset loaders shared across the reproducibility experiments.

These were extracted verbatim from ``reproduce/tables/_fuzzy_models.py`` so a
single import point exists for every experiment. The preprocessing -- which
columns are dropped, which errata are handled, which features are treated as
leaky -- is deliberately unchanged from the originals; ``_fuzzy_models`` now
re-exports these names, so no caller's numbers move. The drop/errata decisions
are documented against ``reproduce/dataset_specs.yaml``, and
``reproduce/test_dataset_loaders.py`` guards that the widths still match it.

Every loader returns ``(X, y)`` (``X`` a numeric DataFrame, ``y`` a Series or
ndarray) except ``load_beth``, which returns a ``{'train','val','test'}`` dict
of ``(X, y)`` pairs. On a missing file or unreadable dataset a loader prints a
diagnostic and returns ``None``, so a table can render that cell as N/A rather
than aborting the whole run.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from ._paths import DATA_DIR, FIS, REPO_ROOT

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
        xls = os.path.join(REPO_ROOT, "AEEM6097", "project-data", "Concrete_Data.xls")
        if os.path.exists(xls):
            try:
                df = pd.read_excel(xls)
                df.columns = CONCRETE_COLS[: len(df.columns)]
                print("  [concrete] read from the local .xls")
            except ImportError:
                print(
                    "  [concrete] the local file is a legacy .xls and needs `xlrd`; "
                    "install it or provide data/Concrete_Data.csv"
                )
            except Exception as exc:  # noqa: BLE001
                print(f"  [concrete] .xls unreadable ({exc.__class__.__name__})")

        if df is None:
            print(
                f"  [concrete] file not found at {os.path.relpath(csv_path, REPO_ROOT)}"
            )
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
    """RT-IOT2022: 123,117 rows × 81 features, 12 classes (open-set detection).

    Returns (X, y) or None if file not found.

    The shipped CSV leads with an unnamed index column, which pandas reads as
    `Unnamed: 0`. It is numeric, so slicing off the label and keeping every
    numeric column kept it as an 82nd FEATURE -- and it is not a harmless row
    number: the file concatenates the twelve per-class captures and the counter
    RESTARTS AT ZERO for each one, so it encodes the label. Any value above
    8,107 belongs to `DOS_SYN_Hping` and to nothing else. `load_bikeshare`
    already drops `instant` for exactly this reason; this loader did not.
    Dropping it here changes every RT-IOT2022 number -- see PROVENANCE_MAP.
    """
    local = os.path.join(DATA_DIR, "RT_IOT2022.csv")
    try:
        df = pd.read_csv(local)
        y = df.iloc[:, -1]  # last column is the target
        X = df.iloc[:, :-1]
        X = X.select_dtypes(include=[np.number]).astype(float)
        leaky_index = [c for c in X.columns if str(c).startswith("Unnamed")]
        if leaky_index:
            X = X.drop(columns=leaky_index)
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
        print(
            f"  [rt-iot2022] failed to load ({exc.__class__.__name__}); column -> N/A"
        )
        return None


def load_beth():
    """BETH host telemetry: 1,141,078 labelled rows, binary anomaly detection.

    The 3.8M figure this docstring used to quote is the size of the full BETH
    capture, not of the three labelled splits that are actually shipped and
    used here (763,144 + 188,967 + 188,967).

    Returns explicit train/validate/test splits:
      dict with keys 'train', 'val', 'test'; each maps to (X, y).
      Returns None if any split is missing.

    Splits are loaded from:
      - data/beth/labelled_training_data.csv
      - data/beth/labelled_validation_data.csv
      - data/beth/labelled_testing_data.csv
    """
    beth_dir = os.path.join(DATA_DIR, "beth")
    splits = {
        "train": "labelled_training_data.csv",
        "val": "labelled_validation_data.csv",
        "test": "labelled_testing_data.csv",
    }

    try:
        result = {}
        for split_name, filename in splits.items():
            path = os.path.join(beth_dir, filename)
            if not os.path.exists(path):
                print(
                    f"  [beth] missing {split_name} split at {os.path.relpath(path, REPO_ROOT)}"
                )
                return None

            df = pd.read_csv(path)
            y = df.iloc[:, -1]
            X = df.iloc[:, :-1]
            X = X.select_dtypes(include=[np.number]).astype(float)
            y = np.asarray(y)
            result[split_name] = (X, y)

        total_rows = sum(len(result[s][0]) for s in splits.keys())
        print(
            f"  [beth] loaded train/val/test from {os.path.relpath(beth_dir, REPO_ROOT)}: "
            f"{total_rows} total rows × {result['train'][0].shape[1]} features"
        )
        return result
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
                idx = np.random.RandomState(42).choice(
                    len(X), sample_size, replace=False
                )
                X, y = X.iloc[idx], y[idx]
            print(
                f"  [shuttle] loaded {os.path.relpath(local, REPO_ROOT)}: "
                f"{len(X)} rows × {X.shape[1]} features"
            )
            return X, y
        except Exception as exc:  # noqa: BLE001
            print(
                f"  [shuttle] failed to load ({exc.__class__.__name__}); column -> N/A"
            )
            return None
    else:
        print(f"  [shuttle] file not found at {os.path.relpath(local, REPO_ROOT)}")
        return None


def load_bikeshare(target_col="cnt", sample_size=None):
    """Bike Sharing Demand: 17,379 rows × 12 features, regression (demand prediction).

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
        X = df.select_dtypes(include=[np.number]).drop(
            columns=[target_col], errors="ignore"
        )
        # Drop obvious ID/index columns if present
        X = X.drop(columns=["instant"], errors="ignore")
        # `casual` and `registered` are the target's two ADDENDS, not features:
        # casual + registered == cnt exactly, on all 17,379 rows. Leaving them in
        # X asks the model to recover a sum it has already been handed, which is
        # why the RF reference on this row read a perfect 1.000. They are dropped
        # for every target, not just `cnt` -- predicting `casual` from
        # `registered` and `cnt` is the same leak wearing a different hat.
        X = X.drop(columns=["casual", "registered"], errors="ignore").astype(float)

        if sample_size and len(X) > sample_size:
            idx = np.random.RandomState(42).choice(len(X), sample_size, replace=False)
            X, y = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(
                drop=True
            )

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


def load_wec(site="Perth", n_wecs=49, target="Total_Power"):
    """Wave Energy Converter farm power output (regression).

    Files: ``data/WEC_{site}_{n_wecs}.csv`` (site in {Perth, Sydney}, n_wecs in
    {49, 100}). Returns (X, y) or None if the file is missing.

    LEAK. The per-converter power columns are the target's own addends: the farm
    total ``Total_Power`` is the sum of the individual ``Power1..PowerN``, so
    handing those to a model that predicts the total is asking it to recover a
    sum it already holds. Every ``Power*`` column is dropped, along with ``qW``
    (the aggregate q-factor, another direct function of the outputs). What
    remains are the converter (X, Y) placement coordinates -- the genuine design
    features. This is the exact exclusion ``quick_wec_baseline.py`` applied
    inline; it now lives here so the WEC arms cannot drift apart on it.
    """
    local = os.path.join(DATA_DIR, f"WEC_{site}_{n_wecs}.csv")
    try:
        df = pd.read_csv(local)
    except FileNotFoundError:
        print(f"  [wec] file not found at {os.path.relpath(local, REPO_ROOT)}")
        return None
    if target not in df.columns:
        print(f"  [wec] target '{target}' not in {os.path.relpath(local, REPO_ROOT)}")
        return None
    y = df[target].astype(float)
    exclude = [target, "qW"] + [c for c in df.columns if c.startswith("Power")]
    X = (
        df.drop(columns=exclude, errors="ignore")
        .select_dtypes(include=[np.number])
        .astype(float)
    )
    print(
        f"  [wec] loaded {os.path.relpath(local, REPO_ROOT)}: "
        f"{len(X)} rows × {X.shape[1]} features"
    )
    return X, y


def load_bodyfat(drop_leak=True):
    """Body Fat (StatLib / Johnson 1996): 13 anthropometric features -> body-fat %.

    File: ``data/bodyfat.csv``. Returns (X, y) or None if the file is missing.

    LEAK. ``Density`` IS the target in another coordinate -- ``BodyFat`` was
    computed from it by Siri's equation ``495/Density - 450`` -- so it reproduces
    the target at R2 0.977 as shipped and R2 1.000 once the errata rows are
    excluded. Both existing arms (``FuzzySystemsExperiments/bodyfat.py`` and
    ``reproduce/experiments/ch5_end_to_end.py``) drop it; ``drop_leak=True`` (the
    default and the protocol) does the same here, leaving 13 features.

    ERRATA are NOT filtered, on purpose: the file is kept byte-identical to the
    canonical one and the errata are counted rather than silently dropped. Full
    provenance, mixed units (weight lb / height in / circumferences cm) and the
    per-case errata list live in ``data/bodyfat.names``.
    """
    local = os.path.join(DATA_DIR, "bodyfat.csv")
    if not os.path.exists(local):
        print(f"  [bodyfat] file not found at {os.path.relpath(local, REPO_ROOT)}")
        return None
    df = pd.read_csv(local)
    y = df["BodyFat"].astype(float)
    drop = ["BodyFat"] + (["Density"] if drop_leak else [])
    X = df.drop(columns=drop)
    print(
        f"  [bodyfat] loaded {os.path.relpath(local, REPO_ROOT)}: "
        f"{len(X)} rows × {X.shape[1]} features"
        + ("  (Density dropped as a leak)" if drop_leak else "")
    )
    return X, y
