"""Dataset loaders for the large-scale-regression pilot.

**Not the canonical sources.** `archive.ics.uci.edu` and
`ndownloader.figshare.com` -- where California Housing and the UCI
Superconductivity set actually live -- were unreachable from the session that
ran this pilot (an egress policy denial, confirmed via that session's proxy
status endpoint, not a transient failure). Both loaders below fetch from a
GitHub-hosted mirror instead, verified only by row/column count against the
known canonical shapes (20,640 rows for housing; 21,263 x 81 for
superconductivity). **Re-point these at the canonical source before any
number drawn from them is treated as a committed measurement** -- see
RESULTS_2026-08-05.md's provenance section.

Fetched files are cached under `DATA_DIR` (default `data/regression_scale/`,
override with `PILOT_DATA_DIR`) and gitignored, the same treatment
`data/.gitignore` already gives PhiUSIIL's 57 MB CSV.
"""

from __future__ import annotations

import os

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.environ.get(
    "PILOT_DATA_DIR", os.path.join(REPO_ROOT, "data", "regression_scale")
)

HOUSING_URL = (
    "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
    "datasets/housing/housing.csv"
)
SUPERCONDUCT_URL = (
    "https://raw.githubusercontent.com/monica110394/"
    "Predicting-the-Critical-Temperature-of-a-Superconductor/"
    "master/train.csv"
)


def _cached(name, url):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
        print(
            f"  [{name}] fetched from mirror, cached -> "
            f"{os.path.relpath(path, REPO_ROOT)}"
        )
    return pd.read_csv(path)


def load_housing():
    """California Housing, the ORIGINAL 1997 StatLib file (20,640 rows, 9
    numeric + 1 categorical column) -- NOT sklearn's `fetch_california_housing`
    derived 8-feature version (AveRooms/AveBedrms/AveOccup ratios), which is
    not reproduced here. `ocean_proximity` (categorical) is dropped; 8 numeric
    features remain, and by coincidence that matches sklearn's own feature
    count despite being a different derivation.
    """
    df = _cached("housing.csv", HOUSING_URL).dropna()
    y = df["median_house_value"].astype(float)
    y.name = "y_value"
    X = df.drop(columns=["median_house_value", "ocean_proximity"]).astype(float)
    return X, y


def load_superconduct():
    """UCI Superconductivity (`train.csv`): 21,263 rows, 81 derived elemental
    features (mean/gmean/entropy/std/range of atomic mass, valence, thermal
    conductivity, etc.), target `critical_temp`. Heavily collinear by
    construction -- see RESULTS_2026-08-05.md's decorrelation section."""
    df = _cached("superconduct_train.csv", SUPERCONDUCT_URL).dropna()
    y = df["critical_temp"].astype(float)
    y.name = "y_value"
    X = df.drop(columns=["critical_temp"]).astype(float)
    return X, y


DATASETS = {
    "California Housing": load_housing,
    "Superconductivity": load_superconduct,
}
