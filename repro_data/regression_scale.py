"""Dataset loaders for the large-scale-regression pilot.

**Re-sourced from canonical locations on 2026-08-11.** RESULTS_2026-08-05.md
found `archive.ics.uci.edu` and `ndownloader.figshare.com` both blocked (403
on CONNECT) from the session that ran the original pilot, and fell back to
GitHub-hosted mirrors verified only by row/column count. Network conditions
had changed by the time this file was revisited: both canonical hosts are
reachable now (`archive.ics.uci.edu` returns 200; the figshare-backed sklearn
fetcher succeeds via GET even though a HEAD probe against it still 403s).
Both loaders below have been re-pointed at their canonical source:

- **California Housing**: `sklearn.datasets.fetch_california_housing()` --
  built into scikit-learn, no mirror needed. This is the derived 8-feature
  version (`MedInc`/`HouseAge`/`AveRooms`/`AveBedrms`/`Population`/
  `AveOccup`/`Latitude`/`Longitude`, Pace & Barry 1997), **not** the raw 1997
  StatLib file the earlier GitHub mirror served (which had `ocean_proximity`
  and un-derived per-block totals rather than per-household ratios). This is
  a genuine change of derivation, not just of source -- see the docstring on
  `load_housing()` -- but it is the version actually cited as "California
  Housing" throughout the ML literature and sklearn's own docs, and it is
  the one reachable canonically.
- **Superconductivity**: `archive.ics.uci.edu/static/public/464/
  superconductivty+data.zip` (UCI dataset id 464), downloaded directly and
  its `train.csv` member extracted -- no `ucimlrepo` dependency needed for a
  single well-known file. Verified 21,263 rows x 81 features + target,
  matching the previous mirror's shape exactly.

Fetched files are cached under `DATA_DIR` (default `data/regression_scale/`,
override with `PILOT_DATA_DIR`) and gitignored, the same treatment
`data/.gitignore` already gives PhiUSIIL's 57 MB CSV.

Moved here from `reproduce/regression_scale/_datasets.py` (which now re-exports
these names) so every experiment shares one definition. Behaviour is unchanged.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile

import pandas as pd

from ._paths import REPO_ROOT

DATA_DIR = os.environ.get(
    "PILOT_DATA_DIR", os.path.join(REPO_ROOT, "data", "regression_scale")
)

SUPERCONDUCT_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/464/superconductivty+data.zip"
)


def load_housing():
    """California Housing via `sklearn.datasets.fetch_california_housing()`
    -- the canonical, built-in source (Pace & Barry 1997, StatLib via
    figshare). 20,640 rows, 8 numeric features, no missing values. This is
    the DERIVED per-household version (`AveRooms`/`AveBedrms`/`AveOccup`
    ratios), not the raw StatLib file with `ocean_proximity` that the
    earlier GitHub mirror served -- see this module's docstring.

    sklearn caches under its own `data_home` (default `~/scikit_learn_data`),
    separate from this pilot's `DATA_DIR`; no local caching duplicated here.
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    X = data.data.astype(float)
    y = data.target.astype(float)
    y.name = "y_value"
    return X, y


def load_superconduct():
    """UCI Superconductivity (dataset id 464, `train.csv`): 21,263 rows, 81
    derived elemental features (mean/gmean/entropy/std/range of atomic mass,
    valence, thermal conductivity, etc.), target `critical_temp`. Heavily
    collinear by construction -- see RESULTS_2026-08-05.md's decorrelation
    section. Fetched from UCI's own static-file host and cached locally."""
    path = os.path.join(DATA_DIR, "superconduct_train.csv")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        raw = urllib.request.urlopen(SUPERCONDUCT_ZIP_URL, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as zf, zf.open("train.csv") as f:
            df = pd.read_csv(f)
        df.to_csv(path, index=False)
        print(
            f"  [superconduct_train.csv] fetched from UCI (id 464), cached -> "
            f"{os.path.relpath(path, REPO_ROOT)}"
        )
    df = pd.read_csv(path).dropna()
    y = df["critical_temp"].astype(float)
    y.name = "y_value"
    X = df.drop(columns=["critical_temp"]).astype(float)
    return X, y


DATASETS = {
    "California Housing": load_housing,
    "Superconductivity": load_superconduct,
}
