"""Shared dataset loaders for the reproducibility experiments.

One import point for every experiment, so a dataset's preprocessing -- its
dropped columns, its errata handling, its leaky features -- is defined once and
cannot drift between callers. The drop/errata decisions are the authoritative
ones recorded in ``reproduce/dataset_specs.yaml``.

    from repro_data import load_concrete, load_beth, DATA_DIR
    X, y = load_concrete()

``load_beth`` returns a ``{'train','val','test'}`` dict of ``(X, y)`` pairs;
every other loader returns ``(X, y)`` or ``None`` when the data is unavailable.

Runs on nothing heavier than numpy/pandas (plus scikit-learn where a loader
delegates to it), so importing it never pulls in a modelling stack.
"""

from __future__ import annotations

from ._paths import DATA_DIR, REPO_ROOT
from .loaders import (
    CONCRETE_COLS,
    load_beth,
    load_bikeshare,
    load_bodyfat,
    load_concrete,
    load_darwin,
    load_phiusiil,
    load_rt_iot2022,
    load_shuttle,
    load_wec,
)
from .regression_scale import load_housing, load_superconduct

__all__ = [
    "DATA_DIR",
    "REPO_ROOT",
    "CONCRETE_COLS",
    "load_beth",
    "load_bikeshare",
    "load_bodyfat",
    "load_concrete",
    "load_darwin",
    "load_phiusiil",
    "load_rt_iot2022",
    "load_shuttle",
    "load_wec",
    "load_housing",
    "load_superconduct",
]
