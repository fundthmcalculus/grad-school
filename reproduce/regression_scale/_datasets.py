"""Backward-compat shim.

The large-scale-regression loaders now live in ``repro_data.regression_scale``
so every experiment shares one definition. They are re-exported here UNCHANGED
so existing ``import _datasets`` callers (table_a7_regression_scale.py,
feature_expansion.py, model_family_pilot.py, mog_top_p_sweep.py) keep working.
See repro_data/regression_scale.py for the loaders, provenance and caching.
"""

from __future__ import annotations

import os
import sys

# repo root -> `import repro_data`
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from repro_data.regression_scale import (  # noqa: E402,F401
    DATA_DIR,
    DATASETS,
    SUPERCONDUCT_ZIP_URL,
    load_housing,
    load_superconduct,
)
