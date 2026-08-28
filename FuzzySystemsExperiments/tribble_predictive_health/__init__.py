"""tribble_predictive_health -- a reusable RUL processing engine.

The whole condition-correction -> features -> RUL-cap -> fuzzy-system ->
monotone-clamp pipeline as one scikit-learn estimator, `TribblePredictiveHealth`,
plus the individual steps (`preprocessing`), scoring (`metrics`), and dataset
loaders (`data`) it is built from.

    from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss

    dev, cond, sensors = load_ncmapss("N-CMAPSS_DS02-006.h5", "dev")
    test, _, _ = load_ncmapss("N-CMAPSS_DS02-006.h5", "test")
    engine = TribblePredictiveHealth().fit(dev, dev["rul"])
    print(engine.score(test))          # both scoring conventions
    curve = engine.predict_frame(test)  # deployable RUL trajectory
"""

from . import cache
from .cache import Bundle, load_or_build, load_or_build_many
from .data import load_ncmapss
from .metrics import nasa_score, per_engine_canonical, rmse
from .pipeline import TribblePredictiveHealth

__all__ = [
    "TribblePredictiveHealth",
    "cache",
    "load_ncmapss",
    "load_or_build",
    "load_or_build_many",
    "Bundle",
    "nasa_score",
    "per_engine_canonical",
    "rmse",
]
