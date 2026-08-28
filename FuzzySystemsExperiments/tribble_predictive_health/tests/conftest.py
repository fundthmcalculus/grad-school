"""Put the `FuzzySystemsExperiments` directory on the path so these tests import
the package the same way the drivers do (`from tribble_predictive_health import
...`), regardless of pytest's invocation directory.
"""

import pathlib
import sys

# tests/ -> tribble_predictive_health/ -> FuzzySystemsExperiments/
_FSE = pathlib.Path(__file__).resolve().parents[2]
if str(_FSE) not in sys.path:
    sys.path.insert(0, str(_FSE))
