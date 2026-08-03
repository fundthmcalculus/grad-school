"""Local `log_transform`, retained after `tribble-fis` PR #67 deleted it.

PR #67 (`a385a1a`) removed `gauss_math.log_transform` along with
`standard_transform` and `detect_and_apply_log_transform`, on the grounds that
they were unused. That is true of the library and its tests, but not of this
directory: eight scripts here call `log_transform`, and `tribblefis.scaling`'s
new `UnitScalar`/`StandardScalar` cannot stand in for it. They differ in three
ways that matter to these callers:

  * the scalers *auto-detect* which columns to log by dynamic range; these
    scripts name their columns explicitly, chosen per dataset;
  * the scalers use `log1p(x - min)`, a shift-then-log1p; this is `log(x + c)`
    for a caller-supplied `c`, and `phiusiil.py` deliberately passes
    `c = 0.0001` to keep ratio features near zero from collapsing;
  * the scalers always normalize afterwards, which these scripts do not want
    here (`nasa.py` normalizes *before* logging, the opposite order).

So this is a verbatim copy of the deleted implementation rather than a rewrite:
the point is that these scripts keep producing the numbers they produced
before the pin moved. New work should prefer `tribblefis.scaling`.
"""

import numpy as np
import pandas as pd


def log_transform(X: pd.DataFrame, column: str | list[str], offset=0):
    """Apply log transformation to a column in the DataFrame with an offset to avoid log(0).

    Verbatim from `tribblefis.gauss_math` @ `d0efefc`, the last pin that had it.
    Note it mutates `X` in place as well as returning it -- `iot-botnet.py`
    relies on the mutation and discards the return value.
    """
    X[column] = np.log(X[column] + offset)
    return X
