#!/usr/bin/env python
"""Dev-time timing check for the dual-form ridge solve in `_baseline_anfis.py`.

Isolates ANFIS / GA-FIS `fit_predict` on Concrete's grid partition (256 rules,
D = R*(M+1) = 2304 params vs N = 824 train rows) -- the D > N case the dual
solve targets -- so the same measurement can be taken before and after that
change, on one machine, without paying for the other Table 4.1 rows or the
MoG/RF arms. Not part of the table suite; a single seed, for sanity only.

Run: uv run python reproduce/quick_option_a_dual_solve_bench.py
"""

import os
import sys
import time

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables"))
import _baseline_anfis as A  # noqa: E402
import _baseline_gafis as G  # noqa: E402
import _fuzzy_models as F  # noqa: E402

X, y = (np.asarray(v) for v in F.load_concrete())
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)

for label, mod in (("ANFIS", A), ("GA-FIS", G)):
    t0 = time.perf_counter()
    pred = mod.fit_predict(Xtr, ytr, Xte, kind="reg", seed=0)
    dt = time.perf_counter() - t0
    r2 = r2_score(yte, pred)
    print(f"{label}: {dt:.2f}s  R2={r2:.4f}")
