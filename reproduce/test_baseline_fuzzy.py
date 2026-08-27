#!/usr/bin/env python
"""Correctness guards for the ANFIS / GA-FIS baselines (Table 4.1, Goal C1).

A baseline is only worth quoting if it is *fair*: a strawman that loses to the
MoG arm by construction would flatter this work rather than test it. Two failure
modes are guarded, both found while building these:

  1. Underfitting on Concrete. Published ANFIS on Concrete reaches test R^2 ~0.8;
     if either arm limps in far below that, its premise search is broken.
  2. High-dimensional firing collapse. A product of 80+ Gaussians underflows to
     exactly 0 in float64, which — before the log-space firing fix — drove the
     scatter classifier to *below chance* (0.06 on a 12-class set whose majority
     is 0.77). This asserts the classifier clears the majority-class baseline,
     which it cannot do if firing has collapsed.

Run:  uv run --project tribble-fis python reproduce/test_baseline_fuzzy.py
"""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables"))

import _baseline_anfis as A  # noqa: E402
import _baseline_gafis as G  # noqa: E402
import _fuzzy_models as F  # noqa: E402

failures = []


def check(name, cond, detail):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")
    if not cond:
        failures.append(name)


# 1. Regression fairness floor on Concrete (grid partition, 8 features).
Xc, yc = (np.asarray(v) for v in F.load_concrete())
Xtr, Xte, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=0)
for label, mod, floor in (("ANFIS", A, 0.75), ("GA-FIS", G, 0.60)):
    r2 = r2_score(yte, mod.fit_predict(Xtr, ytr, Xte, kind="reg", seed=0))
    check(f"{label} Concrete R2", r2 > floor, f"R2={r2:.3f} (floor {floor})")

# 2. High-dimensional classification clears the majority baseline (scatter, 50
#    features). PhiUSIIL subsamples to 20k in its loader; that is plenty to show
#    the firing did not collapse.
Xp, yp = F.load_phiusiil()
Xp, yp = np.asarray(Xp), np.asarray(yp)
Xtr, Xte, ytr, yte = train_test_split(Xp, yp, test_size=0.2, random_state=0)
_, counts = np.unique(yte, return_counts=True)
majority = counts.max() / len(yte)
for label, mod in (("ANFIS", A), ("GA-FIS", G)):
    acc = accuracy_score(yte, mod.fit_predict(Xtr, ytr, Xte, kind="clf", seed=0))
    check(f"{label} PhiUSIIL acc", acc > majority, f"acc={acc:.3f} > majority {majority:.3f}")

print()
if failures:
    print(f"ERROR: {len(failures)} baseline guard(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("SUCCESS: both fuzzy baselines are fair (above floor and above chance).")
sys.exit(0)
