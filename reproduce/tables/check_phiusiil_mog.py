"""Check the MoG PhiUSIIL arm under the Table 4.1 protocol, for the open PhiUSIIL defect.

Current code is expected to give accuracy 0.440 +/- 0.181 at about 0.13 s, below the
0.576 majority rate. Gaussian naive Bayes on the same protocol gives 0.980 +/- 0.002
(table_4_1_gnb.py). If your environment gives a different number, the environment is
the first suspect.

Run (from repo root):  uv run --project tribble-fis python reproduce/tables/check_phiusiil_mog.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import _fuzzy_models as _fm  # noqa: E402


def main():
    X, y = _fm.load_phiusiil()
    accs, ts = [], []
    for seed in C.SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        model = _fm.mog_classifier(seed)
        with C.timed() as t:
            p = np.asarray(model.fit(Xtr, ytr).predict(Xte))
        accs.append(accuracy_score(yte, p))
        ts.append(t.seconds)
    a, t = np.array(accs), np.array(ts)
    print(
        f"MoG PhiUSIIL  acc {a.mean():.3f} ± {a.std(ddof=1):.3f}  "
        f"train {t[1:].mean():.2f} s  per-seed {np.round(a, 3).tolist()}"
    )
    _, counts = np.unique(np.asarray(y), return_counts=True)
    print(f"majority-class rate {counts.max() / counts.sum():.4f}")


if __name__ == "__main__":
    main()
