"""Table 4.1 companion -- Gaussian naive Bayes on the classification rows.

Gaussian naive Bayes is the nearest acknowledged relative of the MoG construction
(Chapter 4 §4.2): one Gaussian per feature and class under independence. It was
missing from Table 4.1, so this fills that row on the same protocol: the same
loaders, the same ``common.SEEDS``, the same 80/20 split per seed, and the same
timer. Only classification rows apply; GNB has no regression form here.

Rows whose data is not present are reported N/A, never imputed.

Run (from repo root):  uv run --project tribble-fis python reproduce/tables/table_4_1_gnb.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
sys.path.insert(0, _TABLES)
import common as C  # noqa: E402
import _fuzzy_models as _fm  # noqa: E402


def _bench(X, y):
    """Per-seed (train_seconds, accuracy) for GNB on identical splits to Table 4.1."""
    # One discarded warm-up fit, as in table_4_1_mog_baselines._bench.
    Xw, Xwte, yw, _ = train_test_split(X, y, test_size=0.2, random_state=C.SEEDS[0])
    GaussianNB().fit(Xw, yw).predict(Xwte)

    times, accs = [], []
    for seed in C.SEEDS:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
        with C.timed() as t:
            p = GaussianNB().fit(Xtr, ytr).predict(Xte)
        times.append(t.seconds)
        accs.append(accuracy_score(yte, p))
    return times, accs


def main():
    print("Table 4.1 companion -- Gaussian naive Bayes (classification rows)")
    phi = _fm.load_phiusiil()
    if phi is None:
        print("  PhiUSIIL: data not present -> N/A")
        return
    X, y = phi
    times, accs = _bench(X, y)
    t = np.array(times)
    a = np.array(accs)
    print(
        f"  PhiUSIIL  accuracy {a.mean():.3f} ± {a.std(ddof=1):.3f}   "
        f"train {t.mean():.2f} ± {t.std(ddof=1):.2f} s   ({len(a)} seeds)"
    )


if __name__ == "__main__":
    main()
