"""PhiUSIIL label polarity: ``label == 1`` is legitimate, ``label == 0`` is phishing.

This is a guard, not a discovery. The shared loader
``repro_data.load_phiusiil`` mapped it the other way until 2026-08-30, having
inherited the inversion from ``tribble-fis/tribble-tree/demo_phishing.py``,
which it was verified byte-identical against. The inversion is invisible in
every accuracy-style metric -- those are invariant under a consistent
relabelling of two classes -- so nothing in the suite failed while it was
wrong. Only outputs that NAME a class showed it, and there is one:
``reproduce/figures/fig_06_fuzzy_tree.py`` renders leaf labels.

A defect that hides from every metric you report needs a test that looks at the
data directly, which is what this is.

The dataset is gitignored (``data/.gitignore`` records how to recover it), so on
CI every test here SKIPS. That is deliberate: the alternative is a synthetic
fixture asserting the mapping against itself, which would pass no matter which
way round the real file is. These tests are meaningful on a host with the data,
which is every host that produces a number from it.

Run: ``python -m pytest experiments/phishing-oneclass/test_phiusiil_labels.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

CSV = os.path.join(REPO_ROOT, "data", "PhiUSIIL_Phishing_URL_Dataset.csv")

# The published split of this dataset, in this order.
N_LEGIT = 134_850
N_PHISH = 100_945

pytestmark = pytest.mark.skipif(
    not os.path.exists(CSV),
    reason="data/PhiUSIIL_Phishing_URL_Dataset.csv is gitignored and absent here",
)


@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(CSV, encoding="utf-8-sig").dropna()


def test_class_counts_identify_which_label_is_legitimate(raw):
    """134,850 legitimate and 100,945 phishing -- so label 1 is the larger."""
    counts = raw["label"].value_counts()
    assert counts.get(1) == N_LEGIT, f"label==1 has {counts.get(1)}, not {N_LEGIT}"
    assert counts.get(0) == N_PHISH, f"label==0 has {counts.get(0)}, not {N_PHISH}"


def test_url_similarity_index_is_exactly_100_on_label_1(raw):
    """The strongest single check.

    ``URLSimilarityIndex`` is a URL's similarity to a whitelist of KNOWN
    LEGITIMATE URLs. A class on which it is exactly 100.0 for every row, with
    zero variance, is the legitimate class by construction -- there is no
    reading of that feature under which the phishing class sits at perfect
    similarity to the legitimate whitelist.
    """
    v = raw.loc[raw.label == 1, "URLSimilarityIndex"].to_numpy(float)
    assert v.std() == 0.0 and np.all(v == 100.0), (
        f"label==1 URLSimilarityIndex is {v.mean():.4f} +- {v.std():.4f}, "
        "not the constant 100.0 that identifies the legitimate class"
    )
    other = raw.loc[raw.label == 0, "URLSimilarityIndex"].to_numpy(float)
    assert other.std() > 1.0, "label==0 should be the dispersed class"


def test_is_https_is_constant_on_label_1(raw):
    """Second independent check: the legitimate corpus is always HTTPS."""
    v = raw.loc[raw.label == 1, "IsHTTPS"].to_numpy(float)
    assert np.all(v == 1.0), "label==1 is not always HTTPS"
    assert raw.loc[raw.label == 0, "IsHTTPS"].to_numpy(float).std() > 0.0


def test_shared_loader_agrees(raw):
    """`repro_data.load_phiusiil` must name the classes the same way round.

    Checked against the FULL file rather than a sample, so the row alignment is
    positional and unambiguous.
    """
    from repro_data import load_phiusiil

    got = load_phiusiil(sample_size=None)
    assert got is not None
    _, y = got
    y = np.asarray(y)
    assert (y == "legit").sum() == N_LEGIT, (
        f"loader labels {(y == 'legit').sum()} rows 'legit'; the legitimate "
        f"class has {N_LEGIT}. The label mapping is inverted."
    )
    assert (y == "phish").sum() == N_PHISH
    assert np.array_equal(y == "legit", raw["label"].to_numpy() == 1)


def test_shared_loader_agrees_with_the_one_class_harness(raw):
    """The two loaders in this repo must not disagree about which class is which.

    ``data.load()`` here returns ``1 = legit``; ``repro_data.load_phiusiil``
    returns the string ``"legit"``. They read the same file and must select the
    same rows -- when they did not, the one-class experiment and every
    ``reproduce/`` table were quietly describing opposite classes.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data as oneclass

    from repro_data import load_phiusiil

    cwd = os.getcwd()
    try:
        os.chdir(REPO_ROOT)  # oneclass.load() reads a repo-relative path
        _, y_oneclass = oneclass.load()
    finally:
        os.chdir(cwd)

    got = load_phiusiil(sample_size=None)
    assert got is not None
    _, y_shared = got
    assert np.array_equal(np.asarray(y_shared) == "legit", np.asarray(y_oneclass) == 1)
