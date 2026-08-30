"""PhiUSIIL leak policy: the three legitimacy-derived features never reach a model.

Issue #215. ``repro_data.load_phiusiil`` drops ``PHIUSIIL_LEAK_COLS`` on load
(``drop_leak=True``, the default), so no caller can train on them by accident.
These tests pin three things:

* the drop happens, and the opt-out restores them;
* the numbers that JUSTIFY the drop still hold on the data, so the policy does
  not outlive its evidence;
* the shared loader's list and this experiment's own ``LEAK`` list agree.

On that last point: ``data.py`` deliberately stays standalone rather than
importing the shared constant. It is the file that had the label polarity right
when the shared loader had it wrong (see ``test_phiusiil_labels.py``), and an
independent second implementation is worth more as a cross-check than as a
duplicate to be eliminated. A test that the two agree buys the safety without
the coupling.

The dataset is gitignored, so the data-dependent tests here carry
``@requires_data`` and skip on CI. The two that need no data -- the leak-list
drift guard and the ``dataset_specs.yaml`` check -- deliberately do NOT, because
they are the ones that catch a future *edit* rather than a change in the data,
and those are exactly the ones worth running on every PR.

Run: ``python -m pytest experiments/phishing-oneclass/test_phiusiil_leak_policy.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

CSV = os.path.join(REPO_ROOT, "data", "PhiUSIIL_Phishing_URL_Dataset.csv")

NUMERIC_COLUMNS = 50  # what the CSV carries once the label and text columns go
MODELLED_FEATURES = 47  # what reaches a model after the leak drop

# Single-feature separation AUC on the full 235,795 rows, measured 2026-08-30.
# These are the numbers the drop is argued from, quoted in the loader docstring,
# dataset_specs.yaml and PROVENANCE_MAP note 31. The tolerance is loose enough
# to survive a float-summation difference and tight enough that a real change in
# the data or the loader would fail.
EXPECTED_SEPARATION = {
    "URLSimilarityIndex": 0.9961,
    "URLCharProb": 0.7679,
    "TLDLegitimateProb": 0.6089,
}
SEPARATION_TOL = 0.002

# Applied PER TEST, not to the module. Two of the assertions below need no data
# at all -- and one of them, the drift guard between the two LEAK lists, is the
# only test here that protects against a future EDIT rather than against the
# data changing. Under a module-level skip it would be the one assertion that
# never runs on CI, which is precisely backwards.
requires_data = pytest.mark.skipif(
    not os.path.exists(CSV),
    reason="data/PhiUSIIL_Phishing_URL_Dataset.csv is gitignored and absent here",
)


@pytest.fixture(scope="module")
def leaky():
    """The full frame WITH the leaks, for measuring what the drop is worth."""
    from repro_data import load_phiusiil

    got = load_phiusiil(sample_size=None, drop_leak=False)
    assert got is not None
    return got


@requires_data
def test_leaks_are_dropped_by_default():
    from repro_data import PHIUSIIL_LEAK_COLS, load_phiusiil

    got = load_phiusiil(sample_size=None)
    assert got is not None
    X, _ = got
    present = [c for c in PHIUSIIL_LEAK_COLS if c in X.columns]
    assert not present, f"leaky features reached a caller by default: {present}"
    assert X.shape[1] == MODELLED_FEATURES


@requires_data
def test_opt_out_restores_them(leaky):
    """`drop_leak=False` has to actually work -- a leak-aware experiment needs it."""
    from repro_data import PHIUSIIL_LEAK_COLS

    X, _ = leaky
    assert X.shape[1] == NUMERIC_COLUMNS
    for col in PHIUSIIL_LEAK_COLS:
        assert col in X.columns


@requires_data
def test_the_evidence_for_the_drop_still_holds(leaky):
    """The separation AUCs the policy is argued from, re-derived from the data.

    A policy whose justification is only in a docstring drifts silently away
    from the data it was measured on. This is the assertion that keeps the
    argument and the file in the same universe.
    """
    X, y = leaky
    yb = (np.asarray(y) == "phish").astype(int)
    for col, expected in EXPECTED_SEPARATION.items():
        auc = roc_auc_score(yb, X[col].to_numpy(float))
        sep = max(auc, 1.0 - auc)
        assert (
            abs(sep - expected) < SEPARATION_TOL
        ), f"{col} separates at {sep:.4f}, not the documented {expected:.4f}"


@requires_data
def test_url_similarity_index_is_the_strongest_feature_in_the_file(leaky):
    """The headline claim: the single most separating feature IS the leak.

    This is what makes the whole PhiUSIIL classification result suspect with it
    present, so it is worth asserting rather than asserting only its AUC.
    """
    X, y = leaky
    yb = (np.asarray(y) == "phish").astype(int)
    best, best_sep = None, -1.0
    for col in X.columns:
        auc = roc_auc_score(yb, X[col].to_numpy(float))
        sep = max(auc, 1.0 - auc)
        if sep > best_sep:
            best, best_sep = col, sep
    assert best == "URLSimilarityIndex", (
        f"the most separating feature is now {best} at {best_sep:.4f}; "
        "the argument in PROVENANCE_MAP note 31 assumes it is URLSimilarityIndex"
    )


def test_the_two_leak_lists_agree():
    """`data.py`'s LEAK and `repro_data.PHIUSIIL_LEAK_COLS` are the same policy.

    They are deliberately separate implementations (see the module docstring);
    they must not become separate *policies*.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import data as oneclass

    from repro_data import PHIUSIIL_LEAK_COLS

    assert set(oneclass.LEAK) == set(PHIUSIIL_LEAK_COLS)


def _phiusiil_spec():
    yaml = pytest.importorskip("yaml", reason="PyYAML not in this environment")
    with open(
        os.path.join(REPO_ROOT, "reproduce", "dataset_specs.yaml"), encoding="utf-8"
    ) as fh:
        return yaml.safe_load(fh)["datasets"]["phiusiil"]


def test_dataset_spec_says_47_not_50():
    """No data needed: the spec is committed, and it must record the drop.

    `features` is what dataset_specs.py's `{{dataset.phiusiil.shape}}`
    substitution renders into the proposal, so a spec still saying 50 would put
    a leak-era feature count in the document.
    """
    spec = _phiusiil_spec()
    assert spec["features"] == MODELLED_FEATURES
    assert spec["numeric_columns"] == NUMERIC_COLUMNS
    assert set(spec["verify"]["drop_columns"]) == set(EXPECTED_SEPARATION)


@requires_data
def test_loader_width_matches_the_spec():
    """And the loader actually returns what the spec promises."""
    from repro_data import load_phiusiil

    got = load_phiusiil(sample_size=None)
    assert got is not None
    assert got[0].shape[1] == _phiusiil_spec()["features"]
