"""Unit tests for ``_ds02_harness.bootstrap``.

``_ds02_harness`` does real work at import time: it calls
``bootstrap("FuzzySystemsExperiments")`` and then imports numpy/sklearn and
``tribble_predictive_health`` (from the ``FuzzySystemsExperiments`` dir at the
repo root). The ``FuzzySystemsExperiments`` argument is a repo-root-relative
path, so it only resolves if the process cwd is the repo root at import time
-- which is how the sibling experiment scripts are meant to be run ("Run from
the repo root", per the module docstring). To make this test file runnable
from *this* directory too, we chdir to the repo root just long enough to
perform the (one-time, cached-after-that) import, then restore both cwd and
sys.path before any test body runs.

Every test only cares about ``bootstrap``'s own sys.path-mutation behaviour,
so an autouse fixture snapshots ``sys.path`` before each test and restores it
afterward -- nothing here should leak into other test files collected in the
same pytest session.
"""

import os
import sys

import pytest

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir, os.pardir))


def _import_ds02_harness():
    """Import _ds02_harness with cwd temporarily pinned to the repo root.

    The module's own top-level ``bootstrap("FuzzySystemsExperiments")`` call
    needs cwd == repo root to resolve. sys.path and cwd are both restored to
    their pre-call state afterward -- the module (and everything it pulled
    into sys.modules along the way) stays importable/cached regardless.
    """
    orig_cwd = os.getcwd()
    orig_sys_path = sys.path[:]
    try:
        os.chdir(_REPO_ROOT)
        import _ds02_harness
    finally:
        os.chdir(orig_cwd)
        sys.path[:] = orig_sys_path
    return _ds02_harness


_ds02_harness = _import_ds02_harness()
bootstrap = _ds02_harness.bootstrap


@pytest.fixture(autouse=True)
def _isolate_sys_path():
    """sys.path is global mutable process state; never let a test's
    bootstrap() calls leak into other tests or other files in this session."""
    snapshot = sys.path[:]
    try:
        yield
    finally:
        sys.path[:] = snapshot


def test_bootstrap_inserts_single_path_at_sys_path_zero():
    bootstrap("some/path")
    assert sys.path[0] == "some/path"


def test_bootstrap_multiple_args_end_up_reverse_order_at_front():
    """Each path is inserted at position 0 in the order given, so the LAST
    argument ends up at sys.path[0] and the FIRST argument is pushed to
    sys.path[1]. This is the real behavior 13 sibling scripts depend on via
    calls like bootstrap("FuzzySystemsExperiments", os.path.dirname(__file__))."""
    bootstrap("a", "b")
    assert sys.path[0] == "b"
    assert sys.path[1] == "a"


def test_bootstrap_does_not_dedupe_repeated_calls():
    before = sys.path.count("x")
    bootstrap("x")
    bootstrap("x")
    after = sys.path.count("x")
    assert after - before == 2
    # and both land at the front, in insertion order (most-recent first)
    assert sys.path[0] == "x"
    assert sys.path[1] == "x"
