"""Tests for the shared ``sys.path`` bootstrap used by the driver scripts.

``_bootstrap.add_repo_paths`` replaced inline ``sys.path.insert`` boilerplate
in eight sibling scripts (see ``find_slow_problem.py`` for a real caller). It
mutates the global, process-wide ``sys.path``, so every test here saves and
restores it to avoid leaking entries into whatever else runs in the same
pytest session.

Run: ``python -m pytest experiments/fis-to-neural-net/test_bootstrap.py``
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _bootstrap import add_repo_paths  # noqa: E402


@pytest.fixture
def saved_sys_path():
    """Snapshot sys.path and restore it exactly, even if the test fails."""
    snapshot = sys.path[:]
    try:
        yield
    finally:
        sys.path[:] = snapshot


def test_no_subpaths_inserts_here_and_returns_here_repo(saved_sys_path):
    here, repo = add_repo_paths(__file__)

    expected_here = os.path.dirname(os.path.abspath(__file__))
    expected_repo = os.path.dirname(os.path.dirname(expected_here))

    assert here == expected_here
    assert repo == expected_repo
    assert here in sys.path


def test_single_subpath_inserts_joined_absolute_path(saved_sys_path):
    here, repo = add_repo_paths(__file__, ("reproduce", "tables"))

    expected = os.path.join(repo, "reproduce", "tables")
    assert expected in sys.path


def test_multiple_subpaths_all_inserted(saved_sys_path):
    here, repo = add_repo_paths(
        __file__, ("reproduce", "tables"), ("AnalyticalDynamics", "chaos")
    )

    assert os.path.join(repo, "reproduce", "tables") in sys.path
    assert os.path.join(repo, "AnalyticalDynamics", "chaos") in sys.path


def test_repeat_call_does_not_duplicate_entries(saved_sys_path):
    # Other test modules in this directory (test_fis2nn.py, this file's own
    # module-level import setup, pytest's own import-mode insertion) may
    # already have their own copy of `here` on sys.path before this test
    # even runs, so the dedup guard can only be judged by whether *repeated*
    # add_repo_paths calls grow that count -- not by an absolute count of 1.
    add_repo_paths(__file__, ("reproduce", "tables"))
    before = sys.path[:]
    here, repo = add_repo_paths(__file__, ("reproduce", "tables"))

    assert sys.path == before, "second call must not append or move anything"

    expected = os.path.join(repo, "reproduce", "tables")
    before_here = before.count(here)
    before_tables = before.count(expected)

    add_repo_paths(__file__, ("reproduce", "tables"))

    assert sys.path.count(here) == before_here
    assert sys.path.count(expected) == before_tables


def test_insertion_order_most_recent_first(saved_sys_path):
    """Mirrors find_slow_problem.py's real usage of two subpaths.

    Each new path is inserted at position 0, so the *last* subpath tuple
    passed ends up ahead of the first subpath tuple, which in turn ends up
    ahead of the caller's own directory (``here``) -- most-recently-inserted
    first.

    Other test modules collected in the same pytest session (e.g. pytest's
    own rootdir-based sys.path insertion, or unrelated fixtures) can leave
    entries anywhere in ``sys.path`` before this test runs, so asserting
    exact absolute indices (``sys.path[0]``, ``sys.path[2]``, ...) is not
    reliable outside this file's own isolated run. Instead, scrub any
    pre-existing copies of the three paths under test right before calling
    ``add_repo_paths``, so their *relative* order after the call is
    unambiguous regardless of what else shares the session.
    """
    expected_here = os.path.dirname(os.path.abspath(__file__))
    expected_repo = os.path.dirname(os.path.dirname(expected_here))
    tables_path = os.path.join(expected_repo, "reproduce", "tables")
    chaos_path = os.path.join(expected_repo, "AnalyticalDynamics", "chaos")

    for stale in (expected_here, tables_path, chaos_path):
        while stale in sys.path:
            sys.path.remove(stale)

    here, repo = add_repo_paths(
        __file__, ("reproduce", "tables"), ("AnalyticalDynamics", "chaos")
    )
    assert here == expected_here
    assert repo == expected_repo

    assert (
        sys.path.index(chaos_path) < sys.path.index(tables_path) < sys.path.index(here)
    )
