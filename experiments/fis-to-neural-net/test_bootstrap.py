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
    passed ends up at ``sys.path[0]``, the first subpath tuple next, and the
    caller's own directory (``here``) after that -- most-recently-inserted
    first.
    """
    here, repo = add_repo_paths(
        __file__, ("reproduce", "tables"), ("AnalyticalDynamics", "chaos")
    )

    tables_path = os.path.join(repo, "reproduce", "tables")
    chaos_path = os.path.join(repo, "AnalyticalDynamics", "chaos")

    assert sys.path[0] == chaos_path
    assert sys.path[1] == tables_path
    assert sys.path[2] == here
