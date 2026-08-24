"""Shared ``sys.path`` bootstrap for the fis-to-neural-net driver scripts.

Every script in this directory is run directly (``python
experiments/fis-to-neural-net/run_foo.py``), needs its own directory on
``sys.path`` to import siblings like ``fis2nn`` and ``simplicial``, and needs
one or two directories elsewhere in the repo -- ``reproduce/tables`` for the
shared dataset/scaling helpers, ``AnalyticalDynamics/chaos`` for the pendulum
operator. This factors that boilerplate into one call.
"""

from __future__ import annotations

import os
import sys


def add_repo_paths(
    caller_file: str, *repo_subpaths: tuple[str, ...]
) -> tuple[str, str]:
    """Insert the caller's directory and repo-relative subpaths into sys.path.

    ``caller_file`` is the calling script's ``__file__``. Each entry in
    ``repo_subpaths`` is a tuple of path components relative to the repo root
    (two directories up from the caller), e.g. ``("reproduce", "tables")``.
    Paths already on ``sys.path`` are left alone rather than re-inserted.

    Returns ``(here, repo)`` so callers can keep using those two paths (for
    ``OUTPUTS``, data files, ``os.path.relpath`` in messages, etc.) without
    recomputing them.
    """
    here = os.path.dirname(os.path.abspath(caller_file))
    repo = os.path.dirname(os.path.dirname(here))

    for path in [here] + [os.path.join(repo, *subpath) for subpath in repo_subpaths]:
        if path not in sys.path:
            sys.path.insert(0, path)

    return here, repo
