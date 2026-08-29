"""Repo-root and data-directory resolution for the shared loaders.

Kept in its own module so every loader agrees on where ``data/`` is without
re-deriving it from its own file location. The old loaders each computed
``REPO_ROOT`` relative to ``reproduce/tables/_fuzzy_models.py``; consolidating
them here means the path is defined once.
"""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_HERE)  # .../grad-school/repro_data -> .../grad-school

# tribble-fis carries the phishing loader (``demo_phishing``) that PhiUSIIL still
# delegates to; ``load_phiusiil`` puts ``tribble-fis/tribble-tree`` on the path.
FIS = os.path.join(REPO_ROOT, "tribble-fis")

# Datasets live in ``data/``, never in a submodule -- tribble-fis deleted its
# bundled ``gaussian_mixture/`` data directory upstream (8484fd6) to keep the
# library pure, and caching a dataset back into a pinned submodule would leave it
# permanently dirty. Override with GRAD_SCHOOL_DATA.
DATA_DIR = os.environ.get("GRAD_SCHOOL_DATA", os.path.join(REPO_ROOT, "data"))
