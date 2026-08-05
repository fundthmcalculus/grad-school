"""Where everything this project reads and writes lives.

One module owns the layout so that no script has to guess it from its own location. That
matters here because the scripts do not all sit at the same depth: the reported pipeline is
at the top level, the exploratory work is in ``experiments/``, and both write into the same
``results/`` tree. A script that derived its output path from ``__file__`` would put its
results somewhere different depending on which folder it happened to be in.

Artifact names are parameterised by the rule-base **scale** (``small`` or ``large``) rather
than fixed, because the two scales are a reported comparison and not a default plus a
variant — see FINDINGS.md §3b. There is deliberately no unscaled ``results.json``: the
previous layout had one, it was a byte-identical copy of the ``small`` file, and a copy that
can silently go stale is worse than no copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
LEGACY = RESULTS / "legacy"
EXPERIMENTS = ROOT / "experiments"

#: The published-optimum index and ``.tsp`` files, which already ship in this repository.
TSPLIB_DIR = ROOT.parent / "ClusteringExperiments" / "tsplib"

#: Fitted per-counter costs for the deterministic time proxy the tuner optimises against.
COSTMODEL = RESULTS / "costmodel.npz"

#: The feature screen, frozen: ``experiments/features_probe.py`` produced it and
#: ``feature_registry.py`` checks itself against it.
FEATURE_SCREEN = RESULTS / "feature_screen.json"

#: LKH as a per-instance yardstick over the test set (one run each, hard timeout).
LKH_REFERENCE = RESULTS / "lkh.json"

#: LKH swept against our own swept solvers, curve against curve.
LKH_COMPARE = RESULTS / "lkh_compare.json"

#: The scaling ladder: both solvers across instance size, with the FIS arm present.
SCALING = RESULTS / "scaling.json"


def tuned(scale: str) -> Path:
    """The fitted rule base for one scale."""
    return RESULTS / f"tuned_{scale}.npz"


def tune_log(scale: str) -> Path:
    """What the fitting run recorded about itself, for one scale."""
    return RESULTS / f"tune_{scale}.json"


def benchmark(scale: str) -> Path:
    """The reported test-set comparison, for one scale."""
    return RESULTS / f"results_{scale}.json"


def ensure() -> None:
    """Create the output tree. Cheap, idempotent, and called by every writing script."""
    for d in (RESULTS, FIGURES, LEGACY):
        d.mkdir(parents=True, exist_ok=True)


def utf8_stdout() -> None:
    """Make stdout able to carry the characters this project's tables are written with.

    On Windows, redirecting a script's output to a file or a pipe gives it a ``cp1252`` stream,
    and ``cp1252`` cannot encode ``ρ``, ``≥``, ``●`` or an em dash. The failure mode is a
    ``UnicodeEncodeError`` *only when redirected* — the console itself replaces unknown
    characters instead of raising — so it hides from interactive use and surfaces the first
    time someone pipes the output into a file or into ``run_all.py``. That is exactly how
    ``feature_registry.py`` came to crash printing its own table.

    Called from ``main()`` rather than at import, so that importing a module for its functions
    never reconfigures the caller's streams.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def on_path() -> str:
    """Put the project root on ``sys.path`` and return it.

    Scripts in ``experiments/`` import the top-level modules, and a subprocess spawned to
    hold a native solver needs the same root. Both call this rather than each recomputing
    the number of ``.parent`` hops between them and the root.
    """
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
