#!/usr/bin/env python3
"""Run a table generator with the `wasserstein_distance` defect corrected.

`tribblefis.stats_numba.wasserstein_distance` returns the MEAN absolute CDF gap
over the union support instead of its integral against dx, which makes it
dimensionless and completely scale-invariant -- see
`diagnose_wasserstein_regression.py` for the derivation and the bisection that
attributes Table 4.1's classification collapse to it.

The fix upstream is one line. Until it lands, this runner answers the question
the proposal actually needs answered: **do the archived numbers still stand, or
were they wrong too?** It substitutes scipy's implementation into
`gauss_math`'s namespace and then runs the generator unmodified, so the only
difference from a stock run is the one function.

    uv run --project tribble-fis python \
        reproduce/experiments/run_with_wasserstein_fix.py \
        reproduce/tables/table_4_1_mog_baselines.py

Nothing is written into a submodule and no file is patched on disk; the
substitution lives only in this process. Point REPRO_OUTPUT_DIR somewhere of its
own so a corrected run cannot be mistaken for a stock one.
"""

from __future__ import annotations

import os
import runpy
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    target = os.path.abspath(sys.argv[1])
    if not os.path.exists(target):
        print(f"no such generator: {target}", file=sys.stderr)
        return 2

    from scipy.stats import wasserstein_distance as scipy_wasserstein

    from tribblefis import gauss_math

    before = gauss_math.wasserstein_distance
    gauss_math.wasserstein_distance = scipy_wasserstein
    print(
        f"[wasserstein-fix] {getattr(before, '__module__', '?')}."
        f"{getattr(before, '__name__', '?')} -> scipy.stats.wasserstein_distance"
    )

    # Sanity: prove the substitution bit, rather than assume it. A runner that
    # silently failed to patch would produce a stock run under a corrected
    # label, which is worse than not running it.
    probe_a = gauss_math.wasserstein_distance([0.0, 1.0], [0.0, 2.0])
    probe_b = gauss_math.wasserstein_distance([0.0, 1000.0], [0.0, 2000.0])
    if not (abs(probe_a - 0.5) < 1e-9 and abs(probe_b - 500.0) < 1e-6):
        print(
            f"[wasserstein-fix] ABORT: patch did not take "
            f"(got {probe_a} and {probe_b}, expected 0.5 and 500.0)",
            file=sys.stderr,
        )
        return 1
    print("[wasserstein-fix] verified: W1=0.5 at unit scale, 500.0 at x1000")

    sys.argv = [target] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(target))
    runpy.run_path(target, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
