"""Remaining-useful-life on N-CMAPSS DS02 with a TRIBBLE fuzzy system.

The single best case, start to finish, in a dozen lines of actual work. All of
the pipeline -- condition correction, memory features, the RUL cap, the fuzzy
system, and the monotone clamp -- now lives in the reusable
`TribblePredictiveHealth` estimator (`tribble_predictive_health/`), so this
script only has to load DS02, fit, and report. What each step does and why it is
the winning choice is documented on the class and its `preprocessing` module.

The estimator's defaults *are* the DS02 best case (memory features, quadratic
consequents, the hamacher norm, the monotone clamp), the configuration that won
a long design-of-experiments; nothing else is tried here.

Loading + condition-correcting + featurising DS02 (2.4 GB) is the slow part, so
it is cached: the first run builds the feature tables and writes them under
`outputs/cmapss-cache/`; later runs read the cache and go straight to the fit,
turning a ~10 s iteration into a sub-second one. Pass `--rebuild-cache` after a
preprocessing change. Because the cache holds the already-condition-corrected,
already-featurised tables, the fit runs through `fit_featurized` with
`condition_correction=False` -- byte-identical numbers to the uncached path
(`test_cache.py` checks this).

Result on the official held-out engines (11, 14, 15): per-sample RMSE ~7.2
cycles, in line with the published DS02 CNN (7.22) and beating the MLP (8.34);
after the monotone clamp the per-cycle curve rises on zero cycles. Uses the 18
real sensors only. This is *not* the same result as the design-of-experiments'
original "best" pipeline (the now-deleted `cmapss_rul_best.py`), which reached
RMSE ~6.5 by adding two "virtual" channels (T40, P30) to the same model -- this
consolidated loader never reads that HDF5 group, so those channels are not
available here. An earlier version of this docstring claimed dropping them
"does not change the result"; that was asserted, not measured, and it is
wrong -- confirmed by re-running the original DOE script with T40/P30, which
reproduces ~6.5. Kept real-sensors-only as a deliberate simplification; see
`research/proposal-defense/prose/04-fast-fis-synthesis-mog.md` §4.4.1 for the
full account.

Needs: h5py, numpy, pandas, scikit-learn, tribble-fis.  Run:

    python cmapss_ds02_rul.py --h5 NASA-CMAPSS/N-CMAPSS_DS02-006.h5
"""

import argparse
import time

from tribble_predictive_health import TribblePredictiveHealth, load_or_build
from tribble_predictive_health.metrics import rmse
from tribble_predictive_health.preprocessing import clamp_monotone, per_cycle

# DS02's own train/test split, as published (the engines the file holds out).
TEST_UNITS = (11, 14, 15)


def _trajectory(engine, test_table):
    """The deployable monotone RUL trajectory from an already-featurised test
    table -- the cached-path equivalent of `engine.predict_frame`."""
    s = engine.predict_samples_featurized(test_table)
    cyc = per_cycle(
        s["unit"].to_numpy(),
        s["cycle"].to_numpy(),
        s["predicted_rul"].to_numpy(),
        true=s["rul"].to_numpy(),
    )
    if engine.monotone:
        cyc = clamp_monotone(cyc)
    return cyc.rename(columns={"pred": "rul"})


def main(h5_path, rebuild_cache=False):
    t0 = time.perf_counter()
    print(
        f"Loading DS02 features (cache under outputs/cmapss-cache/) from {h5_path} ..."
    )
    # The estimator's defaults are the DS02 winning configuration; `raw_memory`
    # is that default aggregation.
    bundle = load_or_build(h5_path, "raw_memory", rebuild=rebuild_cache)
    print(
        f"  {len(bundle.dev):,} train rows, {len(bundle.test):,} test rows, "
        f"{len(bundle.feature_cols)} features"
    )

    # Condition correction is already baked into the cached tables, so the
    # estimator only caps/scales/fits (fit_featurized) and scores.
    engine = TribblePredictiveHealth(condition_correction=False)
    print("Fitting the end-to-end predictive-health engine ...")
    t_fit = time.perf_counter()
    engine.fit_featurized(bundle.dev, bundle.feature_cols)
    fit_seconds = time.perf_counter() - t_fit

    m = engine.score_featurized(bundle.test)
    frame = _trajectory(engine, bundle.test)  # monotone trajectory, with `true`

    print(
        f"\n=== DS02 remaining useful life ({engine.n_rules_} rules,"
        f" fit in {fit_seconds:.2f}s) ==="
    )
    print(
        f"  per-sample RMSE {m['per_sample_rmse']:5.2f}   "
        f"(published baselines: CNN 7.22, MLP 8.34)\n"
    )
    print(
        f"  per cycle, raw       RMSE {m['raw_cycle_rmse']:5.2f}   "
        f"rising cycles {m['raw_rising']:5.1%}"
    )
    print(
        f"  per cycle, monotone  RMSE {m['monotone_cycle_rmse']:5.2f}   "
        f"rising cycles {m['monotone_rising']:5.1%}   <- recommended\n"
    )
    for unit in TEST_UNITS:
        sub = frame[frame["unit"] == unit]
        print(
            f"  engine {unit}: {len(sub):3d} cycles   "
            f"RMSE {rmse(sub['true'], sub['rul']):.2f}"
        )
    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", default="NASA-CMAPSS/N-CMAPSS_DS02-006.h5")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the preprocessing cache instead of reading it.",
    )
    args = parser.parse_args()
    main(args.h5, rebuild_cache=args.rebuild_cache)
