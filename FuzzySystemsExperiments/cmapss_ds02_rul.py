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

Result on the official held-out engines (11, 14, 15): per-sample RMSE ~6.5
cycles, which beats the published DS02 CNN (7.22) and MLP (8.34); after the
monotone clamp the per-cycle curve rises on zero cycles. Uses the 18 real
sensors only -- adding the two "virtual" channels the literature also allows
(T40, P30) does not change the result.

Needs: h5py, numpy, pandas, scikit-learn, tribble-fis.  Run:

    python cmapss_ds02_rul.py --h5 NASA-CMAPSS/N-CMAPSS_DS02-006.h5
"""

import argparse
import time

from tribble_predictive_health import TribblePredictiveHealth, load_ncmapss
from tribble_predictive_health.metrics import rmse

# DS02's own train/test split, as published (the engines the file holds out).
TEST_UNITS = (11, 14, 15)


def main(h5_path):
    t0 = time.perf_counter()
    print(f"Loading {h5_path} ...")
    dev, _, sensor_cols = load_ncmapss(h5_path, "dev")
    test, _, _ = load_ncmapss(h5_path, "test")
    print(
        f"  {len(dev):,} dev rows, {len(test):,} test rows, {len(sensor_cols)} sensors"
    )

    # The estimator's defaults are the DS02 winning configuration.
    engine = TribblePredictiveHealth()
    print("Fitting the end-to-end predictive-health engine ...")
    t_fit = time.perf_counter()
    engine.fit(dev, dev["rul"])
    fit_seconds = time.perf_counter() - t_fit

    m = engine.score(test)
    frame = engine.predict_frame(test, include_true=True)  # monotone trajectory

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
    main(parser.parse_args().h5)
