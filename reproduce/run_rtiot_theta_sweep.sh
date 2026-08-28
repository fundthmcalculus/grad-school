#!/usr/bin/env bash
# Regenerate the RT-IOT2022 theta-sweep (Fig 4.2's missing row) and the Table 4.7b
# headline in ONE invocation, on the de-leaked loader.
#
# WHY ONE INVOCATION: table_4_4_openset.py runs theta_sweep() and then emits the
# headline table from the same fitted folds. Running them separately is exactly
# how PROVENANCE_MAP note 21 got a sweep whose detection (0.777) disagreed with
# its own headline (0.798). Keep them together so they agree by construction.
#
# COST: ~2 to 2.5 hours on the host of record -- the ~94 min headline run happens
# first, then the 7-theta x 12-class x 10-seed sweep. This is why it is a
# deliberate trigger, not part of a routine build.
#
# The theta list is note 21's canonical Fig 4.2 grid. REPRO_THETA_SWEEP is a
# comma-separated LIST OF THETAS, not a boolean: =1 silently emits a one-row
# table at theta=1.0 where the boost saturates and every cell is zero.
#
# Outputs (both overwritten):
#   reproduce/outputs/table_4_4_openset.{md,csv}      -- Table 4.7b headline
#   reproduce/outputs/table_4_4b_theta_sweep.{md,csv} -- Fig 4.2 row
#
# After it lands: re-quote the sweep, fill Fig 4.2's RT-IOT2022 row, update
# section 4.4's "the sweep was not run here" sentence, and supersede note 21's
# pre-de-leak sweep figures. Closes the remainder of grad-school#184.
set -euo pipefail
cd "$(dirname "$0")/.."

REPRO_OPENSET_DATASET=rt-iot2022 \
REPRO_THETA_SWEEP=0.5,0.6,0.7,0.8,0.9,0.99,1.1 \
  uv run --project tribble-fis python reproduce/tables/table_4_4_openset.py
