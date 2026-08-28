#!/usr/bin/env bash
# Goal C1 — the full ten-seed Table 4.1 pass WITH the ANFIS / GA-FIS fuzzy
# baselines, producing both the accuracy table and the companion speed table.
#
# WHY THIS IS A DELIBERATE TRIGGER, NOT A ROUTINE BUILD
# The fuzzy adapters (reproduce/tables/_baseline_{anfis,gafis}.py) are hand-rolled
# and validated (reproduce/test_baseline_fuzzy.py), but running them across ten
# seeds and five dataset rows costs ~1h45m on the host of record -- a 1-seed pass
# measured 10.3 min (2026-08-27), dominated by GA-FIS on the Concrete grid
# (~114 s/seed, and Concrete runs twice) and the RT-IOT2022 scatter arm at 123k
# rows. GA-FIS being the slow arm is the finding, not a defect: it is the
# stochastic-search fuzzy modelling the MoG construction replaces.
#
# WHAT IT PRODUCES (overwrites both):
#   reproduce/outputs/table_4_1.{md,csv}                -- accuracy: baselines are
#       as accurate as MoG, often more so (they are fair, not strawmen).
#   reproduce/outputs/table_4_1b_baseline_timing.{md,csv} -- the C1 deliverable:
#       MoG is 2+ orders of magnitude faster (1-seed Concrete measured ~272x).
#
# 1-seed preview numbers (noisy, for sanity only):
#   Concrete R2  : MoG 0.82/0.88  ANFIS 0.835  GA-FIS 0.903  RF 0.927
#   RT-IOT2022 ac: MoG 0.928      ANFIS 0.985  GA-FIS 0.989  RF 0.999
#   Concrete speedup MoG vs slowest fuzzy: ~272x
#
# AFTER IT LANDS: re-quote Table 4.1 / add Table 4.1b to Chapter 4 (the speed
# claim in Ch 1/4/7/8 finally has a fuzzy baseline to be faster than), flip
# CHECKLIST C1 from open, and add a PROVENANCE_MAP note. Closes Goal C1.
set -euo pipefail
cd "$(dirname "$0")/.."

# Adapters must be fair before a 1h45m run trusts them.
uv run --project tribble-fis python reproduce/test_baseline_fuzzy.py

uv run --project tribble-fis python reproduce/tables/table_4_1_mog_baselines.py
