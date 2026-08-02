#!/usr/bin/env bash
# Run every table generator and archive reproduce/outputs/ under a label.
#
#   reproduce/run_all_tables.sh baseline-d0d6714
#   reproduce/run_all_tables.sh --fast smoke-check     # minutes, NOT citable
#
# Each script runs in its own submodule environment, with its stdout kept
# alongside the tables it produced. A script that fails does not stop the run --
# its log records the failure and the summary at the end says which ones did not
# finish, so "broken" is never confused with "unchanged".
#
# --fast cuts the seed count on the handful of tables that dominate the runtime,
# for smoke-testing a change. It does NOT produce numbers fit for the document:
# moving this project from three seeds to ten retracted one published conclusion,
# refuted another, and exposed a model that diverges on one split in ten. A fast
# run is therefore stamped as such in PROVENANCE.txt, loudly, because the failure
# mode being guarded against is someone quoting one a month from now.
set -uo pipefail

FAST=0
if [ "${1:-}" = "--fast" ]; then FAST=1; shift; fi

LABEL="${1:?usage: run_all_tables.sh [--fast] <label> [table_name ...]}"
shift
ONLY=("$@")          # optional: run just these tables, e.g. to backfill one

# Seeds used for slow tables under --fast. Everything else keeps common.SEEDS:
# the cheap tables gain nothing from a reduction, so they should not pay the
# credibility cost of one.
FAST_SEEDS="${REPRO_FAST_SEEDS:-0,1,2}"

# Runtimes at 10 seeds from outputs/seeds10-2026-08-01/: these four are 1301s,
# 302s, 187s and 112s; the remaining seven total under two minutes combined.
declare -A SLOW_TABLES=(
  [table_concrete_reconciliation]=1
  [table_3_1_pvat_scaling]=1
  [table_norm_conorm_matrix]=1
  [table_hyperparam_normalization]=1
)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/reproduce/outputs"
DEST="$OUT/$LABEL"

mkdir -p "$DEST/logs"

# (project, script) pairs -- the project is the uv environment the script needs.
FIS_TABLES=(
  table_concrete_reconciliation
  table_hyperparam_normalization
  table_g5_output_partitioning
  table_g5b_skew_sweep
  table_4_1_mog_baselines
  table_6_1_model_family
  table_norm_conorm_matrix
  table_4_4_openset
)
CLUSTER_TABLES=(
  table_3_1_pvat_scaling
  table_3_1_reorder_three_arm
)
# Stdlib-only renderers -- no submodule environment, so they run on the host python.
PLAIN_TABLES=(
  table_5_x_ch5_selection
)

declare -A STATUS
declare -A SEEDS_USED

# Resolved once, from common.py, so the provenance file reports the seed list the
# generators will actually use rather than a default repeated here. This line
# used to be hardcoded, and when the default moved from five seeds to ten it
# silently kept recording five.
FULL_SEEDS="$(SEEDLIST_ROOT="$ROOT" python3 -c 'import os,sys;
sys.path.insert(0, os.path.join(os.environ["SEEDLIST_ROOT"], "reproduce"));
import common; print(",".join(map(str, common.SEEDS)))' 2>/dev/null \
    || echo "${REPRO_SEEDS:-unknown}")"

# A script that exits 0 having written no table is NOT a success -- that is what
# a missing dataset looks like (table_4_4_openset prints "no dataset available"
# and returns cleanly). Reporting it as ok would hide the one thing the run is
# supposed to surface, so what the script actually wrote is checked, not just
# the exit code.
#
# This compares an exact snapshot of the output files before and after, rather
# than asking which files are newer than a timestamp. A time window cannot work
# here: consecutive tables finish within the same second or two, so the previous
# table's fresh output falls inside any grace period wide enough to catch a
# same-second write, and every script after the first looks like it produced
# something. Path + mtime + size has no such ambiguity.
snapshot_tables() {
  find "$OUT" -maxdepth 1 -type f \( -name '*.md' -o -name '*.csv' \) \
    -printf '%p %T@ %s\n' 2>/dev/null | sort
}

wanted() {
  [ ${#ONLY[@]} -eq 0 ] && return 0
  local t
  for t in "${ONLY[@]}"; do [ "$t" = "$1" ] && return 0; done
  return 1
}

run_one() {
  local project="$1" name="$2"
  wanted "$name" || return 0
  local log="$DEST/logs/$name.log"
  printf '  %-38s ' "$name"
  local t0 t1 rc before after
  before=$(snapshot_tables)
  t0=$(date +%s)
  # A project of "-" means the script needs no submodule environment (stdlib-only
  # renderers), so it runs on the host python rather than through uv.
  #
  # EXTRA_DEPS covers packages a table needs that the submodule does not declare
  # as a base dependency (tribble-cluster keeps scipy under its `dev` extra, so
  # `uv run --project` alone leaves table_3_1_pvat_scaling unable to import it).
  # Under --fast the slow tables run on a reduced seed set. Recorded per table in
  # PROVENANCE.txt rather than assumed, so the archive says which cells are thin.
  local seed_env=""
  if [ $FAST -eq 1 ] && [ -n "${SLOW_TABLES[$name]:-}" ]; then
    seed_env="$FAST_SEEDS"
    SEEDS_USED[$name]="$FAST_SEEDS (reduced by --fast)"
  else
    SEEDS_USED[$name]="$FULL_SEEDS"
  fi

  # Only override REPRO_SEEDS when --fast actually applies. Exporting it empty
  # would be worse than not setting it: common.py reads os.environ.get(...) with
  # a default, so an empty string is NOT the default -- it splits to [""] and
  # dies on int(""). Pass the variable through untouched otherwise.
  local -a env_prefix=()
  [ -n "$seed_env" ] && env_prefix=(env "REPRO_SEEDS=$seed_env")

  if [ "$project" = "-" ]; then
    "${env_prefix[@]}" python3 "$ROOT/reproduce/tables/$name.py" >"$log" 2>&1
  else
    "${env_prefix[@]}" uv run --project "$ROOT/$project" ${EXTRA_DEPS:-} python \
        "$ROOT/reproduce/tables/$name.py" >"$log" 2>&1
  fi
  rc=$?
  t1=$(date +%s)
  after=$(snapshot_tables)
  if [ $rc -ne 0 ]; then
    STATUS[$name]="FAILED"
  elif [ "$before" = "$after" ]; then
    STATUS[$name]="no-output"
  else
    STATUS[$name]=ok
  fi
  printf '%-10s %4ss\n' "${STATUS[$name]}" "$((t1 - t0))"
}

echo "=== $LABEL ==="
echo "tribble-fis:"
for t in "${FIS_TABLES[@]}"; do run_one tribble-fis "$t"; done
echo "tribble-cluster:"
EXTRA_DEPS="--with scipy"
for t in "${CLUSTER_TABLES[@]}"; do run_one tribble-cluster "$t"; done
EXTRA_DEPS=""
echo "no submodule env:"
for t in "${PLAIN_TABLES[@]}"; do run_one - "$t"; done

# Archive the tables themselves next to the logs.
find "$OUT" -maxdepth 1 -type f \( -name '*.md' -o -name '*.csv' \) \
  -exec cp {} "$DEST/" \;

# Record exactly what produced these numbers. A filtered run appends rather than
# overwrites: backfilling one table must not erase the provenance of the rest.
PROV="$DEST/PROVENANCE.txt"
{
  if [ ${#ONLY[@]} -gt 0 ]; then
    echo
    echo "--- backfill $(date -u +%Y-%m-%dT%H:%M:%SZ): ${ONLY[*]} ---"
  else
    echo "label:       $LABEL"
    echo "generated:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  fi
  if [ $FAST -eq 1 ]; then
    echo
    echo "########################################################################"
    echo "## FAST SWEEP -- REDUCED SEEDS -- NOT CITABLE                         ##"
    echo "##                                                                    ##"
    printf '## %-68s ##\n' "Slow tables ran on seeds $FAST_SEEDS instead of the full set."
    echo "## This archive is a smoke test. Do NOT quote any number from it.     ##"
    echo "## Going from three seeds to ten in this project retracted one        ##"
    echo "## conclusion, refuted another, and exposed a model that diverges on  ##"
    echo "## one split in ten. See reproduce/PROVENANCE_MAP.md.                 ##"
    echo "########################################################################"
    echo
  fi
  echo "tribble-fis: $(git -C "$ROOT/tribble-fis" rev-parse HEAD)"
  echo "tribble-cluster: $(git -C "$ROOT/tribble-cluster" rev-parse HEAD)"
  echo "grad-school: $(git -C "$ROOT" rev-parse HEAD)"
  echo "seeds:       $FULL_SEEDS${REPRO_SEEDS:+ (REPRO_SEEDS override in effect)}"
  echo
  echo "status:"
  for t in "${FIS_TABLES[@]}" "${CLUSTER_TABLES[@]}" "${PLAIN_TABLES[@]}"; do
    # Unset means the filter skipped it; say so rather than claiming a result.
    # The seed set is recorded PER TABLE because --fast makes it vary between
    # them, and a reader must be able to tell which cells are thin.
    printf '  %-38s %-12s seeds=%s\n' \
      "$t" "${STATUS[$t]:-not-run-this-pass}" "${SEEDS_USED[$t]:-—}"
  done
} >> "$PROV"

echo
cat "$DEST/PROVENANCE.txt"
