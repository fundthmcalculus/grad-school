#!/usr/bin/env bash
# Run every table generator and archive reproduce/outputs/ under a label.
#
#   reproduce/run_all_tables.sh baseline-d0d6714
#
# Each script runs in its own submodule environment, with its stdout kept
# alongside the tables it produced. A script that fails does not stop the run --
# its log records the failure and the summary at the end says which ones did not
# finish, so "broken" is never confused with "unchanged".
set -uo pipefail

LABEL="${1:?usage: run_all_tables.sh <label> [table_name ...]}"
shift
ONLY=("$@")          # optional: run just these tables, e.g. to backfill one
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
  table_4_4_openset
)
CLUSTER_TABLES=(
  table_3_1_pvat_scaling
  table_3_1_reorder_three_arm
)

declare -A STATUS

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
  # EXTRA_DEPS covers packages a table needs that the submodule does not declare
  # as a base dependency (tribble-cluster keeps scipy under its `dev` extra, so
  # `uv run --project` alone leaves table_3_1_pvat_scaling unable to import it).
  uv run --project "$ROOT/$project" ${EXTRA_DEPS:-} python \
      "$ROOT/reproduce/tables/$name.py" >"$log" 2>&1
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
  echo "tribble-fis: $(git -C "$ROOT/tribble-fis" rev-parse HEAD)"
  echo "tribble-cluster: $(git -C "$ROOT/tribble-cluster" rev-parse HEAD)"
  echo "grad-school: $(git -C "$ROOT" rev-parse HEAD)"
  echo "seeds:       ${REPRO_SEEDS:-0,1,2,3,4 (default)}"
  echo
  echo "status:"
  for t in "${FIS_TABLES[@]}" "${CLUSTER_TABLES[@]}"; do
    # Unset means the filter skipped it; say so rather than claiming a result.
    printf '  %-38s %s\n' "$t" "${STATUS[$t]:-not-run-this-pass}"
  done
} >> "$PROV"

echo
cat "$DEST/PROVENANCE.txt"
