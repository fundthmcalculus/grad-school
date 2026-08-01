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

LABEL="${1:?usage: run_all_tables.sh <label>}"
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

run_one() {
  local project="$1" name="$2"
  local log="$DEST/logs/$name.log"
  printf '  %-38s ' "$name"
  local t0 t1
  t0=$(date +%s)
  if uv run --project "$ROOT/$project" python "$ROOT/reproduce/tables/$name.py" \
        >"$log" 2>&1; then
    STATUS[$name]=ok
  else
    STATUS[$name]="FAILED"
  fi
  t1=$(date +%s)
  printf '%-8s %4ss\n' "${STATUS[$name]}" "$((t1 - t0))"
}

echo "=== $LABEL ==="
echo "tribble-fis:"
for t in "${FIS_TABLES[@]}"; do run_one tribble-fis "$t"; done
echo "tribble-cluster:"
for t in "${CLUSTER_TABLES[@]}"; do run_one tribble-cluster "$t"; done

# Archive the tables themselves next to the logs.
find "$OUT" -maxdepth 1 -type f \( -name '*.md' -o -name '*.csv' \) \
  -exec cp {} "$DEST/" \;

# Record exactly what produced these numbers.
{
  echo "label:       $LABEL"
  echo "generated:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "tribble-fis: $(git -C "$ROOT/tribble-fis" rev-parse HEAD)"
  echo "tribble-cluster: $(git -C "$ROOT/tribble-cluster" rev-parse HEAD)"
  echo "grad-school: $(git -C "$ROOT" rev-parse HEAD)"
  echo "seeds:       ${REPRO_SEEDS:-0,1,2,3,4 (default)}"
  echo
  echo "status:"
  for t in "${FIS_TABLES[@]}" "${CLUSTER_TABLES[@]}"; do
    printf '  %-38s %s\n' "$t" "${STATUS[$t]}"
  done
} > "$DEST/PROVENANCE.txt"

echo
cat "$DEST/PROVENANCE.txt"
