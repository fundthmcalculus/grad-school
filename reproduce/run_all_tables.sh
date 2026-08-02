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

# Check for submodule SHA divergence before running.
check_submodule_shas

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

check_submodule_shas() {
  # Guard against silent submodule SHA divergence. This failure has happened twice
  # (fix/pin-extreme-bucket-means, resolve-flm-pr), causing the harness to emit
  # tables from a different commit than the last run without warning.
  #
  # If a previous archive exists, compare its recorded SHAs to the current ones.
  # On mismatch, print a loud warning (failure mode: someone quotes old numbers).
  # Don't refuse to emit (the user may have intentionally changed submodules), but
  # make it impossible to miss.
  local last_prov=""
  local last_fis="" last_cluster="" last_grad=""
  local curr_fis="" curr_cluster="" curr_grad=""

  # Find the most recent archive (by modification time).
  if [ -d "$OUT" ]; then
    last_prov=$(find "$OUT" -maxdepth 2 -name "PROVENANCE.txt" -printf '%T@ %p\n' 2>/dev/null \
      | sort -rn | head -1 | cut -d' ' -f2-)
  fi

  if [ -z "$last_prov" ] || [ ! -f "$last_prov" ]; then
    return 0  # no previous archive; nothing to compare
  fi

  # Extract SHAs from the last archive.
  last_fis=$(grep "^tribble-fis:" "$last_prov" | cut -d' ' -f2 | head -1)
  last_cluster=$(grep "^tribble-cluster:" "$last_prov" | cut -d' ' -f2 | head -1)
  last_grad=$(grep "^grad-school:" "$last_prov" | cut -d' ' -f2 | head -1)

  # Get current SHAs.
  curr_fis=$(git -C "$ROOT/tribble-fis" rev-parse HEAD 2>/dev/null || echo "")
  curr_cluster=$(git -C "$ROOT/tribble-cluster" rev-parse HEAD 2>/dev/null || echo "")
  curr_grad=$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "")

  # Compare. If any differ, print a loud warning.
  if [ "$last_fis" != "$curr_fis" ] || [ "$last_cluster" != "$curr_cluster" ] || [ "$last_grad" != "$curr_grad" ]; then
    echo "########################################################################"
    echo "## WARNING: SUBMODULE SHA DIVERGENCE DETECTED                          ##"
    echo "##                                                                    ##"
    echo "## The current submodule SHAs differ from the last archive. This      ##"
    echo "## means the generated tables are from different code than before.    ##"
    echo "##                                                                    ##"
    echo "## Last archive: $last_prov"
    [ -n "$last_fis" ] && echo "##   tribble-fis:    $last_fis"
    [ -n "$last_cluster" ] && echo "##   tribble-cluster: $last_cluster"
    [ -n "$last_grad" ] && echo "##   grad-school:    $last_grad"
    echo "##"
    echo "## Current SHAs:"
    echo "##   tribble-fis:    $curr_fis"
    echo "##   tribble-cluster: $curr_cluster"
    echo "##   grad-school:    $curr_grad"
    echo "##                                                                    ##"
    echo "## Continuing with this run, but VERIFY the code change is           ##"
    echo "## intentional before quoting any numbers from it.                    ##"
    echo "########################################################################"
    echo
  fi
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
  # Machine identity. Recorded because timings are not portable between hosts OR
  # between runs on one host: the same generator, same tribble-cluster commit and
  # same ten seeds moved every wall-clock figure in Table 3.1 by ~45% across two
  # runs hours apart, while the speedup RATIO moved under 3%. Without this block
  # a future reader cannot tell a code change from a thermal one. The governor and
  # boost fields are here for the same reason -- they are the usual culprits.
  echo "machine:"
  printf '  %-16s %s\n' "host"     "$(hostname 2>/dev/null || echo unknown)"
  printf '  %-16s %s\n' "os"       "$( (. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME") || uname -s )"
  printf '  %-16s %s\n' "kernel"   "$(uname -srm 2>/dev/null)"
  printf '  %-16s %s\n' "cpu"      "$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)"
  printf '  %-16s %s\n' "cores"    "$(nproc 2>/dev/null) logical$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Core(s) per socket:[[:space:]]*/, /p' | tr -d '\n' | sed 's/$/ physical per socket/')"
  printf '  %-16s %s\n' "ram"      "$(awk '/MemTotal/ {printf "%.1f GiB (%.1f GB decimal)", $2/1048576, $2*1024/1e9}' /proc/meminfo 2>/dev/null)"
  printf '  %-16s %s\n' "governor" "$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'n/a')"
  printf '  %-16s %s\n' "boost"    "$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 'n/a')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null \
      | while IFS= read -r g; do printf '  %-16s %s\n' "gpu" "$g"; done
  else
    printf '  %-16s %s\n' "gpu" "$(lspci 2>/dev/null | grep -iE 'vga|3d controller' | sed 's/^[^ ]* //' | head -1 || echo 'none detected')"
  fi
  printf '  %-16s %s\n' "python"   "$(python3 -V 2>&1)"
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
