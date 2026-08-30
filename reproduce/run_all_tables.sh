#!/usr/bin/env bash
# Run every table generator and archive reproduce/outputs/ under a label.
#
#   reproduce/run_all_tables.sh baseline-d0d6714
#   reproduce/run_all_tables.sh --fast smoke-check          # minutes, NOT citable
#   reproduce/run_all_tables.sh --archive-only my-label     # runs nothing; see below
#
# Never edit this file while it is running. Bash reads a script incrementally, so an
# insert shifts every later byte offset out from under the interpreter and it resumes
# mid-token. That happened once: thirteen generators finished green and the archive step
# died on a syntax error in code that parses fine. --archive-only exists to recover it.
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

usage() {
  sed -n '2,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# Any unrecognised leading flag is a usage error, not a label. `--help` used to be
# taken as one, which archived a full run under reproduce/outputs/--help/ -- seven
# log files that were then committed.
case "${1:-}" in
  -h|--help) usage 0 ;;
esac

FAST=0
if [ "${1:-}" = "--fast" ]; then FAST=1; shift; fi

# --archive-only: snapshot the CURRENT loose outputs under a label, running nothing.
#
# This exists because a run can complete its whole numeric phase and then fail before
# archiving, which is precisely the Chapter 5 `run_all.py` defect this project already
# fixed once: results written after the figures, a crash in a figure, and every
# invocation silently discarding an hour of correct computation. The same shape bit this
# script on 2026-08-03 -- thirteen generators finished green, including a 1380 s GPU
# sweep, and the archive step died of a syntax error because the file had been edited
# WHILE bash was reading it (bash reads a script incrementally, so an insert shifts every
# later byte offset out from under it).
#
# Re-running 48 minutes of compute to recover a `cp` and a heredoc is the wrong trade.
# Archiving outputs this pass did not produce is a provenance risk, so the archive says
# so, loudly, in the place a reader looks: PROVENANCE.txt records that the generators
# were not run here and that the SHAs are the CURRENT ones rather than necessarily the
# ones that produced the tables. Check them before citing.
ARCHIVE_ONLY=0
if [ "${1:-}" = "--archive-only" ]; then ARCHIVE_ONLY=1; shift; fi

case "${1:-}" in
  -*) echo "error: unknown option '$1'" >&2; usage 2 ;;
esac

LABEL="${1:?usage: run_all_tables.sh [--fast] <label> [table_name ...]}"
shift
ONLY=("$@")          # optional: run just these tables, e.g. to backfill one

# Seeds used for slow tables under --fast. Everything else keeps common.SEEDS:
# the cheap tables gain nothing from a reduction, so they should not pay the
# credibility cost of one.
FAST_SEEDS="${REPRO_FAST_SEEDS:-0,1,2}"

# Every generator writes ± into its cells and several write λ / Δ / σ into their
# headers. Python picks its stdout AND default file encoding from the platform,
# which on Windows is cp1252 -- so a table would finish its whole numeric phase
# and then die in the f.write that emits it. The emitters now pin UTF-8
# explicitly; this covers the generators' own stdout for the same reason.
export PYTHONIOENCODING=utf-8

# Unbuffer the generators' stdout. Python buffers when stdout is a file rather than a
# terminal, and every generator here is run with `>"$log"` -- so a long one shows an
# EMPTY log for its whole duration and then dumps everything at exit. That is bad in
# two ways this project already has scars from. Watching a 25-minute GPU sweep, an
# empty log is indistinguishable from a hung process: on the run that prompted this
# line the only way to tell it was alive was to notice a python.exe holding 12.9 GB.
# And if a generator is killed or dies hard, the buffer goes with it -- so the log of
# the run that failed is blank, which is the exact opposite of what a log is for.
export PYTHONUNBUFFERED=1

# Windows hosts need a C toolchain to build the three compiled submodules, and
# this one lost its MSVC. hostenv.sh is a no-op everywhere else; sourced here so
# the suite bootstraps itself rather than depending on the operator having
# remembered to. It exports CC/DIST_EXTRA_CONFIG/UV_NO_EDITABLE, so it must come
# before the first uv invocation. See its header for the three defects involved.
_hostenv="$(dirname "${BASH_SOURCE[0]}")/hostenv.sh"
if [ -f "$_hostenv" ]; then
  # shellcheck source=/dev/null
  source "$_hostenv" || { echo "error: hostenv.sh failed; no usable compiler" >&2; exit 1; }
fi

# Table 4.4b, the theta operating curve, is emitted by table_4_4_openset ONLY when
# this is set -- so every sweep so far produced 4.4 and silently omitted 4.4b,
# while 4.4b sat in the archives from a hand-run. Defaulted here so the curve is
# part of the sweep rather than a thing someone remembers to do.
#
# It is a LIST of thetas, not a flag. `REPRO_THETA_SWEEP=1` is a valid list of one
# that emits a single row at theta=1.0, where the boost saturates and every cell is
# legitimately zero -- output indistinguishable from a null result. Two sweeps were
# lost to that reading.
export REPRO_THETA_SWEEP="${REPRO_THETA_SWEEP:-0.5,0.6,0.7,0.8,0.9,0.99,1.1}"

# Runtimes at 10 seeds from outputs/seeds10-2026-08-01/: these four are 1301s,
# 302s, 187s and 112s; the remaining seven total under two minutes combined.
declare -A SLOW_TABLES=(
  [table_concrete_reconciliation]=1
  [table_3_1_pvat_scaling]=1
  [table_norm_conorm_matrix]=1
  [table_hyperparam_normalization]=1
  # Added 2026-08-22. The list above was sized from outputs/seeds10-2026-08-01/,
  # which predates RT-IOT2022 landing (2026-08-12). Since then the two tables
  # that actually dominate the suite are these, and neither was in the list --
  # so `--fast` ran them at the full ten seeds and stopped bounding the runtime
  # it exists to bound. Measured on this host: 530 s and 13,084 s.
  [table_4_1_mog_baselines]=1
  [table_4_4_openset]=1
  # The GPU table sweeps four N for the MST/front end, three for FCM and eight
  # (dimension x precision) for distances, with a CPU arm beside every one, so ten
  # seeds is ~25 min on its own -- the slowest single table in the suite.
  [table_3_4_gpu_speedups]=1
  # Added with the ROOT_TABLES wiring below. 146 s measured on this host at ten
  # seeds -- slower than table_hyperparam_normalization (112 s), the cheapest
  # entry above -- so by this list's own threshold it belongs here. It honours the
  # reduction: table_5_4_ch5_g1_scaling.py reads `C.SEEDS`, which common.py builds
  # from REPRO_SEEDS, the variable run_one exports. Its sibling
  # table_5_1_3_ch5_tables is a pure renderer of an existing results.json and is
  # deliberately NOT here: it has no seeds of its own to reduce.
  [table_5_4_ch5_g1_scaling]=1
)
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/reproduce/outputs"
DEST="$OUT/$LABEL"

# ---------------------------------------------------------------------------
# One run at a time. Every generator writes to the SAME `reproduce/outputs/*.csv`
# names before the archive step moves them into $DEST, so two concurrent runs
# interleave their cells and archive whichever landed last -- silently, and with
# a PROVENANCE.txt that describes only one of them.
#
# This is not hypothetical. On 2026-08-27 two runs of this script raced for
# twelve minutes before anyone noticed, because on Windows killing the wrapper
# does not kill the generator: Git Bash's `ps` cannot see native python
# processes, so a live run looks dead, and the natural response -- relaunch --
# produces exactly the pair. `Get-CimInstance Win32_Process` is what shows them.
#
# `set -o noclobber` gives an atomic create-or-fail with no `flock` dependency
# (flock is absent from Git Bash). A stale lock from a killed run is detected by
# checking whether the recorded PID is still alive rather than by age.
# ---------------------------------------------------------------------------
LOCK="$OUT/.run_all_tables.lock"
mkdir -p "$OUT"
if ! (set -o noclobber; echo "$$" > "$LOCK") 2>/dev/null; then
  _holder="$(cat "$LOCK" 2>/dev/null || echo "?")"
  if kill -0 "$_holder" 2>/dev/null; then
    echo "error: another run_all_tables.sh is already running (pid $_holder)." >&2
    echo "       Two runs share reproduce/outputs/*.csv and would interleave." >&2
    echo "       Wait for it, or stop it and remove $LOCK." >&2
    exit 1
  fi
  echo "note: clearing a stale lock from pid $_holder (no such process)" >&2
  echo "$$" > "$LOCK"
fi
trap 'rm -f "$LOCK"' EXIT INT TERM

# Interpreter for the stdlib-only renderers, the seed-list query and the version
# line in PROVENANCE.txt. `python3` is not universally on PATH -- Windows/Git Bash
# ships the `py` launcher instead, and Windows also plants a `python` stub that
# prints a Store advert and exits nonzero. Probing with `-c ''` picks the first
# one that actually runs, so a host without `python3` no longer records its seed
# list as the literal string "unknown" (the same class of silent-provenance bug
# the hardcoded seed list was).
PY=""
for _cand in python3 python "py -3"; do
  if $_cand -c '' >/dev/null 2>&1; then PY="$_cand"; break; fi
done
if [ -z "$PY" ]; then
  echo "error: no working python interpreter (tried python3, python, py -3)" >&2
  exit 1
fi

# (project, script) pairs -- the project is the uv environment the script needs.
#
# table_a1_feature_scoring and table_3_2_memory_precision were absent from these
# lists while their outputs (table_a1_feature_ranking, table_a2_feature_count,
# table_3_2_memory_precision) sat in the run-of-record archive -- they had been
# run by hand. An archive containing a table the orchestrator does not produce is
# the same trap as a hardcoded seed list: the next full sweep silently carries the
# older table forward and reports eleven-of-eleven green.
FIS_TABLES=(
  table_concrete_reconciliation
  table_hyperparam_normalization
  table_g5_output_partitioning
  table_g5b_skew_sweep
  table_4_1_mog_baselines
  table_6_1_model_family
  table_norm_conorm_matrix
  table_4_4_openset
  table_a1_feature_scoring
  table_4_8_mf_dedup
)
CLUSTER_TABLES=(
  table_3_1_pvat_scaling
  table_3_1_reorder_three_arm
  table_3_2_memory_precision
  table_3_4_gpu_speedups
  table_3_7_g2_dtw_nonmetric
  # Added 2026-08-22. Its output sat in every archive with NO LOG beside it,
  # because it had only ever been hand-run -- the same trap B12 records for
  # table_a1_feature_scoring and table_3_2_memory_precision, in a third place.
  # It matters more than those two: this table is the evidence for Chapter 3's
  # Goal G2 downstream claim and for §5.4's corrected coordinate-free claim, so a
  # sweep reporting green while silently carrying it forward asserts a result
  # nobody re-measured.
  table_3_7_g2_downstream
)

# Per-table dependency overrides, for a table needing something the group's
# EXTRA_DEPS does not carry. table_3_4_gpu_speedups needs CuPy, which is an
# OPTIONAL extra of tribble-cluster (`[gpu]`) and must not be forced on the other
# cluster tables -- they run fine without a device, and on a host with no CUDA
# wheel available the resolution failure would take them down with it.
#
# The GPU table is written to survive the absence of a device: it emits N/A cells
# naming the blocker. That graceful path is only reachable if the interpreter
# starts at all, so if the CuPy-bearing invocation fails to resolve, run_one
# retries WITHOUT the GPU dep -- a table that says "no CUDA runtime, here is why"
# is a real deliverable, and a resolution error that emits nothing is not.
declare -A TABLE_DEPS=(
  [table_3_4_gpu_speedups]="--with scipy --with cupy-cuda12x"
  # Goal G2: real non-coordinate (DTW) data via aeon. `uv pip install` does not
  # persist here -- `uv run --project` re-syncs from the lockfile and drops it --
  # so, like the CuPy row above, the dependency has to ride on the invocation.
  # This table also downloads UCR/UEA archives on first run (network + disk).
  [table_3_7_g2_dtw_nonmetric]="--with aeon"
  # Same DTW datasets, same dependency.
  [table_3_7_g2_downstream]="--with aeon"
)
declare -A TABLE_DEPS_FALLBACK=(
  [table_3_4_gpu_speedups]="--with scipy"
)
# Stdlib-only renderers -- no submodule environment, so they run on the host python.
# table_5_1_3_ch5_tables replaced table_5_x_ch5_selection when the Chapter 5 script
# was split (into it and table_5_4_ch5_g1_scaling); the old name lingered here and
# every sweep since FAILED on a file that no longer exists while the split's real
# outputs were carried forward stale from a hand-run -- the exact "archive contains
# a table the orchestrator did not produce" trap this file warns about elsewhere.
PLAIN_TABLES=(
  table_5_1_3_ch5_tables
)

# Root-.venv tables -- repo="." in manifest.py: they do their own computation and
# need numpy/sklearn plus modules imported from gated-minimax-selection/ on sys.path,
# but no submodule package. `run_one .` runs them under `uv run --project <root>`,
# so they get the synced root environment rather than the bare host python (which on
# this Windows host is the Store stub and has no numpy).
#
# Only table_5_4 is here. manifest.py's third repo="." entry,
# table-5-5-7-ch5-hdbscan-baselines, is left out ON PURPOSE: it needs the `hdbscan`
# contrib package, which is in no repo venv (its manifest note says to install it
# into a throwaway venv), and it feeds Appendix A.8 rather than a numbered table.
# Wiring it in would fail every sweep on an import this environment cannot satisfy.
ROOT_TABLES=(
  table_5_4_ch5_g1_scaling
)

# Every listed generator must EXIST before the run starts.
#
# This check is here because of the bug the PLAIN_TABLES/ROOT_TABLES wiring above
# fixes: the Chapter 5 script was split in two, this file kept naming the removed
# `table_5_x_ch5_selection`, and every sweep from then on carried a FAILED table
# whose replacements never ran -- their stale outputs swept into each archive and
# presented as part of the current run. Nothing caught it, because a missing script
# is only discovered when the loop reaches it: for a mid-list FIS table that is
# hours into a sweep, and the evidence is one `FAILED` line in a status block long
# enough to scroll past. See outputs/bump-ae0ef13-2026-08-30/PROVENANCE.txt.
#
# Checking up front costs milliseconds and turns a silent stale-carry-forward into
# an abort before any compute is spent.
_missing=()
for _t in "${FIS_TABLES[@]}" "${CLUSTER_TABLES[@]}" "${PLAIN_TABLES[@]}" "${ROOT_TABLES[@]}"; do
  [ -f "$ROOT/reproduce/tables/$_t.py" ] || _missing+=("$_t")
done
if [ ${#_missing[@]} -gt 0 ]; then
  echo "error: these tables are listed in this script but have no generator on disk:" >&2
  printf '         reproduce/tables/%s.py
' "${_missing[@]}" >&2
  echo "       A renamed or split generator leaves the old name behind; the sweep" >&2
  echo "       then FAILS on it and carries the real outputs forward stale." >&2
  echo "       Fix the list (or restore the file) before running." >&2
  exit 1
fi
unset _missing

# Same guard for a filtered run. `run_all_tables.sh label tabel_4_8_mf_dedup` used
# to run zero generators, archive whatever was already loose in outputs/, and exit
# 0 -- indistinguishable from a successful backfill.
if [ ${#ONLY[@]} -gt 0 ]; then
  _unknown=()
  for _t in "${ONLY[@]}"; do
    case " ${FIS_TABLES[*]} ${CLUSTER_TABLES[*]} ${PLAIN_TABLES[*]} ${ROOT_TABLES[*]} " in
      *" $_t "*) ;;
      *) _unknown+=("$_t") ;;
    esac
  done
  if [ ${#_unknown[@]} -gt 0 ]; then
    echo "error: not a table this script runs: ${_unknown[*]}" >&2
    echo "       A filtered run that matches nothing produces no numbers and still" >&2
    echo "       archives the loose outputs, which reads as a successful backfill." >&2
    exit 1
  fi
  unset _unknown
fi
unset _t

# Only now, with the table lists validated, is the archive directory created --
# a run that aborts on a bad list must not leave an empty archive behind for the
# next reader to find.
mkdir -p "$DEST/logs"

declare -A STATUS
declare -A SEEDS_USED
# N/A cells emitted per table. A generator that reports what it cannot run is
# working as designed -- but `table_norm_conorm_matrix` emitted a THIRD of its
# cells as N/A for weeks, from a class rename it caught and logged, while this
# script printed a bare "ok" and the run went green. The count belongs next to
# the status, so "ran" and "produced a table" stop being the same claim.
declare -A NA_CELLS

# Resolved once, from common.py, so the provenance file reports the seed list the
# generators will actually use rather than a default repeated here. This line
# used to be hardcoded, and when the default moved from five seeds to ten it
# silently kept recording five.
FULL_SEEDS="$(SEEDLIST_ROOT="$ROOT" $PY -c 'import os,sys;
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

  # A table listed in TABLE_DEPS overrides its group's EXTRA_DEPS.
  local deps="${TABLE_DEPS[$name]:-${EXTRA_DEPS:-}}"
  if [ "$project" = "-" ]; then
    "${env_prefix[@]}" $PY "$ROOT/reproduce/tables/$name.py" >"$log" 2>&1
  else
    "${env_prefix[@]}" uv run --project "$ROOT/$project" $deps python \
        "$ROOT/reproduce/tables/$name.py" >"$log" 2>&1
  fi
  rc=$?
  # An optional dependency that cannot be resolved (no CUDA wheel for this
  # platform, no network) must not cost the table entirely: the GPU generator
  # reports its own blocker as N/A cells when the import is missing, so retry
  # once on the reduced dependency set. Both attempts stay in the one log.
  if [ $rc -ne 0 ] && [ -n "${TABLE_DEPS_FALLBACK[$name]:-}" ]; then
    echo "--- retrying without the optional GPU dependency ---" >>"$log"
    "${env_prefix[@]}" uv run --project "$ROOT/$project" \
        ${TABLE_DEPS_FALLBACK[$name]} python \
        "$ROOT/reproduce/tables/$name.py" >>"$log" 2>&1
    rc=$?
  fi
  t1=$(date +%s)
  after=$(snapshot_tables)
  if [ $rc -ne 0 ]; then
    STATUS[$name]="FAILED"
  elif [ "$before" = "$after" ]; then
    STATUS[$name]="no-output"
  else
    STATUS[$name]=ok
  fi
  # Count N/A cells in the CSVs this table just wrote. The changed-file set
  # comes from the same before/after snapshot the no-output check uses, so it
  # needs no timestamp window -- consecutive tables finish inside the same
  # second, and any grace period wide enough to catch that would attribute one
  # table's output to the next.
  local na=0 f
  while read -r f; do
    [ -n "$f" ] && [ "${f##*.}" = "csv" ] || continue
    na=$((na + $(grep -o 'N/A' "$f" 2>/dev/null | wc -l)))
  done < <(comm -13 <(printf '%s\n' "$before") <(printf '%s\n' "$after") | awk '{print $1}')
  NA_CELLS[$name]=$na

  local na_note=""
  [ "$na" -gt 0 ] && na_note="  ($na N/A)"
  printf '%-10s %4ss%s\n' "${STATUS[$name]}" "$((t1 - t0))" "$na_note"
}

# Check for submodule SHA divergence before running. This call used to sit above
# the function's own definition, where bash resolves it as an unknown command --
# `set -e` is off, so the run continued and the guard never once fired.
check_submodule_shas

# Check the upstream contracts the generators silently assume, BEFORE spending
# hours producing numbers under a broken one. Every defect the 2026-08-22 pass
# found had the same shape: an upstream function kept its name and changed its
# behaviour, the generator emitted N/A or a plausible wrong number, exited 0,
# and this script printed "ok".
#
# A failure does NOT abort the run. It stamps the archive NOT CITABLE, the same
# treatment --fast gets and for the same reason: the risk is someone quoting the
# output a month from now, not someone producing it today. Skips are not
# failures -- but an unavailable import is reported as a skip, never as a pass.
PREFLIGHT_STATUS="not run"
run_preflight() {
  [ $ARCHIVE_ONLY -eq 1 ] && return 0
  local out="$DEST/preflight.txt" rc=0
  {
    for suite in fis cluster; do
      local proj="tribble-fis"
      [ "$suite" = "cluster" ] && proj="tribble-cluster"
      uv run --project "$ROOT/$proj" python "$ROOT/reproduce/preflight.py"           --suite "$suite" 2>&1 || rc=1
      echo
    done
  } >"$out"
  if [ $rc -ne 0 ]; then
    PREFLIGHT_STATUS="FAILED -- see preflight.txt"
    echo
    echo "########################################################################"
    echo "## PREFLIGHT FAILED -- AN UPSTREAM INVARIANT IS BROKEN                ##"
    echo "##                                                                    ##"
    grep -E "^\s+\[FAIL\]" "$out" | sed 's/^ *//' | cut -c1-66       | while IFS= read -r _l; do printf '## %-68s ##
' "$_l"; done
    echo "##                                                                    ##"
    echo "## The run continues, and its archive is stamped NOT CITABLE.         ##"
    echo "########################################################################"
    echo
  else
    PREFLIGHT_STATUS="passed"
    echo "preflight: all invariants hold (detail in $(basename "$DEST")/preflight.txt)"
  fi
  return 0
}
run_preflight

echo "=== $LABEL ==="
if [ $ARCHIVE_ONLY -eq 1 ]; then
  echo "--archive-only: running no generators; snapshotting the loose outputs in"
  echo "                $(basename "$OUT")/ as they stand."
  for t in "${FIS_TABLES[@]}" "${CLUSTER_TABLES[@]}" "${PLAIN_TABLES[@]}" "${ROOT_TABLES[@]}"; do
    STATUS[$t]="archived-not-run"
    SEEDS_USED[$t]="$FULL_SEEDS (not verified -- see the notice in PROVENANCE.txt)"
  done
else
  echo "tribble-fis:"
  for t in "${FIS_TABLES[@]}"; do run_one tribble-fis "$t"; done
  echo "tribble-cluster:"
  EXTRA_DEPS="--with scipy"
  for t in "${CLUSTER_TABLES[@]}"; do run_one tribble-cluster "$t"; done
  EXTRA_DEPS=""
  echo "no submodule env:"
  for t in "${PLAIN_TABLES[@]}"; do run_one - "$t"; done
  echo "root .venv:"
  for t in "${ROOT_TABLES[@]}"; do run_one . "$t"; done
fi

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
  if [ $ARCHIVE_ONLY -eq 1 ]; then
    echo
    echo "########################################################################"
    echo "## ARCHIVE ONLY -- NO GENERATOR RAN IN THIS PASS                      ##"
    echo "##                                                                    ##"
    echo "## The tables here were copied from reproduce/outputs/ as they stood.  ##"
    echo "## They were produced by an EARLIER invocation. That invocation may or ##"
    echo "## may not have run at the SHAs recorded below -- this pass reads the  ##"
    echo "## SHAs as they are NOW, and cannot know what produced the files.      ##"
    echo "##                                                                    ##"
    echo "## Use this only to recover a run whose numeric phase completed and    ##"
    echo "## whose archive step failed. Confirm the SHAs and the logs/ contents  ##"
    echo "## match the run you think this is before quoting any number from it.  ##"
    echo "##                                                                    ##"
    echo "## THIS BANNER IS NOT A VERDICT. If an addendum below establishes what ##"
    echo "## produced these tables, read it: the archive may well be citable.    ##"
    echo "## Two independent readers stopped here and concluded it was not.      ##"
    echo "########################################################################"
    echo
  fi
  if [ "$PREFLIGHT_STATUS" != "passed" ] && [ $ARCHIVE_ONLY -eq 0 ]; then
    echo
    echo "########################################################################"
    echo "## PREFLIGHT $(printf '%-56s' "$PREFLIGHT_STATUS")##"
    echo "##                                                                    ##"
    echo "## An upstream contract this harness relies on does not hold, so the  ##"
    echo "## generators ran on a library that does not behave as they assume.   ##"
    echo "## Every number in this archive is suspect. See preflight.txt for     ##"
    echo "## which invariant broke and what it reaches.                         ##"
    echo "########################################################################"
    echo
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
  echo "tribble-opt: $(git -C "$ROOT/tribble-opt" rev-parse HEAD)"
  echo "grad-school: $(git -C "$ROOT" rev-parse HEAD)"
  # The three lines above are the submodule CHECKOUTS. They are not necessarily
  # the code that ran. tribble-fis sources optimizers from a git URL, so
  # `uv run --project tribble-fis` resolves it from tribble-fis's own uv.lock
  # and never looks at the sibling checkout --
  # and for nine days in August 2026 those were five and eleven commits apart
  # respectively, across seeding and correctness fixes, while this file recorded
  # only the checkouts (checklist B18). tribble-clustering was the other half of
  # that pair until tribble-fis#233 moved it out of tribble-fis's dependencies
  # into an optional extra; it is no longer resolved in this environment, so the
  # probe below reports it "not installed" here and that is the correct record,
  # not a gap. The runs that use clustering supply the submodule explicitly
  # (`--with-editable tribble-cluster`), where the checkout SHA above is the
  # truth. So record what was actually imported,
  # read from the installed distribution's own direct_url.json. preflight's
  # PIN-MATCH fails the run when these disagree; this block is what lets a reader
  # check the claim years later instead of trusting that it was checked.
  echo "imported:"
  uv run --project "$ROOT/tribble-fis" python - <<'_PY' 2>/dev/null || echo "  (unavailable)"
import importlib.metadata as md, json
for dist in ("tribble-fis", "tribble-clustering", "optimizers"):
    try:
        raw = md.distribution(dist).read_text("direct_url.json") or "{}"
        info = json.loads(raw)
    except Exception:
        print(f"  {dist:20s} not installed")
        continue
    rev = (info.get("vcs_info") or {}).get("commit_id")
    src = rev if rev else f"dir {info.get('url', '?')}"
    print(f"  {dist:20s} {src}")
_PY
  echo "seeds:       $FULL_SEEDS${REPRO_SEEDS:+ (REPRO_SEEDS override in effect)}"
  echo "thetas:      $REPRO_THETA_SWEEP (table_4_4b operating curve)"
  echo
  # Machine identity. Recorded because timings are not portable between hosts OR
  # between runs on one host: the same generator, same tribble-cluster commit and
  # same ten seeds moved every wall-clock figure in Table 3.1 by ~45% across two
  # runs hours apart, while the speedup RATIO moved under 3%. Without this block
  # a future reader cannot tell a code change from a thermal one. The governor and
  # boost fields are here for the same reason -- they are the usual culprits.
  echo "machine:"
  printf '  %-16s %s\n' "host"     "$(hostname 2>/dev/null || echo unknown)"

  # OS detection: cross-platform
  _os=""
  if [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
    _os="$(sw_vers -productName 2>/dev/null) $(sw_vers -productVersion 2>/dev/null)"
  elif [ -f /etc/os-release ]; then
    _os="$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME" || uname -s)"
  else
    _os="$(uname -s)"
  fi
  printf '  %-16s %s\n' "os"       "$_os"
  printf '  %-16s %s\n' "kernel"   "$(uname -srm 2>/dev/null || echo 'n/a')"

  # CPU detection: cross-platform
  _cpu=""
  if [ -f /proc/cpuinfo ]; then
    _cpu="$(sed -n 's/^model name[[:space:]]*:[[:space:]]*//p' /proc/cpuinfo 2>/dev/null | head -1)"
  elif [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
    _cpu="$(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
  elif [ "$(uname -o 2>/dev/null)" = "Msys" ] || [ -n "$MSYSTEM" ]; then
    _cpu="$(wmic cpu get name /format:list 2>/dev/null | grep -i '^name=' | cut -d= -f2 | head -1)"
  fi
  printf '  %-16s %s\n' "cpu"      "${_cpu:-unknown}"

  # Core detection: cross-platform
  _logical=$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo "unknown")
  _physical=""
  if [ -f /proc/cpuinfo ]; then
    _cores_per=$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Core(s) per socket:[[:space:]]*//p')
    _sockets=$(LC_ALL=C lscpu 2>/dev/null | sed -n 's/^Socket(s):[[:space:]]*//p' | head -1)
    if [ -n "$_cores_per" ] && [ -n "$_sockets" ]; then
      _physical=$((${_cores_per} * ${_sockets}))
    fi
  elif [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
    _physical=$(sysctl -n hw.physicalcpu 2>/dev/null)
  elif [ -n "$MSYSTEM" ] || [ -n "$NUMBER_OF_PROCESSORS" ]; then
    _physical="${NUMBER_OF_PROCESSORS}"
  fi
  if [ -n "$_physical" ]; then
    printf '  %-16s %s\n' "cores"    "$_physical physical, $_logical logical"
  else
    printf '  %-16s %s\n' "cores"    "$_logical logical"
  fi

  # RAM detection: cross-platform
  _ram=""
  if [ -f /proc/meminfo ]; then
    _ram="$(awk '/MemTotal/ {printf "%.1f GiB (%.1f GB decimal)", $2/1048576, $2*1024/1e9}' /proc/meminfo 2>/dev/null)"
  elif [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
    _ram_bytes=$(sysctl -n hw.memsize 2>/dev/null)
    if [ -n "$_ram_bytes" ]; then
      _ram=$(awk "BEGIN {printf \"%.1f GiB (%.1f GB decimal)\", $_ram_bytes/1024/1048576, $_ram_bytes/1e9}")
    fi
  elif [ -n "$MSYSTEM" ]; then
    _ram="$(wmic computersystem get totalphysicalmemory /format:list 2>/dev/null | grep -i '^totalphysicalmemory=' | cut -d= -f2)"
    if [ -n "$_ram" ]; then
      _ram=$(awk "BEGIN {printf \"%.1f GiB (%.1f GB decimal)\", $_ram/1024/1048576, $_ram/1e9}")
    fi
  fi
  printf '  %-16s %s\n' "ram"      "${_ram:-unknown}"

  # Governor detection: Linux only
  _gov="n/a"
  if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    _gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo 'n/a')"
  fi
  printf '  %-16s %s\n' "governor" "$_gov"

  # Boost detection: Linux only
  _boost="n/a"
  if [ -f /sys/devices/system/cpu/cpufreq/boost ]; then
    _boost="$(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo 'n/a')"
  fi
  printf '  %-16s %s\n' "boost"    "$_boost"

  # GPU detection: cross-platform
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version \
               --format=csv,noheader 2>/dev/null \
      | while IFS= read -r g; do printf '  %-16s %s\n' "gpu" "$g"; done
  else
    _gpu=""
    if [ -f /proc/bus/pci/devices ] || command -v lspci >/dev/null 2>&1; then
      _gpu="$(lspci 2>/dev/null | grep -iE 'vga|3d controller' | sed 's/^[^ ]* //' | head -1)"
    elif [ "$(uname -s 2>/dev/null)" = "Darwin" ]; then
      _gpu="$(system_profiler SPDisplaysDataType 2>/dev/null | grep 'Chipset Model' | sed 's/.*Chipset Model: //' | head -1)"
    elif [ -n "$MSYSTEM" ]; then
      _gpu="$(wmic path win32_VideoController get name /format:list 2>/dev/null | grep -i '^name=' | cut -d= -f2 | head -1)"
    fi
    printf '  %-16s %s\n' "gpu"      "${_gpu:-none detected}"
  fi

  printf '  %-16s %s\n' "python"   "$($PY -V 2>&1)"
  # Numeric library versions, from the environment the tables actually ran in --
  # not the host python, which is a different interpreter with different wheels.
  #
  # Added because a difference turned out to be unattributable without them:
  # Table A.2's bhattacharyya accuracies sit up to +0.043 above the main-d0efefc
  # archive at identical code, seeds and feature rankings, while wasserstein and
  # composite agree to four decimals. Two full sweeps on this host reproduce every
  # one of those accuracies exactly, so it is not run-to-run noise -- it is the
  # environment, and no archive recorded enough to say which part of it. The
  # ill-conditioned arm is the one that moved, which is what a BLAS or threading
  # difference would look like.
  _env_probe='
import numpy, scipy, sklearn, sys
print("numpy", numpy.__version__, " scipy", scipy.__version__,
      " sklearn", sklearn.__version__, " python", sys.version.split()[0])
blas = "unknown"
try:
    d = numpy.show_config("dicts")
    b = d["Build Dependencies"]["blas"]
    blas = b.get("name", "?") + " " + b.get("version", "?")
except Exception:
    pass
print("blas", blas)
'
  uv run --project "$ROOT/tribble-fis" python -c "$_env_probe" 2>/dev/null \
    | while IFS= read -r _line; do
        printf '  %-16s %s\n' "${_line%% *}" "${_line#* }"
      done \
    || printf '  %-16s %s\n' "libs" "unavailable"
  echo
  echo "status:"
  for t in "${FIS_TABLES[@]}" "${CLUSTER_TABLES[@]}" "${PLAIN_TABLES[@]}" "${ROOT_TABLES[@]}"; do
    # Unset means the filter skipped it; say so rather than claiming a result.
    # The seed set is recorded PER TABLE because --fast makes it vary between
    # them, and a reader must be able to tell which cells are thin.
    _na="${NA_CELLS[$t]:-}"
    printf '  %-38s %-12s seeds=%s%s\n' \
      "$t" "${STATUS[$t]:-not-run-this-pass}" "${SEEDS_USED[$t]:-—}" \
      "${_na:+  N/A cells=$_na}"
  done
} >> "$PROV"

echo
cat "$DEST/PROVENANCE.txt"
