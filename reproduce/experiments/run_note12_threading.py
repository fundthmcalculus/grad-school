"""Note 12 -- does BLAS thread count move Table A.2's bhattacharyya column?

## The observation this exists to explain

`reproduce/tables/table_a1_feature_scoring.py` emits Tables A.1 and A.2. Against
the archive `reproduce/outputs/main-d0efefc/`, at the same `tribble-fis` commit,
the same ten seeds and a *byte-identical* A.1 feature ranking, every accuracy in
A.2's **bhattacharyya** column sits higher on this host: +0.017 at k=4, +0.033 at
5, +0.043 at 7, +0.040 at 10, +0.029 at 15, +0.030 at 20. The **wasserstein** and
**composite** columns agree to within 0.0002 everywhere.

It is not run-to-run noise: two independent full sweeps on this host
(`full-14900hx-2026-08-02`, `full-14900hx-r2`) reproduce all nine bhattacharyya
accuracies exactly, and only fit times move. So it is an environment difference,
and the standing guess was BLAS/threading -- attractive because the arm that moved
is the ill-conditioned one (bhattacharyya's own ranking scores 0.4267 accuracy at
one feature, so those models are fitted on poor features and sit where a
last-bits difference in a linear-algebra kernel can flip a decision).

## The experiment

Hold *everything* fixed -- code, commit, dataset, sample size, the ten seeds, the
scorer list -- and vary one thing: the number of threads the numeric stack is
allowed. `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` are set
together to each of {1, 2, 8, 32} (this host has 32 logical cores). Each setting
gets its own full generator run in its own subprocess, because these variables are
read when the BLAS loads and cannot be changed inside a live interpreter.

Each run writes a complete table set into its own directory, so nothing is
overwritten and every repetition survives for a later diff.

## Registered in advance: what each outcome means

Read the *accuracy* halves of the A.2 cells only. Fit seconds are expected to move
a lot and mean nothing here -- fewer threads is slower, that is the point of
threads.

* **CONFIRMS the hypothesis** -- bhattacharyya accuracies differ across thread
  counts by an amount on the order of the archive gap (some cell moving >= 0.01),
  while wasserstein and composite stay put to ~0.0002. That is the signature
  claimed: thread count perturbs the ill-conditioned arm and nothing else.
* **REFUTES it** -- all three columns are identical, or differ by < 0.001, at every
  thread count. Thread count on this host then cannot produce a 0.043 shift, and
  the threading explanation is dead regardless of how plausible it looked. Say so.
* **A DIFFERENT AND MORE INTERESTING STORY** -- wasserstein and composite move too.
  They are the internal control; if the control moves, the claim "the harness is
  deterministic on one host with one numeric stack" needs qualifying, because
  thread count is part of a host's configuration and is not currently pinned.

A.1's ranking is a second control: it is computed once per run from the same data
and must come back identical at every thread count. If the *ranking* moves, the
A.2 difference is downstream of a changed input and the whole framing shifts.

A negative result here is a real result. It does not rescue the threading guess by
elimination; it removes it.

## The second axis: which BLAS kernel, not how many threads

Thread count only changes how the same arithmetic is scheduled, and OpenBLAS's
small-matrix paths are often single-threaded regardless, so it is the weaker of
the two things "a BLAS difference" could mean. The stronger one is *which kernel
family runs at all*.

`threadpool_info()` on this host reports the loaded OpenBLAS as a DYNAMIC_ARCH
build that selected `architecture: 'Haswell'` from CPUID. `OPENBLAS_CORETYPE`
overrides that selection, and different families use different vector widths and
different accumulation orders — so this varies the *arithmetic*, which is what
would actually move a last-bits-sensitive result.

This matters for note 12 specifically. `main-d0efefc` recorded no machine block
at all, but Chapter 3 says the pre-workstation numbers came from an i7-1185G7 —
Tiger Lake, which **has** AVX-512, where OpenBLAS would select `SkylakeX` rather
than `Haswell`. A different kernel family on a different CPU is a difference no
amount of thread pinning can reproduce, and unlike a library version it can be
tested on one host by asking for a different family. Run it with

    uv run --project tribble-fis python reproduce/experiments/run_note12_threading.py \
        --vary coretype

Registered outcomes for this axis are the same three as above, with one addition:
if bhattacharyya moves under `--vary coretype` while wasserstein and composite do
not, note 12's mechanism is **identified**, and the practical guidance follows
directly — the cell is a function of the BLAS kernel and cannot be quoted to four
decimals across CPUs. Only sweep *downward* from the native family; naming a
family the CPU cannot execute faults rather than falls back, so `SkylakeX` cannot
be tested from Raptor Lake and the AVX-512 direction stays unmeasured here.

## Usage, from the repo root

    export PYTHONIOENCODING=utf-8
    uv run --project tribble-fis python reproduce/experiments/run_note12_threading.py
    uv run --project tribble-fis python reproduce/experiments/run_note12_threading.py \
        --vary coretype

    # a fast smoke test of the plumbing -- NOT a protocol result, and the script
    # stamps every output as reduced when you do this
    REPRO_SEEDS=0 uv run --project tribble-fis \
        python reproduce/experiments/run_note12_threading.py --threads 1,32

Options:
    --vary threads|coretype   which single variable to sweep (default threads)
    --threads 1,2,8,32        thread counts to sweep
    --coretypes Haswell,...   OPENBLAS_CORETYPE values, native family first
    --out DIR                 destination (default reproduce/outputs/note12-threading)
    --skip-existing           reuse a setting's run if its CSV is already there
    --compare-only            re-do the comparison from CSVs already on disk

Nothing is written inside any submodule. `PYTHONIOENCODING=utf-8` is forced into
the child environment as well as expected in the parent: the generator's Markdown
carries non-cp1252 characters and the platform default here cannot encode them.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "reproduce" / "tables" / "table_a1_feature_scoring.py"
DEFAULT_OUT = ROOT / "reproduce" / "outputs" / "note12-threading"
DEFAULT_THREADS = [1, 2, 8, 32]

# The archive the note is a complaint about. Its bhattacharyya accuracies are the
# thing we are trying to reproduce a shift of.
ARCHIVE = ROOT / "reproduce" / "outputs" / "main-d0efefc" / "table_a2_feature_count.csv"

# Every variable that gates thread count in a stack anyone here might be running.
# Set together so the result does not depend on which BLAS the wheel happened to
# vendor: OpenBLAS reads OPENBLAS_NUM_THREADS then OMP_NUM_THREADS, MKL reads
# MKL_NUM_THREADS, and sklearn's own loops read OMP_NUM_THREADS.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# The second axis, and the better one -- see the docstring's coretype section.
# OpenBLAS here is a DYNAMIC_ARCH build that picks a kernel family from CPUID at
# load time (`threadpool_info()` reports `architecture` on this host as
# 'Haswell'), and `OPENBLAS_CORETYPE` overrides that choice. Different kernel
# families accumulate dot products in a different order, so this varies the
# arithmetic itself rather than only how it is scheduled.
#
# Only DOWNWARD from the host's native family. Asking a CPU for a kernel using
# instructions it does not have -- SkylakeX/AVX-512 on Raptor Lake, say -- does
# not fall back, it faults.
CORETYPES = ["Haswell", "Sandybridge", "Nehalem", "Prescott"]

# Accuracies are quoted to four decimals, so anything at or under half a unit in
# the last place is the same number printed. Above it the cells genuinely differ.
TIED = 0.00005
# The smallest move that would be interesting -- an order below the 0.043 the note
# reports, so a real-but-smaller thread effect still counts as confirmation.
MATERIAL = 0.001
# How much the manipulation check must move before an invariance result counts as
# a refutation. Below this the variable was loaded but did nothing this workload
# can feel, and "no effect on accuracy" is then uninformative rather than
# negative. 5% is comfortably outside the run-to-run timing noise measured here
# (the two full workstation sweeps agreed on accuracy exactly while their fit
# times moved a few percent) and far below the 140% the thread axis produces.
MANIPULATION_FLOOR = 0.05


def parse_a2(path: Path) -> dict[str, dict[int, tuple[float, float]]]:
    """{scorer: {k: (accuracy, fit_seconds)}} from an emitted A.2 CSV.

    Cells are "0.9967 / 0.55". The header names columns "<scorer> (acc / fit s)".
    """
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"{path} is empty")
    header = rows[0]
    scorers = [h.split(" (")[0] for h in header[1:]]
    out: dict[str, dict[int, tuple[float, float]]] = {s: {} for s in scorers}
    for row in rows[1:]:
        if not row or not row[0].strip().isdigit():
            continue
        k = int(row[0])
        for scorer, cellstr in zip(scorers, row[1:]):
            if "/" not in cellstr:
                continue          # N/A column
            acc, secs = cellstr.split("/")
            out[scorer][k] = (float(acc.strip()), float(secs.strip()))
    return out


def run_one(setting, out_dir: Path, skip_existing: bool, axis: str = "threads") -> Path:
    """One full generator run with a single environment variable changed.

    `axis="threads"` pins every thread-count variable to `setting`; `axis="core"`
    pins `OPENBLAS_CORETYPE` to it. One knob per run, everything else inherited.
    """
    if axis == "threads":
        dest = out_dir / f"threads-{int(setting):02d}"
    else:
        dest = out_dir / f"core-{setting}"
    csv_path = dest / "table_a2_feature_count.csv"
    if skip_existing and csv_path.is_file():
        print(f"[{axis}={setting}] reusing {csv_path}")
        return csv_path

    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if axis == "threads":
        for var in THREAD_VARS:
            env[var] = str(setting)
    else:
        env["OPENBLAS_CORETYPE"] = str(setting)
    env["REPRO_OUTPUT_DIR"] = str(dest)
    # Not optional. The generator's Markdown carries characters cp1252 cannot
    # encode and the crash lands in f.write, after the whole numeric phase.
    env["PYTHONIOENCODING"] = "utf-8"

    print(f"[{axis}={setting}] running {GENERATOR.name} -> {dest}")
    log = dest / "run.log"
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.run(
            [sys.executable, str(GENERATOR)],
            cwd=str(ROOT), env=env, stdout=lf,
            stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        print(f"[{axis}={setting}] FAILED rc={proc.returncode}; see {log}")
        return csv_path

    # Exit 0 is not evidence. A generator can complete and write nothing.
    if not csv_path.is_file():
        print(f"[{axis}={setting}] exited 0 but wrote no CSV; see {log}")
    return csv_path


def compare(results: dict, axis: str = "threads") -> tuple[int, bool]:
    """Diff the accuracy cells across settings of one axis.

    Returns (moved, manipulated): moved is 0 if all tied, 1 if at least one
    column moved, or the sentinel 2 if fewer than two runs were usable to
    compare at all. manipulated is False in the sentinel case.
    """
    parsed: dict = {}
    for setting, path in results.items():
        if path.is_file():
            parsed[setting] = parse_a2(path)
        else:
            print(f"  [missing] {axis}={setting}: {path}")
    if len(parsed) < 2:
        print("  fewer than two usable runs; nothing to compare")
        return 2, False

    counts = list(parsed)
    reference = counts[0]
    scorers = [s for s in parsed[reference] if parsed[reference][s]]
    moved_any = False
    label = "thread count" if axis == "threads" else "OPENBLAS_CORETYPE"

    for scorer in scorers:
        ks = sorted(parsed[reference][scorer])
        print(f"\n  {scorer}: accuracy vs {label} "
              f"(reference = {reference})")
        print("    k    " + "".join(f"{str(t):>14}" for t in counts)
              + f"{'max |Δ|':>12}")
        col_max = 0.0
        for k in ks:
            accs = [parsed[t][scorer].get(k, (float('nan'), 0))[0] for t in counts]
            base = accs[0]
            delta = max(abs(a - base) for a in accs)
            col_max = max(col_max, delta)
            print(f"    {k:<5}" + "".join(f"{a:>14.4f}" for a in accs)
                  + f"{delta:>12.4f}")
        verdict = ("identical" if col_max <= TIED else
                   "moved (below the 0.001 materiality floor)"
                   if col_max < MATERIAL else "MOVED")
        print(f"    max |Δ| over the whole column: {col_max:.6f}  -> {verdict}")
        if col_max >= MATERIAL:
            moved_any = True

    # An invariance result is worthless unless the variable was actually varied.
    # `OMP_NUM_THREADS` is trivially easy to set into an environment that ignores
    # it -- a statically single-threaded build reads it and shrugs -- and that
    # failure looks exactly like the finding. Total fit seconds per run is the
    # falsifier: if it does not respond to the setting, the knob never bit and
    # the accuracy invariance says nothing.
    print("\n  Manipulation check -- total A.2 fit seconds per run "
          "(sum over scorers and k, mean per seed):")
    totals = {}
    for t in counts:
        totals[t] = sum(secs for sc in parsed[t].values()
                        for _, secs in sc.values())
        print(f"    {str(t):<14} {totals[t]:8.2f} s")
    lo, hi = min(totals.values()), max(totals.values())
    spread = (hi - lo) / lo if lo else 0.0
    print(f"    spread {spread * 100:.1f}%  "
          f"({'PASS' if spread >= MANIPULATION_FLOOR else 'FAIL'}: needs "
          f">= {MANIPULATION_FLOOR * 100:.0f}% for the invariance to mean "
          f"anything)")
    if spread < MANIPULATION_FLOOR:
        print(f"    {label} loaded but did not measurably change execution, so"
              "\n    an unchanged accuracy is WEAK evidence rather than a"
              "\n    refutation -- the workload may simply not exercise what this"
              "\n    variable controls.")

    return (1 if moved_any else 0), spread >= MANIPULATION_FLOOR


def compare_to_archive(results: dict, axis: str = "threads") -> None:
    """Show each setting against the archive the note is about."""
    if not ARCHIVE.is_file():
        print(f"\n  [skip] archive not found: {ARCHIVE}")
        return
    arch = parse_a2(ARCHIVE)
    print(f"\n  Against the archive ({ARCHIVE.parent.name}): "
          f"max |Δ accuracy| per column")
    print(f"    {axis:<14}" + "".join(f"{s:>18}" for s in arch))
    for setting, path in results.items():
        if not path.is_file():
            continue
        here = parse_a2(path)
        cells = []
        for scorer in arch:
            if scorer not in here or not here[scorer]:
                cells.append("n/a")
                continue
            d = max(abs(here[scorer][k][0] - arch[scorer][k][0])
                    for k in arch[scorer] if k in here[scorer])
            cells.append(f"{d:.4f}")
        print(f"    {str(setting):<14}" + "".join(f"{c:>18}" for c in cells))


def blas_report() -> None:
    """Print what BLAS actually loaded, and which kernel family it chose.

    Recorded per run because `OPENBLAS_CORETYPE` is a request, not a guarantee,
    and because two OpenBLAS copies are loaded here at once -- numpy's and
    scipy's -- at different versions. A note about "the BLAS" that does not say
    which one is the kind of claim this project keeps having to retract.
    """
    try:
        # Both imports matter, and in this order. `threadpool_info()` inspects
        # libraries that are ALREADY LOADED, so calling it before anything pulls
        # in a BLAS returns an empty list -- which reads as "no BLAS found"
        # instead of "you asked too early".
        import numpy  # noqa: F401
        import scipy.linalg  # noqa: F401
        import threadpoolctl
    except ImportError as exc:
        print(f"  [blas] cannot report the backend ({exc})")
        return
    for info in threadpoolctl.threadpool_info():
        print(f"  [blas] {info.get('internal_api')} "
              f"{info.get('version')} arch={info.get('architecture')} "
              f"threads={info.get('num_threads')} "
              f"prefix={info.get('prefix')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vary", choices=("threads", "coretype"), default="threads",
                    help="which single variable to sweep")
    ap.add_argument("--threads", default=",".join(str(t) for t in DEFAULT_THREADS),
                    help="comma-separated thread counts (default 1,2,8,32)")
    ap.add_argument("--coretypes", default=",".join(CORETYPES),
                    help="comma-separated OPENBLAS_CORETYPE values, native first. "
                         "Never name a family your CPU cannot execute.")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--compare-only", action="store_true",
                    help="do not run anything; compare CSVs already on disk")
    args = ap.parse_args()

    axis = "threads" if args.vary == "threads" else "core"
    if axis == "threads":
        settings = [int(t) for t in args.threads.split(",") if t.strip()]
        subdir = lambda s: f"threads-{int(s):02d}"   # noqa: E731
    else:
        settings = [c.strip() for c in args.coretypes.split(",") if c.strip()]
        subdir = lambda s: f"core-{s}"               # noqa: E731
    out_dir = Path(args.out).resolve()

    seeds = os.environ.get("REPRO_SEEDS", "0,1,2,3,4,5,6,7,8,9")
    reduced = [f"REPRO_SEEDS={seeds}"] if seeds != "0,1,2,3,4,5,6,7,8,9" else []
    for var in ("REPRO_SCORERS", "REPRO_PHIUSIIL_N"):
        if os.environ.get(var):
            reduced.append(f"{var}={os.environ[var]}")

    print(f"Note 12 -- {args.vary} vs Table A.2 accuracy")
    print(f"  generator : {GENERATOR}")
    print(f"  varying   : {args.vary} over {settings}")
    print(f"  seeds     : {seeds}")
    print(f"  out       : {out_dir}")
    blas_report()
    if reduced:
        # Loud, because a reduced run reads exactly like a protocol run once the
        # numbers are out of the shell and into a document.
        print("  *** REDUCED RUN -- NOT THE TEN-SEED PROTOCOL: "
              + "; ".join(reduced) + " ***")

    results: dict = {}
    for s in settings:
        dest = out_dir / subdir(s) / "table_a2_feature_count.csv"
        results[s] = dest if args.compare_only else run_one(
            s, out_dir, args.skip_existing, axis=axis)

    print(f"\n--- A.2 accuracy across {args.vary} settings ---")
    moved, manipulated = compare(results, axis=axis)

    # The ranking is the input control: it must not move.
    ranks = {}
    for s, path in results.items():
        rp = path.parent / "table_a1_feature_ranking.csv"
        if rp.is_file():
            ranks[s] = rp.read_text(encoding="utf-8")
    if len(ranks) >= 2:
        distinct = len(set(ranks.values()))
        print(f"\n  A.1 ranking (input control): "
              + ("identical at every setting" if distinct == 1
                 else f"*** {distinct} DISTINCT RANKINGS ***"))

    compare_to_archive(results, axis=axis)

    print("\n--- verdict ---")
    if moved == 2:
        print("  inconclusive: not enough usable runs")
    elif moved:
        print(f"  at least one accuracy column moved by >= {MATERIAL} with "
              f"{args.vary}.\n  Consistent with the note-12 numerical-environment"
              "\n  hypothesis -- check WHICH column moved before concluding it"
              "\n  (see the registered outcomes in this file's docstring).")
    elif not manipulated:
        print(f"  INCONCLUSIVE. Every accuracy column is invariant to {args.vary},"
              "\n  but the manipulation check failed: the variable loaded and"
              "\n  changed no measurable amount of work, so this workload does not"
              "\n  exercise what it controls and the invariance is not evidence"
              "\n  either way. Do not record this as a refutation.")
    else:
        print(f"  every A.2 accuracy column is invariant to {args.vary} on this"
              "\n  host, to the fourth decimal the table prints, while the"
              "\n  manipulation check confirms the variable really changed the"
              "\n  work done. That branch of note 12's hypothesis is REFUTED; it"
              "\n  does not become more likely by elimination, it is removed.")
    if reduced:
        print("  *** reduced run; do not quote these as the ten-seed result ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
