#!/usr/bin/env python3
"""Re-measure the matrix-free reorder, `pvat.vat_prim_mst_seq`.

Chapter 3 §3.3.2, Table 3.3's two `on-demand -- defective` rows, and Appendix A.6
all rest on one measurement: that this function "returns the seed vertex followed
by every other vertex in ascending index order", agreeing with the true ordering
at chance level (0.001 +/- 0.001). A.6 additionally states it "is removed from
the public API at the commit Chapter 3 pins (tribble-cluster e3c27e6)".

Three of those are false at the currently pinned SHA. Upstream clustering #70
(c9be437) repaired the distance computation -- `_get_dist` is now typed for two
scalar indices rather than one scalar and one array -- and added
tests/test_vat_prim_mst_seq.py. The function is still exported.

This script registers its outcomes before it runs, in the manner
`run_note12_threading.py` established:

  PASS-CORRECT   elementwise agreement with `compute_vat` is 1.000 at every size
                 and seed. Table 3.3's ordering column for the on-demand rows
                 becomes 1.000 (exact) rather than 0.001 +/- 0.001.
  PASS-MEMORY    peak working set grows like O(N), not O(N^2): the matrix-free
                 arm's peak stays far below the materialized arm's at a size
                 where the matrix is gigabytes. This is the claim §3.3.2 calls
                 "the idea that is *not* built".
  FAIL-CORRECT   agreement is still at chance. Nothing in the prose changes.
  FAIL-MEMORY    peak tracks the matrix arm, i.e. it materializes after all.

Memory is measured per-arm in a FRESH SUBPROCESS via the Win32
GetProcessMemoryInfo PeakWorkingSetSize (falling back to resource.getrusage
elsewhere), because the compiled kernel allocates outside Python's allocator and
tracemalloc cannot see it -- the same reason a "the picture looks right" test
missed the in-place permutation defect §3.3.2 records.

    uv run --project tribble-cluster python reproduce/experiments/check_matrix_free_reorder.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np


def peak_bytes() -> int:
    """Peak working set of THIS process, in bytes."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # The export lives in psapi.dll on some Windows builds and is
        # re-exported from kernel32 as K32GetProcessMemoryInfo on others.
        # Calling the missing one returns 0 silently, which reads as "this arm
        # used no memory" -- the failure mode this whole script exists to avoid.
        fn = None
        for dll, name in (
            (ctypes.windll.psapi, "GetProcessMemoryInfo"),
            (ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
        ):
            try:
                fn = getattr(dll, name)
                break
            except AttributeError:
                continue
        if fn is None:
            raise RuntimeError("no GetProcessMemoryInfo export found")
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
        fn.restype = wintypes.BOOL

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        handle = ctypes.c_void_p(handle)
        if not fn(handle, ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def make_points(n: int, d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(n, d)))


def child_measure(arm: str, n: int, d: int, seed: int) -> None:
    """Run ONE arm in this (fresh) process and print its peak as JSON."""
    from tribbleclustering import pvat

    x = make_points(n, d, seed)

    if arm.startswith("time-"):
        import time

        which = arm[len("time-") :]
        # One discarded warm-up fit. The Cython kernels JIT/page-in on first
        # call, and PROVENANCE_MAP note 14 records a 3.7x first-seed inflation
        # from exactly this cause -- a cold timing here would be fiction.
        for rep in range(2):
            t0 = time.perf_counter()
            if which == "matrix-free":
                pvat.vat_prim_mst_seq(x)
            else:
                gram = x @ x.T
                sq = np.einsum("ij,ij->i", x, x)
                dm = sq[:, None] + sq[None, :] - 2.0 * gram
                np.maximum(dm, 0.0, out=dm)
                np.sqrt(dm, out=dm)
                del gram, sq
                pvat.compute_vat(dm, inplace=True)
            secs = time.perf_counter() - t0
        print(json.dumps({"arm": arm, "n": n, "secs": secs}))
        return

    base = peak_bytes()
    if arm == "matrix-free":
        pvat.vat_prim_mst_seq(x)
    elif arm == "materialized":
        # What the in-place scheme of §3.3.2 does: build D, then reorder it in
        # place. D is built by the gram identity rather than by broadcasting --
        # `(x[:,None,:] - x[None,:,:])**2` would allocate an (n,n,d) temporary
        # and charge this arm d times the matrix it is supposed to represent,
        # which is E2b's "different formulations, not different hardware" error
        # wearing a different hat. Peak here should be ~one n*n matrix.
        gram = x @ x.T
        sq = np.einsum("ij,ij->i", x, x)
        dm = sq[:, None] + sq[None, :] - 2.0 * gram
        np.maximum(dm, 0.0, out=dm)
        np.sqrt(dm, out=dm)
        del gram, sq
        pvat.compute_vat(dm, inplace=True)
    else:  # pragma: no cover - guarded by the caller
        raise SystemExit(f"unknown arm {arm}")
    print(json.dumps({"arm": arm, "n": n, "peak": peak_bytes(), "base": base}))


def run_child(arm: str, n: int, d: int, seed: int) -> dict | None:
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--child", arm, str(n), str(d), str(seed)],
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    sys.stderr.write(f"  [{arm} n={n}] child produced no result\n{proc.stderr[-800:]}\n")
    return None


def correctness(sizes, dims, seeds) -> list[tuple]:
    from tribbleclustering import pvat

    rows = []
    for n in sizes:
        agreements = []
        ascending = 0
        for s in seeds:
            x = make_points(n, dims, s)
            dm = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
            ref = pvat.compute_vat(dm)
            ref = np.asarray(ref[1] if isinstance(ref, tuple) else ref).ravel()
            got = np.asarray(pvat.vat_prim_mst_seq(x)).ravel()
            agreements.append(float((ref == got).mean()) if ref.shape == got.shape else 0.0)
            # The old defect's fingerprint: seed vertex, then ascending indices.
            if got.size > 2 and bool(np.all(np.diff(got[1:]) == 1)):
                ascending += 1
        arr = np.asarray(agreements, dtype=float)
        rows.append((n, arr.mean(), arr.std(), ascending, len(seeds), 1.0 / n))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=4, metavar=("ARM", "N", "D", "SEED"))
    ap.add_argument("--dims", type=int, default=4)
    args = ap.parse_args()

    if args.child:
        arm, n, d, seed = args.child
        child_measure(arm, int(n), int(d), int(seed))
        return 0

    # Ch 7 G4d's decision rule names these sizes and this seed count verbatim:
    # "verify the ordering elementwise against the serial reference at
    #  N in {1,000, 2,000, 5,000} across ten seeds".
    seeds = list(range(10))
    print("=" * 78)
    print("A. Correctness -- elementwise agreement with compute_vat")
    print("   (G4d decision rule, item 1: N in {1000, 2000, 5000}, ten seeds)")
    print("=" * 78)
    print(f"{'N':>8} {'agreement':>20} {'chance':>10} {'ascending-index runs':>22}")
    rows = correctness([1000, 2000, 5000], args.dims, seeds)
    for n, mean, std, asc, total, chance in rows:
        print(f"{n:>8} {mean:>13.4f} +/- {std:<5.4f} {chance:>10.4f} {f'{asc}/{total}':>22}")
    all_exact = all(abs(m - 1.0) < 1e-12 and s == 0.0 for _, m, s, _, _, _ in rows)
    print(f"\n  => {'PASS-CORRECT' if all_exact else 'FAIL-CORRECT'}")

    print()
    print("=" * 78)
    print("B. Memory -- peak working set, one fresh process per arm")
    print("=" * 78)
    print(f"{'N':>8} {'matrix @ f64':>14} {'matrix-free':>14} {'materialized':>14} {'ratio':>8}")
    mem_rows = []
    for n in (2000, 4000, 8000, 12000):
        free = run_child("matrix-free", n, args.dims, 0)
        full = run_child("materialized", n, args.dims, 0)
        if not free or not full:
            continue
        gb = n * n * 8 / 1e9
        ratio = full["peak"] / free["peak"] if free["peak"] else float("nan")
        mem_rows.append((n, gb, free["peak"], full["peak"], ratio))
        print(
            f"{n:>8} {gb:>12.3f} GB {free['peak']/1e6:>11.1f} MB "
            f"{full['peak']/1e6:>11.1f} MB {ratio:>7.2f}x"
        )

    verdict_mem = "INCONCLUSIVE"
    if len(mem_rows) >= 2:
        # Matrix-free is O(N) if its peak barely moves while the matrix grows
        # quadratically. Compare the largest and smallest sizes measured.
        first, last = mem_rows[0], mem_rows[-1]
        growth_free = last[2] / first[2]
        growth_full = last[3] / first[3]
        size_ratio = (last[0] / first[0]) ** 2
        print(
            f"\n  matrix grew {size_ratio:.0f}x by arithmetic; "
            f"materialized peak grew {growth_full:.1f}x; "
            f"matrix-free peak grew {growth_free:.2f}x"
        )
        verdict_mem = "PASS-MEMORY" if growth_free < 0.25 * growth_full else "FAIL-MEMORY"
    print(f"\n  => {verdict_mem}")

    print()
    print("=" * 78)
    print("C. Wall clock -- G4d's SECOND threshold")
    print("=" * 78)
    print(
        "   The rule: 'if the matrix-free path at 155,000 points is more than an\n"
        "   order of magnitude slower than the in-place path there, the memory wall\n"
        "   was the wrong wall to attack next.' 155,000 points cannot be timed on\n"
        "   this host for the IN-PLACE arm -- its matrix is 96 GB at float32 -- so\n"
        "   what is measured is the RATIO across a size ladder. A flat ratio\n"
        "   extrapolates; a growing one does not. Both arms are O(N^2) in distance\n"
        "   evaluations, so a flat ratio is what the algorithms predict.\n"
    )
    print(f"{'N':>8} {'matrix-free':>14} {'in-place':>14} {'ratio':>9}")
    ratios = []
    for n in (1000, 2000, 4000, 8000):
        free = run_child("time-matrix-free", n, args.dims, 0)
        full = run_child("time-materialized", n, args.dims, 0)
        if not free or not full:
            continue
        ratio = free["secs"] / full["secs"] if full["secs"] else float("nan")
        ratios.append(ratio)
        print(f"{n:>8} {free['secs']:>12.3f} s {full['secs']:>12.3f} s {ratio:>8.2f}x")
    if ratios:
        spread = max(ratios) / min(ratios)
        worst = max(ratios)
        print(
            f"\n  ratio range {min(ratios):.2f}x - {max(ratios):.2f}x "
            f"(spread {spread:.2f}x across an 8x change in N)"
        )
        if worst < 10.0 and spread < 3.0:
            print("\n  => PASS-CLOCK (by extrapolation from a stable ratio, not a")
            print("     measurement at 155,000 points -- state it as such)")
        elif worst >= 10.0:
            print("\n  => FAIL-CLOCK: slower than an order of magnitude; G4d stays a cut candidate")
        else:
            print("\n  => INCONCLUSIVE-CLOCK: the ratio is not stable enough to extrapolate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
