"""Table 3.3 -- memory footprint and reachable problem size, by precision and scheme.

Chapter 3 makes two different memory claims, and an earlier draft of Table 3.2
conflated them: that is how Table 3.1 could quote N = 135,000 while Table 3.3 gave
a ceiling of ~89,000 points on what was described as the same host. Two things were
wrong. The host has 96 GB, not 64 -- 64 GB is a self-imposed working cap that keeps
a large reorder from paging -- so both budgets are reported below. And the schemes
are three, not two, of which only the first two are bounded by the matrix at all.

  classical   materialise D and a reordered copy          k = 3  ->  3 N^2 itemsize
  in-place    materialise D once, permute it in place     k = 1  ->  1 N^2 itemsize
  on-demand   never materialise D; compute D_ij as the    k = 0  ->  no N^2 term
              reorder asks for it (pvat.vat_prim_mst_seq)
              *** DEFECTIVE -- see the ordering column and the note. ***

Two kinds of number live in this table, and they are labelled as such:

  MEMORY    exact arithmetic, NOT a measurement.  N_max = sqrt(budget / (k itemsize)).
  ORDERING  MEASURED.  Halving the precision is only a memory win if the reordering
            survives it, and that is a claim Chapter 3 asserts ("bit-identical to
            the serial reference") without ever varying the precision.  Every cell
            is compared against the float64 in-place ordering on identical points.

Caveat kept visible rather than buried: the on-demand path is the STAGE-ONE
heap algorithm, so its residual memory is the priority queue, not a strict O(N).
The queue is not sized here -- that needs a peak-RSS harness, and is noted as
unmeasured rather than asserted to be small.

Half precision is deliberately out of scope here.  pcvat.pyx ships _64 and _32
kernels only, and reduced precision below float32 belongs with the Boruvka/GPU
path of Chapter 3 §3.3.3 rather than with the CPU memory scheme this table is
about -- that is where a half-precision distance kernel would actually pay.

Run (from repo root):
    uv run --project tribble-cluster python reproduce/tables/table_3_2_memory_precision.py

Knobs:
    REPRO_MEM_BUDGET_GB="64,96" comma-separated budgets; one ceiling column each.
                                Default is the 64 GB working cap and the 96 GB host.
    REPRO_MEM_REF_N="64000"     N at which the footprint column is reported
    REPRO_ORDER_N="2000"        N at which ordering agreement is measured
    REPRO_SEEDS="0,1,2"         seeds for the ordering check
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
import common as C  # noqa: E402

# Two budgets, and the difference between them is a policy, not a hardware limit.
# The host has 96 GB; runs are held to a self-imposed 64 GB working cap so that a
# large reorder cannot start touching the swapfile and turn a memory measurement
# into a disk measurement. Both ceilings are reported because the cap is the one
# that governs day-to-day work while the physical figure is the one that explains
# the 135,000-point result. Decimal GB throughout -- GiB would silently move every
# cell away from the convention the prose was derived under.
BUDGETS = [float(x) for x in os.environ.get("REPRO_MEM_BUDGET_GB", "64,96").split(",")]
REF_N = int(os.environ.get("REPRO_MEM_REF_N", "64000"))
ORDER_N = int(os.environ.get("REPRO_ORDER_N", "2000"))

DTYPES = [("float64", np.float64), ("float32", np.float32)]

# (label, matrices held).  k = 0 marks a scheme with no N^2 term at all.
SCHEMES = [("classical (D + copy + work)", 3),
           ("in-place (D only)", 1),
           ("on-demand (vat_prim_mst_seq) — DEFECTIVE", 0)]


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def points(n, seed, dim=3):
    return np.random.RandomState(seed).rand(n, dim)


def distance_matrix(P):
    G = P @ P.T
    d = np.diag(G)
    return np.sqrt(np.maximum(d[:, None] - 2.0 * G + d[None, :], 0.0))


# --------------------------------------------------------------------------- #
# memory: exact arithmetic
# --------------------------------------------------------------------------- #
def footprint(n, k, itemsize):
    """Bytes held by the N^2 term. k = 0 means the scheme holds no matrix."""
    return None if k == 0 else k * (n ** 2) * itemsize


def ceiling(k, itemsize, budget):
    """Largest N whose N^2 term fits the budget. None when there is no N^2 term."""
    return None if k == 0 else int(math.sqrt(budget * 1e9 / (k * itemsize)))


def gb(nbytes):
    return f"{nbytes / 1e9:,.1f} GB"


# --------------------------------------------------------------------------- #
# ordering: measured
# --------------------------------------------------------------------------- #
def orderings(P, D64, dtype, scheme_k):
    """VAT ordering for one (scheme, dtype), or a string explaining the failure."""
    from tribbleclustering.pvat import compute_vat, vat_prim_mst_seq

    if scheme_k == 0:
        return np.asarray(vat_prim_mst_seq(P.astype(dtype)))
    D = D64.astype(dtype)
    # k = 3 is the classical route: D is preserved and a reordered copy is built.
    # k = 1 permutes D itself.  Both return the same permutation on exact input;
    # whether they still agree in reduced precision is the question.
    return np.asarray(compute_vat(D, inplace=(scheme_k == 1))[1])


def agreement(a, b):
    if a.shape != b.shape:
        return 0.0
    return float(np.mean(a == b))


def main():
    print(f"Table 3.2 -- memory x precision  (budgets "
          f"{', '.join(f'{b:.0f} GB' for b in BUDGETS)}, "
          f"footprint at N={REF_N:,}, ordering at N={ORDER_N:,}, seeds={C.SEEDS})")

    # ---- measured arm: does the ordering survive the precision? ----
    ref = {}
    measured: dict[tuple[str, int], list[float]] = {}
    for seed in C.SEEDS:
        P = points(ORDER_N, seed)
        D64 = distance_matrix(P)
        try:
            ref[seed] = orderings(P, D64, np.float64, 1)
        except Exception as exc:  # noqa: BLE001
            print(f"  [fatal] float64 in-place reference failed on seed {seed}: "
                  f"{exc.__class__.__name__}: {exc}")
            return
        for name, dtype in DTYPES:
            for label, k in SCHEMES:
                key = (name, k)
                try:
                    o = orderings(P, D64, dtype, k)
                    measured.setdefault(key, []).append(agreement(o, ref[seed]))
                except Exception as exc:  # noqa: BLE001
                    measured.setdefault(key, [])
                    if seed == C.SEEDS[0]:
                        print(f"  [skip] {name:<8} {label:<28} "
                              f"{exc.__class__.__name__}")
        print(f"  done: seed {seed}")

    # ---- emit ----
    rows = []
    for name, dtype in DTYPES:
        itemsize = np.dtype(dtype).itemsize
        for label, k in SCHEMES:
            fp = footprint(REF_N, k, itemsize)
            vals = measured.get((name, k), [])
            if not vals:
                order = "no kernel"
            elif all(v == 1.0 for v in vals):
                order = "1.000 (exact)"
            else:
                order = C.cell(vals)
            ceilings = []
            for b in BUDGETS:
                cl = ceiling(k, itemsize, b)
                ceilings.append(f"{cl:,}" if cl is not None else "not memory-bound")
            rows.append([
                name,
                label,
                str(itemsize),
                gb(fp) if fp is not None else "no N^2 term",
                *ceilings,
                order,
            ])

    C.emit(
        "table_3_2_memory_precision",
        "Table 3.3 — memory footprint and reachable N, by precision and scheme",
        ["precision", "scheme", "bytes/entry", f"footprint at N={REF_N:,}",
         *(f"largest N in {b:.0f} GB" for b in BUDGETS),
         f"ordering vs float64 (N={ORDER_N:,})"],
        rows,
        note=(
            "The memory columns are EXACT ARITHMETIC, not measurements: "
            "N_max = sqrt(budget / (k · itemsize)) for a scheme holding k matrices, "
            "with the budget taken as decimal GB. TWO budgets are reported because "
            "they mean different things: the host carries 96 GB, and 64 GB is a "
            "self-imposed working cap that keeps a large reorder from touching the "
            "swapfile and turning a memory measurement into a disk measurement. The "
            "64 GB column governs routine work; the 96 GB column is the one that "
            "explains the 135,000-point result, which needs 72.9 GB at float32 and "
            "so sits inside the physical limit but outside the cap. The ordering "
            "column is MEASURED -- every cell runs "
            "the reorder at that precision and scheme on identical points and "
            "compares the permutation elementwise against the float64 in-place "
            "reference, so 1.000 means bit-identical. This is the column that says "
            "whether a precision reduction is a real memory win or a silent change "
            "of answer. The on-demand scheme has no N^2 term and so no matrix-bound "
            "ceiling; it is the stage-one heap algorithm, so its residual memory is "
            "the priority queue rather than a strict O(N), and that queue is NOT "
            "sized here. ** The on-demand row is a NEGATIVE RESULT. ** "
            "`pvat.vat_prim_mst_seq` is the only matrix-free reorder in the package, "
            "and it does not produce the VAT ordering: it returns the seed vertex "
            "followed by every other vertex in ascending index order, which is why "
            "agreement against the float64 reference sits at chance. The cause is a "
            "vectorised call to `_get_dist(samples, u, vertices[mask])`, a function "
            "typed for scalar indices -- the reduction collapses to one scalar, every "
            "candidate is assigned the same key, and the heap then pops in index "
            "order. Nothing in the package calls this function; it is exported, dead, "
            "and wrong. Until it is fixed there is no matrix-free path here, so the "
            "'not memory-bound' ceiling is arithmetic about a scheme that does not "
            "yet work. Precision below float32 is out of scope: pcvat.pyx ships "
            "_64 and _32 kernels only, and half precision belongs with the Borůvka/GPU "
            "path of §3.3.3, where it would actually pay."
        ),
    )


if __name__ == "__main__":
    main()
