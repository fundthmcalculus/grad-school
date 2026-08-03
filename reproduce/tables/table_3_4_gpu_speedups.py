"""Table 3.4 -- GPU speedups over the CPU, on the card the chapter describes.

Chapter 3 quotes four GPU rows (§3.4, "GPU"): the device-resident Boruvka MST,
the full VAT front end, Fuzzy C-Means, and pairwise distances -- the last of
which is a *negative* result on consumer hardware. Until now those cells came
from findings files (`tribble-cluster/benchmarks/gpu_{fcm,pairwise,vat}.md`) and
`PROVENANCE_MAP.md` listed the table as ungenerated "needs a GPU host". This is
that generator.

WHAT IS TIMED, AND AGAINST WHAT
-------------------------------
Every row is one CPU arm against one GPU arm, measured in the same pass, and
every row carries an EXACTNESS column, because a speedup on a different answer
is not a speedup. The chapter's central GPU claim is that the device MST
reproduces the serial ordering exactly; that claim is tested here rather than
assumed, in the manner of `table_3_2_memory_precision.py`'s ordering column.

  MST (device-resident)   CPU `pcvat.vat_prim_mst_c` (compiled Cython, compact
                          dense Prim) vs `gpu_vat.boruvka_mst_device`. The n x n
                          matrix is ALREADY on the device: this row is the
                          kernel, not the transfer, which is what "device-
                          resident" in the chapter means.
  VAT front end           CPU `pairwise_distances_c` + `vat_prim_mst_c`
                          vs `gpu_vat.vat_gpu` (distances + MST + order, matrix
                          never leaves the device). Both arms produce a VAT
                          ORDER and nothing else, so the work is matched.
                          A second, deliberately UNMATCHED pair is also emitted
                          (`+ compute_vat_c`, which additionally materialises
                          the reordered n x n matrix) because that is the
                          comparison the benchmark note behind the chapter used,
                          and it reads ~3x higher. Both are in the table; the
                          matched one is the honest one.
  Fuzzy C-Means           `fcm.fuzzy_c_means` vs `gpu.fuzzy_c_means_gpu`, from
                          identical initial centers. ** Two CPU arms. ** The
                          library CPU routine is a NumPy broadcasting
                          implementation that materialises (n, k, d) and
                          (n, k, k) temporaries; the GPU routine uses the gram
                          identity and two GEMMs. Those are different
                          ALGORITHMS, not the same algorithm on two devices, so
                          a ratio between them is not a device speedup. The
                          second CPU arm here is the GPU's own formulation
                          written in NumPy/BLAS, which is what isolates the
                          device.
  Pairwise distances      `pcvat.pairwise_distances_c` (C/OpenMP) vs
                          `gpu.pairwise_distances_gpu` (tiled, streamed back to
                          host, because the CPU arm's result is on the host
                          too). Swept over feature dimension and precision,
                          plus the `high_precision=False` fast mode. This is the
                          row the chapter expects to LOSE at low d / float64.

ESTIMATES vs DEMONSTRATIONS  (Chapter 7 §7.2 and the appendix)
--------------------------------------------------------------
Every swept row above is an ESTIMATE: ten seeds (`common.SEEDS`) with the
per-seed spread kept in the CSV. The one row that is not is the reachable-scale
row -- the largest float32 device-resident front end that fits 12 GB of VRAM.
That is a single-shot DEMONSTRATION, recorded with its hardware, precision and
memory footprint instead of a spread, and it is labelled as such in the `kind`
column. Do not read it as an estimate; at the VRAM edge its wall-clock moved by
3x between two probe runs of this very script's arms.

WHAT CANNOT BE MEASURED HERE
----------------------------
The chapter predicts that a datacenter card with full-rate FP64 would flip the
pairwise-distance loss. There is no such card on this host, so that prediction
is recorded as untested and is NOT estimated, extrapolated or implied.

Run (from repo root):
    export PYTHONIOENCODING=utf-8
    uv run --project tribble-cluster --with scipy --with cupy-cuda12x \\
        python reproduce/tables/table_3_4_gpu_speedups.py

Knobs:
    REPRO_GPU_VAT_N="4000,8000,16000,32000"   MST / front-end sweep
    REPRO_GPU_PW_N="16000"                    N for the pairwise sweep
    REPRO_GPU_PW_DIMS="10,50,200,784"         feature dimensions swept
    REPRO_GPU_FCM_N="50000,200000,500000"     FCM sweep
    REPRO_GPU_DEMO_N="48000"                  reachable-scale demonstration ("" skips)
    REPRO_SEEDS="0,1,2"                       smoke only -- NOT citable
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import numpy as np

_TABLES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_TABLES))
import common as C  # noqa: E402

VAT_N = [int(x) for x in os.environ.get("REPRO_GPU_VAT_N", "4000,8000,16000,32000").split(",")]
PW_N = int(os.environ.get("REPRO_GPU_PW_N", "16000"))
PW_DIMS = [int(x) for x in os.environ.get("REPRO_GPU_PW_DIMS", "10,50,200,784").split(",")]
FCM_N = [int(x) for x in os.environ.get("REPRO_GPU_FCM_N", "50000,200000,500000").split(",")]
_demo = os.environ.get("REPRO_GPU_DEMO_N", "48000").strip()
DEMO_N = int(_demo) if _demo else None

FCM_K = int(os.environ.get("REPRO_GPU_FCM_K", "10"))
FCM_D = int(os.environ.get("REPRO_GPU_FCM_D", "20"))
VAT_D = int(os.environ.get("REPRO_GPU_VAT_D", "10"))

EST = "estimate (%d seeds)" % len(C.SEEDS)
DEMO = "demonstration (1 shot)"


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def blobs(n, d, k, seed, dtype=np.float64):
    """Isotropic Gaussian blobs -- the input shape every GPU note behind the
    chapter used (`boruvka_vat.make_blobs`), reimplemented here so the table
    does not depend on a script under ClusteringExperiments/."""
    rng = np.random.RandomState(seed)
    centers = rng.rand(k, d) * 10.0
    labels = rng.randint(0, k, n)
    return np.ascontiguousarray(centers[labels] + rng.randn(n, d) * 0.5, dtype=dtype)


# --------------------------------------------------------------------------- #
# device state -- recorded, not assumed
# --------------------------------------------------------------------------- #
def gpu_state():
    """(ok, description). `ok=False` carries the reason the GPU path is absent.

    The established convention for an unavailable method in this harness is
    `common.NA` plus a printed explanation. A generator that cleanly reports
    "no CUDA runtime" is a result; a table of invented speedups is not.
    """
    try:
        import cupy as cp
    except Exception as exc:  # noqa: BLE001
        return False, f"CuPy not importable ({exc.__class__.__name__}: {exc})"
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False, "CuPy imported but no CUDA device present"
        dev = cp.cuda.Device(0)
        cc = dev.compute_capability
        free, total = cp.cuda.runtime.memGetInfo()
        rt = cp.cuda.runtime.runtimeGetVersion()
    except Exception as exc:  # noqa: BLE001
        return False, (f"CUDA device present but unusable "
                       f"({exc.__class__.__name__}: {exc})")
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        smi = "nvidia-smi unavailable"
    return True, (f"{smi}; compute capability {cc}; VRAM free "
                  f"{free / 1e9:.1f}/{total / 1e9:.1f} GB at start; "
                  f"CuPy {cp.__version__}; CUDA runtime "
                  f"{rt // 1000}.{(rt % 1000) // 10}")


# --------------------------------------------------------------------------- #
# arms
# --------------------------------------------------------------------------- #
def fcm_gram_cpu(x, k, m=2.0, initial_guess=None, max_iter=100, tol=1e-5):
    """The GPU routine's OWN formulation, on the CPU in NumPy/BLAS.

    Line-for-line the algorithm of `gpu.fuzzy_c_means_gpu`: squared distances by
    the gram identity (one GEMM), the closed-form membership, and the centre
    update (a second GEMM), with the identical convergence test. It exists so
    the FCM row can separate the DEVICE speedup from the FORMULATION speedup --
    the library's CPU routine is a broadcasting implementation with (n, k, d)
    temporaries, and comparing that against a GEMM measures the rewrite, not the
    card. Returns (centers, membership, iterations).
    """
    C0 = np.array(initial_guess, dtype=x.dtype, copy=True)
    q = 1.0 / (m - 1.0)
    sqx = np.sum(x * x, axis=1)

    def memb(cen):
        d2 = sqx[:, None] - 2.0 * (x @ cen.T) + np.sum(cen * cen, axis=1)[None, :]
        np.maximum(d2, 0.0, out=d2)
        with np.errstate(divide="ignore"):
            inv = d2 ** (-q)
        return inv / np.sum(inv, axis=1, keepdims=True)

    iters = max_iter
    for i in range(max_iter):
        u = memb(C0)
        um = u ** m
        c_new = (um.T @ x) / np.sum(um, axis=0)[:, None]
        if np.all(np.abs(c_new - C0) <= (1e-8 + tol * np.abs(C0))):
            C0, iters = c_new, i + 1
            break
        C0 = c_new
    return C0, memb(C0), iters


def prim_total(D, order):
    """Total weight of the Prim tree implied by a VAT order: sum_k min_{j<k}
    D[o_k, o_j]. Used ONLY when an ordering disagrees, to say whether the
    divergence is a tie-break between two equally valid MSTs (equal totals) or a
    genuinely different tree. O(n^2); not run otherwise."""
    o = np.asarray(order, dtype=np.int64)
    tot = 0.0
    for k in range(1, len(o)):
        tot += float(np.min(D[o[k], o[:k]].astype(np.float64)))
    return tot


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
class Runner:
    """Holds the imports and the accumulated rows."""

    def __init__(self):
        from tribbleclustering import gpu as tgpu
        from tribbleclustering import gpu_vat as tgpu_vat
        from tribbleclustering.fcm import fuzzy_c_means
        from tribbleclustering.pcvat import (
            compute_vat_c, pairwise_distances_c, vat_prim_mst_c,
        )
        import cupy as cp

        self.cp = cp
        self.gpu = tgpu
        self.gpu_vat = tgpu_vat
        self.fcm_cpu = fuzzy_c_means
        self.pdist_c = pairwise_distances_c
        self.prim_c = vat_prim_mst_c
        self.vat_c = compute_vat_c
        self.rows = []       # CSV rows (absolute seconds)
        self.pairs = []      # (cpu_mean, gpu_mean) per row, for the Markdown view

    # -- plumbing ---------------------------------------------------------- #
    def sync(self):
        self.cp.cuda.Stream.null.synchronize()

    def free(self):
        self.cp.get_default_memory_pool().free_all_blocks()

    def time_gpu(self, fn, *a, **kw):
        """Wall clock of a device call INCLUDING the synchronise. Without the
        sync a CUDA launch returns immediately and the arm times as ~0."""
        self.sync()
        t0 = time.perf_counter()
        out = fn(*a, **kw)
        self.sync()
        return out, time.perf_counter() - t0

    def warm(self):
        """JIT/cuBLAS/RawModule warm-up, per dtype and per kernel actually used.

        The first `boruvka_mst_device` call of a process spends ~0.4 s compiling
        its RawModule -- 13x the n=16,000 kernel time. Timed cold it would make
        the MST row's small-N cells meaningless.
        """
        print("  warming device kernels (RawModule JIT, cuBLAS handles)...")
        for dt in (np.float64, np.float32):
            X = blobs(1024, VAT_D, 4, 0, dtype=dt)
            self.gpu_vat.vat_gpu(X)
            self.gpu.pairwise_distances_gpu(X, high_precision=True)
            self.gpu.pairwise_distances_gpu(X, high_precision=False)
            Xf = blobs(2048, FCM_D, FCM_K, 0, dtype=dt)
            g0 = Xf[:2 * FCM_K].reshape(FCM_K, 2, FCM_D).mean(axis=1)
            self.gpu.fuzzy_c_means_gpu(Xf, FCM_K, m=2.0, initial_guess=g0)
        self.sync()
        self.free()

    def add(self, kernel, conditions, cpu_arm, gpu_arm, cpu_t, gpu_t, exact,
            kind=EST):
        cm, _ = C.agg(cpu_t)
        gm, _ = C.agg(gpu_t)
        speed = f"{cm / gm:.2f}x" if (cm and gm) else C.NA
        # Five decimals, not the emitter's default three. The MST kernel at
        # N=4,000 runs in 3 ms, and at three decimals the CSV reads "0.003" --
        # from which a reader re-deriving the ratio gets 6.0x against the 5.56x
        # this row actually measured. The CSV is the record the Markdown ratios
        # are checked against, so it has to carry enough digits to check them.
        self.rows.append([
            kernel, conditions, cpu_arm, gpu_arm,
            (C.cell(cpu_t, fmt="{:.5f}") if cpu_t else C.NA),
            (C.cell(gpu_t, fmt="{:.5f}") if gpu_t else C.NA),
            speed, exact, kind,
        ])
        self.pairs.append((cm, gm))
        print(f"    {kernel:<22} {conditions:<34} CPU {cm if cm else float('nan'):8.3f}s "
              f"GPU {gm if gm else float('nan'):8.3f}s  ->  {speed:>8}   {exact}")

    # -- rows -------------------------------------------------------------- #
    def vat_rows(self):
        """MST kernel and full front end, swept over N at float64."""
        print("  MST + VAT front end (float64, d=%d)" % VAT_D)
        for n in VAT_N:
            cpu_mst, gpu_mst = [], []
            cpu_e2e, cpu_e2e_gather, gpu_e2e = [], [], []
            agree_mst, agree_e2e = [], []
            for seed in C.SEEDS:
                X = blobs(n, VAT_D, 25, seed)
                with C.timed() as t:
                    D = self.pdist_c(X)
                t_pw = t.seconds
                with C.timed() as t:
                    order_cpu, _ = self.prim_c(D)
                cpu_mst.append(t.seconds)
                cpu_e2e.append(t_pw + t.seconds)
                with C.timed() as t:
                    self.vat_c(D.copy())
                cpu_e2e_gather.append(t_pw + t.seconds)

                # --- MST kernel on an already-resident matrix ---
                Dg = self.cp.asarray(D)
                (mu, mv), t_g = self.time_gpu(self.gpu_vat.boruvka_mst_device, Dg)
                gpu_mst.append(t_g)
                w = self.cp.asnumpy(Dg[mu, mv])
                src = int(self.cp.argmax(Dg).get()) // n
                o_g, _ = self.gpu_vat._order_from_mst(
                    self.cp.asnumpy(mu), self.cp.asnumpy(mv), w, n, src)
                agree_mst.append(float(np.mean(o_g == order_cpu)))
                del Dg, mu, mv
                self.free()

                # --- end to end, matrix born and kept on the device ---
                (o_e2e, _), t_g = self.time_gpu(self.gpu_vat.vat_gpu, X)
                gpu_e2e.append(t_g)
                agree_e2e.append(float(np.mean(o_e2e == order_cpu)))
                del D
                self.free()
            self.add("Boruvka MST (device)", f"N={n:,}, float64, matrix resident",
                     "pcvat.vat_prim_mst_c (Cython dense Prim)",
                     "gpu_vat.boruvka_mst_device", cpu_mst, gpu_mst,
                     exactness(agree_mst))
            self.add("VAT front end", f"N={n:,}, float64, order only",
                     "pairwise_distances_c + vat_prim_mst_c",
                     "gpu_vat.vat_gpu", cpu_e2e, gpu_e2e, exactness(agree_e2e))
            self.add("VAT front end", f"N={n:,}, float64, UNMATCHED work",
                     "pairwise_distances_c + compute_vat_c (also reorders D)",
                     "gpu_vat.vat_gpu (order only)", cpu_e2e_gather, gpu_e2e,
                     exactness(agree_e2e))

    def fcm_rows(self):
        """FCM, swept over N, against BOTH CPU arms."""
        print("  Fuzzy C-Means (k=%d, d=%d, m=2, <=100 iters)" % (FCM_K, FCM_D))
        for n in FCM_N:
            t_bcast, t_gram, t_gpu = [], [], []
            lab_b, lab_g, cdiff_b, cdiff_g, iters = [], [], [], [], []
            for seed in C.SEEDS:
                X = blobs(n, FCM_D, FCM_K, seed)
                rng = np.random.RandomState(seed)
                idx = rng.choice(n, size=FCM_K * 2, replace=False)
                g0 = X[idx].reshape(FCM_K, 2, FCM_D).mean(axis=1)

                with C.timed() as t:
                    c_b, u_b = self.fcm_cpu(X, FCM_K, m=2.0, initial_guess=g0)
                t_bcast.append(t.seconds)
                with C.timed() as t:
                    c_r, u_r, it = fcm_gram_cpu(X, FCM_K, initial_guess=g0)
                t_gram.append(t.seconds)
                iters.append(it)
                (c_g, u_g), tg = self.time_gpu(
                    self.gpu.fuzzy_c_means_gpu, X, FCM_K, m=2.0, initial_guess=g0)
                t_gpu.append(tg)

                lab_b.append(float(np.mean(u_b.argmax(1) == u_g.argmax(1))))
                lab_g.append(float(np.mean(u_r.argmax(1) == u_g.argmax(1))))
                # centres are returned in the same cluster order (same init), so
                # a direct elementwise comparison is meaningful.
                cdiff_b.append(float(np.abs(c_b - c_g).max()))
                cdiff_g.append(float(np.abs(c_r - c_g).max()))
                self.free()
            self.add("Fuzzy C-Means", f"N={n:,}, k={FCM_K}, d={FCM_D}",
                     "fcm.fuzzy_c_means (NumPy broadcasting -- DIFFERENT algorithm)",
                     "gpu.fuzzy_c_means_gpu (gram + 2 GEMM)",
                     t_bcast, t_gpu,
                     f"labels {C.cell(lab_b)}; max abs Δcentre "
                     f"{max(cdiff_b):.1e}" if cdiff_b else C.NA)
            self.add("Fuzzy C-Means", f"N={n:,}, k={FCM_K}, d={FCM_D}, MATCHED formulation",
                     "gram + 2 GEMM in NumPy/BLAS (this file)",
                     "gpu.fuzzy_c_means_gpu (gram + 2 GEMM)",
                     t_gram, t_gpu,
                     f"labels {C.cell(lab_g)}; max abs Δcentre "
                     f"{max(cdiff_g):.1e}" if cdiff_g else C.NA)
            print(f"      (matched-formulation CPU arm converged in "
                  f"{min(iters)}-{max(iters)} iterations)")

    def pairwise_rows(self):
        """The row the chapter expects to lose: distances by dimension and precision."""
        print("  Pairwise distances (N=%d)" % PW_N)
        for dtname, dt in (("float64", np.float64), ("float32", np.float32)):
            for d in PW_DIMS:
                for hp in (True, False):
                    if dt == np.float64 and not hp:
                        continue   # the fast path only differs for float32
                    t_c, t_g, diffs = [], [], []
                    for seed in C.SEEDS:
                        X = blobs(PW_N, d, 25, seed, dtype=dt)
                        with C.timed() as t:
                            Dc = self.pdist_c(X)
                        t_c.append(t.seconds)
                        Dg, tg = self.time_gpu(
                            self.gpu.pairwise_distances_gpu, X, high_precision=hp)
                        t_g.append(tg)
                        diffs.append(float(np.abs(
                            Dc.astype(np.float64) - np.asarray(Dg, dtype=np.float64)
                        ).max()))
                        del Dc, Dg
                        self.free()
                    mode = "high_precision" if hp else "fast (native acc)"
                    self.add("Pairwise distances", f"N={PW_N:,}, d={d}, {dtname}, {mode}",
                             "pcvat.pairwise_distances_c (C/OpenMP)",
                             "gpu.pairwise_distances_gpu (tiled -> host)",
                             t_c, t_g, f"max abs Δ = {max(diffs):.1e}")

    def demo_row(self):
        """Single-shot reachable-scale DEMONSTRATION, not an estimate."""
        if not DEMO_N:
            return
        n = DEMO_N
        nbytes = n * n * 4
        print(f"  reachable-scale demonstration: N={n:,} float32 "
              f"({nbytes / 1e9:.2f} GB resident)")
        X = blobs(n, VAT_D, 25, C.SEEDS[0], dtype=np.float32)
        with C.timed() as t:
            D = self.pdist_c(X)
        t_pw = t.seconds
        with C.timed() as t:
            order_cpu, _ = self.prim_c(D)
        cpu = t_pw + t.seconds
        (o_g, _), gpu_t = self.time_gpu(self.gpu_vat.vat_gpu, X)
        agree = float(np.mean(o_g == order_cpu))
        exact = exactness([agree])
        if agree < 1.0:
            # Say whether the divergence is a tie-break between equally valid
            # MSTs or a genuinely different tree. This is the whole reason the
            # exactness column exists.
            a, b = prim_total(D, order_cpu), prim_total(D, o_g)
            exact += (f"; Prim total CPU {a:.6f} vs GPU {b:.6f} "
                      f"(rel {abs(a - b) / a:.2e})")
        del D
        self.free()
        self.add("VAT front end", f"N={n:,}, float32, {nbytes / 1e9:.2f} GB resident",
                 "pairwise_distances_c + vat_prim_mst_c",
                 "gpu_vat.vat_gpu", [cpu], [gpu_t], exact, kind=DEMO)


def exactness(agrees):
    """Render an ordering-agreement series the way Table 3.3 does."""
    if not agrees:
        return C.NA
    if all(a == 1.0 for a in agrees):
        return "order 1.000 (bit-identical)"
    return f"order {C.cell(agrees, fmt='{:.5f}')}"


# --------------------------------------------------------------------------- #
# emit
# --------------------------------------------------------------------------- #
HEADER = ["kernel", "conditions", "CPU arm", "GPU arm", "CPU (s)", "GPU (s)",
          "GPU speedup (CPU/GPU)", "exactness vs CPU", "kind"]
MD_HEADER = ["kernel", "conditions", "CPU", "GPU", "exactness vs CPU", "kind"]


def note(gpu_desc, blocked=False):
    head = (
        "**Device.** " + gpu_desc + ". "
    )
    if blocked:
        return (head +
                "** Every cell is N/A: the device path could not run at all, for "
                "the reason above, so no arm was timed and the table is a record "
                "of the blocker rather than of a measurement. ** Nothing here is "
                "estimated, extrapolated, or carried over from the findings files "
                "the chapter previously quoted. Install a working CuPy/CUDA "
                "runtime for the device and re-run; the generator needs no other "
                "change.")
    return head + (
        "**The Markdown arms are normalized against the slower arm in each row**: "
        "the loser is the 1.0x baseline and the winner reads as 'this many times "
        "faster', so a row where **the GPU is the 1.0x baseline is a row the GPU "
        "loses**. Absolute seconds and their per-seed spreads are in the "
        "companion CSV; ratios survive a change of machine and seconds do not. "
        "\n>\n"
        "> **Exactness is a column, not an assumption.** A speedup on a different "
        "answer is not a speedup, so every VAT/MST row compares its ordering "
        "elementwise against the compiled Cython serial reference on identical "
        "points, every FCM row compares hard labels and centres from identical "
        "initial centres, and every distance row reports the max absolute "
        "deviation from the CPU kernel. \n>\n"
        "> **The Fuzzy C-Means row is quoted twice on purpose, and the chapter's "
        "version of it overstates the device.** `fcm.fuzzy_c_means` is a NumPy "
        "broadcasting implementation that materialises (n, k, d) and (n, k, k) "
        "temporaries; `gpu.fuzzy_c_means_gpu` uses the gram identity and two "
        "GEMMs. Those are different algorithms, so the ratio between them "
        "measures a rewrite as much as a card. The MATCHED rows run the GPU's own "
        "formulation in NumPy/BLAS on the CPU, and that is the device speedup. "
        "The FCM seconds carry a very large spread by construction: with the same "
        "initial centres and the same convergence test, the number of iterations "
        "to the fixed point varies from about 11 to the 100-iteration cap across "
        "the ten seeds, so a seed that runs nine times as long appears in both "
        "arms. All three arms see the same seeds and the reported figure is a "
        "ratio of means over them, but read the per-seed spread in the CSV before "
        "quoting any single FCM number. \n>\n"
        "> **The VAT front end is also quoted twice.** The matched pair has both "
        "arms produce only an ordering. The UNMATCHED pair additionally has the "
        "CPU arm materialise the reordered n x n matrix (`compute_vat_c`), which "
        "the GPU arm never does; it is included because that is the comparison "
        "behind the chapter's cell, and it reads roughly three times higher. \n>\n"
        "> **Two arms differing in kind is a real hazard here.** Ratios in this "
        "project are not machine-invariant when the arms differ in kind rather "
        "than in device -- a 40% cross-host move was measured in a ratio whose "
        "two arms were interpreted Python versus compiled Cython. The MST, front "
        "end and distance rows compare compiled CPU (Cython / C+OpenMP) against "
        "compiled CUDA and are safe on that count; the unmatched FCM row is not, "
        "which is why the matched one exists. \n>\n"
        "> **The pairwise-distance rows are a NEGATIVE RESULT and are meant to "
        "be.** This card's FP64 throughput is a small fraction of its FP32, and "
        "the O(n^2) result must come back across PCIe, so the GPU loses at low "
        "dimension and at float64 -- the regime VAT actually lives in. The "
        "chapter predicts a datacenter card with full-rate FP64 would flip this. "
        "** That prediction is UNTESTED and untestable on this host **; no cell "
        "here estimates it. \n>\n"
        "> **kind.** Swept rows are ESTIMATES: ten seeds, spread in the CSV. The "
        "reachable-scale row is a single-shot DEMONSTRATION recorded with its "
        "hardware, precision and resident footprint instead of a spread "
        "(Chapter 7 §7.2), and at the VRAM edge its wall clock is volatile -- do "
        "not read it as an estimate. \n>\n"
        "> **What is timed.** The MST row's matrix is already device-resident, so "
        "it times the kernel and not the transfer. Every device timing includes "
        "an explicit stream synchronise; without one a CUDA launch returns "
        "immediately and the arm would time as zero. RawModule JIT and cuBLAS "
        "handle creation are warmed before any measurement (the first "
        "`boruvka_mst_device` call spends ~0.4 s compiling, 13x the N=16,000 "
        "kernel time)."
    )


def main():
    print(f"Table 3.4 -- GPU speedups over the CPU  (seeds={C.SEEDS})")
    ok, desc = gpu_state()
    print(f"  device: {desc}")
    if not ok:
        # No device path: emit the table with every GPU cell N/A and the reason
        # named, rather than skipping the table (silence is not success) or
        # carrying numbers over from a findings file.
        print("  [BLOCKED] the GPU arm cannot run; emitting N/A rows with the reason")
        rows = [[k, cond, "(not run: GPU arm unavailable)", arm, C.NA, C.NA,
                 C.NA, C.NA, "not measured (no device)"]
                for k, cond, arm in (
                    ("Boruvka MST (device)", "float64, matrix resident",
                     "gpu_vat.boruvka_mst_device"),
                    ("VAT front end", "float64, order only", "gpu_vat.vat_gpu"),
                    ("Fuzzy C-Means", f"k={FCM_K}, d={FCM_D}",
                     "gpu.fuzzy_c_means_gpu"),
                    ("Pairwise distances", "float64/float32 x dimension",
                     "gpu.pairwise_distances_gpu"))]
        C.emit("table_3_4_gpu_speedups",
               "Table 3.4 — GPU speedups over the CPU (NOT MEASURED: no usable device)",
               HEADER, rows, md_header=MD_HEADER,
               md_rows=[[r[0], r[1], C.NA, C.NA, C.NA, r[8]] for r in rows],
               note=note(desc, blocked=True))
        return

    r = Runner()
    r.warm()
    r.vat_rows()
    r.fcm_rows()
    r.pairwise_rows()
    r.demo_row()

    md_rows = []
    for row, (cm, gm) in zip(r.rows, r.pairs):
        cn, gn = C.normalized_worst([cm, gm])
        md_rows.append([row[0], row[1], cn, gn, row[7], row[8]])

    C.emit("table_3_4_gpu_speedups",
           "Table 3.4 — GPU speedups over the CPU (RTX 4080 Laptop, 12 GB)",
           HEADER, r.rows, md_header=MD_HEADER, md_rows=md_rows,
           note=note(desc))


if __name__ == "__main__":
    main()
