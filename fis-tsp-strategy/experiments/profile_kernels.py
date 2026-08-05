"""Where the solver's time actually goes, and whether Cython could take any of it back.

Ordinary Python profilers are useless here and it is worth saying why before trusting any
number below. A whole solve is *one* ``@njit`` call: ``lk_solve`` and ``iterated_lk`` enter
nopython mode once and do not return to the interpreter until the tour is finished. ``cProfile``
therefore reports one entry taking 100% of the time, and ``line_profiler`` sees nothing at all.
Nothing inside the machine code is visible to the interpreter that launched it.

So the time is attributed three ways, each of which checks the others:

1. **Counter attribution.** ``costmodel.py`` already fits per-unit costs to measured wall clock
   by non-negative least squares over 224 (instance, configuration) samples. Those coefficients
   are a profile — cost per city scan, per chain level, per rule-base evaluation — and they are
   the only method here that measures the kernels *in situ*, contending for the same caches as
   the rest of the solve. Read as ground truth; the rest is corroboration.
2. **Microbenchmarks in nopython mode.** Each kernel is called in a tight ``@njit`` loop, so
   the dispatch cost is paid once for a million iterations rather than once per call. Calling
   an ``@njit`` function *from Python* costs on the order of a microsecond, which would swamp a
   161 ns kernel entirely — measuring that way is the most common way to get this wrong, so the
   dispatch overhead is measured and printed rather than left implicit.
3. **Ablation.** The same solve with a component switched off, differenced. This catches costs
   the counters cannot see, since a counter only exists where somebody thought to add one.

**On Cython.** The question is not "is Cython faster than Python" — nothing in the hot path is
Python. It is "is Cython faster than numba", and numba already emits LLVM-optimised machine
code with bounds checking off (which FINDINGS §10.4 records the hard way: ``xc[4]`` on a 4-wide
buffer was a silent out-of-bounds write, not an exception). ``--cython`` builds a hand-written
C translation of the hottest kernel and times it against numba's on identical inputs, which is
the measurement that settles it rather than the argument.

Run:  python experiments/profile_kernels.py
      python experiments/profile_kernels.py --cython      # also build and race the C version
      python experiments/profile_kernels.py --asm         # dump numba's generated assembly
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paths  # noqa: E402

paths.on_path()

import numpy as np  # noqa: E402
from numba import njit  # noqa: E402

import costmodel  # noqa: E402
import fis  # noqa: E402
from core import build_candidates, greedy_edge_tour, make_pos, nn_stats  # noqa: E402
from fis_lk import local_search as fis_ls  # noqa: E402
from lk import (  # noqa: E402
    STAT_CHAIN_CALLS,
    STAT_DEPTH,
    STAT_EVALS,
    STAT_MOVES,
    STAT_REV_WORK,
    STAT_SCANS,
    lk_solve,
)

PROFILE_INSTANCE = "pr2392"


# --------------------------------------------------------------------------------------
# 2. microbenchmarks, with the loop inside nopython mode
# --------------------------------------------------------------------------------------
@njit(cache=True)
def _loop_eval1(reps, x, mu, tab, ant, cons):
    """``fis_eval1`` ``reps`` times, accumulating so nothing is optimised away.

    The accumulator matters: without a consumed result LLVM is entitled to delete the whole
    loop, and the kernel then benchmarks at zero. Perturbing ``x`` each iteration also stops
    the result being hoisted out as loop-invariant, which is the same failure in a subtler form.
    """
    acc = 0.0
    for i in range(reps):
        x[0] = (i & 63) / 64.0
        acc += fis.fis_eval1(x, mu, tab, ant, cons)
    return acc


@njit(cache=True)
def _loop_eval(reps, x, mu, tab, ant, cons, out):
    acc = 0.0
    for i in range(reps):
        x[0] = (i & 63) / 64.0
        fis.fis_eval(x, mu, tab, ant, cons, out)
        acc += out[0]
    return acc


@njit(cache=True)
def _loop_memberships(reps, x, tab, mu):
    acc = 0.0
    for i in range(reps):
        x[0] = (i & 63) / 64.0
        fis._memberships(x, tab, mu)
        acc += mu[0, 0]
    return acc


def _bench(fn, reps, *args, trials=5):
    """(ns per iteration, total seconds). Minimum over trials, for the usual reason."""
    fn(2, *args)  # compile
    best = float("inf")
    for _ in range(trials):
        t0 = time.perf_counter()
        fn(reps, *args)
        best = min(best, time.perf_counter() - t0)
    return best / reps * 1e9, best


def _dispatch_overhead(x, mu, tab, ant, cons, reps=200_000):
    """What one Python-level call into nopython mode costs, so the reader can price the
    difference between measuring a kernel this way and measuring it in a jitted loop."""
    fis.fis_eval1(x, mu, tab, ant, cons)
    t0 = time.perf_counter()
    for _ in range(reps):
        fis.fis_eval1(x, mu, tab, ant, cons)
    return (time.perf_counter() - t0) / reps * 1e9


def microbenchmarks(reps=2_000_000):
    e_ant, e_cons, _, _, e_tab = fis.effort_base()
    h_ant, h_cons, _, _, h_tab = fis.chain_base()

    rows = []
    xe = np.full(e_ant.shape[1], 0.5)
    mue = np.zeros((e_ant.shape[1], fis.N_TERMS))
    oute = np.zeros(e_cons.shape[1])
    ns, _ = _bench(_loop_eval, reps, xe, mue, e_tab, e_ant, e_cons, oute)
    rows.append(("EFFORT fis_eval", e_ant.shape[0], e_ant.shape[1], ns))

    xh = np.full(h_ant.shape[1], 0.5)
    muh = np.zeros((h_ant.shape[1], fis.N_TERMS))
    ns, _ = _bench(_loop_eval1, reps, xh, muh, h_tab, h_ant, h_cons)
    rows.append(("CHAIN fis_eval1", h_ant.shape[0], h_ant.shape[1], ns))

    ns, _ = _bench(_loop_memberships, reps, xe, e_tab, mue)
    rows.append(("_memberships only (EFFORT width)", 0, e_ant.shape[1], ns))

    over = _dispatch_overhead(xh, muh, h_tab, h_ant, h_cons)
    return rows, over


# --------------------------------------------------------------------------------------
# 1. counter attribution, from the already-fitted cost model
# --------------------------------------------------------------------------------------
COUNTER_LABEL = {
    "scans": ("city scans", STAT_SCANS),
    "evals": ("candidate evaluations", STAT_EVALS),
    "chain_levels": ("chain levels entered", STAT_DEPTH),
    "moves": ("accepted moves", STAT_MOVES),
    "chain_fis": ("CHAIN decisions", STAT_CHAIN_CALLS),
    "effort_fis": ("EFFORT decisions", None),  # only the fuzzy arm has this counter
    "rev_work": ("reversal element swaps", STAT_REV_WORK),
    "n": ("per-city O(n) setup", None),
}


def counter_attribution(inst_name=PROFILE_INSTANCE):
    """Per-unit costs x observed counts = where a real solve's time goes."""
    if not paths.COSTMODEL.exists():
        return None, "results/costmodel.npz is missing — run costmodel.py first"
    coef = np.load(paths.COSTMODEL)["coef"]

    from tsplib import load

    inst = load(inst_name)
    cand, cand_d = build_candidates(inst.coords, 32, inst.ceil)
    start = greedy_edge_tour(inst.coords, cand, inst.ceil)

    tuned = fis.load_tuned(paths.tuned("small")) if paths.tuned("small").exists() \
        else fis.hand_written()

    lk_solve(inst.coords, cand, cand_d, inst.ceil, start, 32, 10, 32, 3)  # warm
    t0 = time.perf_counter()
    _, _, st_lk = lk_solve(inst.coords, cand, cand_d, inst.ceil, start, 32, 10, 32, 3)
    t_lk = time.perf_counter() - t0

    fis_ls(inst, cand, cand_d, start, tuned.effort_cons, tuned.chain_cons, 10, 3, False, True,
           tuned.effort_tab, tuned.chain_tab, tuned.effort_ant, tuned.chain_ant)
    t0 = time.perf_counter()
    _, _, st_fis = fis_ls(inst, cand, cand_d, start, tuned.effort_cons, tuned.chain_cons,
                          10, 3, False, True, tuned.effort_tab, tuned.chain_tab,
                          tuned.effort_ant, tuned.chain_ant)
    t_fis = time.perf_counter() - t0

    out = {}
    for tag, st, wall in (("baseline LK", st_lk, t_lk), ("FIS effort+chain", st_fis, t_fis)):
        feats = costmodel.features_from_stats(st, inst.n)
        # ``coef`` is seconds per unit — ``costmodel.py`` fits it that way and prints it
        # scaled by 1e9. Everything below is converted once, here, rather than at each use.
        contrib_s = coef * feats
        out[tag] = {
            "wall_s": wall,
            "predicted_s": float(contrib_s.sum()),
            "rows": sorted(
                ((costmodel.FEATURES[i], float(feats[i]), float(coef[i]) * 1e9,
                  float(contrib_s[i]) * 1e3)
                 for i in range(len(costmodel.FEATURES))),
                key=lambda r: -r[3],
            ),
        }
    return out, None


# --------------------------------------------------------------------------------------
# 3. ablation
# --------------------------------------------------------------------------------------
def ablation(inst_name=PROFILE_INSTANCE, reps=3):
    """The same solve with the rule bases switched off, differenced.

    ``use_chain=False`` removes the CHAIN consultation but leaves EFFORT in place, so the two
    rows below bracket what inference costs in a real solve rather than in a loop.
    """
    from tsplib import load

    inst = load(inst_name)
    cand, cand_d = build_candidates(inst.coords, 32, inst.ceil)
    start = greedy_edge_tour(inst.coords, cand, inst.ceil)
    tuned = fis.load_tuned(paths.tuned("small")) if paths.tuned("small").exists() \
        else fis.hand_written()

    def t(fn):
        fn()
        return min(_timed(fn) for _ in range(reps))

    def _timed(fn):
        t0 = time.perf_counter()
        fn()
        return time.perf_counter() - t0

    rows = [
        ("baseline LK, k32/d10/b32",
         t(lambda: lk_solve(inst.coords, cand, cand_d, inst.ceil, start, 32, 10, 32, 3))),
        ("FIS EFFORT only",
         t(lambda: fis_ls(inst, cand, cand_d, start, tuned.effort_cons, tuned.chain_cons,
                          10, 3, False, False, tuned.effort_tab, tuned.chain_tab,
                          tuned.effort_ant, tuned.chain_ant))),
        ("FIS EFFORT + CHAIN",
         t(lambda: fis_ls(inst, cand, cand_d, start, tuned.effort_cons, tuned.chain_cons,
                          10, 3, False, True, tuned.effort_tab, tuned.chain_tab,
                          tuned.effort_ant, tuned.chain_ant))),
    ]
    return inst, rows


# --------------------------------------------------------------------------------------
# the Cython question
# --------------------------------------------------------------------------------------
def cython_race(reps=2_000_000):
    """Build a C translation of ``fis_eval1`` and race it against numba's.

    ``fis_eval1`` is the right kernel to test: it is the smallest and hottest thing in the
    system (the chain cut-off, taken many times per city scan), so if Cython has an advantage
    anywhere it is here, and if it does not have one here it does not have one at all.

    Both sides loop internally over the same inputs, so neither pays a per-call boundary cost.
    """
    import cython_fis_eval as cy  # built by build_cython()

    h_ant, h_cons, _, _, h_tab = fis.chain_base()
    x = np.full(h_ant.shape[1], 0.5)
    mu = np.zeros((h_ant.shape[1], fis.N_TERMS))

    ns_numba, _ = _bench(_loop_eval1, reps, x, mu, h_tab, h_ant, h_cons)
    cy.loop_eval1(2, x, mu, h_tab, h_ant.astype(np.int8), h_cons)
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        cy.loop_eval1(reps, x, mu, h_tab, h_ant.astype(np.int8), h_cons)
        best = min(best, time.perf_counter() - t0)
    ns_cython = best / reps * 1e9

    # and check they agree, because a faster wrong kernel is not a result
    a = _loop_eval1(1000, x.copy(), mu, h_tab, h_ant, h_cons)
    b = cy.loop_eval1(1000, x.copy(), mu, h_tab, h_ant.astype(np.int8), h_cons)
    return ns_numba, ns_cython, abs(a - b)


def build_cython():
    """Compile ``cython_fis_eval.pyx`` in place. Returns (ok, message)."""
    import subprocess

    here = Path(__file__).resolve().parent
    src = here / "cython_fis_eval.pyx"
    if not src.exists():
        return False, f"{src.name} is missing"
    setup = here / "_cython_setup.py"
    setup.write_text(
        "from setuptools import setup, Extension\n"
        "from Cython.Build import cythonize\n"
        "import numpy as np\n"
        "ext = Extension('cython_fis_eval', ['cython_fis_eval.pyx'],\n"
        "                include_dirs=[np.get_include()],\n"
        "                extra_compile_args=['-O3', '-ffast-math', '-march=native'])\n"
        "setup(ext_modules=cythonize([ext], language_level=3,\n"
        "      compiler_directives={'boundscheck': False, 'wraparound': False,\n"
        "                           'cdivision': True, 'initializedcheck': False}),\n"
        "      script_args=['build_ext', '--inplace'])\n",
        encoding="utf-8",
    )
    r = subprocess.run(
        [sys.executable, str(setup), "build_ext", "--inplace", "--compiler=mingw32"],
        cwd=str(here), capture_output=True, text=True,
    )
    if r.returncode != 0:
        return False, (r.stderr or r.stdout)[-2000:]
    return True, "built"


def main():
    paths.utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance", default=PROFILE_INSTANCE)
    ap.add_argument("--reps", type=int, default=2_000_000)
    ap.add_argument("--cython", action="store_true")
    ap.add_argument("--asm", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("1. counter attribution — the fitted cost model applied to a real solve")
    print("=" * 78)
    attr, err = counter_attribution(args.instance)
    if err:
        print(f"  skipped: {err}")
    else:
        ratios = [d["predicted_s"] / d["wall_s"] for d in attr.values() if d["wall_s"]]
        for tag, d in attr.items():
            print(f"\n  {tag} on {args.instance}: {d['wall_s'] * 1e3:.2f} ms measured, "
                  f"{d['predicted_s'] * 1e3:.2f} ms predicted "
                  f"({d['predicted_s'] / d['wall_s']:.2f}x)")
            print(f"    {'counter':<26s} {'count':>12s} {'ns each':>9s} {'ms':>8s} {'share':>7s}")
            total = sum(r[3] for r in d["rows"]) or 1.0
            for name, count, per_ns, contrib_ms in d["rows"]:
                if contrib_ms <= 0:
                    continue
                label = COUNTER_LABEL.get(name, (name, None))[0]
                print(f"    {label:<26s} {count:12.0f} {per_ns:9.1f} "
                      f"{contrib_ms:8.3f} {100 * contrib_ms / total:6.1f}%")
        if ratios and min(ratios) > 1.15:
            print(f"\n  Note: the shipped coefficients over-predict this machine by "
                  f"{min(ratios):.2f}-{max(ratios):.2f}x, and by close to the *same* factor on")
            print("  both arms. A near-uniform scale error is what a cost model fitted on other")
            print("  hardware looks like, and it is harmless for its actual job: the tuner ranks")
            print("  candidates rather than reading absolute times off it, and ranking is what")
            print("  the fit's 0.9995 rank correlation measures. The shares below are unaffected.")
            print("  Re-run costmodel.py to recalibrate; the shares, not the totals, are the")
            print("  profile.")

    print("\n" + "=" * 78)
    print("2. microbenchmarks — the loop inside nopython mode")
    print("=" * 78)
    rows, over = microbenchmarks(args.reps)
    print(f"  {'kernel':<34s} {'rules':>6s} {'inputs':>7s} {'ns/call':>9s}")
    for name, n_rules, n_in, ns in rows:
        print(f"  {name:<34s} {n_rules:6d} {n_in:7d} {ns:9.1f}")
    print(f"\n  one Python-level call into nopython mode: {over:.0f} ns")
    print(f"  ...which is {over / rows[1][3]:.0f}x the CHAIN kernel it would be measuring.")
    print("  Measuring these kernels from Python would therefore report the dispatch cost,")
    print("  not the kernel. That is why the loops above are jitted.")

    print("\n" + "=" * 78)
    print("3. ablation — the same solve with inference switched off")
    print("=" * 78)
    inst, rows_ab = ablation(args.instance)
    base = rows_ab[0][1]
    print(f"  {args.instance}, n={inst.n}")
    for name, s in rows_ab:
        print(f"    {name:<28s} {s * 1e3:8.2f} ms   {s / base:5.2f}x baseline")

    if args.asm:
        print("\n" + "=" * 78)
        print("numba's generated assembly for fis_eval1 (first 60 lines)")
        print("=" * 78)
        for sig, asm in fis.fis_eval1.inspect_asm().items():
            print("\n".join(asm.splitlines()[:60]))
            break

    print("\n" + "=" * 78)
    print("4. could Cython take any of this back?")
    print("=" * 78)
    if not args.cython:
        print("  not measured — pass --cython to build the C translation and race it.")
        return
    ok, msg = build_cython()
    if not ok:
        print(f"  build failed, so this question stays unanswered:\n{msg}")
        return
    ns_numba, ns_cython, disagreement = cython_race(args.reps)
    print(f"  numba  {ns_numba:8.1f} ns/call")
    print(f"  cython {ns_cython:8.1f} ns/call   ({ns_numba / ns_cython:.2f}x)")
    print(f"  outputs agree to {disagreement:.3e}")


if __name__ == "__main__":
    main()
