"""Build script for the ode_kernels Cython extensions.

    python setup.py build_ext --inplace

Compiler flags, and why:

* ``-O3 -funroll-loops`` -- standard aggressive optimization; loop unrolling
  helps the small, fixed-trip-count stage loops (n_stages is often <= 13).
* ``-march=native`` -- lets GCC/Clang target whatever SIMD/FMA extensions the
  build machine actually has, which matters here specifically because the
  stage-combination kernels are written around ``libc.math.fma``: without at
  least ``-mfma`` that call still gets a *correctly rounded* software fma
  (still more accurate than separate multiply+add), but only with an FMA3-
  capable target and this flag does it lower to the single hardware
  instruction. Probed at build time and dropped if the compiler rejects it
  (e.g. cross-compiling for a different target), rather than hard-failing
  the build.
* ``-fno-math-errno`` -- safe to always enable (it only stops libm from
  setting ``errno`` on domain errors, which nothing here checks) and it lets
  the compiler treat calls like ``exp``/``log2`` as pure enough to reorder
  and inline more aggressively.
* Deliberately *not* ``-ffast-math``: it would relax exactly the IEEE
  semantics the adaptive step controller depends on (finite/NaN comparisons
  used to detect blow-up, exact ``error_norm == 0`` checks), trading step-
  control correctness for a speedup that doesn't actually help here since
  the hot loop is bandwidth- and call-overhead-bound, not
  transcendental-function-bound.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from setuptools import Extension, setup


def _compiler_supports(flag: str) -> bool:
    import distutils.ccompiler

    compiler = distutils.ccompiler.new_compiler()
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.c"
        src.write_text("int main(void) { return 0; }\n")
        try:
            compiler.compile([str(src)], output_dir=d, extra_postargs=[flag])
            return True
        except Exception:
            return False


def _extra_compile_args() -> list[str]:
    args = ["-O3", "-funroll-loops", "-fno-math-errno"]
    for candidate in ("-march=native",):
        if _compiler_supports(candidate):
            args.append(candidate)
        else:
            print(
                f"[ode_kernels/setup.py] compiler rejects {candidate!r}, skipping",
                file=sys.stderr,
            )
    return args


EXTRA_COMPILE_ARGS = _extra_compile_args()

extensions = [
    Extension(
        "_rk_kernels",
        ["_rk_kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
    ),
    Extension(
        "_expint",
        ["_expint.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=EXTRA_COMPILE_ARGS,
    ),
]

if __name__ == "__main__":
    from Cython.Build import cythonize

    setup(
        name="ode_kernels",
        ext_modules=cythonize(
            extensions,
            compiler_directives={"language_level": "3"},
            annotate=True,
        ),
        zip_safe=False,
    )
