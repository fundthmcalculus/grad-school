#!/usr/bin/env bash
# Source this before running anything in reproduce/ on a Windows host.
#
#   source reproduce/hostenv.sh
#
# It is a no-op on Linux/macOS, and a no-op on Windows hosts that still have
# MSVC. It exists because this host lost its Visual C++ toolchain (the
# `Microsoft Visual Studio/2022` directory is present but empty), and three of
# the four environments the harness uses cannot be built without a C compiler:
#
#   tribble-opt      optimizers.combinatorial._tsp_cython, optimizers.benchmarks._bench_cython
#   tribble-cluster  tribbleclustering.pcvat / .cfcm / .clk
#   tribble-fis      pulls both of the above in as dependencies
#
# Without this, every `uv run --project ...` invocation in reproduce/ fails
# during dependency resolution and no generator gets as far as importing numpy.
#
# ---------------------------------------------------------------------------
# Three separate defects have to be worked around to get a build. Each was
# reproduced before being worked around; none is a guess.
#
# (1) `optional=True` does not survive cythonize().  tribble-opt declares both
#     of its extensions optional so a missing compiler degrades to the numba
#     fallback rather than failing the install -- setup.py's docstring says so
#     explicitly. Cython's cythonize() constructs a NEW Extension from a fixed
#     list of distutils settings that does not include `optional`, so the flag
#     is silently dropped. Measured directly:
#
#         before cythonize: optional = True
#         after  cythonize: optional = False
#         same object: False
#
#     Since Cython is in tribble-opt's build-system.requires it is always
#     present, so the documented graceful degradation has never actually been
#     reachable. Filed as checklist item B14.
#
# (2) MSVC flags are chosen by platform, not by compiler.  tribble-opt's
#     setup.py branches on platform.system(), so on Windows it emits `/O2
#     /openmp` whatever compiler is in use. gcc reads those as input filenames
#     ("linker input file not found: /O2"). tribble-cluster gets this right --
#     its build_ext subclass branches on self.compiler.compiler_type -- which
#     is why only tribble-opt needs the shim. Also B14.
#
#     tools/ccshim/cc_shim.exe translates the MSVC spellings to their gcc
#     equivalents and adds the -fopenmp that gcc needs at LINK time and MSVC
#     does not (setup.py sets extra_link_args=[] for exactly that reason).
#
# (3) The editable build path drops DIST_EXTRA_CONFIG.  `[build_ext] compiler =
#     mingw32` supplied via DIST_EXTRA_CONFIG is honoured by
#     setuptools.build_meta.build_wheel and ignored by build_editable, so
#     tribble-clustering built fine as a git dependency of tribble-fis and
#     failed as the project itself, in the same shell, seconds apart. UV_NO_EDITABLE
#     forces the wheel path for both. The only cost is that submodule source
#     edits need a re-run to take effect, which is correct for reproduction
#     work anyway.
# ---------------------------------------------------------------------------

_hostenv_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

_hostenv_skip() {
    case "$(uname -s 2>/dev/null)" in
        MINGW*|MSYS*|CYGWIN*|Windows_NT) return 1 ;;
        *) return 0 ;;
    esac
}

if _hostenv_skip; then
    return 0 2>/dev/null || exit 0
fi

# A host with a real MSVC needs none of this, and forcing mingw on one would be
# a downgrade -- it would silently change the compiler behind every archived
# number. Detect and stand down.
if command -v cl >/dev/null 2>&1; then
    echo "hostenv: MSVC found on PATH; no shim needed."
    return 0 2>/dev/null || exit 0
fi

# Locate a gcc. w64devkit is what this host has; anything mingw-w64 works.
if ! command -v gcc >/dev/null 2>&1; then
    for _cand in /c/bin/w64devkit/bin /c/msys64/mingw64/bin /c/mingw64/bin; do
        if [ -x "$_cand/gcc.exe" ]; then export PATH="$_cand:$PATH"; break; fi
    done
fi
if ! command -v gcc >/dev/null 2>&1; then
    echo "hostenv: ERROR -- no MSVC and no gcc. Compiled extensions cannot be built." >&2
    echo "hostenv: install MSVC Build Tools, or w64devkit, then re-source this file." >&2
    return 1 2>/dev/null || exit 1
fi

# Build the flag-translating shim on first use. It is a single file with no
# dependencies; rebuilding it is cheaper than reasoning about whether the
# checked-in binary matches the source.
_hostenv_shim="$_hostenv_root/tools/ccshim/cc_shim.exe"
if [ ! -x "$_hostenv_shim" ] || [ "$_hostenv_root/tools/ccshim/cc_shim.c" -nt "$_hostenv_shim" ]; then
    echo "hostenv: building cc_shim.exe"
    gcc -O2 -Wall -o "$_hostenv_shim" "$_hostenv_root/tools/ccshim/cc_shim.c" || {
        echo "hostenv: ERROR -- could not build cc_shim.exe" >&2
        return 1 2>/dev/null || exit 1
    }
fi

# distutils splits CC with shlex, which eats Windows backslashes -- a
# backslashed path arrives as C:personalgrad-school... and spawns
# "[WinError 2] The system cannot find the file specified". Forward slashes
# only.
_hostenv_shim_fwd="$(echo "$_hostenv_shim" | tr '\\' '/')"
case "$_hostenv_shim_fwd" in
    /c/*) _hostenv_shim_fwd="C:${_hostenv_shim_fwd#/c}" ;;
esac

_hostenv_cfg="$_hostenv_root/tools/ccshim/dist-extra.cfg"
printf '[build_ext]\ncompiler = mingw32\n' > "$_hostenv_cfg"

export DIST_EXTRA_CONFIG="$(cygpath -w "$_hostenv_cfg" 2>/dev/null || echo "$_hostenv_cfg")"
export CC="$_hostenv_shim_fwd"
export CXX="$_hostenv_shim_fwd"
export CCSHIM_REAL_CC=gcc
export UV_NO_EDITABLE=1

echo "hostenv: mingw32 + cc_shim active (CC=$CC)"
