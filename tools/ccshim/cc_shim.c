/* cc_shim.c -- translate MSVC-style compiler flags to GCC equivalents.
 *
 * Why this exists. `tribble-opt`'s setup.py chooses its optimizer flags from
 * platform.system(), not from the compiler actually in use, so on Windows it
 * always emits MSVC's `/O2 /openmp` -- flags gcc reads as input FILENAMES and
 * dies on ("linker input file not found: /O2"). The extension is declared
 * `optional=True` precisely so a compile failure degrades to the numba
 * fallback, but Cython's cythonize() rebuilds the Extension object and does
 * not carry `optional` across (measured: True in, False out), so the failure
 * is fatal instead. Both are upstream defects; this shim is the host-side
 * workaround that lets the extension build for real rather than be skipped.
 *
 * Installed as CC (not on PATH as `gcc`) so it cannot recurse into itself.
 * The real compiler is read from CCSHIM_REAL_CC, defaulting to `gcc`.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <process.h>

#define BACKSLASH 92

int main(int argc, char **argv) {
    const char *real = getenv("CCSHIM_REAL_CC");
    const char **out;
    int n = 0, i;
    int compiling = 0, linking = 0;

    if (!real) real = "gcc";

    out = (const char **)calloc((size_t)argc + 8, sizeof(char *));
    if (!out) return 1;
    out[n++] = real;

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "-c")) compiling = 1;
        if (!strcmp(a, "-shared")) linking = 1;

        if (!strcmp(a, "/O2") || !strcmp(a, "/Ox")) { out[n++] = "-O2"; continue; }
        if (!strcmp(a, "/O1"))                      { out[n++] = "-Os"; continue; }
        if (!strcmp(a, "/Od"))                      { out[n++] = "-O0"; continue; }
        if (!strcmp(a, "/openmp")) { out[n++] = "-fopenmp"; continue; }
        if (!strcmp(a, "/W3") || !strcmp(a, "/W4")) { out[n++] = "-Wall"; continue; }
        if (!strcmp(a, "/EHsc") || !strcmp(a, "/MD") || !strcmp(a, "/MT")
            || !strcmp(a, "/nologo") || !strcmp(a, "/GL")) { continue; }

        /* Any other MSVC-looking switch is dropped rather than handed to gcc,
           which would treat it as a path and fail. A dropped flag costs a
           slower object file; a passed-through one costs the whole build. The
           guards on '.', a backslash and length keep real POSIX-style paths
           (which distutils does emit) from matching. */
        if (a[0] == '/' && a[1] != '\0' && a[1] != '/'
            && !strchr(a, '.') && !strchr(a, BACKSLASH) && strlen(a) < 12) {
            fprintf(stderr, "cc_shim: dropping unhandled MSVC flag %s\n", a);
            continue;
        }
        out[n++] = a;
    }

    /* setup.py sets extra_link_args=[] on Windows because MSVC's /openmp needs
       no link-time counterpart. gcc does need one: without -fopenmp on the
       link line the .pyd carries undefined GOMP_* symbols and fails to
       import -- which would be worse than not building it, because it fails at
       import time rather than at build time. */
    if (linking && !compiling) out[n++] = "-fopenmp";

    out[n] = NULL;
    return _spawnvp(_P_WAIT, real, (const char *const *)out);
}
