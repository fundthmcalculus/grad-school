# `wasserstein-fixed-2026-08-22` — a diagnostic arm, NOT a run of record

Every `gauss_math`-based generator here was run at the current pins with exactly
one change: `tribblefis.stats_numba.wasserstein_distance` substituted for
`scipy.stats.wasserstein_distance`, **in process**, by
`reproduce/experiments/run_with_reference_stats.py`. Nothing on disk was patched
and no submodule was touched.

It exists to answer one question — *are the document's numbers wrong, or is the
pin?* — table by table, without waiting for the upstream fix. See checklist
**B14** and [`../WASSERSTEIN_REGRESSION.md`](../WASSERSTEIN_REGRESSION.md).

**Do not cite it.** It is a mixed archive in two ways, both deliberate:

* the tables that do not run through `gauss_math` (`table_3_*`, `table_5_*`,
  `table_a7_*`, `table_4_4*`) are **copied** from `full-2026-08-22`, because the
  substitution cannot reach them and re-running them would only add noise;
* it therefore describes a build of `tribble-fis` that does not exist. When the
  fix lands upstream, re-run the suite normally and delete this.

Produced by the same host and toolchain as `full-2026-08-22`, including the
gcc-built kernels of checklist **B15**.
