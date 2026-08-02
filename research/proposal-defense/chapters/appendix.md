# Appendix

**Status:** Outline · placeholder
**Mirrors:** Pickering Appendix (EU AI Act excerpt; supplementary figures).

---

## A.1 Supplementary figures

- **A.1 VAT/iVAT RDI galleries** — shuttle 58K, psych-eval 135K, synthetic circular-cities; Prim-MST block diagram.
- **A.2 Selection & multi-scale figures** — fig1–fig11 from `gated-minimax-selection/outputs/` (datasets, transform heatmaps, persistence curves, membership plots, multiscale hierarchy, selection comparison, scaling).
- **A.3 FIS visualizations** — GMM membership plots; fuzzy-tree renderings (`plot_fuzzy_tree`); HME structure; Ruspini partitions.

## A.2 Extended results tables

- Full adversarial-eval ARI grids; stitch-ablation grids; selection-method bake-off; scaling tables; full benchmark suite (turbine, WEC, wine, DARWIN, BETH, IoT-botnet, power consumption).

## A.3 Optimization-engine details (tribble-opt) — CONFIRMED HOME

Per author decision, the optimization engine is supporting infrastructure, not core dissertation, and lives here (not Ch 2). Contents:
- **Metaheuristics:** continuous/combinatorial ACO, GA, PSO, gradient descent.
- **TSP local search:** Nearest-Neighbor, 2-opt/3-opt/Or-opt, **Lin-Kernighan** (candidate-restricted), dual numba/Cython backends; LK quality + timing tables (`LK_PERFORMANCE_REPORT.md`).
- **Quality-Diversity:** MAP-Elites / CVT-MAP-Elites (Vassiliades 2018), Iso+LineDD variation; Pareto reporting (NSGA-II/SPEA2/MOEA/D) as a reporting layer (`QD_PARETO_PLAN.md`).
- **Performance-engineering study:** 15-item ranked findings — truncnorm 177×, ship-once 7.5×, njit local search ~370× (`PERFORMANCE_REPORT.md`); Cython-vs-numba decision analysis (`CYTHON_ANALYSIS.md`).
- Cross-reference the brief motivating mention in Ch 2 §2.5.
- These are also the **standalone-paper candidates** listed in the master outline (perf study; QD layer; exact GPU VAT; LK + VAT-blocked TSP).

## A.4 Feature scoring: why the composite metric earned its keep

- The Chapter 4 construction ranks features before it builds anything; rule
  count, clause count and readability all follow from that ranking.
- The original scorer combined four separation measures via the mean of their
  arithmetic and geometric means — a consensus rule, since the geometric mean
  collapses unless every measure agrees.
- An upstream refactor replaced it with a single metric (Bhattacharyya). On
  PhiUSIIL that metric ranks the most informative feature outside its top 20.
- Table A.1: rankings by scorer. Table A.2: accuracy and fit time vs features
  kept. One feature reaches 0.9969 under the composite or Wasserstein; the
  default's top two score 0.4251.
- Mechanism: Bhattacharyya is the only one of the three that assumes a
  distribution (Gaussian fit per class), and the key feature is not Gaussian.
- The point for the thesis: interpretability is a property of the ranking, not
  of the architecture. A bad ranking costs readability far more than accuracy.
- Written from tribble-fis #49/#50.

## A.5 Reproducibility

- Repo/commit pointers: `tribble-cluster`, `tribble-fis`, `tribble-opt`, `gated-minimax-selection`; master drivers (`run_all.py`, benchmark scripts); hardware notes.

---

### Open items
- ~~Decide whether tribble-opt is §2.5 or A.3~~ → **RESOLVED: Appendix A.3** (author decision).
- Regenerate publication-quality figures at consistent style/size.
