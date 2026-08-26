# The Hard-Case Map: Geometric Bridges, Where No Repair Applies

**Date**: 2026-08-25
**Status**: Complete — two dose-response sweeps, reproducible via `python run_hard_cases.py`
**Code**: `nonmetric_data.knn_graph_hubs` / `heavy_tailed_blobs`, `run_hard_cases.py`
**Outputs**: `outputs/hard_cases_results.json`, `outputs/fig18_hard_cases.png`

---

## The point

The violation sweep (NONMETRIC_FINDINGS E3) and the repair study (BRIDGE_REPAIR)
covered **corruption**: entries inconsistent with the rest of the matrix, which
the reverse-TI repair detects and fixes. This note maps the *other* failure
regime — **geometric bridges**, where every distance is a real distance (both
families are exactly metric; `estimate_corruption_rate` reads 0 and
`auto_repair` is verifiably inert on both, pinned in tests) and the damage
comes from actual points sitting between clusters. Nothing here is repairable
by any consistency argument; these cases bound what the pipeline can do
without point-level defenses (ConiVAT-style constraints or hub/outlier
removal).

Scoring convention: hub nodes are labeled −1 and masked from ARI while every
method sees the full matrix — `run_all.py`'s noise convention. Cells average
3 dataset seeds × 5 NERFCM restarts.

## H1 — kNN-graph hubs (masked ARI)

| n_hubs | NERFCM(D) | NERFCM(D*) | gap-cover (k, coverage) |
|---|---|---|---|
| 0 | 1.00 | 1.00 | 1.00 (3, 1.00) |
| 1 | 1.00 | 0.51 | 0.37 (3.3, 0.30) |
| 2 | 0.68 | 0.21 | 0.22 (4.0, 0.26) |
| 3 | 0.54 | 0.17 | 0.22 (4.3, 0.28) |
| 6 | 0.54 | 0.12 | 0.17 (4.3, 0.27) |

- With **zero** hubs the kNN-graph geometry is benign — every method is
  perfect. The entire failure is attributable to the hub nodes.
- **One hub** already halves the minimax methods while NERFCM(D) still scores
  1.00; by **two hubs** relational averaging degrades too (0.68 → 0.54).
  Unlike random shortcut corruption — where NERFCM(D) held 1.00 through
  everything — geometric hubs eventually defeat *both* aggregation
  strategies. A hub is not one bad matrix entry; it is a point whose entire
  row is plausible.
- Ordering of vulnerability: minimax first (a hub chains blocks at merge
  height), averaging second (enough hubs shift every cluster's relational
  mean).

## H2 — Heavy-tailed cluster noise (Student-t, lower df = heavier)

| df | NERFCM(D) | NERFCM(D*) | gap-cover (k, coverage, abstentions/3) |
|---|---|---|---|
| 5.0 | 0.95 | 0.70 | 0.34 (2.0, 0.48, 0) |
| 3.0 | 0.87 | 0.45 | 0.39 (2.3, 0.51, 0) |
| 2.0 | 0.75 | 0.54 | 0.30 (1.3, 0.55, 1) |
| 1.5 | 0.66 | 0.35 | 0.00 (0.7, 0.12, 1) |
| 1.2 | 0.68 | 0.34 | 0.26 (3.0, 0.36, 0) |

- Relational averaging degrades **gracefully** with tail weight (0.95 → 0.66);
  the minimax methods collapse early — tail draws are genuine members sitting
  in the void, i.e. bridge points, and single linkage chains through them at
  even mild tail weight (df=5: cover already at 0.34).
- The persistence gate partially self-defends by **abstaining**: at df ≤ 2 it
  repeatedly selects k=0–2 blocks covering only 12–55% of points, declining to
  assert structure in outlier soup. That is its designed noise behavior
  (SELECTION_METHODS_COMPARISON), now measured in a regime between "clean
  structure" and "pure noise".

## What this settles for the chapter

The 2×2 the thesis can now draw, every cell measured:

| noise model | best representation | evidence |
|---|---|---|
| clean / stretch-type non-metricity | D* + gated selection | NONMETRIC E2–E3 |
| sparse deflation corruption | D* + reverse-TI repair (auto-q) | BRIDGE_REPAIR R1–R4 |
| geometric bridges: few hubs | NERFCM on raw D | H1 (1 hub: 1.00 vs 0.37) |
| geometric bridges: many hubs / heavy tails | nothing here survives; point-level defenses needed | H1 (≥2 hubs), H2 |

Follow-ups: ConiVAT-style constraint pruning is the natural candidate for the
last row (its whole design targets chaining through real points); hub detection
via degree/centrality of the kNN graph is the cheaper heuristic. Neither is
attempted here.
