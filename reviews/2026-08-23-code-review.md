# Code review — 2026-08-23 (grad-school + submodules)

Autonomous review pass across `reproduce/`, `tribble-fis`, `tribble-opt`, `tribble-cluster`.
Findings gathered by parallel review agents, each finding then verified against source.
Severity: high / med / low. **QW** = safe self-contained quick win.

PR plan at the bottom. tribble-cluster findings appended when its agent returns.

---

## tribble-opt (optimization engine) — highest-severity bugs found here

| # | file:line | sev | QW | finding |
|---|---|---|---|---|
| O1 | combinatorial/ga.py:254-261 | **high** | ✔ | GA crossover always returns the identity permutation — `np.unique(..., return_index=True)` sorts, so `child = child[idx]` rebuilds `[0..N-1]`, discarding both parents. GA is effectively mutation-only. Fix: `child[np.sort(idx)]`. |
| O2 | combinatorial/{aco,aco_mst,ga}.py | **high** | �’ | Whole combinatorial family uses thread-unsafe legacy `np.random` under `joblib(prefer=threads)`; ignores `core.random`/`spawn_streams`. Non-reproducible at n_jobs>1; no seed inheritance under loky. Cross-cutting (not QW). |
| O3 | continuous/step.py:83,96 | **high** | ✔ | `optimize_whole_solution_deck=True` crashes on entry: `min([])` on first iter, `best_soln_value[-1]` before the empty guard, and `local_perturb_optim` never called. |
| O4 | combinatorial/mtsp.py:65,83 | med | ✔ | KMeans/SpectralClustering constructed with no `random_state` → MTSP tours vary run-to-run. Pass `random_state=get_seed()`. |
| O5 | combinatorial/ga.py:194-198 | med | ✔ | 2-opt gain test measures segment reversal but the applied move only swaps endpoints → "improving" move can lengthen the tour. Reverse the segment. |
| O6 | continuous/gd.py:151 | med | ✔ | `parallel_discrete_search` draws via legacy `np.random.randint`, ignoring active worker stream. Use `rng().integers`. |
| O7 | solution_deck.py:321-345 | low | ✔ | `lloyds_algorithm_points` dead + global np.random + lru_cache. Delete. |
| O8 | combinatorial/strategy.py:695,709 | low | ✔ | `ConvexHullTSP` "retry once" guard `restarted` never set True. |
| O9 | continuous/optimizer_strategy.py:69,81 | low | ✔ | uses Python global `random`; mutable default `RandomOptimizerSelection()` at import. |
| O10 | base.py:526 + 3 others | low | ✔ | early-stop compares only window endpoints w/ rtol 1e-2 → mis-stops. |
| O11 | combinatorial/aco_mst.py:170-172 | low | ✔ | inverse-CDF `argmin(new_p>cum_p)` picks index 0 on tail overflow. Use searchsorted like aco.py. |

Solid: `core/random.py`, `core/parallel.py`, continuous GA/ACO/PSO determinism, cvt/descriptor/variation seeding, PSO velocity clamp.

## tribble-fis numeric core

| # | file:line | sev | QW | finding |
|---|---|---|---|---|
| C1 | gauss_math.py:1476,1485 | high | ✔ | `generate_synthetic_data` uses unseeded global np.random. Add `random_state` + default_rng. |
| C2 | gauss_data.py:509 (+829,1101) | high | ✔ | `all_output_labels` = `list(set(...))` → hash-order for string labels; drives rule order + argmax tie-break. `sorted(set(...))` (no-op for int labels). |
| C3 | kernel.py:400-450 | high | �’ | `firing_strengths backend="auto"` switches cython/numpy exp per host (~1 ULP), amplified by argmax/BIC. Pin backend for repro runs. Not QW. |
| C4 | regression.py/refine.py lstsq | med | �’ | LAPACK gelsd rank-deficient truncation platform-dependent → refine trajectory diverges. Pin BLAS / record stack. Not QW. |
| C5 | stats_numba.py:139-169 | med | ✔ | `_wasserstein_distance_jit` dead + wrong interval formula. Delete. |
| C6 | stats_numba.py:217-276,313-335 | low | ✔ | the two `parallel=True` kernels are NOT a determinism hazard (disjoint writes, sequential reductions, no fastmath) — verified. May drop parallel if cold. |
| C7 | refine.py:335-337 | low | ✔ | `_norm_fs_grad` hardcodes `1e-6` vs shared `ZERO_FIRING_THRESHOLD`. |
| C8 | regression.py:57-60 | low | ✔ | `_rsquared` div-by-zero on constant target → nan silently skips hyperparam update. |
| C9 | gauss_math.py:1392 | low | ✔ | `calculate_top_k_accuracy` unstable argsort on tied firing. `kind="stable"`. |
| C10 | gauss_math.py:485 | low | ✔ | `ThreadPoolExecutor(max_workers=1)` is pure overhead. Use map. |
| C11 | gauss_data.py:417,427 | low | ✔ | `all_features`/`get_mfs_for_feature` set-ordered. sorted()/dict.fromkeys. |
| C12 | gauss_math.py:941 | low | �’ | `compile_model` rebuilt every `tsk_firing_strengths` call (CV/predict). Cache. Not QW. |

## tribble-fis model wrappers / trees

| # | file:line | sev | QW | finding |
|---|---|---|---|---|
| M1 | gaussian_classifier.py:79 | high | ✔ | `TribbleClassifier` caches `anomaly_params` in `__init__`; `set_params`/clone leaves it stale → **GridSearch over norm_conorm is a silent no-op**. Resolve on demand (siblings already do). |
| M2 | it2_classifier.py:238,283; it2_regressor.py:323,390 | high | ✔ | `pd.DataFrame(X.values, columns=feature_names_in_)` relabels columns positionally → predict on reordered columns silently wrong. Select by name. |
| M3 | bsp_fuzzy_classifier.py:219,223 | med | ✔ | `predict` returns object array → `score`/accuracy raises "mix of multiclass and unknown". `.astype(classes_.dtype)`. |
| M4 | ruspini.py:316-361 | med | �’ | no-rule-fired rows get uniform proba → argmax→class 0. Route to explicit fallback. |
| M5 | anfis.py:405-408,466 | med | ✔ | empty/NaN val fold → `best_snapshot` None → AssertionError. Init snapshot before loop. |
| M6 | report.py:149 | med | ✔ | per-label rule count `np.prod` int64 overflows (~5^30). `dtype=float` like line 135. |
| M7 | gauss_plot.py:80-81 | med | ✔ | dropna'd Series indexed by full-length bool → IndexingError. Align first. |
| M8 | it2_refine.py:655 | med | �’ | regressor sub-problem lacks the l2_shrink anchor the classifier/docstring promise. |
| M9 | gt2_regressor.py:229-238 | med | ✔ | `refine_gt2_regressor_antecedents` called without `seed=self.random_state` → random_state ignored. |
| M10 | it2_refine.py:528; gt2_refine.py:461 | med | ~ | bare `except Exception: return 1e6` hides real bugs. Narrow. |
| M11 | trapz_math_fast.py:331 | low | ✔ | iterates `y.unique()` not `sorted()` → row-order-dependent tie-break. |
| M12 | ensemble_fuzzy_classifier.py:90,94 | low | ✔ | every member gets identical seed → members converge identical, killing decorrelation. `+ i`. |
| M13 | gauss_plot.py:339-341 | low | ✔ | random top-k baseline formula wrong for k≥2; should be `k/n_classes`. |
| M14 | gaussian_regressor_memory.py:145-157 | low | �’ | interior-NaN mispairs X/y (positional head slice). |
| M15 | it2_refine/gt2_refine/scaling.py | low | ✔ | dead accepted params: `sub_maxfun`, `max_iterations`, `self.var_`. |

## grad-school reproduce harness

| # | file:line | sev | QW | finding |
|---|---|---|---|---|
| H1 | tables/table_4_8_mf_dedup.py:90-107 | high | �’ | CI uses pop-std + z=1.96 for n=10 → ~22% too narrow → "max-lossless ×" boundary reported smaller than data supports. Needs sample std + t(2.262). Re-run. |
| H2 | tables/table_3_1_pvat_scaling.py:56-77 | high | ✔ | `_resolve_pvat` times whichever of 4 entry points resolves first (different work); not the one preflight checks. Pin one; N/A otherwise; record resolved name. |
| H3 | _fuzzy_models.py:18 (+8 files) | med | ✔ | blanket `warnings.filterwarnings("ignore")` masks the exact convergence/BLAS warnings that flag cross-platform divergence. Narrow to categories / capture to log. |
| H4 | common.py:66 | med | �’ | `agg` uses `pstdev` (÷n) for the "± std" cells → ~5% understated. Switch to sample std (coordinate w/ H1, re-run). |
| H5 | table_4_1_mog_baselines.py:46 | med | �’ | scaler `fit_transform` before split → MoG headline transductively leaked. Document in emitted note / offer leak-free variant. |
| H6 | table_a7_regression_scale.py:83-114 | med | ✔ | feature-survivor chosen by |corr with y| over all rows incl test. Surface in note. |
| H7 | table_4_4_openset.py:222 vs 300 | med | ✔ | θ-sweep refits complement rule on the SAME split the headline already fit (suite's slowest table). Reuse headline `memb`. Extends PR #141. |
| H8 | ~15 table scripts | low | ✔ | duplicated sys.path bootstrap + import prologue. Extract `_bootstrap`. |
| H9 | table_4_1:120; table_5_4:65 | low | ✔ | dead conditional branch; unused `import statistics`. |
| H10 | common.py:271,289 | low | ✔ | `normalized_worst` filters `if v` → drops genuine 0.0 timings as N/A; ties both render "worst". Use `is not None`. |

---

## PR plan (stacked per repo)

**tribble-fis** (stack A→B→C off main):
- PR-A determinism hardening: C1,C2,C7,C9,C11,M11,M12,M9 (+C5 dead). *numerically neutral except tie-order.*
- PR-B wrapper correctness: M1,M2,M3,M6,M7,M13.
- PR-C small cleanups: C8,C10,M15.

**tribble-opt** (stack D→E off main):
- PR-D combinatorial correctness: O1,O3,O5,O11 (+ O8).
- PR-E determinism: O4,O6,O7,O9.
- (O2 large; separate follow-up issue.)

**grad-school** (off quality/review-2026-08-23):
- PR-F harness robustness: H2,H3,H9,H10 (+H6 note).
- PR-G openset fit reuse: H7 (extends #141).
- H1/H4/H5 (stat/leak) → flagged for a re-run decision (change reported numbers) — issue, not silent PR.

Deferred to issues (not quick/safe): C3, C4, C12, O2, M4, M8, M10, M14, H1, H4, H5.

---

## Outcomes (updated as PRs land)

**Shipped:**
- tribble-fis **#181** — determinism hardening: C1,C2,C5,C7,C9,C11,M9,M11,M12 (+tests).
- tribble-fis **#182** (stacked on #181) — wrapper correctness: M1,M3,M6,M7,M13.
- optimizers **#115** — combinatorial correctness: O1 (GA identity),O5,O8,O11.
- grad-school (this branch) — H9 dead-code; findings doc.

**Rejected after testing (NOT bugs):**
- M2 — IT2/GT2 `predict` positional column mapping is the intended sklearn
  positional-predict contract; name-based selection broke `test_gt2_regressor`.

**Deferred to issues (real but not safe/quick — change reported numbers or need a test of an unused path):**
- H1/H4 (pop-std vs sample-std in CI + "± std") — changes reported spreads; needs a coordinated re-run.
- H2 (pvat resolver pin) — changes which function is timed → moves Ch3 numbers.
- H3 (blanket warnings filter) — narrowing is a log-policy call across 9 files.
- H5/H6 (transductive scaling / label-informed feature survivor) — document in emitted notes.
- H10 (0.0-timing truthiness) — current guard doubles as a div-by-zero guard; reworking risks inf.
- C3 (firing backend="auto" ULP drift), C4 (BLAS lstsq), C12 (compile_model caching).
- O2 (combinatorial thread-unsafe np.random), O3 (dead whole-deck branch), O4/O6/O9/O10 (opt determinism).
- M4 (ruspini no-rule argmax), M8 (it2 regressor l2 anchor), M10 (bare except), M14 (memory mispair).
- tribble-cluster: GPU MST tie-break (#1), IVATMeans/FCM/NERFCM (#2-#5) — see next PR.

### Final PR/issue map (this session)
- tribble-fis **#181** determinism · **#182** wrapper correctness (stacked)
- optimizers **#115** combinatorial correctness · **#116** determinism (stacked)
- clustering **#82** NERFCM/FCM/pvat determinism+correctness
- grad-school **#142** review doc + harness dead-code
- Tracking issues for deferred high-value items: optimizers **#117** (thread-unsafe RNG), clustering **#83** (FCM init crash + cfcm parity + ivat over-seg + GPU MST safety)
- Verified-and-rejected: M2 (positional predict contract).
