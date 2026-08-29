# Large-scale regression pilot

Appendix A.7 (`research/proposal-defense/prose/appendix.md`) found that this
document's regression story has no large dataset in any form — Concrete
(1,030 rows) carries every regression table on its own. This directory is the
first pass at closing that gap: a **pilot**, not a committed measurement.

**Status: exploratory, single-seed, not yet a `reproduce/tables/` generator.**
Nothing here has run the ten-seed floor `reproduce/tables/` generators hold to
(Goal G4a), and neither dataset comes from its canonical source (see below).
Read `RESULTS_2026-08-05.md` before quoting anything from this directory in
the prose.

## What's here

- `_datasets.py` — loaders for California Housing and UCI Superconductivity.
  Fetch-and-cache, same pattern as `reproduce/tables/_fuzzy_models.py`'s
  `load_concrete`.
- `mog_top_p_sweep.py` — decorrelate (`sklearn.cluster.FeatureAgglomeration`)
  + sweep `MixtureOfGaussiansFuzzyRegressor`'s `top_p` threshold.
- `model_family_pilot.py` — the same model family Table 6.1 compares on
  Concrete (flat MoG, fuzzy tree, HME, CART, Random Forest, M5), run on both
  candidates at their raw features, Table 6.1's own convention.
- `feature_expansion.py` — `AgglomerativeFeatureExpansion`: agglomerative
  decorrelation + differentiation-score ranking, then an **iterative expansion**
  over the nested top-k prefixes to find the *smallest* model that is good
  enough — by bisection when a target is given, by plateau detection otherwise.
  See "Iterative feature expansion" below.
- `test_feature_expansion.py` — network-free tests for the above (synthetic
  data only): they assert the searched k equals the full-scan k while touching
  fewer candidates. `uv run --project tribble-fis --with pytest python -m
  pytest reproduce/regression_scale/test_feature_expansion.py`.
- `RESULTS_2026-08-05.md` — the write-up: findings, a caught-and-fixed bug,
  and what is still open.

## Iterative feature expansion

`mog_top_p_sweep.py` sweeps a threshold and `table_a1_feature_scoring.py`
sweeps a fixed grid of feature counts — both pay one model fit per point they
report. When the only question is *"what is the smallest feature set that still
clears the bar?"*, most of those fits are wasted: on PhiUSIIL the answer is one
feature (Appendix A.2), yet a linear scan fits at every count to discover it.

`feature_expansion.py` answers that question directly. It reuses the same two
pieces as the rest of this directory — `FeatureAgglomeration` decorrelation and
the `calculate_gaussian_correlation` differentiation ranking — then, because
the ranking is fixed once (on the training split only) and the top-k prefixes
are therefore *nested*, it searches the accuracy-vs-k curve instead of scanning
it:

- **target mode** (`select(target=...)`) — galloping bracket + bisection for the
  smallest k reaching the target, ~log₂(k\*) fits instead of k\* of them.
- **plateau mode** (`select()`) — expand one feature at a time, stop once
  `patience` consecutive additions each buy less than `plateau_tol`, and report
  the *knee*: the smallest k already within `plateau_tol` of the best score
  seen.

Every evaluated k is cached, so the two strategies (and repeated `select()`
calls at different targets on the same fitted object) never re-fit a k they have
already seen. `select(..., verify_scan=True)` additionally fits every k and
asserts the searched answer matches a full scan — a guard for datasets whose
curve may not be monotone enough for bisection.

```python
sel = AgglomerativeFeatureExpansion(task="classification")
sel.fit(X, y)
res = sel.select(target=0.97)      # smallest k with accuracy >= 0.97
print(res.k, res.features, res.savings)   # e.g. 2  [...]  2/19 fits
res = sel.select()                 # or: smallest k at the plateau knee
```

The class is model-agnostic — pass `model_factory=lambda seed: MyEstimator(...)`
to drive any estimator that fits on the selected columns; the default is a
Tribble classifier/regressor configured to use every column it is handed.

## Data provenance — read this before re-running anything

Both loaders pull from a GitHub-hosted mirror, **not** the canonical source:

| Dataset | Canonical source | Blocked how | Mirror used |
|---|---|---|---|
| California Housing | `archive.ics.uci.edu` (via `sklearn.fetch_california_housing`'s figshare backend) | egress policy denial (403 on CONNECT, confirmed via the proxy status endpoint — not transient) | `raw.githubusercontent.com/ageron/handson-ml2` |
| Superconductivity | `archive.ics.uci.edu/dataset/464` | same | `raw.githubusercontent.com/monica110394/...` |

Both mirrors were verified only by row/column count against the known
canonical shapes (20,640 rows; 21,263 × 81) — not by checksum, not by
provenance. **Re-point `_datasets.py` at the canonical UCI source on a host
that can reach it before either dataset goes into `reproduce/tables/`.**

Also note: the California Housing mirror is the *original* 1997 StatLib file
(9 numeric + 1 categorical column), not sklearn's derived 8-feature version
(`AveRooms`/`AveBedrms`/`AveOccup` ratios). `load_housing()` drops the
categorical column and keeps the 8 remaining numeric ones, which happen to
match sklearn's feature count without being the same derivation.

## Running

Needs the `tribble-fis` submodule checked out (or `PILOT_TRIBBLE_FIS` pointed
at a clone of it) and, for `model_family_pilot.py`, its `tribble-tree`
subdirectory for `fuzzytree`:

```
PILOT_TRIBBLE_FIS=/path/to/tribble-fis \
    uv run --project tribble-fis python reproduce/regression_scale/mog_top_p_sweep.py
PILOT_TRIBBLE_FIS=/path/to/tribble-fis \
    uv run --project tribble-fis python reproduce/regression_scale/model_family_pilot.py
uv run --project tribble-fis python reproduce/regression_scale/feature_expansion.py
```

`feature_expansion.py`'s built-in demonstration self-runs on synthetic data
(both classification and regression) with no network; its PhiUSIIL and
California-Housing/Superconductivity demos run additionally when those datasets
are reachable, and skip with a printed reason when they are not.
