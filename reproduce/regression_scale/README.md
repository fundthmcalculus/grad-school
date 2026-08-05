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
- `RESULTS_2026-08-05.md` — the write-up: findings, a caught-and-fixed bug,
  and what is still open.

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
```
