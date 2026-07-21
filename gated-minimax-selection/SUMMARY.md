# Project Summary — Gated-Minimax Selection & Multi-Scale Persistence

*Prepared for advisor review. One-page orientation; links point to the detailed
documents.*

---

## The problem

Given only a dissimilarity structure (no assumption of vector coordinates or
Gaussian clusters), extract **fuzzy membership functions** and a **cluster count**
from the single-linkage / iVAT minimax hierarchy — without asking the user to
pre-specify *k*, and without the chaining and scale artifacts that break naive
single-linkage.

Everything is dependency-light (numpy + scipy + scikit-learn) and fully
reproducible: `python run_all.py` regenerates every number and figure from seed.

---

## What has been established

1. **The minimax (iVAT/bottleneck) transform D\* is the workhorse.** It sharpens
   block structure and rescues methods that fail on raw distances — e.g. NERFCM
   goes from ARI 0.02 → 1.00 on concentric rings under D\*. (`FINDINGS.md`)

2. **Selection is reframed as a persistence-gated set-cover.** Instead of "pick
   the top-*k* persistent blocks," `coverage_cover` admits blocks whose absolute
   persistence is a statistical outlier (MAD gate) and greedily covers the data,
   with *k* an **output**. This is scale-invariant for single-level structure:
   ARI holds ≈0.98 across a **30× cluster-spread ratio**.
   (`SELECTION_METHODS_COMPARISON.md`)

3. **Membership-function extraction has three validated variants** (Ruspini
   partition-of-unity, auto-tuned, and interpretable feature-space rules), with
   honest scope limits. (`EXPLORATION_SUMMARY.md`, `FINDINGS_exploration.md`)

4. **The approach extends to purely relational data** (distance-matrix-only, no
   coordinates). (`RELATIONDATA.md`)

---

## Headline new contribution — multi-scale persistence selection (Option D)

**The gap:** the gated set-cover returns a single *flat* partition. It must commit
to one granularity, so it **cannot represent nested structure** (a diffuse
super-cluster containing tighter sub-clusters, where both levels are real).

**The idea:** in single-linkage, a block's *birth height* is an inverse proxy for
the local density at which it becomes a distinct cluster. Stratifying blocks by
gaps in the (log-)birth-height axis therefore stratifies by **density scale**. We
run the *same* persistence gate + set-cover **within each band**, returning a
**hierarchy of partitions** — the number of scales is itself an output.

**Result** (mean ARI averaged over *all* ground-truth levels):

| dataset | levels | flat set-cover | **multi-scale** |
|---|---|---|---|
| `nested_gaussians` | 6 fine / 2 coarse | 0.66 | **1.00** |
| `three_level_hierarchy` | 8 / 4 / 2 | 0.58 | **1.00** |
| `density_hierarchy` | 4 / 2 | 0.75 | **1.00** |

On `three_level_hierarchy` it recovers granularities **[8, 4, 2]** — each
discovered band lands on exactly one ground-truth level at ARI 1.0 — *without
being told the number of levels or clusters*. On single-scale data it discovers
one band and reproduces the flat baseline (strict generalization); on noise it
returns zero bands.

**Why it is defensible, not a tuned demo:**
- the significance test is *literally* the flat selector's gate, factored out —
  so this generalizes the existing method with a proved single-band reduction;
- an included **falsification experiment** documents the regime where flat is
  *not* beaten (single-level scale contrast), so the nested-structure claim is
  not a strawman;
- band discovery is justified by the birth-height ↔ density correspondence,
  linking it to persistent-homology and density-aware directions.

Full write-up, theory, limitations, and dissertation roadmap:
**`OPTION_D_MULTISCALE.md`**.

---

## Key documents (reading order for review)

| Document | What it covers |
|---|---|
| **`SUMMARY.md`** | this page |
| **`OPTION_D_MULTISCALE.md`** | ⭐ the new multi-scale contribution — theory, results, limits, future work |
| `SELECTION_METHODS_COMPARISON.md` | how block selection was reframed as a gated set-cover; method bake-off |
| `FINDINGS.md` | core battery, the two mappings, why D\* matters |
| `EXPLORATION_SUMMARY.md` | Options A–D at a glance (membership-function variants) |
| `RELATIONDATA.md` | extension to distance-matrix-only (relational) data |

## Code map

| File | Role |
|---|---|
| `run_all.py` | ⭐ master reproducible pipeline — regenerates `outputs/results.json` + all figures |
| `ivat_mf.py` | minimax (iVAT) transform, membership mappings, defuzzifiers |
| `selection.py` | persistence-gated set-cover (`select_coverage_cover`) and baselines |
| `multiscale_persistence.py` | multi-scale selector (`select_multiscale`), band discovery, shared gate |
| `battery_hierarchical.py` | nested/multi-level synthetic datasets with per-level ground truth |
| `run_multiscale.py` | standalone Option D experiment harness (subset of `run_all.py`) |

## Reproducing

```bash
python run_all.py            # all numbers -> outputs/results.json, all figures -> outputs/
python run_all.py --high-res # 300-dpi figures for reports
python run_multiscale.py     # just the Option D experiments, verbose
```

The multi-scale headline figure is `outputs/fig8_multiscale_hierarchy.png`.
