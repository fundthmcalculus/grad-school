# HDBSCAN\* baselines — what survives of contributions 2 and 3

**Date:** 2026-08-27
**Driver:** `run_hdbscan_baselines.py` → `outputs/hdbscan_baselines.{json,md}` (gitignored; regenerate to audit)
**Discharges:** the "required before claiming" item in `research/proposal-defense/PRIOR_ART_CH5.md` §3.
**Environment note:** needs the `hdbscan` contrib package for `leaf` extraction and
`dbscan_clustering`. scikit-learn's `HDBSCAN` has eom/leaf but no cut-distance
accessor. Not in any repo venv; run under an interpreter that has it.

The comparison is anchored on **TKDD 2015 Corollary 3.5**: at `mpts ∈ {1,2}` mutual
reachability equals the input dissimilarity and HDBSCAN\* *is* single-linkage on
those distances. At `min_samples=1` both sides therefore consume the **identical
hierarchy**, and the only thing under test is the extraction rule.

---

## Correction to PRIOR_ART_CH5.md first

That document's §3 "what survives" table reported HDBSCAN EOM at concentric_rings
**0.061 / k=30**, bridged_gaussians **0.452 / k=14**, varying_density **0.750 / k=9**
and presented them as *the* HDBSCAN result. Those are all the
**`min_cluster_size=3`** rows. Sweeping the parameter:

| dataset | eom mcs=3 | eom mcs=5 | eom mcs=10 |
|---|---:|---:|---:|
| concentric_rings | 0.061 / 30 | 0.090 / 20 | **0.863 / 3** |
| bridged_gaussians | 0.452 / 14 | 0.613 / 6 | **0.964 / 2** |
| varying_density | 0.750 / 9 | 0.753 / 8 | **0.980 / 3** |

So "EOM over-segments badly — 30 clusters on rings" described HDBSCAN at its worst
setting, not HDBSCAN. This is the same error class as the earlier SIMD DTW
speedup claim (an 8-thread kernel measured against a library's single-threaded
default): **a baseline reported at an unfavourable configuration.** The
size-weighted-versus-unweighted-stability story in that section is not supported by
these numbers and should not be claimed.

## The fair accuracy comparison — one fixed setting per method

Our gate runs at its module default (`gap_sigma=2.0`) on every dataset, so
HDBSCAN\* is scored at one fixed `(min_samples, method, min_cluster_size)` held
across all of them too. Mean ARI over the 12 datasets that have a flat ground
truth:

| method | setting | mean ARI | k correct |
|---|---|---:|---:|
| **ours (gated set-cover)** | `gap_sigma=2.0`, identical everywhere | **0.887** | 9/12 |
| HDBSCAN\* eom | mpts=5, mcs=3 | 0.820 | 9/12 |
| HDBSCAN\* eom | mpts=1, mcs=5 | 0.782 | 8/12 |
| HDBSCAN\* eom | mpts=1, mcs=10 | 0.772 | 8/12 |
| HDBSCAN\* leaf | mpts=1, mcs=10 | 0.680 | 7/12 |
| HDBSCAN\* leaf | mpts=1, mcs=3 | 0.437 | 3/12 |

**+0.067 over the best single HDBSCAN\* setting** (+0.105 against the best at
`mpts=1`, where the hierarchies are provably identical), at equal *k*-accuracy.
Real, but modest, and it is one seed per dataset with no spread — well short of a
pillar.

**Allow HDBSCAN\* a per-dataset choice of `min_cluster_size` and `min_samples`, and
the accuracy advantage disappears entirely.** Scored against the best of its 12
configurations on each dataset (threshold 0.02 ARI for a verdict):

| | wins | losses | ties |
|---|---:|---:|---:|
| flat gate vs tuned HDBSCAN\* | 1 | 2 | 9 |
| band selector vs tuned HDBSCAN\* | 1 | 1 | 10 |

The flat gate's one win is `graph_communities` (0.331 vs 0.253 — though both miss
the true *k*=3, ours returning 4 and the baseline 6). Its losses are
`cosine_topics` (0.803 vs 0.903) and `multi_scale_hierarchy` (0.551 vs 1.000) —
and the second of those is a flat-versus-multiscale artifact, not a defeat: that
dataset's ground truth is its *fine* level of 6 sub-clusters, and the band selector
recovers bands [6, 3] at **ARI 1.000**, tying the baseline. `bridged_gaussians` is
a wash on any reading (ours 0.982 at the wrong count *k*=3; eom at mcs=10 scores
0.964 at the right *k*=2).

So the honest summary of accuracy is **parity with a tuned HDBSCAN\*, and a
+0.067 advantage over an untuned one.** The advantage is in not needing the tuning,
which is the next section.

## Where the durable difference actually is: parameter sensitivity

HDBSCAN\*'s score is a strong function of `min_cluster_size`, for which no
unsupervised criterion exists on an unlabelled dissimilarity matrix:

| dataset | eom range over mcs ∈ {3,5,10} | eom spread | leaf spread | ours (one setting) |
|---|---|---:|---:|---:|
| concentric_rings | [0.061, 0.863] | **0.802** | 0.175 | 1.000 |
| three_clusters_tree | [0.000, 1.000] | **1.000** | 1.000 | 1.000 |
| bridged_gaussians | [0.452, 0.964] | 0.512 | 0.914 | 0.982 |
| relational_nested_hierarchy | [0.544, 1.000] | 0.456 | 0.456 | 0.544 |
| multi_scale_hierarchy | [0.551, 1.000] | 0.449 | 0.449 | 0.551 |
| varying_density | [0.750, 0.980] | 0.230 | 0.398 | 0.980 |
| two_gaussians | [1.000, 1.000] | 0.000 | 0.951 | 1.000 |
| chain_then_ring | [1.000, 1.000] | 0.000 | 0.859 | 1.000 |

(Full table, all 17 datasets, in the generated markdown. On the four *nested*
datasets the `ours` column is the **flat** gate scored against the finest level,
which it does not target; the band selector is the like-for-like comparison and is
in the next section.)

This is the claim worth making about the gate, and it is *not* an accuracy claim:
**one fixed persistence-outlier threshold delivers those scores across 17
dissimilarity matrices of five different kinds, where matching them with HDBSCAN\*
requires a per-dataset `min_cluster_size` and the wrong choice costs up to 1.000
ARI.** A methods-section result about parameter robustness.

## Nested structure — the one place a baseline cannot follow

| dataset | truth granularities | ours: bands / granularities | ours per-level ARI | HDBSCAN leaf per-level ARI | eps-sweep oracle per level |
|---|---|---|---|---|---|
| nested_gaussians | [6, 2] | 2 / [6, 2] | 1.000, 1.000 | 1.000, 0.324 | 1.000, 1.000 |
| three_level_hierarchy | [8, 4, 2] | 3 / [8, 4, 2] | 1.000, 1.000, 1.000 | 1.000, 0.581, 0.236 | 1.000, 1.000, 1.000 |
| density_hierarchy | [4, 2] | 2 / [4, 2] | 1.000, 1.000 | 1.000, 0.492 | 1.000, 1.000 |
| relational_nested_hierarchy | [6, 3] | 2 / [6, 3] | 1.000, 1.000 | 1.000, 1.000 | 1.000, 1.000 |

Two separate findings here, and they point in opposite directions.

**Against a single-output extractor, the structural claim holds.** EOM and leaf
return exactly one partition. On all four nested sets they lock onto the finest
level (0.955–1.000) and then necessarily score 0.24–0.58 on the coarser levels,
because a flat partition cannot be two granularities at once. The band selector
returns the whole stack, correct at every level, and reports the *number* of levels.

**Against a cut-distance sweep, it does not.** `dbscan_clustering(eps)` swept over
400 cut heights on `three_level_hierarchy` yields **7 distinct partitions**, and
three of them are exactly *k*=8, *k*=4, *k*=2 at **ARI 1.000 each**:

| eps | k | fine | medium | coarse |
|---:|---:|---:|---:|---:|
| 0.032 | 0 | 0.000 | 0.000 | 0.000 |
| 0.553 | 8 | 0.948 | 0.542 | 0.216 |
| 1.074 | 8 | **1.000** | 0.581 | 0.236 |
| 3.680 | 7 | 0.862 | 0.702 | 0.300 |
| 4.201 | 4 | 0.581 | **1.000** | 0.492 |
| 27.129 | 2 | 0.236 | 0.492 | **1.000** |
| 193.883 | 1 | 0.000 | 0.000 | 0.000 |

The reviewer's objection lands, and harder than PRIOR_ART_CH5.md anticipated: the
candidate set is not large, it is **seven**, and a human reading that table picks
the three real scales without difficulty. So "recovering a hierarchy of partitions"
is not the contribution.

**What is left is narrow and precise.** The eps family *contains* the right three
partitions; it supplies no criterion for which of its seven members are real, and
three of them are degenerate (all-noise, all-one-cluster, and a mixed *k*=7). Band
discovery emits exactly three, the right three, with no cut parameter and nothing
told to it. That is **automatic model selection over a candidate set the
flat-cut family already contains** — a selection claim, not a discovery claim, and
the same *shape* as the contribution-2 result above. Contributions 2 and 3 are
therefore one claim about an extraction rule, not two.

## The "no density estimator is available" angle is dead

`min_samples=5` engages the kNN core distance and mutual-reachability machinery.
It **ran without error on all 17 matrices, 0/6 failures each**, including every
non-metric family. A kNN distance is computable from any dissimilarity matrix, so
the framing "on a non-metric D\* no density estimator is available" is false and
must not be used.

The defensible version is that the estimate is computable but *unreliable* there.
Turning it on helps on the coordinate sets (concentric_rings 0.863 → 1.000 at
mcs=10) and hurts precisely on the non-metric ones:

| dataset | best mpts=1 | best mpts=5 |
|---|---:|---:|
| graph_communities | 0.253 | **0.011** |
| cosine_topics | 0.903 | **0.458** |
| multi_scale_hierarchy | 1.000 | 0.895 |
| varying_density | 0.980 | 0.970 |
| concentric_rings | 0.863 | 1.000 |

Note also that `mpts=5, mcs=3` is the best *single* HDBSCAN setting overall
(0.820) — the density estimate is a net positive across the battery as a whole.
The non-metric degradation is a real effect on a small number of datasets, not a
general result, and should be reported that way.

## Net effect on Chapter 5

- Contribution 2's claimed lever (unweighted vs size-weighted stability) is **not
  supported**; what replaces it is a **parameter-robustness** result.
- Contribution 3's claimed lever (recovering a hierarchy of partitions) is
  **pre-empted by a 7-member eps sweep**; what replaces it is **automatic
  selection of which cuts are real**.
- 2 and 3 collapse into **one** methods claim about an extraction rule.
- The `mpts=5` results kill the "no density estimator available" framing outright.
- Contribution 4 (memberships from merge heights on a dissimilarity matrix) is
  untouched by any of this and remains the one pillar.

## Caveats

- **One seed per dataset, no spread.** Every number is a single realisation at the
  generator's fixed seed. `run_all.py`'s ten-seed floor is not met here.
- `min_cluster_size` swept over only {3, 5, 10}; `min_samples` over only {1, 5}.
- Small *n* throughout (30–160). `min_cluster_size=10` on `three_clusters_tree`
  (n=30) is a tenth of the data, which is why its eom spread reaches 1.000.
- HDBSCAN noise (-1) is scored as a label of its own; `ari_noise_excluded` is also
  recorded in the JSON and is uniformly kinder to the baseline.
- Contrib `hdbscan` only; not cross-checked against scikit-learn's implementation.

See [[NONMETRIC_FINDINGS]] and `research/proposal-defense/PRIOR_ART_CH5.md`.
