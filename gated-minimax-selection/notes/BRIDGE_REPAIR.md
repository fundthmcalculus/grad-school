# One-Sided Metric Repair: Defending the Minimax Pipeline Against Shortcuts

**Date**: 2026-08-24
**Status**: Complete — follows directly from NONMETRIC_FINDINGS.md finding 2
**Code**: `metric_repair.py` (the defense), `run_bridge_repair.py` (driver), `test_metric_repair.py` (13 tests)
**Outputs**: `outputs/bridge_repair_results.json`, `outputs/fig16_repair_doseresponse.png`, `outputs/fig17_repair_noharm.png`

---

## Executive Summary

NONMETRIC_FINDINGS.md established that shortcut-type corruption (deflated
entries acting as single-linkage bridges) collapses every D*-based method
(ARI 1.0 → 0.11) while stretch-type corruption is harmless. This note closes
that loop with a defense matched to the asymmetry:

**`reverse_ti_repair(D, q)`** — lift every entry to at least the q-quantile of
its witness lower bounds `|D_ik − D_jk|` (the reverse triangle inequality).
One-sided by design: it only raises entries, and only ones provably
inconsistent with the rest of the matrix.

Results:

1. **Collapse repaired.** Gap-cover at the worst sweep cell (shortcut,
   rate 0.2, strength 1.0): raw **0.11** → q=0.5 **0.85** → q=0.75 **0.98**.
   The full dose-response curve flattens back to near-clean performance.
2. **Multi-scale restored.** At the cells where `select_multiscale` was
   measured to break (fine band lost at r=0.1/s=1.0; full collapse at
   r=0.2/s=1.0, ARI 0.15/0.17), repair at q=0.75 restores bands [6, 3] and
   **1.0/1.0 at both truth levels** in both cells.
3. **Provably and empirically harmless where it should be.** Identity on
   metric inputs at *any* quantile (every witness bound is ≤ D_ij — pinned in
   tests on Euclidean, edit, Hamming, and graph matrices). Zero entries lifted
   at q=0.5 on every synthetic battery family, on clean blobs, and under
   stretch corruption. Critically, graph shortest paths pass through
   untouched: **real thin bridges are metrically consistent, so the repair
   distinguishes corruption from structure** — geometric bridges remain
   ConiVAT's problem, deliberately.
4. **Scope boundary found (the honest limit).** Real flight-profile DTW gets
   **50% of its entries lifted** (cover ARI −0.07). This is not a bug: a deep
   upper-bound violation D_ij > D_ik + D_jk *is* a lower-bound violation at
   the pair (i, k) with witness j — the two are views of one triple. Sparse
   deep violations don't reach the median witness (that robustness is the
   design); *dense* deep violations (real DTW: 70% of pairs, depth ~2) do.
   **The repair is for metric-plus-sparse-corruption data, not for
   intrinsically non-metric dissimilarities.** Both regimes are pinned in
   tests.

## The quantile knob

With a fraction r of pairs corrupted, ~2r(1−r) of a pair's witness bounds are
themselves inflated (exactly one leg deflated), so q must sit below
~1 − 2r(1−r): at r = 0.2 that is q ≲ 0.68, and q = 0.9 was observed to
over-repair (over-merging). Recommended: **q = 0.5** when corruption is
unknown (zero false lifts in every no-harm test), **q = 0.75** when corruption
is believed sparse (≤ ~15% of pairs) — it repaired hardest in every
experiment here. One caveat recorded in the JSON: at strength 0.6 (mild,
partially-bridging corruption) q=0.5's repair estimate moved one replicate the
wrong way (0.97 raw → 0.85 repaired); q=0.75 did not (0.98).

## Relation to prior work — checked 2026-08-26, and it is mostly known

The earlier version of this section guessed at two citations and flagged them
unverified. The check came back: **the problem, the operator, and the
application are all published.** What follows is the corrected account.

**The problem has a name.** Repairing a matrix by only *increasing* entries is
**Increase Only Metric Repair (IOMR)** — Gilbert & Jain, *"If it ain't broke,
don't fix it: Sparse metric repair"*, Allerton 2017, 612–619 — equivalently the
increase-only variant of **Metric Violation Distance** (Fan, Raichel & Van
Buskirk, SODA 2018, 196–209; *Algorithmica* 84(5):1441–1465, 2022). Brickell,
Dhillon, Sra & Tropp (*SIAM J. Matrix Anal. Appl.* 30(1):375–396, 2008) had
already floated it in §7.1 as an open variation of metric nearness. Its
complexity is settled and unfavourable: **increase-only is NP-complete and
cannot be approximated better than minimum vertex cover** (Fan et al.).

**The operator is theirs too.** Gilbert & Jain's Algorithm 3 (IOMR) "updates
D_ik with D_ij − D_jk whenever ijk is broken" — which is the reverse-TI witness
bound of this module, at a single witness, iterated.

**At q = 1 this reduces to a classical embedding.** Because the witness set
includes k = i, the bound |D_ii − D_ji| = D_ij is always present, so
`max(D_ij, max_k |D_ik − D_jk|)` collapses to `max_k |D_ik − D_jk|` =
‖row_i(D) − row_j(D)‖_∞ — the **Fréchet/Kuratowski isometric embedding of a
finite metric into ℓ∞**, 1910/1935. Verified numerically here (exact equality
on Euclidean, DTW and corrupted matrices). All three "properties" this module
claimed therefore come for free and are classical at q = 1: identity on
metrics *is* isometry, one-sidedness *is* the IOMR definition, and the output
is a genuine metric because ℓ∞ is a norm.

**A limitation that fell out of the same check.** That last guarantee holds
**only at q = 1**. At the recommended default q = 0.5 the output is *not* a
metric (measured: triangle violations remain). So "repair" is loose language
for what this does at its recommended setting — it removes shortcuts, it does
not restore metricity. That is fine for the minimax pipeline, which needs no
metric, but the name oversells it and the chapter should not use it unguarded.

**The application has an incumbent with the same algebraic shape.** Lifting
entries one-sidedly to break bridges before single-linkage is exactly the
**mutual reachability distance** of HDBSCAN — `mrd(a,b) = max(core_k(a),
core_k(b), d(a,b))` (Campello, Moulavi & Sander, PAKDD 2013) — same
`max(D_ij, ·)` form, differing only in where the lower bound comes from (kNN
density vs. reverse-TI witnesses). ConiVAT (Rathore, Bezdek, Santi & Ratti,
2020) attacks the same failure from the opposite direction, with a
*decrease*-only minimum-transitive transform.

**And a direct challenge worth engaging.** Etgar & Gilbert, *"Metric repair is
two problems: Which edges, and what weights"* (arXiv:2608.07715, Aug 2026),
tests whether repair helps downstream and largely finds it does not —
"finding the correct set of edges … is critical … setting the weights is not
an implementation detail; it is half the problem." Any claim that this repair
helps has to answer that paper.

### What is left unclaimed

Searched and **not found** (which is weak evidence of absence, not proof): the
**quantile-over-witnesses aggregation** as a robustness knob for repair, the
**corruption-rate estimator** r̂ from median-witness-bound exceedance, and the
resulting **auto-tuning + abstention rule**. Nearest precedent for the
aggregation idea is Moerel & Grootswagers (arXiv:2506.00484, 2025), who take a
**median over third points** — but to *impute missing* entries, not repair
corrupted ones.

So the defensible framing, if this is ever written up, is narrow:

> a robust, quantile-aggregated estimator for the increase-only metric repair
> problem of [Gilbert & Jain 2017; Fan et al. 2018], with an auto-tuning rule
> and an abstention criterion

— not a new method, and emphatically not a new problem. *Bibliographic note:*
Brickell 2008, Gilbert & Jain 2017, Fan et al. 2018/2022, the SWAT 2020
generalisation and Etgar & Gilbert 2026 were verified against primary sources;
Campello 2013 page numbers and the Vidal/AESA and Dress–Havel bound-smoothing
references were verified only via search summaries and need a check before
they enter a bibliography.

## Reproduction

```bash
cd gated-minimax-selection
python run_bridge_repair.py      # ~3 min; JSON written before figures
python -m pytest test_metric_repair.py -q
```

Same seed conventions as run_nonmetric.py (NERFCM restarts [0..4], sweep
dataset seeds [0..2]).

## R4 — Auto-q: the quantile set from the data itself

`metric_repair.auto_repair(D)` closes the "how do I pick q" loop:

1. **Estimate** r̂ = `estimate_corruption_rate(D)`: the fraction of pairs whose
   *median* witness bound exceeds the entry. Exactly 0 on metric data; tracks
   planted shortcut corruption monotonically but conservatively (~0.6× —
   deflated intra-cluster pairs are invisible, and only the cross-cluster
   deflations that actually threaten the minimax transform are counted).
2. **Calibrate** q = clip(1 − 2r̂(1−r̂) − margin, 0.5, 0.9) — the
   corrupted-witness bound with a 0.1 margin.
3. **Decline** when r̂ > 0.35: a matrix where the median witness disagrees with
   over a third of the entries is not "metric plus sparse corruption"; it is
   intrinsically non-metric, and auto_repair returns it unchanged. The
   threshold is calibrated between the sweep's largest accepted r̂ (~0.28 at
   true rate 0.4) and real flight-DTW's reading (0.51 → declined).

Result on the shortcut rate sweep at strength 1.0: **auto = 1.0 at every
rate** — matching or beating both fixed quantiles (q=0.5 dips to 0.85 and
q=0.75 to 0.97–0.98 at rate 0.2, where auto's calibrated q=0.68 threads the
needle). Every battery family gets q=0.9 (free: identity on all of them);
real DTW is declined rather than half-rewritten.

One estimator quirk worth recording: r̂ is *not* monotone in dense
inflation-type violation (uniform 4× inflation of 70% of pairs reads r̂=0.12,
less than 50% of pairs at 0.26) — dense inflation shifts the median witness
itself. The decline test therefore exercises the mechanism with an explicit
threshold, and the default's calibration rests on the measured regimes above,
not on a synthetic reconstruction of real DTW.

## Follow-ups this opens
- The chapter framing writes itself as a 2×2: {mean vs bottleneck aggregation}
  × {raw vs repaired}, with each cell's failure mode now measured.
- ConiVAT comparison on *geometric* bridges (bridged_gaussians), where this
  repair is — correctly — inert.
