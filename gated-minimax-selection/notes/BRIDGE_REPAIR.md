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

## Relation to prior work

This is a cheap, one-sided special case of the **metric nearness problem**
(Brickell, Dhillon, Sra & Tropp, SIAM J. Matrix Anal. 2008; sparse variant:
Gilbert & Jain, Allerton 2017). Those project onto the full metric cone; this
lifts only below-lower-bound entries — the only direction the minimax
transform is sensitive to (NONMETRIC_FINDINGS finding 2). *Citations from
memory — verify before quoting in a chapter.*

## Reproduction

```bash
cd gated-minimax-selection
python run_bridge_repair.py      # ~3 min; JSON written before figures
python -m pytest test_metric_repair.py -q
```

Same seed conventions as run_nonmetric.py (NERFCM restarts [0..4], sweep
dataset seeds [0..2]).

## Follow-ups this opens

- A corruption-rate estimator to set q automatically (the 2r(1−r) bound makes
  this a one-line plug-in once r is estimated, e.g. from the fraction of
  entries whose median witness bound exceeds them).
- The chapter framing writes itself as a 2×2: {mean vs bottleneck aggregation}
  × {raw vs repaired}, with each cell's failure mode now measured.
- ConiVAT comparison on *geometric* bridges (bridged_gaussians), where this
  repair is — correctly — inert.
