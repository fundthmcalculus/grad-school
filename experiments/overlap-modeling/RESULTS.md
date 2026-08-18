# Results — the laboratory record

[`README.md`](README.md) states the question, the arms and the protocol. This
file is the record: the hypotheses as registered before the run of record, and
how each one scored.

**Run of record:** [`outputs/results.json`](outputs/results.json) — 15,120 fits,
240 cells × 63 arms, 0 errors, 21.7 min on 3 workers. Repo `44f59a4`,
`tribble-fis` `058501f`, Python 3.11.15, numpy 2.4.6, pandas 3.0.5, ten seeds
(`common.SEEDS`' standard), `l2_reg=1e-2`, `output_partition="quantile"`.

Every table below is generated, not transcribed —
[`outputs/headline.md`](outputs/headline.md),
[`outputs/curves.md`](outputs/curves.md),
[`outputs/by_buckets.md`](outputs/by_buckets.md),
[`outputs/fusion.md`](outputs/fusion.md),
[`outputs/mechanism.md`](outputs/mechanism.md),
[`outputs/overlap_curves.png`](outputs/overlap_curves.png). Regenerate with
`python experiments/overlap-modeling/analyze.py`.

---

## Short version

**The idea is right about the mechanism and the code already collects it.**

Overlapping each rule's fitting slice with its neighbours' does what it was
predicted to do, and the effect is large — but only against the baseline the idea
implicitly assumes, which is per-bucket consequent fitting. Against `local-hard`
(the same per-rule fit with hard edges), overlap is worth **+0.17 test R² on
concrete**, +0.12 on `synth-piecewise`, +0.10 on `synth-smooth`, +0.07 on
bikeshare — 60/60 cells, p ≈ 1.6e-11, monotone in the overlap width all the way
to τ=1. That is as clean a confirmation as this kind of sweep produces.

**And it is not enough to matter, because TRIBBLE does not fit consequents that
way any more.** `solve_tsk_consequents_from_firing` stacks a firing-weighted
design block for *every* rule across *every* sample and solves one ridge system,
so every sample already contributes to every rule's coefficients, weighted
continuously. It is a soft-boundary fit arrived at from the other direction, and
a strictly better one: the weights come from the antecedents that predict time
also uses, rather than from a rank cut on `y`. The best overlapped local fit
still loses to it on every rung — **−0.044 on concrete, −0.162 on bikeshare** —
and the gap *widens* with bucket count, which is the opposite of the prediction.

**The one arm that did beat the baseline turned out not to be about boundaries.**
Overlapping the *antecedent* fits (`soft-ante`) beat the baseline on bikeshare by
+0.0175 test R², 48/60 cells, p=4.1e-07. A control that borrows the same number
of rows with the same weights from the same neighbours but draws them
**uniformly instead of at the shared edge** reproduces +0.0151 of that +0.0175
(p=4.4e-06). The residue attributable to the boundary is +0.0024, p=0.20 — not
significant. Bikeshare's gain is more rows per membership fit, not a softer edge.

**The registered mechanism prediction is falsified.** The boundary-specific
residue is *larger* on `synth-piecewise` (+0.0056, p=0.012) than on
`synth-smooth` (+0.0008, p=0.0092) — bigger where the target genuinely has
regimes and hard edges are approximately correct. Whatever small effect survives
the control, it is not "the response surface does not change at the cut".

**Bottom line for the shipped model:** don't adopt this. The largest honest
effect available anywhere in the sweep, after controls, is ≈0.5% R² on one
synthetic rung. The idea's real value is diagnostic — it explains *why* the
closed-form firing-weighted solver was such a large win over the per-bucket
correction path it replaced, and it says the remaining hard edges in the pipeline
(`fit_gaussians`' per-bucket `(μ,σ)`) are not where the accuracy is.

---

## Hypotheses, as registered

| | hypothesis | verdict |
|---|---|---|
| H1 | Overlap improves per-rule (local) consequent fitting, monotonically in τ, one-sided at the ends | **confirmed**, and large: +0.07 to +0.17 R², 60/60 cells, p≈1.6e-11 |
| H2 | Overlapped local fitting reaches or beats the shipped global solve | **falsified**: −0.008 to −0.162 R², 0–2 wins of 60 |
| H3 | Overlapping the antecedent fits improves the shipped model | **falsified once controlled**: the bikeshare gain survives a random-row band, so it is not the boundary |
| H4 | The benefit is largest where the target is smooth and smallest/negative where it is genuinely piecewise | **falsified**, and backwards: the boundary-specific residue is larger on `synth-piecewise` |
| H5 | The benefit grows with `n_output_buckets`, since more buckets means more edges | **falsified**: the local family's deficit grows with bucket count (concrete −0.015 → −0.077 from 3 to 7) |
| H6 | The ramp (trapezoidal) profile beats the flat one, being the smoother of the two | **split, and falsified where it matters**: flat wins 56/56 local-family cells; ramp wins 22/28 `soft-ante` cells by a negligible margin |
| H7 | Neighbour agreement as a penalty (`fusion_reg`) is a cheap closed-form substitute for shared data | **falsified**: flat to negative in λ on all four rungs |

H1–H2 are the substantive pair, and they point in opposite directions on purpose:
the request's mechanism is real, and the thing it would improve is not what
TRIBBLE runs.

---

## H1 — overlap does what it was predicted to do, against the right baseline

`local-hard` is the same per-rule ridge fit on the hard bucket, so this
comparison holds the local/global change fixed and varies only the edge.

| family | dataset | local-hard test R² | validation-selected test R² | Δ | wins | Wilcoxon p |
|---|---|---|---|---|---:|---:|
| local-overlap | concrete | 0.6657 ± 0.0770 | 0.8309 ± 0.0316 | **+0.1651 ± 0.0542** | 60/60 | 1.6e-11 |
| local-overlap | synth-piecewise | 0.6268 ± 0.0516 | 0.7507 ± 0.0416 | **+0.1240 ± 0.0562** | 60/60 | 1.6e-11 |
| local-overlap | synth-smooth | 0.8694 ± 0.0666 | 0.9657 ± 0.0086 | **+0.0963 ± 0.0624** | 60/60 | 1.6e-11 |
| local-overlap | bikeshare | 0.4431 ± 0.0490 | 0.5104 ± 0.0404 | **+0.0672 ± 0.0204** | 60/60 | 1.6e-11 |

The diagnostic curves are monotone in τ on every rung and had not saturated at
τ=1, the widest band the parameterization allows (each rule fitted on its own
bucket plus both neighbours entire). On concrete the local family climbs
0.679 → 0.831 across the sweep. Nothing about this is marginal, and the
one-sided treatment of the two end buckets never showed up as a defect: no cell
produced a non-finite prediction, and rule counts were identical across all arms.

**Read the limit, though.** τ=1 being the best available width is the tell. If
sharing *all* of both neighbours is better than sharing a quarter of them, the
quantity being recovered is not "a smoother transition at the edge" — it is
simply more data per fit. A per-bucket fit at 7 buckets has an eighth of the
training rows and is fitting a full-2nd polynomial with them; widening the slice
mostly buys back degrees of freedom. That reading is what the H4 and H5 results
independently support.

## H2 — but it never reaches the solver TRIBBLE actually ships

Same selected models, paired against `baseline` instead of `local-hard`:

| family | dataset | baseline | selected | Δ | wins |
|---|---|---|---|---|---:|
| local-overlap | synth-smooth | 0.9731 ± 0.0072 | 0.9657 ± 0.0086 | −0.0075 ± 0.0037 | 0/60 |
| local-overlap | concrete | 0.8752 ± 0.0216 | 0.8309 ± 0.0316 | −0.0443 ± 0.0333 | 2/60 |
| local-overlap | synth-piecewise | 0.8095 ± 0.0388 | 0.7507 ± 0.0416 | −0.0587 ± 0.0441 | 2/60 |
| local-overlap | bikeshare | 0.6727 ± 0.0404 | 0.5104 ± 0.0404 | −0.1623 ± 0.0609 | 0/60 |

All at p ≤ 3.5e-11. `full-overlap` — overlapped antecedents *and* overlapped
local consequents — is worse still on three of four rungs, so the two
interventions do not compound.

This is the finding that decides the question, and it is a structural fact rather
than a tuning result. For fixed firing strengths the TSK output is linear in the
consequent coefficients, so the stacked ridge solve is the *exact* minimizer of
the firing-weighted objective the model is scored on. A per-rule fit — however
its slice is drawn — optimizes a different objective: it asks each rule to
predict `y` alone in its own region, when at predict time the output is a
firing-weighted blend of all rules. Overlap narrows that mismatch (H1) without
removing it.

## H3 — the antecedent arm's gain is not the boundary

This is the arm that looked like a win, and the control is why it is reported as
one that isn't.

| dataset | baseline | soft-random | soft-ante | Δ random−base | Δ ante−random | wins | p |
|---|---|---|---|---|---|---:|---:|
| bikeshare | 0.6727 ± 0.0404 | 0.6878 ± 0.0406 | 0.6902 ± 0.0350 | **+0.0151** | +0.0024 ± 0.0149 | 34/60 | 0.20 |
| concrete | 0.8752 ± 0.0216 | 0.8741 ± 0.0213 | 0.8777 ± 0.0215 | −0.0011 | +0.0035 ± 0.0146 | 31/60 | 0.12 |
| synth-piecewise | 0.8095 ± 0.0388 | 0.8104 ± 0.0378 | 0.8160 ± 0.0368 | +0.0009 | +0.0056 ± 0.0138 | 36/60 | 0.012 |
| synth-smooth | 0.9731 ± 0.0072 | 0.9729 ± 0.0070 | 0.9738 ± 0.0068 | −0.0002 | +0.0008 ± 0.0021 | 39/60 | 0.0092 |

`soft-random` draws the same number of rows from the same neighbours and hands
them the same multiset of weights, but uniformly rather than at the shared edge,
and is selected over the same 14 candidates. It therefore holds fixed everything
widening a slice does *except* the boundary structure — rows per membership fit,
the weight distribution, the widened slice's effect on BIC component selection,
and the selection freedom. `Δ ante−random` is what is left for the idea.

What is left is 0.08% to 0.56% R², at 31–39 wins out of 60, on two rungs of four
at p<0.05 — before any correction for testing four datasets. On bikeshare, where
the raw gain was largest and its p-value smallest, **86% of it is reproduced by
borrowing rows at random**.

That the control itself gains +0.0151 on bikeshare and nothing anywhere else is
worth keeping. Bikeshare is the rung whose target is heavily tied and whose
buckets at 7 are thinnest relative to the model being fitted; more rows per
`(μ, σ)` fit is exactly what would help there. If the goal is better antecedents
on that kind of target, "fit each bucket's membership functions on more data" is
the intervention the evidence supports, and it does not need a boundary story.

**Mechanism check passed, which is why the null is readable.** The adjacent-band
overlap does soften the membership functions as claimed — mean adjacent-envelope
overlap coefficient rises monotonically with τ on all four rungs (`synth-smooth`
0.662 → 0.765, `synth-piecewise` 0.609 → 0.750). The antecedents got measurably
softer and test R² did not move. That is a null result, not a plumbing failure.

## H4 — and the softening is not helping for the reason claimed

The two synthetic rungs exist to make the story falsifiable, and it fails. The
boundary-specific residue is **larger on `synth-piecewise` (+0.0056) than on
`synth-smooth` (+0.0008)** — bigger where the response surface genuinely changes
at the cuts and a hard edge is approximately the correct model.

The registered prediction was the reverse, and it was the prediction that would
have supported the story: if hard edges are an artifact of cutting a smooth
surface, softening them should pay most where the surface is smoothest. It pays
least there. Combined with H1's τ=1 optimum, the consistent reading of the whole
sweep is that overlap acts as a **variance reduction on small per-bucket
samples**, not as a correction to a misplaced boundary.

## H5 — more edges did not mean more benefit

| family | dataset | 3 buckets | 5 buckets | 7 buckets |
|---|---|---|---|---|
| local-overlap | concrete | −0.015 ± 0.014 | −0.041 ± 0.015 | −0.077 ± 0.031 |
| local-overlap | bikeshare | −0.089 ± 0.019 | −0.171 ± 0.023 | −0.226 ± 0.024 |
| local-overlap | synth-piecewise | −0.030 ± 0.028 | −0.051 ± 0.036 | −0.094 ± 0.041 |
| soft-ante | bikeshare | +0.024 ± 0.023 | +0.013 ± 0.023 | +0.015 ± 0.018 |

Δ against baseline. The local family's deficit *grows* monotonically with bucket
count on every rung — the more edges there are to soften, the further behind the
global solve overlap falls. `soft-ante`'s bikeshare Δ is flat to declining, and
its control tracks it (`soft-random` 0.013 / 0.013 / 0.020), so the bucket-count
axis carries no boundary signal either.

## H6 — flat beat ramp wherever the effect was large

Across every local-family cell of [`outputs/curves.md`](outputs/curves.md), the
flat band beats the trapezoidal ramp: concrete at τ=1, 0.831 vs 0.802; bikeshare
0.510 vs 0.490; `synth-smooth` 0.965 vs 0.958. The ramp is the smoother profile
and the one with a boundary-continuous limit, so if smoothness at the edge were
the operative quantity it should have won. It loses by exactly the margin the
variance reading predicts — a ramp's borrowed rows carry mean weight ½, so at
matched τ it buys strictly less effective data than flat does.

For `soft-ante` the ordering reverses — ramp is ahead in 22 of 28 cells — but the
margin is negligible: mean +0.0015 R², largest +0.0091, against per-dataset
standard deviations of 0.007 (`synth-smooth`) to 0.043 (bikeshare). So the ramp
wins consistently and by 4–20% of one standard deviation, on the arm whose total
effect the H3 control already showed to be null. H6 is only decided where the
effect is large, and there flat wins outright.

## H7 — the fusion penalty is not a substitute

| dataset | baseline | λ=1e-3 | λ=1e-2 | λ=0.1 | λ=1 | λ=10 |
|---|---|---|---|---|---|---|
| bikeshare | 0.673 ± 0.040 | 0.673 | 0.672 | 0.670 | 0.663 | 0.648 |
| concrete | 0.875 ± 0.022 | 0.876 | 0.878 | 0.876 | 0.868 | 0.860 |
| synth-piecewise | 0.809 ± 0.039 | 0.809 | 0.808 | 0.800 | 0.785 | 0.771 |
| synth-smooth | 0.973 ± 0.007 | 0.973 | 0.973 | 0.973 | 0.972 | 0.971 |

`fusion_reg` weights `Σ_r ‖c_{r+1} − c_r‖²` inside the exact global solve — the
same intent as shared data, expressed as a penalty, and λ→∞ is the limit of total
sharing between neighbours. Validation selection picks a λ small enough to be
nearly a no-op and the paired Δ is +0.0030 on concrete (p=0.0036) and ≈0
elsewhere. Every λ large enough to change the model makes it worse, monotonically.

Worth stating plainly because it closes the door from the other side: the global
solve does not want its adjacent consequents pulled together. They differ because
the rules cover different regions, and that is information, not noise.

---

## Costs

No arm buys accuracy with capacity or time. Within every one of the 240 cells all
63 arms agree exactly on rule count and kept-feature count (0 cells disagree on
either), by construction: feature ranking is deliberately left on the hard
partition so every arm shares one input space, and the overlap never adds a rule.
Pooled means are 5.00 rules and 6.89 features, the mix of the `n_output_buckets ∈
{3,5,7}` and per-dataset settings. Mean fit time: `local-hard` and `local-overlap` 0.203 s, `baseline`
0.255 s, `soft-ante` and `soft-random` 0.273 s, `fusion` 0.260 s. The overlapped
antecedent arms cost ~7% more than the baseline for no accuracy; the local arms
are ~20% cheaper and much worse.

## Two things found along the way

**WEC cannot be used as a regression rung by this pipeline, and the reason is not
the pipeline.** `Total_Power` is the sum of the 100 `Power*` columns, so those
have to be dropped — but with them gone, `calculate_gaussian_correlation` scores
every one of the 198 remaining buoy-coordinate columns at exactly 0.0000.
`take_top_features` therefore keeps all 198 and the model predicts the training
mean: test R² = −0.000 for all 31 arms in a probe run, at 4–8 s a fit. Excluded
from the default run (`--datasets wec` reproduces it).

`experiments/fis-to-neural-net/run_experiment.py`'s `load_wec` drops only
`Total_Power` and keeps the `Power*` columns, so its WEC rows (R² ≈ 0.91) are
measured on a target the features sum to. That experiment found and documented
the identical problem in bikeshare's shared loader and deliberately left the
shared loader alone; its own WEC loader appears to have the same defect
unnoticed. Not patched here — same reasoning, it would move numbers a table
elsewhere quotes — but it should be checked before that row is cited again.

**`overlap_means` is arithmetically inert under the global solve without
`pin_extremes`.** `solve_tsk_consequents_from_firing` re-derives every unpinned
intercept as part of its exact optimum, so a centroid handed to it survives only
where it is pinned. This is not a weak effect to be measured, it is exactly zero,
and it is pinned as a test
(`test_overlap_means_is_inert_under_the_global_solve_without_pinning`) so the
next person to reach for that knob finds out in a second rather than in a sweep.

## Caveats

- **Selection asymmetry remains in the headline table.** Each family is chosen
  from 14 candidates and the baseline is one fixed model, which flatters the
  families. `soft-random` neutralizes this for `soft-ante` (same candidate count)
  and it does not matter for the local families, whose Δ is negative and large.
  It does mean `fusion`'s +0.0030 on concrete should not be read as a win.
- **Four datasets, four significance tests, no multiplicity correction.** At
  Bonferroni α=0.0125 the two significant `Δ ante−random` results (0.012 and
  0.0092) survive only marginally. Nothing here should be treated as a
  1%-level result.
- **`pin_extremes=False` throughout.** The pinned-extremes configuration is the
  one place `overlap_means` can act, and it is unmeasured. Given the antecedent
  overlap's null, softening the two pinned rungs — which trades output range for
  a less outlier-driven centroid — is unlikely to be where the win is, but it is
  a genuine gap.
- **One t-norm family and one basis.** All arms run the `min/max` default and the
  `raw` monomial basis. The orthogonal basis conditions high-order consequents
  much better, and better-conditioned local fits are the case where overlap has
  the least left to fix — so the H1 gap would likely narrow, not widen.
- **Rank-space ties.** A band edge landing inside a run of equal `y` splits it
  arbitrarily (reproducibly, by stable sort). This is worst on bikeshare, the
  most tied target. `shape="flat"` is insensitive to within-band rank and is the
  profile that won anyway, so this is unlikely to have mattered.
