# Results — the laboratory record

[`README.md`](README.md) states the question, the arms and the protocol. This
file is the record: the hypotheses as registered before each run, and how each
one scored.

Three stages, each answering the objection the previous one raised:

| stage | question | driver | verdict |
|---|---|---|---|
| **1** | Does overlapping the buckets improve the model? | `run_experiment.py` | mechanism real, no usable gain |
| **2** | With a real per-bucket consequent solve, is the deficit the fit or the blend? | `run_local.py` | **the blend** — rules get much better locally, the model gets worse |
| **3** | Does compact antecedent support fix the blend? | `run_support.py` | no — tightening support does not close the gap at all |

Stage 1 is below; [stage 2](#stage-2--per-bucket-consequent-solving-fit-or-aggregation)
and [stage 3](#stage-3--compact-support) follow it.

**Stage 1 run of record:** [`outputs/results.json`](outputs/results.json) — 15,120 fits,
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


---

# Stage 2 — per-bucket consequent solving: fit or aggregation?

**Run of record:** [`outputs/local_results.json`](outputs/local_results.json) —
10,800 fits, 240 cells × 45 arms, 0 errors, 15.2 min. Same provenance, seeds,
protocol and `l2_reg` as stage 1. Tables:
[`local_fit_vs_blend.md`](outputs/local_fit_vs_blend.md),
[`local_aggregation.md`](outputs/local_aggregation.md),
[`local_sharpen.md`](outputs/local_sharpen.md),
[`local_fit_vs_blend.png`](outputs/local_fit_vs_blend.png). Regenerate with
`analyze_local.py`.

## The objection

Stage 1 scored only the *blended* prediction, so it could not distinguish two
explanations for the local family's deficit. Either each rule is a poor
approximator of its own region — in which case no way of combining them helps —
or each rule is a **good** approximator and the firing-weighted blend is what
loses the accuracy. The diagnostic that separates them is `local_r2`: the R² of
each row's own-bucket rule alone, blend ignored.

## The answer: the rules are excellent and the blend destroys them

| | local R² (own-bucket rule, test) | test R² (blend) |
|---|---|---|
| **concrete** — global solve | 0.608 | **0.875** |
| **concrete** — per-bucket solve, τ=0 | **0.940** | 0.666 |
| **bikeshare** — global solve | 0.021 | **0.673** |
| **bikeshare** — per-bucket solve, τ=0 | **0.872** | 0.443 |

Bikeshare is the clearest statement of what the global solve is actually doing.
Its rules are worthless as local models — R² 0.021, barely better than predicting
the mean — and they blend to 0.673. Being the exact minimizer of the
firing-weighted objective means **spreading the function across rules that are
individually meaningless and only correct in superposition**. Per-bucket solving
gives the opposite: rules that are individually excellent and blend badly.

And overlap trades the first for the second, which is why stage 1 saw it help:
across τ = 0 → 1 on concrete, local R² *falls* 0.940 → 0.906 while test R²
*rises* 0.666 → 0.821. Overlap buys blend-compatibility by making the rules less
locally specialized. That is the opposite of the story it was proposed under.

## Which aggregation fixes work

Width and γ chosen per cell on validation, paired against the baseline in the
same cell. `recovered` is the share of `local-free`'s own deficit closed.

| arm | concrete | bikeshare | synth-piecewise | synth-smooth |
|---|---|---|---|---|
| `local-recal` (per-rule affine blend) | −0.026 (51%) | −0.064 (56%) | −0.052 (48%) | −0.001 (88%) |
| `local-wta` (winner-take-all) | −0.103 | −0.212 | −0.257 | −0.016 |
| `local-sharp` (concentrate the blend) | −0.116 | −0.201 | −0.125 | −0.023 |
| `shrink-local` (global solve, local prior) | −0.000 (99%) | −0.000 (100%) | +0.000 (100%) | **+0.002** (122%) |
| `global-wta` *(control)* | −0.148 | −0.363 | −0.123 | −0.048 |
| `global-recal` *(control)* | −0.001 | −0.000 | +0.000 | +0.000 |
| `global-sharp` *(control)* | +0.002 | **+0.021** | +0.002 | +0.001 |

Three things to take from this.

**About half the deficit really is the blend.** `local-recal` adds two parameters
per rule and cannot change what any rule computes, only how it is combined — and
it closes 48–88% of the gap. That is direct confirmation of the diagnosis. It
still never reaches the baseline.

**`shrink-local` is free and marginally positive.** Using the local fit as the
ridge's *prior* instead of zero keeps the exact global objective and has the
baseline as its λ=0 limit, so it cannot lose by construction. On `synth-smooth`
it wins in 59 of 60 cells (+0.002, p=1.7e-11). Tiny, but it is the only arm in
three stages that beats the shipped model without a control killing it.

**The largest win went to a control, which is what controls are for.**
`global-sharp` — raising firing strengths to a power before normalizing —
gives **+0.021 on bikeshare at 60/60 cells, p=1.6e-11**. It helps the *global*
solve, so it is a better aggregation in general and says nothing about local
fitting. Worth noting on its own account: it is one scalar, it costs nothing, and
it is the only change measured here with a large, unambiguous effect on the
shipped model. Its `local` twin is strongly negative, so the two are not the same
mechanism.

`global-wta` being catastrophic (−0.15 to −0.36) settles the winner-take-all
question separately: the blend is not merely tolerable, it is load-bearing.

---

# Stage 3 — compact support

**Run of record:** [`outputs/support_results.json`](outputs/support_results.json)
— 11,520 fits, 240 cells × 48 arms, 0 errors, 17.4 min. Tables:
[`support_shapes.md`](outputs/support_shapes.md),
[`support_paired.md`](outputs/support_paired.md),
[`support_locality.md`](outputs/support_locality.md),
[`support_clamp.png`](outputs/support_clamp.png). Regenerate with
`analyze_support.py`.

## The objection

> We have full support everywhere because the membership functions are Gaussian,
> which have effectively infinite support. What if we switched to trapezoids, or
> applied a non-linear clamp to zero at 2.75–3 sd?

This is the right diagnosis of *why* stage 2's blend can ruin a good local model.
A Gaussian is strictly positive everywhere, so every rule fires — however faintly
— at every point, and a per-bucket consequent gets weight a long way from any
data it was fitted on. It also has independent support in the library: 
`gauss_data.mf_interval` already treats `μ ± 3σ` as a Gaussian's effective
support, so clamping there makes an existing convention literal.

Arms: a clamped Gaussian at k ∈ {2, 2.5, 2.75, 3, 3.5, 4} σ meeting the axis
continuously (the "non-linear clamp"), a hard truncation at 2.75 and 3 σ so the
discontinuity is priced rather than assumed, the library's fast histogram
trapezoid fitter, and a Ruspini triangular partition — crossed with
`consequent_fit ∈ {global, local}` and τ ∈ {0, 0.5}.

## The trade-off, measured

`active_frac` is the share of rules firing on a typical row (locality);
`uncovered` is the share of rows **no** rule covers — answered with exactly 0, a
finite number that passes every NaN filter. Concrete, τ=0:

| membership | active_frac | uncovered | test R² (global) | test R² (per-bucket) |
|---|---:|---:|---|---|
| gaussian | 0.927 | 0.001 | **0.875** | 0.666 |
| clamped 4σ | 0.911 | 0.001 | 0.875 | 0.665 |
| clamped 3σ | 0.841 | 0.010 | 0.868 | 0.655 |
| clamped 2.75σ | 0.808 | 0.014 | 0.863 | 0.650 |
| clamped 2.5σ | 0.755 | 0.022 | 0.853 | 0.640 |
| clamped 2σ | 0.567 | 0.107 | 0.775 | 0.557 |
| trapezoid | 0.129 | 0.800 | 0.105 | 0.057 |
| ruspini (tol 0.02) | 0.032 | 0.889 | 0.109 | 0.085 |

k=4 reproduces the Gaussian to three decimals, so the clamp is a proper
continuum with infinite support as its limit — which is what makes the rest of
the column readable as an effect of the clamp and nothing else.

## H8 — clamping improves the model: **falsified**

Paired against Gaussian in the same cell, global solve:

| k | concrete | bikeshare | synth-piecewise | synth-smooth |
|---|---|---|---|---|
| 4 | +0.0003 (23/60) | −0.0010 (0/60) | 0.0000 (33/60) | 0.0000 (39/60) |
| 3.5 | −0.0017 (25/60) | −0.0018 (0/60) | 0.0000 (34/60) | 0.0000 (41/60) |
| 3 | −0.0058 (25/60) | −0.0041 (2/60) | 0.0000 (30/60) | 0.0000 (35/60) |
| 2.75 | −0.0087 (21/60) | −0.0068 (3/60) | +0.0007 (37/60) | −0.0001 (28/60) |
| 2.5 | −0.0202 (9/60) | −0.0125 (3/60) | +0.0022 (44/60) | −0.0003 (16/60) |

**This retracts a number I reported earlier from a single seed.** A one-seed probe
on concrete had clamping at 2.75σ at 0.8898 against Gaussian's 0.8746, and I
called it the best arm measured while flagging it as provisional. Over ten seeds
and four datasets it is **−0.0087 on concrete and −0.0068 on bikeshare**, winning
3 of 60 bikeshare cells at p=5.7e-11. The +0.015 was seed noise. The effect is
monotone in the wrong direction: the tighter the clamp, the worse the model, on
both real datasets. `synth-piecewise` shows a consistent but tiny positive
(+0.0022 at k=2.5, 44/60) — the one rung where genuine regime boundaries make a
hard edge approximately correct, which is at least coherent with stage 1's H4.

The hard truncation is indistinguishable from the smooth clamp at matched k
(−0.0100 vs −0.0087 on concrete at 2.75σ), so the discontinuity is not what costs
anything. The clamp itself is.

## H9 — compact support closes the local/global gap: **falsified outright**

This is the counterpoint's actual prediction, and the cleanest null in the whole
experiment. `gap` is local minus global test R², same cell, τ=0, pooled:

| membership | active_frac | uncovered | gap |
|---|---:|---:|---|
| gaussian | 0.925 | 0.000 | −0.1814 ± 0.0833 |
| clamped 3.5σ | 0.878 | 0.001 | −0.1817 ± 0.0833 |
| clamped 3σ | 0.836 | 0.004 | −0.1818 ± 0.0840 |
| clamped 2.75σ | 0.805 | 0.006 | −0.1812 ± 0.0837 |
| clamped 2.5σ | 0.760 | 0.009 | −0.1810 ± 0.0835 |
| clamped 2σ | 0.619 | 0.041 | −0.1855 ± 0.0858 |

The gap is **flat to four decimals** across the whole range. Taking rules-firing-
per-row from 93% down to 62% moves the local family's deficit by 0.004 R², in the
wrong direction. Whatever the blend is doing to a per-bucket consequent, it is not
being done by the far tails of the Gaussians.

The two shapes that *do* narrow the gap narrow it for the wrong reason:
`trapezoid` (−0.103) and `ruspini/0.02` (−0.038) collapse **both** arms toward
zero R², so the gap closes because the global solve fell to meet the local one,
not because the local one improved.

The clearest single statement of the null: `local R² (per-bucket fit)` is
**0.940 on concrete for every membership shape in the table**, including the ones
that score 0.105 test R². The per-bucket consequent solve reads only the y-bucket
slices and never touches the antecedents, so the shape cannot affect the local
fit at all — only how that fit gets blended. Compact support changes the blend
and does not help.

## H10 — trapezoids give compact support usably: **falsified**

| dataset | trapezoid test R² (global) | Δ vs gaussian | uncovered |
|---|---|---|---:|
| concrete | 0.1138 ± 0.0274 | −0.7619 (0/60) | 0.800 |
| bikeshare | −0.0003 ± 0.0005 | −0.6786 (0/60) | **1.000** |
| synth-piecewise | 0.7599 ± 0.0478 | −0.0483 (1/60) | 0.459 |
| synth-smooth | 0.9415 ± 0.0214 | −0.0320 (0/60) | — |

On bikeshare the histogram fitter leaves **every single test row uncovered** and
the model predicts 0 throughout. The cause is not dimensionality: **42% of the
axis is uncovered at one feature**, measured directly. The fitter places
trapezoids on histogram modes and never tiles the feature range, so the AND across
features then compounds gaps that already exist in 1-D.

Switched to `trapz_math_fast.create_trapz_membership_dict_fast` on request —
**0.01 s against 42 s** for the EM fitter, a 4,000× speedup that is what made
these arms affordable at all. The EM path stays reachable as
`membership="trapezoid-em"` and is far too slow for an arm matrix, especially on
the overlap path, which refits once per bucket.

## A prediction of mine that was wrong

I added the Ruspini arm expecting compact support **with** guaranteed coverage,
because a Ruspini partition's terms sum to exactly 1 at every point of the axis.
It measured 89% uncovered on concrete and 90% on bikeshare.

The error: the partition tiles the axis, but each bucket is assigned only the
term(s) nearest its own centres, so **coverage by the term set does not imply
coverage by any one rule** — and the AND across features shrinks each rule's
support again. Partition of unity is a property of the whole term family, not of
a per-bucket selection from it. Now
`test_ruspini_terms_partition_the_axis_but_one_bucket_does_not`, which asserts
both halves: the term set passes `ruspini.verify_partition_of_unity`, and the
model still has uncovered rows.

## The structural reading

Locality and coverage are in direct conflict in this architecture, and Gaussians
resolve it by giving up locality. Rules are indexed by output bucket, and every
rule must be evaluated at every input point, so a rule's support has to span the
whole input region its bucket's samples occupy — which for overlapping buckets
means overlapping support. Infinite support is not an oversight in that design; it
is what makes the blend total, and the blend is load-bearing (`global-wta`,
−0.15 to −0.36).

Getting real locality would need the firing strengths to be *bucket posteriors*
rather than per-bucket density fits — a different antecedent model, not a narrower
one. That is a genuine direction and it is outside what this experiment can
settle.

## Costs

Rule counts and kept features are identical across all 48 arms within every cell.
Mean fit time is dominated by the fitter, not the shape: trapezoid-fast is the
cheapest arm measured, the clamped arms cost the same as Gaussian plus one
`np.where`, and ruspini adds a per-feature partition build.

## Caveats carried into stage 3

- `pin_extremes=False` and the `raw` basis throughout, as in stages 1 and 2.
- The clamp is symmetric in σ. An asymmetric or per-feature cutoff is unmeasured.
- `active_frac` counts rules firing above 1e-6, so it is a threshold statistic,
  not a support measurement: an unclamped Gaussian model scores 0.93 rather than
  1.00 because memberships past ~5σ fall below that floor.
- Ruspini used the "nearest apex" matching heuristic that `ruspini.ruspinize_model`
  documents. A better assignment (give each bucket every term its samples occupy)
  would raise coverage and is untested — though it would also give up the locality
  the arm existed to test.
