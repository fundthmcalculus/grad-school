# Reproducing arXiv:2504.13453 with a Collection of Fuzzy Inference Systems

Target paper: Ramachandruni et al., *Using Machine Learning and Neural Networks to
Analyze and Predict Chaos in Multi-Pendulum and Chaotic Systems*, arXiv:2504.13453
(18 Apr 2025). Reference code: `CTNN-ASDRP/ICBBT-IEEE-Xplore-Research-Time-Step-Neural-Operator-Codebase-`.

The paper's reported numbers are taken at face value throughout, as instructed.
Method and every training parameter are documented separately in
`METHOD_AND_PARAMETERS.md`; generated ranking tables are in
`results/comparison.md`. This document is what worked, what didn't, and how the
comparison came out.

---

## 1. Headline result

A collection of Takagi–Sugeno fuzzy inference systems — one FIS per output angle,
Gaussian antecedents fitted from data, consequents solved in closed form by
firing-weighted ridge least squares — **beats every one of the paper's eight
time-step models in six of the seven cells the paper reports**, and places second
in the seventh. The eighth cell below, the frictionless triple pendulum at a
held-out IC, the paper never ran.

| Cell | Paper's best | FIS (ours) | Outcome |
|---|---|---|---|
| double, frictionless, trained IC | LSTM 0.02701 / R² 0.9915 | **0.01264 / R² 0.9981** | FIS 1st of 9 |
| double, frictionless, holdout IC | LSTM 0.26 / R² 0.23 | **0.1839 / R² 0.6199** | FIS 1st (paper ran only LSTM) |
| double, friction, trained IC | LSTM 0.009546 / R² 0.9987 | **0.005503 / R² 0.9996** | FIS 1st of 9 |
| double, friction, holdout IC | LSTM 0.01529 / R² 0.9964 | **0.01201 / R² 0.9979** | FIS 1st of 9 learned |
| triple, frictionless, trained IC | GRU 0.017 / R² 0.9853 | 0.01908 / R² 0.9937 | FIS 2nd of 9 |
| triple, frictionless, holdout IC | *not run in paper* | 0.1620 / R² 0.6091 | — |
| triple, friction, trained IC | GRU 0.009113 / R² 0.9982 | **0.007181 / R² 0.9989** | FIS 1st of 9 |
| triple, friction, holdout IC | GRU 0.006497 / R² 0.9991 | **0.004770 / R² 0.9995** | FIS 1st of 9 learned |

RMSE is in the paper's own units — each trajectory min-max scaled to [0, 1] per
angle column before pooling — so it is a fraction of that trajectory's angular
range, not degrees. The FIS degree figures are 4.97°, 118.78°, 1.26°, 3.11°,
7.29°, 59.29°, 1.41° and 0.95° respectively.

The single loss is the triple-pendulum frictionless trained IC, where the paper's
GRU records RMSE 0.017 against the FIS's 0.01908 — but the FIS's R² is *higher*
(0.9937 vs 0.9853). The two metrics disagree because the paper's Fig. 18B RMSE
labels and its Fig. 22 heatmap disagree with each other for that panel (see
`paper_results.py`); we used the heatmap.

**But the headline is not the interesting result.** See §3.

---

## 2. What worked

### 2.1 Recognising the problem is not autoregressive

The decisive early finding. The paper's "time-step based approach" is described in
prose that sounds recurrent, and five of the eight models it carries into that
approach are recurrent architectures (LSTM, GRU, VRNN, BIRNN, StackedRNN), but the
mapping being fit is

```
(θ₁(0), …, θₙ(0), t)  →  (θ₁(t), …, θₙ(t))
```

with `t` as an ordinary input feature. There is no rollout and no error
accumulation. In the reference code the LSTM even receives `sequences.unsqueeze(1)`
— a sequence of length **one** — so no recurrence over time is exercised at all.

This matters enormously for a FIS. This repository's earlier pendulum surrogate
work (`../DOUBLE_PENDULUM_REPORT.md`, `../N3_N5_FUZZY_REGRESSION_REPORT.md`)
attacked the autoregressive framing and topped out at 0.3–4 s of usable horizon,
because a fuzzy TSK rollout leaves its antecedents' support and flatlines. Here
the same library, on the same physical system, reaches R² 0.9995 — because it is
being asked to interpolate a 2-input surface rather than to integrate.

### 2.2 Rule count, then consequent order, then ridge strength

Only three knobs mattered, in that order:

| Knob | Effect |
|---|---|
| `n_output_buckets` (rules per output) | Dominant. Double-friction holdout R² climbs 0.567 → 0.994 from 2 to 300 rules per output. See `figures/fig_capacity_vs_holdout.png`. |
| `tsk_order="full-2nd"` | Consistently better than `"1st"` at equal rule count; e.g. double friction holdout 0.0350 → 0.0172 at nb=120. |
| `l2_reg=1e-9` | Best config on three of four datasets. The default 1e-6 over-regularises once there are 300 well-populated rules. |

`output_partition="quantile"` mattered only for `triple_friction`, where it was
the *only* way to reach 300 rules (§4.3). Everything else — `norm_conorm`,
`consequent_basis`, `output_partition` elsewhere — moved the fourth decimal place.

This confirms and extends the finding in `../N3_N5_FUZZY_REGRESSION_REPORT.md` §4.7
that `n_output_buckets` is the dominant hyperparameter and that the value 3 used
throughout the earlier ablations was far too low. The useful range here is 200–300.

### 2.3 Dropping zero-variance inputs

θ₁(0) is 120° for every trajectory in the sweep, and θ₂(0) is 0° for every triple
pendulum trajectory. The reference notebooks feed these in anyway. A neural net
absorbs a dead input; a Gaussian membership function on a constant feature has
σ = 0 and a degenerate firing strength. Dropping them is required, not optional.

---

## 3. What the reproduction actually found: the benchmark is not measuring chaos

This is the finding worth keeping.

### 3.1 A predictor with no parameters beats every model in the paper

The held-out initial condition `[120°, 2.05°]` sits **exactly halfway between two
initial conditions that are in the training grid**: `[120°, 2.0°]` and
`[120°, 2.1°]`. So take those two trained trajectories and average them. No
fitting, no parameters, no model. Scored in the paper's own metric:

| Friction cell | bracket midpoint | nearest trained IC | paper's best | FIS |
|---|---|---|---|---|
| double, holdout | **0.00250** (R² 0.99991) | 0.01111 | LSTM 0.01529 | 0.01201 |
| triple, holdout | **0.000359** (R² 0.999997) | 0.001865 | GRU 0.006497 | 0.004770 |

The no-learning midpoint is **6× better than the paper's best double-pendulum
model and 18× better than its best triple-pendulum model.** Even *copying a single
neighbouring trajectory verbatim* beats the paper's best in both cells.

The paper's friction results are its central claim — §3.2 states the shift to
friction was made because frictionless holdout performance was unusable, and the
conclusion generalises from the friction numbers to "the majority of systems with
two or three variable features using the LSTM model for predictions will achieve
an RMSE anywhere between 0.009 and 0.1". Those numbers do not demonstrate learned
chaotic dynamics. They demonstrate that with damping applied, the
initial-condition-to-trajectory map is smooth enough over a 0.1° grid that
one-dimensional interpolation solves it.

Our FIS is in the same position: it beats every learned model and still loses to
the baseline. We report that rather than quoting only the rank among learned
models.

### 3.2 Why: measured, not asserted

`bracket_diagnostic.py` measures how long the bracketing pair stays coherent, and
`figures/fig_bracket.png` plots it:

| Dataset | holdout angular range | bracket separation at t=10 s | λ (fitted) |
|---|---|---|---|
| double, friction | 268.1° | **8.8°** | 0.56 /s |
| triple, friction | 205.3° | **8.2°** | 0.23 /s |
| double, frictionless | 936.6° | **627.6°** | 1.81 /s |
| triple, frictionless | 580.6° | **649.4°** | 0.83 /s |

With damping the two neighbours never separate by more than ~4% of the target's
own range over the full window — the interpolation problem is genuinely easy.
Without damping they separate by 0.1° → 628°, three orders of magnitude, and the
training set then contains two mutually contradictory answers to the same query
with nothing to choose between them.

That is a property of the dataset, not of any model. It caps achievable R² far
below 1 regardless of architecture, and it is exactly why the paper's frictionless
holdout LSTM scores R² 0.23 and why nobody's number in that cell can be good.
`figures/fig_angles_double_frictionless_holdout.png` shows the FIS tracking the
truth cleanly to about 3.5 s and then losing phase — against a measured
decorrelation time of 3.98 s. The model fails exactly when the data stops
containing the answer.

### 3.3 The frictionless case is where a model can still be distinguished

Because the friction problems are solved by interpolation, the frictionless
holdout is the only cell that discriminates. There the FIS is the best result
available:

| double, frictionless, holdout | RMSE | R² |
|---|---|---|
| **FIS (ours)** | **0.1839** | **0.6199** |
| bracket midpoint (no learning) | 0.2256 | 0.4398 |
| LSTM (paper) | 0.26 | 0.23 |
| nearest trained IC (no learning) | 0.3644 | −0.4217 |

The FIS is the only entrant that beats the no-learning baseline on the chaotic
problem. It does so by regressing toward the conditional mean over all 31 initial
conditions instead of committing to one diverged neighbour — which is the correct
behaviour once the bracket has decorrelated, and is what the fuzzy blend across
many rules gives you for free.

The paper never ran the frictionless triple pendulum holdout; the FIS scores
0.1620 / R² 0.6091 there, again ahead of both baselines.

### 3.4 A third, smaller caveat: the metric hides the scale

Per-trajectory min-max scaling means a reported RMSE is a fraction of that
trajectory's own range. The frictionless double-pendulum holdout spans −105° to
+120° in θ₁ (225.5°) and −147° to +790° in θ₂ (936.6°). Spreading a scaled error of
0.26 evenly over those two ranges puts the paper's LSTM at ≈177°; our 0.1839 is a
measured 118.8°. Neither is a useful prediction, and the scaled figure makes both
look like a small error. Every metric in this reproduction is reported in degrees
alongside the scaled value for that reason.

The protocol also requires the held-out trajectory's own min and max in order to
build its target, so even the honest generalisation test is handed two statistics
of the answer. We kept the protocol for comparability and flagged it rather than
silently changing it.

---

## 4. What didn't work

### 4.1 Harmonic time features — the obvious idea, and it fails hard

A TSK consequent is affine in its inputs, so a single rule can only draw a
straight line through whatever slice of input space it fires on. The target
oscillates 15–30 times over the 10 s window. The natural fix is to hand the FIS a
periodic basis: append `sin(k ω₀ t)`, `cos(k ω₀ t)` for k = 1…K, with
ω₀ = √(g/l), which uses no knowledge of the trajectories being fit.

It makes things worse, and past K=16 it collapses. Double pendulum, frictionless,
40 rules per output held fixed so the only variable is the input encoding:

| Encoding | inputs | pooled R² | trained-IC R² | holdout-IC R² |
|---|---|---|---|---|
| raw | 2 | **0.6462** | **0.8640** | 0.4909 |
| harmonic, K=4 | 10 | 0.5127 | 0.6983 | **0.5059** |
| harmonic, K=8 | 18 | 0.4856 | 0.6676 | 0.5098 |
| harmonic, K=16 | 34 | 0.3270 | 0.4893 | 0.3820 |
| harmonic, K=24 | 50 | **−2.8474** | −2.6000 | −2.6564 |

Diagnosis: the binding constraint is **antecedent dimensionality, not consequent
expressiveness**. Firing strength is a t-norm product over one Gaussian membership
per feature. With 50 features every product collapses toward zero, the normalised
firing weights become numerical noise, and the ridge solve fits nothing. Adding
capacity to the consequent by widening the input vector destroys the partition
that selects which consequent to use.

The one place harmonics are not harmful is the held-out IC at small K, where they
move R² from 0.4909 to ~0.51 — but that is the same capacity-control effect as
§4.2, not the oscillation-tracking gain they were added for, and it is well short
of the 0.6199 that coarse memberships reach on the same cell.

The right way to buy oscillation resolution in this library is more *rules*
(§2.2), not more *inputs*. Kept in `sweep.py` at K=8 so the result stays on the
record.

### 4.2 Coarse fixed memberships — helps one metric, hurts three

`n_gaussians=8` (fixed, instead of BIC-selected) was the only setting that
improved frictionless holdout score, and it is the winning configuration in that
cell (R² 0.6199 vs 0.4300 for nb300). But it costs a great deal everywhere else:
on the same dataset it drops trained-IC R² from 0.9981 to 0.7196.

This is not a tuning success, it is capacity control substituting for information
that is not in the data. It is reported as the frictionless-holdout winner because
the assignment asked for the best R² and lowest RMSE, but it should not be read as
a better model — only as a model that commits less where committing is punished.

### 4.3 A library bug at 300 rules with uniform output partitioning

Five `triple_friction` configurations failed outright:

```
IndexError: index 298 is out of bounds for axis 0 with size 298
```

preceded by the library's own warning that `partition_output(method='uniform')`
left buckets 224 and 231 empty and under-filled two more. `triple_friction`'s
scaled target distribution is concentrated enough that a uniform 300-way split of
the output range produces empty buckets, and the downstream consequent indexing
does not tolerate them.

`output_partition="quantile"` avoids it and produced that dataset's best result in
both cells (RMSE 0.004770 holdout / 0.007181 trained). Recorded in
`results/sweep.csv` as errors rather than dropped, per the harness convention that
a failed configuration is a result. Nothing was changed inside the submodule.

### 4.4 The pooled 80/20 split is not worth optimising

The paper's own protocol pools all 62,000 rows and takes a random 80/20 split,
which places samples 5 ms apart on opposite sides of the split. Best pooled R²
reached 0.9249–0.9993, but the number measures within-trajectory interpolation and
tracks trained-IC score almost exactly. It is reported in `results/sweep.csv` for
completeness and used for nothing.

---

## 5. Discrepancies found in the paper and reference code

Documented in full in `METHOD_AND_PARAMETERS.md` §2. The ones that affected this
work:

1. **Two typos in the initial-angle list.** The reference `angles` array contains
   `[122, 0.7]` and `[122, 1.8]` where the pattern requires 120, so the published
   training set has two trajectories from a different θ₁(0) and is missing
   θ₂(0) = 0.7° and 1.8°. We used the intended grid.
2. **31 initial conditions, not 30.** The paper says 30 and 60,000 rows; the code
   produces 31 and 62,000.
3. **The sliding-window dataset size cannot be right.** Paper: 10,000,000 points
   over 1000 s at h = 0.001. 1000/0.001 = 1,000,000, and `Preprocessing.py` uses
   `N=3000`. Approach 1 is superseded by approach 2 in the paper's own
   conclusions, so it was not reproduced.
4. **The PDF's equations (1) and (2) do not match the code**, which implements the
   standard point-mass form. We transcribed the code.
5. **Learning rate**: paper §2 says 10⁻⁴, code uses 10⁻³.
6. **Fig. 18B's RMSE labels contradict the Fig. 22 heatmap** for the frictionless
   triple pendulum (e.g. GRU 9.85e-3 vs 0.017).
7. **The recurrent models are not recurrent** in the time-step experiments
   (`unsqueeze(1)` ⇒ sequence length 1), which undercuts the paper's stated reason
   for expecting RNNs to win.
8. **The friction term is dimensionally irregular** — damping is subtracted inside
   the numerator before dividing by `Lᵢ(2m₁+m₂−m₂cos(2θ₁−2θ₂))`, making the
   effective coefficient a function of θ₁−θ₂. Reproduced as written, since the
   paper's friction numbers depend on it.

## 6. Provenance

- Double-pendulum equations were transcribed from the authors' code, then
  **cross-checked against this repository's independent SymPy Lagrangian model**
  (`../n_pendulum_symbolic.py`, itself validated in `../n_pendulum_validation.py`).
  Maximum disagreement over 200 random states: **1.07e-14**. This check runs on
  every invocation of `pendulum_data.py` and raises if it fails.
- Triple-pendulum dynamics come from that symbolic model rather than from the
  paper's cited Yesilyurt formulation, since it is the derivation this repository
  can verify.
- RK4 at h = 0.005 s over 10 s: maximum relative energy drift **6.6e-7** on the
  undamped `[120°, 0°]` run.
- Every reported FIS number is produced by refitting the selected configuration
  and asserting the refit reproduces the swept score to 1e-9 (`run_all.py`).
- 167 scored configurations across the two sweeps: `results/sweep.csv` — 76 rows,
  71 scored, 5 failed per §4.3 — plus `results/sweep_lowcap.csv`, 96 rows all
  scored. Failures are written to the CSV as error rows, not dropped.

## 7. Figures

Paper-format comparison plots, in `figures/`. No animations, by request.

| Figure | Paper analogue |
|---|---|
| `fig_compare_{system}_{friction}_{setting}.png` (8) | Figs. 11, 12, 13, 18B–D — dual-axis RMSE/R² grouped bars, with the FIS and the no-learning baselines appended |
| `fig_angles_{system}_{friction}_{setting}.png` (8) | the θ(t) truth-vs-prediction plot every reference notebook ends with |
| `fig_trajectory_{system}_{friction}_{setting}.png` (8) | Figs. 14, 15, 16, 19 — bob paths in the plane over 10 s |
| `fig_rmse_heatmap_friction_holdout.png` | Fig. 22 |
| `fig_capacity_vs_holdout.png` | not in the paper — held-out R² against rule count, showing the two regimes |
| `fig_bracket.png` | not in the paper — the §3.2 measurement |

## 8. If this were taken further

The interesting question this reproduction surfaces is not which regressor wins.
It is that **the benchmark as constructed cannot distinguish models**, because the
friction variant is solved by interpolation and the frictionless variant is
information-limited. Two changes would fix that:

1. **Widen the initial-condition spacing** until the bracket-midpoint baseline
   fails. At 0.1° with damping it achieves R² 0.99991; the grid is far finer than
   the problem requires. `bracket_diagnostic.py` gives the criterion directly.
2. **Report the no-learning baseline alongside every model.** It costs nothing and
   it is the difference between "our model achieves R² 0.996" and "our model
   achieves R² 0.996 on a task where averaging two training rows achieves 0.9999".

For the FIS specifically, the open item is §4.1: the product t-norm over
per-feature Gaussians limits this library to a handful of inputs. A firing rule
that degrades gracefully in dimension would make richer input encodings usable,
and that is a change to the fuzzy machinery rather than to the pendulum problem.
