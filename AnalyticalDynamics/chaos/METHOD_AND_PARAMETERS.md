# arXiv:2504.13453 — Method Outline and Training Parameters

Source paper: Ramachandruni, Nara, Lalu, Yang, Ramesh Kumar, Jain, Mehta, Koo,
Damonte, Akl. *Using Machine Learning and Neural Networks to Analyze and Predict
Chaos in Multi-Pendulum and Chaotic Systems.* arXiv:2504.13453 [cs.LG],
18 April 2025. 35 pages, ~20 figures. Aspiring Scholars Directed Research
Program (ASDRP), Fremont CA.

Reference code: <https://github.com/CTNN-ASDRP/ICBBT-IEEE-Xplore-Research-Time-Step-Neural-Operator-Codebase->

Everything below marked **[code]** was read out of the reference notebooks, not
the PDF. Where the two disagree, the code is what produced the published numbers
and the discrepancy is flagged. The paper's own text is internally inconsistent
in several places (§1 "Steps taken", item 5 below), so the code is the authority.

---

## 1. Steps the paper takes

1. **Derive / adopt equations of motion.** Double pendulum: two coupled
   second-order ODEs in the absolute angles θ₁, θ₂ measured from the downward
   vertical, point masses on massless rods. Triple pendulum: equations taken
   from Yesilyurt (arXiv:1910.12610), an n-point-mass formulation.
2. **Generate synthetic data with fixed-step classical RK4** ("ODE-RK4"). Angles
   are converted to degrees and stacked into a NumPy array.
3. **Characterise chaos before modelling.** Sweep initial angles, plot θ(t) over
   0–40 s and 0–1000 s, and identify where motion crosses from quasi-periodic to
   fully rotating. Conclusion: θ₁ correlates with chaos more strongly than θ₂
   (frictionless), and the ordering reverses under friction.
4. **Approach 1 — sliding window (later abandoned).** 49 consecutive angle pairs
   in, the 50th out, stepped along a single 10,000,000-point trajectory from
   [90°, 90°]. Abandoned because it never changes the initial condition, so it
   fits an erratic curve rather than learning chaotic dependence on ICs, and it
   cannot produce a visualisable trajectory.
5. **Approach 2 — the "time-step based approach"** (the paper's claimed novel
   contribution). Train a *direct operator*

   ```
   (θ₁(0), θ₂(0) [, θ₃(0)], t)  ->  (θ₁(t), θ₂(t) [, θ₃(t)])
   ```

   on a grid of initial conditions, then evaluate on an "in-between" initial
   condition never trained on. **This is not autoregressive** — there is no
   rollout, no error accumulation, and `t` is an ordinary input feature.
6. **Run both frictionless and friction variants.** Friction was introduced
   specifically because the frictionless holdout results were unusable (§3.2:
   "the LSTM returned an RMSE of 0.26 and an R² of 0.23 on the unknown initial
   angle... it was not worth continuing to test new models").
7. **Evaluate 10 models** on R² and RMSE, plus trajectory overlay plots.
8. **Stability analysis.** Lyapunov exponent as the slope of the regression of
   (t, ln error(t)) with error(t) = |θ_perturbed − θ_actual|, over a
   θ₁, θ₂ ∈ [0°, 180°] grid at 15° increments with a 0.1° perturbation in θ₁;
   plus Jacobian eigenvalues at the [0,0] and [180°,180°] equilibria.

---

## 2. Data generation parameters

| Parameter | Value | Source |
|---|---|---|
| Integrator | classical RK4, fixed step | paper §2 + **[code]** |
| g | 9.81 m/s² | paper Fig. 5A |
| l₁ = l₂ = l₃ | 1 m | paper Fig. 5A |
| m₁ = m₂ = m₃ | 1 kg | paper Fig. 5A |
| Initial angular velocities | 0 (released from rest) | **[code]** |
| Angle convention | absolute, from downward vertical, stored in **degrees** | **[code]** |
| State ordering | `[θ₁, ω₁, θ₂, ω₂]` | **[code]** |

### Sliding-window dataset (approach 1, not reproduced here)

| Parameter | Value |
|---|---|
| Initial condition | [90°, 90°] (double), [90°, 60°, 45°] (triple) |
| Duration | 1000 s |
| Step size | paper says 0.001 s; **[code]** `Preprocessing.py` uses `N=3000` over `b=1000`, i.e. h = 1/3 s |
| Dataset size | paper says 10,000,000 points; 1000 s at h = 0.001 is 1,000,000, and the code produces 3,000 |

The three numbers above cannot all be true. This is the single largest
unresolved inconsistency in the paper. Approach 1 is superseded by approach 2 in
the paper's own conclusions, so it is not reproduced.

### Time-step dataset (approach 2 — the one reproduced)

| Parameter | Value | Source |
|---|---|---|
| Duration | 10 s | paper §2.2 + **[code]** `b=10` |
| Samples per trajectory | 2000 | paper §2.2 + **[code]** `N=2000` |
| Step size h | 0.005 s | **[code]** `h=(b-a)/N` |
| Double pendulum ICs | θ₁(0) = 120° fixed; θ₂(0) = 0.0° … 3.0° step 0.1° | paper §2.2 + **[code]** |
| Triple pendulum ICs | θ₁(0) = 120°, θ₂(0) = 0° fixed; θ₃(0) = 0.1° … 3.0° step 0.1° | paper Fig. 18B caption + §3.3 |
| Number of training ICs | paper says 30; **[code]** `angles` list has 31 entries | — |
| Held-out "in-between" IC | [120°, 2.05°] / [120°, 0°, 2.05°] | paper §2.2 |
| Total training rows | 60,000 (paper) / 62,000 (**[code]**, 31 × 2000) | — |
| Friction | `damping1 = damping2 = 0.15` | **[code]** |

**Friction form [code].** The damping term is subtracted *inside the numerator*
of the ω̇ᵢ equation, before dividing by `Lᵢ(2m₁+m₂−m₂cos(2θ₁−2θ₂))`:

```
ω̇₁ = ( −g(2m₁+m₂)sin θ₁ − m₂g sin(θ₁−2θ₂)
        − 2 sin(θ₁−θ₂) m₂ (ω₂²L₂ + ω₁²L₁cos(θ₁−θ₂))
        − damping1·ω₁ ) / ( L₁(2m₁+m₂−m₂cos(2θ₁−2θ₂)) )
```

That is not a clean viscous joint torque — dividing a torque by the
configuration-dependent denominator makes the effective damping coefficient a
function of θ₁−θ₂. It is reproduced as written, because the paper's friction
numbers depend on it.

**Two typos in the reference `angles` list [code].** Entries 8 and 19 are
`[122, 0.7]` and `[122, 1.8]` where the pattern requires `120`. Consequently the
published double-pendulum training set contains two trajectories from a
different θ₁(0), and omits θ₂(0) = 0.7° and 1.8° from the intended grid. Our
reproduction uses the intended grid throughout.

**Equations in the PDF do not match the code.** The PDF's displayed equations (1)
and (2) have a bare `L` in one denominator, an unbalanced bracket in the other,
and a different numerator structure from the code. The code implements the
standard point-mass double-pendulum form, which is what we transcribed.

---

## 3. Preprocessing (this is the load-bearing part)

**[code]**, identical across all time-step notebooks:

1. Integrate each IC to a `(2000, 2)` array of degrees; save as `.npy`.
2. **Per-trajectory `MinMaxScaler`** — `scaler.fit_transform(loaded_data)` is
   called *inside the per-IC loop*, so each trajectory is independently scaled
   to [0, 1] per angle column. Every trajectory therefore spans exactly [0, 1]
   in every output, regardless of whether its true angular range is 40° or
   3000°.
3. Flatten to rows `(θ₁(0) in radians, θ₂(0) in radians, t)` → `(θ̃₁, θ̃₂)`.
   Note the inputs are in **radians** while the targets derive from degrees.
4. Global `MinMaxScaler` on the pooled inputs and on the pooled targets. Because
   step 2 already put every trajectory in [0, 1], the pooled target min is
   exactly 0 and max exactly 1 — the target scaler is the identity map.
5. `torch.utils.data.random_split` 80/20 on the **pooled rows**.

Three consequences worth stating plainly:

- **RMSE is dimensionless and per-trajectory-relative.** A reported RMSE of
  0.027 means 2.7% of *that trajectory's own* angular range. For the frictionless
  [120°, 2.05°] holdout the true range of θ₂ is −146.6° to +790.0°, so scaled
  RMSE 0.027 would be ≈ 25°. RMSE values are not comparable across trajectories
  and are not in physical units. Our reproduction reports degrees alongside.
- **The pooled 80/20 random split interleaves neighbours 5 ms apart** between
  train and test. Any smooth interpolator scores well on it. It measures
  interpolation within trajectories, not generalisation.
- **Per-trajectory target scaling needs the test trajectory's own min and max**,
  so even the held-out-IC evaluation is handed two statistics of the answer.
  We keep the protocol for comparability and report degrees as the honest number.

The **held-out IC test itself is sound** — [120°, 2.05°] is never in training.
That is the paper's real contribution and the number worth reproducing.

---

## 4. Training hyperparameters (the paper's models)

**[code]**, `NEW/NEWLSTM (5).ipynb` and the friction twin:

| Parameter | Value |
|---|---|
| Framework | PyTorch (NNs), scikit-learn (RF, SGD, MLPRegressor) |
| Input size | 3 (double) / 4 (triple) |
| Output size | 2 (double) / 3 (triple) |
| LSTM hidden size | 50 |
| LSTM layers | 3, `batch_first=True`, final-timestep linear head |
| Sequence length | **1** — `sequences.unsqueeze(1)` makes a length-1 sequence |
| Loss | `nn.MSELoss` |
| Optimizer | Adam |
| Learning rate | 0.001 **[code]**; paper §2 says "minimum learning rate of 10⁻⁴" |
| Batch size | 10 |
| Max epochs | 100 |
| Early stopping | patience 10 on validation loss, best weights restored |
| Metrics | `r2_score(..., multioutput='uniform_average')`, `sqrt(mean_squared_error(...))` |
| Hyperparameters | shared across all models "to ensure fairness" (paper §Limitations 4) |

**The recurrent models are not recurrent here.** With `unsqueeze(1)` the
sequence length is 1, so the LSTM/GRU/RNN cells see a single timestep and carry
no state between samples. They are gated feed-forward blocks. This matters for
interpreting the paper's conclusion that "Recurrent Neural Networks are known to
be best suited for time-series problems" — no recurrence over time was exercised
in the time-step experiments.

Models evaluated: RF, MLPRegressor, LSTM, VRNN, BIRNN, StackedRNN (SRNN),
SGDRegressor, GRU, Autoregressive (AR), FFNN. RF and SGD were dropped from the
time-step experiments.

---

## 5. Reported results, time-step approach

Read off Figs. 11, 12, 13, 18B, 18C, 18D and the §4 heatmaps. RMSE is in the
scaled units of §3.

### Double pendulum

| Model | Trained IC [120°,0°], no friction |  | Trained IC, friction |  | Holdout [120°,2.05°], friction |  |
|---|---|---|---|---|---|---|
| | RMSE | R² | RMSE | R² | RMSE | R² |
| LSTM | **0.02701** | **0.99153** | **0.009546** | **0.99873** | **0.01529** | **0.99643** |
| GRU | 0.03838 | 0.98209 | 0.01496 | 0.99685 | 0.01813 | 0.99524 |
| VRNN | 0.04073 | 0.98118 | 0.01498 | 0.99686 | 0.01663 | 0.99597 |
| BIRNN | 0.04015 | 0.98049 | 0.02151 | 0.99349 | 0.01791 | 0.99532 |
| FFNN | 0.06010 | 0.96370 | 0.04353 | 0.97269 | 0.03136 | 0.98570 |
| MLP | 0.07547 | 0.93821 | 0.02356 | 0.99226 | 0.02356 | 0.99259 |
| SRNN | 0.07760 | 0.92720 | 0.03890 | 0.97874 | 0.02816 | 0.98762 |
| AR | 0.14150 | 0.76997 | 0.11470 | 0.61360 | 0.07868 | 0.91967 |

Holdout **without** friction: only LSTM was run — RMSE 0.26, R² 0.23 (§3.2). The
§4 conclusion quotes 0.31 for the same cell; both appear in the paper.

### Triple pendulum

| Model | Trained IC, no friction RMSE | Trained IC, friction RMSE | Holdout [120°,0°,2.05°], friction RMSE | R² |
|---|---|---|---|---|
| GRU | **0.01689** | 0.009113 | **0.006497** | **0.99909** |
| LSTM | 0.02174 | **0.008365** | 0.009112 | 0.99746 |
| BIRNN | 0.02591 (BiDir LSTM) | 0.01822 | 0.01822 | 0.99230 |
| VRNN | 0.08616 | 0.02436 | 0.02121 | 0.99039 |
| MLP | 0.08926 | 0.01645 | 0.01645 | 0.99201 |
| FFNN | 0.11400 | 0.01807 | 0.02436 | 0.98739 |
| SRNN | 0.11010 | 0.02975 | 0.02965 | 0.98130 |
| AR | 0.14790 | 0.09855 | 0.09143 | 0.81274 |

Paper's headline claims: LSTM best for the double pendulum in both friction
regimes; VRNN best for the triple pendulum under the sliding window and GRU best
under the time-step approach; AR worst throughout.

Chaos characterisation: largest Lyapunov exponent up to ≈ 1.4 s⁻¹ over the IC
grid (Fig. 20 heatmap, values ×1000 over 10 s / 5000 timesteps); [0,0] is an
undamped oscillator (purely imaginary Jacobian eigenvalues ±5.7874i, ±2.3972i)
and [180°,180°] an unstable saddle (real ±5.7874, ±2.3972).

---

## 6. What this reproduction changes

| Paper / reference code | Here | Why |
|---|---|---|
| LSTM, GRU, VRNN, BIRNN, SRNN, FFNN, MLP, AR | Takagi–Sugeno fuzzy inference systems (`tribblefis.MixtureOfGaussiansFuzzyRegressor`), one FIS per output angle | The assignment: reproduce with a collection of FIS |
| 31 ICs including two `122°` typos | Intended 31-point grid, θ₁(0) = 120° throughout | The typos are clearly unintended |
| Triple-pendulum EOM from Yesilyurt | This repo's SymPy Lagrangian `NPendulum`, cross-checked against the paper's own closed-form n=2 RHS every run (agreement 1.07e-14 over 200 random states) | Independent derivation; do not trust one source |
| θ₁(0) fed in as a constant input | Zero-variance inputs dropped | A Gaussian membership on a constant feature has σ = 0 and a degenerate firing strength. A net can absorb a dead input; a fuzzy partition cannot. |
| RMSE reported only in scaled units | Scaled **and** degrees | Scaled RMSE is per-trajectory-relative and not physically interpretable |
| Two metric settings | Three: `pooled`, `trained_ic`, `holdout_ic` | Separates within-trajectory interpolation from generalisation to a new IC |
| Animations | None | Explicitly out of scope |

Integrator provenance for this reproduction: RK4 at h = 0.005 s over 10 s gives
max relative energy drift 6.6e-7 on the undamped [120°, 0°] run.
