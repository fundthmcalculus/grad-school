# What a Fuzzy System Can and Cannot Learn About a Chaotic Pendulum: A Reproduction, a Zero-Parameter Baseline, and the Limits of Time-Indexed Operators

**Scott Phillips**
University of Cincinnati, Cincinnati, Ohio 45221
`phillis8@mail.uc.edu`

**Kelly Cohen, PhD**
University of Cincinnati, Cincinnati, Ohio 45221
`cohenky@ucmail.uc.edu`

---

## Abstract

We reproduce a recent multi-pendulum chaos-prediction study [12] that benchmarks
ten neural and machine-learning models using Takagi–Sugeno–Kang (TSK) fuzzy inference
systems. The FIS attains the best RMSE in six of seven cells and the best $R^2$ in all
seven, demonstrating competitive accuracy. However, we show this ranking does not
reflect learned chaotic dynamics. The benchmark's held-out initial condition lies exactly
midway between two trained ones, and averaging those two trajectories — a predictor with
no parameters and no fitting — beats every published model by 6× on the double pendulum
and 18× on the triple. Extending to a five-link chain reveals this baseline degrades
twentyfold as the bracketing pair decorrelates, establishing a criterion for detecting
interpolation-dominated benchmarks. For extrapolation, time-indexed FIS models have no
validity beyond the training window: predictions diverge by four orders of magnitude
within one timestep of the training edge, and improving in-window accuracy actually
worsens extrapolation by three orders of magnitude. We address target representation,
proving that no bounded continuous scalar can represent an accumulating angle, and show
that a hysteresis wrap reduces representation discontinuities yet fails to improve
accuracy, whereas $(\sin\theta, \cos\theta)$ improves extrapolated error. All numbers,
figures, and findings are reproducible from the accompanying code.

**Keywords:** Takagi–Sugeno–Kang systems; fuzzy inference systems; chaotic dynamics; surrogate modeling;
benchmark validity; extrapolation; time-indexed operators.

---

## 1. Introduction

Fuzzy inference systems are universal approximators [17] and, in
Takagi–Sugeno–Kang form [15], admit closed-form consequent solutions that make
them unusually cheap to fit. That combination makes them an attractive substitute
for neural surrogates in low-dimensional physical modeling. It also makes them a
good instrument for a second purpose: because a TSK system is transparent about
*where* it is interpolating and where it is not, its failures localize the
difficulty in a way that a deep network's do not.

This paper uses a TSK collection on the multi-pendulum
prediction benchmark of Ramachandruni et al. [12], which evaluates ten neural and
classical regressors on predicting double- and triple-pendulum trajectories. Our
contributions:

1. **A competitive FIS reproduction** (§4). A per-output TSK collection with Gaussian
   antecedents and ridge-solved consequents attains the lowest RMSE in six of seven
   benchmark cells and the highest $R^2$ in all seven, in <20 seconds training.

2. **A benchmark-validity result** (§5). The held-out initial condition is the
   exact midpoint of two trained ones. A zero-parameter midpoint interpolant
   outperforms every published model by 6–18×. The benchmark's headline
   configuration measures one-dimensional interpolation of a smooth grid, not
   chaotic prediction. We extend to $n = 5$ and quantify where this degeneracy breaks.

3. **A structural limit of time-indexed operators** (§6). Admitting $t$ as an input
   bounds validity to the training window by construction: predictions diverge by four
   orders of magnitude at the edge, and improving in-window accuracy makes extrapolation
   worse by three orders of magnitude. Within this family, hyperparameter tuning on the
   reported metric harms the model's horizon.

4. **A topological limit on target representation** (§7). No bounded continuous scalar
   can represent an accumulating angle, which rules out the `np.unwrap` family a priori.
   A hysteresis wrap reduces representation discontinuities yet fails to improve accuracy,
   whereas $(\sin\theta, \cos\theta)$ is the only representation that improves extrapolated error.

---

## 2. Background

### 2.1 The system

The planar double pendulum with point masses is the canonical minimal chaotic
mechanical system [13, 14]. With absolute angles $\theta_1, \theta_2$ measured
from the downward vertical, unit masses $m_1 = m_2$, unit lengths
$\ell_1 = \ell_2$, gravity $g$, and viscous joint damping $c$, the benchmark uses

$$
\dot\omega_1 = \frac{N_1}{\ell_1 D}, \qquad
\dot\omega_2 = \frac{N_2}{\ell_2 D},
\tag{1}
$$

$$
D = 2m_1 + m_2 - m_2 \cos 2\Delta, \qquad \Delta = \theta_1 - \theta_2,
$$

with numerators

$$
\begin{aligned}
N_1 &= -g(2m_1 + m_2)\sin\theta_1 - m_2 g \sin(\theta_1 - 2\theta_2) \\
    &\quad - 2\sin\Delta \, m_2\!\left(\omega_2^2 \ell_2 + \omega_1^2 \ell_1 \cos\Delta\right) - c\,\omega_1,
\end{aligned}
\tag{2}
$$

$$
\begin{aligned}
N_2 &= 2\sin\Delta\Big(\omega_1^2 \ell_1 (m_1 + m_2) + g(m_1 + m_2)\cos\theta_1 \\
    &\quad + \omega_2^2 \ell_2 m_2 \cos\Delta\Big) - c\,\omega_2 .
\end{aligned}
\tag{3}
$$

For $n > 2$ links we use a symbolic Lagrangian construction, forming
$M(q)\ddot q = f(q, \dot q)$ and solving numerically per evaluation; this
reproduces (1) at $n = 2$ and generalizes the $n$-point-mass formulation of
Yesilyurt [19]. Agreement between the two derivations is $1.07 \times 10^{-14}$
in $\max|\dot{\mathbf{x}}|$ over 200 random states, asserted on every
data-generation run.

### 2.2 The benchmark protocol

Ramachandruni et al. evaluate two framings and retain the second, which they term
the *time-step approach* and identify as their principal contribution. It fits a
direct operator

$$
\mathcal{G}: \big(\theta_1(0), \dots, \theta_n(0),\, t\big)
\longmapsto \big(\theta_1(t), \dots, \theta_n(t)\big),
\tag{4}
$$

trained on a grid of initial conditions and evaluated at an unseen "in-between"
one. Crucially, (4) is **not** autoregressive: $t$ is an ordinary input feature,
there is no rollout, and no error accumulates. In the authors' implementation the
recurrent models receive a sequence of length one, so no recurrence over time is
exercised at all — a point worth noting given the study's conclusion favors
recurrent architectures [7].

Data are generated by fixed-step RK4 at $h = 5$ ms over 10 s (2000 samples), for
31 initial conditions with $\theta_1(0) = 120^{\circ}$ and the final link's angle swept
over $[0^{\circ}, 3^{\circ}]$ in $0.1^{\circ}$ steps. The held-out condition is $2.05^{\circ}$. Targets are
min–max scaled **per trajectory**, independently per angle, before pooling; every
trajectory therefore spans exactly $[0, 1]$ in every output. A reported RMSE is
thus a fraction of that trajectory's own angular range, not an angle — a point we
return to in the Threats to Validity section.

### 2.3 Related work

The benchmark frames operator learning: fitting a direct map from initial condition and
time to trajectory state (4). This is distinct from learning dynamics (the vector field),
which is pursued by SINDy [1], neural ODEs [2], and physics-informed neural networks [11].
The time-indexed formulation (4) is computationally attractive—a single call answers any
time within the window—but as we show, this convenience comes at a structural cost.

---

## 3. A TSK Collection for the Time-Step Operator

We approximate (4) with one first-order TSK system per output angle. For output
$j$, rules are indexed by a partition of the target range into $R$ buckets; rule
$r$ has Gaussian antecedent memberships

$$
\mu_{r,i}(x_i) = \exp\!\left(-\frac{(x_i - c_{r,i})^2}{2\sigma_{r,i}^2}\right)
$$

fitted from data per (feature, bucket) pair, firing strength
$\beta_r(\mathbf{x}) = \prod_i \mu_{r,i}(x_i)$ combined by a product t-norm, and
an affine consequent. The prediction is the firing-weighted blend

$$
\hat y_j(\mathbf{x}) = \sum_{r=1}^{R}
\frac{\beta_r(\mathbf{x})}{\sum_{s} \beta_s(\mathbf{x})}
\Big(\bar y_r + \mathbf{\phi}(\mathbf{x})^\top \mathbf{a}_r\Big).
\tag{5}
$$

For fixed antecedents, (5) is linear in $(\bar y_r, \mathbf{a}_r)$, so a single
firing-weighted ridge least-squares solution yields the exact minimizer of the
weighted MSE — no iterative optimizer, unlike gradient-trained TSK variants
or ANFIS [8]. Antecedent placement from data rather than a grid follows the
cluster-estimation tradition [3, 18].

Three implementation details matter.

First, $\theta_1(0)$ is constant across the sweep, so a Gaussian membership on it
has $\sigma = 0$ and a degenerate firing strength; **zero-variance inputs must be
dropped**. A network absorbs a dead input, a fuzzy partition cannot.

Second, the **input** scaler is unclipped: at $t = 20$ s it returns $2.0$ rather
than saturating at $1.0$. This is deliberate and load-bearing — §6 depends on the
failure past the window being visible rather than disguised as a plateau.

Third, the **target** scaler is fitted on the training window and then applied to
the whole test span. Per-trajectory min–max only makes training and test
commensurable when both are normalized over the same duration: training targets
span exactly $[0,1]$ by construction, so fitting a 20 s holdout's scaler over its
full length would leave its first 10 s spanning only $[0, 0.678]$ on the
frictionless double pendulum, and a model trained to emit over $[0,1]$ would
overshoot by ${\sim}1.5\times$ for reasons unrelated to its dynamics. Fitting on
the window keeps every in-window number comparable to the 10 s protocol and leaks
nothing the protocol does not already leak — whereas fitting over 20 s would
additionally leak the range of the region being extrapolated into. Scaled truth
beyond 10 s may consequently exceed $[0,1]$, which is correct: the chain does leave
the window it was normalized against. Friction datasets are indifferent to the
choice, their 20 s and 10 s ranges being bitwise equal; only the frictionless ones
move.

We report three metric families: **pooled** (the original protocol's random 80/20
split over pooled rows), **trained-IC** (all samples of a trajectory that was in
training), and **holdout-IC** (the never-trained $2.05^{\circ}$ trajectory). Only the
last tests the claim the operator makes.

---

## 4. Reproduction Results

Table 1 gives the comparison, taking the original numbers at face value.
Hyperparameters were selected over 253 configurations; capacity ($R$, the rules
per output), consequent order, and ridge strength were the only knobs that
mattered, in that order.

**Table 1.** FIS against the best of the ten models in [12]. RMSE in the
benchmark's scaled units. The FIS attains the lowest RMSE in six of seven
populated cells and the highest $R^2$ in all seven. Cell 6 was not run in the
original study.

| Cell                         | Best published | FIS         | Winner |
|------------------------------|----------------|-------------|--------|
| 2-link, no friction, trained | LSTM 0.0270    | **0.0126**  | FIS    |
| 2-link, no friction, holdout | LSTM 0.26      | **0.184**   | FIS    |
| 2-link, friction, trained    | LSTM 0.00955   | **0.00550** | FIS    |
| 2-link, friction, holdout    | LSTM 0.0153    | **0.0120**  | FIS    |
| 3-link, no friction, trained | GRU 0.0170     | 0.0191      | GRU    |
| 3-link, no friction, holdout | —              | 0.162       | —      |
| 3-link, friction, trained    | GRU 0.00911    | **0.00718** | FIS    |
| 3-link, friction, holdout    | GRU 0.00650    | **0.00477** | FIS    |

The single loss is the frictionless triple-pendulum trained cell, where the
published GRU records a lower RMSE (0.0170 vs. 0.0191) but a lower $R^2$ (0.985 vs. 0.994).
The two metrics disagree because the original work's Fig. 18B bar labels
and its Fig. 22 summary heatmap disagree with one another for that panel; we used
the heatmap.

A closed-form TSK collection therefore matches or beats eight tuned neural
baselines on this task. That is the result the benchmark was designed to produce.
The remainder of this paper argues it should not be believed.

---

## 5. How Much of the Benchmark Is Interpolation?

### 5.1 A predictor with no parameters

The held-out initial condition $2.05^{\circ}$ lies exactly midway between the trained
conditions $2.0^{\circ}$ and $2.1^{\circ}$. Consider therefore

$$
\hat{\mathbf{\theta}}_{\text{mid}}(t) = \tfrac{1}{2}\!\left(
\mathbf{\theta}^{(2.0^{\circ})}(t) + \mathbf{\theta}^{(2.1^{\circ})}(t)\right),
\tag{6}
$$

the mean of two **training** trajectories. It has no parameters, no fitting, and
no model. Scored in the benchmark's own metric (Table 2), it beats every
published model and our own FIS.

**Table 2.** Held-out-IC RMSE with friction, benchmark units. The zero-parameter
midpoint interpolant (6) outperforms every learned model.

| Predictor          | 2-link      | 3-link       | 5-link     |
|--------------------|-------------|--------------|------------|
| Midpoint (6)       | **0.00250** | **0.000359** | **0.0515** |
| Nearest trained IC | 0.0111      | 0.00187      | 0.0520     |
| FIS (ours)         | 0.0120      | 0.00477      | 0.0672     |
| Best published     | 0.0153      | 0.00650      | —          |

The margin is 6× on the double pendulum and 18× on the triple. Even *copying a
single neighboring trajectory verbatim* beats the best published model in both
cells. Since the original study's conclusions generalize from precisely these
friction numbers, this matters: they do not demonstrate learned chaotic dynamics.
They demonstrate that with damping applied, the map from initial condition to
trajectory is smooth enough over a $0.1^{\circ}$ grid that one-dimensional interpolation
suffices.

### 5.2 Why: bracket decorrelation, measured

The reason is quantifiable. Let

$$
s(t) = \max_i \left|\theta_i^{(2.1^{\circ})}(t) - \theta_i^{(2.0^{\circ})}(t)\right|
$$

be the separation of the bracketing pair, and $\rho$ the held-out trajectory's
own angular range. Interpolation (6) can only work while $s(t) \ll \rho$.
Table 3 reports both.

**Table 3.** Separation of the bracketing trained pair at $t = 20$ s relative to
the held-out trajectory's range, with the fitted exponential rate $\lambda$.
Damping keeps the pair coherent; without it the pair decorrelates well inside the
test window.

| Dataset             | $s(20)$ | $s/\rho$ | $\lambda$ (s⁻¹) | Midpoint RMSE |
|---------------------|---------|----------|-----------------|---------------|
| 2-link, friction    | 1.7°    | 0.6 %    | 0.56            | 0.00250       |
| 3-link, friction    | 12.3°   | 6.0 %    | 0.34            | 0.000359      |
| 5-link, friction    | 31.4°   | 7.0 %    | 1.25            | 0.0515        |
| 2-link, no friction | 1780°   | 129 %    | 1.85            | 0.226         |
| 3-link, no friction | 2267°   | 243 %    | 0.87            | 0.166         |
| 5-link, no friction | 844°    | 70 %     | 1.41            | 0.227         |

With damping the pair stays under 7 % separation across $n = 2, 3, 5$; the
interpolation problem is genuinely easy at every chain length we tested.

![Separation of the two trained ICs bracketing the 2.05° holdout, showing how damping keeps
the trajectories coherent while frictionless systems diverge exponentially.](figures/trajectory_snapshots.png)

*Figure 1. Bracket separation over time, by chain length and regime.* Without damping, it grows
from $0.1^{\circ}$ to over $600^{\circ}$ — the training set then contains two mutually
contradictory answers to the same query, with nothing to choose between them. Any
model is pushed toward the conditional mean, which caps achievable $R^2$
regardless of architecture. This is a property of the dataset, not of any model,
and it explains the original study's frictionless holdout result ($R^2 = 0.23$)
without reference to the LSTM at all.

![Separation of the two trained ICs bracketing the 2.05° holdout, log scale. The
three friction datasets stay coherent for the whole window; the frictionless ones
grow three orders of magnitude.](figures/fig_bracket.png)

*Figure 1. Bracket separation against time.*

### 5.3 The degeneracy is chain-length-dependent

Extending the protocol to $n = 5$ — which the original work does not attempt —
shows the degeneracy continuing to grow with chain length, though the separation
metric grows more gently than the RMSE it produces: by $n = 5$ the bracketing pair
separates to 7.0 % of the range (up from 6.0 % at $n = 3$ and 0.6 % at $n = 2$),
while the midpoint baseline's RMSE degrades twentyfold, from 0.0025 to 0.0515. It
remains ahead of our FIS (0.0672), but by 1.3× rather than 4.8×. Across all three chain
lengths the FIS loses to the baseline on every friction holdout and beats both
baselines on every frictionless one: it wins where the answer is a blend over
many training conditions and loses where it is one nearby trajectory copied
accurately.

The prescription is straightforward. Report a zero-parameter baseline alongside
every model, and widen the initial-condition spacing until that baseline fails.
Table 3 supplies the criterion directly.

---

## 6. Time-Indexed Operators Have No Horizon

### 6.1 Why, analytically

We now hold training fixed at 10 s and evaluate to 20 s. The failure of (4) past
its window is structural, not a tuning deficiency.

Inputs are min–max scaled on the training range, so $t \in [0, T]$ maps to
$\tilde t \in [0, 1]$ and $t = 2T$ maps to $\tilde t = 2$. Two mechanisms then
act together. First, each antecedent is a Gaussian centred in $[0, 1]$ with
$\sigma = O(1/R)$, so at $\tilde t = 2$

$$
\mu_r(\tilde t) \sim
\exp\!\left(-\frac{(2 - c_r)^2}{2\sigma_r^2}\right) \longrightarrow 0
\tag{7}
$$

for **every** rule, and the normalization in (5) divides a vanishing numerator by
a vanishing denominator: the weights become the ratio of two underflowing
quantities and rule selection is numerical noise. Second, the consequents are
affine in $\tilde t$ and unbounded, so whichever rule wins contributes
$\bar y_r + a_r \tilde t$ evaluated at twice its fitted range. The composition
diverges rather than saturating.

Note the direction of failure is an accident of the consequent, not a property of
TSK systems. Where a rolled-out state leaves antecedent support and the surviving
rule contributes $\approx 0$, the same mechanism produces *flat-lining* instead.
Outside the antecedents' support a TSK model has no defined behavior; which way
it fails is incidental.

### 6.2 Measured

Table 4 confirms the prediction and adds something we did not expect.

**Table 4.** Double pendulum FIS, trained on 10 s, scored to 20 s against a
DOP853 reference. Angle RMSE in degrees. With the time variable as input,
*in-window accuracy and extrapolation error are inversely related.*

| Capacity    | Variables | Train (s) | 0–10 s    | 10–20 s  |
|-------------|-----------|-----------|-----------|----------|
| 40 rules    | 2         | 7.1       | 22.0°     | 443°     |
| 120 rules   | 2         | 20.6      | 4.35°     | 20 070°  |
| 300 rules   | 2         | 71.7      | **3.11°** | 338 555° |
| 8 harmonics | 18        | 175.8     | 20.1°     | 146°     |

Refining $R$ from 40 to 300 improves the in-window RMSE from $22.0^{\circ}$ to $3.11^{\circ}$
while degrading the extrapolation from $443^{\circ}$ to $338\,555^{\circ}$ — three orders of
magnitude worse. A sharper in-window fit requires steeper affine consequents, and
those are precisely what is extrapolated off the end of the window. **Tuning the
model for its reported metric makes its horizon worse.** We are not aware of this
trade-off being stated for time-indexed surrogates, and it inverts the usual
reading of a capacity sweep.

![FIS predictions diverging sharply at the training-window edge (t = 10 s), illustrating the
structural failure of time-indexed operators beyond their training domain.](figures/n2_rollout_comparison_all.png)

*Figure 2. Extrapolation failure across FIS capacity levels.*

The one mitigation within the family is a periodic input encoding: appending
$\sin k\omega_0 t, \cos k\omega_0 t$ for $k \le 8$ with $\omega_0 = \sqrt{g/\ell}$
yields the family's best extrapolation ($146^{\circ}$) at the cost of the worst
in-window accuracy and, at 18 variables and 176 s, the highest training cost in
the study. Pushed further it collapses: at 24 harmonics (50 inputs) $R^2$ falls
to $-2.85$, because the product t-norm over 50 memberships drives every firing
strength to underflow. **Antecedent dimensionality, not consequent
expressiveness, is the binding constraint.**

![θ(t) truth versus FIS prediction for the friction double pendulum at the
held-out IC, with the training-window edge marked. The prediction leaves the axis
within one timestep of t = 10 s.](figures/fig_angles_double_friction_holdout.png)

*Figure 2. Prediction against truth across the training-window edge.*

---

## 7. Angle Representation and the Topology of the Target

Section 6 concerns what the model is indexed by. This one concerns what it
is asked to *emit*. The frictionless chains spin: $\theta$ reaches $1501^{\circ}$ on
the double pendulum and $1589^{\circ}$ on the quintuple over the 20 s window. The
per-trajectory min–max scaling of §2.2 therefore normalises against a range
dominated by monotone accumulation rather than by oscillation, and the FIS is
asked to fit an unbounded, drifting quantity. Bounding the representation is an
obvious thing to try.

### 8.1 A topological obstruction

There is a limit on what any such attempt can achieve, and it is worth stating
before the experiments rather than discovering it in them.

**Claim.** No bounded, continuous, single-valued scalar function of time can
represent an angle that accumulates without bound.

*Proof sketch.* Let $\theta(t)$ be continuous with
$\theta(T) - \theta(0) = N\!\cdot\!360^{\circ}$ for some $N \ge 1$, and let
$\phi(t) = \theta(t) - 360^{\circ} k(t)$ be a representation confined to
$[-L, L]$. If $\phi$ is continuous then $k$ is continuous and integer-valued,
hence constant, so $\phi = \theta - \text{const}$ is unbounded whenever $\theta$
is — contradiction. Therefore, $k$ must jump, and each jump is a $360^{\circ}$
discontinuity in $\phi$. ∎

So bounded and continuous cannot both hold for a scalar. Any scheme must give up
one, and the practical question is only *where* it gives it up.

A corollary settles a natural first idea. `np.unwrap` and its relatives *recover*
continuity from a wrapped signal; they cannot impose a bound, because by the claim
the continuous representative is the unbounded one. Empirically the point is
sharper still: applied to our raw trajectories `np.unwrap` is a no-op, changing
them by $1 \times 10^{-13}$ degrees, because RK4 at $h = 5$ ms never moves an
angle more than $4.6^{\circ}$ per step and so no wrap is ever detected. Wrapping
and then unwrapping returns the original to $2 \times 10^{-13}$ degrees.

### 8.2 Three representations

**Pointwise wrap.** Subtract $360^{\circ} \lfloor \cdot \rceil$ whenever
$|\theta| > L$, choosing the branch independently at each sample. For $L = 180^{\circ}$
this is the ordinary modulo; for $L > 180^{\circ}$ there is an overlap band of
width $L - 180^{\circ}$ in which a configuration has two admissible
representatives.

**Hysteresis wrap.** Make the branch *stateful*: retain it until the value leaves
$[-L, L]$, then re-branch. A re-branch from $+L$ lands at $L - 360^{\circ}$, deep
inside the window when $L$ is large.

**Sine–cosine pair.** Emit $(\sin\theta, \cos\theta)$ and recover
$\hat\theta = \operatorname{atan2}(\hat s, \hat c)$. This is bounded in $[-1,1]$
*and* continuous, escaping the topological obstruction by spending a second output rather than by
contradicting it: a circle does not embed in an interval without a cut, but it
embeds in the plane without one. The winding number is discarded, which costs
nothing because (1)–(3) depend on the angles only through $\sin$ and $\cos$.

Because wrapping changes what an error means, we score in **circular** angular
error, $((\hat\theta - \theta + 180^{\circ}) \bmod 360^{\circ}) - 180^{\circ}$,
which is the physically meaningful quantity and the only one comparable across
wrapped and unwrapped representations. It is more forgiving than the benchmark's
metric — a prediction of $430^{\circ}$ against a truth of $790^{\circ}$ scores
zero, being the same configuration — so it is used only within this section.

### 8.3 The overlap band works but does not help

Table 6 counts representation-induced discontinuities: sample-to-sample jumps
exceeding $180^{\circ}$, which the true dynamics never produce at this sampling
rate and which are therefore artifacts.

**Table 6.** Discontinuities in the frictionless double-pendulum training set.
Pointwise wrapping gets *worse* as the overlap band widens; hysteresis gets
better.

| $L$           | pointwise | hysteresis |
|---------------|-----------|------------|
| $180^{\circ}$ | 66        | 66         |
| $240^{\circ}$ | 76        | **53**     |
| $300^{\circ}$ | 94        | **51**     |
| $360^{\circ}$ | 111       | **48**     |
| $420^{\circ}$ | 93        | **47**     |

The pointwise trend is explained by its statelessness: a trajectory oscillating
inside the overlap band flips branch on every crossing, and a wider band is a
wider flip zone. Hysteresis removes the flip-flop and the count falls
monotonically. The two agree at $L = 180^{\circ}$, as they must, there being no
band to exploit. **The mechanism therefore works exactly as intended.**

It does not follow that accuracy improves, and Table 7 shows it does not.

**Table 7.** Circular error, in-window / past-window, in degrees. 120 rules,
quantile partitioning throughout so the partition is not a second variable.

| dataset          | no wrap         | $\pm180^{\circ}$ | pointwise $\pm360^{\circ}$ | hysteresis $\pm360^{\circ}$ | sin/cos             |
|------------------|-----------------|------------------|----------------------------|-----------------------------|---------------------|
| 2-link, no fric. | 81.9 / 107.4    | 73.7 / 106.9     | 68.5 / 108.0               | **62.5** / 108.9            | 74.1 / **92.1**     |
| 3-link, no fric. | 59.0 / 108.7    | **49.8** / 106.9 | 54.9 / 108.3               | 59.5 / 109.1                | 48.0 / **80.1**     |
| 5-link, no fric. | 51.9 / 107.7    | 50.5 / 111.4     | 59.0 / 103.7               | 51.2 / 104.2                | **43.6** / **71.9** |
| 5-link, friction | **12.3** / 92.2 | 13.7 / 107.2     | 38.3 / 92.8                | 15.8 / 95.8                 | 12.1 / 109.1        |

Hysteresis at $\pm360^{\circ}$ cuts the double pendulum's in-window error from
$68.5^{\circ}$ to $62.5^{\circ}$, but *raises* the triple pendulum's from
$54.9^{\circ}$ to $59.5^{\circ}$ while cutting its discontinuity count from 93 to 56.
**Representation jitter is not the binding constraint on accuracy here**, and a
mechanism can be validated on its own terms while failing to move the metric it
was introduced to move.

![Circular error by representation scheme and chain length, showing that hysteresis wrapping
reduces discontinuities but fails to improve (and sometimes worsens) prediction accuracy.](figures/rollout_error_vs_n.png)

*Figure 3. Representation effects on prediction error.* Where plain wrapping does help, smaller is better:
$\pm180^{\circ}$ is the best wrap on two of four datasets, consistent with the
range penalty a wider window carries.

### 8.4 Sine–cosine is the representation to use

It is best in-window on three of four datasets and best past the window on three
of four, and it is **the only representation that improves the extrapolated error
at all** — $92.1^{\circ}$, $80.1^{\circ}$, $71.9^{\circ}$ on the three
frictionless sets against $104$–$111^{\circ}$ for every wrap variant. The key insight
is that only a representation that changes the *output geometry* improves
extrapolation past the training window.

One exception: on the damped five-link chain — the sole dataset where an angle
spins but the dynamics dissipate — sin/cos has the best in-window error
($12.1^{\circ}$) and the worst extrapolation ($109.1^{\circ}$).

### 8.5 The unit-circle constraint is not a lever

The predicted pair is not confined to the unit circle: its mean radius is $0.843$,
$0.865$, $0.900$ and $0.979$ across the four datasets, so up to 16 % of the
output is not a valid angle. Renormalizing before $\operatorname{atan2}$ is the
obvious remedy and **cannot work**: $\operatorname{atan2}$ is invariant under
positive scaling of both arguments,
$\operatorname{atan2}(ks, kc) = \operatorname{atan2}(s, c)$, verified to
$4 \times 10^{-16}$ rad. The angular error lives in the ratio $s/c$, which no
rescaling of the magnitude alters.

What the radius does carry is shrinkage, and it carries it in a way that
corroborates the rest of the paper.

**Table 8.** Sine–cosine targets, frictionless double pendulum, against capacity.

| rules | in-window | past-window | mean radius | $\mathrm{corr}(1-r, \lvert e \rvert)$ |
|-------|-----------|-------------|-------------|---------------------------------------|
| 40    | **67.8°** | 102.3°      | 0.723       | **0.541**                             |
| 120   | 74.1°     | **92.1°**   | 0.843       | 0.287                                 |
| 300   | 74.0°     | 110.1°      | 0.835       | 0.240                                 |

The radius rises from $0.72$ to $0.84$ between 40 and 120 rules and then
*saturates* — 300 rules gives $0.835$, not something approaching $1$. The residual
shortfall is therefore not underfitting that capacity would remove; it is the
model hedging toward the conditional mean, the same behavior identified in §5.2 from
bracket decorrelation. A shrunken radius is how the sine–cosine representation displays it.

The per-sample correlation between radius shortfall and angular error is positive
at every capacity ($+0.24$ to $+0.54$), so low-radius samples really are the wrong
ones — but it explains only 6–29 % of error variance. That is a usable soft flag,
not calibrated uncertainty, and we record it as such rather than as a free
confidence estimate.

Finally, note that sin/cos is capacity-saturated on this dataset: in-window error
is best at 40 rules and no better at 300. The Table 7 figures, held at 120 rules
for comparability, accordingly understate it.

---

## 9. Threats to Validity

**Scaled RMSE is not an angle.** Per-trajectory min–max scaling makes a reported
RMSE a fraction of that trajectory's own range. The frictionless double-pendulum
holdout spans $225^{\circ}$ and $937^{\circ}$ in its two angles, so the original study's RMSE of
0.26 corresponds to roughly $177^{\circ}$. We report degrees throughout alongside scaled
values.

**The normalization assumes fixed-length trajectories.** Per-trajectory min–max
scaling is only well-posed when every trajectory is normalized over the same
duration, and the benchmark never has to confront this because all its trajectories
are 10 s. Testing on a longer horizon does, and §3 explains how we resolve it —
fit the target scaler on the training window. We flag the assumption because it is
latent in the published protocol, not because it affects any number here.

**The frictionless reference is not converged.** Energy drift attests that an
integrator is self-consistent, not that a trajectory is converged — on a chaotic
system these differ sharply. Refining $h$ by 2×, 4×, 8× leaves the friction
references unchanged to $0.00^{\circ}$ over 20 s, but moves the frictionless ones by
hundreds of degrees, with successive refinements parting company from
$t \approx 8.9$–11.5 s. An independent check against DOP853 [10, 6, 16] at
rtol $= 10^{-12}$ agrees on those times to two decimals. Consequently the
frictionless 10–20 s figures should be read as "the surrogate does not track a
reference that is itself not reproducible there" — weaker than the friction rows
support. All headline claims in §6 are friction-only for this reason.

**Scope.** The FIS results cover $n \in \{2, 3, 5\}$ in both friction and frictionless regimes.

---

## 10. Conclusion

A closed-form TSK collection matches or beats eight tuned neural baselines on a
published multi-pendulum chaos benchmark, at seconds of training. That result is
also the least informative one in this paper, because a zero-parameter midpoint
interpolant beats all of them by 6–18×: the benchmark's friction configuration at
$n = 2$ and $n = 3$ measures interpolation of a smooth initial-condition grid, not
chaotic prediction. Extending to five links shows where that degeneracy breaks
and supplies a criterion — bracket separation relative to target range — for
designing a grid that does not admit it.

On extrapolation the finding is sharper. A time-indexed FIS that admits $t$ as an input
has no validity outside its training window, and within that family in-window accuracy trades
*against* extrapolation: refining from 40 to 300 rules improves the in-window RMSE from $22.0^{\circ}$
to $3.11^{\circ}$ while degrading the extrapolation from $443^{\circ}$ to $338\,555^{\circ}$ — three
orders of magnitude worse. This inverse relationship means hyperparameter tuning on the reported
metric actively harms the model's horizon.

The representation of the target obeys its own limit. No bounded continuous scalar
can represent an angle that accumulates, which disposes of the `np.unwrap` family
before any experiment is run. Of the schemes that remain, the one that reduces
representation discontinuities monotonically — a hysteresis wrap — does exactly
that and still fails to improve accuracy on half the datasets, whereas emitting
$(\sin\theta, \cos\theta)$ escapes the obstruction by spending a second output and
is the only representation that improves error past the training window. The
general lesson is one this paper meets three times over: a mechanism can be
validated on its own terms and still not move the quantity it was introduced to
move, and only measuring both tells you which happened.

For fuzzy modeling of chaotic mechanical systems, the practical
recommendations are: report a zero-parameter baseline before claiming any model success;
recognize that time-indexed operators have no horizon beyond the training window; and when
representing unbounded angles, emit $(\sin\theta, \cos\theta)$ rather than a bounded scalar.

**Reproducibility.** All datasets, 253 scored configurations, figures, and negative
results are generated by the accompanying code, with provenance assertions on
every run: agreement between two independent derivations of the equations of
motion ($1.07 \times 10^{-14}$), RK4 convergence order (≈16× per halving), and
refit reproduction of every swept score to $10^{-9}$.

---

## References

Keys in `monospace` match `references.bib`.

1. `brunton2016sindy` — S. L. Brunton, J. L. Proctor, J. N. Kutz.
   Discovering governing equations from data by sparse identification of
   nonlinear dynamical systems. *PNAS* 113(15):3932–3937, 2016.
   doi:10.1073/pnas.1517384113
   https://www.pnas.org/doi/10.1073/pnas.1517384113
2. `chen2018neuralode` — R. T. Q. Chen, Y. Rubanova, J. Bettencourt,
   D. Duvenaud. Neural ordinary differential equations. *NeurIPS* 31, 2018.
   https://papers.nips.cc/paper_files/paper/2018/file/69386f6bb1dfed68692a24c8686939b9-Paper.pdf
3. `chiu1994fuzzy` — S. L. Chiu. Fuzzy model identification based on
   cluster estimation. *J. Intelligent and Fuzzy Systems* 2(3):267–278, 1994.
   doi:10.3233/IFS-1994-2306
4. `greydanus2019hamiltonian` — S. Greydanus, M. Dzamba, J. Yosinski.
   Hamiltonian neural networks. *NeurIPS* 32, 2019.
   https://proceedings.neurips.cc/paper_files/paper/2019/file/26cd8ecadce0d4efd6cc8a8725cbd1f8-Paper.pdf
5. `cranmer2020lagrangian` — M. Cranmer, S. Greydanus, S. Hoyer,
   P. Battaglia, D. Spergel, S. Ho. Lagrangian neural networks. *ICLR 2020 Deep
   Differential Equations Workshop*, 2020.
   https://hunterheidenreich.com/notes/machine-learning/model-architectures/lagrangian-neural-networks/
6. `hairer1993solving` — E. Hairer, S. P. Nørsett, G. Wanner. *Solving
   Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Springer, 1993.
   doi:10.1007/978-3-540-78862-1
   https://link.springer.com/book/10.1007/978-3-540-78862-1
7. `hochreiter1997lstm` — S. Hochreiter, J. Schmidhuber. Long short-term
   memory. *Neural Computation* 9(8):1735–1780, 1997.
   doi:10.1162/neco.1997.9.8.1735
   https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735
8. `jang1993anfis` — J.-S. R. Jang. ANFIS: adaptive-network-based fuzzy
   inference system. *IEEE Trans. SMC* 23(3):665–685, 1993. doi:10.1109/21.256541
   https://www.researchgate.net/publication/3113825_ANFIS_adaptive-network-based_fuzzy_inference_system
9. `lu2021deeponet` — L. Lu, P. Jin, G. Pang, Z. Zhang,
   G. E. Karniadakis. Learning nonlinear operators via DeepONet based on the
   universal approximation theorem of operators. *Nature Machine Intelligence*
   3:218–229, 2021. doi:10.1038/s42256-021-00302-5
   https://arxiv.org/pdf/1910.03193
10. `prince1981high` — P. J. Prince, J. R. Dormand. High order embedded
    Runge–Kutta formulae. *J. Computational and Applied Mathematics* 7(1):67–75, 1981.
    doi:10.1016/0771-050X(81)90010-3
    https://www.sciencedirect.com/science/article/pii/0771050X81900103
11. `raissi2019pinn` — M. Raissi, P. Perdikaris, G. E. Karniadakis.
    Physics-informed neural networks. *J. Computational Physics* 378:686–707, 2019.
    doi:10.1016/j.jcp.2018.10.045
    https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125
12. `ramachandruni2025chaos` — V. Ramachandruni, S. H. R. Nara, G. Lalu,
    S. Yang, M. Ramesh Kumar, A. Jain, P. Mehta, H. Koo, J. Damonte, M. Akl.
    Using machine learning and neural networks to analyze and predict chaos in
    multi-pendulum and chaotic systems. arXiv:2504.13453 [cs.LG], 2025.
    https://arxiv.org/pdf/2504.13453
13. `shinbrot1992chaos` — T. Shinbrot, C. Grebogi, J. Wisdom,
    J. A. Yorke. Chaos in a double pendulum. *American Journal of Physics*
    60(6):491–499, 1992. doi:10.1119/1.16860
    https://www.researchgate.net/publication/245641141_Chaos_in_a_Double_Pendulum
14. `stachowiak2006numerical` — T. Stachowiak, T. Okada. A numerical
    analysis of chaos in the double pendulum. *Chaos, Solitons & Fractals*
    29(2):417–422, 2006. doi:10.1016/j.chaos.2005.08.032
    https://www.sciencedirect.com/science/article/abs/pii/S0960077905006703
15. `takagi1985fuzzy` — T. Takagi, M. Sugeno. Fuzzy identification of
    systems and its applications to modeling and control. *IEEE Trans. SMC*
    SMC-15(1):116–132, 1985. doi:10.1109/TSMC.1985.6313399
    https://ieeexplore.ieee.org/document/6313399
16. `virtanen2020scipy` — P. Virtanen, R. Gommers, T. E. Oliphant, et al.
    SciPy 1.0: fundamental algorithms for scientific computing in Python.
    *Nature Methods* 17:261–272, 2020. doi:10.1038/s41592-019-0686-2
    https://pubmed.ncbi.nlm.nih.gov/32015543/
17. `wang1998universal` — L.-X. Wang. Universal approximation by
    hierarchical fuzzy systems. *Fuzzy Sets and Systems* 93(2):223–230, 1998.
    doi:10.1016/S0165-0114(96)00197-2
    https://www.sciencedirect.com/science/article/abs/pii/S0165011496001972
18. `wang1992generating` — L.-X. Wang, J. M. Mendel. Generating fuzzy
    rules by learning from examples. *IEEE Trans. SMC* 22(6):1414–1427, 1992.
    doi:10.1109/21.199466
    https://ieeexplore.ieee.org/document/199466
19. `yesilyurt2019npendulum` — B. Yesilyurt. Equations of motion
    formulation of a pendulum containing N-point masses. arXiv:1910.12610
    [physics.class-ph], 2019.
    https://arxiv.org/pdf/1910.12610


### Paper Action Items
> To be removed before publication, but captured here for completeness at this stage.

1. Implement complete reproducibility harness in conjunction with proposal defense.
2. Evaluate performance optimizations on auto-derivation.
3. Compare training time of LSTM/GRU with the TRIBBLE-FIS approach.
4. Evaluate $(\sin\theta, \cos\theta)$ stability without $t$ as an input variable.
5. External, non-interleaved test sets.