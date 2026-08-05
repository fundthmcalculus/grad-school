# What a Fuzzy System Can and Cannot Learn About a Chaotic Pendulum: A Reproduction, a Zero-Parameter Baseline, and the Limits of Time-Indexed Operators

**Scott Phillips**
University of Cincinnati, Cincinnati, Ohio 45221
`sphillips@nexigen.com`

> Affiliation is as it appears in *NAFIPS-2025-VAT-ACO-Phillips*. Co-authorship is
> deliberately not assumed here, though the 2025 NAFIPS submission was joint with
> K. Cohen.

---

## Abstract

We reproduce a recent multi-pendulum chaos-prediction study [12] that benchmarks
ten neural and machine-learning models, replacing all of them with a collection
of Takagi–Sugeno–Kang (TSK) fuzzy inference systems. The FIS attains the best
RMSE in six of the seven cells the original work reports, and the best $R^2$ in
seven of seven. We then show that this ranking is not evidence of learned
dynamics. The benchmark's held-out initial condition lies exactly midway between
two *trained* ones, and averaging those two trajectories — a predictor with no
parameters and no fitting — beats every published model by 6× on the double
pendulum and 18× on the triple. We extend the benchmark to a five-link chain and
show this dominance is chain-length dependent, degrading twentyfold by $n=5$ as
the bracketing pair decorrelates. Turning to extrapolation, we prove and then
measure that the study's "time-step" formulation, which admits $t$ as an input
feature, has no horizon whatsoever: predictions diverge by four orders of
magnitude within one timestep of the training-window edge, and —
counter-intuitively — improving the in-window fit makes the divergence *worse* by
three orders of magnitude. Reformulating the surrogate to learn the equations of
motion instead removes the window but introduces a different obstruction, which
we quantify: in the damped regime the dissipative term is 1.2 % of the
acceleration, so a TSK fit at $R^2 = 0.990$ leaves a residual 8× larger than the
entire physics that stabilises the system. We derive the accuracy required
($R^2 > 1 - 10^{-6}$), show empirically that rule refinement does not reach it,
and give two structural remedies with their costs. Finally we address the target
representation, proving that no bounded continuous scalar can represent an
accumulating angle and showing that the consequences are not the expected ones: a
hysteresis wrap reduces representation discontinuities monotonically, exactly as
designed, and yet does not reliably improve accuracy, whereas emitting
$(\sin\theta, \cos\theta)$ — the one representation that escapes the obstruction —
is the only one that improves error *past* the training window. All numbers,
figures and negative results are reproducible from the accompanying code.

**Keywords:** Takagi–Sugeno–Kang systems; chaotic dynamics; surrogate modelling;
benchmark validity; extrapolation; system identification.

---

## 1. Introduction

Fuzzy inference systems are universal approximators [17] and, in
Takagi–Sugeno–Kang form [15], admit closed-form consequent solutions that make
them unusually cheap to fit. That combination makes them an attractive substitute
for neural surrogates in low-dimensional physical modelling. It also makes them a
good instrument for a second purpose: because a TSK system is transparent about
*where* it is interpolating and where it is not, its failures localise the
difficulty in a way that a deep network's do not.

This paper uses a TSK collection for both purposes on the multi-pendulum
prediction benchmark of Ramachandruni et al. [12], which evaluates ten neural and
classical regressors on predicting double- and triple-pendulum trajectories. Our
contributions:

1. **A competitive reproduction** (§4). A per-output TSK collection with Gaussian
   antecedents and ridge-solved consequents attains the lowest RMSE in six of the
   benchmark's seven populated cells, at a training cost of tens of seconds.

2. **A benchmark-validity result** (§5). The held-out initial condition is the
   exact midpoint of two trained ones. A zero-parameter midpoint interpolant
   scores RMSE 0.0025 against the best published 0.0153. The benchmark's headline
   configuration measures one-dimensional interpolation of a smooth grid, not
   chaotic prediction. We extend to $n = 5$ and show where this degeneracy breaks.

3. **A no-go result for time-indexed operators** (§6). We show analytically why
   admitting $t$ as an input bounds validity to the training window, verify a
   four-order-of-magnitude divergence at the edge, and report an inverse relation
   between in-window accuracy and extrapolation error that makes hyperparameter
   tuning actively harmful.

4. **A quantitative obstruction for black-box identification of damped dynamics**
   (§7). We derive the fit accuracy required for a learned right-hand side to
   reproduce the sign of dissipation, show it is $R^2 > 1 - 10^{-6}$ for this
   system, and demonstrate that rule refinement does not attain it while a
   structured basis does — exactly, and 7000× faster to train.

5. **A topological limit on target representation** (§8). We prove no bounded
   continuous scalar can represent an accumulating angle, which rules out the
   `np.unwrap` family a priori, and show that the mechanism which *does* reduce
   representation discontinuities monotonically nonetheless fails to improve
   accuracy — while $(\sin\theta,\cos\theta)$, which escapes the obstruction by
   spending an output, is the only representation that improves extrapolated error.

6. **Negative results reported as such** (§7, §8, §9), including one intuitive
   stabiliser that makes matters 25× worse, one that stabilises by freezing the
   system, and one proposed fix that is provably a no-op.

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
reproduces (1) at $n = 2$ and generalises the $n$-point-mass formulation of
Yesilyurt [20]. Agreement between the two derivations is $1.07 \times 10^{-14}$
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
exercised at all — a point worth noting given the study's conclusion favours
recurrent architectures [7].

Data are generated by fixed-step RK4 at $h = 5$ ms over 10 s (2000 samples), for
31 initial conditions with $\theta_1(0) = 120^{\circ}$ and the final link's angle swept
over $[0^{\circ}, 3^{\circ}]$ in $0.1^{\circ}$ steps. The held-out condition is $2.05^{\circ}$. Targets are
min–max scaled **per trajectory**, independently per angle, before pooling; every
trajectory therefore spans exactly $[0, 1]$ in every output. A reported RMSE is
thus a fraction of that trajectory's own angular range, not an angle — a point we
return to in §9.

### 2.3 Related work

Learning dynamics rather than trajectories is a well-established alternative.
SINDy [1] recovers governing equations by sparse regression over a candidate
library; neural ODEs [2] learn a right-hand side and differentiate through the
integrator; Hamiltonian and Lagrangian networks [4, 5] impose conservation
structurally; PINNs [11] penalise residuals of a known operator. Operator
learning in the sense of (4) has its own literature [9]. Our §7 is
methodologically closest to SINDy with a fixed, physically derived library, and
our findings about dissipation magnitude bear directly on why the structural
approaches of [4, 5] are attractive for damped systems.

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
firing-weighted ridge least-squares solve yields the exact minimiser of the
weighted MSE — no iterative optimiser, unlike gradient-trained TSK variants [19]
or ANFIS [8]. Antecedent placement from data rather than a grid follows the
cluster-estimation tradition [3, 18].

Two implementation details matter. First, $\theta_1(0)$ is constant across the
sweep, so a Gaussian membership on it has $\sigma = 0$ and a degenerate firing
strength; **zero-variance inputs must be dropped**. A network absorbs a dead
input, a fuzzy partition cannot. Second, all scalers are **unclipped**: at
$t = 20$ s the input scaler returns $2.0$ rather than saturating at $1.0$. This is
deliberate and load-bearing — §6 depends on the failure being visible rather than
disguised as a plateau.

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

| Cell | Best published | FIS | Winner |
|---|---|---|---|
| 2-link, no friction, trained | LSTM 0.0270 | **0.0126** | FIS |
| 2-link, no friction, holdout | LSTM 0.26 | **0.184** | FIS |
| 2-link, friction, trained | LSTM 0.00955 | **0.00550** | FIS |
| 2-link, friction, holdout | LSTM 0.0153 | **0.0120** | FIS |
| 3-link, no friction, trained | GRU 0.0170 | 0.0191 | GRU |
| 3-link, no friction, holdout | — | 0.162 | — |
| 3-link, friction, trained | GRU 0.00911 | **0.00718** | FIS |
| 3-link, friction, holdout | GRU 0.00650 | **0.00477** | FIS |

The single loss is the frictionless triple-pendulum trained cell, where the
published GRU records a lower RMSE (0.0170 vs 0.0191) but a lower $R^2$ (0.985 vs
0.994). The two metrics disagree because the original work's Fig. 18B bar labels
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

| Predictor | 2-link | 3-link | 5-link |
|---|---|---|---|
| Midpoint (6) | **0.00250** | **0.000359** | **0.0515** |
| Nearest trained IC | 0.0111 | 0.00187 | 0.0520 |
| FIS (ours) | 0.0120 | 0.00477 | 0.0672 |
| Best published | 0.0153 | 0.00650 | — |

The margin is 6× on the double pendulum and 18× on the triple. Even *copying a
single neighbouring trajectory verbatim* beats the best published model in both
cells. Since the original study's conclusions generalise from precisely these
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

| Dataset | $s(20)$ | $s/\rho$ | $\lambda$ (s⁻¹) | Midpoint RMSE |
|---|---|---|---|---|
| 2-link, friction | 8.8° | 3.3 % | 0.56 | 0.00250 |
| 3-link, friction | 8.2° | 4.0 % | 0.23 | 0.000359 |
| 5-link, friction | 105° | 23.7 % | 1.25 | 0.0515 |
| 2-link, no friction | 628° | 67 % | 1.81 | 0.226 |
| 3-link, no friction | 649° | 112 % | 0.83 | 0.166 |
| 5-link, no friction | 709° | 62 % | 1.40 | 0.227 |

With damping the pair never separates by more than 4 % of the target's range at
$n = 2, 3$; the interpolation problem is genuinely easy. Without damping it grows
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

### 5.3 The degeneracy is chain-length dependent

Extending the protocol to $n = 5$ — which the original work does not attempt —
shows that the benchmark's two chain lengths are the two where the degeneracy is
most extreme. By $n = 5$ the bracketing pair separates to 23.7 % of the range and
the midpoint baseline degrades twentyfold, from 0.0025 to 0.0515. It remains
ahead of our FIS (0.0672), but by 1.3× rather than 4.8×. Across all three chain
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

for **every** rule, and the normalisation in (5) divides a vanishing numerator by
a vanishing denominator: the weights become the ratio of two underflowing
quantities and rule selection is numerical noise. Second, the consequents are
affine in $\tilde t$ and unbounded, so whichever rule wins contributes
$\bar y_r + a_r \tilde t$ evaluated at twice its fitted range. The composition
diverges rather than saturating.

Note the direction of failure is an accident of the consequent, not a property of
TSK systems. Where a rolled-out state leaves antecedent support and the surviving
rule contributes $\approx 0$, the same mechanism produces *flat-lining* instead.
Outside the antecedents' support a TSK model has no defined behaviour; which way
it fails is incidental.

### 6.2 Measured

Table 4 confirms the prediction, and adds something we did not anticipate.

**Table 4.** Friction double pendulum, trained on 10 s, scored to 20 s against a
DOP853 reference. Angle RMSE in degrees. Within the time-input family,
*in-window accuracy and extrapolation error are inversely related.*

| Model | Vars | Train (s) | 0–10 s | 10–20 s |
|---|---|---|---|---|
| **With the time variable** | | | | |
| 40 rules | 2 | 7.1 | 22.0° | 443° |
| 120 rules | 2 | 20.6 | 4.35° | 20 070° |
| 300 rules | 2 | 71.7 | **3.11°** | 338 555° |
| 8 harmonics | 18 | 175.8 | 20.1° | 146° |
| **Without the time variable** | | | | |
| state-space, 40 | 4 | 14.4 | 375° | 1 455° |
| state-space, 120 | 4 | 48.5 | 52.2° | 1 173° |
| &nbsp;&nbsp;+ dissipation guard | 4 | 48.1 | 59.0° | 91.7° |
| physics basis | 5 | **0.01** | **0.0004°** | **0.0004°** |

Refining $R$ from 40 to 300 improves the in-window RMSE from $22.0^{\circ}$ to $3.11^{\circ}$
while degrading the extrapolation from $443^{\circ}$ to $338\,555^{\circ}$ — three orders of
magnitude worse. A sharper in-window fit requires steeper affine consequents, and
those are precisely what is extrapolated off the end of the window. **Tuning the
model for its reported metric makes its horizon worse.** We are not aware of this
trade-off being stated for time-indexed surrogates, and it inverts the usual
reading of a capacity sweep.

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

## 7. Learning the Dynamics Instead

### 7.1 Reformulation

Removing $t$ from the inputs removes the window. Let the FIS learn

$$
\mathcal{F}: (\theta_1, \omega_1, \theta_2, \omega_2)
\longmapsto (\dot\omega_1, \dot\omega_2),
\tag{8}
$$

and advance the state with RK4. Time never enters, so validity is governed instead
by whether the rolled-out state remains where training data lay. In the damped
regime the flow is contracting: trajectories decay toward the hanging equilibrium
and the state moves toward *smaller* angles and rates, i.e. into the interior of
the training distribution. Extrapolating in time becomes interpolating in
state — in principle.

### 7.2 An obstruction, derived

In practice the black-box form of (8) fails, and the reason generalises beyond
this system. Write the acceleration as a conservative part plus dissipation,
$\dot\omega = A_{\text{cons}} + A_{\text{damp}}$. From (2)–(3),
$A_{\text{damp}} = -c\,\omega_i / (\ell_i D)$. Measured over the training states,

$$
\frac{\|A_{\text{damp}}\|_{\text{rms}}}{\|\dot\omega\|_{\text{rms}}}
= 1.2 \times 10^{-2},
\tag{9}
$$

so dissipation is 1.2 % of the signal being fitted. A regressor achieving
coefficient of determination $R^2$ on $\dot\omega$ leaves residual r.m.s.
$\sqrt{1 - R^2}\,\sigma_{\dot\omega}$. Requiring that residual to be a factor $k$
**below** the dissipative term gives

$$
R^2 > 1 - \left(
\frac{\|A_{\text{damp}}\|_{\text{rms}}}{k\,\sigma_{\dot\omega}}
\right)^{\!2}.
\tag{10}
$$

With (9) and $k = 10$, (10) demands $R^2 > 1 - 10^{-6}$. At $R^2 = 0.990$ — which
by ordinary standards is a good fit — the residual is 8 to 9× **larger** than the
entire damping term. The surrogate's effective dissipation is fit noise with
arbitrary sign, so the rollout gains energy, amplitude grows past the region
training covered, and the state-space argument of §7.1 collapses.

This is not a fuzzy-specific limitation; (10) applies to any regressor fitting a
total acceleration in which dissipation is a small correction. It is a
quantitative reason to prefer the structurally constrained formulations of [4, 5]
for damped systems, and it explains why the effect resists brute force: Table 5
shows each 2.5× in rules buys roughly one order of magnitude in $1 - R^2$, so
$R^2 = 1 - 10^{-6}$ would require $R \approx 1000$.

**Table 5.** Acceleration-fit accuracy against rule count, and the resulting
20-second rollout. Accuracy improves steadily; stability does not. Amplitude
ratio is predicted over true amplitude in the final 2 s.

| $R$ | $R^2(\dot\omega_1)$ | in-window RMSE | amplitude ratio |
|---|---|---|---|
| 40 | 0.9905 | 375° | ×41 |
| 120 | 0.99931 | 52.2° | ×41 |
| 300 | 0.999987 | 3.87° | ×85 |

Note the last column: refining $R$ improves the in-window rollout by two orders
of magnitude while making the extrapolated amplitude *worse*. Accuracy and
stability are separate properties, and the second is not bought by the first.

### 7.3 Two structural remedies, with costs

**Enforcing dissipation.** Using only the fact that a damped system cannot gain
mechanical energy — not the equations — we may reject any step that increases $E$.
Since kinetic energy is homogeneous of degree two in the rates, scaling both by
$\lambda$ scales $T$ by $\lambda^2$, and the correction that restores
$E_{k+1} = E_k$ is closed-form:

$$
\lambda = \sqrt{\frac{E_k - V(\mathbf{q}_{k+1})}
{T(\mathbf{q}_{k+1}, \dot{\mathbf{q}}_{k+1})}},
\qquad
\dot{\mathbf{q}}_{k+1} \leftarrow \lambda\, \dot{\mathbf{q}}_{k+1}.
\tag{11}
$$

This bounds the rollout: amplitude ratio falls from ×41 to ×1.13 and
extrapolation RMSE from $1173^{\circ}$ to $91.7^{\circ}$. But it constrains the state's
*magnitude*, not its direction on the energy surface, and the distinction is
decisive: the guard fires on 75 % of steps and retains only 2.6 % of the true
oscillation amplitude. The rollout is stable because it has been damped to a
standstill. **Bounded is not predictive**, and a single amplitude statistic cannot
distinguish the two — we report an activity ratio alongside it for exactly this
reason.

**Supplying the structure.** Alternatively, note that $D$ in (1) depends only on
the angles and on masses and lengths, which are known constants of the apparatus.
Dividing (2)–(3) through by $\ell_i D$ makes each acceleration **exactly linear**
in the features

$$
\mathbf{\psi}_1 = \frac{1}{\ell_1 D}
\begin{bmatrix}
\sin\theta_1 \\
\sin(\theta_1 - 2\theta_2) \\
\omega_2^2 \sin\Delta \\
\omega_1^2 \sin\Delta \cos\Delta \\
\omega_1
\end{bmatrix},
\tag{12}
$$

and analogously $\mathbf{\psi}_2 \in \mathbb{R}^4$. Crucially, the damping
feature $\omega_i / (\ell_i D)$ is now *one column among equals* rather than a
1.2 % perturbation buried under fit error, so (10) no longer binds. A
no-intercept least-squares consequent recovers the true coefficients to
$7.8 \times 10^{-14}$, and the resulting model is a degenerate single-rule TSK
system: partitioning cannot improve a fit to a function that is already a plane.

The rollout is then stable and accurate indefinitely. Over 120 s — twelve times
the training window — the pure model error is $4.4 \times 10^{-9}$ degrees, with
amplitude and activity ratios both 1.000. Training is a single least-squares
solve: 0.01 s, against 71.7 s for the best time-input model, i.e. 7000× less
training for eight orders of magnitude less error. This is SINDy [1] with a
physically derived rather than generic library, and we state the assumption
plainly: it presumes the functional *form* of the dynamics, which is what a
surrogate is often meant to avoid.

![Four surrogate families rolled out to 20 s against the truth, with the
training-window edge marked. The physics-basis trace lies exactly under the
reference for the full window.](figures/fig_extrapolation_models.png)

*Figure 3. Rollout comparison across formulations.*

### 7.4 One appealing idea that fails

Since divergence follows from evaluating outside the training box, clipping the
state to that box before evaluation appears to convert blow-up into saturation.
It does the opposite: amplitude ratio ×1468 clipped versus ×41 unclipped, 25×
worse. Clipping freezes the acceleration at the boundary value, so once the state
exits the box the restoring term stops growing with displacement while the rate
keeps integrating — the clip removes precisely the feedback that would have pulled
the state back. **Saturating the input is not the same as saturating the output.**
We report this because the intuition is appealing and wrong.

### 7.5 The families compared

Figure 5 places all eight models on cost against extrapolation accuracy, with
marker area proportional to input variable count. The time-input family is cheap
to query — one batched call answers any $t$ in 0.01 s — whereas the state-space
family must integrate 4000 steps and pays 26–47 s per trajectory. That asymmetry
is real and we report it: the time-input formulation is genuinely better at what
it is for, namely cheap random-access queries inside a known window. It simply
has no validity outside one.

![Error inside versus past the training window, log scale. Within the time-input
family the two bars move in opposite directions as capacity
grows.](figures/fig_family_inwindow_vs_extrap.png)

*Figure 4. In-window versus extrapolated error, by model.*

![Training cost against extrapolation error; marker area is input variable count.
The most accurate model is also the cheapest to train by four orders of
magnitude.](figures/fig_family_tradeoff.png)

*Figure 5. Cost against extrapolation accuracy.*

---

## 8. Angle Representation and the Topology of the Target

Sections 6 and 7 concern what the model is indexed by. This one concerns what it
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
is — contradiction. Therefore $k$ must jump, and each jump is a $360^{\circ}$
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
*and* continuous, escaping §8.1 by spending a second output rather than by
contradicting it: a circle does not embed in an interval without a cut, but it
embeds in the plane without one. The winding number is discarded, which costs
nothing because (1)–(3) depend on the angles only through $\sin$ and $\cos$.

Because wrapping changes what an error means, we score in **circular** angular
error, $((\hat\theta - \theta + 180^{\circ}) \bmod 360^{\circ}) - 180^{\circ}$,
which is the physically meaningful quantity and the only one comparable across
wrapped and unwrapped representations. It is more forgiving than the benchmark's
metric — a prediction of $430^{\circ}$ against a truth of $790^{\circ}$ scores
zero, being the same configuration — so it is used only within this section.

### 8.3 The overlap band works, and does not help

Table 6 counts representation-induced discontinuities: sample-to-sample jumps
exceeding $180^{\circ}$, which the true dynamics never produce at this sampling
rate and which are therefore artefacts.

**Table 6.** Discontinuities in the frictionless double-pendulum training set.
Pointwise wrapping gets *worse* as the overlap band widens; hysteresis gets
better.

| $L$ | pointwise | hysteresis |
|---|---|---|
| $180^{\circ}$ | 66 | 66 |
| $240^{\circ}$ | 76 | **53** |
| $300^{\circ}$ | 94 | **51** |
| $360^{\circ}$ | 111 | **48** |
| $420^{\circ}$ | 93 | **47** |

The pointwise trend is explained by its statelessness: a trajectory oscillating
inside the overlap band flips branch on every crossing, and a wider band is a
wider flip zone. Hysteresis removes the flip-flop and the count falls
monotonically. The two agree at $L = 180^{\circ}$, as they must, there being no
band to exploit. **The mechanism therefore works exactly as intended.**

It does not follow that accuracy improves, and Table 7 shows it does not.

**Table 7.** Circular error, in-window / past-window, in degrees. 120 rules,
quantile partitioning throughout so the partition is not a second variable.

| dataset | no wrap | $\pm180^{\circ}$ | pointwise $\pm360^{\circ}$ | hysteresis $\pm360^{\circ}$ | sin/cos |
|---|---|---|---|---|---|
| 2-link, no fric. | 81.9 / 107.4 | 73.7 / 106.9 | 68.5 / 108.0 | **62.5** / 108.9 | 74.1 / **92.1** |
| 3-link, no fric. | 59.0 / 108.7 | **49.8** / 106.9 | 54.9 / 108.3 | 59.5 / 109.1 | 48.0 / **80.1** |
| 5-link, no fric. | 51.9 / 107.7 | 50.5 / 111.4 | 59.0 / 103.7 | 51.2 / 104.2 | **43.6** / **71.9** |
| 5-link, friction | **12.3** / 92.2 | 13.7 / 107.2 | 38.3 / 92.8 | 15.8 / 95.8 | 12.1 / 109.1 |

Hysteresis at $\pm360^{\circ}$ cuts the double pendulum's in-window error from
$68.5^{\circ}$ to $62.5^{\circ}$, but *raises* the triple pendulum's from
$54.9^{\circ}$ to $59.5^{\circ}$ while cutting its discontinuity count from 93 to
56. **Representation jitter is not the binding constraint on accuracy here**, and a
mechanism can be validated on its own terms while failing to move the metric it
was introduced to move. Where plain wrapping does help, smaller is better:
$\pm180^{\circ}$ is the best wrap on two of four datasets, consistent with the
range penalty a wider window carries.

### 8.4 Sine–cosine is the representation to use

It is best in-window on three of four datasets and best past the window on three
of four, and it is **the only representation that improves the extrapolated error
at all** — $92.1^{\circ}$, $80.1^{\circ}$, $71.9^{\circ}$ on the three
frictionless sets against $104$–$111^{\circ}$ for every wrap variant. That it
helps past the window while wrapping does not is consistent with §6: the horizon
problem is the time input, and only a representation that changes the *output
geometry* touches it.

One exception: on the damped five-link chain — the sole dataset where an angle
spins but the dynamics dissipate — sin/cos has the best in-window error
($12.1^{\circ}$) and the worst extrapolation ($109.1^{\circ}$).

### 8.5 The unit-circle constraint is not a lever

The predicted pair is not confined to the unit circle: its mean radius is $0.843$,
$0.865$, $0.900$ and $0.979$ across the four datasets, so up to 16 % of the
output is not a valid angle. Renormalising before $\operatorname{atan2}$ is the
obvious remedy and **cannot work**: $\operatorname{atan2}$ is invariant under
positive scaling of both arguments,
$\operatorname{atan2}(ks, kc) = \operatorname{atan2}(s, c)$, verified to
$4 \times 10^{-16}$ rad. The angular error lives in the ratio $s/c$, which no
rescaling of the magnitude alters.

What the radius does carry is shrinkage, and it carries it in a way that
corroborates the rest of the paper.

**Table 8.** Sine–cosine targets, frictionless double pendulum, against capacity.

| rules | in-window | past-window | mean radius | $\mathrm{corr}(1-r, \lvert e \rvert)$ |
|---|---|---|---|---|
| 40 | **67.8°** | 102.4° | 0.723 | **0.365** |
| 120 | 74.1° | **92.1°** | 0.843 | 0.252 |
| 300 | 74.0° | 110.1° | 0.835 | 0.218 |

The radius rises from $0.72$ to $0.84$ between 40 and 120 rules and then
*saturates* — 300 rules gives $0.835$, not something approaching $1$. The residual
shortfall is therefore not underfitting that capacity would remove; it is the
model hedging toward the conditional mean, the same behaviour §5.2 identifies from
bracket decorrelation and §7.2 from the dissipation noise floor. A shrunken radius
is how the sine–cosine representation displays it.

The per-sample correlation between radius shortfall and angular error is positive
at every capacity ($+0.22$ to $+0.37$), so low-radius samples really are the wrong
ones — but it explains only 5–13 % of error variance. That is a usable soft flag,
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

**The normalisation assumes fixed-length trajectories.** Per-trajectory scaling
only makes training and test commensurable when both are normalised over the same
duration. Training targets span exactly $[0, 1]$ by construction; normalising a
20 s holdout over its full length leaves its first 10 s spanning only
$[0, 0.678]$, so a model trained to emit over $[0, 1]$ overshoots by ~1.5× for
reasons unrelated to its dynamics. This is a latent flaw in the protocol that a
10 s test cannot expose. Friction datasets are unaffected (their 20 s range equals
their 10 s range bitwise); frictionless in-window numbers are not comparable
across the choice.

**The frictionless reference is not converged.** Energy drift attests that an
integrator is self-consistent, not that a trajectory is converged — on a chaotic
system these differ sharply. Refining $h$ by 2×, 4×, 8× leaves the friction
references unchanged to $0.00^{\circ}$ over 20 s, but moves the frictionless ones by
hundreds of degrees, with successive refinements parting company from
$t \approx 8.9$–11.5 s. An independent check against DOP853 [10, 6, 16] at
rtol $= 10^{-12}$ agrees on those times to two decimals. Consequently the
frictionless 10–20 s figures should be read as "the surrogate does not track a
reference that is itself not reproducible there" — weaker than the friction rows
support. All headline claims in §6–§7 are friction-only for this reason.

**Error decomposition.** Scoring an RK4 rollout of learned dynamics against an
RK4 rollout of true dynamics lets integration error cancel between them. That
comparison reported $3.3 \times 10^{-11}$ degrees for the physics-basis model when
its actual error is $4.4 \times 10^{-9}$ degrees and the paper's fixed-step grid
contributes $1.8 \times 10^{-3}$ degrees. Crossing {RK4, DOP853} against
{learned, true} separates the three; we recommend the practice generally.

**Scope.** The state-space results are for $n = 2$ with damping. The reproduction
covers $n \in \{2, 3, 5\}$ in both regimes.

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

On extrapolation the finding is sharper. A surrogate that admits $t$ as an input
has no validity outside its training window, for reasons we derive and verify,
and within that family in-window accuracy trades *against* extrapolation:
refining capacity improves the reported metric by 7× while worsening the horizon
by three orders of magnitude. Learning the equations of motion instead removes the
window but exposes a different obstruction, which we quantify in (10): when
dissipation is a 1 % correction to the acceleration, identifying it black-box
requires $R^2 > 1 - 10^{-6}$, and rule refinement does not get there. A physically
structured basis does, exactly, and trains four orders of magnitude faster than
the black-box alternative.

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

For fuzzy modelling of dissipative mechanical systems the practical
recommendations are: never index the model by time if a horizon is wanted; emit
angles on the circle rather than as a bounded scalar; give the consequents the
structure that makes dissipation a first-class term rather than a residual; report
a zero-parameter baseline; and separate model error from integrator error before
claiming either.

**Reproducibility.** All datasets, 253 scored configurations, figures and negative
results are generated by the accompanying code, with provenance assertions on
every run: agreement between two independent derivations of the equations of
motion ($1.07 \times 10^{-14}$), RK4 convergence order (≈16× per halving), and
refit reproduction of every swept score to $10^{-9}$.

---

## References

Keys in brackets match `references.bib`. Verification status: **[V]** checked
against a publisher index or arXiv listing this session; **[S]** seminal work
taken from working knowledge, to be spot-checked at proof stage.

1. **[S]** `brunton2016sindy` — S. L. Brunton, J. L. Proctor, J. N. Kutz.
   Discovering governing equations from data by sparse identification of
   nonlinear dynamical systems. *PNAS* 113(15):3932–3937, 2016.
   doi:10.1073/pnas.1517384113
2. **[S]** `chen2018neuralode` — R. T. Q. Chen, Y. Rubanova, J. Bettencourt,
   D. Duvenaud. Neural ordinary differential equations. *NeurIPS* 31, 2018.
3. **[S]** `chiu1994fuzzy` — S. L. Chiu. Fuzzy model identification based on
   cluster estimation. *J. Intelligent and Fuzzy Systems* 2(3):267–278, 1994.
   doi:10.3233/IFS-1994-2306
4. **[S]** `greydanus2019hamiltonian` — S. Greydanus, M. Dzamba, J. Yosinski.
   Hamiltonian neural networks. *NeurIPS* 32, 2019.
5. **[S]** `cranmer2020lagrangian` — M. Cranmer, S. Greydanus, S. Hoyer,
   P. Battaglia, D. Spergel, S. Ho. Lagrangian neural networks. *ICLR 2020 Deep
   Differential Equations Workshop*, 2020.
6. **[S]** `hairer1993solving` — E. Hairer, S. P. Nørsett, G. Wanner. *Solving
   Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Springer, 1993.
   doi:10.1007/978-3-540-78862-1
7. **[S]** `hochreiter1997lstm` — S. Hochreiter, J. Schmidhuber. Long short-term
   memory. *Neural Computation* 9(8):1735–1780, 1997.
   doi:10.1162/neco.1997.9.8.1735
8. **[V]** `jang1993anfis` — J.-S. R. Jang. ANFIS: adaptive-network-based fuzzy
   inference system. *IEEE Trans. SMC* 23(3):665–685, 1993. doi:10.1109/21.256541
9. **[S]** `lu2021deeponet` — L. Lu, P. Jin, G. Pang, Z. Zhang,
   G. E. Karniadakis. Learning nonlinear operators via DeepONet based on the
   universal approximation theorem of operators. *Nature Machine Intelligence*
   3:218–229, 2021. doi:10.1038/s42256-021-00302-5
10. **[S]** `prince1981high` — P. J. Prince, J. R. Dormand. High order embedded
    Runge–Kutta formulae. *J. Computational and Applied Mathematics* 7(1):67–75,
    1981. doi:10.1016/0771-050X(81)90010-3
11. **[S]** `raissi2019pinn` — M. Raissi, P. Perdikaris, G. E. Karniadakis.
    Physics-informed neural networks. *J. Computational Physics* 378:686–707,
    2019. doi:10.1016/j.jcp.2018.10.045
12. **[V]** `ramachandruni2025chaos` — V. Ramachandruni, S. H. R. Nara, G. Lalu,
    S. Yang, M. Ramesh Kumar, A. Jain, P. Mehta, H. Koo, J. Damonte, M. Akl.
    Using machine learning and neural networks to analyze and predict chaos in
    multi-pendulum and chaotic systems. arXiv:2504.13453 [cs.LG], 2025.
13. **[V]** `shinbrot1992chaos` — T. Shinbrot, C. Grebogi, J. Wisdom,
    J. A. Yorke. Chaos in a double pendulum. *American Journal of Physics*
    60(6):491–499, 1992. doi:10.1119/1.16860
14. **[S]** `stachowiak2006numerical` — T. Stachowiak, T. Okada. A numerical
    analysis of chaos in the double pendulum. *Chaos, Solitons & Fractals*
    29(2):417–422, 2006. doi:10.1016/j.chaos.2005.08.032
15. **[V]** `takagi1985fuzzy` — T. Takagi, M. Sugeno. Fuzzy identification of
    systems and its applications to modeling and control. *IEEE Trans. SMC*
    SMC-15(1):116–132, 1985. doi:10.1109/TSMC.1985.6313399
16. **[S]** `virtanen2020scipy` — P. Virtanen, R. Gommers, T. E. Oliphant, et al.
    SciPy 1.0: fundamental algorithms for scientific computing in Python.
    *Nature Methods* 17:261–272, 2020. doi:10.1038/s41592-019-0686-2
17. **[S]** `wang1998universal` — L.-X. Wang. Universal approximation by
    hierarchical fuzzy systems. *Fuzzy Sets and Systems* 93(2):223–230, 1998.
    doi:10.1016/S0165-0114(96)00197-2
18. **[V]** `wang1992generating` — L.-X. Wang, J. M. Mendel. Generating fuzzy
    rules by learning from examples. *IEEE Trans. SMC* 22(6):1414–1427, 1992.
    doi:10.1109/21.199466
19. **[S]** `wu2020optimize` — D. Wu, Y. Yuan, J. Huang, Y. Tan. Optimize
    Takagi–Sugeno–Kang fuzzy systems for regression problems (MBGD-RDA).
    *IEEE Trans. Fuzzy Systems* 28(5):1003–1015, 2020.
    doi:10.1109/TFUZZ.2019.2958559
20. **[V]** `yesilyurt2019npendulum` — B. Yesilyurt. Equations of motion
    formulation of a pendulum containing N-point masses. arXiv:1910.12610
    [physics.class-ph], 2019.

Two further entries are present in `references.bib` but not cited in this draft
and should be dropped or cited before submission: `hairer2006geometric`
(structure-preserving integrators) and `benettin1980lyapunov` (Lyapunov exponent
computation). The $\lambda$ values in Table 3 are fitted from bracket separation
rather than computed by the Benettin method, so the latter is background rather
than method.
