# Certified Fuzzy Control by Least Action

**An $H^1$ variational formulation for TSK controllers, with sum-of-squares stability certificates and a calibrated performance/certificate trade**

Scott Phillips, Kelly Cohen
Department of Aerospace Engineering and Engineering Mechanics, University of Cincinnati

---

## Abstract

We develop a variational formulation of Takagi–Sugeno–Kang (TSK) [@takagi1985fuzzy; @sugenokang1988structure] fuzzy control in which controller fitting is posed as minimization of an $H^1$ (Sobolev) action functional, and we carry the formulation through to closed-loop control with machine-checked stability certificates.

Four results are analytic. **First**, with antecedents fixed the consequent block is the *global* minimizer of the action in closed form, so the only non-convexity in the problem is the antecedent placement; variable projection then yields an exact reduced gradient via the envelope theorem. **Second**, the natural design question — when does sequential (greedy) rule identification coincide with the joint fit — has the exact answer "iff the $H^1$ Gram matrix is block-diagonal," and for two equal-width Gaussians the slope weight achieving $H^1$ orthogonality has the closed form $\lambda^{*}=3b^{4}/(8c^{2})$, equivalently $\ell^{*}=\sqrt{6}\,w$ with $w$ the crossover width. A factor-of-four obstruction shows no single $\lambda$ can orthogonalize three or more uniformly spaced rules. **Third**, the closed-loop suboptimality gap is *exactly* an integral of a Bregman divergence for any convex control cost, not merely the quadratic case. **Fourth**, switching to $\pi$-shaped (rational) membership functions makes the closed loop a rational vector field, and a sum-of-squares S-procedure certificate for the region of attraction is then verified in exact rational arithmetic, with no floating-point step in the chain.

Empirically, on a two-mass–spring benchmark scored jointly on settling time, peak force and energy, the least-action fit converges in two rules (10 parameters; score 0.867 against an LQR baseline of 1.000) and then degrades through distribution shift rather than through any deficiency of the fit. Direct policy optimization warm-started from the least-action fit reaches 0.655, improving all three objectives simultaneously, while a cold start fails outright — the variational fit is the initializer that makes the optimizer reachable.

Finally we make the stability certificate an *objective* rather than an afterthought. A sampled necessary condition for the sum-of-squares program, evaluated by ray search, estimates the certified region $2485\times$ faster than the semidefinite program with a residual bias of a single constant $\kappa=0.962$; specifying a certified radius then yields a controller the solver certifies at or above it. The best controller so obtained carries a certified ball of 0.679 at score 0.847, dominating the imitation fit on both axes.

---

## 1. Introduction

This paper develops an idea originating in a handwritten derivation by the second author: that fitting a fuzzy inference system should be posed as a principle of least action rather than as least squares, and that the resulting stationarity conditions should be strong enough to *certify* optimality rather than merely to terminate an optimizer.

The motivation is that a TSK system is a smooth parametric model, and smooth parametric models admit variational treatment. What is unusual here is insisting that every step of the treatment produce **a statement that can be checked**, rather than a number produced by a solver.

The organizing question is a control question:

> If the least-action machinery is turned on a control problem, what is provable about the resulting controller — not measured, *proved*? We ask for suboptimality certificates and for stability certificates, and then we ask what happens when the two are placed in competition with performance.

### 1.1 Contributions

- An $H^1$ action functional for TSK controller fitting whose consequent block has a closed-form global minimizer (Theorem 1), an exact analytic reduced gradient (Theorem 2), and second-order conditions stated correctly on the critical cone (§3.3).
- An exact characterization of when greedy rule identification equals the joint fit (Proposition 3), a closed form for the $H^1$-orthogonalizing slope weight (Theorem 5), and a structural obstruction showing it cannot be achieved for $N\ge3$ uniformly spaced rules (Corollary 6).
- An exact suboptimality certificate for arbitrary convex control cost, expressed as an integral of a Bregman divergence (Theorem 8); the familiar quadratic identity is a special case.
- A sum-of-squares region-of-attraction certificate for a $\pi$-membership TSK closed loop, verified in exact rational arithmetic (§7), together with the practical finding that the binding obstruction was solver accuracy rather than the relaxation.
- Certificate-aware policy optimization, in which the certified radius enters the objective (§8), and a calibration procedure that turns the certified radius from a dial into a **design specification** (§9).

We are explicit throughout about what is proved, what is certified numerically, and what is merely measured; §10 tabulates this.

### 1.2 Scope

This paper is about control. The same action functional applies to fuzzy classification — a ranking without recurrence is a regression model with range $[0,1]$ — and that development, including the conditions under which Mamdani aggregation reduces to summation and mean-of-maxima defuzzification becomes differentiable, is deferred to a companion paper. Nothing in the control results below depends on it.

---

## 2. Model and action functional

### 2.1 The model

On a scalar input universe $X=[x_{lo},x_{hi}]$, an order-$p$ TSK system with $N$ rules is

$$y_c(x)=\sum_{i=1}^{N}\varphi_i(x)f_i(x),\qquad
\varphi_i=\frac{\mu_i}{\sum_{j}\mu_j},\qquad
f_i(x)=\sum_{k=0}^{p}B_{ik}x^{k},$$

where $\mu_i$ is the membership function of rule $i$ with centre $a_i$ and width $b_i$. The normalized firing strengths form a partition of unity, $\sum_i\varphi_i\equiv1$. Writing $\psi_{ik}=\varphi_i x^{k}$ for the **rule regressors** and $\theta$ for the flattened coefficients, the model is linear in $\theta$:

$$y_c(x)=\theta^{\top}\psi(x).\tag{1}$$

This is the structural fact the entire paper exploits: the parameters split into a block in which the model is linear ($\theta$) and a block in which it is not ($a,b$).

> **Implementation note.** The monomial $x^k$ is replaced by a rescaled monomial $t=(x-\text{mid})/\text{half}$. This spans the identical function space, so the model is unchanged, but it prevents the Gram condition number from reaching $10^{7}$ on a universe such as $[-15,15]$.

### 2.2 The action

Let $e=y_d-y_c$ be the error field against a target $y_d$. We take as action the squared $H^1$ norm of the error,

$$S[y_c]=\int_X\!\Big(e^2+\lambda\,(e')^2\Big)\,dx=\|e\|_{H^1}^{2},\qquad \lambda=\ell^{2}\ge0,\tag{2}$$

where $\ell$ has units of $x$ and plays the role of a correlation length. The $L^2$ term penalizes pointwise error; the $\lambda$ term penalizes error in *slope*, which is what makes this a genuinely variational formulation rather than a reweighted regression. In a control context the slope term is not decorative: it penalizes error in the local feedback *gain*, which is the quantity that governs closed-loop stability.

**Euler–Lagrange.** Treating $y_c$ as a free field, the first variation of (2) vanishes when

$$\lambda e''-e=0,\tag{3}$$

the screened-Poisson (modified Helmholtz) equation, with natural boundary conditions $e'=0$ at the endpoints. Its only solution decaying in both directions is $e\equiv0$: in the unconstrained function space, least action recovers the target exactly. The content of the method is therefore entirely in what happens when $y_c$ is restricted to the TSK manifold.

**Stationarity in the model class.** Restricting to (1), $\delta S=0$ in the $\theta$ directions is

$$\langle e,\psi_m\rangle_{H^1}=0\quad\text{for every regressor }\psi_m,\tag{4}$$

i.e. **Galerkin orthogonality**: the error is $H^1$-orthogonal to the model subspace. Equation (4) is the weak form of (3) and is the correct stationarity condition for the finite-dimensional problem. It is what we check numerically, and it is what distinguishes this formulation from generic nonlinear least squares.

> **Convexity.** For $\lambda\ge0$ the map $y_c\mapsto S$ is a strictly convex quadratic on function space, so the variational problem has no conjugate points and the Jacobi condition is vacuous. All non-convexity in the parametric problem enters through the map $(a,b)\mapsto\psi$, not through the action.

---

## 3. The consequent block is globally solvable

### 3.1 Closed form

Define the $H^1$ Gram matrix and right-hand side $G_{mn}=\langle\psi_m,\psi_n\rangle_{H^1}$, $r_m=\langle y_d,\psi_m\rangle_{H^1}$.

> **Theorem 1 (Variable projection [@golubpereyra1973differentiation; @golubpereyra2003separable]).** For fixed antecedents $(a,b)$, the action (2) is a convex quadratic in $\theta$ with Hessian $2G\succeq0$, and every minimizer satisfies the normal equations $G\theta=r$. If the regressors are linearly independent in $H^1$ then $G\succ0$ and $\theta^{*}=G^{-1}r$ is the unique global minimizer. The reduced action is
> $$\widehat S(a,b)=\|y_d\|_{H^1}^{2}-r^{\top}G^{-1}r.\tag{5}$$

*Proof.* Substituting (1) into (2), $S=\|y_d\|^2_{H^1}-2\theta^\top r+\theta^\top G\theta$. The Hessian $2G$ is a Gram matrix of an inner product, hence positive semidefinite, so $S$ is convex and stationarity is sufficient. Stationarity is $G\theta=r$, which is (4) written out. Substituting $\theta^{*}$ gives (5). $\blacksquare$

Theorem 1 is the strongest guarantee in the paper and it is free: no iteration, no initialization, no local minima. It also localizes the difficulty precisely — the problem is non-convex **only** in $(a,b)$.

> **Regularization.** Numerically we solve $(G+\varepsilon\,\mathrm{tr}(G)/m\,I)\theta=r$ with $\varepsilon=10^{-9}$. The ridge is trace-relative so it scales with the problem, and it is a ridge rather than a truncating pseudo-inverse deliberately: a rank-flipping SVD makes the map $(a,b)\mapsto\theta$ discontinuous, which destroys the differentiability §3.2 depends on.

### 3.2 Exact reduced gradient

Finite differences on (5) are unreliable, because $\widehat S$ is computed only to accuracy $\kappa(G)\,\varepsilon_{\text{mach}}$. In our experiments SciPy's default step produced a gradient dominated by round-off, and L-BFGS-B terminated after *zero* iterations reporting success. The envelope theorem [@milgromsegal2002envelope] removes the issue.

> **Theorem 2 (Analytic reduced gradient).** Let $\theta^{*}(a,b)$ be as in Theorem 1. Then for any antecedent parameter $p\in\{a_i,b_i\}$,
> $$\frac{\partial\widehat S}{\partial p}=-2\Big\langle e,\ \frac{\partial y_c}{\partial p}\Big\rangle_{H^1},\qquad \frac{\partial y_c}{\partial p}=\theta^{*\top}\frac{\partial\psi}{\partial p},\tag{6}$$
> with no contribution from $\partial\theta^{*}/\partial p$.

*Proof.* $\widehat S(a,b)=S(\theta^{*}(a,b),a,b)$, so $\partial_p\widehat S=\partial_\theta S\cdot\partial_p\theta^{*}+\partial_p S$. At $\theta^{*}$ the first factor vanishes by Theorem 1. The remaining partial is (6). $\blacksquare$

The derivative $\partial\psi/\partial p$ is available in closed form from the membership-function derivatives $(\mu,\partial_x\mu,\partial_a\mu,\partial_b\mu,\partial^2_{xa}\mu,\partial^2_{xb}\mu)$; the $\partial_x$ terms are required because the $H^1$ inner product differentiates the regressors. Replacing finite differences by (6) was the difference between an optimizer that stalled immediately and one that converged.

> **Implementation note.** The solver regularizes with a trace-scaled ridge $\varepsilon=10^{-9}\,\mathrm{tr}(G)/m$ (§3.1), and $\varepsilon$ itself depends on $(a,b)$ through $G$. The code accordingly differentiates the ridge term too, adding an $O(\varepsilon\|\theta\|^2)$ correction to (6) that is not shown above. It is negligible at the ridge scale used — Theorem 2 as stated is the gradient of the idealized (unregularized) reduced action — but it is the gradient of the objective actually being minimized, and dropping it silently would make the implemented gradient inexact for its own objective.

### 3.3 Second-order conditions, stated correctly

Three errors are easy to make here, and all three occurred during this work.

1. **Test the reduced Hessian, not the full one.** The full-parameter Hessian is never definite: consequent rescaling and rule relabeling guarantee flat and symmetric directions. Only $\widehat S$ has a meaningful Hessian.
2. **Fitted solutions sit on identifiability constraints.** We impose a minimum rule gap ($0.6\times$ pitch) and width bounds ($[0.25,1.5]\times$ pitch) so rules remain distinguishable. At an active constraint the correct second-order condition is definiteness on the **critical cone** — the null space of the active constraint normals — not on all of $\mathbb{R}^{2N}$; and the correct first-order condition is that the *projected* gradient vanishes, the raw gradient being balanced by KKT multipliers.
3. **Curvature is informative only at a stationary point.** Galerkin orthogonality (4) certifies the consequent block by construction and therefore says nothing about $(a,b)$; the antecedent gradient must be checked separately.

Our certificate accordingly reports the KKT residual on the critical cone, the projected-Hessian spectrum, and a falsification probe that walks along the most-negative eigenvector with the first-order term subtracted, to test whether the action actually decreases. Because the Hessian is obtained by finite differences, the verdict is tolerance-based: we compare against an estimated noise floor $\varepsilon_f/h^{2}$ rather than against exact zero, since judging against zero reads numerical dust as a saddle. This is the one place where we certify *numerically* rather than analytically (guarantee G11 in §10).

---

## 4. Rule orthogonality and sequential identification

The original design question was whether rules can be constrained so that $\int f_i f_j\,dx=0$ for $i\ne j$, and whether rules can then be identified sequentially from residuals. Both halves have exact answers, and they are not the expected ones.

### 4.1 Greedy equals joint iff the Gram is block-diagonal

Partition the regressor index set by rule, and block $G$ accordingly.

> **Proposition 3.** Sequential fitting — fit rule 1, then fit rule 2 to the residual, and so on — returns the joint minimizer of (2) for *every* target $y_d$ if and only if $G$ is block-diagonal with respect to the rule partition.

*Proof.* If $G$ is block-diagonal the normal equations decouple into one system per rule and greedy solves each exactly. Conversely, suppose $G_{IJ}\ne0$ for some $I\ne J$. Choose $y_d$ in the span of block $J$'s regressors. Greedy assigns block $I$ a nonzero coefficient (since $r_I=G_{IJ}\theta_J\ne0$), whereas the joint solution assigns block $I$ zero. $\blacksquare$

Note the quantifier: block-diagonality is required for agreement on *all* targets. For a particular $y_d$ greedy may coincide with the joint fit accidentally.

> **Proposition 4.** For $\lambda>0$, block-diagonality of $G$ does **not** require disjoint rule supports.

*Proof sketch.* The block entry is $\int\varphi_i\varphi_j\,dx+\lambda\int\varphi_i'\varphi_j'\,dx$. For adjacent normalized rules the first integral is strictly positive and the second is strictly negative, because the partition of unity forces $\varphi_i'=-\varphi_j'$ in the two-rule crossover. Cancellation is therefore possible at a strictly positive $\lambda$ with overlapping supports. $\blacksquare$

Proposition 4 corrects a claim we initially made in the other direction, obtained by setting $\lambda=0$ in a statement whose hypothesis required $\lambda>0$.

### 4.2 The orthogonalizing slope weight in closed form

Proposition 4 raises the constructive question: what value of $\lambda$ achieves cancellation? For two rules it is elementary.

> **Theorem 5 ($\lambda^{*}$).** For two equal-width Gaussian membership functions with centres $\pm c$ and width $b$ on $\mathbb{R}$, the normalized weights collapse to a logistic, $\varphi_1=\sigma(kx)$ with $k=4c/b^{2}$, and
> $$\int_{\mathbb{R}}\varphi_0\varphi_1\,dx=\frac1k,\qquad \int_{\mathbb{R}}\varphi_0'\varphi_1'\,dx=-\frac{k}{6},$$
> so the unique $\lambda$ making the two rules $H^1$-orthogonal is
> $$\boxed{\ \lambda^{*}=\frac{6}{k^{2}}=\frac{3b^{4}}{8c^{2}}\ }\qquad\Longleftrightarrow\qquad \ell^{*}=\sqrt{6}\,w,\quad w=\frac{1}{k}=\frac{b^{2}}{4c}.$$

*Proof.* The log-ratio $\log(\mu_1/\mu_0)=4cx/b^{2}$ is linear in $x$, hence $\varphi_1=\sigma(kx)$ with $k=4c/b^{2}$, and $\varphi_0\varphi_1=\sigma(1-\sigma)=\sigma'(kx)$, $\varphi_i'=\pm k\sigma'$. Substituting $u=kx$ and then $s=\sigma(u)$, $ds=s(1-s)\,du$: $\int\varphi_0\varphi_1\,dx=k^{-1}\int_\mathbb{R}\sigma'(u)\,du=k^{-1}$ and $\int\varphi_0'\varphi_1'\,dx=-k\int_\mathbb{R}\sigma'(u)^2du=-k\int_0^1 s(1-s)\,ds=-k/6$. Setting $k^{-1}-\lambda k/6=0$ gives the result. $\blacksquare$

**Interpretation.** The target $y_d$ appears nowhere in Theorem 5. Therefore $\lambda^{*}$ is **not** a distinguished correlation length of the data; it is the correlation length of the rule crossover region, fixed entirely by partition geometry. The ratio $\ell^{*}/w=\sqrt{6}\approx2.44949$ was confirmed numerically in every configuration tested.

> **Corollary 6 (Why $N\ge3$ fails).** On a uniform partition of pitch $d$, adjacent pairs (separation $d$) and next-nearest pairs (separation $2d$) demand orthogonalizing weights in the ratio $(2d/d)^{2}=4$, since $\lambda^{*}\propto c^{-2}$. A single scalar $\lambda$ therefore cannot orthogonalize three or more uniformly spaced rules.

The measured ratio was $4.0000$ in every case, which explains why residual off-block coupling plateaus near $10^{-2}$ for $N\ge3$ rather than approaching zero.

**The price of $\lambda^{*}$.** Choosing $\lambda=\lambda^{*}$ trades $L^2$ accuracy for decoupling. Measured against the $L^2$-optimal ($\lambda=0$) fit of the same partition, target $\tanh(x/3)$:

| $c$ | $b$ | $w$ | $\lambda^{*}$ | $\ell^{*}$ | $L^2$ at $\lambda{=}0$ | $L^2$ at $\lambda^{*}$ | cost |
|---|---|---|---|---|---|---|---|
| 5 | 1 | 0.050 | 0.015 | 0.12 | 3.2540 | 3.2542 | 0.01% |
| 5 | 2 | 0.200 | 0.240 | 0.49 | 1.0992 | 1.1036 | 0.40% |
| 5 | 4 | 0.800 | 3.840 | 1.96 | 0.5191 | 0.5394 | 3.91% |
| 5 | 6 | 1.800 | 19.44 | 4.41 | 0.1799 | 0.1938 | 7.72% |
| 5 | 8 | 3.200 | 61.44 | 7.84 | 0.7769 | 0.8204 | 5.60% |
| 3 | 4 | 1.333 | 10.67 | 3.27 | 0.1101 | 0.1174 | 6.59% |

The cost stays below 8% across the sweep and below 2% for sharp crossovers. The closed form explains why: $\ell^{*}=\sqrt6\,b^{2}/(4c)$ is sub-rule-scale for any sensible partition, and weighting slopes at a short correlation length barely perturbs the $L^2$ solution.

### 4.3 The practical alternative

Since exact orthogonality is unavailable for $N\ge3$, the usable route is to orthogonalize the **regressors** rather than the rules: an $H^1$ Gram–Schmidt (orthogonal least squares) pass on $\psi$, selecting rules greedily by residual reduction. In exact arithmetic this restores the greedy/joint agreement of Proposition 3 by construction, at the cost of rules that are no longer individually interpretable in the original basis. In floating point the restoration is conditioning-dependent rather than unconditional: at narrow, well-separated rule widths the residual off-block coupling collapses to $10^{-5}$–$10^{-7}$, but at wide, heavily overlapping placements (where the original regressors are most nearly linearly dependent) it can remain of order 1 — "by construction" describes the exact-arithmetic guarantee, not a numerical one at every width.

---

## 5. Control: an exact suboptimality certificate

### 5.1 The degenerate case

> **Theorem 7 (TSK reproduces LQR exactly).** For an LTI plant with quadratic cost, an order-1 TSK controller with any partition of unity and consequents $f_i(x)=-Kx$ for all $i$ satisfies $u(x)=\sum_i\varphi_i(x)f_i(x)=-Kx$, the LQR [@kalman1960contributions] optimum, identically.

The proof is $\sum_i\varphi_i\equiv1$. We record Theorem 7 because it is the correct baseline and because it is **vacuous as a recommendation**: it says the model class contains the linear optimum, so any benchmark on which LQR is optimal cannot discriminate. This is why §6 scores on settling time, peak force and energy jointly — objectives under which LQR is not optimal — rather than on the quadratic cost LQR was designed to minimize.

### 5.2 The exact certificate, for any convex cost

Consider a control-affine plant $\dot x=f(x)+g(x)u$ with running cost $\ell(x,u)=q(x)+c(u)$, optimal value $V^{*}$ and optimal policy $u^{*}$.

> **Theorem 8 (Bregman suboptimality identity).** Let $c$ be convex and differentiable, let $V^{*}$ be differentiable and satisfy the Hamilton–Jacobi–Bellman equation [@kirk1970optimal; @bardicapuzzodolcetta1997optimal], and let $u(\cdot)$ be admissible (the closed loop converges to the origin with $V^{*}(x(t))\to0$). Then
> $$\boxed{\ J(x_0)-V^{*}(x_0)=\int_0^{\infty} D_c\big(u(t)\,\|\,u^{*}(x(t))\big)\,dt\ }\tag{7}$$
> where $D_c(u\|v)=c(u)-c(v)-c'(v)(u-v)$ is the Bregman divergence [@bregman1967relaxation] of $c$.

*Proof.* HJB stationarity gives $c'(u^{*})=-g^{\top}\nabla V^{*\top}$. Hence

$$\ell(x,u)+\nabla V^{*}(f+gu)=q+c(u)+\nabla V^{*}f+\nabla V^{*}gu=c(u)-c(u^{*})-c'(u^{*})(u-u^{*})=D_c(u\|u^{*}),$$

using $q+\nabla V^{*}f+\nabla V^{*}gu^{*}+c(u^{*})=0$ (HJB). The left side is $\ell+\tfrac{d}{dt}V^{*}(x(t))$ along the closed loop; integrating from $0$ to $\infty$ and using admissibility to kill the boundary term gives (7). $\blacksquare$

> **Corollary 9.** For $c(u)=u^{\top}Ru$, $D_c(u\|u^{*})=(u-u^{*})^{\top}R(u-u^{*})$ and (7) reduces to the familiar $J-V^{*}=\int(u-u^{*})^{\top}R(u-u^{*})\,dt$.

Three features deserve emphasis. The identity is **exact** — no constants, no inequalities. Non-negativity of the gap is **free**, because $D_c\ge0$ *is* convexity of $c$. And it holds for costs that are not polynomial:

| cost | controller | gap | $\int D_c\,dt$ | residual |
|---|---|---|---|---|
| $\cosh u-1$ | $0.5u^{*}$ | 3.099e−1 | 3.099e−1 | 3.2e−12 |
| $\cosh u-1$ | $1.7u^{*}$ | 5.172e−1 | 5.172e−1 | 4.0e−11 |
| $\cosh u-1$ | $-0.4x$ | 7.536e−1 | 7.536e−1 | 2.8e−11 |
| $u^2+u^4$ | $0.5u^{*}$ | 1.857133e−1 | 1.857133e−1 | 1.1e−11 |
| $u^2+u^4$ | $-0.4x$ | 8.955264e−2 | 8.955264e−2 | 1.4e−11 |

**The hypotheses bind.** Admissibility is not decorative. A controller fitted without the constraint $u(0)=0$ leaves a steady-state offset, the closed loop settles to a nearby point rather than the origin, $V^{*}(x(t))\not\to0$, and the identity silently fails. We encountered exactly this, and it is also why the constraint reappears in §7.

**What lies outside.** Theorem 8 certifies *integral functionals* of the closed loop. Peak force $\max_t|u|$ and settling time are not integral functionals — one is a supremum, the other an exit time — so two of the three benchmark objectives in §6 are provably outside the certified envelope and are reported as measurements.

---

## 6. Benchmark: two-mass–spring

### 6.1 Setup

Two unit masses coupled by a unit spring, force applied to the first mass, saturating at $|u|\le1$; state $z=(x_1,x_2,v_1,v_2)$. This is a simplified nominal instance of the two-mass–spring system long used as a robust-control benchmark [@wiebernstein1992benchmark] — that source's canonical form carries an uncertain spring constant and asymmetric masses, which we do not use here; §12 discusses exercising an uncertainty specification, which we do not currently do.

Controllers are scored by the equal-weight normalized scalarization

$$\text{score}=\tfrac13\left(\frac{T_{\text{settle}}}{T^{\text{ref}}}+\frac{\max_t|u|}{|u|^{\text{ref}}}+\frac{\int u^2dt}{E^{\text{ref}}}\right),$$

with the reference the best LQR over a gain sweep, so LQR scores 1.000 by construction. The question asked was how fast the least-action fit converges **with no refinement**: the recipe is applied once, with no tuning — weighted $k$-means rule placement on open-loop optimal trajectories, then variable projection of the consequents (one linear solve, globally optimal by Theorem 1), then simulate.

### 6.2 It converges in two rules

| rules | params | settle | peak $\lvert u\rvert$ | energy | score | shift | settled |
|---|---|---|---|---|---|---|---|
| LQR | 4 | 19.797 | 1.0000 | 0.5375 | 1.0000 | — | yes |
| open-loop ref | — | 10.972 | 0.8615 | 0.5585 | 0.8183 | — | yes |
| 1 | 5 | 26.773 | 0.9575 | 0.4214 | 1.0313 | 3.05 | yes |
| **2** | **10** | 17.830 | 0.9088 | 0.4248 | **0.8666** | 3.07 | yes |
| **3** | **15** | 17.217 | 0.9023 | 0.4390 | **0.8629** | 3.05 | yes |
| 4 | 20 | 18.470 | 0.9088 | 0.4465 | 0.8908 | 3.07 | yes |
| 6 | 30 | 18.257 | 0.8560 | 0.4798 | 0.8903 | 3.07 | yes |
| 8 | 40 | 19.040 | 0.5531 | 0.6121 | ∞ | 3.07 | **no** |
| 12 | 60 | 14.313 | 1.0000 | 1.2041 | 1.3211 | 4.67 | yes |
| 16 | 80 | 34.990 | 1.0000 | 7.8944 | ∞ | 6.73 | **no** |

"shift" is the maximum distance from a closed-loop state to the nearest training sample, in units of the training set's per-axis spread.

Ten parameters take the score from 1.031 (worse than LQR) to 0.867; the third rule adds 0.4% and that is the end of it. Two rules close **73%** of the gap between the best LQR and the open-loop reference, $(1.000-0.8666)/(1.000-0.8183)$. The win is specific rather than uniform: at $N=3$ the fuzzy controller uses **18% less energy than the best LQR** and 21% less than the open-loop reference it was trained on, and 10% less peak force, while being *worse* on settling time (17.2 against 11.0). It trades response speed for actuator economy.

### 6.3 Then it degrades, for a diagnosable reason

Past $N\approx3$ the score degrades monotonically and by $N=8$ the closed loop stops settling. This is not a numerical artifact and not a failure of the variable projection, which remains a globally optimal linear solve at every rule count. It is **distribution shift**, and the shift column tracks the failure exactly: 3.05 while the method works, 4.67 at $N=12$, 6.73 at $N=16$. The controller is fitted only on optimal trajectories, so it has no data off them; more rules mean each rule is supported by fewer samples and extrapolates harder.

> **Rule count and closed-loop quality are not monotonically related.** The least-action fit is globally optimal *for the data it was given*, and the data is the binding constraint — not the model class and not the optimizer.

Off-trajectory augmentation (perturbed tubes, or one DAgger [@rossgordonbagnell2011reduction] round) removes the collapse but does not move the plateau near 0.86; nor do higher-order consequents, which shift it by 1.6% while fit accuracy and closed-loop score become *inversely* related. Both experiments point at the objective rather than the model.

### 6.4 Direct policy optimization breaks the plateau

Testing the implied fix — stop fitting sampled targets, optimize the closed-loop objective directly over the FIS parameters — requires surrendering Theorem 1: the closed-loop objective is not quadratic in $\theta$, so only derivative-free search remains. We use Powell on a shaped surrogate in which each discontinuous term is replaced by a finite analogue (soft time outside a tolerance ball; log-sum-exp peak; unchanged energy; plus a terminal-norm term that makes a non-settling controller *improvable* rather than merely rejected).

| rules | params | imitation | direct | improvement |
|---|---|---|---|---|
| **2** | **10** | 0.8666 | **0.6547** | **24.5%** |
| 3 | 15 | 0.8629 | 0.6729 | 22.0% |
| 4 | 20 | 0.8908 | 0.7565 | 15.1% |

At two rules the optimized controller improves **all three objectives at once** — 31% faster settling, 35% less peak force and 37% less energy than the best LQR — which no imitation controller did. It is 34.5% better than the best LQR, 22.9% better than the best imitation result, and 20.0% better than the open-loop reference.

**The warm start is load-bearing.** The same structure started from zero consequents instead of the least-action fit failed outright: 601 evaluations, no stabilizing controller found. The two stages are complementary rather than competing.

> **The least-action fit is not the final method — it is the initializer that makes the final method reachable.** On its own it caps out at 0.863; without it, direct optimization does not start.

This rehabilitates variable projection in a specific role. Its value is not that it produces the best controller, but that it produces a **stabilizing** one in a single globally optimal linear solve, with no search and no initialization problem of its own.

---

## 7. Stability certification by sum of squares

### 7.1 Rational membership functions

Replacing Gaussians with $\pi$-shaped (Cauchy [@viana2021cauchy]) membership functions $\mu_i(z)=\big(1+\|(z-a_i)/b_i\|^{2}\big)^{-1}$ makes every $\mu_i$ rational, hence $\varphi_i$ rational, hence the TSK control law exactly a ratio of polynomials $u(z)=P(z)/Q(z)$ with $Q>0$ structurally. The closed loop is then a rational vector field and the Lyapunov decrease condition becomes a polynomial inequality after clearing denominators — which is what a sum-of-squares (SOS) [@parrilo2000structured; @parrilo2003semidefinite] program can decide.

### 7.2 The certificate

With $V(z)=z^{\top}Pz$, we certify $\dot V<0$ on $\{V\le\rho\}\setminus\{0\}$ via the S-procedure [@boyd1994linear]: find $\sigma$ SOS with

$$W(z)-\varepsilon\|z\|^{2}-\sigma(z)\big(\rho-V(z)\big)\ \text{ SOS},\tag{8}$$

where $W=-\dot V\cdot Q^{2}$ is the numerator-cleared decrease condition. Two implementation facts were decisive.

**The constraint $u(0)=0$ is not cosmetic.** An unconstrained fit gives $u(0)=O(10^{-4})$, so the origin is *not* an equilibrium, the closed loop settles to a nearby offset point, and no Lyapunov function can certify asymptotic stability to a non-equilibrium. The same defect voids Theorem 8 through its admissibility hypothesis. Because $u(0)=0$ is a single linear equality in $\theta$, it is imposed exactly by a KKT system, so the fit remains one linear solve and remains the **global** optimum subject to the constraint — Theorem 1 survives intact.

**Both SOS bases must start at degree 1.** At $z=0$ the target of (8) is $-\sigma(0)\rho<0$, which cannot be SOS unless $\sigma(0)=0$. Left unconstrained, the solver silently drives $\sigma(0)\to0$ and every Gram matrix sits on the positive-semidefinite boundary at $10^{-10}$. Forcing both bases to begin at degree 1 gives a genuine margin of $3.5\times10^{-3}$ at the same certified $\rho$.

### 7.3 The obstruction was the solver

> **Provenance flag.** The table below has no corresponding entry in `results.txt`; the one recorded solver comparison (`demo_sos.py` §3) runs a single $\rho=0.6$, not the three-point sweep shown here. An attempt to regenerate this exact sweep during review surfaced a genuine bug rather than a replacement table: this repo's dependency spec (`requirements.txt` at the time, since consolidated into `pyproject.toml`) pinned `numpy~=1.26.4`, but `fis_twocart.py`'s `simulate` calls `np.trapezoid`, a name that does not exist before NumPy 2.0 (it superseded `np.trapz`) — the pinned environment could not run the two-cart benchmark at all. The floor is now NumPy $\ge2.4.6$ in `pyproject.toml` (fixed alongside this review; the smoke tests and the 21 symbolic identities of §11 both re-verified passing under it), so that specific blocker is closed.
>
> What is *not* closed, and why: `np.trapezoid` was in `fis_twocart.py` from that file's first commit, so `results.txt` was itself necessarily generated under some NumPy $\ge2$ — the stale pin was wrong from the start, not a later drift. Re-running under NumPy 2.4.6 with every other dependency at its pinned version still reproduces only the *shape* of §6.2's table, not its recorded values — and the same non-matching numbers came back identically whether the run used scipy 1.15.3 (the pin) or 1.17.1 (ruling out the scipy version as the variable). The likely reason is structural rather than a pin at all: `fit()` (`fis_action.py`) is a 24-random-restart search over SLSQP and L-BFGS-B with finite-difference Hessians on a genuinely non-convex objective (§3.3 already documents needing a negative-curvature escape to avoid saddles). That is exactly the kind of pipeline where floating-point-level differences below the level any package-version pin tracks — BLAS/LAPACK build, SIMD codepath, thread count and reduction order — can tip which of 24 local optima a given restart lands closest to, without any single step doing anything "wrong." A seeded RNG guarantees the same *draws*; it does not guarantee the same *optimizer path* once those draws are processed through a chain of tolerance-based nonlinear solves. Given that, the table below is retained as originally drafted rather than replaced with numbers of equally uncertain provenance, and §6.2's headline numbers should be read as a report of what one run of a chaotic search found, not as a bit-exact target every environment must hit.

| solver | $\sigma$ degree | $\rho=0.5$ | $\rho=0.8$ | $\rho=1.0$ |
|---|---|---|---|---|
| SCS (first-order) [@odonoghue2016conic] | 4 | −0.158 | −0.236 | −0.250 |
| **CLARABEL** (interior-point) [@goulartchen2024clarabel] | **2** | −1.7e−10 ✓ | −1.6e−10 ✓ | −4.9e−11 ✓ |
| **CLARABEL** (interior-point) [@goulartchen2024clarabel] | **4** | −8.2e−10 ✓ | −1.1e−09 ✓ | −1.8e−10 ✓ |

(Gram minimum eigenvalue; ✓ = certified.) CLARABEL closes at multiplier degree **2** — the degree at which SCS reported a violation of −0.25 — in well under a second. The diagnostic that identified this was the **relative** violation $|\lambda_{\min}|/\max|G|$: SCS bottoms out at $10^{-6}$ to $10^{-5}$, a first-order method's realistic accuracy on this problem, while CLARABEL reaches $10^{-11}$. Reading the *absolute* eigenvalue suggested gross infeasibility; the relative figure showed it was noise. We record this because it cost four hours and because the conclusion generalizes: on small, badly scaled SOS programs, interior-point and first-order solvers are not interchangeable.

The certified region for the two-rule imitation controller is $\{z^{\top}Pz\le1.1145\}$.

### 7.4 Exact rational verification

An interior-point SDP certifies only to solver tolerance. We upgrade the result by the standard rounding-and-projection procedure [@peyrlparrilo2008computing]: solve with a strictly positive margin; round both Gram matrices to rationals with bounded denominator; rebuild the target polynomial exactly over $\mathbb{Q}$ (IEEE doubles *are* rationals, so only the deliberate rounding introduces error); project the rounded Gram onto the affine set where the polynomial identity holds exactly, which decouples monomial-by-monomial and is available in closed form; and verify positive definiteness in exact arithmetic.

Two practical notes. Rational $LDL^{\top}$ is exact but useless at this size, because denominators square at each elimination step and the $14\times14$ factorization does not finish; integer leading principal minors computed by fraction-free (Bareiss) elimination have no such blow-up. And symbolic "rationalize" helpers that search for *nice* closed forms must be avoided — one returned $2^{818/971}3^{651/971}$ for an ordinary double — in favour of the exact dyadic conversion.

**Result.** At $\rho=1/2$ the projected identity has residual exactly $0$, and all 14 leading principal minors of the $14\times14$ Gram are positive integers, the largest a **27,737-bit** integer. Therefore $z^{\top}Pz$ decreases strictly on $\{z^{\top}Pz\le1/2\}\setminus\{0\}$ for the rationalized controller — **proved rather than measured**, with no floating-point step remaining in the chain.

### 7.5 Certification does not scale in rule count

Since $\deg Q=2(N-1)$, the SDP grows quickly: the Gram is $35\times35$ at $N=2$, $70\times70$ at $N=3$, $126\times126$ at $N=4$ and $715\times715$ at $N=8$. $N=2$ solves in seconds; $N=8$ is out of reach with these solvers. **Certification is tractable only at low rule count.** It happens to land well here, since §6 found convergence at two or three rules, but any future move toward more rules trades certifiability for performance.

---

## 8. Certificate-aware policy optimization

### 8.1 An uncontrolled trade

Certifying the directly optimized controller of §6 exposed a trade that nothing in the objective knew about: performance improved ($1.044\to0.773$) while the certified ball **shrank** ($0.568\to0.274$). The optimizer buys performance by acting more aggressively, and aggression is exactly what a Lyapunov certificate charges for.

### 8.2 A proxy that is safe to optimize against

Putting the certified radius into the objective is obstructed by cost: one SOS solve is ~1.5 s and a bisection over two Lyapunov candidates is 47 s per controller, against a 601-evaluation search. We therefore optimize a sampled surrogate,

$$\hat\rho=\min\{\,z^{\top}Pz\ :\ \dot V(z)\ge0\,\},\qquad \hat r=\sqrt{\hat\rho/\lambda_{\min}(P)}.\tag{9}$$

Its **logical status** is what makes this sound. A state with $\dot V(z)\ge0$ *proves* the SOS program must fail at that $\rho$, since a certificate is a statement about every point of the sublevel set. Hence $\hat\rho$ is an **upper bound** on any achievable certified $\rho$ — a necessary condition, never sufficient. That is the right direction for a search objective and the wrong direction for a claim, so the proxy is optimized against and never reported: every controller is subsequently passed through the real semidefinite program.

The penalized objective is

$$J_w(\theta)=\text{cost}(\theta)+w\,\frac{\max\!\big(0,\,r_t-\hat r(\theta)\big)^{2}}{r_t^{2}},\tag{10}$$

a one-sided squared hinge — exceeding the target is never punished — normalized so $w$ is dimensionless and comparable to the score. The search remains restricted to the $u(0)=0$ null space of §7.

### 8.3 The front, and a result in the opposite direction

| $w$ | score | held out | proxy $\hat r$ | SOS ball | $V$ from |
|---|---|---|---|---|---|
| — (imitation) | 1.0437 | 1.1395 | 0.6718 | **0.5684** | Riccati |
| 0 | 0.7725 | 0.8272 | 0.3900 | 0.2729 | lin |
| 0.1 | 0.8155 | 0.7717 | 0.8676 | 0.3452 | lin |
| **0.3** | **0.7713** | **0.7194** | 0.6124 | **0.3354** | lin |
| 1 | 0.7906 | 0.7230 | 0.7039 | 0.3498 | lin |
| 3 | 0.8265 | 0.8105 | 0.8676 | **0.3579** | lin |

"SOS ball" is the real certificate, best of two Lyapunov candidates; "held out" is the score on six initial conditions not used in the objective.

The expected outcome was a strict trade. Instead, $w=0.3$ **dominates $w=0$ on all three axes**: better score (0.7713 against 0.7725), better held-out score (0.7194 against 0.8272), and a larger certified ball (0.3354 against 0.2729). The held-out column explains why. The unpenalized search overfits its six training initial conditions, degrading 7% out of sample, and the certificate penalty suppresses precisely the aggressive behaviour responsible.

> **A Lyapunov certificate acts as a regularizer.** It charges for the same aggression that costs generalization, so the first increment of certificate-awareness is free.

Past that point the trade is real and $w$ prices it: over $w\in\{0.3,1,3\}$ the certified ball rises $0.3354\to0.3498\to0.3579$ while the score degrades $0.7713\to0.7906\to0.8265$.

---

## 9. Calibrating the proxy

§8 left the certificate a dial rather than a specification: the proxy over-stated the certified ball by 1.18 to 2.51×, so asking for 0.67 delivered 0.35. We had attributed this to S-procedure conservatism, which would have required a higher multiplier degree. It was instead **estimator error**, of two distinct kinds.

### 9.1 Two defects, both in the estimator

**A minimum-order statistic.** Violations are not rare — **37% of cloud points satisfy $\dot V\ge0$** — so the sampled minimum in (9) depends entirely on whether some point happened to land near the inner boundary. Across five seeds it reports 0.6718, 0.6744, 0.7484, 0.6608, 0.7369: a 12.5% spread. Fixing the seed made it deterministic, not accurate.

**The obvious repair is degenerate.** Sharpening (9) by locally solving $\min\{z^{\top}Pz:\dot V(z)\ge0\}$ fails *structurally* rather than numerically: **$z=0$ is feasible**, since $\dot V(0)=0$ exactly. The program has global optimum 0 at the origin and a descent method walks to it. Measured: of six local solves from the best sampled starts, two collapsed to $\|z\|=0$ and all six terminated infeasible on the $\dot V=0$ boundary, so a strict feasibility recheck discards every one and the refined bound never moves.

### 9.2 Ray search

Both defects vanish if the search is organized by direction. Along a fixed unit vector $d$ define the first violation radius

$$r(d)=\min\{\,r>0:\dot V(rd)\ge0\,\},\qquad \hat\rho=\min_{d}\;r(d)^{2}\,d^{\top}Pd,\tag{11}$$

computed by scan-and-bisect over a fixed direction set. There is no optimizer, no degenerate solution, and the origin is excluded by construction because the scan starts at $r>0$. Bisection returns the **outer** bracket endpoint, which is a state genuinely satisfying $\dot V\ge0$, so the necessary-condition property is preserved exactly. This is checked rather than argued: all 973 returned endpoints satisfy $\dot V\ge0$ on re-evaluation, and the binding one is the explicit state $z^{*}=(+0.0237,+0.0778,+0.0628,-0.2563)$ with $\dot V=+2.9\times10^{-9}$.

| estimator | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | spread | vs SOS |
|---|---|---|---|---|---|---|---|
| cloud | 0.6718 | 0.6744 | 0.7484 | 0.6608 | 0.7369 | 12.5% | 1.182× |
| **ray** | **0.5937** | **0.5934** | **0.5849** | **0.5937** | **0.5915** | **1.5%** | **1.044×** |

### 9.3 What remains is a constant

Perturbing the fit along the $u(0)=0$ null space produces controllers spanning a range of aggressiveness, each still with an equilibrium at the origin. Over five that certify,

$$\kappa=\operatorname{median}\!\left(\frac{\text{SOS}}{\text{ray}}\right)=0.9620,\qquad\text{range }[0.9271,\,0.9735].$$

Of the 3.8% shortfall, **2.5 points are the deliberate 0.95 shrink** applied to $\rho$ before re-certifying ($\sqrt{0.95}=0.9747$). Genuine degree-4 S-procedure conservatism is therefore about **1.3%** — confirming that chasing multiplier degree would have bought nothing.

### 9.4 A specified radius, confirmed by the solver

Hinging (10) on the calibrated estimate $\kappa\hat r$ states the target directly in certified-ball units. A pass that misses raises the target by exactly the shortfall; a pass that misses *and* fails to improve on the previous one is not short of target but stuck, and the target is pushed 50% further.

| spec | pass | score | held out | $\kappa\hat r$ | SOS Ric | SOS lin | best | met? |
|---|---|---|---|---|---|---|---|---|
| 0.35 | 1 | 0.7956 | 0.7546 | 0.3478 | 0.0000 | 0.3415 | 0.3415 | no |
| 0.35 | 2 | 0.7962 | 0.7227 | 0.3577 | 0.0000 | 0.3549 | **0.3549** | **yes** |
| 0.45 | 1 | 0.7690 | 0.7296 | 0.5277 | 0.0000 | 0.3452 | 0.3452 | no |
| 0.45 | 2 | 0.7857 | 0.7536 | 0.5865 | 0.0000 | 0.3430 | 0.3430 | no |
| 0.45 | 3 | 0.8487 | 0.8667 | 0.7269 | 0.6782 | 0.3654 | **0.6782** | **yes** |
| 0.55 | 1 | 0.7717 | 0.7466 | 0.5503 | 0.1038 | 0.3368 | 0.3368 | no |
| 0.55 | 2 | 0.8469 | 0.8560 | 0.7278 | 0.6791 | 0.0000 | **0.6791** | **yes** |

All three specifications are met, and **every "yes" is a real SDP certificate**, not a proxy estimate.

**The best controller of the entire study falls out of this procedure**: certified ball **0.6791** at score **0.8469**, which dominates the imitation controller on *both* axes — a larger certified region than 0.5684 and a 19% better score than 1.0437 — and nearly doubles the best certified region obtained in §8.

### 9.5 Two limits calibration does not remove

**A single $\kappa$ can fail.** At specification 0.45, pass 1, $\kappa\hat r=0.5277$ against a certificate of 0.3452: a ratio of 0.65, far outside $[0.927,0.974]$. The per-candidate columns show why — `SOS Ric = 0.0000`. The proxy takes a maximum over Lyapunov candidates and attains it at the Riccati matrix, where $\dot V<0$ genuinely does hold that far out, but which the degree-4 S-procedure cannot certify **at all**. The proxy measures where $\dot V<0$ *is true*; SOS measures where it is *provable at this degree*; the argmax of the two differs.

**The achievable set is bimodal.** Every row lands near either 0.35 or 0.68, according to which Lyapunov function certifies — two basins, not a continuum. Specification 0.45 therefore fails twice and then succeeds, and asking for 0.45 delivers 0.6782: the specification is **met, not matched**. This is a property of the landscape, not of the calibration. Both limits point at the same remedy: searching $V$ jointly with $\theta$, a bilinear problem, would remove the max-over-two-candidates artifact that produces the calibration outlier and the bimodality alike.

---

## 10. What is proved, and what is not

| # | Guarantee | Status | Conditions |
|---|---|---|---|
| G1 | Consequents globally optimal, closed form | **Proven** | antecedents fixed; independent regressors |
| G2 | Action convex in function space | **Proven** | $\lambda\ge0$ |
| G3 | Reduced gradient exact | **Proven** | $\delta$-coverage |
| G4 | Greedy = joint iff Gram block-diagonal | **Proven** | for all targets |
| G5 | $\lambda^{*}=3b^{4}/(8c^{2})$ | **Proven** | 2 equal-width Gaussians |
| G9 | TSK reproduces LQR exactly | **Proven** | LTI, quadratic cost (and vacuous) |
| G10 | Bregman suboptimality identity | **Proven** | control-affine, convex $c$, admissible, $V^{*}$ known |
| G11 | Local optimality of antecedents | Certified numerically | KKT on critical cone + FD Hessian |
| G12 | Stability / region of attraction | **Proven by SOS** | rational MFs, $u(0)=0$, interior-point SDP |
| G13 | Global optimality over antecedents | Not available | — |
| G14 | Certificate for settling time, peak force | Provably outside G10 | not integral functionals |

(G6–G8 cover the classifier defuzzification results and are deferred to a companion paper; the numbering is preserved so it matches the implementation notes.)

Two obstructions did not yield and should be stated plainly. **G13** is genuinely open: the antecedent landscape is multimodal and nothing here bounds the gap to the global optimum. **G14** is not an open problem but a structural exclusion — a supremum and an exit time are not integral functionals, so no argument of the form of Theorem 8 can reach them. The honest route to certifying the benchmark objectives is to *restate* them as integral costs ($\int\phi(u)\,dt$ with $\phi$ a convex barrier; $\int w(t)\|x\|^{2}dt$), which changes the problem in a direction the framework can certify.

---

## 11. Machine verification

Three tiers of checking, weakest to strongest.

**Tier 1 — symbolic identity checking.** The verification suite checks 21 identities in a computer algebra system as *identities* rather than at sampled points; 16 bear on the results in this paper (the Euler–Lagrange equation, the partition-of-unity consequences, Theorem 7, the Bregman identity of Theorem 8 and its quadratic specialization, the second-derivative identity $D_c''(u)=c''(u)$ underlying non-negativity of $D_c$ for two non-quadratic costs — convexity itself, $c''\ge0$, is not separately checked by this suite — the three $\lambda^{*}$ integrals of Theorem 5, the exact rational form $u=P/Q$ and $Q>0$, the Hessian and stationarity claims of Theorem 1, and the envelope-theorem gradient of Theorem 2). All 21 pass. One subtlety cost two of them for a while: a matrix comparison `Matrix(...) == 0` is `False` even for the zero matrix, and using the correct predicate moved the count from 19/21 to 21/21.

**Tier 2 — exact rational proof of the SOS certificate**, as described in §7.4: residual exactly zero, all 14 integer leading principal minors positive.

**Tier 3 — proof assistant.** A from-scratch Lean 4 + Mathlib formalization (`lean/`, alongside this paper) machine-checks the theorems whose entire content is finite-dimensional algebra or single-variable calculus: Theorem 1 (global optimality of the variable-projection solve, and its uniqueness when $G\succ0$), Theorem 7 (TSK reproduces LQR exactly, for any partition of unity and, more generally, any $\mathbb{R}$-module of consequent values), Corollary 6 (the factor-of-four obstruction), and the algebraic core of Theorem 8 — the pointwise HJB identity that produces the Bregman divergence, given HJB stationarity and the HJB equation as hypotheses. All of these compile with zero `sorry`s, and `#print axioms` on every theorem confirms each depends only on Lean's three standard foundational axioms (`propext`, `Classical.choice`, `Quot.sound`) — nothing admitted. Two pieces remain open, for the reason anticipated above: Theorem 8's *temporal* half — integrating the pointwise identity along an actual closed-loop trajectory from $0$ to $\infty$ and using admissibility to discard the boundary term — needs ODE existence/limit machinery beyond what was attempted; and Theorem 5's improper integrals ($\int\sigma'=1$, $\int(\sigma')^2=1/6$) are tractable in principle via Mathlib's `intervalIntegral` limit machinery but were not attempted here. The SOS side is deliberately left alone: it already sits in exact integer arithmetic (§7.4), and formalizing it would mean re-deriving an answer that is already decidable and decided.

---

## 12. Limitations and future work

- **One plant.** The control results use a single benchmark at nominal parameters. The two-mass–spring system is conventionally posed *with* a spring-constant uncertainty range, which we do not exercise; robustness of the certified region under that uncertainty is an obvious and immediate test.
- **Certification does not scale.** At $N=8$ the SDP is $715\times715$ and out of reach with these solvers. This bounds the whole certified story to low rule count.
- **Two fixed Lyapunov candidates.** Every certified radius reported is a maximum over the Riccati matrix and the closed-loop linearization. Searching $V$ jointly with $\theta$ is bilinear, was not attempted, and is the single change that would most improve §9.
- **Saturation is handled outside the certificate.** We report an unsaturated radius separately rather than encoding $|u|\le u_{\max}$ in the SOS program.
- **Relative form of Theorem 8.** The certificate presupposes $V^{*}$, which is what a forward problem lacks. For any $\hat V$ satisfying HJB with residual $\varepsilon$, the same algebra should bound the true gap in terms of $\varepsilon$; the useful part of that derivation is determining in which norm $\varepsilon$ must be measured. Without it, the certificate is restricted to inverse-optimal benchmarks — problems whose answer was known in advance.
- **Analytic Hessian.** G11 is certified numerically only because the Hessian is obtained by finite differences. The same envelope argument differentiated once more would upgrade it.

A detailed plan addressing these, with candidate comparison benchmarks, is given in the accompanying roadmap.

---

## Reproducibility

All results are produced by a deterministic, seeded implementation; recorded output is stored alongside the source, and the numerical constants that matter (quadrature order, ridge scaling, optimizer schedule, finite-difference steps, definiteness tolerances) are documented individually together with the failure each was chosen to prevent. Every table above is transcribed directly from that recorded output. Two exceptions are worth naming rather than quietly matching or leaving implicit.

First, the recorded log itself carries two slightly different values for the inscribed-ball radius across two demo scripts that compute it the same way (0.5685 vs. 0.5684); the tables above use the correct 0.5684 throughout, but "deterministic" describes each script against itself, not full agreement between scripts that should agree.

Second, and more consequentially: "deterministic... should match `results.txt` exactly" was checked here across environments and does not currently hold, for two separable reasons — one now fixed, one structural. This repo's dependency spec used to pin `numpy~=1.26.4` in `requirements.txt` while the code calls `np.trapezoid` (`fis_twocart.py`), a name introduced only in NumPy 2.0 (as the replacement for `np.trapz`) — meaning the pinned environment could not run `demo_twocart.py` at all. That half is fixed: dependencies now live in `pyproject.toml` (`requirements.txt` was consolidated away), NumPy is floored at $\ge2.4.6$, the full pinned stack reinstalls cleanly, and it reproduces the 8/8 smoke tests and 21/21 symbolic identities under it. The other half is not a pin problem and does not resolve by finding the "right" version: re-running the two-cart benchmark under NumPy 2.4.6, with scipy pinned exactly and also with a newer scipy (1.17.1) for comparison, reproduces the *shape* of §6.2's table both times but the same, different-from-recorded values both times — ruling out scipy version as the variable. `fit()`'s search (§3.2's optimizer, `fis_action.py`) is 24 random restarts of SLSQP/L-BFGS-B with finite-difference Hessians on a non-convex objective; a seeded RNG fixes the random *draws* but not the floating-point path a chain of tolerance-based nonlinear solves takes through BLAS/LAPACK, and on a landscape with multiple close local optima that path can determine which one a restart lands nearest. "Deterministic" is true and verified for a fixed environment (the smoke tests confirm bit-identical repeats); it was never true *across* environments, and no version pin makes it so. The honest fix is not chasing the original environment but re-generating and re-committing `results.txt` fresh against a documented, pinned one, and being explicit that §6.2's exact numbers are one search's outcome rather than a target every environment reproduces.

---

## References

::: {#refs}
:::
