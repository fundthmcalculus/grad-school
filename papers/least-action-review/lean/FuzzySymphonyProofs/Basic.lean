/-
Machine-checked verification of the tractable analytic claims in
`papers/least-action/paper.md` ("Certified Fuzzy Control by Least Action").

This file targets the theorems whose *entire* content is finite-dimensional
algebra or single-variable calculus, i.e. the pieces the paper's own §11
"Tier 3" note flags as candidates for Lean/Mathlib formalization. It is
deliberately NOT a formalization of the harder analytic pieces (existence of
trajectories, admissibility limits, the SOS/Positivstellensatz search itself)
-- see the accompanying report for exactly what is and is not covered here.
-/

import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.Deriv.Mul

open Matrix

namespace FuzzySymphony

/-!
## Theorem 1 (paper §3.1) — variable projection is a global minimizer

For fixed antecedents, the action `S(θ) = c - 2 rᵀθ + θᵀGθ` (c = ‖y_d‖²_{H¹})
is a convex quadratic in the consequent coefficients `θ`, with Hessian `2G`.
If `G` is positive semidefinite and `θ*` solves the normal equations `Gθ* = r`,
then `θ*` is a global minimizer of `S`; if `G` is positive definite the
minimizer is unique. This is the complete content of Theorem 1 (the reduced
action formula (5) is then just `S(θ*)`, by direct substitution).
-/

variable {n : ℕ}

/-- The action functional of paper eq. (2), restricted to the model class (1)
and written out after substituting `y_c = θᵀψ` -- this is the expression
appearing right after "Substituting (1) into (2)" in the proof of Theorem 1. -/
def action (G : Matrix (Fin n) (Fin n) ℝ) (r : Fin n → ℝ) (c : ℝ) (θ : Fin n → ℝ) : ℝ :=
  c - 2 * (r ⬝ᵥ θ) + θ ⬝ᵥ (G *ᵥ θ)

/-- A symmetric matrix defines a symmetric bilinear form via `⬝ᵥ` and `*ᵥ`. -/
theorem symm_bilinear_comm (G : Matrix (Fin n) (Fin n) ℝ) (hSymm : G.IsSymm)
    (u v : Fin n → ℝ) : u ⬝ᵥ (G *ᵥ v) = v ⬝ᵥ (G *ᵥ u) := by
  rw [Matrix.dotProduct_mulVec, ← Matrix.vecMul_transpose, hSymm.eq, Matrix.dotProduct_comm]

/-- The key algebraic step in Theorem 1's proof: "completing the square."
Expanding `S(θ)` around any solution `θ*` of the normal equations leaves
exactly the quadratic form `(θ-θ*)ᵀG(θ-θ*)` on top of `S(θ*)` — no linear
term survives, because `Gθ*=r` cancels it. -/
theorem action_eq_min_add_quadratic
    (G : Matrix (Fin n) (Fin n) ℝ) (hSymm : G.IsSymm)
    (r : Fin n → ℝ) (c : ℝ) (θstar θ : Fin n → ℝ) (hstar : G *ᵥ θstar = r) :
    action G r c θ = action G r c θstar + (θ - θstar) ⬝ᵥ (G *ᵥ (θ - θstar)) := by
  have hθ : θ = θstar + (θ - θstar) := by abel
  simp only [action]
  have hlin : G *ᵥ θ = G *ᵥ θstar + G *ᵥ (θ - θstar) := by
    conv_lhs => rw [hθ]
    exact Matrix.mulVec_add G θstar (θ - θstar)
  rw [hlin, hstar]
  have e1 : r ⬝ᵥ θ = r ⬝ᵥ θstar + (θ - θstar) ⬝ᵥ r := by
    conv_lhs => rw [hθ]
    rw [Matrix.dotProduct_add, Matrix.dotProduct_comm r (θ - θstar)]
  have e2 : θ ⬝ᵥ (r + G *ᵥ (θ - θstar))
      = θstar ⬝ᵥ r + θstar ⬝ᵥ (G *ᵥ (θ - θstar))
        + (θ - θstar) ⬝ᵥ r + (θ - θstar) ⬝ᵥ (G *ᵥ (θ - θstar)) := by
    conv_lhs => rw [hθ]
    rw [Matrix.add_dotProduct, Matrix.dotProduct_add, Matrix.dotProduct_add]
    ring
  rw [e1, e2]
  have hsym : θstar ⬝ᵥ (G *ᵥ (θ - θstar)) = (θ - θstar) ⬝ᵥ (G *ᵥ θstar) :=
    symm_bilinear_comm G hSymm θstar (θ - θstar)
  have hstarr : (θ - θstar) ⬝ᵥ (G *ᵥ θstar) = (θ - θstar) ⬝ᵥ r := by rw [hstar]
  rw [hsym, hstarr]
  ring

/-- **Theorem 1, global-optimality half.** Any solution of the normal
equations `G θ* = r` is a global minimizer of the action, provided `G` is
positive semidefinite. This is exactly "the Hessian `2G` is a Gram matrix of
an inner product, hence positive semidefinite, so `S` is convex and
stationarity is sufficient" from the paper's proof. -/
theorem variable_projection_global_min
    (G : Matrix (Fin n) (Fin n) ℝ) (hSymm : G.IsSymm) (hPSD : G.PosSemidef)
    (r : Fin n → ℝ) (c : ℝ) (θstar : Fin n → ℝ) (hstar : G *ᵥ θstar = r) :
    ∀ θ : Fin n → ℝ, action G r c θstar ≤ action G r c θ := by
  intro θ
  rw [action_eq_min_add_quadratic G hSymm r c θstar θ hstar]
  have hnonneg : 0 ≤ (θ - θstar) ⬝ᵥ (G *ᵥ (θ - θstar)) := by simpa using hPSD.2 (θ - θstar)
  linarith

/-- **Theorem 1, uniqueness half.** If `G` is positive *definite*, the
minimizer is unique: any other global minimizer coincides with `θ*`. -/
theorem variable_projection_unique
    (G : Matrix (Fin n) (Fin n) ℝ) (hSymm : G.IsSymm) (hPD : G.PosDef)
    (r : Fin n → ℝ) (c : ℝ) (θstar θ : Fin n → ℝ)
    (hstar : G *ᵥ θstar = r) (hmin : action G r c θ ≤ action G r c θstar) :
    θ = θstar := by
  have hle := variable_projection_global_min G hSymm hPD.posSemidef r c θstar hstar θ
  have heq : action G r c θ = action G r c θstar := le_antisymm hmin hle
  have hexpand := action_eq_min_add_quadratic G hSymm r c θstar θ hstar
  have hzero : (θ - θstar) ⬝ᵥ (G *ᵥ (θ - θstar)) = 0 := by
    have h := heq.symm.trans hexpand
    linarith
  have hdzero : θ - θstar = 0 := by
    by_contra hne
    have hpos : 0 < (θ - θstar) ⬝ᵥ (G *ᵥ (θ - θstar)) := by simpa using hPD.2 (θ - θstar) hne
    linarith
  exact sub_eq_zero.mp hdzero

/-!
## Theorem 7 (paper §5.1) — TSK reproduces LQR exactly

"The proof is `∑_i φ_i ≡ 1`." Stated generally: if every rule's consequent
equals the *same* value `v` (e.g. `v = -Kx`, the LQR law), then the TSK output
— any partition-of-unity-weighted combination of that value — equals `v`
exactly, for any partition of unity and any module (scalar or vector control).
-/

theorem tsk_constant_consequent_reproduces
    {n : ℕ} {M : Type*} [AddCommMonoid M] [Module ℝ M]
    (φ : Fin n → ℝ) (hpou : ∑ i, φ i = 1) (v : M) :
    ∑ i, φ i • v = v := by
  rw [← Finset.sum_smul, hpou, one_smul]

/-- Scalar specialization matching the paper's `u(x) = Σ φ_i(x) f_i(x)`,
`f_i(x) = -Kx` for every `i`. -/
theorem tsk_reproduces_lqr_scalar
    {n : ℕ} (φ : Fin n → ℝ) (hpou : ∑ i, φ i = 1) (K x : ℝ) :
    ∑ i, φ i * (-(K * x)) = -(K * x) := by
  simpa [smul_eq_mul] using tsk_constant_consequent_reproduces φ hpou (-(K * x))

/-!
## Corollary 6 (paper §4.2) — the factor-of-4 obstruction

Theorem 5 gives `λ* ∝ c⁻²` where `c` is (half) the rule separation. Corollary 6
observes that adjacent pairs (separation `d`) and next-nearest pairs
(separation `2d`) therefore demand orthogonalizing weights in ratio
`(2d/d)² = 4`, so a single scalar `λ` cannot orthogonalize both simultaneously.
The entire content is: for `K, d > 0`, `K/d² ≠ K/(2d)²`.
-/

theorem lambda_star_ratio_is_four (K d : ℝ) (hK : 0 < K) (hd : 0 < d) :
    (K / d ^ 2) / (K / (2 * d) ^ 2) = 4 := by
  have hd2 : d ^ 2 ≠ 0 := by positivity
  have h2d2 : (2 * d) ^ 2 ≠ 0 := by positivity
  field_simp
  ring

theorem lambda_star_incompatible_for_three_rules (K d : ℝ) (hK : 0 < K) (hd : 0 < d) :
    K / d ^ 2 ≠ K / (2 * d) ^ 2 := by
  intro heq
  have hd2 : d ^ 2 ≠ 0 := by positivity
  have h2d2 : (2 * d) ^ 2 ≠ 0 := by positivity
  rw [div_eq_div_iff hd2 h2d2] at heq
  nlinarith [mul_pos hK (mul_pos hd hd)]

/-!
## Theorem 8 (paper §5.2) — algebraic core of the Bregman suboptimality identity

This is the pointwise HJB algebra underlying Theorem 8: given HJB stationarity
`c'(u*) = -p` (with `p := ∇V*·g` folded to a scalar for a scalar control) and
the HJB equation itself `q + a + p u* + c(u*) = 0` (with `a := ∇V*·f`), the
running cost plus the time-derivative of `V*` along the closed loop equals
exactly the Bregman divergence `D_c(u‖u*)`.

What this does *not* cover (deliberately, matching the paper's own §11 Tier-3
admission): integrating this identity from `0` to `∞` along an actual closed-
loop trajectory and using admissibility to discard the boundary term. That
step needs ODE trajectory existence/limit machinery and is not attempted here
— only the algebraic identity that step relies on, at a single instant, is
verified.
-/

/-- The Bregman divergence of a differentiable `c`, exactly as in the paper. -/
noncomputable def bregman (c : ℝ → ℝ) (u v : ℝ) : ℝ :=
  c u - c v - deriv c v * (u - v)

theorem bregman_identity_at_stationary_point (c : ℝ → ℝ) (a p q ustar u : ℝ)
    (hHJB : q + a + p * ustar + c ustar = 0) (hstat : deriv c ustar = -p) :
    q + c u + (a + p * u) = bregman c u ustar := by
  have hq : q = -(a + p * ustar + c ustar) := by linarith
  unfold bregman
  rw [hstat, hq]
  ring

/-- Corollary 9's quadratic specialization: for `c(u) = r u²` the Bregman
divergence collapses to `r (u - u*)²`, matching the "familiar" identity the
paper contrasts Theorem 8 against. -/
theorem bregman_quadratic (r u v : ℝ) :
    bregman (fun u => r * u ^ 2) u v = r * (u - v) ^ 2 := by
  unfold bregman
  have hderiv : deriv (fun u : ℝ => r * u ^ 2) v = 2 * r * v := by
    have h1 : HasDerivAt (fun u : ℝ => u ^ 2) (2 * v) v := by
      simpa using (hasDerivAt_pow 2 v)
    have h2 : HasDerivAt (fun u : ℝ => r * u ^ 2) (r * (2 * v)) v := h1.const_mul r
    rw [h2.deriv]
    ring
  simp only [hderiv]
  ring

end FuzzySymphony
