# Lean/Mathlib verification of the tractable theorems

This is a minimal Lean 4 + Mathlib project that machine-checks the parts of
`papers/least-action/paper.md` whose entire content is finite-dimensional
algebra or single-variable calculus. See `FuzzySymphonyProofs/Basic.lean` for
the proofs and `../REVIEW.md` §3 for what is and isn't covered.

## What's proved (all compile with zero `sorry`, axioms = the Lean/Mathlib
standard three: `propext`, `Classical.choice`, `Quot.sound`)

- `variable_projection_global_min` / `variable_projection_unique` — **Theorem
  1** (§3.1): the closed-form consequent solve is the global minimizer of the
  action, uniquely so when the Gram matrix is positive definite.
- `tsk_reproduces_lqr_scalar` (and the general `tsk_constant_consequent_reproduces`)
  — **Theorem 7** (§5.1): TSK reproduces LQR exactly.
- `lambda_star_incompatible_for_three_rules` — **Corollary 6** (§4.2): the
  factor-of-4 obstruction to a single orthogonalizing λ for 3+ rules.
- `bregman_identity_at_stationary_point` / `bregman_quadratic` — the algebraic
  core of **Theorem 8** (§5.2): the pointwise HJB algebra that produces the
  Bregman suboptimality identity, plus its quadratic specialization
  (Corollary 9).

## Building

```
lake exe cache get   # tries to fetch prebuilt Mathlib oleans (fast if it works)
lake build            # falls back to building from source (~30-40 min on 4 cores)
```

Note: in the environment this was developed in, `lake exe cache get` failed
because its target host (`lakecache.blob.core.windows.net`, Azure blob
storage) was blocked by the egress proxy, while `github.com` and
`raw.githubusercontent.com` were reachable. If your environment has the same
restriction, skip straight to `lake build` — it only needs to compile the
narrow import closure of `Mathlib.LinearAlgebra.Matrix.PosDef` and
`Mathlib.Analysis.Calculus.Deriv.{Pow,Mul}`, not all of Mathlib, but that
closure still touches a large chunk of the algebra hierarchy so it takes a
while the first time.

`lake-manifest.json` pins the exact dependency commits used (Mathlib
`v4.14.0-patch1`) for reproducibility.
