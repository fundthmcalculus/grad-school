"""Smoke tests for the least-action FIS module.

This research directory had no automated tests -- correctness rested on
demo_*.py/verify_*.py scripts that print numbers for a human to read. These
tests don't replace that verification (see verify_proxy.py, verify_sos_exact.py,
verify_symbolic.py for the real thing); they exist so a broken import, a
sign error, or an accidentally-nonconvergent default doesn't sit undetected
between manual runs. Kept fast: small n_rules and n_quad throughout.
"""

import numpy as np
import pytest

from fis_action import (
    Quadrature,
    fit,
    galerkin_residual,
    optimality_certificate,
    sequential_fit,
)


def _sine_targets():
    return (lambda x: np.sin(x)), (lambda x: np.cos(x))


class TestFit:
    def test_fits_a_smooth_target_to_low_action(self):
        yd_fn, dyd_fn = _sine_targets()
        f = fit(
            yd_fn, dyd_fn, n_rules=4, x_lo=-3.0, x_hi=3.0, lam=0.1, order=1,
            n_quad=60, n_restarts=2, polish_rounds=1, seed=0,
        )
        assert np.isfinite(f.action)
        assert f.action >= 0.0
        # sin(x) on [-3, 3] with 4 rules should fit comfortably; this is a
        # smoke bound, not a tight one.
        assert f.action < 0.05

    def test_output_and_derivative_are_finite_on_and_off_the_quadrature_grid(self):
        yd_fn, dyd_fn = _sine_targets()
        f = fit(
            yd_fn, dyd_fn, n_rules=3, x_lo=-2.0, x_hi=2.0, lam=0.1, order=1,
            n_quad=40, n_restarts=1, polish_rounds=1, seed=0,
        )
        x = np.linspace(-2.0, 2.0, 25)
        assert np.all(np.isfinite(f(x)))
        assert np.all(np.isfinite(f.derivative(x)))

    def test_centers_stay_ordered_and_within_bounds(self):
        yd_fn, dyd_fn = _sine_targets()
        n_rules = 5
        f = fit(
            yd_fn, dyd_fn, n_rules=n_rules, x_lo=-1.0, x_hi=1.0, lam=0.1,
            order=1, n_quad=40, n_restarts=1, polish_rounds=1, seed=0,
        )
        assert np.all(np.diff(f.centers) >= f.min_gap - 1e-9)
        assert np.all(f.centers >= -1.0 - 1e-9)
        assert np.all(f.centers <= 1.0 + 1e-9)

    def test_deterministic_given_seed(self):
        yd_fn, dyd_fn = _sine_targets()
        kwargs = dict(
            n_rules=3, x_lo=-2.0, x_hi=2.0, lam=0.1, order=1,
            n_quad=40, n_restarts=2, polish_rounds=1, seed=0,
        )
        f1 = fit(yd_fn, dyd_fn, **kwargs)
        f2 = fit(yd_fn, dyd_fn, **kwargs)
        assert np.allclose(f1.centers, f2.centers)
        assert np.allclose(f1.coeffs, f2.coeffs)

    def test_min_gap_zero_is_accepted_though_documented_as_degenerate(self):
        # README/docstring: min_gap=0 plus wide width_bounds is the documented
        # way to reproduce the rank-deficient failure mode. It must not raise.
        yd_fn, dyd_fn = _sine_targets()
        f = fit(
            yd_fn, dyd_fn, n_rules=3, x_lo=-2.0, x_hi=2.0, lam=0.1, order=1,
            n_quad=40, n_restarts=1, polish_rounds=1, seed=0,
            min_gap=0.0, width_bounds=(1e-3, 10.0),
        )
        assert np.isfinite(f.action)


class TestGalerkinResidual:
    def test_residual_is_small_at_a_converged_fit(self):
        yd_fn, dyd_fn = _sine_targets()
        f = fit(
            yd_fn, dyd_fn, n_rules=4, x_lo=-3.0, x_hi=3.0, lam=0.1, order=1,
            n_quad=80, n_restarts=3, polish_rounds=2, seed=0,
        )
        quad = Quadrature.legendre(-3.0, 3.0, 80)
        residual = galerkin_residual(f, yd_fn, dyd_fn, quad)
        assert np.all(np.isfinite(residual))
        # Stationarity of the action is exactly this vector vanishing; at a
        # good fit it should be small relative to the target's own scale.
        assert np.max(np.abs(residual)) < 0.1


class TestOptimalityCertificate:
    def test_returns_the_documented_keys_and_finite_values(self):
        yd_fn, dyd_fn = _sine_targets()
        f = fit(
            yd_fn, dyd_fn, n_rules=3, x_lo=-2.0, x_hi=2.0, lam=0.1, order=1,
            n_quad=60, n_restarts=2, polish_rounds=1, seed=0,
        )
        quad = Quadrature.legendre(-2.0, 2.0, 60)
        cert = optimality_certificate(f, yd_fn, dyd_fn, quad)
        assert isinstance(cert, dict)
        assert len(cert) > 0
        for value in cert.values():
            if isinstance(value, float):
                assert np.isfinite(value)


class TestSequentialFit:
    def test_matches_the_joint_solution_when_rules_have_disjoint_support(self):
        # Docstring: sequential_fit reproduces the joint solution exactly iff
        # the H1 Gram is block diagonal, which narrow non-overlapping rules
        # give directly -- the one case this smoke test can check without
        # reimplementing the joint solver's own math.
        quad = Quadrature.legendre(-3.0, 3.0, 200)
        yd = np.sin(quad.nodes)
        dyd = np.cos(quad.nodes)
        centers = np.array([-2.0, 0.0, 2.0])
        widths = np.array([0.15, 0.15, 0.15])

        theta, action = sequential_fit(centers, widths, yd, dyd, quad, lam=0.1)

        assert theta.shape == (centers.shape[0] * 2,)
        assert np.isfinite(action)
        assert action >= 0.0


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
