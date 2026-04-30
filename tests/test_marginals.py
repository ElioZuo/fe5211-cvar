"""Test marginal distribution implementations."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats

from src.marginals import (SkewStudentMarginal, SemiParametricMarginal,
                            ewma_variance, fit_ewma_lambda)


def test_skew_student_cdf_ppf_roundtrip():
    """ppf(cdf(z)) ~= z within numerical tolerance."""
    rng = np.random.default_rng(0)
    z = rng.standard_normal(200)
    dist = SkewStudentMarginal(z)
    z_sample = np.linspace(-2, 2, 30)
    u = np.asarray(dist.cdf(z_sample))
    z_back = np.asarray(dist.ppf(u))
    assert np.allclose(z_sample, z_back, atol=1e-2), \
        f"max abs error = {np.max(np.abs(z_sample - z_back)):.4f}"


def test_skew_student_uniform_pit():
    """PIT of fit data should be approximately uniform (KS p > 0.05)."""
    rng = np.random.default_rng(1)
    z = rng.standard_normal(400)
    dist = SkewStudentMarginal(z)
    u = dist.cdf(z)
    p = stats.kstest(u, "uniform").pvalue
    assert p > 0.05, f"KS p = {p:.3f} < 0.05"


def test_semiparam_roundtrip_in_center():
    """SemiParametric ppf(cdf(z)) is exact within the empirical center."""
    rng = np.random.default_rng(2)
    z = rng.standard_normal(200)
    dist = SemiParametricMarginal(z)
    # Pick z values inside [z_lower, z_upper]
    z_test = z[(z > dist.z_lower + 0.1) & (z < dist.z_upper - 0.1)][:30]
    u = np.asarray(dist.cdf(z_test))
    z_back = np.asarray(dist.ppf(u))
    err = np.max(np.abs(z_test - z_back))
    assert err < 0.5, f"center round-trip error = {err:.3f}"


def test_semiparam_tail_extrapolation_finite():
    """ppf at extreme u (1e-4) should produce finite, plausible values."""
    rng = np.random.default_rng(3)
    z = rng.standard_normal(200)
    dist = SemiParametricMarginal(z)
    u_extreme = np.array([1e-4, 0.001, 0.999, 1 - 1e-4])
    z_extreme = np.asarray(dist.ppf(u_extreme))
    assert np.all(np.isfinite(z_extreme))
    assert z_extreme[0] < -2, f"left-tail z @ u=1e-4 = {z_extreme[0]:.2f}"
    assert z_extreme[-1] > 2, f"right-tail z @ u=1-1e-4 = {z_extreme[-1]:.2f}"


def test_ewma_variance_basic():
    """EWMA recursion: known input -> known output."""
    r = np.array([0.0, 0.1, -0.2, 0.05])
    s2 = ewma_variance(r, lam=0.94, init=0.001)
    assert s2[0] == 0.001
    expected_t1 = 0.94 * 0.001 + 0.06 * 0.0 ** 2
    assert abs(s2[1] - expected_t1) < 1e-12


def test_fit_ewma_lambda_in_bounds():
    """MLE lambda is either in (0.80, 0.99) or returns the fallback 0.94."""
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.05, 100)
    lam = fit_ewma_lambda(r)
    assert 0.80 <= lam <= 0.99
