"""Test copula models: estimator consistency and analytic identities."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats
from itertools import combinations

from src.copula import (kendall_tau_matrix, tau_to_rho, nearest_pd,
                         GaussianCopula, StudentTCopula, vuong_test)


def test_tau_to_rho_identity():
    """For independent uniforms, tau and rho should both be near zero."""
    rng = np.random.default_rng(0)
    u = rng.random((1000, 3))
    T = kendall_tau_matrix(u)
    R = tau_to_rho(T)
    off = R - np.eye(3)
    assert np.max(np.abs(off)) < 0.10


def test_nearest_pd_keeps_pd():
    """Nearest-PD should be a fixed point on PD inputs."""
    R = np.array([[1.0, 0.3, 0.2],
                  [0.3, 1.0, 0.4],
                  [0.2, 0.4, 1.0]])
    R_pd = nearest_pd(R)
    assert np.allclose(R, R_pd, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(R_pd) > 0)


def test_nearest_pd_fixes_non_pd():
    """A non-PD matrix should be projected onto a PD matrix."""
    R = np.array([[1.0, 0.95, 0.95],
                  [0.95, 1.0, -0.95],
                  [0.95, -0.95, 1.0]])
    R_pd = nearest_pd(R)
    assert np.all(np.linalg.eigvalsh(R_pd) > 0)


def test_gaussian_copula_simulate_correlation():
    """Simulated samples should reproduce the input correlation within MC error."""
    R = np.array([[1.0, 0.5, 0.3],
                  [0.5, 1.0, 0.2],
                  [0.3, 0.2, 1.0]])
    g = GaussianCopula(); g.D = 3; g.R = R
    g._L = np.linalg.cholesky(R); g._Rinv = np.linalg.inv(R)
    g._logdet = np.linalg.slogdet(R)[1]
    u = g.simulate(50_000, seed=1)
    # Recover Pearson R via inverse-CDF
    z = stats.norm.ppf(u)
    R_emp = np.corrcoef(z, rowvar=False)
    err = np.max(np.abs(R_emp - R))
    assert err < 0.03, f"max R error = {err:.3f}"


def test_studentt_recovers_nu():
    """Student-t copula with true nu = 5 should recover nu within +/- 1.5."""
    rng = np.random.default_rng(2)
    n = 1500; nu_true = 5.0
    R = np.array([[1.0, 0.4, 0.3],
                  [0.4, 1.0, 0.2],
                  [0.3, 0.2, 1.0]])
    L = np.linalg.cholesky(R)
    z = rng.standard_normal((n, 3)) @ L.T
    chi2 = rng.chisquare(nu_true, n)
    x = z / np.sqrt(chi2 / nu_true)[:, None]
    u = stats.t.cdf(x, df=nu_true)

    fit = StudentTCopula().fit(u, R_init=R)
    assert abs(fit.nu - nu_true) < 1.5, \
        f"recovered nu = {fit.nu:.2f}, expected ~{nu_true}"


def test_studentt_lambda_L_positive():
    """Student-t implied lambda_L must be strictly positive for nonzero rho, nu < inf."""
    R = np.array([[1.0, 0.4], [0.4, 1.0]])
    t = StudentTCopula(); t.D = 2; t.R = R; t.nu = 5.0
    t._L = np.linalg.cholesky(R); t._Rinv = np.linalg.inv(R)
    t._logdet = np.linalg.slogdet(R)[1]
    L = t.lambda_L()
    assert L[0, 1] > 0
    assert L[0, 0] == 1.0


def test_vuong_test_sane():
    """Vuong of effectively equivalent log-liks (independent noise around equal mean)
    should give a non-significant Z-stat."""
    rng = np.random.default_rng(3)
    n = 100
    ll1 = rng.normal(0, 1, n)
    ll2 = rng.normal(0, 1, n)
    z, p = vuong_test(ll1, ll2, k1=5, k2=5)
    assert abs(z) < 3.0
    assert p > 0.01
