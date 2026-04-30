"""Test the simulation module: closed-form vs analytic and MC convergence."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src import config as cfg
from src.simulation import (var_es, parametric_es, historical_simulation,
                              stationary_block_indices)


def test_var_es_known_distribution():
    """For Gaussian losses, ES should match analytic phi(z)/(1-alpha)."""
    rng = np.random.default_rng(0)
    losses = rng.normal(0, 1, 100_000)
    var, es = var_es(losses, alpha=0.95)
    z_a = stats.norm.ppf(0.95)
    es_analytic = stats.norm.pdf(z_a) / (1 - 0.95)
    assert abs(es - es_analytic) < 0.05, \
        f"ES = {es:.4f}, analytic = {es_analytic:.4f}"


def test_parametric_es_matches_formula():
    """Parametric ES on identity-cov data should equal sqrt(12) * phi(z)/(1-a)
    minus drift, within numerical tolerance."""
    rng = np.random.default_rng(0)
    n_obs = 200
    cov_target = np.eye(4) * 0.01   # var per asset = 0.01
    L = np.linalg.cholesky(cov_target)
    panel = pd.DataFrame(rng.standard_normal((n_obs, 4)) @ L.T,
                          columns=cfg.ASSETS,
                          index=pd.date_range("2000-01-01", periods=n_obs, freq="QE"))
    var, es = parametric_es(panel)
    weights = np.array(cfg.WEIGHTS)
    sigma_p_q = float(np.sqrt(weights @ cov_target @ weights))
    sigma_total = sigma_p_q * np.sqrt(cfg.HORIZON_QUARTERS)
    mu_q = np.log(1 + np.array(cfg.EXPECTED_RETURNS)) / cfg.QUARTERS_PER_YEAR
    mu_total = cfg.HORIZON_QUARTERS * float(weights @ mu_q)
    es_expected = -mu_total + (stats.norm.pdf(stats.norm.ppf(cfg.ALPHA)) /
                                (1 - cfg.ALPHA)) * sigma_total
    assert abs(es - es_expected) < 0.01, \
        f"ES = {es:.4f}, expected = {es_expected:.4f}"


def test_stationary_block_indices_shape_and_range():
    """Indices should be valid, non-trivial, and respect the wrap."""
    idx = stationary_block_indices(n_obs=80, horizon=12, n_paths=1000,
                                     block_size=4, seed=0)
    assert idx.shape == (1000, 12)
    assert idx.min() >= 0 and idx.max() < 80
    # Non-trivial: not all paths same starting index
    assert len(np.unique(idx[:, 0])) > 10


def test_historical_simulation_runs():
    """Smoke: HS should produce a finite ES on synthetic data."""
    rng = np.random.default_rng(0)
    panel = pd.DataFrame(rng.normal(0, 0.05, (80, 4)),
                          columns=cfg.ASSETS,
                          index=pd.date_range("2000-01-01", periods=80, freq="QE"))
    var, es, losses = historical_simulation(panel, n_paths=10_000, seed=1)
    assert np.isfinite(var) and np.isfinite(es)
    assert es >= var
    assert losses.shape == (10_000,)


def test_es_mc_convergence():
    """ES at N=1M should converge to analytic Gaussian ES within 0.5%."""
    rng = np.random.default_rng(0)
    losses = rng.normal(0.5, 1, 1_000_000)
    var, es = var_es(losses, alpha=0.95)
    z_a = stats.norm.ppf(0.95)
    es_analytic = 0.5 + stats.norm.pdf(z_a) / (1 - 0.95)
    assert abs(es - es_analytic) < 0.005, \
        f"ES @ 1M = {es:.4f}, analytic = {es_analytic:.4f}"
