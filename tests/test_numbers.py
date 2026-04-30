"""Strict integration sanity check.

Reads output/numbers.json (produced by `python main.py`) and asserts every
canonical number is reproduced within tiered tolerances:
    - parameters and AR coefficients:   atol = 1e-3
    - Monte-Carlo ES values:            atol = 0.30 percentage points
    - bootstrap percentile CI bounds:   atol = 0.50 percentage points

Run with:
    pytest tests/test_numbers.py -v
"""

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NUMBERS_FILE = ROOT / "output" / "numbers.json"


# ---- Tolerances ----
ATOL_PARAM   = 1e-3       # for parameters (e.g., AR1, lambda, R, eta)
ATOL_VOL     = 0.20       # 0.20 percentage points (e.g., ann_vol_SPY in %)
ATOL_ES      = 0.30       # 0.30 percentage points for ES (Monte-Carlo)
ATOL_CI      = 0.50       # 0.50 percentage points for bootstrap CI bounds


# ---- Canonical values ----
EXPECTED_PARAMS = {
    # AR(1) on log returns (data-only, deterministic)
    "ar1_SPY":  (0.038, ATOL_PARAM),
    "ar1_AGG":  (0.056, ATOL_PARAM),
    "ar1_PE":   (0.773, ATOL_PARAM),
    "ar1_NPI":  (0.901, ATOL_PARAM),
    # Annualized vols (deterministic)
    "ann_vol_SPY": (16.11, ATOL_VOL),
    "ann_vol_AGG": (4.58,  ATOL_VOL),
    "ann_vol_PE":  (8.42,  ATOL_VOL),
    "ann_vol_NPI": (4.61,  ATOL_VOL),
    # Kalman-filter unsmoothing (PE/NPI)
    "pe_kf_alpha":  (0.5446, 0.005),
    "pe_kf_ma1":    (0.7781, 0.01),
    "npi_kf_alpha": (0.8432, 0.01),
    "npi_kf_ma1":   (0.4257, 0.02),
    # Marginal Skew-t parameters for SPY/AGG
    "spy_eta":      (4.67, 0.10),
    "spy_lambda_sk":(-0.52, 0.05),
    "agg_eta":      (8.68, 0.40),
    "agg_lambda_sk":(-0.19, 0.05),
    # PIT KS p-values must all be > 0.50 (well above 0.05 threshold)
    "ks_p_SPY":   (0.95, 0.10),
    "ks_p_AGG":   (0.93, 0.10),
    "ks_p_PE":    (0.99, 0.05),
    "ks_p_NPI":   (0.99, 0.05),
}

EXPECTED_COPULA = {
    "champion":  ("Student-t", None),
    "t_nu":      (4.443, 0.10),
    "R_SPY_PE":  (0.292, 0.02),
    "R_PE_NPI":  (0.379, 0.02),
    "R_AGG_PE":  (0.024, 0.05),
    "R_SPY_NPI": (-0.089, 0.05),
}

EXPECTED_ES = {
    # ES matrix, in % (Monte-Carlo MC noise ~0.05% at N=1M, tolerate 0.30)
    "es_param_reported":   (8.81,  0.05),
    "es_param_unsmoothed": (15.22, 0.05),
    "es_hs_reported":      (22.71, ATOL_ES),
    "es_hs_unsmoothed":    (33.04, ATOL_ES),
    "es_fhs_unsmoothed":   (33.05, ATOL_ES),
    "es_champion":         (18.73, ATOL_ES),
    "var_champion":        (8.10,  ATOL_ES),
}

EXPECTED_DECOMP = {
    "comp_SPY_pct": (87.0, 1.0),
    "comp_AGG_pct": (-12.0, 1.0),
    "comp_PE_pct":  (19.0, 1.0),
    "comp_NPI_pct": (6.0,  1.0),
}

EXPECTED_REPLAY = {
    "replay_GFC_loss":     (11.56, 0.10),
    "replay_GFC_pct":      (96.4,  0.5),
    "replay_COVID_loss":   (-20.51, 0.10),  # 3Y window 2019Q4-2022Q3 was a gain
    "replay_Worst#1_loss": (23.11, 0.10),
    "replay_Worst#1_pct":  (98.8,  0.5),
}

EXPECTED_BACKTEST = {
    "backtest_n_breach":   (8,     0.5),    # integer, but tolerate +/- 0
    "backtest_kupiec_p":   (0.069, 0.005),
    "backtest_chris_p":    (0.002, 0.005),
    "backtest_acerbi_z":   (0.098, 0.05),
}

EXPECTED_REVERSE = {
    "reverse_SPY_loss": (105.5, 5.0),    # large numbers, larger tolerance
    "reverse_AGG_loss": (-10.4, 3.0),
    "reverse_PE_loss":  (37.8,  3.0),
    "reverse_NPI_loss": (13.1,  3.0),
}

EXPECTED_BOOTSTRAP = {
    "tier_a_es_ci_lo":      (18.59, 0.10),
    "tier_a_es_ci_hi":      (18.87, 0.10),
    "tier_b_es_pct_ci_lo":  (12.40, ATOL_CI),
    "tier_b_es_pct_ci_hi":  (19.82, ATOL_CI),
    "tier_b_nu_mean":       (5.21, 0.50),
}


# ---- Loader fixture ----
@pytest.fixture(scope="module")
def numbers():
    if not NUMBERS_FILE.exists():
        pytest.skip(f"numbers.json not found. Run `python main.py` first.")
    with open(NUMBERS_FILE) as f:
        return json.load(f)


# ---- Assertion helper ----
def _assert_close(numbers, key, expected, tol):
    assert key in numbers, f"missing key: {key}"
    actual = numbers[key]
    if expected is None:
        return
    if isinstance(expected, str):
        assert actual == expected, f"{key}: got {actual!r}, expected {expected!r}"
        return
    assert math.isfinite(actual), f"{key}: non-finite value {actual!r}"
    assert abs(actual - expected) <= tol, \
        f"{key}: {actual:.4f} differs from canonical {expected:.4f} by " \
        f"{abs(actual - expected):.4f} > tol {tol:.4f}"


# ---- Tests ----
@pytest.mark.parametrize("key,target", list(EXPECTED_PARAMS.items()))
def test_parameters(numbers, key, target):
    expected, tol = target
    _assert_close(numbers, key, expected, tol)


@pytest.mark.parametrize("key,target", list(EXPECTED_COPULA.items()))
def test_copula(numbers, key, target):
    expected, tol = target
    _assert_close(numbers, key, expected, tol)


@pytest.mark.parametrize("key,target", list(EXPECTED_ES.items()))
def test_es_matrix(numbers, key, target):
    expected, tol = target
    _assert_close(numbers, key, expected, tol)


@pytest.mark.parametrize("key,target", list(EXPECTED_DECOMP.items()))
def test_decomposition(numbers, key, target):
    expected, tol = target
    _assert_close(numbers, key, expected, tol)


@pytest.mark.parametrize("key,target", list(EXPECTED_REPLAY.items()))
def test_historical_replay(numbers, key, target):
    expected, tol = target
    _assert_close(numbers, key, expected, tol)


@pytest.mark.parametrize("key,target", list(EXPECTED_BACKTEST.items()))
def test_backtest(numbers, key, target):
    expected, tol = target
    _assert_close(numbers, key, expected, tol)


@pytest.mark.parametrize("key,target", list(EXPECTED_REVERSE.items()))
def test_reverse_stress(numbers, key, target):
    expected, tol = target
    _assert_close(numbers, key, expected, tol)


@pytest.mark.parametrize("key,target", list(EXPECTED_BOOTSTRAP.items()))
def test_bootstrap_cis(numbers, key, target):
    expected, tol = target
    _assert_close(numbers, key, expected, tol)


def test_total_keys_present(numbers):
    """At least 50 numbers should be tracked."""
    assert len(numbers) >= 50, \
        f"only {len(numbers)} keys in numbers.json (expected >= 50)"
