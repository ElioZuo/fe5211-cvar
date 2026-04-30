"""Test unsmoothing methods on synthetic Geltner-smoothed data."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.unsmoothing import fisher_geltner_webb, kalman_filter, okunev_white


def _make_smoothed(alpha: float, n: int, sigma_true: float, seed: int = 7):
    """Generate true returns r_t ~ N(0, sigma_true^2) and Geltner-smoothed r*_t."""
    rng = np.random.default_rng(seed)
    r_true = rng.normal(0, sigma_true, n)
    r_smoothed = np.zeros(n)
    r_smoothed[0] = (1 - alpha) * r_true[0]
    for t in range(1, n):
        r_smoothed[t] = alpha * r_smoothed[t - 1] + (1 - alpha) * r_true[t]
    idx = pd.date_range("2000-01-01", periods=n, freq="QE")
    return (pd.Series(r_true, index=idx, name="true"),
            pd.Series(r_smoothed, index=idx, name="smoothed"))


def test_fgw_recovers_alpha():
    """FGW alpha estimate should match true alpha within 0.10 on a long sample."""
    true, smooth = _make_smoothed(alpha=0.7, n=400, sigma_true=0.04)
    res = fisher_geltner_webb(smooth)
    assert abs(res.params["alpha"] - 0.7) < 0.10, \
        f"FGW alpha = {res.params['alpha']:.3f}, expected ~0.7"


def test_fgw_recovers_volatility():
    """Unsmoothed vol should be within 20% of true vol on a long sample."""
    true, smooth = _make_smoothed(alpha=0.7, n=400, sigma_true=0.04)
    res = fisher_geltner_webb(smooth)
    ratio = res.series.std() / true.std()
    assert 0.80 < ratio < 1.20, \
        f"Unsmoothed vol ratio = {ratio:.3f}, expected ~1.0"


def test_kf_alpha_in_range():
    """KF alpha should be plausible on real PE-like AR(1) ~ 0.7."""
    true, smooth = _make_smoothed(alpha=0.7, n=200, sigma_true=0.04)
    res = kalman_filter(smooth)
    assert 0.30 < res.params["alpha"] < 0.95, \
        f"KF alpha = {res.params['alpha']:.3f}, out of plausible range"


def test_ow_q_selection():
    """OW should pick q = 1 for AR(1) data (no higher-order smoothing)."""
    true, smooth = _make_smoothed(alpha=0.7, n=400, sigma_true=0.04)
    res = okunev_white(smooth, max_q=3)
    assert res.params["q"] == 1, f"OW picked q = {res.params['q']}, expected 1"


def test_unsmoothed_vol_increases():
    """Unsmoothing must inflate volatility (the whole point)."""
    true, smooth = _make_smoothed(alpha=0.7, n=200, sigma_true=0.04)
    for fn in (fisher_geltner_webb, kalman_filter, okunev_white):
        res = fn(smooth)
        assert res.unsmoothed_vol_ann > res.raw_vol_ann, \
            f"{res.method}: unsmoothed vol {res.unsmoothed_vol_ann:.4f} " \
            f"<= raw vol {res.raw_vol_ann:.4f}"
