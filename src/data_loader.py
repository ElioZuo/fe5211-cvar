"""Load and preprocess the raw data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg


def load_panel() -> pd.DataFrame:
    """Read the raw quarterly level panel from data.xlsx."""
    df = pd.read_excel(cfg.DATA_FILE, sheet_name="quarterly_levels")
    df = df.set_index("QuarterEnd")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def compute_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Log returns for level series, first differences for rate / spread series."""
    log_part  = np.log(panel[cfg.LEVEL_COLUMNS] / panel[cfg.LEVEL_COLUMNS].shift(1))
    diff_part = panel[cfg.DIFF_COLUMNS].diff()
    out = pd.concat([log_part, diff_part], axis=1).iloc[1:]
    return out


def descriptive_stats(returns: pd.DataFrame, columns: list[str] | None = None,
                      freq: int = 4) -> pd.DataFrame:
    """Per-asset summary: annualized mean / vol, skew, kurt, AR(1)."""
    from scipy import stats as sc
    if columns is None:
        columns = [c for c in cfg.ASSETS if c in returns.columns]
    rows = {}
    for c in columns:
        s = returns[c].dropna()
        rows[c] = {
            "n":          len(s),
            "ann_ret":    (np.exp(s.mean() * freq) - 1) * 100,
            "ann_vol":    s.std() * np.sqrt(freq) * 100,
            "skew":       float(sc.skew(s)),
            "ex_kurt":    float(sc.kurtosis(s)),
            "min_q":      s.min() * 100,
            "max_q":      s.max() * 100,
            "ar1":        s.autocorr(lag=1),
        }
    return pd.DataFrame(rows).T


def empirical_chi(x: np.ndarray, y: np.ndarray, q: float, tail: str) -> float:
    """Empirical tail dependence at quantile q. tail in {'lower','upper'}."""
    n = len(x)
    rx = (np.argsort(np.argsort(x)) + 1) / (n + 1)
    ry = (np.argsort(np.argsort(y)) + 1) / (n + 1)
    if tail == "lower":
        cond = rx < q
        if cond.sum() == 0:
            return float("nan")
        return float(((rx < q) & (ry < q)).sum() / cond.sum())
    elif tail == "upper":
        cond = rx > 1 - q
        if cond.sum() == 0:
            return float("nan")
        return float(((rx > 1 - q) & (ry > 1 - q)).sum() / cond.sum())
    raise ValueError(tail)


def build_or_load_returns(force: bool = False) -> pd.DataFrame:
    """Build returns table; cache to parquet."""
    cache = cfg.CACHE_DIR / "panel.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    panel = load_panel()
    returns = compute_returns(panel)
    returns.to_parquet(cache)
    return returns
