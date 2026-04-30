"""Three unsmoothing methods for appraisal-based private-asset returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd

from . import config as cfg


@dataclass
class UnsmoothResult:
    series:   pd.Series
    method:   str
    params:   dict
    raw_vol_ann:        float
    unsmoothed_vol_ann: float


def _ann(s: pd.Series) -> float:
    return float(s.std() * np.sqrt(cfg.QUARTERS_PER_YEAR))


def fisher_geltner_webb(reported: pd.Series) -> UnsmoothResult:
    """FGW (1994): r_t = (r*_t - alpha r*_{t-1}) / (1 - alpha)."""
    s = reported.dropna()
    alpha = float(s.autocorr(lag=1))
    r = (s - alpha * s.shift(1)) / (1 - alpha)
    r = r.dropna()
    return UnsmoothResult(r, "FGW", {"alpha": alpha},
                          _ann(s), _ann(r))


def okunev_white(reported: pd.Series, max_q: int = 3) -> UnsmoothResult:
    """OW (2003): pick AR(q) by BIC, invert."""
    from statsmodels.tsa.arima.model import ARIMA
    s = reported.dropna()
    bics = {}
    for q in range(1, max_q + 1):
        try:
            bics[q] = ARIMA(s, order=(q, 0, 0)).fit().bic
        except Exception:
            bics[q] = np.inf
    q_opt = min(bics, key=bics.get)
    m = ARIMA(s, order=(q_opt, 0, 0)).fit()
    phi = np.asarray(m.arparams)
    sum_phi = float(phi.sum())
    r = s.copy()
    for i, p in enumerate(phi, start=1):
        r = r - p * s.shift(i)
    r = (r / (1 - sum_phi)).dropna()
    return UnsmoothResult(r, "OW",
                          {"phi": phi.tolist(), "q": q_opt, "sum_phi": sum_phi},
                          _ann(s), _ann(r))


def kalman_filter(reported: pd.Series) -> UnsmoothResult:
    """State-space inversion via ARMA(1,1) parameterization (equivalent
    to Geltner SSM under Gaussian innovations)."""
    from statsmodels.tsa.arima.model import ARIMA
    s = reported.dropna()
    m = ARIMA(s, order=(1, 0, 1)).fit()
    alpha_kf = float(m.arparams[0]) if len(m.arparams) > 0 else 0.0
    alpha_kf = float(np.clip(alpha_kf, -0.99, 0.99))
    ma1 = float(m.maparams[0]) if len(m.maparams) > 0 else 0.0
    r = ((s - alpha_kf * s.shift(1)) / (1 - alpha_kf)).dropna()
    return UnsmoothResult(r, "KF",
                          {"alpha": alpha_kf, "ma1": ma1, "loglik": float(m.llf)},
                          _ann(s), _ann(r))


def unsmooth_all(reported: pd.Series) -> dict[str, UnsmoothResult]:
    """Run all three methods on one series."""
    return {
        "FGW": fisher_geltner_webb(reported),
        "OW":  okunev_white(reported),
        "KF":  kalman_filter(reported),
    }


def build_or_load_unsmoothed(returns: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Build a long-form table with PE/NPI raw + 3 unsmoothed versions + proxies."""
    cache = cfg.CACHE_DIR / "unsmoothed.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)

    pe_raw, npi_raw = returns["PE"].dropna(), returns["NPI"].dropna()
    res_pe  = unsmooth_all(pe_raw)
    res_npi = unsmooth_all(npi_raw)

    out = pd.DataFrame({
        "PE_raw":   pe_raw,
        "PE_FGW":   res_pe["FGW"].series,
        "PE_OW":    res_pe["OW"].series,
        "PE_KF":    res_pe["KF"].series,
        "NPI_raw":  npi_raw,
        "NPI_FGW":  res_npi["FGW"].series,
        "NPI_OW":   res_npi["OW"].series,
        "NPI_KF":   res_npi["KF"].series,
        "LPX50":    returns[cfg.PE_PROXY]  if cfg.PE_PROXY  in returns else np.nan,
        "RMZ":      returns[cfg.NPI_PROXY] if cfg.NPI_PROXY in returns else np.nan,
    })
    out.to_parquet(cache)
    return out


def vol_table(unsmoothed: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Annualized vol comparison: raw / FGW / OW / KF / proxy, for PE & NPI."""
    rows = []
    for asset, proxy in [("PE", "LPX50"), ("NPI", "RMZ")]:
        rows.append({
            "asset":   asset,
            "raw":     _ann(unsmoothed[f"{asset}_raw"].dropna()),
            "FGW":     _ann(unsmoothed[f"{asset}_FGW"].dropna()),
            "OW":      _ann(unsmoothed[f"{asset}_OW"].dropna()),
            "KF":      _ann(unsmoothed[f"{asset}_KF"].dropna()),
            "proxy":   _ann(returns[proxy].dropna()),
            "proxy_name": proxy,
        })
    return pd.DataFrame(rows).set_index("asset")
