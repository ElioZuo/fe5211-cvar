"""Marginal modeling: AR mean + EWMA conditional variance + Skew-t / semi-parametric tails."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import minimize, minimize_scalar
from arch.univariate.distribution import SkewStudent
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

from . import config as cfg


# =============================================================================
#  Marginal distribution classes (must remain pickle-stable across versions)
# =============================================================================
class SkewStudentMarginal:
    """Hansen (1994) skewed Student-t fitted on standardized residuals."""

    def __init__(self, z: np.ndarray):
        self.z = np.asarray(z)
        self._dist = SkewStudent()
        self._fit()

    def _neg_loglik(self, params):
        eta, lam = params
        if eta <= 2.05 or eta >= 300 or abs(lam) >= 0.99:
            return 1e10
        try:
            ll = self._dist.loglikelihood(np.array([eta, lam]),
                                          self.z, np.ones_like(self.z),
                                          individual=False)
            return -ll
        except Exception:
            return 1e10

    def _fit(self):
        x0 = np.array([8.0, 0.0])
        bnds = [(2.05, 300), (-0.99, 0.99)]
        res = minimize(self._neg_loglik, x0, bounds=bnds, method="L-BFGS-B")
        self.eta, self.lam = res.x
        self.params = np.array([self.eta, self.lam])
        self.ll = -res.fun

    def cdf(self, z):
        z = np.atleast_1d(z).astype(float)
        u = np.atleast_1d(self._dist.cdf(z, self.params))
        return u[0] if u.size == 1 and np.ndim(z) == 0 else u

    def ppf(self, u):
        u = np.clip(np.atleast_1d(u).astype(float), 1e-8, 1 - 1e-8)
        z = np.atleast_1d(self._dist.ppf(u, self.params))
        return z[0] if z.size == 1 and np.ndim(u) == 0 else z


class SemiParametricMarginal:
    """Empirical center on [lower_q, upper_q] with Student-t tails outside."""

    def __init__(self, z: np.ndarray, lower_q: float = 0.10, upper_q: float = 0.90):
        self.z = np.asarray(z); self.n = len(self.z)
        self.lower_q = lower_q; self.upper_q = upper_q
        self.z_lower = np.quantile(self.z, lower_q)
        self.z_upper = np.quantile(self.z, upper_q)

        z_sorted = np.sort(self.z)
        cdf_emp = (np.arange(1, self.n + 1) - 0.5) / self.n
        self._z_sorted = z_sorted; self._cdf_emp = cdf_emp

        self.left_tail_data  = self.z[self.z < self.z_lower]
        self.right_tail_data = self.z[self.z > self.z_upper]
        self.left_t_params  = self._fit_tail(self.left_tail_data)
        self.right_t_params = self._fit_tail(self.right_tail_data)

        self._cdf_interp = interp1d(z_sorted, cdf_emp, bounds_error=False,
                                    fill_value=(0.0, 1.0))
        self._ppf_interp = interp1d(cdf_emp, z_sorted, bounds_error=False,
                                    fill_value=(z_sorted[0], z_sorted[-1]))

    @staticmethod
    def _fit_tail(d):
        if len(d) >= 5:
            return stats.t.fit(d)
        return (10.0, float(np.mean(d)) if len(d) else 0.0,
                float(np.std(d)) if len(d) else 1.0)

    @staticmethod
    def _safe(out, scalar):
        out = np.atleast_1d(out)
        return float(out[0]) if scalar and out.size == 1 else out

    def cdf(self, z):
        scalar = np.ndim(z) == 0
        z = np.atleast_1d(z).astype(float)
        out = np.empty_like(z, dtype=float)

        mid = (z >= self.z_lower) & (z <= self.z_upper)
        out[mid] = self._cdf_interp(z[mid])

        left = z < self.z_lower
        if left.any():
            tcz = stats.t.cdf(z[left], *self.left_t_params)
            tcl = stats.t.cdf(self.z_lower, *self.left_t_params)
            sf = self.lower_q / max(tcl, 1e-8)
            out[left] = np.clip(tcz * sf, 1e-6, self.lower_q)

        right = z > self.z_upper
        if right.any():
            tsz = stats.t.sf(z[right], *self.right_t_params)
            tsu = stats.t.sf(self.z_upper, *self.right_t_params)
            sf = (1 - self.upper_q) / max(tsu, 1e-8)
            out[right] = np.clip(1 - tsz * sf, self.upper_q, 1 - 1e-6)

        return self._safe(out, scalar)

    def ppf(self, u):
        scalar = np.ndim(u) == 0
        u = np.atleast_1d(u).astype(float)
        out = np.empty_like(u, dtype=float)

        mid = (u >= self.lower_q) & (u <= self.upper_q)
        out[mid] = self._ppf_interp(u[mid])

        left = u < self.lower_q
        if left.any():
            tcl = stats.t.cdf(self.z_lower, *self.left_t_params)
            sf  = self.lower_q / max(tcl, 1e-8)
            tgt = np.clip(u[left] / sf, 1e-8, 1 - 1e-8)
            out[left] = stats.t.ppf(tgt, *self.left_t_params)

        right = u > self.upper_q
        if right.any():
            tsu = stats.t.sf(self.z_upper, *self.right_t_params)
            sf  = (1 - self.upper_q) / max(tsu, 1e-8)
            tgt = np.clip((1 - u[right]) / sf, 1e-8, 1 - 1e-8)
            out[right] = stats.t.isf(tgt, *self.right_t_params)

        return self._safe(out, scalar)


# =============================================================================
#  Helpers: AR mean + EWMA variance
# =============================================================================
def fit_ar_mean(y: pd.Series, p: int):
    if p == 0:
        mu = pd.Series(y.mean(), index=y.index)
        return mu, (y - mu), None
    m = ARIMA(y, order=(p, 0, 0)).fit()
    return m.fittedvalues, y - m.fittedvalues, m


def ewma_variance(resid: np.ndarray, lam: float, init: float | None = None) -> np.ndarray:
    """sigma^2_t = lam * sigma^2_{t-1} + (1-lam) * resid_{t-1}^2."""
    r = np.asarray(resid)
    n = len(r); s = np.empty(n)
    s[0] = float(np.var(r)) if init is None else init
    for t in range(1, n):
        s[t] = lam * s[t - 1] + (1 - lam) * r[t - 1] ** 2
    return s


def fit_ewma_lambda(resid: np.ndarray) -> float:
    """MLE for lambda assuming normal innovations (RiskMetrics convention)."""
    lo, hi = cfg.EWMA_LAMBDA_BOUNDS

    def nll(lam):
        s2 = ewma_variance(resid, lam)
        s  = np.sqrt(s2[1:])
        z  = resid[1:] / s
        return 0.5 * np.sum(np.log(2 * np.pi * s ** 2) + z ** 2)

    try:
        res = minimize_scalar(nll, bounds=(lo, hi), method="bounded")
        lam = float(res.x)
        if lam <= lo + 1e-3 or lam >= hi - 1e-3:
            return cfg.EWMA_LAMBDA_FALLBACK
        return lam
    except Exception:
        return cfg.EWMA_LAMBDA_FALLBACK


# =============================================================================
#  Per-asset fit
# =============================================================================
def fit_marginal(y: pd.Series, force_ar1: bool = False, semiparam: bool = False) -> dict:
    """Fit one asset: AR mean -> EWMA -> standardize -> distribution."""
    if force_ar1:
        p = 1
    else:
        # BIC on p in {0, 1, 2}
        n = len(y); cands = {}
        for p_try in (0, 1, 2):
            try:
                if p_try == 0:
                    var = float(np.var(y - y.mean()))
                    cands[p_try] = n * np.log(var) + np.log(n)
                else:
                    cands[p_try] = ARIMA(y, order=(p_try, 0, 0)).fit().bic
            except Exception:
                pass
        p = min(cands, key=cands.get)

    mu, resid, mean_model = fit_ar_mean(y, p)
    resid = resid.dropna()
    lam = fit_ewma_lambda(resid.values)
    sigma2 = ewma_variance(resid.values, lam)
    z = resid.values / np.sqrt(sigma2)

    if semiparam:
        dist = SemiParametricMarginal(z, cfg.SEMIPARAM_LOWER_Q, cfg.SEMIPARAM_UPPER_Q)
    else:
        dist = SkewStudentMarginal(z)

    u = np.clip(np.asarray(dist.cdf(z)), 1e-6, 1 - 1e-6)

    lb_z  = acorr_ljungbox(z,    lags=[4, 8], return_df=True)
    lb_z2 = acorr_ljungbox(z**2, lags=[4, 8], return_df=True)
    ks_p  = float(stats.kstest(u, "uniform").pvalue)

    return {
        "ar_lags":       p,
        "lambda":        lam,
        "half_life":     float(np.log(0.5) / np.log(lam)),
        "sigma2":        pd.Series(sigma2, index=resid.index),
        "sigma2_uncond": float(np.var(resid.values)),
        "residuals_raw": resid,
        "residuals":     pd.Series(z, index=resid.index),
        "uniforms":      pd.Series(u, index=resid.index),
        "dist":          dist,
        "use_semiparam": semiparam,
        "diagnostics": {
            "resid_mean": float(np.mean(z)),
            "resid_std":  float(np.std(z)),
            "resid_skew": float(stats.skew(z)),
            "resid_kurt": float(stats.kurtosis(z)),
            "lb_z_p4":    float(lb_z.iloc[0, 1]),
            "lb_z2_p4":   float(lb_z2.iloc[0, 1]),
            "ks_p":       ks_p,
            "eta":        getattr(dist, "eta", float("nan")),
            "lambda_skew":getattr(dist, "lam", float("nan")),
        }
    }


# =============================================================================
#  Top-level: fit all four marginals
# =============================================================================
def fit_all_marginals(returns: pd.DataFrame, unsmoothed: pd.DataFrame) -> dict:
    df = pd.DataFrame(index=unsmoothed.index)
    df["SPY"] = returns["SPY"].reindex(unsmoothed.index)
    df["AGG"] = returns["AGG"].reindex(unsmoothed.index)
    df["PE"]  = unsmoothed["PE_KF"]
    df["NPI"] = unsmoothed["NPI_KF"]
    df = df.dropna()

    out = {
        "SPY": fit_marginal(df["SPY"], force_ar1=False, semiparam=False),
        "AGG": fit_marginal(df["AGG"], force_ar1=False, semiparam=False),
        "PE":  fit_marginal(df["PE"],  force_ar1=True,  semiparam=True),
        "NPI": fit_marginal(df["NPI"], force_ar1=True,  semiparam=True),
    }
    return out


def collect_panels(marginals: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (residuals, uniforms, sigma2) panels aligned to the common index."""
    resid = pd.concat([marginals[a]["residuals"].rename(a) for a in cfg.ASSETS], axis=1).dropna()
    unif  = pd.concat([marginals[a]["uniforms"].rename(a)  for a in cfg.ASSETS], axis=1).dropna()
    s2    = pd.concat([marginals[a]["sigma2"].rename(a)    for a in cfg.ASSETS], axis=1).dropna()
    return resid, unif, s2


def build_or_load_marginals(returns: pd.DataFrame, unsmoothed: pd.DataFrame,
                             force: bool = False):
    """Cache marginals + panels."""
    import pickle
    cache_pkl = cfg.CACHE_DIR / "marginals.pkl"
    cache_unf = cfg.CACHE_DIR / "uniforms.parquet"
    cache_res = cfg.CACHE_DIR / "residuals.parquet"
    cache_s2  = cfg.CACHE_DIR / "sigma2.parquet"

    if all(p.exists() for p in [cache_pkl, cache_unf, cache_res, cache_s2]) and not force:
        with open(cache_pkl, "rb") as f:
            marg = pickle.load(f)
        return (marg, pd.read_parquet(cache_res),
                pd.read_parquet(cache_unf), pd.read_parquet(cache_s2))

    marg = fit_all_marginals(returns, unsmoothed)
    resid, unif, s2 = collect_panels(marg)
    with open(cache_pkl, "wb") as f:
        pickle.dump(marg, f)
    resid.to_parquet(cache_res); unif.to_parquet(cache_unf); s2.to_parquet(cache_s2)
    return marg, resid, unif, s2
