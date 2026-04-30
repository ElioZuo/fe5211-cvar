"""Copula models: Gaussian, Student-t (joint MLE), and optional R-Vine."""

from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln

from . import config as cfg

try:
    import pyvinecopulib as pv
    HAS_VINECOPULIB = True
except ImportError:
    HAS_VINECOPULIB = False


# =============================================================================
#  Helpers (Kendall's tau, nearest-PD projection, empirical chi)
# =============================================================================
def kendall_tau_matrix(u: np.ndarray) -> np.ndarray:
    n, d = u.shape
    T = np.eye(d)
    for i, j in combinations(range(d), 2):
        tau, _ = stats.kendalltau(u[:, i], u[:, j])
        T[i, j] = T[j, i] = tau if not np.isnan(tau) else 0.0
    return T


def tau_to_rho(T: np.ndarray) -> np.ndarray:
    """Elliptical copula identity: rho = sin(pi/2 * tau)."""
    return np.sin(np.pi / 2 * T)


def nearest_pd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    A = (A + A.T) / 2
    w, V = np.linalg.eigh(A)
    if (w > eps).all():
        return A
    w = np.maximum(w, eps)
    A = V @ np.diag(w) @ V.T
    d = np.sqrt(np.diag(A))
    return A / np.outer(d, d)


def empirical_chi_L(u: np.ndarray, q: float = 0.10) -> np.ndarray:
    n, d = u.shape
    M = np.eye(d)
    for i, j in combinations(range(d), 2):
        bi = u[:, i] < q
        chi = float(np.mean(u[bi, j] < q)) if bi.sum() else 0.0
        M[i, j] = M[j, i] = chi
    return M


def empirical_chi_U(u: np.ndarray, q: float = 0.10) -> np.ndarray:
    n, d = u.shape
    M = np.eye(d)
    for i, j in combinations(range(d), 2):
        ai = u[:, i] > 1 - q
        chi = float(np.mean(u[ai, j] > 1 - q)) if ai.sum() else 0.0
        M[i, j] = M[j, i] = chi
    return M


def bootstrap_chi_L_ci(u: np.ndarray, q: float = 0.10, n_boot: int = 1000,
                       seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(u)
    boots = np.zeros((n_boot,) + u.shape[1:] + u.shape[1:])
    boots = boots.reshape(n_boot, u.shape[1], u.shape[1])
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = empirical_chi_L(u[idx], q=q)
    return (np.percentile(boots, 2.5, axis=0),
            np.percentile(boots, 97.5, axis=0))


# =============================================================================
#  Gaussian copula
# =============================================================================
class GaussianCopula:
    name = "Gaussian"

    def __init__(self):
        self.R = None; self.D = None

    def fit(self, u: np.ndarray, R_init: np.ndarray | None = None):
        self.D = u.shape[1]
        if R_init is None:
            R_init = nearest_pd(tau_to_rho(kendall_tau_matrix(u)))
        self.R = R_init
        self._L = np.linalg.cholesky(self.R)
        self._Rinv = np.linalg.inv(self.R)
        self._logdet = np.linalg.slogdet(self.R)[1]
        return self

    def per_obs_loglik(self, u: np.ndarray) -> np.ndarray:
        z = stats.norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        quad = np.einsum("ti,ij,tj->t", z, self._Rinv, z) - np.sum(z ** 2, axis=1)
        return -0.5 * (self._logdet + quad)

    def loglik(self, u: np.ndarray) -> float:
        return float(self.per_obs_loglik(u).sum())

    def aic(self, u: np.ndarray) -> float:
        k = self.D * (self.D - 1) / 2
        return 2 * k - 2 * self.loglik(u)

    def bic(self, u: np.ndarray) -> float:
        k = self.D * (self.D - 1) / 2
        return k * np.log(len(u)) - 2 * self.loglik(u)

    def n_params(self) -> int:
        return int(self.D * (self.D - 1) / 2)

    def simulate(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.D)) @ self._L.T
        return stats.norm.cdf(z)

    def lambda_L(self) -> np.ndarray:
        return np.eye(self.D)


# =============================================================================
#  Student-t copula (joint MLE on R via Cholesky parameterization, plus nu)
# =============================================================================
class StudentTCopula:
    name = "Student-t"

    def __init__(self):
        self.R = None; self.nu = None; self.D = None

    @staticmethod
    def _flat_to_corr(L_flat: np.ndarray, D: int):
        L = np.zeros((D, D)); k = 0
        for i in range(D):
            for j in range(i + 1):
                L[i, j] = L_flat[k]; k += 1
        rn = np.maximum(np.linalg.norm(L, axis=1), 1e-10)
        L = L / rn[:, None]
        return L @ L.T, L

    @staticmethod
    def _corr_to_flat(R: np.ndarray) -> np.ndarray:
        L = np.linalg.cholesky(R); D = R.shape[0]
        return np.array([L[i, j] for i in range(D) for j in range(i + 1)])

    def _t_loglik(self, u: np.ndarray, nu: float, R: np.ndarray) -> float:
        D = R.shape[0]
        x = stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=nu)
        try:
            Rinv = np.linalg.inv(R); logdet = np.linalg.slogdet(R)[1]
        except np.linalg.LinAlgError:
            return -1e10
        log_const = (gammaln((nu + D) / 2) - gammaln(nu / 2)
                     - 0.5 * D * np.log(np.pi * nu) - 0.5 * logdet)
        quad = np.einsum("ti,ij,tj->t", x, Rinv, x)
        log_mvt  = log_const - 0.5 * (nu + D) * np.log1p(quad / nu)
        log_marg = stats.t.logpdf(x, df=nu).sum(axis=1)
        return float((log_mvt - log_marg).sum())

    def fit(self, u: np.ndarray, R_init: np.ndarray | None = None):
        self.D = u.shape[1]
        if R_init is None:
            R_init = nearest_pd(tau_to_rho(kendall_tau_matrix(u)))
        self.R = R_init

        L_flat = self._corr_to_flat(self.R)
        n_chol = len(L_flat)
        x0 = np.concatenate([L_flat, [8.0]])

        def neg_ll(p):
            L_part, nu = p[:n_chol], p[-1]
            if nu <= 2.5 or nu >= 100:
                return 1e10
            try:
                R, _ = self._flat_to_corr(L_part, self.D)
                R = nearest_pd(R, eps=1e-6)
                return -self._t_loglik(u, nu, R)
            except Exception:
                return 1e10

        res = minimize(neg_ll, x0, method="L-BFGS-B",
                       options={"maxiter": 200, "ftol": 1e-7})
        self.nu = float(res.x[-1])
        self.R, _ = self._flat_to_corr(res.x[:n_chol], self.D)
        self.R = nearest_pd(self.R)
        self._L = np.linalg.cholesky(self.R)
        self._Rinv = np.linalg.inv(self.R)
        self._logdet = np.linalg.slogdet(self.R)[1]
        return self

    def per_obs_loglik(self, u: np.ndarray) -> np.ndarray:
        D = self.D
        x = stats.t.ppf(np.clip(u, 1e-6, 1 - 1e-6), df=self.nu)
        log_const = (gammaln((self.nu + D) / 2) - gammaln(self.nu / 2)
                     - 0.5 * D * np.log(np.pi * self.nu) - 0.5 * self._logdet)
        quad = np.einsum("ti,ij,tj->t", x, self._Rinv, x)
        log_mvt  = log_const - 0.5 * (self.nu + D) * np.log1p(quad / self.nu)
        log_marg = stats.t.logpdf(x, df=self.nu).sum(axis=1)
        return log_mvt - log_marg

    def loglik(self, u): return float(self.per_obs_loglik(u).sum())
    def aic(self, u):
        k = self.D * (self.D - 1) / 2 + 1; return 2 * k - 2 * self.loglik(u)
    def bic(self, u):
        k = self.D * (self.D - 1) / 2 + 1
        return k * np.log(len(u)) - 2 * self.loglik(u)
    def n_params(self): return int(self.D * (self.D - 1) / 2 + 1)

    def simulate(self, n: int, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        z = rng.standard_normal((n, self.D)) @ self._L.T
        chi2 = rng.chisquare(self.nu, n)
        x = z / np.sqrt(chi2 / self.nu)[:, None]
        return stats.t.cdf(x, df=self.nu)

    def lambda_L(self) -> np.ndarray:
        L = np.eye(self.D)
        for i, j in combinations(range(self.D), 2):
            rho = self.R[i, j]
            arg = -np.sqrt((self.nu + 1) * (1 - rho) / (1 + rho))
            lam = 2 * stats.t.cdf(arg, df=self.nu + 1)
            L[i, j] = L[j, i] = lam
        return L


# =============================================================================
#  R-Vine copula (optional)
# =============================================================================
class RVineCopula:
    """Wraps pyvinecopulib if available, otherwise raises a helpful error."""
    name = "R-Vine"

    def __init__(self, truncation_level: int = 2):
        if not HAS_VINECOPULIB:
            raise ImportError("pyvinecopulib is required for R-Vine; "
                              "install via `pip install -r requirements-vine.txt`.")
        self.truncation_level = truncation_level
        self.vine = None; self.D = None

    def fit(self, u: np.ndarray):
        self.D = u.shape[1]
        controls = pv.FitControlsVinecop(
            family_set=[pv.BicopFamily.gaussian, pv.BicopFamily.student,
                        pv.BicopFamily.clayton, pv.BicopFamily.gumbel,
                        pv.BicopFamily.frank, pv.BicopFamily.joe,
                        pv.BicopFamily.bb1, pv.BicopFamily.bb7],
            trunc_lvl=self.truncation_level,
            selection_criterion="aic",
            preselect_families=True,
        )
        self.vine = pv.Vinecop(d=self.D)
        self.vine.select(data=u, controls=controls)
        return self

    def loglik(self, u): return float(self.vine.loglik(u))
    def per_obs_loglik(self, u): return np.log(self.vine.pdf(u) + 1e-300)
    def aic(self, u): return float(self.vine.aic(u))
    def bic(self, u): return float(self.vine.bic(u))
    def n_params(self): return int(2 * (self.D - 1))   # rough

    def simulate(self, n, seed):
        try:
            return self.vine.simulate(n=n, seeds=[seed])
        except TypeError:
            return self.vine.simulate(n=n)

    def lambda_L(self, n_sim: int = 200_000, q: float = 0.01) -> np.ndarray:
        u_sim = self.simulate(n_sim, seed=cfg.SEED_COPULA_SIM)
        L = np.eye(self.D)
        for i, j in combinations(range(self.D), 2):
            bi = u_sim[:, i] < q
            lam = float(np.mean(u_sim[bi, j] < q)) if bi.sum() else 0.0
            L[i, j] = L[j, i] = lam
        return L

    def describe_structure(self) -> pd.DataFrame:
        out = []
        for tree in range(self.truncation_level):
            for edge in range(self.D - 1 - tree):
                bicop = self.vine.get_pair_copula(tree, edge)
                out.append({
                    "tree": tree + 1, "edge": edge,
                    "family": str(bicop.family).split(".")[-1],
                    "rotation": bicop.rotation,
                    "params": str(np.array(bicop.parameters).flatten().round(3)),
                    "tau": round(bicop.tau, 3),
                })
        return pd.DataFrame(out)


# =============================================================================
#  Vuong test (Schwarz / BIC corrected)
# =============================================================================
def vuong_test(ll1_per_obs: np.ndarray, ll2_per_obs: np.ndarray,
               k1: int, k2: int) -> tuple[float, float]:
    """Vuong (1989) with BIC correction. Z>1.96 -> model 1 preferred."""
    n = len(ll1_per_obs)
    diff = ll1_per_obs - ll2_per_obs
    correction = (k1 - k2) * np.log(n) / (2 * n)
    diff_corr = diff - correction
    sd = max(float(np.std(diff_corr, ddof=1)), 1e-10)
    z = float(np.sqrt(n) * np.mean(diff_corr) / sd)
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    return z, p


# =============================================================================
#  Top-level driver
# =============================================================================
def fit_all_copulas(uniforms: pd.DataFrame, fit_vine: bool = True) -> dict:
    """Fit Gaussian + Student-t (always) and R-Vine (if available and requested)."""
    u_full = uniforms.values
    u_in   = u_full[:cfg.COPULA_OOS_SPLIT]
    u_oos  = u_full[cfg.COPULA_OOS_SPLIT:]

    R_full = nearest_pd(tau_to_rho(kendall_tau_matrix(u_full)))
    R_in   = nearest_pd(tau_to_rho(kendall_tau_matrix(u_in)))

    full_models = {
        "Gaussian":  GaussianCopula().fit(u_full, R_init=R_full),
        "Student-t": StudentTCopula().fit(u_full, R_init=R_full),
    }
    in_models = {
        "Gaussian":  GaussianCopula().fit(u_in, R_init=R_in),
        "Student-t": StudentTCopula().fit(u_in, R_init=R_in),
    }
    if fit_vine and HAS_VINECOPULIB:
        full_models["R-Vine"] = RVineCopula().fit(u_full)
        in_models["R-Vine"]   = RVineCopula().fit(u_in)

    rows = []
    for name, m in full_models.items():
        rows.append({
            "model":      name,
            "n_params":   m.n_params(),
            "ll_full":    m.loglik(u_full),
            "aic_full":   m.aic(u_full),
            "bic_full":   m.bic(u_full),
            "ll_in":      in_models[name].loglik(u_in),
            "ll_oos":     in_models[name].loglik(u_oos),
        })
    comparison = pd.DataFrame(rows).set_index("model")

    # OOS LL ranking selects champion (excludes Gaussian by design as a baseline)
    candidates = {n: comparison.loc[n, "ll_oos"]
                  for n in full_models if n != "Gaussian"}
    if not candidates:
        candidates = {"Gaussian": comparison.loc["Gaussian", "ll_oos"]}
    champion = max(candidates, key=candidates.get)

    return {
        "full":       full_models,
        "in_sample":  in_models,
        "comparison": comparison,
        "champion":   champion,
        "u_oos":      u_oos,
        "R_kendall_full": R_full,
    }


def build_or_load_copulas(uniforms: pd.DataFrame, fit_vine: bool = True,
                           force: bool = False) -> dict:
    import pickle
    cache = cfg.CACHE_DIR / "copulas.pkl"
    if cache.exists() and not force:
        with open(cache, "rb") as f:
            return pickle.load(f)
    result = fit_all_copulas(uniforms, fit_vine=fit_vine)
    with open(cache, "wb") as f:
        pickle.dump(result, f)
    return result
