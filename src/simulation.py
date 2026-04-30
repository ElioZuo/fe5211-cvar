"""VaR / ES computation via four methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg
from .marginals import ewma_variance


WEIGHTS_ARR  = np.array(cfg.WEIGHTS)
MU_Q_LOG     = np.log(1 + np.array(cfg.EXPECTED_RETURNS)) / cfg.QUARTERS_PER_YEAR


# =============================================================================
#  Helpers
# =============================================================================
def var_es(losses: np.ndarray, alpha: float = cfg.ALPHA) -> tuple[float, float]:
    losses = np.asarray(losses)
    var = float(np.quantile(losses, alpha))
    tail = losses[losses >= var]
    es  = float(tail.mean()) if len(tail) else var
    return var, es


def stationary_block_indices(n_obs: int, horizon: int, n_paths: int,
                             block_size: int, seed: int) -> np.ndarray:
    """Stationary block bootstrap (Politis & Romano 1994), geometric block lengths."""
    rng = np.random.default_rng(seed)
    p = 1.0 / block_size
    jumps  = rng.random((n_paths, horizon)) < p
    starts = rng.integers(0, n_obs, (n_paths, horizon))
    idx = np.empty((n_paths, horizon), dtype=np.int64)
    idx[:, 0] = starts[:, 0]
    for t in range(1, horizon):
        idx[:, t] = np.where(jumps[:, t], starts[:, t],
                              (idx[:, t - 1] + 1) % n_obs)
    return idx


def sigma2_uncond_for_simulation(marginals: dict,
                                  unsmoothed_panel: pd.DataFrame) -> dict:
    """For PE/NPI (force_ar1=True), use unsmoothed log-return variance.
    For SPY/AGG (no AR), the marginal's sigma2_uncond is already correct."""
    out = {}
    for a in cfg.ASSETS:
        if marginals[a]["ar_lags"] > 0:
            out[a] = float(unsmoothed_panel[a].var())
        else:
            out[a] = float(marginals[a]["sigma2_uncond"])
    return out


# =============================================================================
#  Method 1 — Parametric (closed-form)
# =============================================================================
def parametric_es(returns_panel: pd.DataFrame) -> tuple[float, float]:
    """Multivariate normal CVaR with sample covariance + theoretical drift."""
    Sigma_q = np.cov(returns_panel.values.T)
    sigma_p_q = float(np.sqrt(WEIGHTS_ARR @ Sigma_q @ WEIGHTS_ARR))
    mu_p_q    = float(WEIGHTS_ARR @ MU_Q_LOG)
    mu_total  = cfg.HORIZON_QUARTERS * mu_p_q
    sig_total = sigma_p_q * np.sqrt(cfg.HORIZON_QUARTERS)
    z_a = stats.norm.ppf(cfg.ALPHA)
    var = -mu_total + z_a * sig_total
    es  = -mu_total + (stats.norm.pdf(z_a) / (1 - cfg.ALPHA)) * sig_total
    return var, es


# =============================================================================
#  Method 2 — Historical Simulation (stationary block bootstrap)
# =============================================================================
def historical_simulation(returns_panel: pd.DataFrame, n_paths: int,
                          seed: int) -> tuple[float, float, np.ndarray]:
    X = returns_panel.values
    n_obs = X.shape[0]
    X_demean = X - X.mean(axis=0, keepdims=True)
    idx = stationary_block_indices(n_obs, cfg.HORIZON_QUARTERS, n_paths,
                                    cfg.BLOCK_BOOTSTRAP_LEN, seed)
    sampled = X_demean[idx] + MU_Q_LOG[None, None, :]
    port_q  = sampled @ WEIGHTS_ARR
    losses  = -port_q.sum(axis=1)
    var, es = var_es(losses)
    return var, es, losses


# =============================================================================
#  Method 3 — Filtered Historical Simulation
# =============================================================================
def filtered_historical_simulation(residuals_panel: pd.DataFrame,
                                    marginals: dict,
                                    sigma2_uncond: dict,
                                    n_paths: int,
                                    seed: int) -> tuple[float, float, np.ndarray]:
    """Bootstrap from standardized residuals, inflate via EWMA recursion."""
    Z = residuals_panel[cfg.ASSETS].values
    n_obs = Z.shape[0]
    rng = np.random.default_rng(seed)
    sample_idx = rng.integers(0, n_obs, (n_paths, cfg.HORIZON_QUARTERS))
    Z_sampled = Z[sample_idx]

    lam_arr = np.array([marginals[a]["lambda"] for a in cfg.ASSETS])
    s2_0    = np.array([sigma2_uncond[a]       for a in cfg.ASSETS])

    sigma2 = np.empty((n_paths, cfg.HORIZON_QUARTERS, len(cfg.ASSETS)))
    r_sim  = np.empty_like(sigma2)
    sigma2[:, 0, :] = s2_0[None, :]
    r_sim[:, 0, :]  = MU_Q_LOG[None, :] + np.sqrt(sigma2[:, 0, :]) * Z_sampled[:, 0, :]
    for t in range(1, cfg.HORIZON_QUARTERS):
        sigma2[:, t, :] = (lam_arr[None, :] * sigma2[:, t - 1, :]
                           + (1 - lam_arr[None, :]) * r_sim[:, t - 1, :] ** 2)
        r_sim[:, t, :]  = MU_Q_LOG[None, :] + np.sqrt(sigma2[:, t, :]) * Z_sampled[:, t, :]

    port_q = r_sim @ WEIGHTS_ARR
    losses = -port_q.sum(axis=1)
    var, es = var_es(losses)
    return var, es, losses


# =============================================================================
#  Method 4 — Champion: Copula + per-asset PPF + EWMA
# =============================================================================
def champion_simulation(copula, marginals: dict, sigma2_uncond: dict,
                         n_paths: int, seed: int,
                         return_paths: bool = False
                         ) -> tuple[float, float, np.ndarray, np.ndarray | None]:
    n_total = n_paths * cfg.HORIZON_QUARTERS
    u_flat = copula.simulate(n_total, seed=seed)
    # Clip to avoid extreme PPF values from semi-parametric tail extrapolation
    u_flat = np.clip(u_flat, 1e-4, 1 - 1e-4)
    u = u_flat.reshape(n_paths, cfg.HORIZON_QUARTERS, len(cfg.ASSETS))

    Z = np.empty_like(u)
    for i, a in enumerate(cfg.ASSETS):
        Z[:, :, i] = np.asarray(marginals[a]["dist"].ppf(u[:, :, i].ravel())) \
                       .reshape(n_paths, cfg.HORIZON_QUARTERS)

    lam_arr = np.array([marginals[a]["lambda"] for a in cfg.ASSETS])
    s2_0    = np.array([sigma2_uncond[a]       for a in cfg.ASSETS])

    sigma2 = np.empty((n_paths, cfg.HORIZON_QUARTERS, len(cfg.ASSETS)))
    r_sim  = np.empty_like(sigma2)
    sigma2[:, 0, :] = s2_0[None, :]
    r_sim[:, 0, :]  = MU_Q_LOG[None, :] + np.sqrt(sigma2[:, 0, :]) * Z[:, 0, :]
    for t in range(1, cfg.HORIZON_QUARTERS):
        sigma2[:, t, :] = (lam_arr[None, :] * sigma2[:, t - 1, :]
                           + (1 - lam_arr[None, :]) * r_sim[:, t - 1, :] ** 2)
        r_sim[:, t, :]  = MU_Q_LOG[None, :] + np.sqrt(sigma2[:, t, :]) * Z[:, t, :]

    port_q = r_sim @ WEIGHTS_ARR
    losses = -port_q.sum(axis=1)
    var, es = var_es(losses)
    paths_tensor = r_sim.astype(np.float32) if return_paths else None
    return var, es, losses, paths_tensor


# =============================================================================
#  Top-level: build full ES matrix
# =============================================================================
def build_es_matrix(returns: pd.DataFrame, unsmoothed: pd.DataFrame,
                    marginals: dict, residuals: pd.DataFrame,
                    copula, n_paths: int = cfg.N_PATHS_MC,
                    save_paths: bool = True) -> dict:
    """Run 4 methods x 2 data versions and return all results + Champion path tensor."""
    # Build the two return panels aligned on common index
    reported = pd.DataFrame(index=unsmoothed.index)
    reported["SPY"] = returns["SPY"].reindex(unsmoothed.index)
    reported["AGG"] = returns["AGG"].reindex(unsmoothed.index)
    reported["PE"]  = returns["PE"].reindex(unsmoothed.index)
    reported["NPI"] = returns["NPI"].reindex(unsmoothed.index)
    reported = reported.dropna()

    unsm = pd.DataFrame(index=unsmoothed.index)
    unsm["SPY"] = returns["SPY"].reindex(unsmoothed.index)
    unsm["AGG"] = returns["AGG"].reindex(unsmoothed.index)
    unsm["PE"]  = unsmoothed["PE_KF"]
    unsm["NPI"] = unsmoothed["NPI_KF"]
    unsm = unsm.dropna()

    s2_uncond = sigma2_uncond_for_simulation(marginals, unsm)

    methods  = ["Parametric", "HS", "FHS", "Champion"]
    versions = ["reported", "unsmoothed"]
    var_mat = pd.DataFrame(index=methods, columns=versions, dtype=float)
    es_mat  = pd.DataFrame(index=methods, columns=versions, dtype=float)
    loss_arrays: dict = {}

    # Parametric — both versions (closed-form)
    var_mat.loc["Parametric", "reported"], es_mat.loc["Parametric", "reported"]   = parametric_es(reported)
    var_mat.loc["Parametric", "unsmoothed"], es_mat.loc["Parametric", "unsmoothed"] = parametric_es(unsm)

    # HS — both versions
    v, e, l = historical_simulation(reported, n_paths, seed=cfg.SEED_SIM_HS)
    var_mat.loc["HS", "reported"], es_mat.loc["HS", "reported"] = v, e
    loss_arrays[("HS", "reported")] = l

    v, e, l = historical_simulation(unsm, n_paths, seed=cfg.SEED_SIM_HS_UNSM)
    var_mat.loc["HS", "unsmoothed"], es_mat.loc["HS", "unsmoothed"] = v, e
    loss_arrays[("HS", "unsmoothed")] = l

    # FHS — unsmoothed only (Block 3 was fit on unsmoothed)
    v, e, l = filtered_historical_simulation(residuals, marginals, s2_uncond,
                                              n_paths, seed=cfg.SEED_SIM_FHS)
    var_mat.loc["FHS", "unsmoothed"], es_mat.loc["FHS", "unsmoothed"] = v, e
    loss_arrays[("FHS", "unsmoothed")] = l

    # Champion — unsmoothed only, save path tensor
    v, e, l, pt = champion_simulation(copula, marginals, s2_uncond,
                                       n_paths, seed=cfg.SEED_SIM_CHAMPION,
                                       return_paths=save_paths)
    var_mat.loc["Champion", "unsmoothed"], es_mat.loc["Champion", "unsmoothed"] = v, e
    loss_arrays[("Champion", "unsmoothed")] = l

    return {
        "var_matrix":   var_mat,
        "es_matrix":    es_mat,
        "loss_arrays":  loss_arrays,
        "paths_champion": pt,
        "reported_panel":   reported,
        "unsmoothed_panel": unsm,
        "sigma2_uncond":    s2_uncond,
    }
