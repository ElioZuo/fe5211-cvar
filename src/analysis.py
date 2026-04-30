"""Risk decomposition and validation analyses."""

from __future__ import annotations

import time
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln

from . import config as cfg
from .copula import (kendall_tau_matrix, tau_to_rho, nearest_pd,
                      empirical_chi_L, empirical_chi_U)
from .simulation import WEIGHTS_ARR, MU_Q_LOG


# =============================================================================
#  Layer 1 — Component CVaR (Euler decomposition)
# =============================================================================
def component_cvar(paths: np.ndarray) -> dict:
    """Per-asset Comp-CVaR_i = w_i * E[r_i^(3Y) | L_p >= VaR_p].
    Sum of components equals total ES (additivity)."""
    asset_3y = paths.sum(axis=1)              # (N, 4)
    port_3y  = asset_3y @ WEIGHTS_ARR          # (N,)
    losses   = -port_3y
    var_total = float(np.quantile(losses, cfg.ALPHA))
    tail_mask = losses >= var_total
    es_total  = float(losses[tail_mask].mean())

    rows = []
    for i, a in enumerate(cfg.ASSETS):
        ret_in_tail = asset_3y[tail_mask, i].mean()
        loss_in_tail = -ret_in_tail
        comp = WEIGHTS_ARR[i] * loss_in_tail
        rows.append({
            "asset":  a,
            "weight": WEIGHTS_ARR[i],
            "mean_3y_ret_in_tail":  float(ret_in_tail),
            "mean_3y_loss_in_tail": float(loss_in_tail),
            "comp_cvar":            float(comp),
            "comp_pct_of_total":    float(comp / es_total * 100),
        })
    table = pd.DataFrame(rows).set_index("asset")

    return {
        "table":     table,
        "var_total": var_total,
        "es_total":  es_total,
        "tail_mask": tail_mask,
        "port_losses":   losses,
        "asset_3y":  asset_3y,
    }


# =============================================================================
#  Layer 2 — Time-of-loss
# =============================================================================
def time_of_loss(paths: np.ndarray, tail_mask: np.ndarray,
                 worst_path_idx: int) -> pd.DataFrame:
    tail_paths = paths[tail_mask]
    tail_port_q = tail_paths @ WEIGHTS_ARR
    mean_loss_q = -tail_port_q.mean(axis=0)

    worst_port_q = paths[worst_path_idx] @ WEIGHTS_ARR
    worst_loss_q = -worst_port_q
    return pd.DataFrame({
        "quarter":            np.arange(1, cfg.HORIZON_QUARTERS + 1),
        "mean_loss_q":        mean_loss_q,
        "cum_mean_loss":      np.cumsum(mean_loss_q),
        "worst_path_loss_q":  worst_loss_q,
        "cum_worst_loss":     np.cumsum(worst_loss_q),
    })


# =============================================================================
#  Layer 2 — Historical tail windows + simulated tail summary
# =============================================================================
def historical_tail_windows(unsmoothed_panel: pd.DataFrame,
                              named_crises: list = None) -> dict:
    """All rolling 3Y windows, plus named anchors and top-3 worst."""
    if named_crises is None:
        named_crises = cfg.NAMED_CRISES
    hist_arr = unsmoothed_panel[cfg.ASSETS].values
    T = len(hist_arr)
    n_win = T - cfg.HORIZON_QUARTERS + 1
    rec = []
    for s in range(n_win):
        e = s + cfg.HORIZON_QUARTERS
        a3y = hist_arr[s:e].sum(axis=0)
        rec.append({
            "start_date": unsmoothed_panel.index[s],
            "end_date":   unsmoothed_panel.index[e - 1],
            "port_loss":  float(-a3y @ WEIGHTS_ARR),
            **{f"{a}_3y_loss": float(-a3y[i]) for i, a in enumerate(cfg.ASSETS)},
            **{f"{a}_3y_ret":  float( a3y[i]) for i, a in enumerate(cfg.ASSETS)},
        })
    windows = pd.DataFrame(rec)

    named = []
    for s_str, e_str, name in named_crises:
        s_d = pd.Timestamp(s_str)
        if s_d < unsmoothed_panel.index.min():
            continue
        loc = unsmoothed_panel.index.searchsorted(s_d)
        if loc + cfg.HORIZON_QUARTERS > T:
            continue
        a3y = hist_arr[loc:loc + cfg.HORIZON_QUARTERS].sum(axis=0)
        named.append({
            "label": name,
            "source": "named",
            "start_date": unsmoothed_panel.index[loc],
            "end_date":   unsmoothed_panel.index[loc + cfg.HORIZON_QUARTERS - 1],
            "port_loss":  float(-a3y @ WEIGHTS_ARR),
            **{f"{a}_3y_loss": float(-a3y[i]) for i, a in enumerate(cfg.ASSETS)},
            **{f"{a}_3y_ret":  float( a3y[i]) for i, a in enumerate(cfg.ASSETS)},
        })
    named_df = pd.DataFrame(named)

    top3 = windows.nlargest(3, "port_loss").reset_index(drop=True)
    top3_rec = []
    for i, r in top3.iterrows():
        top3_rec.append({
            "label":  f"Worst#{i+1}",
            "source": "data-driven",
            "start_date": r["start_date"], "end_date": r["end_date"],
            "port_loss":  r["port_loss"],
            **{f"{a}_3y_loss": r[f"{a}_3y_loss"] for a in cfg.ASSETS},
            **{f"{a}_3y_ret":  r[f"{a}_3y_ret"]  for a in cfg.ASSETS},
        })
    combined = pd.concat([named_df, pd.DataFrame(top3_rec)], ignore_index=True)

    return {"all_windows": windows, "named": named_df,
            "top3": pd.DataFrame(top3_rec), "combined": combined}


# =============================================================================
#  Layer 5A — Short-term backtest (1Q VaR/ES)
# =============================================================================
def in_sample_backtest(unsmoothed_panel: pd.DataFrame,
                        sigma2_panel: pd.DataFrame,
                        marginals: dict) -> dict:
    """In-sample 1-quarter VaR/ES backtest on the unsmoothed portfolio."""
    R_hist = unsmoothed_panel.loc[sigma2_panel.index][cfg.ASSETS].corr().values

    T = len(sigma2_panel)
    port_sigma = np.empty(T)
    for t in range(T):
        s_t = np.sqrt(sigma2_panel.iloc[t][cfg.ASSETS].values)
        port_sigma[t] = float(np.sqrt(WEIGHTS_ARR @ (np.outer(s_t, s_t) * R_hist) @ WEIGHTS_ARR))

    mu_p_q = float(WEIGHTS_ARR @ MU_Q_LOG)
    z_a = stats.norm.ppf(1 - cfg.ALPHA)              # 5th percentile
    es_factor = stats.norm.pdf(z_a) / (1 - cfg.ALPHA)
    var_q = -mu_p_q - port_sigma * z_a
    es_q  = -mu_p_q + port_sigma * es_factor

    # Actual losses
    port_ret = unsmoothed_panel.loc[sigma2_panel.index][cfg.ASSETS].values @ WEIGHTS_ARR
    port_loss = -port_ret
    breaches = port_loss > var_q
    n = len(breaches); n_b = int(breaches.sum())
    expected = n * (1 - cfg.ALPHA)

    # Kupiec POF
    p0 = 1 - cfg.ALPHA
    p_hat = n_b / n if n else 0
    if p_hat in (0, 1):
        lr_pof, p_kupiec = 0.0, 1.0
    else:
        lr_pof = -2 * (n_b * np.log(p0) + (n - n_b) * np.log(1 - p0)
                       - n_b * np.log(p_hat) - (n - n_b) * np.log(1 - p_hat))
        p_kupiec = float(1 - stats.chi2.cdf(lr_pof, df=1))

    # Christoffersen Independence
    b = breaches.astype(int)
    n00 = int(((b[:-1] == 0) & (b[1:] == 0)).sum())
    n01 = int(((b[:-1] == 0) & (b[1:] == 1)).sum())
    n10 = int(((b[:-1] == 1) & (b[1:] == 0)).sum())
    n11 = int(((b[:-1] == 1) & (b[1:] == 1)).sum())
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi   = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    if pi01 in (0, 1) or pi11 in (0, 1) or pi in (0, 1):
        lr_ind, p_chris = 0.0, 1.0
    else:
        lr_ind = -2 * ((n00 + n10) * np.log(1 - pi) + (n01 + n11) * np.log(pi)
                       - n00 * np.log(1 - pi01) - n01 * np.log(pi01)
                       - n10 * np.log(1 - pi11) - n11 * np.log(pi11))
        p_chris = float(1 - stats.chi2.cdf(lr_ind, df=1))

    # Acerbi-Szekely Z2
    in_tail = port_loss * b
    in_tail_es = es_q * b
    ratio = np.where(in_tail_es != 0, in_tail / in_tail_es, 0)
    acerbi_z = float(ratio.sum() / n - (1 - cfg.ALPHA))
    acerbi_se = float(np.sqrt((1 - cfg.ALPHA) * cfg.ALPHA / n))
    acerbi_p = float(1 - stats.norm.cdf(abs(acerbi_z) / acerbi_se))

    table = pd.DataFrame([
        {"test": "Kupiec POF",         "stat": lr_pof,  "p_value": p_kupiec},
        {"test": "Christoffersen Ind", "stat": lr_ind,  "p_value": p_chris},
        {"test": "Acerbi-Szekely Z",   "stat": acerbi_z,"p_value": acerbi_p},
        {"test": "Breach count",       "stat": n_b,     "p_value": np.nan},
    ])

    return {
        "table":      table,
        "n_obs":      n,
        "n_breach":   n_b,
        "expected_breach": expected,
        "var_q":      pd.Series(var_q, index=sigma2_panel.index),
        "es_q":       pd.Series(es_q,  index=sigma2_panel.index),
        "port_loss":  pd.Series(port_loss, index=sigma2_panel.index),
        "breaches":   pd.Series(breaches,  index=sigma2_panel.index),
        "port_sigma": pd.Series(port_sigma, index=sigma2_panel.index),
    }


# =============================================================================
#  Layer 5B — Historical replay
# =============================================================================
def historical_replay(combined_windows: pd.DataFrame,
                       champion_losses: np.ndarray) -> pd.DataFrame:
    rows = []
    for _, ev in combined_windows.iterrows():
        pct = float((champion_losses < ev["port_loss"]).mean() * 100)
        rows.append({
            "label":     ev["label"],
            "source":    ev["source"],
            "start":     ev["start_date"],
            "end":       ev["end_date"],
            "port_loss": ev["port_loss"],
            **{a: ev[f"{a}_3y_loss"] for a in cfg.ASSETS},
            "mc_percentile": pct,
            "implied_3y_freq_years": (1 / max(1 - pct / 100, 1e-6)) * 3 / 4,
        })
    return pd.DataFrame(rows)


# =============================================================================
#  Layer 5C — Hypothetical stress (multipliers on historical paths)
# =============================================================================
def stress_scenarios(combined_windows: pd.DataFrame,
                      es_baseline: float) -> pd.DataFrame:
    rows = []
    name_to_row = {r["label"]: r for _, r in combined_windows.iterrows()}
    for base_label, mult, scen_name in cfg.STRESS_SCENARIOS:
        if base_label not in name_to_row:
            continue
        base = name_to_row[base_label]
        losses = {a: float(base[f"{a}_3y_loss"]) * mult for a in cfg.ASSETS}
        port_loss = float(sum(losses[a] * WEIGHTS_ARR[i]
                                for i, a in enumerate(cfg.ASSETS)))
        rows.append({
            "scenario":           scen_name,
            "base":               base_label,
            "multiplier":         mult,
            "stressed_port_loss": port_loss,
            "pct_of_baseline_es": port_loss / es_baseline,
            **{f"{a}_loss": losses[a] for a in cfg.ASSETS},
        })
    return pd.DataFrame(rows)


# =============================================================================
#  Layer 5D — Reverse stress (deep-tail decomposition)
# =============================================================================
def reverse_stress(paths: np.ndarray, port_losses: np.ndarray,
                    es_baseline: float) -> dict:
    target = 2 * es_baseline
    deep_mask = port_losses >= target
    n_deep = int(deep_mask.sum())
    if n_deep < 100:
        # fallback: top 0.5%
        threshold = float(np.quantile(port_losses, 0.995))
        deep_mask = port_losses >= threshold
        n_deep = int(deep_mask.sum())
        target = threshold

    asset_3y = paths.sum(axis=1)
    deep_asset_loss = -asset_3y[deep_mask].mean(axis=0)        # (4,)
    baseline_mask = port_losses >= np.quantile(port_losses, cfg.ALPHA)
    base_asset_loss = -asset_3y[baseline_mask].mean(axis=0)

    table_rows = []
    for i, a in enumerate(cfg.ASSETS):
        table_rows.append({
            "asset":           a,
            "baseline_3y_loss":     float(base_asset_loss[i]),
            "deep_tail_3y_loss":    float(deep_asset_loss[i]),
            "weight":               float(WEIGHTS_ARR[i]),
            "loss_contribution":    float(deep_asset_loss[i] * WEIGHTS_ARR[i]),
        })
    return {
        "target_loss":     target,
        "n_deep_tail":     n_deep,
        "table":           pd.DataFrame(table_rows).set_index("asset"),
        "mean_port_loss_deep": float(-(asset_3y[deep_mask] @ WEIGHTS_ARR).mean()),
    }


# =============================================================================
#  Layer 5E — Two-tier bootstrap CIs for VaR & ES
# =============================================================================
def tier_a_bootstrap(losses: np.ndarray, n_boot: int = cfg.N_BOOT_TIER_A,
                      seed: int = cfg.SEED_BOOT_TIER_A) -> dict:
    """Resample the 1M loss array; isolates MC sampling noise."""
    rng = np.random.default_rng(seed)
    n = len(losses)
    var_b = np.empty(n_boot); es_b = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        s = losses[idx]
        v = float(np.quantile(s, cfg.ALPHA))
        e = float(s[s >= v].mean())
        var_b[b] = v; es_b[b] = e
    return {
        "var_samples": var_b,
        "es_samples":  es_b,
        "var_ci":      (float(np.percentile(var_b, 2.5)), float(np.percentile(var_b, 97.5))),
        "es_ci":       (float(np.percentile(es_b, 2.5)),  float(np.percentile(es_b, 97.5))),
        "var_mean":    float(var_b.mean()), "es_mean": float(es_b.mean()),
    }


def _t_copula_grid_loglik(u: np.ndarray, R: np.ndarray, nu: float) -> float:
    """Log-lik of a Student-t copula with given (R, nu) at uniforms u."""
    n, d = u.shape
    x = stats.t.ppf(np.clip(u, 1e-8, 1 - 1e-8), df=nu)
    try:
        Rinv = np.linalg.inv(R); logdet = np.linalg.slogdet(R)[1]
    except np.linalg.LinAlgError:
        return -1e10
    log_const = (gammaln((nu + d) / 2) - gammaln(nu / 2)
                 - 0.5 * d * np.log(np.pi * nu) - 0.5 * logdet)
    quad = np.einsum("ti,ij,tj->t", x, Rinv, x)
    log_mvt  = log_const - 0.5 * (nu + d) * np.log1p(quad / nu)
    log_marg = stats.t.logpdf(x, df=nu).sum(axis=1)
    return float((log_mvt - log_marg).sum())


def _refit_t_copula_fast(u: np.ndarray, nu_grid: list = None) -> tuple[np.ndarray, float]:
    """Fast Student-t refit: R via Kendall, nu via grid search."""
    if nu_grid is None:
        nu_grid = cfg.NU_GRID
    R = nearest_pd(tau_to_rho(kendall_tau_matrix(u)))
    lls = [_t_copula_grid_loglik(u, R, nu) for nu in nu_grid]
    nu_hat = float(nu_grid[int(np.argmax(lls))])
    return R, nu_hat


def tier_b_bootstrap(uniforms: pd.DataFrame, marginals: dict, sigma2_uncond: dict,
                      n_boot: int = cfg.N_BOOT_TIER_B,
                      n_paths: int = cfg.N_PATHS_BOOT_TIER_B,
                      seed: int = cfg.SEED_BOOT_TIER_B,
                      verbose: bool = True) -> dict:
    """Iid row bootstrap on PIT uniforms, refit Student-t copula, resimulate.
    Captures parameter (dependence) uncertainty conditional on marginals."""
    U = uniforms[cfg.ASSETS].values
    T_obs = len(U)
    rng = np.random.default_rng(seed)

    sigmas_arr = np.array([np.sqrt(sigma2_uncond[a]) for a in cfg.ASSETS])

    var_b = np.empty(n_boot); es_b = np.empty(n_boot); nu_b = np.empty(n_boot)
    pairs = list(combinations(range(len(cfg.ASSETS)), 2))
    R_b = np.empty((n_boot, len(pairs)))
    n_fail = 0
    t0 = time.time()

    for b in range(n_boot):
        try:
            idx = rng.integers(0, T_obs, size=T_obs)
            U_b = U[idx]
            R, nu = _refit_t_copula_fast(U_b)

            # Sample n_paths * horizon from t-copula
            n_total = n_paths * cfg.HORIZON_QUARTERS
            L = np.linalg.cholesky(R)
            Z = rng.standard_normal((n_total, len(cfg.ASSETS)))
            W = rng.chisquare(df=nu, size=n_total)
            X = (Z @ L.T) * np.sqrt(nu / W)[:, None]
            U_sim = stats.t.cdf(X, df=nu)

            # PPF -> z -> r (constant sigma per asset; EWMA degenerates on quarterly data)
            Z_marg = np.empty_like(U_sim)
            for i, a in enumerate(cfg.ASSETS):
                Z_marg[:, i] = marginals[a]["dist"].ppf(U_sim[:, i])

            R_ret = MU_Q_LOG[None, :] + sigmas_arr[None, :] * Z_marg
            R_ret = R_ret.reshape(n_paths, cfg.HORIZON_QUARTERS, len(cfg.ASSETS))
            port_q = R_ret @ WEIGHTS_ARR
            losses = -port_q.sum(axis=1)

            v = float(np.quantile(losses, cfg.ALPHA))
            e = float(losses[losses >= v].mean())
            var_b[b] = v; es_b[b] = e; nu_b[b] = nu
            for k, (i, j) in enumerate(pairs):
                R_b[b, k] = R[i, j]

        except Exception:
            var_b[b] = es_b[b] = nu_b[b] = np.nan
            R_b[b, :] = np.nan
            n_fail += 1

        if verbose and ((b + 1) % max(1, n_boot // 20) == 0 or b == 0):
            elapsed = time.time() - t0
            eta = elapsed / (b + 1) * (n_boot - b - 1)
            print(f"    [{b+1:>4}/{n_boot}]  ES so far = "
                  f"{np.nanmean(es_b[:b+1])*100:.2f}%  ν = "
                  f"{np.nanmean(nu_b[:b+1]):.2f}  "
                  f"elapsed {elapsed/60:.1f} min, ETA {eta/60:.1f} min")

    valid = ~np.isnan(es_b)
    pair_labels = [f"{cfg.ASSETS[i]}-{cfg.ASSETS[j]}" for i, j in pairs]

    return {
        "var_samples":   var_b,
        "es_samples":    es_b,
        "nu_samples":    nu_b,
        "R_samples":     R_b,
        "pair_labels":   pair_labels,
        "n_valid":       int(valid.sum()),
        "n_failed":      int(n_fail),
        "var_ci":        (float(np.percentile(var_b[valid], 2.5)), float(np.percentile(var_b[valid], 97.5))),
        "es_ci":         (float(np.percentile(es_b[valid], 2.5)),  float(np.percentile(es_b[valid], 97.5))),
        "nu_ci":         (float(np.percentile(nu_b[valid], 2.5)),  float(np.percentile(nu_b[valid], 97.5))),
        "var_mean":      float(np.nanmean(var_b)),
        "es_mean":       float(np.nanmean(es_b)),
        "nu_mean":       float(np.nanmean(nu_b)),
    }


def wald_ci(point: float, lo: float, hi: float) -> tuple[float, float]:
    """Convert a percentile CI to a symmetric Wald CI centered on the point estimate."""
    se = (hi - lo) / (2 * 1.96)
    return point - 1.96 * se, point + 1.96 * se
