"""End-to-end pipeline runner for the FE5211 CVaR project.

Usage:
    python main.py                 # full pipeline (default)
    python main.py --stage data    # data only
    python main.py --stage all --quick   # 100k paths instead of 1M for fast iteration
    python main.py --no-vine       # skip R-Vine copula even if installed
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config as cfg
from src.style import apply_style
from src import (data_loader, unsmoothing, marginals as marg,
                  copula as cop, simulation as sim, analysis as ana,
                  plots as fig)


def _section(title: str):
    print("\n" + "=" * 78); print(f"  {title}"); print("=" * 78)


def _write(p: Path, df: pd.DataFrame, **kwargs):
    df.to_csv(p, **kwargs); print(f"  saved {p.relative_to(cfg.ROOT_DIR)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                         choices=["data", "unsmooth", "marginal", "copula",
                                  "simulate", "analyze", "plot", "all"])
    parser.add_argument("--quick", action="store_true",
                         help="Use 100k paths and 100 Tier-B bootstraps for fast iteration.")
    parser.add_argument("--no-vine", action="store_true")
    parser.add_argument("--force", action="store_true",
                         help="Refit everything from scratch (ignore cache).")
    args = parser.parse_args()

    apply_style()
    n_paths = 100_000 if args.quick else cfg.N_PATHS_MC
    # Quick mode: 10 Tier-B boots * 20k paths each (~30s) instead of 500*100k (~25 min)
    n_boot_b = 10 if args.quick else cfg.N_BOOT_TIER_B
    n_paths_boot_b = 20_000 if args.quick else cfg.N_PATHS_BOOT_TIER_B
    fit_vine = (not args.no_vine) and cop.HAS_VINECOPULIB

    numbers: dict = {}
    t_start = time.time()

    _section("STAGE 1 — Data")
    panel = data_loader.load_panel()
    returns = data_loader.build_or_load_returns(force=args.force)
    desc = data_loader.descriptive_stats(returns)
    print(desc[["ann_ret", "ann_vol", "skew", "ex_kurt", "ar1"]].round(3))
    _write(cfg.TBL_DIR / "stats_descriptive.csv", desc)
    for a in cfg.ASSETS:
        numbers[f"ar1_{a}"]      = float(desc.loc[a, "ar1"])
        numbers[f"ann_vol_{a}"]  = float(desc.loc[a, "ann_vol"])
    if args.stage == "data": return

    _section("STAGE 2 — Unsmoothing")
    unsm = unsmoothing.build_or_load_unsmoothed(returns, force=args.force)
    vol_t = unsmoothing.vol_table(unsm, returns)
    print((vol_t.drop(columns="proxy_name") * 100).round(2))
    _write(cfg.TBL_DIR / "unsmoothing_vol.csv", vol_t)
    pe_kf = unsmoothing.kalman_filter(returns["PE"].dropna())
    npi_kf = unsmoothing.kalman_filter(returns["NPI"].dropna())
    numbers["pe_kf_alpha"]   = float(pe_kf.params["alpha"])
    numbers["pe_kf_ma1"]     = float(pe_kf.params["ma1"])
    numbers["pe_kf_vol"]     = float(pe_kf.unsmoothed_vol_ann * 100)
    numbers["npi_kf_alpha"]  = float(npi_kf.params["alpha"])
    numbers["npi_kf_ma1"]    = float(npi_kf.params["ma1"])
    numbers["npi_kf_vol"]    = float(npi_kf.unsmoothed_vol_ann * 100)
    if args.stage == "unsmooth": return

    _section("STAGE 3 — Marginals (EWMA + Skew-t / Semi-parametric)")
    marginals, residuals, uniforms, sigma2 = marg.build_or_load_marginals(
        returns, unsm, force=args.force)
    rows = {a: marginals[a]["diagnostics"] for a in cfg.ASSETS}
    diag_df = pd.DataFrame(rows).T
    print(diag_df[["resid_skew", "resid_kurt", "ks_p", "lb_z2_p4", "eta", "lambda_skew"]].round(3))
    _write(cfg.TBL_DIR / "marginal_diagnostics.csv", diag_df)
    for a in cfg.ASSETS:
        numbers[f"lambda_{a}"]    = float(marginals[a]["lambda"])
        numbers[f"ks_p_{a}"]      = float(marginals[a]["diagnostics"]["ks_p"])
    numbers["spy_eta"]      = float(marginals["SPY"]["diagnostics"]["eta"])
    numbers["spy_lambda_sk"] = float(marginals["SPY"]["diagnostics"]["lambda_skew"])
    numbers["agg_eta"]      = float(marginals["AGG"]["diagnostics"]["eta"])
    numbers["agg_lambda_sk"] = float(marginals["AGG"]["diagnostics"]["lambda_skew"])
    if args.stage == "marginal": return

    _section("STAGE 4 — Copulas (Gaussian, Student-t, optional R-Vine)")
    print(f"  fit_vine = {fit_vine}  (HAS_VINECOPULIB = {cop.HAS_VINECOPULIB})")
    copulas = cop.build_or_load_copulas(uniforms, fit_vine=fit_vine, force=args.force)
    print(copulas["comparison"].round(3))
    _write(cfg.TBL_DIR / "copula_comparison.csv", copulas["comparison"])
    t_full = copulas["full"]["Student-t"]
    pair_idx = {"SPY": 0, "AGG": 1, "PE": 2, "NPI": 3}
    numbers["champion"]      = copulas["champion"]
    numbers["t_nu"]          = float(t_full.nu)
    numbers["R_SPY_PE"]      = float(t_full.R[pair_idx["SPY"], pair_idx["PE"]])
    numbers["R_SPY_NPI"]     = float(t_full.R[pair_idx["SPY"], pair_idx["NPI"]])
    numbers["R_PE_NPI"]      = float(t_full.R[pair_idx["PE"], pair_idx["NPI"]])
    numbers["R_AGG_PE"]      = float(t_full.R[pair_idx["AGG"], pair_idx["PE"]])
    # Vuong: t vs Gaussian
    in_t  = copulas["in_sample"]["Student-t"]
    in_g  = copulas["in_sample"]["Gaussian"]
    z, p_v = cop.vuong_test(
        in_t.per_obs_loglik(copulas["u_oos"]),
        in_g.per_obs_loglik(copulas["u_oos"]),
        in_t.n_params(), in_g.n_params())
    numbers["vuong_t_vs_g_z"] = float(z); numbers["vuong_t_vs_g_p"] = float(p_v)
    if "R-Vine" in copulas["in_sample"]:
        v_in = copulas["in_sample"]["R-Vine"]
        z2, p2 = cop.vuong_test(
            v_in.per_obs_loglik(copulas["u_oos"]),
            in_t.per_obs_loglik(copulas["u_oos"]),
            v_in.n_params(), in_t.n_params())
        numbers["vuong_v_vs_t_z"] = float(z2); numbers["vuong_v_vs_t_p"] = float(p2)
    if args.stage == "copula": return

    _section("STAGE 5 — Simulation (4 methods × 2 data treatments)")
    cache_paths = cfg.CACHE_DIR / "es_matrix_results.npz"
    if cache_paths.exists() and not args.force:
        with np.load(cache_paths, allow_pickle=True) as d:
            sim_results = {
                "es_matrix":  pd.DataFrame(d["es_matrix"], index=d["methods"],
                                            columns=d["versions"]),
                "var_matrix": pd.DataFrame(d["var_matrix"], index=d["methods"],
                                             columns=d["versions"]),
                "loss_arrays": {tuple(k.split("__")): d[k] for k in d.files
                                 if "__" in k},
                "paths_champion": d["paths_champion"],
                "sigma2_uncond":  json.loads(str(d["sigma2_uncond"])),
            }
            sim_results["unsmoothed_panel"] = pd.read_parquet(
                cfg.CACHE_DIR / "unsmoothed_panel.parquet")
            sim_results["reported_panel"]   = pd.read_parquet(
                cfg.CACHE_DIR / "reported_panel.parquet")
        print(f"  (cached, {n_paths:,} paths)")
    else:
        print(f"  Running {n_paths:,} paths...")
        t0 = time.time()
        sim_results = sim.build_es_matrix(returns, unsm, marginals, residuals,
                                            t_full, n_paths=n_paths, save_paths=True)
        print(f"  done in {time.time() - t0:.1f}s")
        # Cache
        save_dict = {
            "es_matrix":   sim_results["es_matrix"].values.astype(float),
            "var_matrix":  sim_results["var_matrix"].values.astype(float),
            "methods":     np.array(sim_results["es_matrix"].index),
            "versions":    np.array(sim_results["es_matrix"].columns),
            "paths_champion": sim_results["paths_champion"],
            "sigma2_uncond":  json.dumps(sim_results["sigma2_uncond"]),
        }
        for (m, v), arr in sim_results["loss_arrays"].items():
            if arr is not None:
                save_dict[f"{m}__{v}"] = arr.astype(np.float32)
        np.savez_compressed(cache_paths, **save_dict)
        sim_results["unsmoothed_panel"].to_parquet(cfg.CACHE_DIR / "unsmoothed_panel.parquet")
        sim_results["reported_panel"].to_parquet(cfg.CACHE_DIR / "reported_panel.parquet")
    print("\nVaR 95% (3Y, %):")
    print((sim_results["var_matrix"] * 100).round(2).to_string(na_rep="—"))
    print("\nES 95% (3Y, %):")
    print((sim_results["es_matrix"] * 100).round(2).to_string(na_rep="—"))
    _write(cfg.TBL_DIR / "es_matrix.csv",  sim_results["es_matrix"])
    _write(cfg.TBL_DIR / "var_matrix.csv", sim_results["var_matrix"])

    em = sim_results["es_matrix"]
    numbers.update({
        "es_param_reported":   float(em.loc["Parametric", "reported"]   * 100),
        "es_param_unsmoothed": float(em.loc["Parametric", "unsmoothed"] * 100),
        "es_hs_reported":      float(em.loc["HS", "reported"]   * 100),
        "es_hs_unsmoothed":    float(em.loc["HS", "unsmoothed"] * 100),
        "es_fhs_unsmoothed":   float(em.loc["FHS", "unsmoothed"] * 100),
        "es_champion":         float(em.loc["Champion", "unsmoothed"] * 100),
        "var_champion":        float(sim_results["var_matrix"].loc["Champion", "unsmoothed"] * 100),
    })
    if args.stage == "simulate": return

    _section("STAGE 6 — Decomposition + Validation")
    paths = sim_results["paths_champion"].astype(np.float64)
    champion_losses = sim_results["loss_arrays"][("Champion", "unsmoothed")]

    # Component CVaR
    comp = ana.component_cvar(paths)
    print("\nComponent CVaR:")
    print(comp["table"].round(4))
    _write(cfg.TBL_DIR / "component_cvar.csv", comp["table"])
    for a in cfg.ASSETS:
        numbers[f"comp_{a}_pct"] = float(comp["table"].loc[a, "comp_pct_of_total"])
        numbers[f"comp_{a}"]     = float(comp["table"].loc[a, "comp_cvar"] * 100)
    numbers["es_total_recomputed"] = float(comp["es_total"] * 100)

    # Time-of-loss
    worst_idx = int(np.argmax(comp["port_losses"]))
    tl_df = ana.time_of_loss(paths, comp["tail_mask"], worst_idx)
    _write(cfg.TBL_DIR / "time_of_loss.csv", tl_df, index=False)

    # Historical replay
    tail = ana.historical_tail_windows(sim_results["unsmoothed_panel"])
    replay_df = ana.historical_replay(tail["combined"], champion_losses)
    print("\nHistorical replay:")
    print(replay_df[["label", "port_loss", "mc_percentile"]].round(3).to_string(index=False))
    _write(cfg.TBL_DIR / "historical_replay.csv", replay_df, index=False)
    for _, r in replay_df.iterrows():
        numbers[f"replay_{r['label']}_loss"] = float(r["port_loss"] * 100)
        numbers[f"replay_{r['label']}_pct"]  = float(r["mc_percentile"])

    # In-sample backtest
    backtest = ana.in_sample_backtest(sim_results["unsmoothed_panel"],
                                        sigma2, marginals)
    print("\nBacktest:")
    print(backtest["table"].round(4).to_string(index=False))
    _write(cfg.TBL_DIR / "backtest.csv", backtest["table"], index=False)
    p = backtest["table"]
    numbers["backtest_n_breach"]    = int(backtest["n_breach"])
    numbers["backtest_expected"]    = float(backtest["expected_breach"])
    numbers["backtest_kupiec_p"]    = float(p[p["test"] == "Kupiec POF"]["p_value"].iloc[0])
    numbers["backtest_chris_p"]     = float(p[p["test"] == "Christoffersen Ind"]["p_value"].iloc[0])
    numbers["backtest_acerbi_z"]    = float(p[p["test"] == "Acerbi-Szekely Z"]["stat"].iloc[0])

    # Stress
    stress_df = ana.stress_scenarios(tail["combined"], es_baseline=comp["es_total"])
    print("\nStress:")
    print(stress_df[["scenario", "stressed_port_loss", "pct_of_baseline_es"]].round(3).to_string(index=False))
    _write(cfg.TBL_DIR / "stress.csv", stress_df, index=False)
    for _, r in stress_df.iterrows():
        key = r["scenario"].split("(")[0].strip().replace(" ", "_").replace("x", "x")
        numbers[f"stress_{key}_loss"] = float(r["stressed_port_loss"] * 100)

    # Reverse stress
    rev = ana.reverse_stress(paths, comp["port_losses"], es_baseline=comp["es_total"])
    print(f"\nReverse stress (target = {rev['target_loss']*100:.1f}%, n_deep = {rev['n_deep_tail']:,}):")
    print(rev["table"].round(3))
    _write(cfg.TBL_DIR / "reverse_stress.csv", rev["table"])
    numbers["reverse_target"]   = float(rev["target_loss"] * 100)
    numbers["reverse_n_deep"]   = int(rev["n_deep_tail"])
    for a in cfg.ASSETS:
        numbers[f"reverse_{a}_loss"] = float(rev["table"].loc[a, "deep_tail_3y_loss"] * 100)

    # Tier A bootstrap
    print("\n  Running Tier A bootstrap (MC noise)...")
    t0 = time.time()
    tier_a = ana.tier_a_bootstrap(champion_losses)
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"    VaR CI: [{tier_a['var_ci'][0]*100:.3f}%, {tier_a['var_ci'][1]*100:.3f}%]")
    print(f"    ES  CI: [{tier_a['es_ci'][0]*100:.3f}%, {tier_a['es_ci'][1]*100:.3f}%]")
    numbers["tier_a_var_ci_lo"] = float(tier_a["var_ci"][0] * 100)
    numbers["tier_a_var_ci_hi"] = float(tier_a["var_ci"][1] * 100)
    numbers["tier_a_es_ci_lo"]  = float(tier_a["es_ci"][0]  * 100)
    numbers["tier_a_es_ci_hi"]  = float(tier_a["es_ci"][1]  * 100)

    # Tier B bootstrap
    print(f"\n  Running Tier B bootstrap ({n_boot_b} iterations × {n_paths_boot_b:,} paths each)...")
    t0 = time.time()
    s2_for_boot = sim.sigma2_uncond_for_simulation(marginals,
                                                      sim_results["unsmoothed_panel"])
    tier_b = ana.tier_b_bootstrap(uniforms, marginals, s2_for_boot,
                                    n_boot=n_boot_b,
                                    n_paths=n_paths_boot_b,
                                    seed=cfg.SEED_BOOT_TIER_B,
                                    verbose=True)
    print(f"  done in {(time.time()-t0)/60:.1f} min")
    point_var = sim_results["var_matrix"].loc["Champion", "unsmoothed"]
    point_es  = sim_results["es_matrix"].loc["Champion", "unsmoothed"]
    var_w_lo, var_w_hi = ana.wald_ci(point_var, *tier_b["var_ci"])
    es_w_lo,  es_w_hi  = ana.wald_ci(point_es,  *tier_b["es_ci"])
    print(f"    VaR Wald CI: [{var_w_lo*100:.2f}%, {var_w_hi*100:.2f}%]")
    print(f"    ES  Wald CI: [{es_w_lo*100:.2f}%, {es_w_hi*100:.2f}%]")
    print(f"    nu mean = {tier_b['nu_mean']:.3f}")
    numbers.update({
        "tier_b_var_pct_ci_lo":  float(tier_b["var_ci"][0] * 100),
        "tier_b_var_pct_ci_hi":  float(tier_b["var_ci"][1] * 100),
        "tier_b_es_pct_ci_lo":   float(tier_b["es_ci"][0]  * 100),
        "tier_b_es_pct_ci_hi":   float(tier_b["es_ci"][1]  * 100),
        "tier_b_es_wald_ci_lo":  float(es_w_lo * 100),
        "tier_b_es_wald_ci_hi":  float(es_w_hi * 100),
        "tier_b_var_wald_ci_lo": float(var_w_lo * 100),
        "tier_b_var_wald_ci_hi": float(var_w_hi * 100),
        "tier_b_nu_mean":        float(tier_b["nu_mean"]),
        "tier_b_n_valid":        int(tier_b["n_valid"]),
    })

    if args.stage == "analyze":
        _write_numbers(numbers)
        return

    _section("STAGE 7 — Plots (9 main + 16 appendix)")
    F = cfg.FIG_DIR; A = cfg.APP_DIR

    # Main figures
    fig.fig_three_pathologies(returns, unsm,                  F / "01_three_pathologies.png")
    fig.fig_unsmoothing_comparison(unsm, returns,             F / "02_unsmoothing_comparison.png")
    fig.fig_tail_dependence(uniforms, copulas,                F / "03_tail_dependence.png")
    fig.fig_es_matrix(sim_results["es_matrix"],               F / "04_es_matrix.png")
    fig.fig_component_cvar(comp["table"], comp["es_total"],   F / "05_component_cvar.png")
    fig.fig_path_sanity_check(tail["named"], sim_results["unsmoothed_panel"],
                                paths, comp["tail_mask"],        F / "06_path_sanity_check.png")
    fig.fig_historical_replay(replay_df, champion_losses, comp["es_total"],
                                                                F / "07_historical_replay.png")
    fig.fig_reverse_stress(rev, comp["es_total"],             F / "08_reverse_stress.png")
    fig.fig_bootstrap_forest(point_var, point_es, tier_a, tier_b,
                                                                F / "09_bootstrap_forest.png")
    print("  9 main figures written.")

    # Appendix
    fig.app_eda_wealth(panel,                                  A / "A01_eda_wealth.png")
    fig.app_eda_acf(returns,                                   A / "A02_eda_acf.png")
    fig.app_eda_rolling_vol(returns,                           A / "A03_eda_rolling_vol.png")
    fig.app_eda_rolling_corr(returns,                          A / "A04_eda_rolling_corr.png")
    fig.app_eda_qq(returns,                                    A / "A05_eda_qq.png")
    fig.app_eda_chiplot(returns,                               A / "A06_eda_chiplot.png")
    fig.app_unsmoothing_drawdown(unsm, returns,                A / "A07_unsmoothing_drawdown.png")
    fig.app_marg_pit_hist(marginals,                           A / "A08_marg_pit_hist.png")
    fig.app_marg_qq(marginals,                                 A / "A09_marg_qq.png")
    fig.app_marg_ewma(marginals,                               A / "A10_marg_ewma.png")
    fig.app_copula_loglik(copulas["comparison"],               A / "A11_copula_loglik.png")
    fig.app_copula_pair_scatter(uniforms, copulas,             A / "A12_copula_pair_scatter.png")
    fig.app_sim_loss_distribution(sim_results["loss_arrays"],
                                     sim_results["es_matrix"],    A / "A13_sim_loss_distribution.png")
    fig.app_sim_path_examples(paths, comp["port_losses"],      A / "A14_sim_path_examples.png")
    fig.app_decomp_time_loss(tl_df,                            A / "A15_decomp_time_loss.png")
    fig.app_validation_backtest_ts(backtest,                   A / "A16_validation_backtest_ts.png")
    fig.app_validation_stress(stress_df, comp["es_total"],     A / "A17_validation_stress.png")
    fig.app_bootstrap_distributions(tier_a, tier_b, point_var, point_es,
                                                                A / "A18_bootstrap_distributions.png")
    fig.app_bootstrap_param(tier_b,                            A / "A19_bootstrap_param.png")
    print("  19 appendix figures written.")

    _write_numbers(numbers)
    print(f"\nTotal pipeline time: {(time.time() - t_start) / 60:.1f} min")


def _write_numbers(numbers: dict):
    out = cfg.OUTPUT_DIR / "numbers.json"
    with open(out, "w") as f:
        json.dump(numbers, f, indent=2)
    print(f"\n  saved {out.relative_to(cfg.ROOT_DIR)}  ({len(numbers)} keys)")


if __name__ == "__main__":
    main()
