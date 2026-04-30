"""All figures: 9 main (output/figures/) + 16 appendix (output/figures/appendix/).

Each function takes the data it needs and writes a PNG. Style is set globally
in src/style.py — the only colors used are deep red, red-brown, gray, black,
white. No "Figure N" prefix in titles.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from . import config as cfg
from .style import PALETTE, SEQ_CMAP, DIV_CMAP


CRISIS_SHADE = [
    ("2007-12-31", "2009-06-30"),
    ("2020-03-31", "2020-06-30"),
    ("2022-03-31", "2022-12-31"),
]


def _shade_crises(ax):
    for s, e in CRISIS_SHADE:
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   color=PALETTE["light"], alpha=0.5, zorder=0)


# =============================================================================
#  MAIN FIGURE 01 — Three Data Pathologies (S1.3)
# =============================================================================
def fig_three_pathologies(returns: pd.DataFrame, unsmoothed: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # Panel A: smoothing — AR(1) bar chart
    ax = axes[0]
    ar1 = {a: returns[a].dropna().autocorr(1) for a in cfg.ASSETS}
    bars = ax.bar(list(ar1.keys()), list(ar1.values()),
                  color=[PALETTE["primary"] if v > 0.5 else PALETTE["muted"]
                          for v in ar1.values()],
                  edgecolor=PALETTE["tertiary"], linewidth=0.6)
    for b, v in zip(bars, ar1.values()):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", fontsize=9,
                color=PALETTE["tertiary"], fontweight="bold")
    ax.axhline(0, color=PALETTE["tertiary"], lw=0.6)
    ax.axhline(0.5, color=PALETTE["muted"], ls="--", lw=0.7)
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel("AR(1) autocorrelation")
    ax.set_title("Pathology 1: Private-asset smoothing")

    # Panel B: tail asymmetry chi_L vs chi_U for SPY-NPI
    ax = axes[1]
    spy_npi = returns[["SPY", "NPI"]].dropna()
    n = len(spy_npi)
    rx = (spy_npi["SPY"].rank() - 0.5) / n
    ry = (spy_npi["NPI"].rank() - 0.5) / n
    qs = np.linspace(0.05, 0.95, 19)
    chi = []
    for q in qs:
        if q < 0.5:
            mask = rx < q
            chi.append(np.mean((rx < q) & (ry < q)) / max(mask.mean(), 1e-9))
        else:
            mask = rx > q
            chi.append(np.mean((rx > q) & (ry > q)) / max(mask.mean(), 1e-9))
    ax.plot(qs, chi, "o-", color=PALETTE["primary"], lw=1.5, ms=4)
    ax.axvline(0.5, color=PALETTE["muted"], ls="--", lw=0.7)
    ax.fill_between([0, 0.5], 0, 1, color=PALETTE["primary"], alpha=0.05)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("Quantile $u$")
    ax.set_ylabel(r"Empirical $\chi(u)$")
    ax.set_title("Pathology 2: Asymmetric tail dependence (SPY-NPI)")
    ax.text(0.10, 0.92, "Lower-tail\nstrong", color=PALETTE["primary"],
            fontsize=9, fontweight="bold", verticalalignment="top")
    ax.text(0.72, 0.92, "Upper-tail\nweak", color=PALETTE["muted"],
            fontsize=9, fontweight="bold", verticalalignment="top")

    # Panel C: long horizon vs short sample
    ax = axes[2]
    n_obs = 80
    horiz = cfg.HORIZON_QUARTERS
    n_indep_3y = n_obs // horiz
    info = (f"Sample size:           n = {n_obs} quarters\n"
            f"Horizon length:        H = {horiz} quarters (3 yrs)\n\n"
            f"Number of disjoint     n / H = {n_indep_3y}\n"
            f"3-year windows:\n\n"
            f"Expected tail count    {n_indep_3y} × 0.05\n"
            f"at 95% quantile:       = {n_indep_3y * 0.05:.1f} observations")
    ax.text(0.05, 0.95, info, transform=ax.transAxes, fontsize=10,
            family="monospace", verticalalignment="top",
            color=PALETTE["tertiary"])
    ax.text(0.05, 0.18,
            "→ Empirical historical\n"
            "   estimation infeasible.\n"
            "   Simulation required.",
            transform=ax.transAxes, fontsize=10,
            color=PALETTE["primary"], fontweight="bold",
            verticalalignment="top")
    ax.axis("off")
    ax.set_title("Pathology 3: Long horizon, short sample")

    fig.suptitle("Three Data Pathologies Motivating the Model",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
#  MAIN FIGURE 02 — Unsmoothing Comparison (S2.2)
# =============================================================================
def fig_unsmoothing_comparison(unsm: pd.DataFrame, returns: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    methods = ["raw", "FGW", "OW", "KF"]

    for ax, asset, proxy in [(axes[0], "PE", "LPX50"), (axes[1], "NPI", "RMZ")]:
        vols = []
        for m in methods:
            s = unsm[f"{asset}_{m}"].dropna()
            vols.append(float(s.std() * np.sqrt(cfg.QUARTERS_PER_YEAR) * 100))
        proxy_vol = float(returns[proxy].dropna().std() *
                          np.sqrt(cfg.QUARTERS_PER_YEAR) * 100)
        labels = ["Raw", "FGW", "OW", "KF (selected)", f"{proxy}\n(listed proxy)"]
        all_vols = vols + [proxy_vol]
        colors = [PALETTE["muted"], PALETTE["secondary"], PALETTE["secondary"],
                   PALETTE["primary"], PALETTE["tertiary"]]
        bars = ax.bar(labels, all_vols, color=colors,
                       edgecolor=PALETTE["tertiary"], linewidth=0.6)
        for b, v in zip(bars, all_vols):
            ax.text(b.get_x() + b.get_width() / 2, v + max(all_vols) * 0.01,
                    f"{v:.1f}%", ha="center", fontsize=10,
                    fontweight="bold", color=PALETTE["tertiary"])
        ax.set_ylabel("Annualized volatility (%)")
        ax.set_title(f"{asset}: Reported vs Three Unsmoothing Methods vs Listed Proxy")
        ax.grid(True, axis="y", alpha=0.4)
        ax.set_ylim(0, max(all_vols) * 1.15)

    fig.suptitle("Unsmoothing Private-Asset Returns: FGW vs OW vs Kalman Filter",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
#  MAIN FIGURE 03 — Tail Dependence (S2.3)
# =============================================================================
def fig_tail_dependence(uniforms: pd.DataFrame, copulas: dict, path):
    """Empirical chi_L / chi_U at q=0.10 vs Student-t implied lambda_L.
    The story: empirical chi_L can be much larger than chi_U for risk-relevant
    pairs (SPY-NPI, SPY-PE), and the elliptical Student-t under-estimates chi_L."""
    from .copula import empirical_chi_L, empirical_chi_U
    U = uniforms.values
    emp_L = empirical_chi_L(U, q=0.10)
    emp_U = empirical_chi_U(U, q=0.10)
    lam_t = copulas["full"]["Student-t"].lambda_L()

    pairs = list(combinations(range(len(cfg.ASSETS)), 2))
    pair_labels = [f"{cfg.ASSETS[i]}-{cfg.ASSETS[j]}" for i, j in pairs]
    emp_L_v = np.array([emp_L[i, j] for i, j in pairs])
    emp_U_v = np.array([emp_U[i, j] for i, j in pairs])
    lam_t_v = np.array([lam_t[i, j] for i, j in pairs])

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(pairs))
    w = 0.27   # 3 bars * 0.27 = 0.81 < 1, no overlap with neighbours

    b1 = ax.bar(x - w, emp_L_v, w,
                 label=r"Empirical $\chi_L$ (lower tail, q=0.10)",
                 color=PALETTE["primary"], edgecolor=PALETTE["tertiary"], linewidth=0.5)
    b2 = ax.bar(x,       emp_U_v, w,
                 label=r"Empirical $\chi_U$ (upper tail, q=0.10)",
                 color=PALETTE["secondary"], edgecolor=PALETTE["tertiary"], linewidth=0.5)
    b3 = ax.bar(x + w,   lam_t_v, w,
                 label=r"Student-t implied $\lambda_L$",
                 color=PALETTE["tertiary"], edgecolor=PALETTE["tertiary"], linewidth=0.5)

    # Number labels on top of every bar
    for bars, vals in [(b1, emp_L_v), (b2, emp_U_v), (b3, lam_t_v)]:
        for bar, v in zip(bars, vals):
            if v > 0.005:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=8, color=PALETTE["tertiary"])

    # Annotation: explicit reminder that Gaussian implied lambda_L = 0
    ax.axhline(0, color=PALETTE["muted"], lw=0.6,
                label=r"Gaussian implied $\lambda_L \equiv 0$")

    # Highlight the most asymmetric pair (SPY-NPI is canonical exhibit)
    if "SPY-NPI" in pair_labels:
        i_npi = pair_labels.index("SPY-NPI")
        ax.annotate(r"Strong $\chi_L$, zero $\chi_U$" "\n"
                    "(asymmetric tail dependence)",
                    xy=(i_npi - w, emp_L_v[i_npi]),
                    xytext=(i_npi - 0.3, max(emp_L_v) * 1.45),
                    fontsize=9, color=PALETTE["primary"], fontweight="bold",
                    ha="center",
                    arrowprops=dict(arrowstyle="->", color=PALETTE["primary"],
                                     lw=0.8, connectionstyle="arc3,rad=0.2"))

    ax.set_xticks(x); ax.set_xticklabels(pair_labels, rotation=15)
    ax.set_ylabel("Tail dependence")
    ax.set_title("Empirical vs Model-Implied Tail Dependence (Pairs at q=0.10)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    y_max = max(max(emp_L_v), max(emp_U_v), max(lam_t_v)) * 1.7
    ax.set_ylim(-0.02, y_max)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
#  MAIN FIGURE 04 — ES Matrix Heatmap (S3.1)
# =============================================================================
def fig_es_matrix(es_matrix: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    mat = (es_matrix.values * 100)

    # Build a masked array so NaN cells render as the colormap "bad" color
    import numpy.ma as ma
    masked = ma.masked_invalid(mat)
    cmap = plt.get_cmap(SEQ_CMAP).copy()
    cmap.set_bad(color=PALETTE["light"])  # light gray for N/A

    im = ax.imshow(masked, cmap=cmap, aspect="auto",
                    vmin=np.nanmin(mat) * 0.85, vmax=np.nanmax(mat))

    ax.set_xticks(range(es_matrix.shape[1]))
    ax.set_xticklabels([c.capitalize() for c in es_matrix.columns], fontsize=11)
    ax.set_yticks(range(es_matrix.shape[0]))
    ax.set_yticklabels(es_matrix.index, fontsize=11)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if np.isnan(v):
                ax.text(j, i, "N/A\n(by design)", ha="center", va="center",
                        color=PALETTE["muted"], fontsize=10, style="italic")
            else:
                color = "white" if v > (np.nanmin(mat) + np.nanmax(mat)) * 0.55 \
                                  else PALETTE["tertiary"]
                ax.text(j, i, f"{v:.1f}%", ha="center", va="center",
                        color=color, fontsize=14, fontweight="bold")
    ax.set_title("3-Year 95% Expected Shortfall: Methods × Data Treatments")
    plt.colorbar(im, ax=ax, label="ES (%)", shrink=0.8)
    plt.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
#  MAIN FIGURE 05 — Component CVaR (S4.1)
# =============================================================================
def fig_component_cvar(comp_table: pd.DataFrame, es_total: float, path):
    fig, ax = plt.subplots(figsize=(11, 5.8))
    weights_pct  = comp_table["weight"].values * 100      # left column: 40/20/25/15
    comp_pct     = comp_table["comp_pct_of_total"].values  # right column: 86.8/-11.5/18.5/6.2

    width = 0.55
    # ----- LEFT column: capital weights (always positive, stack 0 -> 100) -----
    bottom_left = 0.0
    for i, a in enumerate(cfg.ASSETS):
        v = weights_pct[i]
        ax.bar(0, v, width, bottom=bottom_left, color=PALETTE[a],
                edgecolor="white", linewidth=1.2, label=a)
        ax.text(0, bottom_left + v / 2, f"{a}\n{v:.1f}%",
                ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")
        bottom_left += v

    # ----- RIGHT column: comp CVaR (positives stack up from 0, negatives down) -----
    bottom_pos = 0.0
    bottom_neg = 0.0
    for i, a in enumerate(cfg.ASSETS):
        v = comp_pct[i]
        if v >= 0:
            ax.bar(1, v, width, bottom=bottom_pos, color=PALETTE[a],
                    edgecolor="white", linewidth=1.2)
            ax.text(1, bottom_pos + v / 2, f"{a}\n{v:.1f}%",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white")
            bottom_pos += v
        else:
            ax.bar(1, v, width, bottom=bottom_neg, color=PALETTE[a],
                    edgecolor="white", linewidth=1.2, alpha=0.85,
                    hatch="//")
            ax.text(1, bottom_neg + v / 2, f"{a}\n{v:.1f}%",
                    ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white")
            bottom_neg += v

    # Reference line at 100% (the additivity target on the right column)
    ax.axhline(100, color=PALETTE["muted"], ls=":", lw=0.8, alpha=0.7)
    ax.axhline(0, color=PALETTE["tertiary"], lw=0.8)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Capital Weight", "Risk Contribution\n(Component CVaR)"],
                        fontsize=11)
    ax.set_ylabel("Share of total (%)")
    y_max = max(105, bottom_pos * 1.05)
    y_min = min(-15, bottom_neg * 1.2)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Component CVaR Decomposition by Asset"
                 f"   (Total ES = {es_total*100:.2f}%)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), ncol=1, fontsize=10,
               title="Asset", title_fontsize=10)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
#  MAIN FIGURE 06 — Path sanity check (S4.2)
# =============================================================================
def fig_path_sanity_check(named_windows: pd.DataFrame,
                            unsm_panel: pd.DataFrame,
                            paths: np.ndarray,
                            tail_mask: np.ndarray, path):
    """4-panel cumulative paths: GFC empirical vs simulated worst-5% mean per asset."""
    sim_mean_path = paths[tail_mask].mean(axis=0)   # (12, 4)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    axes = axes.flatten()
    quarters = np.arange(1, cfg.HORIZON_QUARTERS + 1)

    gfc_row = named_windows[named_windows["label"] == "GFC"].iloc[0] \
        if "GFC" in named_windows["label"].values else None
    if gfc_row is not None:
        loc = unsm_panel.index.searchsorted(gfc_row["start_date"])
        gfc_path = unsm_panel.iloc[loc:loc + cfg.HORIZON_QUARTERS][cfg.ASSETS].values
    else:
        gfc_path = None

    for i, a in enumerate(cfg.ASSETS):
        ax = axes[i]
        if gfc_path is not None:
            ax.plot(quarters, np.cumsum(gfc_path[:, i]) * 100,
                    color=PALETTE["primary"], lw=2.0, label="Historical GFC")
        ax.plot(quarters, np.cumsum(sim_mean_path[:, i]) * 100,
                color=PALETTE["tertiary"], lw=2.0, ls="--",
                label="Simulated worst-5% mean")
        ax.axhline(0, color=PALETTE["muted"], lw=0.5)
        ax.set_xlabel("Quarter ahead")
        ax.set_ylabel(f"{a} cumulative log return (%)")
        ax.set_title(a)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.4)
        # Explicit quarterly tick labels (1..12)
        ax.set_xticks(quarters)
        ax.set_xticklabels([str(q) for q in quarters], fontsize=8)

    fig.suptitle("Cumulative Asset Paths: Historical GFC vs Simulated Worst-5%",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
#  MAIN FIGURE 07 — Historical replay distribution (S5.1)
# =============================================================================
def fig_historical_replay(replay_df: pd.DataFrame, champion_losses: np.ndarray,
                            es_total: float, path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    var_total = float(np.quantile(champion_losses, cfg.ALPHA))
    losses_clip = champion_losses[
        (champion_losses > np.percentile(champion_losses, 0.5)) &
        (champion_losses < np.percentile(champion_losses, 99.5))
    ]
    ax.hist(losses_clip * 100, bins=100, density=True,
            color=PALETTE["light"], edgecolor=PALETTE["tertiary"], linewidth=0.3,
            label=f"Champion MC distribution (N={len(champion_losses):,})")
    ax.axvline(var_total * 100, color=PALETTE["muted"], lw=1.5, ls=":",
                label=f"VaR 95% = {var_total*100:.2f}%")
    ax.axvline(es_total * 100, color=PALETTE["primary"], lw=2.0, ls="--",
                label=f"ES 95% = {es_total*100:.2f}%")

    event_styles = [
        ("GFC",     PALETTE["tertiary"], "-"),
        ("COVID",   PALETTE["secondary"], "-"),
        ("Worst#1", PALETTE["primary"],  ":"),
        ("Worst#2", PALETTE["primary"],  ":"),
        ("Worst#3", PALETTE["primary"],  ":"),
    ]
    for lbl, color, ls in event_styles:
        rows = replay_df[replay_df["label"] == lbl]
        if rows.empty: continue
        r = rows.iloc[0]
        ax.axvline(r["port_loss"] * 100, color=color, lw=1.4, ls=ls,
                    alpha=0.85,
                    label=f"{lbl} = {r['port_loss']*100:+.1f}%  "
                          f"({r['mc_percentile']:.1f}%ile)")
    ax.set_xlabel("3-year cumulative log loss (%)")
    ax.set_ylabel("Density")
    ax.set_title("Historical Tail Events on Champion Loss Distribution")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
#  MAIN FIGURE 08 — Reverse Stress (S5.2)
# =============================================================================
def fig_reverse_stress(rev: dict, es_total: float, path):
    table = rev["table"]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(cfg.ASSETS)); w = 0.35

    base = table["baseline_3y_loss"].values * 100
    deep = table["deep_tail_3y_loss"].values * 100

    ax.bar(x - w/2, base, w, color=PALETTE["secondary"],
            edgecolor=PALETTE["tertiary"], linewidth=0.6,
            label=f"Baseline tail (ES 95% = {es_total*100:.1f}%)")
    ax.bar(x + w/2, deep, w, color=PALETTE["primary"],
            edgecolor=PALETTE["tertiary"], linewidth=0.6,
            label=f"Deep tail (loss ≥ {rev['target_loss']*100:.1f}%)")

    for i, a in enumerate(cfg.ASSETS):
        ax.text(x[i] - w/2, base[i] + (3 if base[i] >= 0 else -5),
                f"{base[i]:+.1f}%", ha="center", fontsize=9,
                color=PALETTE["tertiary"])
        ax.text(x[i] + w/2, deep[i] + (3 if deep[i] >= 0 else -5),
                f"{deep[i]:+.1f}%", ha="center", fontsize=9,
                color=PALETTE["tertiary"], fontweight="bold")
    ax.axhline(0, color=PALETTE["tertiary"], lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(cfg.ASSETS, fontsize=11)
    ax.set_ylabel("Mean 3-year cumulative log loss (%)")
    ax.set_title(f"Reverse Stress: Per-Asset Loss for ES to Double  "
                 f"(deep-tail subset n={rev['n_deep_tail']:,})")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
#  MAIN FIGURE 09 — Bootstrap forest plot (S5.3)
# =============================================================================
def fig_bootstrap_forest(point_var: float, point_es: float,
                           tier_a: dict, tier_b: dict, path):
    """Forest plot: VaR & ES under point / Tier A / Tier B."""
    from .analysis import wald_ci
    rows = [
        ("VaR (point)",                point_var, point_var, point_var, PALETTE["tertiary"]),
        ("VaR (Tier A — MC noise)",    tier_a["var_mean"], tier_a["var_ci"][0], tier_a["var_ci"][1], PALETTE["secondary"]),
        ("VaR (Tier B — Wald)",        point_var, *wald_ci(point_var, *tier_b["var_ci"]), PALETTE["primary"]),
        ("ES (point)",                 point_es,  point_es,  point_es,  PALETTE["tertiary"]),
        ("ES  (Tier A — MC noise)",    tier_a["es_mean"], tier_a["es_ci"][0], tier_a["es_ci"][1], PALETTE["secondary"]),
        ("ES  (Tier B — Wald)",        point_es,  *wald_ci(point_es, *tier_b["es_ci"]), PALETTE["primary"]),
    ]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    ypos = np.arange(len(rows))[::-1]
    for i, (lbl, p, lo, hi, col) in enumerate(rows):
        err = [[max(p - lo, 0)], [max(hi - p, 0)]]
        ax.errorbar(p * 100, ypos[i], xerr=np.array(err) * 100,
                    fmt="o", color=col, ecolor=col,
                    elinewidth=2.0, capsize=5, ms=10, mew=1.2)
        ax.text(hi * 100 + 0.4, ypos[i],
                f" {p*100:.2f}%  [{lo*100:.2f}, {hi*100:.2f}]",
                va="center", fontsize=9, color=PALETTE["tertiary"])
    ax.set_yticks(ypos); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.axvline(point_var * 100, color=PALETTE["muted"], ls=":", lw=0.7, alpha=0.6)
    ax.axvline(point_es  * 100, color=PALETTE["muted"], ls=":", lw=0.7, alpha=0.6)
    ax.set_xlabel("Loss (%)")
    ax.set_title("Point Estimates with 95% Bootstrap CIs (MC vs Parameter)")
    ax.grid(True, axis="x", alpha=0.4)
    ax.set_xlim(left=0)
    plt.tight_layout()
    fig.savefig(path); plt.close(fig)


# =============================================================================
#  APPENDIX FIGURES
# =============================================================================
def app_eda_wealth(panel: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    cols = ["SPY", "AGG", "PE", "NPI", "LPX50", "RMZ"]
    cols = [c for c in cols if c in panel.columns]
    wealth = panel[cols].dropna(how="all").copy()
    wealth = wealth / wealth.iloc[0] * 100
    style_map = {"SPY": (PALETTE["primary"], "-"),
                 "AGG": (PALETTE["tertiary"], "-"),
                 "PE":  (PALETTE["secondary"], "-"),
                 "NPI": (PALETTE["muted"], "-"),
                 "LPX50": (PALETTE["secondary"], "--"),
                 "RMZ":   (PALETTE["muted"], "--")}
    for c in cols:
        col, ls = style_map.get(c, (PALETTE["tertiary"], "-"))
        s = wealth[c].dropna()
        ax.plot(s.index, s, label=c, color=col, lw=1.4, linestyle=ls)
    _shade_crises(ax)
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative wealth (rebased to 100, log scale)")
    ax.set_title("Cumulative Wealth: Four Core Assets and Listed Proxies")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(True, which="both", alpha=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_eda_acf(returns: pd.DataFrame, path):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for col_idx, asset in enumerate(cfg.ASSETS):
        s = returns[asset].dropna()
        plot_acf(s,    lags=12, ax=axes[0, col_idx], title=f"{asset}  ACF",
                 color=PALETTE["primary"], vlines_kwargs={"colors": PALETTE["primary"]})
        plot_pacf(s,   lags=12, ax=axes[1, col_idx], title=f"{asset}  PACF",
                  method="ywm", color=PALETTE["primary"],
                  vlines_kwargs={"colors": PALETTE["primary"]})
        for ax in (axes[0, col_idx], axes[1, col_idx]):
            ax.set_ylim(-0.4, 1.05)
    fig.suptitle("Autocorrelation Diagnostics: ACF and PACF (lags 1-12)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_eda_rolling_vol(returns: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    rv = returns[cfg.ASSETS].rolling(4).std() * np.sqrt(4) * 100
    style_map = {"SPY": PALETTE["primary"], "AGG": PALETTE["tertiary"],
                 "PE": PALETTE["secondary"], "NPI": PALETTE["muted"]}
    for a in cfg.ASSETS:
        ax.plot(rv.index, rv[a], color=style_map[a], lw=1.5, label=a)
    _shade_crises(ax)
    ax.set_ylabel("Annualized volatility (%)")
    ax.set_title("Rolling 1-Year Annualized Volatility")
    ax.legend(ncol=4); ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_eda_rolling_corr(returns: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(12, 5))
    win = 8
    R = returns[cfg.ASSETS]
    pairs = [("SPY", "PE"), ("SPY", "NPI"), ("SPY", "AGG"), ("PE", "NPI")]
    for (a, b) in pairs:
        rc = R[a].rolling(win).corr(R[b])
        ax.plot(rc.index, rc, lw=1.5, label=f"{a}-{b}")
    _shade_crises(ax)
    ax.axhline(0, color=PALETTE["tertiary"], lw=0.5)
    ax.set_ylim(-0.5, 1.0)
    ax.set_ylabel("Rolling 2-year Pearson correlation")
    ax.set_title("Rolling 2-Year Pairwise Correlation (Core Assets)")
    ax.legend(ncol=4); ax.grid(True, alpha=0.4)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_eda_qq(returns: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for i, a in enumerate(cfg.ASSETS):
        s = returns[a].dropna()
        stats.probplot(s, dist="norm", plot=axes[i])
        axes[i].set_title(f"{a}  (skew={stats.skew(s):.2f}, "
                          f"ex.kurt={stats.kurtosis(s):.2f})")
        line = axes[i].get_lines()
        if len(line) >= 2:
            line[0].set_marker("o"); line[0].set_ms(4)
            line[0].set_color(PALETTE[a]); line[0].set_mfc(PALETTE[a])
            line[1].set_color(PALETTE["tertiary"]); line[1].set_lw(1.0)
        axes[i].grid(True, alpha=0.4)
    fig.suptitle("Normal Q-Q Plot of Quarterly Log Returns",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_eda_chiplot(returns: pd.DataFrame, path):
    pairs = [("SPY", "PE"), ("SPY", "NPI"), ("SPY", "AGG"),
             ("PE", "NPI"), ("PE", "LPX50"), ("NPI", "RMZ")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    qs = np.linspace(0.05, 0.95, 19)
    for ax, (a, b) in zip(axes.flatten(), pairs):
        d = returns[[a, b]].dropna()
        if len(d) < 20:
            ax.text(0.5, 0.5, f"insufficient data", transform=ax.transAxes,
                    ha="center"); continue
        n = len(d)
        rx = (d[a].rank() - 0.5) / n
        ry = (d[b].rank() - 0.5) / n
        chi = []
        for q in qs:
            if q < 0.5:
                m = rx < q; chi.append(np.mean((rx < q) & (ry < q)) /
                                       max(m.mean(), 1e-9))
            else:
                m = rx > q; chi.append(np.mean((rx > q) & (ry > q)) /
                                       max(m.mean(), 1e-9))
        ax.plot(qs, chi, "o-", color=PALETTE["primary"], lw=1.4, ms=4)
        ax.axvline(0.5, color=PALETTE["muted"], ls="--", lw=0.7)
        ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("u"); ax.set_ylabel(r"$\chi(u)$")
        ax.set_title(f"{a} - {b}   (Pearson ρ = {d.corr().iloc[0,1]:.2f})")
        ax.grid(True, alpha=0.4)
    fig.suptitle("Empirical Chi-Plot: Tail Dependence Diagnostic",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_unsmoothing_drawdown(unsm: pd.DataFrame, returns: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    win = (pd.Timestamp("2007-09-30"), pd.Timestamp("2010-12-31"))
    for ax, asset, proxy in [(axes[0], "PE", "LPX50"), (axes[1], "NPI", "RMZ")]:
        styles = [(f"{asset}_raw", "Raw",  PALETTE["muted"], "-"),
                  (f"{asset}_FGW", "FGW",  PALETTE["secondary"], "-"),
                  (f"{asset}_KF",  "KF",   PALETTE["primary"], "-"),
                  (proxy,          proxy,  PALETTE["tertiary"], "--")]
        src = returns if proxy in returns.columns else unsm
        for col, lbl, c, ls in styles:
            s = (unsm[col] if col in unsm.columns else returns[col]).dropna()
            s = s.loc[(s.index >= win[0]) & (s.index <= win[1])]
            if len(s) < 3: continue
            cum = 100 * np.exp(s.cumsum()); cum = cum / cum.iloc[0] * 100
            ax.plot(cum.index, cum, color=c, ls=ls, lw=1.6, label=lbl)
        ax.axhline(100, color=PALETTE["muted"], ls="--", lw=0.5)
        ax.set_ylabel("Cumulative wealth (rebased to 100)")
        ax.set_title(f"{asset} during GFC")
        ax.legend(); ax.grid(True, alpha=0.4)
    fig.suptitle("GFC Drawdown: Reported vs Unsmoothed vs Listed Proxy",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_marg_pit_hist(marginals: dict, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, a in zip(axes.flatten(), cfg.ASSETS):
        u = marginals[a]["uniforms"].values
        ax.hist(u, bins=15, color=PALETTE["primary"],
                edgecolor=PALETTE["tertiary"], linewidth=0.5, alpha=0.8)
        ax.axhline(len(u) / 15, color=PALETTE["tertiary"], ls="--", lw=0.8,
                    label=r"Expected uniform")
        ks_p = marginals[a]["diagnostics"]["ks_p"]
        ax.set_title(f"{a}  PIT uniforms (KS p = {ks_p:.3f})")
        ax.set_xlabel("u"); ax.set_ylabel("Count")
        ax.legend(); ax.grid(True, alpha=0.4)
    fig.suptitle("Probability Integral Transform: Uniform Diagnostic",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_marg_qq(marginals: dict, path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, a in zip(axes.flatten(), cfg.ASSETS):
        z = marginals[a]["residuals"].values
        dist = marginals[a]["dist"]
        n = len(z); zs = np.sort(z)
        probs = (np.arange(1, n + 1) - 0.5) / n
        theo = np.asarray(dist.ppf(probs))
        ax.scatter(theo, zs, s=18, color=PALETTE["primary"], alpha=0.7,
                    edgecolor=PALETTE["tertiary"], linewidth=0.3)
        lim = max(abs(zs).max(), abs(theo).max()) * 1.05
        ax.plot([-lim, lim], [-lim, lim], color=PALETTE["tertiary"],
                ls="--", lw=1.0)
        ax.set_title(f"{a}  Q-Q vs fitted distribution")
        ax.set_xlabel("Theoretical quantile"); ax.set_ylabel("Empirical quantile")
        ax.grid(True, alpha=0.4)
    fig.suptitle("Residual Q-Q Plot vs Fitted Marginal",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_marg_ewma(marginals: dict, path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    for ax, a in zip(axes, cfg.ASSETS):
        sig = (np.sqrt(marginals[a]["sigma2"]) *
                np.sqrt(cfg.QUARTERS_PER_YEAR) * 100)
        absr = marginals[a]["residuals_raw"].abs() * 100
        ax.plot(sig.index, sig.values, color=PALETTE["primary"], lw=1.4,
                label=f"EWMA σ (annualized %, λ={marginals[a]['lambda']:.3f})")
        ax.plot(absr.index, absr.values, color=PALETTE["muted"], lw=0.7,
                alpha=0.6, label="|residual| (%)")
        ax.set_ylabel(f"{a}"); ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.4)
    axes[0].set_title("EWMA Conditional Volatility Tracking",
                       fontsize=12, fontweight="bold")
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_copula_loglik(comparison: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    metrics = [("ll_full", "In-sample LL ↑"),
                ("aic_full", "AIC ↓"),
                ("bic_full", "BIC ↓"),
                ("ll_oos",  "OOS LL ↑")]
    colors = [PALETTE["secondary"], PALETTE["primary"], PALETTE["tertiary"]]
    for ax, (m, lbl) in zip(axes, metrics):
        vals = comparison[m].values; idx = comparison.index
        ax.bar(idx, vals, color=colors[:len(idx)],
                edgecolor=PALETTE["tertiary"], linewidth=0.5)
        # Place labels OUTSIDE the bar regardless of sign
        spread = max(vals.max() - vals.min(), 1e-3)
        offset = spread * 0.04
        for i, v in enumerate(vals):
            if v >= 0:
                ax.text(i, v + offset, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=10, color=PALETTE["tertiary"], fontweight="bold")
            else:
                ax.text(i, v - offset, f"{v:.1f}", ha="center", va="top",
                        fontsize=10, color=PALETTE["tertiary"], fontweight="bold")
        # Pad y-axis to leave room for labels above/below
        y_min = min(0, vals.min()) - spread * 0.15
        y_max = max(0, vals.max()) + spread * 0.15
        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color=PALETTE["tertiary"], lw=0.5)
        ax.set_title(lbl); ax.grid(True, axis="y", alpha=0.4)
    fig.suptitle("Copula Model Comparison: Information Criteria",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_copula_pair_scatter(uniforms: pd.DataFrame, copulas: dict, path):
    U = uniforms.values
    sim_t = copulas["full"]["Student-t"].simulate(20000, seed=cfg.SEED_COPULA_SIM)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, (i, j) in zip(axes.flatten(), combinations(range(4), 2)):
        # Background: density of simulated draws (hexbin gives readable structure)
        hb = ax.hexbin(sim_t[:, i], sim_t[:, j], gridsize=22,
                        cmap="Reds", mincnt=1, alpha=0.85,
                        extent=(0, 1, 0, 1), edgecolors="none")
        # Foreground: empirical points
        ax.scatter(U[:, i], U[:, j], s=36, color=PALETTE["tertiary"],
                    edgecolor="white", linewidth=0.7,
                    alpha=0.95, label="Empirical (n=80)", zorder=5)
        ax.set_xlabel(f"u_{cfg.ASSETS[i]}")
        ax.set_ylabel(f"u_{cfg.ASSETS[j]}")
        ax.set_title(f"{cfg.ASSETS[i]} vs {cfg.ASSETS[j]}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    handles = [
        plt.Line2D([], [], marker="o", color="w", markerfacecolor=PALETTE["tertiary"],
                   markeredgecolor="white", markersize=9, label="Empirical (n=80)"),
        plt.Line2D([], [], marker="h", color="w", markerfacecolor=PALETTE["primary"],
                   markersize=11, label="Student-t simulated density (n=20,000)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01),
               ncol=2, fontsize=10, frameon=False)
    fig.suptitle("Pairwise Scatters: Empirical vs Student-t Simulated Density",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(rect=[0, 0.02, 1, 1])
    fig.savefig(path); plt.close(fig)


def app_sim_loss_distribution(loss_arrays: dict, es_matrix: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(11, 6))
    style_map = [
        ("HS", PALETTE["muted"], "Historical Simulation"),
        ("FHS", PALETTE["secondary"], "Filtered HS"),
        ("Champion", PALETTE["primary"], "Champion (t-copula MC)"),
    ]
    avail = [k for k in style_map if (k[0], "unsmoothed") in loss_arrays
             and loss_arrays[(k[0], "unsmoothed")] is not None]
    if not avail: return
    concat = np.concatenate([loss_arrays[(k[0], "unsmoothed")] for k in avail])
    x_lo, x_hi = np.percentile(concat * 100, [0.5, 99.5])

    for label, color, full_name in avail:
        arr = loss_arrays[(label, "unsmoothed")]
        clip = arr[(arr * 100 >= x_lo) & (arr * 100 <= x_hi)]
        ax.hist(clip * 100, bins=80, density=True, alpha=0.45,
                color=color, label=f"{full_name} (ES={es_matrix.loc[label, 'unsmoothed']*100:.1f}%)")
        ax.axvline(es_matrix.loc[label, "unsmoothed"] * 100, color=color,
                    lw=1.4, ls="--", alpha=0.8)
    es_p = es_matrix.loc["Parametric", "unsmoothed"] * 100
    ax.axvline(es_p, color=PALETTE["tertiary"], ls=":", lw=1.4,
                label=f"Parametric ES (analytic) = {es_p:.1f}%")
    ax.set_xlim(x_lo, x_hi)
    ax.set_xlabel("3-year cumulative log loss (%)")
    ax.set_ylabel("Density")
    ax.set_title("3-Year Loss Distributions: HS / FHS / Champion (Unsmoothed Data)")
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.4)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_sim_path_examples(paths: np.ndarray, port_losses: np.ndarray, path):
    n_worst = 5
    worst_idx  = np.argsort(port_losses)[-n_worst:][::-1]
    median_idx = np.argsort(port_losses)[len(port_losses) // 2]
    quarters = np.arange(1, cfg.HORIZON_QUARTERS + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for k, idx in enumerate(worst_idx):
        port_q = paths[idx] @ np.array(cfg.WEIGHTS)
        cum = np.exp(np.cumsum(port_q))
        axes[0].plot(quarters, cum, color=PALETTE["primary"], alpha=0.5,
                     lw=1.0, label="Worst 5" if k == 0 else None)
    port_q_med = paths[median_idx] @ np.array(cfg.WEIGHTS)
    axes[0].plot(quarters, np.exp(np.cumsum(port_q_med)),
                  color=PALETTE["tertiary"], lw=1.8, label="Median path")
    axes[0].axhline(1.0, color=PALETTE["muted"], ls="--", lw=0.6)
    axes[0].set_xlabel("Quarter ahead")
    axes[0].set_ylabel("Cumulative wealth (× initial)")
    axes[0].set_title("Worst-5 paths vs Median")
    axes[0].legend(); axes[0].grid(True, alpha=0.4)

    wow = worst_idx[0]
    style_map = {"SPY": PALETTE["primary"], "AGG": PALETTE["tertiary"],
                 "PE": PALETTE["secondary"], "NPI": PALETTE["muted"]}
    for i, a in enumerate(cfg.ASSETS):
        cum = np.exp(np.cumsum(paths[wow, :, i]))
        axes[1].plot(quarters, cum, color=style_map[a], lw=1.6, label=a)
    axes[1].axhline(1.0, color=PALETTE["muted"], ls="--", lw=0.6)
    axes[1].set_xlabel("Quarter ahead")
    axes[1].set_ylabel("Cumulative asset return (× initial)")
    axes[1].set_title("Worst-of-Worst Path: Per-Asset Trajectory")
    axes[1].legend(); axes[1].grid(True, alpha=0.4)
    fig.suptitle("Champion Model: Example Simulated Paths",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_decomp_time_loss(time_loss_df: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.bar(time_loss_df["quarter"], time_loss_df["mean_loss_q"] * 100,
            color=PALETTE["primary"], edgecolor=PALETTE["tertiary"],
            linewidth=0.5, alpha=0.85)
    ax.set_xlabel("Quarter ahead"); ax.set_ylabel("Mean loss (%)")
    ax.set_title("Per-Quarter Mean Loss in Worst-5% Tail")
    ax.grid(True, axis="y", alpha=0.4)

    ax = axes[1]
    ax.plot(time_loss_df["quarter"], time_loss_df["cum_mean_loss"] * 100,
            color=PALETTE["primary"], marker="o", lw=1.6,
            label="Cumulative worst-5% mean loss")
    ax.set_xlabel("Quarter ahead"); ax.set_ylabel("Cumulative loss (%)")
    ax.set_title("Cumulative Loss Trajectory")
    ax.grid(True, alpha=0.4); ax.legend()
    fig.suptitle("Time Profile of Tail Loss (Within 12-Quarter Horizon)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_validation_backtest_ts(backtest: dict, path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    dates = backtest["var_q"].index
    ax = axes[0]
    ax.plot(dates, backtest["var_q"] * 100, color=PALETTE["primary"], lw=1.4,
             label="1Q VaR (95%)")
    ax.plot(dates, backtest["es_q"]  * 100, color=PALETTE["primary"], lw=1.4,
             ls="--", label="1Q ES (95%)")
    ax.plot(dates, backtest["port_loss"] * 100, "o", ms=3.5,
             color=PALETTE["tertiary"], alpha=0.7, label="Actual loss")
    breach_dates = dates[backtest["breaches"]]
    breach_vals  = backtest["port_loss"][backtest["breaches"]] * 100
    ax.plot(breach_dates, breach_vals, "s", color=PALETTE["secondary"],
             ms=8, mec=PALETTE["tertiary"], mew=0.6,
             label=f"Breach (n={backtest['n_breach']})")
    ax.axhline(0, color=PALETTE["muted"], lw=0.5)
    ax.set_ylabel("1Q loss (%)"); ax.legend(loc="upper left")
    ax.grid(True, alpha=0.4)
    p = backtest["table"]
    ax.set_title(
        f"In-Sample 1Q VaR/ES Backtest   "
        f"Kupiec p={p[p['test']=='Kupiec POF']['p_value'].iloc[0]:.3f},  "
        f"Christoffersen p={p[p['test']=='Christoffersen Ind']['p_value'].iloc[0]:.3f},  "
        f"Acerbi Z={p[p['test']=='Acerbi-Szekely Z']['stat'].iloc[0]:.3f}")

    ax = axes[1]
    sig_ann = backtest["port_sigma"] * np.sqrt(cfg.QUARTERS_PER_YEAR) * 100
    ax.plot(dates, sig_ann, color=PALETTE["primary"], lw=1.4,
             label="Portfolio EWMA σ (annualized %)")
    ax.axhline(float(sig_ann.mean()), color=PALETTE["muted"], ls="--", lw=0.7,
                label=f"mean = {sig_ann.mean():.1f}%")
    ax.set_ylabel("Annualized vol (%)"); ax.set_xlabel("Date")
    ax.set_title("Conditional Portfolio Volatility")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.4)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_validation_stress(stress_df: pd.DataFrame, es_total: float, path):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    n = len(stress_df); xpos = np.arange(n); w = 0.65
    contributions = (stress_df[[f"{a}_loss" for a in cfg.ASSETS]].values
                     * np.array(cfg.WEIGHTS)[None, :] * 100)
    bottoms_pos = np.zeros(n); bottoms_neg = np.zeros(n)
    for i, a in enumerate(cfg.ASSETS):
        vals = contributions[:, i]
        pos = vals >= 0; neg = vals < 0
        if pos.any():
            ax.bar(xpos[pos], vals[pos], w, bottom=bottoms_pos[pos],
                    color=PALETTE[a], edgecolor="white", linewidth=0.6,
                    label=a)
            bottoms_pos[pos] += vals[pos]
        if neg.any():
            ax.bar(xpos[neg], vals[neg], w, bottom=bottoms_neg[neg],
                    color=PALETTE[a], edgecolor="white", linewidth=0.6,
                    alpha=0.8)
            bottoms_neg[neg] += vals[neg]
    # Place total-loss markers and labels above (or below) the stack
    for i, (_, r) in enumerate(stress_df.iterrows()):
        pl = r["stressed_port_loss"] * 100
        ax.plot(i, pl, "_", color=PALETTE["tertiary"], ms=22, mew=2.5,
                 zorder=10)
        # Label sits above positive totals, below negative totals
        if pl >= 0:
            ax.text(i, max(pl, bottoms_pos[i]) + 1.5,
                    f"{pl:+.1f}%", ha="center", va="bottom", fontsize=11,
                    fontweight="bold", color=PALETTE["tertiary"])
        else:
            ax.text(i, min(pl, bottoms_neg[i]) - 1.5,
                    f"{pl:+.1f}%", ha="center", va="top", fontsize=11,
                    fontweight="bold", color=PALETTE["tertiary"])
    ax.axhline(es_total * 100, color=PALETTE["primary"], ls="--", lw=1.2,
                label=f"Baseline ES = {es_total*100:.1f}%")
    ax.axhline(0, color=PALETTE["tertiary"], lw=0.5)
    ax.set_xticks(xpos); ax.set_xticklabels(stress_df["scenario"].values,
                                              fontsize=9, rotation=15)
    ax.set_ylabel("Loss contribution (% of portfolio)")
    ax.set_title("Hypothetical Stress Scenarios (Historical × Multiplier)")
    handles = [Patch(color=PALETTE[a], label=a) for a in cfg.ASSETS]
    handles.append(plt.Line2D([], [], color=PALETTE["tertiary"], marker="_",
                                linestyle="None", ms=20, mew=2.5,
                                label="Total stressed loss"))
    handles.append(plt.Line2D([], [], color=PALETTE["primary"], ls="--",
                                label="Baseline ES"))
    # Legend OUTSIDE on the right so it can't overlap any bar
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1),
              fontsize=9, title="Legend", title_fontsize=9)
    # Y-axis padding for labels
    y_top = max(bottoms_pos.max(), (stress_df["stressed_port_loss"] * 100).max()) + 5
    y_bot = min(bottoms_neg.min(), (stress_df["stressed_port_loss"] * 100).min()) - 5
    ax.set_ylim(y_bot, y_top)
    ax.grid(True, axis="y", alpha=0.4)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_bootstrap_distributions(tier_a: dict, tier_b: dict,
                                  point_var: float, point_es: float, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [
        (axes[0, 0], tier_a["var_samples"], "Tier A: VaR — MC noise",
         point_var, "VaR 95% (%)"),
        (axes[0, 1], tier_a["es_samples"],  "Tier A: ES — MC noise",
         point_es, "ES 95% (%)"),
        (axes[1, 0], tier_b["var_samples"][~np.isnan(tier_b["var_samples"])],
         "Tier B: VaR — Parameter uncertainty", point_var, "VaR 95% (%)"),
        (axes[1, 1], tier_b["es_samples"][~np.isnan(tier_b["es_samples"])],
         "Tier B: ES — Parameter uncertainty",  point_es, "ES 95% (%)"),
    ]
    for ax, samples, title, pt, xlab in panels:
        color = PALETTE["primary"] if "Tier A" in title else PALETTE["secondary"]
        ax.hist(samples * 100, bins=40, color=color,
                 edgecolor=PALETTE["tertiary"], linewidth=0.4, alpha=0.8)
        ax.axvline(pt * 100, color=PALETTE["tertiary"], lw=2.0,
                    label=f"Point = {pt*100:.2f}%")
        lo, hi = np.percentile(samples * 100, [2.5, 97.5])
        ax.axvline(lo, color=PALETTE["muted"], ls="--", lw=1.0)
        ax.axvline(hi, color=PALETTE["muted"], ls="--", lw=1.0,
                    label=f"95% CI [{lo:.2f}, {hi:.2f}]")
        ax.set_xlabel(xlab); ax.set_ylabel("Count")
        ax.set_title(title); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.4)
    fig.suptitle("Bootstrap Distributions of VaR and ES",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)


def app_bootstrap_param(tier_b: dict, path):
    pair_labels = tier_b["pair_labels"]
    R_samples = tier_b["R_samples"]
    nu_samples = tier_b["nu_samples"]
    valid = ~np.isnan(nu_samples)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    # Top: 4 R off-diagonals
    show_pairs = ["SPY-PE", "SPY-NPI", "PE-NPI", "AGG-NPI"]
    for col, lbl in enumerate(show_pairs):
        ax = axes[0, col]; k = pair_labels.index(lbl)
        s = R_samples[valid, k]
        ax.hist(s, bins=25, color=PALETTE["primary"],
                 edgecolor=PALETTE["tertiary"], linewidth=0.4, alpha=0.8)
        m = float(s.mean()); lo, hi = np.percentile(s, [2.5, 97.5])
        ax.axvline(m, color=PALETTE["tertiary"], lw=1.5, label=f"Mean = {m:.3f}")
        ax.axvline(lo, color=PALETTE["muted"], ls="--", lw=0.8)
        ax.axvline(hi, color=PALETTE["muted"], ls="--", lw=0.8,
                    label=f"95% CI [{lo:.2f}, {hi:.2f}]")
        ax.set_xlabel(f"R[{lbl}]"); ax.set_title(lbl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    # Bottom row: nu hist + 2 more R + ES vs nu scatter
    ax = axes[1, 0]
    nu_data = nu_samples[valid]
    nu_lo, nu_hi = float(np.percentile(nu_data, 1)), float(np.percentile(nu_data, 99))
    bins = np.linspace(max(2.5, nu_lo - 1), min(30, nu_hi + 1), 16)
    ax.hist(nu_data, bins=bins, color=PALETTE["secondary"],
             edgecolor=PALETTE["tertiary"], linewidth=0.4, alpha=0.8)
    ax.axvline(float(nu_data.mean()), color=PALETTE["tertiary"],
                lw=1.5, label=f"Mean ν = {float(nu_data.mean()):.2f}")
    ax.set_xlabel("ν (df)"); ax.set_title("Student-t copula df ν")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.4)
    ax.set_xlim(bins[0], bins[-1])

    for col, lbl in zip([1, 2], ["SPY-AGG", "AGG-PE"]):
        ax = axes[1, col]; k = pair_labels.index(lbl)
        s = R_samples[valid, k]
        ax.hist(s, bins=25, color=PALETTE["primary"],
                 edgecolor=PALETTE["tertiary"], linewidth=0.4, alpha=0.8)
        ax.axvline(float(s.mean()), color=PALETTE["tertiary"], lw=1.5,
                    label=f"Mean = {float(s.mean()):.3f}")
        ax.set_xlabel(f"R[{lbl}]"); ax.set_title(lbl)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.4)

    ax = axes[1, 3]
    es = tier_b["es_samples"][valid]
    nu = nu_samples[valid]
    ax.scatter(nu, es * 100, alpha=0.4, s=15, color=PALETTE["primary"],
                edgecolor="none")
    ax.set_xlabel("ν"); ax.set_ylabel("ES 95% (%)")
    ax.set_title("ES vs ν")
    ax.grid(True, alpha=0.4)
    fig.suptitle("Tier B Parameter Bootstrap: R Off-Diagonals and ν",
                 fontsize=13, fontweight="bold", y=1.005)
    plt.tight_layout(); fig.savefig(path); plt.close(fig)
