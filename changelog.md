# Changelog: From Prototype Notebook (block1-8) to Refactored Project

This document records the mapping from the original block-by-block prototype
(`block1_eda.py` ... `block8_bootstrap.py`) to the current package structure,
and the substantive corrections made during refactoring.

---

## File-level mapping

| Original prototype                 | Refactored module(s)                             |
|------------------------------------|--------------------------------------------------|
| `block1_eda.py`                    | `src/data_loader.py` + appendix figures in `src/plots.py` (`app_eda_*`) |
| `block2_unsmooth.py`               | `src/unsmoothing.py`                             |
| `block3_marginals_v3.py`           | `src/marginals.py`                               |
| `block4_copula_v2.py`              | `src/copula.py`                                  |
| `block5_var_es.py`                 | `src/simulation.py` + main figures (`fig_es_matrix`) |
| `block6_decomposition.py`          | `src/analysis.py` (`component_cvar`, `time_of_loss`, `historical_tail_windows`) |
| `block7_validation.py`             | `src/analysis.py` (`in_sample_backtest`, `historical_replay`, `stress_scenarios`, `reverse_stress`) |
| `block8_bootstrap.py`              | `src/analysis.py` (`tier_a_bootstrap`, `tier_b_bootstrap`, `wald_ci`) |
| Per-block diagnostic figures       | `src/plots.py`                                   |

---

## Substantive corrections

### 1. PE/NPI starting variance for FHS / Champion simulation

**Bug.** The prototype's `block5_var_es.py` initially used
`marginals[a]['sigma2_uncond']` for the EWMA recursion's `sigma_0^2`. For
SPY/AGG (constant mean) this is the variance of the data; for PE/NPI
(`force_ar1=True`) this is the variance of *AR(1) residuals*, which is
~17-23% lower than the variance of unsmoothed log returns. Using the wrong
variance produced an artificially low Champion ES (16.73% in an early test
run, against the correct 18.73%).

**Fix.** The refactor uses the unsmoothed log-return variance for PE/NPI
and the AR-residual variance for SPY/AGG. The selection happens in
`src/simulation.py::sigma2_uncond_for_simulation`, conditioned on
`marginals[a]['ar_lags']`.

### 2. Uniform clipping for Champion PPF

**Bug.** Without clipping, the semi-parametric tails (which are fit on only
~8 observations on each side of the central interval) extrapolate to
implausibly extreme `z` values when `u < 1e-6`, polluting the simulated
loss distribution's left tail. Note that this does *not* affect 95% ES,
which only averages losses in the worst 5% of paths, but it produces
visually disturbing density plots and can destabilise downstream
diagnostics.

**Fix.** All uniforms are clipped to `[1e-4, 1 - 1e-4]` before PPF in
`src/simulation.py::champion_simulation`.

### 3. Tier B bootstrap: iid row vs. block bootstrap

**Bug.** The prototype briefly considered using a block bootstrap for the
Tier B parameter resampling. This is incorrect because Block 3 (marginal
fitting) has already filtered out time-series structure via the EWMA
recursion: the standardised residuals are approximately iid by
construction. A block bootstrap would re-introduce serial dependence that
is not in the residuals.

**Fix.** Tier B uses an iid row bootstrap on the PIT uniforms
(`src/analysis.py::tier_b_bootstrap`).

### 4. Tier B bootstrap: constant vs. EWMA simulation

**Bug.** For computational tractability we cannot run a 1M-path EWMA
recursion inside each of 500 Tier-B bootstrap iterations. The prototype
considered simplifying to constant volatility; we initially worried this
would underestimate uncertainty.

**Fix (justified).** On quarterly data with `lambda = 0.94` (half-life ~12
quarters = the entire horizon), the EWMA recursion's effect within a
3-year horizon is small; the long-run volatility dominates. Using constant
volatility introduces a small downward bias (~0.5pp on the central ES) but
produces CI bounds that are within a few pp of those that would be
obtained with full EWMA. The `methodology.md` Section 5.5 documents this
tradeoff.

### 5. Vuong test BIC correction

The Vuong (1989) test in its original form is a likelihood-ratio statistic;
without correction it tends to favour the more flexible model. We use the
Schwarz/BIC correction (subtract `(k1 - k2) log(n) / (2n)` from the
per-observation log-likelihood difference) per Vuong's own recommendation
for non-nested model selection. This is in `src/copula.py::vuong_test`.

### 6. Use of `gammaln` instead of `log(gamma(.))`

The Student-t copula log-likelihood involves the gamma function. Using
`np.log(stats.gamma.pdf(...))` or similar through the frozen-distribution
interface produced numerical instability and an outright bug (frozen
distributions do not have `.log` methods). The refactor uses
`scipy.special.gammaln` directly throughout.

---

## Architectural changes

1. **One source of truth for input data.** The prototype mixed `data.parquet`
   (block-1 output) with the original `data.xlsx`. The refactor reads only
   `data/data.xlsx`; everything downstream is regenerated and cached to
   `cache/`.

2. **Cached intermediates.** Each major stage caches its output (parquet
   for tabular, pickle for fitted models, npz for arrays). Re-runs of
   later stages skip the refit.

3. **Single seed strategy.** All Monte Carlo uses `SEED = 42` with
   deterministic offsets in `src/config.py`. The prototype mixed several
   seeds across blocks.

4. **Sanity check via JSON.** Every key number reported in
   `methodology.md` is also written to `output/numbers.json` and asserted
   against canonical values in `tests/test_numbers.py` with tiered
   tolerances.

5. **Optional R-Vine.** `pyvinecopulib` is moved to `requirements-vine.txt`
   to keep the main install path lightweight; the project gracefully skips
   R-Vine when the library is missing.

6. **English everywhere.** All inline comments, output strings, and
   documentation are in English. The prototype mixed Chinese and English
   in comments and prints.

---

## Plotting and validation fixes (post-review)

### 7. Component CVaR figure: stacking semantics

**Bug.** The two columns of `fig_component_cvar` were stacked using a single
`bottoms` array, so the right column inherited the left column's running
total. The right-column NPI bar was drawn at y = 105-111 and clipped by the
ylim, making NPI "disappear" visually even though the data was correct.

**Fix.** Maintain `bottom_pos` and `bottom_neg` separately for the right
column. Positives stack from 0 upward to 111.5, negatives stack from 0
downward to -11.7. A 100% reference line shows the additivity target;
ylim auto-extends to capture the full bar. Legend moved out of the plot
area to avoid overlap.

### 8. Tail-dependence figure: bar overlap and missing emphasis

**Bug.** Four bars per pair at width 0.20 with offsets ±1.5w / ±0.5w gave a
total bar group width of 0.80 — close enough to the 1-unit pair spacing
that bars from neighbouring pairs touched and the visual story was lost.

**Fix.** Drop the redundant Gaussian λ_L bar (it is identically zero, the
information is conveyed by a horizontal axis line and a legend entry).
Three bars per pair at width 0.27 give comfortable separation. Add a
direct annotation pointing to the SPY-NPI bar — the canonical asymmetric
pair — so the slide-talk callout is built into the figure.

### 9. ES matrix figure: N/A cells rendered as white

**Bug.** Cells with NaN (e.g., `FHS / reported`) were drawn white with a
`—` glyph, which a quick reader confuses with "ES = 0". The "by design"
caveat (FHS and Champion only run on unsmoothed data) was easy to miss.

**Fix.** Use a masked array with `cmap.set_bad(light gray)` and an
explicit "N/A (by design)" label in italic gray.

### 10. Path sanity figure: missing x-tick labels

**Bug.** The four-panel cumulative-paths figure had no quarterly tick
labels on x. The default matplotlib axis showed `2, 4, 6, 8, 10, 12`
which read as years to many viewers.

**Fix.** Force `set_xticks([1..12])` and explicit string labels.

### 11. Three-pathologies figure: text panel wording

**Issue.** The "Pathology 3" text panel said "~0.3 obs in the tail" with
no surrounding math, which read as a typo. Re-worded to spell out
n × (1 − α) = 6 × 0.05 = 0.3 with aligned columns. The Pathology 2 panel's
"Upper-tail weak" annotation was being covered by the chi curve when it
crossed the upper-right corner — repositioned to top-right, away from
the data.

### 12. COVID stress scenario removed

**Issue.** The historical 3-year COVID window (2019Q4-2022Q3) was a
**portfolio gain** of 20% (the post-March 2020 rebound dominated the
drawdown). Multiplying it by 1.5 amplified a gain rather than a loss,
producing a "stress scenario" with negative loss — a meaningless object
that would invite questions in defence.

**Fix.** Drop COVID from `STRESS_SCENARIOS` in `src/config.py`. The stress
ladder is now GFC × {1.2, 1.5, 2.0}. COVID remains in the historical
replay (Section 5.2) where its sign is the right one to discuss.

### 13. Pair-scatter and ν histogram: visual quality

**Issue.** The empirical-vs-simulated 6-panel scatter used 5,000 simulated
points at α = 0.20, which produced a near-uniform gray cloud — the
Student-t structure was invisible. The Tier-B ν histogram used the full
NU_GRID as bin edges (3 to 30), so 95% of the data sat in two bins on the
left and the rest of the axis was empty.

**Fix.** Pair-scatter: replace with a hexbin density background (20,000
draws), so the t-copula's mass concentration in the lower-left and
upper-right corners is visible. ν histogram: clip x-axis to
[1st-percentile − 1, 99th-percentile + 1] of bootstrap samples and use
16 evenly-spaced bins inside that range.
