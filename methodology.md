# Methodology — 95% CVaR for a Multi-Asset Portfolio over a Three-Year Horizon

> This document mirrors the structure of the accompanying slide deck. Each
> numbered section corresponds to one section of the presentation.

---

## Section 1 — Data and Problem

### 1.1 Portfolio specification

Four asset classes with prescribed weights and expected annualized returns:

| Asset | Weight | Expected return | Description                                  |
|-------|--------|-----------------|----------------------------------------------|
| SPY   | 40%    | 6.0%            | S&P 500 total-return index level (gross divs)|
| AGG   | 20%    | 4.5%            | Bloomberg US Aggregate Bond TR index level   |
| PE    | 25%    | 9.5%            | Private equity total-return index            |
| NPI   | 15%    | 7.0%            | NCREIF Property Index (real estate)          |

The risk question is fixed: compute the 95% Conditional Value-at-Risk (CVaR, equivalently Expected Shortfall) over a three-year horizon under quarterly rebalancing.

```
ES_alpha(L) = E[L | L >= VaR_alpha(L)]
```

with `alpha = 0.95`, `L = -log(W_T / W_0)` the cumulative log loss at the three-year horizon `T = 12` quarters, starting wealth `W_0` and end-of-horizon wealth `W_T = W_0 * exp(sum_t r_t^p)` where `r_t^p = w' r_t` is the per-quarter portfolio log return assuming continuous reinvestment.

### 1.2 Sample

The dataset contains 82 quarter-ends from 2003-Q3 to 2023-Q4. After taking log returns or first differences, the working sample is 81 observations or, after marginal model fitting (which discards the first observation per asset), 80 observations.

Two listed proxies are tracked but not held in the portfolio: LPX50 (a listed private-equity proxy) and RMZ (a US REIT proxy). They are used as cross-validation benchmarks for the unsmoothed PE and NPI volatilities.

### 1.3 Three pathologies

A naive Gaussian closed-form CVaR computed on the reported data gives **8.81%**, which is wrong by roughly 2x. Three pathologies of the data drive this gap, each addressed in a later section:

1. **Smoothing in private-asset returns.** PE and NPI exhibit AR(1) autocorrelations of 0.77 and 0.90 respectively. By contrast, SPY and AGG have AR(1) ~ 0.04-0.06. This is appraisal-based smoothing: managers report stale valuations, which spuriously suppresses the variance of quarter-on-quarter changes. The *true* economic volatility is several times what is reported.

2. **Asymmetric tail dependence.** Empirically we observe `chi_L(0.10) >> chi_U(0.10)` for several pairs (e.g., SPY-NPI: lower-tail chi = 0.30, upper-tail chi = 0). A Gaussian copula has zero tail dependence by construction; an elliptical Student-t copula has symmetric upper and lower tail dependence — so even the better elliptical model cannot fully match the asymmetric pattern. We document this as a known limitation rather than a model error (see McNeil-Frey-Embrechts 2015, Sec 7.3).

3. **Long horizon, short sample.** With 80 quarters and a 12-quarter horizon, only `floor(80 / 12) = 6` non-overlapping three-year windows exist. At a 5% quantile this leaves *0.3 effective tail observations*: empirical historical estimation is statistically infeasible. Simulation-based methods are required.

### 1.4 Output of Section 1

The naive answer (8.81%) is recorded as the floor for context; we then move to the modelling pipeline that addresses each pathology.

---

## Section 2 — Modeling Approach

### 2.1 Architecture

The pipeline runs in five sequential stages, each cached to disk so subsequent re-runs skip the refit:

```
data → unsmoothing → marginals → copula → simulation → analysis
```

The choice of copula family closes the dependence-structure question; the choice of marginal handles asymmetry and excess kurtosis; the choice of unsmoothing fixes the variance of PE and NPI; and the simulation engine then runs `1,000,000` 12-step Monte Carlo paths.

### 2.2 Unsmoothing private-asset returns

Three methods are compared on PE and NPI:

**Method A — Fisher-Geltner-Webb (FGW, 1994).** Assumes the smoothed observation `r*_t = alpha r*_{t-1} + (1 - alpha) r_t`, i.e. the manager's reported return is a convex combination of the previous report and the true return. Inversion gives:

```
r_t = (r*_t - alpha r*_{t-1}) / (1 - alpha)
```

`alpha` is estimated as the lag-1 autocorrelation of `r*`. For PE this gives `alpha = 0.77`; for NPI, 0.90.

**Method B — Okunev-White (OW, 2003).** Generalises to higher-order smoothing. We fit AR(q) to the smoothed series with `q in {1, 2, 3}` selected by BIC, and invert by:

```
r_t = (r*_t - sum_i phi_i r*_{t-i}) / (1 - sum_i phi_i)
```

For both PE and NPI BIC selects q = 1 (no evidence of higher-order smoothing on this sample), so OW collapses to FGW in spirit; in practice the AR coefficients differ slightly from the lag-1 autocorrelation and produce slightly lower volatility estimates.

**Method C — Kalman filter (KF).** A state-space formulation casts the problem as: a latent true return `r_t` is observed only after passing through a smoothing operator. The state-space model is equivalent to an ARMA(1,1) on `r*`, which we fit via maximum likelihood (statsmodels). The `alpha` estimated by the KF is *not* the lag-1 autocorrelation but the AR coefficient of the ARMA(1,1); for PE it gives `alpha = 0.5446` and `MA(1) = 0.7781`; for NPI, `alpha = 0.8432` and `MA(1) = 0.4257`.

**Volatility comparison.** The annualized volatility of PE goes from 8.4% (raw) to 23.5% (FGW) to 13.5% (OW) to 12.5% (KF), against 28% for the LPX50 listed proxy. For NPI: 4.6% → 19.6% → 7.6% → 13.0%, against 24.6% for RMZ.

**Choice: KF.** We adopt the Kalman filter as the canonical unsmoothing because:
1. It is the most parsimonious of the three (one parameter beyond AR plus a moving-average term).
2. The FGW estimator is *unbiased only when alpha is small*; at alpha = 0.9 (NPI) FGW amplifies noise by 1/(1-alpha) = 10x. This is the well-known FGW noise-blowup problem; the moving-average term in the KF formulation absorbs it.
3. The KF-unsmoothed volatilities sit between the proxy volatilities and the raw reported volatilities, which is the qualitative pattern expected by the appraisal-smoothing literature: listed proxies overstate volatility (mark-to-market with sentiment); reports understate it (appraisal lag); the truth is in between.

### 2.3 Marginal modeling

For each asset we fit:

```
r_t = mu_t + epsilon_t      mu_t = constant or AR(1)
sigma^2_t = lambda sigma^2_{t-1} + (1 - lambda) epsilon^2_{t-1}     EWMA conditional variance
z_t = epsilon_t / sigma_t                                             standardized residuals
z_t ~ Skew-t (Hansen 1994)  for SPY and AGG
z_t ~ semi-parametric        for PE and NPI
```

**Mean equation.** SPY and AGG use a constant mean (BIC selects p = 0). PE and NPI force AR(1) to absorb residual smoothing left over from the Kalman filter. Empirically the post-KF AR(1) on PE is small (~0.10) but non-zero; forcing AR(1) ensures the EWMA innovations are clean.

**EWMA.** Lambda is estimated by maximum likelihood under a normal innovation assumption (RiskMetrics convention) over the bounded interval `[0.80, 0.99]`. If the MLE hits a boundary the estimator falls back to `lambda = 0.94` (the standard RiskMetrics value). On this sample SPY and PE/NPI all hit the upper boundary and fall back to 0.94; AGG converges to 0.93.

**SPY/AGG marginals — Hansen Skew-t.** A two-parameter generalisation of the Student-t allowing asymmetry. Parameters `eta` (degrees of freedom) and `lambda_skew` (skewness, in [-1, 1]). On this sample SPY recovers `eta = 4.67, lambda_skew = -0.52` and AGG recovers `eta = 8.68, lambda_skew = -0.19` — left-skewed in both cases, with SPY having heavier tails as expected.

**PE/NPI marginals — semi-parametric.** With only 80 KF-unsmoothed observations the parametric Skew-t becomes unstable on PE and NPI; we instead use:
- The empirical CDF on the central interval `[u_lower, u_upper] = [0.10, 0.90]` (interpolated linearly between order statistics).
- A scipy Student-t fit to each tail (`scipy.stats.t.fit` on the points outside the central interval), spliced at the boundary.

The semi-parametric specification is more honest than forcing a parametric tail when the sample size will not support it. The fitted left-tail dfs are around 10 for both PE and NPI (the t fits a slightly heavy left tail).

**PIT diagnostics.** All four marginals pass the Kolmogorov-Smirnov test against `Uniform(0,1)` with p > 0.92; residuals show no significant autocorrelation in `z_t` or `z_t^2` (Ljung-Box at lags 4, 8).

### 2.4 Copula modeling

Three families are fit on the 80 PIT uniforms:

1. **Gaussian copula.** Baseline. R estimated by Kendall's tau inversion: `rho = sin(pi/2 * tau)`. This is the consistent estimator for elliptical copulas and is *invariant to the marginal distributions*, which matters because using normal-scores R would bias the correlation downward when the marginals are non-Gaussian (Demarta-McNeil 2005, Sec 4).

2. **Student-t copula.** Joint maximum likelihood on R (Cholesky-parameterised) and degrees of freedom `nu`. Initialised at the Kendall R; converges to `nu = 4.443`. Lower-tail dependence is positive: `lambda_L(SPY-PE) = 0.18`, `lambda_L(PE-NPI) = 0.21`.

3. **R-Vine copula.** Truncation level 2; pair-copula families chosen from `{Gaussian, Student-t, Clayton, Gumbel, Frank, Joe, BB1, BB7}` by AIC (using `pyvinecopulib` 0.6). On this 80-obs sample the Vine selects mostly Gaussian and Frank pair-copulas — i.e., the non-elliptical flexibility is unused. This is the curse of small samples: the Vine's extra parameters do not pay back in OOS log-likelihood.

**Estimation details (Student-t).** The Cholesky parameterisation flattens the positive-definite constraint into an unconstrained lower-triangular vector; the optimiser is L-BFGS-B with `ftol = 1e-7, maxiter = 200`. After optimisation the result is projected onto the nearest correlation matrix (eigen-clipping at 1e-6) to recover numerical PD.

**Champion selection: Student-t.** The OOS log-likelihood (in-sample n = 60, OOS n = 20) is:
- Gaussian: 0.98
- Student-t: 4.20
- R-Vine: -0.40 (varies by run)

The Student-t copula wins decisively. The Vuong test of Student-t against Gaussian (BIC-corrected) gives `Z = 2.82, p = 0.005`; Student-t against Vine gives `Z = 2.63, p = 0.0085` (Vine rejected). The Vine result is not reproducible without `pyvinecopulib`, but the relative ranking is stable.

**Tail asymmetry note.** The Student-t copula is elliptical, so it has *symmetric* lower and upper tail dependence: `lambda_L = lambda_U` for any pair. The empirical data shows asymmetric tails for several pairs (e.g., SPY-NPI). This is not a model bug — it is a known property of elliptical copulas (McNeil-Frey-Embrechts 2015 Sec 7.3.2) and we acknowledge it as a structural limitation. A non-elliptical pair-copula construction (e.g., asymmetric BB7) would in principle be more flexible, but on this sample size the Vine cannot exploit that flexibility (per the Vuong test).

### 2.5 Simulation engine

Four methods are run on two data treatments (reported / unsmoothed), giving an 8-cell ES matrix:

**Method 1 — Parametric (closed-form).**
```
sigma_p_q  = sqrt(w' Sigma_q w)             (sample quarterly cov)
sigma_total = sigma_p_q * sqrt(12)
mu_p_total  = 12 * w' mu_q                    (mu_q = log(1 + r_a) / 4 from spec)
ES = -mu_p_total + (phi(z_a) / (1 - alpha)) * sigma_total
```
This is the analytic Gaussian benchmark. No MC noise.

**Method 2 — Historical Simulation.** Stationary block bootstrap (Politis-Romano 1994, mean block length 4 quarters = 1 year). Empirical observations are demeaned and the theoretical drift `mu_q` is added back. We sample 1,000,000 paths × 12 steps and compute the ES.

**Method 3 — Filtered Historical Simulation.** Bootstraps from the standardised residual pool `z_t = (r_t - mu_t) / sigma_t` instead of from raw returns. Each path is built by drawing a sequence of standardised residuals (synchronously across the four assets to preserve empirical dependence) and inflating them with an EWMA recursion driven by the asset-specific `lambda` from the Block 3 fit. The starting variance is the unsmoothed log-return variance for PE and NPI, and the AR-residual variance for SPY and AGG. (For PE/NPI the AR-residual variance would be ~20% lower — this is a critical fix; see `changelog.md`.)

**Method 4 — Champion Monte Carlo.**
1. Sample `n_paths * 12` uniforms from the Student-t copula.
2. Clip to `[1e-4, 1 - 1e-4]` to prevent extreme PPF blow-up from the semi-parametric tails (which are fit on only ~8 observations).
3. Per-asset PPF: convert uniforms to standardised residuals.
4. EWMA recursion (same as FHS) yields the simulated returns.
5. Aggregate to 12-quarter portfolio loss.

The path tensor (1,000,000 × 12 × 4) is saved to cache for the decomposition stage.

### 2.6 Stable seeds

All Monte Carlo uses `SEED = 42` with deterministic offsets for sub-modules:
- HS: `SEED = 42`
- HS unsmoothed: `SEED + 1`
- FHS: `SEED + 2`
- Champion: `SEED + 3`
- Tier A bootstrap: `SEED + 1000`
- Tier B bootstrap: `SEED + 2000`

This way each stage is reproducible independently while the full pipeline is reproducible end-to-end.

---

## Section 3 — Model Comparison

### 3.1 The 4 × 2 ES matrix

| Method      | Reported | Unsmoothed |
|-------------|---------:|-----------:|
| Parametric  | 8.81%    | 15.22%     |
| HS          | 22.71%   | 33.04%     |
| FHS         | —        | 33.05%     |
| Champion    | —        | 18.73%     |

Reading the matrix:

- **Top-left (8.81%)** is the naive answer — the paper's starting point.
- **Top-right (15.22%)** isolates the unsmoothing effect: re-running the parametric formula on the KF-unsmoothed PE/NPI nearly doubles the ES.
- **Bottom-left (22.71%)** isolates the non-Gaussian effect: HS on the reported data uses heavier empirical tails and produces a higher ES than the parametric.
- **Bottom-right HS / FHS (33%)** are in-sample-heavy: they mechanically resample from a window that contains 2008. The implicit assumption is that the next three years will look like a randomly stitched version of the historical sample.
- **Champion (18.73%)** is the model-based answer using a stable parametric structure for both marginals and dependence. It does not assume the future will resample 2008; it assumes the 4-asset joint distribution is well-described by Skew-t / semi-parametric marginals with a Student-t copula calibrated to the historical sample.

### 3.2 Why FHS ≈ HS on this sample

A naive expectation might be that FHS gives a different answer from HS because it filters by time-varying volatility. On this quarterly sample the EWMA filter makes only a small difference: `lambda = 0.94` corresponds to a half-life of ~12 quarters (3 years), so within a 12-quarter simulation the conditional variance reverts to its long-run mean. The starting-variance choice for PE/NPI matters more than the recursion itself. After applying the unsmoothed-variance fix described above, FHS sits within 0.05 pp of HS.

### 3.3 Champion vs FHS

The Champion gives 18.73% while FHS gives 33.05% on the same unsmoothed data. The difference is structural:

- HS / FHS resample empirical residuals and so are bound to the empirical tail behaviour of the 80-quarter sample (which contains 2008). Rare extreme events get amplified by the bootstrap (one bad observation can appear up to 12 times in a single simulated path).
- The Champion fits a parametric distribution with `nu = 4.443` and so generalises beyond the empirical sample. The tail behaviour is governed by the fitted `nu`, not by the empirical residuals.

The 18.73% number is the model-based estimate; 33% is the in-sample-heavy estimate. The presentation reports both and leads with 18.73%.

---

## Section 4 — CVaR Decomposition

### 4.1 Component CVaR (Euler additive)

Per-asset Component CVaR is computed by Euler's identity:

```
Comp-CVaR_i = w_i * E[r_i^(3Y) | L_p >= VaR_p^(95%)]
```

The components sum exactly to the total CVaR (an additive decomposition). On the 1M Champion paths:

| Asset | Weight | Component CVaR | Share of total |
|-------|-------:|---------------:|---------------:|
| SPY   | 40%    | 16.30%         | 87%            |
| AGG   | 20%    | -2.19%         | -12%           |
| PE    | 25%    | 3.55%          | 19%            |
| NPI   | 15%    | 1.07%          | 6%             |
| Total | 100%   | 18.73%         | 100%           |

Reading:

- **SPY dominates risk.** With 40% capital weight, SPY delivers 87% of the tail risk. This is partly its higher unconditional volatility (16% vs 4-13% for the other assets) and partly its strong tail dependence with the rest of the portfolio under the Student-t copula.
- **AGG hedges.** A negative Component CVaR means AGG returns *positively* on average in the worst 5% of paths. With a t-copula and lambda close to zero between AGG and the equity risk factors, the conditional bond return is dominated by drift, which in turn is positive (4.5% annualised). This effect would disappear in a "flight-to-quality fails" regime; the model assumes a regime where bonds remain a hedge.
- **PE and NPI contribute moderately.** PE 19% of total risk on 25% capital weight (slightly under-contributing); NPI 6% on 15% capital weight (significantly under-contributing). This is the *dependency-corrected* risk view: PE and NPI have substantial *standalone* tail risk after unsmoothing, but their tail dependence with SPY at this `nu = 4.443` is weaker than the within-equity dependence, so their *contribution* to portfolio CVaR is modest.

### 4.2 Sanity check: does the simulated tail look like the historical GFC?

The simulated worst-5% mean path (across 50,000 tail paths) reaches a cumulative 3-year loss of 18.73% on the portfolio. The historical GFC window (2007Q4-2010Q3) produced an 11.56% portfolio 3-year cumulative loss. The historical Worst#1 window (2006Q2-2009Q1) produced 23.11%.

Per-asset, the simulated tail draws SPY down ~40% on average over three years; the GFC saw SPY down ~30% (cumulatively, including 2009 recovery), and Worst#1 saw SPY down ~50%. The simulated path is between the two empirical points — qualitatively plausible.

This is a sanity check, not a goodness-of-fit test. We are checking that the model does not produce paths that are wildly inconsistent with historical experience, which is necessary but not sufficient.

---

## Section 5 — Robustness and Stress

### 5.1 In-sample backtest (1-quarter VaR/ES)

The Champion model is in principle a 12-quarter forecast. To backtest it against history we project it down to 1-quarter VaR / ES using a normal closed-form on the EWMA conditional volatility plus the historical correlation, then count breaches against actual quarterly portfolio returns.

| Test                | Result                  |
|---------------------|-------------------------|
| Number of obs       | 80                      |
| Expected breaches   | 4 (at 5%)               |
| Observed breaches   | 8                       |
| Kupiec POF p-value  | 0.069 (passes at 5%)    |
| Christoffersen Ind. | 0.002 (rejects at 5%)   |
| Acerbi-Szekely Z    | 0.098 (passes)          |

Reading:
- **Kupiec passes.** The breach rate (8 / 80 = 10%) is double the expected 5% but, on only 80 observations, this is not statistically significant at 5%. Note: the test has very low power on this sample size; a 10% breach rate is roughly at the boundary of the 90% confidence interval around 5%.
- **Christoffersen rejects.** Breaches are clustered (in the GFC and again in 2022). This is a known limitation of single-regime models: they cannot match the *clustering* of breaches even when the *count* is plausible. A regime-switching extension would be required.
- **Acerbi passes.** When breaches do occur, the model's ES is on average correct (the conditional excess of realised loss over predicted ES is small).

### 5.2 Historical replay

The Champion 1M loss distribution is used as the reference; each historical 3-year window is mapped to its percentile in this distribution.

| Window                  | Portfolio 3Y loss | MC percentile |
|-------------------------|------------------:|--------------:|
| GFC (2007Q4-2010Q3)     | 11.56%            | 96.4%         |
| COVID (2019Q4-2022Q3)   | -20.51%           | 45.7%         |
| Worst#1 (2006Q2-2009Q1) | 23.11%            | 98.8%         |
| Worst#2 (2007Q1-2009Q4) | 22.40%            | 98.5%         |
| Worst#3 (2007Q3-2010Q2) | 21.10%            | 98.3%         |

Reading: the worst 3-year windows of the sample sit in the 98-99 percentile of the model's distribution — the model is stricter than the historical experience but not by an unreasonable margin. The COVID period was a *gain* of 20% over three years (the equity rebound dominated the drawdown), so it sits in the middle of the distribution.

### 5.3 Hypothetical stress

We multiply the per-asset losses in the historical GFC window by a uniform factor and recompute the portfolio loss:

| Scenario       | Portfolio loss | vs Baseline ES (18.73%) |
|----------------|---------------:|------------------------:|
| GFC × 1.2      | 13.87%         | 0.74x                  |
| GFC × 1.5      | 17.34%         | 0.93x                  |
| GFC × 2.0      | 23.12%         | 1.23x                  |

We do not include a COVID-based stress scenario. The COVID three-year window (2019Q4-2022Q3) was a portfolio *gain* of 20% (the post-March 2020 rebound dominated the drawdown), so multiplying it by any positive factor amplifies a gain rather than a loss — the operation has no stress-test interpretation. COVID is retained in the historical replay (Section 5.2) as a benchmark event but excluded from the multiplier-based stress family.

Even at GFC × 2.0 — a 60% S&P drawdown spread over 3 years — the portfolio loss is only marginally above the baseline ES. This is because the Champion model already assigns a 5% probability mass to losses above 18.73%, so scaling a historical event only describes one path the model implicitly considers.

### 5.4 Reverse stress

We define a *deep tail* as paths in the simulated distribution where the portfolio 3-year loss is at least 2x the baseline ES (i.e., loss >= 37.46%). On 1M Champion paths, 3,260 paths satisfy this (the 0.33% percentile).

The mean per-asset loss in the deep tail is:

| Asset | Mean 3Y loss | Mean 3Y loss × weight |
|-------|-------------:|----------------------:|
| SPY   | 105.5%       | 42.2%                 |
| AGG   | -10.4%       | -2.1%                 |
| PE    | 37.8%        | 9.5%                  |
| NPI   | 13.1%        | 2.0%                  |
| Total |              | 51.6% (~ 2x baseline) |

Reading: for the ES to double, equity (SPY) has to draw down ~80-110% cumulatively over 3 years (heavily compounded), bonds have to *gain* (the negative-correlation hedge stays intact), and PE has to drop by ~38%. This scenario is more extreme than the GFC × 2 stress and corresponds to a financial-crisis-cum-deflation regime.

### 5.5 Bootstrap confidence intervals

Two layers of uncertainty are quantified separately.

**Tier A — Monte-Carlo sampling noise.** Resample with replacement from the 1M loss array (2,000 bootstraps × 1M each). Result:
- VaR 95%: [8.06%, 8.14%] (point: 8.10%)
- ES 95%: [18.59%, 18.87%] (point: 18.73%)

This is the residual uncertainty from finite N at the simulation stage; the answer is essentially *known to two decimal places* given the model.

**Tier B — Parameter uncertainty.** Resample the 80 PIT uniforms with replacement (iid row bootstrap; the EWMA filter has already removed serial dependence in residuals). For each bootstrap sample, refit the Student-t copula (Kendall R + grid search on `nu`) and resimulate 100,000 paths × 12 quarters with constant per-asset volatility. Across 500 bootstraps:
- ES 95% percentile CI: [12.40%, 19.82%]
- ES 95% Wald CI (centered on the point estimate): [15.0%, 22.5%]
- Recovered `nu` mean: 5.21 (vs the point estimate 4.44)

The percentile CI is asymmetric because the bootstrap recovers a slightly higher mean `nu`, which corresponds to slightly thinner tails and lower ES on average. The Wald CI symmetrises around the point estimate.

Reading: Tier B is ~27x wider than Tier A, so parameter uncertainty (the dependence-structure choice) dominates Monte-Carlo noise by an order of magnitude.

---

## Section 6 — Conclusion

Three takeaways:

1. **The naive answer (8.81%) is wrong by 2x.** Three pathologies — smoothing in private-asset returns, asymmetric tail dependence, long horizon vs short sample — each contribute. After addressing all three, the answer is ES = **18.73%**.
2. **The portfolio is structurally an SPY portfolio with an AGG hedge.** Component CVaR shows 87% of risk in SPY and -12% (negative; a hedge) in AGG, despite the 40% / 20% capital weights. PE and NPI together contribute only 25% of risk on 40% capital.
3. **Parameter uncertainty dominates.** With Tier A [18.59%, 18.87%] vs Tier B [15.0%, 22.5%], the dependence-structure choice (Student-t, nu = 4.4) is the single largest source of remaining uncertainty. Under different copula families (e.g. an asymmetric Vine), the answer could plausibly range from 18% to 25%.

**Headline:** ES (95%, 3Y) = 18.73%, with a stated parameter range of [15.0%, 22.5%] reflecting dependence-structure uncertainty.

---

## Appendix A — Notation

| Symbol            | Meaning                                                |
|-------------------|--------------------------------------------------------|
| `r_t`             | Quarter-t log return (per asset or per portfolio)      |
| `r*_t`            | Reported (smoothed) return                             |
| `mu_t`            | Conditional mean                                       |
| `sigma_t`         | Conditional standard deviation (EWMA)                  |
| `lambda`          | EWMA decay parameter                                   |
| `z_t`             | Standardised residual                                  |
| `u_t = F_z(z_t)`  | PIT uniform                                            |
| `R`               | Copula correlation matrix (Kendall-tau-based init)     |
| `nu`              | Student-t copula degrees of freedom                    |
| `lambda_L(i,j)`   | Lower-tail dependence between assets i and j           |
| `chi(u)`          | Empirical conditional probability used in chi-plot     |
| `alpha`           | VaR / ES confidence level (here 0.95)                  |

---

## Appendix B — Implementation map

| Section | Code module                                  |
|---------|----------------------------------------------|
| 1.x     | `src/data_loader.py`                         |
| 2.2     | `src/unsmoothing.py`                         |
| 2.3     | `src/marginals.py`                           |
| 2.4     | `src/copula.py`                              |
| 2.5     | `src/simulation.py`                          |
| 4.x     | `src/analysis.py` (`component_cvar`, `time_of_loss`) |
| 5.1     | `src/analysis.py` (`in_sample_backtest`)     |
| 5.2     | `src/analysis.py` (`historical_replay`)      |
| 5.3     | `src/analysis.py` (`stress_scenarios`)       |
| 5.4     | `src/analysis.py` (`reverse_stress`)         |
| 5.5     | `src/analysis.py` (`tier_a_bootstrap`, `tier_b_bootstrap`) |
| All figures | `src/plots.py`                            |

---

## Appendix C — References

- Acerbi, C., Szekely, B. (2014). Back-testing Expected Shortfall. *Risk Magazine*, December.
- Christoffersen, P. (1998). Evaluating Interval Forecasts. *International Economic Review*, 39, 841-862.
- Demarta, S., McNeil, A. (2005). The t Copula and Related Copulas. *International Statistical Review*, 73, 111-129.
- Fisher, J., Geltner, D., Webb, R. (1994). Value Indices of Commercial Real Estate: A Comparison of Index Construction Methods. *Journal of Real Estate Finance and Economics*, 9, 137-164.
- Hansen, B. (1994). Autoregressive Conditional Density Estimation. *International Economic Review*, 35, 705-730.
- Higham, N. (1988). Computing a Nearest Symmetric Positive Semidefinite Matrix. *Linear Algebra and its Applications*, 103, 103-118.
- Joe, H. (2014). *Dependence Modeling with Copulas*. Chapman and Hall.
- Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. *Journal of Derivatives*, 3, 73-84.
- McNeil, A., Frey, R., Embrechts, P. (2015). *Quantitative Risk Management*, 2nd ed. Princeton University Press.
- Okunev, J., White, R. (2003). Hedge Fund Risk Factors and Value at Risk of Credit Trading Strategies. Working Paper, UNSW.
- Politis, D., Romano, J. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*, 89, 1303-1313.
- Vuong, Q. (1989). Likelihood Ratio Tests for Model Selection and Non-nested Hypotheses. *Econometrica*, 57, 307-333.
