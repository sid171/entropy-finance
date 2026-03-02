# Fintropy v2: Making the Entropy Layer Empirically Defensible

*Building a financial tool is one thing. Being able to defend every number in it is another. Here's what changed.*

---

## The Problem with "It Looks About Right"

The first version of Fintropy worked. Entropy scores moved in the right direction. The Entropy-Adjusted WACC responded to regime change. The Monte Carlo stress-tested assumptions. But a version of the same critique applies to financial models as to financial tools: precision without justification is just noise with a decimal point.

Specifically, three calibration choices in the original implementation were reasonable but not rigorously grounded:

1. **Entropy normalization** was linear and hand-tuned — the uncertainty score formula `(entropy - 2.0) / 2.0 × 100` assumed a plausible range for Shannon entropy values, but that range was not derived from data. It was a guess.
2. **Transfer entropy binning** used a fixed value of 10 bins regardless of how much data was available. With only one year of daily returns (~252 observations), 10 bins produces sparse joint probability tables and systematically inflated transfer entropy estimates.
3. **Monte Carlo WACC uncertainty** was set at 15% of the base WACC — a round number with no empirical anchor. A WACC of 10% with a standard deviation of 1.5% might be approximately correct on average, but it ignores the actual volatility profile of the specific stock being analyzed.

This post documents how each of these was fixed, and why the fixes matter for a tool that claims to add intelligence rather than just complexity.

---

## Fix 1: Empirical Entropy Normalization Across 57 Stocks

The entropy score pipeline produces four sub-scores — uncertainty, information flow, regime instability, and relationship stress — each of which needs to be mapped to a common 0–100 scale before weighting and combining. The original implementation used arbitrary linear functions. The new implementation uses empirically derived percentile anchors.

The calibration dataset consists of 57 stocks spanning all 11 GICS sectors, analyzed over a two-year window of daily returns (2022–2023). For each metric, the 5th and 95th percentiles across all stocks define the normalization range: a stock at the 5th percentile receives a score of 0, and a stock at the 95th percentile receives a score of 100.

**Key calibrated ranges:**

| Metric | p5 | p50 | p95 |
|---|---|---|---|
| Shannon entropy (nats, bins=50) | 2.68 | 3.12 | 3.72 |
| \|Net transfer entropy\| (nats) | 0.000 10 | 0.001 20 | 0.006 50 |
| PELT changepoints (2Y window) | 1 | 5 | 11 |
| Rolling correlation vs SPY | 0.15 | 0.58 | 0.85 |

These ranges are now hardcoded in `entropy_calibration.py` with full documentation of the methodology, the stocks used, and the mapping functions. Every normalization step is traceable to a specific empirical distribution rather than to intuition.

The practical effect is significant. Under the old normalization, a stock with a Shannon entropy of 3.0 nats received an uncertainty score of 50 regardless of whether 3.0 was typical or unusual. Under the new normalization, 3.0 nats maps to approximately 31 — correctly reflecting that it sits in the lower-middle of the empirical distribution across real market data.

---

## Fix 2: Adaptive Binning for Transfer Entropy

Transfer entropy is computed by discretizing continuous return series into histogram bins and estimating joint probability distributions from those bins. The accuracy of the estimate depends critically on having enough observations per bin: too many bins on too little data produces empty cells, which the estimator fills with zeros, which inflates the measured information transfer.

The fix follows the Rice rule for adaptive bin selection, with thresholds validated against the empirical calibration dataset:

| Sample size | Bins selected |
|---|---|
| < 100 observations | 4 |
| 100–199 observations | 5 |
| 200–399 observations | 6 |
| ≥ 400 observations | 8 |

For a standard one-year analysis window (~252 trading days), the system now selects 6 bins rather than the previous 10. This reduces the joint state space by 64% (from 100 cells to 36), substantially improving estimation reliability for the data volumes typical in this application. The adjustment follows the methodological guidance in Schreiber (2000), who notes that bin count should be chosen to ensure adequate occupancy of the joint probability tables.

The `adaptive_te_bins()` function is exposed as a public utility so the same logic applies consistently across all transfer entropy computations in the codebase — both in the direct API and through the entropy radar pipeline.

---

## Fix 3: Data-Driven WACC Standard Deviation

The Monte Carlo DCF generates 1,500 scenarios by sampling WACC from a normal distribution. The previous implementation set the standard deviation at 15% of the base WACC — a plausible prior, but one that treats a high-volatility technology stock the same as a stable utility.

The new implementation derives σ(WACC) from two empirical sources specific to each ticker:

**σ(β) from rolling 2-year betas.** Using daily returns for the stock and SPY, 252-day rolling betas are computed over the full two-year window. The standard deviation of these rolling betas estimates how unstable the stock's systematic risk loading has been historically. Under CAPM, this propagates to equity cost uncertainty via: σ(Ke) = ERP × σ(β), where ERP is the long-run equity risk premium of 5%.

**σ(Rf) from 10-year Treasury yield changes.** Daily changes in the 10-year Treasury yield (^TNX) are fetched and annualized. This captures the uncertainty introduced by shifts in the risk-free rate — a source of WACC volatility that is often overlooked but was particularly significant during 2022–2023.

The two components are combined using error propagation:

σ(WACC) ≈ (E/V) × √[ (ERP × σ_β)² + σ_Rf² ]

where E/V is the equity weight derived from the stock's actual market capitalization and debt levels. The result is bounded between 0.5% and 4% to prevent degenerate scenarios, and is displayed in the Monte Carlo UI alongside the simulation results so that analysts can inspect the uncertainty estimate alongside the output it drives.

For a typical large-cap technology stock, this yields a WACC σ of approximately 1.3–1.8% — close to the old fixed assumption, but now derived from actual data and meaningfully different for high-beta or high-duration names.

---

## Fix 4: Parallel Sector Heatmap Fetching

The sector heatmap computes entropy scores across 55+ stocks. In the original implementation this was done sequentially, with each stock requiring a full entropy radar computation involving multiple yfinance calls. Wall-clock time for a full heatmap was 5–8 minutes, making it impractical for interactive use.

The new implementation uses Python's `concurrent.futures.ThreadPoolExecutor` with a pool of 8 workers. Each stock's entropy computation is submitted as an independent task, and results are collected as they complete using `as_completed()`. This reduces heatmap computation time to approximately one minute — an improvement of roughly 6×, achieved without changing any of the underlying entropy computation logic.

The implementation correctly handles failures at the per-stock level: if a single ticker raises an exception (network timeout, missing data), the error is logged and the computation continues for all remaining stocks. The heatmap result always reports both successful analyses and a per-stock error log.

---

## CI/CD and Validation: From 12 Tests to 62

Beyond the four functional improvements, the project's testing infrastructure was substantially expanded to match the standard expected of a production-grade tool.

**Test coverage by module:**

| Module | Tests | Coverage |
|---|---|---|
| `entropy_tools.py` | 12 | 94% |
| `entropy_calibration.py` | 28 | 100% |
| `valuation.py` (pure logic) | 15 | — |
| `backtest.py` | 9 | 98% |

The new test suites cover boundary conditions and edge cases that the original tests did not reach: normalization clamping behavior at distribution extremes, in-sample/out-of-sample isolation in the backtest (verifying that no signal dates precede the split date), WACC standard deviation bounds under mocked network conditions, and monotonicity invariants for all calibration functions.

The CI/CD pipeline was upgraded from a single pytest job to a two-stage workflow:

1. **Lint stage** — `ruff` checks all Python files for import errors, undefined names, and style violations. The test stage does not run if linting fails.
2. **Test stage** — runs on Python 3.11 and 3.12 in parallel, with pip caching for speed, full import verification across all modules, and a 90% coverage threshold enforced on the pure-logic modules.

All 62 tests pass locally and on both Python versions in CI. The repository README now displays a live CI status badge.

---

## Conclusion

The distinction between a working prototype and a defensible tool is empirical grounding. Every number that Fintropy produces now traces back either to a cited methodology (Schreiber 2000, Killick 2012, Damodaran 2023) or to a calibration dataset of real market data. The normalization ranges are not guesses; the transfer entropy bin counts are not arbitrary; the WACC uncertainty is not a fixed percentage applied universally.

This iteration also illustrates a broader principle in applied quantitative finance: the choice of hyperparameters is not a technical detail. It is a modeling assumption, and modeling assumptions should be justified. Building the infrastructure to justify them — cross-stock calibration, adaptive estimation, data-driven uncertainty — is what separates a tool from a toy.

The full source code is available at [github.com/sid171/entropy-finance](https://github.com/sid171/entropy-finance).

---

*References*

- Schreiber, T. (2000). Measuring Information Transfer. *Physical Review Letters*, 85(2), 461–464.
- Killick, R., Fearnhead, P., & Eckley, I.A. (2012). Optimal Detection of Changepoints with a Linear Computational Cost. *Journal of the American Statistical Association*, 107(500), 1590–1598.
- Damodaran, A. (2023). *Cost of Capital by Sector*. NYU Stern School of Business.
- Rice, J.A. (2006). *Mathematical Statistics and Data Analysis* (3rd ed.). Duxbury Press.

*MGMT 69000: Mastering AI for Finance | Purdue MSF*
