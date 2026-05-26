# BL-KIO: Black-Litterman with K-means Idiosyncratic-signal Optimisation

Vietnamese bank stock portfolio backtest (2017–2024, monthly, OOS).

## Portfolios

| Symbol | Description |
|--------|-------------|
| MKT | Market-cap weighted |
| TAN | Tangency (max Sharpe, historical μ) |
| BL | Black-Litterman equilibrium, no views (BL-IO) |
| EW | Equal-weight 1/N |
| BL_KIO | **BL + K-means idiosyncratic signal** (proposed model) |

## Structure

```
config.py       — Config, VN10Y rf table, build_rf_series
data.py         — DataLoader
signal.py       — ClusterSignalGenerator (K-means: idio momentum + low-vol)
bl.py           — BlackLittermanEngine (reverse-optim prior + Bayesian update)
optimise.py     — PortfolioOptimiser (tangency, BL, equal-weight)
perf.py         — PerformanceCalculator, t-test, Jobson-Korkie, JB, LB, drawdown
mc.py           — MonteCarloEngine (BL_KIO vs BL_IO view-noise comparison)
backtest.py     — Backtester (rolling OOS loop + all CSV outputs)
run.py          — Entry point
```

### Analysis scripts
```
sensitivity_analysis.py    — one-at-a-time parameter sweep
kmeans_k_validation.py     — k selection: Elbow, Silhouette, CH, DB, Gap Statistic
dm_bootstrap_test.py       — Diebold-Mariano test + Bootstrap Sharpe CI
plot_monte_carlo.py        — Monte Carlo density figure (BL_KIO vs BL_IO)
plot_density_comparison.py — Static vs MC comparison figure
export_pdf.py              — Full PDF report
```

## Data

Place in `data/`:
- `bank_monthly_close.csv` — adjusted monthly close (date index, ticker columns)
- `bank_monthly_mktcap_bn.csv` — market cap in billion VND (same shape)

## Reproduce

```bash
python run.py                     # main backtest → output/*.csv
python sensitivity_analysis.py    # parameter sweep
python kmeans_k_validation.py     # k-selection analysis
python dm_bootstrap_test.py       # DM test + bootstrap CI
python plot_monte_carlo.py        # Monte Carlo density figure
python export_pdf.py              # full PDF report
```

## Outputs (`output/`)

| File | Description |
|------|-------------|
| `oos_returns_v2.csv` | Monthly OOS returns, 5 portfolios + RF |
| `weights_v2.csv` | Portfolio weights per rebalancing date |
| `cluster_signals_v2.csv` | K-means cluster labels and view positions |
| `monte_carlo_v2.csv` | BL_KIO vs BL_IO Sharpe/Return/Vol under view noise |
| `performance_summary_v2.csv` | Ann. Return, Vol, Sharpe, Sortino, MDD, Calmar |
| `ttest_results_v2.csv` | Paired t-test: BL_KIO vs each benchmark |
| `jobson_korkie_results.csv` | Jobson-Korkie z-test: BL_KIO vs each benchmark |
| `distribution_jb_tests.csv` | Jarque-Bera normality test |
| `distribution_lb_tests.csv` | Ljung-Box autocorrelation test |
| `drawdown_summary.csv` | Drawdown aggregate stats |
| `drawdown_periods.csv` | Individual drawdown events |
| `signal_ic_analysis.csv` | Information coefficient (Spearman rank correlation) |
| `subperiod_analysis.csv` | Full / Period-1 / Period-2 performance |

## Key Results (OOS 2017–2024)

| Portfolio | Ann. Return | Sharpe | MDD |
|-----------|-------------|--------|-----|
| MKT | ~16% | ~0.58 | ~-47% |
| TAN | ~19% | ~0.56 | ~-47% |
| BL | ~27% | ~0.80 | ~-42% |
| EW (1/N) | ~22% | ~0.75 | ~-46% |
| **BL_KIO** | **~30%** | **~0.94** | **~-41%** |

BL_KIO vs TAN: Jobson-Korkie z = 2.33, p = 0.020 (**)

## Config (`config.py`)

| Parameter | Value | Description |
|-----------|-------|-------------|
| LOOKBACK | 36 | Training window months |
| N_CLUSTERS | 4 | K-means clusters |
| MAX_WEIGHT | 0.30 | Per-stock weight cap |
| TAU | 1/36 | BL prior uncertainty scalar |
| N_MC_SIMS | 2000 | Monte Carlo simulations |
| SIGNAL_LOOKBACK | 6 | Momentum lookback months |
| SIGNAL_SKIP_LAST | 1 | Skip last month (short-term reversal) |
