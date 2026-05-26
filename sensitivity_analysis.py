"""
=============================================================================
Sensitivity Analysis  –  One-at-a-time parameter sweep
=============================================================================
Tests robustness of the BL (and other) portfolios by varying ONE parameter
at a time while holding all others at the baseline values.

Parameters swept
----------------
  lookback    : 24, 36 (baseline), 48  months
  max_weight  : 20%, 30% (baseline), 40%  per-stock cap
  k_clusters  : 2, 3, 4, 5, 6 (baseline)  K-means clusters
  rf_rate     : 4%, 5%, 6% (fixed),  VN10Y (baseline, time-varying)

Baseline: lookback=36, max_weight=30%, k=4, rf=VN10Y

Monte Carlo is skipped (N_MC_SIMS=0) and optimizer starts are reduced
to keep each sensitivity run fast (~15-30 s).

Output
------
  output/sensitivity_results.csv  – Ann_Return, Ann_Vol, Sharpe, Sortino,
                                     MDD for every (parameter, value,
                                     portfolio) combination.
=============================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Import core classes from backtest_v2 (same directory)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_v2 import (
    Config, build_rf_series,
    DataLoader, ClusterSignalGenerator,
    BlackLittermanEngine, PortfolioOptimiser,
    PerformanceCalculator,
)


# =============================================================================
#  FAST (SILENT) BACKTESTER  –  no MC, fewer optimizer starts
# =============================================================================

class FastBacktester:
    """
    Stripped-down backtester for sensitivity sweeps.
    Returns a dict {portfolio_name: metrics_dict} after one full run.
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.loader = DataLoader(cfg)
        self.signal = ClusterSignalGenerator(cfg)
        self.bl_eng = BlackLittermanEngine(cfg)
        self.optim  = PortfolioOptimiser(cfg)
        self.perf   = PerformanceCalculator(cfg)

    def run(self) -> dict:
        np.random.seed(self.cfg.RANDOM_SEED)

        _, mktcap, log_ret, simple_ret, rf_series = self.loader.load()
        dates = log_ret.index
        L     = self.cfg.LOOKBACK
        oos_rows = []

        for i in range(L, len(dates) - 1):
            lw       = log_ret.iloc[i - L : i]
            oos_sr   = simple_ret.iloc[i + 1]
            oos_date = dates[i + 1]

            rf_m_train = float(rf_series.iloc[i])
            rf_a_train = rf_m_train * 12
            rf_m_oos   = float(rf_series.iloc[i + 1])

            active_mask = lw.notna().all() & oos_sr.notna()
            tickers     = active_mask[active_mask].index.tolist()
            if len(tickers) < self.cfg.N_CLUSTERS + 2:
                continue

            n     = len(tickers)
            lw_a  = lw[tickers].values
            oos_r = oos_sr[tickers].values

            mc_row = mktcap.iloc[i][tickers].fillna(0.0).clip(lower=0.0)
            mc_sum = mc_row.sum()
            if mc_sum <= 0.0:
                continue
            w_mkt = (mc_row / mc_sum).values

            Sigma   = np.cov(lw_a.T) * 12 + np.eye(n) * 1e-8
            mu_hist = lw_a.mean(axis=0) * 12

            sig  = self.signal.generate(lw_a, tickers, rf_a_train)
            P, q = sig["P"], sig["q"]

            delta = self.bl_eng.compute_delta(mu_hist, Sigma, w_mkt, rf_a_train)
            Pi    = delta * Sigma @ w_mkt
            mu_BL, Sigma_BL, _, _ = self.bl_eng.posterior(Pi, Sigma, P, q)

            w_TAN    = self.optim.tangency(mu_hist, Sigma, rf_a_train)
            w_MV     = self.optim.min_variance(Sigma)
            w_BL     = self.optim.bl_equilibrium(Pi, Sigma, rf_a_train)
            w_BL_KIO = self.optim.bl_tangency(mu_BL, Sigma_BL, rf_a_train)
            w_EW     = self.optim.equal_weight(n)
            w_RP     = self.optim.risk_parity(Sigma)

            oos_rows.append(dict(
                Date=oos_date, RF_monthly=rf_m_oos,
                MKT=float(w_mkt    @ oos_r),
                TAN=float(w_TAN    @ oos_r),
                MV =float(w_MV     @ oos_r),
                BL =float(w_BL     @ oos_r),
                BL_KIO=float(w_BL_KIO @ oos_r),
                EW =float(w_EW     @ oos_r),
                RP =float(w_RP     @ oos_r),
            ))

        if not oos_rows:
            return {}

        df     = pd.DataFrame(oos_rows)
        df_idx = df.set_index("Date")
        rf_s   = df_idx["RF_monthly"]

        return {
            p: self.perf.compute(df_idx[p], rf_s, p)
            for p in ["MKT", "TAN", "MV", "BL", "BL_KIO", "EW", "RP"]
        }


# =============================================================================
#  CONFIG FACTORY
# =============================================================================

def make_config(param: str, value) -> Config:
    """
    Create a Config that differs from the baseline in exactly one parameter.
    Monte Carlo is disabled (N_MC_SIMS=0) and optimizer starts are reduced
    for speed.
    """
    cfg = Config()
    cfg.N_MC_SIMS  = 0   # skip Monte Carlo in sensitivity runs
    cfg.OPT_STARTS = 4   # fewer starts → ~3x faster per run

    if param == "lookback":
        cfg.LOOKBACK = int(value)
        cfg.TAU      = 1.0 / int(value)     # τ = 1/T as in main backtest

    elif param == "max_weight":
        cfg.MAX_WEIGHT = float(value)

    elif param == "k_clusters":
        cfg.N_CLUSTERS = int(value)

    elif param == "rf_rate":
        if value == "dynamic":
            cfg.USE_DYNAMIC_RF = True
        else:
            cfg.USE_DYNAMIC_RF = False
            cfg.RF_ANNUAL      = float(value)
            cfg.RF_MONTHLY     = float(value) / 12.0

    return cfg


# =============================================================================
#  PARAMETER GRID
# =============================================================================

# (parameter_name, list_of_values, baseline_value)
PARAM_GRID = [
    ("lookback",   [24, 36, 48],             36),
    ("max_weight", [0.20, 0.30, 0.40],       0.30),
    ("k_clusters", [2, 3, 4, 5, 6],           4),
    ("rf_rate",    [0.04, 0.05, 0.06, "dynamic"], "dynamic"),
]


# =============================================================================
#  MAIN SENSITIVITY LOOP
# =============================================================================

def run_sensitivity() -> pd.DataFrame:
    records = []

    for param, values, baseline in PARAM_GRID:
        print(f"\n{'─'*60}")
        print(f"  Parameter: {param}  (baseline = {baseline})")
        print(f"{'─'*60}")

        for value in values:
            is_baseline = (value == baseline)

            # Human-readable label
            if param == "max_weight":
                label = f"{value:.0%}"
            elif param == "rf_rate":
                label = "VN10Y (dynamic)" if value == "dynamic" \
                        else f"{value:.0%} fixed"
            else:
                label = str(value)

            tag = f"{label}  ← baseline" if is_baseline else label
            print(f"  Running: {tag:<30}", end=" ", flush=True)

            cfg     = make_config(param, value)
            results = FastBacktester(cfg).run()

            if not results:
                print("  (skipped – too few OOS months)")
                continue

            bl_kio_sharpe = results["BL_KIO"]["Sharpe"]
            ew_sharpe     = results["EW"]["Sharpe"]
            print(f"BL_KIO Sharpe={bl_kio_sharpe:+.4f}  EW Sharpe={ew_sharpe:+.4f}")

            for port, metrics in results.items():
                records.append(dict(
                    Parameter=param,
                    Value=label,
                    Is_Baseline=is_baseline,
                    Portfolio=port,
                    Ann_Return=metrics["Ann_Return"],
                    Ann_Vol=metrics["Ann_Vol"],
                    Sharpe=metrics["Sharpe"],
                    Sortino=metrics["Sortino"],
                    MDD=metrics["MDD"],
                    Calmar=metrics["Calmar"],
                    Win_Rate=metrics["Win_Rate"],
                ))

    df = pd.DataFrame(records)

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/sensitivity_results.csv", index=False)
    print(f"\n{'='*60}")
    print(f"  Saved: output/sensitivity_results.csv  ({len(df):,} rows)")

    # ── Summary pivot: BL_KIO Sharpe across each parameter ───────────────
    print("\n  BL_KIO Sharpe ratio by parameter value:")
    print("  (asterisk = baseline)\n")
    bl_kio = df[df["Portfolio"] == "BL_KIO"].copy()
    for param, values, baseline in PARAM_GRID:
        sub = bl_kio[bl_kio["Parameter"] == param][["Value", "Is_Baseline", "Sharpe"]]
        print(f"  {param}:")
        for _, row in sub.iterrows():
            marker = " *" if row["Is_Baseline"] else "  "
            print(f"    {marker} {str(row['Value']):<22} Sharpe = {row['Sharpe']:+.4f}")

    return df


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Sensitivity Analysis – One-at-a-time parameter sweep")
    print("=" * 60)
    run_sensitivity()
