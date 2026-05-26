"""
=============================================================================
Diebold-Mariano Test & Bootstrap Sharpe CI  –  k=4 vs k=2 for BL_KIO
=============================================================================
Two complementary statistical tests to justify k=4 over k=2:

  A. Diebold-Mariano (1995) + Harvey-Leybourne-Newey (1997) small-sample
       Loss differential: d_t = r_k4_t − r_k2_t
       H₀: E[d_t] ≤ 0  (k=4 not better)
       H₁: E[d_t] > 0  (k=4 better) — one-sided
       HAC variance via Newey-West (max_lag = ⌊T^(1/3)⌋)

  B. Bootstrap Sharpe CI (Efron & Tibshirani 1994, percentile method)
       B = 5,000 resamples; 95% CI for each k
       Non-overlap ↔ statistically distinct Sharpe ratios

Both tests run on IDENTICAL OOS dates (inner join of k=4 and k=2 windows).

Output
------
  output/dm_bootstrap_results.csv  – summary metrics
  output/dm_bootstrap_monthly.csv  – aligned monthly returns k2 / k4
=============================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_v2 import (
    Config, DataLoader,
    ClusterSignalGenerator, BlackLittermanEngine, PortfolioOptimiser,
)

N_BOOTSTRAP = 5_000
RANDOM_SEED = 42


# =============================================================================
#  HELPER: run BL_KIO backtest for a given k, return aligned DataFrame
# =============================================================================

def run_bl_kio_series(k: int) -> pd.DataFrame:
    """
    Run BL_KIO with N_CLUSTERS=k, collect monthly OOS returns.
    Returns DataFrame indexed by Date with columns [r_BL_KIO, rf_monthly].
    """
    cfg = Config()
    cfg.N_CLUSTERS = k
    cfg.N_MC_SIMS  = 0
    cfg.OPT_STARTS = 4

    loader = DataLoader(cfg)
    signal = ClusterSignalGenerator(cfg)
    bl_eng = BlackLittermanEngine(cfg)
    optim  = PortfolioOptimiser(cfg)

    np.random.seed(cfg.RANDOM_SEED)
    _, mktcap, log_ret, simple_ret, rf_series = loader.load()

    dates = log_ret.index
    L     = cfg.LOOKBACK
    rows  = []

    for i in range(L, len(dates) - 1):
        lw       = log_ret.iloc[i - L : i]
        oos_sr   = simple_ret.iloc[i + 1]
        oos_date = dates[i + 1]

        rf_m_train = float(rf_series.iloc[i])
        rf_a_train = rf_m_train * 12
        rf_m_oos   = float(rf_series.iloc[i + 1])

        active_mask = lw.notna().all() & oos_sr.notna()
        tickers     = active_mask[active_mask].index.tolist()
        if len(tickers) < cfg.N_CLUSTERS + 2:
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

        sig  = signal.generate(lw_a, tickers, rf_a_train)
        P, q = sig["P"], sig["q"]

        delta = bl_eng.compute_delta(mu_hist, Sigma, w_mkt, rf_a_train)
        Pi    = delta * Sigma @ w_mkt
        mu_BL, Sigma_BL, _, _ = bl_eng.posterior(Pi, Sigma, P, q)

        w_BL_KIO = optim.bl_tangency(mu_BL, Sigma_BL, rf_a_train)

        rows.append({
            "Date":       oos_date,
            "r_BL_KIO":  float(w_BL_KIO @ oos_r),
            "rf_monthly": rf_m_oos,
        })

    return pd.DataFrame(rows).set_index("Date")


# =============================================================================
#  TEST A: Diebold-Mariano with Newey-West HAC + HLN small-sample correction
# =============================================================================

def _newey_west_var(d: np.ndarray, max_lag: int) -> float:
    """
    Newey-West (1987) long-run variance of d (mean-demeaned).
    σ̂² = γ₀ + 2·Σ w_j·γ_j   (Bartlett kernel, w_j = 1 − j/(h+1))
    """
    T   = len(d)
    dm  = d - d.mean()
    g0  = float(dm @ dm) / T
    nw  = g0
    for j in range(1, max_lag + 1):
        w    = 1.0 - j / (max_lag + 1.0)
        g_j  = float(dm[j:] @ dm[:-j]) / T
        nw  += 2.0 * w * g_j
    return max(nw, 1e-14)   # numerical floor


def diebold_mariano_test(d: np.ndarray) -> dict:
    """
    One-sided DM test on loss differential d_t = r_k4_t − r_k2_t.
    H₀: E[d] ≤ 0  →  p-value = P(t > DM | H₀).

    Harvey, Leybourne & Newey (1997) small-sample correction:
      DM_HLN = DM × √((T+1 − 2h + h(h−1)/T) / T)
      referenced to t(T−1) distribution.
    """
    T     = len(d)
    d_bar = d.mean()
    h     = max(1, int(np.floor(T ** (1.0 / 3.0))))   # optimal lag ≈ T^(1/3)

    nw_var  = _newey_west_var(d, max_lag=h)
    dm_stat = d_bar / np.sqrt(nw_var / T)

    # HLN correction
    hln_fac = np.sqrt((T + 1 - 2*h + h*(h - 1) / T) / T)
    dm_hln  = dm_stat * hln_fac

    p_val  = float(1.0 - stats.t.cdf(dm_hln, df=T - 1))

    return {
        "T":                T,
        "h_lags":           h,
        "d_bar_monthly":    float(d_bar),
        "d_bar_annual":     float(d_bar * 12),
        "DM_stat":          float(dm_stat),
        "DM_HLN":           float(dm_hln),
        "p_value":          float(p_val),
        "sig_10pct":        p_val < 0.10,
        "sig_5pct":         p_val < 0.05,
        "sig_1pct":         p_val < 0.01,
        "NW_max_lag":       h,
    }


# =============================================================================
#  TEST B: Bootstrap CI for Sharpe ratio (percentile method)
# =============================================================================

def _geometric_sharpe(r: np.ndarray, rf: np.ndarray) -> float:
    """
    Annualised Sharpe using geometric compounding — consistent with
    PerformanceCalculator.compute() in backtest_v2.py:
      ann_ret = (1 + mean_monthly_r)^12 − 1
      ann_vol = std(r, ddof=1) × √12
      rf_ann  = mean(rf_monthly) × 12
    """
    ann_ret = (1.0 + r.mean()) ** 12 - 1.0
    ann_vol = r.std(ddof=1) * np.sqrt(12)
    rf_ann  = rf.mean() * 12
    return (ann_ret - rf_ann) / ann_vol if ann_vol > 1e-8 else 0.0


def bootstrap_sharpe(r: np.ndarray, rf: np.ndarray,
                     B: int = N_BOOTSTRAP,
                     seed: int = RANDOM_SEED) -> dict:
    """
    Percentile bootstrap CI for annualised Sharpe ratio.
    Uses geometric formula consistent with main backtest PerformanceCalculator.
    """
    rng      = np.random.default_rng(seed)
    T        = len(r)
    sr_point = _geometric_sharpe(r, rf)

    sr_boot = np.empty(B)
    for b in range(B):
        idx        = rng.integers(0, T, size=T)
        sr_boot[b] = _geometric_sharpe(r[idx], rf[idx])

    ci_lo = float(np.percentile(sr_boot, 2.5))
    ci_hi = float(np.percentile(sr_boot, 97.5))

    return {
        "Sharpe_point":  float(sr_point),
        "Boot_SE":       float(sr_boot.std()),
        "Boot_mean":     float(sr_boot.mean()),
        "CI_95_lo":      ci_lo,
        "CI_95_hi":      ci_hi,
    }


# =============================================================================
#  MAIN
# =============================================================================

def run() -> pd.DataFrame:
    print("=" * 65)
    print("  DM Test & Bootstrap Sharpe CI  –  BL_KIO(k=4) vs BL_KIO(k=2)")
    print("=" * 65)

    # ── Run both series ───────────────────────────────────────────────────────
    print("\n[1/4] Running BL_KIO with k=4 …", flush=True)
    df4 = run_bl_kio_series(k=4)

    print("[2/4] Running BL_KIO with k=2 …", flush=True)
    df2 = run_bl_kio_series(k=2)

    # ── Align on common dates (inner join) ────────────────────────────────────
    common = df4.index.intersection(df2.index)
    r4  = df4.loc[common, "r_BL_KIO"].values
    r2  = df2.loc[common, "r_BL_KIO"].values
    rf4 = df4.loc[common, "rf_monthly"].values
    rf2 = df2.loc[common, "rf_monthly"].values
    T   = len(common)
    print(f"       Common OOS dates: {T}  "
          f"({common[0].date()} → {common[-1].date()})")

    # Save monthly comparison
    monthly = pd.DataFrame({
        "Date":       common,
        "r_k4":       r4,
        "r_k2":       r2,
        "rf":         rf4,
        "d_t":        r4 - r2,
        "excess_k4":  r4 - rf4,
        "excess_k2":  r2 - rf2,
    })
    os.makedirs("output", exist_ok=True)
    monthly.to_csv("output/dm_bootstrap_monthly.csv", index=False)

    # ── Test A: DM test ───────────────────────────────────────────────────────
    print("\n[3/4] Running Diebold-Mariano test …", flush=True)
    d      = r4 - r2
    dm_res = diebold_mariano_test(d)

    print(f"\n  ── A. Diebold-Mariano (HLN correction) ──")
    print(f"     T = {dm_res['T']}  |  Newey-West lags h = {dm_res['h_lags']}")
    print(f"     Mean monthly differential  d̄  = {dm_res['d_bar_monthly']:+.5f}  "
          f"({dm_res['d_bar_annual']:+.4f} ann.)")
    print(f"     DM statistic               = {dm_res['DM_stat']:+.4f}")
    print(f"     DM_HLN (t-corrected)       = {dm_res['DM_HLN']:+.4f}")
    print(f"     p-value (one-sided)        = {dm_res['p_value']:.4f}")
    sig = ("***" if dm_res["sig_1pct"]  else
           "**"  if dm_res["sig_5pct"]  else
           "*"   if dm_res["sig_10pct"] else "n.s.")
    print(f"     Significance               = {sig}  "
          f"({'REJECT H₀' if dm_res['sig_10pct'] else 'FAIL TO REJECT H₀'})")

    # ── Test B: Bootstrap Sharpe CI ───────────────────────────────────────────
    print(f"\n[4/4] Bootstrap Sharpe CI (B={N_BOOTSTRAP:,}) …", flush=True)
    boot4 = bootstrap_sharpe(r4, rf4)
    boot2 = bootstrap_sharpe(r2, rf2, seed=RANDOM_SEED + 1)

    overlap = boot4["CI_95_lo"] < boot2["CI_95_hi"] and boot2["CI_95_lo"] < boot4["CI_95_hi"]

    print(f"\n  ── B. Bootstrap Sharpe CI (95%, percentile method) ──")
    print(f"     k=4: Sharpe = {boot4['Sharpe_point']:+.4f}  "
          f"SE = {boot4['Boot_SE']:.4f}  "
          f"CI = [{boot4['CI_95_lo']:+.4f}, {boot4['CI_95_hi']:+.4f}]")
    print(f"     k=2: Sharpe = {boot2['Sharpe_point']:+.4f}  "
          f"SE = {boot2['Boot_SE']:.4f}  "
          f"CI = [{boot2['CI_95_lo']:+.4f}, {boot2['CI_95_hi']:+.4f}]")
    print(f"     CI overlap: {'YES (not statistically distinct at 95%)' if overlap else 'NO (statistically distinct at 95%)'}")
    print(f"     Effect size ΔSharpe = {boot4['Sharpe_point'] - boot2['Sharpe_point']:+.4f}")

    # ── Combined output table ─────────────────────────────────────────────────
    results = pd.DataFrame([
        # DM test row
        {"Test": "DM_HLN",
         "Metric": "DM_HLN statistic",       "Value": round(dm_res["DM_HLN"], 4)},
        {"Test": "DM_HLN",
         "Metric": "p-value (one-sided)",     "Value": round(dm_res["p_value"], 4)},
        {"Test": "DM_HLN",
         "Metric": "Significant 10%",         "Value": int(dm_res["sig_10pct"])},
        {"Test": "DM_HLN",
         "Metric": "Significant 5%",          "Value": int(dm_res["sig_5pct"])},
        {"Test": "DM_HLN",
         "Metric": "d_bar_annual",            "Value": round(dm_res["d_bar_annual"], 4)},
        {"Test": "DM_HLN",
         "Metric": "T_common",               "Value": T},
        {"Test": "DM_HLN",
         "Metric": "NW_lags_h",              "Value": dm_res["h_lags"]},
        # Bootstrap k=4
        {"Test": "Bootstrap_k4",
         "Metric": "Sharpe_point",            "Value": round(boot4["Sharpe_point"], 4)},
        {"Test": "Bootstrap_k4",
         "Metric": "Boot_SE",                 "Value": round(boot4["Boot_SE"], 4)},
        {"Test": "Bootstrap_k4",
         "Metric": "CI_95_lo",               "Value": round(boot4["CI_95_lo"], 4)},
        {"Test": "Bootstrap_k4",
         "Metric": "CI_95_hi",               "Value": round(boot4["CI_95_hi"], 4)},
        # Bootstrap k=2
        {"Test": "Bootstrap_k2",
         "Metric": "Sharpe_point",            "Value": round(boot2["Sharpe_point"], 4)},
        {"Test": "Bootstrap_k2",
         "Metric": "Boot_SE",                 "Value": round(boot2["Boot_SE"], 4)},
        {"Test": "Bootstrap_k2",
         "Metric": "CI_95_lo",               "Value": round(boot2["CI_95_lo"], 4)},
        {"Test": "Bootstrap_k2",
         "Metric": "CI_95_hi",               "Value": round(boot2["CI_95_hi"], 4)},
        # Summary
        {"Test": "Summary",
         "Metric": "Delta_Sharpe_k4_minus_k2","Value": round(boot4["Sharpe_point"] - boot2["Sharpe_point"], 4)},
        {"Test": "Summary",
         "Metric": "CI_overlap_95pct",        "Value": int(overlap)},
    ])

    results.to_csv("output/dm_bootstrap_results.csv", index=False)
    print(f"\n  Saved: output/dm_bootstrap_results.csv")
    print(f"  Saved: output/dm_bootstrap_monthly.csv  ({T} rows)")
    print()

    print("=" * 65)
    print("  INTERPRETATION")
    print("=" * 65)
    if dm_res["sig_10pct"]:
        print(f"  DM test: REJECT H₀ at {10 if not dm_res['sig_5pct'] else 5}% → "
              f"k=4 monthly returns significantly higher than k=2")
    else:
        print(f"  DM test: FAIL TO REJECT H₀ → cannot reject equality at 10%")
        print(f"  (with T={T} OOS months, power is limited for modest differences)")

    if not overlap:
        print(f"  Bootstrap: CIs DO NOT overlap → Sharpe(k=4) statistically "
              f"distinct from Sharpe(k=2) at 95%")
    else:
        print(f"  Bootstrap: CIs overlap → cannot reject H₀: Sharpe(k=4)=Sharpe(k=2)")
        print(f"  Economic effect size ΔSharpe = "
              f"{boot4['Sharpe_point']-boot2['Sharpe_point']:+.4f} (substantial)")

    return results


if __name__ == "__main__":
    run()
