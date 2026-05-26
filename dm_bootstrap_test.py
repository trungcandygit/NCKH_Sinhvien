"""
=============================================================================
Diebold-Mariano Test & Bootstrap Sharpe CI  –  k=6 vs k=2 and k=6 vs k=4
=============================================================================
Two complementary statistical tests to justify k=6 (new baseline):

  A. Diebold-Mariano (1995) + Harvey-Leybourne-Newey (1997) small-sample
       Loss differential: d_t = r_k6_t − r_kX_t  (X = 2 and X = 4)
       H₀: E[d_t] ≤ 0  (k=6 not better)
       H₁: E[d_t] > 0  (k=6 better) — one-sided
       HAC variance via Newey-West (max_lag = ⌊T^(1/3)⌋)

  B. Bootstrap Sharpe CI (Efron & Tibshirani 1994, percentile method)
       B = 5,000 resamples; 95% CI for k=2, k=4, k=6
       Non-overlap ↔ statistically distinct Sharpe ratios

All tests run on IDENTICAL OOS dates (inner join across k windows).

Output
------
  output/dm_bootstrap_results.csv  – summary metrics
  output/dm_bootstrap_monthly.csv  – aligned monthly returns k2/k4/k6
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

def _dm_pair(label: str, r_new: np.ndarray, r_old: np.ndarray,
             rf_new: np.ndarray, rf_old: np.ndarray) -> dict:
    """Run DM test for one pair (new k vs old k) and return result dict."""
    d      = r_new - r_old
    res    = diebold_mariano_test(d)
    boot_n = bootstrap_sharpe(r_new, rf_new)
    boot_o = bootstrap_sharpe(r_old, rf_old, seed=RANDOM_SEED + 1)
    overlap = (boot_n["CI_95_lo"] < boot_o["CI_95_hi"] and
               boot_o["CI_95_lo"] < boot_n["CI_95_hi"])
    sig = ("***" if res["sig_1pct"] else "**" if res["sig_5pct"]
           else "*" if res["sig_10pct"] else "n.s.")
    print(f"\n  ── {label} ──")
    print(f"     T={res['T']}  h={res['h_lags']}  "
          f"d̄_ann={res['d_bar_annual']:+.4f}  "
          f"DM_HLN={res['DM_HLN']:+.4f}  p={res['p_value']:.4f}  {sig}")
    print(f"     Sharpe new={boot_n['Sharpe_point']:+.4f} "
          f"CI=[{boot_n['CI_95_lo']:+.4f},{boot_n['CI_95_hi']:+.4f}]  "
          f"old={boot_o['Sharpe_point']:+.4f} "
          f"CI=[{boot_o['CI_95_lo']:+.4f},{boot_o['CI_95_hi']:+.4f}]")
    print(f"     ΔSharpe={boot_n['Sharpe_point']-boot_o['Sharpe_point']:+.4f}  "
          f"CI overlap: {'YES' if overlap else 'NO'}")
    return {
        "pair": label,
        "T": res["T"], "h_lags": res["h_lags"],
        "d_bar_annual": res["d_bar_annual"],
        "DM_stat": res["DM_stat"],
        "DM_HLN": res["DM_HLN"],
        "p_value": res["p_value"],
        "sig": sig,
        "sig_10pct": int(res["sig_10pct"]),
        "sig_5pct":  int(res["sig_5pct"]),
        "sig_1pct":  int(res["sig_1pct"]),
        "Sharpe_new": boot_n["Sharpe_point"],
        "Boot_SE_new": boot_n["Boot_SE"],
        "CI_lo_new": boot_n["CI_95_lo"],
        "CI_hi_new": boot_n["CI_95_hi"],
        "Sharpe_old": boot_o["Sharpe_point"],
        "Boot_SE_old": boot_o["Boot_SE"],
        "CI_lo_old": boot_o["CI_95_lo"],
        "CI_hi_old": boot_o["CI_95_hi"],
        "Delta_Sharpe": round(boot_n["Sharpe_point"] - boot_o["Sharpe_point"], 4),
        "CI_overlap_95pct": int(overlap),
    }


def run() -> pd.DataFrame:
    print("=" * 70)
    print("  DM Test & Bootstrap Sharpe CI  –  k=6 vs k=4 vs k=2 for BL_KIO")
    print("=" * 70)

    # ── Run all three series ──────────────────────────────────────────────────
    print("\n[1/3] Running BL_KIO with k=6 (new baseline) …", flush=True)
    df6 = run_bl_kio_series(k=6)
    print("[2/3] Running BL_KIO with k=4 (old baseline) …", flush=True)
    df4 = run_bl_kio_series(k=4)
    print("[3/3] Running BL_KIO with k=2 …", flush=True)
    df2 = run_bl_kio_series(k=2)

    # ── Align on common dates (inner join of all three) ───────────────────────
    common = df6.index.intersection(df4.index).intersection(df2.index)
    r6  = df6.loc[common, "r_BL_KIO"].values
    r4  = df4.loc[common, "r_BL_KIO"].values
    r2  = df2.loc[common, "r_BL_KIO"].values
    rf6 = df6.loc[common, "rf_monthly"].values
    rf4 = df4.loc[common, "rf_monthly"].values
    rf2 = df2.loc[common, "rf_monthly"].values
    T   = len(common)
    print(f"\n  Common OOS dates: {T}  "
          f"({common[0].date()} → {common[-1].date()})")

    # Save monthly comparison
    monthly = pd.DataFrame({
        "Date":      [d.strftime("%Y-%m-%d") for d in common],
        "r_k6":      r6,
        "r_k4":      r4,
        "r_k2":      r2,
        "rf":        rf6,
        "d_k6_k4":  r6 - r4,
        "d_k6_k2":  r6 - r2,
        "d_k4_k2":  r4 - r2,
    })
    os.makedirs("output", exist_ok=True)
    monthly.to_csv("output/dm_bootstrap_monthly.csv", index=False)

    # ── Tests ─────────────────────────────────────────────────────────────────
    print("\n── Diebold-Mariano + Bootstrap (B={:,}) ──".format(N_BOOTSTRAP))
    pairs = []
    pairs.append(_dm_pair("k=6 vs k=4  (new vs old baseline)",
                          r6, r4, rf6, rf4))
    pairs.append(_dm_pair("k=6 vs k=2  (new baseline vs weakest)",
                          r6, r2, rf6, rf2))
    pairs.append(_dm_pair("k=4 vs k=2  (old baseline vs weakest)",
                          r4, r2, rf4, rf2))

    # Also bootstrap for k=6 standalone
    boot6 = bootstrap_sharpe(r6, rf6, seed=RANDOM_SEED + 2)
    boot4 = bootstrap_sharpe(r4, rf4, seed=RANDOM_SEED + 3)
    boot2 = bootstrap_sharpe(r2, rf2, seed=RANDOM_SEED + 4)
    print(f"\n  ── Standalone Bootstrap Sharpe ──")
    for lbl, b in [("k=6", boot6), ("k=4", boot4), ("k=2", boot2)]:
        print(f"     {lbl}: {b['Sharpe_point']:+.4f}  "
              f"SE={b['Boot_SE']:.4f}  "
              f"95%CI=[{b['CI_95_lo']:+.4f},{b['CI_95_hi']:+.4f}]")

    results = pd.DataFrame(pairs)
    results.to_csv("output/dm_bootstrap_results.csv", index=False)

    # Standalone summary
    standalone = pd.DataFrame([
        {"k": 6, **{m: round(v, 4) for m, v in boot6.items()}},
        {"k": 4, **{m: round(v, 4) for m, v in boot4.items()}},
        {"k": 2, **{m: round(v, 4) for m, v in boot2.items()}},
    ])
    standalone.to_csv("output/dm_bootstrap_sharpe_ci.csv", index=False)

    print(f"\n  Saved: output/dm_bootstrap_results.csv  ({len(results)} pairs)")
    print(f"  Saved: output/dm_bootstrap_monthly.csv  ({T} rows)")
    print(f"  Saved: output/dm_bootstrap_sharpe_ci.csv")

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for p in pairs:
        rej = "REJECT H₀" if p["sig_10pct"] else "fail to reject"
        print(f"  {p['pair']:<40} p={p['p_value']:.4f} {p['sig']:<5} "
              f"ΔSharpe={p['Delta_Sharpe']:+.4f}  → {rej}")

    return results


if __name__ == "__main__":
    run()
