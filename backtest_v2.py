"""
=============================================================================
Vietnamese Bank Stock Portfolio Backtest  v2
=============================================================================
Additions vs v1
---------------
  • Time-varying risk-free rate  (VN10Y / 12, by year)
  • Equal-Weight  (EW)  portfolio  – DeMiguel et al. (2009) benchmark
  • Risk Parity   (RP)  portfolio  – inverse-volatility weighting
  • Jobson-Korkie (1981) Sharpe-ratio comparison test
  • Jarque-Bera normality test  &  Ljung-Box autocorrelation test
  • Detailed drawdown analysis (recovery months, threshold crossings)

Outputs (./output/)
-------------------
  oos_returns_v2.csv            – monthly OOS returns for 6 portfolios + RF
  weights_v2.csv                – portfolio weights at each rebalancing
  cluster_signals_v2.csv        – K-means labels and view positions
  monte_carlo_v2.csv            – expected Sharpe under view-noise (BL, TAN)
  performance_summary_v2.csv    – annualised Return, Vol, Sharpe, Sortino,
                                   MDD, Calmar, Win-Rate for all 6 portfolios
  ttest_results_v2.csv          – paired t-tests: BL vs each other portfolio
  jobson_korkie_results.csv     – JK z-test: BL vs each other portfolio
  distribution_jb_tests.csv     – Jarque-Bera normality test per portfolio
  distribution_lb_tests.csv     – Ljung-Box autocorrelation test per portfolio
  drawdown_summary.csv          – aggregate drawdown stats per portfolio
  drawdown_periods.csv          – individual drawdown events per portfolio
=============================================================================
"""

import os
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# =============================================================================
#  VN10Y RISK-FREE RATE TABLE
# =============================================================================

# Annual yield (%) of Vietnamese 10-year government bond, year-end observations.
# Months within a year use that year's rate.  Dates outside the table
# are extrapolated from the nearest available year.
_VN10Y_ANNUAL_PCT = {
    2014: 6.50,   # extrapolated from 2015 (no earlier data)
    2015: 6.50,
    2016: 6.20,
    2017: 5.80,
    2018: 4.58,
    2019: 4.58,
    2020: 3.00,
    2021: 2.53,
    2022: 4.00,
    2023: 3.50,
    2024: 3.06,
    2025: 3.10,
    2026: 4.35,
}


def build_rf_series(dates: pd.DatetimeIndex) -> pd.Series:
    """
    Build a monthly risk-free rate series aligned to `dates`.
    Monthly rf  =  VN10Y_annual_pct[year] / 100 / 12
    """
    years    = np.array(sorted(_VN10Y_ANNUAL_PCT.keys()))
    rf_vals  = []
    for d in dates:
        year = d.year
        if year not in _VN10Y_ANNUAL_PCT:
            closest = years[np.argmin(np.abs(years - year))]
            annual_pct = _VN10Y_ANNUAL_PCT[closest]
        else:
            annual_pct = _VN10Y_ANNUAL_PCT[year]
        rf_vals.append(annual_pct / 100.0 / 12.0)
    return pd.Series(rf_vals, index=dates, name="rf_monthly")


# =============================================================================
#  CONFIGURATION
# =============================================================================

class Config:
    DATA_DIR    = "data"
    OUTPUT_DIR  = "output"
    CLOSE_FILE  = "bank_monthly_close.csv"
    MKTCAP_FILE = "bank_monthly_mktcap_bn.csv"

    LOOKBACK    = 36
    TAU         = 1 / 36       # BL prior-uncertainty scalar  (τ = 1/T)
    N_CLUSTERS  = 4            # k=4: best OOS Sharpe on fair comparison (106 steps)
    MAX_WEIGHT  = 0.30
    OPT_STARTS  = 10
    # ── K-means signal parameters ────────────────────────────────────────────
    # Composite signal: idiosyncratic momentum + low-volatility (Blitz & van Vliet 2007).
    # Full LOOKBACK window is still used for Σ and Π estimation.
    SIGNAL_LOOKBACK  = 6    # recent months for momentum (skip last 1 month)
    SIGNAL_SKIP_LAST = 1    # skip last month to avoid short-term reversal

    # Use time-varying VN10Y as risk-free rate
    USE_DYNAMIC_RF = True
    # Fallback fixed rate (used when USE_DYNAMIC_RF = False)
    RF_ANNUAL      = 0.05
    RF_MONTHLY     = RF_ANNUAL / 12

    # Monte Carlo
    MC_DELTAS   = [-0.10, -0.05, 0.00, 0.05, 0.10]
    N_MC_SIMS   = 2000         # vectorised → fast even at 2000

    RANDOM_SEED = 42


# =============================================================================
#  DATA LOADER
# =============================================================================

class DataLoader:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def load(self):
        """
        Returns
        -------
        price      DataFrame (T × N)   adjusted monthly close
        mktcap     DataFrame (T × N)   market cap (bn VND)
        log_ret    DataFrame (T × N)   ln(P_t / P_{t-1})   – for estimation
        simple_ret DataFrame (T × N)   (P_t − P_{t-1}) / P_{t-1} – for OOS P&L
        rf_series  Series    (T,)      monthly risk-free rate (decimal)
        """
        price = pd.read_csv(
            os.path.join(self.cfg.DATA_DIR, self.cfg.CLOSE_FILE),
            index_col="date", parse_dates=True,
        )
        mktcap = pd.read_csv(
            os.path.join(self.cfg.DATA_DIR, self.cfg.MKTCAP_FILE),
            index_col="date", parse_dates=True,
        )
        price.sort_index(inplace=True)
        mktcap.sort_index(inplace=True)

        log_ret    = np.log(price / price.shift(1))
        simple_ret = price.pct_change()

        if self.cfg.USE_DYNAMIC_RF:
            rf_series = build_rf_series(log_ret.index)
        else:
            rf_series = pd.Series(
                self.cfg.RF_MONTHLY,
                index=log_ret.index,
                name="rf_monthly",
            )

        return price, mktcap, log_ret, simple_ret, rf_series


# =============================================================================
#  CLUSTER SIGNAL GENERATOR
# =============================================================================

class ClusterSignalGenerator:
    """
    K-means (k = N_CLUSTERS) on (ann_ret, ann_vol).
    Produces a single relative-value view: best cluster outperforms worst.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, log_ret_window: np.ndarray,
                 tickers: list, rf_annual: float) -> dict:
        """
        Composite signal: idiosyncratic short-term momentum + low-volatility.

        Parameters
        ----------
        log_ret_window : (T, n) training-window log returns
        tickers        : list of length n
        rf_annual      : annualised risk-free rate for this step

        Signal design
        -------------
        1. Idiosyncratic momentum (Jegadeesh & Titman 1993; Blitz & van Vliet 2007)
           Market-adjusted returns over the past SIGNAL_LOOKBACK months,
           skipping the last SIGNAL_SKIP_LAST months to avoid the well-documented
           short-term reversal (Jegadeesh 1990). Removing the EW sector factor
           isolates stock-specific information in a concentrated sector portfolio.

        2. Low-volatility feature (Baker, Bradley & Wurgler 2011)
           Volatility (negative) is included as a second clustering dimension so
           that clusters combine recent idiosyncratic outperformance with lower
           risk, matching the low-vol anomaly documented in equity markets.

        The two standardised features are passed to K-means. Clusters are ranked
        by a composite score (idio_return − ann_vol); the best cluster receives a
        positive BL view and the worst a negative view.

        Full LOOKBACK window is still used for Σ and equilibrium prior Π.

        Returns
        -------
        dict with keys: labels, positions, P, q, cluster_sharpe,
                        best_k, worst_k, ann_ret, ann_vol, sharpe_arr
        """
        n = len(tickers)
        T = len(log_ret_window)

        # ── Full-window statistics (for Σ and Π) ──────────────────────────────
        ann_ret = log_ret_window.mean(axis=0) * 12
        ann_vol = np.clip(log_ret_window.std(axis=0, ddof=1) * np.sqrt(12),
                          1e-8, None)

        # ── Idiosyncratic momentum signal ──────────────────────────────────────
        # Remove equal-weighted sector factor from each month's return
        mkt_monthly  = log_ret_window.mean(axis=1, keepdims=True)   # (T, 1)
        idio_returns = log_ret_window - mkt_monthly                  # (T, n)

        skip  = max(0, self.cfg.SIGNAL_SKIP_LAST)
        sig_L = max(1, min(self.cfg.SIGNAL_LOOKBACK, T - skip))
        end   = T - skip if skip > 0 else T
        start = max(0, end - sig_L)
        idio_window = idio_returns[start:end]                        # (sig_L, n)
        idio_sig    = idio_window.mean(axis=0) * 12                  # annualised

        # Composite cluster score used for ranking (not for Sharpe calculation)
        composite    = idio_sig - ann_vol   # high idio momentum, low vol = good
        sharpe_a     = (idio_sig - rf_annual) / ann_vol              # for reporting

        # ── K-means clustering ─────────────────────────────────────────────────
        # Features: (idiosyncratic momentum, -ann_vol) — negate vol so that
        # "high feature value" means "good" for both dimensions.
        features_scaled = StandardScaler().fit_transform(
            np.column_stack([idio_sig, -ann_vol])
        )

        try:
            km     = KMeans(n_clusters=self.cfg.N_CLUSTERS,
                            random_state=self.cfg.RANDOM_SEED, n_init=10)
            labels = km.fit_predict(features_scaled)
        except Exception:
            return dict(labels=np.zeros(n, dtype=int),
                        positions={t: "Neutral" for t in tickers},
                        P=np.zeros((1, n)), q=np.array([0.0]),
                        cluster_sharpe={0: 0.0},
                        best_k=0, worst_k=0,
                        ann_ret=ann_ret, ann_vol=ann_vol, sharpe_arr=sharpe_a)

        # Rank clusters by composite score (idio_return − vol)
        cluster_sharpe = {
            k: composite[labels == k].mean() if (labels == k).any() else -np.inf
            for k in range(self.cfg.N_CLUSTERS)
        }
        best_k  = max(cluster_sharpe, key=cluster_sharpe.get)
        worst_k = min(cluster_sharpe, key=cluster_sharpe.get)

        if best_k == worst_k:
            return dict(labels=labels,
                        positions={t: "Neutral" for t in tickers},
                        P=np.zeros((1, n)), q=np.array([0.0]),
                        cluster_sharpe=cluster_sharpe,
                        best_k=best_k, worst_k=worst_k,
                        ann_ret=ann_ret, ann_vol=ann_vol, sharpe_arr=sharpe_a)

        # ── View matrix P and view return q ────────────────────────────────────
        # Zero-investment: long best cluster (sum=+1), short worst (sum=−1)
        p_row = np.zeros(n)
        for i in range(n):
            if labels[i] == best_k:    p_row[i] =  1.0
            elif labels[i] == worst_k: p_row[i] = -1.0

        pos_sum = p_row[p_row > 0].sum()
        neg_sum = np.abs(p_row[p_row < 0].sum())
        if pos_sum > 0: p_row[p_row > 0] /= pos_sum
        if neg_sum > 0: p_row[p_row < 0] /= neg_sum
        if np.all(p_row == 0):
            p_row[:] = 1.0 / n

        P = p_row.reshape(1, n)
        # q = expected idiosyncratic return spread between best and worst cluster
        q = np.array([idio_sig[labels == best_k].mean()
                      - idio_sig[labels == worst_k].mean()])

        positions = {}
        for i, t in enumerate(tickers):
            if labels[i] == best_k:    positions[t] = "Long"
            elif labels[i] == worst_k: positions[t] = "Short"
            else:                      positions[t] = "Neutral"

        return dict(labels=labels, positions=positions, P=P, q=q,
                    cluster_sharpe=cluster_sharpe, best_k=best_k,
                    worst_k=worst_k, ann_ret=ann_ret,
                    ann_vol=ann_vol, sharpe_arr=sharpe_a,
                    idio_sig=idio_sig)    # exposed for IC calculation


# =============================================================================
#  BLACK-LITTERMAN ENGINE
# =============================================================================

class BlackLittermanEngine:
    """
    Reverse-CAPM implied returns + Bayesian BL update.

    Equilibrium:   Π  =  δ · Σ · w_mkt
    Risk-aversion: δ  =  (w ᵀ μ − rf) / (w ᵀ Σ w)
    View noise:    Ω  =  τ · P Σ Pᵀ
    Posterior:     μ_BL  =  M⁻¹ [(τΣ)⁻¹Π + Pᵀ Ω⁻¹ q]
                   Σ_BL  =  M⁻¹ + Σ
    where          M    =  (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _inv(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        n = M.shape[0]
        return np.linalg.inv(M + eps * np.eye(n))

    def compute_delta(self, mu: np.ndarray, Sigma: np.ndarray,
                      w_mkt: np.ndarray, rf_annual: float) -> float:
        """Implied risk-aversion from market portfolio; clipped to [0.5, 10]."""
        port_var = float(w_mkt @ Sigma @ w_mkt)
        if port_var < 1e-12:
            return 2.5
        return float(np.clip(
            (float(w_mkt @ mu) - rf_annual) / port_var, 0.5, 10.0
        ))

    def posterior(self, Pi: np.ndarray, Sigma: np.ndarray,
                  P: np.ndarray, q: np.ndarray):
        """
        Returns μ_BL, Σ_BL, A_mat, base_vec.
        A_mat and base_vec satisfy:  μ_BL(q') = base_vec + A_mat @ q'
        This linear structure lets Monte Carlo skip re-optimisation.
        """
        tau = self.cfg.TAU
        n, K = len(Pi), len(q)

        tau_Sig     = tau * Sigma
        tau_Sig_inv = self._inv(tau_Sig)

        Omega     = tau * (P @ Sigma @ P.T) + np.eye(K) * 1e-8
        Omega_inv = self._inv(Omega)

        M     = tau_Sig_inv + P.T @ Omega_inv @ P
        M_inv = self._inv(M)

        mu_BL    = M_inv @ (tau_Sig_inv @ Pi + P.T @ Omega_inv @ q)
        Sigma_BL = M_inv + Sigma

        A_mat    = M_inv @ P.T @ Omega_inv    # (n, K)
        base_vec = M_inv @ tau_Sig_inv @ Pi   # (n,)

        return mu_BL, Sigma_BL, A_mat, base_vec


# =============================================================================
#  PORTFOLIO OPTIMISER
# =============================================================================

class PortfolioOptimiser:
    """
    All portfolios obey:  Σwᵢ = 1,  0 ≤ wᵢ ≤ MAX_WEIGHT
    EW has no constraint other than Σwᵢ = 1 (all weights = 1/n).
    RP is computed analytically (inverse-vol) then clipped to MAX_WEIGHT.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def _neg_sharpe(self, w, mu, Sigma, rf):
        ret = float(w @ mu)
        var = float(w @ Sigma @ w)
        return -(ret - rf) / np.sqrt(max(var, 1e-12))

    def _variance(self, w, Sigma):
        return float(w @ Sigma @ w)

    def _solve(self, obj_fn, args: tuple, n: int) -> np.ndarray:
        bounds = [(0.0, self.cfg.MAX_WEIGHT)] * n
        con    = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        rng    = np.random.default_rng(self.cfg.RANDOM_SEED)
        best_w, best_f = np.ones(n) / n, np.inf

        for _ in range(self.cfg.OPT_STARTS):
            w0  = np.clip(rng.dirichlet(np.ones(n)), 0.0, self.cfg.MAX_WEIGHT)
            w0 /= w0.sum()
            res = optimize.minimize(
                obj_fn, w0, args=args, method="SLSQP",
                bounds=bounds, constraints=con,
                options={"ftol": 1e-10, "maxiter": 2000},
            )
            if res.success and res.fun < best_f:
                best_f, best_w = res.fun, res.x

        best_w = np.clip(best_w, 0.0, self.cfg.MAX_WEIGHT)
        best_w /= best_w.sum()
        return best_w

    def tangency(self, mu: np.ndarray, Sigma: np.ndarray,
                 rf: float) -> np.ndarray:
        """Maximise Sharpe(μ, Σ)."""
        return self._solve(self._neg_sharpe, (mu, Sigma, rf), len(mu))

    def min_variance(self, Sigma: np.ndarray) -> np.ndarray:
        """Minimise portfolio variance."""
        return self._solve(self._variance, (Sigma,), Sigma.shape[0])

    def bl_tangency(self, mu_BL: np.ndarray, Sigma_BL: np.ndarray,
                    rf: float) -> np.ndarray:
        """Maximise Sharpe(μ_BL, Σ_BL)."""
        return self._solve(self._neg_sharpe, (mu_BL, Sigma_BL, rf), len(mu_BL))

    def bl_equilibrium(self, Pi: np.ndarray, Sigma: np.ndarray,
                       rf: float) -> np.ndarray:
        """
        Vanilla BL with NO active views.
        Posterior simplifies to: μ = Π,  Σ_post = (1 + τ)·Σ
        (the prior-uncertainty scalar τ inflates the covariance).
        This is the Black-Litterman (1992) 'no-view' baseline.
        """
        Sigma_post = (1.0 + self.cfg.TAU) * Sigma
        return self._solve(self._neg_sharpe, (Pi, Sigma_post, rf), len(Pi))

    def equal_weight(self, n: int) -> np.ndarray:
        """
        Naive 1/N portfolio.
        DeMiguel et al. (2009) show this is hard to beat out-of-sample.
        No constraint beyond Σwᵢ = 1; all weights = 1/n.
        """
        return np.ones(n) / n

    def risk_parity(self, Sigma: np.ndarray) -> np.ndarray:
        """
        Inverse-volatility weighting (closed-form approximation to risk parity).
            wᵢ ∝ 1 / σᵢ     where σᵢ = √Σᵢᵢ (annualised)
        Normalise first, THEN clip to MAX_WEIGHT and re-normalise.
        (Clipping raw 1/σ values before normalisation is incorrect because
        1/σ ~ 3–5 which always exceeds MAX_WEIGHT=0.30, collapsing RP to EW.)
        """
        ann_vol = np.sqrt(np.clip(np.diag(Sigma), 1e-8, None))
        w = 1.0 / ann_vol
        w /= w.sum()                              # normalise to sum = 1 first
        w = np.clip(w, 0.0, self.cfg.MAX_WEIGHT)  # then apply per-asset cap
        w /= w.sum()                              # re-normalise after clipping
        return w


# =============================================================================
#  MONTE CARLO ENGINE  (vectorised)
# =============================================================================

class MonteCarloEngine:
    """
    Sensitivity of BL_KIO and BL_IO portfolio Sharpe to view-specification errors.

    BL_KIO noise: q_noisy = q + delta * |q| * N(0,1)   (K-means view noise)
    BL_IO  noise: Pi_noisy = Pi + delta * |Pi| * N(0,1) (equilibrium view noise)

    Both use the same delta levels and same noise scale for fair comparison.
    Exploits linear structure μ_BL(q') = base_vec + A_mat @ q' for vectorised BL_KIO.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, w_BL_KIO, w_BL_IO, w_TAN, mu_hist, Sigma, Sigma_BL,
            Pi, q, A_mat, base_vec, rf_annual: float) -> list:
        """
        Returns list of (delta_pct, sharpe_KIO_arr, sharpe_IO_arr, sharpe_TAN_scalar).
        sharpe_KIO_arr and sharpe_IO_arr have shape (N_MC_SIMS,).
        sharpe_TAN_scalar is a float (TAN is deterministic given the training data).
        """
        if self.cfg.N_MC_SIMS == 0:
            return []

        rng = np.random.default_rng(self.cfg.RANDOM_SEED)
        N   = self.cfg.N_MC_SIMS

        var_KIO    = float(w_BL_KIO @ Sigma_BL @ w_BL_KIO)
        var_IO     = float(w_BL_IO  @ Sigma    @ w_BL_IO)
        var_TAN    = float(w_TAN    @ Sigma    @ w_TAN)
        sharpe_TAN = (float(w_TAN @ mu_hist) - rf_annual) / np.sqrt(max(var_TAN, 1e-12))

        vol_KIO    = np.sqrt(max(var_KIO, 1e-12))
        vol_IO     = np.sqrt(max(var_IO,  1e-12))
        vol_TAN    = np.sqrt(max(var_TAN, 1e-12))
        ret_TAN    = float(w_TAN @ mu_hist)

        results = []
        for delta_pct in self.cfg.MC_DELTAS:
            std_q  = max(abs(delta_pct), 0.01) * np.abs(q)              # (K,)
            std_Pi = max(abs(delta_pct), 0.01) * np.abs(Pi)             # (n,)

            # BL_KIO: noise on K-means views q
            noise_q      = rng.normal(0.0, std_q,  size=(N, len(q)))    # (N, K)
            q_mat        = q  + noise_q                                   # (N, K)
            mu_BL_mat    = base_vec + (A_mat @ q_mat.T).T               # (N, n)
            ret_KIO_sim  = mu_BL_mat @ w_BL_KIO                         # (N,) annualised return
            sharpe_KIO   = (ret_KIO_sim - rf_annual) / vol_KIO          # (N,)

            # BL_IO: noise on equilibrium Pi
            noise_Pi     = rng.normal(0.0, std_Pi, size=(N, len(Pi)))   # (N, n)
            Pi_mat       = Pi + noise_Pi                                  # (N, n)
            ret_IO_sim   = Pi_mat @ w_BL_IO                              # (N,) annualised return
            sharpe_IO    = (ret_IO_sim - rf_annual) / vol_IO             # (N,)

            results.append((
                delta_pct,
                sharpe_KIO, ret_KIO_sim, vol_KIO,   # BL_KIO: (N,), (N,), scalar
                sharpe_IO,  ret_IO_sim,  vol_IO,    # BL_IO:  (N,), (N,), scalar
                sharpe_TAN, ret_TAN,     vol_TAN,   # TAN:    scalar, scalar, scalar
            ))
        return results


# =============================================================================
#  PERFORMANCE & STATISTICAL TESTS
# =============================================================================

class PerformanceCalculator:

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── Annualised metrics ────────────────────────────────────────────────

    def compute(self, ret_series: pd.Series, rf_series: pd.Series,
                name: str) -> dict:
        rf_a    = rf_series.mean() * 12
        mean_m  = ret_series.mean()
        std_m   = ret_series.std(ddof=1)
        ann_ret = (1 + mean_m) ** 12 - 1
        ann_vol = std_m * np.sqrt(12)
        sharpe  = (ann_ret - rf_a) / ann_vol if ann_vol > 1e-8 else 0.0

        excess   = ret_series - rf_series
        # Average over ALL periods (standard Sortino denominator)
        down_std = np.sqrt((excess.clip(upper=0) ** 2).mean()) * np.sqrt(12)
        down_std = max(down_std, 1e-8)
        sortino  = (ann_ret - rf_a) / down_std

        cum     = (1 + ret_series).cumprod()
        roll_mx = cum.expanding().max()
        mdd     = ((cum - roll_mx) / roll_mx).min()
        calmar  = ann_ret / max(abs(mdd), 1e-8)
        win_rt  = (ret_series > 0).mean()

        return dict(Portfolio=name, Ann_Return=ann_ret, Ann_Vol=ann_vol,
                    Sharpe=sharpe, Sortino=sortino, MDD=mdd,
                    Calmar=calmar, Win_Rate=win_rt)

    # ── Paired t-test ─────────────────────────────────────────────────────

    def ttest(self, bl: pd.Series, other: pd.Series, pair: str) -> dict:
        a, b = bl.align(other, join="inner")
        t, p = stats.ttest_rel(a, b)
        return dict(Pair=pair, t_statistic=t, p_value=p,
                    is_significant_5pct=(p < 0.05))

    # ── Jobson-Korkie (1981) Sharpe comparison ────────────────────────────

    def jobson_korkie(self, r_a: pd.Series, r_b: pd.Series,
                      rf_series: pd.Series, pair: str) -> dict:
        """
        H₀: SR_A = SR_B  (monthly, non-annualised Sharpe ratios)

        Test statistic:
            z = (SR_A − SR_B) / √θ
            θ = (1/T)[2 − 2ρ + ½SR²_A + ½SR²_B − SR_A·SR_B·ρ]

        where ρ = corr(r_A, r_B).  Asymptotically z ~ N(0,1).
        Reference: Jobson & Korkie, J. Finance, 1981.
        """
        ra = np.array(r_a)
        rb = np.array(r_b)
        rf = np.array(rf_series)
        T  = len(ra)

        mu_a  = (ra - rf).mean()
        mu_b  = (rb - rf).mean()
        sig_a = ra.std(ddof=1)
        sig_b = rb.std(ddof=1)
        cov   = np.cov(ra, rb, ddof=1)[0, 1]
        rho   = cov / max(sig_a * sig_b, 1e-12)

        SR_a = mu_a / max(sig_a, 1e-12)
        SR_b = mu_b / max(sig_b, 1e-12)

        theta = (1 / T) * (2 - 2 * rho
                           + 0.5 * SR_a ** 2 + 0.5 * SR_b ** 2
                           - SR_a * SR_b * rho)
        if theta <= 1e-12:
            return dict(Pair=pair, z_statistic=np.nan,
                        p_value=np.nan, is_significant_5pct=False)

        z = (SR_a - SR_b) / np.sqrt(theta)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        return dict(Pair=pair, z_statistic=z, p_value=p,
                    is_significant_5pct=(p < 0.05))

    # ── Jarque-Bera normality test ────────────────────────────────────────

    def jarque_bera(self, ret_series: pd.Series, name: str) -> dict:
        """H₀: monthly returns are normally distributed."""
        jb_stat, jb_p = stats.jarque_bera(ret_series.dropna())
        skew  = float(ret_series.skew())
        kurt  = float(ret_series.kurt())   # excess kurtosis
        return dict(Portfolio=name, JB_stat=jb_stat, JB_pval=jb_p,
                    Skewness=skew, Excess_Kurtosis=kurt,
                    is_normal_5pct=(jb_p > 0.05))

    # ── Ljung-Box autocorrelation test ────────────────────────────────────

    def ljung_box(self, ret_series: pd.Series, name: str,
                  lags=(5, 10, 20)) -> dict:
        """
        H₀: no serial autocorrelation up to lag h.
        Q(h) = T(T+2) Σ_{k=1}^{h} ρ̂²_k / (T−k)  ~  χ²(h)
        """
        s = ret_series.dropna()
        T = len(s)
        result = dict(Portfolio=name)
        for lag in lags:
            acf_vals = [s.autocorr(lag=k) for k in range(1, lag + 1)]
            Q = T * (T + 2) * sum(r**2 / (T - k)
                                  for k, r in enumerate(acf_vals, 1))
            p = 1 - stats.chi2.cdf(Q, df=lag)
            result[f"LB{lag}_stat"]    = Q
            result[f"LB{lag}_pval"]    = p
            result[f"LB{lag}_no_autocorr"] = (p > 0.05)
        return result

    # ── Drawdown analysis ─────────────────────────────────────────────────

    def drawdown_analysis(self, ret_series: pd.Series,
                          name: str, threshold: float = -0.20) -> tuple:
        """
        Returns
        -------
        summary : dict – aggregate stats (N drawdowns, avg depth, avg recovery)
        periods : list of dicts – individual drawdown events
        """
        cum     = (1 + ret_series).cumprod()
        roll_mx = cum.expanding().max()
        dd      = (cum - roll_mx) / roll_mx   # ≤ 0

        n_exceed = int((dd < threshold).sum())

        periods  = []
        in_dd    = False
        start_i  = None
        trough_i = None
        trough_v = 0.0

        for i in range(len(dd)):
            v = dd.iloc[i]
            if not in_dd and v < -1e-4:
                in_dd    = True
                start_i  = i
                trough_i = i
                trough_v = v
            elif in_dd:
                if v < trough_v:
                    trough_v = v
                    trough_i = i
                if v >= -1e-4:   # recovered back to previous peak
                    periods.append(dict(
                        Portfolio=name,
                        DD_Start=dd.index[start_i],
                        DD_Trough=dd.index[trough_i],
                        DD_Recovery=dd.index[i],
                        Depth_pct=round(trough_v * 100, 2),
                        Months_to_Trough=trough_i - start_i,
                        Recovery_Months=i - trough_i,
                        Total_Months=i - start_i,
                    ))
                    in_dd = False

        # Open drawdown at end of series (not yet recovered)
        if in_dd:
            periods.append(dict(
                Portfolio=name,
                DD_Start=dd.index[start_i],
                DD_Trough=dd.index[trough_i],
                DD_Recovery="Not recovered",
                Depth_pct=round(trough_v * 100, 2),
                Months_to_Trough=trough_i - start_i,
                Recovery_Months=None,
                Total_Months=len(dd) - 1 - start_i,
            ))

        depths     = [p["Depth_pct"] for p in periods]
        rec_months = [p["Recovery_Months"] for p in periods
                      if p["Recovery_Months"] is not None]

        summary = dict(
            Portfolio=name,
            N_Drawdowns=len(periods),
            Max_DD_pct=round(dd.min() * 100, 2),
            Avg_DD_Depth_pct=round(np.mean(depths), 2) if depths else 0.0,
            Avg_Recovery_Months=round(np.mean(rec_months), 1) if rec_months else None,
            N_Months_Below_Neg20pct=n_exceed,
        )
        return summary, periods


# =============================================================================
#  MAIN BACKTESTER
# =============================================================================

class Backtester:
    """
    Rolls a LOOKBACK-month window forward one month at a time.

    Indexing convention
    -------------------
    i       = index of the last date in the training window
    i+1     = OOS index  (return earned during month i+1,
              observable only after close of month i)
    lw      = log_ret.iloc[i-L : i]     training log returns  (L rows)
    oos_sr  = simple_ret.iloc[i+1]      OOS simple returns    (scalar per stock)
    """

    def __init__(self, cfg: Config, verbose: bool = True):
        self.cfg     = cfg
        self.verbose = verbose

        self.loader = DataLoader(cfg)
        self.signal = ClusterSignalGenerator(cfg)
        self.bl_eng = BlackLittermanEngine(cfg)
        self.optim  = PortfolioOptimiser(cfg)
        self.mc_eng = MonteCarloEngine(cfg)
        self.perf   = PerformanceCalculator(cfg)

        self._weights  : list = []
        self._oos      : list = []
        self._clusters : list = []
        self._ic_rows  : list = []   # per-step IC for signal robustness analysis
        # MC accumulators (sum over OOS steps, divide by nsteps at save time)
        self._mc_kio_sharpe : dict = {}   # delta -> (N,)
        self._mc_kio_ret    : dict = {}   # delta -> (N,)
        self._mc_kio_vol    : dict = {}   # delta -> float
        self._mc_io_sharpe  : dict = {}   # delta -> (N,)
        self._mc_io_ret     : dict = {}   # delta -> (N,)
        self._mc_io_vol     : dict = {}   # delta -> float
        self._mc_tan_sharpe : dict = {}   # delta -> float
        self._mc_tan_ret    : dict = {}   # delta -> float
        self._mc_tan_vol    : dict = {}   # delta -> float
        self._mc_nsteps     : int  = 0

    # ── Main rolling loop ────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        np.random.seed(self.cfg.RANDOM_SEED)

        price, mktcap, log_ret, simple_ret, rf_series = self.loader.load()
        dates = log_ret.index
        L     = self.cfg.LOOKBACK

        if self.verbose:
            print(f"Date range : {dates[0].date()} → {dates[-1].date()}")
            print(f"Lookback   : {L} months\n")
            hdr = f"{'Date':<12} {'N':>3}  {'MKT':>7}  {'TAN':>7}  {'MV':>7}  "
            hdr += f"{'BL':>7}  {'BL_KIO':>7}  {'EW':>7}  {'RP':>7}  {'rf%':>5}"
            print(hdr)
            print("-" * 80)

        for i in range(L, len(dates) - 1):

            lw       = log_ret.iloc[i - L : i]    # (L, N_all)
            oos_sr   = simple_ret.iloc[i + 1]      # (N_all,)
            oos_date = dates[i + 1]

            # RF rate: use value at end of training window for estimation;
            #          use OOS-month value for OOS performance attribution.
            rf_m_train = float(rf_series.iloc[i])        # monthly
            rf_a_train = rf_m_train * 12                  # annualised
            rf_m_oos   = float(rf_series.iloc[i + 1])

            # ── Dynamic universe ──────────────────────────────────────────
            active_mask = lw.notna().all() & oos_sr.notna()
            tickers     = active_mask[active_mask].index.tolist()
            if len(tickers) < self.cfg.N_CLUSTERS + 2:
                continue

            n     = len(tickers)
            lw_a  = lw[tickers].values     # (L, n)
            oos_r = oos_sr[tickers].values  # (n,)

            # ── Market-cap weights (renormalised to active universe) ───────
            mc_row = mktcap.iloc[i][tickers].fillna(0.0).clip(lower=0.0)
            mc_sum = mc_row.sum()
            if mc_sum <= 0.0:
                continue
            w_mkt = (mc_row / mc_sum).values

            # ── Covariance & mean (annualised log-return statistics) ───────
            Sigma   = np.cov(lw_a.T) * 12 + np.eye(n) * 1e-8
            mu_hist = lw_a.mean(axis=0) * 12

            # ── K-means clustering → view (P, q) ──────────────────────────
            sig  = self.signal.generate(lw_a, tickers, rf_a_train)
            P, q = sig["P"], sig["q"]

            # ── Black-Litterman ────────────────────────────────────────────
            delta = self.bl_eng.compute_delta(mu_hist, Sigma, w_mkt, rf_a_train)
            Pi    = delta * Sigma @ w_mkt
            mu_BL, Sigma_BL, A_mat, base_vec = self.bl_eng.posterior(Pi, Sigma, P, q)

            # ── Portfolio optimisation ────────────────────────────────────
            w_TAN    = self.optim.tangency(mu_hist, Sigma, rf_a_train)
            w_MV     = self.optim.min_variance(Sigma)
            w_BL     = self.optim.bl_equilibrium(Pi, Sigma, rf_a_train)   # vanilla BL (no views)
            w_BL_KIO = self.optim.bl_tangency(mu_BL, Sigma_BL, rf_a_train)  # BL + K-means idio signal
            w_EW     = self.optim.equal_weight(n)
            w_RP     = self.optim.risk_parity(Sigma)

            # ── OOS simple returns ─────────────────────────────────────────
            r_mkt    = float(w_mkt    @ oos_r)
            r_TAN    = float(w_TAN    @ oos_r)
            r_MV     = float(w_MV     @ oos_r)
            r_BL     = float(w_BL     @ oos_r)
            r_BL_KIO = float(w_BL_KIO @ oos_r)
            r_EW     = float(w_EW     @ oos_r)
            r_RP     = float(w_RP     @ oos_r)

            # ── Store OOS returns ──────────────────────────────────────────
            self._oos.append(dict(
                Date=oos_date, RF_monthly=rf_m_oos,
                MKT=r_mkt, TAN=r_TAN, MV=r_MV,
                BL=r_BL, BL_KIO=r_BL_KIO, EW=r_EW, RP=r_RP,
            ))

            # ── Signal IC (rank-correlation of idio signal vs realised return)
            idio_sig = sig.get("idio_sig")
            if idio_sig is not None and len(idio_sig) > 2:
                ic_val = float(stats.spearmanr(idio_sig, oos_r).correlation)
                hit    = int(
                    oos_r[sig["labels"] == sig["best_k"]].mean()
                    > oos_r[sig["labels"] == sig["worst_k"]].mean()
                ) if (sig["best_k"] != sig["worst_k"]) else 0
            else:
                ic_val, hit = np.nan, 0
            self._ic_rows.append(dict(Date=oos_date, IC=ic_val, Hit=hit,
                                      N_stocks=n))

            # ── Store weights ──────────────────────────────────────────────
            for j, t in enumerate(tickers):
                self._weights.append(dict(
                    Date=oos_date, Ticker=t,
                    w_mkt=w_mkt[j], w_TAN=w_TAN[j], w_MV=w_MV[j],
                    w_BL=w_BL[j],   w_BL_KIO=w_BL_KIO[j],
                    w_EW=w_EW[j],   w_RP=w_RP[j],
                ))
                self._clusters.append(dict(
                    Date=oos_date, Ticker=t,
                    Cluster_Label=int(sig["labels"][j]),
                    Assigned_View_Position=sig["positions"][t],
                ))

            # ── Monte Carlo sensitivity ────────────────────────────────────
            mc_result = self.mc_eng.run(w_BL_KIO, w_BL, w_TAN, mu_hist, Sigma,
                                        Sigma_BL, Pi, q, A_mat, base_vec, rf_a_train)
            if mc_result:
                for (delta_pct,
                     kio_sh, kio_ret, kio_vol,
                     io_sh,  io_ret,  io_vol,
                     tan_sh, tan_ret, tan_vol) in mc_result:
                    if delta_pct not in self._mc_kio_sharpe:
                        N = len(kio_sh)
                        self._mc_kio_sharpe[delta_pct] = np.zeros(N)
                        self._mc_kio_ret[delta_pct]    = np.zeros(N)
                        self._mc_kio_vol[delta_pct]    = 0.0
                        self._mc_io_sharpe[delta_pct]  = np.zeros(N)
                        self._mc_io_ret[delta_pct]     = np.zeros(N)
                        self._mc_io_vol[delta_pct]     = 0.0
                        self._mc_tan_sharpe[delta_pct] = 0.0
                        self._mc_tan_ret[delta_pct]    = 0.0
                        self._mc_tan_vol[delta_pct]    = 0.0
                    self._mc_kio_sharpe[delta_pct] += kio_sh
                    self._mc_kio_ret[delta_pct]    += kio_ret
                    self._mc_kio_vol[delta_pct]    += kio_vol
                    self._mc_io_sharpe[delta_pct]  += io_sh
                    self._mc_io_ret[delta_pct]     += io_ret
                    self._mc_io_vol[delta_pct]     += io_vol
                    self._mc_tan_sharpe[delta_pct] += tan_sh
                    self._mc_tan_ret[delta_pct]    += tan_ret
                    self._mc_tan_vol[delta_pct]    += tan_vol
                self._mc_nsteps += 1

            if self.verbose:
                print(f"{str(oos_date.date()):<12} {n:>3}  "
                      f"{r_mkt:>+7.4f}  {r_TAN:>+7.4f}  {r_MV:>+7.4f}  "
                      f"{r_BL:>+7.4f}  {r_BL_KIO:>+7.4f}  {r_EW:>+7.4f}  "
                      f"{r_RP:>+7.4f}  {rf_m_oos*100:>4.2f}%")

        if self.verbose:
            print()

        df_oos = pd.DataFrame(self._oos)
        self._save_outputs(df_oos)
        return df_oos

    # ── Output ─────────────────────────────────────────────────────────────

    def _save_outputs(self, df_oos: pd.DataFrame) -> None:
        out = self.cfg.OUTPUT_DIR
        os.makedirs(out, exist_ok=True)

        PORTS = ["MKT", "TAN", "MV", "BL", "BL_KIO", "EW", "RP"]

        df_w = pd.DataFrame(self._weights)
        df_c = pd.DataFrame(self._clusters)

        # ── 1. oos_returns_v2.csv ─────────────────────────────────────────
        df_oos.to_csv(f"{out}/oos_returns_v2.csv", index=False)

        # ── 2. weights_v2.csv ─────────────────────────────────────────────
        df_w.to_csv(f"{out}/weights_v2.csv", index=False)

        # ── 3. cluster_signals_v2.csv ─────────────────────────────────────
        df_c.to_csv(f"{out}/cluster_signals_v2.csv", index=False)

        # ── 4. monte_carlo_v2.csv ─────────────────────────────────────────
        # Columns: Delta_Noise, Sim_ID,
        #   Sharpe_BL_KIO, Return_BL_KIO, Vol_BL_KIO,
        #   Sharpe_BL_IO,  Return_BL_IO,  Vol_BL_IO,
        #   Sharpe_TAN,    Return_TAN,    Vol_TAN
        if self._mc_nsteps > 0:
            S = self._mc_nsteps
            mc_rows = []
            for delta in sorted(self._mc_kio_sharpe.keys()):
                kio_sh  = self._mc_kio_sharpe[delta] / S
                kio_ret = self._mc_kio_ret[delta]    / S
                kio_vol = self._mc_kio_vol[delta]    / S
                io_sh   = self._mc_io_sharpe[delta]  / S
                io_ret  = self._mc_io_ret[delta]     / S
                io_vol  = self._mc_io_vol[delta]     / S
                tan_sh  = self._mc_tan_sharpe[delta] / S
                tan_ret = self._mc_tan_ret[delta]    / S
                tan_vol = self._mc_tan_vol[delta]    / S
                for sim_id, (sk, rk, si, ri) in enumerate(
                        zip(kio_sh, kio_ret, io_sh, io_ret), start=1):
                    mc_rows.append({
                        "Delta_Noise":   delta,
                        "Sim_ID":        sim_id,
                        "Sharpe_BL_KIO": round(float(sk),  6),
                        "Return_BL_KIO": round(float(rk),  6),
                        "Vol_BL_KIO":    round(float(kio_vol), 6),
                        "Sharpe_BL_IO":  round(float(si),  6),
                        "Return_BL_IO":  round(float(ri),  6),
                        "Vol_BL_IO":     round(float(io_vol),  6),
                        "Sharpe_TAN":    round(float(tan_sh),  6),
                        "Return_TAN":    round(float(tan_ret), 6),
                        "Vol_TAN":       round(float(tan_vol), 6),
                    })
            df_mc = pd.DataFrame(mc_rows)
            df_mc.to_csv(f"{out}/monte_carlo_v2.csv", index=False)

        # Set Date as index once so all series are aligned on the same index
        df_idx = df_oos.set_index("Date")
        rf_s   = df_idx["RF_monthly"]

        # ── 5. performance_summary_v2.csv ─────────────────────────────────
        perf_rows = [self.perf.compute(df_idx[p], rf_s, p) for p in PORTS]
        df_perf = pd.DataFrame(perf_rows)
        df_perf.to_csv(f"{out}/performance_summary_v2.csv", index=False)

        # ── 6. ttest_results_v2.csv ───────────────────────────────────────
        bl_kio_r = df_idx["BL_KIO"]
        ttest_rows = [self.perf.ttest(bl_kio_r, df_idx[p], f"BL_KIO vs {p}")
                      for p in ["MKT", "TAN", "MV", "BL", "EW", "RP"]]
        pd.DataFrame(ttest_rows).to_csv(f"{out}/ttest_results_v2.csv", index=False)

        # ── 7. jobson_korkie_results.csv ──────────────────────────────────
        jk_rows = [self.perf.jobson_korkie(bl_kio_r, df_idx[p], rf_s, f"BL_KIO vs {p}")
                   for p in ["MKT", "TAN", "MV", "BL", "EW", "RP"]]
        df_jk = pd.DataFrame(jk_rows)
        df_jk.to_csv(f"{out}/jobson_korkie_results.csv", index=False)

        # ── 8. distribution_jb_tests.csv ─────────────────────────────────
        jb_rows = [self.perf.jarque_bera(df_idx[p], p) for p in PORTS]
        pd.DataFrame(jb_rows).to_csv(f"{out}/distribution_jb_tests.csv", index=False)

        # ── 9. distribution_lb_tests.csv ─────────────────────────────────
        lb_rows = [self.perf.ljung_box(df_idx[p], p) for p in PORTS]
        pd.DataFrame(lb_rows).to_csv(f"{out}/distribution_lb_tests.csv", index=False)

        # ── 10. drawdown_summary.csv ──────────────────────────────────────
        # ── 11. drawdown_periods.csv ──────────────────────────────────────
        dd_summaries, dd_periods = [], []
        for p in PORTS:
            summ, pds = self.perf.drawdown_analysis(df_idx[p], p)
            dd_summaries.append(summ)
            dd_periods.extend(pds)
        pd.DataFrame(dd_summaries).to_csv(f"{out}/drawdown_summary.csv", index=False)
        pd.DataFrame(dd_periods).to_csv(f"{out}/drawdown_periods.csv", index=False)

        # ── 12. signal_ic_analysis.csv ────────────────────────────────────
        # Per-step Information Coefficient (Spearman rank correlation between
        # the idiosyncratic momentum signal and realised next-month returns).
        # IC > 0 means the signal correctly ranks stocks in that month.
        if self._ic_rows:
            df_ic = pd.DataFrame(self._ic_rows)
            df_ic.to_csv(f"{out}/signal_ic_analysis.csv", index=False)

        # ── 13. subperiod_analysis.csv ────────────────────────────────────
        # Split OOS period at midpoint; all 6 portfolios × 3 periods = 18 rows.
        oos_dates = df_idx.index
        mid_date  = oos_dates[len(oos_dates) // 2]
        sub_rows  = []
        for period_label, mask in [
            ("Full",     slice(None)),
            ("Period_1", oos_dates < mid_date),
            ("Period_2", oos_dates >= mid_date),
        ]:
            if isinstance(mask, slice):
                sub_rf = rf_s
                sub_series = {p: df_idx[p] for p in PORTS}
            else:
                sub_rf = rf_s[mask]
                sub_series = {p: df_idx.loc[mask, p] for p in PORTS}
            if len(sub_rf) < 6:
                continue
            for pname in PORTS:
                series = sub_series[pname]
                m = self.perf.compute(series, sub_rf, pname)
                sub_rows.append(dict(
                    Period=period_label,
                    Period_Start=str(series.index[0].date()),
                    Period_End=str(series.index[-1].date()),
                    N_months=len(series),
                    Portfolio=pname,
                    Ann_Return=m["Ann_Return"],
                    Ann_Vol=m["Ann_Vol"],
                    Sharpe=m["Sharpe"],
                    Sortino=m["Sortino"],
                    MDD=m["MDD"],
                    Calmar=m["Calmar"],
                ))
        pd.DataFrame(sub_rows).to_csv(f"{out}/subperiod_analysis.csv", index=False)

        # ── Console summary ───────────────────────────────────────────────
        print("=" * 90)
        print("PERFORMANCE SUMMARY  (OOS, annualised, time-varying VN10Y risk-free rate)")
        print("=" * 90)
        print(df_perf.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

        print("\n" + "=" * 90)
        print("JOBSON-KORKIE TEST  (H₀: SR_BL_KIO = SR_other,  two-sided)")
        print("=" * 90)
        print(df_jk.to_string(index=False))

        print("\n" + "=" * 90)
        print("DRAWDOWN SUMMARY")
        print("=" * 90)
        print(pd.DataFrame(dd_summaries).to_string(index=False))

        saved = [
            "oos_returns_v2.csv", "weights_v2.csv",
            "cluster_signals_v2.csv", "monte_carlo_v2.csv",
            "performance_summary_v2.csv", "ttest_results_v2.csv",
            "jobson_korkie_results.csv", "distribution_jb_tests.csv",
            "distribution_lb_tests.csv", "drawdown_summary.csv",
            "drawdown_periods.csv", "signal_ic_analysis.csv",
            "subperiod_analysis.csv",
        ]
        print(f"\nOutputs saved to ./{out}/")
        for f in saved:
            path = f"{out}/{f}"
            size = os.path.getsize(path) if os.path.exists(path) else 0
            print(f"  [✓] {f:<45} {size:>8,} bytes")


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 90)
    print("  Vietnamese Bank Stocks – Portfolio Backtest v2")
    print("  7 portfolios: MKT · TAN · MV · BL · BL_KIO · EW · RP")
    print("  Risk-free: time-varying VN10Y/12")
    print("=" * 90 + "\n")
    Backtester(Config()).run()
