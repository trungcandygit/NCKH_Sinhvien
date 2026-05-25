"""
=============================================================================
Vietnamese Bank Stock Portfolio Backtest
Strategy: K-means Clustering Signal → Black-Litterman (Inverse Optimisation)
          → Monte Carlo Robustness → Out-of-Sample Performance
=============================================================================

Mathematical pipeline:
  Phase 1  – Rolling 36-month window, dynamic universe (no NaN constraint)
  Phase 2  – K-means(k=3) on (ann_ret, ann_vol); derive view matrix P, q
  Phase 3  – Reverse-CAPM → Π; Bayesian update → μ_BL, Σ_BL
  Phase 4  – Constrained mean-variance optimisation (TAN, MV, BL)
  Phase 5  – Vectorised Monte Carlo: perturb q, measure Sharpe sensitivity
  Phase 6  – OOS portfolio returns, annualised metrics, paired t-tests

No plots are produced.  All results are written to CSV under ./output/.
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
#  CONFIGURATION
# =============================================================================

class Config:
    """All hyper-parameters in one place."""

    DATA_DIR    = "data"
    OUTPUT_DIR  = "output"
    CLOSE_FILE  = "bank_monthly_close.csv"
    MKTCAP_FILE = "bank_monthly_mktcap_bn.csv"

    # Rolling-window
    LOOKBACK    = 36          # months of training data required

    # Risk-free rate: approx. Vietnamese 1-yr gov-bond yield (annualised)
    RF_ANNUAL   = 0.05
    RF_MONTHLY  = RF_ANNUAL / 12

    # Black-Litterman
    # τ scales uncertainty in the prior: smaller → stronger prior on Π
    TAU         = 1 / 36

    # K-means
    N_CLUSTERS  = 3

    # Optimisation constraints
    MAX_WEIGHT  = 0.30        # maximum allocation per stock (30 %)
    OPT_STARTS  = 10          # random initialisations for SLSQP robustness

    # Monte Carlo
    # δ is the noise scale as a fraction of |q|; 5 deterministic levels
    MC_DELTAS   = [-0.10, -0.05, 0.00, 0.05, 0.10]
    N_MC_SIMS   = 2000        # simulations per (step, δ) – vectorised, fast

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
        price      : DataFrame (T × N) adjusted close prices
        mktcap     : DataFrame (T × N) market capitalisation (bn VND)
        log_ret    : DataFrame (T × N) log returns  ln(P_t / P_{t-1})
        simple_ret : DataFrame (T × N) simple returns  (P_t - P_{t-1}) / P_{t-1}

        Log returns are used for covariance / mean estimation (better
        normal-distribution properties).  Simple returns are used for OOS
        portfolio P&L arithmetic, since
            R_portfolio = Σ w_i · r_i   holds exactly for simple returns.
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

        return price, mktcap, log_ret, simple_ret


# =============================================================================
#  CLUSTER SIGNAL GENERATOR  (Phase 2)
# =============================================================================

class ClusterSignalGenerator:
    """
    Applies K-means (k=3) clustering to stocks based on their annualised
    return and annualised volatility over the training window, then
    constructs a single relative-value view:
        "Best-cluster stocks outperform worst-cluster stocks by q."
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def generate(self, log_ret_window: np.ndarray, tickers: list) -> dict:
        """
        Parameters
        ----------
        log_ret_window : (T, n) array of log returns for the active universe
        tickers        : list of n ticker strings

        Returns
        -------
        dict with keys: labels, positions, P, q, cluster_sharpe,
                        best_k, worst_k, ann_ret, ann_vol, sharpe_arr
        """
        n = len(tickers)
        cfg = self.cfg

        # ── Annualised features ───────────────────────────────────────────
        ann_ret  = log_ret_window.mean(axis=0) * 12              # (n,)
        ann_vol  = log_ret_window.std(axis=0, ddof=1) * np.sqrt(12)
        ann_vol  = np.clip(ann_vol, 1e-8, None)
        sharpe_a = (ann_ret - cfg.RF_ANNUAL) / ann_vol            # (n,)

        # ── K-means on (ann_ret, ann_vol) – standardised ─────────────────
        features        = np.column_stack([ann_ret, ann_vol])     # (n, 2)
        features_scaled = StandardScaler().fit_transform(features)

        km     = KMeans(n_clusters=cfg.N_CLUSTERS,
                        random_state=cfg.RANDOM_SEED, n_init=10)
        labels = km.fit_predict(features_scaled)                  # (n,) int

        # ── Cluster-level Sharpe ratios ───────────────────────────────────
        cluster_sharpe = {}
        for k in range(cfg.N_CLUSTERS):
            mask = labels == k
            cluster_sharpe[k] = sharpe_a[mask].mean() if mask.any() else -np.inf

        best_k  = max(cluster_sharpe, key=cluster_sharpe.get)
        worst_k = min(cluster_sharpe, key=cluster_sharpe.get)

        # Degenerate case: all stocks in one cluster → neutral view
        if best_k == worst_k:
            P = np.zeros((1, n))
            q = np.array([0.0])
            positions = {t: "Neutral" for t in tickers}
            return dict(labels=labels, positions=positions, P=P, q=q,
                        cluster_sharpe=cluster_sharpe, best_k=best_k,
                        worst_k=worst_k, ann_ret=ann_ret,
                        ann_vol=ann_vol, sharpe_arr=sharpe_a)

        # ── View matrix P  (1 × n) and view scalar q ─────────────────────
        # Long best-cluster / Short worst-cluster, zero-investment:
        #   positive entries normalised to sum +1
        #   negative entries normalised to sum -1
        p_row = np.zeros(n)
        for i in range(n):
            if labels[i] == best_k:
                p_row[i] =  1.0
            elif labels[i] == worst_k:
                p_row[i] = -1.0

        pos_sum = p_row[p_row > 0].sum()
        neg_sum = np.abs(p_row[p_row < 0].sum())
        if pos_sum > 0:
            p_row[p_row > 0] /= pos_sum
        if neg_sum > 0:
            p_row[p_row < 0] /= neg_sum

        P = p_row.reshape(1, n)   # (1, n)

        # q = spread in annualised mean returns between best and worst cluster
        q_val = (ann_ret[labels == best_k].mean()
                 - ann_ret[labels == worst_k].mean())
        q = np.array([q_val])     # (1,)

        # If P is still all-zero (edge case), fall back to market-weight view
        if np.all(p_row == 0):
            p_row[:] = 1.0 / n
            P = p_row.reshape(1, n)
            q = np.array([ann_ret.mean()])

        # ── View positions for CSV output ─────────────────────────────────
        positions = {}
        for i, t in enumerate(tickers):
            if labels[i] == best_k:
                positions[t] = "Long"
            elif labels[i] == worst_k:
                positions[t] = "Short"
            else:
                positions[t] = "Neutral"

        return dict(
            labels=labels, positions=positions,
            P=P, q=q,
            cluster_sharpe=cluster_sharpe, best_k=best_k, worst_k=worst_k,
            ann_ret=ann_ret, ann_vol=ann_vol, sharpe_arr=sharpe_a,
        )


# =============================================================================
#  BLACK-LITTERMAN ENGINE  (Phase 3)
# =============================================================================

class BlackLittermanEngine:
    """
    Implements the Bayesian Black-Litterman update.

    Equilibrium (Reverse-CAPM):
        Π  =  δ · Σ · w_mkt

    where the risk-aversion coefficient is derived from the observed market
    portfolio (Inverse Optimisation / Implied Returns):
        δ  =  (w_mkt ᵀ μ − rf) / (w_mkt ᵀ Σ w_mkt)

    Bayesian update (He & Litterman, 1999):
        Ω      = τ · P Σ Pᵀ           (proportional view uncertainty)
        M      = (τΣ)⁻¹ + Pᵀ Ω⁻¹ P
        μ_BL   = M⁻¹ [(τΣ)⁻¹ Π  +  Pᵀ Ω⁻¹ q]
        Σ_BL   = M⁻¹ + Σ              (estimation + sampling uncertainty)

    The linear structure  μ_BL = base + A·q  is exploited for O(1) Monte Carlo.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def _inv(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Tikhonov-regularised matrix inverse for numerical stability."""
        n = M.shape[0]
        return np.linalg.inv(M + eps * np.eye(n))

    def compute_delta(self, mu: np.ndarray,
                      Sigma: np.ndarray, w_mkt: np.ndarray) -> float:
        """
        Implied risk aversion from market portfolio.
        Clipped to [0.5, 10] to avoid unreasonable values near zero variance.
        """
        port_ret = float(w_mkt @ mu)
        port_var = float(w_mkt @ Sigma @ w_mkt)
        if port_var < 1e-12:
            return 2.5
        delta = (port_ret - self.cfg.RF_ANNUAL) / port_var
        return float(np.clip(delta, 0.5, 10.0))

    def posterior(self, Pi: np.ndarray, Sigma: np.ndarray,
                  P: np.ndarray, q: np.ndarray):
        """
        Compute BL posterior.

        Returns
        -------
        mu_BL    : (n,) posterior expected returns
        Sigma_BL : (n, n) posterior covariance
        A_mat    : (n, K)  d(μ_BL)/d(q)  – precomputed for vectorised MC
        base_vec : (n,)    μ_BL when q ≡ 0  – constant part
        """
        tau = self.cfg.TAU
        n   = len(Pi)
        K   = len(q)

        tau_Sig     = tau * Sigma
        tau_Sig_inv = self._inv(tau_Sig)

        # View uncertainty: Ω is (K × K); for our single view K=1 it's a scalar
        Omega     = tau * (P @ Sigma @ P.T)         # (K, K)
        Omega    += np.eye(K) * 1e-8
        Omega_inv = self._inv(Omega)

        # Posterior precision matrix
        M     = tau_Sig_inv + P.T @ Omega_inv @ P   # (n, n)
        M_inv = self._inv(M)

        # Posterior mean:  μ_BL = M⁻¹ [(τΣ)⁻¹ Π  +  Pᵀ Ω⁻¹ q]
        mu_BL    = M_inv @ (tau_Sig_inv @ Pi + P.T @ Omega_inv @ q)

        # Posterior covariance:  Σ_BL = M⁻¹ + Σ
        Sigma_BL = M_inv + Sigma

        # ── Pre-compute linear coefficients for Monte Carlo ───────────────
        # μ_BL(q') = base_vec + A_mat @ q'    (linear in q')
        A_mat    = M_inv @ P.T @ Omega_inv      # (n, K)
        base_vec = M_inv @ tau_Sig_inv @ Pi     # (n,)

        return mu_BL, Sigma_BL, A_mat, base_vec


# =============================================================================
#  PORTFOLIO OPTIMISER  (Phase 4)
# =============================================================================

class PortfolioOptimiser:
    """
    Constrained mean-variance optimisation via SLSQP.

    Constraints (applied to all three portfolios):
        Σ w_i = 1   (fully invested)
        0 ≤ w_i ≤ 0.30   (long-only, maximum 30 % per asset)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # ── Objective functions ───────────────────────────────────────────────

    @staticmethod
    def _neg_sharpe(w, mu, Sigma, rf):
        """Negative Sharpe ratio (minimised → maximised Sharpe)."""
        ret = float(w @ mu)
        var = float(w @ Sigma @ w)
        return -(ret - rf) / np.sqrt(max(var, 1e-12))

    @staticmethod
    def _variance(w, Sigma):
        """Portfolio variance (minimised for MV portfolio)."""
        return float(w @ Sigma @ w)

    # ── Generic solver ────────────────────────────────────────────────────

    def _solve(self, obj_fn, args: tuple, n: int) -> np.ndarray:
        """
        Multi-start SLSQP with Dirichlet initialisation for robustness.
        Returns weights clipped to [0, MAX_WEIGHT] and renormalised.
        """
        bounds = [(0.0, self.cfg.MAX_WEIGHT)] * n
        con    = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
        rng    = np.random.default_rng(self.cfg.RANDOM_SEED)

        best_w, best_f = np.ones(n) / n, np.inf

        for _ in range(self.cfg.OPT_STARTS):
            w0  = rng.dirichlet(np.ones(n))
            w0  = np.clip(w0, 0.0, self.cfg.MAX_WEIGHT)
            w0 /= w0.sum()

            res = optimize.minimize(
                obj_fn, w0, args=args,
                method="SLSQP",
                bounds=bounds,
                constraints=con,
                options={"ftol": 1e-10, "maxiter": 2000},
            )
            if res.success and res.fun < best_f:
                best_f, best_w = res.fun, res.x

        best_w  = np.clip(best_w, 0.0, self.cfg.MAX_WEIGHT)
        best_w /= best_w.sum()
        return best_w

    # ── Named portfolios ──────────────────────────────────────────────────

    def tangency(self, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
        """Markowitz tangency: maximise Sharpe(μ, Σ)."""
        return self._solve(self._neg_sharpe, (mu, Sigma, self.cfg.RF_ANNUAL),
                           len(mu))

    def min_variance(self, Sigma: np.ndarray) -> np.ndarray:
        """Minimum-variance portfolio."""
        return self._solve(self._variance, (Sigma,), Sigma.shape[0])

    def bl_tangency(self, mu_BL: np.ndarray, Sigma_BL: np.ndarray) -> np.ndarray:
        """Black-Litterman tangency: maximise Sharpe(μ_BL, Σ_BL)."""
        return self._solve(self._neg_sharpe,
                           (mu_BL, Sigma_BL, self.cfg.RF_ANNUAL),
                           len(mu_BL))


# =============================================================================
#  MONTE CARLO ROBUSTNESS ENGINE  (Phase 5)
# =============================================================================

class MonteCarloEngine:
    """
    In-sample sensitivity of BL portfolio Sharpe to view-specification error.

    For each noise level δ ∈ MC_DELTAS:
        q_sim[j]  = q + ε_j    where ε_j ~ N(0, (max(|δ|, 0.01) · |q|)²)
        μ_BL_sim  = base_vec + A_mat @ q_sim[j]   (closed-form, no re-optimisation)
        Sharpe_BL = (w_BL ᵀ μ_BL_sim − rf) / √(w_BL ᵀ Σ_BL w_BL)

    The key insight is that μ_BL is **linear** in q, so the entire simulation
    batch is evaluated with a single matrix multiplication: O(N · n · K).

    TAN portfolio Sharpe is unaffected by view noise (reported as a constant).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, w_BL: np.ndarray, w_TAN: np.ndarray,
            mu_hist: np.ndarray, Sigma: np.ndarray, Sigma_BL: np.ndarray,
            q: np.ndarray, A_mat: np.ndarray, base_vec: np.ndarray) -> list:
        """
        Returns list of dicts {delta, port_type, expected_sharpe}.
        """
        cfg = self.cfg
        rng = np.random.default_rng(cfg.RANDOM_SEED)
        rf  = cfg.RF_ANNUAL
        N   = cfg.N_MC_SIMS

        # Pre-compute variance scalars (constant across simulations)
        var_BL  = float(w_BL  @ Sigma_BL @ w_BL)
        var_TAN = float(w_TAN @ Sigma    @ w_TAN)

        # TAN Sharpe: deterministic (no views), same for all δ levels
        ret_TAN    = float(w_TAN @ mu_hist)
        sharpe_TAN = (ret_TAN - rf) / np.sqrt(max(var_TAN, 1e-12))

        records = []

        for delta_pct in cfg.MC_DELTAS:
            # Noise std: proportional to |q|, with a 1 % floor for δ = 0
            noise_std = max(abs(delta_pct), 0.01) * np.abs(q)   # (K,)

            # ── Vectorised simulation ─────────────────────────────────────
            noise_mat  = rng.normal(0.0, noise_std, size=(N, len(q)))  # (N, K)
            q_mat      = q + noise_mat                                   # (N, K)

            # μ_BL for all N simulations at once:  (N, n)
            # μ_BL(q') = base_vec  +  A_mat · q'
            mu_BL_mat  = base_vec + (A_mat @ q_mat.T).T                 # (N, n)

            # Sharpe of fixed w_BL under each perturbed μ_BL:  (N,)
            ret_BL_sim    = mu_BL_mat @ w_BL
            sharpe_BL_sim = (ret_BL_sim - rf) / np.sqrt(max(var_BL, 1e-12))

            records.append(dict(
                delta=delta_pct,
                port_type="BL",
                expected_sharpe=float(sharpe_BL_sim.mean()),
            ))
            records.append(dict(
                delta=delta_pct,
                port_type="TAN",
                expected_sharpe=sharpe_TAN,
            ))

        return records


# =============================================================================
#  PERFORMANCE CALCULATOR  (Phase 6)
# =============================================================================

class PerformanceCalculator:
    """Annualised performance metrics and paired t-tests."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def compute(self, ret_series: pd.Series, name: str) -> dict:
        """
        Annualised Return  : (1 + mean_monthly)^12 − 1
        Annualised Vol     : std_monthly · √12
        Sharpe             : (Ann_Ret − rf_annual) / Ann_Vol
        Sortino            : (Ann_Ret − rf_annual) / (downside_ann_std)
                             where downside is computed on monthly excess returns
        MDD                : peak-to-trough drawdown of cumulative wealth
        """
        rf_m = self.cfg.RF_MONTHLY
        rf_a = self.cfg.RF_ANNUAL

        mean_m = ret_series.mean()
        std_m  = ret_series.std(ddof=1)

        ann_ret = (1 + mean_m) ** 12 - 1
        ann_vol = std_m * np.sqrt(12)

        sharpe  = (ann_ret - rf_a) / ann_vol if ann_vol > 1e-8 else 0.0

        # Sortino: annualised downside deviation of monthly excess returns
        excess   = ret_series - rf_m
        neg_exc  = excess[excess < 0]
        down_std = (np.sqrt((neg_exc ** 2).mean()) * np.sqrt(12)
                    if len(neg_exc) > 0 else 1e-8)
        sortino  = (ann_ret - rf_a) / down_std if down_std > 1e-8 else 0.0

        # Maximum drawdown: (cumulative_peak − current) / cumulative_peak
        cum     = (1 + ret_series).cumprod()
        roll_mx = cum.expanding().max()
        mdd     = ((cum - roll_mx) / roll_mx).min()

        return dict(Portfolio=name,
                    Ann_Return=ann_ret, Ann_Vol=ann_vol,
                    Sharpe=sharpe, Sortino=sortino, MDD=mdd)

    def ttest(self, bl: pd.Series, other: pd.Series, pair: str) -> dict:
        """Paired t-test (two-sided).  BL column vs other portfolio column."""
        bl_a, ot_a = bl.align(other, join="inner")
        t_stat, p_val = stats.ttest_rel(bl_a, ot_a)
        return dict(
            Pair=pair,
            t_statistic=t_stat,
            p_value=p_val,
            is_significant_5pct=(p_val < 0.05),
        )


# =============================================================================
#  MAIN BACKTESTER
# =============================================================================

class Backtester:
    """
    Orchestrates the full pipeline month by month.

    Loop index semantics
    --------------------
    i   = last index of the training window (inclusive)
    i+1 = OOS index: simple_ret.iloc[i+1] is the return earned in month i+1,
          formed at the close of month i using weights computed from
          the window  log_ret.iloc[i-L : i]  (L = 36 months).
    """

    def __init__(self, cfg: Config):
        self.cfg    = cfg
        self.loader = DataLoader(cfg)
        self.signal = ClusterSignalGenerator(cfg)
        self.bl_eng = BlackLittermanEngine(cfg)
        self.optim  = PortfolioOptimiser(cfg)
        self.mc_eng = MonteCarloEngine(cfg)
        self.perf   = PerformanceCalculator(cfg)

        # Accumulators
        self._weights  : list = []
        self._oos      : list = []
        self._clusters : list = []
        self._mc_rows  : list = []

    # ── Main loop ─────────────────────────────────────────────────────────

    def run(self) -> None:
        np.random.seed(self.cfg.RANDOM_SEED)

        price, mktcap, log_ret, simple_ret = self.loader.load()
        dates = log_ret.index
        L     = self.cfg.LOOKBACK

        print(f"Date range   : {dates[0].date()} → {dates[-1].date()}")
        print(f"Total months : {len(dates)}")
        print(f"Lookback     : {L} months")
        print(f"Max OOS steps: {len(dates) - L - 1}\n")
        print(f"{'Date':<12} {'N':>4}  {'δ':>7}  {'MKT':>8}  "
              f"{'TAN':>8}  {'MV':>8}  {'BL':>8}")
        print("-" * 62)

        for i in range(L, len(dates) - 1):

            # ── Training window log returns ────────────────────────────
            lw      = log_ret.iloc[i - L : i]   # (L, N_all)
            oos_sr  = simple_ret.iloc[i + 1]     # (N_all,)  OOS simple ret
            oos_date = dates[i + 1]

            # ── Dynamic universe: full L-month history + OOS data ──────
            active_mask = lw.notna().all() & oos_sr.notna()
            tickers     = active_mask[active_mask].index.tolist()

            # Need at least k+2 stocks to form 3 clusters with a view
            if len(tickers) < self.cfg.N_CLUSTERS + 2:
                continue

            n    = len(tickers)
            lw_a = lw[tickers].values        # (L, n) – no NaN guaranteed
            oos_r = oos_sr[tickers].values    # (n,)

            # ── Market-cap weights (renormalised to active universe) ────
            # Use market cap at the last date of the training window (index i)
            mc_row = mktcap.iloc[i][tickers].fillna(0.0).clip(lower=0.0)
            mc_sum = mc_row.sum()
            if mc_sum <= 0.0:
                continue
            w_mkt = (mc_row / mc_sum).values   # (n,) sums to 1.0

            # ── Covariance matrix (annualised) and mean ─────────────────
            # Annualise monthly log-return statistics  (×12 for mean, ×√12 std)
            Sigma   = np.cov(lw_a.T) * 12      # (n, n)
            Sigma  += np.eye(n) * 1e-8          # regularise for PD guarantee
            mu_hist = lw_a.mean(axis=0) * 12    # (n,) annualised sample mean

            # ── Phase 2: K-means clustering → view (P, q) ───────────────
            sig  = self.signal.generate(lw_a, tickers)
            P    = sig["P"]   # (1, n)
            q    = sig["q"]   # (1,)

            # ── Phase 3: Black-Litterman ─────────────────────────────────
            # Step 1 – Implied risk aversion from market portfolio
            delta = self.bl_eng.compute_delta(mu_hist, Sigma, w_mkt)

            # Step 2 – Equilibrium implied returns (Reverse-CAPM)
            #          Π = δ · Σ · w_mkt
            Pi = delta * Sigma @ w_mkt           # (n,)

            # Step 3 – Bayesian update
            mu_BL, Sigma_BL, A_mat, base_vec = self.bl_eng.posterior(
                Pi, Sigma, P, q
            )

            # ── Phase 4: Portfolio optimisation ─────────────────────────
            w_TAN = self.optim.tangency(mu_hist, Sigma)
            w_MV  = self.optim.min_variance(Sigma)
            w_BL  = self.optim.bl_tangency(mu_BL, Sigma_BL)

            # ── Phase 6: OOS simple returns ──────────────────────────────
            r_mkt = float(w_mkt @ oos_r)
            r_TAN = float(w_TAN @ oos_r)
            r_MV  = float(w_MV  @ oos_r)
            r_BL  = float(w_BL  @ oos_r)

            # ── Phase 5: Monte Carlo sensitivity ────────────────────────
            mc_res = self.mc_eng.run(
                w_BL, w_TAN, mu_hist, Sigma, Sigma_BL,
                q, A_mat, base_vec,
            )

            # ── Accumulate records ────────────────────────────────────────

            self._oos.append(dict(
                Date=oos_date,
                MKT_return=r_mkt, TAN_return=r_TAN,
                MV_return=r_MV,   BL_return=r_BL,
            ))

            for j, t in enumerate(tickers):
                self._weights.append(dict(
                    Date=oos_date, Ticker=t,
                    w_mkt=w_mkt[j], w_TAN=w_TAN[j],
                    w_MV=w_MV[j],   w_BL=w_BL[j],
                ))
                self._clusters.append(dict(
                    Date=oos_date, Ticker=t,
                    Cluster_Label=int(sig["labels"][j]),
                    Assigned_View_Position=sig["positions"][t],
                ))

            for row in mc_res:
                self._mc_rows.append(dict(
                    Date=oos_date,
                    Delta_Noise=row["delta"],
                    Port_Type=row["port_type"],
                    Expected_Sharpe=row["expected_sharpe"],
                ))

            print(f"{str(oos_date.date()):<12} {n:>4}  {delta:>7.3f}  "
                  f"{r_mkt:>+8.4f}  {r_TAN:>+8.4f}  "
                  f"{r_MV:>+8.4f}  {r_BL:>+8.4f}")

        print()
        self._save_outputs()

    # ── Output ────────────────────────────────────────────────────────────

    def _save_outputs(self) -> None:
        out = self.cfg.OUTPUT_DIR
        os.makedirs(out, exist_ok=True)

        df_w = pd.DataFrame(self._weights)
        df_o = pd.DataFrame(self._oos)
        df_c = pd.DataFrame(self._clusters)
        df_m = pd.DataFrame(self._mc_rows)

        # ── 1. weights_history.csv ────────────────────────────────────────
        df_w.to_csv(f"{out}/weights_history.csv", index=False)
        print(f"[1] weights_history.csv          ({len(df_w):,} rows)")

        # ── 2. oos_returns_series.csv ─────────────────────────────────────
        df_o.to_csv(f"{out}/oos_returns_series.csv", index=False)
        print(f"[2] oos_returns_series.csv        ({len(df_o):,} rows)")

        # ── 3. cluster_signals_history.csv ───────────────────────────────
        df_c.to_csv(f"{out}/cluster_signals_history.csv", index=False)
        print(f"[3] cluster_signals_history.csv   ({len(df_c):,} rows)")

        # ── 4. monte_carlo_sensitivity.csv ───────────────────────────────
        df_m.to_csv(f"{out}/monte_carlo_sensitivity.csv", index=False)
        print(f"[4] monte_carlo_sensitivity.csv   ({len(df_m):,} rows)")

        # ── 5. performance_metrics_summary.csv ───────────────────────────
        perf_rows = [
            self.perf.compute(df_o["MKT_return"], "MKT"),
            self.perf.compute(df_o["TAN_return"], "TAN"),
            self.perf.compute(df_o["MV_return"],  "MV"),
            self.perf.compute(df_o["BL_return"],  "BL"),
        ]
        df_p = pd.DataFrame(perf_rows)
        df_p.to_csv(f"{out}/performance_metrics_summary.csv", index=False)
        print(f"[5] performance_metrics_summary.csv (4 portfolios)")

        # ── 6. ttest_results.csv ─────────────────────────────────────────
        bl_r = df_o["BL_return"]
        ttest_rows = [
            self.perf.ttest(bl_r, df_o["MKT_return"], "BL vs MKT"),
            self.perf.ttest(bl_r, df_o["TAN_return"], "BL vs TAN"),
            self.perf.ttest(bl_r, df_o["MV_return"],  "BL vs MV"),
        ]
        df_t = pd.DataFrame(ttest_rows)
        df_t.to_csv(f"{out}/ttest_results.csv", index=False)
        print(f"[6] ttest_results.csv             (3 pairs)")

        # ── Console summary ───────────────────────────────────────────────
        print("\n" + "=" * 72)
        print("PERFORMANCE SUMMARY  (OOS, annualised, rf = 5 % p.a.)")
        print("=" * 72)
        print(df_p.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))

        print("\n" + "=" * 72)
        print("PAIRED T-TEST RESULTS  (two-sided, H₀: BL return = other)")
        print("=" * 72)
        print(df_t.to_string(index=False))

        print(f"\nAll outputs saved to: ./{out}/\n")


# =============================================================================
#  ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("=" * 72)
    print(" Vietnamese Bank Stocks – BL + K-means + Monte Carlo Backtest")
    print("=" * 72 + "\n")

    cfg = Config()
    Backtester(cfg).run()
