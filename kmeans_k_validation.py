"""
=============================================================================
K-means Optimal-K Validation
=============================================================================
Determines the best number of clusters (k) for the K-means signal used in
the Black-Litterman model via two complementary methods:

  1. Elbow method   – within-cluster sum of squares (inertia) vs k
  2. Silhouette analysis – average silhouette score vs k (Rousseeuw 1987)

For each rebalancing step in the rolling-window backtest we compute both
metrics.  The final result is the average over all steps.

k is evaluated over the range 2 … MAX_K (default 6).

Output
------
  output/kmeans_k_validation.csv  – per-step inertia + silhouette for each k
  output/kmeans_k_summary.csv     – mean inertia + silhouette across all steps

Conclusion: the k that maximises the average silhouette score is recommended.
=============================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_v2 import Config, DataLoader

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_K          = 6
MIN_STOCKS_REQ = MAX_K + 2   # need at least k+2 stocks to form meaningful clusters


def build_signal_features(lw_a: np.ndarray,
                           sig_lookback: int,
                           sig_skip: int) -> np.ndarray:
    """
    Build the composite feature matrix used by the BL K-means signal:
      col 0: idiosyncratic 6-1 momentum  (market-adjusted, skip-last-month)
      col 1: negative annualised vol      (low vol = high feature value)

    Returns
    -------
    features : (n,) raw (unscaled) feature array; caller handles scaling
    """
    T, n = lw_a.shape
    ann_vol = np.clip(lw_a.std(axis=0, ddof=1) * np.sqrt(12), 1e-8, None)

    mkt_m   = lw_a.mean(axis=1, keepdims=True)
    idio    = lw_a - mkt_m

    skip  = max(0, sig_skip)
    sl    = max(1, min(sig_lookback, T - skip))
    end   = T - skip if skip > 0 else T
    start = max(0, end - sl)

    idio_sig = idio[start:end].mean(axis=0) * 12
    return np.column_stack([idio_sig, -ann_vol])


def run_validation() -> pd.DataFrame:
    cfg = Config()
    loader = DataLoader(cfg)
    _, _, log_ret, simple_ret, _ = loader.load()

    dates  = log_ret.index
    L      = cfg.LOOKBACK
    k_vals = list(range(2, MAX_K + 1))

    records = []

    for i in range(L, len(dates) - 1):
        lw       = log_ret.iloc[i - L : i]
        oos_sr   = simple_ret.iloc[i + 1]
        active   = lw.notna().all() & oos_sr.notna()
        tickers  = active[active].index.tolist()

        if len(tickers) < MIN_STOCKS_REQ:
            continue

        raw_feat = build_signal_features(lw[tickers].values,
                                         cfg.SIGNAL_LOOKBACK,
                                         cfg.SIGNAL_SKIP_LAST)
        feat     = StandardScaler().fit_transform(raw_feat)

        row = {"Date": dates[i + 1], "N_stocks": len(tickers)}

        for k in k_vals:
            if len(tickers) <= k:
                row[f"inertia_k{k}"]    = np.nan
                row[f"silhouette_k{k}"] = np.nan
                continue

            km      = KMeans(n_clusters=k, random_state=cfg.RANDOM_SEED,
                             n_init=10)
            labels  = km.fit_predict(feat)
            inertia = km.inertia_

            # silhouette undefined when all labels are the same
            if len(set(labels)) < 2:
                sil = np.nan
            else:
                sil = silhouette_score(feat, labels)

            row[f"inertia_k{k}"]    = inertia
            row[f"silhouette_k{k}"] = sil

        records.append(row)

    df = pd.DataFrame(records)

    os.makedirs("output", exist_ok=True)
    df.to_csv("output/kmeans_k_validation.csv", index=False)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = []
    for k in k_vals:
        inertia_mean = df[f"inertia_k{k}"].mean()
        sil_mean     = df[f"silhouette_k{k}"].mean()
        sil_std      = df[f"silhouette_k{k}"].std()
        summary_rows.append(dict(
            k=k,
            Mean_Inertia=round(inertia_mean, 4),
            Mean_Silhouette=round(sil_mean, 4),
            Std_Silhouette=round(sil_std, 4),
        ))
    summary = pd.DataFrame(summary_rows)
    summary.to_csv("output/kmeans_k_summary.csv", index=False)

    # ── Console report ────────────────────────────────────────────────────────
    best_k_sil = int(summary.loc[summary["Mean_Silhouette"].idxmax(), "k"])

    print("=" * 60)
    print("  K-means Optimal-K Validation")
    print(f"  Steps evaluated: {len(df)}  |  k range: {k_vals[0]}–{k_vals[-1]}")
    print("=" * 60)
    print()
    print(summary.to_string(index=False))
    print()
    print(f"  Elbow method: inspect Mean_Inertia — look for the 'knee'")
    print(f"  Silhouette:   higher = better separation")
    print()
    print(f"  → Recommended k (max silhouette): k = {best_k_sil}")
    print()
    print(f"  Saved: output/kmeans_k_validation.csv  ({len(df)} rows)")
    print(f"  Saved: output/kmeans_k_summary.csv")

    return df, summary


if __name__ == "__main__":
    run_validation()
