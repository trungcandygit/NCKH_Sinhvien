"""
=============================================================================
K-means Optimal-K Validation  –  4 complementary criteria
=============================================================================
Determines the best number of clusters (k) for the K-means signal used in
the Black-Litterman model via four statistical methods:

  1. Elbow (Inertia / WCSS)   – within-cluster sum of squares vs k
  2. Silhouette score          – Rousseeuw (1987); higher = better
  3. Calinski-Harabasz Index   – Variance Ratio Criterion; higher = better
  4. Davies-Bouldin Index      – average cluster similarity; lower = better
  5. Gap Statistic             – Tibshirani, Walther & Hastie (2001);
                                  k* = smallest k with Gap(k) ≥ Gap(k+1) − s_{k+1}

For each rebalancing step in the rolling-window backtest we compute all
metrics. The final result is the average over all steps.

k is evaluated over the range 2 … MAX_K (default 6).

Output
------
  output/kmeans_k_validation.csv  – per-step metrics for each k
  output/kmeans_k_summary.csv     – mean metrics across all steps + votes
=============================================================================
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (silhouette_score,
                             calinski_harabasz_score,
                             davies_bouldin_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_v2 import Config, DataLoader

# ── Configuration ─────────────────────────────────────────────────────────────
MAX_K          = 6
N_GAP_REF      = 100   # reference datasets for Gap statistic
MIN_STOCKS_REQ = MAX_K + 2


def build_signal_features(lw_a: np.ndarray,
                           sig_lookback: int,
                           sig_skip: int) -> np.ndarray:
    """
    Composite feature matrix used by BL K-means signal:
      col 0: idiosyncratic 6-1 momentum  (market-adjusted, skip last month)
      col 1: negative annualised vol      (low vol = high feature value)
    """
    T, n    = lw_a.shape
    ann_vol = np.clip(lw_a.std(axis=0, ddof=1) * np.sqrt(12), 1e-8, None)
    mkt_m   = lw_a.mean(axis=1, keepdims=True)
    idio    = lw_a - mkt_m
    skip    = max(0, sig_skip)
    sl      = max(1, min(sig_lookback, T - skip))
    end     = T - skip if skip > 0 else T
    start   = max(0, end - sl)
    idio_sig = idio[start:end].mean(axis=0) * 12
    return np.column_stack([idio_sig, -ann_vol])


def _gap_statistic(feat: np.ndarray, k: int,
                   n_ref: int, rng: np.random.Generator) -> tuple:
    """
    Compute Gap(k) for one step.

    Gap(k) = E*[log Wk_ref] - log Wk
    where E* is the mean over n_ref uniform reference datasets.

    Returns (gap_k, sk) where sk = std(log Wk_ref) * sqrt(1 + 1/n_ref).
    """
    km  = KMeans(n_clusters=k, random_state=0, n_init=5)
    km.fit(feat)
    log_wk = np.log(km.inertia_ + 1e-10)

    # Reference: uniform over bounding box of feat
    mins = feat.min(axis=0)
    maxs = feat.max(axis=0)
    log_wk_refs = []
    for _ in range(n_ref):
        ref = rng.uniform(mins, maxs, size=feat.shape)
        km_r = KMeans(n_clusters=k, random_state=0, n_init=3)
        km_r.fit(ref)
        log_wk_refs.append(np.log(km_r.inertia_ + 1e-10))

    log_wk_refs = np.array(log_wk_refs)
    gap = log_wk_refs.mean() - log_wk
    sk  = log_wk_refs.std() * np.sqrt(1 + 1.0 / n_ref)
    return float(gap), float(sk)


def run_validation() -> tuple:
    cfg    = Config()
    loader = DataLoader(cfg)
    _, _, log_ret, simple_ret, _ = loader.load()

    dates  = log_ret.index
    L      = cfg.LOOKBACK
    k_vals = list(range(2, MAX_K + 1))
    rng    = np.random.default_rng(cfg.RANDOM_SEED)

    records = []

    print("=" * 70)
    print("  K-means Optimal-K Validation  (4 criteria + Gap statistic)")
    print("=" * 70)

    for i in range(L, len(dates) - 1):
        lw      = log_ret.iloc[i - L : i]
        oos_sr  = simple_ret.iloc[i + 1]
        active  = lw.notna().all() & oos_sr.notna()
        tickers = active[active].index.tolist()

        if len(tickers) < MIN_STOCKS_REQ:
            continue

        raw_feat = build_signal_features(lw[tickers].values,
                                         cfg.SIGNAL_LOOKBACK,
                                         cfg.SIGNAL_SKIP_LAST)
        feat     = StandardScaler().fit_transform(raw_feat)

        row = {"Date": dates[i + 1], "N_stocks": len(tickers)}

        for k in k_vals:
            if len(tickers) <= k:
                for m in ["inertia","silhouette","calinski_harabasz",
                          "davies_bouldin","gap","gap_sk"]:
                    row[f"{m}_k{k}"] = np.nan
                continue

            km     = KMeans(n_clusters=k, random_state=cfg.RANDOM_SEED,
                            n_init=10)
            labels = km.fit_predict(feat)

            row[f"inertia_k{k}"] = km.inertia_

            if len(set(labels)) < 2:
                row[f"silhouette_k{k}"]         = np.nan
                row[f"calinski_harabasz_k{k}"]  = np.nan
                row[f"davies_bouldin_k{k}"]     = np.nan
            else:
                row[f"silhouette_k{k}"]        = silhouette_score(feat, labels)
                row[f"calinski_harabasz_k{k}"] = calinski_harabasz_score(feat, labels)
                row[f"davies_bouldin_k{k}"]    = davies_bouldin_score(feat, labels)

            gap, sk = _gap_statistic(feat, k, N_GAP_REF, rng)
            row[f"gap_k{k}"]    = gap
            row[f"gap_sk_k{k}"] = sk

        records.append(row)
        if len(records) % 20 == 0:
            print(f"  Step {len(records)}/{len(range(L, len(dates)-1))} done …")

    df = pd.DataFrame(records)
    os.makedirs("output", exist_ok=True)
    df.to_csv("output/kmeans_k_validation.csv", index=False)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = []
    for k in k_vals:
        gap_arr = df[f"gap_k{k}"].values
        sk_arr  = df[f"gap_sk_k{k}"].values

        # Gap statistic criterion: Gap(k) - sk vs Gap(k+1) — computed globally
        summary_rows.append(dict(
            k                    = k,
            Mean_Inertia         = round(df[f"inertia_k{k}"].mean(), 4),
            Mean_Silhouette      = round(df[f"silhouette_k{k}"].mean(), 4),
            Std_Silhouette       = round(df[f"silhouette_k{k}"].std(),  4),
            Mean_Calinski_Harabasz = round(df[f"calinski_harabasz_k{k}"].mean(), 4),
            Mean_Davies_Bouldin  = round(df[f"davies_bouldin_k{k}"].mean(), 4),
            Mean_Gap             = round(np.nanmean(gap_arr), 4),
            Mean_Gap_sk          = round(np.nanmean(sk_arr),  4),
        ))

    summary = pd.DataFrame(summary_rows)

    # ── Gap statistic: k* = smallest k with Gap(k) >= Gap(k+1) - sk_{k+1} ─────
    best_k_gap = None
    for idx, row in summary.iterrows():
        if idx + 1 < len(summary):
            next_row = summary.iloc[idx + 1]
            if row["Mean_Gap"] >= next_row["Mean_Gap"] - next_row["Mean_Gap_sk"]:
                best_k_gap = int(row["k"])
                break
    if best_k_gap is None:
        best_k_gap = int(summary.iloc[-1]["k"])

    best_k_sil = int(summary.loc[summary["Mean_Silhouette"].idxmax(),      "k"])
    best_k_ch  = int(summary.loc[summary["Mean_Calinski_Harabasz"].idxmax(),"k"])
    best_k_db  = int(summary.loc[summary["Mean_Davies_Bouldin"].idxmin(),   "k"])

    # Add vote column
    votes = {k: 0 for k in k_vals}
    for best in [best_k_sil, best_k_ch, best_k_db, best_k_gap]:
        if best in votes:
            votes[best] += 1
    summary["Votes_Best_k"] = summary["k"].map(votes)

    summary.to_csv("output/kmeans_k_summary.csv", index=False)

    # ── Console report ────────────────────────────────────────────────────────
    print()
    print(summary[["k","Mean_Inertia","Mean_Silhouette",
                   "Mean_Calinski_Harabasz","Mean_Davies_Bouldin",
                   "Mean_Gap","Votes_Best_k"]].to_string(index=False))
    print()
    print("  Criteria results:")
    print(f"    Elbow (inertia)      → inspect 'knee' visually")
    print(f"    Silhouette (↑)       → best k = {best_k_sil}")
    print(f"    Calinski-Harabasz (↑)→ best k = {best_k_ch}")
    print(f"    Davies-Bouldin (↓)   → best k = {best_k_db}")
    print(f"    Gap Statistic        → best k = {best_k_gap}")
    print()
    best_vote = max(votes, key=votes.get)
    print(f"  → Majority vote: k = {best_vote}  "
          f"(chosen baseline: k = {cfg.N_CLUSTERS})")
    print()
    print(f"  Saved: output/kmeans_k_validation.csv  ({len(df)} rows)")
    print(f"  Saved: output/kmeans_k_summary.csv")

    return df, summary


if __name__ == "__main__":
    run_validation()
