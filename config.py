import numpy as np
import pandas as pd

_VN10Y = {2014:6.50,2015:6.50,2016:6.20,2017:5.80,2018:4.58,2019:4.58,2020:3.00,2021:2.53,2022:4.00,2023:3.50,2024:3.06,2025:3.10,2026:4.35}

def build_rf_series(dates):
    years = np.array(sorted(_VN10Y))
    vals = []
    for d in dates:
        y = d.year
        pct = _VN10Y.get(y, _VN10Y[years[np.argmin(np.abs(years-y))]])
        vals.append(pct/100/12)
    return pd.Series(vals, index=dates, name="rf_monthly")

class Config:
    DATA_DIR="data"; OUTPUT_DIR="output"
    CLOSE_FILE="bank_monthly_close.csv"; MKTCAP_FILE="bank_monthly_mktcap_bn.csv"
    LOOKBACK=36; TAU=1/36; N_CLUSTERS=4; MAX_WEIGHT=0.30; OPT_STARTS=10
    SIGNAL_LOOKBACK=6; SIGNAL_SKIP_LAST=1
    USE_DYNAMIC_RF=True; RF_ANNUAL=0.05; RF_MONTHLY=RF_ANNUAL/12
    MC_DELTAS=[-0.10,-0.05,0.00,0.05,0.10]; N_MC_SIMS=2000; RANDOM_SEED=42
