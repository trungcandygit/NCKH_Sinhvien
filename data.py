import os
import numpy as np
import pandas as pd
from config import Config, build_rf_series

class DataLoader:
    def __init__(self, cfg): self.cfg=cfg
    def load(self):
        cfg=self.cfg
        price=pd.read_csv(os.path.join(cfg.DATA_DIR,cfg.CLOSE_FILE),index_col="date",parse_dates=True)
        mktcap=pd.read_csv(os.path.join(cfg.DATA_DIR,cfg.MKTCAP_FILE),index_col="date",parse_dates=True)
        price.sort_index(inplace=True); mktcap.sort_index(inplace=True)
        log_ret=np.log(price/price.shift(1)); simple_ret=price.pct_change()
        rf=build_rf_series(log_ret.index) if cfg.USE_DYNAMIC_RF else pd.Series(cfg.RF_MONTHLY,index=log_ret.index,name="rf_monthly")
        return price,mktcap,log_ret,simple_ret,rf
