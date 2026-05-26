import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from config import Config

class ClusterSignalGenerator:
    def __init__(self,cfg): self.cfg=cfg
    def generate(self,log_ret_window,tickers,rf_annual):
        cfg=self.cfg; n=len(tickers); T=len(log_ret_window)
        ann_ret=log_ret_window.mean(0)*12
        ann_vol=np.clip(log_ret_window.std(0,ddof=1)*np.sqrt(12),1e-8,None)
        mkt_m=log_ret_window.mean(1,keepdims=True)
        idio=log_ret_window-mkt_m
        skip=max(0,cfg.SIGNAL_SKIP_LAST); sig_L=max(1,min(cfg.SIGNAL_LOOKBACK,T-skip))
        end=T-skip if skip>0 else T; start=max(0,end-sig_L)
        idio_sig=idio[start:end].mean(0)*12
        composite=idio_sig-ann_vol
        sharpe_a=(idio_sig-rf_annual)/ann_vol
        feats=StandardScaler().fit_transform(np.column_stack([idio_sig,-ann_vol]))
        _base=dict(ann_ret=ann_ret,ann_vol=ann_vol,sharpe_arr=sharpe_a)
        try:
            labels=KMeans(n_clusters=cfg.N_CLUSTERS,random_state=cfg.RANDOM_SEED,n_init=10).fit_predict(feats)
        except:
            return {**_base,**dict(labels=np.zeros(n,int),positions={t:"Neutral" for t in tickers},P=np.zeros((1,n)),q=np.array([0.0]),cluster_sharpe={0:0.0},best_k=0,worst_k=0)}
        cs={k:composite[labels==k].mean() if (labels==k).any() else -np.inf for k in range(cfg.N_CLUSTERS)}
        bk=max(cs,key=cs.get); wk=min(cs,key=cs.get)
        if bk==wk:
            return {**_base,**dict(labels=labels,positions={t:"Neutral" for t in tickers},P=np.zeros((1,n)),q=np.array([0.0]),cluster_sharpe=cs,best_k=bk,worst_k=wk)}
        p=np.zeros(n)
        for i in range(n):
            if labels[i]==bk: p[i]=1.0
            elif labels[i]==wk: p[i]=-1.0
        ps=p[p>0].sum(); ns=abs(p[p<0].sum())
        if ps>0: p[p>0]/=ps
        if ns>0: p[p<0]/=ns
        if np.all(p==0): p[:]=1/n
        q=np.array([idio_sig[labels==bk].mean()-idio_sig[labels==wk].mean()])
        pos={t:("Long" if labels[i]==bk else "Short" if labels[i]==wk else "Neutral") for i,t in enumerate(tickers)}
        return {**_base,**dict(labels=labels,positions=pos,P=p.reshape(1,n),q=q,cluster_sharpe=cs,best_k=bk,worst_k=wk,idio_sig=idio_sig)}
