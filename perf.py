import numpy as np
import pandas as pd
from scipy import stats
from config import Config

class PerformanceCalculator:
    def __init__(self,cfg): self.cfg=cfg
    def compute(self,r,rf,name):
        rf_a=rf.mean()*12; m=r.mean(); s=r.std(ddof=1)
        ar=(1+m)**12-1; av=s*np.sqrt(12)
        sh=(ar-rf_a)/av if av>1e-8 else 0.0
        ex=r-rf; ds=np.sqrt((ex.clip(upper=0)**2).mean())*np.sqrt(12); ds=max(ds,1e-8)
        cum=(1+r).cumprod(); mx=cum.expanding().max(); mdd=((cum-mx)/mx).min()
        return dict(Portfolio=name,Ann_Return=ar,Ann_Vol=av,Sharpe=sh,Sortino=(ar-rf_a)/ds,MDD=mdd,Calmar=ar/max(abs(mdd),1e-8),Win_Rate=(r>0).mean())
    def ttest(self,a,b,pair):
        x,y=a.align(b,join="inner"); t,p=stats.ttest_rel(x,y)
        return dict(Pair=pair,t_statistic=t,p_value=p,is_significant_5pct=(p<0.05))
    def jobson_korkie(self,ra,rb,rf,pair):
        a,b,f=np.array(ra),np.array(rb),np.array(rf); T=len(a)
        ma,mb=(a-f).mean(),(b-f).mean(); sa,sb=a.std(ddof=1),b.std(ddof=1)
        rho=np.cov(a,b,ddof=1)[0,1]/max(sa*sb,1e-12)
        Sa,Sb=ma/max(sa,1e-12),mb/max(sb,1e-12)
        th=(1/T)*(2-2*rho+0.5*Sa**2+0.5*Sb**2-Sa*Sb*rho)
        if th<=1e-12: return dict(Pair=pair,z_statistic=np.nan,p_value=np.nan,is_significant_5pct=False)
        z=(Sa-Sb)/np.sqrt(th); p=2*(1-stats.norm.cdf(abs(z)))
        return dict(Pair=pair,z_statistic=z,p_value=p,is_significant_5pct=(p<0.05))
    def jarque_bera(self,r,name):
        jb,p=stats.jarque_bera(r.dropna())
        return dict(Portfolio=name,JB_stat=jb,JB_pval=p,Skewness=float(r.skew()),Excess_Kurtosis=float(r.kurt()),is_normal_5pct=(p>0.05))
    def ljung_box(self,r,name,lags=(5,10,20)):
        s=r.dropna(); T=len(s); res=dict(Portfolio=name)
        for lag in lags:
            acf=[s.autocorr(lag=k) for k in range(1,lag+1)]
            Q=T*(T+2)*sum(v**2/(T-k) for k,v in enumerate(acf,1)); p=1-stats.chi2.cdf(Q,df=lag)
            res[f"LB{lag}_stat"]=Q; res[f"LB{lag}_pval"]=p; res[f"LB{lag}_no_autocorr"]=(p>0.05)
        return res
    def drawdown_analysis(self,r,name,thr=-0.20):
        cum=(1+r).cumprod(); mx=cum.expanding().max(); dd=(cum-mx)/mx
        n_ex=int((dd<thr).sum()); periods=[]; in_dd=False; si=ti=None; tv=0.0
        for i in range(len(dd)):
            v=dd.iloc[i]
            if not in_dd and v<-1e-4: in_dd=True; si=ti=i; tv=v
            elif in_dd:
                if v<tv: tv=v; ti=i
                if v>=-1e-4:
                    periods.append(dict(Portfolio=name,DD_Start=dd.index[si],DD_Trough=dd.index[ti],DD_Recovery=dd.index[i],Depth_pct=round(tv*100,2),Months_to_Trough=ti-si,Recovery_Months=i-ti,Total_Months=i-si)); in_dd=False
        if in_dd: periods.append(dict(Portfolio=name,DD_Start=dd.index[si],DD_Trough=dd.index[ti],DD_Recovery="Not recovered",Depth_pct=round(tv*100,2),Months_to_Trough=ti-si,Recovery_Months=None,Total_Months=len(dd)-1-si))
        deps=[p["Depth_pct"] for p in periods]; rec=[p["Recovery_Months"] for p in periods if p["Recovery_Months"] is not None]
        return dict(Portfolio=name,N_Drawdowns=len(periods),Max_DD_pct=round(dd.min()*100,2),Avg_DD_Depth_pct=round(np.mean(deps),2) if deps else 0.0,Avg_Recovery_Months=round(np.mean(rec),1) if rec else None,N_Months_Below_Neg20pct=n_ex),periods
