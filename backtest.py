import os
import warnings
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
from config import Config
from data import DataLoader
from kmeans_signal import ClusterSignalGenerator
from bl import BlackLittermanEngine
from optimise import PortfolioOptimiser
from mc import MonteCarloEngine
from perf import PerformanceCalculator

PORTS = ["MKT","TAN","BL","BL_KIO","EW"]

class Backtester:
    def __init__(self,cfg,verbose=True):
        self.cfg=cfg; self.verbose=verbose
        self.loader=DataLoader(cfg); self.signal=ClusterSignalGenerator(cfg)
        self.bl=BlackLittermanEngine(cfg); self.opt=PortfolioOptimiser(cfg)
        self.mc=MonteCarloEngine(cfg); self.perf=PerformanceCalculator(cfg)
        self._oos=[]; self._w=[]; self._cl=[]; self._ic=[]
        self._mks={}; self._mkr={}; self._mkv={}
        self._mis={}; self._mir={}; self._miv={}
        self._mts={}; self._mtr={}; self._mtv={}
        self._mn=0

    def run(self):
        cfg=self.cfg; np.random.seed(cfg.RANDOM_SEED)
        price,mktcap,lr,sr,rf=self.loader.load()
        dates=lr.index; L=cfg.LOOKBACK
        if self.verbose:
            print(f"Range: {dates[0].date()} to {dates[-1].date()}  Lookback: {L}\n")
            print(f"{'Date':<12}{'N':>4}{'MKT':>8}{'TAN':>8}{'BL':>8}{'BL_KIO':>8}{'EW':>8}{'rf%':>6}")
            print("-"*62)
        for i in range(L,len(dates)-1):
            lw=lr.iloc[i-L:i]; osr=sr.iloc[i+1]; od=dates[i+1]
            rft=float(rf.iloc[i])*12; rfo=float(rf.iloc[i+1])
            mask=lw.notna().all()&osr.notna(); tks=mask[mask].index.tolist()
            if len(tks)<cfg.N_CLUSTERS+2: continue
            n=len(tks); lwa=lw[tks].values; oor=osr[tks].values
            mc=mktcap.iloc[i][tks].fillna(0).clip(lower=0); mcs=mc.sum()
            if mcs<=0: continue
            wm=(mc/mcs).values
            S=np.cov(lwa.T)*12+np.eye(n)*1e-8; mh=lwa.mean(0)*12
            sig=self.signal.generate(lwa,tks,rft); P,q=sig["P"],sig["q"]
            dlt=self.bl.compute_delta(mh,S,wm,rft); Pi=dlt*S@wm
            muBL,SBL,A,base=self.bl.posterior(Pi,S,P,q)
            wT=self.opt.tangency(mh,S,rft); wB=self.opt.bl_equilibrium(Pi,S,rft)
            wK=self.opt.bl_tangency(muBL,SBL,rft); wE=self.opt.equal_weight(n)
            rm,rt,rb,rk,re=float(wm@oor),float(wT@oor),float(wB@oor),float(wK@oor),float(wE@oor)
            self._oos.append(dict(Date=od,RF_monthly=rfo,MKT=rm,TAN=rt,BL=rb,BL_KIO=rk,EW=re))
            isig=sig.get("idio_sig")
            if isig is not None and len(isig)>2:
                ic=float(stats.spearmanr(isig,oor).correlation)
                hit=int(oor[sig["labels"]==sig["best_k"]].mean()>oor[sig["labels"]==sig["worst_k"]].mean()) if sig["best_k"]!=sig["worst_k"] else 0
            else: ic,hit=np.nan,0
            self._ic.append(dict(Date=od,IC=ic,Hit=hit,N_stocks=n))
            for j,t in enumerate(tks):
                self._w.append(dict(Date=od,Ticker=t,w_mkt=wm[j],w_TAN=wT[j],w_BL=wB[j],w_BL_KIO=wK[j],w_EW=wE[j]))
                self._cl.append(dict(Date=od,Ticker=t,Cluster_Label=int(sig["labels"][j]),Assigned_View_Position=sig["positions"][t]))
            mcr=self.mc.run(wK,wB,wT,mh,S,SBL,Pi,q,A,base,rft)
            if mcr:
                for (d,ks,kr,kv,is_,ir,iv,ts,tr,tv) in mcr:
                    if d not in self._mks:
                        N2=len(ks)
                        self._mks[d]=np.zeros(N2);self._mkr[d]=np.zeros(N2);self._mkv[d]=0.0
                        self._mis[d]=np.zeros(N2);self._mir[d]=np.zeros(N2);self._miv[d]=0.0
                        self._mts[d]=0.0;self._mtr[d]=0.0;self._mtv[d]=0.0
                    self._mks[d]+=ks;self._mkr[d]+=kr;self._mkv[d]+=kv
                    self._mis[d]+=is_;self._mir[d]+=ir;self._miv[d]+=iv
                    self._mts[d]+=ts;self._mtr[d]+=tr;self._mtv[d]+=tv
                self._mn+=1
            if self.verbose:
                print(f"{str(od.date()):<12}{n:>4}{rm:>+8.4f}{rt:>+8.4f}{rb:>+8.4f}{rk:>+8.4f}{re:>+8.4f}{rfo*100:>5.2f}%")
        if self.verbose: print()
        df=pd.DataFrame(self._oos); self._save(df); return df

    def _save(self,df):
        cfg=self.cfg; out=cfg.OUTPUT_DIR; os.makedirs(out,exist_ok=True)
        df.to_csv(f"{out}/oos_returns_v2.csv",index=False)
        pd.DataFrame(self._w).to_csv(f"{out}/weights_v2.csv",index=False)
        pd.DataFrame(self._cl).to_csv(f"{out}/cluster_signals_v2.csv",index=False)
        if self._mn>0:
            S=self._mn; rows=[]
            for d in sorted(self._mks):
                ks=self._mks[d]/S;kr=self._mkr[d]/S;kv=self._mkv[d]/S
                is_=self._mis[d]/S;ir=self._mir[d]/S;iv=self._miv[d]/S
                ts=self._mts[d]/S;tr=self._mtr[d]/S;tv=self._mtv[d]/S
                for sid,(sk,rk,si,ri) in enumerate(zip(ks,kr,is_,ir),1):
                    rows.append({"Delta_Noise":d,"Sim_ID":sid,"Sharpe_BL_KIO":round(float(sk),6),"Return_BL_KIO":round(float(rk),6),"Vol_BL_KIO":round(float(kv),6),"Sharpe_BL_IO":round(float(si),6),"Return_BL_IO":round(float(ri),6),"Vol_BL_IO":round(float(iv),6),"Sharpe_TAN":round(float(ts),6),"Return_TAN":round(float(tr),6),"Vol_TAN":round(float(tv),6)})
            pd.DataFrame(rows).to_csv(f"{out}/monte_carlo_v2.csv",index=False)
        di=df.set_index("Date"); rf=di["RF_monthly"]; p=self.perf
        pd.DataFrame([p.compute(di[x],rf,x) for x in PORTS]).to_csv(f"{out}/performance_summary_v2.csv",index=False)
        bk=di["BL_KIO"]
        pd.DataFrame([p.ttest(bk,di[x],f"BL_KIO vs {x}") for x in ["MKT","TAN","BL","EW"]]).to_csv(f"{out}/ttest_results_v2.csv",index=False)
        jk=pd.DataFrame([p.jobson_korkie(bk,di[x],rf,f"BL_KIO vs {x}") for x in ["MKT","TAN","BL","EW"]]); jk.to_csv(f"{out}/jobson_korkie_results.csv",index=False)
        pd.DataFrame([p.jarque_bera(di[x],x) for x in PORTS]).to_csv(f"{out}/distribution_jb_tests.csv",index=False)
        pd.DataFrame([p.ljung_box(di[x],x) for x in PORTS]).to_csv(f"{out}/distribution_lb_tests.csv",index=False)
        sums,pds=[],[]
        for x in PORTS:
            s,ps=p.drawdown_analysis(di[x],x); sums.append(s); pds.extend(ps)
        pd.DataFrame(sums).to_csv(f"{out}/drawdown_summary.csv",index=False)
        pd.DataFrame(pds).to_csv(f"{out}/drawdown_periods.csv",index=False)
        if self._ic: pd.DataFrame(self._ic).to_csv(f"{out}/signal_ic_analysis.csv",index=False)
        idx=di.index; mid=idx[len(idx)//2]; rows2=[]
        for lbl,mask in [("Full",slice(None)),("Period_1",idx<mid),("Period_2",idx>=mid)]:
            srf=rf if isinstance(mask,slice) else rf[mask]
            if len(srf)<6: continue
            for x in PORTS:
                s=di[x] if isinstance(mask,slice) else di.loc[mask,x]
                m=p.compute(s,srf,x)
                rows2.append(dict(Period=lbl,Period_Start=str(s.index[0].date()),Period_End=str(s.index[-1].date()),N_months=len(s),Portfolio=x,Ann_Return=m["Ann_Return"],Ann_Vol=m["Ann_Vol"],Sharpe=m["Sharpe"],Sortino=m["Sortino"],MDD=m["MDD"],Calmar=m["Calmar"]))
        pd.DataFrame(rows2).to_csv(f"{out}/subperiod_analysis.csv",index=False)
        pf=pd.DataFrame([p.compute(di[x],rf,x) for x in PORTS])
        print("="*70+"\nPERFORMANCE SUMMARY\n"+"="*70)
        print(pf.to_string(index=False,float_format=lambda x:f"{x:+.4f}"))
        print("\n"+"="*70+"\nJOBSON-KORKIE TEST\n"+"="*70)
        print(jk.to_string(index=False))
        saved=["oos_returns_v2.csv","weights_v2.csv","cluster_signals_v2.csv","monte_carlo_v2.csv","performance_summary_v2.csv","ttest_results_v2.csv","jobson_korkie_results.csv","distribution_jb_tests.csv","distribution_lb_tests.csv","drawdown_summary.csv","drawdown_periods.csv","signal_ic_analysis.csv","subperiod_analysis.csv"]
        print(f"\nOutputs saved to ./{out}/")
        for f in saved:
            fp=f"{out}/{f}"; sz=os.path.getsize(fp) if os.path.exists(fp) else 0
            print(f"  [✓] {f:<45} {sz:>8,} bytes")
