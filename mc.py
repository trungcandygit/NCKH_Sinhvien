import numpy as np
from config import Config

class MonteCarloEngine:
    def __init__(self,cfg): self.cfg=cfg
    def run(self,w_KIO,w_IO,w_TAN,mu_hist,Sigma,Sigma_BL,Pi,q,A,base,rf):
        cfg=self.cfg
        if cfg.N_MC_SIMS==0: return []
        rng=np.random.default_rng(cfg.RANDOM_SEED); N=cfg.N_MC_SIMS
        vK=float(w_KIO@Sigma_BL@w_KIO); vI=float(w_IO@Sigma@w_IO); vT=float(w_TAN@Sigma@w_TAN)
        volK,volI,volT=np.sqrt(max(vK,1e-12)),np.sqrt(max(vI,1e-12)),np.sqrt(max(vT,1e-12))
        shT=(float(w_TAN@mu_hist)-rf)/volT; retT=float(w_TAN@mu_hist)
        out=[]
        for d in cfg.MC_DELTAS:
            sq=max(abs(d),0.01)*np.abs(q); sP=max(abs(d),0.01)*np.abs(Pi)
            qm=q+rng.normal(0,sq,size=(N,len(q)))
            muKm=base+(A@qm.T).T; rKs=muKm@w_KIO; shKs=(rKs-rf)/volK
            Pm=Pi+rng.normal(0,sP,size=(N,len(Pi))); rIs=Pm@w_IO; shIs=(rIs-rf)/volI
            out.append((d,shKs,rKs,volK,shIs,rIs,volI,shT,retT,volT))
        return out
