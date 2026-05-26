import numpy as np
from scipy import optimize
from config import Config

class PortfolioOptimiser:
    def __init__(self,cfg): self.cfg=cfg
    def _neg_sharpe(self,w,mu,Sigma,rf): return -(float(w@mu)-rf)/np.sqrt(max(float(w@Sigma@w),1e-12))
    def _solve(self,obj,args,n):
        cfg=self.cfg; bnd=[(0,cfg.MAX_WEIGHT)]*n; con={"type":"eq","fun":lambda w:w.sum()-1}
        rng=np.random.default_rng(cfg.RANDOM_SEED); bw,bf=np.ones(n)/n,np.inf
        for _ in range(cfg.OPT_STARTS):
            w0=np.clip(rng.dirichlet(np.ones(n)),0,cfg.MAX_WEIGHT); w0/=w0.sum()
            r=optimize.minimize(obj,w0,args=args,method="SLSQP",bounds=bnd,constraints=con,options={"ftol":1e-10,"maxiter":2000})
            if r.success and r.fun<bf: bf,bw=r.fun,r.x
        bw=np.clip(bw,0,cfg.MAX_WEIGHT); bw/=bw.sum(); return bw
    def tangency(self,mu,Sigma,rf): return self._solve(self._neg_sharpe,(mu,Sigma,rf),len(mu))
    def bl_tangency(self,mu_BL,Sigma_BL,rf): return self._solve(self._neg_sharpe,(mu_BL,Sigma_BL,rf),len(mu_BL))
    def bl_equilibrium(self,Pi,Sigma,rf): return self._solve(self._neg_sharpe,(Pi,(1+self.cfg.TAU)*Sigma,rf),len(Pi))
    def equal_weight(self,n): return np.ones(n)/n
