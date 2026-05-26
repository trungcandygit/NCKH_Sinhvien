import numpy as np
from config import Config

class BlackLittermanEngine:
    def __init__(self,cfg): self.cfg=cfg
    @staticmethod
    def _inv(M,eps=1e-8): return np.linalg.inv(M+eps*np.eye(M.shape[0]))
    def compute_delta(self,mu,Sigma,w_mkt,rf):
        v=float(w_mkt@Sigma@w_mkt)
        return 2.5 if v<1e-12 else float(np.clip((float(w_mkt@mu)-rf)/v,0.5,10.0))
    def posterior(self,Pi,Sigma,P,q):
        tau=self.cfg.TAU; K=len(q)
        tSi=self._inv(tau*Sigma)
        Om=self._inv(tau*(P@Sigma@P.T)+np.eye(K)*1e-8)
        Mi=self._inv(tSi+P.T@Om@P)
        mu_BL=Mi@(tSi@Pi+P.T@Om@q)
        return mu_BL,Mi+Sigma,Mi@P.T@Om,Mi@tSi@Pi
