import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import sys, time, json, numpy as np

from vrk_fast import Setup
from scipy.optimize import least_squares
a=float(sys.argv[1]); p=float(sys.argv[2]); tries=int(sys.argv[3]); cmax=float(sys.argv[4]); seed=int(sys.argv[5]) if len(sys.argv)>5 else 0
S=Setup(5,a,p); K=S.keys(); n=S.nunk()
lb=np.full(n,-np.inf); ub=np.full(n,np.inf); lb[:4]=1e-8; ub[:4]=cmax
rng=np.random.default_rng(seed); best=None; t=time.time()
for i in range(tries):
    x0=rng.uniform(-1,2,n); x0[:4]=rng.uniform(0.05,cmax,4)
    r=least_squares(lambda v: S.residual(v,K), x0, bounds=(lb,ub), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
    if best is None or r.cost<best.cost: best=r
    if np.sqrt(2*best.cost)<1e-10: break
res=float(np.sqrt(2*best.cost))
out=dict(alpha=a,p=p,cmax=cmax,ok=res<1e-10,res=res,ncond=len(K),maxcoef=float(np.abs(best.x).max()),c=[round(float(v),4) for v in best.x[:4]],sec=round(time.time()-t))
print(json.dumps(out),flush=True)
with open(_os.path.join(DATA,'s5_cb.jsonl'),'a') as f: f.write(json.dumps(out)+'\n')
