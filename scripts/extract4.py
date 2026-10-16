import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import sys, json, numpy as np

from vrk_fast import Setup
from scipy.optimize import least_squares
def best_solution(s, a, p, tries=80, seed=3):
    S=Setup(s,a,p); keys=S.keys(); n=S.nunk(); rng=np.random.default_rng(seed); sols=[]
    lb=np.full(n,-np.inf); lb[:s-1]=1e-3; ub=np.full(n,np.inf); ub[:s-1]=1.0   # nodes in (0,1]
    for t in range(tries):
        x0=rng.uniform(-1,1.5,n); x0[:s-1]=np.sort(rng.uniform(0.1,1.0,s-1))
        r=least_squares(lambda v:S.residual(v,keys),x0,bounds=(lb,ub),xtol=1e-15,ftol=1e-15,gtol=1e-15,max_nfev=4000)
        if np.sqrt(2*r.cost)<1e-12: sols.append(r.x)
    sols.sort(key=lambda x: np.abs(x).max())
    c,A,b=S.unpack(sols[0]); return dict(c=c.tolist(),A=A.tolist(),b=b.tolist(),n_found=len(sols),maxcoef=float(np.abs(sols[0]).max()))
out={}
for a,p in [(0.3,2.5),(0.7,3.7)]:
    out[str(a)]=best_solution(4,a,p); print(a, json.dumps(out[str(a)]), flush=True)
json.dump(out,open(_os.path.join(DATA,'tab4.json'),'w'))
