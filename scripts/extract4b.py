import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import sys, json, numpy as np

from vrk_fast import Setup
from scipy.optimize import least_squares
a=float(sys.argv[1]); p=float(sys.argv[2]); s=4
S=Setup(s,a,p); keys=S.keys(); n=S.nunk(); rng=np.random.default_rng(11); sols=[]
lb=np.full(n,-np.inf); lb[:s-1]=1e-3; ub=np.full(n,np.inf); ub[:s-1]=1.5
for t in range(400):
    x0=rng.uniform(-1,1.5,n); x0[:s-1]=np.sort(rng.uniform(0.1,1.2,s-1))
    r=least_squares(lambda v:S.residual(v,keys),x0,bounds=(lb,ub),xtol=1e-15,ftol=1e-15,gtol=1e-15,max_nfev=1500)
    if np.sqrt(2*r.cost)<1e-12:
        sols.append(r.x); print('found',len(sols),'maxcoef',np.abs(r.x).max(),flush=True)
        if len(sols)>=6: break
sols.sort(key=lambda x: np.abs(x).max()); c,A,b=S.unpack(sols[0])
print(a, json.dumps(dict(c=c.tolist(),A=A.tolist(),b=b.tolist())), flush=True)
