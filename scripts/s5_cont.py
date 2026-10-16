import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import sys, time, json, numpy as np

from vrk_fast import Setup
from scipy.optimize import least_squares
x=np.load(_os.path.join(DATA,'s5_x_0.7_4.4_0.npy'))
def solve(a, x0):
    p=1+5*a-0.03; S=Setup(5,a,p); K=S.keys(); n=S.nunk()
    lb=np.full(n,-np.inf); lb[:4]=1e-8
    r=least_squares(lambda v: S.residual(v,K), x0, bounds=(lb,np.inf), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    return r.x, float(np.sqrt(2*r.cost)), len(K)
direction=float(sys.argv[1]); a=0.7; xc=x.copy()
c,A,b=Setup(5,0.7,4.4).unpack(x); print('a=0.7 c=',np.round(c,4)); print('b=',np.round(b,3)); print('A=\n',np.round(A,3)); sys.stdout.flush()
for i in range(60):
    a=round(a+direction,4)
    if a<=0.6667 or a>=0.75: break
    t=time.time(); xn,res,nk=solve(a,xc)
    mc=float(np.abs(xn).max()); print(json.dumps(dict(alpha=a,res=res,maxcoef=mc,ncond=nk,sec=round(time.time()-t,1))),flush=True)
    if res>1e-10: break
    xc=xn
