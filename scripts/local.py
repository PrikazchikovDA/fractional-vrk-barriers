"""Direct measurement of the local error exponent: one step from t_n with EXACT lag term."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')

import json, numpy as np, sys, warnings
from scipy.integrate import quad
from scipy.special import gamma as G
from vrk_global3 import problems, tab2, tab3A, tab3B
warnings.filterwarnings('ignore')
def step_err(tab, a, f, yex, tn, h):
    c,A,b=(np.array(x,float) for x in tab); s=len(c)
    g=lambda u: np.atleast_1d(f(u, yex(u)))
    dim=g(tn).size
    def Phi(t):  # y(t) - (1/G(a)) int_{tn}^{t} (t-s)^{a-1} g(s) ds
        L=t-tn
        if L<=0: return np.atleast_1d(yex(t)).astype(float)
        I=np.array([quad(lambda u,k=k: g(t-u)[k], 0, L, weight='alg', wvar=(a-1,0), epsabs=1e-17, epsrel=1e-14, limit=200)[0] for k in range(dim)])
        return np.atleast_1d(yex(t)).astype(float)-I/G(a)
    F=[]
    for i in range(s):
        Y=Phi(tn+c[i]*h)+h**a*sum(A[i,j]*F[j] for j in range(i)) if i else Phi(tn)
        F.append(np.atleast_1d(f(tn+c[i]*h,Y)))
    y1=Phi(tn+h)+h**a*sum(b[j]*F[j] for j in range(s))
    return float(np.abs(y1-np.atleast_1d(yex(tn+h))).max())
if __name__=='__main__':
    a=float(sys.argv[1]); t4=json.load(open(_os.path.join(DATA,'tab4.json')))[str(a)]
    M={'2':tab2(a),'3A':tab3A(a),'3B':tab3B(a),'4':(t4['c'],t4['A'],t4['b'])}
    hs=[1/20,1/40,1/80,1/160,1/320]
    for pn,(f,yex) in problems(a).items():
        for m,tb in M.items():
            e=[step_err(tb,a,f,yex,0.5,h) for h in hs]
            r=[round(float(np.log2(e[i]/e[i+1])),3) for i in range(len(e)-1)]
            rec=dict(alpha=a,method=m,problem=pn,hs=hs,err=e,rates=r); print(json.dumps(rec),flush=True)
            open(_os.path.join(DATA,'local.jsonl'),'a').write(json.dumps(rec)+'\n')
