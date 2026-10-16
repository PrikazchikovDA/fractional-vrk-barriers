import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import sys, time, json, numpy as np
from fractions import Fraction

from vrk_fast import exists, Setup
a=float(sys.argv[1]); af=Fraction(a).limit_denominator(1000)
qf=max(1+5*af, 2+3*af)
E=sorted({m*af+k for m in range(1,40) for k in range(0,7) if m*af+k<=qf})
below=E[-2]; at=E[-1]
out={'alpha':a,'formula':float(qf)}
for tag,p in [('below',below),('at',at)]:
    t=time.time(); S=Setup(5,a,float(p)); nk=len(S.keys())
    ok,res,x=exists(5,a,float(p),tries=10)
    out[tag]=dict(p=float(p),ok=bool(ok),res=float(res),nconds=nk,maxcoef=(float(np.abs(x).max()) if x is not None else None),sec=time.time()-t)
    print(json.dumps(out[tag]),flush=True)
    json.dump(out,open(_os.path.join(DATA,f's5_high_{a}.json'),'w'))
