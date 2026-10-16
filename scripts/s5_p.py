import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import sys, time, json, numpy as np

from vrk_fast import exists, Setup
a=float(sys.argv[1]); p=float(sys.argv[2]); tries=int(sys.argv[3]) if len(sys.argv)>3 else 10; seed=int(sys.argv[4]) if len(sys.argv)>4 else 0
t=time.time(); nk=len(Setup(5,a,p).keys())
ok,res,x=exists(5,a,p,tries=tries,seed=seed)
r=dict(alpha=a,p=p,seed=seed,tries=tries,ok=bool(ok),res=float(res),nconds=nk,maxcoef=(float(np.abs(x).max()) if x is not None else None),sec=time.time()-t)
print(json.dumps(r),flush=True)
np.save(_os.path.join(DATA,f"s5_x_{a}_{p}_{seed}.npy"), x) if x is not None else None
with open(_os.path.join(DATA,'s5_scan.jsonl'),'a') as f: f.write(json.dumps(r)+'\n')
