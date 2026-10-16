import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import json, numpy as np, sys
from vrk_global3 import solve, problems, tab2, tab3A, tab3B
a=float(sys.argv[1]); Ns=[int(x) for x in sys.argv[2].split(',')]; out=sys.argv[3]
t4=json.load(open(_os.path.join(DATA,'tab4.json')))[str(a)]
M={'2':tab2(a),'3A':tab3A(a),'3B':tab3B(a),'4':(t4['c'],t4['A'],t4['b'])}
for pn,(f,yex) in problems(a).items():
    for m,tb in M.items():
        e=[float(np.abs(solve(tb,a,f,yex,1.0,N)-yex(1.0)).max()) for N in Ns]
        r=[round(float(np.log2(e[i]/e[i+1])),3) for i in range(len(e)-1)]
        rec=dict(alpha=a,method=m,problem=pn,Ns=Ns,err=e,rates=r); print(json.dumps(rec),flush=True)
        open(out,'a').write(json.dumps(rec)+'\n')
