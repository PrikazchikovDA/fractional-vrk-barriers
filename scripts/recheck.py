import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import json, numpy as np, sys
from vrk_global3 import solve, problems, tab3B
a=float(sys.argv[1]); which=sys.argv[2].split(','); probs=sys.argv[3].split(','); Ns=[int(x) for x in sys.argv[4].split(',')]
t4=json.load(open(_os.path.join(DATA,'tab4.json')))[str(a)]
M={'3B':tab3B(a),'4':(t4['c'],t4['A'],t4['b'])}
P=problems(a)
for pn in probs:
    f,yex=P[pn]
    for m in which:
        e=[float(np.abs(solve(M[m],a,f,yex,1.0,N)-yex(1.0)).max()) for N in Ns]
        r=[np.log2(e[i]/e[i+1]) for i in range(len(e)-1)]
        out=dict(alpha=a,method=m,problem=pn,Ns=Ns,err=e,rates=[round(x,3) for x in r])
        print(json.dumps(out),flush=True)
        open(_os.path.join(DATA,'recheck.jsonl'),'a').write(json.dumps(out)+'\n')
