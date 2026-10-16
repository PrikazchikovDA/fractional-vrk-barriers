import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')
import json, numpy as np, sys

from vrk_global3 import solve, problems, tab2, tab3A, tab3B
a=0.7; tab4=json.load(open(_os.path.join(DATA,'tab4.json')))['0.7']; Ns=[40,80,160,320]
meths=[('2 стадии',tab2(a),1+2*a),('3 стадии, A',tab3A(a),2+a),('3 стадии, B',tab3B(a),1+3*a),('4 стадии',(tab4['c'],tab4['A'],tab4['b']),1+4*a)]
out={}
for pname,(f,yex) in problems(a).items():
    out[pname]={m:[float(np.abs(solve(tb,a,f,yex,1.0,N)-yex(1.0)).max()) for N in Ns] for m,tb,q in meths}
    print(pname, out[pname], flush=True)
json.dump(dict(Ns=Ns,q={m:q for m,_,q in meths},err=out),open(_os.path.join(DATA,'data_a07.json'),'w'),ensure_ascii=False)
