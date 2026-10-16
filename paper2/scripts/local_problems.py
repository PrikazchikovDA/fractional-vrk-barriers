"""Local exponent of a tableau on the test problems (L), (N), (S) of paper 1 (exact lag via quadrature),
as scripts/local.py does. usage: python local_problems.py tableau.json s alpha"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'src'), os.path.join(ROOT, 'scripts'), os.path.join(ROOT, 'paper2', 'src')]
import numpy as np
from vrk_global3 import problems
from local import step_err
from vtrees import unpack
v = np.array([float(x) for x in json.load(open(sys.argv[1]))['v']]); s = int(sys.argv[2]); a = float(sys.argv[3])
c, A, b = unpack(v, s, a)
hs = [1/10, 1/20, 1/40, 1/80, 1/160]
for pn, (f, yex) in problems(a).items():
    e = [step_err((c, A, b), a, f, yex, 0.5, h) for h in hs]
    r = [round(float(np.log2(e[i]/e[i+1])), 3) for i in range(len(e)-1)]
    rec = dict(tableau=os.path.basename(sys.argv[1]), alpha=a, problem=pn, err=e, rates=r)
    print(json.dumps(rec), flush=True)
    open(os.path.join(ROOT, 'paper2', 'data', 'local_problems.jsonl'), 'a').write(json.dumps(rec) + '\n')
