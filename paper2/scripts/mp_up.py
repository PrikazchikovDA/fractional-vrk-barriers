"""High-precision continuation of the S family toward alpha -> 1 (Gauss-Newton in mpmath, 50 digits)."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import numpy as np, mpmath as mp
import polish
mp.mp.dps = 50
pts = [json.loads(l) for l in open(os.path.join(ROOT, 'paper2', 'data', 'cont_S16.jsonl'))]
v = np.array([p for p in pts if abs(p['alpha'] - 0.972) < 1e-9][0]['x'])
log = open(os.path.join(ROOT, 'paper2', 'data', 'mp_up.jsonl'), 'a')
a = 0.972
while a < 0.9985:
    vm, res, nt = polish.polish(v, 5, mp.mpf(str(round(a, 4))), 3 + 2*a, iters=30)
    c, A, b = polish.build(list(vm), 5, mp.mpf(str(round(a, 4))))
    mc = max(abs(x) for x in list(b) + [y for r in A for y in r])
    rec = dict(alpha=round(a, 4), res=res, ncond=nt, maxcoef=float(mc), cmin=float(min(c[1:])), cmax=float(max(c)), v=[mp.nstr(x, 45) for x in vm])
    log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps({k: rec[k] for k in ('alpha', 'res', 'maxcoef', 'cmin', 'cmax')}), flush=True)
    if res > 1e-30: break
    v = np.array([float(x) for x in vm]); a += 0.002
