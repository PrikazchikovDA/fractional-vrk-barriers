"""From continuation points of S (cont_S16.jsonl), try to additionally zero D_4 (tree g_4) and/or the chain B_4,
with warm start plus random restarts around the point. Logs paper2/data/extend_S16.jsonl."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import numpy as np
from scipy.optimize import least_squares
from vtrees import trees_by_order, NK, height, sigma, unpack, LEAF
from continue_S16 import CS, S_trees
from search import _to_v, _to_z
D4 = ((), (), (), ())                 # root with four leaves: g_4
B4 = ((((((),),),),),)                # chain of 5 stage vertices: J^4 g_1
pts = {round(json.loads(l)['alpha'], 6): json.loads(l) for l in open(os.path.join(ROOT, 'paper2', 'data', 'cont_S16.jsonl'))}
log = open(os.path.join(ROOT, 'paper2', 'data', 'extend_S16.jsonl'), 'a')
for a in [0.7, 0.72, 0.75, 0.8, 0.85, 0.9, 0.95]:
    p = pts[a]
    for extra_name, extra in [('D4', [D4]), ('B4', [B4]), ('D4+B4', [D4, B4])]:
        C = CS(a); C.trees = S_trees() + extra; C.groups = [[t] for t in C.trees]
        C.sig = [[float(sigma(t))] for t in C.trees]; C.ncond = len(C.trees)
        F = lambda z: C.residual(_to_v(z, 5, 1.5))
        rng = np.random.default_rng(0); best = None
        z0 = _to_z(np.array(p['x']), 5, 1.5)
        for k in range(30):
            z = z0 if k == 0 else z0 + rng.normal(scale=0.3 if k < 15 else 1.0, size=z0.size)
            r = least_squares(F, z, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
            if best is None or r.cost < best.cost: best = r
            if np.sqrt(2*best.cost) < 1e-11: break
        v = _to_v(best.x, 5, 1.5)
        rec = dict(alpha=a, extra=extra_name, ncond=C.ncond, res=float(np.sqrt(2*best.cost)), tries=k+1,
                   maxcoef=float(np.abs(v).max()), x=v.tolist())
        print(json.dumps({kk: vv for kk, vv in rec.items() if kk != 'x'}), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
