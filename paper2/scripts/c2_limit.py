"""alpha=0.7, nodes <= 1: fix c2 = eps (decreasing) and minimise the S residual over the other 14 parameters.
If min residual -> 0 only as eps -> 0, the 'solution' lies on the boundary c2 = 0 (confluent limit)."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import numpy as np
from scipy.optimize import least_squares
from continue_S16 import CS
R = [json.loads(l) for l in open(os.path.join(ROOT, 'paper2', 'data', 'S16_cbound.jsonl'))]
a = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7
start = [r for r in R if r['alpha'] == a and r['cm'] == 1.0][0]['x_r']
C = CS(a); x = np.array(start)
def F(z, eps):
    v = np.concatenate([[eps], 1.0*np.sin(z[:3])**2, z[3:]]); return C.residual(v)
z = np.concatenate([np.arcsin(np.sqrt(np.clip(x[1:4], 0, 1))), x[4:]])
log = open(os.path.join(ROOT, 'paper2', 'data', 'c2_limit.jsonl'), 'a')
for eps in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5]:
    best = None
    for k in range(6):
        z0 = z if k == 0 else z + np.random.default_rng(k).normal(scale=0.05, size=z.size)
        r = least_squares(lambda zz: F(zz, eps), z0, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        if best is None or r.cost < best.cost: best = r
    z = best.x; res = float(np.linalg.norm(best.fun))
    c = np.concatenate([[0, eps], np.sin(z[:3])**2]); co = np.abs(z[3:]).max()
    rec = dict(alpha=a, c2=eps, res=res, c=c.round(4).tolist(), maxcoef=float(co))
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
