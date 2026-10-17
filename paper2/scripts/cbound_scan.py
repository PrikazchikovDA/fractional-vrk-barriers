"""S (16 system conditions, s=5) with nodes c <= cm: (a) continuation in cm downward from every known solution
branch (step 0.005); (b) 100 random LM starts at cm = 1.0 and 1.05 for alpha in {0.7,0.75,0.8,0.85,0.9}.
Log paper2/data/cbound_scan.jsonl."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import numpy as np
from scipy.optimize import least_squares
from continue_S16 import CS
from search import _to_v, _to_z
log = open(os.path.join(ROOT, 'paper2', 'data', 'cbound_scan.jsonl'), 'a')
def lm(C, cm, z0, nfev=3000):
    return least_squares(lambda z: C.residual(_to_v(z, 5, cm)), z0, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=nfev)
def emit(rec):
    log.write(json.dumps(rec) + '\n'); log.flush(); print(json.dumps({k: v for k, v in rec.items() if k != 'x'}), flush=True)
# (a) continuation in cm from the second branch found at alpha=0.8, cm=1.2
R = [json.loads(l) for l in open(os.path.join(ROOT, 'paper2', 'data', 'S16_cbound.jsonl'))]
seeds = [(r['alpha'], np.array(r['x_r'])) for r in R if r['random_res'] < 1e-11]
for a, v in seeds:
    C = CS(a); cm = float(v[:4].max()) + 0.005
    while cm > 0.95:
        r = lm(C, cm, _to_z(np.minimum(v, np.r_[np.full(4, cm*0.9999), v[4:]]), 5, cm)); res = float(np.sqrt(2*r.cost))
        if res > 1e-10: emit(dict(part='a', alpha=a, cm_stop=round(cm, 4), res=res)); break
        v = _to_v(r.x, 5, cm); cm -= 0.005
    else:
        emit(dict(part='a', alpha=a, cm_stop=round(cm, 4), res=res, reached_1=True, x=v.tolist()))
# (b) random search
for a in [0.7, 0.75, 0.8, 0.85, 0.9]:
    C = CS(a)
    for cm in [1.0, 1.05]:
        rng = np.random.default_rng(int(1000*a + 100*cm)); best = None
        for k in range(100):
            z0 = rng.uniform(-1.5, 1.5, 15); z0[:4] = rng.uniform(0.05, 1.5, 4)
            r = lm(C, cm, z0)
            if best is None or r.cost < best.cost: best = r
            if np.sqrt(2*best.cost) < 1e-11: break
        v = _to_v(best.x, 5, cm)
        emit(dict(part='b', alpha=a, cm=cm, res=float(np.sqrt(2*best.cost)), tries=k+1, maxcoef=float(np.abs(v).max()),
                  c=[round(x, 4) for x in v[:4]], x=v.tolist()))
