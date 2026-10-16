"""Is S (16 system conditions, s=5) solvable with nodes c_i <= cm?  For cm in {1.0, 1.05, 1.1, 1.2},
alpha in {0.6, 0.7, 0.8, 0.9}: (a) homotopy from the known family (nodes up to ~1.25) by shrinking the bound
gradually; (b) 60 random LM starts with c = cm*sin^2(theta). Log paper2/data/S16_cbound.jsonl."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import numpy as np
from scipy.optimize import least_squares
from continue_S16 import CS
from search import _to_v, _to_z
pts = {round(json.loads(l)['alpha'], 4): json.loads(l) for l in open(os.path.join(ROOT, 'paper2', 'data', 'cont_S16.jsonl')) if json.loads(l)['res'] < 1e-11}
log = open(os.path.join(ROOT, 'paper2', 'data', 'S16_cbound.jsonl'), 'a')
def lm(C, cm, z0):
    F = lambda z: C.residual(_to_v(z, 5, cm))
    return least_squares(F, z0, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
for a in [0.7, 0.8, 0.9, 0.6]:
    C = CS(a); v0 = np.array(pts[a]['x'])
    for cm in [1.2, 1.1, 1.05, 1.0]:
        # (a) homotopy: shrink bound from max(c)+0.01 to cm in 20 steps
        v = v0.copy(); ok = True
        for b_ in np.linspace(v0[:4].max() + 0.01, cm, 21):
            z = _to_z(np.minimum(v, np.concatenate([np.full(4, b_ * 0.999), v[4:] + 0 * v[4:]])), 5, b_)
            r = lm(C, b_, z); v = _to_v(r.x, 5, b_); res_h = float(np.sqrt(2 * r.cost))
            if res_h > 1e-10: ok = False; break
        # (b) random
        rng = np.random.default_rng(int(100 * a + 10 * cm)); best = None
        for k in range(60):
            z0 = rng.uniform(-1.5, 1.5, 15); z0[:4] = rng.uniform(0.05, 1.5, 4)
            r = lm(C, cm, z0)
            if best is None or r.cost < best.cost: best = r
            if np.sqrt(2 * best.cost) < 1e-11: break
        vb = _to_v(best.x, 5, cm)
        rec = dict(alpha=a, cm=cm, homotopy_ok=ok, homotopy_res=res_h, homotopy_bound_reached=float(b_),
                   random_res=float(np.sqrt(2 * best.cost)), random_tries=k + 1, random_maxcoef=float(np.abs(vb).max()),
                   random_c=[round(x, 4) for x in vb[:4]], x_h=v.tolist(), x_r=vb.tolist())
        log.write(json.dumps(rec) + '\n'); log.flush()
        print(json.dumps({kk: vv for kk, vv in rec.items() if not kk.startswith('x_')}), flush=True)
        if ok: v0 = v
