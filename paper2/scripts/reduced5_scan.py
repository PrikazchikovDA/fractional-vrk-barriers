"""All solutions of the reduced s=5 system (7 eqs, 6 unknowns) for alpha in a grid, from many random starts.
Reports node range, max |b|, max |a_ij| (excluding the free column 2). Log paper2/data/reduced5_scan.jsonl."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src')]
import numpy as np
from scipy.optimize import least_squares
from reduced5 import residual, weights
log = open(os.path.join(ROOT, 'paper2', 'data', 'reduced5_scan.jsonl'), 'a')
ALPHAS = [float(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
for a in ALPHAS:
    rng = np.random.default_rng(int(1000*a)); found = {}
    for t in range(500):
        p0 = np.r_[rng.uniform(0.02, 1.5, 3), rng.normal(scale=2.0, size=3)]
        try:
            r = least_squares(lambda p: residual(p, a), p0, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=1500)
        except Exception: continue
        c = r.x[:3]
        if np.linalg.norm(r.fun) < 1e-12 and np.all(c > 1e-6) and min(abs(c[0]-c[1]), abs(c[0]-c[2]), abs(c[1]-c[2])) > 1e-5:
            key = tuple(np.round(r.x, 5))
            if key not in found:
                b = weights(*c, a)
                found[key] = dict(alpha=a, p=r.x.tolist(), cmax=float(c.max()), cmin=float(c.min()),
                                  maxb=float(np.abs(b).max()), maxa=float(np.abs(r.x[3:]).max()))
    for rec in sorted(found.values(), key=lambda d: max(d['maxb'], d['maxa'])):
        log.write(json.dumps(rec) + '\n'); log.flush()
    best = sorted(found.values(), key=lambda d: max(d['maxb'], d['maxa']))
    print(a, 'n_solutions', len(found), 'best:', [(round(d['cmin'], 3), round(d['cmax'], 3), round(max(d['maxb'], d['maxa']), 1)) for d in best[:4]], flush=True)
