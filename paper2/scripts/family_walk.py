"""At fixed alpha, walk along the 1-dim solution family of S (kernel direction of the Jacobian),
then from each family point try to continue in alpha upward (0.975 ... 0.995). Double precision LM.
Detects whether the alpha-continuation stops at a fold (alpha* ~ 0.9725) for the whole family."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import numpy as np
from scipy.optimize import least_squares
from continue_S16 import CS
from vtrees import unpack
def solve(a, v):
    r = least_squares(CS(a).residual, v, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
    return r.x, float(np.linalg.norm(r.fun)), r.jac
pts = {round(json.loads(l)['alpha'], 4): json.loads(l) for l in open(os.path.join(ROOT, 'paper2', 'data', 'cont_S16.jsonl')) if json.loads(l)['res'] < 1e-11}
log = open(os.path.join(ROOT, 'paper2', 'data', 'family_walk.jsonl'), 'a')
a0 = 0.96
v, res, J = solve(a0, np.array(pts[a0]['x']))
for sgn in (+1, -1):
    w = v.copy(); prev = None
    for step in range(60):
        _, _, J = solve(a0, w)
        _, sv, Vt = np.linalg.svd(J); t = Vt[-1]
        if prev is not None and t @ prev < 0: t = -t
        prev = t
        w, res, _ = solve(a0, w + sgn * 0.05 * t / np.linalg.norm(t))
        if res > 1e-10: break
        # try to push alpha up from this family point
        x = w.copy(); amax = a0
        for a in np.arange(a0 + 0.0025, 0.9951, 0.0025):
            x2, r2, _ = solve(a, x)
            if r2 > 1e-10: break
            x, amax = x2, a
        c, A, b = unpack(w, 5, a0)
        rec = dict(sgn=sgn, step=step, res=res, alpha_reached=round(float(amax), 4), maxcoef=float(np.abs(np.concatenate([A.ravel(), b])).max()),
                   cmin=float(c[1:].min()), cmax=float(c.max()), sv_last=[float(x) for x in sv[-3:]], x=w.tolist())
        log.write(json.dumps(rec) + '\n'); log.flush()
        print(json.dumps({k: v for k, v in rec.items() if k != 'x'}), flush=True)
        if c.max() > 1.5 or c[1:].min() < 0: break
