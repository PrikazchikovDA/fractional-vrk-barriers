"""Branch A of the reduced s=5 system: (1) continuation in alpha toward 1 (and alpha=1 itself);
(2) for alpha in {0.6,0.7,0.8,0.9} choose the free node c2 minimising max|coefficient| and save the full tableau."""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src')]
import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from reduced5 import residual, full_tableau
R = [json.loads(l) for l in open(os.path.join(ROOT, 'paper2', 'data', 'reduced5_scan.jsonl'))]
A_pts = {}
for r in R:
    if r['cmax'] > 0.9 and (r['alpha'] not in A_pts or r['maxb'] < A_pts[r['alpha']]['maxb']): A_pts[r['alpha']] = r
out = {}
# (1) continuation
p = np.array(A_pts[0.95]['p']); a = 0.95; cont = []
for a in list(np.arange(0.95, 0.999, 0.005)) + [0.999, 0.9999, 1.0]:
    r = least_squares(lambda q: residual(q, a), p, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
    res = float(np.linalg.norm(r.fun)); p = r.x
    cont.append(dict(alpha=round(float(a), 5), res=res, p=p.tolist()))
    print('cont', round(float(a), 5), '%.1e' % res, np.round(p, 5), flush=True)
out['continuation'] = cont
# (2) tableaux
tabs = {}
for a in [0.6, 0.7, 0.8, 0.9]:
    p = np.array(A_pts[a]['p'])
    f = lambda c2: np.abs(np.concatenate([full_tableau(p, a, c2)[1].ravel(), full_tableau(p, a, c2)[2]])).max()
    m = minimize_scalar(f, bounds=(0.02, 1.0), method='bounded')
    c, A, b = full_tableau(p, a, m.x)
    tabs[str(a)] = dict(c=c.tolist(), A=A.tolist(), b=b.tolist(), maxcoef=float(m.fun), reduced_res=float(np.linalg.norm(residual(p, a))))
    print('tab', a, 'c2=%.4f' % m.x, 'max|coef|=%.2f' % m.fun, 'c=', np.round(c, 4), flush=True)
out['tableaux'] = tabs
json.dump(out, open(os.path.join(ROOT, 'paper2', 'data', 'branchA.json'), 'w'), indent=1)
