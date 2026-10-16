"""Continuation in alpha of a solution of the 16 system conditions S = {trees with K<=3, N+K<=5, ht<=5}
(everything with |T| <= 3+2alpha for alpha>2/3 except the chain B_4). Logs nodes, max coefficient, residual,
and Jacobian singular values. usage: python continue_S16.py start.json alpha0 alpha1 step"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src')]
import numpy as np
from scipy.optimize import least_squares
from vtrees import trees_by_order, NK, height, sigma, unpack
from search import Conds
s = 5
def S_trees():
    out = []
    for n in range(1, 7):
        for t in trees_by_order(n):
            N, K = NK(t)
            if K <= 3 and N + K <= 5 and height(t) <= s: out.append(t)
    return out
class CS(Conds):
    def __init__(self, a):
        self.s, self.a, self.mode = s, a, 'system'
        self.trees = S_trees(); self.groups = [[t] for t in self.trees]
        self.sig = [[float(sigma(t))] for t in self.trees]; self.ncond = len(self.trees); self.nunk = 15
if __name__ == '__main__':
    st = json.load(open(sys.argv[1])); v = np.array([float(x) for x in st['v']])
    a0, a1, h = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    log = open(os.path.join(ROOT, 'paper2', 'data', 'cont_S16.jsonl'), 'a')
    a = a0
    while (a1 - a) * np.sign(a1 - a0) >= -1e-12:
        C = CS(a)
        r = least_squares(C.residual, v, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=5000)
        res = float(np.linalg.norm(r.fun))
        J = r.jac; sv = np.linalg.svd(J, compute_uv=False)
        c, A, b = unpack(r.x, s, a)
        rec = dict(alpha=round(a, 6), res=res, ncond=C.ncond, c=c.round(4).tolist(), cmax=float(c.max()), cmin=float(c[1:].min()),
                   maxcoef=float(np.abs(np.concatenate([A.ravel(), b])).max()), sv_min=float(sv[-1]), sv_2=float(sv[-2]), x=r.x.tolist())
        print(json.dumps({k: v for k, v in rec.items() if k != 'x'}), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if res > 1e-9: break
        v = r.x; a += h * np.sign(a1 - a0)
