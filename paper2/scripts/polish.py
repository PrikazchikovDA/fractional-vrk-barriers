"""Polish a tableau in high precision: Gauss-Newton (mpmath) on all system conditions with |T| <= L.
usage: python polish.py in.npy s alpha L out.json"""
import os, sys, json
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src'), os.path.join(ROOT, 'paper2', 'scripts')]
import numpy as np, mpmath as mp
from vtrees import all_trees, NK, height, sigma
from verify_systems import psi_mp
mp.mp.dps = 60
def build(v, s, a):
    c = [mp.mpf(0)] + list(v[:s-1]); A = [[mp.mpf(0)]*s for _ in range(s)]; idx = s-1
    for i in range(1, s):
        for j in range(1, i): A[i][j] = v[idx]; idx += 1
        A[i][0] = c[i]**a/mp.gamma(a+1) - sum(A[i][1:i])
    return c, A, list(v[idx:idx+s])
def polish(v0, s, a, L, iters=40):
    am = mp.mpf(a) if not isinstance(a, str) else mp.mpf(a)
    trees = [t for t in all_trees(float(a), L + 0.5) if height(t) <= s and NK(t)[0]*float(a) + NK(t)[1] <= L + 1e-9]
    v = mp.matrix([mp.mpf(float(x)) for x in v0])
    def R(v):
        c, A, b = build(list(v), s, am)
        return mp.matrix([psi_mp(t, c, A, b, am)/sigma(t) for t in trees])
    for it in range(iters):
        r = R(v); nr = mp.norm(r)
        if nr < mp.mpf(10)**(-50): break
        n = len(v); J = mp.matrix(len(r), n); hstep = mp.mpf(10)**(-30)
        for k in range(n):
            vk = v.copy(); vk[k] += hstep; J[:, k] = (R(vk) - r)/hstep
        dv = mp.lu_solve(J.T*J + mp.eye(n)*mp.mpf(10)**(-40), -(J.T*r))   # minimum-change GN step
        lam = mp.mpf(1)
        for _ in range(30):   # backtracking: keep nodes positive and decrease the residual
            w = v + lam*dv
            if all(w[k] > 0 for k in range(s-1)):
                try:
                    if mp.norm(R(w)) < nr: break
                except TypeError: pass
            lam /= 2
        v = w
    return v, float(mp.norm(R(v))), len(trees)
if __name__ == '__main__':
    v0 = np.load(sys.argv[1]); s = int(sys.argv[2]); a = mp.mpf(sys.argv[3]); L = float(sys.argv[4])
    v, res, nt = polish(v0, s, a, L)
    print('conditions', nt, 'residual', res)
    json.dump(dict(s=s, alpha=sys.argv[3], L=L, residual=res, v=[mp.nstr(x, 50) for x in v]), open(sys.argv[5], 'w'), indent=0)
