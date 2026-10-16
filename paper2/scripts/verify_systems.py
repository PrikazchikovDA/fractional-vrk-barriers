"""High-precision verification of the tree B-series for SYSTEMS (no quadrature needed).

By the free-atom construction, a problem is fixed by g(t) (polynomial) and
f(t,u) = g(t) + sum_{r>=1,k} (t-t_n)^k/k! * (1/r!) T_{r,k}[(u-y(t))^r],  T_{r,k} symmetric r-linear maps.
Then g_k = g^{(k)}(t_n), D_{r,k} = T_{r,k}, and the stage errors e_i = Y_i - y(t_n+c_i h) satisfy
e_i = h^a sum_j a_ij [f(t_j, y(t_j)+e_j) - g(t_j)] + h^a [sum_j a_ij g(t_j) - I_i],  I_i = exact RL integral of g,
independently of y. delta is computed exactly (mpmath, 60 digits) and compared with the truncated tree series.
Expected: |delta - series_P| / h^{P+} -> const as h -> 0, where P+ = smallest tree exponent > P.
Writes jsonl log paper2/data/verify_systems.jsonl."""
import os, sys, json, itertools
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [os.path.join(ROOT, 'paper2', 'src')]
import numpy as np
import mpmath as mp
from math import factorial
from vtrees import all_trees, NK, sigma, split, LEAF, unpack, height
mp.mp.dps = 60

def sym_tensor(rng, d, r):
    T = rng.normal(size=(d,) * (r + 1))
    S = np.zeros_like(T)
    perms = list(itertools.permutations(range(1, r + 1)))
    for p in perms: S += np.transpose(T, (0,) + p)
    return S / len(perms)

def apply(T, vecs):
    """T[v1,...,vr] for numpy object arrays of mpf."""
    out = T
    for v in reversed(vecs): out = np.tensordot(out, v, axes=([out.ndim - 1], [0]))
    return out

class Problem:
    def __init__(self, d, rmax, kmax, seed):
        rng = np.random.default_rng(seed); self.d = d; self.rmax = rmax; self.kmax = kmax
        self.g = [np.array([mp.mpf(x) for x in rng.normal(size=d)], dtype=object) for _ in range(kmax + 1)]
        self.T = {(r, k): np.vectorize(mp.mpf, otypes=[object])(sym_tensor(rng, d, r))
                  for r in range(1, rmax + 1) for k in range(kmax + 1)}
    def F(self, t):
        if t == LEAF: return self.g[0]
        k, ch = split(t)
        if not ch: return self.g[k]
        return apply(self.T[(len(ch), k)], [self.F(c) for c in ch])
    def defect_f(self, tau, e):
        """f(t_n+tau, y+e) - g(t_n+tau)."""
        out = np.array([mp.mpf(0)] * self.d, dtype=object)
        for (r, k), T in self.T.items():
            out = out + tau**k / factorial(k) / factorial(r) * apply(T, [e] * r)
        return out

def delta_exact(P, c, A, b, a, h):
    s = len(c); h = mp.mpf(h); a = mp.mpf(a); ha = h**a
    def quad_defect(w, ci):   # sum_j w_j g(t_j) - exact integral at ci (all in units of h^a)
        out = np.array([mp.mpf(0)] * P.d, dtype=object)
        for k in range(P.kmax + 1):
            num = sum(w[j] * (c[j] * h)**k for j in range(s)) / factorial(k)
            ex = (ci**(k + a) if ci > 0 else 0) * h**k / mp.gamma(k + a + 1)
            out = out + (num - ex) * P.g[k]
        return out
    e = []; Fd = []
    for i in range(s):
        ei = ha * (quad_defect(A[i], c[i]) + sum((A[i][j] * Fd[j] for j in range(i)), np.array([mp.mpf(0)] * P.d, dtype=object)))
        e.append(ei); Fd.append(P.defect_f(c[i] * h, ei))
    return ha * (quad_defect(b, mp.mpf(1)) + sum((b[j] * Fd[j] for j in range(s)), np.array([mp.mpf(0)] * P.d, dtype=object)))

def psi_mp(t, c, A, b, a):
    s = len(c)
    def dk(k): return [sum(A[i][j] * c[j]**k for j in range(s)) - factorial(k) / mp.gamma(k + a + 1) * (c[i]**(k + a) if c[i] > 0 else 0) for i in range(s)]
    def st(t):
        k, ch = split(t)
        if not ch: return dk(k)
        v = [c[i]**k for i in range(s)]
        for cc in ch:
            w = st(cc); v = [v[i] * w[i] for i in range(s)]
        return [sum(A[i][j] * v[j] for j in range(s)) for i in range(s)]
    if t == LEAF: return sum(b) - 1 / mp.gamma(a + 1)
    k, ch = split(t)
    if not ch: return sum(b[i] * c[i]**k for i in range(s)) - factorial(k) / mp.gamma(k + a + 1)
    v = [c[i]**k for i in range(s)]
    for cc in ch:
        w = st(cc); v = [v[i] * w[i] for i in range(s)]
    return sum(b[i] * v[i] for i in range(s))

CASES = [(4, 0.3, 3.0, 0), (5, 0.7, 4.4, 1), (5, 0.45, 3.6, 2), (6, 0.6, 4.2, 3)]
if __name__ == "__main__":
    if len(sys.argv) > 1: CASES = json.loads(sys.argv[1])
    log = open(os.path.join(ROOT, 'paper2', 'data', 'verify_systems.jsonl'), 'a')
    for (s, a, P_, seed) in CASES:
        rng = np.random.default_rng(100 + seed)
        n = (s-1) + (s-1)*(s-2)//2 + s; v = rng.uniform(0.1, 1.2, n)
        c, A, b = unpack(v, s, a)
        am = mp.mpf(a)
        # exact row-sum in high precision
        cm = [mp.mpf(x) for x in c]; Am = [[mp.mpf(x) for x in row] for row in A]; bm = [mp.mpf(x) for x in b]
        for i in range(1, s): Am[i][0] = cm[i]**am / mp.gamma(am + 1) - sum(Am[i][1:i])
        trees = [t for t in all_trees(a, P_ + 1.5) if height(t) <= s]
        exps = sorted({round(NK(t)[0]*a + NK(t)[1], 12) for t in trees})
        Pplus = min(x for x in exps if x > P_ + 1e-12)
        kmax = int(P_) + 2; rmax = max(1, int(P_ / (1 + a)))
        Pr = Problem(3, rmax, kmax, seed)
        inc = [t for t in trees if NK(t)[0]*a + NK(t)[1] <= P_ + 1e-12]
        coef = [(NK(t)[0]*am + NK(t)[1], psi_mp(t, cm, Am, bm, am) / sigma(t) * Pr.F(t)) for t in inc]
        ratios = []
        for j in range(4, 15):
            h = mp.mpf(2)**(-j)
            dex = delta_exact(Pr, cm, Am, bm, am, h)
            ser = sum((h**ex * vec for ex, vec in coef), np.array([mp.mpf(0)]*3, dtype=object))
            r = max(abs(x) for x in (dex - ser)); ratios.append(float(r / h**Pplus))
        rates = [float(mp.log(ratios[i] / ratios[i+1], 2)) + Pplus for i in range(len(ratios)-1)]
        rec = dict(s=s, alpha=a, P=P_, Pplus=Pplus, ntrees=len(inc), remainder_over_hPplus=ratios, observed_rates=rates)
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n')
