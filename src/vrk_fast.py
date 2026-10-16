"""Fast numeric version of vrk_order.py: same expansion, but method parameters are numbers and
atoms are formal monomials (dict-based polynomials). Residual = all atom-monomial coefficients of
the local error at exponents m*alpha+k <= p. Used for least-squares existence search."""
import numpy as np
from fractions import Fraction
from math import factorial
from scipy.special import gamma as G
from scipy.optimize import least_squares

class Setup:
    def __init__(self, s, alpha, p):
        self.s, self.alpha = s, alpha
        self.a_fr = Fraction(alpha).limit_denominator(1000); self.pf = Fraction(p).limit_denominator(1000)
        self.K = int(p) + 1
        self.rmax = max(1, int(self.pf / (1 + self.a_fr)))  # row-sum condition => stage errors start at h^{1+alpha}
        # atom ids: ('g',k) and ('D',r,k)
    def ok(self, m, k): return m*self.a_fr + k <= self.pf
    # polynomial: dict (m,k,atom_tuple_sorted) -> float
    def mul(self, X, Y):
        Z = {}
        for (m1, k1, a1), v1 in X.items():
            for (m2, k2, a2), v2 in Y.items():
                m, k = m1+m2, k1+k2
                if not self.ok(m, k): continue
                key = (m, k, tuple(sorted(a1+a2)))
                Z[key] = Z.get(key, 0.0) + v1*v2
        return Z
    @staticmethod
    def add(X, Y, w=1.0):
        Z = dict(X)
        for key, v in Y.items(): Z[key] = Z.get(key, 0.0) + w*v
        return Z
    def fval(self, cj, e):
        F = {(0, k, (('g', k),)): cj**k/factorial(k) for k in range(self.K+1) if self.ok(0, k)}
        P = {(0, 0, ()): 1.0}
        for r in range(1, self.rmax+1):
            P = self.mul(P, e)
            if not P: break
            Dr = {(0, k, (('D', r, k),)): cj**k/factorial(k) for k in range(self.K+1)}
            F = self.add(F, self.mul(P, Dr), 1.0/factorial(r))
        return F
    def shift(self, X): return {(m+1, k, a): v for (m, k, a), v in X.items() if self.ok(m+1, k)}
    def exact(self, ci):
        al = self.alpha
        return {(1, k, (('g', k),)): -(ci**(k+al) if ci > 0 else 0.0)/G(k+al+1) for k in range(self.K+1) if self.ok(1, k)}
    def unpack(self, v):
        s = self.s; c = np.concatenate([[0.0], v[:s-1]]); idx = s-1
        A = np.zeros((s, s))
        for i in range(1, s):
            for j in range(1, i): A[i, j] = v[idx]; idx += 1
            A[i, 0] = c[i]**self.alpha/G(self.alpha+1) - A[i, 1:i].sum()   # row-sum (Lubich (9))
        b = v[idx:idx+s]
        return c, A, b
    def nunk(self): return (self.s-1) + (self.s-1)*(self.s-2)//2 + self.s
    def delta(self, v):
        c, A, b = self.unpack(v); Fs = []
        for i in range(self.s):
            S = {}
            for j in range(i): S = self.add(S, Fs[j], A[i, j])
            ei = self.add(self.shift(S), self.exact(c[i])) if i > 0 else {}
            Fs.append(self.fval(c[i], ei))
        S = {}
        for j in range(self.s): S = self.add(S, Fs[j], b[j])
        return self.add(self.shift(S), self.exact(1.0))
    def keys(self):
        # fixed ordering of condition keys (exponent value, atom monomial), from a random parameter point
        rng = np.random.default_rng(123); v = rng.uniform(0.1, 1.0, self.nunk())
        d = self.delta(v); ks = set()
        for (m, k, a) in d: ks.add((m*self.a_fr+k, a))
        return sorted(ks, key=lambda x: (x[0], str(x[1])))
    def residual(self, v, keys):
        d = self.delta(v); R = {}
        for (m, k, a), val in d.items():
            key = (m*self.a_fr+k, a); R[key] = R.get(key, 0.0) + val
        return np.array([R.get(kk, 0.0) for kk in keys])

def exists(s, alpha, p, tries=60, seed=0, tol=1e-10):
    S = Setup(s, alpha, p); keys = S.keys(); n = S.nunk()
    if not keys: return True, 0.0, None
    rng = np.random.default_rng(seed); best = None
    lb = np.full(n, -np.inf); lb[:s-1] = 1e-8
    for t in range(tries):
        x0 = rng.uniform(-1, 2, n); x0[:s-1] = rng.uniform(0.05, 1.2, s-1)
        r = least_squares(lambda v: S.residual(v, keys), x0, bounds=(lb, np.inf),
                          xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=3000)
        if best is None or r.cost < best.cost: best = r
        if np.sqrt(2*best.cost) < tol: break
    res = np.sqrt(2*best.cost)
    return res < tol, res, (best.x if res < tol else None)

if __name__ == '__main__':
    import sys, time
    a = float(Fraction(sys.argv[1])); s = int(sys.argv[2]); p = float(Fraction(sys.argv[3]))
    t0 = time.time(); ok, res, x = exists(s, a, p)
    S = Setup(s, a, p)
    print(f'alpha={sys.argv[1]} s={s} p={sys.argv[3]}: {len(S.keys())} conds, {S.nunk()} unk ->',
          'EXISTS' if ok else 'none', f'res={res:.1e}', f'({time.time()-t0:.0f}s)')
