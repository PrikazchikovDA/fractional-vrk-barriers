"""Linear-core order conditions (test problem y = Phi + J^a(lambda y + g(t)), constant lambda):
   Q_k(z) = D_k + sum_{m>=1} z^m b A^{m-1} d_k ; term exponent k + a + m a.
   Row-sum built in. Checks feasibility of all conditions with exponent < or <= target."""
import numpy as np, sys
from scipy.special import gamma as G
from scipy.optimize import least_squares
def unpack(v, s, a):
    c = np.concatenate([[0.0], v[:s-1]]); A = np.zeros((s, s)); idx = s-1
    for i in range(1, s):
        for j in range(1, i): A[i, j] = v[idx]; idx += 1
        A[i, 0] = c[i]**a/G(a+1) - A[i, 1:i].sum()
    return c, A, v[idx:idx+s]
def nunk(s): return (s-1) + (s-1)*(s-2)//2 + s
def resid(v, s, a, q, strict):
    c, A, b = unpack(v, s, a); out = []
    K = int(np.ceil(q)) + 1
    for k in range(K+1):
        if k + a > q + 1e-9 or (strict and k + a > q - 1e-9): continue
        out.append(b @ c**k - G(k+1)/G(k+a+1))
        dk = A @ c**k - G(k+1)*c**(k+a)/G(k+a+1)
        if k == 0: continue
        v_ = dk.copy()
        for m in range(1, s):
            e = k + a + m*a
            if e > q + 1e-9 or (strict and e > q - 1e-9): break
            out.append(b @ v_); v_ = A @ v_
    return np.array(out)
def feasible(s, a, q, strict, tries=40, seed=0):
    rng = np.random.default_rng(seed); n = nunk(s); lb = np.full(n, -np.inf); lb[:s-1] = 1e-8; best = None
    for t in range(tries):
        x0 = rng.uniform(-1, 2, n); x0[:s-1] = rng.uniform(0.05, 1.5, s-1)
        r = least_squares(lambda v: resid(v, s, a, q, strict), x0, bounds=(lb, np.inf), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        if best is None or r.cost < best.cost: best = r
        if np.sqrt(2*best.cost) < 1e-12: break
    return np.sqrt(2*best.cost), np.abs(best.x).max()
if __name__ == '__main__':
    for s in [3, 4, 5, 6]:
        for a in [0.3, 0.4, 0.6, 0.7, 0.8]:
            qs = max(1+s*a, 2+(s-2)*a)
            r_lt, m_lt = feasible(s, a, qs, strict=True)
            r_le, m_le = feasible(s, a, qs, strict=False)
            print(f's={s} a={a} q*={qs:.2f}: below q* res={r_lt:.1e} | incl. q* res={r_le:.1e} (max|par|={m_le:.0e})', flush=True)
