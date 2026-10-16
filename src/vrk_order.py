"""Order conditions for explicit Volterra-RK (Pouzet/Lubich type) methods for Caputo FDE
   y(t) = y0 + J^alpha f(t,y),  step from interior t_n (smooth solution), EXACT lag term Phi_n:
     Y_i     = Phi_n(t_n+c_i h) + h^a sum_{j<i} a_ij f(t_n+c_j h, Y_j)
     y_{n+1} = Phi_n(t_n+h)     + h^a sum_j   b_j  f(t_n+c_j h, Y_j)
   Local error is expanded in h^{m*alpha + k}.  Atoms (scalar case):
     g_k   = d^k/dt^k f(t,y(t))            at t_n
     D_r_k = d^k/dt^k (d^r f/dy^r)(t,y(t)) at t_n
   Exact local integral: (1/G(a)) int_0^c (c-s)^{a-1} s^k ds = c^{k+a} k!/G(k+a+1).
"""
import sympy as sp, numpy as np, itertools
from fractions import Fraction
from scipy.optimize import least_squares

def build(s, alpha, p, rmax=None):
    a_fr = Fraction(alpha).limit_denominator(1000)
    if rmax is None: rmax = max(1, int(Fraction(p).limit_denominator(1000)/a_fr))
    al = sp.Rational(a_fr.numerator, a_fr.denominator)
    pf = Fraction(p).limit_denominator(1000)
    ok = lambda m, k: m*a_fr + k <= pf
    K = int(p) + 1
    g = sp.symbols(f'g0:{K+1}')
    D = {(r, k): sp.Symbol(f'D{r}_{k}') for r in range(1, rmax+1) for k in range(K+1)}
    c = [sp.Integer(0)] + list(sp.symbols(f'c2:{s+1}', positive=True))
    A = {(i, j): sp.Symbol(f'a{i+1}{j+1}') for i in range(s) for j in range(i)}
    b = sp.symbols(f'b1:{s+1}')
    def mul(X, Y):
        Z = {}
        for (m1, k1), v1 in X.items():
            for (m2, k2), v2 in Y.items():
                m, k = m1+m2, k1+k2
                if ok(m, k): Z[(m, k)] = Z.get((m, k), 0) + v1*v2
        return Z
    def add(X, Y, s_=1):
        Z = dict(X)
        for key, v in Y.items(): Z[key] = Z.get(key, 0) + s_*v
        return Z
    def fval(j, e):
        # f(t_n + c_j h, y(t_n+c_j h) + e_j) as series; keys (m,k) mean H1^m H^k (H1=h^a, H=h)
        F = {(0, k): g[k]*c[j]**k/sp.factorial(k) for k in range(K+1) if ok(0, k)}
        P = {(0, 0): sp.Integer(1)}
        for r in range(1, rmax+1):
            P = mul(P, e)
            if not P: break
            Dr = {(0, k): D[(r, k)]*c[j]**k/sp.factorial(k) for k in range(K+1)}
            F = add(F, {kk: v/sp.factorial(r) for kk, v in mul(P, Dr).items()})
        return F
    def shiftH1(X):  # multiply by h^alpha
        return {(m+1, k): v for (m, k), v in X.items() if ok(m+1, k)}
    def exact(ci):
        return {(1, k): -g[k]*ci**(k+al)/sp.gamma(k+al+1) for k in range(K+1) if ok(1, k)}
    e, Fs = [], []
    for i in range(s):
        S = {}
        for j in range(i): S = add(S, {kk: A[(i, j)]*v for kk, v in Fs[j].items()})
        ei = add(shiftH1(S), exact(c[i])) if i > 0 else {}
        e.append(ei); Fs.append(fval(i, ei))
    S = {}
    for j in range(s): S = add(S, {kk: b[j]*v for kk, v in Fs[j].items()})
    delta = add(shiftH1(S), exact(sp.Integer(1)))
    # group by exponent value, split by atom monomials
    groups = {}
    for (m, k), v in delta.items(): groups.setdefault(m*a_fr + k, 0); groups[m*a_fr+k] += v
    atoms = list(g) + list(D.values())
    conds = []
    for E in sorted(groups):
        ex = sp.expand(groups[E])
        if ex == 0: continue
        P = sp.Poly(ex, *atoms)
        for mon, cf in zip(P.monoms(), P.coeffs()):
            conds.append((E, mon, cf))
    unk = list(c[1:]) + list(A.values()) + list(b)
    return conds, unk, atoms

def solve(s, alpha, p, tries=200, seed=0, tol=1e-11):
    conds, unk, atoms = build(s, alpha, p)
    if not conds: return True, None, 0
    fs = [sp.lambdify(unk, cf, 'numpy') for _, _, cf in conds]
    fun = lambda v: np.array([f(*v) for f in fs], dtype=float)
    rng = np.random.default_rng(seed); best = None
    nc = s - 1
    for t in range(tries):
        x0 = rng.uniform(-1, 2, len(unk)); x0[:nc] = rng.uniform(0.05, 1.5, nc)
        lb = np.full(len(unk), -np.inf); lb[:nc] = 1e-6   # c_i > 0
        try:
            r = least_squares(fun, x0, bounds=(lb, np.inf), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=4000)
        except Exception:
            continue
        if best is None or r.cost < best.cost: best = r
        if np.sqrt(2*best.cost) < tol: break
    res = np.sqrt(2*best.cost)
    return res < tol, (dict(zip([str(u) for u in unk], best.x)) if res < tol else None), res

if __name__ == '__main__':
    import sys
    alpha = float(Fraction(sys.argv[1])); s = int(sys.argv[2]); p = float(Fraction(sys.argv[3]))
    conds, unk, _ = build(s, alpha, p)
    print(f'alpha={alpha} s={s} p={p}: {len(conds)} scalar conditions, {len(unk)} unknowns')
    ok, sol, res = solve(s, alpha, p)
    print('  exists' if ok else '  NOT found', f'(min residual {res:.2e})')
    if sol: print('  ', {k: round(v, 6) for k, v in sol.items()})
