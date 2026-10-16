"""Existence search for explicit Volterra-form methods via the tree conditions.

residual(v) = [Psi(T)/sigma(T) for trees with |T| < p and ht(T) <= s]           (mode='system')
            = [sum over trees with the same atom monomial of Psi/sigma]           (mode='scalar')
Parametrisation as in src/vrk_fast.py (row sums imposed). Nodes in [0, cmax] (cmax=None: c>=0 only).
Failure of the search is numerical evidence only, never a proof of non-existence."""
import numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
from vtrees import all_trees, NK, sigma, atoms, height, split, LEAF, unpack
from math import factorial
from scipy.special import gamma as G

class Conds:
    def __init__(self, s, alpha, p, mode='system', strict=True):
        self.s, self.a, self.mode = s, alpha, mode
        eps = 1e-9
        ts = [t for t in all_trees(alpha, p + 1.0) if height(t) <= s and
              ((NK(t)[0]*alpha + NK(t)[1] < p - eps) if strict else (NK(t)[0]*alpha + NK(t)[1] <= p + eps))]
        self.trees = ts
        if mode == 'scalar':
            g = defaultdict(list)
            for t in ts: g[atoms(t)].append(t)
            self.groups = list(g.values())
        else:
            self.groups = [[t] for t in ts]
        self.sig = [[float(sigma(t)) for t in grp] for grp in self.groups]
        self.ncond = len(self.groups)
        self.nunk = (s-1) + (s-1)*(s-2)//2 + s
    def residual(self, v):
        c, A, b = unpack(v, self.s, self.a); a = self.a
        cpos = np.where(c > 0, c, 0.0)
        dcache = {}; memo = {}
        def d(k):
            if k not in dcache: dcache[k] = A @ c**k - factorial(k)/G(k+a+1) * cpos**(k+a)
            return dcache[k]
        def st(t):
            if t in memo: return memo[t]
            k, ch = split(t)
            if not ch: r = d(k)
            else:
                w = c**k
                for u in ch: w = w * st(u)
                r = A @ w
            memo[t] = r; return r
        def psi(t):
            if t == LEAF: return b.sum() - 1/G(a+1)
            k, ch = split(t)
            if not ch: return b @ c**k - factorial(k)/G(k+a+1)
            w = c**k
            for u in ch: w = w * st(u)
            return b @ w
        return np.array([sum(psi(t)/sg for t, sg in zip(grp, sgs)) for grp, sgs in zip(self.groups, self.sig)])

def search(s, alpha, p, mode='system', cmax=1.5, tries=40, seed=0, tol=1e-11, max_nfev=4000):
    C = Conds(s, alpha, p, mode)
    n = C.nunk
    lb = np.full(n, -np.inf); ub = np.full(n, np.inf); lb[:s-1] = 0.0
    if cmax is not None: ub[:s-1] = cmax
    rng = np.random.default_rng(seed); best = None
    for t in range(tries):
        x0 = rng.uniform(-1.5, 1.5, n); x0[:s-1] = rng.uniform(0.02, (cmax or 1.5), s-1)
        r = least_squares(C.residual, x0, bounds=(lb, ub), xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
        if best is None or r.cost < best.cost: best = r
        if np.sqrt(2*best.cost) < tol: break
    res = float(np.sqrt(2*best.cost))
    return dict(found=res < tol, res=res, ncond=C.ncond, nunk=n, tries=t+1,
                maxcoef=float(np.abs(best.x).max()), x=best.x.tolist())

def _to_v(z, s, cmax):
    v = z.copy()
    v[:s-1] = cmax*np.sin(z[:s-1])**2 if cmax is not None else z[:s-1]**2
    return v
def _to_z(v, s, cmax):
    z = np.array(v, float).copy()
    if cmax is not None: z[:s-1] = np.arcsin(np.sqrt(np.clip(z[:s-1]/cmax, 0, 1)))
    else: z[:s-1] = np.sqrt(np.maximum(z[:s-1], 0))
    return z

def search_lm(s, alpha, L, mode='system', cmax=1.5, tries=60, seed=0, tol=1e-11, max_nfev=3000, warm=None, inclusive=True):
    """Levenberg-Marquardt (MINPACK) with node reparametrisation c = cmax*sin^2(theta);
    conditions |T| <= L (inclusive) or < L."""
    C = Conds(s, alpha, L + (1e-7 if inclusive else 0.0), mode)
    n = C.nunk; pad = max(0, n - C.ncond)
    F = lambda z: np.concatenate([C.residual(_to_v(z, s, cmax)), np.zeros(pad)])
    rng = np.random.default_rng(seed); best = None; t = 0
    starts = ([_to_z(warm, s, cmax)] if warm is not None else []) + [None]*tries
    for t, z0 in enumerate(starts):
        if z0 is None:
            z0 = rng.uniform(-1.5, 1.5, n); z0[:s-1] = rng.uniform(0.05, 1.5, s-1)
        r = least_squares(F, z0, method='lm', xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=max_nfev)
        if best is None or r.cost < best.cost: best = r
        if np.sqrt(2*best.cost) < tol: break
    v = _to_v(best.x, s, cmax); res = float(np.sqrt(2*best.cost))
    return dict(found=res < tol, res=res, ncond=C.ncond, nunk=n, tries=t+1,
                maxcoef=float(np.abs(v).max()), x=v.tolist())
