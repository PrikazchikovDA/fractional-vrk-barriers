"""Fractional Volterra trees and the B-series of the local error of explicit Volterra-form RK methods.

Tree = ordinary rooted tree, canonical form: sorted tuple of child trees; leaf = ().
For a vertex v: k_v = number of leaf children ("time" vertices, weight 1),
r_v = number of non-leaf children. Non-leaf vertices (incl. the root) are "stage" vertices (weight alpha).
|T| = N(T)*alpha + K(T),  N = #non-leaf vertices (root always counted), K = #leaves (single vertex: N=1,K=0).
Elementary differential: vertex v -> atom D_{r_v,k_v} (D_{0,k}=g_k), applied to the children's differentials.

Local error (theorem in notes):  delta = sum_T h^{|T|} Psi(T)/sigma(T) F(T),
   stage weight  phi(v) = d_k                          if r_v = 0   (d_k = A c^k - k!/G(k+a+1) c^{k+a})
                 phi(v) = A( c^k * prod phi(children) ) if r_v >= 1
   root weight   Psi = D_k (r=0) or b^T( c^k * prod phi(children) ).
sigma = Butcher symmetry factor (identical leaf children included).
"""
from functools import lru_cache
from math import factorial
from collections import Counter
import numpy as np
from scipy.special import gamma as G

LEAF = ()

def trees_by_order(n):
    """All rooted trees with n vertices (canonical sorted tuples)."""
    return _trees(n)

@lru_cache(None)
def _trees(n):
    if n == 1: return (LEAF,)
    out = set()
    # children forests with n-1 vertices: multisets of trees
    for forest in _forests(n - 1, n - 1):
        out.add(tuple(sorted(forest)))
    return tuple(sorted(out))

@lru_cache(None)
def _forests(n, maxsize):
    """multisets (as sorted tuples) of trees with total n vertices, each tree size <= maxsize, nonincreasing key."""
    if n == 0: return ((),)
    res = []
    for m in range(min(n, maxsize), 0, -1):
        for t in _trees(m):
            for rest in _forests(n - m, m):
                # enforce canonical ordering to avoid duplicates: rest trees must be <= t in (size, tuple) order
                if rest and (_size(rest[0]), rest[0]) > (m, t): continue
                res.append((t,) + rest)
    return tuple(res)

@lru_cache(None)
def _size(t): return 1 + sum(_size(c) for c in t)

def split(t):
    k = sum(1 for c in t if c == LEAF); return k, [c for c in t if c != LEAF]

@lru_cache(None)
def NK(t):
    """(N, K): number of stage (non-leaf, root counted) vertices and time leaves."""
    if t == LEAF: return (1, 0)          # single-vertex tree (only possible as root)
    k, ch = split(t); N, K = 1, k
    for c in ch:
        n2, k2 = NK(c); N += n2; K += k2
    return (N, K)

@lru_cache(None)
def sigma(t):
    s = 1
    for c, mu in Counter(t).items(): s *= factorial(mu) * sigma(c) ** mu
    return s

@lru_cache(None)
def label(t):
    """Elementary differential as a nested string, e.g. 'D1,0[g1]'."""
    if t == LEAF: return 'g0'
    k, ch = split(t)
    if not ch: return f'g{k}'
    inner = ','.join(sorted(label(c) for c in ch))
    return f'D{len(ch)},{k}[{inner}]'

@lru_cache(None)
def atoms(t):
    """Scalar monomial: sorted tuple of atoms (('g',k) or ('D',r,k)), same convention as src/vrk_fast.py."""
    if t == LEAF: return (('g', 0),)
    k, ch = split(t)
    if not ch: return (('g', k),)
    a = [('D', len(ch), k)]
    for c in ch: a += list(atoms(c))
    return tuple(sorted(a))

def all_trees(alpha, pmax, nmax=14):
    """All trees with |T| <= pmax (alpha float), sorted by exponent."""
    out = []
    for n in range(1, nmax + 1):
        hit = False
        for t in trees_by_order(n):
            N, K = NK(t)
            if N * alpha + K <= pmax + 1e-12: out.append(t); hit = True
        # a tree with n vertices has N+K >= n/?; stop when no tree of size n qualifies and min exponent exceeds
        if not hit and n > 2 and min(NK(t)[0]*alpha+NK(t)[1] for t in trees_by_order(n)) > pmax + 1: break
    return sorted(out, key=lambda t: (NK(t)[0]*alpha + NK(t)[1], label(t)))

# ---------------- weights ----------------
class Tab:
    def __init__(self, c, A, b, alpha):
        self.c = np.asarray(c, float); self.A = np.asarray(A, float); self.b = np.asarray(b, float); self.a = alpha
    def d(self, k):
        c, a = self.c, self.a
        return self.A @ c**k - factorial(k) / G(k + a + 1) * np.where(c > 0, c, 0.0)**(k + a)
    def D(self, k): return self.b @ self.c**k - factorial(k) / G(k + self.a + 1)

def stage_weight(T, t):
    k, ch = split(t)
    if not ch: return T.d(k)
    v = T.c**k
    for c in ch: v = v * stage_weight(T, c)
    return T.A @ v

def Psi(T, t):
    if t == LEAF: return T.D(0)
    k, ch = split(t)
    if not ch: return T.D(k)
    v = T.c**k
    for c in ch: v = v * stage_weight(T, c)
    return T.b @ v

def unpack(v, s, a):
    """Same parametrisation as src/vrk_fast.py."""
    c = np.concatenate([[0.0], v[:s-1]]); A = np.zeros((s, s)); idx = s-1
    for i in range(1, s):
        for j in range(1, i): A[i, j] = v[idx]; idx += 1
        A[i, 0] = c[i]**a/G(a+1) - A[i, 1:i].sum()
    return c, A, v[idx:idx+s]

# ---------------- formatting ----------------
def _cpow(k): return '' if k == 0 else ('c' if k == 1 else f'c^{k}')
def _stage_expr(t):
    k, ch = split(t)
    if not ch: return f'd_{k}'
    parts = ([_cpow(k)] if k else []) + [_stage_expr(c) for c in sorted(ch)]
    inner = '·'.join(parts)
    return f'A{inner}' if len(parts) == 1 and not inner.startswith('A') and '·' not in inner and not k else f'A({inner})'
def weight_expr(t):
    """Matrix form of Psi(T); '·' = componentwise product."""
    if t == LEAF: return 'D_0'
    k, ch = split(t)
    if not ch: return f'D_{k}'
    parts = ([_cpow(k)] if k else []) + [_stage_expr(c) for c in sorted(ch)]
    return 'bᵀ(' + '·'.join(parts) + ')'
def diff_expr(t):
    """Elementary differential in tensor notation: f^{(r)}_{k} = D_{r,k}."""
    if t == LEAF: return 'g_0'
    k, ch = split(t)
    if not ch: return f'g_{k}'
    args = ','.join(diff_expr(c) for c in sorted(ch))
    return f'D_{{{len(ch)},{k}}}({args})'

@lru_cache(None)
def height(t):
    """Stage height: number of stage (non-leaf) vertices on the longest root path."""
    if t == LEAF: return 1
    k, ch = split(t)
    return 1 + max((height(c) for c in ch), default=0)

def count_conditions(s, alpha, p, scalar=True):
    """Number of order conditions with exponent <= p for an explicit s-stage method
    (trees of stage height > s vanish identically)."""
    ts = [t for t in all_trees(alpha, p) if height(t) <= s]
    return len({atoms(t) for t in ts}) if scalar else len(ts)
