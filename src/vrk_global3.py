"""Explicit VRK for Caputo systems D^a y = f(t,y), y = y0 + J^a f.
Lag term: product integration of piecewise polynomial (degree r) interpolation of ACCEPTED grid
values F_m = f(t_m, y_m) (window of r+1 grid points inside [0, t_n]).  Starting values y_0..y_r exact
(isolates the step order from the starting procedure)."""
import os as _os, sys as _sys
_ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path.insert(0,_os.path.join(_ROOT,'src')); DATA=_os.path.join(_ROOT,'data')

import numpy as np, sys, json

from vrk_global import moments, lagrange_coeffs, ML, tab2, tab3A, tab3B
from scipy.special import gamma as G

def solve(tab, a, f, yex, T, N, r=4):
    c, A, b = (np.array(t, float) for t in tab); s = len(c); h = T/N
    targets = list(c) + [1.0]
    # W[o, i, d, l]: weight of F at node (m - o + l) for interval m=n-d, target i ; nodes theta = l - o
    W = np.zeros((r+1, s+1, N+1, r+1))
    for o in range(r+1):
        L = lagrange_coeffs(np.arange(r+1, dtype=float) - o)
        for i, ci in enumerate(targets):
            for d in range(1, N+1): W[o, i, d] = L @ moments(d + ci, a, r)
    y0 = np.atleast_1d(yex(0.0)).astype(float); dim = y0.size
    Y = np.zeros((N+1, dim)); Fg = np.zeros((N+1, dim))
    for m in range(r+1):
        Y[m] = yex(m*h); Fg[m] = f(m*h, Y[m])
    for n in range(r, N):
        m = np.arange(n); j0 = np.clip(m - r//2, 0, n - r); o = m - j0; d = n - m
        idx = j0[:, None] + np.arange(r+1)[None, :]
        Fw = Fg[idx]                                      # (n, r+1, dim)
        lag = np.empty((s+1, dim))
        for i in range(s+1):
            w = W[o, i, d, :]                             # (n, r+1)
            lag[i] = y0 + h**a * np.einsum('ml,mld->d', w, Fw)
        Fs = np.zeros((s, dim))
        for i in range(s):
            Fs[i] = f(n*h + c[i]*h, lag[i] + h**a * (A[i, :i] @ Fs[:i]))
        Y[n+1] = lag[s] + h**a * (b @ Fs); Fg[n+1] = f((n+1)*h, Y[n+1])
    return Y[-1]

def problems(a):
    g = lambda k: G(k+1)/G(k+1-a)
    P = {}
    P['linear (Ex.1)'] = (lambda t, y: -y + t**(4-a)/G(5-a), lambda t: np.array([t**4*ML(-t**a, a, 5)]))
    P['nonlinear y=t^4'] = (lambda t, y: g(4)*t**(4-a) + y**2 - t**8, lambda t: np.array([t**4]))
    P['2D system'] = (lambda t, y: np.array([g(4)*t**(4-a) + y[0]*y[1] - t**9 - np.sin(y[1]) + np.sin(t**5),
                                             g(5)*t**(5-a) - y[0]**2 + t**8 + t*(y[0]-t**4)]),
                      lambda t: np.array([t**4, t**5]))
    return P

if __name__ == '__main__':
    tab4 = json.load(open(_os.path.join(DATA,_os.path.join(DATA,'tab4.json'))))
    Ns = [40, 80, 160, 320]
    import sys as _s
    for a in [float(x) for x in _s.argv[1:]] or [0.3, 0.7]:
        meths = [('2-st', tab2(a), 1+2*a), ('3-st A', tab3A(a), 2+a), ('3-st B', tab3B(a), 1+3*a),
                 ('4-st opt', (tab4[str(a)]['c'], tab4[str(a)]['A'], tab4[str(a)]['b']), max(1+4*a, 2+2*a))]
        for pname, (f, yex) in problems(a).items():
            print(f'alpha={a}  problem: {pname}')
            for mname, tb, q in meths:
                errs = [np.abs(solve(tb, a, f, yex, 1.0, N) - yex(1.0)).max() for N in Ns]
                rates = [np.log2(errs[i]/errs[i+1]) for i in range(len(Ns)-1)]
                print(f'   {mname:9s} predicted {q:.2f}: ' + ' '.join(f'{e:.1e}' for e in errs) + '  rates ' + ' '.join(f'{x:.2f}' for x in rates), flush=True)
